//! The Qwenodyssey pipeline (port of `src/core/orchestrator.ts`):
//!
//! ```text
//!   scan → plan → context → code → patch → review → test → fix loop → summary
//! ```
//!
//! Modes tune which stages run (see [`AgentMode`]). Everything the TS printed
//! or asked goes through [`PipelineUi`]; structured events go to the JSONL
//! session [`Logger`]; every model call and check run honours the turn's Esc
//! token (taken from the tool registry's context unless overridden).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::agents::{coder, edit_coder, error_fixer, planner, reviewer, tester};
use crate::context_builder::{build_context, guess_relevant_files, ContextInput, DEFAULT_RELEVANT_LIMIT};
use crate::deps::{load_prompt, render, resolve_writable};
use crate::edit_pipeline::{EventSink, PipelineEvent, Resolver};
use crate::memory::MemoryStore;
use crate::patch_engine::{apply_edits, mode_name, preview_edit_or_note, AppliedEdit};
use crate::render::{render_plan, render_review};
use crate::repo_scanner::scan_repo;
use crate::ui::PipelineUi;
use qo_core::config::{Config, EditProtocol};
use qo_core::logger::Logger;
use qo_core::{AgentMode, EditMode, FileEdit, Message, ModelOptions, Plan, RepoInfo, SharedProvider, ToolRegistry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    pub mode: AgentMode,
    /// `--yes`: skip every confirmation.
    pub auto_confirm: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub edits: Vec<FileEdit>,
    pub applied: bool,
    pub tests_passed: Option<bool>,
    pub summary: String,
}

impl RunResult {
    fn not_applied(edits: Vec<FileEdit>, summary: &str) -> Self {
        Self { edits, applied: false, tests_passed: None, summary: summary.to_string() }
    }
}

/// One line of an apply report.
struct ApplyLine {
    path: String,
    mode: String,
    ok: bool,
    error: Option<String>,
}

impl From<&AppliedEdit> for ApplyLine {
    fn from(a: &AppliedEdit) -> Self {
        Self { path: a.path.clone(), mode: a.mode.clone(), ok: a.ok, error: a.error.clone() }
    }
}

pub struct Orchestrator {
    cwd: PathBuf,
    provider: SharedProvider,
    config: Config,
    tools: ToolRegistry,
    log: Logger,
    memory: MemoryStore,
    ui: Arc<dyn PipelineUi>,
    cancel: Option<CancellationToken>,
}

impl Orchestrator {
    /// The Esc token defaults to the registry context's `cancel` (the CLI
    /// installs a fresh one per turn); see [`Orchestrator::with_cancel`].
    pub fn new(
        cwd: impl Into<PathBuf>,
        provider: SharedProvider,
        config: Config,
        tools: ToolRegistry,
        log: Logger,
        memory: MemoryStore,
        ui: Arc<dyn PipelineUi>,
    ) -> Self {
        let cancel = tools.context().cancel.clone();
        Self { cwd: cwd.into(), provider, config, tools, log, memory, ui, cancel }
    }

    /// Use `cancel` for every model call and check run of this orchestrator.
    pub fn with_cancel(mut self, cancel: Option<CancellationToken>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    fn budget_tokens(&self) -> usize {
        (self.config.model.context_tokens as f64 * 0.6).floor() as usize
    }

    fn start_log(&self, task: &str) {
        if let Err(e) = self.log.start(task) {
            self.ui.debug(&format!("session log unavailable: {e}"));
        }
    }

    pub async fn run_code(&self, task: &str, opts: RunOptions) -> anyhow::Result<RunResult> {
        let RunOptions { mode, auto_confirm } = opts;
        let cancel = self.cancel.as_ref();
        self.start_log(task);
        self.log.event(json!({ "type": "task", "task": task, "mode": mode.as_str() }));

        // 1. Scan
        self.ui.step("Scanning repo…");
        let repo = scan_repo(&self.cwd).await;
        self.ui.debug(&serde_json::to_string(&repo).unwrap_or_default());
        self.log.event(json!({ "type": "scan", "repo": repo }));
        if repo.dirty {
            self.ui.warn("Working tree has uncommitted changes — commit or stash first if you want clean rollback points.");
        }

        // 2. Plan (skipped in fast mode)
        let file_list = self.tools.run("list_files", json!({ "pattern": "**/*" })).await;
        let files = string_array(file_list.data.as_ref());
        let plan = if mode == AgentMode::Fast {
            Plan {
                goal: task.to_string(),
                steps: vec![qo_core::PlanStep {
                    index: 1,
                    title: task.to_string(),
                    detail: Some(task.to_string()),
                    files: Vec::new(),
                }],
                files: Vec::new(),
                assumptions: Vec::new(),
            }
        } else {
            self.ui.step("Planning…");
            let plan = planner::plan(self.provider.as_ref(), task, &repo, &files, cancel).await?;
            self.ui.block(&render_plan(&plan));
            self.log.event(json!({ "type": "plan", "plan": plan }));
            if mode != AgentMode::Deep && !auto_confirm && !self.ui.ask_yes_no("Proceed with this plan?", true).await {
                return Ok(RunResult::not_applied(Vec::new(), "Aborted by user at plan stage."));
            }
            plan
        };

        // 3. Context
        let relevant =
            if plan.files.is_empty() { guess_relevant_files(&repo, task, DEFAULT_RELEVANT_LIMIT) } else { plan.files.clone() };
        let git_diff = if repo.has_git { Some(self.tools.run("git_diff", json!({})).await.output) } else { None };
        let memory = if self.config.memory.enabled { Some(self.memory.digest()) } else { None };
        let context = build_context(
            self.provider.as_ref(),
            &ContextInput {
                user_request: task,
                repo: &repo,
                files: &relevant,
                errors: None,
                git_diff: git_diff.as_deref(),
                memory: memory.as_deref(),
                budget_tokens: self.budget_tokens(),
            },
        );

        // 4. Code
        self.ui.step("Generating edits…");

        // SEARCH/REPLACE block pipeline (best-of-N + verify + self-repair) —
        // the default for small models and where most of the reliability gain
        // comes from. Falls back to the legacy JSON-diff coder when configured.
        if self.config.agent.edit_protocol == EditProtocol::Blocks {
            return self.run_blocks(task, &context, mode, auto_confirm).await;
        }

        let coder_out = coder::code(self.provider.as_ref(), task, &plan, &context, cancel).await?;
        if coder_out.edits.is_empty() {
            return Ok(RunResult::not_applied(Vec::new(), "Model produced no edits. Try rephrasing or use --deep."));
        }
        let edit_list: Vec<Value> =
            coder_out.edits.iter().map(|e| json!({ "path": e.path, "mode": mode_name(e.mode) })).collect();
        self.log.event(json!({ "type": "code", "edits": edit_list }));
        self.show_edits(&coder_out.edits);

        // 5. Review (safe/deep)
        let careful = matches!(mode, AgentMode::Safe | AgentMode::Deep);
        if careful && self.config.agent.review_before_apply {
            self.ui.step("Reviewing…");
            let review = reviewer::review(self.provider.as_ref(), &self.cwd, task, &coder_out.edits, cancel).await?;
            self.ui.block(&render_review(&review));
            self.log.event(json!({ "type": "review", "review": review }));
            if !review.approve
                && !auto_confirm
                && !self.ui.ask_yes_no("Reviewer flagged issues. Apply anyway?", false).await
            {
                return Ok(RunResult::not_applied(coder_out.edits, "Held back after review."));
            }
        }

        // 6. Apply (with confirmation unless auto)
        let should_apply = auto_confirm || self.config.agent.auto_apply || self.confirm_apply().await;
        if !should_apply {
            return Ok(RunResult::not_applied(
                coder_out.edits,
                "Edits generated but not applied. Re-run with --yes or `qwenodyssey apply`.",
            ));
        }
        let outcome = apply_edits(&self.cwd, &coder_out.edits)?;
        self.report_apply(outcome.applied.iter().map(ApplyLine::from));
        self.log.event(json!({ "type": "apply", "journal": path_str(outcome.journal_file.as_deref()), "ok": outcome.all_ok }));

        // 7. Test + fix loop (deep retries; safe runs once)
        let mut tests_passed = None;
        if self.config.agent.auto_test && careful {
            tests_passed = Some(self.test_and_fix_loop(task, &repo, mode).await?);
        }

        // 8. Summary
        let summary = self.summarize(task, &coder_out.edits, tests_passed).await;
        self.log.event(json!({ "type": "summary", "summary": summary, "testsPassed": tests_passed }));
        Ok(RunResult { edits: coder_out.edits, applied: true, tests_passed, summary })
    }

    /// The SEARCH/REPLACE path of [`Orchestrator::run_code`]. The pipeline
    /// applies as it goes, so the apply confirmation comes first.
    async fn run_blocks(&self, task: &str, context: &str, mode: AgentMode, auto_confirm: bool) -> anyhow::Result<RunResult> {
        let cwd = self.cwd.clone();
        let self_root = self.tools.context().self_root.clone();
        let resolve: Resolver = Arc::new(move |rel: &str| resolve_writable(&cwd, self_root.as_deref(), rel));
        let careful = matches!(mode, AgentMode::Safe | AgentMode::Deep);
        let run_verify = self.config.agent.verify_after_edit && careful;
        let apply_first = auto_confirm || self.config.agent.auto_apply || self.confirm_apply().await;
        if !apply_first {
            return Ok(RunResult::not_applied(Vec::new(), "Held back before editing. Re-run with --yes."));
        }
        let ui = self.ui.clone();
        let on_event: EventSink =
            Arc::new(move |e: &PipelineEvent| ui.debug(&serde_json::to_string(e).unwrap_or_default()));
        let max_repair_rounds = if mode == AgentMode::Deep {
            self.config.agent.max_retries as usize
        } else if run_verify {
            1
        } else {
            0
        };
        let result = edit_coder::code_with_blocks(edit_coder::EditCoderOptions {
            provider: self.provider.as_ref(),
            task,
            context,
            root: self.cwd.clone(),
            resolve,
            candidates: self.config.agent.candidates.max(1) as usize,
            max_repair_rounds,
            run_verify,
            verify_only: None,
            base_temperature: Some(self.config.model.temperature),
            cancel: self.cancel.clone(),
            on_event: Some(on_event),
        })
        .await?;
        self.log.event(json!({
            "type": "code_blocks",
            "applied": result.applied,
            "verified": result.verified,
            "rounds": result.rounds,
        }));
        if !result.applied {
            return Ok(RunResult::not_applied(Vec::new(), &result.message));
        }
        let edits: Vec<FileEdit> = result
            .blocks
            .iter()
            .map(|b| FileEdit { path: b.path.clone(), mode: EditMode::Patch, diff: None, content: None, rationale: None })
            .collect();
        self.report_apply(result.apply_results.iter().map(|r| ApplyLine {
            path: r.path.clone(),
            mode: r.strategy.map_or_else(|| "edit".to_string(), |s| s.as_str().to_string()),
            ok: r.ok,
            error: r.error.clone(),
        }));
        let tests_passed = result.verify.as_ref().map(|v| v.all_passed);
        let summary = self.summarize(task, &edits, tests_passed).await;
        Ok(RunResult { edits, applied: true, tests_passed, summary: format!("{}\n\n{summary}", result.message) })
    }

    /// `--autofix`: run tests, diagnose failures, patch, repeat. No initial codegen.
    pub async fn run_autofix(&self, task: &str) -> anyhow::Result<RunResult> {
        let goal = if task.trim().is_empty() { "Make the failing tests pass." } else { task };
        self.start_log(&format!("autofix: {goal}"));
        self.ui.step("Scanning repo…");
        let repo = scan_repo(&self.cwd).await;
        if repo.test_command.as_deref().map_or(true, str::is_empty) {
            return Ok(RunResult::not_applied(Vec::new(), "No test command detected; nothing to autofix."));
        }
        let passed = self.test_and_fix_loop(goal, &repo, AgentMode::Deep).await?;
        let summary = if passed {
            "Autofix succeeded: tests now pass."
        } else {
            "Autofix could not get tests passing within the retry budget."
        };
        Ok(RunResult { edits: Vec::new(), applied: true, tests_passed: Some(passed), summary: summary.to_string() })
    }

    async fn test_and_fix_loop(&self, task: &str, repo: &RepoInfo, mode: AgentMode) -> anyhow::Result<bool> {
        let max_retries = if mode == AgentMode::Deep { self.config.agent.max_retries } else { 0 };
        for attempt in 0..=max_retries {
            self.ui.step(&if attempt == 0 {
                "Running tests…".to_string()
            } else {
                format!("Re-running tests (attempt {attempt})…")
            });
            let result = tester::run_tests(&self.tools, repo, None).await;
            self.log.event(json!({ "type": "test", "attempt": attempt, "ran": result.ran, "passed": result.passed }));
            if !result.ran {
                self.ui.warn(&result.output);
                return Ok(false);
            }
            if result.passed {
                self.ui.success("Tests passed.");
                return Ok(true);
            }
            self.ui.error("Tests failed.");
            if attempt >= max_retries {
                self.ui.warn(&format!("Giving up after {} attempt(s).", max_retries + 1));
                return Ok(false);
            }
            if self.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(qo_core::ProviderError::Interrupted.into());
            }

            // Fix loop
            self.ui.step("Analyzing failure and patching…");
            let relevant = guess_relevant_files(repo, task, DEFAULT_RELEVANT_LIMIT);
            let context = build_context(
                self.provider.as_ref(),
                &ContextInput {
                    user_request: task,
                    repo,
                    files: &relevant,
                    errors: Some(&result.output),
                    git_diff: None,
                    memory: None,
                    budget_tokens: self.budget_tokens(),
                },
            );
            let fix =
                error_fixer::fix_errors(self.provider.as_ref(), task, &result.output, &context, self.cancel.as_ref()).await?;
            if fix.edits.is_empty() {
                self.ui.warn("No fix proposed.");
                return Ok(false);
            }
            self.show_edits(&fix.edits);
            let outcome = apply_edits(&self.cwd, &fix.edits)?;
            self.report_apply(outcome.applied.iter().map(ApplyLine::from));
            self.log.event(json!({
                "type": "fix_apply",
                "attempt": attempt,
                "journal": path_str(outcome.journal_file.as_deref()),
            }));
        }
        Ok(false)
    }

    /// One short model-written summary; any failure falls back to a count.
    async fn summarize(&self, task: &str, edits: &[FileEdit], tests_passed: Option<bool>) -> String {
        let fallback = format!("Applied {} edit(s).", edits.len());
        let edit_lines = edits.iter().map(|e| format!("{} {}", mode_name(e.mode), e.path)).collect::<Vec<_>>().join("\n");
        let tests = match tests_passed {
            None => "not run",
            Some(true) => "passed",
            Some(false) => "failed",
        };
        let (Ok(template), Ok(system)) = (load_prompt("summarizer"), load_prompt("system")) else {
            return fallback;
        };
        let prompt = render(&template, &[("task", task), ("edits", &edit_lines), ("tests", tests)]);
        let opts = ModelOptions {
            temperature: Some(0.2),
            max_tokens: Some(400),
            cancel: self.cancel.clone(),
            ..Default::default()
        };
        match self.provider.generate(&[Message::system(system), Message::user(prompt)], &opts).await {
            Ok(res) => res.text.trim().to_string(),
            Err(_) => fallback,
        }
    }

    /* ── UI helpers ── */

    fn show_edits(&self, edits: &[FileEdit]) {
        for e in edits {
            self.ui.show_edit_diff(e, &preview_edit_or_note(&self.cwd, e));
        }
    }

    fn report_apply(&self, lines: impl Iterator<Item = ApplyLine>) {
        for a in lines {
            if a.ok {
                self.ui.success(&format!("{} {}", a.mode, a.path));
            } else {
                self.ui.error(&format!("{} {} — {}", a.mode, a.path, a.error.unwrap_or_default()));
            }
        }
    }

    async fn confirm_apply(&self) -> bool {
        self.ui.ask_yes_no("Apply these edits?", true).await
    }
}

/// Strings of a JSON array (the `list_files` tool's `data`).
fn string_array(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn path_str(p: Option<&Path>) -> Option<String> {
    p.map(|p| p.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{registry_with, scratch_repo, FakeProvider, FakeTool};
    use crate::ui::RecordingUi;
    use qo_core::ToolResult;
    use std::fs;

    struct Rig {
        dir: tempfile::TempDir,
        provider: Arc<FakeProvider>,
        ui: Arc<RecordingUi>,
        shell: Arc<FakeTool>,
    }

    fn rig(replies: Vec<&str>, answers: Vec<bool>, shell: ToolResult) -> Rig {
        let dir = scratch_repo();
        Rig {
            dir,
            provider: Arc::new(FakeProvider::new(replies.into_iter().map(str::to_string).collect()).with_fallback("Done.")),
            ui: Arc::new(RecordingUi::new(false).with_answers(answers)),
            shell: FakeTool::new("run_shell", shell),
        }
    }

    fn orch(r: &Rig, config: Config) -> Orchestrator {
        let files = FakeTool::new("list_files", ToolResult::ok("").with_data(json!(["a.py"])));
        let diff = FakeTool::new("git_diff", ToolResult::ok("(no changes)"));
        let reg = registry_with(r.dir.path(), vec![files, diff, r.shell.clone()]);
        Orchestrator::new(
            r.dir.path(),
            r.provider.clone(),
            config,
            reg,
            Logger::new(r.dir.path(), false),
            MemoryStore::new(r.dir.path().join(".qwenodyssey/memory")),
            r.ui.clone(),
        )
    }

    const PLAN: &str = r#"{"goal":"bump x","files":["a.py"],"steps":[{"index":1,"title":"edit a.py"}]}"#;
    const BLOCK: &str = "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE";

    #[tokio::test]
    async fn blocks_path_plans_applies_and_summarizes() {
        let r = rig(vec![PLAN, BLOCK, "Bumped x."], vec![], ToolResult::ok(""));
        fs::write(r.dir.path().join("a.py"), "x = 1\n").unwrap();
        let o = orch(&r, Config::default());
        let res = o.run_code("bump x", RunOptions { mode: AgentMode::Safe, auto_confirm: true }).await.unwrap();
        assert!(res.applied, "{}", res.summary);
        assert_eq!(fs::read_to_string(r.dir.path().join("a.py")).unwrap(), "x = 2\n");
        assert_eq!(res.edits.len(), 1);
        assert_eq!(res.edits[0].mode, EditMode::Patch);
        // No checks in the scratch project: applied, verification reports "no checks".
        assert!(res.summary.starts_with("Edits applied (no checks detected).\n\nBumped x."), "{}", res.summary);
        assert_eq!(res.tests_passed, Some(true));
        let t = r.ui.transcript();
        assert!(t.contains("Scanning repo…") && t.contains("Planning…") && t.contains("Proposed plan: "));
        assert!(t.contains("exact a.py"), "{t}");
        assert!(r.ui.questions().is_empty(), "--yes asks nothing");
        // The planner saw the listed files; the coder saw the file content.
        let reqs = r.provider.requests();
        assert!(reqs[0].messages[1].content.contains("a.py"));
        assert!(reqs[1].messages[1].content.contains("x = 1"));
        let summary_req = reqs.last().unwrap();
        assert_eq!((summary_req.temperature, summary_req.max_tokens), (Some(0.2), Some(400)));
        // A session log was written.
        assert!(r.dir.path().join(".qwenodyssey/logs").exists());
    }

    #[tokio::test]
    async fn declining_the_plan_aborts() {
        let r = rig(vec![PLAN], vec![false], ToolResult::ok(""));
        let o = orch(&r, Config::default());
        let res = o.run_code("bump x", RunOptions { mode: AgentMode::Safe, auto_confirm: false }).await.unwrap();
        assert!(!res.applied);
        assert_eq!(res.summary, "Aborted by user at plan stage.");
        assert_eq!(r.ui.questions(), vec!["Proceed with this plan?"]);
    }

    #[tokio::test]
    async fn fast_mode_skips_planning_and_can_hold_back() {
        let r = rig(vec![], vec![false], ToolResult::ok(""));
        let o = orch(&r, Config::default());
        let res = o.run_code("bump x", RunOptions { mode: AgentMode::Fast, auto_confirm: false }).await.unwrap();
        assert_eq!(res.summary, "Held back before editing. Re-run with --yes.");
        assert!(r.provider.requests().is_empty());
        assert_eq!(r.ui.questions(), vec!["Apply these edits?"]);
    }

    #[tokio::test]
    async fn json_path_reviews_applies_and_tests() {
        let edits = r#"{"edits":[{"path":"a.py","mode":"rewrite","content":"x = 3\n","rationale":"bump"}]}"#;
        let review = r#"{"approve":true,"summary":"fine","issues":[]}"#;
        let r = rig(vec![PLAN, edits, review, "Summary."], vec![], ToolResult::ok("1 passed"));
        fs::write(r.dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(r.dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        let mut cfg = Config::default();
        cfg.agent.edit_protocol = EditProtocol::Json;
        let o = orch(&r, cfg);
        let res = o.run_code("bump x", RunOptions { mode: AgentMode::Safe, auto_confirm: true }).await.unwrap();
        assert!(res.applied);
        assert_eq!(res.tests_passed, Some(true));
        assert_eq!(res.summary, "Summary.");
        assert_eq!(fs::read_to_string(r.dir.path().join("a.py")).unwrap(), "x = 3\n");
        assert_eq!(r.shell.calls()[0]["command"], "cargo test");
        let t = r.ui.transcript();
        assert!(t.contains("REWRITE a.py") && t.contains("Review: ") && t.contains("Tests passed."), "{t}");
        assert!(crate::patch_engine::latest_journal(r.dir.path()).is_some());
    }

    #[tokio::test]
    async fn json_path_holds_back_after_negative_review() {
        let edits = r#"{"edits":[{"path":"a.py","mode":"rewrite","content":"bad"}]}"#;
        let review = r#"{"approve":false,"issues":[{"severity":"error","message":"breaks it"}]}"#;
        let r = rig(vec![PLAN, edits, review], vec![true, false], ToolResult::ok(""));
        fs::write(r.dir.path().join("a.py"), "x = 1\n").unwrap();
        let mut cfg = Config::default();
        cfg.agent.edit_protocol = EditProtocol::Json;
        let o = orch(&r, cfg);
        let res = o.run_code("bump x", RunOptions { mode: AgentMode::Safe, auto_confirm: false }).await.unwrap();
        assert!(!res.applied);
        assert_eq!(res.summary, "Held back after review.");
        assert_eq!(res.edits.len(), 1, "edits are returned so `apply` can finish later");
        assert_eq!(fs::read_to_string(r.dir.path().join("a.py")).unwrap(), "x = 1\n");
    }

    #[tokio::test]
    async fn autofix_loops_until_tests_pass_or_budget() {
        // Tests always fail; the fixer proposes a rewrite each time → gives up
        // after max_retries + 1 attempts.
        let fix = r#"{"edits":[{"path":"a.py","mode":"rewrite","content":"x = 9\n"}]}"#;
        let r = rig(vec![fix, fix], vec![], ToolResult::err("1 failed"));
        fs::write(r.dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        let mut cfg = Config::default();
        cfg.agent.max_retries = 2;
        let o = orch(&r, cfg);
        let res = o.run_autofix("").await.unwrap();
        assert_eq!(res.tests_passed, Some(false));
        assert_eq!(res.summary, "Autofix could not get tests passing within the retry budget.");
        assert_eq!(r.shell.calls().len(), 3);
        assert!(r.ui.transcript().contains("Giving up after 3 attempt(s)."));
        assert_eq!(fs::read_to_string(r.dir.path().join("a.py")).unwrap(), "x = 9\n");
    }

    #[tokio::test]
    async fn autofix_without_tests_does_nothing() {
        let r = rig(vec![], vec![], ToolResult::ok(""));
        let o = orch(&r, Config::default());
        let res = o.run_autofix("x").await.unwrap();
        assert_eq!(res.summary, "No test command detected; nothing to autofix.");
    }

    /// The binary spawns / `select!`s on these futures, so they must be `Send`.
    #[test]
    fn pipeline_futures_are_send() {
        fn assert_send<T: Send>(_: &T) {}
        let r = rig(vec![], vec![], ToolResult::ok(""));
        let o = orch(&r, Config::default());
        let f = o.run_code("x", RunOptions { mode: AgentMode::Safe, auto_confirm: true });
        assert_send(&f);
        let g = o.run_autofix("x");
        assert_send(&g);
        let h = crate::commands::code_command(&o, "x", AgentMode::Safe, true);
        assert_send(&h);
        let p = FakeProvider::new(vec![]);
        let ui = RecordingUi::new(false);
        let reg = registry_with(r.dir.path(), vec![]);
        let i = crate::commands::review_command(&p, &reg, &ui);
        assert_send(&i);
        let j = crate::commands::plan_command(&p, &reg, r.dir.path(), "g", &ui);
        assert_send(&j);
    }

    #[tokio::test]
    async fn interrupt_propagates_as_an_error() {
        let r = rig(vec!["__interrupt__"], vec![], ToolResult::ok(""));
        let o = orch(&r, Config::default());
        let e = o.run_code("x", RunOptions { mode: AgentMode::Safe, auto_confirm: true }).await.unwrap_err();
        assert!(qo_core::error::is_interrupt(&e));
    }
}
