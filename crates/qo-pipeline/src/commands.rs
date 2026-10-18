//! The logic of the pipeline CLI commands (ports of
//! `src/cli/commands/{code,plan,edit,apply,review,test,memory,init}.ts`) as
//! library functions: the binary parses flags, builds the session (provider,
//! config, registry, logger, memory), calls these, and renders the outcome.
//! Interactive questions and progress lines go through [`PipelineUi`].

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::agents::{coder, reviewer, tester};
use crate::context_builder::{build_context, ContextInput};
use crate::deps::{extract_json, load_prompt, render, resolve_inside, workspace_path};
use crate::memory::{MemoryCategory, MemoryEntry, MemoryStore};
use crate::orchestrator::{Orchestrator, RunOptions, RunResult};
use crate::patch_engine::{apply_edits, latest_journal, preview_edit_or_note, rollback, ApplyOutcome};
use crate::repo_scanner::scan_repo;
use crate::ui::PipelineUi;
use qo_core::config::{config_path, default_config, save_config, Config};
use qo_core::logger::Logger;
use qo_core::{AgentMode, FileEdit, Plan, PlanStep, Provider, ReviewResult, TestResult, ToolRegistry};

/* ── backend check (code) ── */

/// Result of the pre-flight health check `code` runs before any work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendCheck {
    /// Reachable; `warning` is a detail to show as `! …` (e.g. "model not pulled yet").
    Ready { warning: Option<String> },
    /// Not reachable: show `Model backend not reachable (<provider>): <detail>`
    /// and the hint `Check your config: qwenodyssey config list`; exit 1.
    Down { provider: String, detail: String },
}

pub async fn check_backend(provider: &dyn Provider) -> BackendCheck {
    let health = provider.health_check().await;
    if health.ok {
        BackendCheck::Ready { warning: health.detail.filter(|d| !d.is_empty()) }
    } else {
        BackendCheck::Down { provider: provider.name().to_string(), detail: health.detail.unwrap_or_default() }
    }
}

/* ── code ── */

#[derive(Clone, Debug, PartialEq)]
pub struct CodeCommandResult {
    pub run: RunResult,
    /// Set when edits were generated but not applied and were stashed for
    /// `apply` (show "Pending edits saved — run `qwenodyssey apply` to write them.").
    pub pending_saved: Option<PathBuf>,
}

/// `qwenodyssey code "<task>"` after the backend check: run the pipeline (or
/// autofix) and stash unapplied edits. Render `run.summary` under
/// "── Summary ──" and the logger's file as `Log: …`.
pub async fn code_command(orch: &Orchestrator, task: &str, mode: AgentMode, auto_confirm: bool) -> anyhow::Result<CodeCommandResult> {
    let run = if mode == AgentMode::Autofix {
        orch.run_autofix(task).await?
    } else {
        orch.run_code(task, RunOptions { mode, auto_confirm }).await?
    };
    // If edits were generated but not applied, stash them so `apply` can finish.
    let pending_saved =
        if !run.applied && !run.edits.is_empty() { Some(save_pending(orch.cwd(), &run.edits)?) } else { None };
    Ok(CodeCommandResult { run, pending_saved })
}

/* ── plan ── */

/// `qwenodyssey plan "<goal>"`: scan, list files, plan. Render with
/// [`crate::render::render_plan_command`].
pub async fn plan_command(
    provider: &dyn Provider,
    tools: &ToolRegistry,
    cwd: &Path,
    goal: &str,
    ui: &dyn PipelineUi,
) -> anyhow::Result<Plan> {
    let repo = scan_repo(cwd).await;
    let listed = tools.run("list_files", json!({ "pattern": "**/*" })).await;
    let files: Vec<String> = listed
        .data
        .as_ref()
        .and_then(|d| d.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    ui.step("Planning…");
    crate::agents::planner::plan(provider, goal, &repo, &files, tools.context().cancel.as_ref()).await
}

/* ── edit ── */

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditCommandOutcome {
    /// The file does not exist (show `Not found: <file>`; exit 1).
    NotFound(String),
    /// The model produced no edit (show "No edit produced.").
    NoEdit,
    /// The user declined (show "Not applied.").
    NotApplied,
    /// Applied: show `✓ <mode> <path>` / `✗ <path>: <error>` per entry.
    Applied(ApplyOutcome),
}

/// Everything the `edit` command needs from the session.
pub struct EditSession<'a> {
    pub cwd: &'a Path,
    pub provider: &'a dyn Provider,
    pub config: &'a Config,
    pub logger: &'a Logger,
    pub ui: &'a dyn PipelineUi,
    pub auto_confirm: bool,
    pub cancel: Option<&'a CancellationToken>,
}

/// `qwenodyssey edit <file> "<instruction>"`: one-file edit through the JSON
/// coder, diff preview, confirm, apply.
pub async fn edit_command(s: &EditSession<'_>, file: &str, instruction: &str) -> anyhow::Result<EditCommandOutcome> {
    let exists = resolve_inside(s.cwd, file).map(|abs| abs.exists()).unwrap_or(false);
    if !exists {
        return Ok(EditCommandOutcome::NotFound(file.to_string()));
    }
    if let Err(e) = s.logger.start(&format!("edit {file}: {instruction}")) {
        s.ui.debug(&format!("session log unavailable: {e}"));
    }
    let repo = scan_repo(s.cwd).await;
    let task = format!("Edit the file \"{file}\". Instruction: {instruction}");
    let files = [file.to_string()];
    let context = build_context(
        s.provider,
        &ContextInput {
            user_request: &task,
            repo: &repo,
            files: &files,
            errors: None,
            git_diff: None,
            memory: None,
            budget_tokens: (s.config.model.context_tokens as f64 * 0.6).floor() as usize,
        },
    );
    s.ui.step("Generating edit…");
    let plan = Plan {
        goal: task.clone(),
        steps: vec![PlanStep { index: 1, title: instruction.to_string(), detail: None, files: Vec::new() }],
        files: vec![file.to_string()],
        assumptions: Vec::new(),
    };
    let out = coder::code(s.provider, &task, &plan, &context, s.cancel).await?;
    if out.edits.is_empty() {
        return Ok(EditCommandOutcome::NoEdit);
    }
    for e in &out.edits {
        // The edit command's header has no rationale (unlike the pipeline's).
        let bare = FileEdit { rationale: None, ..e.clone() };
        s.ui.show_edit_diff(&bare, &preview_edit_or_note(s.cwd, e));
    }
    let go = s.auto_confirm || s.config.agent.auto_apply || s.ui.ask_yes_no("Apply?", true).await;
    if !go {
        return Ok(EditCommandOutcome::NotApplied);
    }
    Ok(EditCommandOutcome::Applied(apply_edits(s.cwd, &out.edits)?))
}

/* ── apply ── */

#[derive(Serialize, Deserialize)]
struct Pending {
    edits: Vec<FileEdit>,
}

/// `.qwenodyssey/cache/pending.json`.
pub fn pending_file(cwd: &Path) -> PathBuf {
    workspace_path(cwd).join("cache").join("pending.json")
}

/// Stash generated-but-unapplied edits for `apply`. Returns the file written.
pub fn save_pending(cwd: &Path, edits: &[FileEdit]) -> anyhow::Result<PathBuf> {
    let f = pending_file(cwd);
    if let Some(dir) = f.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&f, serde_json::to_string_pretty(&Pending { edits: edits.to_vec() })?)?;
    Ok(f)
}

/// `qwenodyssey apply`: write the pending edits and remove the stash.
/// `None` = nothing pending (show "No pending edits. Generate some with
/// `qwenodyssey code "…"` first.").
pub fn apply_pending(cwd: &Path) -> anyhow::Result<Option<ApplyOutcome>> {
    let f = pending_file(cwd);
    if !f.exists() {
        return Ok(None);
    }
    let pending: Pending = serde_json::from_str(&fs::read_to_string(&f)?)?;
    let outcome = apply_edits(cwd, &pending.edits)?;
    fs::remove_file(&f)?;
    Ok(Some(outcome))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RollbackOutcome {
    pub journal: PathBuf,
    pub restored: Vec<String>,
}

/// `qwenodyssey apply --rollback`: restore the latest patch journal. `None` =
/// no journal (show "No patch journal to roll back."); otherwise show
/// `✓ Rolled back N file(s) from <journal file name>` and `  - <path>` lines.
pub fn rollback_latest(cwd: &Path) -> anyhow::Result<Option<RollbackOutcome>> {
    let Some(journal) = latest_journal(cwd) else { return Ok(None) };
    let restored = rollback(cwd, &journal)?;
    Ok(Some(RollbackOutcome { journal, restored }))
}

/* ── review ── */

#[derive(Clone, Debug, PartialEq)]
pub enum ReviewCommandOutcome {
    /// `git_diff` failed (e.g. not a repo): show the message in yellow.
    Unavailable(String),
    /// Show "No uncommitted changes to review." (gray).
    NoChanges,
    /// The reviewer's reply was not JSON: print it raw.
    Raw(String),
    /// Render with [`crate::render::render_review`].
    Review(ReviewResult),
}

/// Characters of the diff sent to the reviewer.
const REVIEW_DIFF_CHARS: usize = 12_000;

/// `qwenodyssey review`: review the current uncommitted git diff.
pub async fn review_command(provider: &dyn Provider, tools: &ToolRegistry, ui: &dyn PipelineUi) -> anyhow::Result<ReviewCommandOutcome> {
    let diff_res = tools.run("git_diff", json!({})).await;
    if !diff_res.ok {
        return Ok(ReviewCommandOutcome::Unavailable(diff_res.output));
    }
    let diff = diff_res.output;
    if diff.is_empty() || diff == "(no changes)" {
        return Ok(ReviewCommandOutcome::NoChanges);
    }
    ui.step("Reviewing current diff…");
    let clipped: String = diff.chars().take(REVIEW_DIFF_CHARS).collect();
    let edits = format!("```diff\n{clipped}\n```");
    let prompt = render(&load_prompt("reviewer")?, &[("task", "Review the current uncommitted changes."), ("edits", &edits)]);
    let text = crate::agents::ask(provider, prompt, true, 0.1, None, tools.context().cancel.as_ref()).await?;
    Ok(match extract_json(&text) {
        Some(parsed) => ReviewCommandOutcome::Review(reviewer::review_from_json(&parsed)),
        None => ReviewCommandOutcome::Raw(text),
    })
}

/* ── test ── */

#[derive(Clone, Debug, PartialEq)]
pub enum TestCommandOutcome {
    /// Show "No test command detected for this project." and
    /// `Detected languages: …` (gray).
    NoTestCommand { languages: Vec<String> },
    /// Print `result.output`, then `✓ Tests passed.` or `✗ Tests failed.` (exit 1).
    Ran(TestResult),
}

/// `qwenodyssey test`: run the detected test command via `run_shell`.
pub async fn test_command(tools: &ToolRegistry, cwd: &Path, ui: &dyn PipelineUi) -> TestCommandOutcome {
    let repo = scan_repo(cwd).await;
    let Some(cmd) = repo.test_command.clone().filter(|c| !c.is_empty()) else {
        return TestCommandOutcome::NoTestCommand { languages: repo.languages };
    };
    ui.step(&format!("Running: {cmd}"));
    TestCommandOutcome::Ran(tester::run_tests(tools, &repo, None).await)
}

/* ── memory ── */

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryCommandOutcome {
    /// Show `✓ Remembered in [<category>].`
    Added(MemoryCategory),
    /// Show the hits one per line, or "(no matches)" (gray) when empty.
    Hits(Vec<String>),
    /// Show `\n## <category>` (bold) + content per entry, or "(memory is empty)".
    Listing(Vec<MemoryEntry>),
    /// Show "✓ Memory cleared."
    Cleared,
    /// A usage / input error: show in red and exit 1.
    Usage(String),
}

/// `qwenodyssey memory [add|search|list|clear] [text] [--category c]`.
pub fn memory_command(
    store: &MemoryStore,
    action: Option<&str>,
    text: Option<&str>,
    category: Option<&str>,
) -> anyhow::Result<MemoryCommandOutcome> {
    let category = match category.filter(|c| !c.is_empty()) {
        None => MemoryCategory::Project,
        Some(c) => match c.parse::<MemoryCategory>() {
            Ok(c) => c,
            Err(e) => return Ok(MemoryCommandOutcome::Usage(e.to_string())),
        },
    };
    let text = text.filter(|t| !t.is_empty());
    Ok(match action {
        Some("add") => match text {
            None => MemoryCommandOutcome::Usage("Usage: qwenodyssey memory add \"fact\"".into()),
            Some(t) => {
                store.add(t, category)?;
                MemoryCommandOutcome::Added(category)
            }
        },
        Some("search") => match text {
            None => MemoryCommandOutcome::Usage("Usage: qwenodyssey memory search \"query\"".into()),
            Some(q) => MemoryCommandOutcome::Hits(store.search(q)),
        },
        Some("list") | None => MemoryCommandOutcome::Listing(store.list()),
        Some("clear") => {
            store.clear()?;
            MemoryCommandOutcome::Cleared
        }
        Some(other) => MemoryCommandOutcome::Usage(format!("Unknown memory action: {other}. Use add|search|list|clear.")),
    })
}

/* ── init ── */

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitOutcome {
    /// Show "<config> already exists. Use --force to overwrite." (yellow).
    AlreadyExists(PathBuf),
    /// Show "✓ Initialized Qwenodyssey workspace", `  Config: <path>` and the
    /// next-steps hints (see [`INIT_NEXT_STEPS`]).
    Initialized(PathBuf),
}

/// The hints `init` prints under "Next steps:" (gray).
pub const INIT_NEXT_STEPS: &[&str] = &[
    "  qwenodyssey config set model.provider ollama",
    "  qwenodyssey config set model.model qwen2.5:7b",
    "  qwenodyssey code \"your first task\"",
];

/// `qwenodyssey init [--force]`: workspace layout, default config, seeded memory files.
pub fn init_workspace(cwd: &Path, force: bool) -> anyhow::Result<InitOutcome> {
    let ws = workspace_path(cwd);
    let cfg_file = config_path(cwd);
    if cfg_file.exists() && !force {
        return Ok(InitOutcome::AlreadyExists(cfg_file));
    }
    for sub in ["memory", "patches", "logs", "cache"] {
        fs::create_dir_all(ws.join(sub))?;
    }
    save_config(cwd, &default_config())?;
    let mem_dir = ws.join("memory");
    for (f, header) in [
        ("project.md", "# Project memory\n"),
        ("user.md", "# User preferences\n"),
        ("decisions.md", "# Architecture decisions\n"),
        ("commands.md", "# Useful commands\n"),
    ] {
        let fp = mem_dir.join(f);
        if !fp.exists() {
            fs::write(&fp, header)?;
        }
    }
    Ok(InitOutcome::Initialized(cfg_file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{registry_with, scratch_repo, FakeProvider, FakeTool};
    use crate::ui::RecordingUi;
    use qo_core::{EditMode, ToolResult};

    #[tokio::test]
    async fn backend_check() {
        let mut p = FakeProvider::new(vec![]);
        assert_eq!(check_backend(&p).await, BackendCheck::Ready { warning: None });
        p.health = qo_core::HealthStatus { ok: false, detail: Some("connection refused".into()) };
        assert_eq!(
            check_backend(&p).await,
            BackendCheck::Down { provider: "fake".into(), detail: "connection refused".into() }
        );
    }

    #[test]
    fn pending_round_trip_and_rollback() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(apply_pending(dir.path()).unwrap(), None);
        assert_eq!(rollback_latest(dir.path()).unwrap(), None);
        let edits = [FileEdit { path: "p.txt".into(), mode: EditMode::Create, diff: None, content: Some("p".into()), rationale: None }];
        let f = save_pending(dir.path(), &edits).unwrap();
        assert!(f.ends_with("pending.json"));
        let out = apply_pending(dir.path()).unwrap().unwrap();
        assert!(out.all_ok);
        assert!(!f.exists(), "the stash is consumed");
        assert_eq!(fs::read_to_string(dir.path().join("p.txt")).unwrap(), "p");
        let rb = rollback_latest(dir.path()).unwrap().unwrap();
        assert_eq!(rb.restored, vec!["p.txt"]);
        assert!(!dir.path().join("p.txt").exists());
    }

    #[test]
    fn memory_actions() {
        let dir = tempfile::tempdir().unwrap();
        let m = MemoryStore::new(dir.path().join("mem"));
        assert_eq!(memory_command(&m, None, None, None).unwrap(), MemoryCommandOutcome::Listing(vec![]));
        assert!(matches!(memory_command(&m, Some("add"), None, None).unwrap(), MemoryCommandOutcome::Usage(_)));
        assert_eq!(
            memory_command(&m, Some("add"), Some("uses pnpm"), Some("commands")).unwrap(),
            MemoryCommandOutcome::Added(MemoryCategory::Commands)
        );
        let MemoryCommandOutcome::Hits(h) = memory_command(&m, Some("search"), Some("pnpm"), None).unwrap() else {
            panic!("hits")
        };
        assert_eq!(h.len(), 1);
        assert!(matches!(memory_command(&m, Some("add"), Some("x"), Some("bogus")).unwrap(), MemoryCommandOutcome::Usage(_)));
        assert!(matches!(memory_command(&m, Some("frob"), None, None).unwrap(), MemoryCommandOutcome::Usage(u) if u.contains("frob")));
        assert_eq!(memory_command(&m, Some("clear"), None, None).unwrap(), MemoryCommandOutcome::Cleared);
        assert_eq!(memory_command(&m, Some("list"), None, None).unwrap(), MemoryCommandOutcome::Listing(vec![]));
    }

    #[test]
    fn init_creates_layout_and_respects_force() {
        let dir = tempfile::tempdir().unwrap();
        let InitOutcome::Initialized(cfg) = init_workspace(dir.path(), false).unwrap() else { panic!("init") };
        assert!(cfg.exists());
        for sub in ["memory", "patches", "logs", "cache"] {
            assert!(dir.path().join(".qwenodyssey").join(sub).is_dir());
        }
        let user = dir.path().join(".qwenodyssey/memory/user.md");
        assert_eq!(fs::read_to_string(&user).unwrap(), "# User preferences\n");
        fs::write(&user, "# User preferences\n- (2026-01-01) tabs\n").unwrap();
        assert_eq!(init_workspace(dir.path(), false).unwrap(), InitOutcome::AlreadyExists(cfg.clone()));
        assert_eq!(init_workspace(dir.path(), true).unwrap(), InitOutcome::Initialized(cfg));
        assert!(fs::read_to_string(&user).unwrap().contains("tabs"), "existing memory is kept");
        assert!(qo_core::config::load_config(dir.path()).is_ok());
    }

    #[tokio::test]
    async fn review_outcomes() {
        let ui = RecordingUi::new(false);
        let p = FakeProvider::new(vec![r#"{"approve":false,"summary":"s","issues":[]}"#.into(), "plain words".into()]);
        let none = registry_with(std::env::temp_dir(), vec![FakeTool::new("git_diff", ToolResult::ok("(no changes)"))]);
        assert_eq!(review_command(&p, &none, &ui).await.unwrap(), ReviewCommandOutcome::NoChanges);
        let not_repo = registry_with(std::env::temp_dir(), vec![FakeTool::new("git_diff", ToolResult::err("Not a git repository."))]);
        assert_eq!(
            review_command(&p, &not_repo, &ui).await.unwrap(),
            ReviewCommandOutcome::Unavailable("Not a git repository.".into())
        );
        let diff = registry_with(std::env::temp_dir(), vec![FakeTool::new("git_diff", ToolResult::ok("-a\n+b"))]);
        let ReviewCommandOutcome::Review(r) = review_command(&p, &diff, &ui).await.unwrap() else { panic!("review") };
        assert!(!r.approve);
        assert_eq!(review_command(&p, &diff, &ui).await.unwrap(), ReviewCommandOutcome::Raw("plain words".into()));
        assert!(p.requests()[0].messages[1].content.contains("```diff\n-a\n+b\n```"));
    }

    #[tokio::test]
    async fn test_command_outcomes() {
        let ui = RecordingUi::new(false);
        let dir = scratch_repo();
        let reg = registry_with(dir.path(), vec![FakeTool::new("run_shell", ToolResult::ok("ok"))]);
        assert!(matches!(test_command(&reg, dir.path(), &ui).await, TestCommandOutcome::NoTestCommand { .. }));
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        let TestCommandOutcome::Ran(r) = test_command(&reg, dir.path(), &ui).await else { panic!("ran") };
        assert!(r.passed);
        assert!(ui.transcript().contains("Running: cargo test"));
    }

    #[tokio::test]
    async fn edit_command_flow() {
        let dir = scratch_repo();
        fs::write(dir.path().join("a.txt"), "old\n").unwrap();
        let reply = r#"{"edits":[{"path":"a.txt","mode":"rewrite","content":"new\n","rationale":"r"}]}"#;
        let p = FakeProvider::new(vec![reply.into(), reply.into()]);
        let cfg = Config::default();
        let logger = Logger::new(dir.path(), false);
        let ui = RecordingUi::new(false).with_answers([false]);
        let mut s = EditSession { cwd: dir.path(), provider: &p, config: &cfg, logger: &logger, ui: &ui, auto_confirm: false, cancel: None };
        assert_eq!(edit_command(&s, "missing.txt", "x").await.unwrap(), EditCommandOutcome::NotFound("missing.txt".into()));
        assert_eq!(edit_command(&s, "a.txt", "make it new").await.unwrap(), EditCommandOutcome::NotApplied);
        assert_eq!(ui.questions(), vec!["Apply?"]);
        assert!(ui.transcript().contains("REWRITE a.txt"));
        assert!(!ui.transcript().contains(" — r"), "edit headers carry no rationale");
        s.auto_confirm = true;
        let EditCommandOutcome::Applied(out) = edit_command(&s, "a.txt", "make it new").await.unwrap() else {
            panic!("applied")
        };
        assert!(out.all_ok);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "new\n");
        assert!(p.requests()[0].messages[1].content.contains("Edit the file \"a.txt\". Instruction: make it new"));
    }

    #[tokio::test]
    async fn code_command_stashes_unapplied_edits() {
        let dir = scratch_repo();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        let plan = r#"{"goal":"g","files":["a.py"],"steps":[{"title":"t"}]}"#;
        let edits = r#"{"edits":[{"path":"a.py","mode":"rewrite","content":"x = 2\n"}]}"#;
        let review = r#"{"approve":true}"#;
        let p = std::sync::Arc::new(FakeProvider::new(vec![plan.into(), edits.into(), review.into()]));
        let ui = std::sync::Arc::new(RecordingUi::new(false).with_answers([true, false]));
        let reg = registry_with(dir.path(), vec![FakeTool::new("list_files", ToolResult::ok(""))]);
        let mut cfg = Config::default();
        cfg.agent.edit_protocol = qo_core::config::EditProtocol::Json;
        let orch = Orchestrator::new(
            dir.path(),
            p,
            cfg,
            reg,
            Logger::new(dir.path(), false),
            MemoryStore::new(dir.path().join("mem")),
            ui.clone(),
        );
        let r = code_command(&orch, "t", AgentMode::Safe, false).await.unwrap();
        assert!(!r.run.applied);
        let f = r.pending_saved.unwrap();
        assert!(f.exists());
        assert_eq!(ui.questions(), vec!["Proceed with this plan?", "Apply these edits?"]);
        assert!(apply_pending(dir.path()).unwrap().unwrap().all_ok);
        assert_eq!(fs::read_to_string(dir.path().join("a.py")).unwrap(), "x = 2\n");
    }
}
