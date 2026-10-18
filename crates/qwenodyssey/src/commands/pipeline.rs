//! The pipeline commands — code, plan, edit, apply, review, test, memory —
//! (ports of src/cli/commands/{code,plan,edit,apply,review,test,memory}.ts).
//! The logic lives in qo-pipeline; these functions build the session, render
//! the outcome, and implement [`PipelineUi`] for the terminal.
//!
//! These commands do NOT start the chat's input owner or raw mode: the only
//! interactive moment is a yes/no question, read as one cooked line.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::Arc;

use async_trait::async_trait;
use qo_core::{AgentMode, FileEdit};
use qo_pipeline::commands::{self as pc, BackendCheck, EditCommandOutcome, MemoryCommandOutcome, ReviewCommandOutcome, TestCommandOutcome};
use qo_pipeline::patch_engine::ApplyOutcome;
use qo_pipeline::{Level, Orchestrator, PipelineUi};

use crate::cli::GlobalOpts;
use crate::outln;
use crate::render::render_markdown;
use crate::session::Session;
use crate::ui::style;

/// Terminal implementation of the pipeline's UI trait (the TS logger glyphs).
pub struct TermUi {
    verbose: bool,
}

impl TermUi {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

#[async_trait]
impl PipelineUi for TermUi {
    fn line(&self, level: Level, msg: &str) {
        match level {
            Level::Step => outln!("{}{msg}", style::cyan("→ ")),
            Level::Info => outln!("{msg}"),
            Level::Success => outln!("{}{msg}", style::green("✓ ")),
            Level::Warn => outln!("{}", style::yellow(&format!("! {msg}"))),
            Level::Error => outln!("{}", style::red(&format!("✗ {msg}"))),
            Level::Debug => {
                if self.verbose {
                    outln!("{}", style::gray(&format!("· {msg}")));
                }
            }
        }
    }

    fn block(&self, text: &str) {
        outln!("{text}");
    }

    fn show_edit_diff(&self, edit: &FileEdit, diff: &str) {
        outln!("\n{}", qo_pipeline::render::render_edit_header(edit));
        outln!("{}", crate::render::colorize_diff(diff));
    }

    async fn ask_yes_no(&self, question: &str, initial: bool) -> bool {
        ask_yes_no(question, initial).await
    }
}

/// A one-line yes/no question in cooked mode (the `prompts` confirm of 1.x).
pub async fn ask_yes_no(question: &str, initial: bool) -> bool {
    let hint = if initial { "(Y/n)" } else { "(y/N)" };
    crate::ui::out::print(&format!("{} {} {} ", style::green("?"), style::bold(question), style::gray(hint)));
    let line = tokio::task::spawn_blocking(|| {
        let mut s = String::new();
        match std::io::stdin().lock().read_line(&mut s) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(s),
        }
    })
    .await
    .ok()
    .flatten();
    if !std::io::stdin().is_terminal() {
        outln!();
    }
    let _ = std::io::stdout().flush();
    match line.map(|l| l.trim().to_lowercase()) {
        Some(a) if a.is_empty() => initial,
        Some(a) => matches!(a.as_str(), "y" | "yes"),
        None => initial,
    }
}

fn print_applied(outcome: &ApplyOutcome) {
    for a in &outcome.applied {
        if a.ok {
            outln!("{}", style::green(&format!("✓ {} {}", a.mode, a.path)));
        } else {
            outln!("{}", style::red(&format!("✗ {}: {}", a.path, a.error.clone().unwrap_or_default())));
        }
    }
}

/// `qwenodyssey code "<task>"`.
pub async fn code(opts: &GlobalOpts, task: &str, mode: Option<AgentMode>) -> anyhow::Result<i32> {
    if task.trim().is_empty() {
        outln!("{}", style::red("Usage: qwenodyssey code \"describe the task\""));
        return Ok(1);
    }
    let s = Session::create(opts, mode)?;
    match pc::check_backend(s.provider.as_ref()).await {
        BackendCheck::Down { provider, detail } => {
            outln!("{}", style::red(&format!("Model backend not reachable ({provider}): {detail}")));
            outln!("{}", style::gray("Check your config: qwenodyssey config list"));
            return Ok(1);
        }
        BackendCheck::Ready { warning: Some(w) } => outln!("{}", style::yellow(&format!("! {w}"))),
        BackendCheck::Ready { warning: None } => {}
    }
    let ui: Arc<dyn PipelineUi> = Arc::new(TermUi::new(opts.verbose));
    let orch = Orchestrator::new(&s.cwd, s.provider.clone(), s.config.clone(), s.default_tools(), s.logger.clone(), s.memory(), ui);
    let result = pc::code_command(&orch, task, s.mode, s.auto_confirm).await?;
    if result.pending_saved.is_some() {
        outln!("{}", style::gray("Pending edits saved — run `qwenodyssey apply` to write them."));
    }
    outln!("{}", style::bold("\n── Summary ──"));
    outln!("{}", render_markdown(&result.run.summary));
    if let Some(f) = s.logger.log_file() {
        outln!("{}", style::gray(&format!("\nLog: {}", f.display())));
    }
    Ok(0)
}

/// `qwenodyssey plan "<goal>"`.
pub async fn plan(opts: &GlobalOpts, goal: &str) -> anyhow::Result<i32> {
    if goal.trim().is_empty() {
        outln!("{}", style::red("Usage: qwenodyssey plan \"goal\""));
        return Ok(1);
    }
    let s = Session::create(opts, None)?;
    let ui = TermUi::new(opts.verbose);
    let plan = pc::plan_command(s.provider.as_ref(), &s.default_tools(), &s.cwd, goal, &ui).await?;
    outln!("{}", qo_pipeline::render::render_plan_command(&plan));
    Ok(0)
}

/// `qwenodyssey edit <file> "<instruction>"`.
pub async fn edit(opts: &GlobalOpts, file: &str, instruction: &str) -> anyhow::Result<i32> {
    if file.is_empty() || instruction.is_empty() {
        outln!("{}", style::red("Usage: qwenodyssey edit <file> \"instruction\""));
        return Ok(1);
    }
    let s = Session::create(opts, None)?;
    let ui = TermUi::new(opts.verbose);
    let es = pc::EditSession {
        cwd: &s.cwd,
        provider: s.provider.as_ref(),
        config: &s.config,
        logger: &s.logger,
        ui: &ui,
        auto_confirm: s.auto_confirm,
        cancel: None,
    };
    Ok(match pc::edit_command(&es, file, instruction).await? {
        EditCommandOutcome::NotFound(f) => {
            outln!("{}", style::red(&format!("Not found: {f}")));
            1
        }
        EditCommandOutcome::NoEdit => {
            outln!("{}", style::yellow("No edit produced."));
            0
        }
        EditCommandOutcome::NotApplied => {
            outln!("{}", style::gray("Not applied."));
            0
        }
        EditCommandOutcome::Applied(o) => {
            print_applied(&o);
            0
        }
    })
}

/// `qwenodyssey apply [--rollback]`.
pub fn apply(opts: &GlobalOpts, rollback: bool) -> anyhow::Result<i32> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    if rollback {
        match pc::rollback_latest(&cwd)? {
            None => outln!("{}", style::yellow("No patch journal to roll back.")),
            Some(r) => {
                let name = r.journal.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                outln!("{}", style::green(&format!("✓ Rolled back {} file(s) from {name}", r.restored.len())));
                for p in r.restored {
                    outln!("  - {p}");
                }
            }
        }
        return Ok(0);
    }
    match pc::apply_pending(&cwd)? {
        None => outln!("{}", style::yellow("No pending edits. Generate some with `qwenodyssey code \"…\"` first.")),
        Some(o) => print_applied(&o),
    }
    Ok(0)
}

/// `qwenodyssey review`.
pub async fn review(opts: &GlobalOpts) -> anyhow::Result<i32> {
    let s = Session::create(opts, None)?;
    let ui = TermUi::new(opts.verbose);
    match pc::review_command(s.provider.as_ref(), &s.default_tools(), &ui).await? {
        ReviewCommandOutcome::Unavailable(msg) => outln!("{}", style::yellow(&msg)),
        ReviewCommandOutcome::NoChanges => outln!("{}", style::gray("No uncommitted changes to review.")),
        ReviewCommandOutcome::Raw(text) => outln!("{text}"),
        ReviewCommandOutcome::Review(r) => outln!("{}", qo_pipeline::render::render_review(&r)),
    }
    Ok(0)
}

/// `qwenodyssey test`.
pub async fn test(opts: &GlobalOpts) -> anyhow::Result<i32> {
    let s = Session::create(opts, None)?;
    let ui = TermUi::new(opts.verbose);
    Ok(match pc::test_command(&s.default_tools(), &s.cwd, &ui).await {
        TestCommandOutcome::NoTestCommand { languages } => {
            outln!("{}", style::yellow("No test command detected for this project."));
            outln!("{}", style::gray(&format!("Detected languages: {}", languages.join(", "))));
            0
        }
        TestCommandOutcome::Ran(r) => {
            outln!("{}", r.output);
            if r.passed {
                outln!("{}", style::green("✓ Tests passed."));
                0
            } else {
                outln!("{}", style::red("✗ Tests failed."));
                1
            }
        }
    })
}

/// `qwenodyssey memory [add|search|list|clear] [text] [--category c]`.
pub fn memory(opts: &GlobalOpts, action: Option<&str>, text: Option<&str>, category: &str) -> anyhow::Result<i32> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    let config = qo_core::config::load_config(&cwd)?;
    let store = qo_pipeline::memory::MemoryStore::for_project(&cwd, &config.memory.path);
    Ok(match pc::memory_command(&store, action, text, Some(category))? {
        MemoryCommandOutcome::Added(c) => {
            outln!("{}", style::green(&format!("✓ Remembered in [{c}].")));
            0
        }
        MemoryCommandOutcome::Hits(h) => {
            outln!("{}", if h.is_empty() { style::gray("(no matches)") } else { h.join("\n") });
            0
        }
        MemoryCommandOutcome::Listing(all) => {
            if all.is_empty() {
                outln!("{}", style::gray("(memory is empty)"));
            }
            for e in all {
                outln!("{}", style::bold(&format!("\n## {}", e.category)));
                outln!("{}", e.content);
            }
            0
        }
        MemoryCommandOutcome::Cleared => {
            outln!("{}", style::green("✓ Memory cleared."));
            0
        }
        MemoryCommandOutcome::Usage(msg) => {
            outln!("{}", style::red(&msg));
            1
        }
    })
}
