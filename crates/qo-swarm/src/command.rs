//! `qwenodyssey swarm <task>` — run a complex task across a COORDINATED team of
//! frontier agents. A lead model splits the task into dependency-aware subtasks; the
//! agents share a blackboard (each sees its dependencies' results) and run in
//! parallel waves, streaming live into a split-pane TUI (one terminal per agent).
//! Finally a lead model integrates the board into one answer.
//!
//!   --plain      classic mode (ensemble, or --divide shards) with no coordination
//!   --no-live    coordinated, but log progress as plain lines (no TUI)
//!   --divide …   supply the subtasks yourself (skips auto-decomposition)
//!   --list       show the roster and exit (no models are called)
//!   --demo       rehearse the live TUI with fake streaming agents (no models called)
//!   --exec …     let agents run commands: off | auto | bare | daytona
//!
//! Roster is FRONTIER-FIRST: cloud models when keys exist, else local, else primary.
//!
//! The binary owns the terminal, so this module writes its line output to the
//! `out` writer it is handed (normally stdout); only the live TUI draws directly.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use crossterm::style::Stylize;
use qo_core::{
    CancellationToken, GenerateResult, HealthStatus, Message, ModelOptions, Provider, ProviderError,
};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::coordinator::{create_coordinated_swarm, CoordinateOptions, CoordinatedSwarm, Subtask};
use crate::env::SwarmEnv;
use crate::events::SwarmEvent;
use crate::exec::{BareExecutorOptions, ExecMode};
use crate::swarm::{
    frontier_workers, synthesize, RosterEntry, Swarm, SwarmOptions, SwarmRun, SwarmWorker, WorkerKind,
};
use crate::tui::{SwarmTui, SwarmTuiOptions, TuiControls};

/// Flags of the `swarm` subcommand.
#[derive(Clone, Debug)]
pub struct SwarmCommandOptions {
    /// The task (required unless `list` or `demo`).
    pub task: Option<String>,
    /// `--divide <subtasks...>`: supply the subtasks yourself.
    pub divide: Vec<String>,
    /// `--no-synth` → false: don't integrate a final answer.
    pub synth: bool,
    /// `--local` (plain mode): also enlist local fallback models.
    pub local: bool,
    /// `--list`: show the roster and exit (no models are called).
    pub list: bool,
    /// `--no-live` → false: plain line log instead of the TUI.
    pub live: bool,
    /// `--plain`: classic uncoordinated ensemble/divide.
    pub plain: bool,
    /// `--demo`: synthetic workers, no network.
    pub demo: bool,
    /// `--exec <mode>`: off | auto | bare | daytona (overrides `[swarm].exec`).
    pub exec: Option<String>,
    /// Working directory (bare-metal exec root, result file location, pane dashboards).
    pub cwd: PathBuf,
    /// Colour the line output (ANSI).
    pub color: bool,
}

impl Default for SwarmCommandOptions {
    fn default() -> Self {
        Self {
            task: None,
            divide: Vec::new(),
            synth: true,
            local: false,
            list: false,
            live: true,
            plain: false,
            demo: false,
            exec: None,
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            color: true,
        }
    }
}

/// What happened, for the binary's exit status.
#[derive(Clone, Debug, Default)]
pub struct SwarmCommandOutcome {
    /// 0 ok, 1 usage/config error, 130 aborted by Ctrl-C (plain path).
    pub exit_code: i32,
    pub run: Option<SwarmRun>,
    /// Where the full result was saved (TUI path).
    pub saved_to: Option<PathBuf>,
    pub aborted: bool,
}

/// ANSI styling that switches off cleanly.
#[derive(Clone, Copy)]
struct Pen {
    color: bool,
}

macro_rules! pen_fn {
    ($name:ident, $how:ident) => {
        fn $name(&self, s: impl AsRef<str>) -> String {
            let s = s.as_ref();
            if self.color {
                s.$how().to_string()
            } else {
                s.to_string()
            }
        }
    };
}

impl Pen {
    pen_fn!(red, red);
    pen_fn!(green, green);
    pen_fn!(yellow, yellow);
    pen_fn!(cyan, cyan);
    pen_fn!(gray, dark_grey);
    pen_fn!(bold, bold);
}

fn secs(ms: u64) -> String {
    format!("{:.1}", ms as f64 / 1000.0)
}

/// Run the `swarm` subcommand against an environment built from the config
/// (see `run_swarm_command` in compat for the `&Config` entry point).
pub async fn run_swarm_command_with_env(
    env: Arc<SwarmEnv>,
    opts: SwarmCommandOptions,
    out: &mut (dyn Write + Send),
) -> anyhow::Result<SwarmCommandOutcome> {
    let pen = Pen { color: opts.color };
    let task_text = opts.task.clone().unwrap_or_default().trim().to_string();
    if task_text.is_empty() && !opts.list && !opts.demo {
        writeln!(out, "{}", pen.red("A <task> is required (or use --list to just show the roster)."))?;
        return Ok(SwarmCommandOutcome { exit_code: 1, ..Default::default() });
    }

    // ── demo: synthetic streaming workers, zero network — verifies the TUI ──
    if opts.demo {
        return demo_swarm(env, &opts, out).await;
    }
    // ── classic (uncoordinated) path: --plain ──
    if opts.plain {
        return plain_swarm(env, &task_text, &opts, out).await;
    }

    // Where agents may run commands: CLI flag overrides config ([swarm] exec).
    let exec_mode = match opts.exec.as_deref() {
        None => env.swarm.exec,
        Some(s) => match ExecMode::parse(s) {
            Some(m) => m,
            None => {
                writeln!(out, "{}", pen.red(format!("Unknown --exec mode '{s}' (use off | auto | bare | daytona).")))?;
                return Ok(SwarmCommandOutcome { exit_code: 1, ..Default::default() });
            }
        },
    };

    // ── coordinated path (default) ──
    let Some(built) = create_coordinated_swarm(Arc::clone(&env), None) else {
        writeln!(
            out,
            "{}{}",
            pen.red("No swarm workers available."),
            pen.gray(" Add cloud models to model.fallback_models (nvidia:… / openrouter:…) and set the API key(s), or configure a local primary.")
        )?;
        return Ok(SwarmCommandOutcome { exit_code: 1, ..Default::default() });
    };
    let swarm = built.swarm;
    let note = built.local.then(|| "(local fallback — no frontier key configured)".to_string());

    let roster = swarm.roster();
    writeln!(out, "{}", pen.bold(format!("\n🜂 Coordinated swarm — {} agent(s):", roster.len())))?;
    for w in &roster {
        writeln!(out, "  • {} {}", pen.cyan(&w.model), pen.gray(format!("({})", w.backend)))?;
    }
    if let Some(n) = &note {
        writeln!(out, "  {}", pen.yellow(n))?;
    }
    writeln!(out)?;

    if opts.list {
        writeln!(out, "{}", pen.gray("(--list: roster only; no models were called.)"))?;
        return Ok(SwarmCommandOutcome::default());
    }

    // Explicit --divide subtasks skip auto-decomposition (still coordinated/streamed).
    let presupplied = supplied_subtasks(&opts.divide);
    let cancel = CancellationToken::new();
    let run_opts = CoordinateOptions {
        synthesize: opts.synth,
        subtasks: presupplied,
        cancel: Some(cancel.clone()),
        exec_mode,
        bare: BareExecutorOptions {
            cwd: Some(opts.cwd.clone()),
            allow_commands: env.allow_commands.clone(),
            deny_commands: env.deny_commands.clone(),
            default_timeout_ms: Some(env.swarm.exec_timeout_s * 1000),
        },
        ..Default::default()
    };

    let use_tui = opts.live && env.swarm.live && SwarmTui::supported();
    let started = Instant::now();

    if use_tui {
        // Ctrl-C/q are keys in raw mode → abort; Esc stops the focused agent; the
        // panes stay up at the end in review mode (scrollable) until the user presses q.
        let events = swarm.subscribe();
        let abort_token = cancel.clone();
        let stopper = Arc::clone(&swarm);
        let controls = TuiControls {
            on_abort: Some(Box::new(move || abort_token.cancel())),
            on_stop_pane: Some(Box::new(move |pane| stopper.stop_pane(pane))),
        };
        let tui_opts = SwarmTuiOptions {
            task: task_text.clone(),
            panes: roster.len(),
            note: note.clone(),
            roster: roster.clone(),
            cwd: opts.cwd.display().to_string(),
        };
        let (run, outcome) = SwarmTui::run(tui_opts, events, controls, swarm.run(&task_text, run_opts)).await;
        // No raw dump to the terminal — persist the result to a file instead.
        let file = save_result(&run, &task_text, &opts.cwd);
        if outcome.aborted {
            writeln!(out, "{}", pen.yellow("(swarm aborted)"))?;
        } else {
            let ok = run.results.iter().filter(|r| r.ok).count();
            let files = run.artifacts.as_ref().map_or(0, |a| a.files.len());
            writeln!(
                out,
                "{}{}",
                pen.green("✓ swarm complete"),
                pen.gray(format!(
                    " · {ok}/{} subtasks · {}s{}",
                    run.results.len(),
                    started.elapsed().as_secs(),
                    if files > 0 { format!(" · {files} files") } else { String::new() }
                ))
            )?;
            if let Some(f) = &file {
                writeln!(out, "{}{}", pen.gray("  full result + file list saved to "), pen.cyan(f.display().to_string()))?;
            }
        }
        return Ok(SwarmCommandOutcome {
            exit_code: 0,
            aborted: outcome.aborted,
            saved_to: file,
            run: Some(run),
        });
    }

    // Plain / non-TTY path: line logger + durable text summary; Ctrl-C aborts.
    let mut events = swarm.subscribe();
    let (run, aborted) = drive_plain(&swarm, &task_text, run_opts, &cancel, &mut events, pen, out).await?;
    if aborted {
        writeln!(out, "{}", pen.yellow("\n(swarm aborted by Ctrl-C)"))?;
    }
    if let Some(run) = &run {
        print_summary(run, started.elapsed(), pen, out)?;
    }
    Ok(SwarmCommandOutcome { exit_code: if aborted { 130 } else { 0 }, run, saved_to: None, aborted })
}

fn supplied_subtasks(divide: &[String]) -> Option<Vec<Subtask>> {
    let clean: Vec<&str> = divide.iter().map(|d| d.trim()).filter(|d| !d.is_empty()).collect();
    if clean.is_empty() {
        return None;
    }
    Some(clean.iter().enumerate().map(|(i, d)| Subtask::supplied(i, d)).collect())
}

/// Run the coordinated swarm while printing its events as lines. The first
/// Ctrl-C cancels the run (it winds down and still reports what finished); a
/// second one stops waiting (the run's cleanup continues in the background).
async fn drive_plain(
    swarm: &Arc<CoordinatedSwarm>,
    task: &str,
    run_opts: CoordinateOptions,
    cancel: &CancellationToken,
    events: &mut UnboundedReceiver<SwarmEvent>,
    pen: Pen,
    out: &mut (dyn Write + Send),
) -> anyhow::Result<(Option<SwarmRun>, bool)> {
    let run_fut = swarm.run(task, run_opts);
    tokio::pin!(run_fut);
    let mut aborted = false;
    loop {
        tokio::select! {
            run = &mut run_fut => {
                while let Ok(ev) = events.try_recv() {
                    log_event(&ev, pen, out)?;
                }
                return Ok((Some(run), aborted));
            }
            Some(ev) = events.recv() => log_event(&ev, pen, out)?,
            r = tokio::signal::ctrl_c() => {
                if r.is_err() {
                    // No signal support here: just wait for the run.
                    let run = (&mut run_fut).await;
                    while let Ok(ev) = events.try_recv() {
                        log_event(&ev, pen, out)?;
                    }
                    return Ok((Some(run), aborted));
                }
                if aborted {
                    return Ok((None, true));
                }
                aborted = true;
                cancel.cancel();
            }
        }
    }
}

/// Plain line logger for --no-live / non-TTY: announce preflight, plan, each subtask result.
fn log_event(ev: &SwarmEvent, pen: Pen, out: &mut (dyn Write + Send)) -> std::io::Result<()> {
    match ev {
        SwarmEvent::PreflightStart { count } => {
            writeln!(out, "{}", pen.gray(format!("Checking {count} agent key(s)…")))?;
        }
        SwarmEvent::Preflight(r) => {
            if r.ok {
                writeln!(out, "{}", pen.gray(format!("  ✓ {} ({}ms)", r.model, r.ms)))?;
            } else {
                writeln!(out, "{}", pen.red(format!("  ✗ {} — {}", r.model, r.detail.as_deref().unwrap_or("unreachable"))))?;
            }
        }
        SwarmEvent::PreflightDone { ok, broken } if *broken > 0 => {
            let msg = if *ok > 0 {
                format!("  {broken} broken key(s) dropped; {ok} agent(s) continue")
            } else {
                "  every agent failed its key check".to_string()
            };
            writeln!(out, "{}", pen.yellow(msg))?;
        }
        SwarmEvent::Plan(e) => {
            let by = match e.planned_by.as_deref() {
                Some(b) if b != "(supplied)" => pen.gray(format!(" (planned by {b})")),
                _ => String::new(),
            };
            writeln!(out, "{}{by}:", pen.bold(format!("\nPlan — {} subtask(s)", e.subtasks.len())))?;
            for st in &e.subtasks {
                let deps = if st.depends_on.is_empty() {
                    String::new()
                } else {
                    pen.gray(format!(" ← {}", st.depends_on.join(", ")))
                };
                writeln!(out, "  {} {}{deps}", pen.cyan(format!("[{}]", st.id)), st.title)?;
            }
            if let Some(x) = &e.exec {
                let c = e.complexity.map_or("?", |c| c.as_str());
                writeln!(out, "{}", pen.gray(format!("  complexity: {c} · exec: {x}")))?;
            }
            if let Some(n) = &e.note {
                writeln!(out, "{}", pen.yellow(format!("  ⚠ {n}")))?;
            }
            writeln!(out)?;
        }
        SwarmEvent::ExecCmd(e) => {
            let cmd: String = e.command.chars().take(100).collect();
            writeln!(out, "{}", pen.gray(format!("  $ [{}] {cmd}{}", e.subtask_id, if e.ok { "" } else { "  (failed)" })))?;
        }
        SwarmEvent::Assign(e) => writeln!(out, "{}", pen.gray(format!("▸ {} → [{}] {}", e.model, e.subtask_id, e.title)))?,
        SwarmEvent::Status(e) => match e.status {
            crate::coordinator::SubtaskStatus::Done => {
                writeln!(out, "{}", pen.green(format!("  ✓ [{}] ({}s)", e.subtask_id, secs(e.ms.unwrap_or(0)))))?
            }
            crate::coordinator::SubtaskStatus::Failed => {
                writeln!(out, "{}", pen.red(format!("  ✗ [{}] {}", e.subtask_id, e.error.as_deref().unwrap_or(""))))?
            }
            crate::coordinator::SubtaskStatus::Stopped => {
                writeln!(out, "{}", pen.yellow(format!("  ■ [{}] stopped", e.subtask_id)))?
            }
            _ => {}
        },
        SwarmEvent::Synth { model } => writeln!(out, "{}", pen.gray(format!("\nSynthesizing with {model}…")))?,
        _ => {}
    }
    out.flush()
}

fn print_summary(run: &SwarmRun, elapsed: Duration, pen: Pen, out: &mut (dyn Write + Send)) -> std::io::Result<()> {
    writeln!(out)?;
    for r in &run.results {
        let head = format!(
            "{} {}",
            pen.bold(format!("[{}] {}", r.id.as_deref().unwrap_or("?"), r.title.as_deref().unwrap_or(&r.model))),
            pen.gray(format!("{} · {}s", r.model, secs(r.ms)))
        );
        if r.ok {
            writeln!(out, "{}{head}", pen.green("✓ "))?;
            writeln!(out, "{}\n", r.text)?;
        } else {
            writeln!(out, "{}{head}", pen.red("✗ "))?;
            writeln!(out, "{}\n", pen.red(format!("  {}", r.error.as_deref().unwrap_or(""))))?;
        }
    }
    if let Some(s) = &run.synthesis {
        let by = run.synthesized_by.as_deref().map(|m| format!(" (by {m})")).unwrap_or_default();
        writeln!(out, "{}\n{s}\n", pen.bold(format!("═══ Integrated answer{by} ═══")))?;
    }
    writeln!(out, "{}", pen.gray(format!("(swarm finished in {:.1}s)", elapsed.as_secs_f64())))
}

/// Persist the final result + per-agent details + generated-file list to a
/// Markdown file in `cwd` (`swarm-result-<iso timestamp>.md`).
pub fn save_result(run: &SwarmRun, task: &str, cwd: &Path) -> Option<PathBuf> {
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S-%3fZ");
    let file = cwd.join(format!("swarm-result-{ts}.md"));
    std::fs::write(&file, render_result_markdown(run, task)).ok()?;
    Some(file)
}

/// The Markdown body [`save_result`] writes.
pub fn render_result_markdown(run: &SwarmRun, task: &str) -> String {
    let mut parts: Vec<String> = vec![format!("# Swarm result\n\n**Task:** {task}\n")];
    if let Some(s) = &run.synthesis {
        parts.push(format!("\n## Integrated result\n\n{s}\n"));
    }
    if let Some(a) = &run.artifacts {
        parts.push(format!("\n## Files generated\n\nLocation: {}\n", a.location));
        parts.push(if a.files.is_empty() {
            "_(none detected)_".to_string()
        } else {
            a.files.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n")
        });
        parts.push(String::new());
    }
    parts.push("\n---\n\n## Per-agent results\n".to_string());
    for r in &run.results {
        let head = format!(
            "### [{}] {} — {} {}s{}",
            r.id.as_deref().unwrap_or("?"),
            r.title.as_deref().unwrap_or(""),
            r.model,
            secs(r.ms),
            if r.ok { "" } else { " (FAILED)" }
        );
        let body = if r.ok { r.text.clone() } else { format!("error: {}", r.error.as_deref().unwrap_or("")) };
        parts.push(format!("\n{head}\n\n{body}\n"));
    }
    parts.join("\n")
}

/* ── demo mode: fake agents streaming through the REAL coordinator + TUI ── */

const DEMO_TEXTS: [&str; 4] = [
    "Surveying the landscape: enumerating the key requirements, constraints, and edge cases this design must satisfy before anything is built. ",
    "Drafting the core architecture: components, interfaces between them, and the data that flows across each boundary, with failure modes called out. ",
    "Writing concrete examples: a quick-start snippet, a typical production configuration, and one advanced scenario exercising the tricky paths. ",
    "Integrating the team's results into one coherent deliverable, reconciling overlaps and making the terminology consistent throughout. ",
];

/// A canned, word-by-word streaming provider (no network) for `--demo`.
pub struct DemoProvider {
    model: String,
    text: String,
    word_delay: Duration,
}

impl DemoProvider {
    pub fn new(model: &str, text: &str) -> Self {
        Self { model: model.to_string(), text: text.to_string(), word_delay: Duration::from_millis(55) }
    }
}

#[async_trait]
impl Provider for DemoProvider {
    fn name(&self) -> &str {
        "demo"
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn set_model(&self, _model: &str) {}

    async fn generate(&self, _messages: &[Message], _opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult { text: self.text.clone(), model: self.model.clone(), ..Default::default() })
    }

    async fn stream(
        &self,
        _messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        // Word-by-word so the panes visibly stream, ~8s per agent.
        let full = self.text.repeat(3);
        let never = CancellationToken::new();
        let cancel = opts.cancel.as_ref().unwrap_or(&never);
        for word in full.split_inclusive(char::is_whitespace) {
            tokio::select! {
                _ = tokio::time::sleep(self.word_delay) => {}
                _ = cancel.cancelled() => return Err(ProviderError::Interrupted.into()),
            }
            on_chunk(word);
        }
        Ok(GenerateResult { text: full, model: self.model.clone(), ..Default::default() })
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus { ok: true, detail: None }
    }
}

/// No-network rehearsal: 4 fake agents run the real plan→waves→synthesis flow.
async fn demo_swarm(
    env: Arc<SwarmEnv>,
    opts: &SwarmCommandOptions,
    out: &mut (dyn Write + Send),
) -> anyhow::Result<SwarmCommandOutcome> {
    let pen = Pen { color: opts.color };
    let models = ["demo/kimi", "demo/nemotron", "demo/llama", "demo/deepseek"];
    let workers: Vec<SwarmWorker> = models
        .iter()
        .enumerate()
        .map(|(i, m)| SwarmWorker {
            model_ref: m.to_string(),
            kind: WorkerKind::Ollama,
            model: m.to_string(),
            label: m[5..].to_string(),
            provider: Arc::new(DemoProvider::new(m, DEMO_TEXTS[i % DEMO_TEXTS.len()])),
        })
        .collect();
    let roster: Vec<RosterEntry> = workers
        .iter()
        .map(|w| RosterEntry { label: w.label.clone(), model: w.model.clone(), backend: w.kind.as_str().to_string() })
        .collect();
    let lead = Arc::new(DemoProvider::new("demo/lead", DEMO_TEXTS[3]));
    let swarm = Arc::new(CoordinatedSwarm::new(env, workers, lead, Some(400)));

    let st = |id: &str, title: &str, deps: &[&str]| Subtask {
        id: id.to_string(),
        title: title.to_string(),
        detail: "demo".to_string(),
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
    };
    let subtasks = vec![
        st("requirements", "Gather requirements", &[]),
        st("architecture", "Draft architecture", &[]),
        st("examples", "Write examples", &[]),
        st("integrate", "Integrate everything", &["requirements", "architecture", "examples"]),
    ];
    let task = "DEMO — four fake agents rehearsing the coordinated swarm (no models called)";
    writeln!(out, "{}", pen.bold("\n🜂 Swarm demo — no models are called; this rehearses the live TUI.\n"))?;

    let cancel = CancellationToken::new();
    let run_opts = CoordinateOptions {
        synthesize: true,
        subtasks: Some(subtasks),
        cancel: Some(cancel.clone()),
        ..Default::default()
    };

    if opts.live && SwarmTui::supported() {
        let events = swarm.subscribe();
        let abort_token = cancel.clone();
        let stopper = Arc::clone(&swarm);
        let controls = TuiControls {
            on_abort: Some(Box::new(move || abort_token.cancel())),
            on_stop_pane: Some(Box::new(move |pane| stopper.stop_pane(pane))),
        };
        let tui_opts = SwarmTuiOptions {
            task: task.to_string(),
            panes: roster.len(),
            note: Some("demo mode (no tokens spent)".to_string()),
            roster,
            cwd: opts.cwd.display().to_string(),
        };
        let (run, outcome) = SwarmTui::run(tui_opts, events, controls, swarm.run(task, run_opts)).await;
        writeln!(
            out,
            "{}{}",
            pen.green("✓ demo finished."),
            pen.gray(" 4 agent panes + a results pane; Tab/↑↓ scroll, Esc stops an agent, q to quit.")
        )?;
        return Ok(SwarmCommandOutcome { exit_code: 0, run: Some(run), saved_to: None, aborted: outcome.aborted });
    }

    let mut events = swarm.subscribe();
    let (run, aborted) = drive_plain(&swarm, task, run_opts, &cancel, &mut events, pen, out).await?;
    writeln!(
        out,
        "{}",
        pen.yellow("\n(no TTY here — ran with the plain logger; run from a real terminal to see the panes)")
    )?;
    Ok(SwarmCommandOutcome { exit_code: if aborted { 130 } else { 0 }, run, saved_to: None, aborted })
}

/* ── classic uncoordinated swarm (--plain): preserved from the original command ── */

async fn plain_swarm(
    env: Arc<SwarmEnv>,
    task_text: &str,
    opts: &SwarmCommandOptions,
    out: &mut (dyn Write + Send),
) -> anyhow::Result<SwarmCommandOutcome> {
    let pen = Pen { color: opts.color };
    let cancel = CancellationToken::new();
    let sw_opts = SwarmOptions {
        max_workers: Some(env.swarm.max_workers),
        include_local: opts.local || env.swarm.include_local,
        max_tokens: Some(env.swarm.max_tokens),
        cancel: Some(cancel.clone()),
    };
    let workers = frontier_workers(&env, &sw_opts);
    if workers.is_empty() {
        writeln!(
            out,
            "{}{}",
            pen.red("No frontier workers available."),
            pen.gray(" Add cloud models to model.fallback_models (nvidia:… / openrouter:…) and set the API key(s).")
        )?;
        return Ok(SwarmCommandOutcome { exit_code: 1, ..Default::default() });
    }
    writeln!(out, "{}", pen.bold(format!("\n🜂 Agent swarm (plain) — {} model(s) in parallel:", workers.len())))?;
    for w in &workers {
        writeln!(out, "  • {} {}", pen.cyan(&w.model), pen.gray(format!("({})", w.kind)))?;
    }
    writeln!(out)?;
    if opts.list {
        writeln!(out, "{}", pen.gray("(--list: roster only; no models were called.)"))?;
        return Ok(SwarmCommandOutcome::default());
    }

    let swarm = Swarm::new(workers, sw_opts);
    let divide: Vec<String> = opts.divide.iter().map(|d| d.trim().to_string()).filter(|d| !d.is_empty()).collect();
    let started = Instant::now();
    let work = async {
        if divide.is_empty() {
            swarm.ensemble(task_text, None).await
        } else {
            swarm.divide(&divide, Some(task_text)).await
        }
    };
    let (run, aborted) = until_ctrl_c(work, &cancel).await;
    let Some(mut run) = run else {
        writeln!(out, "{}", pen.yellow("\n(swarm aborted by Ctrl-C)"))?;
        return Ok(SwarmCommandOutcome { exit_code: 130, aborted: true, ..Default::default() });
    };

    for r in &run.results {
        let head = format!("{} {}", pen.bold(&r.model), pen.gray(format!("{}s", secs(r.ms))));
        writeln!(out, "{}{head}", if r.ok { pen.green("✓ ") } else { pen.red("✗ ") })?;
        if run.mode == crate::swarm::SwarmMode::Divide {
            writeln!(out, "{}", pen.gray(format!("  subtask: {}", r.task)))?;
        }
        if r.ok {
            writeln!(out, "{}\n", r.text)?;
        } else {
            writeln!(out, "{}\n", pen.red(format!("  {}", r.error.as_deref().unwrap_or(""))))?;
        }
    }

    if opts.synth && !aborted && run.results.iter().any(|r| r.ok) {
        let lead = env.factory.primary();
        writeln!(out, "{}", pen.gray(format!("Synthesizing best answer with {}…\n", lead.model())))?;
        let synth = synthesize(lead.as_ref(), task_text, &run, env.swarm.max_tokens, Some(&cancel));
        let (answer, _) = until_ctrl_c(synth, &cancel).await;
        if let Some(answer) = answer {
            writeln!(out, "{}\n{answer}\n", pen.bold("═══ Synthesized answer ═══"))?;
            run.synthesis = Some(answer);
            run.synthesized_by = Some(lead.model());
        }
    }
    if aborted {
        writeln!(out, "{}", pen.yellow("\n(swarm aborted by Ctrl-C)"))?;
    }
    writeln!(out, "{}", pen.gray(format!("(swarm finished in {:.1}s)", started.elapsed().as_secs_f64())))?;
    Ok(SwarmCommandOutcome { exit_code: if aborted { 130 } else { 0 }, run: Some(run), saved_to: None, aborted })
}

/// Await `fut`; the first Ctrl-C cancels `cancel` (the work winds down and its
/// partial result is returned), a second stops waiting (`None`).
async fn until_ctrl_c<T, F: std::future::Future<Output = T>>(fut: F, cancel: &CancellationToken) -> (Option<T>, bool) {
    tokio::pin!(fut);
    let mut aborted = cancel.is_cancelled();
    loop {
        tokio::select! {
            v = &mut fut => return (Some(v), aborted),
            r = tokio::signal::ctrl_c() => {
                if r.is_err() {
                    return ((&mut fut).await.into(), aborted);
                }
                if aborted {
                    return (None, true);
                }
                aborted = true;
                cancel.cancel();
            }
        }
    }
}
