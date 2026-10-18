//! Live split-pane TUI for the coordinated swarm (ratatui).
//!
//! - One fully-boxed pane per agent (grid) + a full-width RESULTS pane that shows
//!   the final integrated result, so the deliverable lives on the lead model's
//!   screen rather than being dumped raw to the terminal.
//! - Each pane is a mini Qwenodyssey dashboard (logo + model · backend + cwd) with a
//!   live token counter on its bottom border (↑ prompt ↓ completion · tok/s).
//! - The panes STAY UP when the run finishes (alt-screen review mode); you can focus
//!   any pane (Tab / 1-9) and scroll its full session history (↑/↓, PgUp/PgDn, g/G),
//!   then press q to leave. Nothing is printed raw to the scrollback.
//! - While the run is going, Esc (or x) stops the FOCUSED agent only (its subtask
//!   becomes "stopped" and dependents still run); q / Ctrl-C abort the whole run.
//!
//! The TUI owns the screen while it runs (raw mode + alternate screen) and always
//! restores the terminal: on normal exit, on early return, and from a panic hook.
//! TTY-gated — when stdout is not a TTY, [`SwarmTui::supported`] is false and the
//! caller uses the plain line logger instead.

use std::future::Future;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::{cursor, execute, terminal};
use futures::StreamExt;
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc::UnboundedReceiver;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::coordinator::{clean_agent_text, SubtaskStatus};
use crate::events::SwarmEvent;
use crate::swarm::{RosterEntry, SwarmArtifacts, SwarmRun};

const SPIN: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// The Qwenodyssey block-glyph mark (3 rows), shared with the launch banner.
const MARK: [&str; 3] = [" ▟█▜▛█▙ ", " ▜█▟▙█▛ ", "  ▀▘▝▀  "];
/// Per-pane scrollback cap (chars).
const MAX_BUFFER: usize = 200_000;
const VERSION: &str = env!("CARGO_PKG_VERSION");
const TICK: Duration = Duration::from_millis(100);

/// What the TUI shows before the plan arrives.
#[derive(Clone, Debug, Default)]
pub struct SwarmTuiOptions {
    pub task: String,
    /// Number of agent panes (the roster size).
    pub panes: usize,
    /// Footer warning (e.g. "local fallback — no frontier key configured").
    pub note: Option<String>,
    pub roster: Vec<RosterEntry>,
    /// Shown on every pane's dashboard.
    pub cwd: String,
}

/// Callbacks into the running swarm.
#[derive(Default)]
pub struct TuiControls {
    /// q / Ctrl-C while the run is still going: abort the whole run.
    pub on_abort: Option<Box<dyn Fn() + Send + Sync>>,
    /// Esc / x on a running agent pane: stop just that agent.
    pub on_stop_pane: Option<Box<dyn Fn(usize) + Send + Sync>>,
}

/// How the TUI session ended.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TuiOutcome {
    /// The user aborted the run before it completed.
    pub aborted: bool,
    /// The RESULTS pane text (final answer + generated-file list).
    pub final_result: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaneKind {
    Agent,
    Synth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaneStatus {
    Idle,
    Pending,
    Running,
    Done,
    Failed,
    Stopped,
}

impl From<SubtaskStatus> for PaneStatus {
    fn from(s: SubtaskStatus) -> Self {
        match s {
            SubtaskStatus::Pending => PaneStatus::Pending,
            SubtaskStatus::Running => PaneStatus::Running,
            SubtaskStatus::Done => PaneStatus::Done,
            SubtaskStatus::Failed => PaneStatus::Failed,
            SubtaskStatus::Stopped => PaneStatus::Stopped,
        }
    }
}

#[derive(Clone, Debug)]
struct PaneState {
    kind: PaneKind,
    subtask_id: Option<String>,
    title: String,
    model: String,
    full_model: Option<String>,
    backend: Option<String>,
    status: PaneStatus,
    buffer: String,
    started_at: Option<Instant>,
    ms: Option<u64>,
    // token accounting
    out_chars: usize,
    first_token_at: Option<Instant>,
    up_tokens: Option<u64>,
    /// Exact completion tokens once known.
    out_tokens: Option<u64>,
    /// Lines from the bottom; 0 = follow tail.
    scroll_offset: usize,
    /// Body rows at the last render (page size for PgUp/PgDn).
    page_rows: usize,
    /// Bumped on every buffer change (wrap-cache key; length alone goes stale at the cap).
    generation: u64,
    cache: Option<(usize, u64, Vec<String>)>,
}

impl PaneState {
    fn new(kind: PaneKind, r: Option<&RosterEntry>) -> Self {
        Self {
            kind,
            subtask_id: None,
            title: String::new(),
            model: r.map(|r| r.label.clone()).unwrap_or_default(),
            full_model: r.map(|r| if r.model.is_empty() { r.label.clone() } else { r.model.clone() }),
            backend: r.map(|r| r.backend.clone()).filter(|b| !b.is_empty()),
            status: PaneStatus::Idle,
            buffer: String::new(),
            started_at: None,
            ms: None,
            out_chars: 0,
            first_token_at: None,
            up_tokens: None,
            out_tokens: None,
            scroll_offset: 0,
            page_rows: 6,
            generation: 0,
            cache: None,
        }
    }

    fn touch(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn append(&mut self, delta: &str) {
        if self.out_chars == 0 {
            self.first_token_at = Some(Instant::now());
        }
        self.buffer.push_str(delta);
        self.out_chars += delta.chars().count();
        if self.buffer.len() > MAX_BUFFER {
            let mut cut = self.buffer.len() - MAX_BUFFER;
            while !self.buffer.is_char_boundary(cut) {
                cut += 1;
            }
            self.buffer.drain(..cut);
        }
        self.touch();
    }

    fn set_buffer(&mut self, s: String) {
        self.buffer = s;
        self.touch();
    }

    /// Cleaned + wrapped lines for the pane's full buffer (cached per width/generation).
    fn wrapped(&mut self, width: usize) -> &[String] {
        let fresh = matches!(&self.cache, Some((w, g, _)) if *w == width && *g == self.generation);
        if !fresh {
            let cleaned = clean_agent_text(&self.buffer);
            let plain = strip_ansi_escapes::strip_str(&cleaned);
            self.cache = Some((width, self.generation, wrap_all(&plain, width)));
        }
        self.cache.as_ref().map(|(_, _, l)| l.as_slice()).unwrap_or(&[])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Preflight,
    Planning,
    Running,
    Complete,
}

/// What a key press asks the event loop to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyAction {
    None,
    Redraw,
    Quit,
    StopPane(usize),
}

/// All TUI state; rendering is a pure function of it (testable on a TestBackend).
struct TuiState {
    task: String,
    note: Option<String>,
    cwd: String,
    panes: Vec<PaneState>,
    synth: PaneState,
    started_at: Instant,
    frame: usize,
    wave_index: usize,
    subtask_count: usize,
    done_count: usize,
    fail_count: usize,
    phase: Phase,
    planning_with: Option<String>,
    plan_note: Option<String>,
    preflight_note: Option<String>,
    preflight_count: usize,
    preflight_broken: Vec<String>,
    planned: bool,
    exec_label: Option<String>,
    artifacts: Option<SwarmArtifacts>,
    /// Focused pane index; `panes.len()` = the results pane.
    focus: usize,
}

impl TuiState {
    fn new(opts: &SwarmTuiOptions) -> Self {
        let n = opts.panes.max(1);
        let panes = (0..n).map(|i| PaneState::new(PaneKind::Agent, opts.roster.get(i))).collect();
        let mut synth = PaneState::new(PaneKind::Synth, None);
        synth.title = "results".to_string();
        Self {
            task: opts.task.clone(),
            note: opts.note.clone(),
            cwd: opts.cwd.clone(),
            panes,
            synth,
            started_at: Instant::now(),
            frame: 0,
            wave_index: 0,
            subtask_count: 0,
            done_count: 0,
            fail_count: 0,
            phase: Phase::Planning,
            planning_with: None,
            plan_note: None,
            preflight_note: None,
            preflight_count: 0,
            preflight_broken: Vec::new(),
            planned: false,
            exec_label: None,
            artifacts: None,
            focus: 0,
        }
    }

    fn pane_mut(&mut self, pane: usize) -> &mut PaneState {
        let n = self.panes.len();
        &mut self.panes[pane % n]
    }

    /* ── event wiring ── */

    fn apply(&mut self, ev: &SwarmEvent) {
        match ev {
            SwarmEvent::PreflightStart { count } => {
                self.phase = Phase::Preflight;
                self.preflight_count = *count;
            }
            SwarmEvent::Preflight(r) => {
                if r.pane < self.panes.len() {
                    let p = &mut self.panes[r.pane];
                    if r.ok {
                        p.set_buffer(format!("✓ key ok ({}ms)", r.ms));
                    } else {
                        p.status = PaneStatus::Failed;
                        p.set_buffer(format!("✗ key check failed: {}", r.detail.as_deref().unwrap_or("unreachable")));
                    }
                }
                if !r.ok {
                    self.preflight_broken.push(format!("{} ({})", r.label, r.detail.as_deref().unwrap_or("unreachable")));
                }
            }
            SwarmEvent::PreflightDone { ok, broken } => {
                self.phase = Phase::Planning;
                if *broken > 0 {
                    let which = self.preflight_broken.join(", ");
                    self.preflight_note = Some(if *ok > 0 {
                        format!("{broken} broken key(s) dropped: {which}")
                    } else {
                        format!("every key failed preflight: {which}")
                    });
                }
            }
            SwarmEvent::Planner { model } => self.planning_with = Some(model.clone()),
            SwarmEvent::Plan(e) => {
                self.subtask_count = e.subtasks.len();
                self.planning_with = None;
                self.planned = true;
                self.phase = Phase::Running;
                if e.exec.is_some() {
                    self.exec_label = e.exec.clone();
                }
                if e.note.is_some() {
                    self.plan_note = e.note.clone();
                }
                // Preflight may have dropped broken workers: panes follow the live roster.
                if !e.roster.is_empty() && e.roster.len() != self.panes.len() {
                    self.panes = e.roster.iter().map(|r| PaneState::new(PaneKind::Agent, Some(r))).collect();
                    if self.focus > self.panes.len() {
                        self.focus = self.panes.len();
                    }
                } else {
                    for (p, r) in self.panes.iter_mut().zip(&e.roster) {
                        p.model = r.label.clone();
                        p.full_model = Some(if r.model.is_empty() { r.label.clone() } else { r.model.clone() });
                        p.backend = Some(r.backend.clone()).filter(|b| !b.is_empty());
                        if p.status == PaneStatus::Idle || p.status == PaneStatus::Failed {
                            p.status = PaneStatus::Idle;
                            p.set_buffer(String::new());
                        }
                    }
                }
            }
            SwarmEvent::Wave(e) => self.wave_index = e.index,
            SwarmEvent::Assign(e) => {
                let p = self.pane_mut(e.pane);
                p.subtask_id = Some(e.subtask_id.clone());
                p.title = e.title.clone();
                if !e.worker_label.is_empty() {
                    p.model = e.worker_label.clone();
                }
                if !e.model.is_empty() {
                    p.full_model = Some(e.model.clone());
                }
                p.status = PaneStatus::Running;
                p.set_buffer(String::new());
                p.out_chars = 0;
                p.first_token_at = None;
                p.out_tokens = None;
                p.up_tokens = None;
                p.scroll_offset = 0;
                p.started_at = Some(Instant::now());
                p.ms = None;
            }
            SwarmEvent::Delta(e) => self.pane_mut(e.pane).append(&e.delta),
            SwarmEvent::Status(e) => {
                let status = PaneStatus::from(e.status);
                let p = self.pane_mut(e.pane);
                p.status = status;
                if e.ms.is_some() {
                    p.ms = e.ms;
                }
                match e.status {
                    SubtaskStatus::Done => self.done_count += 1,
                    SubtaskStatus::Failed => self.fail_count += 1,
                    _ => {}
                }
            }
            SwarmEvent::Result(r) => {
                if let Some(p) = self.panes.iter_mut().find(|x| x.subtask_id.is_some() && x.subtask_id == r.id) {
                    if r.prompt_tokens.is_some() {
                        p.up_tokens = r.prompt_tokens;
                    }
                    if r.completion_tokens.is_some() {
                        p.out_tokens = r.completion_tokens;
                    }
                }
            }
            SwarmEvent::ExecCmd(_) => {}
            SwarmEvent::Synth { model } => {
                let s = &mut self.synth;
                s.status = PaneStatus::Running;
                s.model = model.clone();
                s.full_model = Some(model.clone());
                s.started_at = Some(Instant::now());
                s.set_buffer(String::new());
                s.out_chars = 0;
                s.first_token_at = None;
            }
            SwarmEvent::SynthDelta { delta } => self.synth.append(delta),
            SwarmEvent::Artifacts(a) => self.artifacts = Some(a.clone()),
            SwarmEvent::Done(_) => {}
        }
    }

    /// Mark the run finished and surface the final output + files on the RESULTS pane.
    fn complete(&mut self, run: &SwarmRun) {
        self.phase = Phase::Complete;
        self.synth.status = if run.results.iter().any(|r| r.ok) { PaneStatus::Done } else { PaneStatus::Failed };
        if let Some(by) = &run.synthesized_by {
            self.synth.model = by.clone();
            self.synth.full_model = Some(by.clone());
        } else if self.synth.model.is_empty() {
            self.synth.model = "results".to_string();
        }
        if let Some(a) = &run.artifacts {
            self.artifacts = Some(a.clone());
        }
        let ok: Vec<String> = run
            .results
            .iter()
            .filter(|r| r.ok)
            .map(|r| format!("## {}\n{}", r.title.as_deref().unwrap_or(&r.model), r.text))
            .collect();
        let mut out = match &run.synthesis {
            Some(s) if !s.is_empty() => s.clone(),
            _ if !ok.is_empty() => ok.join("\n\n"),
            _ => "All agents failed — no result.".to_string(),
        };
        if let Some(arts) = &self.artifacts {
            out.push_str("\n\n──────── Files generated ────────\n");
            out.push_str(&format!("Location: {}\n", arts.location));
            if arts.files.is_empty() {
                out.push_str("  (no new files detected)");
            } else {
                out.push_str(&arts.files.iter().map(|f| format!("  • {f}")).collect::<Vec<_>>().join("\n"));
            }
        }
        self.synth.set_buffer(out);
        self.focus = self.panes.len(); // focus the results pane
        self.synth.scroll_offset = usize::MAX; // show it from the top
    }

    fn final_result(&self) -> String {
        self.synth.buffer.clone()
    }

    fn focus_count(&self) -> usize {
        self.panes.len() + 1
    }

    fn focused_mut(&mut self) -> &mut PaneState {
        if self.focus >= self.panes.len() {
            &mut self.synth
        } else {
            &mut self.panes[self.focus]
        }
    }

    /* ── keyboard ── */

    fn handle_key(&mut self, key: KeyEvent) -> KeyAction {
        let all = self.focus_count();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => return KeyAction::Quit,
            KeyCode::Char('q') | KeyCode::Char('Q') => return KeyAction::Quit,
            KeyCode::Esc | KeyCode::Char('x') => {
                if self.phase != Phase::Complete && self.focus < self.panes.len() {
                    let p = &mut self.panes[self.focus];
                    if p.status == PaneStatus::Running {
                        p.append("\n⏹ stop requested…\n");
                        return KeyAction::StopPane(self.focus);
                    }
                }
                return KeyAction::None;
            }
            KeyCode::Tab => self.focus = (self.focus + 1) % all,
            KeyCode::BackTab => self.focus = (self.focus + all - 1) % all,
            KeyCode::Up | KeyCode::Char('k') => {
                let p = self.focused_mut();
                p.scroll_offset = p.scroll_offset.saturating_add(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let p = self.focused_mut();
                p.scroll_offset = p.scroll_offset.saturating_sub(1);
            }
            KeyCode::PageUp => {
                let p = self.focused_mut();
                p.scroll_offset = p.scroll_offset.saturating_add(p.page_rows.max(1));
            }
            KeyCode::PageDown => {
                let p = self.focused_mut();
                p.scroll_offset = p.scroll_offset.saturating_sub(p.page_rows.max(1));
            }
            KeyCode::Char('g') | KeyCode::Home => self.focused_mut().scroll_offset = usize::MAX,
            KeyCode::Char('G') | KeyCode::End => self.focused_mut().scroll_offset = 0,
            KeyCode::Char(d @ '1'..='9') => {
                let idx = (d as usize) - ('1' as usize);
                if idx < all {
                    self.focus = idx;
                }
            }
            _ => return KeyAction::None, // unknown key, no redraw
        }
        KeyAction::Redraw
    }

    /* ── rendering ── */

    fn render(&mut self, f: &mut Frame) {
        self.frame = self.frame.wrapping_add(1);
        let area = f.area();
        if area.width < 4 || area.height < 3 {
            return;
        }
        let width = area.width as i64;
        let height = area.height as i64;
        let header = self.header(area.width as usize);
        f.render_widget(Paragraph::new(header), Rect::new(area.x, area.y, area.width, 1));

        let n = self.panes.len() as i64;
        let cols = 1.max(n.min((width / 24).max(1)).min((n as f64).sqrt().ceil() as i64));
        let grid_rows = (n + cols - 1) / cols;
        let gap: i64 = 1;
        let pane_w = 20.max((width - gap * (cols - 1)) / cols);
        let full_w = pane_w * cols + gap * (cols - 1);
        let avail = 8.max(height - 2); // minus header + footer
        let synth_h = 4.max((avail - grid_rows * 3).min(avail * 32 / 100));
        let grid_area = avail - synth_h;
        let pane_h = 3.max(grid_area / grid_rows.max(1));

        // Everything between the header and the footer row.
        let body = Rect::new(area.x, area.y + 1, area.width, area.height.saturating_sub(2));
        let clip = |x: i64, y: i64, w: i64, h: i64| -> Option<Rect> {
            let r = Rect::new(
                area.x.saturating_add(x.clamp(0, u16::MAX as i64) as u16),
                area.y.saturating_add(y.clamp(0, u16::MAX as i64) as u16),
                w.clamp(0, u16::MAX as i64) as u16,
                h.clamp(0, u16::MAX as i64) as u16,
            )
            .intersection(body);
            (r.width >= 3 && r.height >= 2).then_some(r)
        };

        // Agent grid.
        let focus = self.focus;
        let frame = self.frame;
        for idx in 0..self.panes.len() {
            let (r, c) = (idx as i64 / cols, idx as i64 % cols);
            if let Some(rect) = clip(c * (pane_w + gap), 1 + r * pane_h, pane_w, pane_h) {
                let ctx = PaneCtx { frame, phase: self.phase, synth_running: false, cwd: &self.cwd };
                render_pane(f, &mut self.panes[idx], rect, focus == idx, &ctx);
            }
        }
        for idx in self.panes.len()..(grid_rows * cols) as usize {
            let (r, c) = (idx as i64 / cols, idx as i64 % cols);
            if let Some(rect) = clip(c * (pane_w + gap), 1 + r * pane_h, pane_w, pane_h) {
                f.render_widget(Block::bordered().border_style(Style::default().fg(Color::DarkGray)), rect);
            }
        }

        // Results pane (full width).
        if let Some(rect) = clip(0, 1 + grid_rows * pane_h, full_w, synth_h) {
            let ctx = PaneCtx {
                frame,
                phase: self.phase,
                synth_running: self.synth.status == PaneStatus::Running,
                cwd: &self.cwd,
            };
            let focused = focus == self.panes.len();
            render_pane(f, &mut self.synth, rect, focused, &ctx);
        }

        let footer = self.footer(area.width as usize);
        f.render_widget(Paragraph::new(footer), Rect::new(area.x, area.bottom() - 1, area.width, 1));
    }

    fn header(&self, width: usize) -> Line<'static> {
        let elapsed = fmt_elapsed(self.started_at.elapsed().as_secs());
        let spin = SPIN[self.frame % SPIN.len()];
        let (phase, phase_style) = match self.phase {
            Phase::Complete => ("✓ complete".to_string(), Style::default().fg(Color::Green)),
            Phase::Preflight => (format!("{spin} checking {} key(s)…", self.preflight_count), gray()),
            _ if self.planned => (format!("wave {}", self.wave_index + 1), gray()),
            _ => (
                format!(
                    "{spin} planning split{}…",
                    self.planning_with.as_deref().map(|m| format!(" with {m}")).unwrap_or_default()
                ),
                gray(),
            ),
        };
        let right = format!(" · {elapsed}");
        let right_w = phase.width() + right.width();
        let left_budget = width.saturating_sub(right_w + 1);
        let title = "🜂 swarm";
        let task = fit(&format!("  {}", self.task.replace(['\n', '\r'], " ")), left_budget.saturating_sub(title.width()));
        let used = title.width().min(left_budget) + task.width();
        let pad = width.saturating_sub(used + right_w).max(1);
        Line::from(vec![
            Span::styled(fit(title, left_budget), Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(task, gray()),
            Span::raw(" ".repeat(pad)),
            Span::styled(phase, phase_style),
            Span::styled(right, gray()),
        ])
    }

    fn footer(&self, width: usize) -> Line<'static> {
        let notes: Vec<&str> =
            [self.note.as_deref(), self.preflight_note.as_deref(), self.plan_note.as_deref()].into_iter().flatten().collect();
        let total = if self.subtask_count > 0 { self.subtask_count } else { self.panes.len() };
        let mut spans: Vec<(String, Style)> = vec![(format!("{}/{} done", self.done_count, total), gray())];
        if self.fail_count > 0 {
            spans.push((format!(" · {} failed", self.fail_count), Style::default().fg(Color::Red)));
        }
        if let Some(e) = &self.exec_label {
            spans.push((format!(" · exec: {e}"), Style::default().fg(Color::Cyan)));
        }
        if !notes.is_empty() {
            spans.push((format!(" ⚠ {}", notes.join(" · ")), Style::default().fg(Color::Yellow)));
        }
        if self.phase == Phase::Complete {
            spans.push(("   ".to_string(), gray()));
            spans.push(("Tab".to_string(), cyan()));
            spans.push((" focus · ".to_string(), gray()));
            spans.push(("↑↓/PgUp/PgDn/g/G".to_string(), cyan()));
            spans.push((" scroll · ".to_string(), gray()));
            spans.push(("q".to_string(), cyan()));
            spans.push((" quit".to_string(), gray()));
        } else {
            spans.push(("   Tab focus · ↑↓ scroll · Esc stop agent · Ctrl-C abort".to_string(), gray()));
        }
        // Truncate to the visible width, span by span.
        let mut left = width;
        let mut out = Vec::new();
        for (text, style) in spans {
            if left == 0 {
                break;
            }
            let t = fit(&text, left);
            left = left.saturating_sub(t.width());
            out.push(Span::styled(t, style));
        }
        Line::from(out)
    }
}

struct PaneCtx<'a> {
    frame: usize,
    phase: Phase,
    synth_running: bool,
    cwd: &'a str,
}

fn gray() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn cyan() -> Style {
    Style::default().fg(Color::Cyan)
}

fn status_style(s: PaneStatus) -> Style {
    match s {
        PaneStatus::Done => Style::default().fg(Color::Green),
        PaneStatus::Failed => Style::default().fg(Color::Red),
        PaneStatus::Running => Style::default().fg(Color::Cyan),
        PaneStatus::Stopped => Style::default().fg(Color::Yellow),
        PaneStatus::Idle | PaneStatus::Pending => gray(),
    }
}

fn status_glyph(s: PaneStatus, frame: usize) -> &'static str {
    match s {
        PaneStatus::Running => SPIN[frame % SPIN.len()],
        PaneStatus::Done => "✓",
        PaneStatus::Failed => "✗",
        PaneStatus::Stopped => "■",
        PaneStatus::Idle | PaneStatus::Pending => "·",
    }
}

/// Render one fully-boxed pane into `rect`.
fn render_pane(f: &mut Frame, p: &mut PaneState, rect: Rect, focused: bool, ctx: &PaneCtx<'_>) {
    let border = if focused { Style::default().fg(Color::LightCyan) } else { gray() };
    let inner_w = rect.width.saturating_sub(2) as usize;
    let mut body_rows = rect.height.saturating_sub(2) as usize;

    // The bottom border is overwritten below with the token counter (it needs the scroll math).
    let block = Block::bordered().border_style(border).title(title_bar(p, inner_w, ctx));
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    let mut y = inner.y;

    // Per-agent Qwenodyssey dashboard when tall enough.
    if p.kind == PaneKind::Agent && body_rows >= MARK.len() + 2 {
        let model = p.full_model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| if p.model.is_empty() { "—".into() } else { p.model.clone() });
        let rows: [Vec<Span<'static>>; 3] = [
            vec![
                Span::styled("Qwenodyssey".to_string(), Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(format!(" v{VERSION}"), gray()),
            ],
            vec![
                Span::styled(model, Style::default().fg(Color::White)),
                Span::styled(p.backend.as_ref().map(|b| format!(" · {b}")).unwrap_or_default(), gray()),
            ],
            vec![Span::styled(ctx.cwd.to_string(), gray())],
        ];
        for (mark, texts) in MARK.iter().zip(rows) {
            let mut spans = vec![Span::styled(format!("{mark}  "), Style::default().fg(Color::LightCyan))];
            spans.extend(texts);
            f.render_widget(Paragraph::new(clip_spans(spans, inner_w)), Rect::new(inner.x, y, inner.width, 1));
            y += 1;
        }
        // Separator joined to the border: ├────┤
        let sep_y = y;
        f.render_widget(Paragraph::new(Span::styled("─".repeat(inner_w), border)), Rect::new(inner.x, sep_y, inner.width, 1));
        let buf = f.buffer_mut();
        if let Some(c) = buf.cell_mut((rect.x, sep_y)) {
            c.set_symbol("├").set_style(border);
        }
        if let Some(c) = buf.cell_mut((rect.right().saturating_sub(1), sep_y)) {
            c.set_symbol("┤").set_style(border);
        }
        y += 1;
        body_rows -= MARK.len() + 1;
    }
    p.page_rows = body_rows.max(1);

    let total = p.wrapped(inner_w).len();
    let max_off = total.saturating_sub(body_rows);
    if p.scroll_offset > max_off {
        p.scroll_offset = max_off;
    }
    let end = total - p.scroll_offset;
    let start = end.saturating_sub(body_rows);
    // Cached: only the visible window is copied per frame.
    let lines: Vec<Line> = p.wrapped(inner_w)[start..end]
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().add_modifier(Modifier::DIM))))
        .collect();
    if body_rows > 0 {
        f.render_widget(Paragraph::new(lines), Rect::new(inner.x, y, inner.width, body_rows as u16));
    }

    // Bottom border: live token counter + scroll indicator.
    let bottom = bottom_bar(p, inner_w, start > 0, p.scroll_offset > 0);
    let by = rect.bottom().saturating_sub(1);
    f.render_widget(Paragraph::new(bottom), Rect::new(rect.x + 1, by, inner_w as u16, 1));
}

fn title_bar(p: &PaneState, inner: usize, ctx: &PaneCtx<'_>) -> Line<'static> {
    let glyph = status_glyph(p.status, ctx.frame);
    let elapsed = match (p.status, p.started_at, p.ms) {
        (PaneStatus::Running, Some(t), _) => fmt_elapsed(t.elapsed().as_secs()),
        (_, _, Some(ms)) => fmt_elapsed((ms + 500) / 1000),
        _ => String::new(),
    };
    let name = match p.kind {
        PaneKind::Synth => {
            let stage = if ctx.phase == Phase::Complete {
                ""
            } else if ctx.synth_running {
                " · compiling"
            } else {
                " · waiting"
            };
            format!("📋 results{stage}{}", if p.model.is_empty() { String::new() } else { format!(" · {}", p.model) })
        }
        PaneKind::Agent => if p.model.is_empty() { "—".to_string() } else { p.model.clone() },
    };
    let subtitle = if p.kind == PaneKind::Agent && !p.title.is_empty() { format!(" · {}", p.title) } else { String::new() };
    let elap_tag = if elapsed.is_empty() { String::new() } else { format!("{elapsed} ") };
    let title_plain = format!(" {glyph} {name}{subtitle}");
    let budget = inner.saturating_sub(elap_tag.width());
    let title_fit = pad_to(&fit(title_plain.trim_end(), budget), budget);
    Line::from(vec![Span::styled(title_fit, status_style(p.status)), Span::styled(elap_tag, gray())])
}

fn bottom_bar(p: &PaneState, inner: usize, more_above: bool, scrolled_up: bool) -> Line<'static> {
    let up = p.up_tokens.map(|t| format!("↑ {} ", fmt_tok(t))).unwrap_or_default();
    let out_tok = p.out_tokens.unwrap_or_else(|| p.out_chars.div_ceil(4) as u64);
    let down = if out_tok > 0 || p.status != PaneStatus::Idle { format!("↓ {} ", fmt_tok(out_tok)) } else { String::new() };
    let mut rate = 0.0;
    if let (Some(t0), PaneStatus::Running) = (p.first_token_at, p.status) {
        let secs = t0.elapsed().as_secs_f64();
        if secs > 0.3 {
            rate = out_tok as f64 / secs;
        }
    }
    let rate_tag = if rate > 0.0 {
        if rate >= 10.0 {
            format!("· {} tok/s ", rate.round() as u64)
        } else {
            format!("· {rate:.1} tok/s ")
        }
    } else {
        String::new()
    };
    let left = fit(&format!("─ {up}{down}{rate_tag}"), inner);
    let scroll = if scrolled_up {
        " ⇡scroll "
    } else if more_above {
        " ▲more "
    } else {
        ""
    };
    let scroll = fit(scroll, inner.saturating_sub(left.width()));
    let fill = inner.saturating_sub(left.width() + scroll.width());
    Line::from(vec![
        Span::styled(left, gray()),
        Span::styled("─".repeat(fill), gray()),
        Span::styled(scroll, Style::default().fg(Color::Yellow)),
    ])
}

/* ──────────────────────────── helpers ──────────────────────────── */

/// Wrap a multi-line string to `width` display columns, returning ALL visual lines.
fn wrap_all(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() || width < 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for ln in text.replace('\r', "").split('\n') {
        let ln = ln.replace('\t', "    ");
        if ln.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut cur = String::new();
        let mut cur_w = 0usize;
        for ch in ln.chars() {
            if ch.is_control() {
                continue;
            }
            let w = ch.width().unwrap_or(0);
            if cur_w + w > width && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            cur.push(ch);
            cur_w += w;
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// Truncate `s` to at most `w` display columns.
fn fit(s: &str, w: usize) -> String {
    if s.width() <= w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if used + cw > w {
            break;
        }
        out.push(ch);
        used += cw;
    }
    out
}

fn pad_to(s: &str, w: usize) -> String {
    let n = s.width();
    if n >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - n))
    }
}

fn clip_spans(spans: Vec<Span<'static>>, w: usize) -> Line<'static> {
    let mut left = w;
    let mut out = Vec::new();
    for s in spans {
        if left == 0 {
            break;
        }
        let t = fit(&s.content, left);
        left = left.saturating_sub(t.width());
        out.push(Span::styled(t, s.style));
    }
    Line::from(out)
}

fn fmt_elapsed(seconds: u64) -> String {
    if seconds >= 60 {
        format!("{}m{}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn fmt_tok(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = format!("{:.1}", n as f64 / 1000.0);
    format!("{}k", s.strip_suffix(".0").unwrap_or(&s))
}

/* ──────────────────────────── terminal ownership ──────────────────────────── */

static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

fn restore_terminal() {
    let _ = terminal::disable_raw_mode();
    let mut out = std::io::stdout();
    let _ = execute!(out, cursor::Show, terminal::LeaveAlternateScreen);
    let _ = out.flush();
}

/// Raw mode + alternate screen for the TUI's lifetime; restores on drop and
/// (via a chained panic hook) on panic, so a crash never leaves the shell raw.
struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> std::io::Result<Self> {
        PANIC_HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if TUI_ACTIVE.swap(false, Ordering::SeqCst) {
                    restore_terminal();
                }
                prev(info);
            }));
        });
        terminal::enable_raw_mode()?;
        TUI_ACTIVE.store(true, Ordering::SeqCst);
        let mut out = std::io::stdout();
        if let Err(e) = execute!(out, terminal::EnterAlternateScreen, cursor::Hide, terminal::Clear(terminal::ClearType::All)) {
            TUI_ACTIVE.store(false, Ordering::SeqCst);
            restore_terminal();
            return Err(e);
        }
        Ok(TerminalGuard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if TUI_ACTIVE.swap(false, Ordering::SeqCst) {
            restore_terminal();
        }
    }
}

/// The live dashboard.
pub struct SwarmTui;

impl SwarmTui {
    /// stdout is a TTY at least 40 columns wide.
    pub fn supported() -> bool {
        std::io::stdout().is_terminal() && terminal::size().map(|(c, _)| c >= 40).unwrap_or(false)
    }

    /// Own the screen while `run` executes: render `events` live, route keys
    /// (Esc → `on_stop_pane`, q/Ctrl-C → `on_abort`), then keep the panes up in
    /// review mode until the user presses q (skipped when stdin is not a TTY or
    /// the user already quit). `run` is always driven to completion, so the
    /// swarm's cleanup (sandbox teardown) happens before this returns.
    ///
    /// If the terminal can't be taken over, the run still completes, headless.
    pub async fn run<F>(
        opts: SwarmTuiOptions,
        mut events: UnboundedReceiver<SwarmEvent>,
        controls: TuiControls,
        run: F,
    ) -> (SwarmRun, TuiOutcome)
    where
        F: Future<Output = SwarmRun>,
    {
        let mut state = TuiState::new(&opts);
        let guard = match TerminalGuard::enter() {
            Ok(g) => g,
            Err(_) => {
                drop(events);
                let r = run.await;
                state.complete(&r);
                return (r, TuiOutcome { aborted: false, final_result: state.final_result() });
            }
        };
        let mut term = match Terminal::new(CrosstermBackend::new(std::io::stdout())) {
            Ok(t) => t,
            Err(_) => {
                drop(guard);
                drop(events);
                let r = run.await;
                state.complete(&r);
                return (r, TuiOutcome { aborted: false, final_result: state.final_result() });
            }
        };
        let review = std::io::stdin().is_terminal();
        let (result, aborted) = event_loop(&mut state, &mut term, &mut events, &controls, run, review).await;
        let _ = term.show_cursor();
        drop(term);
        drop(guard);
        (result, TuiOutcome { aborted, final_result: state.final_result() })
    }
}

async fn event_loop<B: Backend, F: Future<Output = SwarmRun>>(
    state: &mut TuiState,
    term: &mut Terminal<B>,
    events: &mut UnboundedReceiver<SwarmEvent>,
    controls: &TuiControls,
    run: F,
    review: bool,
) -> (SwarmRun, bool) {
    tokio::pin!(run);
    let mut keys = EventStream::new();
    let mut keys_alive = true;
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut result: Option<SwarmRun> = None;
    let mut quit = false;
    let mut aborted = false;
    let _ = term.clear();
    let _ = term.draw(|f| state.render(f));
    loop {
        if let Some(r) = result.take() {
            if quit || !review || !keys_alive {
                return (r, aborted);
            }
            result = Some(r);
        }
        let mut redraw = false;
        tokio::select! {
            r = &mut run, if result.is_none() => {
                while let Ok(ev) = events.try_recv() {
                    state.apply(&ev);
                }
                state.complete(&r);
                result = Some(r);
                redraw = true;
            }
            Some(ev) = events.recv() => {
                state.apply(&ev);
                // Batch a burst of deltas into one wake-up.
                for _ in 0..512 {
                    match events.try_recv() {
                        Ok(ev) => state.apply(&ev),
                        Err(_) => break,
                    }
                }
            }
            k = keys.next(), if keys_alive => match k {
                Some(Ok(Event::Key(key))) if key.kind != KeyEventKind::Release => {
                    match state.handle_key(key) {
                        KeyAction::Quit => {
                            quit = true;
                            if result.is_none() {
                                aborted = true;
                                if let Some(f) = &controls.on_abort {
                                    f();
                                }
                            }
                        }
                        KeyAction::StopPane(i) => {
                            if let Some(f) = &controls.on_stop_pane {
                                f(i);
                            }
                            redraw = true;
                        }
                        KeyAction::Redraw => redraw = true,
                        KeyAction::None => {}
                    }
                }
                Some(Ok(Event::Resize(_, _))) => {
                    let _ = term.autoresize();
                    redraw = true;
                }
                Some(Ok(_)) => {}
                Some(Err(_)) | None => keys_alive = false,
            },
            _ = tick.tick() => redraw = true,
        }
        if redraw {
            let _ = term.draw(|f| state.render(f));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::Subtask;
    use crate::events::{AssignEvent, DeltaEvent, PlanEvent, StatusEvent};
    use crate::swarm::{SwarmMode, WorkerKind, WorkerResult};
    use ratatui::backend::TestBackend;

    fn opts(n: usize) -> SwarmTuiOptions {
        SwarmTuiOptions {
            task: "build a parser".into(),
            panes: n,
            note: Some("demo mode".into()),
            roster: (0..n)
                .map(|i| RosterEntry { label: format!("w{i}"), model: format!("org/w{i}"), backend: "nvidia".into() })
                .collect(),
            cwd: "C:\\work".into(),
        }
    }

    fn screen(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut s = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    fn feed(state: &mut TuiState) {
        state.apply(&SwarmEvent::Plan(PlanEvent {
            task: "build a parser".into(),
            subtasks: vec![Subtask { id: "a".into(), title: "Lexer".into(), detail: "d".into(), depends_on: vec![] }],
            roster: opts(3).roster,
            planned_by: Some("lead".into()),
            ..Default::default()
        }));
        state.apply(&SwarmEvent::Assign(AssignEvent {
            pane: 0,
            subtask_id: "a".into(),
            title: "Lexer".into(),
            worker_label: "w0".into(),
            model: "org/w0".into(),
        }));
        for i in 0..200 {
            state.apply(&SwarmEvent::Delta(DeltaEvent { pane: 0, subtask_id: "a".into(), delta: format!("line {i}\n") }));
        }
    }

    #[test]
    fn renders_grid_results_pane_and_counters_at_many_sizes() {
        for (w, h) in [(120u16, 40u16), (80, 24), (40, 10), (200, 60), (10, 4)] {
            let mut state = TuiState::new(&opts(3));
            feed(&mut state);
            let mut term = Terminal::new(TestBackend::new(w, h)).expect("test terminal");
            term.draw(|f| state.render(f)).expect("draw");
            if w >= 80 {
                let s = screen(&term);
                assert!(s.contains("swarm"), "header at {w}x{h}");
                assert!(s.contains("results"), "results pane at {w}x{h}");
                assert!(s.contains("line 199"), "tail followed at {w}x{h}");
                assert!(s.contains("↓"), "token counter at {w}x{h}");
            }
        }
    }

    #[test]
    fn complete_focuses_results_and_lists_files() {
        let mut state = TuiState::new(&opts(2));
        feed(&mut state);
        let run = SwarmRun {
            mode: SwarmMode::Divide,
            results: vec![WorkerResult {
                ok: true,
                text: "done".into(),
                model: "org/w0".into(),
                backend: WorkerKind::Nvidia,
                id: Some("a".into()),
                title: Some("Lexer".into()),
                ..Default::default()
            }],
            synthesis: Some("FINAL ANSWER".into()),
            synthesized_by: Some("lead".into()),
            artifacts: Some(SwarmArtifacts { location: "bare metal: C:\\work".into(), files: vec!["C:\\work\\a.py".into()] }),
        };
        state.complete(&run);
        assert_eq!(state.focus, state.panes.len());
        let out = state.final_result();
        assert!(out.starts_with("FINAL ANSWER"));
        assert!(out.contains("Files generated"));
        assert!(out.contains("a.py"));
        let mut term = Terminal::new(TestBackend::new(100, 30)).expect("test terminal");
        term.draw(|f| state.render(f)).expect("draw");
        let s = screen(&term);
        assert!(s.contains("FINAL ANSWER"), "results shown from the top");
        assert!(s.contains("complete"));
    }

    #[test]
    fn keys_focus_scroll_stop_and_quit() {
        let mut state = TuiState::new(&opts(2));
        feed(&mut state);
        let key = |c| KeyEvent::new(c, KeyModifiers::NONE);
        // Esc on the running focused pane asks to stop that pane only.
        assert_eq!(state.handle_key(key(KeyCode::Esc)), KeyAction::StopPane(0));
        // Tab cycles through agents + results.
        assert_eq!(state.handle_key(key(KeyCode::Tab)), KeyAction::Redraw);
        assert_eq!(state.focus, 1);
        assert_eq!(state.handle_key(key(KeyCode::Esc)), KeyAction::None, "idle pane: nothing to stop");
        state.handle_key(key(KeyCode::Char('1')));
        assert_eq!(state.focus, 0);
        state.handle_key(key(KeyCode::Up));
        assert_eq!(state.panes[0].scroll_offset, 1);
        state.handle_key(key(KeyCode::Char('g')));
        assert_eq!(state.panes[0].scroll_offset, usize::MAX);
        state.handle_key(key(KeyCode::Char('G')));
        assert_eq!(state.panes[0].scroll_offset, 0);
        assert_eq!(state.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)), KeyAction::Quit);
        assert_eq!(state.handle_key(key(KeyCode::Char('q'))), KeyAction::Quit);
    }

    #[test]
    fn status_and_result_events_update_counters() {
        let mut state = TuiState::new(&opts(2));
        feed(&mut state);
        state.apply(&SwarmEvent::Status(StatusEvent {
            pane: 0,
            subtask_id: "a".into(),
            status: SubtaskStatus::Done,
            ms: Some(1500),
            error: None,
        }));
        state.apply(&SwarmEvent::Result(WorkerResult {
            id: Some("a".into()),
            ok: true,
            prompt_tokens: Some(1234),
            completion_tokens: Some(56),
            ..Default::default()
        }));
        assert_eq!(state.done_count, 1);
        assert_eq!(state.panes[0].up_tokens, Some(1234));
        let bar = bottom_bar(&state.panes[0], 40, false, false);
        let text: String = bar.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("↑ 1.2k"), "{text}");
        assert!(text.contains("↓ 56"), "{text}");
    }

    #[test]
    fn wrap_and_format_helpers() {
        assert_eq!(wrap_all("abcdef", 4), vec!["abcd", "ef"]);
        assert_eq!(wrap_all("a\n\nb", 4), vec!["a", "", "b"]);
        assert_eq!(fmt_tok(999), "999");
        assert_eq!(fmt_tok(1000), "1k");
        assert_eq!(fmt_tok(1234), "1.2k");
        assert_eq!(fmt_tok(15_000), "15k");
        assert_eq!(fmt_elapsed(75), "1m15s");
        assert_eq!(fit("héllo wörld", 5), "héllo");
    }
}
