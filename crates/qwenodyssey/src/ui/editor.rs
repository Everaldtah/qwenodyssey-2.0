//! The prompt line editor with a live slash-command palette (port of
//! src/cli/prompt.ts `createPrompt`).
//!
//! In a real terminal this consumes key events from the single input owner
//! ([`super::input`]) and renders a filtering menu beneath the input line: type
//! `/` to see every command, keep typing to filter (e.g. `/s` → /settings),
//! ↑/↓ to highlight, Tab/Enter to accept. When stdin is not a TTY (piped input,
//! tests) it is a plain line reader with no menu.
//!
//! Unlike 1.x, the editor never touches the terminal mode or stdin: raw mode is
//! already on for the whole session, and the editor only borrows the input
//! stream (a [`Lease`]) while it waits for a line. All editing logic lives in
//! [`EditorState`], a pure state machine that turns events into redraw strings,
//! so it is unit-tested without a terminal.

use std::collections::HashMap;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::input::{is_text_key, Input, InputEvent, Lease};
use super::{out, style};

/// One entry of the `/` palette (and the static /help menu).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlashCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// Argument hint, e.g. `"[name|#]"`. A command with args is completed to
    /// `"<name> "` on Enter instead of being submitted, so the user can type them.
    pub args: Option<&'static str>,
    pub desc: &'static str,
}

impl SlashCommand {
    /// `"/model [name|#]"` — the palette's left column.
    pub fn label(&self) -> String {
        match self.args {
            Some(a) => format!("{} {a}", self.name),
            None => self.name.to_string(),
        }
    }
}

/// Max menu rows drawn at once (keeps us clear of bottom-of-screen scroll).
pub const MAX_MENU: usize = 8;

/// A typed or pasted chunk longer than this (or containing a newline) becomes
/// a collapsed one-line chip so the input line never wraps the terminal.
const CHIP_MIN_CHARS: usize = 40;

/// Number of EXTRA physical rows a line of `width` visible columns wraps onto in
/// a terminal `cols` wide (0 if it fits on one row). Accounts for deferred-wrap:
/// a line exactly `cols` wide stays on one row until one more char is written.
pub fn wrap_rows(width: usize, cols: usize) -> usize {
    if width == 0 || cols == 0 {
        return 0;
    }
    (width - 1) / cols
}

/// Collapse a multi-line paste to a one-line chip: first line + "[+N lines]".
pub fn paste_chip(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let more = lines.len().saturating_sub(1);
    let first = lines.first().copied().unwrap_or("");
    let head: String = first.chars().take(48).collect();
    let ell = if first.chars().count() > 48 { "…" } else { "" };
    format!(
        "{}{}",
        style::cyan(&format!("{head}{ell}")),
        style::gray(&format!(" [+{more} line{}]", if more == 1 { "" } else { "s" }))
    )
}

/// Palette entries matching the buffer (only when it starts with `/`).
pub fn filter(buf: &str, commands: &[SlashCommand]) -> Vec<SlashCommand> {
    if !buf.starts_with('/') {
        return Vec::new();
    }
    let q = buf.to_lowercase();
    commands
        .iter()
        .filter(|c| std::iter::once(&c.name).chain(c.aliases.iter()).any(|n| n.to_lowercase().starts_with(&q)))
        .take(MAX_MENU)
        .copied()
        .collect()
}

/// What an input event did to the editor.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    /// Nothing visible changed.
    Idle,
    /// The line must be redrawn.
    Redraw,
    /// The user submitted this (already paste-expanded, trimmed) line.
    Submit(String),
}

/// Sentinel bytes that stand for a stored paste inside the buffer:
/// `\x01<id>\x02`. They are rendered as a chip and expanded on submit.
const CHIP_OPEN: char = '\x01';
const CHIP_CLOSE: char = '\x02';

/// The editor's pure state: buffer, palette selection, stored pastes, and how
/// much of the screen the last render occupied.
pub struct EditorState {
    label: String,
    label_cols: usize,
    commands: &'static [SlashCommand],
    /// The logical input. Multi-line pastes are stored out-of-line in `pastes`
    /// and represented here by a sentinel rendered as a chip.
    buf: String,
    sel: usize,
    /// Menu rows currently on screen.
    drawn: usize,
    pastes: HashMap<usize, String>,
    paste_seq: usize,
    /// Extra physical rows the input line wrapped onto in the LAST render. When
    /// the prompt + buffer is wider than the terminal it wraps over several
    /// rows; the cursor must go back up to the first row before clearing, or the
    /// earlier wrapped rows survive and the prompt text stacks across the screen.
    prev_rows: usize,
}

impl EditorState {
    pub fn new(label: &str, commands: &'static [SlashCommand]) -> Self {
        Self {
            label: label.to_string(),
            label_cols: style::visible_width(label),
            commands,
            buf: String::new(),
            sel: 0,
            drawn: 0,
            pastes: HashMap::new(),
            paste_seq: 0,
            prev_rows: 0,
        }
    }

    /// The raw buffer (with chip sentinels) — for tests.
    #[cfg(test)]
    pub fn buffer(&self) -> &str {
        &self.buf
    }

    /// Replace sentinels using `f(id)`.
    fn map_chips(&self, f: impl Fn(&str) -> String) -> String {
        let mut out = String::with_capacity(self.buf.len());
        let mut chars = self.buf.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch != CHIP_OPEN {
                out.push(ch);
                continue;
            }
            let mut id = String::new();
            let mut closed = false;
            for c in chars.by_ref() {
                if c == CHIP_CLOSE {
                    closed = true;
                    break;
                }
                id.push(c);
            }
            if closed {
                let text = id.parse::<usize>().ok().and_then(|n| self.pastes.get(&n)).cloned().unwrap_or_default();
                out.push_str(&f(&text));
            }
        }
        out
    }

    /// The buffer as shown: chips instead of sentinels.
    pub fn display(&self) -> String {
        self.map_chips(paste_chip)
    }

    /// The buffer as submitted: sentinels expanded back to the pasted text.
    pub fn expand(&self) -> String {
        self.map_chips(|t| t.to_string())
    }

    /// Fold a chunk of input into the buffer. A short single-line chunk (normal
    /// typing, or a scripted write) goes in literally; anything multi-line OR
    /// long (a paste) becomes a collapsed one-line chip.
    fn commit_paste(&mut self, raw: &str) {
        let text: String = raw.replace("\r\n", "\n").replace('\r', "\n").chars().filter(|c| *c != CHIP_OPEN && *c != CHIP_CLOSE).collect();
        if text.is_empty() {
            return;
        }
        if text.contains('\n') || text.chars().count() > CHIP_MIN_CHARS {
            let id = self.paste_seq;
            self.paste_seq += 1;
            self.pastes.insert(id, text);
            self.buf.push(CHIP_OPEN);
            self.buf.push_str(&id.to_string());
            self.buf.push(CHIP_CLOSE);
        } else {
            self.buf.push_str(&text);
        }
        self.sel = 0;
    }

    /// Backspace: removing a chip drops the whole paste, not one sentinel byte.
    fn backspace(&mut self) {
        if self.buf.ends_with(CHIP_CLOSE) {
            if let Some(start) = self.buf.rfind(CHIP_OPEN) {
                if let Ok(id) = self.buf[start + 1..self.buf.len() - 1].parse::<usize>() {
                    self.pastes.remove(&id);
                }
                self.buf.truncate(start);
                return;
            }
        }
        self.buf.pop();
    }

    /// Apply one input event.
    pub fn handle(&mut self, ev: InputEvent) -> Step {
        match ev {
            InputEvent::Paste(text) => {
                self.commit_paste(&text);
                Step::Redraw
            }
            InputEvent::Resize(..) => Step::Redraw,
            InputEvent::Line(l) => Step::Submit(l.trim().to_string()),
            InputEvent::Eof => Step::Submit("/exit".into()),
            InputEvent::Key(k) => self.key(k),
        }
    }

    fn key(&mut self, k: KeyEvent) -> Step {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL) && !k.modifiers.contains(KeyModifiers::ALT);
        if ctrl && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('C')) {
            return Step::Submit("/exit".into());
        }
        if ctrl && matches!(k.code, KeyCode::Char('d') | KeyCode::Char('D')) {
            return if self.buf.is_empty() { Step::Submit("/exit".into()) } else { Step::Idle };
        }
        if let Some(c) = is_text_key(&k) {
            self.buf.push(c);
            self.sel = 0;
            return Step::Redraw;
        }
        match k.code {
            KeyCode::Esc => {
                // Dismiss the palette by clearing the (slash) buffer.
                self.buf.clear();
                self.sel = 0;
                Step::Redraw
            }
            KeyCode::Enter => {
                let items = filter(&self.buf, self.commands);
                if self.drawn > 0 && !items.is_empty() {
                    let c = items[self.sel.min(items.len() - 1)];
                    if c.args.is_some() {
                        // Command takes arguments — complete it and keep editing.
                        self.buf = format!("{} ", c.name);
                        return Step::Redraw;
                    }
                    self.buf = c.name.to_string();
                }
                Step::Submit(self.expand().trim().to_string())
            }
            KeyCode::Tab => {
                let items = filter(&self.buf, self.commands);
                if !items.is_empty() {
                    let c = items[self.sel.min(items.len() - 1)];
                    self.buf = format!("{}{}", c.name, if c.args.is_some() { " " } else { "" });
                    self.sel = 0;
                }
                Step::Redraw
            }
            KeyCode::Backspace => {
                self.backspace();
                Step::Redraw
            }
            KeyCode::Up => {
                if self.drawn > 0 {
                    self.sel = (self.sel + self.drawn - 1) % self.drawn;
                }
                Step::Redraw
            }
            KeyCode::Down => {
                if self.drawn > 0 {
                    self.sel = (self.sel + 1) % self.drawn;
                }
                Step::Redraw
            }
            _ => Step::Idle,
        }
    }

    /// The escape sequence that redraws the input line (+ palette) in place.
    pub fn render(&mut self, cols: usize) -> String {
        let items = filter(&self.buf, self.commands);
        if self.sel >= items.len() {
            self.sel = items.len().saturating_sub(1);
        }
        let cols = cols.max(1);
        // Move up to the FIRST physical row the input occupied last time, then
        // return to column 0 and clear everything below — this erases any rows
        // the previous (possibly wrapped) input line spilled onto.
        let shown = self.display();
        let mut out = String::new();
        if self.prev_rows > 0 {
            out.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        out.push_str("\r\x1b[0J");
        out.push_str(&self.label);
        out.push_str(&shown);

        let input_width = self.label_cols + style::visible_width(&shown);
        self.prev_rows = wrap_rows(input_width, cols);

        if !items.is_empty() {
            let pad = items.iter().map(|c| c.label().chars().count()).max().unwrap_or(0) + 2;
            // Rows must not wrap (the cursor goes back up a fixed count), so the
            // description is cut to what fits: "  " + "▸ " + label column.
            let desc_room = cols.saturating_sub(1 + 4 + pad);
            for (i, c) in items.iter().enumerate() {
                let head = style::pad_end(&c.label(), pad);
                let desc = style::truncate_cols(c.desc, desc_room);
                let row = if i == self.sel {
                    format!("{}{}", style::cyan(&format!("▸ {}", style::bold(&head))), style::gray(&desc))
                } else {
                    format!("{}{}{}", style::gray("  "), style::cyan(&head), style::dim(&desc))
                };
                out.push_str("\n  ");
                out.push_str(&row);
            }
            // Back up to the input line, just past the buffer. (Only shown for
            // slash commands, whose buffer has no chips, so its width is the
            // true on-screen width.)
            let col = self.label_cols + style::visible_width(&self.buf);
            out.push_str(&format!("\x1b[{}A\r", items.len()));
            if col > 0 {
                out.push_str(&format!("\x1b[{col}C"));
            }
        }
        self.drawn = items.len();
        out
    }

    /// Erase the input line (+ palette) so other output can be printed in its
    /// place; the next [`EditorState::render`] draws the prompt afresh.
    pub fn erase(&mut self) -> String {
        let mut out = String::new();
        if self.prev_rows > 0 {
            out.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        out.push_str("\r\x1b[0J");
        self.prev_rows = 0;
        out
    }

    /// The final redraw on submit: the line as typed (palette erased) + newline.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if self.prev_rows > 0 {
            out.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        out.push_str("\r\x1b[0J");
        out.push_str(&self.label);
        out.push_str(&self.display());
        out.push('\n');
        self.prev_rows = 0;
        self.drawn = 0;
        out
    }
}

/// Read one line with the palette, holding an already-acquired lease.
pub async fn read_line(lease: &mut Lease, input: &Input, label: &str, commands: &'static [SlashCommand]) -> String {
    if !input.is_tty() {
        // Plain line reader: print the label only when no line is queued (a
        // piped script's lines are all there at once — echoing a prompt for
        // each would interleave noise into captured output, as in 1.x).
        let ev = match lease.try_next() {
            Some(ev) => ev,
            None => {
                out::raw(label);
                lease.next().await
            }
        };
        return match ev {
            InputEvent::Line(l) => l,
            InputEvent::Paste(p) => p,
            InputEvent::Eof => "/exit".into(),
            _ => String::new(),
        };
    }
    let mut st = EditorState::new(label, commands);
    out::raw(&st.render(input.cols() as usize));
    // Background output that arrives while we wait is printed above the
    // prompt (see `out::set_prompt_active`).
    out::set_prompt_active(true);
    let line = loop {
        let ev = tokio::select! {
            ev = lease.next() => ev,
            _ = out::deferred_ready() => {
                let text = out::take_deferred();
                if !text.is_empty() {
                    let mut s = st.erase();
                    s.push_str(&text);
                    if !text.ends_with('\n') {
                        s.push('\n');
                    }
                    s.push_str(&st.render(input.cols() as usize));
                    out::raw(&s);
                }
                continue;
            }
        };
        if let InputEvent::Resize(c, r) = ev {
            input.note_resize(c, r);
        }
        match st.handle(ev) {
            Step::Idle => {}
            Step::Redraw => out::raw(&st.render(input.cols() as usize)),
            Step::Submit(v) => {
                out::raw(&st.finish());
                break v;
            }
        }
    };
    out::set_prompt_active(false);
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    static CMDS: &[SlashCommand] = &[
        SlashCommand { name: "/help", aliases: &["/commands", "/?"], args: None, desc: "Show help" },
        SlashCommand { name: "/settings", aliases: &["/config"], args: None, desc: "Show settings" },
        SlashCommand { name: "/models", aliases: &[], args: None, desc: "Pick a model" },
        SlashCommand { name: "/model", aliases: &[], args: Some("[name|#]"), desc: "Switch model" },
        SlashCommand { name: "/exit", aliases: &["/quit"], args: None, desc: "Quit" },
    ];

    fn plain(s: &str) -> String {
        style::strip_ansi(s)
    }

    fn typed(st: &mut EditorState, s: &str) {
        for c in s.chars() {
            st.handle(InputEvent::key(KeyCode::Char(c)));
            st.render(80);
        }
    }

    #[test]
    fn wrap_rows_matches_ts() {
        assert_eq!(wrap_rows(0, 80), 0);
        assert_eq!(wrap_rows(1, 80), 0);
        assert_eq!(wrap_rows(79, 80), 0);
        // Exactly full width: deferred-wrap keeps the cursor on the same row.
        assert_eq!(wrap_rows(80, 80), 0);
        assert_eq!(wrap_rows(81, 80), 1);
        assert_eq!(wrap_rows(160, 80), 1);
        assert_eq!(wrap_rows(161, 80), 2);
        assert_eq!(wrap_rows(240, 80), 2);
        assert_eq!(wrap_rows(50, 0), 0);
    }

    #[test]
    fn paste_chip_matches_ts() {
        assert_eq!(plain(&paste_chip("import asyncio, time\nasync def main():\n    pass")), "import asyncio, time [+2 lines]");
        assert_eq!(plain(&paste_chip("first\nsecond")), "first [+1 line]");
        let long = format!("{}\nmore", "x".repeat(80));
        assert_eq!(plain(&paste_chip(&long)), format!("{}… [+1 line]", "x".repeat(48)));
        let chip = paste_chip("first line\nsecond\nthird");
        assert!(chip.contains("first line"));
        assert!(chip.contains("[+2 lines]"));
    }

    #[test]
    fn filter_by_name_and_alias() {
        let names = |b: &str| filter(b, CMDS).iter().map(|c| c.name).collect::<Vec<_>>();
        assert_eq!(names("/s"), vec!["/settings"]);
        assert_eq!(names("/con"), vec!["/settings"]); // alias /config
        assert_eq!(names("/model"), vec!["/models", "/model"]);
        assert!(names("hello").is_empty());
        assert_eq!(names("/").len(), CMDS.len());
    }

    #[test]
    fn typing_and_submit() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "hello");
        assert_eq!(st.handle(InputEvent::key(KeyCode::Enter)), Step::Submit("hello".into()));
    }

    #[test]
    fn enter_on_palette_accepts_highlighted_command() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "/se");
        assert_eq!(st.handle(InputEvent::key(KeyCode::Enter)), Step::Submit("/settings".into()));
    }

    #[test]
    fn enter_on_command_with_args_completes_instead() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "/model");
        st.handle(InputEvent::key(KeyCode::Down)); // /models → /model
        st.render(80);
        assert_eq!(st.handle(InputEvent::key(KeyCode::Enter)), Step::Redraw);
        assert_eq!(st.buffer(), "/model ");
        st.render(80);
        typed(&mut st, "3");
        assert_eq!(st.handle(InputEvent::key(KeyCode::Enter)), Step::Submit("/model 3".into()));
    }

    #[test]
    fn tab_completes_and_up_wraps() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "/");
        st.handle(InputEvent::key(KeyCode::Up)); // wraps to the last entry
        st.render(80);
        st.handle(InputEvent::key(KeyCode::Tab));
        assert_eq!(st.buffer(), "/exit");
    }

    #[test]
    fn esc_clears_and_backspace_edits() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "/hel");
        st.handle(InputEvent::key(KeyCode::Backspace));
        assert_eq!(st.buffer(), "/he");
        st.handle(InputEvent::key(KeyCode::Esc));
        assert_eq!(st.buffer(), "");
    }

    #[test]
    fn multiline_paste_becomes_chip_and_expands_on_submit() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "fix ");
        st.handle(InputEvent::Paste("line one\r\nline two\nline three".into()));
        let shown = plain(&st.display());
        assert_eq!(shown, "fix line one [+2 lines]");
        assert_eq!(st.handle(InputEvent::key(KeyCode::Enter)), Step::Submit("fix line one\nline two\nline three".into()));
    }

    #[test]
    fn backspace_removes_whole_chip() {
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "a");
        st.handle(InputEvent::Paste("x\ny".into()));
        st.handle(InputEvent::key(KeyCode::Backspace));
        assert_eq!(st.buffer(), "a");
        assert_eq!(st.expand(), "a");
    }

    #[test]
    fn short_paste_is_literal_long_paste_is_chip() {
        let mut st = EditorState::new("❯ ", CMDS);
        st.handle(InputEvent::Paste("short".into()));
        assert_eq!(st.buffer(), "short");
        st.handle(InputEvent::Paste("y".repeat(41)));
        assert!(st.buffer().contains(CHIP_OPEN));
        assert_eq!(st.expand(), format!("short{}", "y".repeat(41)));
    }

    #[test]
    fn ctrl_c_and_ctrl_d_exit() {
        let mut st = EditorState::new("❯ ", CMDS);
        assert_eq!(st.handle(InputEvent::ctrl('d')), Step::Submit("/exit".into()));
        let mut st = EditorState::new("❯ ", CMDS);
        typed(&mut st, "x");
        assert_eq!(st.handle(InputEvent::ctrl('d')), Step::Idle);
        assert_eq!(st.handle(InputEvent::ctrl('c')), Step::Submit("/exit".into()));
    }

    #[test]
    fn render_erases_wrapped_rows_and_positions_cursor() {
        style::set_enabled(false);
        let mut st = EditorState::new("> ", CMDS);
        st.handle(InputEvent::Paste("a".repeat(30)));
        let first = st.render(20); // 32 columns on a 20-col terminal → 1 extra row
        assert!(first.starts_with("\r\x1b[0J> "));
        let second = st.render(20);
        assert!(second.starts_with("\x1b[1A\r\x1b[0J"), "must move up over the wrapped row: {second:?}");

        let mut st = EditorState::new("> ", CMDS);
        typed(&mut st, "/m");
        let r = st.render(80);
        // Two menu rows, then back up two rows and right past "> /m".
        assert!(r.ends_with("\x1b[2A\r\x1b[4C"), "{r:?}");
        let fin = st.finish();
        assert_eq!(fin, "\r\x1b[0J> /m\n");
    }

    #[test]
    fn menu_rows_never_wrap() {
        style::set_enabled(false);
        let mut st = EditorState::new("> ", CMDS);
        typed(&mut st, "/");
        let r = st.render(24);
        for row in r.split('\n').skip(1) {
            let row = row.split('\x1b').next().unwrap_or("");
            assert!(style::visible_width(row) < 24, "row too wide: {row:?}");
        }
    }

    #[tokio::test]
    async fn read_line_non_tty_queues_every_line() {
        let (input, tx) = Input::from_channel(false);
        tx.send(InputEvent::Line("first".into())).unwrap();
        tx.send(InputEvent::Line("second".into())).unwrap();
        drop(tx);
        let mut lease = input.lease().await;
        assert_eq!(read_line(&mut lease, &input, "> ", CMDS).await, "first");
        assert_eq!(read_line(&mut lease, &input, "> ", CMDS).await, "second");
        assert_eq!(read_line(&mut lease, &input, "> ", CMDS).await, "/exit");
    }

    #[tokio::test]
    async fn read_line_tty_scripted_write() {
        let (input, tx) = Input::from_channel(true);
        tx.send(InputEvent::Paste("hi".into())).unwrap();
        tx.send(InputEvent::key(KeyCode::Enter)).unwrap();
        let mut lease = input.lease().await;
        assert_eq!(read_line(&mut lease, &input, "> ", CMDS).await, "hi");
    }
}
