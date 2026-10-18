//! Arrow-key list picker (port of src/cli/prompt.ts `selectFromList`):
//! ↑/↓ (or Ctrl-P/Ctrl-N) to move, Enter to choose, Esc/Ctrl-C/q to cancel.
//! Redraws in place. Like every other consumer it only borrows the input
//! stream; it never switches the terminal mode.

use crossterm::event::{KeyCode, KeyModifiers};

use super::input::{Input, InputEvent};
use super::{out, style};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SelectItem {
    pub label: String,
    pub hint: Option<String>,
    pub current: bool,
}

/// Result of one event.
#[derive(Debug, PartialEq, Eq)]
pub enum PickStep {
    Idle,
    Redraw,
    /// Chosen index, or `None` when cancelled.
    Done(Option<usize>),
}

/// Pure picker state (rendering + key handling), unit-testable.
pub struct PickerState {
    title: String,
    items: Vec<SelectItem>,
    sel: usize,
    /// Lines on screen after the last render.
    drawn: usize,
}

impl PickerState {
    pub fn new(title: &str, items: Vec<SelectItem>, initial: usize) -> Self {
        let sel = initial.min(items.len().saturating_sub(1));
        Self { title: title.to_string(), items, sel, drawn: 0 }
    }

    #[cfg(test)]
    pub fn selected(&self) -> usize {
        self.sel
    }

    pub fn handle(&mut self, ev: &InputEvent) -> PickStep {
        let n = self.items.len();
        if n == 0 {
            return PickStep::Done(None);
        }
        match ev {
            InputEvent::Eof => PickStep::Done(None),
            InputEvent::Resize(..) => PickStep::Redraw,
            InputEvent::Key(k) => {
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                match k.code {
                    KeyCode::Up => {
                        self.sel = (self.sel + n - 1) % n;
                        PickStep::Redraw
                    }
                    KeyCode::Char('p') if ctrl => {
                        self.sel = (self.sel + n - 1) % n;
                        PickStep::Redraw
                    }
                    KeyCode::Down => {
                        self.sel = (self.sel + 1) % n;
                        PickStep::Redraw
                    }
                    KeyCode::Char('n') if ctrl => {
                        self.sel = (self.sel + 1) % n;
                        PickStep::Redraw
                    }
                    KeyCode::Enter => PickStep::Done(Some(self.sel)),
                    KeyCode::Esc => PickStep::Done(None),
                    KeyCode::Char('c') if ctrl => PickStep::Done(None),
                    KeyCode::Char('q') if !ctrl => PickStep::Done(None),
                    _ => PickStep::Idle,
                }
            }
            _ => PickStep::Idle,
        }
    }

    /// Redraw in place. Only a window of items that fits the terminal is shown
    /// (a list taller than the screen would scroll and break the in-place
    /// redraw), and no line may wrap for the same reason.
    pub fn render(&mut self, cols: usize, rows: usize) -> String {
        let cols = cols.max(10);
        let room = rows.saturating_sub(3).max(3);
        let n = self.items.len();
        let (start, end) = if n <= room {
            (0, n)
        } else {
            let start = self.sel.saturating_sub(room / 2).min(n - room);
            (start, start + room)
        };
        let mut lines = vec![style::bold(&style::truncate_cols(&self.title, cols - 1))];
        for i in start..end {
            let it = &self.items[i];
            let dot = if it.current { style::green("●") } else { style::gray("○") };
            // "❯ ● " prefix = 4 columns.
            let avail = cols.saturating_sub(5);
            let label = style::truncate_cols(&it.label, avail);
            let hint_room = avail.saturating_sub(style::visible_width(&label) + 2);
            let hint = it
                .hint
                .as_deref()
                .filter(|h| !h.is_empty() && hint_room > 1)
                .map(|h| style::gray(&format!("  {}", style::truncate_cols(h, hint_room))))
                .unwrap_or_default();
            let text = format!("{label}{hint}");
            lines.push(if i == self.sel {
                format!("{} {dot} {}", style::cyan("❯"), style::bold(&text))
            } else {
                format!("  {dot} {text}")
            });
        }
        let more = if n > end - start { format!(" · {}/{}", self.sel + 1, n) } else { String::new() };
        lines.push(style::gray(&style::truncate_cols(&format!("  ↑/↓ move · Enter switch · Esc cancel{more}"), cols - 1)));
        let mut s = String::new();
        if self.drawn > 0 {
            s.push_str(&format!("\x1b[{}A", self.drawn - 1));
        }
        s.push_str("\r\x1b[0J");
        s.push_str(&lines.join("\n"));
        self.drawn = lines.len();
        s
    }
}

/// Show the picker and wait for a choice. `None` when cancelled, when stdin is
/// not a TTY, or when there is nothing to pick.
pub async fn select_from_list(input: &Input, title: &str, items: Vec<SelectItem>, initial: usize) -> Option<usize> {
    if !input.is_tty() || items.is_empty() {
        return None;
    }
    let mut st = PickerState::new(title, items, initial);
    let mut lease = input.lease().await;
    out::raw(&st.render(input.cols() as usize, input.rows() as usize));
    loop {
        let ev = lease.next().await;
        if let InputEvent::Resize(c, r) = ev {
            input.note_resize(c, r);
        }
        match st.handle(&ev) {
            PickStep::Idle => {}
            PickStep::Redraw => out::raw(&st.render(input.cols() as usize, input.rows() as usize)),
            PickStep::Done(choice) => {
                out::raw("\n");
                return choice;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: usize) -> Vec<SelectItem> {
        (0..n).map(|i| SelectItem { label: format!("model-{i}"), hint: Some("cloud".into()), current: i == 1 }).collect()
    }

    #[test]
    fn moves_wraps_and_chooses() {
        let mut p = PickerState::new("Pick", items(3), 1);
        assert_eq!(p.handle(&InputEvent::key(KeyCode::Down)), PickStep::Redraw);
        assert_eq!(p.handle(&InputEvent::key(KeyCode::Down)), PickStep::Redraw);
        assert_eq!(p.selected(), 0);
        assert_eq!(p.handle(&InputEvent::ctrl('p')), PickStep::Redraw);
        assert_eq!(p.selected(), 2);
        assert_eq!(p.handle(&InputEvent::key(KeyCode::Enter)), PickStep::Done(Some(2)));
    }

    #[test]
    fn cancels() {
        let mut p = PickerState::new("Pick", items(3), 0);
        assert_eq!(p.handle(&InputEvent::key(KeyCode::Esc)), PickStep::Done(None));
        assert_eq!(p.handle(&InputEvent::key(KeyCode::Char('q'))), PickStep::Done(None));
        assert_eq!(p.handle(&InputEvent::ctrl('c')), PickStep::Done(None));
        assert_eq!(p.handle(&InputEvent::Eof), PickStep::Done(None));
    }

    #[test]
    fn renders_in_place_and_windows_long_lists() {
        style::set_enabled(false);
        let mut p = PickerState::new("Pick", items(3), 0);
        let first = p.render(80, 24);
        assert!(first.starts_with("\r\x1b[0JPick\n❯ ○ model-0  cloud"));
        let second = p.render(80, 24);
        assert!(second.starts_with("\x1b[4A\r\x1b[0J"), "{second:?}");

        let mut p = PickerState::new("Pick", items(40), 30);
        let r = p.render(80, 12);
        assert_eq!(r.split('\n').count(), 1 + 9 + 1);
        assert!(r.contains("model-30"));
        assert!(r.contains("31/40"));
    }

    #[tokio::test]
    async fn select_reads_from_the_shared_input() {
        let (input, tx) = Input::from_channel(true);
        tx.send(InputEvent::key(KeyCode::Down)).unwrap();
        tx.send(InputEvent::key(KeyCode::Enter)).unwrap();
        assert_eq!(select_from_list(&input, "Pick", items(3), 0).await, Some(1));
        let (input, _tx) = Input::from_channel(false);
        assert_eq!(select_from_list(&input, "Pick", items(3), 0).await, None);
    }
}
