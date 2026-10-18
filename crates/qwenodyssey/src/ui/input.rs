//! The ONE input owner of the interactive session.
//!
//! A dedicated OS thread reads terminal events (keys, bracketed paste, resize)
//! and forwards them into a channel. Nothing else in the process reads stdin.
//! Consumers — the prompt editor, the model picker, the turn-time
//! interjection capture, the y/N confirmer — take turns: each holds a
//! [`Lease`] (an async mutex guard over the receiver) while it is active and
//! pulls events from it. Because the reader never stops and never toggles the
//! terminal mode, there is no window in which keystrokes go nowhere, and a turn
//! that is waiting on the network can always see Esc.
//!
//! When stdin is not a TTY (pipes, tests) the thread is a plain line reader
//! that queues EVERY line as it arrives (the v0.4.1 fix: readline used to drop
//! lines that arrived while no question was pending, so only the first prompt
//! of a piped script ran).

use std::collections::VecDeque;
use std::io::BufRead;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, OwnedMutexGuard};

/// One unit of user input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputEvent {
    /// A key press (releases are filtered out; Windows reports both).
    Key(KeyEvent),
    /// Text pasted as one unit: a bracketed paste, or a key flood the reader
    /// recognised as a paste (terminals without bracketed paste).
    Paste(String),
    /// The terminal was resized to (cols, rows).
    Resize(u16, u16),
    /// A whole line from a non-TTY stdin.
    Line(String),
    /// Input is gone (stdin closed / the reader failed).
    Eof,
}

impl InputEvent {
    /// Convenience constructor for tests: a plain key press.
    pub fn key(code: KeyCode) -> Self {
        InputEvent::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    #[cfg(test)]
    pub fn ctrl(c: char) -> Self {
        InputEvent::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }
}

/// Does this key press insert text (as opposed to a control/meta chord)?
/// AltGr arrives as Ctrl+Alt on Windows and still types a character.
pub fn is_text_key(k: &KeyEvent) -> Option<char> {
    let KeyCode::Char(c) = k.code else { return None };
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    let alt = k.modifiers.contains(KeyModifiers::ALT);
    if ctrl ^ alt {
        return None;
    }
    if c.is_control() {
        return None;
    }
    Some(c)
}

struct Inner {
    rx: UnboundedReceiver<InputEvent>,
    /// Events a consumer looked at but handed back (see [`Lease::unread`]).
    stash: VecDeque<InputEvent>,
}

/// The shared input source.
pub struct Input {
    inner: Arc<Mutex<Inner>>,
    tty: bool,
    cols: AtomicU16,
    rows: AtomicU16,
}

/// Exclusive access to the input stream for one consumer at a time.
pub struct Lease {
    guard: OwnedMutexGuard<Inner>,
}

impl Lease {
    /// Next event; `Eof` forever once input is gone.
    pub async fn next(&mut self) -> InputEvent {
        if let Some(ev) = self.guard.stash.pop_front() {
            return ev;
        }
        self.guard.rx.recv().await.unwrap_or(InputEvent::Eof)
    }

    /// An already-available event, without waiting.
    pub fn try_next(&mut self) -> Option<InputEvent> {
        if let Some(ev) = self.guard.stash.pop_front() {
            return Some(ev);
        }
        self.guard.rx.try_recv().ok()
    }

    /// Hand an event back so the next consumer sees it first (type-ahead
    /// during a model probe is kept for the prompt instead of being lost).
    pub fn unread(&mut self, ev: InputEvent) {
        self.guard.stash.push_back(ev);
    }

    /// Hand an event back AHEAD of anything already handed back (text typed
    /// before a stashed `Eof` must still reach the next consumer first).
    pub fn unread_front(&mut self, ev: InputEvent) {
        self.guard.stash.push_front(ev);
    }
}

impl Input {
    fn with_receiver(rx: UnboundedReceiver<InputEvent>, tty: bool) -> Arc<Self> {
        let (cols, rows) = if tty { crossterm::terminal::size().unwrap_or((80, 24)) } else { (80, 24) };
        Arc::new(Self {
            inner: Arc::new(Mutex::new(Inner { rx, stash: VecDeque::new() })),
            tty,
            cols: AtomicU16::new(cols.max(1)),
            rows: AtomicU16::new(rows.max(1)),
        })
    }

    /// Start the terminal event reader thread. Call after raw mode is on.
    pub fn start_tty() -> Arc<Self> {
        let (tx, rx) = unbounded_channel();
        std::thread::Builder::new()
            .name("qo-input".into())
            .spawn(move || tty_reader(tx))
            .ok();
        Self::with_receiver(rx, true)
    }

    /// Start the non-TTY line reader thread.
    pub fn start_lines() -> Arc<Self> {
        let (tx, rx) = unbounded_channel();
        std::thread::Builder::new()
            .name("qo-input-lines".into())
            .spawn(move || line_reader(tx))
            .ok();
        Self::with_receiver(rx, false)
    }

    /// An input fed from a channel (tests, scripted runs).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_channel(tty: bool) -> (Arc<Self>, UnboundedSender<InputEvent>) {
        let (tx, rx) = unbounded_channel();
        let me = Arc::new(Self {
            inner: Arc::new(Mutex::new(Inner { rx, stash: VecDeque::new() })),
            tty,
            cols: AtomicU16::new(80),
            rows: AtomicU16::new(24),
        });
        (me, tx)
    }

    /// Interactive terminal (raw keys) vs plain lines.
    pub fn is_tty(&self) -> bool {
        self.tty
    }

    /// Wait for exclusive access to the input stream.
    pub async fn lease(&self) -> Lease {
        Lease { guard: self.inner.clone().lock_owned().await }
    }

    /// Terminal width in columns (kept current from resize events).
    pub fn cols(&self) -> u16 {
        self.cols.load(Ordering::Relaxed)
    }

    pub fn rows(&self) -> u16 {
        self.rows.load(Ordering::Relaxed)
    }

    /// Record a resize (consumers call this when they see `Resize`).
    pub fn note_resize(&self, cols: u16, rows: u16) {
        self.cols.store(cols.max(1), Ordering::Relaxed);
        self.rows.store(rows.max(1), Ordering::Relaxed);
    }
}

/// Max events gathered into one batch before it is flushed.
const MAX_BATCH: usize = 8192;

fn tty_reader(tx: UnboundedSender<InputEvent>) {
    loop {
        let first = match event::read() {
            Ok(ev) => ev,
            Err(_) => {
                let _ = tx.send(InputEvent::Eof);
                return;
            }
        };
        // Everything already queued behind this event arrived together — a
        // paste on a terminal without bracketed paste, or a scripted write.
        let mut batch = vec![first];
        while batch.len() < MAX_BATCH && matches!(event::poll(Duration::ZERO), Ok(true)) {
            match event::read() {
                Ok(ev) => batch.push(ev),
                Err(_) => break,
            }
        }
        for ev in coalesce(batch) {
            if tx.send(ev).is_err() {
                return; // nobody is listening any more
            }
        }
    }
}

fn line_reader(tx: UnboundedSender<InputEvent>) {
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut buf = String::new();
    loop {
        buf.clear();
        match lock.read_line(&mut buf) {
            Ok(0) | Err(_) => {
                let _ = tx.send(InputEvent::Eof);
                return;
            }
            Ok(_) => {
                let line = buf.trim_end_matches(['\n', '\r']).to_string();
                if tx.send(InputEvent::Line(line)).is_err() {
                    return;
                }
            }
        }
    }
}

/// Text a key contributes to a flood (`None` = not a text key).
fn flood_text(k: &KeyEvent) -> Option<char> {
    match k.code {
        KeyCode::Enter => Some('\n'),
        KeyCode::Tab => Some('\t'),
        _ => is_text_key(k),
    }
}

/// Turn a batch of raw terminal events into input events.
///
/// Key releases are dropped. A batch made only of text keys (chars, Enter,
/// Tab) with two or more presses is a paste or a scripted write and becomes a
/// single [`InputEvent::Paste`] — EXCEPT that a lone trailing Enter stays a
/// real Enter, so `"hi\r"` written at once still submits "hi". A newline in
/// the middle of the batch can only come from a paste, so those newlines stay
/// literal and nothing is submitted (the 1.x flood ran every pasted line as
/// its own prompt).
pub fn coalesce(batch: Vec<Event>) -> Vec<InputEvent> {
    let mut keys: Vec<KeyEvent> = Vec::new();
    let mut only_text_keys = true;
    for ev in &batch {
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => {
                if flood_text(k).is_none() {
                    only_text_keys = false;
                }
                keys.push(*k);
            }
            Event::Key(_) => {}
            Event::Paste(_) | Event::Resize(..) => only_text_keys = false,
            _ => {}
        }
    }

    if only_text_keys && keys.len() >= 2 {
        let mut text: String = keys.iter().filter_map(flood_text).collect();
        let inner_newline = text.trim_end_matches('\n').contains('\n');
        let trailing_enter = matches!(keys.last(), Some(k) if k.code == KeyCode::Enter);
        if !inner_newline && trailing_enter {
            text.pop();
            let mut out = Vec::new();
            if !text.is_empty() {
                out.push(InputEvent::Paste(text));
            }
            out.push(InputEvent::key(KeyCode::Enter));
            return out;
        }
        return vec![InputEvent::Paste(text)];
    }

    batch
        .into_iter()
        .filter_map(|ev| match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => Some(InputEvent::Key(k)),
            Event::Paste(s) => Some(InputEvent::Paste(s)),
            Event::Resize(c, r) => Some(InputEvent::Resize(c, r)),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventState;

    fn press(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    fn release(code: KeyCode) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        })
    }

    #[test]
    fn single_key_passes_through_and_releases_drop() {
        let out = coalesce(vec![press(KeyCode::Char('a')), release(KeyCode::Char('a'))]);
        assert_eq!(out, vec![InputEvent::key(KeyCode::Char('a'))]);
    }

    #[test]
    fn scripted_line_with_trailing_enter_still_submits() {
        let out = coalesce(vec![press(KeyCode::Char('h')), press(KeyCode::Char('i')), press(KeyCode::Enter)]);
        assert_eq!(out, vec![InputEvent::Paste("hi".into()), InputEvent::key(KeyCode::Enter)]);
    }

    #[test]
    fn flood_with_inner_newline_is_one_paste() {
        let out = coalesce(vec![
            press(KeyCode::Char('a')),
            press(KeyCode::Enter),
            press(KeyCode::Char('b')),
            press(KeyCode::Enter),
        ]);
        assert_eq!(out, vec![InputEvent::Paste("a\nb\n".into())]);
    }

    #[test]
    fn mixed_batch_is_not_a_paste() {
        let out = coalesce(vec![press(KeyCode::Char('a')), press(KeyCode::Esc)]);
        assert_eq!(out, vec![InputEvent::key(KeyCode::Char('a')), InputEvent::key(KeyCode::Esc)]);
    }

    #[test]
    fn ctrl_chords_are_not_text() {
        let k = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(is_text_key(&k), None);
        let altgr = KeyEvent::new(KeyCode::Char('@'), KeyModifiers::CONTROL | KeyModifiers::ALT);
        assert_eq!(is_text_key(&altgr), Some('@'));
    }

    #[tokio::test]
    async fn lease_unread_replays_first() {
        let (input, tx) = Input::from_channel(true);
        tx.send(InputEvent::key(KeyCode::Char('x'))).unwrap();
        {
            let mut l = input.lease().await;
            let ev = l.next().await;
            l.unread(ev);
        }
        let mut l = input.lease().await;
        assert_eq!(l.next().await, InputEvent::key(KeyCode::Char('x')));
        drop(tx);
        assert_eq!(l.next().await, InputEvent::Eof);
    }
}
