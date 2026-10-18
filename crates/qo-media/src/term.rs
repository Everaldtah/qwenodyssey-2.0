//! Terminal plumbing for the commands that own the screen while they run
//! (`live`, `mictest`, `miclevel`): raw-mode key input restored on exit *and*
//! on panic, a thinking spinner with a live token meter, and line printing that
//! works whether or not raw mode is on.
//!
//! Output goes through `print!` (never a raw stdout handle) so the test
//! harness captures it.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle as ThreadHandle;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::Stylize;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Print one line. `\r\n` so it also lands at column 0 under raw mode
/// (which turns off output post-processing on Unix).
pub(crate) fn say(s: impl std::fmt::Display) {
    print!("{s}\r\n");
    flush();
}

/// Print without a newline (prompts, in-place meter redraws).
pub(crate) fn put(s: impl std::fmt::Display) {
    print!("{s}");
    flush();
}

pub(crate) fn flush() {
    let _ = std::io::stdout().flush();
}

/// Clear the current line (`\r\x1b[2K`).
pub(crate) const CLEAR_LINE: &str = "\r\x1b[2K";

/// chalk.gray (bright black).
pub(crate) fn gray(s: &str) -> String {
    s.dark_grey().to_string()
}

/// The keys the media commands react to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Key {
    Char(char),
    Enter,
    Esc,
    CtrlC,
}

impl Key {
    /// q / Esc / Ctrl-C.
    pub fn is_quit(self) -> bool {
        matches!(self, Key::Char('q') | Key::Char('Q') | Key::Esc | Key::CtrlC)
    }
}

fn map_key(code: KeyCode, mods: KeyModifiers) -> Option<Key> {
    match code {
        KeyCode::Char('c') | KeyCode::Char('C') if mods.contains(KeyModifiers::CONTROL) => Some(Key::CtrlC),
        KeyCode::Char(c) => Some(Key::Char(c.to_ascii_lowercase())),
        KeyCode::Enter => Some(Key::Enter),
        KeyCode::Esc => Some(Key::Esc),
        _ => None,
    }
}

/// Raw-mode keyboard for one command run. Created with [`Keys::start`];
/// dropping it (normal exit, `?` early return, or a panic unwinding through
/// the command) stops the reader thread and restores the terminal.
///
/// When stdin is not a TTY nothing is read and [`Keys::next`] never resolves
/// (the TS skipped key handling the same way).
pub(crate) struct Keys {
    rx: Option<mpsc::UnboundedReceiver<Key>>,
    stop: Arc<AtomicBool>,
    thread: Option<ThreadHandle<()>>,
    raw: bool,
}

impl Keys {
    pub fn start() -> Keys {
        let stop = Arc::new(AtomicBool::new(false));
        if !std::io::stdin().is_terminal() || crossterm::terminal::enable_raw_mode().is_err() {
            return Keys { rx: None, stop, thread: None, raw: false };
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let flag = Arc::clone(&stop);
        // crossterm's blocking reader on its own thread, polled with a short
        // timeout so `stop` is noticed within 100 ms.
        let thread = std::thread::Builder::new()
            .name("qo-media-keys".into())
            .spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    match event::poll(Duration::from_millis(100)) {
                        Ok(true) => match event::read() {
                            // Windows reports Press and Release; act on presses only.
                            Ok(Event::Key(k)) if k.kind != KeyEventKind::Release => {
                                if let Some(key) = map_key(k.code, k.modifiers) {
                                    if tx.send(key).is_err() {
                                        break;
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(_) => break,
                        },
                        Ok(false) => {}
                        Err(_) => break,
                    }
                }
            })
            .ok();
        Keys { rx: Some(rx), stop, thread, raw: true }
    }

    /// Is a keyboard attached (stdin is a TTY in raw mode)?
    pub fn interactive(&self) -> bool {
        self.rx.is_some()
    }

    /// The next key press; pends forever when there is no keyboard.
    pub async fn next(&mut self) -> Key {
        match self.rx.as_mut() {
            Some(rx) => match rx.recv().await {
                Some(k) => k,
                None => std::future::pending().await,
            },
            None => std::future::pending().await,
        }
    }

    /// Stop reading and leave raw mode (idempotent).
    pub fn restore(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.rx = None;
        if self.raw {
            let _ = crossterm::terminal::disable_raw_mode();
            self.raw = false;
        }
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Whimsical gerunds shown while the model is working (one per request).
const THINKING_WORDS: [&str; 18] = [
    "Herding", "Pondering", "Noodling", "Conjuring", "Percolating", "Ruminating", "Synthesizing", "Cogitating",
    "Tinkering", "Brewing", "Scheming", "Musing", "Crunching", "Divining", "Untangling", "Marinating", "Spelunking",
    "Wrangling",
];

pub(crate) fn thinking_word() -> &'static str {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    THINKING_WORDS[nanos as usize % THINKING_WORDS.len()]
}

/// "9601" → "9.6k", "512" → "512", "12000" → "12k".
pub(crate) fn format_tokens(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = format!("{:.1}", n as f64 / 1000.0);
    format!("{}k", s.strip_suffix(".0").unwrap_or(&s))
}

/// "85" → "1m 25s", "42" → "42s".
pub(crate) fn format_elapsed(seconds: u64) -> String {
    if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

#[derive(Default)]
struct SpinState {
    out_tokens: u64,
    first_token_at: Option<Instant>,
}

/// Animated "Pondering… (3s · ↑ 0 ↓ 42 tokens · 18 tok/s)" line while the
/// model works (port of render.ts `Spinner`, ↑ is always 0 in live mode).
/// No-op when stdout is not a TTY.
pub(crate) struct Spinner {
    state: Arc<Mutex<SpinState>>,
    task: Option<JoinHandle<()>>,
}

impl Spinner {
    pub fn begin(word: &str) -> Spinner {
        let state = Arc::new(Mutex::new(SpinState::default()));
        if !std::io::stdout().is_terminal() {
            return Spinner { state, task: None };
        }
        let word = word.to_string();
        let st = Arc::clone(&state);
        let task = tokio::spawn(async move {
            const FRAMES: [&str; 6] = ["·", "∘", "○", "◌", "○", "∘"];
            let started = Instant::now();
            let mut frame = 0usize;
            let mut tick = tokio::time::interval(Duration::from_millis(140));
            loop {
                tick.tick().await;
                let (out, first) = {
                    let s = st.lock().unwrap_or_else(PoisonError::into_inner);
                    (s.out_tokens, s.first_token_at)
                };
                let mut parts = vec![
                    format_elapsed(started.elapsed().as_secs()),
                    format!("↑ 0{} tokens", if out > 0 { format!(" ↓ {}", format_tokens(out)) } else { String::new() }),
                ];
                if let Some(t0) = first {
                    let secs = t0.elapsed().as_secs_f64();
                    if out >= 2 && secs > 0.2 {
                        let r = out as f64 / secs;
                        parts.push(if r >= 10.0 { format!("{} tok/s", r.round()) } else { format!("{r:.1} tok/s") });
                    }
                }
                let glyph = FRAMES[frame % FRAMES.len()];
                frame += 1;
                put(format!(
                    "{CLEAR_LINE}{} {}{}",
                    glyph.dark_magenta(),
                    word.as_str().dark_magenta(),
                    format!("… ({})", parts.join(" · ")).dark_grey()
                ));
            }
        });
        Spinner { state, task: Some(task) }
    }

    /// Set the absolute streamed-output token count.
    pub fn set_out(&self, tokens: u64) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if s.out_tokens == 0 && tokens > 0 {
            s.first_token_at = Some(Instant::now());
        }
        s.out_tokens = tokens;
    }

    /// A handle the stream callback can update from inside `on_chunk`.
    pub fn counter(&self) -> SpinCounter {
        SpinCounter(Arc::clone(&self.state))
    }

    pub fn stop(&mut self) {
        if let Some(t) = self.task.take() {
            t.abort();
            put(CLEAR_LINE);
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Cloneable, `Send` handle to bump the spinner's output-token count.
#[derive(Clone)]
pub(crate) struct SpinCounter(Arc<Mutex<SpinState>>);

impl SpinCounter {
    pub fn bump(&self, tokens: u64) {
        let mut s = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if s.out_tokens == 0 {
            s.first_token_at = Some(Instant::now());
        }
        s.out_tokens += tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping() {
        assert_eq!(map_key(KeyCode::Char('c'), KeyModifiers::CONTROL), Some(Key::CtrlC));
        assert_eq!(map_key(KeyCode::Char('C'), KeyModifiers::SHIFT), Some(Key::Char('c')));
        assert_eq!(map_key(KeyCode::Enter, KeyModifiers::NONE), Some(Key::Enter));
        assert_eq!(map_key(KeyCode::Esc, KeyModifiers::NONE), Some(Key::Esc));
        assert_eq!(map_key(KeyCode::Up, KeyModifiers::NONE), None);
        assert!(Key::Char('q').is_quit() && Key::Esc.is_quit() && Key::CtrlC.is_quit());
        assert!(!Key::Char('c').is_quit() && !Key::Enter.is_quit());
    }

    #[test]
    fn token_and_time_formatting() {
        assert_eq!(format_tokens(512), "512");
        assert_eq!(format_tokens(9601), "9.6k");
        assert_eq!(format_tokens(12000), "12k");
        assert_eq!(format_elapsed(42), "42s");
        assert_eq!(format_elapsed(85), "1m 25s");
        assert!(THINKING_WORDS.contains(&thinking_word()));
    }

    #[tokio::test]
    async fn spinner_counts_without_a_tty() {
        let mut s = Spinner::begin("Testing");
        let c = s.counter();
        c.bump(3);
        c.bump(2);
        assert_eq!(s.state.lock().unwrap().out_tokens, 5);
        s.set_out(9);
        assert_eq!(s.state.lock().unwrap().out_tokens, 9);
        s.stop();
    }
}
