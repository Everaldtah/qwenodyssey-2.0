//! The single UI writer. Everything the interactive CLI prints goes through
//! here, for two reasons:
//!
//! 1. **Raw mode.** The chat enables raw mode once for the whole session. On
//!    Unix that turns off output post-processing, so a bare `"\n"` moves down
//!    without returning to column 0 and text staircases across the screen.
//!    While raw mode is on, every lone `"\n"` is written as `"\r\n"`.
//! 2. **The bottom overlay line.** The spinner ("· Herding… (3s · ↑ 1.2k …)")
//!    and the "btw ›" aside the user types during a turn both live on the last
//!    line. Printing a line clears the overlay first and redraws it afterwards,
//!    so output and the live status never overwrite each other.
//!
//! 3. **The prompt.** While the prompt editor is waiting for a line, a line
//!    printed by a background task (the GitHub login probe, an LM Studio load
//!    notice) is held back and handed to the editor, which erases its input
//!    line, prints it, and redraws the prompt below it — instead of the text
//!    landing under the input line and the prompt stacking on screen.
//!
//! When stdout is not a terminal the overlay is never drawn (status lines would
//! pollute captured output), matching the TS spinner's non-TTY no-op.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};

/// Who currently owns the overlay line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    Spinner,
    Aside,
}

#[derive(Default)]
struct State {
    /// Owner + text of the overlay line, when one is shown.
    overlay: Option<(Overlay, String)>,
    /// A prompt editor is active: prints are deferred to it.
    prompt_active: bool,
    /// Text printed while the prompt was active, waiting for the editor.
    deferred: String,
}

/// Wakes the active prompt editor when deferred output arrives.
static DEFERRED_READY: LazyLock<tokio::sync::Notify> = LazyLock::new(tokio::sync::Notify::new);

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));
static RAW: AtomicBool = AtomicBool::new(false);
static STDOUT_TTY: LazyLock<bool> = LazyLock::new(|| std::io::stdout().is_terminal());

fn state() -> MutexGuard<'static, State> {
    // A panic while printing must not wedge every later print.
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Is stdout an interactive terminal?
pub fn is_tty() -> bool {
    *STDOUT_TTY
}

/// Tell the writer whether raw mode is active (set by the terminal guard).
pub fn set_raw(on: bool) {
    RAW.store(on, Ordering::SeqCst);
}

pub fn raw_mode() -> bool {
    RAW.load(Ordering::SeqCst)
}

/// `"\n"` → `"\r\n"` (a `"\r\n"` already present is left alone).
pub fn normalize_newlines(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut prev = '\0';
    for ch in s.chars() {
        if ch == '\n' && prev != '\r' {
            out.push('\r');
        }
        out.push(ch);
        prev = ch;
    }
    out
}

fn emit(s: &str) {
    // Unit tests drive the UI headlessly; keep their output clean.
    if cfg!(test) {
        return;
    }
    let text = if raw_mode() { normalize_newlines(s) } else { s.to_string() };
    let mut so = std::io::stdout().lock();
    // A closed stdout (e.g. `| head`) must never crash the agent mid-turn.
    let _ = so.write_all(text.as_bytes());
    let _ = so.flush();
}

/// Print text that ends a line (or several). The overlay is cleared first and
/// redrawn below the new text.
pub fn print(s: &str) {
    let mut st = state();
    if st.prompt_active {
        st.deferred.push_str(s);
        drop(st);
        DEFERRED_READY.notify_one();
        return;
    }
    let mut buf = String::new();
    let overlay = st.overlay.as_ref().filter(|_| is_tty());
    if overlay.is_some() {
        buf.push_str("\r\x1b[2K");
    }
    buf.push_str(s);
    if let Some((_, line)) = overlay {
        if s.ends_with('\n') {
            buf.push_str(line);
        }
    }
    emit(&buf);
    drop(st);
}

/// `print(s + "\n")` — the `console.log` of the Rust CLI.
pub fn line(s: &str) {
    let mut t = String::with_capacity(s.len() + 1);
    t.push_str(s);
    t.push('\n');
    print(&t);
}

/// Write terminal control output verbatim (newline-normalized) without
/// touching the overlay. Used by the prompt editor and the picker, which run
/// when no overlay is active.
pub fn raw(s: &str) {
    let _st = state();
    emit(s);
}

/// Mark the prompt editor active/inactive. Deactivating flushes anything
/// still deferred straight to the terminal.
pub fn set_prompt_active(on: bool) {
    let mut st = state();
    st.prompt_active = on;
    if !on && !st.deferred.is_empty() {
        let text = std::mem::take(&mut st.deferred);
        emit(&text);
    }
}

/// Take the output deferred while the prompt was active.
pub fn take_deferred() -> String {
    std::mem::take(&mut state().deferred)
}

/// Resolves when deferred output is waiting for the prompt editor.
pub async fn deferred_ready() {
    DEFERRED_READY.notified().await
}

/// Show (or replace) the overlay line for `owner`. The aside takes the line
/// from a spinner, but a spinner never takes it from an aside the user is
/// still typing (a new request's spinner starting mid-aside must not wipe it).
pub fn set_overlay(owner: Overlay, text: &str) {
    if !is_tty() {
        return;
    }
    let mut st = state();
    if owner == Overlay::Spinner && matches!(&st.overlay, Some((Overlay::Aside, _))) {
        return;
    }
    st.overlay = Some((owner, text.to_string()));
    emit(&format!("\r\x1b[2K{text}"));
}

/// Remove the overlay line if `owner` still owns it.
pub fn clear_overlay(owner: Overlay) {
    let mut st = state();
    if matches!(&st.overlay, Some((o, _)) if *o == owner) {
        st.overlay = None;
        if is_tty() {
            emit("\r\x1b[2K");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_lone_newlines_only() {
        assert_eq!(normalize_newlines("a\nb"), "a\r\nb");
        assert_eq!(normalize_newlines("a\r\nb"), "a\r\nb");
        assert_eq!(normalize_newlines("\n\n"), "\r\n\r\n");
        assert_eq!(normalize_newlines("plain"), "plain");
    }
}
