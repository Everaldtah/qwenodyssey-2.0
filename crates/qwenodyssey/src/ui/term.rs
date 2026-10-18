//! Terminal mode ownership.
//!
//! In 1.x every prompt, picker and interjection capture called `setRawMode`
//! and attached its own stdin listener, so a component that forgot to undo it
//! (or an error between the two calls) left the terminal stuck. In 2.0 raw
//! mode and bracketed paste are switched on ONCE, when the interactive chat
//! starts, and switched off by [`restore`] — which runs from the guard's
//! `Drop`, from the panic hook, and before any `process::exit`. Nothing else
//! touches the terminal mode.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::{cursor, execute, terminal};

use super::out;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static HOOK: Once = Once::new();

/// Holds the terminal in raw mode + bracketed paste for its lifetime.
pub struct TerminalGuard {
    _private: (),
}

impl TerminalGuard {
    /// Enter raw mode and enable bracketed paste. Installs the panic hook.
    pub fn enter() -> std::io::Result<Self> {
        install_panic_hook();
        terminal::enable_raw_mode()?;
        ACTIVE.store(true, Ordering::SeqCst);
        out::set_raw(true);
        let mut so = std::io::stdout();
        // Bracketed paste is best-effort: a terminal that ignores it still works
        // (pastes then arrive as a key flood, which the input thread coalesces).
        let _ = execute!(so, EnableBracketedPaste);
        let _ = so.flush();
        Ok(Self { _private: () })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
    }
}

/// Is the interactive raw-mode session active?
pub fn active() -> bool {
    ACTIVE.load(Ordering::SeqCst)
}

/// Undo everything [`TerminalGuard::enter`] did. Idempotent and safe to call
/// from a panic hook or right before `process::exit`.
pub fn restore() {
    if !ACTIVE.swap(false, Ordering::SeqCst) {
        return;
    }
    let mut so = std::io::stdout();
    let _ = execute!(so, DisableBracketedPaste, cursor::Show);
    let _ = terminal::disable_raw_mode();
    out::set_raw(false);
    let _ = so.flush();
}

/// Restore the terminal, then exit with `code` (destructors don't run on
/// `process::exit`, so the guard alone isn't enough).
pub fn exit(code: i32) -> ! {
    restore();
    std::process::exit(code);
}

/// Restore the terminal before the default panic message prints, so a panic
/// never leaves the user's shell in raw mode. A panic while the interactive
/// session is active is fatal (even inside a background task that tokio would
/// otherwise survive): continuing with a half-restored terminal is exactly the
/// "can't type" state 2.0 exists to rule out.
pub fn install_panic_hook() {
    HOOK.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let was_active = active();
            restore();
            prev(info);
            if was_active {
                std::process::exit(101);
            }
        }));
    });
}
