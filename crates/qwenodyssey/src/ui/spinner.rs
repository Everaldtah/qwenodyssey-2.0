//! The live one-line status shown while the model works (port of the TS
//! `Spinner` in src/cli/render.ts), e.g.
//!
//! ```text
//! · Herding… (12s · ↑ 9.6k ↓ 47 tokens · 18 tok/s · esc to interrupt)
//! ```
//!
//! The glyph pulses, the elapsed time ticks, and the ↓ count climbs as streamed
//! deltas arrive ([`Spinner::bump_out`] / [`Spinner::set_out`]). tok/s is the
//! live generation rate measured from the first streamed token. It draws on the
//! UI writer's overlay line, so printed output and the status never overwrite
//! each other. A no-op when stdout is not a TTY (the counters still update).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

use super::out::{self, Overlay};
use super::style;
use crate::render::{format_elapsed, format_tokens};

const FRAMES: [&str; 6] = ["·", "∘", "○", "◌", "○", "∘"];
/// Immediate-redraw throttle for token ticks.
const MIN_REDRAW: Duration = Duration::from_millis(60);
const TICK: Duration = Duration::from_millis(140);

struct State {
    word: String,
    hint: String,
    up_tokens: u64,
    out_tokens: u64,
    started_at: Instant,
    first_token_at: Option<Instant>,
    last_draw: Option<Instant>,
    frame: usize,
    paused: bool,
    running: bool,
}

/// Cheap to clone; all clones drive the same status line.
#[derive(Clone)]
pub struct Spinner {
    st: Arc<Mutex<State>>,
    task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl Spinner {
    /// `hint` is a trailing key reminder, e.g. "esc to interrupt".
    pub fn new(word: &str, up_tokens: u64, hint: &str) -> Self {
        Self {
            st: Arc::new(Mutex::new(State {
                word: word.to_string(),
                hint: hint.to_string(),
                up_tokens,
                out_tokens: 0,
                started_at: Instant::now(),
                first_token_at: None,
                last_draw: None,
                frame: 0,
                paused: false,
                running: false,
            })),
            task: Arc::new(Mutex::new(None)),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start the animation (needs a tokio runtime; no-op without a TTY).
    pub fn begin(&self) {
        {
            let mut st = self.state();
            st.started_at = Instant::now();
            if !out::is_tty() {
                return;
            }
            st.running = true;
            draw(&mut st);
        }
        self.start_ticker();
    }

    fn start_ticker(&self) {
        let st = self.st.clone();
        let handle = tokio::spawn(async move {
            let mut iv = tokio::time::interval(TICK);
            iv.tick().await; // the first tick fires immediately
            loop {
                iv.tick().await;
                let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
                if !g.running {
                    break;
                }
                draw(&mut g);
            }
        });
        let mut slot = self.task.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = slot.replace(handle) {
            old.abort();
        }
    }

    /// Update the ↑ (input/context) count, e.g. once the prompt size is known.
    pub fn set_up(&self, tokens: u64) {
        self.state().up_tokens = tokens;
    }

    /// Add streamed output tokens — call from the stream's on_chunk handler.
    pub fn bump_out(&self, tokens: u64) {
        let mut st = self.state();
        if st.out_tokens == 0 {
            st.first_token_at = Some(Instant::now());
        }
        st.out_tokens += tokens;
        tick_draw(&mut st);
    }

    /// Set the absolute streamed-output token count (reconciles to exact usage).
    pub fn set_out(&self, tokens: u64) {
        let mut st = self.state();
        if st.out_tokens == 0 && tokens > 0 {
            st.first_token_at = Some(Instant::now());
        }
        st.out_tokens = tokens;
        tick_draw(&mut st);
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn out_tokens(&self) -> u64 {
        self.state().out_tokens
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn up_tokens(&self) -> u64 {
        self.state().up_tokens
    }

    /// Stop and clear the line.
    pub fn stop(&self) {
        self.state().running = false;
        if let Some(h) = self.task.lock().unwrap_or_else(|e| e.into_inner()).take() {
            h.abort();
        }
        out::clear_overlay(Overlay::Spinner);
    }

    /// Temporarily clear + halt the animation (while the user types an aside).
    pub fn pause(&self) {
        self.state().paused = true;
        out::clear_overlay(Overlay::Spinner);
    }

    /// Resume after `pause` without resetting the elapsed clock.
    pub fn resume(&self) {
        let mut st = self.state();
        st.paused = false;
        if st.running {
            draw(&mut st);
        }
    }

    /// The status text as it would be drawn now (tests, non-TTY logs).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn line(&self) -> String {
        let mut st = self.state();
        status_line(&mut st)
    }
}

/// Live tok/s once streaming has begun; 0 until then.
fn rate(st: &State) -> f64 {
    let Some(first) = st.first_token_at else { return 0.0 };
    if st.out_tokens < 2 {
        return 0.0;
    }
    let secs = first.elapsed().as_secs_f64();
    if secs > 0.2 {
        st.out_tokens as f64 / secs
    } else {
        0.0
    }
}

fn status_line(st: &mut State) -> String {
    let secs = st.started_at.elapsed().as_secs();
    let glyph = FRAMES[st.frame % FRAMES.len()];
    st.frame = st.frame.wrapping_add(1);
    let r = rate(st);
    let mut tokens = format!("↑ {}", format_tokens(st.up_tokens));
    if st.out_tokens > 0 {
        tokens.push_str(&format!(" ↓ {}", format_tokens(st.out_tokens)));
    }
    tokens.push_str(" tokens");
    let mut parts = vec![format_elapsed(secs), tokens];
    if r > 0.0 {
        parts.push(if r >= 10.0 { format!("{} tok/s", r.round() as u64) } else { format!("{r:.1} tok/s") });
    }
    if !st.hint.is_empty() {
        parts.push(st.hint.clone());
    }
    format!("{} {}{}", style::magenta(glyph), style::magenta(&st.word), style::gray(&format!("… ({})", parts.join(" · "))))
}

fn draw(st: &mut State) {
    if st.paused || !st.running {
        return;
    }
    st.last_draw = Some(Instant::now());
    let line = status_line(st);
    out::set_overlay(Overlay::Spinner, &line);
}

/// Redraw immediately on a token tick, throttled so fast streams stay cheap.
fn tick_draw(st: &mut State) {
    if !st.running || st.paused {
        return;
    }
    if st.last_draw.is_some_and(|t| t.elapsed() < MIN_REDRAW) {
        return;
    }
    draw(st);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_update_without_a_tty() {
        // Not a TTY in tests, so nothing draws, but the counters still update —
        // which is what the meter reconciliation relies on.
        let sp = Spinner::new("Testing", 100, "");
        sp.bump_out(1);
        sp.bump_out(2);
        assert_eq!(sp.out_tokens(), 3);
        sp.set_out(50);
        sp.set_up(120);
        assert_eq!(sp.out_tokens(), 50);
        assert_eq!(sp.up_tokens(), 120);
        sp.stop();
    }

    #[test]
    fn status_line_shape() {
        style::set_enabled(false);
        let sp = Spinner::new("Herding", 9601, "esc to interrupt");
        let l = sp.line();
        assert!(l.starts_with("· Herding… (0s · ↑ 9.6k tokens · esc to interrupt)"), "{l}");
        sp.set_out(47);
        assert!(sp.line().contains("↑ 9.6k ↓ 47 tokens"));
    }
}
