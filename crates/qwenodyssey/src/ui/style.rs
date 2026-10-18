//! Tiny ANSI styling helpers (the chalk subset the TS CLI used).
//!
//! Colour is decided once at startup: on when stdout is a terminal and
//! `NO_COLOR` is unset (`FORCE_COLOR` forces it on), exactly like chalk's
//! auto-detection, so piped output and test captures stay plain text. Close
//! codes match chalk's (39 for colours, 22 for bold/dim, 24 for underline) so
//! styles nest the same way.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

use regex::Regex;
use unicode_width::UnicodeWidthStr;

static COLOR: AtomicBool = AtomicBool::new(false);

/// Decide whether to emit colour. Call once at startup.
pub fn init() {
    let forced = std::env::var_os("FORCE_COLOR").is_some_and(|v| v != "0");
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    let on = forced || (!no_color && std::io::stdout().is_terminal());
    COLOR.store(on, Ordering::Relaxed);
}

/// Force colour on/off (tests).
#[cfg_attr(not(test), allow(dead_code))]
pub fn set_enabled(on: bool) {
    COLOR.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    COLOR.load(Ordering::Relaxed)
}

fn paint(open: &str, close: &str, s: &str) -> String {
    if enabled() {
        format!("\x1b[{open}m{s}\x1b[{close}m")
    } else {
        s.to_string()
    }
}

pub fn red(s: &str) -> String {
    paint("31", "39", s)
}
pub fn green(s: &str) -> String {
    paint("32", "39", s)
}
pub fn yellow(s: &str) -> String {
    paint("33", "39", s)
}
pub fn magenta(s: &str) -> String {
    paint("35", "39", s)
}
pub fn cyan(s: &str) -> String {
    paint("36", "39", s)
}
pub fn gray(s: &str) -> String {
    paint("90", "39", s)
}
pub fn cyan_bright(s: &str) -> String {
    paint("96", "39", s)
}
pub fn bold(s: &str) -> String {
    paint("1", "22", s)
}
pub fn dim(s: &str) -> String {
    paint("2", "22", s)
}
pub fn underline(s: &str) -> String {
    paint("4", "24", s)
}

static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[A-Za-z]").expect("static regex"));

/// Remove ANSI escape sequences.
pub fn strip_ansi(s: &str) -> String {
    ANSI.replace_all(s, "").into_owned()
}

/// Terminal columns a string occupies once escapes are removed (wide CJK = 2).
pub fn visible_width(s: &str) -> usize {
    UnicodeWidthStr::width(strip_ansi(s).as_str())
}

/// Left-justify `s` to `width` visible columns.
pub fn pad_end(s: &str, width: usize) -> String {
    let w = visible_width(s);
    if w >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - w))
    }
}

/// Cut PLAIN text (no escapes) to at most `max` terminal columns, ending in
/// "…" when something was cut. Menus and pickers redraw in place by moving the
/// cursor up a fixed number of rows, so none of their rows may wrap.
pub fn truncate_cols(s: &str, max: usize) -> String {
    if UnicodeWidthStr::width(s) <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw + 1 > max {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_and_measures() {
        assert_eq!(strip_ansi("\x1b[36mhi\x1b[39m"), "hi");
        assert_eq!(visible_width("\x1b[1mab\x1b[22m"), 2);
        assert_eq!(visible_width("日本"), 4);
        assert_eq!(pad_end("ab", 4), "ab  ");
    }

    #[test]
    fn truncates_by_columns() {
        assert_eq!(truncate_cols("hello", 10), "hello");
        assert_eq!(truncate_cols("hello world", 6), "hello…");
        assert_eq!(truncate_cols("日本語", 4), "日…");
        assert_eq!(truncate_cols("abc", 0), "");
    }
}
