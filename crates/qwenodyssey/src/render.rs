//! Terminal rendering helpers (port of src/cli/render.ts): the launch banner,
//! rules, token/elapsed formatting, diff coloring and a tiny markdown renderer.
//! The live spinner lives in [`crate::ui::spinner`].

use std::sync::LazyLock;

use rand::seq::SliceRandom;
use regex::Regex;

use crate::ui::style;

pub struct BannerInfo<'a> {
    pub version: &'a str,
    pub model: &'a str,
    pub provider: &'a str,
    pub mode: &'a str,
    pub cwd: &'a str,
}

/// Claude-Code-style launch banner: a small block-glyph mark on the left with
/// version / model / cwd lines on the right, followed by an input hint.
pub fn banner(info: &BannerInfo) -> String {
    let mark = style::cyan_bright;
    let dim = style::gray;
    let rows = [
        format!("{}  {} {}", mark(" ▟█▜▛█▙ "), style::bold("Qwenodyssey"), dim(&format!("v{}", info.version))),
        format!("{}  {} {}", mark(" ▜█▟▙█▛ "), info.model, dim(&format!("· {}", info.provider))),
        format!("{}  {}", mark("  ▀▘▝▀  "), dim(info.cwd)),
    ];
    let mut out = vec![String::new()];
    out.extend(rows);
    out.push(dim(&format!("\n   {} mode · {}:{}", info.mode, info.provider, info.model)));
    out.push(dim("   Type a message · @path inlines a file · Esc interrupts · /models · /help · /exit"));
    out.push(String::new());
    out.join("\n")
}

/// Terminal width in columns (100 when unknown, like `process.stdout.columns || 100`).
pub fn term_cols() -> usize {
    if crate::ui::out::is_tty() {
        crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(100)
    } else {
        100
    }
}

/// A full-width horizontal rule sized to the current terminal.
pub fn hrule() -> String {
    let width = term_cols().saturating_sub(2).min(120);
    style::gray(&"─".repeat(width.max(40)))
}

/// Whimsical gerunds shown while the model is working (one per request).
const THINKING_WORDS: [&str; 18] = [
    "Herding", "Pondering", "Noodling", "Conjuring", "Percolating", "Ruminating", "Synthesizing", "Cogitating", "Tinkering",
    "Brewing", "Scheming", "Musing", "Crunching", "Divining", "Untangling", "Marinating", "Spelunking", "Wrangling",
];

pub fn thinking_word() -> &'static str {
    THINKING_WORDS.choose(&mut rand::thread_rng()).copied().unwrap_or("Herding")
}

/// "9601" → "9.6k", "512" → "512", "12000" → "12k".
pub fn format_tokens(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = format!("{:.1}", n as f64 / 1000.0);
    format!("{}k", s.strip_suffix(".0").unwrap_or(&s))
}

/// "85" → "1m 25s", "42" → "42s".
pub fn format_elapsed(seconds: u64) -> String {
    if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

/// Color a unified diff: headers bold, hunks cyan, +green, -red, context gray.
pub fn colorize_diff(diff: &str) -> String {
    diff.split('\n')
        .map(|line| {
            if line.starts_with("+++") || line.starts_with("---") {
                style::bold(line)
            } else if line.starts_with("@@") {
                style::cyan(line)
            } else if line.starts_with('+') {
                style::green(line)
            } else if line.starts_with('-') {
                style::red(line)
            } else {
                style::gray(line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(#{1,6})\s+(.*)$").expect("static regex"));
static BOLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*(.+?)\*\*").expect("static regex"));
static CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`([^`]+)`").expect("static regex"));

/// Very small markdown renderer for the terminal: headings, bold, inline code,
/// and fenced code blocks. Avoids a heavy dependency.
pub fn render_markdown(md: &str) -> String {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in md.split('\n') {
        if line.starts_with("```") {
            in_fence = !in_fence;
            out.push(style::gray(line));
            continue;
        }
        if in_fence {
            out.push(style::yellow(line));
            continue;
        }
        let l = HEADING.replace(line, |c: &regex::Captures| style::bold(&style::underline(&c[2]))).into_owned();
        let l = BOLD.replace_all(&l, |c: &regex::Captures| style::bold(&c[1])).into_owned();
        let l = CODE.replace_all(&l, |c: &regex::Captures| style::cyan(&c[1])).into_owned();
        out.push(l);
    }
    out.join("\n")
}

/// Prefix every line with a dim gutter ("  │ ").
pub fn indent(text: &str) -> String {
    text.split('\n').map(|l| format!("  │ {l}")).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_tokens_like_ts() {
        assert_eq!(format_tokens(5), "5");
        assert_eq!(format_tokens(512), "512");
        assert_eq!(format_tokens(9601), "9.6k");
        assert_eq!(format_tokens(12000), "12k");
    }

    #[test]
    fn formats_elapsed() {
        assert_eq!(format_elapsed(85), "1m 25s");
        assert_eq!(format_elapsed(42), "42s");
    }

    #[test]
    fn banner_says_v2() {
        style::set_enabled(false);
        let b = banner(&BannerInfo { version: "2.0.0", model: "m", provider: "nvidia", mode: "safe", cwd: "C:/x" });
        assert!(b.contains("Qwenodyssey v2.0.0"));
        assert!(b.contains("safe mode · nvidia:m"));
        assert!(b.contains("Esc interrupts"));
    }

    #[test]
    fn markdown_and_diff_plain() {
        style::set_enabled(false);
        assert_eq!(render_markdown("# Title\n**b** `c`\n```\ncode\n```"), "Title\nb c\n```\ncode\n```");
        assert_eq!(colorize_diff("+a\n-b"), "+a\n-b");
        assert_eq!(indent("a\nb"), "  │ a\n  │ b");
        assert!(THINKING_WORDS.contains(&thinking_word()));
    }
}
