//! Edit-block engine — the single biggest lever for small-model coding accuracy.
//!
//! Small models (7B–20B) produce *unreliable* unified diffs: off-by-one hunk
//! headers, drifted context lines, wrong line counts. They are far more reliable
//! at "here is the exact block I want to find, here is what to replace it with".
//! This module implements that protocol (inspired by Aider's SEARCH/REPLACE
//! blocks, reimplemented from scratch) with several layers of tolerance so a
//! near-miss is repaired deterministically instead of failing the turn:
//!
//!   1. Exact match.
//!   2. Whitespace-normalised match (leading indent / trailing space differences).
//!   3. Anchored fuzzy match (first & last non-blank lines must match, the middle
//!      is matched by similarity above a threshold).
//!   4. Indentation re-flow: the replacement is re-indented to match the matched
//!      region, so a model that dropped or guessed indentation still lands right.
//!
//! The wire format the model emits (parsed by [`parse_edit_blocks`]):
//!
//! ```text
//! path/to/file.ts
//! <<<<<<< SEARCH
//! original code
//! =======
//! replacement code
//! >>>>>>> REPLACE
//! ```
//!
//! A block with an empty SEARCH section creates/appends; a block whose file does
//! not exist creates it. Multiple blocks per file are applied in order.
//!
//! Lives in `qo-tools` (not `qo-pipeline`) because the `apply_edit` tool needs it
//! and `qo-pipeline` already depends on `qo-tools`; the pipeline re-uses it.

use std::path::PathBuf;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::util::read_text_lossy;

/// One SEARCH/REPLACE edit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditBlock {
    /// Target file path (relative to project root).
    pub path: String,
    /// The text to find. Empty string ⇒ create file / append.
    pub search: String,
    /// The text to substitute in.
    pub replace: String,
}

impl EditBlock {
    pub fn new(path: impl Into<String>, search: impl Into<String>, replace: impl Into<String>) -> Self {
        Self { path: path.into(), search: search.into(), replace: replace.into() }
    }
}

/// How a block was matched / applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MatchStrategy {
    Exact,
    Whitespace,
    AnchoredFuzzy,
    Reindented,
    Created,
    Appended,
    ReplacedWhole,
}

impl MatchStrategy {
    /// The TS string form ("exact", "anchored-fuzzy", "replaced-whole", …).
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchStrategy::Exact => "exact",
            MatchStrategy::Whitespace => "whitespace",
            MatchStrategy::AnchoredFuzzy => "anchored-fuzzy",
            MatchStrategy::Reindented => "reindented",
            MatchStrategy::Created => "created",
            MatchStrategy::Appended => "appended",
            MatchStrategy::ReplacedWhole => "replaced-whole",
        }
    }
}

impl std::fmt::Display for MatchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of applying all blocks for one file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BlockApplyResult {
    pub path: String,
    pub ok: bool,
    pub strategy: Option<MatchStrategy>,
    /// Similarity score 0..1 for fuzzy matches (1 = exact).
    pub confidence: Option<f64>,
    pub error: Option<String>,
    /// Pre-image, for journaling / rollback. `None` = file didn't exist.
    pub before: Option<String>,
    pub after: Option<String>,
}

/// A located SEARCH region: `[start, end)` line range plus how it matched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockLocation {
    pub start: usize,
    pub end: usize,
    pub strategy: MatchStrategy,
    pub confidence: f64,
}

/// Result of [`apply_block_to_content`] (pure, in-memory).
#[derive(Clone, Debug, PartialEq)]
pub struct ContentApplyResult {
    pub ok: bool,
    pub result: Option<String>,
    pub strategy: MatchStrategy,
    pub confidence: f64,
    pub error: Option<String>,
}

static FENCE_SEARCH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^<{5,9}\s*SEARCH\s*$").unwrap());
static FENCE_DIVIDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^={5,9}\s*$").unwrap());
static FENCE_REPLACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^>{5,9}\s*REPLACE\s*$").unwrap());
static BACKTICKS_EDGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^`+|`+$").unwrap());
static HEADING_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#+\s*").unwrap());
static FENCE_HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^```+\s*[A-Za-z0-9_]*\s*").unwrap());
// JS `\w` is ASCII-only, so spell the class out.
static PATHISH_FILE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9_./\-@ ]+\.[A-Za-z0-9]+$").unwrap());
static WS_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// A path-ish token: has a slash or a file extension, no spaces in the core.
fn looks_like_path(s: &str) -> bool {
    let t = BACKTICKS_EDGE.replace_all(s.trim(), "");
    let t = HEADING_PREFIX.replace(&t, "");
    if t.is_empty() || t.chars().count() > 200 {
        return false;
    }
    PATHISH_FILE.is_match(&t) || t.contains('/')
}

/// Parse SEARCH/REPLACE blocks out of a model response. Tolerant of:
///  - the path appearing on the line above the fence, or as a ```` ```lang path ```` header
///  - fences of 5–9 markers (models vary)
///  - surrounding prose and code fences
pub fn parse_edit_blocks(text: &str) -> Vec<EditBlock> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut blocks = Vec::new();
    let mut i = 0usize;
    let mut pending_path = String::new();

    while i < lines.len() {
        let line = lines[i];
        if FENCE_SEARCH.is_match(line.trim()) {
            // Walk backwards (skipping blank / fence lines) to find the path.
            let mut p = pending_path.clone();
            let lo = i.saturating_sub(4);
            let mut j = i;
            while j > lo {
                j -= 1;
                let cand = lines[j].trim();
                if cand.is_empty() {
                    continue;
                }
                if cand.starts_with("```") {
                    let after = FENCE_HEADER.replace(cand, "");
                    let after = after.trim();
                    if !after.is_empty() && looks_like_path(after) {
                        p = after.to_string();
                        break;
                    }
                    continue;
                }
                if looks_like_path(cand) {
                    let c = BACKTICKS_EDGE.replace_all(cand, "");
                    p = HEADING_PREFIX.replace(&c, "").trim().to_string();
                    break;
                }
                break;
            }

            // Collect SEARCH lines until the divider.
            i += 1;
            let mut search: Vec<&str> = Vec::new();
            while i < lines.len() && !FENCE_DIVIDER.is_match(lines[i].trim()) {
                search.push(lines[i]);
                i += 1;
            }
            i += 1; // skip divider
            let mut replace: Vec<&str> = Vec::new();
            while i < lines.len() && !FENCE_REPLACE.is_match(lines[i].trim()) {
                replace.push(lines[i]);
                i += 1;
            }
            i += 1; // skip closing REPLACE fence
            if !p.is_empty() {
                blocks.push(EditBlock { path: p.clone(), search: search.join("\n"), replace: replace.join("\n") });
            }
            pending_path = p; // subsequent blocks may target the same file
            continue;
        }
        // Track a bare path line so a block that omits it can inherit it.
        if looks_like_path(line) {
            pending_path = BACKTICKS_EDGE.replace_all(line, "").trim().to_string();
        }
        i += 1;
    }
    blocks
}

/// Levenshtein-ratio similarity in [0,1] (1 = identical). Cheap, line-level.
pub fn similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dist = levenshtein_chars(&a, &b);
    let max = a.len().max(b.len());
    1.0 - dist as f64 / max as f64
}

/// Two-row Levenshtein over chars.
pub(crate) fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

fn normalize_ws(s: &str) -> String {
    s.split('\n')
        .map(|l| WS_RUN.replace_all(l, " ").trim().to_string())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Find where `search` occurs in `content`, returning the `[start, end)` line
/// range and the strategy/confidence used, or `None`. Tries progressively looser
/// matches.
pub fn locate_block(content: &str, search: &str) -> Option<BlockLocation> {
    let content_lines: Vec<&str> = content.split('\n').collect();
    let search_lines: Vec<&str> = search.split('\n').collect();
    let win = search_lines.len();
    if win == 0 {
        return None;
    }

    // 1. Exact contiguous match.
    if win <= content_lines.len() {
        for i in 0..=(content_lines.len() - win) {
            if content_lines[i..i + win].join("\n") == search {
                return Some(BlockLocation { start: i, end: i + win, strategy: MatchStrategy::Exact, confidence: 1.0 });
            }
        }
    }

    // 2. Whitespace-normalised match.
    let n_search = normalize_ws(search);
    if win <= content_lines.len() {
        for i in 0..=(content_lines.len() - win) {
            if normalize_ws(&content_lines[i..i + win].join("\n")) == n_search {
                return Some(BlockLocation {
                    start: i,
                    end: i + win,
                    strategy: MatchStrategy::Whitespace,
                    confidence: 0.97,
                });
            }
        }
    }

    // 3. Anchored fuzzy match: the first and last non-blank search lines must
    //    appear (normalised); score the window between them by similarity.
    let first_anchor = search_lines.iter().find(|l| !l.trim().is_empty());
    let last_anchor = search_lines.iter().rev().find(|l| !l.trim().is_empty());
    if let (Some(fa), Some(la)) = (first_anchor, last_anchor) {
        let fa = fa.trim();
        let la = la.trim();
        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..content_lines.len() {
            if content_lines[i].trim() != fa {
                continue;
            }
            // Find the matching last-anchor within a reasonable span.
            let max_end = content_lines.len().min(i + win + 6);
            for k in (i + 1)..=max_end {
                if content_lines[k - 1].trim() != la {
                    continue;
                }
                let score = similarity(&normalize_ws(&content_lines[i..k].join("\n")), &n_search);
                if best.map_or(true, |(_, _, s)| score > s) {
                    best = Some((i, k, score));
                }
            }
        }
        if let Some((start, end, score)) = best {
            if score >= 0.75 {
                return Some(BlockLocation { start, end, strategy: MatchStrategy::AnchoredFuzzy, confidence: score });
            }
        }
    }

    None
}

fn leading_indent(l: &str) -> &str {
    let end = l.find(|c: char| c != ' ' && c != '\t').unwrap_or(l.len());
    &l[..end]
}

/// Detect the common leading-indent of a block of lines (spaces or tabs).
fn common_indent(lines: &[&str]) -> String {
    let non_blank: Vec<&str> = lines.iter().copied().filter(|l| !l.trim().is_empty()).collect();
    let Some(first) = non_blank.first() else {
        return String::new();
    };
    let mut indent: Vec<char> = leading_indent(first).chars().collect();
    for l in &non_blank[1..] {
        let cur: Vec<char> = leading_indent(l).chars().collect();
        let mut k = 0;
        while k < indent.len() && k < cur.len() && indent[k] == cur[k] {
            k += 1;
        }
        indent.truncate(k);
    }
    indent.into_iter().collect()
}

/// Re-indent `replace` so its base indent matches the region it replaces.
fn reindent(replace: &str, target_indent: &str, search_indent: &str) -> String {
    if target_indent == search_indent {
        return replace.to_string();
    }
    replace
        .split('\n')
        .map(|l| {
            if l.trim().is_empty() {
                return l.to_string();
            }
            // Strip the replacement's own base indent, then prefix the target's.
            let stripped = match l.strip_prefix(search_indent) {
                Some(rest) => rest,
                None => &l[leading_indent(l).len()..],
            };
            format!("{target_indent}{stripped}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Apply a single edit block to in-memory content. Pure — no filesystem.
pub fn apply_block_to_content(content: &str, block: &EditBlock) -> ContentApplyResult {
    // Empty SEARCH ⇒ append (or full content if file is empty). Exception: when
    // the replacement already CONTAINS the whole current file, the model meant
    // "here is the complete new file" — appending would duplicate every existing
    // definition (a 9B model did exactly that: two `def add`s). Replace instead.
    if block.search.trim().is_empty() {
        let cur = content.trim();
        if !cur.is_empty() && block.replace.contains(cur) {
            return ContentApplyResult {
                ok: true,
                result: Some(block.replace.clone()),
                strategy: MatchStrategy::ReplacedWhole,
                confidence: 1.0,
                error: None,
            };
        }
        let sep = if !content.is_empty() && !content.ends_with('\n') { "\n" } else { "" };
        let (result, strategy) = if content.is_empty() {
            (block.replace.clone(), MatchStrategy::Created)
        } else {
            (format!("{content}{sep}{}", block.replace), MatchStrategy::Appended)
        };
        return ContentApplyResult { ok: true, result: Some(result), strategy, confidence: 1.0, error: None };
    }

    let Some(loc) = locate_block(content, &block.search) else {
        return ContentApplyResult {
            ok: false,
            result: None,
            strategy: MatchStrategy::Exact,
            confidence: 0.0,
            error: Some("SEARCH block not found in file".to_string()),
        };
    };

    let content_lines: Vec<&str> = content.split('\n').collect();
    let region = &content_lines[loc.start..loc.end];
    let target_indent = common_indent(region);
    let search_lines: Vec<&str> = block.search.split('\n').collect();
    let search_indent = common_indent(&search_lines);
    let mut replacement = block.replace.clone();
    let mut strategy = loc.strategy;
    if loc.strategy != MatchStrategy::Exact && target_indent != search_indent {
        replacement = reindent(&block.replace, &target_indent, &search_indent);
        strategy = MatchStrategy::Reindented;
    }

    let mut out: Vec<&str> = Vec::with_capacity(content_lines.len());
    out.extend_from_slice(&content_lines[..loc.start]);
    out.extend(replacement.split('\n'));
    out.extend_from_slice(&content_lines[loc.end..]);

    ContentApplyResult { ok: true, result: Some(out.join("\n")), strategy, confidence: loc.confidence, error: None }
}

/// Apply edit blocks to the filesystem (grouped by file, applied in order),
/// journaling pre-images so a batch can be rolled back. `resolve` maps a
/// relative path to an absolute one and enforces the write sandbox.
pub fn apply_edit_blocks<F>(blocks: &[EditBlock], resolve: F) -> Vec<BlockApplyResult>
where
    F: Fn(&str) -> anyhow::Result<PathBuf>,
{
    let mut results = Vec::new();
    // Group blocks per file (first-seen order) so multiple edits compose on live content.
    let mut order: Vec<&str> = Vec::new();
    let mut by_file: std::collections::HashMap<&str, Vec<&EditBlock>> = std::collections::HashMap::new();
    for b in blocks {
        let entry = by_file.entry(b.path.as_str()).or_default();
        if entry.is_empty() {
            order.push(b.path.as_str());
        }
        entry.push(b);
    }

    for rel in order {
        let file_blocks = &by_file[rel];
        let abs = match resolve(rel) {
            Ok(a) => a,
            Err(err) => {
                results.push(BlockApplyResult {
                    path: rel.to_string(),
                    ok: false,
                    error: Some(err.to_string()),
                    ..Default::default()
                });
                continue;
            }
        };
        let before = if abs.exists() {
            match read_text_lossy(&abs) {
                Ok(s) => Some(s),
                Err(err) => {
                    results.push(BlockApplyResult {
                        path: rel.to_string(),
                        ok: false,
                        error: Some(err.to_string()),
                        ..Default::default()
                    });
                    continue;
                }
            }
        } else {
            None
        };
        let mut content = before.clone().unwrap_or_default();
        let mut all_ok = true;
        let mut last_strategy = MatchStrategy::Exact;
        let mut last_confidence = 1.0;

        for block in file_blocks {
            let r = apply_block_to_content(&content, block);
            if !r.ok {
                results.push(BlockApplyResult {
                    path: rel.to_string(),
                    ok: false,
                    before: before.clone(),
                    strategy: Some(r.strategy),
                    confidence: Some(r.confidence),
                    error: r.error,
                    after: None,
                });
                all_ok = false;
                break;
            }
            content = r.result.unwrap_or_default();
            last_strategy = r.strategy;
            last_confidence = r.confidence;
        }

        if !all_ok {
            continue;
        }

        let write = (|| -> std::io::Result<()> {
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&abs, content.as_bytes())
        })();
        match write {
            Ok(()) => results.push(BlockApplyResult {
                path: rel.to_string(),
                ok: true,
                before,
                after: Some(content),
                strategy: Some(last_strategy),
                confidence: Some(last_confidence),
                error: None,
            }),
            Err(err) => results.push(BlockApplyResult {
                path: rel.to_string(),
                ok: false,
                before,
                error: Some(err.to_string()),
                ..Default::default()
            }),
        }
    }

    results
}

/// Produce targeted feedback for blocks that failed to match, so the model can
/// correct on the next turn. Includes the closest line in the file (by anchor
/// similarity) to nudge it toward the right region without dumping the whole file.
pub fn explain_failure(content: &str, block: &EditBlock) -> String {
    let Some(anchor) = block.search.split('\n').find(|l| !l.trim().is_empty()) else {
        return "SEARCH block was empty.".to_string();
    };
    let anchor = anchor.trim();
    let content_lines: Vec<&str> = content.split('\n').collect();
    let mut best_line: Option<usize> = None;
    let mut best_score = 0.0;
    for (i, l) in content_lines.iter().enumerate() {
        let sc = similarity(l.trim(), anchor);
        if sc > best_score {
            best_score = sc;
            best_line = Some(i);
        }
    }
    let hint = match best_line {
        Some(bl) if best_score > 0.5 => format!(
            " Closest line in the file is L{}: \"{}\". Your SEARCH must match the file EXACTLY (copy it verbatim).",
            bl + 1,
            content_lines[bl].trim().chars().take(120).collect::<String>()
        ),
        _ => " No similar line found — the file may not contain that code at all.".to_string(),
    };
    format!("SEARCH block did not match {}.{hint}", block.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(path: &str, search: &str, replace: &str) -> EditBlock {
        EditBlock::new(path, search, replace)
    }

    #[test]
    fn parses_single_block_with_path_line() {
        let text = [
            "Here's the fix:",
            "src/foo.ts",
            "<<<<<<< SEARCH",
            "const x = 1;",
            "=======",
            "const x = 2;",
            ">>>>>>> REPLACE",
        ]
        .join("\n");
        let blocks = parse_edit_blocks(&text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].path, "src/foo.ts");
        assert_eq!(blocks[0].search, "const x = 1;");
        assert_eq!(blocks[0].replace, "const x = 2;");
    }

    #[test]
    fn parses_path_from_fence_header() {
        let text = ["```ts src/bar.ts", "<<<<<<< SEARCH", "old", "=======", "new", ">>>>>>> REPLACE", "```"].join("\n");
        let blocks = parse_edit_blocks(&text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].path, "src/bar.ts");
    }

    #[test]
    fn parses_multiple_blocks_inheriting_path() {
        let text = [
            "lib/util.py",
            "<<<<<<< SEARCH",
            "a = 1",
            "=======",
            "a = 2",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "b = 3",
            "=======",
            "b = 4",
            ">>>>>>> REPLACE",
        ]
        .join("\n");
        let blocks = parse_edit_blocks(&text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].path, "lib/util.py");
        assert_eq!(blocks[1].path, "lib/util.py");
    }

    #[test]
    fn parses_empty_search_block() {
        let text = ["newfile.txt", "<<<<<<< SEARCH", "=======", "hello world", ">>>>>>> REPLACE"].join("\n");
        let blocks = parse_edit_blocks(&text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].search, "");
        assert_eq!(blocks[0].replace, "hello world");
    }

    const CONTENT: &str = "function add(a, b) {\n    return a + b;\n}\n\nconst z = 10;";

    #[test]
    fn locate_exact() {
        let loc = locate_block(CONTENT, "    return a + b;").unwrap();
        assert_eq!(loc.strategy, MatchStrategy::Exact);
        assert_eq!(loc.start, 1);
    }

    #[test]
    fn locate_whitespace_tolerant() {
        let loc = locate_block(CONTENT, "return a + b;").unwrap();
        assert!(matches!(loc.strategy, MatchStrategy::Whitespace | MatchStrategy::AnchoredFuzzy));
    }

    #[test]
    fn locate_none_when_unrelated() {
        assert!(locate_block(CONTENT, "completely unrelated code here xyz").is_none());
    }

    #[test]
    fn apply_replaces_exact() {
        let r = apply_block_to_content("a\nb\nc", &blk("x", "b", "B"));
        assert!(r.ok);
        assert_eq!(r.result.as_deref(), Some("a\nB\nc"));
    }

    #[test]
    fn apply_reindents() {
        let content = ["class C {", "    method() {", "        return 1;", "    }", "}"].join("\n");
        let r = apply_block_to_content(&content, &blk("x", "return 1;", "return 42;"));
        assert!(r.ok);
        assert!(r.result.unwrap().contains("        return 42;"));
    }

    #[test]
    fn apply_appends_on_empty_search() {
        let r = apply_block_to_content("line1", &blk("x", "", "line2"));
        assert_eq!(r.result.as_deref(), Some("line1\nline2"));
        assert_eq!(r.strategy, MatchStrategy::Appended);
    }

    #[test]
    fn apply_reports_not_found() {
        let r = apply_block_to_content("a\nb", &blk("x", "zzz", "q"));
        assert!(!r.ok);
        assert!(r.error.unwrap().to_lowercase().contains("not found"));
    }

    // bugfixes2026_09: apply_edit with an empty search
    #[test]
    fn empty_search_replaces_whole_when_replacement_contains_file() {
        let cur = "def add(a, b):\n    return a + b\n";
        let r = apply_block_to_content(cur, &blk("calc.py", "", &format!("{cur}\ndef multiply(a, b):\n    return a * b\n")));
        assert!(r.ok);
        assert_eq!(r.strategy, MatchStrategy::ReplacedWhole);
        let out = r.result.unwrap();
        assert_eq!(out.matches("def add").count(), 1);
        assert!(out.contains("def multiply"));
    }

    #[test]
    fn empty_search_still_appends_new_text() {
        let cur = "def add(a, b):\n    return a + b\n";
        let r = apply_block_to_content(cur, &blk("calc.py", "", "def multiply(a, b):\n    return a * b\n"));
        assert_eq!(r.strategy, MatchStrategy::Appended);
        assert_eq!(r.result.unwrap(), format!("{cur}def multiply(a, b):\n    return a * b\n"));
    }

    /// Same resolver as the TS test: `path.resolve(dir, rel)` (lexical) and
    /// refuse anything outside `dir`.
    fn sandbox(dir: &std::path::Path) -> impl Fn(&str) -> anyhow::Result<PathBuf> + '_ {
        move |rel: &str| {
            let mut abs = dir.to_path_buf();
            for comp in std::path::Path::new(rel).components() {
                match comp {
                    std::path::Component::ParentDir => {
                        abs.pop();
                    }
                    std::path::Component::Normal(c) => abs.push(c),
                    _ => {}
                }
            }
            if abs != dir && !abs.starts_with(dir) {
                anyhow::bail!("escape");
            }
            Ok(abs)
        }
    }

    #[test]
    fn fs_applies_multiple_blocks_sequentially() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("code.js"), "let a = 1;\nlet b = 2;\n").unwrap();
        let results = apply_edit_blocks(
            &[blk("code.js", "let a = 1;", "let a = 100;"), blk("code.js", "let b = 2;", "let b = 200;")],
            sandbox(dir.path()),
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].ok);
        let out = std::fs::read_to_string(dir.path().join("code.js")).unwrap();
        assert!(out.contains("let a = 100;"));
        assert!(out.contains("let b = 200;"));
    }

    #[test]
    fn fs_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let results = apply_edit_blocks(&[blk("fresh.txt", "", "brand new")], sandbox(dir.path()));
        assert!(results[0].ok);
        assert_eq!(std::fs::read_to_string(dir.path().join("fresh.txt")).unwrap(), "brand new");
    }

    #[test]
    fn fs_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let results = apply_edit_blocks(&[blk("../escape.txt", "", "x")], sandbox(dir.path()));
        assert!(!results[0].ok);
    }

    #[test]
    fn similarity_scores() {
        assert_eq!(similarity("hello", "hello"), 1.0);
        assert!(similarity("hello", "xyzzy") < 0.4);
    }

    #[test]
    fn explain_failure_points_to_closest_line() {
        let content = "function compute(value) {\n  return value * 2;\n}";
        let msg = explain_failure(content, &blk("x", "function compute(val) {", "..."));
        assert!(msg.contains("compute"));
    }
}
