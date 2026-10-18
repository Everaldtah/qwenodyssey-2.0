//! Unified diffs in exactly the dialect of jsdiff 5 (the npm `diff` package the
//! TypeScript patch engine used): `createTwoFilesPatch`, `parsePatch` and
//! `applyPatch` (fuzz factor 0).
//!
//! Matching jsdiff matters for two reasons: the preview the user approves is
//! byte-for-byte the format they know (Index line, `===` rule, 4 lines of
//! context, `\ No newline at end of file` markers), and a model-written diff
//! that applied in the TS version applies here too — including jsdiff's
//! leniency of sliding a hunk to the nearest offset where its context fits.
//! The line diff itself is Myers (via `similar`), as in jsdiff.

use std::sync::LazyLock;

use regex::Regex;
use similar::{Algorithm, DiffOp, TextDiff};

/// Lines of context around each hunk (jsdiff default).
const CONTEXT: usize = 4;
const NO_NEWLINE: &str = "\\ No newline at end of file";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Equal,
    Removed,
    Added,
}

/// One run of the line diff (jsdiff "component"): the lines without their
/// terminating `\n` (a `\r` of a CRLF file stays on the line, as in jsdiff).
struct Run {
    kind: Kind,
    lines: Vec<String>,
}

fn value_lines(value: &str) -> Vec<String> {
    value.strip_suffix('\n').unwrap_or(value).split('\n').map(str::to_string).collect()
}

/// jsdiff `diffLines`: runs of equal / removed / added lines, removals first
/// inside each change.
fn line_runs(old: &str, new: &str) -> Vec<Run> {
    let diff = TextDiff::configure().algorithm(Algorithm::Myers).diff_lines(old, new);
    let olds = diff.old_slices();
    let news = diff.new_slices();
    let mut out: Vec<Run> = Vec::new();
    let mut removed = String::new();
    let mut added = String::new();
    let flush = |out: &mut Vec<Run>, removed: &mut String, added: &mut String| {
        if !removed.is_empty() {
            out.push(Run { kind: Kind::Removed, lines: value_lines(removed) });
            removed.clear();
        }
        if !added.is_empty() {
            out.push(Run { kind: Kind::Added, lines: value_lines(added) });
            added.clear();
        }
    };
    for op in diff.ops() {
        match *op {
            DiffOp::Equal { old_index, len, .. } => {
                flush(&mut out, &mut removed, &mut added);
                let value: String = olds[old_index..old_index + len].concat();
                out.push(Run { kind: Kind::Equal, lines: value_lines(&value) });
            }
            DiffOp::Delete { old_index, old_len, .. } => {
                removed.push_str(&olds[old_index..old_index + old_len].concat());
            }
            DiffOp::Insert { new_index, new_len, .. } => {
                added.push_str(&news[new_index..new_index + new_len].concat());
            }
            DiffOp::Replace { old_index, old_len, new_index, new_len } => {
                removed.push_str(&olds[old_index..old_index + old_len].concat());
                added.push_str(&news[new_index..new_index + new_len].concat());
            }
        }
    }
    flush(&mut out, &mut removed, &mut added);
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutHunk {
    old_start: usize,
    old_lines: usize,
    new_start: usize,
    new_lines: usize,
    lines: Vec<String>,
}

/// jsdiff `structuredPatch` hunks (context 4).
fn structured_hunks(old: &str, new: &str) -> Vec<OutHunk> {
    let mut runs = line_runs(old, new);
    // Append an empty run to make cleanup easier (as jsdiff does).
    runs.push(Run { kind: Kind::Equal, lines: Vec::new() });
    let n = runs.len();
    let ctx = |lines: &[String]| lines.iter().map(|l| format!(" {l}")).collect::<Vec<_>>();

    let mut hunks = Vec::new();
    let (mut old_range_start, mut new_range_start) = (0usize, 0usize);
    let mut cur: Vec<String> = Vec::new();
    let (mut old_line, mut new_line) = (1usize, 1usize);

    for i in 0..n {
        let lines = &runs[i].lines;
        match runs[i].kind {
            Kind::Added | Kind::Removed => {
                if old_range_start == 0 {
                    old_range_start = old_line;
                    new_range_start = new_line;
                    if i > 0 {
                        let prev = &runs[i - 1].lines;
                        cur = ctx(&prev[prev.len().saturating_sub(CONTEXT)..]);
                        old_range_start -= cur.len();
                        new_range_start -= cur.len();
                    }
                }
                let sign = if runs[i].kind == Kind::Added { '+' } else { '-' };
                cur.extend(lines.iter().map(|l| format!("{sign}{l}")));
                if runs[i].kind == Kind::Added {
                    new_line += lines.len();
                } else {
                    old_line += lines.len();
                }
            }
            Kind::Equal => {
                if old_range_start != 0 {
                    if lines.len() <= CONTEXT * 2 && i + 2 < n {
                        // Overlapping context: keep the hunk open.
                        cur.extend(ctx(lines));
                    } else {
                        let context_size = lines.len().min(CONTEXT);
                        cur.extend(ctx(&lines[..context_size]));
                        let mut hunk = OutHunk {
                            old_start: old_range_start,
                            old_lines: old_line - old_range_start + context_size,
                            new_start: new_range_start,
                            new_lines: new_line - new_range_start + context_size,
                            lines: std::mem::take(&mut cur),
                        };
                        if i + 2 >= n && lines.len() <= CONTEXT {
                            // EOF is inside this hunk.
                            let old_eof_nl = old.ends_with('\n');
                            let new_eof_nl = new.ends_with('\n');
                            let no_nl_before_adds = lines.is_empty() && hunk.lines.len() > hunk.old_lines;
                            if !old_eof_nl && no_nl_before_adds && !old.is_empty() {
                                // Old has no EOL and no trailing context: the marker
                                // goes before the additions (unless old is empty).
                                hunk.lines.insert(hunk.old_lines, NO_NEWLINE.to_string());
                            }
                            if (!old_eof_nl && !no_nl_before_adds) || !new_eof_nl {
                                hunk.lines.push(NO_NEWLINE.to_string());
                            }
                        }
                        hunks.push(hunk);
                        old_range_start = 0;
                        new_range_start = 0;
                    }
                }
                old_line += lines.len();
                new_line += lines.len();
            }
        }
    }
    hunks
}

/// jsdiff `createTwoFilesPatch(oldName, newName, old, new, oldHeader, newHeader)`.
/// A `Some("")` header still emits the tab after the file name, like jsdiff.
pub fn create_two_files_patch(
    old_name: &str,
    new_name: &str,
    old: &str,
    new: &str,
    old_header: Option<&str>,
    new_header: Option<&str>,
) -> String {
    let mut ret: Vec<String> = Vec::new();
    if old_name == new_name {
        ret.push(format!("Index: {old_name}"));
    }
    ret.push("=".repeat(67));
    ret.push(format!("--- {old_name}{}", old_header.map(|h| format!("\t{h}")).unwrap_or_default()));
    ret.push(format!("+++ {new_name}{}", new_header.map(|h| format!("\t{h}")).unwrap_or_default()));
    for mut h in structured_hunks(old, new) {
        // Unified diff quirk: a zero-length side starts one line lower.
        if h.old_lines == 0 {
            h.old_start = h.old_start.saturating_sub(1);
        }
        if h.new_lines == 0 {
            h.new_start = h.new_start.saturating_sub(1);
        }
        ret.push(format!("@@ -{},{} +{},{} @@", h.old_start, h.old_lines, h.new_start, h.new_lines));
        ret.extend(h.lines);
    }
    ret.join("\n") + "\n"
}

/* ── parsePatch ── */

static LINE_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("\r\n|[\n\x0B\x0C\r\u{85}]").expect("static regex"));
static FILE_OR_HUNK_START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(---|\+\+\+|@@)\s").expect("static regex"));
static NEXT_INDEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(Index:|diff|---|\+\+\+)\s").expect("static regex"));
static FILE_HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(---|\+\+\+)\s+").expect("static regex"));
static HUNK_HEADER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@").expect("static regex"));

/// Split like JS `s.split(/\r\n|[\n\v\f\r\x85]/)`, also returning the delimiters.
fn split_lines(s: &str) -> (Vec<String>, Vec<String>) {
    let mut lines = Vec::new();
    let mut delims = Vec::new();
    let mut last = 0;
    for m in LINE_BREAK.find_iter(s) {
        lines.push(s[last..m.start()].to_string());
        delims.push(m.as_str().to_string());
        last = m.end();
    }
    lines.push(s[last..].to_string());
    (lines, delims)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Hunk {
    old_start: isize,
    old_lines: isize,
    new_start: isize,
    new_lines: isize,
    lines: Vec<String>,
    delimiters: Vec<String>,
    /// Set by `apply_patch` once the hunk's fit position is known.
    offset: isize,
    /// The `@@` header did not parse; such a hunk can never apply.
    malformed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FilePatch {
    hunks: Vec<Hunk>,
}

fn first_char(s: &str) -> Option<char> {
    s.chars().next()
}

/// jsdiff `parsePatch` (non-strict): one entry per file in the diff.
fn parse_patch(uni: &str) -> Vec<FilePatch> {
    let (diffstr, delimiters) = split_lines(uni);
    let len = diffstr.len();
    let mut list = Vec::new();
    let mut i = 0usize;

    while i < len {
        let mut index = FilePatch::default();
        // Metadata (Index:/diff lines; the file name is not needed here) up to
        // the first file or hunk header.
        while i < len {
            if FILE_OR_HUNK_START.is_match(&diffstr[i]) {
                break;
            }
            i += 1;
        }
        // Up to two file headers (--- / +++).
        for _ in 0..2 {
            if i < len && FILE_HEADER.is_match(&diffstr[i]) {
                i += 1;
            }
        }
        // Hunks.
        while i < len {
            let line = &diffstr[i];
            if NEXT_INDEX.is_match(line) {
                break;
            } else if line.starts_with("@@") {
                index.hunks.push(parse_hunk(&diffstr, &delimiters, &mut i));
            } else {
                i += 1;
            }
        }
        list.push(index);
    }
    list
}

fn parse_hunk(diffstr: &[String], delimiters: &[String], i: &mut usize) -> Hunk {
    let len = diffstr.len();
    let header = &diffstr[*i];
    *i += 1;
    let mut hunk = Hunk::default();
    match HUNK_HEADER.captures(header) {
        Some(c) => {
            let num = |k: usize, default: isize| {
                c.get(k).and_then(|m| m.as_str().parse::<isize>().ok()).unwrap_or(default)
            };
            hunk.old_start = num(1, 0);
            hunk.old_lines = num(2, 1);
            hunk.new_start = num(3, 0);
            hunk.new_lines = num(4, 1);
        }
        None => hunk.malformed = true,
    }
    // Unified diff quirk: a zero-length side starts one line lower.
    if hunk.old_lines == 0 {
        hunk.old_start += 1;
    }
    if hunk.new_lines == 0 {
        hunk.new_start += 1;
    }

    let (mut add_count, mut remove_count) = (0, 0);
    while *i < len {
        let line = &diffstr[*i];
        // `--- ` could be a removed line or the next file's header: prune the latter.
        if line.starts_with("--- ")
            && *i + 2 < len
            && diffstr[*i + 1].starts_with("+++ ")
            && diffstr[*i + 2].starts_with("@@")
        {
            break;
        }
        let op = if line.is_empty() && *i != len - 1 { Some(' ') } else { first_char(line) };
        match op {
            Some(c @ ('+' | '-' | ' ' | '\\')) => {
                hunk.lines.push(line.clone());
                hunk.delimiters.push(delimiters.get(*i).cloned().unwrap_or_else(|| "\n".to_string()));
                match c {
                    '+' => add_count += 1,
                    '-' => remove_count += 1,
                    ' ' => {
                        add_count += 1;
                        remove_count += 1;
                    }
                    _ => {}
                }
            }
            _ => break,
        }
        *i += 1;
    }
    // Handle the empty block count case.
    if add_count == 0 && hunk.new_lines == 1 {
        hunk.new_lines = 0;
    }
    if remove_count == 0 && hunk.old_lines == 1 {
        hunk.old_lines = 0;
    }
    hunk
}

/* ── applyPatch ── */

/// jsdiff's distance iterator: offsets 0, +1, -1, +2, -2 … within [min, max].
struct DistanceIter {
    start: isize,
    min: isize,
    max: isize,
    want_forward: bool,
    backward_exhausted: bool,
    forward_exhausted: bool,
    local_offset: isize,
}

impl DistanceIter {
    fn new(start: isize, min: isize, max: isize) -> Self {
        Self {
            start,
            min,
            max,
            want_forward: true,
            backward_exhausted: false,
            forward_exhausted: false,
            local_offset: 1,
        }
    }

    fn next_offset(&mut self) -> Option<isize> {
        loop {
            if self.want_forward && !self.forward_exhausted {
                if self.backward_exhausted {
                    self.local_offset += 1;
                } else {
                    self.want_forward = false;
                }
                if self.start + self.local_offset <= self.max {
                    return Some(self.local_offset);
                }
                self.forward_exhausted = true;
            }
            if !self.backward_exhausted {
                if !self.forward_exhausted {
                    self.want_forward = true;
                }
                if self.min <= self.start - self.local_offset {
                    let r = -self.local_offset;
                    self.local_offset += 1;
                    return Some(r);
                }
                self.backward_exhausted = true;
                continue;
            }
            return None;
        }
    }
}

fn op_and_content(line: &str) -> (char, &str) {
    match first_char(line) {
        Some(c) => (c, &line[c.len_utf8()..]),
        None => (' ', line),
    }
}

/// Does `hunk` fit exactly (fuzz 0) at line index `to_pos`?
fn hunk_fits(lines: &[String], hunk: &Hunk, mut to_pos: isize) -> bool {
    for line in &hunk.lines {
        let (op, content) = op_and_content(line);
        if op == ' ' || op == '-' {
            let actual = usize::try_from(to_pos).ok().and_then(|p| lines.get(p));
            if actual.map(String::as_str) != Some(content) {
                return false;
            }
            to_pos += 1;
        }
    }
    true
}

/// jsdiff `applyPatch(source, uniDiff)`: `Ok(Some(patched))`, `Ok(None)` when a
/// hunk does not fit anywhere (jsdiff's `false`), `Err` for a multi-file diff.
pub fn apply_patch(source: &str, uni_diff: &str) -> anyhow::Result<Option<String>> {
    let mut patches = parse_patch(uni_diff);
    if patches.len() > 1 {
        anyhow::bail!("applyPatch only works with a single input.");
    }
    let Some(mut patch) = patches.pop() else {
        return Ok(Some(source.to_string()));
    };

    let (mut lines, mut delimiters) = split_lines(source);
    let (mut min_line, mut offset) = (0isize, 0isize);

    // Search the best-fit offset for each hunk, given the previous ones.
    for hunk in patch.hunks.iter_mut() {
        if hunk.malformed {
            return Ok(None);
        }
        let max_line = lines.len() as isize - hunk.old_lines;
        let to_pos = offset + hunk.old_start - 1;
        let mut iter = DistanceIter::new(to_pos, min_line, max_line);
        let mut local = Some(0isize);
        while let Some(lo) = local {
            if hunk_fits(&lines, hunk, to_pos + lo) {
                offset += lo;
                hunk.offset = offset;
                break;
            }
            local = iter.next_offset();
        }
        if local.is_none() {
            return Ok(None);
        }
        // Later hunks may not fit over already patched text.
        min_line = hunk.offset + hunk.old_start + hunk.old_lines;
    }

    // Apply the hunks.
    let mut diff_offset = 0isize;
    let (mut remove_eofnl, mut add_eofnl) = (false, false);
    for hunk in &patch.hunks {
        let mut to_pos = (hunk.old_start + hunk.offset + diff_offset - 1).max(0) as usize;
        diff_offset += hunk.new_lines - hunk.old_lines;
        for (j, line) in hunk.lines.iter().enumerate() {
            let (op, content) = op_and_content(line);
            match op {
                ' ' => to_pos += 1,
                '-' => {
                    if to_pos < lines.len() {
                        lines.remove(to_pos);
                    }
                    if to_pos < delimiters.len() {
                        delimiters.remove(to_pos);
                    }
                }
                '+' => {
                    let delim = hunk.delimiters.get(j).cloned().unwrap_or_else(|| "\n".to_string());
                    lines.insert(to_pos.min(lines.len()), content.to_string());
                    delimiters.insert(to_pos.min(delimiters.len()), delim);
                    to_pos += 1;
                }
                '\\' => {
                    let prev = j.checked_sub(1).and_then(|p| hunk.lines.get(p)).and_then(|l| first_char(l));
                    match prev {
                        Some('+') => remove_eofnl = true,
                        Some('-') => add_eofnl = true,
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // EOF newline insertion / removal.
    if remove_eofnl {
        while lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
            delimiters.pop();
        }
    } else if add_eofnl {
        lines.push(String::new());
        delimiters.push("\n".to_string());
    }

    let mut out = String::new();
    let last = lines.len().saturating_sub(1);
    for (k, line) in lines.iter().enumerate() {
        out.push_str(line);
        if k < last {
            out.push_str(delimiters.get(k).map(String::as_str).unwrap_or(""));
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected strings were produced by jsdiff 5.2.2 (node) for the same inputs.
    const RULE: &str = "===================================================================";

    fn patch(old_name: &str, new_name: &str, old: &str, new: &str) -> String {
        create_two_files_patch(old_name, new_name, old, new, Some(""), Some(""))
    }

    #[test]
    fn rewrite_patch_matches_jsdiff() {
        assert_eq!(
            patch("b.txt", "b.txt", "one\n", "two\n"),
            format!("Index: b.txt\n{RULE}\n--- b.txt\t\n+++ b.txt\t\n@@ -1,1 +1,1 @@\n-one\n+two\n")
        );
    }

    #[test]
    fn create_from_empty_matches_jsdiff() {
        assert_eq!(
            patch("a.txt", "a.txt", "", "hi\n"),
            format!("Index: a.txt\n{RULE}\n--- a.txt\t\n+++ a.txt\t\n@@ -0,0 +1,1 @@\n+hi\n")
        );
    }

    #[test]
    fn delete_to_dev_null_matches_jsdiff() {
        assert_eq!(
            patch("a.txt", "/dev/null", "x\ny\n", ""),
            format!("{RULE}\n--- a.txt\t\n+++ /dev/null\t\n@@ -1,2 +0,0 @@\n-x\n-y\n\\ No newline at end of file\n")
        );
    }

    #[test]
    fn missing_final_newline_markers_match_jsdiff() {
        assert_eq!(
            patch("f", "f", "a\nb", "a\nc"),
            format!("Index: f\n{RULE}\n--- f\t\n+++ f\t\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n")
        );
        assert_eq!(
            patch("f", "f", "x", "x\ny"),
            format!("Index: f\n{RULE}\n--- f\t\n+++ f\t\n@@ -1,1 +1,2 @@\n-x\n\\ No newline at end of file\n+x\n+y\n\\ No newline at end of file\n")
        );
    }

    #[test]
    fn pure_removal_matches_jsdiff() {
        assert_eq!(
            patch("f", "f", "a\nb\nc\n", "a\nc\n"),
            format!("Index: f\n{RULE}\n--- f\t\n+++ f\t\n@@ -1,3 +1,2 @@\n a\n-b\n c\n")
        );
    }

    #[test]
    fn identical_inputs_have_no_hunks() {
        assert_eq!(patch("f", "f", "same\n", "same\n"), format!("Index: f\n{RULE}\n--- f\t\n+++ f\t\n"));
    }

    #[test]
    fn far_apart_changes_make_two_hunks() {
        let old: String = (1..=30).map(|i| format!("l{i}\n")).collect();
        let new = old.replace("l2\n", "L2\n").replace("l28\n", "L28\n");
        assert_eq!(
            patch("f", "f", &old, &new),
            format!(
                "Index: f\n{RULE}\n--- f\t\n+++ f\t\n@@ -1,6 +1,6 @@\n l1\n-l2\n+L2\n l3\n l4\n l5\n l6\n\
                 @@ -24,7 +24,7 @@\n l24\n l25\n l26\n l27\n-l28\n+L28\n l29\n l30\n"
            )
        );
    }

    #[test]
    fn round_trip_apply() {
        let old = "fn a() {}\nfn b() {}\nfn c() {}\n";
        let new = "fn a() {}\nfn b2() {}\nfn c() {}\nfn d() {}\n";
        let p = patch("x.rs", "x.rs", old, new);
        assert_eq!(apply_patch(old, &p).unwrap().as_deref(), Some(new));
    }

    #[test]
    fn apply_slides_hunk_to_matching_offset() {
        // The header says line 1, but the context really sits at line 3.
        let diff = "--- f\n+++ f\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n";
        assert_eq!(apply_patch("x\ny\nalpha\nbeta\n", diff).unwrap().as_deref(), Some("x\ny\nalpha\nBETA\n"));
    }

    #[test]
    fn apply_rejects_non_matching_context() {
        let diff = "Index: c.txt\n===\n--- c.txt\n+++ c.txt\n@@ -1,1 +1,1 @@\n-nonexistent line\n+replacement\n";
        assert_eq!(apply_patch("real content\n", diff).unwrap(), None);
    }

    #[test]
    fn apply_rejects_multi_file_diffs() {
        let diff = "--- a\n+++ a\n@@ -1 +1 @@\n-x\n+y\n--- b\n+++ b\n@@ -1 +1 @@\n-x\n+y\n";
        assert!(apply_patch("x\n", diff).is_err());
    }

    #[test]
    fn apply_to_empty_source_creates_content() {
        let diff = "--- /dev/null\n+++ n.txt\n@@ -0,0 +1,2 @@\n+one\n+two\n";
        assert_eq!(apply_patch("", diff).unwrap().as_deref(), Some("one\ntwo\n"));
    }

    #[test]
    fn text_without_hunks_is_a_no_op_like_jsdiff() {
        assert_eq!(apply_patch("keep\n", "hello this is not a diff").unwrap().as_deref(), Some("keep\n"));
    }

    #[test]
    fn apply_handles_no_newline_markers() {
        let (a, b) = ("a\nb", "a\nc\n");
        assert_eq!(apply_patch(a, &patch("f", "f", a, b)).unwrap().as_deref(), Some(b));
        assert_eq!(apply_patch(b, &patch("f", "f", b, a)).unwrap().as_deref(), Some(a));
    }

    #[test]
    fn crlf_lines_keep_their_delimiters_like_jsdiff() {
        let diff = "--- f\n+++ f\n@@ -2,1 +2,1 @@\n-b\n+B\n";
        assert_eq!(apply_patch("a\r\nb\r\nc\r\n", diff).unwrap().as_deref(), Some("a\r\nB\nc\r\n"));
    }

    #[test]
    fn distance_iterator_order() {
        let mut it = DistanceIter::new(2, 0, 4);
        let got: Vec<isize> = std::iter::from_fn(|| it.next_offset()).collect();
        // From start 2 within [0, 4]: 3, 1, 4, 0 → offsets +1, -1, +2, -2.
        assert_eq!(got, vec![1, -1, 2, -2]);
    }
}
