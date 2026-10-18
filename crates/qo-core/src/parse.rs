//! Tolerant JSON extraction for small-model output (ported from
//! `src/core/parse.ts`). 7B models often wrap JSON in prose or ```json fences,
//! or emit trailing commas — recover what we can. Falls back to
//! [`crate::repair`] for the harder malformations (single quotes, unquoted keys,
//! unterminated strings at the token limit).

use std::sync::LazyLock;

use regex::Regex;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::repair::repair_json;

fn re(pattern: &str) -> Regex {
    // The patterns are compile-time constants covered by the tests below.
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad parse regex {pattern}: {e}"))
}

static FENCE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)```(?:json)?\s*([\s\S]*?)```"));
static LEADING_FENCE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)^\s*```(?:json)?"));
static TRAILING_FENCE: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)```\s*$"));
static TRAILING_COMMA: LazyLock<Regex> = LazyLock::new(|| re(r",\s*([}\]])"));

/// Extract the first JSON value from model output: a fenced ```json block, else
/// the first balanced `{…}`/`[…]` span, else the whole string — each tried as-is
/// (trailing commas tolerated), then with deterministic repair. `None` when
/// nothing parses.
pub fn extract_json(text: &str) -> Option<Value> {
    if text.is_empty() {
        return None;
    }

    let mut candidates: Vec<String> = Vec::new();
    // 1. Fenced ```json block
    if let Some(c) = FENCE.captures(text) {
        candidates.push(c.get(1).map_or("", |m| m.as_str()).to_string());
    }
    // 2. First balanced {...} or [...] span
    if let Some(span) = balanced_span(text) {
        candidates.push(span.to_string());
    }
    // 3. Whole string
    candidates.push(text.trim().to_string());

    for c in &candidates {
        if let Some(v) = try_parse(c) {
            return Some(v);
        }
    }
    // Escalate: deterministic repair of the harder malformations (single-quoted
    // strings, bare keys, Python literals, unterminated trailing string).
    for c in &candidates {
        let r = repair_json(c);
        if r.ok {
            return r.value;
        }
    }
    None
}

/// [`extract_json`] deserialized into `T` (`None` if nothing parses or the
/// first parsed value doesn't fit `T`).
pub fn extract_json_as<T: DeserializeOwned>(text: &str) -> Option<T> {
    extract_json(text).and_then(|v| serde_json::from_value(v).ok())
}

/// Strip stray fences and trailing commas, then parse strictly.
fn try_parse(s: &str) -> Option<Value> {
    let cleaned = LEADING_FENCE.replace(s, "");
    let cleaned = TRAILING_FENCE.replace(&cleaned, "");
    let cleaned = TRAILING_COMMA.replace_all(&cleaned, "$1");
    serde_json::from_str(cleaned.trim()).ok()
}

/// The first balanced `{…}`/`[…]` span in `text`.
fn balanced_span(text: &str) -> Option<&str> {
    let start = text.find(['{', '['])?;
    balanced_span_at(text, start).map(|(span, _)| span)
}

/// The balanced `{…}`/`[…]` span starting at byte `start`, plus the byte index
/// just past it. Only the opening delimiter's own kind is counted, and
/// double-quoted strings (with escapes) are skipped.
fn balanced_span_at(text: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = text.as_bytes();
    let open = *bytes.get(start)?;
    if open != b'{' && open != b'[' {
        return None;
    }
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    // Delimiters are ASCII, so scanning bytes is safe on UTF-8 (multi-byte
    // sequences never contain ASCII bytes) and the slice bounds are char bounds.
    for (i, &ch) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if ch == b'\\' {
                esc = true;
            } else if ch == b'"' {
                in_str = false;
            }
            continue;
        }
        if ch == b'"' {
            in_str = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some((&text[start..=i], i + 1));
            }
        }
    }
    None
}

/// Every top-level balanced JSON value embedded in free text, parsed loosely.
/// Small models often emit one or more tool-call objects inline (sometimes in
/// separate ```json fences); this recovers all of them in document order.
pub fn extract_all_json(text: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < text.len() {
        let Some(rel) = text[i..].find(['{', '[']) else { break };
        let start = i + rel;
        let Some((span, end)) = balanced_span_at(text, start) else {
            i = start + 1;
            continue;
        };
        let parsed = try_parse(span).or_else(|| {
            // Near-valid span (single quotes, bare keys…) — try deterministic repair
            // before giving up, so a malformed inline tool call still executes.
            let r = repair_json(span);
            if r.ok {
                r.value
            } else {
                None
            }
        });
        if let Some(v) = parsed {
            out.push(v);
        }
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── extractJson ──

    #[test]
    fn parses_bare_json() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(json!({"a": 1})));
    }

    #[test]
    fn parses_fenced_json_with_prose() {
        let text = "Sure! Here it is:\n```json\n{\"edits\": [], \"notes\": \"ok\"}\n```\nDone.";
        assert_eq!(extract_json(text), Some(json!({"edits": [], "notes": "ok"})));
    }

    #[test]
    fn recovers_balanced_object_in_prose() {
        let text = r#"The plan is {"goal":"x","steps":[{"index":1,"title":"t"}]} and that is all."#;
        let out = extract_json(text).unwrap();
        assert_eq!(out["goal"], "x");
        assert_eq!(out["steps"][0]["title"], "t");
    }

    #[test]
    fn tolerates_trailing_commas() {
        assert_eq!(extract_json(r#"{"a":1,"b":2,}"#), Some(json!({"a": 1, "b": 2})));
    }

    #[test]
    fn none_when_no_json() {
        assert_eq!(extract_json("just words"), None);
        assert_eq!(extract_json(""), None);
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_span() {
        let text = r#"x {"s": "a } b { c", "n": [1]} y"#;
        assert_eq!(extract_json(text), Some(json!({"s": "a } b { c", "n": [1]})));
    }

    #[test]
    fn typed_extraction() {
        #[derive(serde::Deserialize)]
        struct P {
            goal: String,
        }
        let p: P = extract_json_as(r#"ok {"goal": "g"}"#).unwrap();
        assert_eq!(p.goal, "g");
        assert!(extract_json_as::<P>(r#"{"other": 1}"#).is_none());
    }

    // ── repair escalation (smallModelOptims.test.ts) ──

    #[test]
    fn recovers_single_quoted_json() {
        assert_eq!(
            extract_json("{'name': 'run_shell', 'arguments': {'command': 'ls'}}"),
            Some(json!({"name": "run_shell", "arguments": {"command": "ls"}}))
        );
    }

    #[test]
    fn recovers_bare_keys_and_python_literals() {
        assert_eq!(
            extract_json(r#"{name: "x", ok: True, missing: None}"#),
            Some(json!({"name": "x", "ok": true, "missing": null}))
        );
    }

    // ── extractAllJson ──

    #[test]
    fn all_recovers_fenced_tool_call() {
        let text = "```json\n{\"name\": \"shell_help\", \"arguments\": {\"query\": \"network tests\"}}\n```";
        assert_eq!(
            extract_all_json(text),
            vec![json!({"name": "shell_help", "arguments": {"query": "network tests"}})]
        );
    }

    #[test]
    fn all_recovers_multiple_fences_in_order() {
        let text = concat!(
            "first:\n```json\n{\"name\":\"run_shell\",\"arguments\":{\"command\":\"ls\"}}\n```\n",
            "then:\n```json\n{\"name\":\"update_plan\",\"arguments\":{}}\n```"
        );
        let names: Vec<Value> = extract_all_json(text).into_iter().map(|o| o["name"].clone()).collect();
        assert_eq!(names, vec![json!("run_shell"), json!("update_plan")]);
    }

    #[test]
    fn all_recovers_bare_object_in_prose() {
        let text = r#"Sure! {"name":"run_shell","arguments":{"command":"echo hi"}}"#;
        assert_eq!(
            extract_all_json(text),
            vec![json!({"name": "run_shell", "arguments": {"command": "echo hi"}})]
        );
    }

    #[test]
    fn all_empty_for_plain_prose() {
        assert!(extract_all_json("the time is 14:05, no json here").is_empty());
        assert!(extract_all_json("").is_empty());
    }

    #[test]
    fn all_repairs_malformed_inline_call_among_valid_ones() {
        let text = r#"First {"name":"read_file","arguments":{"path":"a.ts"}} then {'name': 'run_shell', 'arguments': {'command': 'ls'}}"#;
        let all = extract_all_json(text);
        assert_eq!(all.len(), 2);
        assert_eq!(all[1], json!({"name": "run_shell", "arguments": {"command": "ls"}}));
    }

    #[test]
    fn all_skips_unbalanced_openers_and_handles_unicode() {
        let text = "ünïcödé [ not closed {\"a\": \"é\"}";
        assert_eq!(extract_all_json(text), vec![json!({"a": "é"})]);
    }
}
