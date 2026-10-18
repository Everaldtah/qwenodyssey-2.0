//! Deterministic repair for small-model structured output (ported from
//! `src/core/repair.ts`).
//!
//! 7B–20B models frequently emit *almost* valid JSON: trailing commas, single
//! quotes, unquoted keys, ```json fences, smart quotes, a stray prose preamble,
//! an unterminated string at the token limit. Rather than burn a turn asking the
//! model to "try again", we repair the common failure shapes deterministically
//! and only escalate to a re-ask if repair genuinely fails.
//!
//! This complements [`crate::parse`] (which extracts a JSON span); here we focus
//! on *fixing* a span that didn't parse.

use std::sync::LazyLock;

use regex::Regex;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Outcome of [`repair_json`].
#[derive(Clone, Debug, PartialEq)]
pub struct RepairResult {
    pub ok: bool,
    /// The parsed value when `ok`.
    pub value: Option<Value>,
    /// False only on the fast path (the input was already valid JSON).
    pub repaired: bool,
    /// What was done, in order (e.g. "stripped fences/prose", "quote bare keys").
    pub notes: Vec<String>,
}

impl RepairResult {
    /// The value deserialized into `T`, when repair succeeded and it fits.
    pub fn value_as<T: DeserializeOwned>(&self) -> Option<T> {
        self.value.clone().and_then(|v| serde_json::from_value(v).ok())
    }
}

fn re(pattern: &str) -> Regex {
    // The patterns are compile-time constants covered by the tests below.
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad repair regex {pattern}: {e}"))
}

static FENCE_OPEN: LazyLock<Regex> = LazyLock::new(|| re(r"^```[a-zA-Z]*\s*\n?"));
static FENCE_CLOSE: LazyLock<Regex> = LazyLock::new(|| re(r"\n?```\s*$"));
static TRAILING_COMMA: LazyLock<Regex> = LazyLock::new(|| re(r",\s*([}\]])"));
// JS `\b` is an ASCII word boundary.
static PY_TRUE: LazyLock<Regex> = LazyLock::new(|| re(r"(?-u:\b)True(?-u:\b)"));
static PY_FALSE: LazyLock<Regex> = LazyLock::new(|| re(r"(?-u:\b)False(?-u:\b)"));
static PY_NONE: LazyLock<Regex> = LazyLock::new(|| re(r"(?-u:\b)None(?-u:\b)"));
static BARE_KEY: LazyLock<Regex> =
    LazyLock::new(|| re(r"([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)\s*:"));

/// Strip markdown code fences and any prose before the first JSON delimiter.
pub(crate) fn strip_to_json(text: &str) -> String {
    let mut s = text.trim().to_string();
    // Remove ```json ... ``` fences.
    s = FENCE_OPEN.replace(&s, "").into_owned();
    s = FENCE_CLOSE.replace(&s, "").into_owned();
    // Cut to the first { or [ and the last } or ].
    if let Some(start) = s.find(['{', '[']) {
        if start > 0 {
            s = s[start..].to_string();
        }
    }
    if let Some(end) = s.rfind(['}', ']']) {
        if end < s.len() - 1 {
            s.truncate(end + 1);
        }
    }
    s.trim().to_string()
}

/// Normalise quotes/commas/keys that small models commonly get wrong.
fn normalize_jsonish(s: &str) -> String {
    // Smart quotes → straight quotes.
    let out: String = s
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            other => other,
        })
        .collect();
    // Trailing commas before } or ].
    let out = TRAILING_COMMA.replace_all(&out, "$1");
    // Python literals.
    let out = PY_TRUE.replace_all(&out, "true");
    let out = PY_FALSE.replace_all(&out, "false");
    PY_NONE.replace_all(&out, "null").into_owned()
}

/// Convert single-quoted strings/keys to double-quoted, *outside* of existing
/// double-quoted strings. A small state machine so we don't corrupt apostrophes
/// inside valid strings.
fn single_to_double_quotes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_double = false;
    let mut in_single = false;
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if c == '"' && prev != Some('\\') && !in_single {
            in_double = !in_double;
            out.push(c);
        } else if c == '\'' && prev != Some('\\') && !in_double {
            in_single = !in_single;
            out.push('"'); // re-emit as a double quote
        } else {
            out.push(c);
        }
        prev = Some(c);
    }
    out
}

/// Quote bare object keys: `{key: 1}` → `{"key": 1}`.
fn quote_bare_keys(s: &str) -> String {
    BARE_KEY.replace_all(s, "${1}\"${2}\":").into_owned()
}

/// Best-effort close of an unterminated trailing string + open brackets.
fn close_unterminated(s: &str) -> String {
    let mut out = s.to_string();
    // Count unescaped double quotes; if odd, the last string is unterminated.
    let bytes = s.as_bytes();
    let quotes = bytes
        .iter()
        .enumerate()
        .filter(|&(i, &b)| b == b'"' && (i == 0 || bytes[i - 1] != b'\\'))
        .count();
    if quotes % 2 == 1 {
        out.push('"');
    }
    // Balance braces/brackets.
    let count = |ch: char| out.matches(ch).count();
    let (opens_curly, closes_curly) = (count('{'), count('}'));
    let (opens_sq, closes_sq) = (count('['), count(']'));
    out.push_str(&"]".repeat(opens_sq.saturating_sub(closes_sq)));
    out.push_str(&"}".repeat(opens_curly.saturating_sub(closes_curly)));
    out
}

/// Parse JSON, repairing common small-model malformations. Returns the parsed
/// value plus whether repair was needed and what was done.
pub fn repair_json(text: &str) -> RepairResult {
    let mut notes: Vec<String> = Vec::new();

    // Fast path: already valid.
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return RepairResult { ok: true, value: Some(v), repaired: false, notes };
    }

    let s = strip_to_json(text);
    if s != text.trim() {
        notes.push("stripped fences/prose".into());
    }

    type Fix = fn(&str) -> String;
    let attempts: [(&str, Fix); 4] = [
        ("normalize quotes/commas", normalize_jsonish),
        ("single→double quotes", |x| single_to_double_quotes(&normalize_jsonish(x))),
        ("quote bare keys", |x| quote_bare_keys(&single_to_double_quotes(&normalize_jsonish(x)))),
        ("close unterminated", |x| {
            close_unterminated(&quote_bare_keys(&single_to_double_quotes(&normalize_jsonish(x))))
        }),
    ];

    for (label, fix) in attempts {
        if let Ok(v) = serde_json::from_str::<Value>(&fix(&s)) {
            notes.push(label.into());
            return RepairResult { ok: true, value: Some(v), repaired: true, notes };
        }
    }

    notes.push("all repair strategies failed".into());
    RepairResult { ok: false, value: None, repaired: true, notes }
}

/// Keys [`coerce_array`] unwraps by default.
pub const DEFAULT_ARRAY_KEYS: &[&str] = &["items", "results", "edits", "steps", "issues"];

/// Coerce a value to an array. Small models sometimes return a single object
/// where an array was expected, or wrap the array in a key (`keys`, tried in
/// order; pass [`DEFAULT_ARRAY_KEYS`] for the TS default). Scalars and null → [].
pub fn coerce_array(value: Value, keys: &[&str]) -> Vec<Value> {
    match value {
        Value::Array(items) => items,
        Value::Object(mut map) => {
            for k in keys {
                if matches!(map.get(*k), Some(Value::Array(_))) {
                    if let Some(Value::Array(items)) = map.remove(*k) {
                        return items;
                    }
                }
            }
            vec![Value::Object(map)] // single object → one-element array
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_json_is_not_marked_repaired() {
        let r = repair_json(r#"{"a":1}"#);
        assert!(r.ok);
        assert!(!r.repaired);
        assert_eq!(r.value.unwrap()["a"], 1);
    }

    #[test]
    fn strips_code_fences_and_prose() {
        let r = repair_json("Sure! Here you go:\n```json\n{\"x\": 5}\n```");
        assert!(r.ok);
        assert_eq!(r.value.unwrap()["x"], 5);
        assert_eq!(r.notes[0], "stripped fences/prose");
    }

    #[test]
    fn removes_trailing_commas() {
        let r = repair_json(r#"{"a": 1, "b": [1, 2, 3,],}"#);
        assert!(r.ok);
        assert_eq!(r.value.unwrap()["b"], json!([1, 2, 3]));
    }

    #[test]
    fn converts_single_quotes_and_bare_keys() {
        let r = repair_json("{name: 'qwen', n: 7}");
        assert!(r.ok);
        let v = r.value.unwrap();
        assert_eq!(v["name"], "qwen");
        assert_eq!(v["n"], 7);
        assert_eq!(r.notes.last().unwrap(), "quote bare keys");
    }

    #[test]
    fn closes_unterminated_object_and_string() {
        let r = repair_json(r#"{"path": "src/a.ts", "content": "unterminated"#);
        assert!(r.ok);
        let v = r.value.unwrap();
        assert_eq!(v["path"], "src/a.ts");
        assert_eq!(v["content"], "unterminated");
    }

    #[test]
    fn normalizes_python_literals() {
        let r = repair_json(r#"{"ok": True, "bad": False, "x": None}"#);
        assert!(r.ok);
        let v = r.value.unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["bad"], false);
        assert_eq!(v["x"], Value::Null);
    }

    #[test]
    fn smart_quotes_are_straightened() {
        let r = repair_json("{\u{201C}a\u{201D}: \u{201C}b\u{201D}}");
        assert!(r.ok);
        assert_eq!(r.value.unwrap()["a"], "b");
    }

    #[test]
    fn apostrophes_inside_double_quoted_strings_survive() {
        let r = repair_json(r#"{"msg": "it's fine",}"#);
        assert!(r.ok);
        assert_eq!(r.value.unwrap()["msg"], "it's fine");
    }

    #[test]
    fn hopeless_input_fails_with_note() {
        let r = repair_json("no json here at all");
        assert!(!r.ok);
        assert!(r.repaired);
        assert_eq!(r.notes.last().unwrap(), "all repair strategies failed");
    }

    #[test]
    fn value_as_deserializes() {
        #[derive(serde::Deserialize)]
        struct S {
            n: u32,
        }
        assert_eq!(repair_json("{n: 3}").value_as::<S>().unwrap().n, 3);
    }

    #[test]
    fn coerce_array_cases() {
        assert_eq!(coerce_array(json!([1, 2]), DEFAULT_ARRAY_KEYS), vec![json!(1), json!(2)]);
        assert_eq!(coerce_array(json!({"edits": [{"p": 1}]}), DEFAULT_ARRAY_KEYS), vec![json!({"p": 1})]);
        assert_eq!(coerce_array(json!({"path": "x"}), DEFAULT_ARRAY_KEYS), vec![json!({"path": "x"})]);
        assert!(coerce_array(Value::Null, DEFAULT_ARRAY_KEYS).is_empty());
        assert!(coerce_array(json!("s"), DEFAULT_ARRAY_KEYS).is_empty());
        assert_eq!(coerce_array(json!({"list": [1]}), &["list"]), vec![json!(1)]);
    }
}
