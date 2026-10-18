//! Small argument / string helpers shared by the tools in this crate.
//!
//! Tool arguments come from models, so every reader is forgiving: the TS code
//! leaned on JavaScript coercions (`String(x || "")`, `Number(x) || d`), and these
//! helpers reproduce those semantics for `serde_json::Value`.

use std::future::Future;

use qo_core::ToolContext;
use serde_json::Value;

/// What a tool reports when the user interrupted the turn (Esc).
pub(crate) const INTERRUPTED: &str = "interrupted";

/// Run `fut` unless the turn's cancellation token fires first. `None` means the
/// user interrupted: the future is dropped on the spot, which aborts any HTTP
/// request in flight (reqwest futures are cancel-safe) and kills children
/// spawned with `kill_on_drop`.
pub(crate) async fn cancellable<F: Future>(ctx: &ToolContext, fut: F) -> Option<F::Output> {
    match &ctx.cancel {
        None => Some(fut.await),
        Some(token) => {
            if token.is_cancelled() {
                return None;
            }
            tokio::select! {
                biased;
                _ = token.cancelled() => None,
                out = fut => Some(out),
            }
        }
    }
}

/// JavaScript's `\s` class (what the TS regexes and `String.trim()` treat as
/// whitespace). Differs from `char::is_whitespace` on U+0085 and U+FEFF.
pub(crate) fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{000B}' | '\u{000C}' | '\r' | ' ' | '\u{00A0}' | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'
    )
}

/// `String.prototype.trim()`.
pub(crate) fn js_trim(s: &str) -> &str {
    s.trim_matches(is_js_space)
}

/// JS-style truthiness of a JSON value (`null`, `false`, `0`, `""` are falsy).
pub(crate) fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(_)) | Some(Value::Object(_)) => true,
    }
}

/// `String(v)` for a JSON value: strings verbatim, scalars printed, null → "".
pub(crate) fn value_to_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `String(args[key] || "")` — the first truthy key wins (e.g. `query` then `q`).
pub(crate) fn str_arg(args: &Value, keys: &[&str]) -> String {
    for k in keys {
        let v = args.get(*k);
        if truthy(v) {
            return v.map(value_to_string).unwrap_or_default();
        }
    }
    String::new()
}

/// `Number(v)`: numbers as-is, numeric strings parsed, booleans 1/0; `None` for NaN.
pub(crate) fn num_arg(args: &Value, key: &str) -> Option<f64> {
    let n = match args.get(key)? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else {
                t.parse::<f64>().ok()?
            }
        }
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Null => 0.0,
        _ => return None,
    };
    if n.is_nan() {
        None
    } else {
        Some(n)
    }
}

/// `Number(args[key]) || default` — zero / NaN / missing fall back to `default`.
pub(crate) fn num_or(args: &Value, key: &str, default: f64) -> f64 {
    match num_arg(args, key) {
        Some(n) if n != 0.0 => n,
        _ => default,
    }
}

/// `String(n)` for a JS number: integral values print without a decimal point.
pub(crate) fn js_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// First `n` characters of `s` (the TS `slice(0, n)` on mostly-BMP text).
pub(crate) fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Character count (TS `.length` for BMP text).
pub(crate) fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `s.slice(0, n)` plus `suffix` when something was cut.
pub(crate) fn truncate_chars(s: &str, n: usize, suffix: &str) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}{suffix}", &s[..i]),
        None => s.to_string(),
    }
}

/// Collapse every whitespace run to one space (`replace(/\s+/g, " ")`).
pub(crate) fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for ch in s.chars() {
        if is_js_space(ch) {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(ch);
            in_ws = false;
        }
    }
    out
}

/// `new Date().toISOString()` — UTC, millisecond precision, `Z` suffix. The
/// knowledge index compares these strings for equality, so the format must
/// match what the TypeScript version wrote.
pub(crate) fn iso_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn js_coercions() {
        let a = json!({"q": "x", "query": "", "n": "4", "z": 0, "f": 2.7});
        assert_eq!(str_arg(&a, &["query", "q"]), "x");
        assert_eq!(num_or(&a, "n", 3.0), 4.0);
        assert_eq!(num_or(&a, "z", 3.0), 3.0);
        assert_eq!(num_or(&a, "missing", 3.0), 3.0);
        assert_eq!(num_or(&a, "f", 3.0), 2.7);
        assert_eq!(take_chars("héllo", 2), "hé");
        assert_eq!(collapse_ws(" a \n\t b "), " a b ");
        let now = iso_now();
        assert!(now.ends_with('Z') && now.len() == 24, "{now}");
    }
}
