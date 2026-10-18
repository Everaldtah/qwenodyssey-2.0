//! Pre-dispatch hardening for tool calls coming from small models. Two jobs:
//!
//!  1. FUZZY NAME RESOLUTION — small models hallucinate tool names ("read" for
//!     "read_file", "search" for "grep"). We map an unknown name onto the nearest
//!     advertised tool within a small edit distance, or list suggestions.
//!  2. SCHEMA COERCION + VALIDATION — they also send the wrong JSON types
//!     ("true"/"1" for booleans, "5" for numbers), alias keys, or omit a required
//!     field. We coerce each argument to the type its schema declares and return a
//!     TARGETED message ("read_file needs `path`") instead of letting the tool
//!     fail with a cryptic error the model can't act on.
//!
//! The result feeds back into the chat loop: on a hard problem (unresolvable name
//! / missing required arg) we hand the model a corrective tool message rather than
//! dispatching a doomed call.

use std::sync::LazyLock;

use qo_core::ToolSpec;
use regex::Regex;
use serde_json::{Map, Number, Value};

use crate::util::parse_js_number;

/// A tool call after name resolution and argument coercion.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedCall {
    /// Resolved tool name (may differ from the requested one after fuzzy-match).
    pub name: String,
    /// Coerced arguments (always a JSON object).
    pub arguments: Value,
    /// Set when the call cannot proceed; this string is fed back to the model.
    pub error: Option<String>,
    /// Non-fatal note (e.g. "interpreted `read` as `read_file`") to surface.
    pub note: Option<String>,
}

/// Curated aliases for tool names small models hallucinate. These come from
/// OTHER harnesses' tool vocabularies baked into model training data (Claude
/// Code's str_replace_editor, OpenHands' execute_command, generic "bash"/"cat"/
/// "ls"). Deterministic table lookup beats fuzzy matching for these: "bash" is
/// nowhere near "run_shell" by edit distance, yet the intent is unambiguous.
/// Only applied when the target tool is actually advertised.
pub const TOOL_ALIASES: &[(&str, &str)] = &[
    // shell
    ("bash", "run_shell"), ("shell", "run_shell"), ("sh", "run_shell"), ("powershell", "run_shell"),
    ("terminal", "run_shell"), ("exec", "run_shell"), ("execute", "run_shell"),
    ("execute_command", "run_shell"), ("run_command", "run_shell"), ("run_terminal_cmd", "run_shell"),
    ("shell_command", "run_shell"), ("cmd", "run_shell"),
    // file reads
    ("cat", "read_file"), ("open", "read_file"), ("open_file", "read_file"), ("view", "read_file"),
    ("view_file", "read_file"), ("get_file", "read_file"), ("show_file", "read_file"),
    // file writes/edits
    ("write", "write_file"), ("create_file", "write_file"), ("save_file", "write_file"),
    ("new_file", "write_file"), ("edit_file", "apply_edit"), ("str_replace_editor", "apply_edit"),
    ("str_replace", "apply_edit"), ("replace_in_file", "apply_edit"), ("edit", "apply_edit"),
    // listing / search
    ("ls", "list_files"), ("dir", "list_files"), ("list_dir", "list_files"),
    ("list_directory", "list_files"), ("find", "list_files"), ("glob", "list_files"),
    ("search", "grep"), ("search_files", "grep"), ("search_code", "grep"), ("code_search", "grep"),
    ("rg", "grep"), ("ripgrep", "grep"), ("search_file_content", "grep"),
    // web
    ("search_web", "web_search"), ("google", "web_search"), ("internet_search", "web_search"),
    ("browse", "web_fetch"), ("fetch", "web_fetch"), ("fetch_url", "web_fetch"),
    ("get_url", "web_fetch"), ("curl", "web_fetch"), ("open_url", "web_fetch"),
    // memory
    ("remember", "knowledge_save"), ("save_memory", "knowledge_save"), ("memorize", "knowledge_save"),
    ("recall", "knowledge_search"), ("memory_search", "knowledge_search"), ("search_memory", "knowledge_search"),
    // planning
    ("todo", "update_plan"), ("todo_write", "update_plan"), ("plan", "update_plan"),
    ("update_todo", "update_plan"), ("create_plan", "update_plan"),
    // deliberation
    ("thought", "think"), ("thinking", "think"), ("reflect", "think"), ("reflection", "think"),
    ("reason", "think"), ("internal_monologue", "think"),
];

/// Aliases for ARGUMENT keys small models substitute for the schema's real key
/// ({"file": "x.ts"} for read_file, {"cmd": "…"} for run_shell). For each schema
/// property that is missing, we adopt the value of the first alias present —
/// but never steal a key that is itself a valid property of the same tool.
pub const ARG_ALIASES: &[(&str, &[&str])] = &[
    ("path", &["file", "filename", "file_path", "filepath", "file_name", "directory", "dir", "folder", "target"]),
    ("command", &["cmd", "script", "shell_command", "command_line", "input"]),
    ("content", &["text", "contents", "body", "data", "file_text", "new_content", "code"]),
    ("pattern", &["regex", "regexp", "search", "search_pattern", "search_term", "search_string"]),
    ("query", &["q", "search_query", "question", "topic", "keywords"]),
    ("url", &["link", "uri", "website", "address"]),
    ("search", &["old_str", "old_string", "old_text", "find", "original"]),
    ("replace", &["new_str", "new_string", "new_text", "replacement", "updated"]),
    ("name", &["symbol", "symbol_name", "function_name", "identifier"]),
    ("timeout_ms", &["timeout", "timeout_seconds", "max_time"]),
    ("thought", &["thinking", "reasoning", "reflection", "analysis"]),
];

fn tool_alias(name: &str) -> Option<&'static str> {
    TOOL_ALIASES.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}

/// Classic Levenshtein edit distance (small strings — tool names).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut cur = vec![i; n + 1];
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[n]
}

/// Outcome of [`resolve_tool_name`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolNameResolution {
    pub name: Option<String>,
    pub suggestions: Vec<String>,
}

static NAMESPACE_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(functions?|tools?|default_api|api|assistant)[._:/]").expect("static regex"));

/// Resolve a (possibly hallucinated) tool name against the advertised set.
/// Exact match wins. Otherwise we accept the single nearest name within an edit
/// distance that scales with length (so "read"→"read_file" via substring, and
/// "grpe"→"grep" via distance), and only when it's unambiguously closest.
pub fn resolve_tool_name<S: AsRef<str>>(requested: &str, known: &[S]) -> ToolNameResolution {
    let known: Vec<&str> = known.iter().map(AsRef::as_ref).collect();
    let found = |n: &str| ToolNameResolution { name: Some(n.to_string()), suggestions: Vec::new() };
    if known.contains(&requested) {
        return found(requested);
    }
    let lower = requested.to_lowercase();

    // Curated alias table first — deterministic and cheap. Models trained on other
    // harnesses emit "bash"/"cat"/"str_replace_editor"; edit distance can't bridge
    // those, a lookup can. Also try with a hallucinated namespace prefix stripped
    // ("functions.run_shell", "tools.read_file", "default_api.grep").
    let deprefixed = NAMESPACE_PREFIX.replace(&lower, "").into_owned();
    for cand in [lower.as_str(), deprefixed.as_str()] {
        if let Some(exact) = known.iter().find(|k| k.to_lowercase() == cand) {
            return found(exact);
        }
        if let Some(alias) = tool_alias(cand) {
            if known.contains(&alias) {
                return found(alias);
            }
        }
    }

    // Substring / prefix aliasing: "read" → "read_file", "web" → "web_search".
    let subs: Vec<&str> = known
        .iter()
        .copied()
        .filter(|k| {
            let kl = k.to_lowercase();
            kl.starts_with(&lower) || kl.contains(&lower) || lower.contains(&kl)
        })
        .collect();
    if subs.len() == 1 {
        return found(subs[0]);
    }

    // Edit-distance: pick the closest within a length-scaled threshold.
    let mut scored: Vec<(&str, usize)> = known.iter().map(|k| (*k, levenshtein(&lower, &k.to_lowercase()))).collect();
    scored.sort_by_key(|(_, d)| *d);
    let threshold = 2usize.max(requested.chars().count() / 3);
    if let Some(&(best, d)) = scored.first() {
        let unambiguous = scored.get(1).map_or(true, |(_, d2)| *d2 > d);
        if d <= threshold && unambiguous {
            return found(best);
        }
    }
    // Ambiguous: hand back a few suggestions for the model to pick from.
    let suggestions: Vec<String> = if subs.is_empty() {
        scored.iter().take(3).map(|(k, _)| k.to_string()).collect()
    } else {
        subs.iter().map(|s| s.to_string()).collect()
    };
    ToolNameResolution { name: None, suggestions: suggestions.into_iter().take(4).collect() }
}

/// A JSON number from an f64, integral values as integers (like JS JSON).
fn json_number(n: f64) -> Value {
    if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
        Value::Number(Number::from(n as i64))
    } else {
        Number::from_f64(n).map(Value::Number).unwrap_or(Value::Null)
    }
}

/// JS `String(value)` for coercion purposes (`JSON.stringify` for non-strings).
fn stringify(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

static TRUTHY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(true|1|yes|on)$").expect("static regex"));
static FALSY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^(false|0|no|off)$").expect("static regex"));

/// Coerce a single value to the JSON-schema `type` a small model probably fumbled.
fn coerce_value(value: Value, schema: &Value) -> Value {
    if value.is_null() || !schema.is_object() {
        return value;
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => match value {
            Value::String(_) => value,
            other => Value::String(stringify(&other)),
        },
        Some(t @ ("integer" | "number")) => {
            if value.is_number() {
                return value;
            }
            let n = parse_js_number(&crate::util::js_string(Some(&value)));
            if n.is_finite() {
                json_number(if t == "integer" { n.trunc() } else { n })
            } else {
                value
            }
        }
        Some("boolean") => {
            if value.is_boolean() {
                return value;
            }
            let s = crate::util::js_string(Some(&value));
            let s = s.trim();
            if TRUTHY.is_match(s) {
                Value::Bool(true)
            } else if FALSY.is_match(s) {
                Value::Bool(false)
            } else {
                value
            }
        }
        Some("array") => match value {
            Value::Array(items) => match schema.get("items") {
                Some(item_schema) => Value::Array(items.into_iter().map(|v| coerce_value(v, item_schema)).collect()),
                None => Value::Array(items),
            },
            // A model that sent a single scalar where a list was wanted → wrap it.
            Value::String(s) if s.contains(',') => Value::Array(
                s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| Value::String(x.to_string())).collect(),
            ),
            other => Value::Array(vec![other]),
        },
        Some("object") => match &value {
            Value::String(s) => serde_json::from_str(s).unwrap_or(value),
            _ => value,
        },
        _ => value,
    }
}

/// Validate + coerce a tool call against its advertised schema, after resolving
/// the tool name. Returns a [`PreparedCall`] the chat loop can either dispatch or
/// feed back to the model as a correction.
pub fn prepare_tool_call(name: &str, args: Option<&Value>, specs: &[ToolSpec]) -> PreparedCall {
    let known: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let original: Map<String, Value> = match args {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    let resolved = resolve_tool_name(name, &known);
    let (Some(resolved_name), Some(spec)) =
        (resolved.name.clone(), resolved.name.as_deref().and_then(|n| specs.iter().find(|s| s.name == n)))
    else {
        let hint = if resolved.suggestions.is_empty() {
            format!(" Available tools: {}.", known.join(", "))
        } else {
            format!(" Did you mean: {}?", resolved.suggestions.join(", "))
        };
        return PreparedCall {
            name: name.to_string(),
            arguments: Value::Object(original),
            error: Some(format!("No tool named \"{name}\".{hint}")),
            note: None,
        };
    };

    let mut note = (resolved_name != name).then(|| format!("interpreted tool \"{name}\" as \"{resolved_name}\""));
    let empty = Map::new();
    let props: &Map<String, Value> = spec.parameters.get("properties").and_then(Value::as_object).unwrap_or(&empty);
    let required: Vec<String> = spec
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let mut out = original;

    // Argument-key aliasing: a small model sent {"file": …} where the schema says
    // `path`, or {"cmd": …} for `command`. For each MISSING schema property, adopt
    // the first alias the model did send — unless that alias is itself a real
    // property of this tool (never steal a legitimate argument).
    let mut alias_notes: Vec<String> = Vec::new();
    for (canonical, aliases) in ARG_ALIASES {
        if !props.contains_key(*canonical) || out.contains_key(*canonical) {
            continue;
        }
        for a in aliases.iter() {
            if props.contains_key(*a) {
                continue;
            }
            if let Some(v) = out.get(*a).cloned() {
                // Order-preserving delete (Map::remove would swap entries around).
                out.retain(|k, _| k.as_str() != *a);
                out.insert((*canonical).to_string(), v);
                alias_notes.push(format!("`{a}` → `{canonical}`"));
                break;
            }
        }
    }
    // Drop keys the schema doesn't declare — noise for the tool, and some
    // dispatchers hard-reject extras. Skipped for open schemas (no declared
    // properties, or additionalProperties allowed — common for MCP tools).
    let open_schema = props.is_empty() || spec.parameters.get("additionalProperties") == Some(&Value::Bool(true));
    if !open_schema {
        out.retain(|k, _| props.contains_key(k));
    }

    for (key, sub) in props {
        if let Some(v) = out.get_mut(key) {
            let taken = std::mem::take(v);
            *v = coerce_value(taken, sub);
        }
    }

    if !alias_notes.is_empty() {
        let a = format!("mapped argument {}", alias_notes.join(", "));
        note = Some(match note {
            Some(n) => format!("{n}; {a}"),
            None => a,
        });
    }

    let missing: Vec<&str> = required
        .iter()
        .filter(|r| match out.get(r.as_str()) {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.trim().is_empty(),
            Some(_) => false,
        })
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        let shape = required
            .iter()
            .map(|r| {
                let t = props.get(r).and_then(|p| p.get("type")).and_then(Value::as_str).unwrap_or("value");
                format!("{r}: {t}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let error = format!(
            "Tool \"{resolved_name}\" is missing required argument(s): {}. It needs {{ {shape} }}. \
             Call it again with those fields filled in.",
            missing.join(", ")
        );
        return PreparedCall { name: resolved_name, arguments: Value::Object(out), note, error: Some(error) };
    }

    PreparedCall { name: resolved_name, arguments: Value::Object(out), note, error: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str, params: Value) -> ToolSpec {
        ToolSpec { name: name.to_string(), description: name.to_string(), parameters: params }
    }

    // harnessUpgrades.test.ts SPECS
    fn specs_a() -> Vec<ToolSpec> {
        vec![
            spec("read_file", json!({ "type": "object", "properties": { "path": { "type": "string" }, "offset": { "type": "integer" } }, "required": ["path"] })),
            spec("grep", json!({ "type": "object", "properties": { "pattern": { "type": "string" }, "ignoreCase": { "type": "boolean" } }, "required": ["pattern"] })),
            spec("web_research", json!({ "type": "object", "properties": { "query": { "type": "string" }, "pages": { "type": "integer" } }, "required": ["query"] })),
        ]
    }

    // smallModelOptims.test.ts SPECS
    fn specs_b() -> Vec<ToolSpec> {
        vec![
            spec("run_shell", json!({ "type": "object", "properties": { "command": { "type": "string" }, "timeout_ms": { "type": "integer" } }, "required": ["command"] })),
            spec("read_file", json!({ "type": "object", "properties": { "path": { "type": "string" }, "offset": { "type": "integer" } }, "required": ["path"] })),
            spec("apply_edit", json!({ "type": "object", "properties": { "path": { "type": "string" }, "search": { "type": "string" }, "replace": { "type": "string" } }, "required": ["path", "replace"] })),
            spec("web_search", json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })),
        ]
    }

    fn names(s: &[ToolSpec]) -> Vec<String> {
        s.iter().map(|x| x.name.clone()).collect()
    }

    #[test]
    fn keeps_exact_name() {
        assert_eq!(resolve_tool_name("grep", &["grep", "read_file"]).name.as_deref(), Some("grep"));
    }

    #[test]
    fn resolves_substring_alias() {
        assert_eq!(resolve_tool_name("read", &["read_file", "grep"]).name.as_deref(), Some("read_file"));
    }

    #[test]
    fn resolves_typo_by_edit_distance() {
        assert_eq!(resolve_tool_name("grpe", &["grep", "read_file"]).name.as_deref(), Some("grep"));
    }

    #[test]
    fn suggestions_when_unknown() {
        let r = resolve_tool_name("zzzzzz", &["grep", "read_file", "web_search"]);
        assert!(r.name.is_none());
        assert!(!r.suggestions.is_empty());
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("grep", "grep"), 0);
        assert_eq!(levenshtein("grpe", "grep"), 2);
    }

    #[test]
    fn coerces_boolean_strings() {
        let p = prepare_tool_call("grep", Some(&json!({ "pattern": "TODO", "ignoreCase": "true" })), &specs_a());
        assert!(p.error.is_none());
        assert_eq!(p.arguments["ignoreCase"], json!(true));
    }

    #[test]
    fn coerces_numeric_strings() {
        let p = prepare_tool_call("web_research", Some(&json!({ "query": "x", "pages": "5" })), &specs_a());
        assert_eq!(p.arguments["pages"], json!(5));
    }

    #[test]
    fn reports_missing_required_with_shape() {
        let p = prepare_tool_call("read_file", Some(&json!({})), &specs_a());
        let e = p.error.unwrap();
        assert!(e.to_lowercase().contains("missing required"));
        assert!(e.contains("path"));
        assert!(e.contains("{ path: string }"));
    }

    #[test]
    fn blank_required_string_is_missing() {
        let p = prepare_tool_call("read_file", Some(&json!({ "path": "   " })), &specs_a());
        assert!(p.error.unwrap().to_lowercase().contains("missing required"));
    }

    #[test]
    fn notes_corrected_name() {
        let p = prepare_tool_call("read", Some(&json!({ "path": "a.ts" })), &specs_a());
        assert_eq!(p.name, "read_file");
        assert!(p.note.unwrap().contains("interpreted"));
    }

    #[test]
    fn errors_on_unknown_tool() {
        let p = prepare_tool_call("frobnicate", Some(&json!({})), &specs_a());
        assert!(p.error.unwrap().contains("No tool named"));
    }

    #[test]
    fn maps_cross_harness_shell_aliases() {
        let n = names(&specs_b());
        for alias in ["bash", "shell", "execute_command", "terminal", "cmd"] {
            assert_eq!(resolve_tool_name(alias, &n).name.as_deref(), Some("run_shell"), "{alias}");
        }
    }

    #[test]
    fn maps_unix_file_aliases() {
        let n = names(&specs_b());
        assert_eq!(resolve_tool_name("cat", &n).name.as_deref(), Some("read_file"));
        assert_eq!(resolve_tool_name("str_replace_editor", &n).name.as_deref(), Some("apply_edit"));
        assert_eq!(resolve_tool_name("google", &n).name.as_deref(), Some("web_search"));
    }

    #[test]
    fn strips_namespace_prefixes() {
        let n = names(&specs_b());
        assert_eq!(resolve_tool_name("functions.run_shell", &n).name.as_deref(), Some("run_shell"));
        assert_eq!(resolve_tool_name("default_api.read_file", &n).name.as_deref(), Some("read_file"));
        assert_eq!(resolve_tool_name("tools.bash", &n).name.as_deref(), Some("run_shell"));
    }

    #[test]
    fn never_aliases_to_unadvertised_tool() {
        assert!(resolve_tool_name("bash", &["read_file", "web_search"]).name.is_none());
    }

    #[test]
    fn maps_file_to_path() {
        let p = prepare_tool_call("read_file", Some(&json!({ "file": "a.ts" })), &specs_b());
        assert!(p.error.is_none());
        assert_eq!(p.arguments["path"], json!("a.ts"));
        assert!(p.note.unwrap().contains("mapped argument"));
    }

    #[test]
    fn maps_cmd_to_command() {
        let p = prepare_tool_call("run_shell", Some(&json!({ "cmd": "git status" })), &specs_b());
        assert!(p.error.is_none());
        assert_eq!(p.arguments["command"], json!("git status"));
    }

    #[test]
    fn maps_old_new_str_onto_search_replace() {
        let p = prepare_tool_call(
            "str_replace_editor",
            Some(&json!({ "path": "a.ts", "old_str": "x", "new_str": "y" })),
            &specs_b(),
        );
        assert_eq!(p.name, "apply_edit");
        assert_eq!(p.arguments["search"], json!("x"));
        assert_eq!(p.arguments["replace"], json!("y"));
    }

    #[test]
    fn never_steals_a_real_property() {
        let p = prepare_tool_call("apply_edit", Some(&json!({ "path": "a.ts", "search": "x", "replace": "y" })), &specs_b());
        assert_eq!(p.arguments["search"], json!("x"));
    }

    #[test]
    fn drops_undeclared_keys() {
        let p = prepare_tool_call("read_file", Some(&json!({ "path": "a.ts", "recursive": true })), &specs_b());
        assert!(p.arguments.get("recursive").is_none());
    }

    #[test]
    fn alias_fills_missing_required() {
        let p = prepare_tool_call("run_shell", Some(&json!({ "script": "ls" })), &specs_b());
        assert!(p.error.is_none());
        assert_eq!(p.arguments["command"], json!("ls"));
    }

    #[test]
    fn think_aliases_only_when_advertised() {
        let mut s = specs_b();
        s.push(crate::think_tool::think_tool_spec());
        let n = names(&s);
        for alias in ["thought", "reflect", "reason", "internal_monologue"] {
            assert_eq!(resolve_tool_name(alias, &n).name.as_deref(), Some("think"), "{alias}");
        }
        assert!(resolve_tool_name("internal_monologue", &names(&specs_b())).name.is_none());
        let p = prepare_tool_call("think", Some(&json!({ "reasoning": "check the exit code first" })), &s);
        assert!(p.error.is_none());
        assert_eq!(p.arguments["thought"], json!("check the exit code first"));
    }

    #[test]
    fn coerces_arrays_and_objects() {
        let s = vec![spec(
            "t",
            json!({ "type": "object", "properties": {
                "tags": { "type": "array", "items": { "type": "string" } },
                "opts": { "type": "object" },
                "n": { "type": "number" }
            } }),
        )];
        let p = prepare_tool_call("t", Some(&json!({ "tags": "a, b", "opts": "{\"x\":1}", "n": "2.5" })), &s);
        assert_eq!(p.arguments["tags"], json!(["a", "b"]));
        assert_eq!(p.arguments["opts"], json!({ "x": 1 }));
        assert_eq!(p.arguments["n"], json!(2.5));
        let p = prepare_tool_call("t", Some(&json!({ "tags": 7 })), &s);
        assert_eq!(p.arguments["tags"], json!([7]));
    }
}
