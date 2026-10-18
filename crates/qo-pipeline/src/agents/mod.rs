//! The pipeline's model-driven agents (ports of `src/agents/*.ts`):
//! planner, legacy JSON coder, SEARCH/REPLACE edit coder, reviewer, tester
//! and error fixer. Each is a thin prompt + parse layer over [`Provider`];
//! every request carries the turn's Esc [`CancellationToken`].

pub mod coder;
pub mod edit_coder;
pub mod error_fixer;
pub mod planner;
pub mod reviewer;
pub mod tester;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::deps::load_prompt;
use qo_core::{EditMode, FileEdit, Message, ModelOptions, Provider};

/// One system + user request (the shape every agent sends), returning the text.
pub(crate) async fn ask(
    provider: &dyn Provider,
    user_prompt: String,
    json: bool,
    temperature: f64,
    max_tokens: Option<u32>,
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<String> {
    let messages = [Message::system(load_prompt("system")?), Message::user(user_prompt)];
    let opts = ModelOptions {
        json,
        temperature: Some(temperature),
        max_tokens,
        cancel: cancel.cloned(),
        ..Default::default()
    };
    Ok(provider.generate(&messages, &opts).await?.text)
}

/// A JSON string field, if present and a string.
pub(crate) fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

/// JS `String(x)`-ish rendering of a non-null JSON value for display fields.
pub(crate) fn display_value(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// Strings of a JSON array field (non-strings dropped); empty if absent.
pub(crate) fn string_list(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// The coder / error-fixer `normalizeEdits`: keep entries with a string path,
/// coerce unknown modes to "rewrite", keep only string diff/content/rationale.
pub(crate) fn normalize_edits(raw: Option<&Value>) -> Vec<FileEdit> {
    let Some(edits) = raw.and_then(|r| r.get("edits")).and_then(Value::as_array) else {
        return Vec::new();
    };
    edits
        .iter()
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?.to_string();
            let mode = match e.get("mode").and_then(Value::as_str) {
                Some("patch") => EditMode::Patch,
                Some("create") => EditMode::Create,
                Some("delete") => EditMode::Delete,
                _ => EditMode::Rewrite,
            };
            Some(FileEdit {
                path,
                mode,
                diff: str_field(e, "diff"),
                content: str_field(e, "content"),
                rationale: str_field(e, "rationale"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_edits_like_the_ts() {
        let raw = json!({"edits": [
            {"path": "a.rs", "mode": "patch", "diff": "@@", "rationale": "r"},
            {"path": "b.rs", "mode": "weird", "content": "x", "diff": 3},
            {"mode": "create"},
            null,
            {"path": 5}
        ]});
        let e = normalize_edits(Some(&raw));
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].mode, EditMode::Patch);
        assert_eq!(e[0].diff.as_deref(), Some("@@"));
        assert_eq!(e[1].mode, EditMode::Rewrite);
        assert_eq!(e[1].diff, None);
        assert!(normalize_edits(Some(&json!({"edits": "nope"}))).is_empty());
        assert!(normalize_edits(None).is_empty());
    }
}
