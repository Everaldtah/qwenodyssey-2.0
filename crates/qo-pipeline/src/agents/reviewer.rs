//! Reviewer agent (port of `src/agents/reviewer.ts`): check proposed edits
//! before they are applied. An unparseable review never blocks.

use std::path::Path;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{ask, display_value, str_field};
use crate::deps::{extract_json, load_prompt, render};
use crate::patch_engine::{mode_name, preview_edit_or_note};
use qo_core::{FileEdit, Provider, ReviewIssue, ReviewResult, Severity};

/// Turn the reviewer's JSON into a [`ReviewResult`] (approve unless
/// `approve: false`; unknown severities become info).
pub fn review_from_json(parsed: &Value) -> ReviewResult {
    let issues = parsed
        .get("issues")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|i| ReviewIssue {
                    severity: match i.get("severity").and_then(Value::as_str) {
                        Some("error") => Severity::Error,
                        Some("warning") => Severity::Warning,
                        _ => Severity::Info,
                    },
                    file: str_field(i, "file"),
                    message: i.get("message").and_then(display_value).unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default();
    ReviewResult {
        approve: parsed.get("approve") != Some(&Value::Bool(false)),
        summary: str_field(parsed, "summary"),
        issues,
    }
}

/// The review used when the model's output is unstructured.
pub fn unparseable_review() -> ReviewResult {
    ReviewResult {
        approve: true,
        issues: vec![ReviewIssue {
            severity: Severity::Warning,
            file: None,
            message: "reviewer output was unstructured".into(),
        }],
        summary: Some("review skipped (unparseable)".into()),
    }
}

pub async fn review(
    provider: &dyn Provider,
    cwd: &Path,
    task: &str,
    edits: &[FileEdit],
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<ReviewResult> {
    let diffs = edits
        .iter()
        .map(|e| format!("### {} ({})\n```diff\n{}\n```", e.path, mode_name(e.mode), preview_edit_or_note(cwd, e)))
        .collect::<Vec<_>>()
        .join("\n\n");
    let prompt = render(&load_prompt("reviewer")?, &[("task", task), ("edits", &diffs)]);
    let text = ask(provider, prompt, true, 0.1, None, cancel).await?;
    // If the reviewer is unparseable, do not block — flag it as a warning.
    Ok(extract_json(&text).map(|p| review_from_json(&p)).unwrap_or_else(unparseable_review))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeProvider;
    use qo_core::EditMode;
    use serde_json::json;

    #[test]
    fn parses_review_json() {
        let r = review_from_json(&json!({"approve": false, "summary": "s", "issues": [
            {"severity": "error", "file": "a", "message": "m"},
            {"severity": "loud", "message": 3}
        ]}));
        assert!(!r.approve);
        assert_eq!(r.issues[0].severity, Severity::Error);
        assert_eq!(r.issues[1].severity, Severity::Info);
        assert_eq!(r.issues[1].message, "3");
        assert!(review_from_json(&json!({"approve": "no"})).approve, "only literal false blocks");
    }

    #[tokio::test]
    async fn unparseable_review_does_not_block() {
        let dir = tempfile::tempdir().unwrap();
        let fp = FakeProvider::new(vec!["looks fine to me".into()]);
        let edits =
            [FileEdit { path: "n.txt".into(), mode: EditMode::Create, diff: None, content: Some("x\n".into()), rationale: None }];
        let r = review(&fp, dir.path(), "t", &edits, None).await.unwrap();
        assert!(r.approve);
        assert_eq!(r.summary.as_deref(), Some("review skipped (unparseable)"));
        let prompt = &fp.requests()[0].messages[1].content;
        assert!(prompt.contains("### n.txt (create)\n```diff\n"));
        assert!(prompt.contains("+x"));
    }
}
