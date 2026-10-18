//! Error fixer (port of `src/agents/errorFixer.ts`): given failing test
//! output, propose JSON file edits that fix the cause.

use tokio_util::sync::CancellationToken;

use super::{ask, display_value, normalize_edits};
use crate::deps::{extract_json, load_prompt, render};
use qo_core::{CoderResult, Provider};

pub async fn fix_errors(
    provider: &dyn Provider,
    task: &str,
    errors: &str,
    context: &str,
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<CoderResult> {
    let prompt = render(&load_prompt("error_fix")?, &[("task", task), ("errors", errors), ("context", context)]);
    let text = ask(provider, prompt, true, 0.2, None, cancel).await?;
    let parsed = extract_json(&text);
    Ok(CoderResult {
        edits: normalize_edits(parsed.as_ref()),
        notes: parsed.as_ref().and_then(|p| p.get("notes")).and_then(display_value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeProvider;

    #[tokio::test]
    async fn returns_no_edits_for_unparseable_output() {
        let fp = FakeProvider::new(vec!["sorry, no idea".into()]);
        let r = fix_errors(&fp, "t", "boom", "ctx", None).await.unwrap();
        assert!(r.edits.is_empty());
        assert!(fp.requests()[0].messages[1].content.contains("boom"));
    }
}
