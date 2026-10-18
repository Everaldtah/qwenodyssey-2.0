//! Legacy JSON coder (port of `src/agents/coder.ts`): asks for whole-file /
//! unified-diff edits as JSON. Used when `agent.edit_protocol = "json"` and by
//! the `edit` command; the default path is [`super::edit_coder`].

use tokio_util::sync::CancellationToken;

use super::{ask, display_value, normalize_edits};
use crate::deps::{extract_json, load_prompt, render};
use qo_core::{CoderResult, Plan, Provider};

pub async fn code(
    provider: &dyn Provider,
    task: &str,
    plan: &Plan,
    context: &str,
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<CoderResult> {
    let plan_json = serde_json::to_string_pretty(plan)?;
    let prompt = render(&load_prompt("coder")?, &[("task", task), ("plan", &plan_json), ("context", context)]);
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
    use qo_core::EditMode;

    #[tokio::test]
    async fn parses_edits_and_notes() {
        let fp = FakeProvider::new(vec![
            r#"```json
{"edits":[{"path":"a.txt","mode":"create","content":"hi"}],"notes":"n"}
```"#
                .into(),
        ]);
        let plan = Plan { goal: "g".into(), ..Default::default() };
        let r = code(&fp, "task", &plan, "ctx", None).await.unwrap();
        assert_eq!(r.edits.len(), 1);
        assert_eq!(r.edits[0].mode, EditMode::Create);
        assert_eq!(r.notes.as_deref(), Some("n"));
        let req = &fp.requests()[0];
        assert!(req.json);
        assert_eq!(req.temperature, Some(0.2));
    }
}
