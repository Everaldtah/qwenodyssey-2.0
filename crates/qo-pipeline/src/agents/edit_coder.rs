//! Edit-block coder — the upgraded code-generation agent for small models
//! (port of `src/agents/editCoder.ts`).
//!
//! Unlike the legacy JSON-diff coder ([`super::coder`]), this agent:
//!   - asks the model for SEARCH/REPLACE blocks (which small models get right
//!     far more often than unified diffs or whole-file JSON),
//!   - drives the reliable edit pipeline (best-of-N + apply + verify + repair),
//!   - injects compiler/test feedback verbatim on repair rounds.
//!
//! The orchestrator selects it when `agent.edit_protocol = "blocks"` (the
//! default for small models).

use std::path::PathBuf;

use tokio_util::sync::CancellationToken;

use crate::deps::{load_prompt, render};
use crate::edit_pipeline::{run_edit_pipeline, EditPipelineOptions, EditPipelineResult, EventSink, Resolver};
use crate::verify_loop::CheckKind;
use qo_core::{Message, ModelOptions, Provider};

/// Repair-round feedback preamble (sent as a user turn, verbatim from the TS).
pub const REPAIR_PREAMBLE: &str = "[system] Your previous edits were applied but the checks FAILED with the output below. \
Fix the cause with new SEARCH/REPLACE blocks. The SEARCH text must match the CURRENT file \
contents exactly (your earlier edits are already applied).\n\n";

/// Diversity nudge for additional best-of-N samples (no feedback yet).
pub const ALTERNATIVE_NUDGE: &str = "[system] Provide an alternative implementation approach for the same task.";

#[derive(Clone)]
pub struct EditCoderOptions<'a> {
    pub provider: &'a dyn Provider,
    pub task: &'a str,
    pub context: &'a str,
    pub root: PathBuf,
    pub resolve: Resolver,
    /// best-of-N candidate count (config `agent.candidates`).
    pub candidates: usize,
    /// Self-repair rounds.
    pub max_repair_rounds: usize,
    pub run_verify: bool,
    pub verify_only: Option<Vec<CheckKind>>,
    /// Temperature for the first sample; later samples nudge up for diversity.
    pub base_temperature: Option<f64>,
    pub cancel: Option<CancellationToken>,
    pub on_event: Option<EventSink>,
}

/// Build the conversation for one generation. On repair rounds, `feedback`
/// (compiler/test digest) is appended so the model fixes the real failure.
pub fn build_messages(task: &str, context: &str, attempt: usize, feedback: Option<&str>) -> anyhow::Result<Vec<Message>> {
    let user_prompt = render(&load_prompt("coder_blocks")?, &[("task", task), ("context", context)]);
    let mut messages = vec![Message::system(load_prompt("system")?), Message::user(user_prompt)];
    if let Some(fb) = feedback {
        messages.push(Message::user(format!("{REPAIR_PREAMBLE}{fb}")));
    } else if attempt > 0 {
        messages.push(Message::user(ALTERNATIVE_NUDGE));
    }
    Ok(messages)
}

/// Sampling temperature: later best-of-N samples run slightly hotter for
/// diversity; repair rounds (feedback present) stay cold for precision.
pub fn sample_temperature(base: f64, attempt: usize, has_feedback: bool) -> f64 {
    if has_feedback {
        0.1
    } else {
        (base + attempt as f64 * 0.15).min(0.7)
    }
}

pub async fn code_with_blocks(opts: EditCoderOptions<'_>) -> anyhow::Result<EditPipelineResult> {
    let base = opts.base_temperature.unwrap_or(0.2);
    let provider = opts.provider;
    let (task, context) = (opts.task, opts.context);
    let cancel = opts.cancel.clone();

    let generate = move |attempt: usize, feedback: Option<String>| {
        let cancel = cancel.clone();
        async move {
            let messages = build_messages(task, context, attempt, feedback.as_deref())?;
            let mopts = ModelOptions {
                temperature: Some(sample_temperature(base, attempt, feedback.is_some())),
                cancel,
                ..Default::default()
            };
            Ok(provider.generate(&messages, &mopts).await?.text)
        }
    };

    let popts = EditPipelineOptions {
        root: opts.root,
        resolve: opts.resolve,
        candidates: opts.candidates,
        max_repair_rounds: opts.max_repair_rounds,
        run_verify: opts.run_verify,
        verify_only: opts.verify_only,
        cancel: opts.cancel,
        on_event: opts.on_event,
    };
    run_edit_pipeline(generate, &popts).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeProvider;
    use std::sync::Arc;

    #[test]
    fn temperatures() {
        assert_eq!(sample_temperature(0.2, 0, false), 0.2);
        assert!((sample_temperature(0.2, 1, false) - 0.35).abs() < 1e-9);
        assert_eq!(sample_temperature(0.2, 9, false), 0.7);
        assert_eq!(sample_temperature(0.6, 0, true), 0.1);
    }

    #[test]
    fn messages_for_samples_and_repairs() {
        let m = build_messages("t", "c", 0, None).unwrap();
        assert_eq!(m.len(), 2);
        let m = build_messages("t", "c", 1, None).unwrap();
        assert_eq!(m[2].content, ALTERNATIVE_NUDGE);
        let m = build_messages("t", "c", 1, Some("### test FAILED")).unwrap();
        assert!(m[2].content.starts_with("[system] Your previous edits were applied"));
        assert!(m[2].content.ends_with("exactly (your earlier edits are already applied).\n\n### test FAILED"));
    }

    #[tokio::test]
    async fn drives_the_pipeline_with_the_provider() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        let fp = FakeProvider::new(vec![
            "nothing useful".into(),
            "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE".into(),
        ]);
        let root = dir.path().to_path_buf();
        let r2 = root.clone();
        let res = code_with_blocks(EditCoderOptions {
            provider: &fp,
            task: "set x to 2",
            context: "ctx",
            root,
            resolve: Arc::new(move |rel: &str| Ok(r2.join(rel))),
            candidates: 2,
            max_repair_rounds: 0,
            run_verify: false,
            verify_only: None,
            base_temperature: Some(0.3),
            cancel: None,
            on_event: None,
        })
        .await
        .unwrap();
        assert!(res.applied, "{}", res.message);
        assert_eq!(std::fs::read_to_string(dir.path().join("a.py")).unwrap(), "x = 2\n");
        let reqs = fp.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].temperature, Some(0.3));
        assert!((reqs[1].temperature.unwrap() - 0.45).abs() < 1e-9);
        assert!(!reqs[0].json);
        assert_eq!(reqs[1].messages.last().unwrap().content, ALTERNATIVE_NUDGE);
    }
}
