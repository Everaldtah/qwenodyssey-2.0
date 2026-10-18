//! Evolution engine: the agent gets smarter by learning from its mistakes.
//!
//! After each turn the chat layer collects failure signals (tools that errored,
//! retries, the step-limit being hit). When a turn went badly, the engine asks
//! the model to reflect and distill ONE durable, reusable lesson, then stores it
//! in the knowledge vault tagged "lesson". Those lessons are auto-recalled in
//! future turns (semantic retrieval), so the same mistake isn't repeated —
//! learning accumulates across sessions without retraining weights.

use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use qo_core::{GenerateResult, Message, ModelOptions, SharedProvider};
use regex::Regex;
use serde_json::Value;

use crate::knowledge::{KnowledgeBase, Note, NoteInput};
use crate::util::{js_trim, take_chars, truthy, value_to_string};

/// What happened in the turn that just finished.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnSignals {
    pub user_message: String,
    /// One line per failed tool call / retry.
    pub failures: Vec<String>,
    pub final_answer: String,
    pub step_limit_hit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvolutionOptions {
    pub enabled: bool,
    /// Only reflect when the turn had trouble (`evolution.reflect_on_failure`).
    pub reflect_on_failure: bool,
}

impl Default for EvolutionOptions {
    fn default() -> Self {
        Self { enabled: true, reflect_on_failure: true }
    }
}

/// The model call used for reflection. A closure (rather than a fixed
/// provider) so the chat layer can route it to whatever backend is active now.
pub type GenerateFn = Arc<dyn Fn(Vec<Message>, ModelOptions) -> BoxFuture<'static, anyhow::Result<GenerateResult>> + Send + Sync>;

pub struct EvolutionEngine {
    kb: Arc<KnowledgeBase>,
    generate: GenerateFn,
    opts: EvolutionOptions,
}

const REFLECT_SYSTEM: &str = "You are Qwenodyssey reflecting on the turn you just finished, in order to get smarter. \
From the problems and the final outcome, extract ONE concise, GENERAL, reusable lesson that \
would prevent repeating the mistake next time — name the correct tool/command/approach. \
Reply with ONLY JSON: {\"title\":\"short title\",\"lesson\":\"1-2 sentence rule\"} \
or {\"skip\":true} if nothing durable is worth saving.";

impl EvolutionEngine {
    pub fn new(kb: Arc<KnowledgeBase>, generate: GenerateFn, opts: EvolutionOptions) -> Self {
        Self { kb, generate, opts }
    }

    /// Reflect with a fixed provider (its active model can still be switched).
    pub fn with_provider(kb: Arc<KnowledgeBase>, provider: SharedProvider, opts: EvolutionOptions) -> Self {
        let generate: GenerateFn = Arc::new(move |messages, options| {
            let p = provider.clone();
            Box::pin(async move { p.generate(&messages, &options).await })
        });
        Self::new(kb, generate, opts)
    }

    /// Saved lessons (tag "lesson"), newest first.
    pub fn lessons(&self) -> Vec<Note> {
        self.kb.list().into_iter().filter(|n| n.meta.tags.iter().any(|t| t == "lesson")).collect()
    }

    /// Reflect on a finished turn and, if there's something to learn, save a
    /// lesson. Returns the lesson title when one was saved. Cheap: one short
    /// JSON-only model call, and only when the turn actually had trouble (so
    /// smooth turns cost nothing). Never fails — reflection is best-effort.
    pub async fn reflect(&self, turn: &TurnSignals) -> Option<String> {
        if !self.opts.enabled {
            return None;
        }
        let troubled = !turn.failures.is_empty() || turn.step_limit_hit;
        if self.opts.reflect_on_failure && !troubled {
            return None;
        }

        let problems = turn.failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n");
        let problems = take_chars(&problems, 1500);
        let user = format!(
            "User asked: {}\n\nProblems this turn:\n{}\n\nFinal answer: {}",
            take_chars(&turn.user_message, 300),
            if problems.is_empty() { "(hit the tool-step limit without resolving)" } else { problems },
            take_chars(&turn.final_answer, 400)
        );
        let opts = ModelOptions { temperature: Some(0.2), max_tokens: Some(250), json: true, ..Default::default() };
        let res = (self.generate)(vec![Message::system(REFLECT_SYSTEM), Message::user(user)], opts).await.ok()?;

        let parsed = safe_json(&res.text)?;
        if truthy(parsed.get("skip")) || !truthy(parsed.get("lesson")) {
            return None;
        }
        let raw_title = parsed.get("title").filter(|t| truthy(Some(t))).map(value_to_string).unwrap_or_else(|| "lesson".into());
        let title = js_trim(take_chars(&raw_title, 80)).to_string();
        let lesson = js_trim(&parsed.get("lesson").map(value_to_string).unwrap_or_default()).to_string();
        self.kb
            .write(NoteInput {
                title: format!("Lesson: {title}"),
                content: lesson,
                tags: vec!["lesson".into(), "evolution".into()],
                ..Default::default()
            })
            .await
            .ok()?;
        Some(title)
    }
}

static JSON_OBJ: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\{.*\}").expect("static regex"));

/// Extract the first JSON object from a model reply, tolerating stray text.
fn safe_json(text: &str) -> Option<Value> {
    let m = JSON_OBJ.find(text)?;
    serde_json::from_str(m.as_str()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::KnowledgeOptions;
    use crate::testutil::{serve, Reply};
    use std::sync::Mutex;

    fn canned(reply: &'static str, seen: Arc<Mutex<Vec<Vec<Message>>>>) -> GenerateFn {
        Arc::new(move |messages, opts| {
            assert!(opts.json && opts.max_tokens == Some(250));
            seen.lock().unwrap().push(messages);
            Box::pin(async move { Ok(GenerateResult { text: reply.into(), model: "stub".into(), ..Default::default() }) })
        })
    }

    #[tokio::test]
    async fn saves_a_lesson_only_for_troubled_turns() {
        let dir = tempfile::tempdir().unwrap();
        let ollama = serve(Arc::new(|_| Reply::json(404, serde_json::json!({})))).await;
        let kb = Arc::new(KnowledgeBase::new(
            &dir.path().to_string_lossy(),
            KnowledgeOptions { base_url: ollama.base.clone(), ..Default::default() },
        ));
        kb.ensure();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let engine = EvolutionEngine::new(
            kb.clone(),
            canned("Sure! {\"title\":\"Use Get-WinEvent\",\"lesson\":\"Query logs with Get-WinEvent.\"} done", seen.clone()),
            EvolutionOptions::default(),
        );

        // Smooth turn: no model call at all.
        let smooth = TurnSignals { user_message: "hi".into(), final_answer: "hello".into(), ..Default::default() };
        assert_eq!(engine.reflect(&smooth).await, None);
        assert!(seen.lock().unwrap().is_empty());

        let bad = TurnSignals {
            user_message: "show login logs".into(),
            failures: vec!["run_shell: eventquery not found".into()],
            final_answer: "could not".into(),
            step_limit_hit: false,
        };
        assert_eq!(engine.reflect(&bad).await.as_deref(), Some("Use Get-WinEvent"));
        // Clone: a borrowed `&seen.lock()…[0]` would keep the guard alive and
        // deadlock the next reflect() call.
        let prompt = seen.lock().unwrap()[0].clone();
        assert!(prompt[1].content.contains("Problems this turn:\nrun_shell: eventquery not found"));
        let lessons = engine.lessons();
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].meta.title, "Lesson: Use Get-WinEvent");
        assert_eq!(lessons[0].body, "Query logs with Get-WinEvent.");

        // Step limit with no failures uses the placeholder line; {"skip":true} saves nothing.
        let engine = EvolutionEngine::new(kb, canned("{\"skip\":true}", seen.clone()), EvolutionOptions::default());
        let limit = TurnSignals { step_limit_hit: true, ..Default::default() };
        assert_eq!(engine.reflect(&limit).await, None);
        let second = seen.lock().unwrap()[1][1].content.clone();
        assert!(second.contains("(hit the tool-step limit without resolving)"));
    }

    #[test]
    fn safe_json_tolerates_stray_text() {
        assert_eq!(safe_json("x {\"a\":1} y").unwrap()["a"], 1);
        assert!(safe_json("no json").is_none());
        assert!(safe_json("{broken").is_none());
    }
}
