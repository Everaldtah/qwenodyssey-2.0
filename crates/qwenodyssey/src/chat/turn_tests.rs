//! Chat-loop tests with a scripted provider (headless; no network, no models).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use crossterm::event::KeyCode;
use qo_core::config::Config;
use qo_core::logger::Logger;
use qo_core::{AgentMode, GenerateResult, HealthStatus, Message, ModelOptions, Provider, ProviderError, Role, SharedProvider, ToolCall, ToolContext, ToolRegistry};
use tokio::sync::mpsc::UnboundedSender;

use super::*;
use crate::session::Session;
use crate::ui::input::InputEvent;
use crate::ui::{Input, Ui};

/// What the scripted provider does for one request.
enum Step {
    Text(&'static str),
    Calls(Vec<ToolCall>),
    /// Accept the request and never answer (until cancelled).
    Hang,
    Fail(ProviderError),
}

struct Scripted {
    name: String,
    model: Mutex<String>,
    steps: Mutex<VecDeque<Step>>,
    seen: Mutex<Vec<Vec<Message>>>,
}

impl Scripted {
    fn new(name: &str, steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self { name: name.into(), model: Mutex::new("test-model".into()), steps: Mutex::new(steps.into()), seen: Mutex::new(Vec::new()) })
    }

    async fn next(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        self.seen.lock().unwrap().push(messages.to_vec());
        let step = self.steps.lock().unwrap().pop_front().unwrap_or(Step::Text("(script exhausted)"));
        let model = self.model();
        match step {
            Step::Text(t) => Ok(GenerateResult { text: t.into(), model, ..Default::default() }),
            Step::Calls(c) => Ok(GenerateResult { tool_calls: c, model, ..Default::default() }),
            Step::Fail(e) => Err(e.into()),
            Step::Hang => {
                match &opts.cancel {
                    Some(c) => c.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
                Err(ProviderError::Interrupted.into())
            }
        }
    }
}

#[async_trait]
impl Provider for Scripted {
    fn name(&self) -> &str {
        &self.name
    }
    fn model(&self) -> String {
        self.model.lock().unwrap().clone()
    }
    fn set_model(&self, model: &str) {
        *self.model.lock().unwrap() = model.into();
    }
    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        self.next(messages, opts).await
    }
    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let r = self.next(messages, opts).await?;
        on_chunk(&r.text);
        Ok(r)
    }
    async fn health_check(&self) -> HealthStatus {
        HealthStatus { ok: true, detail: None }
    }
}

struct Harness {
    chat: Chat,
    tx: UnboundedSender<InputEvent>,
    _dir: tempfile::TempDir,
}

fn harness(provider: SharedProvider, tty: bool, mutate: impl FnOnce(&mut Config)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    config.model.fallback_models.clear();
    config.lmstudio.enabled = false;
    config.knowledge.enabled = false;
    config.web.enabled = false;
    mutate(&mut config);
    let (input, tx) = Input::from_channel(tty);
    let ui = Ui::new(input);
    let s = Session {
        cwd: dir.path().to_path_buf(),
        config,
        logger: Logger::new(dir.path(), false),
        provider: provider.clone(),
        auto_confirm: false,
        mode: AgentMode::Safe,
        self_root: dir.path().to_path_buf(),
        lms: Arc::new(qo_providers::LmStudioControl::new(None)),
        lms_model_keys: Vec::new(),
        lms_models: Some(Vec::new()),
        model_pinned: false,
    };
    let mut tools = ToolRegistry::new(ToolContext::new(dir.path()));
    qo_tools::register_defaults(&mut tools);
    let plan_state = qo_tools::PlanState::new();
    tools.register(Arc::new(qo_tools::create_plan_tool(plan_state.clone())));
    let kb = Arc::new(qo_net::KnowledgeBase::new(&dir.path().join("vault").display().to_string(), Default::default()));
    let chat = Chat {
        s,
        ui,
        history: vec![Message::system("SYSTEM")],
        tools,
        tool_specs: qo_tools::chat_tool_specs(),
        kb,
        memory_enabled: false,
        evolution: None,
        provider_cell: Arc::new(Mutex::new(provider)),
        meter: TokenMeter::default(),
        seen_files: Mutex::new(qo_tools::BlindOverwriteGuard::new()),
        plan_state,
        last_models: Vec::new(),
        lms_job: None,
        startup_job: None,
        startup_resolved: true,
        anthropic_auth: Arc::new(std::sync::OnceLock::new()),
    };
    Harness { chat, tx, _dir: dir }
}

#[tokio::test]
async fn esc_interrupts_a_model_that_never_answers() {
    let p = Scripted::new("nvidia", vec![Step::Hang]);
    let mut h = harness(p.clone(), true, |_| {});
    h.chat.history.push(Message::user("hi"));
    let tx = h.tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        tx.send(InputEvent::key(KeyCode::Esc)).unwrap();
    });
    let specs = h.chat.tool_specs.clone();
    let sig = tokio::time::timeout(Duration::from_secs(5), h.chat.run_assistant_turn(&specs))
        .await
        .expect("Esc must end the turn promptly")
        .expect("an interrupt is not an error");
    assert!(sig.final_answer.is_empty() && !sig.step_limit_hit);
    // No assistant message was recorded, the prompt owns the input again, and
    // the turn's cancel token was removed from the tool context.
    assert_eq!(h.chat.history.last().map(|m| m.role), Some(Role::User));
    assert!(h.chat.tools.context().cancel.is_none());
    h.tx.send(InputEvent::Line("after".into())).unwrap();
    let mut lease = h.chat.ui.input().lease().await;
    assert_eq!(lease.next().await, InputEvent::Line("after".into()));
}

#[tokio::test]
async fn text_emitted_tool_call_is_executed_then_answered() {
    let p = Scripted::new(
        "ollama",
        vec![Step::Text("```json\n{\"name\": \"update_plan\", \"arguments\": {\"plan\": [{\"step\": \"look\", \"status\": \"in_progress\"}]}}\n```"), Step::Text("All planned.")],
    );
    let mut h = harness(p.clone(), false, |_| {});
    h.chat.history.push(Message::user("plan it"));
    let specs = h.chat.tool_specs.clone();
    let sig = h.chat.run_assistant_turn(&specs).await.unwrap();
    assert_eq!(sig.final_answer, "All planned.");
    assert!(h.chat.plan_state.render().contains("look"));
    let roles: Vec<Role> = h.chat.history.iter().map(|m| m.role).collect();
    assert_eq!(roles, vec![Role::System, Role::User, Role::Assistant, Role::Tool, Role::Assistant]);
    // The raw JSON was not fed back as assistant content.
    assert_eq!(h.chat.history[2].content, "");
}

#[tokio::test]
async fn fenced_command_gets_one_nudge_before_anything_runs() {
    let p = Scripted::new("ollama", vec![Step::Text("Run this:\n```powershell\nGet-Date\n```"), Step::Text("It is noon.")]);
    let mut h = harness(p.clone(), false, |_| {});
    h.chat.history.push(Message::user("what time is it"));
    let specs = h.chat.tool_specs.clone();
    let sig = h.chat.run_assistant_turn(&specs).await.unwrap();
    assert_eq!(sig.final_answer, "It is noon.");
    assert!(sig.failures.iter().any(|f| f.contains("``` fence")));
    assert!(h.chat.history.iter().any(|m| m.role == Role::User && m.content.contains("fenced code is NOT executed")));
    assert!(!h.chat.history.iter().any(|m| m.role == Role::Tool), "nothing ran on the first regression");
}

#[tokio::test]
async fn repeated_identical_call_trips_the_loop_guard_then_tools_go_off() {
    let call = || vec![ToolCall { id: "c".into(), name: "update_plan".into(), arguments: serde_json::json!({"plan": [{"step": "x"}]}) }];
    let p = Scripted::new("ollama", vec![Step::Calls(call()), Step::Calls(call()), Step::Calls(call()), Step::Text("done")]);
    let mut h = harness(p.clone(), false, |_| {});
    h.chat.history.push(Message::user("go"));
    let specs = h.chat.tool_specs.clone();
    let sig = h.chat.run_assistant_turn(&specs).await.unwrap();
    assert_eq!(sig.final_answer, "done");
    assert_eq!(sig.failures.iter().filter(|f| f.contains("(loop)")).count(), 2);
    assert!(h.chat.history.iter().any(|m| m.content.contains("Tools are now switched off")));
}

#[tokio::test]
async fn non_availability_stream_error_falls_back_to_blocking_once() {
    // A 400 is not "unavailable": the turn retries on the blocking path with
    // the same model instead of walking the chain.
    let p = Scripted::new("nvidia", vec![Step::Fail(ProviderError::http("nvidia", 400, "bad request shape")), Step::Text("recovered")]);
    let mut h = harness(p.clone(), false, |c| c.model.fallback_models = vec!["nvidia:other/model".into()]);
    h.chat.history.push(Message::user("hello there friend, explain x"));
    let sig = h.chat.run_assistant_turn(&[]).await.unwrap();
    assert_eq!(sig.final_answer, "recovered");
    assert_eq!(h.chat.s.provider.model(), "test-model", "no fallback switch happened");
}

#[tokio::test]
async fn interrupt_error_never_falls_back() {
    let p = Scripted::new("nvidia", vec![Step::Fail(ProviderError::Interrupted)]);
    let mut h = harness(p.clone(), false, |c| c.model.fallback_models = vec!["nvidia:other/model".into()]);
    h.chat.history.push(Message::user("x"));
    let sig = h.chat.run_assistant_turn(&[]).await.unwrap();
    assert!(sig.final_answer.is_empty());
    assert_eq!(h.chat.s.provider.model(), "test-model");
    assert_eq!(p.seen.lock().unwrap().len(), 1, "exactly one request");
}

#[tokio::test]
async fn fallback_chain_filters_active_dupes_and_keyless_clouds() {
    let p = Scripted::new("nvidia", vec![]);
    let h = harness(p.clone(), false, |c| {
        c.model.fallback_models = vec![
            "nvidia:test-model".into(), // the active model → dropped
            "qwen2.5:7b".into(),
            "qwen2.5:7b".into(), // duplicate → dropped
            "".into(),
            "anthropic:claude-x".into(), // auth unknown → dropped
        ];
        c.anthropic.include_as_fallback = true;
    });
    assert_eq!(h.chat.fallback_chain(), vec!["qwen2.5:7b".to_string()]);
    assert_eq!(h.chat.active_ref(), "nvidia:test-model");
}

#[tokio::test]
async fn switching_identity_updates_the_system_prompt() {
    let p = Scripted::new("nvidia", vec![]);
    let mut h = harness(p.clone(), false, |_| {});
    let sa = h.chat.self_awareness_text();
    h.chat.history[0].content.push('\n');
    h.chat.history[0].content.push_str(&sa);
    p.set_model("nvidia/nemotron-3-ultra-550b-a55b");
    h.chat.refresh_identity();
    assert!(h.chat.history[0].content.starts_with("SYSTEM\n"));
    assert!(h.chat.history[0].content.contains("\"nvidia/nemotron-3-ultra-550b-a55b\""));
    assert!(!h.chat.history[0].content.contains("\"test-model\""));
}

#[tokio::test]
async fn smalltalk_turn_sends_no_tools() {
    let p = Scripted::new("nvidia", vec![Step::Text("Hey!")]);
    let mut h = harness(p.clone(), false, |_| {});
    h.chat.history.push(Message::user("hi"));
    let sig = h.chat.run_assistant_turn(&[]).await.unwrap();
    assert_eq!(sig.final_answer, "Hey!");
    assert_eq!(h.chat.history.last().unwrap().content, "Hey!");
}
