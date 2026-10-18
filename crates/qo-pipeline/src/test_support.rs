//! Test doubles: a canned-response [`Provider`] and recording [`Tool`]s, so
//! agent / orchestrator tests never touch a real model or shell.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use qo_core::{
    GenerateResult, HealthStatus, Message, ModelOptions, Provider, ProviderError, Tool, ToolContext, ToolRegistry,
    ToolResult,
};

/// What one `generate` call received.
#[derive(Clone, Debug)]
pub struct Request {
    pub messages: Vec<Message>,
    pub json: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// Replies with the queued texts in order; when the queue is empty it replies
/// with `fallback` (or errors when that is `None`). A reply of
/// `"__interrupt__"` returns [`ProviderError::Interrupted`]; `"__error__"` a
/// generic error.
pub struct FakeProvider {
    replies: Mutex<VecDeque<String>>,
    fallback: Option<String>,
    requests: Mutex<Vec<Request>>,
    pub health: HealthStatus,
}

impl FakeProvider {
    pub fn new(replies: Vec<String>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            fallback: None,
            requests: Mutex::new(Vec::new()),
            health: HealthStatus { ok: true, detail: None },
        }
    }

    pub fn with_fallback(mut self, text: &str) -> Self {
        self.fallback = Some(text.to_string());
        self
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }

    fn next_reply(&self) -> anyhow::Result<String> {
        let next = self.replies.lock().ok().and_then(|mut q| q.pop_front()).or_else(|| self.fallback.clone());
        match next.as_deref() {
            Some("__interrupt__") => Err(ProviderError::Interrupted.into()),
            Some("__error__") => Err(anyhow::anyhow!("fake provider failure")),
            Some(t) => Ok(t.to_string()),
            None => Err(anyhow::anyhow!("fake provider: no reply queued")),
        }
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }
    fn model(&self) -> String {
        "fake-model".into()
    }
    fn set_model(&self, _model: &str) {}

    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        if let Ok(mut r) = self.requests.lock() {
            r.push(Request {
                messages: messages.to_vec(),
                json: opts.json,
                temperature: opts.temperature,
                max_tokens: opts.max_tokens,
            });
        }
        if opts.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(ProviderError::Interrupted.into());
        }
        let text = self.next_reply()?;
        Ok(GenerateResult { text, model: "fake-model".into(), ..Default::default() })
    }

    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let r = self.generate(messages, opts).await?;
        on_chunk(&r.text);
        Ok(r)
    }

    async fn health_check(&self) -> HealthStatus {
        self.health.clone()
    }
}

/// A tool that records its arguments and returns a fixed result.
#[derive(Clone)]
pub struct FakeTool {
    name: String,
    result: ToolResult,
    calls: Arc<Mutex<Vec<Value>>>,
}

impl FakeTool {
    pub fn new(name: &str, result: ToolResult) -> Arc<FakeTool> {
        Arc::new(FakeTool { name: name.to_string(), result, calls: Arc::new(Mutex::new(Vec::new())) })
    }

    pub fn calls(&self) -> Vec<Value> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "fake"
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        if let Ok(mut c) = self.calls.lock() {
            c.push(args);
        }
        self.result.clone()
    }
}

/// A temp dir that is its own git repo, so git probes stay scoped to this
/// small tree (the temp dir may live inside a huge parent repo, e.g. a
/// git-init'ed home directory). Works without git too (no repo then).
pub fn scratch_repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    let _ = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(dir.path())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    dir
}

/// A registry rooted at `cwd` containing `tools`.
pub fn registry_with(cwd: impl Into<PathBuf>, tools: Vec<Arc<FakeTool>>) -> ToolRegistry {
    let mut reg = ToolRegistry::new(ToolContext::new(cwd));
    for t in tools {
        reg.register(t);
    }
    reg
}
