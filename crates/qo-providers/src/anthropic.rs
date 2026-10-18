//! Anthropic (Claude) backend — the NATIVE Messages API (`/v1/messages`), not
//! an OpenAI-compatible shim (port of anthropic.ts). Claude's wire format
//! differs from OpenAI's: the system prompt is a top-level field, tools use
//! `input_schema`, and tool calls/results are `tool_use` / `tool_result`
//! content blocks. Qwenodyssey's internal OpenAI-shaped messages/tool calls
//! are translated to and from that format here.
//!
//! Auth supports both worlds:
//!  - **OAuth bearer** (a Claude Pro/Max subscription, the way Claude Code /
//!    `ant` authenticate): `Authorization: Bearer <token>` +
//!    `anthropic-beta: oauth-2025-04-20`. The token is short-lived, so a
//!    [`TokenSource`](crate::provider_config::TokenSource) (wired to
//!    `ant auth print-credentials`) re-mints it on a 401.
//!  - **API key** (console.anthropic.com, pay-as-you-go): `x-api-key: <key>`.
//!
//! We deliberately send NO sampling params (temperature/top_p) and NO
//! `thinking` config: Opus 4.7/4.8 and Fable 5 reject temperature/top_p and
//! `budget_tokens` with a 400, and omitting `thinking` is valid on every
//! current model (Fable always thinks; its thinking blocks carry empty text and
//! we read text only).

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use async_trait::async_trait;
use qo_core::tokens::estimate_tokens;
use qo_core::{
    CancellationToken, GenerateResult, HealthStatus, Message, ModelInfo, ModelOptions, Provider, ProviderError, Role,
    ToolCall,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Response};
use serde_json::{json, Map, Value};

use crate::http::{self, LineBuffer, FALLBACK_READ_MS};
use crate::openai::{default_api_base, post_json, recover_affordable, u64_at};
use crate::provider_config::ProviderConfig;
use crate::wire::{self, arguments_value, describe_api_error, truthy};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const OAUTH_BETA: &str = "oauth-2025-04-20";

pub struct AnthropicProvider {
    cfg: ProviderConfig,
    model: RwLock<String>,
    client: Client,
    /// Live OAuth bearer token (re-minted on 401 via `cfg.refresh_auth`).
    token: RwLock<Option<String>>,
    /// Whether the token source has been asked for an initial token yet.
    token_checked: AtomicBool,
}

impl AnthropicProvider {
    pub fn new(cfg: ProviderConfig) -> Self {
        let token = RwLock::new(cfg.auth_token.clone().filter(|t| !t.is_empty()));
        let model = RwLock::new(cfg.model.clone());
        Self { cfg, model, client: http::build_client(), token, token_checked: AtomicBool::new(false) }
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    fn current_model(&self) -> String {
        self.model.read().map(|m| m.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }

    fn current_token(&self) -> Option<String> {
        self.token.read().map(|t| t.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }

    fn set_token(&self, tok: Option<String>) {
        match self.token.write() {
            Ok(mut t) => *t = tok,
            Err(e) => *e.into_inner() = tok,
        }
    }

    /// `https://api.anthropic.com` → `https://api.anthropic.com/v1`.
    fn api_base(&self) -> String {
        default_api_base(&self.cfg.base_url)
    }

    /// Is an OAuth bearer in use (vs. an API key)?
    pub fn uses_oauth(&self) -> bool {
        self.current_token().is_some()
    }

    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(HeaderName::from_static("anthropic-version"), HeaderValue::from_static(ANTHROPIC_VERSION));
        if let Some(tok) = self.current_token() {
            // OAuth (subscription) path — same as Claude Code / `ant`.
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {tok}")) {
                h.insert(AUTHORIZATION, v);
            }
            h.insert(HeaderName::from_static("anthropic-beta"), HeaderValue::from_static(OAUTH_BETA));
        } else if !self.cfg.api_key.is_empty() {
            if let Ok(v) = HeaderValue::from_str(&self.cfg.api_key) {
                h.insert(HeaderName::from_static("x-api-key"), v);
            }
        }
        h
    }

    /// First use with no configured token: ask the token source (the `ant` CLI)
    /// once. The TS factory did this synchronously at construction; doing it
    /// lazily keeps the factory non-blocking and makes the `ant` child
    /// cancellable by Esc.
    async fn ensure_token(&self, cancel: Option<&CancellationToken>) -> Result<(), ProviderError> {
        if self.current_token().is_some() || self.token_checked.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let Some(src) = self.cfg.refresh_auth.clone() else { return Ok(()) };
        let tok = tokio::select! {
            biased;
            _ = http::cancelled(cancel) => {
                self.token_checked.store(false, Ordering::SeqCst);
                return Err(ProviderError::Interrupted);
            }
            t = src.token(false) => t,
        };
        if let Some(t) = tok.filter(|t| !t.is_empty()) {
            self.set_token(Some(t));
        }
        Ok(())
    }

    /// Re-mint the OAuth token (if a refresher is wired) and report whether it changed.
    async fn try_refresh(&self, cancel: Option<&CancellationToken>) -> Result<bool, ProviderError> {
        let Some(src) = self.cfg.refresh_auth.clone() else { return Ok(false) };
        let fresh = tokio::select! {
            biased;
            _ = http::cancelled(cancel) => return Err(ProviderError::Interrupted),
            t = src.token(true) => t,
        };
        match fresh.filter(|t| !t.is_empty()) {
            Some(t) if Some(&t) != self.current_token().as_ref() => {
                self.set_token(Some(t));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Build the Anthropic Messages request body from internal messages.
    pub fn anthropic_body(&self, messages: &[Message], opts: &ModelOptions, stream: bool, model: &str) -> Map<String, Value> {
        let (system, msgs) = to_anthropic_messages(messages);
        let max_tokens = opts.max_tokens.unwrap_or(self.cfg.max_tokens);
        let mut body = Map::new();
        body.insert("model".into(), json!(model));
        body.insert("max_tokens".into(), json!(if max_tokens == 0 { 4096 } else { max_tokens }));
        body.insert("messages".into(), Value::Array(msgs));
        body.insert("stream".into(), json!(stream));
        if !system.is_empty() {
            body.insert("system".into(), json!(system));
        }
        if let Some(tools) = opts.tools.as_ref().filter(|t| !t.is_empty()) {
            let tools: Vec<Value> = tools
                .iter()
                .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.parameters }))
                .collect();
            body.insert("tools".into(), Value::Array(tools));
        }
        body
    }

    /// POST `/messages`, refreshing an expired OAuth token (401) once.
    async fn open(&self, messages: &[Message], opts: &ModelOptions, stream: bool) -> anyhow::Result<(Response, String)> {
        let cancel = opts.cancel.as_ref();
        let timeout = self.cfg.timeout_ms();
        self.ensure_token(cancel).await?;
        let url = format!("{}/messages", self.api_base());
        let model = self.current_model();
        let body = Value::Object(self.anthropic_body(messages, opts, stream, &model));
        let mut res = http::send("anthropic", post_json(&self.client, &url, self.headers(), &body), cancel, timeout).await?;
        if res.status().as_u16() == 401 && self.try_refresh(cancel).await? {
            res = http::send("anthropic", post_json(&self.client, &url, self.headers(), &body), cancel, timeout).await?;
        }
        if !res.status().is_success() {
            let status = res.status().as_u16();
            let detail = http::safe_text("anthropic", res, cancel, timeout).await?;
            return Err(ProviderError::http("anthropic", status, detail).into());
        }
        Ok((res, model))
    }

    async fn get_models(&self) -> Result<Response, ProviderError> {
        self.ensure_token(None).await?;
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        let url = format!("{}/models", self.api_base());
        http::send("anthropic", self.client.get(url).headers(self.headers()), None, ms).await
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> String {
        self.current_model()
    }

    fn set_model(&self, model: &str) {
        match self.model.write() {
            Ok(mut m) => *m = model.to_string(),
            Err(e) => *e.into_inner() = model.to_string(),
        }
    }

    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        let (res, model) = self.open(messages, opts, false).await?;
        let json = http::read_json("anthropic", res, opts.cancel.as_ref(), self.cfg.timeout_ms()).await?;
        if let Some(err) = json.get("error").filter(|e| truthy(e)) {
            return Err(ProviderError::Api { provider: "anthropic".into(), message: describe_api_error(err) }.into());
        }
        let (text, tool_calls) = parse_content(json.get("content"));
        let usage = json.get("usage").cloned().unwrap_or(Value::Null);
        Ok(GenerateResult {
            text,
            tool_calls,
            thinking: None,
            prompt_tokens: u64_at(&usage, "input_tokens"),
            completion_tokens: u64_at(&usage, "output_tokens"),
            model,
        })
    }

    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let (mut res, model) = self.open(messages, opts, true).await?;
        let cancel = opts.cancel.as_ref();
        let mut state = StreamState::default();
        let mut lines = LineBuffer::default();
        // Same stall watchdog as the OpenAI path (Anthropic sends `ping`
        // events while a long answer is being produced, so silence is real).
        while let Some(chunk) = http::next_chunk("anthropic", &mut res, cancel, self.cfg.timeout_ms()).await? {
            for line in lines.push(&chunk) {
                state.apply_line(&line, on_chunk)?;
            }
        }
        if let Some(rest) = lines.finish() {
            state.apply_line(&rest, on_chunk)?;
        }
        Ok(state.finish(model))
    }

    async fn health_check(&self) -> HealthStatus {
        match self.get_models().await {
            Ok(res) if res.status().is_success() => HealthStatus { ok: true, detail: None },
            Ok(res) => HealthStatus { ok: false, detail: Some(format!("HTTP {}", res.status().as_u16())) },
            Err(e) => HealthStatus { ok: false, detail: Some(e.to_string()) },
        }
    }

    /// The Models API works under both auth modes.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let res = self.get_models().await?;
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        if !res.status().is_success() {
            let status = res.status().as_u16();
            let detail = http::safe_text("anthropic", res, None, ms).await?;
            return Err(ProviderError::http("anthropic", status, detail).into());
        }
        let json = http::read_json("anthropic", res, None, ms).await?;
        Ok(json
            .get("data")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()))
                    .map(|id| ModelInfo { name: id.to_string(), size: None })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Tiny streaming `/messages` request; an expired OAuth token (401) is
    /// refreshed and the probe retried once.
    async fn probe(&self, cancel: &CancellationToken) -> anyhow::Result<()> {
        self.ensure_token(Some(cancel)).await?;
        let url = format!("{}/messages", self.api_base());
        let opts = ModelOptions { max_tokens: Some(16), ..Default::default() };
        let ping = [Message::user("ping")];
        let timeout = self.cfg.timeout_ms();
        let mut retried = false;
        loop {
            let model = self.current_model();
            let body = Value::Object(self.anthropic_body(&ping, &opts, true, &model));
            let res = http::send("anthropic", post_json(&self.client, &url, self.headers(), &body), Some(cancel), timeout)
                .await?;
            let status = res.status().as_u16();
            if res.status().is_success() {
                drop(res); // body dropped unread
                return Ok(());
            }
            let detail = http::safe_text("anthropic", res, Some(cancel), timeout).await?;
            if !retried {
                let recover = if status == 401 {
                    self.try_refresh(Some(cancel)).await?
                } else {
                    recover_affordable(status, &detail, &ModelOptions::default()).is_some()
                };
                if recover {
                    retried = true;
                    continue;
                }
            }
            return Err(ProviderError::http("anthropic", status, detail).into());
        }
    }
}

/// One content block being streamed.
#[derive(Default)]
struct Block {
    kind: String,
    id: String,
    name: String,
    json: String,
}

/// Incremental state of one Messages SSE stream.
#[derive(Default)]
struct StreamState {
    text: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    /// Per content-block accumulator: text blocks stream `text_delta`;
    /// tool_use blocks stream their JSON arguments as `input_json_delta`.
    blocks: BTreeMap<u64, Block>,
}

impl StreamState {
    fn apply_line(&mut self, line: &str, on_chunk: &mut (dyn FnMut(&str) + Send)) -> Result<(), ProviderError> {
        // `event:` lines are ignored; the payload's `type` says the same.
        let Some(obj) = http::sse_payload(line) else { return Ok(()) };
        let idx = obj.get("index").and_then(Value::as_u64).unwrap_or(0);
        match obj.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message_start" => {
                self.prompt_tokens = obj.pointer("/message/usage/input_tokens").and_then(Value::as_u64).or(self.prompt_tokens);
            }
            "content_block_start" => {
                let cb = obj.get("content_block").cloned().unwrap_or(Value::Null);
                let s = |k: &str| cb.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                self.blocks.insert(idx, Block { kind: s("type"), id: s("id"), name: s("name"), json: String::new() });
            }
            "content_block_delta" => {
                let d = obj.get("delta").cloned().unwrap_or(Value::Null);
                match d.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "text_delta" => {
                        if let Some(t) = d.get("text").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                            self.text.push_str(t);
                            on_chunk(t);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(p) = d.get("partial_json").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                            if let Some(b) = self.blocks.get_mut(&idx) {
                                b.json.push_str(p);
                            }
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                self.completion_tokens = obj.pointer("/usage/output_tokens").and_then(Value::as_u64).or(self.completion_tokens);
            }
            // An in-band `error` event (e.g. `overloaded_error` mid-stream):
            // fail loudly so the chat layer can fall back, instead of returning
            // a truncated reply (same rule as the OpenAI SSE path).
            "error" => {
                let err = obj.get("error").cloned().unwrap_or(obj.clone());
                return Err(ProviderError::Api { provider: "anthropic".into(), message: describe_api_error(&err) });
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self, model: String) -> GenerateResult {
        let completion_tokens = match self.completion_tokens {
            None if !self.text.is_empty() => Some(estimate_tokens(&self.text) as u64),
            other => other,
        };
        let tool_calls = self
            .blocks
            .into_iter()
            .filter(|(_, b)| b.kind == "tool_use" && !b.name.is_empty())
            .map(|(idx, b)| ToolCall {
                id: if b.id.is_empty() { format!("call_{idx}") } else { b.id },
                name: b.name,
                arguments: wire::parse_arguments(&b.json),
            })
            .collect();
        GenerateResult {
            text: self.text,
            tool_calls,
            thinking: None,
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            model,
        }
    }
}

/// Parse a non-streamed `content` array into text + tool calls.
pub fn parse_content(content: Option<&Value>) -> (String, Vec<ToolCall>) {
    let Some(arr) = content.and_then(Value::as_array) else {
        return (String::new(), Vec::new());
    };
    let mut text = String::new();
    let mut calls = Vec::new();
    for (i, block) in arr.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or_default()),
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("call_{i}"));
                let name = block.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                calls.push(ToolCall { id, name, arguments: arguments_value(block.get("input")) });
            }
            _ => {}
        }
    }
    (text, calls)
}

/// Translate internal messages → Anthropic `(system, messages)`. System turns
/// are folded into the top-level `system` string; assistant tool calls become
/// `tool_use` blocks; `tool` results become `tool_result` blocks in a user
/// turn. Orphan tool results (whose tool_use was dropped by compaction)
/// degrade to plain user text so the request stays valid instead of 400-ing.
/// Empty assistant turns (no text, no calls) are dropped.
pub fn to_anthropic_messages(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut msgs = Vec::new();
    let mut known_tool_use_ids: HashSet<&str> = HashSet::new();
    for m in messages {
        match m.role {
            Role::System => {
                if !m.content.trim().is_empty() {
                    system_parts.push(&m.content);
                }
            }
            Role::Assistant => {
                let mut content = Vec::new();
                if !m.content.trim().is_empty() {
                    content.push(json!({ "type": "text", "text": m.content }));
                }
                for tc in m.tool_calls.iter().flatten() {
                    known_tool_use_ids.insert(&tc.id);
                    content.push(json!({ "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments }));
                }
                if !content.is_empty() {
                    msgs.push(json!({ "role": "assistant", "content": content }));
                }
            }
            Role::Tool => match m.tool_call_id.as_deref().filter(|id| known_tool_use_ids.contains(id)) {
                Some(id) => msgs.push(json!({
                    "role": "user",
                    "content": [{ "type": "tool_result", "tool_use_id": id, "content": m.content }],
                })),
                // Orphaned result (e.g. after history compaction) — keep it as context.
                None => msgs.push(json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": format!("[tool result] {}", m.content) }],
                })),
            },
            Role::User => msgs.push(json!({ "role": "user", "content": [{ "type": "text", "text": m.content }] })),
        }
    }
    (system_parts.join("\n\n"), msgs)
}
