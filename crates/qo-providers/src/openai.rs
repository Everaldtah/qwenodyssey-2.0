//! Shared OpenAI-compatible provider (port of base.ts). Most local servers
//! (LM Studio, vLLM, llama.cpp) and the hosted gateways (OpenAI, NVIDIA NIM,
//! OpenRouter) expose an OpenAI-style `/v1/chat/completions` endpoint, so the
//! request/stream logic is implemented once here and each backend only tweaks
//! a few hooks ([`OpenAiBackend`]): API base, extra headers, system-role
//! support, extra body fields and error recovery.
//!
//! Reliability (v0.4.2): every request goes through [`crate::http`], so the
//! request timeout covers connect + headers, a streamed body that goes silent
//! is aborted as stalled, and the caller's cancel token (Esc) interrupts every
//! await. A failed request may be retried ONCE with adjusted options
//! ([`OpenAiBackend::recover_from_error`]).

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

use async_trait::async_trait;
use qo_core::{
    CancellationToken, GenerateResult, HealthStatus, Message, ModelInfo, ModelOptions, Provider, ProviderError,
};
use qo_core::tokens::estimate_tokens;
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, RequestBuilder};
use serde_json::{json, Map, Value};

use crate::http::{self, LineBuffer, FALLBACK_READ_MS};
use crate::profile;
use crate::provider_config::ProviderConfig;
use crate::wire::{self, describe_api_error, truthy, ToolCallAccumulator};

/* ───────────────────────── shared request shaping ───────────────────────── */

/// OpenAI-style base: `http://host:port` → `http://host:port/v1` (kept as is
/// when it already ends in `/v1`).
pub(crate) fn default_api_base(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if b.ends_with("/v1") {
        b.to_string()
    } else {
        format!("{b}/v1")
    }
}

/// Whether the model should expose a chain-of-thought this turn. An explicit
/// per-request option wins; otherwise the configured policy (`model.think`) is
/// resolved against the model's family. `None` means the model has no thinking
/// mode at all, so the flag stays off the wire entirely.
pub(crate) fn thinking_enabled(cfg: &ProviderConfig, model: &str, opts: &ModelOptions) -> Option<bool> {
    if let Some(t) = opts.think {
        return Some(t);
    }
    profile::resolve_thinking(model, cfg.think)
}

/// Decoding knobs that are only sent when actually set (0/unset = leave the
/// choice to the server). These are the levers that keep SMALL models from
/// looping; big hosted models normally leave them unset.
pub(crate) fn sampling_extras(cfg: &ProviderConfig, opts: &ModelOptions) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(k) = opts.top_k.or(cfg.top_k).filter(|k| *k > 0) {
        out.insert("top_k".into(), json!(k));
    }
    if let Some(p) = opts.presence_penalty.or(cfg.presence_penalty).filter(|p| *p > 0.0) {
        out.insert("presence_penalty".into(), json!(p));
    }
    if let Some(r) = opts.repeat_penalty.or(cfg.repeat_penalty).filter(|r| *r > 0.0) {
        out.insert("repetition_penalty".into(), json!(r));
    }
    out
}

/// The OpenAI chat-completions request body (without backend extras).
pub(crate) fn openai_body(
    cfg: &ProviderConfig,
    model: &str,
    messages: &[Message],
    opts: &ModelOptions,
    stream: bool,
    system_role_supported: bool,
) -> Map<String, Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(wire::wire_messages(messages, system_role_supported)));
    body.insert("temperature".into(), json!(opts.temperature.unwrap_or(cfg.temperature)));
    body.insert("top_p".into(), json!(opts.top_p.unwrap_or(cfg.top_p)));
    body.insert("max_tokens".into(), json!(opts.max_tokens.unwrap_or(cfg.max_tokens)));
    body.extend(sampling_extras(cfg, opts));
    body.insert("stream".into(), json!(stream));
    // Hybrid thinking models (Qwen3 / Qwen3.5) are toggled through the chat
    // template: `chat_template_kwargs.enable_thinking`, which LM Studio, vLLM and
    // llama.cpp all honour. The old approach appended a literal " /no_think" to
    // the user message — Qwen 3.5 no longer recognises that switch and small
    // models answered the text itself ("No, I'm not thinking…") instead of the
    // question. A backend's extra_body() may still override this.
    if let Some(think) = thinking_enabled(cfg, model, opts) {
        if profile::hybrid_thinking(model) {
            body.insert("chat_template_kwargs".into(), json!({ "enable_thinking": think }));
        }
    }
    if let Some(stop) = &opts.stop {
        body.insert("stop".into(), json!(stop));
    }
    if opts.json {
        body.insert("response_format".into(), json!({ "type": "json_object" }));
    }
    if let Some(tools) = opts.tools.as_ref().filter(|t| !t.is_empty()) {
        body.insert("tools".into(), Value::Array(wire::wire_tools(tools)));
    }
    body
}

static AFFORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)can only afford (\d+)").expect("valid regex"));

/// Base recovery rule: OpenRouter's HTTP 402 "You requested up to 8192 tokens,
/// but can only afford 3436" — retry once within the affordable budget (N − 64,
/// at least 256) instead of failing the turn.
pub fn recover_affordable(status: u16, detail: &str, opts: &ModelOptions) -> Option<ModelOptions> {
    if status != 402 {
        return None;
    }
    let n: i64 = AFFORD.captures(detail)?.get(1)?.as_str().parse().ok()?;
    let max = n - 64;
    if max >= 256 {
        let max = u32::try_from(max).ok()?;
        return Some(ModelOptions { max_tokens: Some(max), ..opts.clone() });
    }
    None
}

/// The `Content-Type` + optional bearer headers every OpenAI-style request carries.
pub(crate) fn bearer_headers(api_key: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if !api_key.is_empty() {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
            h.insert(AUTHORIZATION, v);
        }
    }
    h
}

/// A JSON POST with the given headers.
pub(crate) fn post_json(client: &Client, url: &str, headers: HeaderMap, body: &Value) -> RequestBuilder {
    client
        .post(url)
        .headers(headers)
        .body(serde_json::to_vec(body).unwrap_or_default())
}

/// `x.y` as u64 when it is a number.
pub(crate) fn u64_at(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

/// The tiny-streaming-request liveness check shared by every backend: resolve
/// as soon as the server answers with HTTP 200 (the body is dropped unread —
/// a model that accepts the request and then never sends a byte is exactly
/// the failure this catches, while reasoning models that need many tokens to
/// finish still pass quickly). A non-200 may be retried once via `recover`.
pub(crate) async fn probe_with<R, F>(
    provider: &str,
    mut request: R,
    cancel: &CancellationToken,
    timeout_ms: Option<u64>,
    mut recover: F,
) -> anyhow::Result<()>
where
    R: FnMut() -> RequestBuilder,
    F: FnMut(u16, &str) -> bool,
{
    let mut retried = false;
    loop {
        let res = http::send(provider, request(), Some(cancel), timeout_ms).await?;
        let status = res.status().as_u16();
        if res.status().is_success() {
            drop(res); // body dropped unread: closes the connection
            return Ok(());
        }
        let detail = http::safe_text(provider, res, Some(cancel), timeout_ms).await?;
        if !retried && recover(status, &detail) {
            retried = true;
            continue;
        }
        return Err(ProviderError::http(provider, status, detail).into());
    }
}

/* ───────────────────────────── backend hooks ───────────────────────────── */

/// Per-backend hooks over the shared OpenAI-compatible implementation (the
/// overridable methods of the TS base class).
pub trait OpenAiBackend: Send + Sync + 'static {
    /// Provider id (`Provider::name`).
    fn name(&self) -> &'static str;

    /// OpenAI-style base URL; default appends `/v1` when missing.
    fn api_base(&self, base_url: &str) -> String {
        default_api_base(base_url)
    }

    /// Headers added on top of `Content-Type` + bearer key.
    fn extra_headers(&self) -> Vec<(&'static str, &'static str)> {
        Vec::new()
    }

    /// Whether this backend/model honours the `system` role. Some hosted models
    /// (e.g. Kimi K2.x on NVIDIA NIM) degenerate into repetition when given a
    /// system message, so a backend can opt to fold it into the user turn.
    fn system_role_supported(&self, model: &str) -> bool {
        let _ = model;
        true
    }

    /// Extra provider-specific request-body fields merged (last, overriding)
    /// into every request, e.g. NVIDIA NIM's `chat_template_kwargs`.
    fn extra_body(&self, cfg: &ProviderConfig, model: &str) -> Map<String, Value> {
        let _ = (cfg, model);
        Map::new()
    }

    /// Decide whether a failed request can be retried with adjusted options:
    /// the options to retry with, or `None` to surface the error. Called at
    /// most once per request. Default: [`recover_affordable`] (OpenRouter 402).
    fn recover_from_error(&self, status: u16, detail: &str, opts: &ModelOptions, model: &str) -> Option<ModelOptions> {
        let _ = model;
        recover_affordable(status, detail, opts)
    }
}

/// LM Studio's local server (default `http://localhost:1234`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LmStudio;
impl OpenAiBackend for LmStudio {
    fn name(&self) -> &'static str {
        "lmstudio"
    }
}

/// vLLM's OpenAI server (default `http://localhost:8000`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Vllm;
impl OpenAiBackend for Vllm {
    fn name(&self) -> &'static str {
        "vllm"
    }
}

/// llama.cpp server (default `http://localhost:8080`), `/v1/chat/completions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LlamaCpp;
impl OpenAiBackend for LlamaCpp {
    fn name(&self) -> &'static str {
        "llamacpp"
    }
}

/// Generic OpenAI-compatible endpoint (OpenAI itself, Groq, …).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiEndpoint;
impl OpenAiBackend for OpenAiEndpoint {
    fn name(&self) -> &'static str {
        "openai"
    }
}

/// OpenRouter (`https://openrouter.ai/api/v1`): many providers behind one key,
/// with OpenRouter's optional app-attribution headers.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenRouter;
impl OpenAiBackend for OpenRouter {
    fn name(&self) -> &'static str {
        "openrouter"
    }

    /// OpenRouter recommends HTTP-Referer / X-Title for app attribution (optional).
    fn extra_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("HTTP-Referer", "https://github.com/Everaldtah/Qwenodyssey"), ("X-Title", "Qwenodyssey")]
    }
}

static NIM_KIMI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(^|[-_/:.])(kimi|k2)").expect("valid regex"));
static NIM_NEMOTRON: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)nemotron").expect("valid regex"));
static NIM_THINKING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(^|[-_/:.])(r1|qwq|o1|o3|thinking|reason|kimi|k2)|deepseek-(r1|v4)").expect("valid regex")
});
static NIM_BUDGET_ERR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)thinking_token_budget|reasoning_budget").expect("valid regex"));

/// NVIDIA NIM (`https://integrate.api.nvidia.com/v1`): strong hosted models
/// (e.g. moonshotai/kimi-k2.6, nvidia/nemotron-*) usable as the primary brain
/// or a fallback. Bearer API key resolved by the factory from config/env.
#[derive(Debug, Default)]
pub struct Nvidia {
    /// Set once NIM rejects `reasoning_budget`; it is then left off the wire
    /// for the rest of the session.
    budget_rejected: AtomicBool,
}

impl Nvidia {
    /// Nemotron reasoning models use a different thinking toggle + reasoning_budget.
    fn is_nemotron(model: &str) -> bool {
        NIM_NEMOTRON.is_match(model)
    }

    /// Models whose NIM chat template exposes a toggleable internal "thinking" mode.
    fn is_thinking_model(model: &str) -> bool {
        NIM_THINKING.is_match(model)
    }

    /// Has NIM rejected the reasoning budget this session?
    pub fn budget_rejected(&self) -> bool {
        self.budget_rejected.load(Ordering::Relaxed)
    }
}

impl OpenAiBackend for Nvidia {
    fn name(&self) -> &'static str {
        "nvidia"
    }

    /// Kimi K2/K2.6 on NVIDIA NIM degenerates into repetition (NIM reports
    /// finish_reason: "repetition") when given a `system` message —
    /// reproducible across temperatures. Fold the system prompt into the user
    /// turn for those models; other NIM models handle system fine.
    fn system_role_supported(&self, model: &str) -> bool {
        !NIM_KIMI.is_match(model)
    }

    /// Per-model NIM request extras:
    ///  - Nemotron: `chat_template_kwargs.enable_thinking` (+ reasoning_budget).
    ///    Its reasoning comes back in a separate `reasoning_content` field, so
    ///    the answer stays clean. Reasoning is on by default (NVIDIA's recipe).
    ///  - Other thinking models (kimi, deepseek-v4, r1…): `thinking=false`,
    ///    because on NIM their thinking path degenerates / leaks raw CoT into
    ///    `content`.
    ///  - Non-thinking models (qwen-coder, llama, gpt-oss): nothing extra.
    fn extra_body(&self, cfg: &ProviderConfig, model: &str) -> Map<String, Value> {
        let mut out = Map::new();
        if Self::is_nemotron(model) {
            out.insert(
                "chat_template_kwargs".into(),
                json!({ "enable_thinking": cfg.nemotron_thinking != Some(false) }),
            );
            // NIM's current Nemotron runner rejects the budget outright
            // ("thinking_token_budget is not yet supported by the V2 model
            // runner" → HTTP 400), so it is sent only when the user explicitly
            // configures one — and dropped for the rest of the session once NIM
            // rejects it (see recover_from_error).
            if !self.budget_rejected() {
                if let Some(b) = cfg.reasoning_budget.filter(|b| *b > 0) {
                    out.insert("reasoning_budget".into(), json!(b));
                }
            }
            return out;
        }
        if cfg.disable_thinking != Some(false) && Self::is_thinking_model(model) {
            out.insert("chat_template_kwargs".into(), json!({ "thinking": false }));
        }
        out
    }

    fn recover_from_error(&self, status: u16, detail: &str, opts: &ModelOptions, model: &str) -> Option<ModelOptions> {
        if status == 400 && !self.budget_rejected() && Self::is_nemotron(model) && NIM_BUDGET_ERR.is_match(detail) {
            self.budget_rejected.store(true, Ordering::Relaxed);
            return Some(opts.clone());
        }
        recover_affordable(status, detail, opts)
    }
}

/* ─────────────────────────────── provider ─────────────────────────────── */

/// The shared OpenAI-compatible provider, parameterised by its backend hooks.
pub struct OpenAiCompatibleProvider<B: OpenAiBackend> {
    backend: B,
    cfg: ProviderConfig,
    model: RwLock<String>,
    client: Client,
}

/// LM Studio (`lmstudio`).
pub type LmStudioProvider = OpenAiCompatibleProvider<LmStudio>;
/// vLLM (`vllm`).
pub type VllmProvider = OpenAiCompatibleProvider<Vllm>;
/// llama.cpp (`llamacpp`).
pub type LlamaCppProvider = OpenAiCompatibleProvider<LlamaCpp>;
/// Generic OpenAI-compatible endpoint (`openai`).
pub type OpenAiCompatibleEndpointProvider = OpenAiCompatibleProvider<OpenAiEndpoint>;
/// NVIDIA NIM (`nvidia`).
pub type NvidiaNimProvider = OpenAiCompatibleProvider<Nvidia>;
/// OpenRouter (`openrouter`).
pub type OpenRouterProvider = OpenAiCompatibleProvider<OpenRouter>;

impl<B: OpenAiBackend + Default> OpenAiCompatibleProvider<B> {
    pub fn new(cfg: ProviderConfig) -> Self {
        Self::with_backend(B::default(), cfg)
    }
}

impl<B: OpenAiBackend> OpenAiCompatibleProvider<B> {
    pub fn with_backend(backend: B, cfg: ProviderConfig) -> Self {
        let model = RwLock::new(cfg.model.clone());
        Self { backend, cfg, model, client: http::build_client() }
    }

    /// The settings this provider was built with (the live model is [`Provider::model`]).
    pub fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn current_model(&self) -> String {
        self.model.read().map(|m| m.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }

    fn api_base(&self) -> String {
        self.backend.api_base(&self.cfg.base_url)
    }

    fn headers(&self) -> HeaderMap {
        let mut h = bearer_headers(&self.cfg.api_key);
        for (k, v) in self.backend.extra_headers() {
            if let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
                h.insert(k, v);
            }
        }
        h
    }

    /// Full request body for `model`, backend extras merged last.
    pub fn body(&self, messages: &[Message], opts: &ModelOptions, stream: bool, model: &str) -> Map<String, Value> {
        let mut body =
            openai_body(&self.cfg, model, messages, opts, stream, self.backend.system_role_supported(model));
        body.extend(self.backend.extra_body(&self.cfg, model));
        body
    }

    fn post(&self, url: &str, body: &Value) -> RequestBuilder {
        post_json(&self.client, url, self.headers(), body)
    }

    /// GET `{api_base}/models`, bounded by the request timeout (or 30 s).
    async fn get_models(&self) -> Result<reqwest::Response, ProviderError> {
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        let url = format!("{}/models", self.api_base());
        http::send(self.backend.name(), self.client.get(url).headers(self.headers()), None, ms).await
    }
}

#[async_trait]
impl<B: OpenAiBackend> Provider for OpenAiCompatibleProvider<B> {
    fn name(&self) -> &str {
        self.backend.name()
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
        let name = self.backend.name();
        let cancel = opts.cancel.as_ref();
        let timeout = self.cfg.timeout_ms();
        let url = format!("{}/chat/completions", self.api_base());
        let mut current: Cow<'_, ModelOptions> = Cow::Borrowed(opts);
        let mut retried = false;
        loop {
            let model = self.current_model();
            let body = Value::Object(self.body(messages, &current, false, &model));
            let res = http::send(name, self.post(&url, &body), cancel, timeout).await?;
            let status = res.status().as_u16();
            if !res.status().is_success() {
                let detail = http::safe_text(name, res, cancel, timeout).await?;
                if !retried {
                    if let Some(retry) = self.backend.recover_from_error(status, &detail, &current, &model) {
                        current = Cow::Owned(retry);
                        retried = true;
                        continue;
                    }
                }
                return Err(ProviderError::http(name, status, detail).into());
            }
            let json = http::read_json(name, res, cancel, timeout).await?;
            // Some gateways (NVIDIA NIM) answer HTTP 200 with an error envelope
            // instead of a completion — surface it as a real error so the
            // fallback chain runs.
            if let Some(err) = json.get("error").filter(|e| truthy(e)) {
                return Err(ProviderError::Api { provider: name.to_string(), message: describe_api_error(err) }.into());
            }
            return Ok(parse_completion(&json, model));
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let name = self.backend.name();
        let cancel = opts.cancel.as_ref();
        let timeout = self.cfg.timeout_ms();
        let url = format!("{}/chat/completions", self.api_base());
        let mut current: Cow<'_, ModelOptions> = Cow::Borrowed(opts);
        let mut retried = false;
        loop {
            let model = self.current_model();
            let mut body = self.body(messages, &current, true, &model);
            // Ask OpenAI-compatible servers to include a final usage frame (vLLM,
            // llama.cpp, LM Studio, OpenRouter all honour this), so we get EXACT
            // prompt/completion counts even while streaming.
            body.insert("stream_options".into(), json!({ "include_usage": true }));
            let res = http::send(name, self.post(&url, &Value::Object(body)), cancel, timeout).await?;
            let status = res.status().as_u16();
            if !res.status().is_success() {
                let detail = http::safe_text(name, res, cancel, timeout).await?;
                if !retried {
                    if let Some(retry) = self.backend.recover_from_error(status, &detail, &current, &model) {
                        current = Cow::Owned(retry);
                        retried = true;
                        continue;
                    }
                }
                return Err(ProviderError::http(name, status, detail).into());
            }
            return read_sse_stream(name, res, on_chunk, cancel, timeout, model).await;
        }
    }

    async fn health_check(&self) -> HealthStatus {
        match self.get_models().await {
            Ok(res) if res.status().is_success() => HealthStatus { ok: true, detail: None },
            Ok(res) => HealthStatus { ok: false, detail: Some(format!("HTTP {}", res.status().as_u16())) },
            Err(e) => HealthStatus { ok: false, detail: Some(e.to_string()) },
        }
    }

    /// List models via the OpenAI-style `/models` endpoint.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let name = self.backend.name();
        let res = self.get_models().await?;
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        if !res.status().is_success() {
            let status = res.status().as_u16();
            let detail = http::safe_text(name, res, None, ms).await?;
            return Err(ProviderError::http(name, status, detail).into());
        }
        let json = http::read_json(name, res, None, ms).await?;
        Ok(parse_model_list(&json))
    }

    async fn probe(&self, cancel: &CancellationToken) -> anyhow::Result<()> {
        let url = format!("{}/chat/completions", self.api_base());
        let opts = ModelOptions { max_tokens: Some(16), ..Default::default() };
        let ping = [Message::user("ping")];
        probe_with(
            self.backend.name(),
            || {
                let model = self.current_model();
                self.post(&url, &Value::Object(self.body(&ping, &opts, true, &model)))
            },
            cancel,
            self.cfg.timeout_ms(),
            |status, detail| {
                self.backend
                    .recover_from_error(status, detail, &ModelOptions::default(), &self.current_model())
                    .is_some()
            },
        )
        .await
    }
}

/// `{data|models: [{id|name}]}` → model names.
pub(crate) fn parse_model_list(json: &Value) -> Vec<ModelInfo> {
    let data = json
        .get("data")
        .filter(|v| !v.is_null())
        .or_else(|| json.get("models"))
        .and_then(Value::as_array);
    data.map(|arr| {
        arr.iter()
            .filter_map(|m| {
                let name = m
                    .get("id")
                    .filter(|v| !v.is_null())
                    .or_else(|| m.get("name"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())?;
                Some(ModelInfo { name: name.to_string(), size: None })
            })
            .collect()
    })
    .unwrap_or_default()
}

/// Parse a non-streamed chat completion.
pub(crate) fn parse_completion(json: &Value, model: String) -> GenerateResult {
    let msg = json.pointer("/choices/0/message").cloned().unwrap_or(Value::Null);
    let text = msg.get("content").and_then(Value::as_str).unwrap_or_default().to_string();
    let tool_calls = wire::parse_tool_calls(msg.get("tool_calls"));
    // Reasoning models return their chain-of-thought in a separate field
    // (NIM/vLLM: reasoning_content, OpenRouter: reasoning). Keep it so the chat
    // layer can show it and tell "thought but never answered" from "no response".
    let reasoning = msg.get("reasoning_content").filter(|v| !v.is_null()).or_else(|| msg.get("reasoning"));
    let thinking = reasoning
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string);
    let usage = json.get("usage").cloned().unwrap_or(Value::Null);
    GenerateResult {
        text,
        tool_calls,
        thinking,
        prompt_tokens: u64_at(&usage, "prompt_tokens"),
        completion_tokens: u64_at(&usage, "completion_tokens"),
        model,
    }
}

/// Incremental state of one OpenAI SSE stream.
#[derive(Default)]
pub(crate) struct SseState {
    pub text: String,
    pub thinking: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub tools: ToolCallAccumulator,
}

impl SseState {
    /// Apply one SSE line. Errors on an in-band error frame.
    pub(crate) fn apply_line(
        &mut self,
        provider: &str,
        line: &str,
        on_chunk: &mut (dyn FnMut(&str) + Send),
    ) -> Result<(), ProviderError> {
        let Some(obj) = http::sse_payload(line) else {
            return Ok(()); // keep-alive / comment / partial frame / [DONE]
        };
        // An in-band error frame (e.g. NIM "503 Service temporarily overloaded"
        // delivered inside an HTTP 200 stream) used to be silently skipped,
        // leaving an empty reply with 0 tokens. Fail loudly instead so the chat
        // layer can fall back to another model.
        if let Some(err) = obj.get("error").filter(|e| truthy(e)) {
            return Err(ProviderError::Api {
                provider: provider.to_string(),
                message: describe_api_error(err),
            });
        }
        if let Some(usage) = obj.get("usage").filter(|u| truthy(u)) {
            self.prompt_tokens = u64_at(usage, "prompt_tokens").or(self.prompt_tokens);
            self.completion_tokens = u64_at(usage, "completion_tokens").or(self.completion_tokens);
        }
        let Some(delta) = obj.pointer("/choices/0/delta") else {
            return Ok(());
        };
        if let Some(content) = delta.get("content").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            self.text.push_str(content);
            on_chunk(content);
        }
        let reason = delta.get("reasoning_content").filter(|v| !v.is_null()).or_else(|| delta.get("reasoning"));
        if let Some(r) = reason.and_then(Value::as_str).filter(|s| !s.is_empty()) {
            self.thinking.push_str(r);
        }
        if let Some(tcs) = delta.get("tool_calls").filter(|v| v.is_array()) {
            self.tools.push(tcs);
        }
        Ok(())
    }

    pub(crate) fn finish(self, model: String) -> GenerateResult {
        // No usage frame from the server: fall back to the heuristic count so
        // callers still get a number (better than nothing for budgets).
        let completion_tokens = match self.completion_tokens {
            None if !self.text.is_empty() => Some(estimate_tokens(&self.text) as u64),
            other => other,
        };
        let thinking = self.thinking.trim();
        GenerateResult {
            thinking: (!thinking.is_empty()).then(|| thinking.to_string()),
            text: self.text,
            tool_calls: self.tools.finish(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            model,
        }
    }
}

/// Consume an OpenAI SSE body under the cancel + stall rules.
async fn read_sse_stream(
    provider: &str,
    mut res: reqwest::Response,
    on_chunk: &mut (dyn FnMut(&str) + Send),
    cancel: Option<&CancellationToken>,
    stall_ms: Option<u64>,
    model: String,
) -> anyhow::Result<GenerateResult> {
    let mut state = SseState::default();
    let mut lines = LineBuffer::default();
    // Stall watchdog: headers arrived but the body can still go silent (an
    // overloaded gateway holding the socket open). next_chunk aborts when no
    // bytes arrive for the request timeout, so the turn fails over instead of
    // spinning forever.
    while let Some(chunk) = http::next_chunk(provider, &mut res, cancel, stall_ms).await? {
        for line in lines.push(&chunk) {
            state.apply_line(provider, &line, on_chunk)?;
        }
    }
    if let Some(rest) = lines.finish() {
        state.apply_line(provider, &rest, on_chunk)?;
    }
    Ok(state.finish(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(model: &str) -> ProviderConfig {
        ProviderConfig::new(model, "http://localhost:8000")
    }

    #[test]
    fn api_base_appends_v1_once() {
        assert_eq!(default_api_base("http://x:1234"), "http://x:1234/v1");
        assert_eq!(default_api_base("http://x:1234/"), "http://x:1234/v1");
        assert_eq!(default_api_base("https://openrouter.ai/api/v1/"), "https://openrouter.ai/api/v1");
    }

    #[test]
    fn body_key_order_and_extras() {
        let mut c = cfg("llama3");
        c.top_k = Some(20);
        c.repeat_penalty = Some(1.05);
        c.presence_penalty = Some(0.0);
        let p = VllmProvider::new(c);
        let opts = ModelOptions { json: true, stop: Some(vec!["X".into()]), ..Default::default() };
        let body = p.body(&[Message::user("hi")], &opts, false, "llama3");
        let keys: Vec<&str> = body.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["model", "messages", "temperature", "top_p", "max_tokens", "top_k", "repetition_penalty", "stream", "stop", "response_format"]
        );
        assert_eq!(body["response_format"], json!({"type": "json_object"}));
        assert!(!body.contains_key("presence_penalty"));
        assert!(!body.contains_key("tools"));
    }

    #[test]
    fn hybrid_thinking_flag() {
        let p = LmStudioProvider::new(cfg("qwen3.5-9b"));
        let body = p.body(&[Message::user("hi")], &ModelOptions::default(), false, "qwen3.5-9b");
        assert_eq!(body["chat_template_kwargs"], json!({"enable_thinking": false}));
        let body = p.body(&[Message::user("hi")], &ModelOptions { think: Some(true), ..Default::default() }, false, "qwen3.5-9b");
        assert_eq!(body["chat_template_kwargs"], json!({"enable_thinking": true}));
        // Non-hybrid models never get the flag.
        let body = p.body(&[Message::user("hi")], &ModelOptions { think: Some(true), ..Default::default() }, false, "llama3");
        assert!(!body.contains_key("chat_template_kwargs"));
    }

    #[test]
    fn nvidia_extras() {
        let mut c = cfg("moonshotai/kimi-k2.6");
        c.reasoning_budget = Some(2048);
        let p = NvidiaNimProvider::new(c);
        let body = p.body(&[Message::system("rules"), Message::user("hi")], &ModelOptions::default(), false, "moonshotai/kimi-k2.6");
        assert_eq!(body["chat_template_kwargs"], json!({"thinking": false}));
        // Kimi: system folded into the user turn.
        assert_eq!(body["messages"], json!([{"role": "user", "content": "rules\n\n---\n\nhi"}]));
        let body = p.body(&[Message::user("hi")], &ModelOptions::default(), false, "nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(body["chat_template_kwargs"], json!({"enable_thinking": true}));
        assert_eq!(body["reasoning_budget"], json!(2048));
        let body = p.body(&[Message::user("hi")], &ModelOptions::default(), false, "meta/llama-3.3-70b-instruct");
        assert!(!body.contains_key("chat_template_kwargs"));
    }

    #[test]
    fn affordable_retry_rule() {
        let o = ModelOptions::default();
        assert_eq!(recover_affordable(402, "can only afford 3436", &o).and_then(|o| o.max_tokens), Some(3372));
        assert!(recover_affordable(402, "can only afford 300", &o).is_none());
        assert!(recover_affordable(400, "can only afford 3436", &o).is_none());
        assert!(recover_affordable(402, "insufficient credits", &o).is_none());
    }

    #[test]
    fn model_list_shapes() {
        let v = json!({"data": [{"id": "a"}, {"name": "b"}, {"x": 1}]});
        let names: Vec<String> = parse_model_list(&v).into_iter().map(|m| m.name).collect();
        assert_eq!(names, ["a", "b"]);
        let v = json!({"models": [{"name": "c"}]});
        assert_eq!(parse_model_list(&v)[0].name, "c");
    }
}
