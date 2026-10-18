//! Per-provider settings (port of `ProviderConfig` in base.ts).
//!
//! The factory ([`crate::factory`]) builds one of these from the user's
//! `Config`; tests and embedders can also build one directly with
//! [`ProviderConfig::new`] and struct-update syntax.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::profile::ThinkMode;

/// Source of a short-lived OAuth bearer token (Anthropic subscription auth).
///
/// The TypeScript version had a synchronous `refreshAuth` thunk around
/// `execa.sync("ant", …)`. Here it is async so the `ant` child process can be
/// cancelled (the provider races it against the request's cancel token and the
/// child is killed when the future is dropped).
#[async_trait]
pub trait TokenSource: Send + Sync {
    /// The current token (a cached one is fine), or — when `force` is set — a
    /// freshly minted one (used after an HTTP 401). `None` when unavailable.
    async fn token(&self, force: bool) -> Option<String>;
}

pub type SharedTokenSource = Arc<dyn TokenSource>;

#[derive(Clone)]
pub struct ProviderConfig {
    /// Initial model id (the live one is kept by the provider and changed with `set_model`).
    pub model: String,
    pub base_url: String,
    /// Bearer key (OpenAI-compatible) / `x-api-key` (Anthropic). Empty = none.
    pub api_key: String,
    pub temperature: f64,
    pub top_p: f64,
    /// Optional small-model decoding knobs; omitted from the wire when unset/0.
    pub top_k: Option<u32>,
    pub repeat_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    /// Chain-of-thought policy for thinking-capable models (config `model.think`).
    pub think: ThinkMode,
    pub max_tokens: u32,
    /// Ollama `num_ctx` (defaults to 8192 on the wire when unset).
    pub context_tokens: Option<u32>,
    /// Ollama `num_gpu`; `None` or negative = auto offload.
    pub gpu_layers: Option<i32>,
    pub low_vram: bool,
    /// Ollama `keep_alive` (e.g. "30m"); `None`/empty = server default.
    pub keep_alive: Option<String>,
    /// NIM: send `chat_template_kwargs.thinking=false` to thinking models.
    /// `None` behaves like `Some(true)` (the TS `!== false` check).
    pub disable_thinking: Option<bool>,
    /// Abort a request after this many ms without response headers, and a
    /// streamed body that goes silent for this long (`None`/0 = no timeout).
    pub request_timeout_ms: Option<u64>,
    /// Nemotron reasoning models: `enable_thinking` toggle (`None` = on).
    pub nemotron_thinking: Option<bool>,
    /// Nemotron `reasoning_budget`; sent only when > 0.
    pub reasoning_budget: Option<u32>,
    /// Anthropic OAuth bearer token (Claude subscription); preferred over `api_key`.
    pub auth_token: Option<String>,
    /// Mints / re-mints the OAuth token (e.g. `ant auth print-credentials`).
    /// When `auth_token` is empty the provider asks it for a token on first use;
    /// after a 401 it asks for a fresh one and retries once.
    pub refresh_auth: Option<SharedTokenSource>,
}

impl ProviderConfig {
    /// A config with the built-in `[model]` defaults (temperature 0.2, top_p 0.9,
    /// max_tokens 8192, no timeout) for `model` at `base_url`.
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: base_url.into(),
            api_key: String::new(),
            temperature: 0.2,
            top_p: 0.9,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            think: ThinkMode::default(),
            max_tokens: 8192,
            context_tokens: None,
            gpu_layers: None,
            low_vram: false,
            keep_alive: None,
            disable_thinking: None,
            request_timeout_ms: None,
            nemotron_thinking: None,
            reasoning_budget: None,
            auth_token: None,
            refresh_auth: None,
        }
    }

    /// The request timeout, if one is configured (0 = none).
    pub fn timeout_ms(&self) -> Option<u64> {
        self.request_timeout_ms.filter(|ms| *ms > 0)
    }
}

impl fmt::Debug for ProviderConfig {
    /// Secrets are redacted so a config can be logged safely.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redact = |s: &str| if s.is_empty() { "" } else { "<redacted>" };
        f.debug_struct("ProviderConfig")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &redact(&self.api_key))
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("presence_penalty", &self.presence_penalty)
            .field("think", &self.think)
            .field("max_tokens", &self.max_tokens)
            .field("context_tokens", &self.context_tokens)
            .field("gpu_layers", &self.gpu_layers)
            .field("low_vram", &self.low_vram)
            .field("keep_alive", &self.keep_alive)
            .field("disable_thinking", &self.disable_thinking)
            .field("request_timeout_ms", &self.request_timeout_ms)
            .field("nemotron_thinking", &self.nemotron_thinking)
            .field("reasoning_budget", &self.reasoning_budget)
            .field("auth_token", &self.auth_token.as_deref().map(redact))
            .field("refresh_auth", &self.refresh_auth.is_some())
            .finish()
    }
}
