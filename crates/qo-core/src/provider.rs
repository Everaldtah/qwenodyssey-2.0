//! The uniform interface every model backend implements.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::ProviderError;
use crate::tokens::estimate_tokens;
use crate::types::{GenerateResult, Message, ModelInfo, ModelOptions};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HealthStatus {
    pub ok: bool,
    pub detail: Option<String>,
}

/// A model backend. Implementations use interior mutability for the active
/// model so a shared `Arc<dyn Provider>` can be re-pointed by `/model`.
///
/// Errors are `anyhow::Error` wrapping a [`ProviderError`] wherever the
/// failure is a known kind (interrupt, timeout, HTTP status …), so the chat
/// layer can classify them with [`crate::error`].
#[async_trait]
pub trait Provider: Send + Sync {
    /// Backend id: "ollama", "lmstudio", "nvidia", "openrouter", "anthropic", "openai", "vllm", "llamacpp".
    fn name(&self) -> &str;

    /// The active model id.
    fn model(&self) -> String;

    /// Switch the active model for subsequent requests.
    fn set_model(&self, model: &str);

    /// Blocking (non-streaming) completion. Must honour `opts.cancel`.
    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult>;

    /// Streaming completion: `on_chunk` receives visible text deltas as they
    /// arrive; the full result (incl. assembled tool calls and exact usage when
    /// reported) is returned at the end. Must honour `opts.cancel` at every
    /// stage and abort a body that goes silent for the request timeout.
    async fn stream(
        &self,
        messages: &[Message],
        // Explicit HRTB: async-trait rewrites an elided `&str` inside Fn sugar
        // into a caller-chosen lifetime, which no impl could satisfy.
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult>;

    /// Token estimate for budgeting. Backends with a real tokenizer may override.
    fn count_tokens(&self, text: &str) -> usize {
        estimate_tokens(text)
    }

    /// Is the backend reachable at all (e.g. GET /models)?
    async fn health_check(&self) -> HealthStatus;

    /// Models the backend currently has available, if it can tell.
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Err(ProviderError::Unsupported(format!("{} cannot list models", self.name())).into())
    }

    /// Liveness check after a manual `/model` switch: resolve as soon as the
    /// backend answers a tiny streaming request with HTTP 200 (body dropped),
    /// error otherwise. Cancelling `cancel` returns `ProviderError::Interrupted`.
    /// Default: nothing to check.
    async fn probe(&self, cancel: &CancellationToken) -> anyhow::Result<()> {
        let _ = cancel;
        Ok(())
    }
}

pub type SharedProvider = Arc<dyn Provider>;
