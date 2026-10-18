//! Qwenodyssey 2.0 model backends (port of `src/providers/*.ts`).
//!
//! - [`openai`]: the shared OpenAI-compatible core and its backends — LM Studio,
//!   vLLM, llama.cpp, generic OpenAI endpoint, NVIDIA NIM, OpenRouter.
//! - [`ollama`]: Ollama's native `/api/chat` (NDJSON streaming, `/api/tags`).
//! - [`anthropic`]: Claude's native Messages API (OAuth bearer or API key).
//! - [`lmstudio_control`]: headless LM Studio control through the `lms` CLI.
//! - [`auth`]: the `ant` CLI OAuth token source for Anthropic.
//! - [`http`]-level reliability rules shared by every backend (request timeout
//!   covers connect + headers, stall watchdog on bodies, Esc honoured at every
//!   await) live in one private module.
//!
//! Every provider implements [`qo_core::Provider`]; the factory functions hand
//! them out as [`qo_core::SharedProvider`] (`Arc<dyn Provider>`), and the
//! active model is interior-mutable (`set_model`).

mod http;

pub mod anthropic;
pub mod auth;
pub mod factory;
pub mod lmstudio_control;
pub mod ollama;
pub mod openai;
pub mod profile;
pub mod provider_config;
pub mod wire;

pub use anthropic::{parse_content, to_anthropic_messages, AnthropicProvider};
pub use auth::{ant_print_credentials, ant_print_credentials_async, AntCliTokenSource};
pub use factory::{
    anthropic_auth_available, anthropic_auth_available_async, create_anthropic_provider, create_lmstudio_provider,
    create_lmstudio_provider_at, create_nvidia_provider, create_ollama_provider, create_openrouter_provider, create_provider,
    create_provider_for_ref, default_base_url, is_cloud_provider, resolve_anthropic_auth, resolve_anthropic_auth_async,
    resolve_nvidia_key, resolve_openrouter_key, tune_for_model, AnthropicAuth,
};
pub use lmstudio_control::{lms_bin, GpuMem, LmStudioControl, LmsModel, LoadedModel, SafeLoadOpts, SafeLoadResult};
pub use ollama::OllamaProvider;
pub use openai::{
    LlamaCppProvider, LmStudioProvider, NvidiaNimProvider, OpenAiBackend, OpenAiCompatibleEndpointProvider,
    OpenAiCompatibleProvider, OpenRouterProvider, VllmProvider,
};
pub use profile::ThinkMode;
pub use provider_config::{ProviderConfig, SharedTokenSource, TokenSource};
pub use wire::describe_api_error;
