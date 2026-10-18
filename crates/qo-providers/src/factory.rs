//! Provider factory (port of providers/index.ts): build the configured backend,
//! or a specific backend for a model picked at runtime (picker / fallback chain
//! / swarm), resolving keys from config or the environment so a secret never
//! has to live in a committed file.

use std::sync::Arc;

use qo_core::config::{Config, ProviderKind, MODEL_DEFAULTS};
use qo_core::model_profile::sampling_for_model;
use qo_core::{ModelKind, ModelRef, SharedProvider};

use crate::anthropic::AnthropicProvider;
use crate::auth::{ant_print_credentials, ant_print_credentials_async, AntCliTokenSource};
use crate::ollama::OllamaProvider;
use crate::openai::{
    LlamaCppProvider, LmStudioProvider, NvidiaNimProvider, OpenAiCompatibleEndpointProvider, OpenRouterProvider,
    VllmProvider,
};
use crate::provider_config::ProviderConfig;

/// Sensible default base URL per provider when the config leaves it blank.
pub fn default_base_url(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Ollama => "http://localhost:11434",
        ProviderKind::LmStudio => "http://localhost:1234",
        ProviderKind::OpenAi => "https://api.openai.com",
        ProviderKind::Vllm => "http://localhost:8000",
        ProviderKind::LlamaCpp => "http://localhost:8080",
        ProviderKind::Nvidia => "https://integrate.api.nvidia.com",
        ProviderKind::OpenRouter => "https://openrouter.ai/api",
        ProviderKind::Anthropic => "https://api.anthropic.com",
    }
}

/// Cloud backends (reached over the internet; a hard 0 temperature risks
/// repetition loops there, and the chat layer probes them after a switch).
pub fn is_cloud_provider(name: &str) -> bool {
    matches!(name, "nvidia" | "openai" | "openrouter" | "anthropic")
}

/// A non-empty environment variable (JS `process.env[x] || …` semantics).
fn env(name: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// First non-empty of an explicit value and a list of env var names.
fn first_of(explicit: &str, envs: &[&str]) -> String {
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    envs.iter().find_map(|e| env(e)).unwrap_or_default()
}

/// NVIDIA NIM key: explicit config first, then the configured env var, then
/// `NVIDIA_API_KEY`.
pub fn resolve_nvidia_key(config: &Config) -> String {
    let n = &config.nvidia;
    let env_name = if n.api_key_env.is_empty() { "NVIDIA_API_KEY" } else { n.api_key_env.as_str() };
    first_of(&n.api_key, &[env_name, "NVIDIA_API_KEY"])
}

/// OpenRouter key: explicit config first, then the configured env var, then
/// `OPENROUTER_API_KEY`.
pub fn resolve_openrouter_key(config: &Config) -> String {
    let o = &config.openrouter;
    let env_name = if o.api_key_env.is_empty() { "OPENROUTER_API_KEY" } else { o.api_key_env.as_str() };
    first_of(&o.api_key, &[env_name, "OPENROUTER_API_KEY"])
}

/// Resolved Anthropic credentials (either may be empty).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct AnthropicAuth {
    /// OAuth bearer (Claude subscription).
    pub auth_token: String,
    /// Console API key (pay-as-you-go).
    pub api_key: String,
}

impl std::fmt::Debug for AnthropicAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = |s: &str| if s.is_empty() { "" } else { "<redacted>" };
        f.debug_struct("AnthropicAuth").field("auth_token", &r(&self.auth_token)).field("api_key", &r(&self.api_key)).finish()
    }
}

/// The config/env part of the Anthropic credentials (never spawns anything).
fn anthropic_static_auth(config: &Config) -> AnthropicAuth {
    let a = &config.anthropic;
    let token_env = if a.auth_token_env.is_empty() { "ANTHROPIC_AUTH_TOKEN" } else { a.auth_token_env.as_str() };
    let key_env = if a.api_key_env.is_empty() { "ANTHROPIC_API_KEY" } else { a.api_key_env.as_str() };
    AnthropicAuth { auth_token: first_of(&a.auth_token, &[token_env]), api_key: first_of(&a.api_key, &[key_env]) }
}

/// Resolve Anthropic auth WITHOUT requiring a committed secret:
///  - OAuth bearer: explicit config token, then the env var, then the `ant`
///    CLI (auto-refreshing) when `use_cli_auth` is on;
///  - API key: explicit config, then the env var.
///
/// May run `ant` (cached for a minute, ≤ 15 s) and BLOCKS while it does; from
/// async code prefer [`resolve_anthropic_auth_async`].
pub fn resolve_anthropic_auth(config: &Config) -> AnthropicAuth {
    let mut auth = anthropic_static_auth(config);
    if auth.auth_token.is_empty() && config.anthropic.use_cli_auth {
        auth.auth_token = ant_print_credentials(false).unwrap_or_default();
    }
    auth
}

/// Async [`resolve_anthropic_auth`] (the `ant` child is killed if the future is dropped).
pub async fn resolve_anthropic_auth_async(config: &Config) -> AnthropicAuth {
    let mut auth = anthropic_static_auth(config);
    if auth.auth_token.is_empty() && config.anthropic.use_cli_auth {
        auth.auth_token = ant_print_credentials_async(false).await.unwrap_or_default();
    }
    auth
}

/// Is some Anthropic credential available (OAuth token or API key)? Only asks
/// the `ant` CLI when config/env have nothing (blocking; see
/// [`anthropic_auth_available_async`]).
pub fn anthropic_auth_available(config: &Config) -> bool {
    if !config.anthropic.enabled {
        return false;
    }
    let auth = anthropic_static_auth(config);
    if !auth.auth_token.is_empty() || !auth.api_key.is_empty() {
        return true;
    }
    config.anthropic.use_cli_auth && ant_print_credentials(false).is_some()
}

/// Async [`anthropic_auth_available`].
pub async fn anthropic_auth_available_async(config: &Config) -> bool {
    if !config.anthropic.enabled {
        return false;
    }
    let auth = anthropic_static_auth(config);
    if !auth.auth_token.is_empty() || !auth.api_key.is_empty() {
        return true;
    }
    config.anthropic.use_cli_auth && ant_print_credentials_async(false).await.is_some()
}

/// The `[model]` knobs every backend shares.
fn base_cfg(config: &Config, model: &str, base_url: String, api_key: String) -> ProviderConfig {
    let m = &config.model;
    ProviderConfig {
        api_key,
        temperature: m.temperature,
        top_p: m.top_p,
        max_tokens: m.max_tokens,
        context_tokens: Some(m.context_tokens).filter(|c| *c > 0),
        ..ProviderConfig::new(model, base_url)
    }
}

/// The local-runtime knobs (Ollama GPU policy, keep-alive).
fn with_runtime(config: &Config, cfg: ProviderConfig) -> ProviderConfig {
    let m = &config.model;
    ProviderConfig {
        gpu_layers: Some(m.gpu_layers),
        low_vram: m.low_vram,
        keep_alive: Some(m.keep_alive.clone()).filter(|k| !k.is_empty()),
        ..cfg
    }
}

/// Model-aware tuning for LOCAL backends: take the decoding parameters the
/// model family's authors recommend (see `qo_core::model_profile`) and apply
/// them wherever the user left `[model]` at its defaults. An explicitly
/// configured value always wins, and `model.auto_tune = false` disables the
/// whole mechanism.
///
/// This is what makes a 9B-class local model behave: Qwen3/3.5 gets its
/// anti-repetition sampling instead of degenerate greedy decoding,
/// qwen2.5-coder keeps the near-greedy settings that maximise tool adherence,
/// and gemma gets top_k 64 — without the user tuning anything by hand.
pub fn tune_for_model(config: &Config, model: &str, cfg: ProviderConfig) -> ProviderConfig {
    let m = &config.model;
    let d = &*MODEL_DEFAULTS;
    let out = ProviderConfig { think: m.think, ..cfg };
    if !m.auto_tune {
        return ProviderConfig {
            top_k: Some(m.top_k),
            repeat_penalty: Some(m.repeat_penalty),
            presence_penalty: Some(m.presence_penalty),
            ..out
        };
    }
    let s = sampling_for_model(model, m.think);
    ProviderConfig {
        temperature: if m.temperature == d.temperature { s.temperature } else { m.temperature },
        top_p: if m.top_p == d.top_p { s.top_p } else { m.top_p },
        top_k: Some(if m.top_k == d.top_k { s.top_k.unwrap_or(0) } else { m.top_k }),
        repeat_penalty: Some(if m.repeat_penalty == d.repeat_penalty {
            s.repeat_penalty.unwrap_or(0.0)
        } else {
            m.repeat_penalty
        }),
        presence_penalty: Some(if m.presence_penalty == d.presence_penalty {
            s.presence_penalty.unwrap_or(0.0)
        } else {
            m.presence_penalty
        }),
        ..out
    }
}

/// The configured primary backend (`[model].provider` / `[model].model`).
pub fn create_provider(config: &Config) -> SharedProvider {
    let m = &config.model;
    match m.provider {
        ProviderKind::Nvidia => return create_nvidia_provider(config, &m.model),
        ProviderKind::OpenRouter => return create_openrouter_provider(config, &m.model),
        ProviderKind::Anthropic => return create_anthropic_provider(config, &m.model),
        // LM Studio as the PRIMARY backend must honour the [lmstudio] section
        // (URL + API key), exactly like a runtime switch does. Before, it used
        // [model].base_url — which defaults to Ollama's port — so an
        // LM Studio-primary config died at startup with "Model backend not
        // reachable". An explicit non-Ollama [model].base_url still wins.
        ProviderKind::LmStudio => {
            let explicit = Some(m.base_url.as_str()).filter(|b| !b.is_empty() && *b != MODEL_DEFAULTS.base_url);
            return create_lmstudio_provider_at(config, &m.model, explicit);
        }
        _ => {}
    }
    let base_url =
        if m.base_url.is_empty() { default_base_url(m.provider).to_string() } else { m.base_url.clone() };
    let api_key = first_of(&m.api_key, &["QWENODYSSEY_API_KEY"]);
    let cfg = tune_for_model(config, &m.model, with_runtime(config, base_cfg(config, &m.model, base_url, api_key)));
    match m.provider {
        ProviderKind::OpenAi => Arc::new(OpenAiCompatibleEndpointProvider::new(cfg)),
        ProviderKind::Vllm => Arc::new(VllmProvider::new(cfg)),
        ProviderKind::LlamaCpp => Arc::new(LlamaCppProvider::new(cfg)),
        _ => Arc::new(OllamaProvider::new(cfg)),
    }
}

/// An Ollama provider for a specific tag, independent of the configured
/// primary backend (model picker / fallback while the primary is something else).
pub fn create_ollama_provider(config: &Config, model: &str) -> SharedProvider {
    let m = &config.model;
    let base_url = if !m.base_url.is_empty() && m.provider == ProviderKind::Ollama {
        m.base_url.clone()
    } else {
        default_base_url(ProviderKind::Ollama).to_string()
    };
    let cfg = with_runtime(config, base_cfg(config, model, base_url, String::new()));
    Arc::new(OllamaProvider::new(tune_for_model(config, model, cfg)))
}

/// An NVIDIA NIM provider for a specific hosted model (primary or fallback).
pub fn create_nvidia_provider(config: &Config, model: &str) -> SharedProvider {
    let n = &config.nvidia;
    let base_url =
        if n.base_url.is_empty() { default_base_url(ProviderKind::Nvidia).to_string() } else { n.base_url.clone() };
    let cfg = ProviderConfig {
        disable_thinking: Some(n.disable_thinking),
        request_timeout_ms: Some(n.request_timeout_ms),
        nemotron_thinking: Some(n.nemotron_thinking),
        reasoning_budget: Some(n.reasoning_budget),
        ..base_cfg(config, model, base_url, resolve_nvidia_key(config))
    };
    Arc::new(NvidiaNimProvider::new(cfg))
}

/// An OpenRouter provider for a specific hosted model (primary or fallback).
pub fn create_openrouter_provider(config: &Config, model: &str) -> SharedProvider {
    let o = &config.openrouter;
    let base_url = if o.base_url.is_empty() {
        default_base_url(ProviderKind::OpenRouter).to_string()
    } else {
        o.base_url.clone()
    };
    let cfg = ProviderConfig {
        request_timeout_ms: Some(o.request_timeout_ms),
        ..base_cfg(config, model, base_url, resolve_openrouter_key(config))
    };
    Arc::new(OpenRouterProvider::new(cfg))
}

/// An Anthropic (Claude) provider for a specific model (primary or fallback).
/// Auth is OAuth-first (Claude subscription via the `ant` CLI) with API-key
/// fallback. Unlike the TS factory this never spawns `ant` here: with no
/// configured token the provider asks the CLI on its first request (so the
/// child is cancellable), and re-mints the token on a 401.
pub fn create_anthropic_provider(config: &Config, model: &str) -> SharedProvider {
    let a = &config.anthropic;
    let auth = anthropic_static_auth(config);
    let base_url = if a.base_url.is_empty() {
        default_base_url(ProviderKind::Anthropic).to_string()
    } else {
        a.base_url.clone()
    };
    let cfg = ProviderConfig {
        auth_token: Some(auth.auth_token).filter(|t| !t.is_empty()),
        // Only the CLI source can refresh; a config/env token can't be re-minted here.
        refresh_auth: if a.use_cli_auth { Some(Arc::new(AntCliTokenSource)) } else { None },
        request_timeout_ms: Some(a.request_timeout_ms),
        ..base_cfg(config, model, base_url, auth.api_key)
    };
    Arc::new(AnthropicProvider::new(cfg))
}

/// An LM Studio provider for a specific model id (runtime switch / fallback),
/// at `[lmstudio].base_url`.
pub fn create_lmstudio_provider(config: &Config, model: &str) -> SharedProvider {
    create_lmstudio_provider_at(config, model, None)
}

/// [`create_lmstudio_provider`] with an explicit server URL (`Some` overrides
/// `[lmstudio].base_url`; the TS optional third parameter).
pub fn create_lmstudio_provider_at(config: &Config, model: &str, base_url: Option<&str>) -> SharedProvider {
    let l = &config.lmstudio;
    let base_url = match base_url.filter(|b| !b.is_empty()) {
        Some(b) => b.to_string(),
        None if !l.base_url.is_empty() => l.base_url.clone(),
        None => default_base_url(ProviderKind::LmStudio).to_string(),
    };
    let api_key = if l.api_key.is_empty() { config.model.api_key.clone() } else { l.api_key.clone() };
    let cfg = base_cfg(config, model, base_url, api_key);
    Arc::new(LmStudioProvider::new(tune_for_model(config, model, cfg)))
}

/// The provider for a fallback/picker ref (`nvidia:<model>`, `lmstudio:<key>`,
/// `openrouter:<model>`, `anthropic:<model>`, else a bare Ollama tag).
pub fn create_provider_for_ref(config: &Config, r: &ModelRef) -> SharedProvider {
    match r.kind {
        ModelKind::Ollama => create_ollama_provider(config, &r.model),
        ModelKind::LmStudio => create_lmstudio_provider(config, &r.model),
        ModelKind::Nvidia => create_nvidia_provider(config, &r.model),
        ModelKind::OpenRouter => create_openrouter_provider(config, &r.model),
        ModelKind::Anthropic => create_anthropic_provider(config, &r.model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qo_core::config::{default_config, ThinkMode};

    #[test]
    fn tune_applies_family_sampling_only_at_defaults() {
        let config = default_config();
        let cfg = tune_for_model(&config, "qwen3.5:9b", ProviderConfig::new("qwen3.5:9b", "x"));
        // Qwen3 non-thinking (auto) profile.
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.8);
        assert_eq!(cfg.top_k, Some(20));
        assert_eq!(cfg.presence_penalty, Some(1.5));
        assert_eq!(cfg.repeat_penalty, Some(1.05));

        let mut config = default_config();
        config.model.temperature = 0.1;
        config.model.top_k = 5;
        let cfg = tune_for_model(&config, "qwen3.5:9b", ProviderConfig::new("qwen3.5:9b", "x"));
        assert_eq!(cfg.temperature, 0.1);
        assert_eq!(cfg.top_k, Some(5));

        let mut config = default_config();
        config.model.auto_tune = false;
        config.model.think = ThinkMode::Always;
        let base = ProviderConfig { temperature: 0.2, ..ProviderConfig::new("qwen3.5:9b", "x") };
        let cfg = tune_for_model(&config, "qwen3.5:9b", base);
        assert_eq!(cfg.temperature, 0.2);
        assert_eq!(cfg.top_k, Some(0));
        assert_eq!(cfg.think, ThinkMode::Always);
    }

    #[test]
    fn factory_picks_backend_and_urls() {
        let mut config = default_config();
        let p = create_provider(&config);
        assert_eq!(p.name(), "ollama");
        assert_eq!(p.model(), "qwen3.5:9b");

        config.model.provider = ProviderKind::LmStudio;
        config.model.model = "qwen3.5-9b".into();
        let p = create_provider(&config);
        assert_eq!(p.name(), "lmstudio");

        for (kind, name) in [
            (ProviderKind::Nvidia, "nvidia"),
            (ProviderKind::OpenRouter, "openrouter"),
            (ProviderKind::Anthropic, "anthropic"),
            (ProviderKind::OpenAi, "openai"),
            (ProviderKind::Vllm, "vllm"),
            (ProviderKind::LlamaCpp, "llamacpp"),
        ] {
            config.model.provider = kind;
            assert_eq!(create_provider(&config).name(), name);
        }
        let r = ModelRef::parse("nvidia:moonshotai/kimi-k2.6");
        let p = create_provider_for_ref(&config, &r);
        assert_eq!((p.name(), p.model().as_str()), ("nvidia", "moonshotai/kimi-k2.6"));
        p.set_model("nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(p.model(), "nvidia/nemotron-3-ultra-550b-a55b");
    }

    #[test]
    fn key_resolution_prefers_config_then_env() {
        let mut config = default_config();
        config.nvidia.api_key = "cfg-key".into();
        assert_eq!(resolve_nvidia_key(&config), "cfg-key");
        // A custom env var name that is certainly unset falls through to the
        // standard one (whatever the machine has) — never panics.
        config.nvidia.api_key.clear();
        config.nvidia.api_key_env = "QO_TEST_UNSET_VAR_7f3a".into();
        let _ = resolve_nvidia_key(&config);
        config.openrouter.api_key = "or-key".into();
        assert_eq!(resolve_openrouter_key(&config), "or-key");
    }

    #[test]
    fn anthropic_availability_short_circuits_on_static_auth() {
        let mut config = default_config();
        config.anthropic.api_key = "sk-ant-api-key".into();
        config.anthropic.use_cli_auth = false;
        assert!(anthropic_auth_available(&config));
        config.anthropic.enabled = false;
        assert!(!anthropic_auth_available(&config));
    }

    #[test]
    fn cloud_names() {
        assert!(is_cloud_provider("nvidia") && is_cloud_provider("anthropic") && is_cloud_provider("openai"));
        assert!(!is_cloud_provider("ollama") && !is_cloud_provider("lmstudio"));
    }
}
