//! The ONE place this crate touches other crates' evolving APIs (config fields,
//! the provider factory, prompts, model profiles, the shell safety gate). Every
//! other module reads plain settings from [`SwarmEnv`], so an upstream rename is
//! a one-file fix.

use std::sync::Arc;

use qo_core::config::{Config, SwarmExec};
use qo_core::SharedProvider;

use crate::env::{DaytonaSettings, ProviderFactory, SwarmEnv, SwarmSettings};
use crate::exec::ExecMode;
use crate::swarm::WorkerKind;

/* ── config → settings ── */

fn exec_mode(e: SwarmExec) -> ExecMode {
    match e {
        SwarmExec::Off => ExecMode::Off,
        SwarmExec::Auto => ExecMode::Auto,
        SwarmExec::Bare => ExecMode::Bare,
        SwarmExec::Daytona => ExecMode::Daytona,
    }
}

/// `[swarm]` → [`SwarmSettings`].
pub fn swarm_settings(config: &Config) -> SwarmSettings {
    let s = &config.swarm;
    SwarmSettings {
        enabled: s.enabled,
        max_workers: s.max_workers as usize,
        include_local: s.include_local,
        synthesizer: s.synthesizer.clone(),
        max_tokens: s.max_tokens,
        coordinate: s.coordinate,
        live: s.live,
        exec: exec_mode(s.exec),
        exec_max_steps: s.exec_max_steps,
        exec_timeout_s: s.exec_timeout_s,
        preflight: s.preflight,
    }
}

/// `[daytona]` → [`DaytonaSettings`].
pub fn daytona_settings(config: &Config) -> DaytonaSettings {
    let d = &config.daytona;
    DaytonaSettings {
        enabled: d.enabled,
        base_url: d.base_url.clone(),
        api_key: d.api_key.clone(),
        api_key_env: d.api_key_env.clone(),
        snapshot: d.snapshot.clone(),
        target: d.target.clone(),
        auto_stop_minutes: d.auto_stop_minutes,
        auto_delete_minutes: d.auto_delete_minutes,
    }
}

impl SwarmEnv {
    /// Everything the swarm needs from the loaded config, with the real
    /// qo-providers factory behind [`ProviderFactory`].
    pub fn from_config(config: &Config) -> Self {
        SwarmEnv {
            swarm: swarm_settings(config),
            daytona: daytona_settings(config),
            primary_backend: config.model.provider.as_str().to_string(),
            primary_model: config.model.model.clone(),
            fallback_models: config.model.fallback_models.clone(),
            allow_commands: config.tools.allow_commands.clone(),
            deny_commands: config.tools.deny_commands.clone(),
            have_nvidia: config.nvidia.enabled && !qo_providers::resolve_nvidia_key(config).is_empty(),
            have_openrouter: config.openrouter.enabled && !qo_providers::resolve_openrouter_key(config).is_empty(),
            factory: Arc::new(ConfigProviderFactory { config: Arc::new(config.clone()) }),
        }
    }
}

/* ── provider factory ── */

/// [`ProviderFactory`] over the qo-providers factory functions.
pub struct ConfigProviderFactory {
    config: Arc<Config>,
}

impl ProviderFactory for ConfigProviderFactory {
    fn primary(&self) -> SharedProvider {
        qo_providers::create_provider(&self.config)
    }

    fn create(&self, kind: WorkerKind, model: &str) -> SharedProvider {
        let c = &self.config;
        match kind {
            WorkerKind::Nvidia => qo_providers::create_nvidia_provider(c, model),
            WorkerKind::OpenRouter => qo_providers::create_openrouter_provider(c, model),
            WorkerKind::LmStudio => qo_providers::create_lmstudio_provider(c, model),
            WorkerKind::Ollama => qo_providers::create_ollama_provider(c, model),
        }
    }
}

/* ── prompts / model profiles ── */

/// The shipped `system.md` harness prompt (used by the plain ensemble/divide path).
pub fn system_prompt() -> String {
    qo_core::prompts::load_prompt("system").unwrap_or_default()
}

/// Reasoning/thinking models need heat — at temp 0 they loop. One classifier for
/// the whole harness: the family profile in `qo_core::model_profile`.
pub fn is_reasoning_model(model: &str) -> bool {
    qo_core::model_profile::is_reasoning_model(model)
}

/* ── shell safety gate (shared with run_shell) ── */

/// Safety class of a shell command (mirrors `qo_tools::CommandClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandClass {
    Blocked,
    Destructive,
    Safe,
}

/// The run_shell gate: deny-list → built-in hard-block → allow-list → destructive.
pub fn classify_command(cmd: &str, allow: &[String], deny: &[String]) -> CommandClass {
    let opts = qo_tools::shell_tools::ClassifyOptions { allow: allow.to_vec(), deny: deny.to_vec() };
    match qo_tools::shell_tools::classify_command(cmd, Some(&opts)) {
        qo_tools::shell_tools::CommandClass::Blocked => CommandClass::Blocked,
        qo_tools::shell_tools::CommandClass::Destructive => CommandClass::Destructive,
        qo_tools::shell_tools::CommandClass::Safe => CommandClass::Safe,
    }
}

/// PowerShell preamble (`-UseBasicParsing` defaults, no progress bar).
pub fn ps_preamble() -> &'static str {
    qo_tools::shell_tools::PS_PREAMBLE
}
