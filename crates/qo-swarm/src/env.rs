//! The swarm's view of the configuration: plain settings structs plus a
//! provider factory. Everything in this crate codes against [`SwarmEnv`]
//! instead of `qo_core::config::Config` directly, so
//!
//!  - the only file that knows the real config/provider APIs is `compat.rs`
//!    ([`SwarmEnv::from_config`]), and an upstream rename is a one-file fix;
//!  - tests can build a roster from canned settings and fake providers without
//!    touching env vars, config files or the network.

use std::fmt;
use std::sync::Arc;

use qo_core::SharedProvider;

use crate::exec::ExecMode;
use crate::swarm::WorkerKind;

/// `[swarm]` settings (see config.ts `SwarmConfig`; defaults match it).
#[derive(Clone, Debug, PartialEq)]
pub struct SwarmSettings {
    pub enabled: bool,
    /// Hard cap on how many models run concurrently (protects rate limits / cost).
    pub max_workers: usize,
    /// Also enlist local refs (bare Ollama tags / lmstudio:*) from the fallback
    /// list, not just cloud models. Off by default so the swarm stays frontier-only.
    pub include_local: bool,
    /// Who merges ensemble answers: "auto" (best frontier worker), "primary"
    /// (the active chat model) or an explicit ref like "openrouter:moonshotai/kimi-k2.6".
    pub synthesizer: String,
    /// Per-worker (and synthesis) output token budget.
    pub max_tokens: u32,
    /// COORDINATED mode by default (CLI `swarm` + the agent_swarm tool): a lead
    /// splits the task into dependency-aware subtasks that share a blackboard.
    pub coordinate: bool,
    /// Show the live split-pane TUI when the CLI `swarm` command runs in a TTY.
    pub live: bool,
    /// Where swarm agents may RUN COMMANDS ("off" = text-only agents, the safe default).
    pub exec: ExecMode,
    /// Max run_shell tool steps per agent subtask.
    pub exec_max_steps: u32,
    /// Per-command timeout (seconds).
    pub exec_timeout_s: u64,
    /// Health-check each agent's API key/endpoint before the run (drops broken ones).
    pub preflight: bool,
}

impl Default for SwarmSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_workers: 4,
            include_local: false,
            synthesizer: "primary".to_string(),
            max_tokens: 1500,
            coordinate: true,
            live: true,
            exec: ExecMode::Off,
            exec_max_steps: 5,
            exec_timeout_s: 120,
            preflight: true,
        }
    }
}

/// `[daytona]` settings (see config.ts `DaytonaConfig`; defaults match it).
/// The key is a SECRET: normally `api_key` is blank and the env var is used
/// (and `Debug` redacts it).
#[derive(Clone, PartialEq)]
pub struct DaytonaSettings {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    /// Env var holding the key when `api_key` is blank (default `DAYTONA_API_KEY`).
    pub api_key_env: String,
    /// Optional snapshot/image name for new sandboxes.
    pub snapshot: String,
    /// Optional target region for new sandboxes.
    pub target: String,
    /// Lifecycle backstops (minutes): auto-stop when idle, auto-delete after.
    pub auto_stop_minutes: u64,
    pub auto_delete_minutes: u64,
}

impl Default for DaytonaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://app.daytona.io/api".to_string(),
            api_key: String::new(),
            api_key_env: "DAYTONA_API_KEY".to_string(),
            snapshot: String::new(),
            target: String::new(),
            auto_stop_minutes: 15,
            auto_delete_minutes: 120,
        }
    }
}

impl fmt::Debug for DaytonaSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaytonaSettings")
            .field("enabled", &self.enabled)
            .field("base_url", &self.base_url)
            .field("api_key", &if self.api_key.is_empty() { "" } else { "<redacted>" })
            .field("api_key_env", &self.api_key_env)
            .field("snapshot", &self.snapshot)
            .field("target", &self.target)
            .field("auto_stop_minutes", &self.auto_stop_minutes)
            .field("auto_delete_minutes", &self.auto_delete_minutes)
            .finish()
    }
}

/// Creates the model backends the swarm runs on. The real implementation wraps
/// the qo-providers factory (see `compat.rs`); tests supply canned providers.
pub trait ProviderFactory: Send + Sync {
    /// The configured primary chat provider (`create_provider(config)`).
    fn primary(&self) -> SharedProvider;
    /// A provider for `model` on backend `kind`, using that backend's configured key/URL.
    fn create(&self, kind: WorkerKind, model: &str) -> SharedProvider;
}

/// Everything the swarm needs to know about the user's setup.
#[derive(Clone)]
pub struct SwarmEnv {
    pub swarm: SwarmSettings,
    pub daytona: DaytonaSettings,
    /// `[model].provider` — the primary backend id ("nvidia", "openrouter", "ollama", …).
    pub primary_backend: String,
    /// `[model].model` — the primary model id.
    pub primary_model: String,
    /// `[model].fallback_models` — refs like `nvidia:<model>`, `openrouter:<model>`,
    /// `lmstudio:<model>` or a bare Ollama tag.
    pub fallback_models: Vec<String>,
    /// `[tools].allow_commands` — auto-approved by the shell safety gate.
    pub allow_commands: Vec<String>,
    /// `[tools].deny_commands` — hard-blocked by the shell safety gate.
    pub deny_commands: Vec<String>,
    /// NVIDIA NIM is enabled and a key resolves (config or env).
    pub have_nvidia: bool,
    /// OpenRouter is enabled and a key resolves (config or env).
    pub have_openrouter: bool,
    pub factory: Arc<dyn ProviderFactory>,
}

impl SwarmEnv {
    /// Default settings around `factory`, with no cloud keys and no fallback
    /// models (tests / embedding). Adjust the public fields as needed.
    pub fn with_factory(factory: Arc<dyn ProviderFactory>) -> Self {
        Self {
            swarm: SwarmSettings::default(),
            daytona: DaytonaSettings::default(),
            primary_backend: "ollama".to_string(),
            primary_model: String::new(),
            fallback_models: Vec::new(),
            allow_commands: Vec::new(),
            deny_commands: Vec::new(),
            have_nvidia: false,
            have_openrouter: false,
            factory,
        }
    }
}

impl fmt::Debug for SwarmEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SwarmEnv")
            .field("swarm", &self.swarm)
            .field("daytona", &self.daytona)
            .field("primary_backend", &self.primary_backend)
            .field("primary_model", &self.primary_model)
            .field("fallback_models", &self.fallback_models)
            .field("have_nvidia", &self.have_nvidia)
            .field("have_openrouter", &self.have_openrouter)
            .finish_non_exhaustive()
    }
}
