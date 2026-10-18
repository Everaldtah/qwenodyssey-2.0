//! Configuration: load/save `.qwenodyssey/config.toml` with defaults + validation
//! (ported from `src/core/config.ts`).
//!
//! Every section and field of the TS zod schema is here with the SAME TOML key
//! (snake_case) and the SAME default. Every struct is `#[serde(default)]`, so a
//! partial file works, and unknown keys are ignored (zod strips them), so an
//! older/newer config never fails to load just because of an extra key.
//!
//! Layering ([`load_config`]): built-in defaults ← user-level
//! `~/.qwenodyssey/config.toml` ← project `<cwd>/.qwenodyssey/config.toml`,
//! deep-merged at the TOML-table level exactly like the TS `deepMerge` (nested
//! tables merge, arrays/scalars replace).
//!
//! Numbers: the TS schema uses `z.number()` everywhere. Here integer-valued knobs
//! are integers (`u32` counts/tokens, `u64` times, `i32` for `gpu_layers`) and
//! the rest `f64`. Integer fields also accept an integral float (`8192.0`) like
//! the TS did, and `f64` fields serialize integral values without a `.0`
//! (`repeat_penalty = 0`), so `config list`/`save_config` output matches the TS.

use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ordered map used for `[mcp.servers.*]` and server `env` tables, so servers
/// keep the order they were written in (the TS objects preserve insertion order).
pub use indexmap::IndexMap;

/// Name of the per-project (and per-user) workspace directory.
pub const WORKSPACE_DIR: &str = ".qwenodyssey";

/* ───────────────────────────── number helpers ───────────────────────────── */

/// Serde helpers that make numbers behave like the TS `z.number()` fields.
mod num {
    use std::fmt;
    use std::marker::PhantomData;

    use serde::de::{self, Deserializer, Visitor};
    use serde::Serializer;

    /// Integer field: accepts any integer in range, or an integral float.
    pub(super) fn int<'de, D, T>(d: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: TryFrom<i64> + TryFrom<u64>,
    {
        struct V<T>(PhantomData<T>);
        impl<'de, T: TryFrom<i64> + TryFrom<u64>> Visitor<'de> for V<T> {
            type Value = T;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an integer")
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<T, E> {
                T::try_from(v).map_err(|_| E::custom(format!("number {v} is out of range")))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<T, E> {
                T::try_from(v).map_err(|_| E::custom(format!("number {v} is out of range")))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<T, E> {
                if !v.is_finite() || v.fract() != 0.0 {
                    return Err(E::custom(format!("expected an integer, got {v}")));
                }
                if v < 0.0 {
                    self.visit_i64(v as i64)
                } else {
                    self.visit_u64(v as u64)
                }
            }
        }
        d.deserialize_any(V(PhantomData))
    }

    /// Float field: write integral values as integers (`0`, `20`) the way a JS
    /// number serializes, so JSON/TOML output matches the TS byte-for-byte.
    pub(super) fn float<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        const SAFE: f64 = 9_007_199_254_740_991.0; // Number.MAX_SAFE_INTEGER
        if v.is_finite() && v.fract() == 0.0 && v.abs() <= SAFE {
            s.serialize_i64(*v as i64)
        } else {
            s.serialize_f64(*v)
        }
    }
}

/* ───────────────────────────── enum helper ───────────────────────────── */

/// A closed set of lowercase string values (a zod `z.enum([...])`), with
/// `as_str`, `Display`, `FromStr` and serde as the lowercase name.
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident { $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)? }
        default = $default:ident
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $( $(#[$vmeta])* #[serde(rename = $text)] $variant, )+
        }

        impl $name {
            /// Every accepted value, in schema order.
            pub const ALL: &'static [$name] = &[ $( $name::$variant ),+ ];

            /// The TOML/JSON spelling.
            pub fn as_str(&self) -> &'static str {
                match self { $( $name::$variant => $text, )+ }
            }
        }

        impl Default for $name {
            fn default() -> Self { $name::$default }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = anyhow::Error;
            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $( $text => Ok($name::$variant), )+
                    other => Err(anyhow!(
                        "Invalid {} \"{}\": expected one of {}",
                        stringify!($name),
                        other,
                        [$( $text ),+].join(" | ")
                    )),
                }
            }
        }
    };
}

string_enum! {
    /// Model backend (`model.provider`).
    pub enum ProviderKind {
        Ollama => "ollama",
        LmStudio => "lmstudio",
        OpenAi => "openai",
        Vllm => "vllm",
        LlamaCpp => "llamacpp",
        Nvidia => "nvidia",
        OpenRouter => "openrouter",
        Anthropic => "anthropic",
    }
    default = Ollama
}

string_enum! {
    /// Chain-of-thought policy (`model.think`).
    pub enum ThinkMode {
        /// Per family — hybrids (Qwen3/3.5) run with thinking OFF for fast,
        /// tool-shaped coding/shell turns; pure reasoners (R1, QwQ) think.
        Auto => "auto",
        /// Always deliberate (slower, better on hard one-shot problems).
        Always => "always",
        /// Disable thinking wherever the model supports being turned off.
        Never => "never",
    }
    default = Auto
}

string_enum! {
    /// Edit protocol (`agent.edit_protocol`).
    pub enum EditProtocol {
        /// SEARCH/REPLACE blocks (reliable for small models).
        Blocks => "blocks",
        /// The legacy JSON-diff coder.
        Json => "json",
    }
    default = Blocks
}

string_enum! {
    /// Search backend (`web.provider`).
    pub enum WebProvider {
        DuckDuckGo => "duckduckgo",
        Tavily => "tavily",
        Brave => "brave",
        Searxng => "searxng",
    }
    default = DuckDuckGo
}

string_enum! {
    /// Where swarm agents may run commands (`swarm.exec`).
    pub enum SwarmExec {
        /// Text-only agents (default — safe for cloners).
        Off => "off",
        /// Bare metal for simple plans, a Daytona sandbox for complex plans
        /// (when `[daytona]` is configured).
        Auto => "auto",
        /// Forced bare metal.
        Bare => "bare",
        /// Forced Daytona sandbox.
        Daytona => "daytona",
    }
    default = Off
}

string_enum! {
    /// Backend that analyzes a captured webcam frame (`vision.provider`).
    pub enum VisionProvider {
        Nvidia => "nvidia",
        OpenAi => "openai",
        Ollama => "ollama",
    }
    default = Nvidia
}

string_enum! {
    /// Backend for omni mode (`omni.provider`).
    pub enum OmniProvider {
        Nvidia => "nvidia",
        OpenAi => "openai",
    }
    default = Nvidia
}

string_enum! {
    /// Text-to-speech engine (`tts.engine`).
    pub enum TtsEngine {
        /// Local neural TTS.
        Piper => "piper",
        /// Built-in Windows voice.
        Sapi => "sapi",
        /// Text only.
        None => "none",
    }
    default = Piper
}

/* ───────────────────────────── [model] ───────────────────────────── */

/// `[model]` — the active model and its decoding/runtime knobs (TS `ModelSettings`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSettings {
    pub provider: ProviderKind,
    pub model: String,
    /// Ordered fallback chain: tried in turn if `model` isn't installed at launch,
    /// or when a request fails because the active model is unavailable. The first
    /// installed/working one wins. Empty list disables fallback.
    /// Refs are bare Ollama tags, `lmstudio:<key>`, `nvidia:<model>` (cloud NIM), or
    /// `openrouter:<model>` (cloud OpenRouter) — cloud refs are only used when the
    /// matching API key is configured; skipped otherwise.
    pub fallback_models: Vec<String>,
    pub base_url: String,
    pub api_key: String,
    #[serde(serialize_with = "num::float")]
    pub temperature: f64,
    #[serde(serialize_with = "num::float")]
    pub top_p: f64,
    /// Sampling knobs that matter most for SMALL local models. Left at 0 they are
    /// omitted from the request and the model-profile tuner fills in the values the
    /// model's authors recommend (see `auto_tune` and `model_profile`). Set any of
    /// them explicitly to pin it and opt that knob out of auto-tuning.
    #[serde(deserialize_with = "num::int")]
    pub top_k: u32,
    #[serde(serialize_with = "num::float")]
    pub repeat_penalty: f64,
    #[serde(serialize_with = "num::float")]
    pub presence_penalty: f64,
    /// Model-aware tuning: detect the model family (qwen3/3.5, qwen2.5-coder,
    /// deepseek-r1, gemma…) and apply its recommended decoding parameters wherever
    /// the settings above are still at their defaults. Turn off to use the literal
    /// config values for every model.
    pub auto_tune: bool,
    /// Chain-of-thought policy for models that HAVE one (see [`ThinkMode`]).
    pub think: ThinkMode,
    /// GPU acceleration (Ollama). -1 = AUTO: load as many layers onto the GPU as
    /// fit in VRAM and run the overflow on CPU/RAM — i.e. automatic offload when
    /// the model + context/KV cache is too large for VRAM. Set a positive number
    /// to force a layer count; 0 = CPU only.
    #[serde(deserialize_with = "num::int")]
    pub gpu_layers: i32,
    /// Shrink GPU working buffers so bigger models/contexts still fit (Ollama low_vram).
    pub low_vram: bool,
    /// How long Ollama keeps the model resident in VRAM between turns.
    pub keep_alive: String,
    /// Generous output budget so deep reasoning / long `<think>` chains aren't cut off.
    #[serde(deserialize_with = "num::int")]
    pub max_tokens: u32,
    /// Ollama context window (num_ctx). 16384 holds far more code/docs than the
    /// 4096 Ollama default; raise toward the model's max (qwen2.5 32K, deepseek-r1
    /// & gemma-12B ~128K) if you have the RAM/VRAM for the larger KV cache.
    #[serde(deserialize_with = "num::int")]
    pub context_tokens: u32,
}

/// Alias matching the TS schema constant's name.
pub type ModelConfig = ModelSettings;

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            provider: ProviderKind::Ollama,
            model: "qwen3.5:9b".into(),
            fallback_models: vec![
                "nvidia:deepseek-ai/deepseek-v4-pro".into(),
                "nvidia:moonshotai/kimi-k2.6".into(),
                "qwen2.5-coder:7b".into(),
                "deepseek-r1:7b".into(),
                "igorls/gemma-4-12B-it-heretic-GGUF".into(),
            ],
            base_url: "http://localhost:11434".into(),
            api_key: String::new(),
            temperature: 0.2,
            top_p: 0.9,
            top_k: 0,
            repeat_penalty: 0.0,
            presence_penalty: 0.0,
            auto_tune: true,
            think: ThinkMode::Auto,
            gpu_layers: -1,
            low_vram: false,
            keep_alive: "30m".into(),
            max_tokens: 8192,
            context_tokens: 16384,
        }
    }
}

/// The built-in `[model]` defaults. Auto-tuning only fills in a knob that still
/// equals its default, so anything a user (or `qwenodyssey config set`) writes
/// explicitly always wins over the model profile.
pub static MODEL_DEFAULTS: LazyLock<ModelSettings> = LazyLock::new(ModelSettings::default);

/* ───────────────────────────── [agent] ───────────────────────────── */

/// `[agent]` — pipeline and chat-loop behaviour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    #[serde(deserialize_with = "num::int")]
    pub max_retries: u32,
    pub auto_apply: bool,
    pub auto_test: bool,
    pub review_before_apply: bool,
    pub small_model_mode: bool,
    /// Edit protocol: "blocks" = SEARCH/REPLACE (reliable for small models),
    /// "json" = the legacy JSON-diff coder. Default to blocks.
    pub edit_protocol: EditProtocol,
    /// best-of-N candidate sampling for the block pipeline (1 disables it).
    #[serde(deserialize_with = "num::int")]
    pub candidates: u32,
    /// Run typecheck/lint/test after edits and self-repair from the output.
    pub verify_after_edit: bool,
    /// Build the symbol index at startup so grounding tools are instant.
    pub prebuild_symbol_index: bool,
    /// Budget of PRODUCTIVE tool steps per user turn (turns that actually ran a
    /// tool). Corrective nudges don't count against this — they're bounded by a
    /// separate hard cap (3× this) so the model can't loop forever. Raised from the
    /// old fixed 8 because research / multi-file shell work needs more headroom.
    #[serde(deserialize_with = "num::int")]
    pub max_tool_steps: u32,
    /// Adaptive turn budget / "thinking" mode: allow the model to request extra
    /// reasoning turns before committing to a tool call or final answer. When
    /// enabled, the model can call the `think` tool to spend a "thinking turn"
    /// (separate from tool steps) to deliberate. Each thinking turn consumes 1
    /// from the thinking budget but does NOT count toward max_tool_steps.
    pub thinking_mode: bool,
    /// Maximum thinking turns per user turn (independent of tool steps).
    /// Defaults to max_tool_steps so thinking capacity scales with complexity.
    #[serde(deserialize_with = "num::int")]
    pub max_thinking_turns: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_retries: 4,
            auto_apply: false,
            auto_test: true,
            review_before_apply: true,
            small_model_mode: true,
            edit_protocol: EditProtocol::Blocks,
            candidates: 2,
            verify_after_edit: true,
            prebuild_symbol_index: true,
            max_tool_steps: 16,
            thinking_mode: true,
            max_thinking_turns: 16,
        }
    }
}

/* ───────────────────────────── [tools] ───────────────────────────── */

/// `[tools]` — local tool safety switches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub allow_shell: bool,
    pub confirm_destructive: bool,
    pub sandbox: bool,
    /// Let the agent write into ITS OWN install/source tree (self-modification).
    /// Off by default: a 0.5B model once "added a function to calc.py" by writing
    /// C:/.../Qwenodyssey/src/calc.py, a path it hallucinated from the prompt.
    pub allow_self_edit: bool,
    /// Regex patterns (case-insensitive). Commands matching allow_commands skip the
    /// destructive-confirm prompt (handy for unattended/headless runs of known-safe
    /// families like "^git (status|log|diff)" or "^npm (test|run)"). Commands
    /// matching deny_commands are hard-blocked. deny wins over allow; neither can
    /// override the built-in catastrophic hard-block list.
    pub allow_commands: Vec<String>,
    pub deny_commands: Vec<String>,
    /// Default wall-clock timeout for a single run_shell command, in ms. Raised from
    /// the old fixed 120s so ordinary slow jobs (git clone, npm/pip install, a build)
    /// don't get killed. A command can request more via its `timeout_ms` arg (capped
    /// at 10 min); genuinely long-running / interactive work should use shell_session.
    #[serde(deserialize_with = "num::int")]
    pub shell_timeout_ms: u64,
    /// Enable the persistent shell-session tools (shell_session / _read / _reset),
    /// backed by a real pseudo-terminal. Keeps cwd/env/processes alive across calls
    /// and lets long-running commands be polled. Opt-in because a live shell
    /// weakens the one-shot isolation of run_shell.
    pub shell_session: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            allow_shell: true,
            confirm_destructive: true,
            sandbox: true,
            allow_self_edit: false,
            allow_commands: Vec::new(),
            deny_commands: Vec::new(),
            shell_timeout_ms: 240_000,
            shell_session: false,
        }
    }
}

/* ───────────────────────────── [memory] ───────────────────────────── */

/// `[memory]` — per-project memory store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub path: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self { enabled: true, path: ".qwenodyssey/memory".into() }
    }
}

/* ───────────────────────────── [knowledge] ───────────────────────────── */

/// `[knowledge]` — Obsidian-style long-term knowledge vault + retrieval (RAG).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeConfig {
    pub enabled: bool,
    /// Empty => a global vault at ~/.qwenodyssey/knowledge (permanent, cross-project).
    pub path: String,
    /// Pull relevant notes into context automatically before each turn.
    pub auto_recall: bool,
    #[serde(deserialize_with = "num::int")]
    pub recall_k: u32,
    /// Ollama embedding model for semantic search; falls back to keyword if absent.
    pub embed_model: String,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: String::new(),
            auto_recall: true,
            recall_k: 4,
            embed_model: "nomic-embed-text".into(),
        }
    }
}

/* ───────────────────────────── [lmstudio] ───────────────────────────── */

/// `[lmstudio]` — headless LM Studio backend: use its installed models via the
/// `lms` CLI + `/v1` server.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LmStudioConfig {
    pub enabled: bool,
    /// OpenAI-compatible server. Use the Tailscale IP to reach it across the tailnet.
    pub base_url: String,
    /// Bearer token if the server's "Require API Key" is on; empty if disabled.
    pub api_key: String,
    /// Path to lms.exe; empty = auto-detect (~/.lmstudio/bin/lms.exe or PATH).
    pub cli_path: String,
    /// Ensure the headless server is up at launch.
    pub start_server: bool,
    /// Include LM Studio models in the fallback chain (as `lmstudio:<key>`).
    pub include_as_fallback: bool,
    /// Crash-safe loading for big models.
    pub safe_load: bool,
    /// >= this many B params => treat as big.
    #[serde(serialize_with = "num::float")]
    pub big_params_b: f64,
    /// >= this on-disk size (GB) => big.
    #[serde(serialize_with = "num::float")]
    pub big_size_gb: f64,
    /// Capped context for big models.
    #[serde(deserialize_with = "num::int")]
    pub big_context: u32,
    /// Auto-unload idle big models.
    #[serde(deserialize_with = "num::int")]
    pub ttl_seconds: u64,
}

impl Default for LmStudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "http://localhost:1234".into(),
            api_key: String::new(),
            cli_path: String::new(),
            start_server: true,
            include_as_fallback: true,
            safe_load: true,
            big_params_b: 20.0,
            big_size_gb: 14.0,
            big_context: 8192,
            ttl_seconds: 1800,
        }
    }
}

/* ───────────────────────────── [evolution] ───────────────────────────── */

/// `[evolution]` — self-improvement: reflect on mistakes and save lessons to the
/// knowledge vault.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EvolutionConfig {
    pub enabled: bool,
    /// Only reflect when a turn had failures / hit the step limit (cheap). If false,
    /// it could reflect on every turn (more thorough, more tokens).
    pub reflect_on_failure: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self { enabled: true, reflect_on_failure: true }
    }
}

/* ───────────────────────────── [nvidia] ───────────────────────────── */

/// `[nvidia]` — NVIDIA NIM cloud endpoint (OpenAI-compatible,
/// <https://integrate.api.nvidia.com>). Lets a powerful hosted model (e.g.
/// moonshotai/kimi-k2.6) act as the primary brain or a fallback. The API key is a
/// SECRET — never commit it: leave it blank here and set the NVIDIA_API_KEY
/// environment variable, or put it only in your user-level
/// ~/.qwenodyssey/config.toml (which lives outside the repo).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NvidiaConfig {
    pub enabled: bool,
    pub base_url: String,
    /// Prefer leaving blank + using the env var below. If set here, keep it OUT of
    /// any committed config file (public repo).
    pub api_key: String,
    /// Environment variable to read the key from when api_key is blank.
    pub api_key_env: String,
    /// Include the nvidia:* refs from fallback_models in the runtime fallback chain.
    pub include_as_fallback: bool,
    /// For thinking models (kimi, deepseek-v4, r1…) send chat_template_kwargs.thinking=false.
    /// On NIM the thinking path degenerates/leaks raw CoT; disabling it gives clean answers.
    pub disable_thinking: bool,
    /// Abort a NIM request after this many ms so a hung/slow hosted model (e.g.
    /// deepseek-v4-pro can stall) fails over to the next model in the chain.
    #[serde(deserialize_with = "num::int")]
    pub request_timeout_ms: u64,
    /// Nemotron reasoning models use chat_template_kwargs.enable_thinking (a
    /// different toggle than kimi/v4's `thinking`). When true, Qwenodyssey enables
    /// their reasoning.
    pub nemotron_thinking: bool,
    /// Caps the Nemotron thinking tokens — 0 (default) sends nothing, because
    /// NIM's current runner rejects the parameter with HTTP 400.
    #[serde(deserialize_with = "num::int")]
    pub reasoning_budget: u32,
}

impl Default for NvidiaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://integrate.api.nvidia.com".into(),
            api_key: String::new(),
            api_key_env: "NVIDIA_API_KEY".into(),
            include_as_fallback: true,
            disable_thinking: true,
            request_timeout_ms: 90_000,
            nemotron_thinking: true,
            reasoning_budget: 0,
        }
    }
}

/* ───────────────────────────── [openrouter] ───────────────────────────── */

/// `[openrouter]` — OpenRouter cloud endpoint (OpenAI-compatible,
/// <https://openrouter.ai/api>). One key aggregates many providers/models (e.g.
/// moonshotai/kimi-k2.6) that can act as the primary brain or a fallback. The API
/// key is a SECRET — never commit it: leave it blank here and set the
/// OPENROUTER_API_KEY environment variable, or put it only in your user-level
/// ~/.qwenodyssey/config.toml (which lives outside the repo).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterConfig {
    pub enabled: bool,
    pub base_url: String,
    /// Prefer leaving blank + using the env var below. If set here, keep it OUT of
    /// any committed config file (public repo).
    pub api_key: String,
    /// Environment variable to read the key from when api_key is blank.
    pub api_key_env: String,
    /// Include the openrouter:* refs from fallback_models in the runtime fallback chain.
    pub include_as_fallback: bool,
    /// Abort a request after this many ms so a hung/slow hosted model fails over to
    /// the next model in the chain.
    #[serde(deserialize_with = "num::int")]
    pub request_timeout_ms: u64,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://openrouter.ai/api".into(),
            api_key: String::new(),
            api_key_env: "OPENROUTER_API_KEY".into(),
            include_as_fallback: true,
            request_timeout_ms: 90_000,
        }
    }
}

/* ───────────────────────────── [anthropic] ───────────────────────────── */

/// `[anthropic]` — Anthropic (Claude) cloud endpoint, the NATIVE Messages API
/// (<https://api.anthropic.com/v1/messages>). Lets a Claude model act as the
/// primary brain or a fallback. TWO auth modes:
///  - **Claude Pro/Max subscription** (OAuth): the agent reads a short-lived bearer
///    token from the `ant` CLI (`ant auth print-credentials`) when use_cli_auth is
///    on — the same path Claude Code uses — or from the ANTHROPIC_AUTH_TOKEN env var.
///  - **API key** (console.anthropic.com, pay-as-you-go): ANTHROPIC_API_KEY.
///
/// Secrets are NEVER required in a committed file — leave the fields blank and use
/// the env vars or `ant auth login`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicConfig {
    pub enabled: bool,
    pub base_url: String,
    /// OAuth bearer token for a Claude subscription. Prefer leaving blank + `ant auth login`.
    pub auth_token: String,
    pub auth_token_env: String,
    /// Source (and auto-refresh) the OAuth token from the `ant` CLI when no explicit
    /// token is set. Turn off if you don't have the Anthropic CLI installed.
    pub use_cli_auth: bool,
    /// Developer API key (pay-as-you-go). Prefer the env var.
    pub api_key: String,
    pub api_key_env: String,
    /// Include anthropic:* refs from fallback_models in the runtime fallback chain.
    pub include_as_fallback: bool,
    /// Abort a request after this many ms so a stall fails over to the next model.
    #[serde(deserialize_with = "num::int")]
    pub request_timeout_ms: u64,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://api.anthropic.com".into(),
            auth_token: String::new(),
            auth_token_env: "ANTHROPIC_AUTH_TOKEN".into(),
            use_cli_auth: true,
            api_key: String::new(),
            api_key_env: "ANTHROPIC_API_KEY".into(),
            include_as_fallback: true,
            request_timeout_ms: 120_000,
        }
    }
}

/* ───────────────────────────── [vision] ───────────────────────────── */

/// `[vision]` — webcam vision: capture a frame and describe it with a
/// vision-language model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub enabled: bool,
    /// Which backend analyzes the captured frame. nvidia/openai use the OpenAI-style
    /// image_url payload; ollama uses its native images field.
    pub provider: VisionProvider,
    pub model: String,
    /// DirectShow camera device name; "" = auto-detect the first dshow video device.
    pub device: String,
    /// Ask for confirmation before each capture (the camera is privacy-sensitive).
    pub confirm: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: VisionProvider::Nvidia,
            model: "nvidia/nemotron-nano-12b-v2-vl".into(),
            device: String::new(),
            confirm: true,
        }
    }
}

/* ───────────────────────────── [audio] ───────────────────────────── */

/// `[audio]` — microphone speech-to-text via a local whisper.cpp binary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub enabled: bool,
    /// DirectShow audio device; "" = auto-detect a mic (prefers one matching the camera).
    pub device: String,
    /// whisper.cpp CLI; "" = auto (whisper-cli / whisper-cpp / main on PATH).
    pub whisper_bin: String,
    /// ggml model path; "" = ~/.qwenodyssey/models/ggml-base.en.bin.
    pub whisper_model: String,
    /// Safety cap on a single push-to-talk recording, in seconds.
    #[serde(deserialize_with = "num::int")]
    pub max_seconds: u64,
    pub language: String,
    /// Voice-activity-detection threshold for continuous listening (whisper-stream).
    /// Higher = needs louder/clearer speech to trigger; lower = more sensitive.
    #[serde(serialize_with = "num::float")]
    pub vad_threshold: f64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: String::new(),
            whisper_bin: String::new(),
            whisper_model: String::new(),
            max_seconds: 30,
            language: "en".into(),
            vad_threshold: 0.6,
        }
    }
}

/* ───────────────────────────── [omni] ───────────────────────────── */

/// `[omni]` — omni mode: a single multimodal NIM model does everything — it
/// ingests the mic audio AND the camera frame directly and replies (no separate
/// local STT/vision). Audio + frames are uploaded to the cloud model in this mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OmniConfig {
    pub enabled: bool,
    /// Multimodal model that accepts input_audio (+ images) and reasons in one call.
    /// nemotron-3-nano-omni (NVIDIA) is verified audio+vision+reasoning and reliable
    /// on NIM; microsoft/phi-4-multimodal-instruct also works when not degraded.
    pub model: String,
    pub provider: OmniProvider,
    /// Send the camera frame alongside the audio each turn.
    pub send_image: bool,
}

impl Default for OmniConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning".into(),
            provider: OmniProvider::Nvidia,
            send_image: true,
        }
    }
}

/* ───────────────────────────── [tts] ───────────────────────────── */

/// `[tts]` — spoken replies (text-to-speech).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub enabled: bool,
    /// piper = local neural TTS; sapi = built-in Windows voice; none = text only.
    pub engine: TtsEngine,
    /// "" = ~/.qwenodyssey/piper/piper/piper.exe
    pub piper_bin: String,
    /// "" = ~/.qwenodyssey/piper/voices/en_US-amy-medium.onnx
    pub piper_voice: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: TtsEngine::Piper,
            piper_bin: String::new(),
            piper_voice: String::new(),
        }
    }
}

/* ───────────────────────────── [swarm] ───────────────────────────── */

/// `[swarm]` — agent swarm: fan a complex task out to MANY frontier models at
/// once, each on its own backend + API key (the cloud refs already in
/// `model.fallback_models`). "ensemble" = every model answers the same task then a
/// lead synthesizes the best answer; "divide" = independent subtasks are sharded
/// across the models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SwarmConfig {
    pub enabled: bool,
    /// Hard cap on how many models run concurrently (protects rate limits / cost).
    #[serde(deserialize_with = "num::int")]
    pub max_workers: u32,
    /// Also enlist local refs (bare Ollama tags / lmstudio:*) from the fallback list,
    /// not just cloud models. Off by default so the swarm stays frontier-only.
    pub include_local: bool,
    /// Who merges the workers' answers: "primary" (the active model) or an explicit
    /// ref like "openrouter:moonshotai/kimi-k2.6".
    pub synthesizer: String,
    /// Per-worker (and synthesis) output token budget.
    #[serde(deserialize_with = "num::int")]
    pub max_tokens: u32,
    /// COORDINATED mode: a lead model splits the task into dependency-aware subtasks
    /// and the agents share a blackboard (each sees its dependencies' results), so the
    /// team builds on each other instead of answering blind. On by default for the CLI
    /// `swarm` command and the agent_swarm tool; the plain ensemble/divide path is still
    /// reachable via --plain / mode:"ensemble".
    pub coordinate: bool,
    /// Show the live split-pane TUI (one streaming terminal per agent) when the CLI
    /// `swarm` command runs in a TTY. Falls back to a plain line log when off / non-TTY.
    pub live: bool,
    /// Where swarm agents may RUN COMMANDS (see [`SwarmExec`]).
    pub exec: SwarmExec,
    /// Max run_shell tool steps per agent subtask.
    #[serde(deserialize_with = "num::int")]
    pub exec_max_steps: u32,
    /// Per-command timeout, in seconds.
    #[serde(deserialize_with = "num::int")]
    pub exec_timeout_s: u64,
    /// Health-check each agent's API key/endpoint before the run (drops broken ones).
    pub preflight: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_workers: 4,
            include_local: false,
            synthesizer: "primary".into(),
            max_tokens: 1500,
            coordinate: true,
            live: true,
            exec: SwarmExec::Off,
            exec_max_steps: 5,
            exec_timeout_s: 120,
            preflight: true,
        }
    }
}

/* ───────────────────────────── [daytona] ───────────────────────────── */

/// `[daytona]` — Daytona (<https://daytona.io>) isolated cloud sandboxes where swarm
/// agents run commands for COMPLEX projects (dependency installs, multi-file
/// builds, servers) without touching the host. One sandbox per swarm run,
/// auto-deleted afterwards. Key is a SECRET: leave `api_key` blank and set the env
/// var instead.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DaytonaConfig {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    pub api_key_env: String,
    /// Optional snapshot/image name for new sandboxes.
    pub snapshot: String,
    /// Optional target region for new sandboxes.
    pub target: String,
    /// Sandbox lifecycle backstop (minutes): auto-stop when idle.
    #[serde(deserialize_with = "num::int")]
    pub auto_stop_minutes: u64,
    /// Sandbox lifecycle backstop (minutes): auto-delete after.
    #[serde(deserialize_with = "num::int")]
    pub auto_delete_minutes: u64,
}

impl Default for DaytonaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://app.daytona.io/api".into(),
            api_key: String::new(),
            api_key_env: "DAYTONA_API_KEY".into(),
            snapshot: String::new(),
            target: String::new(),
            auto_stop_minutes: 15,
            auto_delete_minutes: 120,
        }
    }
}

/* ───────────────────────────── [github] ───────────────────────────── */

/// `[github]` — gives the agent the user's GitHub login so it can read, write,
/// upload and review code. The token is a SECRET — never commit it. Leave `token`
/// blank and the tools fall back to the GITHUB_TOKEN env var, then to the GitHub
/// CLI's own login (`gh auth token`), so an already-`gh auth login`'d machine needs
/// zero config here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GithubConfig {
    pub enabled: bool,
    /// Personal access token (classic or fine-grained with `repo` scope). Prefer
    /// leaving blank + using `gh auth login` or the env var below.
    pub token: String,
    /// Env var to read the token from when `token` is blank.
    pub token_env: String,
    /// Default owner for `owner/repo` args; blank = the authenticated user.
    pub default_owner: String,
    /// REST API base (change for GitHub Enterprise).
    pub api_base: String,
}

impl Default for GithubConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            token: String::new(),
            token_env: "GITHUB_TOKEN".into(),
            default_owner: String::new(),
            api_base: "https://api.github.com".into(),
        }
    }
}

/* ───────────────────────────── [mcp] ───────────────────────────── */

/// One `[mcp.servers.<name>]` entry: a stdio MCP server whose tools are surfaced
/// to the agent, namespaced `mcp__<name>__<tool>`.
///
/// Example (~/.qwenodyssey/config.toml):
/// ```toml
/// [mcp]
/// enabled = true
/// [mcp.servers.filesystem]
/// command = "npx"
/// args = ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me"]
/// [mcp.servers.git]
/// command = "uvx"
/// args = ["mcp-server-git"]
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServerConfig {
    /// Skip this server when false (keep the definition without launching it).
    pub enabled: bool,
    /// Executable to spawn (npx, uvx, node, python, or an absolute path).
    pub command: String,
    pub args: Vec<String>,
    /// Extra env vars merged over the process env (e.g. API tokens the server needs).
    pub env: IndexMap<String, String>,
    /// Working directory for the server; blank = current dir.
    pub cwd: String,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: String::new(),
            args: Vec::new(),
            env: IndexMap::new(),
            cwd: String::new(),
        }
    }
}

/// `[mcp]` — Model Context Protocol servers. This is how Qwenodyssey plugs into
/// the wider MCP ecosystem (filesystem, git, Slack, Postgres, Puppeteer, …)
/// without bespoke integrations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct McpConfig {
    /// Master switch. Off by default — MCP launches external server processes, so
    /// it's opt-in. Turn on and define at least one [mcp.servers.*] to use it.
    pub enabled: bool,
    pub servers: IndexMap<String, McpServerConfig>,
    /// Timeout for the initialize handshake + tools/list of each server.
    #[serde(deserialize_with = "num::int")]
    pub init_timeout_ms: u64,
    /// Timeout for a single tools/call.
    #[serde(deserialize_with = "num::int")]
    pub call_timeout_ms: u64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            servers: IndexMap::new(),
            init_timeout_ms: 20_000,
            call_timeout_ms: 60_000,
        }
    }
}

/* ───────────────────────────── [web] ───────────────────────────── */

/// `[web]` — internet search + page fetch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    pub enabled: bool,
    pub provider: WebProvider,
    pub api_key: String,
    pub searxng_url: String,
    #[serde(deserialize_with = "num::int")]
    pub max_results: u32,
    #[serde(deserialize_with = "num::int")]
    pub fetch_chars: u32,
    /// Abort a single search/page fetch after this many ms so a hung site can't
    /// stall the turn (0 = no timeout).
    #[serde(deserialize_with = "num::int")]
    pub fetch_timeout_ms: u64,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: WebProvider::DuckDuckGo,
            api_key: String::new(),
            searxng_url: String::new(),
            max_results: 5,
            fetch_chars: 8000,
            fetch_timeout_ms: 15_000,
        }
    }
}

/* ───────────────────────────── Config ───────────────────────────── */

/// The whole configuration (TS `ConfigSchema`). Section order matches the TS so
/// `config list` / `save_config` output is in the same order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelSettings,
    pub agent: AgentConfig,
    pub tools: ToolsConfig,
    pub memory: MemoryConfig,
    pub knowledge: KnowledgeConfig,
    pub web: WebConfig,
    pub mcp: McpConfig,
    pub github: GithubConfig,
    pub swarm: SwarmConfig,
    pub daytona: DaytonaConfig,
    pub evolution: EvolutionConfig,
    pub lmstudio: LmStudioConfig,
    pub nvidia: NvidiaConfig,
    pub openrouter: OpenRouterConfig,
    pub anthropic: AnthropicConfig,
    pub vision: VisionConfig,
    pub audio: AudioConfig,
    pub tts: TtsConfig,
    pub omni: OmniConfig,
}

impl Config {
    /// The config as a JSON value (what `config list` prints, pretty-printed).
    pub fn to_json(&self) -> Value {
        // Serializing plain structs/maps/strings/finite numbers cannot fail.
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// The config as TOML text (what [`save_config`] writes).
    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string(self).context("Failed to serialize configuration to TOML")
    }

    /// Parse one TOML document into a config (defaults fill whatever it omits).
    pub fn from_toml_str(text: &str) -> Result<Config> {
        let table = parse_toml_table(text)?;
        Ok(toml::Value::Table(table).try_into::<Config>()?)
    }
}

/// Built-in defaults (TS `defaultConfig()`).
pub fn default_config() -> Config {
    Config::default()
}

/* ───────────────────────────── paths ───────────────────────────── */

/// The user's home directory, resolved like Node's `os.homedir()`:
/// `USERPROFILE` (Windows) / `HOME` (elsewhere) when set, else the OS profile dir.
pub fn home_dir() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    match std::env::var_os(var) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")),
    }
}

/// `<cwd>/.qwenodyssey` — the project workspace (logs, memory, sessions, config).
pub fn workspace_path(cwd: impl AsRef<Path>) -> PathBuf {
    cwd.as_ref().join(WORKSPACE_DIR)
}

/// `<cwd>/.qwenodyssey/config.toml` — the project config file.
pub fn config_path(cwd: impl AsRef<Path>) -> PathBuf {
    workspace_path(cwd).join("config.toml")
}

/// `~/.qwenodyssey` — the user-level workspace (global config, knowledge vault,
/// models, piper voices…).
pub fn global_workspace_path() -> PathBuf {
    home_dir().join(WORKSPACE_DIR)
}

/// User-level config (~/.qwenodyssey/config.toml). Acts as the cross-project base:
/// settings here (e.g. the knowledge vault path, API keys) apply everywhere, while
/// a project's own .qwenodyssey/config.toml overrides them per-repo.
pub fn global_config_path() -> PathBuf {
    global_workspace_path().join("config.toml")
}

/* ───────────────────────────── load / save ───────────────────────────── */

fn parse_toml_table(text: &str) -> Result<toml::Table> {
    // Tolerate a UTF-8 BOM (Windows PowerShell 5.1's `Out-File` writes one).
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    text.parse::<toml::Table>().map_err(|e| anyhow!("{}", e.to_string().trim_end()))
}

/// Read + parse a TOML file; `Ok(None)` when it doesn't exist.
fn read_toml_raw(file: &Path) -> Result<Option<toml::Table>> {
    if !file.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    match parse_toml_table(&text) {
        Ok(t) => Ok(Some(t)),
        Err(e) => bail!(
            "Failed to parse {}: {}. Run \"qwenodyssey init --force\" to regenerate it.",
            file.display(),
            e
        ),
    }
}

/// Deep-merge `source` over `base`: nested tables merge, arrays/scalars replace.
/// An existing key keeps its position; new keys are appended (JS object order).
pub fn deep_merge(base: &mut toml::Table, source: toml::Table) {
    for (k, v) in source {
        let both_tables = v.is_table() && matches!(base.get(&k), Some(toml::Value::Table(_)));
        if both_tables {
            if let (Some(toml::Value::Table(b)), toml::Value::Table(s)) = (base.get_mut(&k), v) {
                deep_merge(b, s);
            }
        } else {
            base.insert(k, v);
        }
    }
}

/// Lexically normalize an absolute version of `p` (like Node's `path.resolve`).
fn resolve_path(p: &Path) -> PathBuf {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn same_file(a: &Path, b: &Path) -> bool {
    let (a, b) = (resolve_path(a), resolve_path(b));
    if cfg!(windows) {
        // NTFS paths are case-insensitive; C:\Users\x and c:/users/x are one file.
        a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

/// Load configuration, layering the project's .qwenodyssey/config.toml ON TOP of
/// the user-level ~/.qwenodyssey/config.toml. The global file holds machine-wide
/// defaults (knowledge vault, keys, backend); the project file overrides only what
/// it sets. With neither present, built-in defaults are used.
pub fn load_config(cwd: impl AsRef<Path>) -> Result<Config> {
    load_config_layered(&global_config_path(), &config_path(cwd))
}

/// [`load_config`] with explicit file paths (global base, project override).
/// When both name the same file (cwd IS the home directory) it is read once.
pub fn load_config_layered(global_file: &Path, project_file: &Path) -> Result<Config> {
    let global_raw = read_toml_raw(global_file)?;
    // Avoid reading the same file twice when cwd IS the home directory.
    let project_raw = if same_file(project_file, global_file) {
        None
    } else {
        read_toml_raw(project_file)?
    };

    if global_raw.is_none() && project_raw.is_none() {
        return Ok(default_config());
    }

    let mut merged = global_raw.unwrap_or_default();
    deep_merge(&mut merged, project_raw.unwrap_or_default());
    toml::Value::Table(merged).try_into::<Config>().map_err(|e| {
        anyhow!(
            "Invalid configuration (merged {} + {}): {}. Run \"qwenodyssey init --force\" to regenerate the project config.",
            global_file.display(),
            project_file.display(),
            e.to_string().trim_end()
        )
    })
}

/// Write `config` to `<cwd>/.qwenodyssey/config.toml` (creating the directory).
pub fn save_config(cwd: impl AsRef<Path>, config: &Config) -> Result<()> {
    let dir = workspace_path(&cwd);
    fs::create_dir_all(&dir).with_context(|| format!("Failed to create {}", dir.display()))?;
    let text = config.to_toml_string()?;
    let file = config_path(&cwd);
    fs::write(&file, text).with_context(|| format!("Failed to write {}", file.display()))
}

/* ───────────────────────────── dotted get / set ───────────────────────────── */

/// Read a dotted key like "model.provider". `None` when the path doesn't exist
/// (the TS returned `undefined`, which `config get` prints as `undefined`).
/// A numeric segment indexes into an array ("model.fallback_models.0").
pub fn get_by_path(config: &Config, dotted: &str) -> Option<Value> {
    let mut node = config.to_json();
    for key in dotted.split('.') {
        node = match node {
            Value::Object(mut map) => map.remove(key)?,
            Value::Array(mut arr) => {
                let i: usize = key.parse().ok()?;
                if i < arr.len() {
                    arr.swap_remove(i)
                } else {
                    return None;
                }
            }
            _ => return None,
        };
    }
    Some(node)
}

/// Alias of [`get_by_path`].
pub fn get_value(config: &Config, dotted: &str) -> Option<Value> {
    get_by_path(config, dotted)
}

/// Set a dotted key, coercing simple scalar types to match the existing value,
/// and return the re-validated config (the input is untouched).
///
/// - existing number → the string must parse as a JS `Number` ("Expected a number")
/// - existing boolean → true|1|yes|on / false|0|no|off, any case ("Expected a boolean")
/// - anything else (string, missing key, …) → stored as the raw string
///
/// Missing intermediate tables are created (so `mcp.servers.git.command` works);
/// an unknown leaf key is dropped by validation, exactly like zod's strip.
pub fn set_by_path(config: &Config, dotted: &str, value: &str) -> Result<Config> {
    let parts: Vec<&str> = dotted.split('.').collect();
    let mut root = config.to_json();
    let mut node = &mut root;
    for part in &parts[..parts.len() - 1] {
        node = child_slot(node, part)?;
        if !node.is_object() && !node.is_array() {
            *node = Value::Object(serde_json::Map::new());
        }
    }
    let leaf = parts[parts.len() - 1];
    let slot = child_slot(node, leaf)?;
    *slot = coerce(value, slot)?;
    serde_json::from_value::<Config>(root).map_err(|e| anyhow!("Invalid value for {dotted}: {e}"))
}

/// In-place variant of [`set_by_path`]: `config` changes only on success.
pub fn set_value(config: &mut Config, dotted: &str, value: &str) -> Result<()> {
    *config = set_by_path(config, dotted, value)?;
    Ok(())
}

/// The child slot `key` of an object/array node, created (as `null`) if missing.
fn child_slot<'a>(node: &'a mut Value, key: &str) -> Result<&'a mut Value> {
    match node {
        Value::Object(map) => Ok(map.entry(key.to_string()).or_insert(Value::Null)),
        Value::Array(arr) => {
            let i: usize = key
                .parse()
                .map_err(|_| anyhow!("\"{key}\" is not an index into a list"))?;
            if i == arr.len() {
                arr.push(Value::Null);
            }
            arr.get_mut(i).ok_or_else(|| anyhow!("index {i} is past the end of the list"))
        }
        _ => Err(anyhow!("cannot set \"{key}\" on a non-table value")),
    }
}

fn coerce(value: &str, prev: &Value) -> Result<Value> {
    match prev {
        Value::Number(_) => {
            let n = js_number(value).ok_or_else(|| anyhow!("Expected a number, got \"{value}\""))?;
            if n.is_finite() && n.fract() == 0.0 && n.abs() <= 9_007_199_254_740_991.0 {
                Ok(Value::from(n as i64))
            } else {
                serde_json::Number::from_f64(n)
                    .map(Value::Number)
                    .ok_or_else(|| anyhow!("Expected a finite number, got \"{value}\""))
            }
        }
        Value::Bool(_) => {
            let v = value.to_ascii_lowercase();
            match v.as_str() {
                "true" | "1" | "yes" | "on" => Ok(Value::Bool(true)),
                "false" | "0" | "no" | "off" => Ok(Value::Bool(false)),
                _ => Err(anyhow!("Expected a boolean, got \"{value}\"")),
            }
        }
        _ => Ok(Value::String(value.to_string())),
    }
}

/// JavaScript `Number(str)`: trims, "" → 0, hex/octal/binary prefixes,
/// `Infinity`; `None` where JS would give `NaN`.
fn js_number(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return Some(0.0);
    }
    for (prefix, radix) in [("0x", 16), ("0X", 16), ("0o", 8), ("0O", 8), ("0b", 2), ("0B", 2)] {
        if let Some(rest) = t.strip_prefix(prefix) {
            return u64::from_str_radix(rest, radix).ok().map(|n| n as f64);
        }
    }
    match t {
        "Infinity" | "+Infinity" => return Some(f64::INFINITY),
        "-Infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    // Only plain decimal literals: Rust's parser also takes "inf"/"nan", JS doesn't.
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    let (mantissa, exponent) = match body.find(['e', 'E']) {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let digits_ok = {
        let mut parts = mantissa.splitn(2, '.');
        let int = parts.next().unwrap_or("");
        let frac = parts.next().unwrap_or("");
        (!int.is_empty() || !frac.is_empty())
            && int.bytes().all(|b| b.is_ascii_digit())
            && frac.bytes().all(|b| b.is_ascii_digit())
    };
    let exp_ok = exponent.map_or(true, |e| {
        let e = e.strip_prefix(['+', '-']).unwrap_or(e);
        !e.is_empty() && e.bytes().all(|b| b.is_ascii_digit())
    });
    if digits_ok && exp_ok {
        t.parse::<f64>().ok()
    } else {
        None
    }
}

/* ───────────────────────────── tests ───────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    const TS_DEFAULTS: &str = include_str!("../testdata/ts_default_config.json");

    /// A representative user-level config (shape of a real one, fake values):
    /// underscored integers, a literal-string Windows path, comments inside an
    /// array, sections in a different order than the schema, missing keys.
    const SAMPLE_GLOBAL: &str = r#"
[model]
provider = "nvidia"
model = "moonshotai/kimi-k2.6"
# Frontier fallbacks
fallback_models = [
  "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
  # removed: an EOL model
  "openrouter:deepseek/deepseek-chat",
  "qwen2.5-coder:7b",
]
base_url = "http://localhost:11434"
api_key = ""
temperature = 0.2
top_p = 0.9
gpu_layers = -1
low_vram = false
keep_alive = "30m"
max_tokens = 8_192
context_tokens = 16_384

[agent]
max_retries = 4
auto_apply = false

[tools]
allow_shell = true
shell_session = true

[swarm]
exec = "auto"

[daytona]
enabled = true

[knowledge]
enabled = true
path = 'C:\notes\vault'
recall_k = 4

[web]
provider = "duckduckgo"
fetch_chars = 8_000

[lmstudio]
base_url = "http://100.64.0.1:1234"
big_params_b = 20
big_size_gb = 14
big_context = 8_192
ttl_seconds = 1_800

[nvidia]
api_key = "nvapi-fake"
request_timeout_ms = 90_000
reasoning_budget = 4_096

[vision]
provider = "ollama"
model = "moondream"
confirm = false

[audio]
vad_threshold = 0.6

[tts]
engine = "piper"

[omni]
enabled = true
provider = "nvidia"
send_image = true

[some_future_section]
whatever = 1
"#;

    fn write(dir: &Path, rel: &str, text: &str) -> PathBuf {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn defaults_match_the_typescript_schema_exactly() {
        let ts: Value = serde_json::from_str(TS_DEFAULTS).unwrap();
        assert_eq!(default_config().to_json(), ts);
        // Key order matches too (serde_json preserve_order).
        assert_eq!(
            serde_json::to_string_pretty(&default_config()).unwrap(),
            serde_json::to_string_pretty(&ts).unwrap()
        );
    }

    #[test]
    fn model_defaults_equal_default_model_section() {
        assert_eq!(*MODEL_DEFAULTS, default_config().model);
        assert_eq!(MODEL_DEFAULTS.top_k, 0);
        assert_eq!(MODEL_DEFAULTS.temperature, 0.2);
    }

    #[test]
    fn empty_and_partial_toml_fill_defaults() {
        assert_eq!(Config::from_toml_str("").unwrap(), default_config());
        let c = Config::from_toml_str("[model]\nmodel = \"x\"\n").unwrap();
        assert_eq!(c.model.model, "x");
        assert_eq!(c.model.max_tokens, 8192);
        assert_eq!(c.agent, AgentConfig::default());
    }

    #[test]
    fn sample_user_config_parses() {
        let c = Config::from_toml_str(SAMPLE_GLOBAL).unwrap();
        assert_eq!(c.model.provider, ProviderKind::Nvidia);
        assert_eq!(c.model.fallback_models.len(), 3);
        assert_eq!(c.model.max_tokens, 8192);
        assert_eq!(c.swarm.exec, SwarmExec::Auto);
        assert_eq!(c.knowledge.path, r"C:\notes\vault");
        assert_eq!(c.lmstudio.big_params_b, 20.0);
        assert_eq!(c.nvidia.reasoning_budget, 4096);
        assert_eq!(c.vision.provider, VisionProvider::Ollama);
        assert!(!c.vision.confirm);
        assert!(c.omni.enabled);
        assert!(c.tools.shell_session);
        // untouched keys keep their defaults
        assert_eq!(c.agent.max_tool_steps, 16);
        assert_eq!(c.anthropic, AnthropicConfig::default());
    }

    #[test]
    fn bom_is_tolerated() {
        let c = Config::from_toml_str("\u{feff}[model]\nmodel = \"bom\"\n").unwrap();
        assert_eq!(c.model.model, "bom");
    }

    #[test]
    fn integral_float_accepted_for_integer_fields_but_fraction_rejected() {
        let c = Config::from_toml_str("[model]\nmax_tokens = 4096.0\ntemperature = 1\n").unwrap();
        assert_eq!(c.model.max_tokens, 4096);
        assert_eq!(c.model.temperature, 1.0);
        assert!(Config::from_toml_str("[model]\nmax_tokens = 4096.5\n").is_err());
        assert!(Config::from_toml_str("[model]\nmax_tokens = -1\n").is_err());
    }

    #[test]
    fn invalid_enum_and_wrong_types_are_rejected() {
        assert!(Config::from_toml_str("[model]\nprovider = \"bogus\"\n").is_err());
        assert!(Config::from_toml_str("[model]\ntemperature = \"0.2\"\n").is_err());
        assert!(Config::from_toml_str("[model]\nfallback_models = \"x\"\n").is_err());
    }

    #[test]
    fn enum_round_trip() {
        for p in ProviderKind::ALL {
            assert_eq!(p.as_str().parse::<ProviderKind>().unwrap(), *p);
        }
        assert_eq!(ProviderKind::LmStudio.to_string(), "lmstudio");
        assert_eq!(TtsEngine::None.as_str(), "none");
        assert!("Ollama".parse::<ProviderKind>().is_err());
    }

    #[test]
    fn deep_merge_semantics() {
        let mut base: toml::Table = r#"
a = 1
arr = [1, 2, 3]
[t]
x = 1
y = 2
[t.inner]
k = "base"
"#
        .parse()
        .unwrap();
        let src: toml::Table = r#"
arr = [9]
new = true
[t]
y = 20
[t.inner]
j = "src"
"#
        .parse()
        .unwrap();
        deep_merge(&mut base, src);
        assert_eq!(base["a"].as_integer(), Some(1));
        assert_eq!(base["arr"].as_array().unwrap().len(), 1, "arrays replace");
        assert_eq!(base["new"].as_bool(), Some(true));
        let t = base["t"].as_table().unwrap();
        assert_eq!(t["x"].as_integer(), Some(1));
        assert_eq!(t["y"].as_integer(), Some(20));
        let inner = t["inner"].as_table().unwrap();
        assert_eq!(inner["k"].as_str(), Some("base"));
        assert_eq!(inner["j"].as_str(), Some("src"));
        // existing keys keep position, new ones append
        let keys: Vec<&String> = base.keys().collect();
        assert_eq!(keys, ["a", "arr", "t", "new"]);
    }

    #[test]
    fn scalar_replaces_table_and_table_replaces_scalar() {
        let mut base: toml::Table = "[t]\nx = 1\n".parse().unwrap();
        deep_merge(&mut base, "t = 5\n".parse().unwrap());
        assert_eq!(base["t"].as_integer(), Some(5));
        deep_merge(&mut base, "[t]\nz = 1\n".parse().unwrap());
        assert_eq!(base["t"].as_table().unwrap().len(), 1);
    }

    #[test]
    fn layered_load_project_overrides_global() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(dir.path(), "home/.qwenodyssey/config.toml", SAMPLE_GLOBAL);
        let project = write(
            dir.path(),
            "proj/.qwenodyssey/config.toml",
            "[model]\nmodel = \"qwen2.5-coder:7b\"\nprovider = \"ollama\"\n[nvidia]\nrequest_timeout_ms = 1000\n",
        );
        let c = load_config_layered(&global, &project).unwrap();
        assert_eq!(c.model.model, "qwen2.5-coder:7b");
        assert_eq!(c.model.provider, ProviderKind::Ollama);
        // global values not overridden survive
        assert_eq!(c.model.fallback_models.len(), 3);
        assert_eq!(c.nvidia.api_key, "nvapi-fake");
        assert_eq!(c.nvidia.request_timeout_ms, 1000);
        assert_eq!(c.nvidia.reasoning_budget, 4096);
    }

    #[test]
    fn layered_load_neither_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let c = load_config_layered(&dir.path().join("g.toml"), &dir.path().join("p.toml")).unwrap();
        assert_eq!(c, default_config());
    }

    #[test]
    fn layered_load_only_project() {
        let dir = tempfile::tempdir().unwrap();
        let project = write(dir.path(), "p/.qwenodyssey/config.toml", "[web]\nmax_results = 9\n");
        let c = load_config_layered(&dir.path().join("missing.toml"), &project).unwrap();
        assert_eq!(c.web.max_results, 9);
    }

    #[test]
    fn same_file_is_read_once() {
        let dir = tempfile::tempdir().unwrap();
        let f = write(dir.path(), ".qwenodyssey/config.toml", "[model]\nmodel = \"m\"\n");
        let alt = dir.path().join(".qwenodyssey").join(".").join("config.toml");
        assert!(same_file(&f, &alt));
        let c = load_config_layered(&f, &alt).unwrap();
        assert_eq!(c.model.model, "m");
    }

    #[test]
    fn parse_error_message_names_file_and_fix() {
        let dir = tempfile::tempdir().unwrap();
        let bad = write(dir.path(), "bad.toml", "[model\nx = ");
        let err = load_config_layered(&bad, &dir.path().join("p.toml")).unwrap_err().to_string();
        assert!(err.starts_with("Failed to parse "), "{err}");
        assert!(err.contains("bad.toml"), "{err}");
        assert!(err.contains("qwenodyssey init --force"), "{err}");
    }

    #[test]
    fn invalid_merged_config_message() {
        let dir = tempfile::tempdir().unwrap();
        let g = write(dir.path(), "g.toml", "[model]\nprovider = \"nope\"\n");
        let err = load_config_layered(&g, &dir.path().join("p.toml")).unwrap_err().to_string();
        assert!(err.starts_with("Invalid configuration (merged "), "{err}");
        assert!(err.contains("regenerate the project config"), "{err}");
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = Config::from_toml_str(SAMPLE_GLOBAL).unwrap();
        c.mcp.enabled = true;
        let mut srv = McpServerConfig { command: "npx".into(), ..Default::default() };
        srv.args = vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()];
        srv.env.insert("TOKEN".into(), "abc".into());
        c.mcp.servers.insert("filesystem".into(), srv);
        c.mcp.servers.insert("git".into(), McpServerConfig { command: "uvx".into(), ..Default::default() });
        save_config(dir.path(), &c).unwrap();
        let text = fs::read_to_string(config_path(dir.path())).unwrap();
        assert!(text.contains("[model]"));
        assert!(text.contains("repeat_penalty = 0\n"), "integral floats written like JS: {text}");
        let back = load_config_layered(&dir.path().join("none.toml"), &config_path(dir.path())).unwrap();
        assert_eq!(back, c);
        let names: Vec<&String> = back.mcp.servers.keys().collect();
        assert_eq!(names, ["filesystem", "git"]);
    }

    #[test]
    fn default_config_round_trips_through_toml() {
        let text = default_config().to_toml_string().unwrap();
        assert_eq!(Config::from_toml_str(&text).unwrap(), default_config());
    }

    #[test]
    fn paths() {
        let cwd = Path::new("proj");
        assert_eq!(workspace_path(cwd), Path::new("proj").join(".qwenodyssey"));
        assert_eq!(config_path(cwd), Path::new("proj").join(".qwenodyssey").join("config.toml"));
        assert!(global_config_path().ends_with(Path::new(".qwenodyssey").join("config.toml")));
    }

    #[test]
    fn get_by_path_reads_dotted_keys() {
        let c = default_config();
        assert_eq!(get_by_path(&c, "model.provider"), Some(Value::from("ollama")));
        assert_eq!(get_by_path(&c, "model.temperature"), Some(serde_json::json!(0.2)));
        assert_eq!(get_by_path(&c, "model.repeat_penalty"), Some(serde_json::json!(0)));
        assert_eq!(get_by_path(&c, "model.max_tokens"), Some(serde_json::json!(8192)));
        assert_eq!(
            get_by_path(&c, "model.fallback_models.0"),
            Some(Value::from("nvidia:deepseek-ai/deepseek-v4-pro"))
        );
        assert!(get_by_path(&c, "agent").unwrap().is_object());
        assert_eq!(get_by_path(&c, "model.nope"), None);
        assert_eq!(get_by_path(&c, "model.provider.deeper"), None);
        assert_eq!(get_value(&c, "tools.allow_shell"), Some(Value::Bool(true)));
    }

    #[test]
    fn set_by_path_coerces_to_existing_type() {
        let c = default_config();
        let c2 = set_by_path(&c, "model.temperature", "0.7").unwrap();
        assert_eq!(c2.model.temperature, 0.7);
        assert_eq!(c.model.temperature, 0.2, "input untouched");
        let c3 = set_by_path(&c2, "model.max_tokens", "4096").unwrap();
        assert_eq!(c3.model.max_tokens, 4096);
        let c4 = set_by_path(&c3, "tools.allow_shell", "OFF").unwrap();
        assert!(!c4.tools.allow_shell);
        let c5 = set_by_path(&c4, "tools.allow_shell", "yes").unwrap();
        assert!(c5.tools.allow_shell);
        let c6 = set_by_path(&c5, "model.model", "qwen3:8b").unwrap();
        assert_eq!(c6.model.model, "qwen3:8b");
        let c7 = set_by_path(&c6, "model.provider", "nvidia").unwrap();
        assert_eq!(c7.model.provider, ProviderKind::Nvidia);
        let c8 = set_by_path(&c7, "model.gpu_layers", "-1").unwrap();
        assert_eq!(c8.model.gpu_layers, -1);
        let c9 = set_by_path(&c8, "lmstudio.big_size_gb", "0x10").unwrap();
        assert_eq!(c9.lmstudio.big_size_gb, 16.0);
    }

    #[test]
    fn set_by_path_errors_like_the_ts() {
        let c = default_config();
        let e = set_by_path(&c, "model.temperature", "hot").unwrap_err().to_string();
        assert_eq!(e, "Expected a number, got \"hot\"");
        let e = set_by_path(&c, "tools.sandbox", "maybe").unwrap_err().to_string();
        assert_eq!(e, "Expected a boolean, got \"maybe\"");
        assert!(set_by_path(&c, "model.provider", "bogus").is_err());
        assert!(set_by_path(&c, "model.max_tokens", "1.5").is_err());
        assert!(set_by_path(&c, "model.fallback_models", "a,b").is_err());
    }

    #[test]
    fn set_by_path_unknown_key_is_stripped_and_tables_are_created() {
        let c = default_config();
        assert_eq!(set_by_path(&c, "model.nonexistent", "x").unwrap(), c);
        let c2 = set_by_path(&c, "mcp.servers.git.command", "uvx").unwrap();
        let git = &c2.mcp.servers["git"];
        assert_eq!(git.command, "uvx");
        assert!(git.enabled, "defaults fill the new server");
        let c3 = set_by_path(&c2, "mcp.servers.git.env.TOKEN", "t").unwrap();
        assert_eq!(c3.mcp.servers["git"].env["TOKEN"], "t");
        let c4 = set_by_path(&c3, "model.fallback_models.0", "qwen3:4b").unwrap();
        assert_eq!(c4.model.fallback_models[0], "qwen3:4b");
    }

    #[test]
    fn set_value_mutates_only_on_success() {
        let mut c = default_config();
        set_value(&mut c, "agent.candidates", "3").unwrap();
        assert_eq!(c.agent.candidates, 3);
        assert!(set_value(&mut c, "agent.candidates", "lots").is_err());
        assert_eq!(c.agent.candidates, 3);
    }

    #[test]
    fn js_number_semantics() {
        assert_eq!(js_number("12"), Some(12.0));
        assert_eq!(js_number(" 1.5 "), Some(1.5));
        assert_eq!(js_number(""), Some(0.0));
        assert_eq!(js_number("1e3"), Some(1000.0));
        assert_eq!(js_number(".5"), Some(0.5));
        assert_eq!(js_number("-2"), Some(-2.0));
        assert_eq!(js_number("0b101"), Some(5.0));
        assert_eq!(js_number("Infinity"), Some(f64::INFINITY));
        assert_eq!(js_number("inf"), None);
        assert_eq!(js_number("nan"), None);
        assert_eq!(js_number("1.2.3"), None);
        assert_eq!(js_number("12px"), None);
        assert_eq!(js_number("."), None);
        assert_eq!(js_number("1e"), None);
    }

    /// Read-only check that the real user-level config on this machine parses.
    /// Ignored by default (tests must not depend on the user's machine):
    /// `cargo test -p qo-core -- --ignored real_user_config`.
    #[test]
    #[ignore]
    fn real_user_config_loads() {
        let c = load_config(home_dir()).expect("user config should parse");
        assert!(!c.model.model.is_empty());
    }
}
