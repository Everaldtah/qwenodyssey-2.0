//! Qwenodyssey 2.0 core: the types and traits every other crate codes against.
//!
//! - [`types`]: messages, tool calls/specs, model options/results, pipeline types.
//! - [`provider`]: the [`Provider`] trait every model backend implements.
//! - [`tool`]: the [`Tool`] trait, [`ToolContext`] and [`ToolRegistry`].
//! - [`error`]: [`ProviderError`] plus the classifiers the chat layer uses
//!   (interrupt / unavailable / context overflow / no tool support).
//! - [`tokens`]: dependency-free token estimate (BPE-ish heuristic).
//! - [`model_ref`]: fallback/picker refs like `nvidia:<model>` or a bare Ollama tag.
//! - [`config`]: `.qwenodyssey/config.toml` (global + project layering).
//! - [`model_profile`]: model family detection + recommended sampling / thinking policy.
//! - [`parse`] / [`repair`]: tolerant JSON extraction and repair for small-model output.
//! - [`logger`]: JSONL session log under `.qwenodyssey/logs`.
//! - [`prompts`]: the embedded prompt templates + `{{placeholder}}` rendering.

pub mod config;
pub mod error;
pub mod logger;
pub mod model_profile;
pub mod model_ref;
pub mod parse;
pub mod prompts;
pub mod provider;
pub mod repair;
pub mod tokens;
pub mod tool;
pub mod types;

pub use error::ProviderError;
pub use model_ref::{ModelKind, ModelRef};
pub use provider::{HealthStatus, Provider, SharedProvider};
pub use tool::{Confirmer, Tool, ToolContext, ToolRegistry, ToolResult};
pub use types::*;

/// Re-exported so every crate uses the same cancellation primitive (Esc interrupt).
pub use tokio_util::sync::CancellationToken;
