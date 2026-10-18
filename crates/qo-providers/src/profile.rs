//! Thin adapter over the model-family profile ([`qo_core::model_profile`]):
//! the two questions the backends ask — "does this model have a toggleable
//! thinking mode?" and "should it think this turn?".

use qo_core::model_profile::{model_profile, resolve_thinking as resolve};

/// How the harness decides whether a thinking-capable model should think
/// (config `model.think`).
pub use qo_core::model_profile::ThinkMode;

/// Thinking can be toggled per request (Qwen3/3.5 hybrids).
pub(crate) fn hybrid_thinking(model: &str) -> bool {
    model_profile(model).hybrid_thinking
}

/// Should this model expose a chain-of-thought this turn? `None` for models
/// with no thinking mode at all (the flag then stays off the wire).
pub(crate) fn resolve_thinking(model: &str, mode: ThinkMode) -> Option<bool> {
    resolve(&model_profile(model), mode)
}
