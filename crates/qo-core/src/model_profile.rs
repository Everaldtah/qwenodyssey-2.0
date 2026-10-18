//! Model profiles: what a given model IS, and how it should be sampled
//! (ported from `src/core/modelProfile.ts`).
//!
//! Small local models (≤ ~9B) live or die by their decoding parameters. Running
//! Qwen3/3.5 greedily makes it repeat itself; running qwen2.5-coder hot makes it
//! botch tool calls. Rather than one global temperature for every backend, we
//! classify the model by family and hand back the sampling its authors actually
//! recommend, split by whether the model is thinking this turn.
//!
//! Everything here is derived from the model NAME alone (no network calls), so it
//! works for Ollama tags, LM Studio keys and hosted refs alike.

use std::sync::LazyLock;

use regex::Regex;

/// How the harness decides whether a thinking-capable model should think
/// (the same enum as `model.think` in the config).
pub use crate::config::ThinkMode;

/// Recommended decoding parameters for one thinking state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingProfile {
    /// Base temperature for one-shot generation (edit pipeline, summaries, …).
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub presence_penalty: Option<f64>,
    /// Ollama `repeat_penalty` — the main anti-loop lever for small models.
    pub repeat_penalty: Option<f64>,
    /// Floor for agent TOOL turns. Tool turns want determinism, but families that
    /// degenerate at temperature 0 (Qwen3, R1, QwQ) need a little heat regardless.
    pub min_turn_temp: f64,
}

/// What a model is, classified from its name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelProfile {
    /// Family slug, e.g. "qwen3" / "qwen2.5-coder" / "deepseek-r1".
    pub family: &'static str,
    /// Trained to deliberate with an internal chain-of-thought.
    pub reasoning: bool,
    /// Thinking can be toggled per request (Qwen3/3.5 hybrids: Ollama `think`,
    /// or `chat_template_kwargs.enable_thinking` on OpenAI-compatible servers).
    /// Pure reasoning models (R1, QwQ) always think and must NOT be sent a
    /// disable flag.
    pub hybrid_thinking: bool,
    /// What think mode "auto" resolves to for this family.
    pub think_by_default: bool,
    /// Recommended sampling while thinking.
    pub thinking: SamplingProfile,
    /// Recommended sampling while not thinking.
    pub non_thinking: SamplingProfile,
    /// Context window the family can comfortably use (num_ctx hint).
    pub context_suggestion: u32,
}

/// Generic decoding for a non-reasoning instruct model.
pub const GENERIC: SamplingProfile = SamplingProfile {
    temperature: 0.2,
    top_p: 0.9,
    top_k: None,
    presence_penalty: None,
    repeat_penalty: None,
    min_turn_temp: 0.0,
};

/// Deliberating models loop at temperature 0; ~0.6 is the published guidance.
pub const REASONING_SAMPLING: SamplingProfile = SamplingProfile {
    temperature: 0.6,
    top_p: 0.95,
    top_k: None,
    presence_penalty: None,
    repeat_penalty: None,
    min_turn_temp: 0.6,
};

/// Qwen3 / Qwen3.5 THINKING mode — the sampling Qwen publishes for these hybrids.
/// Greedy decoding is explicitly discouraged: it causes endless repetition.
pub const QWEN3_THINKING: SamplingProfile = SamplingProfile {
    temperature: 0.6,
    top_p: 0.95,
    top_k: Some(20),
    presence_penalty: None,
    repeat_penalty: Some(1.05),
    min_turn_temp: 0.6,
};

/// Qwen3 / Qwen3.5 NON-thinking mode — what the agent loop uses by default, since
/// a coding/shell harness wants fast, tool-shaped turns and already scaffolds its
/// own reasoning (DEEP_THINK + the `think` tool). The presence penalty is Qwen's
/// recommended anti-repetition setting for this mode; the 0.3 turn floor keeps
/// tool calls near-deterministic without tipping into a greedy loop.
pub const QWEN3_NON_THINKING: SamplingProfile = SamplingProfile {
    temperature: 0.7,
    top_p: 0.8,
    top_k: Some(20),
    presence_penalty: Some(1.5),
    repeat_penalty: Some(1.05),
    min_turn_temp: 0.3,
};

/// Qwen2.x (incl. the coder tunes): stable under greedy decoding, which is the
/// best setting for tool adherence, so the turn floor stays at 0.
pub const QWEN2_SAMPLING: SamplingProfile = SamplingProfile {
    temperature: 0.2,
    top_p: 0.8,
    top_k: Some(20),
    presence_penalty: None,
    repeat_penalty: Some(1.05),
    min_turn_temp: 0.0,
};

/// Google's published defaults for Gemma.
pub const GEMMA_SAMPLING: SamplingProfile = SamplingProfile {
    temperature: 0.2,
    top_p: 0.95,
    top_k: Some(64),
    presence_penalty: None,
    repeat_penalty: Some(1.05),
    min_turn_temp: 0.0,
};

/// The base profile every family starts from (TS `profileOf` defaults).
const BASE: ModelProfile = ModelProfile {
    family: "generic",
    reasoning: false,
    hybrid_thinking: false,
    think_by_default: false,
    thinking: REASONING_SAMPLING,
    non_thinking: GENERIC,
    context_suggestion: 16384,
};

fn re(pattern: &str) -> Regex {
    // The patterns are compile-time constants covered by the tests below.
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad model-profile regex {pattern}: {e}"))
}

/// Matches qwen3, qwen-3, qwen3.5, qwen_3.5-9b, qwen3.5-coder…
static QWEN3_RE: LazyLock<Regex> = LazyLock::new(|| re(r"qwen[\s._-]?3(\.\d+)?"));
static DEEPSEEK_R1_RE: LazyLock<Regex> = LazyLock::new(|| re(r"deepseek[\s._-]?r1"));
static BARE_R1_RE: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[-_/:.])r1([-_/:.]|$)"));
static REASONER_RE: LazyLock<Regex> =
    LazyLock::new(|| re(r"(^|[-_/:.])(o1|o3|kimi|k2|nemotron|thinking|reason)"));
static DEEPSEEK_V34_RE: LazyLock<Regex> = LazyLock::new(|| re(r"deepseek[\s._-]?v[34]"));

/// Classify a model by name. Order matters: the most specific families are tested
/// first (qwen3 before the generic qwen rule, deepseek-r1 before deepseek).
pub fn model_profile(model: &str) -> ModelProfile {
    let m = model.to_lowercase();

    // ── Qwen 3 / 3.5 (incl. 3.5 9B) — hybrid thinking ────────────────────────
    if QWEN3_RE.is_match(&m) {
        return ModelProfile {
            family: if m.contains("coder") { "qwen3-coder" } else { "qwen3" },
            reasoning: true,
            hybrid_thinking: true,
            // Auto = think OFF: for coding/shell turns this is markedly faster and
            // sticks to tool calls better. The harness still reasons via its own
            // scaffold, and `model.think = "always"` restores native deliberation.
            think_by_default: false,
            thinking: QWEN3_THINKING,
            non_thinking: QWEN3_NON_THINKING,
            context_suggestion: 32768,
        };
    }

    // ── DeepSeek-R1 / QwQ — always-thinking reasoners ────────────────────────
    if DEEPSEEK_R1_RE.is_match(&m) || BARE_R1_RE.is_match(&m) {
        return ModelProfile {
            family: "deepseek-r1",
            reasoning: true,
            think_by_default: true,
            thinking: REASONING_SAMPLING,
            non_thinking: REASONING_SAMPLING,
            context_suggestion: 32768,
            ..BASE
        };
    }
    if m.contains("qwq") {
        return ModelProfile {
            family: "qwq",
            reasoning: true,
            think_by_default: true,
            context_suggestion: 32768,
            ..BASE
        };
    }

    // ── Qwen 2.x ─────────────────────────────────────────────────────────────
    if m.contains("qwen") {
        return ModelProfile {
            family: if m.contains("coder") { "qwen2.5-coder" } else { "qwen2.5" },
            thinking: QWEN2_SAMPLING,
            non_thinking: QWEN2_SAMPLING,
            context_suggestion: 32768,
            ..BASE
        };
    }

    // ── Other deliberating models (hosted or local) ──────────────────────────
    // Keeps parity with the classifier this replaced: kimi/k2, nemotron, o1/o3,
    // and anything self-describing as thinking/reasoning.
    if REASONER_RE.is_match(&m) || DEEPSEEK_V34_RE.is_match(&m) {
        return ModelProfile {
            family: "reasoning",
            reasoning: true,
            think_by_default: true,
            context_suggestion: 32768,
            ..BASE
        };
    }

    // ── Gemma ────────────────────────────────────────────────────────────────
    if m.contains("gemma") {
        return ModelProfile {
            family: "gemma",
            thinking: GEMMA_SAMPLING,
            non_thinking: GEMMA_SAMPLING,
            context_suggestion: 32768,
            ..BASE
        };
    }

    // ── Everything else: llama / mistral / codestral / phi / granite / … ─────
    ModelProfile { family: "generic", thinking: GENERIC, non_thinking: GENERIC, ..BASE }
}

/// Should this model expose a chain-of-thought this turn?
/// Returns `None` for models with no thinking mode at all, so callers know to
/// omit the flag entirely rather than sending `think: false` (which some
/// backends reject for non-thinking models).
pub fn resolve_thinking(profile: &ModelProfile, mode: ThinkMode) -> Option<bool> {
    if !profile.reasoning {
        return None;
    }
    match mode {
        ThinkMode::Always => Some(true),
        // A pure reasoning model always thinks — asking it not to is not supported.
        ThinkMode::Never => Some(!profile.hybrid_thinking),
        ThinkMode::Auto => Some(profile.think_by_default),
    }
}

/// The sampling profile that applies given the resolved thinking state.
pub fn sampling_for(profile: &ModelProfile, thinking: Option<bool>) -> SamplingProfile {
    if thinking == Some(true) {
        profile.thinking
    } else {
        profile.non_thinking
    }
}

/// Convenience for callers that only have a model name and the configured think
/// mode: the sampling actually in force.
pub fn sampling_for_model(model: &str, mode: ThinkMode) -> SamplingProfile {
    let p = model_profile(model);
    sampling_for(&p, resolve_thinking(&p, mode))
}

/// Trained to deliberate with an internal chain-of-thought.
pub fn is_reasoning_model(model: &str) -> bool {
    model_profile(model).reasoning
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── model family detection ──

    #[test]
    fn classifies_qwen35_9b_as_hybrid_thinking() {
        let p = model_profile("qwen3.5:9b");
        assert_eq!(p.family, "qwen3");
        assert!(p.reasoning);
        assert!(p.hybrid_thinking);
        // Coding/shell turns default to thinking OFF for speed + tool adherence.
        assert!(!p.think_by_default);
    }

    #[test]
    fn matches_qwen3_family_across_naming_styles() {
        for tag in ["qwen3:8b", "qwen-3.5-9b", "Qwen3.5-Coder-9B-GGUF", "qwen_3.5:9b-q4"] {
            assert!(model_profile(tag).hybrid_thinking, "{tag}");
        }
        assert_eq!(model_profile("Qwen3.5-Coder-9B-GGUF").family, "qwen3-coder");
    }

    #[test]
    fn keeps_qwen2_separate_and_greedy_stable() {
        let p = model_profile("qwen2.5-coder:7b");
        assert_eq!(p.family, "qwen2.5-coder");
        assert!(!p.reasoning);
        assert_eq!(p.non_thinking.min_turn_temp, 0.0);
        assert_eq!(model_profile("qwen2.5:7b").family, "qwen2.5");
    }

    #[test]
    fn preserves_reasoning_classification() {
        for m in [
            "deepseek-r1:7b",
            "qwq:32b",
            "moonshotai/kimi-k2.6",
            "nvidia/nemotron-nano-12b",
            "o1-mini",
            "o3",
            "some-thinking-model",
        ] {
            assert!(is_reasoning_model(m), "{m}");
        }
        for m in ["qwen2.5:7b", "llama3.1:8b", "gemma3:12b", "mistral:7b"] {
            assert!(!is_reasoning_model(m), "{m}");
        }
    }

    #[test]
    fn families_for_other_shapes() {
        assert_eq!(model_profile("deepseek-ai/deepseek-v4-pro").family, "reasoning");
        assert_eq!(model_profile("nvidia:deepseek-ai/deepseek-v4-pro").family, "reasoning");
        assert_eq!(model_profile("hf.co/foo/r1:latest").family, "deepseek-r1");
        assert_eq!(model_profile("igorls/gemma-4-12B-it-heretic-GGUF").family, "gemma");
        assert_eq!(model_profile("").family, "generic");
        assert_eq!(model_profile("llama3.1:8b").context_suggestion, 16384);
        assert_eq!(model_profile("qwq:32b").thinking, REASONING_SAMPLING);
        assert_eq!(model_profile("qwq:32b").non_thinking, GENERIC);
    }

    #[test]
    fn gemma_gets_googles_top_k() {
        assert_eq!(model_profile("gemma3:12b").non_thinking.top_k, Some(64));
    }

    // ── thinking policy ──

    #[test]
    fn auto_turns_qwen3_off_but_reasoners_think() {
        assert_eq!(resolve_thinking(&model_profile("qwen3.5:9b"), ThinkMode::Auto), Some(false));
        assert_eq!(resolve_thinking(&model_profile("deepseek-r1:7b"), ThinkMode::Auto), Some(true));
    }

    #[test]
    fn always_never_override_where_supported() {
        assert_eq!(resolve_thinking(&model_profile("qwen3.5:9b"), ThinkMode::Always), Some(true));
        assert_eq!(resolve_thinking(&model_profile("qwen3.5:9b"), ThinkMode::Never), Some(false));
        // R1 has no off switch — "never" must not produce an unsupported request.
        assert_eq!(resolve_thinking(&model_profile("deepseek-r1:7b"), ThinkMode::Never), Some(true));
    }

    #[test]
    fn none_for_models_without_thinking() {
        assert_eq!(resolve_thinking(&model_profile("qwen2.5-coder:7b"), ThinkMode::Always), None);
    }

    #[test]
    fn swaps_sampling_by_thinking_state() {
        let p = model_profile("qwen3.5:9b");
        let t = sampling_for(&p, Some(true));
        assert_eq!((t.temperature, t.top_p, t.min_turn_temp), (0.6, 0.95, 0.6));
        let n = sampling_for(&p, Some(false));
        assert_eq!(n.temperature, 0.7);
        assert_eq!(n.top_p, 0.8);
        assert_eq!(n.top_k, Some(20));
        assert_eq!(n.presence_penalty, Some(1.5));
        assert_eq!(n.min_turn_temp, 0.3);
        assert_eq!(sampling_for(&p, None), n);
        // Never greedy: Qwen3 loops at temperature 0.
        assert!(sampling_for_model("qwen3.5:9b", ThinkMode::Auto).min_turn_temp > 0.0);
    }
}
