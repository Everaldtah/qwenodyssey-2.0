//! Model refs as used by `[model].fallback_models`, the `/model` picker and the
//! swarm: `"lmstudio:<key>"`, `"nvidia:<model>"`, `"openrouter:<model>"`,
//! `"anthropic:<model>"`, else a bare Ollama tag.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Ollama,
    LmStudio,
    Nvidia,
    OpenRouter,
    Anthropic,
}

impl ModelKind {
    /// Provider name this kind maps to (matches `Provider::name()`).
    pub fn provider_name(&self) -> &'static str {
        match self {
            ModelKind::Ollama => "ollama",
            ModelKind::LmStudio => "lmstudio",
            ModelKind::Nvidia => "nvidia",
            ModelKind::OpenRouter => "openrouter",
            ModelKind::Anthropic => "anthropic",
        }
    }

    /// Reached over the internet (probed after a manual switch, needs a key).
    pub fn is_cloud(&self) -> bool {
        matches!(self, ModelKind::Nvidia | ModelKind::OpenRouter | ModelKind::Anthropic)
    }

    pub fn from_provider_name(name: &str) -> Option<Self> {
        Some(match name {
            "ollama" => ModelKind::Ollama,
            "lmstudio" => ModelKind::LmStudio,
            "nvidia" => ModelKind::Nvidia,
            "openrouter" => ModelKind::OpenRouter,
            "anthropic" => ModelKind::Anthropic,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelRef {
    pub kind: ModelKind,
    pub model: String,
}

impl ModelRef {
    pub fn parse(r: &str) -> Self {
        let r = r.trim();
        for (prefix, kind) in [
            ("lmstudio:", ModelKind::LmStudio),
            ("nvidia:", ModelKind::Nvidia),
            ("openrouter:", ModelKind::OpenRouter),
            ("anthropic:", ModelKind::Anthropic),
        ] {
            if let Some(model) = r.strip_prefix(prefix) {
                return ModelRef { kind, model: model.to_string() };
            }
        }
        ModelRef { kind: ModelKind::Ollama, model: r.to_string() }
    }

    /// The ref of an active provider/model pair (bare tag for Ollama).
    pub fn of(provider_name: &str, model: &str) -> String {
        if provider_name == "ollama" {
            model.to_string()
        } else {
            format!("{provider_name}:{model}")
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&ModelRef::of(self.kind.provider_name(), &self.model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let r = ModelRef::parse("nvidia:z-ai/glm-5.3");
        assert_eq!(r.kind, ModelKind::Nvidia);
        assert_eq!(r.model, "z-ai/glm-5.3");
        assert_eq!(r.to_string(), "nvidia:z-ai/glm-5.3");
        let r = ModelRef::parse("qwen2.5-coder:7b");
        assert_eq!(r.kind, ModelKind::Ollama);
        assert_eq!(r.to_string(), "qwen2.5-coder:7b");
    }
}
