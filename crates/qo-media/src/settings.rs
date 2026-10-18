//! The slice of `.qwenodyssey/config.toml` the media layer reads.
//!
//! The `[vision]` / `[audio]` / `[omni]` / `[tts]` sections are the config
//! crate's own structs (same fields, same TS defaults); the endpoint bits of
//! `[model]` and `[nvidia]` that vision/omni calls need are flattened next to
//! them, so every function here is testable without a full
//! [`qo_core::config::Config`]. [`MediaSettings::from_config`] is the only
//! place that knows the real `Config` layout.

use qo_core::config::Config;

pub use qo_core::config::{
    AudioConfig as AudioSettings, OmniConfig as OmniSettings, OmniProvider, TtsConfig as TtsSettings, TtsEngine,
    VisionConfig as VisionSettings, VisionProvider,
};

/// An OpenAI-compatible endpoint: base URL (with or without `/v1`) + bearer key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Endpoint {
    pub base_url: String,
    /// Empty = send no `Authorization` header.
    pub api_key: String,
}

/// Everything the media layer reads from the config.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaSettings {
    pub vision: VisionSettings,
    pub audio: AudioSettings,
    pub omni: OmniSettings,
    pub tts: TtsSettings,
    /// `[model].base_url` — the Ollama URL for `vision.provider = "ollama"` and
    /// the endpoint for the `openai` vision/omni providers.
    pub model_base_url: String,
    /// `[model].api_key` (falls back to `QWENODYSSEY_API_KEY` for openai calls).
    pub model_api_key: String,
    /// `[model].max_tokens` — reply budget in live mode.
    pub model_max_tokens: u32,
    /// `[nvidia].base_url`.
    pub nvidia_base_url: String,
    /// The NIM key, resolved like `resolveNvidiaKey`: `nvidia.api_key`, then the
    /// env var named by `nvidia.api_key_env`, then `NVIDIA_API_KEY`.
    pub nvidia_api_key: String,
}

impl Default for MediaSettings {
    fn default() -> Self {
        Self {
            vision: VisionSettings::default(),
            audio: AudioSettings::default(),
            omni: OmniSettings::default(),
            tts: TtsSettings::default(),
            model_base_url: "http://localhost:11434".into(),
            model_api_key: String::new(),
            model_max_tokens: 8192,
            nvidia_base_url: DEFAULT_NVIDIA_BASE.into(),
            nvidia_api_key: String::new(),
        }
    }
}

pub(crate) const DEFAULT_NVIDIA_BASE: &str = "https://integrate.api.nvidia.com";
pub(crate) const DEFAULT_OPENAI_BASE: &str = "https://api.openai.com";
pub(crate) const DEFAULT_OLLAMA_BASE: &str = "http://localhost:11434";

/// `config.model.api_key || process.env.QWENODYSSEY_API_KEY || ""`.
pub fn resolve_openai_key(model_api_key: &str, env_key: Option<String>) -> String {
    if !model_api_key.is_empty() {
        return model_api_key.to_string();
    }
    env_key.filter(|k| !k.is_empty()).unwrap_or_default()
}

/// `n.api_key || process.env[n.api_key_env || "NVIDIA_API_KEY"] || process.env.NVIDIA_API_KEY || ""`
/// (TS `resolveNvidiaKey`), with the environment injected for tests.
pub fn resolve_nvidia_key_with(api_key: &str, api_key_env: &str, env: impl Fn(&str) -> Option<String>) -> String {
    if !api_key.is_empty() {
        return api_key.to_string();
    }
    let name = if api_key_env.is_empty() { "NVIDIA_API_KEY" } else { api_key_env };
    env(name).filter(|k| !k.is_empty()).or_else(|| env("NVIDIA_API_KEY").filter(|k| !k.is_empty())).unwrap_or_default()
}

fn or_default(s: &str, default: &str) -> String {
    if s.is_empty() {
        default.to_string()
    } else {
        s.to_string()
    }
}

impl MediaSettings {
    /// Build the media slice from the full config (the NIM key is resolved
    /// from the process environment here).
    pub fn from_config(config: &Config) -> Self {
        let n = &config.nvidia;
        Self {
            vision: config.vision.clone(),
            audio: config.audio.clone(),
            omni: config.omni.clone(),
            tts: config.tts.clone(),
            model_base_url: config.model.base_url.clone(),
            model_api_key: config.model.api_key.clone(),
            model_max_tokens: config.model.max_tokens,
            nvidia_base_url: n.base_url.clone(),
            nvidia_api_key: resolve_nvidia_key_with(&n.api_key, &n.api_key_env, |k| std::env::var(k).ok()),
        }
    }

    /// NIM endpoint (`nvidia.base_url` || the public NIM URL, resolved NIM key).
    fn nvidia_endpoint(&self) -> Endpoint {
        Endpoint { base_url: or_default(&self.nvidia_base_url, DEFAULT_NVIDIA_BASE), api_key: self.nvidia_api_key.clone() }
    }

    /// OpenAI-compatible endpoint (`model.base_url` || api.openai.com,
    /// `model.api_key` || `QWENODYSSEY_API_KEY`).
    fn openai_endpoint(&self) -> Endpoint {
        Endpoint {
            base_url: or_default(&self.model_base_url, DEFAULT_OPENAI_BASE),
            api_key: resolve_openai_key(&self.model_api_key, std::env::var("QWENODYSSEY_API_KEY").ok()),
        }
    }

    /// Where `describe_image` sends the frame. For Ollama the base is
    /// `model.base_url` || localhost:11434 (no key).
    pub fn vision_endpoint(&self) -> Endpoint {
        match self.vision.provider {
            VisionProvider::Nvidia => self.nvidia_endpoint(),
            VisionProvider::OpenAi => self.openai_endpoint(),
            VisionProvider::Ollama => {
                Endpoint { base_url: or_default(&self.model_base_url, DEFAULT_OLLAMA_BASE), api_key: String::new() }
            }
        }
    }

    /// Where `ask_omni` sends the audio (+ frame).
    pub fn omni_endpoint(&self) -> Endpoint {
        match self.omni.provider {
            OmniProvider::Nvidia => self.nvidia_endpoint(),
            OmniProvider::OpenAi => self.openai_endpoint(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_ts() {
        let s = MediaSettings::default();
        assert!(s.vision.enabled && s.vision.confirm);
        assert_eq!(s.vision.model, "nvidia/nemotron-nano-12b-v2-vl");
        assert_eq!(s.audio.max_seconds, 30);
        assert_eq!(s.audio.vad_threshold, 0.6);
        assert_eq!(s.audio.language, "en");
        assert!(!s.omni.enabled && s.omni.send_image);
        assert_eq!(s.omni.model, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning");
        assert_eq!(s.tts.engine, TtsEngine::Piper);
    }

    #[test]
    fn from_config_copies_sections_and_endpoints() {
        let mut c = Config::default();
        c.vision.provider = VisionProvider::Ollama;
        c.vision.model = "moondream".into();
        c.audio.device = "Mic X".into();
        c.omni.enabled = true;
        c.tts.engine = TtsEngine::Sapi;
        c.model.base_url = "http://127.0.0.1:1234".into();
        c.model.max_tokens = 2048;
        c.nvidia.base_url = "https://nim.example".into();
        c.nvidia.api_key = "nv-cfg".into();
        let s = MediaSettings::from_config(&c);
        assert_eq!(s.vision.model, "moondream");
        assert_eq!(s.audio.device, "Mic X");
        assert!(s.omni.enabled);
        assert_eq!(s.tts.engine, TtsEngine::Sapi);
        assert_eq!(s.model_max_tokens, 2048);
        assert_eq!(s.vision_endpoint(), Endpoint { base_url: "http://127.0.0.1:1234".into(), api_key: String::new() });
        assert_eq!(s.omni_endpoint(), Endpoint { base_url: "https://nim.example".into(), api_key: "nv-cfg".into() });
    }

    #[test]
    fn nvidia_key_resolution_order() {
        let env = |k: &str| match k {
            "MY_NIM" => Some("from-custom".to_string()),
            "NVIDIA_API_KEY" => Some("from-default".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        };
        assert_eq!(resolve_nvidia_key_with("cfg", "MY_NIM", env), "cfg");
        assert_eq!(resolve_nvidia_key_with("", "MY_NIM", env), "from-custom");
        assert_eq!(resolve_nvidia_key_with("", "", env), "from-default");
        assert_eq!(resolve_nvidia_key_with("", "EMPTY", env), "from-default");
        assert_eq!(resolve_nvidia_key_with("", "MISSING", env), "from-default");
        assert_eq!(resolve_nvidia_key_with("", "MISSING", |_| None), "");
    }

    #[test]
    fn endpoints() {
        let mut s = MediaSettings { nvidia_api_key: "nv-key".into(), ..Default::default() };
        assert_eq!(
            s.vision_endpoint(),
            Endpoint { base_url: "https://integrate.api.nvidia.com".into(), api_key: "nv-key".into() }
        );
        s.nvidia_base_url.clear();
        assert_eq!(s.omni_endpoint().base_url, DEFAULT_NVIDIA_BASE);

        s.vision.provider = VisionProvider::Ollama;
        s.model_base_url = "http://127.0.0.1:11434/".into();
        let e = s.vision_endpoint();
        assert_eq!(e.base_url, "http://127.0.0.1:11434/");
        assert!(e.api_key.is_empty());
        s.model_base_url.clear();
        assert_eq!(s.vision_endpoint().base_url, DEFAULT_OLLAMA_BASE);

        s.vision.provider = VisionProvider::OpenAi;
        s.model_api_key = "sk-x".into();
        assert_eq!(s.vision_endpoint(), Endpoint { base_url: DEFAULT_OPENAI_BASE.into(), api_key: "sk-x".into() });
    }

    #[test]
    fn openai_key_fallback() {
        assert_eq!(resolve_openai_key("cfg", Some("env".into())), "cfg");
        assert_eq!(resolve_openai_key("", Some("env".into())), "env");
        assert_eq!(resolve_openai_key("", Some(String::new())), "");
        assert_eq!(resolve_openai_key("", None), "");
    }
}
