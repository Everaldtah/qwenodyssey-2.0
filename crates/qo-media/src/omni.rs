//! Omni call: send any combination of text + audio + image to a single
//! multimodal model (NVIDIA NIM / OpenAI-compatible) and get a reply. This is
//! the "brain does everything" path — the model ingests the raw mic audio and
//! camera frame itself, instead of local whisper + a separate vision model.
//!
//! Uses the OpenAI multimodal content format:
//! `content: [ {type:"text"}, {type:"input_audio", input_audio:{data,format}},
//!             {type:"image_url", image_url:{url}} ]`

use std::path::PathBuf;

use anyhow::Result;
use base64::Engine as _;
use qo_core::CancellationToken;
use serde_json::{json, Value};

use crate::http::{api_base, openai_content, post_json, REQUEST_TIMEOUT};
use crate::settings::MediaSettings;
use crate::vision::{data_uri, read_file};

/// Error bodies are clipped to this many characters.
const ERROR_BODY_CHARS: usize = 300;

/// One omni request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OmniInput {
    pub model: String,
    /// Instruction/text part ("" = no text part).
    pub text: String,
    pub audio_path: Option<PathBuf>,
    /// "wav" | "mp3" | … (default "wav").
    pub audio_format: Option<String>,
    pub image_path: Option<PathBuf>,
    /// Default 512.
    pub max_tokens: Option<u32>,
    /// Default 0.3.
    pub temperature: Option<f64>,
}

/// Build the `/chat/completions` body from already-read audio/image bytes.
pub fn omni_body(input: &OmniInput, audio: Option<&[u8]>, image: Option<&[u8]>) -> Value {
    let b64 = &base64::engine::general_purpose::STANDARD;
    let mut content: Vec<Value> = Vec::new();
    if !input.text.is_empty() {
        content.push(json!({ "type": "text", "text": input.text }));
    }
    if let Some(a) = audio {
        let format = input.audio_format.as_deref().filter(|f| !f.is_empty()).unwrap_or("wav");
        content.push(json!({ "type": "input_audio", "input_audio": { "data": b64.encode(a), "format": format } }));
    }
    if let (Some(img), Some(path)) = (image, input.image_path.as_deref()) {
        content.push(json!({ "type": "image_url", "image_url": { "url": data_uri(path, img) } }));
    }
    json!({
        "model": input.model,
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": input.max_tokens.unwrap_or(512),
        "temperature": input.temperature.unwrap_or(0.3),
    })
}

/// Send `input` to the omni model on the configured endpoint (`[omni].provider`)
/// and return the trimmed reply. Cancellable.
pub async fn ask_omni(input: &OmniInput, settings: &MediaSettings, cancel: &CancellationToken) -> Result<String> {
    let endpoint = settings.omni_endpoint();
    let audio = match &input.audio_path {
        Some(p) => Some(read_file(p, "audio").await?),
        None => None,
    };
    let image = match &input.image_path {
        Some(p) => Some(read_file(p, "image").await?),
        None => None,
    };
    let body = omni_body(input, audio.as_deref(), image.as_deref());
    let label = format!("omni ({})", settings.omni.provider.as_str());
    let json = post_json(
        &format!("{}/chat/completions", api_base(&endpoint.base_url)),
        &endpoint.api_key,
        &body,
        &label,
        ERROR_BODY_CHARS,
        REQUEST_TIMEOUT,
        cancel,
    )
    .await?;
    Ok(openai_content(&json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::mock;
    use crate::settings::OmniProvider;

    #[test]
    fn body_parts_only_for_present_inputs() {
        let input = OmniInput { model: "m".into(), ..Default::default() };
        let b = omni_body(&input, None, None);
        assert_eq!(b, json!({"model": "m", "messages": [{"role": "user", "content": []}], "max_tokens": 512, "temperature": 0.3}));

        let input = OmniInput {
            model: "m".into(),
            text: "hi".into(),
            audio_path: Some("a.wav".into()),
            audio_format: Some("mp3".into()),
            image_path: Some("f.PNG".into()),
            max_tokens: Some(8192),
            temperature: Some(0.5),
        };
        let b = omni_body(&input, Some(b"RIFF"), Some(b"img"));
        let c = &b["messages"][0]["content"];
        assert_eq!(c[0], json!({"type": "text", "text": "hi"}));
        assert_eq!(c[1], json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "mp3"}}));
        assert_eq!(c[2], json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}));
        assert_eq!(b["max_tokens"], 8192);
        assert_eq!(b["temperature"], 0.5);
    }

    #[tokio::test]
    async fn sends_audio_and_parses_reply() {
        let (base, rx) = mock::serve(200, r#"{"choices":[{"message":{"content":"You said hello. Hi!"}}]}"#).await;
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("utt-1.wav");
        std::fs::write(&wav, crate::micstream::wav_bytes(&[0, 0, 1, 0])).unwrap();
        let mut settings = MediaSettings { model_base_url: format!("{base}/v1"), model_api_key: "sk-o".into(), ..Default::default() };
        settings.omni.provider = OmniProvider::OpenAi;
        let input = OmniInput {
            model: "phi-4-multimodal".into(),
            text: "repeat what you heard".into(),
            audio_path: Some(wav),
            ..Default::default()
        };
        let reply = ask_omni(&input, &settings, &CancellationToken::new()).await.unwrap();
        assert_eq!(reply, "You said hello. Hi!");
        let req = rx.await.unwrap();
        assert_eq!(req.request_line, "POST /v1/chat/completions HTTP/1.1");
        assert_eq!(req.header("authorization"), Some("Bearer sk-o"));
        let c = &req.body["messages"][0]["content"];
        assert_eq!(c.as_array().map(Vec::len), Some(2));
        assert_eq!(c[1]["input_audio"]["format"], "wav");
        assert!(c[1]["input_audio"]["data"].as_str().unwrap().starts_with("UklGR"));
    }

    #[tokio::test]
    async fn http_error_mentions_provider() {
        let (base, _rx) = mock::serve(400, "audio input not supported").await;
        let settings = MediaSettings { nvidia_base_url: base, ..Default::default() };
        let input = OmniInput { model: "x".into(), text: "t".into(), ..Default::default() };
        let e = ask_omni(&input, &settings, &CancellationToken::new()).await.unwrap_err();
        assert_eq!(e.to_string(), "omni (nvidia) HTTP 400: audio input not supported");
    }
}
