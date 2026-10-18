//! Vision: describe a captured image with a vision-language model. NVIDIA NIM
//! and other OpenAI-compatible endpoints take an `image_url` data URI; Ollama
//! takes a base64 `images` array. Returns the model's text description.

use std::path::Path;

use anyhow::{anyhow, Result};
use base64::Engine as _;
use qo_core::CancellationToken;
use serde_json::{json, Value};

use crate::http::{api_base, openai_content, post_json, REQUEST_TIMEOUT};
use crate::settings::{MediaSettings, VisionProvider};

/// Used when the caller passes no (or a blank) question.
pub const DEFAULT_PROMPT: &str = "Describe what you see in this webcam image concisely and concretely: people, objects, \
text, gestures, and anything notable. If the user asked something specific, answer that.";

/// Error bodies are clipped to this many characters.
const ERROR_BODY_CHARS: usize = 200;

/// `image/png` for a `.png` path (any case), else `image/jpeg`.
pub fn image_mime_ext(path: &Path) -> &'static str {
    let s = path.to_string_lossy();
    let is_png = s.len() >= 4 && s.is_char_boundary(s.len() - 4) && s[s.len() - 4..].eq_ignore_ascii_case(".png");
    if is_png {
        "png"
    } else {
        "jpeg"
    }
}

/// `data:image/<ext>;base64,<bytes>`.
pub fn data_uri(path: &Path, bytes: &[u8]) -> String {
    format!("data:image/{};base64,{}", image_mime_ext(path), base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub(crate) async fn read_file(path: &Path, what: &str) -> Result<Vec<u8>> {
    tokio::fs::read(path).await.map_err(|e| anyhow!("cannot read {what} {}: {e}", path.display()))
}

/// The question actually asked: the trimmed prompt, else [`DEFAULT_PROMPT`].
pub fn vision_question(prompt: &str) -> String {
    let q = prompt.trim();
    if q.is_empty() {
        DEFAULT_PROMPT.to_string()
    } else {
        q.to_string()
    }
}

/// Ollama `/api/chat` body (non-streaming, base64 `images`).
pub fn ollama_vision_body(model: &str, question: &str, image: &[u8]) -> Value {
    json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": question,
            "images": [base64::engine::general_purpose::STANDARD.encode(image)],
        }],
        "stream": false,
    })
}

/// OpenAI-style `/chat/completions` body with an `image_url` data URI.
pub fn openai_vision_body(model: &str, question: &str, image_data_uri: &str) -> Value {
    json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": question },
                { "type": "image_url", "image_url": { "url": image_data_uri } },
            ],
        }],
        "max_tokens": 1024,
        "temperature": 0.2,
    })
}

/// Describe `image` with the configured vision model (`[vision]`). `prompt` is
/// the user's question (blank → [`DEFAULT_PROMPT`]). Cancellable.
pub async fn describe_image(
    image: &Path,
    prompt: &str,
    settings: &MediaSettings,
    cancel: &CancellationToken,
) -> Result<String> {
    let v = &settings.vision;
    let question = vision_question(prompt);
    let endpoint = settings.vision_endpoint();
    let bytes = read_file(image, "image").await?;

    if v.provider == VisionProvider::Ollama {
        let base = endpoint.base_url.trim_end_matches('/');
        let json = post_json(
            &format!("{base}/api/chat"),
            "",
            &ollama_vision_body(&v.model, &question, &bytes),
            "vision (ollama)",
            ERROR_BODY_CHARS,
            REQUEST_TIMEOUT,
            cancel,
        )
        .await?;
        return Ok(json.pointer("/message/content").and_then(Value::as_str).unwrap_or("").trim().to_string());
    }

    // nvidia / openai-compatible: image_url data URI.
    let label = format!("vision ({})", v.provider.as_str());
    let json = post_json(
        &format!("{}/chat/completions", api_base(&endpoint.base_url)),
        &endpoint.api_key,
        &openai_vision_body(&v.model, &question, &data_uri(image, &bytes)),
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

    fn jpeg(dir: &Path) -> std::path::PathBuf {
        let p = dir.join("frame-1.jpg");
        std::fs::write(&p, [0xFFu8, 0xD8, 0xFF, 0xE0]).unwrap();
        p
    }

    #[test]
    fn mime_and_data_uri() {
        assert_eq!(image_mime_ext(Path::new("a.PNG")), "png");
        assert_eq!(image_mime_ext(Path::new("a.jpg")), "jpeg");
        assert_eq!(data_uri(Path::new("x.png"), b"hi"), "data:image/png;base64,aGk=");
        assert_eq!(vision_question("  what colour is it? "), "what colour is it?");
        assert_eq!(vision_question("   "), DEFAULT_PROMPT);
    }

    #[tokio::test]
    async fn nvidia_request_shape_and_reply() {
        let (base, rx) = mock::serve(200, r#"{"choices":[{"message":{"content":"  A person waving.  "}}]}"#).await;
        let dir = tempfile::tempdir().unwrap();
        let img = jpeg(dir.path());
        let settings = MediaSettings { nvidia_base_url: base, nvidia_api_key: "nv-123".into(), ..Default::default() };
        let out = describe_image(&img, "", &settings, &CancellationToken::new()).await.unwrap();
        assert_eq!(out, "A person waving.");

        let req = rx.await.unwrap();
        assert_eq!(req.request_line, "POST /v1/chat/completions HTTP/1.1");
        assert_eq!(req.header("authorization"), Some("Bearer nv-123"));
        let b = &req.body;
        assert_eq!(b["model"], "nvidia/nemotron-nano-12b-v2-vl");
        assert_eq!(b["max_tokens"], 1024);
        assert_eq!(b["temperature"], 0.2);
        let content = &b["messages"][0]["content"];
        assert_eq!(content[0], json!({"type": "text", "text": DEFAULT_PROMPT}));
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/jpeg;base64,/9j/4A==");
    }

    #[tokio::test]
    async fn ollama_request_shape_and_reply() {
        let (base, rx) = mock::serve(200, r#"{"message":{"role":"assistant","content":"a mug\n"}}"#).await;
        let dir = tempfile::tempdir().unwrap();
        let img = jpeg(dir.path());
        let mut settings = MediaSettings { model_base_url: format!("{base}/"), ..Default::default() };
        settings.vision.provider = VisionProvider::Ollama;
        settings.vision.model = "moondream".into();
        let out = describe_image(&img, "what is on the desk?", &settings, &CancellationToken::new()).await.unwrap();
        assert_eq!(out, "a mug");
        let req = rx.await.unwrap();
        assert_eq!(req.request_line, "POST /api/chat HTTP/1.1");
        assert_eq!(req.header("authorization"), None);
        assert_eq!(
            req.body,
            json!({
                "model": "moondream",
                "messages": [{"role": "user", "content": "what is on the desk?", "images": ["/9j/4A=="]}],
                "stream": false
            })
        );
    }

    #[tokio::test]
    async fn http_failure_message_matches_ts() {
        let (base, _rx) = mock::serve(404, r#"{"error":"model not found"}"#).await;
        let dir = tempfile::tempdir().unwrap();
        let img = jpeg(dir.path());
        let mut settings = MediaSettings { model_base_url: base, model_api_key: "sk".into(), ..Default::default() };
        settings.vision.provider = VisionProvider::OpenAi;
        let e = describe_image(&img, "x", &settings, &CancellationToken::new()).await.unwrap_err();
        assert_eq!(e.to_string(), r#"vision (openai) HTTP 404: {"error":"model not found"}"#);
    }

    #[tokio::test]
    async fn missing_image_is_an_error() {
        let e = describe_image(Path::new("Z:/nope/x.jpg"), "", &MediaSettings::default(), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(e.to_string().starts_with("cannot read image Z:/nope/x.jpg"));
    }
}
