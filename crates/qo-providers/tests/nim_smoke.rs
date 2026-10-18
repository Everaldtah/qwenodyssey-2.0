//! Optional manual smoke test against the real NVIDIA NIM API. Ignored by
//! default (tests must stay headless and free); run explicitly with
//! `cargo test -p qo-providers --test nim_smoke -- --ignored` when
//! NVIDIA_API_KEY is set. Uses nemotron — never z-ai/glm-5.3 (it hangs).

use std::time::Duration;

use qo_core::{CancellationToken, Message, ModelOptions, Provider};
use qo_providers::{NvidiaNimProvider, ProviderConfig};

#[tokio::test]
#[ignore = "calls the real NVIDIA NIM API; run with --ignored when NVIDIA_API_KEY is set"]
async fn nim_nemotron_tiny_request() {
    let Ok(key) = std::env::var("NVIDIA_API_KEY") else { return };
    if key.is_empty() {
        return;
    }
    let p = NvidiaNimProvider::new(ProviderConfig {
        api_key: key,
        max_tokens: 256,
        request_timeout_ms: Some(90_000),
        nemotron_thinking: Some(true),
        ..ProviderConfig::new("nvidia/nemotron-3-ultra-550b-a55b", "https://integrate.api.nvidia.com")
    });
    let cancel = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(30), p.probe(&cancel)).await.expect("probe in time").expect("probe ok");
    let mut text = String::new();
    let r = tokio::time::timeout(
        Duration::from_secs(120),
        p.stream(&[Message::user("Reply with the single word: pong")], &mut |d: &str| text.push_str(d), &ModelOptions::default()),
    )
    .await
    .expect("stream in time")
    .expect("stream ok");
    assert!(!r.text.is_empty() || r.thinking.is_some(), "{r:?}");
    eprintln!("nim smoke: text={:?} thinking_chars={} usage={:?}/{:?}", r.text, r.thinking.map_or(0, |t| t.len()), r.prompt_tokens, r.completion_tokens);
}
