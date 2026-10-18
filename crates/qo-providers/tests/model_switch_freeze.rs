//! Port of tests/modelSwitchFreeze.test.ts — regressions for the "switch
//! models → terminal stuck" bug (v0.4.2): a hosted model that accepts a request
//! and never answers held the turn for the full timeout, Esc was ignored on the
//! blocking path, and failed requests were not self-healed.

mod common;

use std::time::{Duration, Instant};

use common::{serve, sse, Reply};
use qo_core::error::{is_interrupt, looks_unavailable};
use qo_core::{CancellationToken, Message, ModelOptions, Provider, ProviderError};
use qo_providers::{NvidiaNimProvider, OpenRouterProvider, ProviderConfig, VllmProvider};
use serde_json::json;

fn cfg(model: &str, base_url: &str) -> ProviderConfig {
    ProviderConfig { top_p: 1.0, max_tokens: 8192, ..ProviderConfig::new(model, base_url) }
}

/// A token that fires after `ms`.
fn cancel_after(ms: u64) -> CancellationToken {
    let t = CancellationToken::new();
    let t2 = t.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        t2.cancel();
    });
    t
}

fn provider_error(e: &anyhow::Error) -> &ProviderError {
    e.chain().find_map(|c| c.downcast_ref::<ProviderError>()).expect("a ProviderError")
}

#[tokio::test]
async fn generate_esc_interrupts_a_request_that_never_answers() {
    let srv = serve(|_| Reply::Hang).await;
    let p = VllmProvider::new(ProviderConfig { request_timeout_ms: Some(90_000), ..cfg("test-model", &srv.base_url) });
    let opts = ModelOptions { cancel: Some(cancel_after(20)), ..Default::default() };
    let t0 = Instant::now();
    let err = p.generate(&[Message::user("hi")], &opts).await.unwrap_err();
    assert!(is_interrupt(&err), "{err:#}");
    assert!(t0.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn stream_esc_interrupts_before_headers_and_mid_body() {
    let srv = serve(|_| Reply::Hang).await;
    let p = VllmProvider::new(cfg("test-model", &srv.base_url));
    let opts = ModelOptions { cancel: Some(cancel_after(20)), ..Default::default() };
    let err = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &opts).await.unwrap_err();
    assert!(is_interrupt(&err), "{err:#}");

    // Mid-body: some text arrives, then Esc while the body is silent (no timeout configured).
    let srv = serve(|_| Reply::StreamThenStall(vec![sse(r#"{"choices":[{"delta":{"content":"he"}}]}"#)])).await;
    let p = VllmProvider::new(cfg("test-model", &srv.base_url));
    let opts = ModelOptions { cancel: Some(cancel_after(150)), ..Default::default() };
    let mut seen = String::new();
    let err = p.stream(&[Message::user("hi")], &mut |d: &str| seen.push_str(d), &opts).await.unwrap_err();
    assert!(is_interrupt(&err), "{err:#}");
    assert_eq!(seen, "he");
}

#[tokio::test]
async fn an_already_cancelled_token_never_sends() {
    let srv = serve(|_| Reply::Json(200, json!({"choices": [{"message": {"content": "x"}}]}))).await;
    let p = VllmProvider::new(cfg("test-model", &srv.base_url));
    let t = CancellationToken::new();
    t.cancel();
    let err = p.generate(&[Message::user("hi")], &ModelOptions { cancel: Some(t), ..Default::default() }).await.unwrap_err();
    assert!(is_interrupt(&err));
    assert!(srv.requests().is_empty());
}

#[tokio::test]
async fn generate_times_out_when_headers_never_arrive() {
    let srv = serve(|_| Reply::Hang).await;
    let p = VllmProvider::new(ProviderConfig { request_timeout_ms: Some(100), ..cfg("test-model", &srv.base_url) });
    let err = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap_err();
    assert!(matches!(provider_error(&err), ProviderError::Timeout { ms: 100, .. }), "{err:#}");
    assert!(looks_unavailable(&err));
}

#[tokio::test]
async fn probe_a_backend_that_never_sends_headers_is_aborted_by_the_callers_token() {
    let srv = serve(|_| Reply::Hang).await;
    let p = NvidiaNimProvider::new(cfg("z-ai/glm-5.3", &srv.base_url));
    let err = p.probe(&cancel_after(20)).await.unwrap_err();
    assert!(is_interrupt(&err), "{err:#}");
}

#[tokio::test]
async fn probe_resolves_on_http_200_without_reading_the_body() {
    let srv = serve(|_| Reply::HeadersThenHang).await;
    let p = NvidiaNimProvider::new(cfg("nvidia/nemotron-3-ultra-550b-a55b", &srv.base_url));
    let never = CancellationToken::new();
    tokio::time::timeout(Duration::from_secs(5), p.probe(&never)).await.expect("probe returned").expect("probe ok");
    // The body was dropped unread: the client hung up on the open stream.
    let t0 = Instant::now();
    while !srv.client_closed() && t0.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(srv.client_closed());
    // A tiny streaming request.
    let body = &srv.bodies()[0];
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["messages"], json!([{"role": "user", "content": "ping"}]));
}

#[tokio::test]
async fn probe_surfaces_the_server_error() {
    let srv = serve(|_| Reply::Text(404, "Function not found for account".into())).await;
    let p = NvidiaNimProvider::new(cfg("moonshotai/kimi-k2.6", &srv.base_url));
    let err = p.probe(&CancellationToken::new()).await.unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("HTTP 404"), "{msg}");
    assert!(msg.contains("Function not found"), "{msg}");
}

#[tokio::test]
async fn nim_nemotron_a_rejected_reasoning_budget_is_dropped_and_the_request_retried() {
    let srv = serve(|req| {
        if req.json().get("reasoning_budget").is_some() {
            Reply::Text(400, "ValueError: thinking_token_budget is not yet supported by the V2 model runner.".into())
        } else {
            Reply::Json(200, json!({"choices": [{"message": {"content": "hey"}}]}))
        }
    })
    .await;
    let p = NvidiaNimProvider::new(ProviderConfig {
        reasoning_budget: Some(4096),
        ..cfg("nvidia/nemotron-3-ultra-550b-a55b", &srv.base_url)
    });
    let r = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(r.text, "hey");
    let bodies = srv.bodies();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["reasoning_budget"], 4096);
    assert!(bodies[1].get("reasoning_budget").is_none());
    // Remembered for the rest of the session: the next request skips the budget.
    p.generate(&[Message::user("again")], &ModelOptions::default()).await.unwrap();
    let bodies = srv.bodies();
    assert_eq!(bodies.len(), 3);
    assert!(bodies[2].get("reasoning_budget").is_none());
    assert!(p.backend().budget_rejected());
}

#[tokio::test]
async fn openrouter_402_can_only_afford_n_retried_once_within_the_affordable_budget() {
    let srv = serve(|req| {
        if req.json()["max_tokens"].as_u64().unwrap_or(0) > 3436 {
            Reply::Text(402, "You requested up to 8192 tokens, but can only afford 3436.".into())
        } else {
            Reply::Json(200, json!({"choices": [{"message": {"content": "ok"}}]}))
        }
    })
    .await;
    let p = OpenRouterProvider::new(cfg("qwen/qwen-2.5-coder-32b-instruct", &srv.base_url));
    let r = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(r.text, "ok");
    assert_eq!(srv.bodies()[1]["max_tokens"], 3436 - 64);
}

#[tokio::test]
async fn openrouter_402_streaming_path_also_retries_but_only_once() {
    let srv = serve(|_| Reply::Text(402, "You requested up to 8192 tokens, but can only afford 3436.".into())).await;
    let p = OpenRouterProvider::new(cfg("qwen/qwen-2.5-coder-32b-instruct", &srv.base_url));
    let err = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("HTTP 402"));
    // One original request + exactly one retry.
    assert_eq!(srv.requests().len(), 2);
    assert_eq!(srv.bodies()[1]["max_tokens"], 3372);
}

#[tokio::test]
async fn stream_a_body_that_goes_silent_after_the_headers_is_aborted_as_stalled() {
    let srv = serve(|_| Reply::StreamThenStall(vec![sse(r#"{"choices":[{"delta":{"content":"he"}}]}"#)])).await;
    let p = VllmProvider::new(ProviderConfig { request_timeout_ms: Some(150), ..cfg("test-model", &srv.base_url) });
    let err = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("stalled"), "{err:#}");
    assert!(matches!(provider_error(&err), ProviderError::Stalled { .. }));
    assert!(looks_unavailable(&err));
}
