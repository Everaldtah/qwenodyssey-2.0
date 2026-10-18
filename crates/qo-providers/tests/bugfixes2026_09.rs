//! Port of the provider parts of tests/bugfixes2026_09.test.ts: in-band API
//! error frames, reasoning capture, describeApiError and NIM request shaping.

mod common;

use common::{serve, sse, Reply};
use qo_core::{Message, ModelOptions, Provider};
use qo_providers::{describe_api_error, NvidiaNimProvider, OpenRouterProvider, ProviderConfig, VllmProvider};
use serde_json::json;

fn cfg(base_url: &str) -> ProviderConfig {
    ProviderConfig { top_p: 1.0, max_tokens: 256, ..ProviderConfig::new("test-model", base_url) }
}

/* ── in-band API error frames (NIM 503 inside an HTTP 200 stream) ── */

#[tokio::test]
async fn stream_an_error_frame_errors_instead_of_yielding_an_empty_reply() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"error":{"message":"Service temporarily overloaded","type":"service_unavailable","code":503}}"#),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let err = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("overloaded"), "{msg}");
    assert!(qo_core::error::looks_unavailable(&err));
}

#[tokio::test]
async fn generate_an_error_envelope_with_http_200_errors() {
    let srv = serve(|_| Reply::Json(200, json!({"error": {"message": "Function not found for account", "code": 404}}))).await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let err = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("404 Function not found"), "{msg}");
}

#[tokio::test]
async fn stream_reasoning_content_deltas_are_kept_as_thinking() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"choices":[{"delta":{"reasoning_content":"user said hi. "}}]}"#),
            sse(r#"{"choices":[{"delta":{"reasoning_content":"greet back."}}]}"#),
            sse(r#"{"choices":[{"delta":{"content":"Hey!"}}]}"#),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let mut seen = String::new();
    let res = p.stream(&[Message::user("hi")], &mut |d: &str| seen.push_str(d), &ModelOptions::default()).await.unwrap();
    assert_eq!(res.text, "Hey!");
    assert_eq!(seen, "Hey!"); // reasoning never reaches on_chunk
    assert_eq!(res.thinking.as_deref(), Some("user said hi. greet back."));
}

#[tokio::test]
async fn stream_openrouter_reasoning_field_is_kept_too() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"choices":[{"delta":{"reasoning":"hmm"}}]}"#),
            sse(r#"{"choices":[{"delta":{"content":"A"}}]}"#),
        ])
    })
    .await;
    let p = OpenRouterProvider::new(cfg(&srv.base_url));
    let res = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap();
    assert_eq!(res.thinking.as_deref(), Some("hmm"));
    // OpenRouter attribution headers + bearer key ride along.
    let req = &srv.requests()[0];
    assert_eq!(req.header("x-title"), Some("Qwenodyssey"));
    assert!(req.header("http-referer").is_some_and(|v| v.contains("Qwenodyssey")));
}

#[tokio::test]
async fn generate_reasoning_content_is_surfaced_as_thinking() {
    let srv = serve(|_| {
        Reply::Json(
            200,
            json!({
                "choices": [{"message": {"content": "Hey!", "reasoning_content": "casual greeting"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }),
        )
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let res = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(res.text, "Hey!");
    assert_eq!(res.thinking.as_deref(), Some("casual greeting"));
    assert_eq!((res.prompt_tokens, res.completion_tokens), (Some(1), Some(2)));
}

#[tokio::test]
async fn generate_parses_tool_calls_and_null_content() {
    let srv = serve(|_| {
        Reply::Json(
            200,
            json!({"choices": [{"message": {"content": null, "tool_calls": [
                {"id": "call_9", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"}}
            ]}}]}),
        )
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let res = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(res.text, "");
    assert_eq!(res.tool_calls[0].id, "call_9");
    assert_eq!(res.tool_calls[0].arguments, json!({"path": "a.py"}));
}

#[test]
fn describe_api_error_renders_objects_and_strings() {
    assert_eq!(describe_api_error(&json!("boom")), "boom");
    assert_eq!(
        describe_api_error(&json!({"code": 503, "type": "service_unavailable", "message": "overloaded"})),
        "503 service_unavailable overloaded"
    );
    assert_eq!(describe_api_error(&json!({"weird": 1})), r#"{"weird":1}"#);
}

/* ── NVIDIA NIM request shaping ── */

fn nim_cfg(base_url: &str, model: &str) -> ProviderConfig {
    ProviderConfig {
        api_key: "k".into(),
        top_p: 1.0,
        max_tokens: 64,
        nemotron_thinking: Some(true),
        ..ProviderConfig::new(model, base_url)
    }
}

async fn nim_server() -> common::MockServer {
    serve(|_| Reply::Json(200, json!({"choices": [{"message": {"content": "ok"}}], "usage": {}}))).await
}

#[tokio::test]
async fn omits_reasoning_budget_for_nemotron_by_default() {
    let srv = nim_server().await;
    let p = NvidiaNimProvider::new(nim_cfg(&srv.base_url, "nvidia/nemotron-3-ultra-550b-a55b"));
    p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    let body = &srv.bodies()[0];
    assert_eq!(body["chat_template_kwargs"], json!({"enable_thinking": true}));
    assert!(body.get("reasoning_budget").is_none());
    assert_eq!(srv.requests()[0].header("authorization"), Some("Bearer k"));
}

#[tokio::test]
async fn sends_reasoning_budget_only_when_explicitly_configured() {
    let srv = nim_server().await;
    let p = NvidiaNimProvider::new(ProviderConfig {
        reasoning_budget: Some(2048),
        ..nim_cfg(&srv.base_url, "nvidia/nemotron-3-ultra-550b-a55b")
    });
    p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(srv.bodies()[0]["reasoning_budget"], 2048);
}

#[tokio::test]
async fn kimi_on_nim_folds_the_system_prompt_and_disables_thinking() {
    let srv = nim_server().await;
    let p = NvidiaNimProvider::new(nim_cfg(&srv.base_url, "moonshotai/kimi-k2.6"));
    p.generate(&[Message::system("rules"), Message::system("more"), Message::user("hi")], &ModelOptions::default())
        .await
        .unwrap();
    let body = &srv.bodies()[0];
    assert_eq!(body["messages"], json!([{"role": "user", "content": "rules\n\nmore\n\n---\n\nhi"}]));
    assert_eq!(body["chat_template_kwargs"], json!({"thinking": false}));
}

#[tokio::test]
async fn set_model_redirects_subsequent_requests() {
    let srv = nim_server().await;
    let p = NvidiaNimProvider::new(nim_cfg(&srv.base_url, "moonshotai/kimi-k2.6"));
    p.set_model("meta/llama-3.3-70b-instruct");
    let r = p.generate(&[Message::system("sys"), Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(r.model, "meta/llama-3.3-70b-instruct");
    let body = &srv.bodies()[0];
    assert_eq!(body["model"], "meta/llama-3.3-70b-instruct");
    // Non-kimi model keeps the system role and gets no thinking extras.
    assert_eq!(body["messages"][0]["role"], "system");
    assert!(body.get("chat_template_kwargs").is_none());
}
