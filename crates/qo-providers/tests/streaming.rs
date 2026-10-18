//! Port of tests/streaming.test.ts (OpenAI-compatible SSE streaming + the
//! token estimate), against a local mock server.

mod common;

use common::{serve, sse, Reply};
use qo_core::{Message, ModelOptions, Provider};
use qo_providers::{ProviderConfig, VllmProvider};
use serde_json::json;

fn cfg(base_url: &str) -> ProviderConfig {
    ProviderConfig { top_p: 1.0, max_tokens: 256, ..ProviderConfig::new("test-model", base_url) }
}

async fn stream_collect(p: &VllmProvider, msg: &str) -> (Vec<String>, qo_core::GenerateResult) {
    let mut seen = Vec::new();
    let res = p
        .stream(&[Message::user(msg)], &mut |d: &str| seen.push(d.to_string()), &ModelOptions::default())
        .await
        .expect("stream ok");
    (seen, res)
}

#[tokio::test]
async fn accumulates_content_deltas_and_emits_them_via_on_chunk() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#),
            sse(r#"{"choices":[{"delta":{"content":", "}}]}"#),
            sse(r#"{"choices":[{"delta":{"content":"world"}}]}"#),
            sse(r#"{"usage":{"prompt_tokens":11,"completion_tokens":3}}"#),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (seen, res) = stream_collect(&p, "hi").await;
    assert_eq!(seen, ["Hello", ", ", "world"]);
    assert_eq!(res.text, "Hello, world");
    assert_eq!(res.prompt_tokens, Some(11));
    assert_eq!(res.completion_tokens, Some(3));
    // The request asked for a usage frame and went to /v1/chat/completions.
    let req = &srv.requests()[0];
    assert_eq!(req.path, "/v1/chat/completions");
    assert_eq!(req.json()["stream"], true);
    assert_eq!(req.json()["stream_options"], json!({"include_usage": true}));
}

#[tokio::test]
async fn survives_a_content_delta_split_across_two_network_chunks() {
    // The JSON frame is broken mid-way between two network chunks; the parser
    // must buffer the partial line and only parse once the newline arrives.
    let srv = serve(|_| {
        Reply::Stream(vec![
            r#"data: {"choices":[{"delta":{"con"#.into(),
            "tent\":\"split\"}}]}\n\n".into(),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (seen, res) = stream_collect(&p, "x").await;
    assert_eq!(res.text, "split");
    assert_eq!(seen, ["split"]);
}

#[tokio::test]
async fn assembles_tool_calls_streamed_as_fragments() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run_shell"}}]}}]}"#),
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"ls"}}]}}]}"#),
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" -la\"}"}}]}}]}"#),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (_, res) = stream_collect(&p, "list").await;
    assert_eq!(res.tool_calls.len(), 1);
    assert_eq!(res.tool_calls[0].id, "c1");
    assert_eq!(res.tool_calls[0].name, "run_shell");
    assert_eq!(res.tool_calls[0].arguments, json!({"command": "ls -la"}));
}

#[tokio::test]
async fn falls_back_to_a_heuristic_completion_count_when_no_usage_frame_arrives() {
    let srv = serve(|_| {
        Reply::Stream(vec![sse(r#"{"choices":[{"delta":{"content":"some output text here"}}]}"#), "data: [DONE]\n\n".into()])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (_, res) = stream_collect(&p, "x").await;
    assert!(res.completion_tokens.unwrap_or(0) > 0);
    assert_eq!(res.prompt_tokens, None);
}

#[tokio::test]
async fn parses_a_final_frame_without_trailing_newline() {
    let srv = serve(|_| Reply::Stream(vec![r#"data: {"choices":[{"delta":{"content":"tail"}}]}"#.into()])).await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (_, res) = stream_collect(&p, "x").await;
    assert_eq!(res.text, "tail");
}

#[tokio::test]
async fn ignores_keepalives_comments_and_garbage_frames() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            ": keep-alive\n\n".into(),
            "event: ping\n".into(),
            "data: {not json}\n\n".into(),
            sse(r#"{"choices":[{"delta":{"content":"ok"}}],"usage":null}"#),
            "data: [DONE]\n\n".into(),
        ])
    })
    .await;
    let p = VllmProvider::new(cfg(&srv.base_url));
    let (_, res) = stream_collect(&p, "x").await;
    assert_eq!(res.text, "ok");
}

/* ── countTokens heuristic ── */

#[test]
fn count_tokens_returns_0_for_empty() {
    let p = VllmProvider::new(cfg("http://localhost:8000"));
    assert_eq!(p.count_tokens(""), 0);
}

#[test]
fn count_tokens_monotonically_increases_with_length() {
    let p = VllmProvider::new(cfg("http://localhost:8000"));
    assert!(p.count_tokens("hello world this is longer") > p.count_tokens("hello"));
}

#[test]
fn count_tokens_counts_cjk_roughly_per_character() {
    let p = VllmProvider::new(cfg("http://localhost:8000"));
    assert!(p.count_tokens("你好世界") >= 4);
}

#[test]
fn count_tokens_is_in_a_sane_range_for_a_code_snippet() {
    let p = VllmProvider::new(cfg("http://localhost:8000"));
    let n = p.count_tokens("function add(a, b) { return a + b; }");
    assert!(n > 8 && n < 30, "{n}");
}
