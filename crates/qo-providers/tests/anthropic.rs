//! Port of tests/anthropic.test.ts plus mock-server tests of the native
//! Messages API path (streaming events, OAuth refresh on 401, probe).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use common::{serve, sse, Reply};
use qo_core::error::is_interrupt;
use qo_core::{CancellationToken, Message, ModelOptions, Provider, ToolCall, ToolSpec};
use qo_providers::{parse_content, to_anthropic_messages, AnthropicProvider, ProviderConfig, TokenSource};
use serde_json::json;

/* ── toAnthropicMessages — OpenAI-shape → Claude Messages ── */

#[test]
fn folds_system_turns_into_the_top_level_system_field() {
    let (system, msgs) = to_anthropic_messages(&[Message::system("You are Qwenodyssey."), Message::user("hi")]);
    assert_eq!(system, "You are Qwenodyssey.");
    assert_eq!(msgs, vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})]);
}

#[test]
fn maps_assistant_tool_calls_to_tool_use_blocks_and_tool_results_to_tool_result() {
    let call = ToolCall { id: "call_1".into(), name: "get_weather".into(), arguments: json!({"city": "Paris"}) };
    let history = vec![
        Message::user("what's the weather"),
        Message::assistant_calls("", vec![call]),
        Message::tool_result("call_1", "get_weather", "18C sunny"),
        Message::assistant("It's 18C and sunny in Paris."),
    ];
    let (_, msgs) = to_anthropic_messages(&history);
    assert_eq!(
        msgs[1],
        json!({"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}]})
    );
    assert_eq!(
        msgs[2],
        json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "18C sunny"}]})
    );
    assert_eq!(msgs[3], json!({"role": "assistant", "content": [{"type": "text", "text": "It's 18C and sunny in Paris."}]}));
}

#[test]
fn degrades_an_orphaned_tool_result_to_plain_user_text() {
    let (_, msgs) = to_anthropic_messages(&[Message::user("continue"), Message::tool_result("gone", "x", "stale result")]);
    assert_eq!(msgs[1], json!({"role": "user", "content": [{"type": "text", "text": "[tool result] stale result"}]}));
}

#[test]
fn drops_empty_assistant_turns() {
    let (_, msgs) = to_anthropic_messages(&[Message::user("hi"), Message::assistant(""), Message::user("still there?")]);
    assert_eq!(msgs.len(), 2);
    assert!(msgs.iter().all(|m| m["role"] == "user"));
}

/* ── parseContent — Claude content blocks → text + tool calls ── */

#[test]
fn extracts_text_and_tool_use_blocks() {
    let content = json!([
        {"type": "text", "text": "Let me check. "},
        {"type": "tool_use", "id": "toolu_1", "name": "grep", "input": {"pattern": "TODO"}},
    ]);
    let (text, calls) = parse_content(Some(&content));
    assert_eq!(text, "Let me check. ");
    assert_eq!(calls, vec![ToolCall { id: "toolu_1".into(), name: "grep".into(), arguments: json!({"pattern": "TODO"}) }]);
}

#[test]
fn returns_no_tool_calls_when_there_are_none() {
    let (text, calls) = parse_content(Some(&json!([{"type": "text", "text": "done"}])));
    assert_eq!(text, "done");
    assert!(calls.is_empty());
}

/* ── AnthropicProvider ── */

fn cfg(base_url: &str) -> ProviderConfig {
    ProviderConfig { auth_token: Some("test-token".into()), ..ProviderConfig::new("claude-opus-4-8", base_url) }
}

#[test]
fn identifies_as_the_anthropic_backend() {
    let p = AnthropicProvider::new(cfg("https://api.anthropic.com"));
    assert_eq!(p.name(), "anthropic");
    assert_eq!(p.model(), "claude-opus-4-8");
    assert!(p.uses_oauth());
}

#[tokio::test]
async fn generate_sends_oauth_headers_and_no_sampling_params() {
    let srv = serve(|_| {
        Reply::Json(
            200,
            json!({
                "content": [{"type": "text", "text": "hi there"}, {"type": "tool_use", "id": "toolu_2", "name": "ls", "input": {}}],
                "usage": {"input_tokens": 12, "output_tokens": 5},
            }),
        )
    })
    .await;
    let p = AnthropicProvider::new(cfg(&srv.base_url));
    let tools = vec![ToolSpec { name: "ls".into(), description: "list".into(), parameters: json!({"type": "object"}) }];
    let r = p
        .generate(&[Message::system("sys"), Message::user("hi")], &ModelOptions { tools: Some(tools), ..Default::default() })
        .await
        .unwrap();
    assert_eq!(r.text, "hi there");
    assert_eq!(r.tool_calls[0].name, "ls");
    assert_eq!((r.prompt_tokens, r.completion_tokens), (Some(12), Some(5)));
    let req = &srv.requests()[0];
    assert_eq!(req.path, "/v1/messages");
    assert_eq!(req.header("authorization"), Some("Bearer test-token"));
    assert_eq!(req.header("anthropic-beta"), Some("oauth-2025-04-20"));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
    let body = req.json();
    assert_eq!(body["system"], "sys");
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["tools"][0]["input_schema"], json!({"type": "object"}));
    assert!(body.get("temperature").is_none() && body.get("top_p").is_none() && body.get("thinking").is_none());
}

#[tokio::test]
async fn api_key_mode_uses_x_api_key() {
    let srv = serve(|_| Reply::Json(200, json!({"content": [{"type": "text", "text": "ok"}]}))).await;
    let p = AnthropicProvider::new(ProviderConfig { api_key: "sk-ant-api".into(), ..ProviderConfig::new("claude-x", &srv.base_url) });
    p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    let req = &srv.requests()[0];
    assert_eq!(req.header("x-api-key"), Some("sk-ant-api"));
    assert!(req.header("authorization").is_none());
}

#[tokio::test]
async fn stream_assembles_text_tool_use_and_usage() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            "event: message_start\n".to_string() + &sse(r#"{"type":"message_start","message":{"usage":{"input_tokens":21}}}"#),
            sse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
            sse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me "}}"#),
            sse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"look."}}"#),
            sse(r#"{"type":"ping"}"#),
            sse(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_7","name":"grep","input":{}}}"#),
            sse(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pattern\":"}}"#),
            sse(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"TODO\"}"}}"#),
            sse(r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#),
            sse(r#"{"type":"message_stop"}"#),
        ])
    })
    .await;
    let p = AnthropicProvider::new(cfg(&srv.base_url));
    let mut seen = Vec::new();
    let r = p
        .stream(&[Message::user("find todos")], &mut |d: &str| seen.push(d.to_string()), &ModelOptions::default())
        .await
        .unwrap();
    assert_eq!(seen, ["Let me ", "look."]);
    assert_eq!(r.text, "Let me look.");
    assert_eq!(r.tool_calls, vec![ToolCall { id: "toolu_7".into(), name: "grep".into(), arguments: json!({"pattern": "TODO"}) }]);
    assert_eq!((r.prompt_tokens, r.completion_tokens), (Some(21), Some(9)));
    assert_eq!(srv.bodies()[0]["stream"], true);
}

#[tokio::test]
async fn stream_in_band_error_event_fails_loudly() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            sse(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}"#),
            "event: error\n".to_string() + &sse(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#),
        ])
    })
    .await;
    let p = AnthropicProvider::new(cfg(&srv.base_url));
    let err = p.stream(&[Message::user("x")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("Overloaded"), "{err:#}");
}

/// Token source that mints a new token per forced call.
struct CountingSource {
    calls: AtomicUsize,
}

#[async_trait]
impl TokenSource for CountingSource {
    async fn token(&self, force: bool) -> Option<String> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Some(if force { format!("fresh-token-{n}") } else { "initial-token".into() })
    }
}

#[tokio::test]
async fn an_expired_oauth_token_401_is_refreshed_and_retried_once() {
    let srv = serve(|req| {
        if req.header("authorization").is_some_and(|a| a.starts_with("Bearer fresh-token")) {
            Reply::Json(200, json!({"content": [{"type": "text", "text": "ok"}]}))
        } else {
            Reply::Text(401, r#"{"type":"error","error":{"type":"authentication_error","message":"expired"}}"#.into())
        }
    })
    .await;
    let src = Arc::new(CountingSource { calls: AtomicUsize::new(0) });
    let p = AnthropicProvider::new(ProviderConfig { refresh_auth: Some(src.clone()), ..ProviderConfig::new("claude-x", &srv.base_url) });
    // No configured token: the source is asked lazily, then re-minted on the 401.
    let r = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap();
    assert_eq!(r.text, "ok");
    let reqs = srv.requests();
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[0].header("authorization"), Some("Bearer initial-token"));
    assert_eq!(reqs[1].header("authorization"), Some("Bearer fresh-token-1"));
    assert_eq!(src.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_401_without_a_refresher_surfaces() {
    let srv = serve(|_| Reply::Text(401, "invalid x-api-key".into())).await;
    let p = AnthropicProvider::new(ProviderConfig { api_key: "bad".into(), ..ProviderConfig::new("claude-x", &srv.base_url) });
    let err = p.generate(&[Message::user("hi")], &ModelOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("HTTP 401"));
    assert_eq!(srv.requests().len(), 1);
}

#[tokio::test]
async fn probe_resolves_on_200_and_refreshes_on_401() {
    let srv = serve(|req| {
        if req.header("authorization") == Some("Bearer test-token") {
            Reply::Text(401, "expired".into())
        } else {
            Reply::HeadersThenHang
        }
    })
    .await;
    let src = Arc::new(CountingSource { calls: AtomicUsize::new(0) });
    let p = AnthropicProvider::new(ProviderConfig { refresh_auth: Some(src), ..cfg(&srv.base_url) });
    p.probe(&CancellationToken::new()).await.unwrap();
    let reqs = srv.requests();
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[1].path, "/v1/messages");
    assert_eq!(reqs[1].json()["max_tokens"], 16);
}

#[tokio::test]
async fn probe_and_generate_honour_esc() {
    let srv = serve(|_| Reply::Hang).await;
    let p = AnthropicProvider::new(cfg(&srv.base_url));
    let t = CancellationToken::new();
    let t2 = t.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        t2.cancel();
    });
    let err = p.probe(&t).await.unwrap_err();
    assert!(is_interrupt(&err));
    let err = p.generate(&[Message::user("x")], &ModelOptions { cancel: Some(t), ..Default::default() }).await.unwrap_err();
    assert!(is_interrupt(&err));
}
