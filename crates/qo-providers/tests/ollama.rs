//! Ollama native API (NDJSON streaming, /api/tags) and LM Studio control,
//! against a local mock server.

mod common;

use common::{serve, Reply};
use qo_core::error::is_interrupt;
use qo_core::{CancellationToken, Message, ModelOptions, Provider};
use qo_providers::{LmStudioControl, LmStudioProvider, OllamaProvider, ProviderConfig};
use serde_json::json;

fn ollama(base_url: &str, model: &str) -> OllamaProvider {
    OllamaProvider::new(ProviderConfig { context_tokens: Some(16384), ..ProviderConfig::new(model, base_url) })
}

#[tokio::test]
async fn stream_parses_ndjson_deltas_thinking_tools_and_usage() {
    let srv = serve(|_| {
        Reply::Stream(vec![
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"plan \"}}\n".into(),
            "{\"message\":{\"content\":\"Hel\"}}\n{\"message\":{\"content\":\"lo\"}}\n".into(),
            "{\"message\":{\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"ls\",\"arguments\":{\"path\":\".\"}}}]}}\n".into(),
            "{\"message\":{\"content\":\"\"},\"done\":true,\"prompt_eval_count\":30,\"eval_count\":4}\n".into(),
        ])
    })
    .await;
    let p = ollama(&srv.base_url, "qwen3.5:9b");
    let mut seen = Vec::new();
    let r = p.stream(&[Message::user("hi")], &mut |d: &str| seen.push(d.to_string()), &ModelOptions::default()).await.unwrap();
    assert_eq!(seen, ["Hel", "lo"]);
    assert_eq!(r.text, "Hello");
    assert_eq!(r.thinking.as_deref(), Some("plan"));
    assert_eq!(r.tool_calls[0].name, "ls");
    assert_eq!(r.tool_calls[0].id, "call_0");
    assert_eq!(r.tool_calls[0].arguments, json!({"path": "."}));
    assert_eq!((r.prompt_tokens, r.completion_tokens), (Some(30), Some(4)));
    let req = &srv.requests()[0];
    assert_eq!(req.path, "/api/chat");
    let body = req.json();
    assert_eq!(body["options"]["num_ctx"], 16384);
    assert_eq!(body["think"], false);
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn stream_error_line_fails_loudly() {
    let srv = serve(|_| Reply::Stream(vec!["{\"error\":\"model runner has unexpectedly stopped\"}\n".into()])).await;
    let p = ollama(&srv.base_url, "llama3");
    let err = p.stream(&[Message::user("hi")], &mut |_: &str| {}, &ModelOptions::default()).await.unwrap_err();
    assert!(format!("{err:#}").contains("unexpectedly stopped"));
}

#[tokio::test]
async fn generate_uses_native_chat_and_reports_http_errors() {
    let srv = serve(|req| {
        if req.json()["model"] == "missing:1b" {
            Reply::Text(404, "{\"error\":\"model 'missing:1b' not found, try pulling it first\"}".into())
        } else {
            Reply::Json(200, json!({"message": {"content": "pong", "thinking": "t"}, "prompt_eval_count": 3, "eval_count": 1, "done": true}))
        }
    })
    .await;
    let p = ollama(&srv.base_url, "llama3");
    let r = p.generate(&[Message::user("ping")], &ModelOptions::default()).await.unwrap();
    assert_eq!(r.text, "pong");
    assert_eq!(r.thinking.as_deref(), Some("t"));
    assert_eq!((r.prompt_tokens, r.completion_tokens), (Some(3), Some(1)));
    assert_eq!(srv.bodies()[0]["stream"], false);

    p.set_model("missing:1b");
    let err = p.generate(&[Message::user("ping")], &ModelOptions::default()).await.unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("ollama HTTP 404") && msg.contains("try pulling"), "{msg}");
    assert!(qo_core::error::looks_unavailable(&err));
}

#[tokio::test]
async fn generate_honours_esc() {
    let srv = serve(|_| Reply::Hang).await;
    let p = ollama(&srv.base_url, "llama3");
    let t = CancellationToken::new();
    let t2 = t.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        t2.cancel();
    });
    let err = p.generate(&[Message::user("x")], &ModelOptions { cancel: Some(t), ..Default::default() }).await.unwrap_err();
    assert!(is_interrupt(&err));
}

#[tokio::test]
async fn list_models_and_health_check_use_api_tags() {
    let srv = serve(|_| {
        Reply::Json(200, json!({"models": [{"name": "qwen3.5:9b", "size": 6_600_000_000u64}, {"name": "deepseek-r1:7b", "size": 4_700_000_000u64}]}))
    })
    .await;
    let p = ollama(&srv.base_url, "qwen3.5:9b");
    let models = p.list_models().await.unwrap();
    assert_eq!(models[0].name, "deepseek-r1:7b");
    assert_eq!(models[0].size.as_deref(), Some("4.4 GB"));
    assert_eq!(srv.requests()[0].path, "/api/tags");
    let h = p.health_check().await;
    assert!(h.ok && h.detail.is_none());
    p.set_model("gemma:2b");
    let h = p.health_check().await;
    assert!(h.ok);
    assert_eq!(h.detail.as_deref(), Some("reachable, but \"gemma:2b\" is not pulled (have: qwen3.5:9b, deepseek-r1:7b)"));
}

#[tokio::test]
async fn health_check_reports_an_unreachable_server() {
    // Bind then drop a listener so the port is (almost certainly) closed.
    let addr = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let p = ollama(&format!("http://{addr}"), "llama3");
    let h = p.health_check().await;
    assert!(!h.ok);
    assert!(h.detail.is_some());
}

#[tokio::test]
async fn probe_goes_through_the_openai_compatible_endpoint() {
    let srv = serve(|_| Reply::HeadersThenHang).await;
    let p = ollama(&srv.base_url, "qwen3.5:9b");
    p.probe(&CancellationToken::new()).await.unwrap();
    assert_eq!(srv.requests()[0].path, "/v1/chat/completions");
}

/* ── LM Studio ── */

#[tokio::test]
async fn lmstudio_provider_lists_models_and_checks_health_with_bearer() {
    let srv = serve(|_| Reply::Json(200, json!({"data": [{"id": "qwen3.5-9b"}, {"id": "gemma-4-12b"}]}))).await;
    let p = LmStudioProvider::new(ProviderConfig { api_key: "lm-key".into(), ..ProviderConfig::new("qwen3.5-9b", &srv.base_url) });
    let names: Vec<String> = p.list_models().await.unwrap().into_iter().map(|m| m.name).collect();
    assert_eq!(names, ["qwen3.5-9b", "gemma-4-12b"]);
    assert!(p.health_check().await.ok);
    let req = &srv.requests()[0];
    assert_eq!(req.path, "/v1/models");
    assert_eq!(req.header("authorization"), Some("Bearer lm-key"));
}

#[tokio::test]
async fn ensure_server_takes_the_fast_path_when_the_server_answers_401() {
    let srv = serve(|_| Reply::Text(401, "api key required".into())).await;
    // A bogus CLI path: the fast path must never reach for it.
    let ctl = LmStudioControl::new(Some("Z:/definitely/missing/lms.exe"));
    let up = ctl.ensure_server(1234, "0.0.0.0", &format!("{}/v1", srv.base_url)).await;
    assert!(up);
    assert_eq!(srv.requests()[0].path, "/v1/models");
}
