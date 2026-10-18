//! Ollama backend (port of ollama.ts). Ollama serves an OpenAI-compatible API
//! at `{base}/v1` and its native API at `{base}/api`; generation goes through
//! the NATIVE `/api/chat` because the `/v1` endpoint silently ignores
//! `num_ctx` and caps context at ~4096 tokens, which truncates long
//! conversations / file reads and makes the model return nothing.

use std::cmp::Ordering;
use std::sync::RwLock;

use async_trait::async_trait;
use qo_core::tokens::estimate_tokens;
use qo_core::{
    CancellationToken, GenerateResult, HealthStatus, Message, ModelInfo, ModelOptions, Provider, ProviderError, Role,
    ToolCall,
};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use serde_json::{json, Map, Value};

use crate::http::{self, LineBuffer, FALLBACK_READ_MS};
use crate::openai::{self, bearer_headers, default_api_base, post_json, u64_at};
use crate::profile;
use crate::provider_config::ProviderConfig;
use crate::wire::{self, truthy};

pub struct OllamaProvider {
    cfg: ProviderConfig,
    model: RwLock<String>,
    client: Client,
}

impl OllamaProvider {
    pub fn new(cfg: ProviderConfig) -> Self {
        let model = RwLock::new(cfg.model.clone());
        Self { cfg, model, client: http::build_client() }
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    fn current_model(&self) -> String {
        self.model.read().map(|m| m.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }

    /// `http://host:11434/v1/` → `http://host:11434`.
    fn root(&self) -> String {
        let b = self.cfg.base_url.trim_end_matches('/');
        b.strip_suffix("/v1").unwrap_or(b).to_string()
    }

    fn json_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h
    }

    /// Build the native `/api/chat` request. Shared by generate() and stream()
    /// so the sampling knobs, GPU policy, thinking flag and tools can never
    /// drift apart.
    pub fn chat_body(&self, messages: &[Message], opts: &ModelOptions, stream: bool, model: &str) -> Map<String, Value> {
        let cfg = &self.cfg;
        let mut o = Map::new();
        o.insert("temperature".into(), json!(opts.temperature.unwrap_or(cfg.temperature)));
        o.insert("top_p".into(), json!(opts.top_p.unwrap_or(cfg.top_p)));
        o.insert("num_predict".into(), json!(opts.max_tokens.unwrap_or(cfg.max_tokens)));
        o.insert("num_ctx".into(), json!(cfg.context_tokens.unwrap_or(8192)));
        if let Some(stop) = &opts.stop {
            o.insert("stop".into(), json!(stop));
        }
        // Small-model decoding levers (top_k / repeat_penalty / presence_penalty).
        // Only sent when set — 0 means "leave it to Ollama's own default".
        if let Some(k) = opts.top_k.or(cfg.top_k).filter(|k| *k > 0) {
            o.insert("top_k".into(), json!(k));
        }
        if let Some(r) = opts.repeat_penalty.or(cfg.repeat_penalty).filter(|r| *r > 0.0) {
            o.insert("repeat_penalty".into(), json!(r));
        }
        if let Some(p) = opts.presence_penalty.or(cfg.presence_penalty).filter(|p| *p > 0.0) {
            o.insert("presence_penalty".into(), json!(p));
        }
        // GPU acceleration: only pin num_gpu when the user forces a layer count.
        // Left unset (gpu_layers = -1), Ollama auto-loads as many layers as fit
        // in VRAM and runs the rest on CPU/RAM — automatic offload for large
        // contexts.
        if let Some(n) = cfg.gpu_layers.filter(|n| *n >= 0) {
            o.insert("num_gpu".into(), json!(n));
        }
        if cfg.low_vram {
            o.insert("low_vram".into(), json!(true));
        }

        let mut body = Map::new();
        body.insert("model".into(), json!(model));
        body.insert("messages".into(), Value::Array(native_messages(messages)));
        body.insert("stream".into(), json!(stream));
        body.insert("options".into(), Value::Object(o));
        if let Some(ka) = cfg.keep_alive.as_deref().filter(|k| !k.is_empty()) {
            body.insert("keep_alive".into(), json!(ka));
        }
        if opts.json {
            body.insert("format".into(), json!("json"));
        }
        // Thinking control. Ollama returns a reasoning model's chain-of-thought
        // in a separate `thinking` field when asked. Hybrids (Qwen3/3.5) also
        // accept think:false, which skips deliberation entirely — faster, more
        // tool-shaped turns. Models with no thinking mode get no flag at all,
        // since Ollama rejects `think` for them.
        match openai::thinking_enabled(cfg, model, opts) {
            Some(true) => {
                body.insert("think".into(), json!(true));
            }
            Some(false) if profile::hybrid_thinking(model) => {
                body.insert("think".into(), json!(false));
            }
            _ => {}
        }
        if let Some(tools) = opts.tools.as_ref().filter(|t| !t.is_empty()) {
            body.insert("tools".into(), Value::Array(wire::wire_tools(tools)));
        }
        body
    }

    async fn get_tags(&self) -> Result<reqwest::Response, ProviderError> {
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        http::send("ollama", self.client.get(format!("{}/api/tags", self.root())), None, ms).await
    }

    async fn open_chat(&self, messages: &[Message], opts: &ModelOptions, stream: bool) -> anyhow::Result<(reqwest::Response, String)> {
        let model = self.current_model();
        let body = Value::Object(self.chat_body(messages, opts, stream, &model));
        let url = format!("{}/api/chat", self.root());
        let cancel = opts.cancel.as_ref();
        let timeout = self.cfg.timeout_ms();
        let res = http::send("ollama", post_json(&self.client, &url, Self::json_headers(), &body), cancel, timeout).await?;
        if !res.status().is_success() {
            let status = res.status().as_u16();
            let detail = http::safe_text("ollama", res, cancel, timeout).await?;
            return Err(ProviderError::http("ollama", status, detail).into());
        }
        Ok((res, model))
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn model(&self) -> String {
        self.current_model()
    }

    fn set_model(&self, model: &str) {
        match self.model.write() {
            Ok(mut m) => *m = model.to_string(),
            Err(e) => *e.into_inner() = model.to_string(),
        }
    }

    /// Generate via the native `/api/chat` (honours `options.num_ctx`).
    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        let (res, model) = self.open_chat(messages, opts, false).await?;
        let json = http::read_json("ollama", res, opts.cancel.as_ref(), self.cfg.timeout_ms()).await?;
        if let Some(err) = json.get("error").filter(|e| truthy(e)) {
            return Err(ProviderError::Api { provider: "ollama".into(), message: wire::describe_api_error(err) }.into());
        }
        let msg = json.get("message").cloned().unwrap_or(Value::Null);
        Ok(GenerateResult {
            text: msg.get("content").and_then(Value::as_str).unwrap_or_default().to_string(),
            thinking: msg.get("thinking").and_then(Value::as_str).map(str::to_string),
            tool_calls: parse_native_tool_calls(msg.get("tool_calls")),
            prompt_tokens: u64_at(&json, "prompt_eval_count"),
            completion_tokens: u64_at(&json, "eval_count"),
            model,
        })
    }

    /// Stream via the native `/api/chat` (newline-delimited JSON, NOT SSE).
    /// Each line carries `message.content` deltas; the final line has
    /// `done: true` plus `prompt_eval_count` / `eval_count` for exact usage.
    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let (mut res, model) = self.open_chat(messages, opts, true).await?;
        let cancel = opts.cancel.as_ref();
        let mut state = NdjsonState::default();
        let mut lines = LineBuffer::default();
        while let Some(chunk) = http::next_chunk("ollama", &mut res, cancel, self.cfg.timeout_ms()).await? {
            for line in lines.push(&chunk) {
                state.apply_line(&line, on_chunk)?;
            }
        }
        if let Some(rest) = lines.finish() {
            state.apply_line(&rest, on_chunk)?;
        }
        Ok(state.finish(model))
    }

    async fn health_check(&self) -> HealthStatus {
        let res = match self.get_tags().await {
            Ok(r) => r,
            Err(e) => return HealthStatus { ok: false, detail: Some(e.to_string()) },
        };
        if !res.status().is_success() {
            return HealthStatus { ok: false, detail: Some(format!("HTTP {}", res.status().as_u16())) };
        }
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        let json = match http::read_json("ollama", res, None, ms).await {
            Ok(j) => j,
            Err(e) => return HealthStatus { ok: false, detail: Some(e.to_string()) },
        };
        let names = tag_names(&json).join(", ");
        let model = self.current_model();
        HealthStatus {
            ok: true,
            detail: (!names.contains(&model)).then(|| {
                format!(
                    "reachable, but \"{model}\" is not pulled (have: {})",
                    if names.is_empty() { "none" } else { &names }
                )
            }),
        }
    }

    /// Native `/api/tags`, so sizes come alongside names (sorted by name).
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let res = self.get_tags().await?;
        let ms = Some(self.cfg.timeout_ms().unwrap_or(FALLBACK_READ_MS));
        if !res.status().is_success() {
            let status = res.status().as_u16();
            let detail = http::safe_text("ollama", res, None, ms).await?;
            return Err(ProviderError::http("ollama", status, detail).into());
        }
        let json = http::read_json("ollama", res, None, ms).await?;
        Ok(parse_tags(&json))
    }

    /// The shared liveness check against Ollama's OpenAI-compatible `/v1`
    /// endpoint (as the TS class inherited it from the base provider).
    async fn probe(&self, cancel: &CancellationToken) -> anyhow::Result<()> {
        let url = format!("{}/chat/completions", default_api_base(&self.cfg.base_url));
        let opts = ModelOptions { max_tokens: Some(16), ..Default::default() };
        let ping = [Message::user("ping")];
        openai::probe_with(
            "ollama",
            || {
                let model = self.current_model();
                let body = openai::openai_body(&self.cfg, &model, &ping, &opts, true, true);
                post_json(&self.client, &url, bearer_headers(&self.cfg.api_key), &Value::Object(body))
            },
            cancel,
            self.cfg.timeout_ms(),
            |status, detail| openai::recover_affordable(status, detail, &ModelOptions::default()).is_some(),
        )
        .await
    }
}

/// Incremental state of one native NDJSON stream.
#[derive(Default)]
struct NdjsonState {
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

impl NdjsonState {
    fn apply_line(&mut self, line: &str, on_chunk: &mut (dyn FnMut(&str) + Send)) -> Result<(), ProviderError> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let Ok(obj) = serde_json::from_str::<Value>(trimmed) else {
            return Ok(()); // partial frame
        };
        // Ollama reports a mid-stream failure (e.g. the model runner crashed)
        // as an `{"error": …}` line. Fail loudly rather than return an empty
        // reply, so the chat layer can fall back (same rule as the SSE path).
        if let Some(err) = obj.get("error").filter(|e| truthy(e)) {
            return Err(ProviderError::Api { provider: "ollama".into(), message: wire::describe_api_error(err) });
        }
        if let Some(msg) = obj.get("message") {
            if let Some(c) = msg.get("content").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                self.text.push_str(c);
                on_chunk(c);
            }
            if let Some(t) = msg.get("thinking").and_then(Value::as_str) {
                self.thinking.push_str(t);
            }
            let tc = parse_native_tool_calls(msg.get("tool_calls"));
            if !tc.is_empty() {
                self.tool_calls = tc;
            }
        }
        if obj.get("done").is_some_and(truthy) {
            self.prompt_tokens = u64_at(&obj, "prompt_eval_count").or(self.prompt_tokens);
            self.completion_tokens = u64_at(&obj, "eval_count").or(self.completion_tokens);
        }
        Ok(())
    }

    fn finish(self, model: String) -> GenerateResult {
        let completion_tokens = match self.completion_tokens {
            None if !self.text.is_empty() => Some(estimate_tokens(&self.text) as u64),
            other => other,
        };
        let thinking = self.thinking.trim();
        GenerateResult {
            thinking: (!thinking.is_empty()).then(|| thinking.to_string()),
            text: self.text,
            tool_calls: self.tool_calls,
            prompt_tokens: self.prompt_tokens,
            completion_tokens,
            model,
        }
    }
}

/// Map internal messages to Ollama's native `/api/chat` wire format.
pub fn native_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            if m.role == Role::Tool {
                let mut o = Map::new();
                o.insert("role".into(), json!("tool"));
                o.insert("content".into(), json!(m.content));
                if let Some(name) = m.name.as_deref().filter(|n| !n.is_empty()) {
                    o.insert("tool_name".into(), json!(name));
                }
                return Value::Object(o);
            }
            if m.role == Role::Assistant {
                if let Some(calls) = m.tool_calls.as_ref().filter(|c| !c.is_empty()) {
                    let calls: Vec<Value> = calls
                        .iter()
                        .map(|tc| json!({ "function": { "name": tc.name, "arguments": tc.arguments } }))
                        .collect();
                    return json!({ "role": "assistant", "content": m.content, "tool_calls": calls });
                }
            }
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            json!({ "role": role, "content": m.content })
        })
        .collect()
}

/// Native `tool_calls` carry `arguments` as an object (sometimes a JSON string).
pub fn parse_native_tool_calls(raw: Option<&Value>) -> Vec<ToolCall> {
    // Same normalisation as the OpenAI shape: string-or-object arguments,
    // nameless calls dropped, `call_<i>` ids when missing.
    wire::parse_tool_calls(raw)
}

fn tag_names(json: &Value) -> Vec<String> {
    json.get("models")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|m| m.get("name").and_then(Value::as_str).map(str::to_string)).collect())
        .unwrap_or_default()
}

/// `/api/tags` → models with human sizes, sorted by name.
pub fn parse_tags(json: &Value) -> Vec<ModelInfo> {
    let mut out: Vec<ModelInfo> = json
        .get("models")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let name = m.get("name").and_then(Value::as_str).filter(|s| !s.is_empty())?;
                    Some(ModelInfo { name: name.to_string(), size: format_bytes(m.get("size").and_then(Value::as_f64)) })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| match a.name.to_lowercase().cmp(&b.name.to_lowercase()) {
        Ordering::Equal => a.name.cmp(&b.name),
        o => o,
    });
    out
}

/// Bytes → "4.7 GB" (one decimal below 10, none at/above 10 or for bytes).
pub fn format_bytes(bytes: Option<f64>) -> Option<String> {
    let bytes = bytes.filter(|b| *b > 0.0)?;
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut n = bytes;
    let mut i = 0;
    while n >= 1024.0 && i < UNITS.len() - 1 {
        n /= 1024.0;
        i += 1;
    }
    let digits = if n >= 10.0 || i == 0 { 0 } else { 1 };
    Some(format!("{n:.digits$} {}", UNITS[i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qo_core::ToolSpec;

    #[test]
    fn human_sizes() {
        assert_eq!(format_bytes(None), None);
        assert_eq!(format_bytes(Some(0.0)), None);
        assert_eq!(format_bytes(Some(512.0)).as_deref(), Some("512 B"));
        assert_eq!(format_bytes(Some(4_700_000_000.0)).as_deref(), Some("4.4 GB"));
        assert_eq!(format_bytes(Some(19_000_000_000.0)).as_deref(), Some("18 GB"));
    }

    #[test]
    fn tags_sorted_with_sizes() {
        let v = json!({"models": [{"name": "qwen3.5:9b", "size": 6_600_000_000u64}, {"name": "Llama3:8b"}, {"size": 1}]});
        let m = parse_tags(&v);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].name, "Llama3:8b");
        assert_eq!(m[0].size, None);
        assert_eq!(m[1].size.as_deref(), Some("6.1 GB"));
    }

    #[test]
    fn chat_body_options_mapping() {
        let mut cfg = ProviderConfig::new("qwen3.5:9b", "http://localhost:11434/v1/");
        cfg.gpu_layers = Some(20);
        cfg.low_vram = true;
        cfg.keep_alive = Some("30m".into());
        cfg.context_tokens = Some(16384);
        cfg.top_k = Some(20);
        let p = OllamaProvider::new(cfg);
        assert_eq!(p.root(), "http://localhost:11434");
        let tools = vec![ToolSpec { name: "ls".into(), description: "list".into(), parameters: json!({"type": "object"}) }];
        let opts = ModelOptions { json: true, tools: Some(tools), ..Default::default() };
        let b = p.chat_body(&[Message::user("hi")], &opts, true, "qwen3.5:9b");
        assert_eq!(b["options"]["num_ctx"], 16384);
        assert_eq!(b["options"]["num_gpu"], 20);
        assert_eq!(b["options"]["low_vram"], true);
        assert_eq!(b["options"]["top_k"], 20);
        assert!(b["options"].get("repeat_penalty").is_none());
        assert_eq!(b["keep_alive"], "30m");
        assert_eq!(b["format"], "json");
        // Qwen3.5 hybrid, think auto → explicit think:false.
        assert_eq!(b["think"], false);
        assert_eq!(b["tools"][0]["function"]["name"], "ls");

        let p = OllamaProvider::new(ProviderConfig::new("llama3", "http://localhost:11434"));
        let b = p.chat_body(&[Message::user("hi")], &ModelOptions::default(), false, "llama3");
        assert!(b.get("think").is_none());
        assert!(b["options"].get("num_gpu").is_none());
        assert_eq!(b["options"]["num_ctx"], 8192);
    }

    #[test]
    fn native_message_shapes() {
        let call = ToolCall { id: "c1".into(), name: "ls".into(), arguments: json!({"p": "."}) };
        let w = native_messages(&[Message::assistant_calls("", vec![call]), Message::tool_result("c1", "ls", "a.txt")]);
        assert_eq!(w[0]["tool_calls"][0], json!({"function": {"name": "ls", "arguments": {"p": "."}}}));
        assert_eq!(w[1], json!({"role": "tool", "content": "a.txt", "tool_name": "ls"}));
    }
}
