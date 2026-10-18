//! OpenAI wire-format helpers shared by the OpenAI-compatible backends (the
//! free functions of base.ts): message mapping, system-prompt merging/folding,
//! tool-call parsing and streamed tool-call assembly, error-envelope rendering.

use std::collections::BTreeMap;

use qo_core::{Message, Role, ToolCall, ToolSpec};
use serde_json::{json, Map, Value};

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Map internal messages to the OpenAI chat wire format (incl. tool fields).
/// System messages are merged first; when `system_role_supported` is false the
/// merged system prompt is folded into the first user turn.
pub fn wire_messages(messages: &[Message], system_role_supported: bool) -> Vec<Value> {
    let mut prepared = merge_system(messages);
    if !system_role_supported {
        prepared = fold_system_into_user(&prepared);
    }
    prepared
        .iter()
        .map(|m| {
            if m.role == Role::Tool {
                return json!({
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                });
            }
            if m.role == Role::Assistant {
                if let Some(calls) = m.tool_calls.as_ref().filter(|c| !c.is_empty()) {
                    let calls: Vec<Value> = calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            })
                        })
                        .collect();
                    return json!({ "role": "assistant", "content": m.content, "tool_calls": calls });
                }
            }
            json!({ "role": role_str(m.role), "content": m.content })
        })
        .collect()
}

/// Collapse multiple `system` messages into one (at the first system's
/// position). Some chat templates (e.g. LM Studio's qwen3.5) error with "No
/// user query found" when given more than one system message.
pub fn merge_system(messages: &[Message]) -> Vec<Message> {
    let systems: Vec<&str> =
        messages.iter().filter(|m| m.role == Role::System).map(|m| m.content.as_str()).collect();
    if systems.len() <= 1 {
        return messages.to_vec();
    }
    let merged = systems.join("\n\n");
    let mut out = Vec::with_capacity(messages.len());
    let mut placed = false;
    for m in messages {
        if m.role == Role::System {
            if !placed {
                out.push(Message::system(merged.clone()));
                placed = true;
            }
        } else {
            out.push(m.clone());
        }
    }
    out
}

/// Demote a leading `system` message into the conversation as user content,
/// for backends/models that mishandle the system role (Kimi K2.x on NIM loops
/// with one). Assumes [`merge_system`] already ran. The system text is
/// prepended to the first user message, or inserted as a standalone user
/// message when there is none, so the model still receives the instructions.
pub fn fold_system_into_user(messages: &[Message]) -> Vec<Message> {
    let Some(sys) = messages.iter().find(|m| m.role == Role::System) else {
        return messages.to_vec();
    };
    let mut rest: Vec<Message> = messages.iter().filter(|m| m.role != Role::System).cloned().collect();
    match rest.iter().position(|m| m.role == Role::User) {
        None => {
            rest.insert(0, Message::user(sys.content.clone()));
            rest
        }
        Some(i) => {
            rest[i].content = format!("{}\n\n---\n\n{}", sys.content, rest[i].content);
            rest
        }
    }
}

/// OpenAI-style `tools` array.
pub fn wire_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.parameters },
            })
        })
        .collect()
}

/// Human-readable text for an OpenAI-style `error` envelope (object or string):
/// `code type message`, falling back to the raw JSON.
pub fn describe_api_error(err: &Value) -> String {
    if let Value::String(s) = err {
        return s.clone();
    }
    let parts: Vec<String> = ["code", "type", "message"]
        .iter()
        .filter_map(|k| err.get(*k))
        .filter(|v| !v.is_null() && v.as_str() != Some(""))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .collect();
    if parts.is_empty() {
        err.to_string()
    } else {
        parts.join(" ")
    }
}

/// Is this JSON value "truthy" in the JS sense (used for `obj.error` checks)?
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Parse tool-call arguments sent as a JSON string. The result is always an
/// object: an empty string is `{}`, garbage is kept as `{"_raw": text}`, and a
/// double-encoded object (`"{\"a\":1}"`) is unwrapped.
pub fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(Map::new());
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(v @ Value::Object(_)) => v,
        Ok(Value::String(inner)) => match serde_json::from_str::<Value>(&inner) {
            Ok(v @ Value::Object(_)) => v,
            _ => json!({ "_raw": raw }),
        },
        _ => json!({ "_raw": raw }),
    }
}

/// Arguments that may arrive either as a JSON string or as an object already.
pub(crate) fn arguments_value(v: Option<&Value>) -> Value {
    match v {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::String(s)) => parse_arguments(s),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(other) => json!({ "_raw": other }),
    }
}

/// Normalize OpenAI-style `tool_calls` (non-streamed) into [`ToolCall`]s.
/// Calls without a function name are dropped; missing ids become `call_<i>`.
pub fn parse_tool_calls(raw: Option<&Value>) -> Vec<ToolCall> {
    let Some(arr) = raw.and_then(Value::as_array) else {
        return Vec::new();
    };
    arr.iter()
        .enumerate()
        .filter_map(|(i, tc)| {
            let func = tc.get("function")?;
            let name = func.get("name").and_then(Value::as_str).filter(|n| !n.is_empty())?;
            let id = tc
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("call_{i}"));
            Some(ToolCall { id, name: name.to_string(), arguments: arguments_value(func.get("arguments")) })
        })
        .collect()
}

/// Accumulator for tool calls that arrive as streamed fragments (OpenAI sends
/// the name on the first frame and the arguments byte-by-byte after), keyed
/// by the fragment `index`.
#[derive(Default, Debug)]
pub struct ToolCallAccumulator {
    calls: BTreeMap<u64, PartialCall>,
}

#[derive(Default, Debug)]
struct PartialCall {
    id: String,
    name: String,
    args: String,
}

impl ToolCallAccumulator {
    /// Feed one `delta.tool_calls` array.
    pub fn push(&mut self, fragments: &Value) {
        let Some(arr) = fragments.as_array() else { return };
        for tc in arr {
            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
            let cur = self.calls.entry(idx).or_default();
            if let Some(id) = tc.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                cur.id = id.to_string();
            }
            if let Some(func) = tc.get("function") {
                if let Some(name) = func.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    cur.name = name.to_string();
                }
                if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                    cur.args.push_str(args);
                }
            }
        }
    }

    /// Finished calls in index order (nameless fragments dropped).
    pub fn finish(self) -> Vec<ToolCall> {
        self.calls
            .into_iter()
            .filter(|(_, c)| !c.name.is_empty())
            .map(|(idx, c)| ToolCall {
                id: if c.id.is_empty() { format!("call_{idx}") } else { c.id },
                name: c.name,
                arguments: parse_arguments(&c.args),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_system_collapses_multiple() {
        let msgs = vec![
            Message::system("a"),
            Message::user("hi"),
            Message::system("b"),
        ];
        let out = merge_system(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Role::System);
        assert_eq!(out[0].content, "a\n\nb");
        assert_eq!(out[1].content, "hi");
    }

    #[test]
    fn fold_system_prepends_to_first_user() {
        let out = fold_system_into_user(&[Message::system("rules"), Message::user("hi")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, Role::User);
        assert_eq!(out[0].content, "rules\n\n---\n\nhi");
        let out = fold_system_into_user(&[Message::system("rules"), Message::assistant("x")]);
        assert_eq!(out[0].role, Role::User);
        assert_eq!(out[0].content, "rules");
        assert_eq!(out[1].role, Role::Assistant);
    }

    #[test]
    fn wire_messages_maps_tool_fields() {
        let call = ToolCall { id: "c1".into(), name: "ls".into(), arguments: json!({"p": "."}) };
        let w = wire_messages(
            &[Message::assistant_calls("", vec![call]), Message::tool_result("c1", "ls", "a.txt")],
            true,
        );
        assert_eq!(w[0]["tool_calls"][0]["function"]["arguments"], "{\"p\":\".\"}");
        assert_eq!(w[0]["tool_calls"][0]["type"], "function");
        assert_eq!(w[1], json!({"role":"tool","content":"a.txt","tool_call_id":"c1","name":"ls"}));
    }

    #[test]
    fn describe_api_error_renders_objects_and_strings() {
        assert_eq!(describe_api_error(&json!("boom")), "boom");
        assert_eq!(
            describe_api_error(&json!({"code": 503, "type": "service_unavailable", "message": "overloaded"})),
            "503 service_unavailable overloaded"
        );
        assert_eq!(describe_api_error(&json!({"weird": 1})), "{\"weird\":1}");
    }

    #[test]
    fn parse_arguments_always_object() {
        assert_eq!(parse_arguments(""), json!({}));
        assert_eq!(parse_arguments("{\"a\":1}"), json!({"a": 1}));
        assert_eq!(parse_arguments("{bad"), json!({"_raw": "{bad"}));
        assert_eq!(parse_arguments("\"{\\\"a\\\":1}\""), json!({"a": 1}));
        assert_eq!(parse_arguments("[1]"), json!({"_raw": "[1]"}));
    }

    #[test]
    fn parse_tool_calls_handles_string_and_object_args() {
        let raw = json!([
            {"id": "x", "function": {"name": "a", "arguments": "{\"k\":1}"}},
            {"function": {"name": "b", "arguments": {"k": 2}}},
            {"function": {"arguments": "{}"}},
        ]);
        let calls = parse_tool_calls(Some(&raw));
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "x");
        assert_eq!(calls[0].arguments, json!({"k": 1}));
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments, json!({"k": 2}));
    }
}
