//! `think` — a scratchpad tool for deliberation between actions. Small models
//! benefit from an explicit place to reason: without one they either skip
//! thinking (and act wrong) or leak chain-of-thought into their answer. A think
//! call spends a THINKING turn (agent.max_thinking_turns), not a tool step, so
//! deliberating never starves the budget for real work — see the chat loop.
//!
//! The thought is preserved in history via the tool_call arguments; the tool
//! result is a short ack so the thought isn't fed back twice. Unlike the TS
//! (which printed "qwen ⟂ thinking" itself), this library never writes to the
//! terminal: the thought is returned in `ToolResult.data` as
//! `{"thought": "..."}` so the UI layer renders it.
//!
//! Clean-room implementation — inspired by the "think" tool pattern published
//! for Claude-class agents, written from scratch for Qwenodyssey.

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult, ToolSpec};
use serde_json::{json, Value};

use crate::util::js_string_or_empty;

const THINK_DESCRIPTION: &str = "Pause and reason privately before your next action. Use it to analyse a tool result, weigh \
options, or plan the next step when the task is tricky. The thought is not shown as your \
answer and runs nothing. After thinking, make the actual tool call or give your final answer.";

/// The `think` tool's schema (advertised only when agent.thinking_mode is on).
pub fn think_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "think".to_string(),
        description: THINK_DESCRIPTION.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "thought": { "type": "string", "description": "Your reasoning about what you learned and what to do next." }
            },
            "required": ["thought"]
        }),
    }
}

pub struct ThinkTool;

/// The `think` tool.
pub fn create_think_tool() -> ThinkTool {
    ThinkTool
}

/// The ack fed back to the model after a recorded thought.
pub const THINK_ACK: &str = "Thought noted. Now take the next step: call the tool you need, or give your final answer.";

#[async_trait]
impl Tool for ThinkTool {
    fn name(&self) -> &str {
        "think"
    }
    fn description(&self) -> &str {
        THINK_DESCRIPTION
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let thought = js_string_or_empty(args.get("thought")).trim().to_string();
        if thought.is_empty() {
            return ToolResult::err("Provide `thought` — the reasoning you want to record.");
        }
        ctx.log(json!({ "tool": "think", "chars": thought.encode_utf16().count() }));
        ToolResult::ok(THINK_ACK).with_data(json!({ "thought": thought }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // smallModelOptims.test.ts: think tool
    #[tokio::test]
    async fn records_a_thought_and_points_at_next_step() {
        let r = create_think_tool().run(json!({ "thought": "the test failed because of X" }), &ToolContext::new(".")).await;
        assert!(r.ok);
        assert!(r.output.to_lowercase().contains("next step"));
        assert_eq!(r.data.unwrap()["thought"], json!("the test failed because of X"));
    }

    #[tokio::test]
    async fn rejects_empty_thought() {
        let r = create_think_tool().run(json!({ "thought": "  " }), &ToolContext::new(".")).await;
        assert!(!r.ok);
    }
}
