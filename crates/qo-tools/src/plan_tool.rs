//! In-session plan / TODO tracking. Gives the model an `update_plan` tool so it
//! can lay out the steps of a multi-step task and keep their status current as it
//! works — which markedly improves a small model's coherence over long tool
//! chains (it stops "losing the thread"). The current plan is also shown by the
//! /plan chat command (via [`PlanState::items`] + [`render_plan`]).
//!
//! Clean-room implementation — inspired by the to-do/checklist tool in
//! Claude-Code-class agents, written from scratch for Qwenodyssey.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::util::{js_string, js_truthy};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    #[default]
    Pending,
    InProgress,
    Done,
}

impl PlanStatus {
    fn mark(&self) -> &'static str {
        match self {
            PlanStatus::Pending => "[ ]",
            PlanStatus::InProgress => "[~]",
            PlanStatus::Done => "[x]",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanItem {
    pub step: String,
    pub status: PlanStatus,
}

/// Shared, mutable holder so chat can render the latest plan on demand.
/// Cloning shares the same underlying plan.
#[derive(Clone, Debug, Default)]
pub struct PlanState {
    items: Arc<Mutex<Vec<PlanItem>>>,
}

impl PlanState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of the current plan.
    pub fn items(&self) -> Vec<PlanItem> {
        match self.items.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }

    /// Replace the plan.
    pub fn set(&self, items: Vec<PlanItem>) {
        match self.items.lock() {
            Ok(mut g) => *g = items,
            Err(p) => *p.into_inner() = items,
        }
    }

    pub fn clear(&self) {
        self.set(Vec::new());
    }

    /// The plan rendered as a checklist ([`render_plan`]).
    pub fn render(&self) -> String {
        render_plan(&self.items())
    }
}

/// Render a plan as `[x] 1. step` lines ("(no plan yet)" when empty).
pub fn render_plan(items: &[PlanItem]) -> String {
    if items.is_empty() {
        return "(no plan yet)".to_string();
    }
    items.iter().enumerate().map(|(i, it)| format!("{} {}. {}", it.status.mark(), i + 1, it.step)).collect::<Vec<_>>().join("\n")
}

/// `String(x ?? fallback)` treating only null/undefined as absent.
fn first_present<'a>(o: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| o.get(*k).filter(|v| !v.is_null()))
}

/// Coerce whatever list the model sent into plan items: objects with
/// step/title/task + status, or bare strings.
fn coerce(raw: Option<&Value>) -> Vec<PlanItem> {
    let Some(Value::Array(list)) = raw else {
        return Vec::new();
    };
    list.iter()
        .map(|it| {
            let step = match it {
                Value::Object(_) => match first_present(it, &["step", "title", "task"]) {
                    Some(v) => js_string(Some(v)),
                    None => js_string(Some(it)),
                },
                Value::Null => String::new(),
                other => js_string(Some(other)),
            };
            let status_raw = match it.get("status") {
                Some(v) if js_truthy(Some(v)) => js_string(Some(v)).to_lowercase(),
                _ => "pending".to_string(),
            };
            let status = match status_raw.as_str() {
                "done" | "completed" => PlanStatus::Done,
                "in_progress" | "active" => PlanStatus::InProgress,
                _ => PlanStatus::Pending,
            };
            PlanItem { step: step.trim().to_string(), status }
        })
        .filter(|i| !i.step.is_empty())
        .collect()
}

pub struct PlanTool {
    state: PlanState,
}

/// The `update_plan` tool writing into `state`.
pub fn create_plan_tool(state: PlanState) -> PlanTool {
    PlanTool { state }
}

#[async_trait]
impl Tool for PlanTool {
    fn name(&self) -> &str {
        "update_plan"
    }
    fn description(&self) -> &str {
        "Create or update your step-by-step plan for a multi-step task and keep each step's status \
         current as you work. Pass the FULL plan each time (the list is replaced)."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = first_present(&args, &["plan", "steps", "todos"]);
        let items = coerce(raw);
        if items.is_empty() {
            return ToolResult::err("Provide `plan` as a non-empty array of { step, status }.");
        }
        let rendered = render_plan(&items);
        let n = items.len();
        self.state.set(items);
        ctx.log(json!({ "tool": "update_plan", "steps": n }));
        ToolResult::ok(format!("Plan updated:\n{rendered}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // upgrades.test.ts: planTool
    #[tokio::test]
    async fn stores_coerces_and_renders() {
        let state = PlanState::new();
        let tool = create_plan_tool(state.clone());
        let r = tool
            .run(
                json!({ "plan": [
                    { "step": "scan repo", "status": "done" },
                    { "step": "edit file", "status": "in_progress" },
                    { "step": "run tests", "status": "todo" },
                ] }),
                &ToolContext::new("."),
            )
            .await;
        assert!(r.ok);
        let items = state.items();
        assert_eq!(items.len(), 3);
        assert_eq!(items[2].status, PlanStatus::Pending);
        let out = render_plan(&items);
        assert!(out.contains("[x] 1. scan repo"));
        assert!(out.contains("[~] 2. edit file"));
        assert!(out.contains("[ ] 3. run tests"));
    }

    #[tokio::test]
    async fn rejects_empty_plan() {
        let tool = create_plan_tool(PlanState::new());
        let r = tool.run(json!({ "plan": [] }), &ToolContext::new(".")).await;
        assert!(!r.ok);
    }

    #[tokio::test]
    async fn accepts_bare_strings_and_aliases() {
        let state = PlanState::new();
        let tool = create_plan_tool(state.clone());
        let r = tool.run(json!({ "todos": ["one", { "title": "two", "status": "completed" }] }), &ToolContext::new(".")).await;
        assert!(r.ok);
        assert_eq!(state.render(), "[ ] 1. one\n[x] 2. two");
    }
}
