//! Shared data types (ported from src/types/index.ts).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A model's request to invoke a tool (OpenAI-style function call).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Always a JSON object (`{}` when the model sent nothing / garbage).
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Present on assistant messages that requested tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on tool-result messages: which call this answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name, for tool-result messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), tool_calls: None, tool_call_id: None, name: None }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
    /// Assistant turn that requested tool calls.
    pub fn assistant_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self { tool_calls: Some(calls), ..Self::assistant(content) }
    }
    /// Tool-result message answering `call_id`.
    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            name: Some(name.into()),
        }
    }
}

/// OpenAI-style tool/function schema advertised to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema object for the arguments.
    pub parameters: Value,
}

/// Per-request knobs. `None` means "use the provider's configured default" and,
/// for the optional sampling knobs, "leave it off the wire".
#[derive(Clone, Debug, Default)]
pub struct ModelOptions {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub presence_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
    /// Hint that the response should be valid JSON.
    pub json: bool,
    /// Tools the model may call this turn (`None`/empty = no tools on the wire;
    /// OpenAI-compatible APIs reject `tools: []`).
    pub tools: Option<Vec<ToolSpec>>,
    /// `Some(true)` requests thinking, `Some(false)` explicitly disables it on
    /// hybrid models (Qwen3/3.5), `None` leaves the flag off the wire.
    pub think: Option<bool>,
    /// Esc interrupt: cancelling aborts the request immediately (connect, headers,
    /// body) and the provider returns [`crate::ProviderError::Interrupted`].
    pub cancel: Option<CancellationToken>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateResult {
    pub text: String,
    /// Tool calls the model requested, if any.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Separate chain-of-thought, when the backend returns it.
    #[serde(default)]
    pub thinking: Option<String>,
    /// Exact usage when the backend reports it.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    pub model: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    /// Human-readable size, e.g. "4.7 GB", when the backend reports it.
    #[serde(default)]
    pub size: Option<String>,
}

/* ── Agent pipeline ── */

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub index: u32,
    pub title: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub assumptions: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    Patch,
    Create,
    Rewrite,
    Delete,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileEdit {
    pub path: String,
    pub mode: EditMode,
    /// Unified diff (for mode == Patch).
    #[serde(default)]
    pub diff: Option<String>,
    /// Full file contents (for Create / Rewrite).
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub rationale: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CoderResult {
    pub edits: Vec<FileEdit>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewIssue {
    pub severity: Severity,
    #[serde(default)]
    pub file: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewResult {
    pub approve: bool,
    #[serde(default)]
    pub issues: Vec<ReviewIssue>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub ran: bool,
    pub passed: bool,
    #[serde(default)]
    pub command: Option<String>,
    pub output: String,
}

/* ── Repo scanning ── */

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RepoInfo {
    pub root: String,
    pub languages: Vec<String>,
    pub framework: Option<String>,
    pub package_manager: Option<String>,
    pub test_command: Option<String>,
    pub lint_command: Option<String>,
    pub build_command: Option<String>,
    pub entrypoints: Vec<String>,
    pub key_files: Vec<String>,
    pub file_count: usize,
    pub has_git: bool,
    pub dirty: bool,
}

/* ── Modes ── */

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentMode {
    Fast,
    #[default]
    Safe,
    Deep,
    Pair,
    Autofix,
}

impl AgentMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentMode::Fast => "fast",
            AgentMode::Safe => "safe",
            AgentMode::Deep => "deep",
            AgentMode::Pair => "pair",
            AgentMode::Autofix => "autofix",
        }
    }
}
