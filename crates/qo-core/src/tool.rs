//! Tools: the trait, the per-session context, and the registry agents call into.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolResult {
    pub ok: bool,
    pub output: String,
    pub data: Option<Value>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self { ok: true, output: output.into(), data: None }
    }
    pub fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into(), data: None }
    }
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Asks the user a yes/no question (destructive-command gate). The interactive
/// CLI implements this on top of its single input owner, so a confirmation can
/// never fight the prompt for the terminal. Non-interactive runs pass `None`
/// (and then destructive actions are refused unless `auto_confirm`).
#[async_trait]
pub trait Confirmer: Send + Sync {
    async fn confirm(&self, question: &str) -> bool;
}

pub type LogFn = Arc<dyn Fn(Value) + Send + Sync>;

#[derive(Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
    /// Skip confirmation prompts when true (e.g. --yes).
    pub auto_confirm: bool,
    pub confirm_destructive: bool,
    pub allow_shell: bool,
    pub sandbox: bool,
    /// Config tools.allow_commands — auto-approved (skip the destructive gate).
    pub allow_commands: Vec<String>,
    /// Config tools.deny_commands — hard-blocked.
    pub deny_commands: Vec<String>,
    /// Default run_shell timeout in ms (config tools.shell_timeout_ms).
    pub shell_timeout_ms: Option<u64>,
    /// The agent's own source root — writable only when tools.allow_self_edit.
    pub self_root: Option<PathBuf>,
    /// Structured event log (JSONL session log).
    pub log: LogFn,
    pub confirmer: Option<Arc<dyn Confirmer>>,
    /// Esc interrupt for the current turn: long-running tools (run_shell, shell
    /// session, web fetch) stop and report "interrupted" when it is cancelled.
    /// The chat layer installs a fresh token per turn.
    pub cancel: Option<CancellationToken>,
}

impl ToolContext {
    /// A context for `cwd` with safe defaults and a no-op logger.
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            auto_confirm: false,
            confirm_destructive: true,
            allow_shell: true,
            sandbox: true,
            allow_commands: Vec::new(),
            deny_commands: Vec::new(),
            shell_timeout_ms: None,
            self_root: None,
            log: Arc::new(|_| {}),
            confirmer: None,
            cancel: None,
        }
    }

    /// Has the current turn been interrupted?
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.is_cancelled())
    }

    pub fn log(&self, entry: Value) {
        (self.log)(entry)
    }

    /// Ask before a destructive action. Auto-confirm wins; without a
    /// confirmer the answer is "no".
    pub async fn confirm(&self, question: &str) -> bool {
        if self.auto_confirm {
            return true;
        }
        match &self.confirmer {
            Some(c) => c.confirm(question).await,
            None => false,
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Whether the tool can modify the filesystem / run commands (mutating
    /// calls run serially; read-only ones may run in parallel).
    fn mutating(&self) -> bool;
    /// `args` is always a JSON object.
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult;
}

pub type SharedTool = Arc<dyn Tool>;

/// Central catalog of tools, keyed by name (later registrations replace earlier
/// ones with the same name). Default tool sets are added by the crates that
/// own them, e.g. `qo_tools::register_defaults(&mut registry)`.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, SharedTool>,
    ctx: ToolContext,
}

impl ToolRegistry {
    pub fn new(ctx: ToolContext) -> Self {
        Self { tools: BTreeMap::new(), ctx }
    }

    pub fn register(&mut self, tool: SharedTool) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn register_all(&mut self, tools: impl IntoIterator<Item = SharedTool>) {
        for t in tools {
            self.register(t);
        }
    }

    pub fn context(&self) -> &ToolContext {
        &self.ctx
    }

    pub fn context_mut(&mut self) -> &mut ToolContext {
        &mut self.ctx
    }

    pub fn get(&self, name: &str) -> Option<SharedTool> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn list(&self) -> Vec<SharedTool> {
        self.tools.values().cloned().collect()
    }

    pub async fn run(&self, name: &str, args: Value) -> ToolResult {
        let Some(tool) = self.get(name) else {
            return ToolResult::err(format!("Unknown tool: {name}"));
        };
        let args = if args.is_object() { args } else { Value::Object(Default::default()) };
        tool.run(args, &self.ctx).await
    }
}
