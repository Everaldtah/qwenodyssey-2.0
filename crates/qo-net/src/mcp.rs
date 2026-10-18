//! Minimal Model Context Protocol (MCP) client over the stdio transport, and
//! the bridge that exposes MCP servers' tools as native harness tools.
//!
//! MCP servers speak JSON-RPC 2.0 as newline-delimited JSON on stdin/stdout. We
//! spawn the server, do the initialize handshake (`initialize` →
//! `notifications/initialized`), then expose `tools/list` and `tools/call`. A
//! clean-room implementation of the open wire protocol — no SDK dependency.
//! Supports stdio servers (the common case: `npx -y @modelcontextprotocol/...`);
//! HTTP/SSE transports are out of scope.
//!
//! Tool names are namespaced `mcp__<server>__<tool>` so two servers can both
//! have a "search" tool without colliding, and so the model can tell at a glance
//! a call goes out to an MCP server.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use qo_core::tool::SharedTool;
use qo_core::{CancellationToken, Tool, ToolContext, ToolResult, ToolSpec};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{oneshot, Notify};

use crate::util::{cancellable, take_chars, truthy, value_to_string, INTERRUPTED};

/// The protocol revision we advertise; the server echoes whatever it supports.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// How much server stderr is kept for error messages.
const STDERR_KEEP: usize = 8000;

/// One configured server (`[mcp.servers.<name>]`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct McpServerSpec {
    /// Stable short name, used to namespace the server's tools (e.g. "filesystem").
    pub name: String,
    /// Executable to launch (e.g. "npx", "uvx", "node", an absolute path).
    pub command: String,
    pub args: Vec<String>,
    /// Extra environment variables merged over the parent process env.
    pub env: BTreeMap<String, String>,
    /// Working directory for the server process (default: ours).
    pub cwd: Option<PathBuf>,
}

/// Per-request deadlines (`mcp.init_timeout_ms` / `mcp.call_timeout_ms`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct McpOptions {
    /// Deadline for `initialize` and `tools/list`.
    pub init_timeout_ms: u64,
    /// Deadline for each `tools/call`.
    pub call_timeout_ms: u64,
}

impl Default for McpOptions {
    fn default() -> Self {
        Self { init_timeout_ms: 20_000, call_timeout_ms: 60_000 }
    }
}

/// A tool as the server describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments (MCP calls this `inputSchema`).
    pub input_schema: Value,
}

/// A `tools/call` result with its content flattened to text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpCallResult {
    pub text: String,
    pub is_error: bool,
}

type Reply = Result<Value, String>;

/// State shared between the client handle and its reader / waiter tasks.
struct Shared {
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
    stderr: Mutex<String>,
    /// `dispose()` was called.
    closed: AtomicBool,
    /// Why the server is gone (spawn failure / exit) — new requests fail fast with it.
    dead: Mutex<Option<String>>,
    /// The child process is alive.
    running: AtomicBool,
    /// Fired by `dispose()`: the waiter task kills the process (tree).
    kill: CancellationToken,
    /// The stdout reader hit EOF (so every response the server wrote was seen).
    stdout_done: Notify,
    stdout_eof: AtomicBool,
}

impl Shared {
    fn fail_all(&self, msg: &str) {
        let drained: Vec<_> = match self.pending.lock() {
            Ok(mut p) => p.drain().collect(),
            Err(_) => return,
        };
        for (_, tx) in drained {
            let _ = tx.send(Err(msg.to_string()));
        }
    }

    fn tail_stderr(&self) -> String {
        let s = self.stderr.lock().map(|s| s.clone()).unwrap_or_default();
        let lines: Vec<&str> = s.split('\n').map(|l| l.trim_end_matches('\r')).filter(|l| !l.is_empty()).collect();
        lines[lines.len().saturating_sub(3)..].join(" | ")
    }

    fn dead_reason(&self) -> Option<String> {
        self.dead.lock().ok().and_then(|d| d.clone())
    }

    fn set_dead(&self, msg: &str) {
        if let Ok(mut d) = self.dead.lock() {
            d.get_or_insert_with(|| msg.to_string());
        }
    }

    async fn write_line(&self, payload: &str) -> std::io::Result<()> {
        let mut guard = self.stdin.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "stdin closed"));
        };
        stdin.write_all(payload.as_bytes()).await?;
        stdin.flush().await
    }
}

/// Removes a request's pending slot if the request future is dropped before
/// its response arrives (timeout, Esc, tool cancelled) and tells the server
/// with `notifications/cancelled`, as the spec asks.
struct PendingGuard {
    shared: Arc<Shared>,
    id: u64,
    done: bool,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let was_pending = self.shared.pending.lock().map(|mut p| p.remove(&self.id).is_some()).unwrap_or(false);
        if !was_pending || self.shared.closed.load(Ordering::SeqCst) || self.shared.dead_reason().is_some() {
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let shared = self.shared.clone();
            let payload = json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": self.id, "reason": "cancelled by client"},
            })
            .to_string()
                + "\n";
            handle.spawn(async move {
                let _ = shared.write_line(&payload).await;
            });
        }
    }
}

/// A connection to one stdio MCP server.
pub struct McpClient {
    spec: McpServerSpec,
    opts: McpOptions,
    shared: Arc<Shared>,
    initialized: AtomicBool,
    pid: Mutex<Option<u32>>,
    /// The `initialize` result (protocolVersion, capabilities, serverInfo).
    server_info: Mutex<Option<Value>>,
}

impl McpClient {
    pub fn new(spec: McpServerSpec, opts: McpOptions) -> Self {
        Self {
            spec,
            opts,
            shared: Arc::new(Shared {
                next_id: AtomicU64::new(1),
                pending: Mutex::new(HashMap::new()),
                stdin: tokio::sync::Mutex::new(None),
                stderr: Mutex::new(String::new()),
                closed: AtomicBool::new(false),
                dead: Mutex::new(None),
                running: AtomicBool::new(false),
                kill: CancellationToken::new(),
                stdout_done: Notify::new(),
                stdout_eof: AtomicBool::new(false),
            }),
            initialized: AtomicBool::new(false),
            pid: Mutex::new(None),
            server_info: Mutex::new(None),
        }
    }

    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// The initialize handshake completed.
    pub fn initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// The server process is alive.
    pub fn running(&self) -> bool {
        self.shared.running.load(Ordering::SeqCst)
    }

    /// OS process id of the spawned server (the `cmd.exe` wrapper for a
    /// `.cmd`/`.bat` launcher on Windows).
    pub fn pid(&self) -> Option<u32> {
        self.pid.lock().ok().and_then(|p| *p)
    }

    /// What the server answered to `initialize` (its `serverInfo`, protocol
    /// version and capabilities), once started.
    pub fn server_info(&self) -> Option<Value> {
        self.server_info.lock().ok().and_then(|s| s.clone())
    }

    /// Spawn the server and complete the MCP initialize handshake.
    pub async fn start(&self) -> anyhow::Result<()> {
        if self.shared.closed.load(Ordering::SeqCst) {
            anyhow::bail!("client closed");
        }
        let cwd = match &self.spec.cwd {
            Some(c) if !c.as_os_str().is_empty() => c.clone(),
            _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        };
        let program = resolve_program(&self.spec.command, &self.spec.env, &cwd);
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&self.spec.args)
            .envs(&self.spec.env)
            .current_dir(&cwd)
            // stdin is the protocol channel, so it is piped (never the terminal).
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("spawn failed: {e}");
                self.shared.set_dead(&msg);
                anyhow::bail!(msg);
            }
        };
        self.shared.running.store(true, Ordering::SeqCst);
        if let Ok(mut p) = self.pid.lock() {
            *p = child.id();
        }
        *self.shared.stdin.lock().await = child.stdin.take();
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(read_stdout(self.shared.clone(), stdout));
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(read_stderr(self.shared.clone(), stderr));
        }
        tokio::spawn(wait_child(self.shared.clone(), child));

        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "qwenodyssey", "version": env!("CARGO_PKG_VERSION")},
        });
        let info = self.request("initialize", params, self.opts.init_timeout_ms).await?;
        if let Ok(mut s) = self.server_info.lock() {
            *s = Some(info);
        }
        // Per spec, follow up with the initialized notification before any other call.
        self.notify("notifications/initialized", None).await;
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Discover the tools this server exposes.
    pub async fn list_tools(&self) -> anyhow::Result<Vec<McpToolDef>> {
        let res = self.request("tools/list", json!({}), self.opts.init_timeout_ms).await?;
        let tools = res.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(tools
            .iter()
            .map(|t| McpToolDef {
                name: t.get("name").map(value_to_string).unwrap_or_default(),
                description: t
                    .get("description")
                    .filter(|d| truthy(Some(d)))
                    .map(value_to_string)
                    .unwrap_or_default(),
                input_schema: match t.get("inputSchema") {
                    Some(s) if s.is_object() => s.clone(),
                    _ => json!({"type": "object", "properties": {}}),
                },
            })
            .collect())
    }

    /// Invoke a tool; returns its content flattened to a single text string.
    pub async fn call_tool(&self, name: &str, args: Value) -> anyhow::Result<McpCallResult> {
        let args = if args.is_object() { args } else { json!({}) };
        let res = self.request("tools/call", json!({"name": name, "arguments": args}), self.opts.call_timeout_ms).await?;
        let parts = res.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
        let text = parts.iter().map(flatten_part).collect::<Vec<_>>().join("\n");
        let text = text.trim();
        Ok(McpCallResult {
            text: if text.is_empty() { "(no output)".into() } else { text.to_string() },
            is_error: truthy(res.get("isError")),
        })
    }

    /// Terminate the server process (tree) and reject any in-flight calls.
    /// Idempotent; also runs on drop.
    pub fn dispose(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        self.shared.fail_all("client disposed");
        self.shared.kill.cancel();
    }

    /* ── JSON-RPC plumbing ── */

    async fn request(&self, method: &str, params: Value, timeout_ms: u64) -> anyhow::Result<Value> {
        let shared = &self.shared;
        if shared.closed.load(Ordering::SeqCst) {
            anyhow::bail!("client closed");
        }
        if let Some(why) = shared.dead_reason() {
            anyhow::bail!(why);
        }
        let id = shared.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string() + "\n";
        let (tx, rx) = oneshot::channel::<Reply>();
        if let Ok(mut p) = shared.pending.lock() {
            p.insert(id, tx);
        }
        let mut guard = PendingGuard { shared: shared.clone(), id, done: false };

        // The write is inside the deadline too: a server that stopped reading
        // its stdin must not wedge us on a full pipe.
        let exchange = async {
            if let Err(e) = shared.write_line(&payload).await {
                return Err(format!("write failed: {e}"));
            }
            match rx.await {
                Ok(r) => r,
                Err(_) => Err("client disposed".to_string()),
            }
        };
        let outcome = tokio::time::timeout(Duration::from_millis(timeout_ms), exchange).await;
        match outcome {
            Err(_) => anyhow::bail!("\"{method}\" timed out after {timeout_ms}ms"), // guard sends notifications/cancelled
            Ok(r) => {
                guard.done = true;
                if let Ok(mut p) = shared.pending.lock() {
                    p.remove(&id);
                }
                r.map_err(anyhow::Error::msg)
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) {
        if self.shared.closed.load(Ordering::SeqCst) {
            return;
        }
        let payload =
            json!({"jsonrpc": "2.0", "method": method, "params": params.unwrap_or_else(|| json!({}))}).to_string() + "\n";
        let _ = self.shared.write_line(&payload).await;
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.dispose();
    }
}

/// Flatten one `content` part to text.
fn flatten_part(p: &Value) -> String {
    let ty = p.get("type").and_then(Value::as_str);
    let non_null = |v: Option<&Value>| v.filter(|x| !x.is_null()).map(value_to_string);
    match ty {
        Some("text") => non_null(p.get("text")).unwrap_or_default(),
        Some("resource") if truthy(p.get("resource")) => {
            let r = &p["resource"];
            non_null(r.get("text")).or_else(|| non_null(r.get("uri"))).unwrap_or_default()
        }
        Some("image") => "[image content omitted]".into(),
        _ => match p {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
    }
}

/// Newline-delimited JSON: route every response line to its pending request.
async fn read_stdout(shared: Arc<Shared>, stdout: tokio::process::ChildStdout) {
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        // Ignore non-JSON noise (some servers log to stdout).
        let Ok(msg) = serde_json::from_str::<Value>(text) else { continue };
        // A notification/log from the server (no id) — ignore.
        let Some(id) = msg.get("id").and_then(Value::as_u64) else { continue };
        let tx = shared.pending.lock().ok().and_then(|mut p| p.remove(&id));
        let Some(tx) = tx else { continue };
        let reply = match msg.get("error").filter(|e| truthy(Some(e))) {
            Some(err) => Err(err
                .get("message")
                .filter(|m| truthy(Some(m)))
                .map(value_to_string)
                .unwrap_or_else(|| format!("JSON-RPC error {}", err.get("code").map(value_to_string).unwrap_or_default()))),
            None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
        };
        let _ = tx.send(reply);
    }
    shared.stdout_eof.store(true, Ordering::SeqCst);
    shared.stdout_done.notify_waiters();
}

async fn read_stderr(shared: Arc<Shared>, mut stderr: tokio::process::ChildStderr) {
    let mut buf = [0u8; 4096];
    loop {
        match stderr.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(mut s) = shared.stderr.lock() {
                    s.push_str(&String::from_utf8_lossy(&buf[..n]));
                    let len = s.chars().count();
                    if len > STDERR_KEEP {
                        let cut = s.char_indices().nth(len - STDERR_KEEP).map_or(0, |(i, _)| i);
                        s.drain(..cut);
                    }
                }
            }
        }
    }
}

/// Owns the child: reports its exit to every pending request, or kills it
/// (with its whole process tree) when the client is disposed.
async fn wait_child(shared: Arc<Shared>, mut child: Child) {
    tokio::select! {
        status = child.wait() => {
            shared.running.store(false, Ordering::SeqCst);
            // Let the reader drain whatever the server wrote before exiting, so a
            // response sent just before a clean exit still reaches its caller.
            let drained = shared.stdout_done.notified();
            tokio::pin!(drained);
            drained.as_mut().enable(); // registered before the flag check: no lost wakeup
            if !shared.stdout_eof.load(Ordering::SeqCst) {
                let _ = tokio::time::timeout(Duration::from_millis(1000), drained).await;
            }
            if !shared.closed.load(Ordering::SeqCst) {
                let code = status.ok().and_then(|s| s.code()).map_or_else(|| "null".to_string(), |c| c.to_string());
                let tail = shared.tail_stderr();
                let msg = if tail.is_empty() {
                    format!("server exited (code {code})")
                } else {
                    format!("server exited (code {code}): {tail}")
                };
                shared.set_dead(&msg);
                shared.fail_all(&msg);
            }
            *shared.stdin.lock().await = None;
        }
        _ = shared.kill.cancelled() => {
            *shared.stdin.lock().await = None;
            kill_tree(&mut child).await;
            shared.running.store(false, Ordering::SeqCst);
        }
    }
}

/// Kill the server and everything it spawned. On Windows a server launched
/// through `npx.cmd`/`uvx.cmd` runs under `cmd.exe`, and killing only that
/// wrapper would orphan the real server — `taskkill /T` takes the tree.
async fn kill_tree(child: &mut Child) {
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        let tk = tokio::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status();
        let _ = tokio::time::timeout(Duration::from_secs(5), tk).await;
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

/// Resolve the executable the way a shell would. On Windows `npx`/`uvx` are
/// `.cmd` shims that `CreateProcess` can't find by bare name, so look the
/// command up through PATH × PATHEXT (honouring a PATH override in the
/// server's env); Rust's std then runs a `.cmd`/`.bat` through `cmd.exe` with
/// proper argument escaping. Elsewhere the OS does the PATH search itself.
fn resolve_program(command: &str, env: &BTreeMap<String, String>, cwd: &Path) -> PathBuf {
    if cfg!(windows) {
        let path_var = env
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
            .map(|(_, v)| std::ffi::OsString::from(v))
            .or_else(|| std::env::var_os("PATH"));
        if let Ok(found) = which::which_in(command, path_var, cwd) {
            return found;
        }
    }
    PathBuf::from(command)
}

/* ── harness bridge ── */

/// Everything [`create_mcp_tools`] produced.
pub struct McpSetup {
    /// Handlers to register on the chat `ToolRegistry`.
    pub tools: Vec<SharedTool>,
    /// Schemas to advertise to the model.
    pub specs: Vec<ToolSpec>,
    /// Live clients — dispose these when the chat exits (dropping them works too).
    pub clients: Vec<Arc<McpClient>>,
    /// One human-readable status line per server (connected or failed).
    pub infos: Vec<String>,
}

impl McpSetup {
    /// Names of the servers that connected and contributed tools, in config order.
    pub fn servers(&self) -> Vec<String> {
        self.clients.iter().map(|c| c.name().to_string()).collect()
    }

    /// Stop every server process.
    pub fn dispose_all(&self) {
        for c in &self.clients {
            c.dispose();
        }
    }
}

/// MCP tool names must be function-call-safe; sanitize and namespace them.
pub fn mcp_tool_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect::<String>();
    format!("mcp__{}__{}", clean(server), clean(tool))
}

/// Connect to every server in `specs`, in parallel, and build the tool set. A
/// server that fails to start or list tools is skipped with a status line — it
/// never blocks the others or crashes chat startup. Results are assembled in
/// config order (deterministic, unlike the TS completion order).
pub async fn create_mcp_tools(specs: Vec<McpServerSpec>, opts: McpOptions) -> McpSetup {
    let outcomes = futures::future::join_all(specs.into_iter().map(|spec| async move {
        let client = McpClient::new(spec.clone(), opts);
        let defs = match client.start().await {
            Ok(()) => client.list_tools().await,
            Err(e) => Err(e),
        };
        (spec, client, defs)
    }))
    .await;

    let mut setup = McpSetup { tools: Vec::new(), specs: Vec::new(), clients: Vec::new(), infos: Vec::new() };
    for (spec, client, defs) in outcomes {
        let defs = match defs {
            Ok(d) => d,
            Err(e) => {
                setup.infos.push(format!("MCP {}: failed — {e:#}", spec.name));
                client.dispose();
                continue;
            }
        };
        if defs.is_empty() {
            setup.infos.push(format!("MCP {}: connected, but exposes no tools", spec.name));
            client.dispose();
            continue;
        }
        let client = Arc::new(client);
        for def in &defs {
            let fq_name = mcp_tool_name(&spec.name, &def.name);
            let label = if def.description.is_empty() { def.name.as_str() } else { def.description.as_str() };
            let description = take_chars(&format!("[MCP:{}] {label}", spec.name), 1024).to_string();
            setup.specs.push(ToolSpec { name: fq_name.clone(), description: description.clone(), parameters: def.input_schema.clone() });
            setup.tools.push(Arc::new(McpTool {
                fq_name,
                description,
                server: spec.name.clone(),
                tool: def.name.clone(),
                client: client.clone(),
            }));
        }
        let n = defs.len();
        setup.infos.push(format!("MCP {}: {n} tool{}", spec.name, if n == 1 { "" } else { "s" }));
        setup.clients.push(client);
    }
    setup
}

struct McpTool {
    fq_name: String,
    description: String,
    server: String,
    tool: String,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.fq_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    /// External, possibly side-effectful — run serially, never auto-batched.
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Esc drops the request: its pending slot is released and the server
        // gets notifications/cancelled; the server process keeps running.
        match cancellable(ctx, self.client.call_tool(&self.tool, args)).await {
            None => ToolResult::err(INTERRUPTED),
            Some(Ok(r)) => {
                ctx.log(json!({"tool": self.fq_name, "isError": r.is_error}));
                ToolResult { ok: !r.is_error, output: r.text, data: None }
            }
            Some(Err(e)) => ToolResult::err(format!("MCP {}/{} failed: {e:#}", self.server, self.tool)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_namespaced_and_sanitized() {
        assert_eq!(mcp_tool_name("filesystem", "read_file"), "mcp__filesystem__read_file");
        assert_eq!(mcp_tool_name("my server", "do-it!"), "mcp__my_server__do_it_");
    }

    #[test]
    fn content_parts_flatten_like_the_ts() {
        assert_eq!(flatten_part(&json!({"type": "text", "text": "hi"})), "hi");
        assert_eq!(flatten_part(&json!({"type": "text"})), "");
        assert_eq!(flatten_part(&json!({"type": "resource", "resource": {"uri": "file:///a"}})), "file:///a");
        assert_eq!(flatten_part(&json!({"type": "resource", "resource": {"text": "", "uri": "u"}})), "");
        assert_eq!(flatten_part(&json!({"type": "image", "data": "..."})), "[image content omitted]");
        assert_eq!(flatten_part(&json!("plain")), "plain");
        assert_eq!(flatten_part(&json!({"type": "audio"})), r#"{"type":"audio"}"#);
    }

    #[tokio::test]
    async fn spawn_failure_is_reported_and_requests_fail_fast() {
        let c = McpClient::new(
            McpServerSpec { name: "broken".into(), command: "qo_no_such_binary_xyz".into(), ..Default::default() },
            McpOptions { init_timeout_ms: 2000, call_timeout_ms: 2000 },
        );
        let e = c.start().await.unwrap_err().to_string();
        assert!(e.starts_with("spawn failed"), "{e}");
        let e = c.list_tools().await.unwrap_err().to_string();
        assert!(e.starts_with("spawn failed"), "{e}");
        assert!(!c.running());
    }
}
