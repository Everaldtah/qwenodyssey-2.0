//! Execution backends for swarm agents. The coordinated swarm routes a run's
//! shell work to one of two places, chosen by the PLANNER's complexity judgment:
//!
//!  - "bare"    — the user's own machine (host CLI). For LESS complex work.
//!                Every command goes through the same safety gate as run_shell
//!                (`classify_command`: hard-blocks, destructive detection, the
//!                config allow/deny lists) — and DESTRUCTIVE commands are flatly
//!                refused (a swarm agent has no interactive confirm).
//!  - "daytona" — an isolated Daytona sandbox (<https://daytona.io>). For MORE
//!                complex work (dependency installs, multi-file builds, servers).
//!                One sandbox per swarm run, shared by all agents so their files
//!                compose; deleted on dispose. Plain REST integration:
//!                `POST /sandbox`, `POST /toolbox/{id}/toolbox/process/execute`,
//!                `DELETE /sandbox/{id}`; bearer auth via `DAYTONA_API_KEY`.
//!
//! Mode "auto" = bare for simple plans, daytona for complex plans when a key is
//! configured (falls back to bare with a note when not).
//!
//! Every command is cancellable: the bare executor kills the child's whole
//! process tree, the Daytona executor aborts the in-flight request. Children
//! never inherit the terminal's stdin.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use qo_core::CancellationToken;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;

use crate::compat::{self, CommandClass};
use crate::env::{DaytonaSettings, SwarmEnv};
use crate::swarm::SwarmArtifacts;

/// Which backend actually runs commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecKind {
    Bare,
    Daytona,
}

impl ExecKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecKind::Bare => "bare",
            ExecKind::Daytona => "daytona",
        }
    }
}

/// `[swarm].exec` / `--exec`: where swarm agents may run commands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecMode {
    /// Text-only agents (default — safe for cloners).
    #[default]
    Off,
    /// Bare metal for simple plans, a Daytona sandbox for complex ones.
    Auto,
    /// Forced bare metal.
    Bare,
    /// Forced Daytona (falls back to bare metal with a note when no key).
    Daytona,
}

impl ExecMode {
    /// Parse "off" | "auto" | "bare" | "daytona" (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(ExecMode::Off),
            "auto" => Some(ExecMode::Auto),
            "bare" => Some(ExecMode::Bare),
            "daytona" => Some(ExecMode::Daytona),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ExecMode::Off => "off",
            ExecMode::Auto => "auto",
            ExecMode::Bare => "bare",
            ExecMode::Daytona => "daytona",
        }
    }
}

impl fmt::Display for ExecMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The planner's judgment: "complex" = real project execution (installs,
/// multi-file builds, services); "simple" = analysis, writing, single-file code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanComplexity {
    #[default]
    Simple,
    Complex,
}

impl PlanComplexity {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanComplexity::Simple => "simple",
            PlanComplexity::Complex => "complex",
        }
    }
}

/// Outcome of one command. Never an `Err`: failures (refusals, HTTP errors,
/// timeouts, interrupts) come back as `ok: false` with the reason in `output`,
/// so the agent sees them as a tool result.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ExecResult {
    pub ok: bool,
    pub exit_code: Option<i32>,
    pub output: String,
}

impl ExecResult {
    fn fail(output: impl Into<String>) -> Self {
        Self { ok: false, exit_code: None, output: output.into() }
    }
}

/// Per-command knobs.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Working directory (relative paths resolve against the executor's root).
    pub cwd: Option<String>,
    pub timeout_ms: Option<u64>,
    /// Stop button: kills the command (bare) / aborts the request (daytona).
    pub cancel: Option<CancellationToken>,
}

/// A place where swarm agents run shell commands.
#[async_trait]
pub trait AgentExecutor: Send + Sync {
    fn kind(&self) -> ExecKind;
    /// Short label for UI ("bare metal" / "daytona sandbox").
    fn label(&self) -> &str;
    async fn run(&self, command: &str, opts: RunOptions) -> ExecResult;
    /// Files created/modified during this run, for the results pane. Call before `dispose`.
    async fn artifacts(&self) -> SwarmArtifacts;
    /// Release resources (delete the sandbox). Safe to call more than once.
    async fn dispose(&self);
}

pub type SharedExecutor = Arc<dyn AgentExecutor>;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
/// Keep at most this much combined output per command (the agent only sees the tail).
const MAX_OUTPUT_BYTES: usize = 1 << 20;

/* ─────────────────────────── bare metal (host) ─────────────────────────── */

/// Host-side settings for the bare executor.
#[derive(Clone, Debug, Default)]
pub struct BareExecutorOptions {
    /// Root working directory (default: the process cwd).
    pub cwd: Option<PathBuf>,
    /// Config tools.allow_commands — auto-approved by the safety gate.
    pub allow_commands: Vec<String>,
    /// Config tools.deny_commands — hard-blocked.
    pub deny_commands: Vec<String>,
    pub default_timeout_ms: Option<u64>,
}

/// Runs commands on the user's machine: PowerShell on Windows, `sh -c` elsewhere.
pub struct BareExecutor {
    opts: BareExecutorOptions,
    started_at: SystemTime,
}

impl BareExecutor {
    pub fn new(opts: BareExecutorOptions) -> Self {
        Self { opts, started_at: SystemTime::now() }
    }

    fn root(&self) -> PathBuf {
        match &self.opts.cwd {
            Some(c) => c.clone(),
            None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    fn command_for(&self, command: &str) -> tokio::process::Command {
        #[cfg(windows)]
        {
            let mut c = tokio::process::Command::new("powershell.exe");
            c.args(["-NoProfile", "-NonInteractive", "-InputFormat", "None", "-ExecutionPolicy", "Bypass", "-Command"]);
            c.arg(format!("{}{}", compat::ps_preamble(), command));
            // No console of its own: nothing can pop up on the user's desktop or
            // read the terminal, and a console Ctrl-C does not reach it (we kill it).
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            c.creation_flags(CREATE_NO_WINDOW);
            c
        }
        #[cfg(not(windows))]
        {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        }
    }
}

impl Default for BareExecutor {
    fn default() -> Self {
        Self::new(BareExecutorOptions::default())
    }
}

#[async_trait]
impl AgentExecutor for BareExecutor {
    fn kind(&self) -> ExecKind {
        ExecKind::Bare
    }

    fn label(&self) -> &str {
        "bare metal"
    }

    async fn run(&self, command: &str, o: RunOptions) -> ExecResult {
        match compat::classify_command(command, &self.opts.allow_commands, &self.opts.deny_commands) {
            CommandClass::Blocked => {
                return ExecResult::fail("Refused: command is hard-blocked on this machine.");
            }
            CommandClass::Destructive => {
                return ExecResult::fail(
                    "Refused: command looks destructive and swarm agents cannot confirm interactively. \
                     Rephrase non-destructively, or the user can add it to tools.allow_commands.",
                );
            }
            CommandClass::Safe => {}
        }
        let timeout = o.timeout_ms.or(self.opts.default_timeout_ms).unwrap_or(DEFAULT_TIMEOUT_MS);
        let root = self.root();
        let cwd = match o.cwd.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
            Some(c) => root.join(c),
            None => root,
        };
        let mut cmd = self.command_for(command);
        cmd.current_dir(&cwd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        run_child(cmd, Duration::from_millis(timeout.max(1)), o.cancel).await
    }

    /// Files under cwd modified since this executor was created (depth-limited).
    async fn artifacts(&self) -> SwarmArtifacts {
        let root = self.root();
        let root = std::fs::canonicalize(&root).map(strip_verbatim).unwrap_or(root);
        let since = self.started_at.checked_sub(Duration::from_secs(1)).unwrap_or(self.started_at);
        let walk_root = root.clone();
        let files = tokio::task::spawn_blocking(move || {
            let mut files = Vec::new();
            walk_recent(&walk_root, 0, since, &mut files);
            files.sort();
            files
        })
        .await
        .unwrap_or_default();
        SwarmArtifacts { location: format!("bare metal: {}", root.display()), files }
    }

    async fn dispose(&self) {
        // Nothing to release on the host.
    }
}

/// `\\?\C:\x` → `C:\x` (canonicalize on Windows returns verbatim paths).
fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => p,
    }
}

const ARTIFACT_SKIP: [&str; 6] = ["node_modules", ".git", "dist", ".qwenodyssey", "__pycache__", ".venv"];
const ARTIFACT_MAX_FILES: usize = 60;
const ARTIFACT_MAX_DEPTH: usize = 3;

fn walk_recent(dir: &Path, depth: usize, since: SystemTime, files: &mut Vec<String>) {
    if depth > ARTIFACT_MAX_DEPTH || files.len() >= ARTIFACT_MAX_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        if files.len() >= ARTIFACT_MAX_FILES {
            break;
        }
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || ARTIFACT_SKIP.contains(&name.as_ref()) {
            continue;
        }
        let full = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            walk_recent(&full, depth + 1, since, files);
        } else if let Ok(meta) = std::fs::metadata(&full) {
            if meta.modified().is_ok_and(|m| m >= since) {
                files.push(full.to_string_lossy().into_owned());
            }
        }
    }
}

/// Spawn `cmd`, collect stdout+stderr interleaved (like execa's `all`), and
/// enforce the timeout / cancellation by killing the whole process tree.
async fn run_child(mut cmd: tokio::process::Command, timeout: Duration, cancel: Option<CancellationToken>) -> ExecResult {
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ExecResult::fail(format!("failed to start shell: {e}")),
    };
    let buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        readers.push(tokio::spawn(pump(out, Arc::clone(&buf))));
    }
    if let Some(err) = child.stderr.take() {
        readers.push(tokio::spawn(pump(err, Arc::clone(&buf))));
    }

    let cancel = cancel.unwrap_or_default();
    enum End {
        Exited(std::io::Result<std::process::ExitStatus>),
        TimedOut,
        Interrupted,
    }
    let end = tokio::select! {
        s = child.wait() => End::Exited(s),
        _ = tokio::time::sleep(timeout) => End::TimedOut,
        _ = cancel.cancelled() => End::Interrupted,
    };
    if !matches!(end, End::Exited(_)) {
        kill_tree(&mut child).await;
    }
    // Pipes close when the tree is gone; a detached grandchild (a server started
    // in the background) may keep them open, so don't wait on it forever.
    for r in readers {
        let abort = r.abort_handle();
        if tokio::time::timeout(Duration::from_secs(2), r).await.is_err() {
            abort.abort();
        }
    }
    let bytes = std::mem::take(&mut *buf.lock().unwrap_or_else(|e| e.into_inner()));
    let mut output = String::from_utf8_lossy(&bytes).into_owned();
    match end {
        End::Exited(Ok(status)) => {
            let code = status.code();
            ExecResult { ok: code == Some(0), exit_code: code, output }
        }
        End::Exited(Err(e)) => ExecResult::fail(format!("{output}{}failed waiting for shell: {e}", sep(&output))),
        End::TimedOut => {
            output.push_str(&format!("{}[timed out after {}s — process killed]", sep(&output), timeout.as_secs().max(1)));
            ExecResult::fail(output)
        }
        End::Interrupted => {
            output.push_str(&format!("{}[interrupted — process killed]", sep(&output)));
            ExecResult::fail(output)
        }
    }
}

fn sep(s: &str) -> &'static str {
    if s.is_empty() || s.ends_with('\n') {
        ""
    } else {
        "\n"
    }
}

async fn pump<R: tokio::io::AsyncRead + Unpin>(mut r: R, buf: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 8192];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut b = buf.lock().unwrap_or_else(|e| e.into_inner());
                b.extend_from_slice(&chunk[..n]);
                if b.len() > MAX_OUTPUT_BYTES {
                    let cut = b.len() - MAX_OUTPUT_BYTES;
                    b.drain(..cut);
                }
            }
        }
    }
}

/// Kill the child and everything it spawned. On Windows `taskkill /T` takes
/// the tree down (killing powershell.exe alone would orphan `node`, `npm`, …).
async fn kill_tree(child: &mut tokio::process::Child) {
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        let mut tk = tokio::process::Command::new("taskkill");
        tk.args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        tk.creation_flags(CREATE_NO_WINDOW);
        if let Ok(mut p) = tk.spawn() {
            let _ = tokio::time::timeout(Duration::from_secs(5), p.wait()).await;
        }
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

/* ───────────────────────── daytona sandbox (cloud) ─────────────────────── */

/// The Daytona key: `[daytona].api_key`, else the env var it names (default `DAYTONA_API_KEY`).
pub fn resolve_daytona_key(d: &DaytonaSettings) -> String {
    if !d.api_key.is_empty() {
        return d.api_key.clone();
    }
    let var = if d.api_key_env.is_empty() { "DAYTONA_API_KEY" } else { d.api_key_env.as_str() };
    std::env::var(var).unwrap_or_default()
}

/// Daytona is enabled and a key resolves.
pub fn daytona_available(d: &DaytonaSettings) -> bool {
    d.enabled && !resolve_daytona_key(d).is_empty()
}

/// One sandbox per swarm run, shared by every agent, deleted on dispose.
pub struct DaytonaExecutor {
    settings: DaytonaSettings,
    http: reqwest::Client,
    /// The run's sandbox id once created. Holding the async lock while creating
    /// makes concurrent callers share one creation; a failed creation leaves it
    /// `None`, so the next `run` retries.
    sandbox: tokio::sync::Mutex<Option<String>>,
    disposed: AtomicBool,
    poll_interval: Duration,
    start_timeout: Duration,
}

impl DaytonaExecutor {
    pub fn new(settings: DaytonaSettings) -> Self {
        Self::with_client(settings, reqwest::Client::new())
    }

    /// Use a specific HTTP client (tests point it at a local mock server).
    pub fn with_client(settings: DaytonaSettings, http: reqwest::Client) -> Self {
        Self {
            settings,
            http,
            sandbox: tokio::sync::Mutex::new(None),
            disposed: AtomicBool::new(false),
            poll_interval: Duration::from_millis(1500),
            start_timeout: Duration::from_secs(120),
        }
    }

    /// How often to poll a starting sandbox (default 1.5 s).
    pub fn with_poll_interval(mut self, every: Duration) -> Self {
        self.poll_interval = every;
        self
    }

    /// The sandbox id, once created.
    pub async fn sandbox_id(&self) -> Option<String> {
        self.sandbox.lock().await.clone()
    }

    fn base(&self) -> String {
        let b = if self.settings.base_url.is_empty() { "https://app.daytona.io/api" } else { &self.settings.base_url };
        b.trim_end_matches('/').to_string()
    }

    async fn api(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Value> {
        let mut req = self
            .http
            .request(method.clone(), format!("{}{}", self.base(), path))
            .bearer_auth(resolve_daytona_key(&self.settings))
            .header("Content-Type", "application/json")
            .timeout(timeout);
        if let Some(b) = body {
            req = req.body(b.to_string());
        }
        let call = async {
            let res = req.send().await.with_context(|| format!("daytona request failed on {method} {path}"))?;
            let status = res.status();
            let text = res.text().await.unwrap_or_else(|_| "<no body>".to_string());
            if !status.is_success() {
                return Err(anyhow!(
                    "daytona HTTP {} on {method} {path}: {}",
                    status.as_u16(),
                    truncate_chars(&text, 200)
                ));
            }
            if text.is_empty() {
                return Ok(json!({}));
            }
            Ok(serde_json::from_str(&text).unwrap_or_else(|_| json!({ "raw": text })))
        };
        tokio::select! {
            r = call => r,
            _ = cancel.cancelled() => Err(qo_core::ProviderError::Interrupted.into()),
        }
    }

    /// Create the run's sandbox once; concurrent callers share the same creation.
    async fn ensure_sandbox(&self, cancel: &CancellationToken) -> anyhow::Result<String> {
        let mut slot = self.sandbox.lock().await;
        if let Some(id) = slot.as_ref() {
            return Ok(id.clone());
        }
        let d = &self.settings;
        let mut body = json!({
            "name": format!("qod-swarm-{}", base36(chrono::Utc::now().timestamp_millis().max(0) as u64)),
            "labels": { "app": "qwenodyssey-swarm" },
            "autoStopInterval": d.auto_stop_minutes,
            "autoDeleteInterval": d.auto_delete_minutes,
        });
        if !d.snapshot.is_empty() {
            body["snapshot"] = json!(d.snapshot);
        }
        if !d.target.is_empty() {
            body["target"] = json!(d.target);
        }
        let short = Duration::from_secs(60);
        let sb = self.api(reqwest::Method::POST, "/sandbox", Some(body), short, cancel).await?;
        let id = sb
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| anyhow!("daytona sandbox create returned no id"))?;
        // Wait until the sandbox is running (state "started").
        let deadline = Instant::now() + self.start_timeout;
        let mut state = sb.get("state").and_then(|v| v.as_str()).unwrap_or("creating").to_string();
        while state != "started" && Instant::now() < deadline {
            tokio::select! {
                _ = tokio::time::sleep(self.poll_interval) => {}
                _ = cancel.cancelled() => return Err(qo_core::ProviderError::Interrupted.into()),
            }
            let cur = self.api(reqwest::Method::GET, &format!("/sandbox/{id}"), None, short, cancel).await?;
            if let Some(s) = cur.get("state").and_then(|v| v.as_str()) {
                state = s.to_string();
            }
            let reason = cur.get("errorReason").filter(|v| !v.is_null() && v.as_str() != Some(""));
            if state == "error" || reason.is_some() {
                let why = reason.and_then(|v| v.as_str()).map(str::to_string).unwrap_or_else(|| state.clone());
                return Err(anyhow!("daytona sandbox failed to start: {why}"));
            }
        }
        if state != "started" {
            return Err(anyhow!("daytona sandbox start timed out ({}s)", self.start_timeout.as_secs()));
        }
        // Drop a sentinel so artifacts() can list files created after this point.
        let _ = self
            .api(
                reqwest::Method::POST,
                &format!("/toolbox/{id}/toolbox/process/execute"),
                Some(json!({ "command": "touch /tmp/.qod_swarm_start", "timeout": 10 })),
                short,
                cancel,
            )
            .await;
        *slot = Some(id.clone());
        Ok(id)
    }

    async fn execute(&self, command: &str, o: &RunOptions, cancel: &CancellationToken) -> anyhow::Result<ExecResult> {
        let id = self.ensure_sandbox(cancel).await?;
        let timeout_ms = o.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
        let mut body = json!({ "command": command, "timeout": timeout_ms.div_ceil(1000).max(1) });
        if let Some(cwd) = o.cwd.as_deref().filter(|c| !c.is_empty()) {
            body["cwd"] = json!(cwd);
        }
        let http_timeout = Duration::from_millis(timeout_ms) + Duration::from_secs(30);
        let res = self
            .api(reqwest::Method::POST, &format!("/toolbox/{id}/toolbox/process/execute"), Some(body), http_timeout, cancel)
            .await?;
        let exit_code = res.get("exitCode").and_then(|v| v.as_f64()).map(|n| n as i32);
        let output = match res.get("result") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        Ok(ExecResult { ok: exit_code == Some(0), exit_code, output })
    }
}

#[async_trait]
impl AgentExecutor for DaytonaExecutor {
    fn kind(&self) -> ExecKind {
        ExecKind::Daytona
    }

    fn label(&self) -> &str {
        "daytona sandbox"
    }

    async fn run(&self, command: &str, o: RunOptions) -> ExecResult {
        if self.disposed.load(Ordering::SeqCst) {
            return ExecResult::fail("daytona executor already disposed");
        }
        let cancel = o.cancel.clone().unwrap_or_default();
        match self.execute(command, &o, &cancel).await {
            Ok(r) => r,
            Err(e) if qo_core::error::is_interrupt(&e) => ExecResult::fail("[interrupted — request aborted]"),
            Err(e) => ExecResult::fail(format!("{e:#}")),
        }
    }

    /// List files created in the sandbox since the sentinel (before it's deleted).
    async fn artifacts(&self) -> SwarmArtifacts {
        let Some(id) = self.sandbox_id().await else {
            return SwarmArtifacts { location: "daytona sandbox".to_string(), files: Vec::new() };
        };
        let r = self
            .run(
                "find . ~ -maxdepth 3 -type f -newer /tmp/.qod_swarm_start -not -path '*/.*' 2>/dev/null | sort -u | head -60",
                RunOptions { timeout_ms: Some(30_000), ..Default::default() },
            )
            .await;
        let files = if r.ok || r.exit_code.is_some() {
            r.output.lines().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
        } else {
            Vec::new()
        };
        SwarmArtifacts { location: format!("daytona sandbox {id}"), files }
    }

    async fn dispose(&self) {
        if self.disposed.swap(true, Ordering::SeqCst) {
            return;
        }
        let id = self.sandbox.lock().await.take();
        if let Some(id) = id {
            // Best-effort cleanup; autoDeleteInterval is the backstop.
            let never = CancellationToken::new();
            let _ = self
                .api(reqwest::Method::DELETE, &format!("/sandbox/{id}"), None, Duration::from_secs(30), &never)
                .await;
        }
    }
}

fn base36(mut n: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

pub(crate) fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/* ───────────────────────────── mode routing ────────────────────────────── */

/// The backend picked for a run.
pub struct ExecutorChoice {
    pub executor: Option<SharedExecutor>,
    /// Diagnostic for the UI (e.g. why daytona was requested but bare was used).
    pub note: Option<String>,
}

impl fmt::Debug for ExecutorChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutorChoice")
            .field("executor", &self.executor.as_ref().map(|e| e.kind()))
            .field("note", &self.note)
            .finish()
    }
}

/// Pick the execution backend for a swarm run.
///   off            → none (agents are text-only, the previous behavior)
///   bare / daytona → forced (daytona without a key falls back to bare + note)
///   auto           → complexity "complex" + Daytona configured → daytona,
///                    otherwise bare
pub fn choose_executor(env: &SwarmEnv, mode: ExecMode, complexity: PlanComplexity, bare: BareExecutorOptions) -> ExecutorChoice {
    let bare_exec = |o: BareExecutorOptions| -> Option<SharedExecutor> { Some(Arc::new(BareExecutor::new(o))) };
    let daytona = || -> Option<SharedExecutor> { Some(Arc::new(DaytonaExecutor::new(env.daytona.clone()))) };
    match mode {
        ExecMode::Off => ExecutorChoice { executor: None, note: None },
        ExecMode::Bare => ExecutorChoice { executor: bare_exec(bare), note: None },
        ExecMode::Daytona => {
            if daytona_available(&env.daytona) {
                ExecutorChoice { executor: daytona(), note: None }
            } else {
                ExecutorChoice {
                    executor: bare_exec(bare),
                    note: Some("daytona requested but no DAYTONA_API_KEY — using bare metal".to_string()),
                }
            }
        }
        ExecMode::Auto => {
            if complexity == PlanComplexity::Complex {
                if daytona_available(&env.daytona) {
                    return ExecutorChoice { executor: daytona(), note: None };
                }
                return ExecutorChoice {
                    executor: bare_exec(bare),
                    note: Some("complex plan — daytona not configured, using bare metal".to_string()),
                };
            }
            ExecutorChoice { executor: bare_exec(bare), note: None }
        }
    }
}
