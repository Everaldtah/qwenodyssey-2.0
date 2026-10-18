//! Persistent, stateful shell session backed by a real pseudo-terminal
//! (`portable-pty`: ConPTY on Windows, a Unix pty elsewhere).
//!
//! The default `run_shell` tool spawns a FRESH shell per command, so `cd`,
//! activated venvs, exported env vars, and long-running processes (servers,
//! watchers, REPLs) don't survive between calls. A ShellSession keeps ONE shell
//! process alive for the whole chat session: state persists, long-running
//! commands can be started and polled instead of blocking-or-dying, and
//! interactive prompts can be answered by sending more input.
//!
//! The TS used node-pty, an optional native module, and degraded gracefully
//! when it was missing. `portable-pty` is always compiled in, but spawning can
//! still fail (no ConPTY on an old Windows, no shell on PATH): `start()` then
//! reports the problem and the tools return a clean error, while the rest of
//! the harness keeps working with the one-shot run_shell tool.
//!
//! Command boundaries: a pty is just a byte stream with no notion of "command
//! finished", so after each command we write a unique sentinel that echoes the
//! exit code (`<TOKEN>:<code>`). Seeing that line on its own means the command
//! returned to the prompt; not seeing it within the timeout means it's still
//! running, and the caller can poll with read().
//!
//! Threads: one blocking reader thread drains the pty (ConPTY stalls the child
//! when nobody reads), one waits for the shell to exit; both tag their updates
//! with a generation number so a reset session ignores a dead shell's tail.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, MasterPty, PtySize};
use rand::Rng;
use regex::Regex;
use tokio::sync::Notify;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::ps_compat::adapt_chains_for_powershell;

/// Where a command (or the session) stands after a run/read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunStatus {
    /// The command finished (its sentinel was seen).
    Done,
    /// Still running after the wait; poll with read().
    Running,
    /// A previous command is still running; nothing was sent.
    Busy,
    /// The shell could not be started.
    Error,
    /// read() before any command was run.
    Idle,
    /// The wait was cancelled (Esc); Ctrl+C was sent to the shell.
    Interrupted,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Done => "done",
            RunStatus::Running => "running",
            RunStatus::Busy => "busy",
            RunStatus::Error => "error",
            RunStatus::Idle => "idle",
            RunStatus::Interrupted => "interrupted",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunResult {
    pub status: RunStatus,
    pub output: String,
    /// Exit code of a finished command (`None` while running, or when the
    /// shell reported none — e.g. pure PowerShell cmdlets leave $LASTEXITCODE empty).
    pub exit_code: Option<i32>,
}

impl RunResult {
    fn new(status: RunStatus, output: impl Into<String>) -> Self {
        Self { status, output: output.into(), exit_code: None }
    }
}

struct Job {
    marker: String,
    marker_re: Regex,
    /// The command the user sent, so we can drop its terminal echo.
    command: String,
    /// All pty output for this command, pre-clean.
    raw: String,
    done: bool,
    exit_code: Option<i32>,
    /// How much of the cleaned output the caller has already seen (bytes).
    clean_consumed: usize,
}

struct Pty {
    /// Never read, but owning it keeps the pseudo-terminal open: dropping the
    /// master closes the ConPTY / pty and ends the shell.
    #[allow(dead_code)]
    master: Box<dyn MasterPty + Send>,
    writer: Box<dyn Write + Send>,
    killer: Box<dyn ChildKiller + Send + Sync>,
}

#[derive(Default)]
struct State {
    pty: Option<Pty>,
    alive: bool,
    start_error: Option<String>,
    warm: bool,
    saw_data: bool,
    current: Option<Job>,
    generation: u64,
}

const IS_WIN: bool = cfg!(windows);

static OSC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)").expect("static regex"));
static CSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b[\[0-9;?]*[ -/]*[@-~]").expect("static regex"));
static KEYPAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b[=>]").expect("static regex"));
static PROMPT_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^PS[^>\n]*>\s?").expect("static regex"));
static INIT_ECHO: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"function prompt \{|^PS1=''").expect("static regex"));
static STALE_SENTINEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^QOD[0-9A-Z]{8}:-?\d*\s*$").expect("static regex"));
static BLANK_RUNS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").expect("static regex"));

/// Strip ANSI escape / control sequences a pty interleaves into the stream:
/// OSC sequences (window title), CSI sequences, keypad modes, and lone
/// carriage returns (a CR not followed by LF). Backspaces are applied (they
/// erase the previous character on the line, as a terminal would) and stray
/// BELs dropped: PSReadLine repaints its input line with them under load
/// ("e\x08echo …"), which used to leak into the cleaned output.
pub fn strip_ansi(s: &str) -> String {
    let s = OSC.replace_all(s, "");
    let s = CSI.replace_all(&s, "");
    let s = KEYPAD.replace_all(&s, "");
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' if chars.peek() != Some(&'\n') => {}
            '\x08' => {
                if out.chars().next_back().is_some_and(|last| last != '\n') {
                    out.pop();
                }
            }
            '\x07' => {}
            _ => out.push(c),
        }
    }
    out
}

/// Is `line` the terminal echo of `cmd`? Exact match, or a repaint artefact:
/// a partial prefix of the command glued in front of the full command
/// ("echo $enecho $env:X" — PSReadLine redrew the line after moving the cursor).
fn is_echo_of(line: &str, cmd: &str) -> bool {
    if cmd.is_empty() {
        return false;
    }
    if line == cmd {
        return true;
    }
    match line.strip_suffix(cmd) {
        Some(prefix) => !prefix.is_empty() && cmd.starts_with(prefix),
        None => false,
    }
}

fn marker_cmd(token: &str) -> String {
    // Keep this SHORT and parse-trivial — a long/paren-heavy marker is fragile to
    // conpty echo mangling. `$LASTEXITCODE` is the native-exe exit code (empty for
    // pure cmdlets, which we treat as success). On failure of a native command it
    // carries the real non-zero code.
    if IS_WIN {
        format!("Write-Output \"{token}:$LASTEXITCODE\"")
    } else {
        format!("echo \"{token}:$?\"")
    }
}

/// Locate the sentinel RESULT line (token at line start, then the exit code).
/// A (blanked) prompt may be glued in front of the result line: "PS> QOD…:0".
fn marker_regex(token: &str) -> Option<Regex> {
    Regex::new(&format!(r"(?:^|\n)(?:PS[^>\n]*>\s?)?{}:(-?\d*)\r?\n", regex::escape(token))).ok()
}

fn find_marker(re: &Regex, raw: &str) -> Option<(usize, Option<i32>)> {
    let caps = re.captures(raw)?;
    let index = caps.get(0)?.start();
    let code = caps.get(1).map(|m| m.as_str()).filter(|s| !s.is_empty()).and_then(|s| s.parse::<i32>().ok());
    Some((index, code))
}

/// Decode a byte stream as UTF-8 across chunk boundaries (a code point split
/// between two reads is carried over; genuinely invalid bytes become U+FFFD).
fn decode_utf8_stream(pending: &mut Vec<u8>) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return out;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                out.push_str(std::str::from_utf8(&pending[..valid]).unwrap_or_default());
                match e.error_len() {
                    None => {
                        // Incomplete sequence at the end: keep it for the next chunk.
                        pending.drain(..valid);
                        return out;
                    }
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        pending.drain(..valid + bad);
                    }
                }
            }
        }
    }
}

impl State {
    fn on_data(&mut self, d: &str) {
        self.saw_data = true;
        let Some(job) = self.current.as_mut() else { return };
        if job.done {
            return;
        }
        job.raw.push_str(d);
        // conpty interleaves ANSI (e.g. "\x1b[m") between the newline and the marker,
        // so match on the STRIPPED text — matching the raw bytes never saw the
        // sentinel, every command "ran" until its timeout, and the session then
        // reported "busy" for the rest of the chat.
        let plain = strip_ansi(&job.raw);
        if let Some((index, code)) = find_marker(&job.marker_re, &plain) {
            job.done = true;
            job.exit_code = code;
            job.raw = plain[..index].to_string();
        }
    }

    /// Cleaned output the caller hasn't seen yet, advancing the consumed pointer.
    fn delta_result(&mut self) -> RunResult {
        let Some(job) = self.current.as_mut() else {
            return RunResult::new(RunStatus::Idle, "No command has been run in this session yet.");
        };
        let cleaned = cleaned_output(job);
        let delta = cleaned.get(job.clean_consumed..).unwrap_or("").to_string();
        job.clean_consumed = cleaned.len();
        RunResult {
            status: if job.done { RunStatus::Done } else { RunStatus::Running },
            output: delta,
            exit_code: if job.done { job.exit_code } else { None },
        }
    }

    fn write(&mut self, data: &str) -> std::io::Result<()> {
        match self.pty.as_mut() {
            Some(p) => {
                p.writer.write_all(data.as_bytes())?;
                p.writer.flush()
            }
            None => Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "shell is not running")),
        }
    }
}

/// Clean a job's raw buffer: drop ANSI, the echoed command/marker lines, prompts.
fn cleaned_output(job: &Job) -> String {
    let text = strip_ansi(&job.raw);
    let cmd = job.command.trim();
    // The line actually sent to PowerShell may differ (bash-chain rewrite).
    let wire_cmd = if IS_WIN { adapt_chains_for_powershell(cmd).trim().to_string() } else { cmd.to_string() };
    let mut out: Vec<String> = Vec::new();
    for line in text.split('\n') {
        // Strip a leading PowerShell prompt ("PS C:\path> " or the blanked "PS> "),
        // which conpty often glues onto the echoed command line.
        let line = PROMPT_PREFIX.replace(line, "");
        let line = line.strip_suffix('\r').unwrap_or(&line);
        if line.contains(&job.marker) {
            continue; // echoed marker command / result line
        }
        let t = line.trim();
        if is_echo_of(t, cmd) || is_echo_of(t, &wire_cmd) {
            continue; // the terminal-echoed input command
        }
        // Startup init echo can bleed into the first command's buffer (it's written
        // before any job exists) — drop it. Likewise a stale sentinel result from
        // an interrupted earlier command.
        if INIT_ECHO.is_match(line) || STALE_SENTINEL.is_match(t) {
            continue;
        }
        out.push(line.to_string());
    }
    // Collapse leading/trailing blank lines and runs of blanks.
    BLANK_RUNS.replace_all(&out.join("\n"), "\n\n").trim().to_string()
}

fn random_token() -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    let tail: String = (0..8).map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char).collect();
    format!("QOD{tail}")
}

/// A persistent shell on a real pty. Share it as `Arc<ShellSession>`; the shell
/// is killed when the last handle drops (or on [`ShellSession::dispose`]).
pub struct ShellSession {
    cwd: PathBuf,
    state: Arc<Mutex<State>>,
    notify: Arc<Notify>,
    /// Serialises run/read so two callers never interleave on one job.
    op: tokio::sync::Mutex<()>,
}

impl ShellSession {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            state: Arc::new(Mutex::new(State::default())),
            notify: Arc::new(Notify::new()),
            op: tokio::sync::Mutex::new(()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock_state(&self.state)
    }

    pub fn is_alive(&self) -> bool {
        self.lock().alive
    }

    /// The program the session runs (PowerShell 5.1 on Windows, `$SHELL`/bash elsewhere).
    fn shell_program() -> (String, Vec<&'static str>) {
        if IS_WIN {
            ("powershell.exe".to_string(), vec!["-NoLogo", "-NoProfile"])
        } else {
            (std::env::var("SHELL").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "bash".to_string()), vec![])
        }
    }

    /// Spawn the persistent shell. Idempotent while alive.
    pub fn start(&self) -> Result<(), String> {
        let mut st = self.lock();
        if st.alive {
            return Ok(());
        }
        let fail = |st: &mut State, msg: String| {
            st.start_error = Some(msg.clone());
            Err(msg)
        };
        let pair = match native_pty_system().openpty(PtySize {
            // Very wide so long command/marker lines never wrap (wrapping under conpty
            // corrupts the echoed input and can swallow the completion sentinel).
            cols: 1000,
            rows: 30,
            pixel_width: 0,
            pixel_height: 0,
        }) {
            Ok(p) => p,
            Err(e) => return fail(&mut st, format!("failed to open a pseudo-terminal: {e}")),
        };
        let (program, args) = Self::shell_program();
        let mut cmd = CommandBuilder::new(&program);
        cmd.args(&args);
        cmd.cwd(&self.cwd);
        if !IS_WIN {
            cmd.env("TERM", "xterm-color");
        }
        let mut child = match pair.slave.spawn_command(cmd) {
            Ok(c) => c,
            Err(e) => return fail(&mut st, format!("failed to spawn shell: {e}")),
        };
        drop(pair.slave); // the child owns its end; keeping ours open would hide EOF
        let reader = match pair.master.try_clone_reader() {
            Ok(r) => r,
            Err(e) => {
                let _ = child.kill();
                return fail(&mut st, format!("failed to read from the shell: {e}"));
            }
        };
        let writer = match pair.master.take_writer() {
            Ok(w) => w,
            Err(e) => {
                let _ = child.kill();
                return fail(&mut st, format!("failed to write to the shell: {e}"));
            }
        };
        let killer = child.clone_killer();

        st.generation += 1;
        let generation = st.generation;
        st.pty = Some(Pty { master: pair.master, writer, killer });
        st.alive = true;
        st.warm = false;
        st.saw_data = false;
        st.current = None;
        st.start_error = None;
        drop(st);

        // Reader: drain the pty forever (until EOF/error or the session moves on).
        let state = Arc::clone(&self.state);
        let notify = Arc::clone(&self.notify);
        let spawned_reader = std::thread::Builder::new().name("qo-shell-session-reader".into()).spawn(move || {
            let mut reader = reader;
            let mut buf = [0u8; 8192];
            let mut pending: Vec<u8> = Vec::new();
            loop {
                let n = match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                pending.extend_from_slice(&buf[..n]);
                let text = decode_utf8_stream(&mut pending);
                {
                    let mut st = lock_state(&state);
                    if st.generation != generation {
                        break;
                    }
                    if !text.is_empty() {
                        st.on_data(&text);
                    }
                }
                notify.notify_waiters();
            }
        });

        // Exit watcher: unblock any waiter when the shell dies.
        let state = Arc::clone(&self.state);
        let notify = Arc::clone(&self.notify);
        let spawned_waiter = std::thread::Builder::new().name("qo-shell-session-wait".into()).spawn(move || {
            let _ = child.wait();
            let mut st = lock_state(&state);
            if st.generation == generation {
                st.alive = false;
                if let Some(job) = st.current.as_mut() {
                    if !job.done {
                        job.done = true;
                        job.exit_code = None;
                    }
                }
            }
            drop(st);
            notify.notify_waiters();
        });

        if spawned_reader.is_err() || spawned_waiter.is_err() {
            self.dispose();
            let msg = "failed to start shell-session threads".to_string();
            self.lock().start_error = Some(msg.clone());
            return Err(msg);
        }
        Ok(())
    }

    /// Wait for the shell to be ready for input, then quiet it (blank prompt, web
    /// cmdlet defaults). Writing the first command before conpty has attached the
    /// shell's input reader silently drops it, so the first run() awaits this.
    async fn warmup(&self) {
        if self.lock().warm {
            return;
        }
        // Proceed after the first output, or after 1.5 s even if it never comes.
        let deadline = Instant::now() + Duration::from_millis(1500);
        loop {
            let notified = self.notify.notified();
            if self.lock().saw_data {
                break;
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(deadline) => break,
            }
        }
        // Blank the prompt for clean output. Do NOT remove PSReadLine — doing so
        // mid-session wedges the input reader and the next command is lost; its
        // syntax-highlight ANSI is harmless (stripped on read). Also default
        // Invoke-WebRequest/RestMethod to -UseBasicParsing + silence progress so
        // web cmdlets don't stall on the legacy IE engine.
        let init = if IS_WIN {
            "function prompt { '' }; $ProgressPreference='SilentlyContinue'; \
             $PSDefaultParameterValues['Invoke-WebRequest:UseBasicParsing']=$true; \
             $PSDefaultParameterValues['Invoke-RestMethod:UseBasicParsing']=$true\r"
        } else {
            "PS1=''\r"
        };
        let _ = self.lock().write(init);
        // Let the init line run; its output is discarded (no active job).
        tokio::time::sleep(Duration::from_millis(400)).await;
        self.lock().warm = true;
    }

    /// Run a command in the persistent shell. Waits up to `timeout` for it to
    /// finish; if it's still running by then, returns status Running with the
    /// output so far — poll [`ShellSession::read`] for the rest. Cancelling
    /// `cancel` sends Ctrl+C to the shell and returns status Interrupted.
    pub async fn run(&self, command: &str, timeout: Duration, cancel: Option<&CancellationToken>) -> RunResult {
        let _op = self.op.lock().await;
        if !self.is_alive() {
            if let Err(e) = self.start() {
                return RunResult::new(RunStatus::Error, e);
            }
        }
        self.warmup().await; // ensure the shell is attached + quieted before the first command
        {
            let mut st = self.lock();
            if st.current.as_ref().is_some_and(|j| !j.done) {
                return RunResult::new(
                    RunStatus::Busy,
                    "A previous command is still running in this session. Use shell_session_read to read its output, \
                     or wait for it to finish before sending another command.",
                );
            }
            let token = random_token();
            let Some(marker_re) = marker_regex(&token) else {
                return RunResult::new(RunStatus::Error, "internal error: bad sentinel");
            };
            // Windows PowerShell 5.1 has no `&&`/`||` — rewrite bash-style chains.
            let wire = if IS_WIN { adapt_chains_for_powershell(command) } else { command.to_string() };
            let marker_line = marker_cmd(&token);
            st.current = Some(Job {
                marker: token,
                marker_re,
                command: command.to_string(),
                raw: String::new(),
                done: false,
                exit_code: None,
                clean_consumed: 0,
            });
            // Single write of both lines — avoids an inter-write race where the marker
            // arrives while the shell is still echoing the command.
            if let Err(e) = st.write(&format!("{wire}\r{marker_line}\r")) {
                st.current = None;
                return RunResult::new(RunStatus::Error, format!("failed to write to the shell: {e}"));
            }
        }
        self.wait_for_delta(timeout, cancel).await
    }

    /// Poll a still-running command for more output (and whether it has finished).
    pub async fn read(&self, timeout: Duration, cancel: Option<&CancellationToken>) -> RunResult {
        let _op = self.op.lock().await;
        {
            let mut st = self.lock();
            match st.current.as_ref() {
                None => return RunResult::new(RunStatus::Idle, "No command has been run in this session yet."),
                Some(j) if j.done => return st.delta_result(), // remaining tail + done
                Some(_) => {}
            }
        }
        self.wait_for_delta(timeout, cancel).await
    }

    /// Send raw input to the live shell (e.g. answer an interactive prompt).
    pub fn write(&self, input: &str) {
        let mut st = self.lock();
        if st.alive {
            let _ = st.write(input);
        }
    }

    async fn wait_for_delta(&self, timeout: Duration, cancel: Option<&CancellationToken>) -> RunResult {
        let deadline = Instant::now() + timeout;
        let never = CancellationToken::new();
        let cancel = cancel.unwrap_or(&never);
        loop {
            let notified = self.notify.notified();
            {
                let mut st = self.lock();
                if st.current.as_ref().map_or(true, |j| j.done) {
                    return st.delta_result();
                }
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(deadline) => {
                    return self.lock().delta_result(); // status Running with whatever arrived
                }
                _ = cancel.cancelled() => {
                    let mut st = self.lock();
                    let _ = st.write("\x03"); // Ctrl+C the foreground command
                    let mut r = st.delta_result();
                    // Free the session for the next command even if the sentinel
                    // never arrives after the interrupt.
                    if let Some(job) = st.current.as_mut() {
                        job.done = true;
                        job.exit_code = None;
                    }
                    r.status = RunStatus::Interrupted;
                    r.exit_code = None;
                    return r;
                }
            }
        }
    }

    /// Kill and forget the shell; a later run() will start a fresh one.
    pub fn reset(&self) {
        self.dispose();
    }

    pub fn dispose(&self) {
        let pty = {
            let mut st = self.lock();
            st.generation += 1; // orphan the old reader/waiter threads
            st.current = None;
            st.alive = false;
            st.warm = false;
            st.saw_data = false;
            st.pty.take()
        };
        self.notify.notify_waiters();
        if let Some(mut p) = pty {
            let _ = p.killer.kill();
            // Closing a ConPTY can block until its output is drained; never do
            // that on the caller's (async) thread.
            let _ = std::thread::Builder::new().name("qo-shell-session-close".into()).spawn(move || drop(p));
        }
    }

    /// Why the last start() failed, if it did.
    pub fn start_error(&self) -> Option<String> {
        self.lock().start_error.clone()
    }
}

impl Drop for ShellSession {
    fn drop(&mut self) {
        self.dispose();
    }
}

fn lock_state(m: &Mutex<State>) -> MutexGuard<'_, State> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_osc_csi_and_lone_cr() {
        let s = "\x1b]0;title\x07hello\x1b[32m green\x1b[0m\r\nline\rover\x1b=";
        assert_eq!(strip_ansi(s), "hello green\r\nlineover");
    }

    #[test]
    fn applies_backspaces_and_drops_repaint_echo() {
        assert_eq!(strip_ansi("e\x08echo $env:QO_T\r\npersisted"), "echo $env:QO_T\r\npersisted");
        assert_eq!(strip_ansi("\n\x08x"), "\nx");
        assert!(is_echo_of("echo $enecho $env:X", "echo $env:X"));
        assert!(is_echo_of("ls", "ls"));
        assert!(!is_echo_of("als", "ls"));
        assert!(!is_echo_of("persisted", "echo $env:X"));
    }

    #[test]
    fn finds_marker_after_glued_prompt_and_ansi() {
        let re = marker_regex("QODABCDEFGH").unwrap();
        let raw = strip_ansi("out\r\nPS> \x1b[mQODABCDEFGH:3\r\n");
        let (idx, code) = find_marker(&re, &raw).unwrap();
        assert_eq!(code, Some(3));
        assert_eq!(&raw[..idx], "out\r");
        // The echoed marker COMMAND must not count as the result line.
        let echo = "Write-Output \"QODABCDEFGH:$LASTEXITCODE\"\r\n";
        assert!(find_marker(&re, echo).is_none());
        let (_, code) = find_marker(&re, "\nQODABCDEFGH:\n").unwrap();
        assert_eq!(code, None);
    }

    #[test]
    fn cleans_echo_marker_and_prompts() {
        let job = Job {
            marker: "QODABCDEFGH".into(),
            marker_re: marker_regex("QODABCDEFGH").unwrap(),
            command: "python x.py && echo ok".into(),
            raw: "PS C:\\p> python x.py; if ($?) { echo ok }\r\nWrite-Output \"QODABCDEFGH:$LASTEXITCODE\"\r\nhello\r\n\r\n\r\n\r\nok\r\n".into(),
            done: true,
            exit_code: Some(0),
            clean_consumed: 0,
        };
        let out = cleaned_output(&job);
        if IS_WIN {
            assert_eq!(out, "hello\n\nok");
        } else {
            assert!(out.contains("hello") && out.contains("ok") && !out.contains("QOD"));
        }
    }

    #[test]
    fn decodes_split_utf8() {
        let bytes = "héllo".as_bytes();
        let mut pending = bytes[..2].to_vec(); // "h" + first byte of "é"
        assert_eq!(decode_utf8_stream(&mut pending), "h");
        pending.extend_from_slice(&bytes[2..]);
        assert_eq!(decode_utf8_stream(&mut pending), "éllo");
        let mut bad = vec![b'a', 0xff, b'b'];
        assert_eq!(decode_utf8_stream(&mut bad), "a\u{FFFD}b");
    }

    #[test]
    fn tokens_are_distinct_and_well_formed() {
        let a = random_token();
        let b = random_token();
        assert_ne!(a, b);
        assert!(STALE_SENTINEL.is_match(&format!("{a}:0")));
    }

    #[tokio::test]
    async fn read_before_run_is_idle() {
        let s = ShellSession::new(std::env::temp_dir());
        let r = s.read(Duration::from_millis(100), None).await;
        assert_eq!(r.status, RunStatus::Idle);
    }

    // harnessUpgrades: "ShellSession graceful degradation" — headless pty run.
    // Either it runs (done/running) or it reports a clean error; never panics.
    #[tokio::test]
    async fn runs_echo_in_a_real_pty_or_degrades_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let s = ShellSession::new(dir.path());
        let r = s.run("echo qo-pty-hi", Duration::from_secs(20), None).await;
        assert!(matches!(r.status, RunStatus::Done | RunStatus::Running | RunStatus::Error), "{r:?}");
        // ConPTY ships with every supported Windows: there it must really run.
        if IS_WIN {
            assert_eq!(r.status, RunStatus::Done, "{r:?}");
        }
        if r.status == RunStatus::Done {
            assert_eq!(r.output, "qo-pty-hi", "{r:?}");
            // State persists between calls in the same shell.
            let set = if IS_WIN { "$env:QO_T='persisted'" } else { "export QO_T=persisted" };
            let get = if IS_WIN { "echo $env:QO_T" } else { "echo $QO_T" };
            assert_eq!(s.run(set, Duration::from_secs(20), None).await.status, RunStatus::Done);
            let r2 = s.run(get, Duration::from_secs(20), None).await;
            assert_eq!(r2.status, RunStatus::Done, "{r2:?}");
            assert_eq!(r2.output, "persisted", "{r2:?}");
            // A native command's exit code comes back through the sentinel.
            let fail = if IS_WIN { "cmd /c exit 3" } else { "sh -c 'exit 3'" };
            let r3 = s.run(fail, Duration::from_secs(20), None).await;
            assert_eq!((r3.status, r3.exit_code), (RunStatus::Done, Some(3)), "{r3:?}");
        }
        s.dispose();
        assert!(!s.is_alive());
    }

    #[tokio::test]
    async fn long_command_reports_running_then_can_be_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let s = ShellSession::new(dir.path());
        let cmd = if IS_WIN { "Start-Sleep -Seconds 30" } else { "sleep 30" };
        let r = s.run(cmd, Duration::from_millis(1500), None).await;
        if r.status == RunStatus::Error {
            return; // no pty on this machine — covered by the degradation test
        }
        assert_eq!(r.status, RunStatus::Running, "{r:?}");
        let busy = s.run("echo nope", Duration::from_millis(500), None).await;
        assert_eq!(busy.status, RunStatus::Busy);
        let token = CancellationToken::new();
        token.cancel();
        let r = s.read(Duration::from_secs(5), Some(&token)).await;
        assert_eq!(r.status, RunStatus::Interrupted);
        s.dispose();
    }
}
