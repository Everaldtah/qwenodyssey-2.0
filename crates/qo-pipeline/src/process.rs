//! Child-process runner shared by the verify loop and the repo scanner's git
//! probes.
//!
//! Rules (the TS bug class "something stole the keyboard" must be impossible):
//! - stdin is always null;
//! - every run has a timeout and an optional Esc [`CancellationToken`];
//! - on timeout / cancel the whole process TREE is killed (`npm run test`
//!   spawns node grandchildren that would otherwise keep running and keep the
//!   output pipes open);
//! - stdout and stderr are captured interleaved in arrival order (execa `all`),
//!   and separately for probes that parse stdout.

use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

/// Result of a captured run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcOutput {
    /// Exit code; `None` when killed (timeout / cancel) or terminated by a signal.
    pub exit_code: Option<i32>,
    /// stdout + stderr interleaved in arrival order, lossily decoded as UTF-8
    /// (CRLF normalised to LF).
    pub output: String,
    /// stdout alone (same decoding).
    pub stdout: String,
    /// stderr alone (same decoding).
    pub stderr: String,
    pub timed_out: bool,
    pub interrupted: bool,
}

impl ProcOutput {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out && !self.interrupted
    }
}

/// How long to keep draining the pipes after the child exits (grandchildren
/// that inherited the pipes may keep them open indefinitely).
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Windows `CREATE_NO_WINDOW`: never flash a console window for a check run.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Resolve `program` the way a shell would (PATHEXT on Windows, so `npm`
/// finds `npm.cmd`). Falls back to the bare name, letting spawn report it.
pub fn resolve_program(program: &str) -> std::ffi::OsString {
    match which::which(program) {
        Ok(p) => p.into_os_string(),
        Err(_) => program.into(),
    }
}

type Buf = Arc<Mutex<Vec<u8>>>;

/// Spawn `program args…` in `cwd` and capture its output, honouring `timeout`
/// and `cancel`. Spawn failures (e.g. program not found) are returned as `Err`.
pub async fn run_captured(
    program: &str,
    args: &[String],
    cwd: &Path,
    envs: &[(&str, &str)],
    timeout: Duration,
    cancel: Option<&CancellationToken>,
) -> std::io::Result<ProcOutput> {
    let mut cmd = Command::new(resolve_program(program));
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        // Own process group so the whole tree can be signalled at once.
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd.spawn()?;
    let all: Buf = Arc::new(Mutex::new(Vec::new()));
    let out_buf: Buf = Arc::new(Mutex::new(Vec::new()));
    let err_buf: Buf = Arc::new(Mutex::new(Vec::new()));
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        readers.push(tokio::spawn(pump(out, all.clone(), out_buf.clone())));
    }
    if let Some(err) = child.stderr.take() {
        readers.push(tokio::spawn(pump(err, all.clone(), err_buf.clone())));
    }

    let never = CancellationToken::new();
    let cancel = cancel.unwrap_or(&never);
    let mut result = ProcOutput::default();
    tokio::select! {
        status = child.wait() => {
            result.exit_code = status.ok().and_then(|s| s.code());
        }
        _ = tokio::time::sleep(timeout) => {
            result.timed_out = true;
            kill_tree(&mut child).await;
        }
        _ = cancel.cancelled() => {
            result.interrupted = true;
            kill_tree(&mut child).await;
        }
    }

    // Drain whatever the readers still have, bounded; a grandchild that kept
    // the pipe open must not keep a reader task alive forever.
    let _ = tokio::time::timeout(DRAIN_GRACE, async {
        for r in readers.iter_mut() {
            let _ = r.await;
        }
    })
    .await;
    for r in &readers {
        r.abort();
    }
    result.output = decode(&all);
    result.stdout = decode(&out_buf);
    result.stderr = decode(&err_buf);
    Ok(result)
}

fn decode(buf: &Buf) -> String {
    let bytes = buf.lock().map(|b| b.clone()).unwrap_or_default();
    String::from_utf8_lossy(&bytes).replace("\r\n", "\n")
}

async fn pump<R: tokio::io::AsyncRead + Unpin>(mut r: R, all: Buf, own: Buf) {
    let mut chunk = [0u8; 8192];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(mut b) = all.lock() {
                    b.extend_from_slice(&chunk[..n]);
                }
                if let Ok(mut b) = own.lock() {
                    b.extend_from_slice(&chunk[..n]);
                }
            }
        }
    }
}

/// Kill the child and everything it spawned.
pub async fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .status()
                .await;
        }
        #[cfg(unix)]
        {
            // Negative pid = the process group created at spawn.
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{pid}")])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
        }
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_output_and_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let out =
            run_captured("git", &["--version".to_string()], dir.path(), &[], Duration::from_secs(30), None).await;
        // git is a hard requirement of the TS tool too; skip quietly if absent.
        if let Ok(out) = out {
            assert_eq!(out.exit_code, Some(0));
            assert!(out.output.contains("git version"));
            assert!(out.stdout.contains("git version"));
            assert!(out.stderr.is_empty());
            assert!(out.success());
        }
    }

    #[tokio::test]
    async fn missing_program_is_a_spawn_error() {
        let dir = tempfile::tempdir().unwrap();
        let r = run_captured("definitely-not-a-real-program-xyz", &[], dir.path(), &[], Duration::from_secs(5), None)
            .await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn cancel_kills_the_child() {
        if which::which("node").is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            c2.cancel();
        });
        let t0 = std::time::Instant::now();
        let out = run_captured(
            "node",
            &["-e".to_string(), "setTimeout(()=>{}, 60000)".to_string()],
            dir.path(),
            &[],
            Duration::from_secs(60),
            Some(&cancel),
        )
        .await
        .unwrap();
        assert!(out.interrupted);
        assert!(!out.success());
        assert!(t0.elapsed() < Duration::from_secs(20));
    }

    #[tokio::test]
    async fn timeout_kills_the_child() {
        if which::which("node").is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let out = run_captured(
            "node",
            &["-e".to_string(), "console.log('hi'); setTimeout(()=>{}, 60000)".to_string()],
            dir.path(),
            &[],
            Duration::from_millis(1500),
            None,
        )
        .await
        .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.exit_code, None);
    }
}
