//! Child-process plumbing shared by every external tool (ffmpeg, whisper,
//! piper, ffplay, PowerShell).
//!
//! Rules (PORTING.md): stdin is `Stdio::null()` unless the tool genuinely needs
//! input (so a child can never steal the keyboard from the terminal UI), every
//! child is killed when its future is dropped, and every run honours a
//! [`CancellationToken`] (Esc / quit) plus an optional timeout.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{anyhow, Result};
use qo_core::{CancellationToken, ProviderError};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Captured result of a finished child.
#[derive(Debug, Default)]
pub(crate) struct ProcOutput {
    /// Exit code (None when killed by a signal).
    pub code: Option<i32>,
    pub success: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl ProcOutput {
    /// stdout + stderr as one lossy string (execa's `all`).
    pub fn all(&self) -> String {
        let mut s = String::from_utf8_lossy(&self.stdout).into_owned();
        s.push_str(&String::from_utf8_lossy(&self.stderr));
        s
    }
}

/// A command with the house defaults: stdin null, stdout/stderr piped,
/// killed when dropped.
pub(crate) fn command<S: AsRef<OsStr>>(bin: impl AsRef<OsStr>, args: &[S]) -> Command {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    cmd
}

/// Display name of a binary for error messages (file name, not the full path).
pub(crate) fn bin_label(bin: &Path) -> String {
    bin.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| bin.display().to_string())
}

/// Run `bin args` to completion. `input`, when given, is piped to stdin and
/// then stdin is closed (piper / SAPI read the text to speak this way).
/// Does NOT fail on a non-zero exit (execa `reject: false`); use [`check`].
///
/// Errors: spawn failure, the timeout (child killed), or cancellation
/// (child killed, [`ProviderError::Interrupted`]).
pub(crate) async fn run<S: AsRef<OsStr>>(
    bin: &Path,
    args: &[S],
    input: Option<&[u8]>,
    timeout: Option<Duration>,
    cancel: &CancellationToken,
) -> Result<ProcOutput> {
    if cancel.is_cancelled() {
        return Err(ProviderError::Interrupted.into());
    }
    let mut cmd = command(bin, args);
    if input.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd.spawn().map_err(|e| anyhow!("failed to start {}: {e}", bin.display()))?;
    if let (Some(data), Some(mut stdin)) = (input, child.stdin.take()) {
        let data = data.to_vec();
        // Feed stdin concurrently so a chatty child can't deadlock on a full stdout pipe.
        tokio::spawn(async move {
            let _ = stdin.write_all(&data).await;
            let _ = stdin.shutdown().await;
        });
    }
    // `wait_with_output` owns the child: dropping this future (timeout/cancel)
    // drops the child, and `kill_on_drop` terminates it.
    let wait = child.wait_with_output();
    let out = match timeout {
        Some(limit) => tokio::select! {
            r = tokio::time::timeout(limit, wait) => match r {
                Ok(r) => r?,
                Err(_) => return Err(anyhow!("{} timed out after {}ms", bin_label(bin), limit.as_millis())),
            },
            _ = cancel.cancelled() => return Err(ProviderError::Interrupted.into()),
        },
        None => tokio::select! {
            r = wait => r?,
            _ = cancel.cancelled() => return Err(ProviderError::Interrupted.into()),
        },
    };
    Ok(ProcOutput { code: out.status.code(), success: out.status.success(), stdout: out.stdout, stderr: out.stderr })
}

/// execa `reject: true`: turn a non-zero exit into an error carrying the tail
/// of stderr (where ffmpeg/whisper explain themselves).
pub(crate) fn check(out: ProcOutput, bin: &Path) -> Result<ProcOutput> {
    if out.success {
        return Ok(out);
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let tail = tail_chars(stderr.trim(), 400);
    let code = out.code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into());
    if tail.is_empty() {
        Err(anyhow!("{} exited with code {code}", bin_label(bin)))
    } else {
        Err(anyhow!("{} exited with code {code}: {tail}", bin_label(bin)))
    }
}

/// First `n` chars of `s` (JS `.slice(0, n)` for messages).
pub(crate) fn head_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Last `n` chars of `s`.
pub(crate) fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    s.chars().skip(count.saturating_sub(n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_slicing() {
        assert_eq!(head_chars("héllo", 2), "hé");
        assert_eq!(tail_chars("héllo", 3), "llo");
        assert_eq!(tail_chars("ab", 5), "ab");
    }

    #[test]
    fn check_reports_stderr_tail() {
        let out = ProcOutput { code: Some(1), success: false, stdout: vec![], stderr: b"  boom\n".to_vec() };
        let e = check(out, Path::new("C:/x/ffmpeg.exe")).unwrap_err();
        assert_eq!(e.to_string(), "ffmpeg.exe exited with code 1: boom");
        let ok = ProcOutput { code: Some(0), success: true, ..Default::default() };
        assert!(check(ok, Path::new("ffmpeg")).is_ok());
    }

    #[tokio::test]
    async fn missing_binary_is_an_error_not_a_panic() {
        let cancel = CancellationToken::new();
        let r = run(Path::new("qo-media-definitely-missing-binary-xyz"), &["--help"], None, None, &cancel).await;
        assert!(r.unwrap_err().to_string().contains("failed to start"));
    }

    #[tokio::test]
    async fn cancelled_before_spawn_is_interrupt() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let r = run(Path::new("qo-media-definitely-missing-binary-xyz"), &["x"], None, None, &cancel).await;
        assert!(qo_core::error::is_interrupt(&r.unwrap_err()));
    }
}
