//! Anthropic OAuth via the `ant` CLI (port of `antPrintCredentials` in
//! providers/index.ts).
//!
//! `ant auth print-credentials --access-token` is the documented way to use a
//! Claude Pro/Max subscription: it refreshes the token if needed and prints
//! just the bearer. The lookup is cached for a minute so the per-turn
//! fallback/picker checks don't spawn a subprocess every time; OAuth tokens
//! live far longer than that, and a 401 forces a fresh mint.
//!
//! Two entry points share the cache: [`ant_print_credentials`] (blocking, for
//! the synchronous factory helpers, like the TS `execa.sync`) and
//! [`ant_print_credentials_async`] (tokio; the child is killed if the future
//! is dropped, so a provider can race it against the Esc token).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::provider_config::TokenSource;

/// How long a looked-up token (or a failed lookup) is reused.
const CLI_TOKEN_TTL: Duration = Duration::from_secs(60);
/// Upper bound for one `ant` invocation.
const ANT_TIMEOUT: Duration = Duration::from_secs(15);

static CLI_TOKEN_CACHE: Mutex<Option<(Instant, Option<String>)>> = Mutex::new(None);

fn cached(force: bool) -> Option<Option<String>> {
    if force {
        return None;
    }
    let guard = CLI_TOKEN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some((at, tok)) if at.elapsed() < CLI_TOKEN_TTL => Some(tok.clone()),
        _ => None,
    }
}

fn store(token: Option<String>) {
    let mut guard = CLI_TOKEN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some((Instant::now(), token));
}

/// Forget the cached lookup (tests, `/login`-style flows).
pub fn clear_ant_token_cache() {
    let mut guard = CLI_TOKEN_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// A real bearer token is a single whitespace-free string. `ant` exits 0 even
/// when not logged in (printing a message to stderr), so stray output must
/// never be treated as a credential.
pub(crate) fn valid_token(stdout: &str) -> Option<String> {
    let tok = stdout.trim();
    (!tok.is_empty() && !tok.chars().any(char::is_whitespace) && tok.len() >= 20).then(|| tok.to_string())
}

/// Binaries to try, in order: whatever `ant` resolves to on PATH (incl. the
/// `.cmd` shim npm installs on Windows), then the bare names.
fn ant_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = which::which("ant") {
        out.push(p);
    }
    out.push(PathBuf::from("ant"));
    out.push(PathBuf::from("ant.cmd"));
    out
}

const ANT_ARGS: [&str; 3] = ["auth", "print-credentials", "--access-token"];

/// Blocking lookup (≤ 15 s per binary). Best-effort: `None` when `ant` isn't
/// installed or the user isn't logged in. Cached for a minute unless `force`.
///
/// This blocks the calling thread; from async code prefer
/// [`ant_print_credentials_async`].
pub fn ant_print_credentials(force: bool) -> Option<String> {
    if let Some(hit) = cached(force) {
        return hit;
    }
    let mut token = None;
    for bin in ant_candidates() {
        if let Some(out) = run_blocking(&bin) {
            if let Some(tok) = valid_token(&out) {
                token = Some(tok);
                break;
            }
        }
    }
    store(token.clone());
    token
}

/// Run `bin auth print-credentials --access-token` with a timeout; stdout on
/// a clean exit.
fn run_blocking(bin: &PathBuf) -> Option<String> {
    use std::io::Read;
    let mut child = std::process::Command::new(bin)
        .args(ANT_ARGS)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + ANT_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                child.stdout.take()?.read_to_string(&mut out).ok()?;
                return status.success().then_some(out);
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(25)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Async lookup sharing the same cache. Dropping the future kills the child.
pub async fn ant_print_credentials_async(force: bool) -> Option<String> {
    if let Some(hit) = cached(force) {
        return hit;
    }
    let mut token = None;
    for bin in ant_candidates() {
        let child = tokio::process::Command::new(&bin)
            .args(ANT_ARGS)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn();
        let Ok(child) = child else { continue };
        match tokio::time::timeout(ANT_TIMEOUT, child.wait_with_output()).await {
            Ok(Ok(out)) if out.status.success() => {
                if let Some(tok) = valid_token(&String::from_utf8_lossy(&out.stdout)) {
                    token = Some(tok);
                    break;
                }
            }
            _ => {}
        }
    }
    store(token.clone());
    token
}

/// [`TokenSource`] backed by the `ant` CLI: the cached token normally, a fresh
/// mint when forced (after a 401).
#[derive(Clone, Copy, Debug, Default)]
pub struct AntCliTokenSource;

#[async_trait]
impl TokenSource for AntCliTokenSource {
    async fn token(&self, force: bool) -> Option<String> {
        ant_print_credentials_async(force).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_validation_rejects_stray_output() {
        assert_eq!(valid_token("  sk-ant-oat01-abcdefghijklmnop\n").as_deref(), Some("sk-ant-oat01-abcdefghijklmnop"));
        assert!(valid_token("").is_none());
        assert!(valid_token("not logged in, run ant auth login").is_none());
        assert!(valid_token("short").is_none());
    }
}
