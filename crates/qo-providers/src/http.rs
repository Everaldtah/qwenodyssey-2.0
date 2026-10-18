//! HTTP plumbing shared by every backend — the reliability rules of the v0.4.2
//! "model switch freeze" fix live here, in one place:
//!
//! - the request timeout covers connect + response headers ([`send`]);
//! - after the headers a stall watchdog aborts a body that goes silent for the
//!   timeout ([`next_chunk`] → [`ProviderError::Stalled`]);
//! - the caller's cancel token (Esc) is honoured at EVERY await — connect,
//!   headers, each body chunk — and yields [`ProviderError::Interrupted`].
//!
//! Every wait is a `tokio::select!` over the cancel token, a timer and the I/O,
//! so a backend that accepts a request and never answers can never hold the
//! terminal hostage.

use std::future::pending;
use std::time::Duration;

use bytes::Bytes;
use qo_core::{CancellationToken, ProviderError};
use reqwest::{Client, RequestBuilder, Response};

/// Connect timeout for the shared client (mirrors undici's 10 s default, which
/// is what the Node fetch of the TS version used).
pub(crate) const CONNECT_TIMEOUT_MS: u64 = 10_000;

/// Bound for reading an error body / metadata response when no request timeout
/// is configured (local backends), so a half-dead server can't wedge a turn.
pub(crate) const FALLBACK_READ_MS: u64 = 30_000;

/// Error bodies are clipped to this many characters (TS `safeText`).
const ERROR_BODY_CHARS: usize = 300;

/// Build the HTTP client a provider uses. One per provider instance: a reqwest
/// connection pool is tied to the tokio runtime that created its connections.
/// Proxies are not consulted (Node's fetch ignored them too, and a system proxy
/// must never intercept `localhost` model servers).
pub(crate) fn build_client() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_millis(CONNECT_TIMEOUT_MS))
        .no_proxy()
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Resolve when `cancel` fires; never when there is no token.
pub(crate) async fn cancelled(cancel: Option<&CancellationToken>) {
    match cancel {
        Some(c) => c.cancelled().await,
        None => pending::<()>().await,
    }
}

/// Sleep for `ms` (when set and > 0); never resolve otherwise.
pub(crate) async fn sleep_ms(ms: Option<u64>) {
    match ms.filter(|m| *m > 0) {
        Some(ms) => tokio::time::sleep(Duration::from_millis(ms)).await,
        None => pending::<()>().await,
    }
}

/// Render a reqwest error with its whole source chain ("error sending request
/// … : tcp connect error: Connection refused"), which the fallback classifier
/// and the user both need.
pub(crate) fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut out = err.to_string();
    let mut src = err.source();
    while let Some(s) = src {
        let part = s.to_string();
        if !out.contains(&part) {
            out.push_str(": ");
            out.push_str(&part);
        }
        src = s.source();
    }
    out
}

pub(crate) fn network_error(provider: &str, err: &reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        return ProviderError::Timeout { provider: provider.to_string(), ms: CONNECT_TIMEOUT_MS };
    }
    ProviderError::Network { provider: provider.to_string(), message: error_chain(err) }
}

/// Send `req` and resolve once the response headers arrive.
///
/// `timeout_ms` bounds connect + headers ([`ProviderError::Timeout`]); the
/// cancel token aborts at any point ([`ProviderError::Interrupted`]). Dropping
/// the in-flight future closes the connection.
pub(crate) async fn send(
    provider: &str,
    req: RequestBuilder,
    cancel: Option<&CancellationToken>,
    timeout_ms: Option<u64>,
) -> Result<Response, ProviderError> {
    if cancel.is_some_and(|c| c.is_cancelled()) {
        return Err(ProviderError::Interrupted);
    }
    tokio::select! {
        biased;
        _ = cancelled(cancel) => Err(ProviderError::Interrupted),
        _ = sleep_ms(timeout_ms) => Err(ProviderError::Timeout {
            provider: provider.to_string(),
            ms: timeout_ms.unwrap_or(0),
        }),
        r = req.send() => r.map_err(|e| network_error(provider, &e)),
    }
}

/// Next body chunk (`None` at end of body). `stall_ms` is the watchdog: no
/// bytes for that long → [`ProviderError::Stalled`]. Re-armed on every call, so
/// it measures silence, not total duration.
pub(crate) async fn next_chunk(
    provider: &str,
    res: &mut Response,
    cancel: Option<&CancellationToken>,
    stall_ms: Option<u64>,
) -> Result<Option<Bytes>, ProviderError> {
    tokio::select! {
        biased;
        _ = cancelled(cancel) => Err(ProviderError::Interrupted),
        _ = sleep_ms(stall_ms) => Err(ProviderError::Stalled {
            provider: provider.to_string(),
            ms: stall_ms.unwrap_or(0),
        }),
        r = res.chunk() => r.map_err(|e| network_error(provider, &e)),
    }
}

/// Read a whole (non-streamed) body, chunk by chunk, under the same cancel +
/// stall rules as a stream.
pub(crate) async fn read_body(
    provider: &str,
    mut res: Response,
    cancel: Option<&CancellationToken>,
    stall_ms: Option<u64>,
) -> Result<Vec<u8>, ProviderError> {
    let mut out = Vec::new();
    while let Some(chunk) = next_chunk(provider, &mut res, cancel, stall_ms).await? {
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Read a JSON body. Garbage is reported as an API error (the TS `res.json()`
/// threw a SyntaxError there).
pub(crate) async fn read_json(
    provider: &str,
    res: Response,
    cancel: Option<&CancellationToken>,
    stall_ms: Option<u64>,
) -> Result<serde_json::Value, ProviderError> {
    let bytes = read_body(provider, res, cancel, stall_ms).await?;
    serde_json::from_slice(&bytes).map_err(|e| ProviderError::Api {
        provider: provider.to_string(),
        message: format!("invalid JSON in response: {e}"),
    })
}

/// The first 300 characters of an error response body (TS `safeText`).
/// Esc still interrupts; any other read failure becomes `"<no body>"`.
pub(crate) async fn safe_text(
    provider: &str,
    res: Response,
    cancel: Option<&CancellationToken>,
    stall_ms: Option<u64>,
) -> Result<String, ProviderError> {
    let stall = stall_ms.filter(|m| *m > 0).or(Some(FALLBACK_READ_MS));
    match read_body(provider, res, cancel, stall).await {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            Ok(text.chars().take(ERROR_BODY_CHARS).collect())
        }
        Err(ProviderError::Interrupted) => Err(ProviderError::Interrupted),
        Err(_) => Ok("<no body>".to_string()),
    }
}

/// Splits a byte stream into lines. Splitting happens on raw bytes, so a
/// multi-byte UTF-8 character cut across two network chunks is reassembled
/// before decoding (the TS `TextDecoder({stream:true})` behaviour).
#[derive(Default)]
pub(crate) struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    /// Append a chunk and return every line it completed (without the `\n`).
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            lines.push(String::from_utf8_lossy(&line[..line.len() - 1]).into_owned());
        }
        lines
    }

    /// Whatever is left after the body ended (a final frame without newline).
    pub(crate) fn finish(&mut self) -> Option<String> {
        if self.buf.iter().all(|b| b.is_ascii_whitespace()) {
            self.buf.clear();
            return None;
        }
        let rest = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        Some(rest)
    }
}

/// The JSON payload of an SSE `data:` line, or `None` for comments, `event:`
/// lines, keep-alives, `[DONE]` and partial/garbage frames.
pub(crate) fn sse_payload(line: &str) -> Option<serde_json::Value> {
    let trimmed = line.trim();
    let payload = trimmed.strip_prefix("data:")?.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    serde_json::from_str(payload).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_buffer_reassembles_split_frames_and_utf8() {
        let mut lb = LineBuffer::default();
        assert!(lb.push(b"data: {\"a\":").is_empty());
        let lines = lb.push(b"1}\n\ndata: x");
        assert_eq!(lines, vec!["data: {\"a\":1}".to_string(), String::new()]);
        // "é" = C3 A9 split across chunks
        assert!(lb.push(b"\xC3").is_empty());
        let lines = lb.push(b"\xA9\n");
        assert_eq!(lines, vec!["data: x\u{e9}".to_string()]);
        assert_eq!(lb.finish(), None);
        lb.push(b"tail");
        assert_eq!(lb.finish().as_deref(), Some("tail"));
    }

    #[test]
    fn sse_payload_filters_non_data() {
        assert!(sse_payload("event: ping").is_none());
        assert!(sse_payload("data: [DONE]").is_none());
        assert!(sse_payload(": keep-alive").is_none());
        assert!(sse_payload("data: {\"partial\"").is_none());
        assert_eq!(sse_payload("  data:{\"x\":1}  ").unwrap()["x"], 1);
    }
}
