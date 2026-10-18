//! The one HTTP call shape the media layer makes: POST a JSON body, get a JSON
//! body back. Every await races the cancel token (quit / Esc) and a timeout,
//! and failures come back as typed [`ProviderError`]s whose messages match the
//! TS (`vision (nvidia) HTTP 401: …`).

use std::time::Duration;

use anyhow::{anyhow, Result};
use qo_core::{CancellationToken, ProviderError};
use reqwest::Client;
use serde_json::Value;

use crate::proc::head_chars;

/// Connect timeout (undici's default, which the Node fetch used).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Whole-request cap. The TS fetch had none; a vision/omni call that never
/// answers must not hold a live session forever (the user can also quit).
pub(crate) const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// A fresh client per call (these calls are rare; a pool is tied to the runtime
/// that created it). Proxies are not consulted — Node's fetch ignored them and
/// a system proxy must never intercept a `localhost` Ollama.
fn client() -> Client {
    Client::builder().connect_timeout(CONNECT_TIMEOUT).no_proxy().build().unwrap_or_else(|_| Client::new())
}

/// `base` without trailing slashes, with `/v1` appended unless already there.
pub fn api_base(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if b.ends_with("/v1") {
        b.to_string()
    } else {
        format!("{b}/v1")
    }
}

/// POST `body` as JSON to `url` (bearer `api_key` when non-empty) and parse the
/// JSON response. `label` names the caller in errors ("vision (nvidia)");
/// error bodies are clipped to `body_chars` characters.
pub(crate) async fn post_json(
    url: &str,
    api_key: &str,
    body: &Value,
    label: &str,
    body_chars: usize,
    timeout: Duration,
    cancel: &CancellationToken,
) -> Result<Value> {
    if cancel.is_cancelled() {
        return Err(ProviderError::Interrupted.into());
    }
    let mut req = client().post(url).json(body);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let call = async {
        let res = req.send().await.map_err(|e| network(label, &e))?;
        let status = res.status();
        let text = res.text().await.map_err(|e| network(label, &e))?;
        if !status.is_success() {
            return Err(ProviderError::Http {
                provider: label.to_string(),
                status: status.as_u16(),
                body: head_chars(&text, body_chars),
            }
            .into());
        }
        serde_json::from_str::<Value>(&text)
            .map_err(|e| anyhow!("{label}: invalid JSON response ({e}): {}", head_chars(&text, body_chars)))
    };
    tokio::select! {
        r = tokio::time::timeout(timeout, call) => match r {
            Ok(r) => r,
            Err(_) => Err(ProviderError::Timeout { provider: label.to_string(), ms: timeout.as_millis() as u64 }.into()),
        },
        _ = cancel.cancelled() => Err(ProviderError::Interrupted.into()),
    }
}

fn network(label: &str, e: &reqwest::Error) -> anyhow::Error {
    // reqwest's Display hides the root cause ("error sending request"); walk the chain.
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        msg.push_str(": ");
        msg.push_str(&s.to_string());
        src = s.source();
    }
    ProviderError::Network { provider: label.to_string(), message: msg }.into()
}

/// `json.choices[0].message.content`, trimmed ("" when absent / not a string).
pub(crate) fn openai_content(json: &Value) -> String {
    json.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("").trim().to_string()
}

#[cfg(test)]
pub(crate) mod mock {
    //! A one-shot local HTTP server for tests: records the request, replies
    //! with a canned status + body. No paid API is ever contacted.

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[derive(Debug, Default)]
    pub struct Captured {
        pub request_line: String,
        pub headers: Vec<(String, String)>,
        pub body: serde_json::Value,
    }

    impl Captured {
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
        }
    }

    /// Start the server; returns its base URL and a receiver for the request.
    pub async fn serve(status: u16, body: &str) -> (String, oneshot::Receiver<Captured>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
        let addr = listener.local_addr().expect("mock addr");
        let body = body.to_string();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let (head_end, content_len) = loop {
                let Ok(n) = sock.read(&mut tmp).await else { return };
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..pos]).to_string();
                    let len = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                        })
                        .unwrap_or(0);
                    break (pos + 4, len);
                }
            };
            while buf.len() < head_end + content_len {
                let Ok(n) = sock.read(&mut tmp).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let head = String::from_utf8_lossy(&buf[..head_end - 4]).to_string();
            let mut lines = head.lines();
            let request_line = lines.next().unwrap_or("").to_string();
            let headers = lines
                .filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
                .collect();
            let end = (head_end + content_len).min(buf.len());
            let body_json = serde_json::from_slice(&buf[head_end..end]).unwrap_or(serde_json::Value::Null);
            let reason = if status == 200 { "OK" } else { "Error" };
            let resp = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
            let _ = tx.send(Captured { request_line, headers, body: body_json });
        });
        (format!("http://{addr}"), rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn api_base_appends_v1_once() {
        assert_eq!(api_base("https://integrate.api.nvidia.com"), "https://integrate.api.nvidia.com/v1");
        assert_eq!(api_base("https://integrate.api.nvidia.com/v1/"), "https://integrate.api.nvidia.com/v1");
        assert_eq!(api_base("http://localhost:1234//"), "http://localhost:1234/v1");
    }

    #[test]
    fn content_extraction() {
        assert_eq!(openai_content(&json!({"choices":[{"message":{"content":"  hi \n"}}]})), "hi");
        assert_eq!(openai_content(&json!({"choices":[{"message":{"content":null}}]})), "");
        assert_eq!(openai_content(&json!({})), "");
    }

    #[tokio::test]
    async fn http_error_is_typed_and_clipped() {
        let long = "x".repeat(500);
        let (base, _rx) = mock::serve(401, &long).await;
        let cancel = CancellationToken::new();
        let e = post_json(&format!("{base}/v1/chat/completions"), "k", &json!({}), "vision (nvidia)", 200, REQUEST_TIMEOUT, &cancel)
            .await
            .unwrap_err();
        assert_eq!(e.to_string(), format!("vision (nvidia) HTTP 401: {}", "x".repeat(200)));
        assert!(matches!(e.downcast_ref::<ProviderError>(), Some(ProviderError::Http { status: 401, .. })));
    }

    #[tokio::test]
    async fn cancel_and_timeout() {
        // A listener that accepts but never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/x", listener.local_addr().unwrap());
        let hold = tokio::spawn(async move {
            let mut socks = Vec::new();
            while let Ok((s, _)) = listener.accept().await {
                socks.push(s);
            }
        });
        let cancel = CancellationToken::new();
        let e = post_json(&url, "", &json!({}), "omni (nvidia)", 300, Duration::from_millis(200), &cancel)
            .await
            .unwrap_err();
        assert_eq!(e.to_string(), "omni (nvidia) request timed out after 200ms");

        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            c2.cancel();
        });
        let e = post_json(&url, "", &json!({}), "omni (nvidia)", 300, REQUEST_TIMEOUT, &cancel).await.unwrap_err();
        assert!(qo_core::error::is_interrupt(&e));
        hold.abort();
    }
}
