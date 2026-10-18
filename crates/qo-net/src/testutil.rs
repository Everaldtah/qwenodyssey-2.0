//! Test-only helpers: a tiny HTTP/1.1 mock server on a loopback port, so the
//! web / GitHub / embedding code paths run headless without real network.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// One request the mock received.
#[derive(Clone, Debug, Default)]
pub struct Recorded {
    pub method: String,
    /// Path including the query string.
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// What the mock answers.
#[derive(Clone, Debug)]
pub struct Reply {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub headers: Vec<(String, String)>,
    /// Wait this long before answering (timeout / cancellation tests).
    pub delay_ms: u64,
}

impl Reply {
    pub fn text(status: u16, content_type: &str, body: &str) -> Self {
        Self { status, content_type: content_type.into(), body: body.as_bytes().to_vec(), headers: vec![], delay_ms: 0 }
    }
    pub fn json(status: u16, v: Value) -> Self {
        Self::text(status, "application/json", &v.to_string())
    }
    pub fn html(body: &str) -> Self {
        Self::text(200, "text/html; charset=utf-8", body)
    }
    pub fn delayed(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self
    }
    pub fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
}

pub type Handler = Arc<dyn Fn(&Recorded) -> Reply + Send + Sync>;

pub struct MockServer {
    /// `http://127.0.0.1:<port>`
    pub base: String,
    log: Arc<Mutex<Vec<Recorded>>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub fn requests(&self) -> Vec<Recorded> {
        self.log.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub async fn serve(handler: Handler) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    let log: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));
    let log2 = log.clone();
    let task = tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = listener.accept().await else { break };
            let handler = handler.clone();
            let log = log2.clone();
            tokio::spawn(async move {
                let _ = handle(sock, handler, log).await;
            });
        }
    });
    MockServer { base, log, task }
}

async fn handle(mut sock: TcpStream, handler: Handler, log: Arc<Mutex<Vec<Recorded>>>) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let method = first.next().unwrap_or("").to_string();
    let path = first.next().unwrap_or("").to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[header_end..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    let rec = Recorded { method, path, headers, body: String::from_utf8_lossy(&body).into_owned() };
    if let Ok(mut l) = log.lock() {
        l.push(rec.clone());
    }
    let reply = handler(&rec);
    if reply.delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(reply.delay_ms)).await;
    }
    let mut out = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reply.status,
        reply.content_type,
        reply.body.len()
    );
    for (k, v) in &reply.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    sock.write_all(out.as_bytes()).await?;
    sock.write_all(&reply.body).await?;
    sock.flush().await?;
    let _ = sock.shutdown().await;
    Ok(())
}
