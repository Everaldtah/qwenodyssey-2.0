//! A tiny local HTTP/1.1 mock server on 127.0.0.1 for the provider tests (no
//! real APIs are ever called). Each connection gets one request and one
//! scripted [`Reply`]; the connection closes afterwards.

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

pub enum Reply {
    /// Status + JSON body.
    Json(u16, Value),
    /// Status + plain-text body.
    Text(u16, String),
    /// 200 streaming body: each chunk written and flushed separately.
    Stream(Vec<String>),
    /// 200 streaming body, then the socket stays open and silent.
    StreamThenStall(Vec<String>),
    /// 200 headers, then nothing (records when the client hangs up).
    HeadersThenHang,
    /// Never answer at all.
    Hang,
}

pub struct MockServer {
    pub base_url: String,
    requests: Arc<Mutex<Vec<Request>>>,
    client_closed: Arc<AtomicBool>,
}

impl MockServer {
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    pub fn bodies(&self) -> Vec<Value> {
        self.requests().iter().map(Request::json).collect()
    }

    /// Did a client close a `HeadersThenHang` connection (i.e. drop the body)?
    pub fn client_closed(&self) -> bool {
        self.client_closed.load(Ordering::SeqCst)
    }
}

pub async fn serve<F>(handler: F) -> MockServer
where
    F: Fn(&Request) -> Reply + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let client_closed = Arc::new(AtomicBool::new(false));
    let handler = Arc::new(handler);
    {
        let requests = requests.clone();
        let client_closed = client_closed.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else { return };
                let handler = handler.clone();
                let requests = requests.clone();
                let client_closed = client_closed.clone();
                tokio::spawn(async move {
                    let _ = handle(sock, handler.as_ref(), requests, client_closed).await;
                });
            }
        });
    }
    MockServer { base_url: format!("http://{addr}"), requests, client_closed }
}

async fn read_request(sock: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or_default().split_whitespace();
    let method = first.next().unwrap_or_default().to_string();
    let path = first.next().unwrap_or_default().to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Ok(Some(Request { method, path, headers, body }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

async fn write_head(sock: &mut TcpStream, status: u16, content_type: &str, len: Option<usize>) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n", reason(status));
    if let Some(n) = len {
        head.push_str(&format!("Content-Length: {n}\r\n"));
    }
    head.push_str("\r\n");
    sock.write_all(head.as_bytes()).await?;
    sock.flush().await
}

async fn handle<F>(
    mut sock: TcpStream,
    handler: &F,
    requests: Arc<Mutex<Vec<Request>>>,
    client_closed: Arc<AtomicBool>,
) -> std::io::Result<()>
where
    F: Fn(&Request) -> Reply + Send + Sync + 'static,
{
    let Some(req) = read_request(&mut sock).await? else { return Ok(()) };
    let reply = handler(&req);
    requests.lock().unwrap().push(req);
    match reply {
        Reply::Json(status, v) => {
            let body = serde_json::to_vec(&v).unwrap();
            write_head(&mut sock, status, "application/json", Some(body.len())).await?;
            sock.write_all(&body).await?;
        }
        Reply::Text(status, t) => {
            write_head(&mut sock, status, "text/plain", Some(t.len())).await?;
            sock.write_all(t.as_bytes()).await?;
        }
        Reply::Stream(chunks) => {
            write_head(&mut sock, 200, "text/event-stream", None).await?;
            for c in chunks {
                sock.write_all(c.as_bytes()).await?;
                sock.flush().await?;
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        }
        Reply::StreamThenStall(chunks) => {
            write_head(&mut sock, 200, "text/event-stream", None).await?;
            for c in chunks {
                sock.write_all(c.as_bytes()).await?;
                sock.flush().await?;
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
            std::future::pending::<()>().await;
        }
        Reply::HeadersThenHang => {
            write_head(&mut sock, 200, "text/event-stream", None).await?;
            let mut tmp = [0u8; 64];
            // read() returns 0 once the client drops the connection.
            loop {
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            client_closed.store(true, Ordering::SeqCst);
            return Ok(());
        }
        Reply::Hang => std::future::pending::<()>().await,
    }
    sock.flush().await?;
    sock.shutdown().await
}

/// One SSE `data:` frame.
pub fn sse(json: &str) -> String {
    format!("data: {json}\n\n")
}
