//! Test doubles: canned providers, a recording factory, a fake executor and a
//! tiny local HTTP server (never real models, never real Daytona).
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use qo_core::{GenerateResult, HealthStatus, Message, ModelOptions, Provider, ProviderError, Role, SharedProvider};
use qo_swarm::{
    AgentExecutor, ExecKind, ExecResult, ProviderFactory, RunOptions, SwarmArtifacts, SwarmEnv, SwarmWorker, WorkerKind,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub type GenFn = Box<dyn Fn(&[Message]) -> anyhow::Result<String> + Send + Sync>;
pub type StreamFn = Box<dyn Fn(&[Message], &ModelOptions, usize) -> anyhow::Result<GenerateResult> + Send + Sync>;

/// A provider driven by closures. `hang` makes the FIRST `stream` call emit one
/// chunk and then block until its cancel token fires (for stop/abort tests);
/// later calls use the normal stream closure.
pub struct Fake {
    pub model: String,
    pub generate: Option<GenFn>,
    pub stream: Option<StreamFn>,
    pub health: HealthStatus,
    pub hang: bool,
    pub delay: Option<Duration>,
    pub calls: AtomicUsize,
}

impl Fake {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            generate: None,
            stream: None,
            health: HealthStatus { ok: true, detail: None },
            hang: false,
            delay: None,
            calls: AtomicUsize::new(0),
        }
    }

    pub fn on_generate(mut self, f: impl Fn(&[Message]) -> anyhow::Result<String> + Send + Sync + 'static) -> Self {
        self.generate = Some(Box::new(f));
        self
    }

    /// Text-only stream reply.
    pub fn on_stream(mut self, f: impl Fn(&[Message]) -> anyhow::Result<String> + Send + Sync + 'static) -> Self {
        self.stream = Some(Box::new(move |m, _o, _i| {
            f(m).map(|text| GenerateResult { text, ..Default::default() })
        }));
        self
    }

    /// Full control over the stream result (tool calls, usage), with the call index.
    pub fn on_stream_full(
        mut self,
        f: impl Fn(&[Message], &ModelOptions, usize) -> anyhow::Result<GenerateResult> + Send + Sync + 'static,
    ) -> Self {
        self.stream = Some(Box::new(f));
        self
    }

    pub fn health(mut self, ok: bool, detail: Option<&str>) -> Self {
        self.health = HealthStatus { ok, detail: detail.map(str::to_string) };
        self
    }

    pub fn hang(mut self) -> Self {
        self.hang = true;
        self
    }

    pub fn delay(mut self, d: Duration) -> Self {
        self.delay = Some(d);
        self
    }

    pub fn shared(self) -> SharedProvider {
        Arc::new(self)
    }

    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

async fn wait_or_cancel(d: Option<Duration>, opts: &ModelOptions) -> anyhow::Result<()> {
    if let Some(d) = d {
        match &opts.cancel {
            Some(c) => tokio::select! {
                _ = tokio::time::sleep(d) => {}
                _ = c.cancelled() => return Err(ProviderError::Interrupted.into()),
            },
            None => tokio::time::sleep(d).await,
        }
    }
    Ok(())
}

#[async_trait]
impl Provider for Fake {
    fn name(&self) -> &str {
        "fake"
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn set_model(&self, _model: &str) {}

    async fn generate(&self, messages: &[Message], opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        wait_or_cancel(self.delay, opts).await?;
        let text = match &self.generate {
            Some(f) => f(messages)?,
            None => String::new(),
        };
        Ok(GenerateResult { text, model: self.model.clone(), ..Default::default() })
    }

    async fn stream(
        &self,
        messages: &[Message],
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
        opts: &ModelOptions,
    ) -> anyhow::Result<GenerateResult> {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.hang && idx == 0 {
            on_chunk("partial work ");
            match &opts.cancel {
                Some(c) => c.cancelled().await,
                None => std::future::pending::<()>().await,
            }
            return Err(ProviderError::Interrupted.into());
        }
        wait_or_cancel(self.delay, opts).await?;
        let mut res = match &self.stream {
            Some(f) => f(messages, opts, idx)?,
            None => GenerateResult::default(),
        };
        if !res.text.is_empty() {
            on_chunk(&res.text.clone()); // single-chunk stream is enough to exercise the path
        }
        res.model = self.model.clone();
        Ok(res)
    }

    async fn health_check(&self) -> HealthStatus {
        self.health.clone()
    }
}

pub fn worker(p: SharedProvider) -> SwarmWorker {
    let model = p.model();
    SwarmWorker { model_ref: model.clone(), kind: WorkerKind::Nvidia, label: model.clone(), model, provider: p }
}

pub fn user_of(msgs: &[Message]) -> String {
    msgs.iter().find(|m| m.role == Role::User).map(|m| m.content.clone()).unwrap_or_default()
}

/// Hands out `Fake` providers (or registered ones) and records every creation.
#[derive(Default)]
pub struct FakeFactory {
    pub created: Mutex<Vec<(WorkerKind, String)>>,
    pub registered: Mutex<HashMap<String, SharedProvider>>,
    pub primary_model: String,
}

impl FakeFactory {
    pub fn new(primary_model: &str) -> Self {
        Self { primary_model: primary_model.to_string(), ..Default::default() }
    }

    /// Serve `p` whenever `model` is requested.
    pub fn register(&self, p: SharedProvider) {
        self.registered.lock().unwrap().insert(p.model(), p);
    }

    pub fn created(&self) -> Vec<(WorkerKind, String)> {
        self.created.lock().unwrap().clone()
    }

    fn get(&self, model: &str) -> SharedProvider {
        if let Some(p) = self.registered.lock().unwrap().get(model) {
            return Arc::clone(p);
        }
        Fake::new(model).on_generate(|_| Ok("generated".into())).on_stream(|_| Ok("streamed".into())).shared()
    }
}

impl ProviderFactory for FakeFactory {
    fn primary(&self) -> SharedProvider {
        self.get(&self.primary_model)
    }

    fn create(&self, kind: WorkerKind, model: &str) -> SharedProvider {
        self.created.lock().unwrap().push((kind, model.to_string()));
        self.get(model)
    }
}

/// An environment around a fresh `FakeFactory` (returned for inspection).
pub fn env_with(f: impl FnOnce(&mut SwarmEnv)) -> (Arc<SwarmEnv>, Arc<FakeFactory>) {
    let factory = Arc::new(FakeFactory::new("primary-model"));
    let mut env = SwarmEnv::with_factory(factory.clone());
    env.primary_model = "primary-model".into();
    f(&mut env);
    (Arc::new(env), factory)
}

/// Records commands; every command succeeds with `OUT(<cmd>)`.
#[derive(Default)]
pub struct FakeExec {
    pub log: Mutex<Vec<String>>,
}

impl FakeExec {
    pub fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentExecutor for FakeExec {
    fn kind(&self) -> ExecKind {
        ExecKind::Bare
    }
    fn label(&self) -> &str {
        "fake exec"
    }
    async fn run(&self, command: &str, _opts: RunOptions) -> ExecResult {
        self.log.lock().unwrap().push(command.to_string());
        ExecResult { ok: true, exit_code: Some(0), output: format!("OUT({command})") }
    }
    async fn artifacts(&self) -> SwarmArtifacts {
        SwarmArtifacts { location: "fake".into(), files: vec!["a.py".into()] }
    }
    async fn dispose(&self) {
        self.log.lock().unwrap().push("__disposed__".into());
    }
}

/// One recorded HTTP request.
#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub body: String,
    pub authorization: Option<String>,
}

pub type Handler = Arc<dyn Fn(&Req) -> (u16, String) + Send + Sync>;

/// A minimal HTTP/1.1 server on 127.0.0.1 (one request per connection).
pub struct MockServer {
    pub url: String,
    pub log: Arc<Mutex<Vec<Req>>>,
}

impl MockServer {
    pub async fn start(handler: Handler) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
        let addr = listener.local_addr().expect("local addr");
        let log: Arc<Mutex<Vec<Req>>> = Arc::new(Mutex::new(Vec::new()));
        let log2 = Arc::clone(&log);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { break };
                let handler = Arc::clone(&handler);
                let log = Arc::clone(&log2);
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let head_end = loop {
                        let Ok(n) = sock.read(&mut chunk).await else { return };
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break i + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let mut lines = head.lines();
                    let first = lines.next().unwrap_or_default().to_string();
                    let mut parts = first.split_whitespace();
                    let method = parts.next().unwrap_or_default().to_string();
                    let path = parts.next().unwrap_or_default().to_string();
                    let mut len = 0usize;
                    let mut authorization = None;
                    for l in lines {
                        let lower = l.to_ascii_lowercase();
                        if let Some(v) = lower.strip_prefix("content-length:") {
                            len = v.trim().parse().unwrap_or(0);
                        }
                        if lower.starts_with("authorization:") {
                            authorization = Some(l["authorization:".len()..].trim().to_string());
                        }
                    }
                    while buf.len() < head_end + len {
                        let Ok(n) = sock.read(&mut chunk).await else { return };
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[head_end..(head_end + len).min(buf.len())]).to_string();
                    let req = Req { method, path, body, authorization };
                    log.lock().unwrap().push(req.clone());
                    let (status, resp) = handler(&req);
                    let out = format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp}",
                        resp.len()
                    );
                    let _ = sock.write_all(out.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        MockServer { url: format!("http://{addr}"), log }
    }

    pub fn requests(&self) -> Vec<Req> {
        self.log.lock().unwrap().clone()
    }
}
