//! MCP integration tests (port of tests/mcp.test.ts plus timeout / cancel /
//! crash / dispose coverage).
//!
//! `harness = false`: this binary is both the test runner and the mock stdio
//! MCP server. Run with `--mock-mcp-server` it speaks newline-delimited JSON-RPC
//! on stdin/stdout (echo / add / slow / crash tools), so the tests need no
//! network and no Node.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use qo_core::{CancellationToken, ToolContext};
use qo_net::mcp::{create_mcp_tools, McpClient, McpOptions, McpServerSpec};
use serde_json::{json, Value};

/* ── mock server ── */

fn mock_server() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    // Noise a real server might print: a non-JSON log line and a notification.
    let _ = writeln!(out, "mock MCP server starting up");
    let _ = writeln!(out, r#"{{"jsonrpc":"2.0","method":"notifications/message","params":{{"level":"info"}}}}"#);
    let _ = out.flush();
    let mut initialized = false;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(line) else { continue };
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let reply = |result: Value| {
            let mut o = std::io::stdout();
            let _ = writeln!(o, "{}", json!({"jsonrpc": "2.0", "id": id, "result": result}));
            let _ = o.flush();
        };
        match method {
            "initialize" => reply(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                // The pid lets the dispose test check the process is really gone.
                "serverInfo": {"name": "mock", "version": std::process::id().to_string()},
            })),
            "notifications/initialized" => initialized = true,
            "notifications/cancelled" => eprintln!("cancelled {}", msg["params"]["requestId"]),
            "tools/list" if !initialized => {
                let mut o = std::io::stdout();
                let _ = writeln!(o, "{}", json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32002, "message": "not initialized"}}));
                let _ = o.flush();
            }
            "tools/list" => reply(json!({"tools": [
                {"name": "echo", "description": "Echo text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
                {"name": "add", "description": "Add two numbers", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}},
                {"name": "slow", "description": "Sleeps", "inputSchema": {"type": "object", "properties": {}}},
                {"name": "crash", "inputSchema": "not-an-object"},
            ]})),
            "tools/call" => {
                let name = msg["params"]["name"].as_str().unwrap_or("");
                let a = &msg["params"]["arguments"];
                match name {
                    "echo" => reply(json!({"content": [{"type": "text", "text": format!("echoed: {}", a["text"].as_str().unwrap_or(""))}]})),
                    "add" => {
                        let sum = a["a"].as_f64().unwrap_or(0.0) + a["b"].as_f64().unwrap_or(0.0);
                        reply(json!({"content": [{"type": "text", "text": format!("{sum}")}]}))
                    }
                    "slow" => {
                        std::thread::sleep(Duration::from_millis(1500));
                        reply(json!({"content": [{"type": "text", "text": "slept"}]}))
                    }
                    "crash" => {
                        eprintln!("boom: fatal error");
                        std::process::exit(3);
                    }
                    _ => reply(json!({"content": [{"type": "text", "text": "unknown"}], "isError": true})),
                }
            }
            _ => {
                if id.is_some() {
                    let mut o = std::io::stdout();
                    let _ = writeln!(o, "{}", json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}}));
                    let _ = o.flush();
                }
            }
        }
    }
}

/* ── helpers ── */

fn self_exe() -> PathBuf {
    std::env::current_exe().expect("current exe")
}

fn mock_spec(name: &str) -> McpServerSpec {
    McpServerSpec {
        name: name.into(),
        command: self_exe().to_string_lossy().into_owned(),
        args: vec!["--mock-mcp-server".into()],
        ..Default::default()
    }
}

const OPTS: McpOptions = McpOptions { init_timeout_ms: 8000, call_timeout_ms: 8000 };

/// Is a process with this pid still alive?
fn process_alive(pid: u32) -> bool {
    if cfg!(windows) {
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .stdin(std::process::Stdio::null())
            .output();
        out.map(|o| String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\""))).unwrap_or(false)
    } else {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
}

async fn wait_until(mut f: impl FnMut() -> bool, ms: u64) -> bool {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(ms) {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    f()
}

/* ── tests ── */

async fn connects_discovers_and_invokes() {
    let setup = create_mcp_tools(vec![mock_spec("mock")], OPTS).await;
    assert_eq!(setup.infos, ["MCP mock: 4 tools"]);
    assert_eq!(
        setup.specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["mcp__mock__echo", "mcp__mock__add", "mcp__mock__slow", "mcp__mock__crash"]
    );
    // The server's inputSchema is advertised verbatim; a non-object one is replaced.
    assert_eq!(setup.specs[0].parameters["properties"]["text"]["type"], "string");
    assert_eq!(setup.specs[3].parameters, json!({"type": "object", "properties": {}}));
    assert_eq!(setup.specs[0].description, "[MCP:mock] Echo text back");
    assert_eq!(setup.specs[3].description, "[MCP:mock] crash");
    assert_eq!(setup.servers(), ["mock"]);

    let ctx = ToolContext::new(".");
    let tool = |n: &str| setup.tools.iter().find(|t| t.name() == n).cloned().unwrap();
    assert!(tool("mcp__mock__echo").mutating());
    let r = tool("mcp__mock__echo").run(json!({"text": "hi"}), &ctx).await;
    assert_eq!((r.ok, r.output.as_str()), (true, "echoed: hi"));
    let r = tool("mcp__mock__add").run(json!({"a": 2, "b": 40}), &ctx).await;
    assert_eq!((r.ok, r.output.as_str()), (true, "42"));
    setup.dispose_all();
}

async fn degrades_gracefully_when_a_server_fails_to_start() {
    let broken = McpServerSpec { name: "broken".into(), command: "qo_no_such_binary_xyz".into(), ..Default::default() };
    let setup = create_mcp_tools(vec![broken, mock_spec("ok")], McpOptions { init_timeout_ms: 4000, call_timeout_ms: 4000 }).await;
    assert_eq!(setup.tools.len(), 4, "the healthy server still loads");
    assert!(setup.infos[0].starts_with("MCP broken: failed"), "{:?}", setup.infos);
    assert_eq!(setup.infos[1], "MCP ok: 4 tools");
    assert_eq!(setup.servers(), ["ok"]);
    setup.dispose_all();
}

async fn times_out_then_keeps_working() {
    let client = McpClient::new(mock_spec("t"), McpOptions { init_timeout_ms: 8000, call_timeout_ms: 300 });
    client.start().await.unwrap();
    let e = client.call_tool("slow", json!({})).await.unwrap_err().to_string();
    assert_eq!(e, "\"tools/call\" timed out after 300ms");
    // The late reply to the abandoned request is ignored; new calls still work.
    tokio::time::sleep(Duration::from_millis(1400)).await;
    let r = client.call_tool("echo", json!({"text": "after"})).await.unwrap();
    assert_eq!(r.text, "echoed: after");
    client.dispose();
}

async fn esc_interrupts_a_call() {
    let setup = create_mcp_tools(vec![mock_spec("c")], OPTS).await;
    let slow = setup.tools.iter().find(|t| t.name() == "mcp__c__slow").cloned().unwrap();
    let echo = setup.tools.iter().find(|t| t.name() == "mcp__c__echo").cloned().unwrap();
    let token = CancellationToken::new();
    let mut ctx = ToolContext::new(".");
    ctx.cancel = Some(token.clone());
    let t2 = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        t2.cancel();
    });
    let start = Instant::now();
    let r = slow.run(json!({}), &ctx).await;
    assert_eq!((r.ok, r.output.as_str()), (false, "interrupted"));
    assert!(start.elapsed() < Duration::from_millis(1200), "returned promptly");
    // The server survives an interrupt.
    let r = echo.run(json!({"text": "still here"}), &ToolContext::new(".")).await;
    assert_eq!(r.output, "echoed: still here");
    setup.dispose_all();
}

async fn crash_is_reported_with_stderr_and_fails_fast_after() {
    let setup = create_mcp_tools(vec![mock_spec("x")], OPTS).await;
    let tool = |n: &str| setup.tools.iter().find(|t| t.name() == n).cloned().unwrap();
    let ctx = ToolContext::new(".");
    let r = tool("mcp__x__crash").run(json!({}), &ctx).await;
    assert!(!r.ok);
    assert!(r.output.starts_with("MCP x/crash failed: server exited (code 3)"), "{}", r.output);
    assert!(r.output.contains("boom: fatal error"), "{}", r.output);
    let start = Instant::now();
    let r = tool("mcp__x__echo").run(json!({"text": "?"}), &ctx).await;
    assert!(r.output.contains("server exited (code 3)"), "{}", r.output);
    assert!(start.elapsed() < Duration::from_millis(500), "no wait for a timeout");
    assert!(!setup.clients[0].running());
    setup.dispose_all();
}

async fn dispose_kills_the_server() {
    let client = Arc::new(McpClient::new(mock_spec("d"), OPTS));
    client.start().await.unwrap();
    assert!(client.initialized() && client.running());
    let pid = client.pid().unwrap();
    assert!(process_alive(pid));
    client.dispose();
    let c2 = client.clone();
    assert!(wait_until(move || !c2.running(), 5000).await, "waiter saw the exit");
    assert!(wait_until(|| !process_alive(pid), 5000).await, "pid {pid} still alive");
    let e = client.call_tool("echo", json!({"text": "x"})).await.unwrap_err().to_string();
    assert_eq!(e, "client closed");
}

/// npx/uvx on Windows are `.cmd` shims: the bare command name must resolve
/// through PATHEXT, and dispose must take down the real server under cmd.exe.
#[cfg(windows)]
async fn spawns_cmd_shims_and_kills_the_tree() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("qo-mock-mcp.cmd"),
        format!("@echo off\r\n\"{}\" --mock-mcp-server\r\n", self_exe().display()),
    )
    .unwrap();
    let path = format!("{};{}", dir.path().display(), std::env::var("PATH").unwrap_or_default());
    let spec = McpServerSpec {
        name: "shim".into(),
        command: "qo-mock-mcp".into(),
        env: [("PATH".to_string(), path)].into_iter().collect(),
        ..Default::default()
    };
    let client = McpClient::new(spec, OPTS);
    client.start().await.unwrap();
    let r = client.call_tool("echo", json!({"text": "via cmd"})).await.unwrap();
    assert_eq!(r.text, "echoed: via cmd");
    assert_eq!(client.list_tools().await.unwrap().len(), 4);
    // The real server is a grandchild under cmd.exe; it reports its own pid.
    let info = client.server_info().unwrap();
    let server_pid: u32 = info["serverInfo"]["version"].as_str().unwrap().parse().unwrap();
    let wrapper = client.pid().unwrap();
    assert_ne!(server_pid, wrapper, "launched through the cmd.exe wrapper");
    assert!(process_alive(server_pid));
    client.dispose();
    assert!(wait_until(|| !process_alive(wrapper), 5000).await, "wrapper {wrapper} still alive");
    assert!(wait_until(|| !process_alive(server_pid), 5000).await, "server {server_pid} orphaned");
}

#[cfg(not(windows))]
async fn spawns_cmd_shims_and_kills_the_tree() {}

/* ── runner ── */

type TestFn = fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>;

macro_rules! tests {
    ($($name:ident),* $(,)?) => {
        vec![$((stringify!($name), (|| Box::pin($name()) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>) as TestFn)),*]
    };
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--mock-mcp-server") {
        mock_server();
        return;
    }
    if args.iter().any(|a| a == "--list") {
        return;
    }
    let filter: Option<&String> = args.iter().skip(1).find(|a| !a.starts_with('-'));
    let all = tests![
        connects_discovers_and_invokes,
        degrades_gracefully_when_a_server_fails_to_start,
        times_out_then_keeps_working,
        esc_interrupts_a_call,
        crash_is_reported_with_stderr_and_fails_fast_after,
        dispose_kills_the_server,
        spawns_cmd_shims_and_kills_the_tree,
    ];
    let selected: Vec<_> = all.into_iter().filter(|(n, _)| filter.map_or(true, |f| n.contains(f.as_str()))).collect();
    println!("\nrunning {} tests", selected.len());
    let mut failed = Vec::new();
    for (name, f) in &selected {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("runtime");
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.block_on(f())));
        rt.shutdown_timeout(Duration::from_secs(5));
        match outcome {
            Ok(()) => println!("test {name} ... ok"),
            Err(_) => {
                println!("test {name} ... FAILED");
                failed.push(*name);
            }
        }
    }
    if failed.is_empty() {
        println!("\ntest result: ok. {} passed; 0 failed\n", selected.len());
    } else {
        println!("\ntest result: FAILED. {} passed; {} failed: {failed:?}\n", selected.len() - failed.len(), failed.len());
        std::process::exit(101);
    }
}
