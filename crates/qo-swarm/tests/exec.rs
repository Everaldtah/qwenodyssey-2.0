//! Port of tests/agentExec.test.ts (executors + routing + the agent tool loop),
//! with a local mock server standing in for Daytona.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{env_with, worker, Fake, FakeExec, Handler, MockServer};
use qo_core::{GenerateResult, Role, ToolCall};
use qo_swarm::{
    choose_executor, daytona_available, parse_complexity, AgentExecutor, BareExecutor, BareExecutorOptions,
    CoordinateOptions, CoordinatedSwarm, DaytonaExecutor, DaytonaSettings, ExecKind, ExecMode, PlanComplexity,
    RunOptions, Subtask, SwarmEvent,
};
use serde_json::{json, Value};

const UNSET_ENV: &str = "QO_SWARM_TEST_UNSET_DAYTONA_KEY";

fn no_key() -> DaytonaSettings {
    DaytonaSettings { api_key: String::new(), api_key_env: UNSET_ENV.into(), ..Default::default() }
}

#[test]
fn parse_complexity_reads_the_planners_judgment_and_defaults_to_simple() {
    assert_eq!(parse_complexity(r#"{"complexity":"complex","subtasks":[]}"#), PlanComplexity::Complex);
    assert_eq!(parse_complexity(r#"{"complexity":"simple","subtasks":[]}"#), PlanComplexity::Simple);
    assert_eq!(parse_complexity(r#"{ "Complexity" : "COMPLEX" }"#), PlanComplexity::Complex);
    assert_eq!(parse_complexity(r#"{"subtasks":[]}"#), PlanComplexity::Simple);
}

#[test]
fn exec_mode_parses_config_spellings() {
    assert_eq!(ExecMode::parse("Daytona"), Some(ExecMode::Daytona));
    assert_eq!(ExecMode::parse(" auto "), Some(ExecMode::Auto));
    assert_eq!(ExecMode::parse("sandbox"), None);
}

#[test]
fn routing_off_none_bare_bare_auto_simple_bare() {
    let (env, _) = env_with(|e| e.daytona = no_key());
    let kind = |m, c| choose_executor(&env, m, c, BareExecutorOptions::default()).executor.map(|e| e.kind());
    assert_eq!(kind(ExecMode::Off, PlanComplexity::Complex), None);
    assert_eq!(kind(ExecMode::Bare, PlanComplexity::Complex), Some(ExecKind::Bare));
    assert_eq!(kind(ExecMode::Auto, PlanComplexity::Simple), Some(ExecKind::Bare));
}

#[test]
fn routing_auto_complex_uses_daytona_with_a_key_and_bare_with_a_note_without() {
    let (env, _) = env_with(|e| e.daytona = no_key());
    assert!(!daytona_available(&env.daytona));
    let no = choose_executor(&env, ExecMode::Auto, PlanComplexity::Complex, BareExecutorOptions::default());
    assert_eq!(no.executor.as_ref().map(|e| e.kind()), Some(ExecKind::Bare));
    assert!(no.note.as_deref().unwrap_or("").contains("daytona not configured"));
    let forced = choose_executor(&env, ExecMode::Daytona, PlanComplexity::Simple, BareExecutorOptions::default());
    assert_eq!(forced.executor.as_ref().map(|e| e.kind()), Some(ExecKind::Bare));
    assert!(forced.note.as_deref().unwrap_or("").contains("no DAYTONA_API_KEY"));

    let (env, _) = env_with(|e| e.daytona = DaytonaSettings { api_key: "dtn_test".into(), ..no_key() });
    assert!(daytona_available(&env.daytona));
    let kind = |m, c| choose_executor(&env, m, c, BareExecutorOptions::default()).executor.map(|e| e.kind());
    assert_eq!(kind(ExecMode::Auto, PlanComplexity::Complex), Some(ExecKind::Daytona));
    // forced daytona honors the key too
    assert_eq!(kind(ExecMode::Daytona, PlanComplexity::Simple), Some(ExecKind::Daytona));

    // disabled = unavailable even with a key
    let (env, _) = env_with(|e| e.daytona = DaytonaSettings { api_key: "dtn_test".into(), enabled: false, ..no_key() });
    assert!(!daytona_available(&env.daytona));
}

/* ── bare metal ── */

#[tokio::test]
async fn bare_refuses_hard_blocked_and_destructive_commands_without_running_them() {
    let ex = BareExecutor::default();
    let blocked = ex.run("Format-Volume -DriveLetter C", RunOptions::default()).await;
    assert!(!blocked.ok);
    assert!(blocked.output.contains("hard-blocked"));

    let destructive = ex.run("Remove-Item -Recurse some/dir", RunOptions::default()).await;
    assert!(!destructive.ok);
    assert!(destructive.output.contains("destructive"));

    // deny list wins; allow list lets a destructive command through the gate.
    let deny = BareExecutor::new(BareExecutorOptions { deny_commands: vec!["^echo".into()], ..Default::default() });
    assert!(deny.run("echo hi", RunOptions::default()).await.output.contains("hard-blocked"));
}

fn echo_42() -> &'static str {
    if cfg!(windows) {
        "Write-Output (40+2)"
    } else {
        "echo $((40+2))"
    }
}

#[tokio::test]
async fn bare_runs_a_safe_command_and_captures_output_and_exit_code() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ex = BareExecutor::new(BareExecutorOptions { cwd: Some(dir.path().to_path_buf()), ..Default::default() });
    let r = ex.run(echo_42(), RunOptions { timeout_ms: Some(60_000), ..Default::default() }).await;
    assert!(r.ok, "{r:?}");
    assert_eq!(r.exit_code, Some(0));
    assert!(r.output.contains("42"));

    let fail = ex.run("exit 3", RunOptions { timeout_ms: Some(60_000), ..Default::default() }).await;
    assert!(!fail.ok);
    assert_eq!(fail.exit_code, Some(3));
}

#[tokio::test]
async fn bare_kills_a_command_on_timeout_and_on_cancel() {
    let ex = BareExecutor::default();
    let sleep = if cfg!(windows) { "Start-Sleep -Seconds 30" } else { "sleep 30" };

    let t0 = Instant::now();
    let r = ex.run(sleep, RunOptions { timeout_ms: Some(1500), ..Default::default() }).await;
    assert!(!r.ok);
    assert_eq!(r.exit_code, None);
    assert!(r.output.contains("timed out"), "{r:?}");
    assert!(t0.elapsed() < Duration::from_secs(20));

    let cancel = qo_core::CancellationToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        c2.cancel();
    });
    let t0 = Instant::now();
    let r = ex.run(sleep, RunOptions { cancel: Some(cancel), timeout_ms: Some(60_000), ..Default::default() }).await;
    assert!(!r.ok);
    assert!(r.output.contains("interrupted"), "{r:?}");
    assert!(t0.elapsed() < Duration::from_secs(20));
}

#[tokio::test]
async fn bare_artifacts_lists_files_created_during_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
    let ex = BareExecutor::new(BareExecutorOptions { cwd: Some(dir.path().to_path_buf()), ..Default::default() });
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src").join("main.py"), "print(1)").unwrap();
    std::fs::write(dir.path().join("node_modules").join("skip.js"), "x").unwrap();
    std::fs::write(dir.path().join(".hidden"), "x").unwrap();
    let a = ex.artifacts().await;
    assert!(a.location.starts_with("bare metal: "));
    assert_eq!(a.files.len(), 1, "{a:?}");
    assert!(a.files[0].ends_with("main.py"));
}

/* ── daytona (mocked REST) ── */

fn daytona_handler(creates: Arc<AtomicUsize>) -> Handler {
    Arc::new(move |req| {
        let ok = |v: Value| (200u16, v.to_string());
        match (req.method.as_str(), req.path.as_str()) {
            ("POST", "/sandbox") => {
                creates.fetch_add(1, Ordering::SeqCst);
                ok(json!({ "id": "sb-1", "state": "creating" }))
            }
            ("GET", "/sandbox/sb-1") => ok(json!({ "id": "sb-1", "state": "started" })),
            ("POST", "/toolbox/sb-1/toolbox/process/execute") => {
                let body: Value = serde_json::from_str(&req.body).unwrap_or(Value::Null);
                let cmd = body["command"].as_str().unwrap_or("");
                if cmd.starts_with("find ") {
                    return ok(json!({ "exitCode": 0, "result": "./app.py\n/root/out.txt\n" }));
                }
                ok(json!({ "exitCode": 0, "result": format!("ran: {cmd}") }))
            }
            ("DELETE", "/sandbox/sb-1") => ok(json!({})),
            _ => (404, "not found".into()),
        }
    })
}

fn settings_for(server: &MockServer) -> DaytonaSettings {
    DaytonaSettings { base_url: format!("{}/", server.url), api_key: "dtn_test".into(), ..no_key() }
}

fn local_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().expect("client")
}

#[tokio::test]
async fn daytona_creates_one_sandbox_executes_commands_and_deletes_it_on_dispose() {
    let creates = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start(daytona_handler(Arc::clone(&creates))).await;
    let ex = Arc::new(
        DaytonaExecutor::with_client(settings_for(&server), local_client()).with_poll_interval(Duration::from_millis(10)),
    );

    // Concurrent first commands share ONE sandbox creation.
    let (r1, r2) = tokio::join!(ex.run("echo one", RunOptions::default()), ex.run("echo two", RunOptions::default()));
    assert!(r1.ok, "{r1:?}");
    assert_eq!(r1.output, "ran: echo one");
    assert_eq!(r2.output, "ran: echo two");
    assert_eq!(creates.load(Ordering::SeqCst), 1, "exactly ONE sandbox creation despite two commands");

    let a = ex.artifacts().await;
    assert_eq!(a.location, "daytona sandbox sb-1");
    assert_eq!(a.files, vec!["./app.py", "/root/out.txt"]);

    ex.dispose().await;
    ex.dispose().await; // idempotent
    let reqs = server.requests();
    assert_eq!(reqs.iter().filter(|r| r.method == "DELETE" && r.path == "/sandbox/sb-1").count(), 1);
    let after = ex.run("echo three", RunOptions::default()).await;
    assert!(!after.ok);
    assert!(after.output.contains("disposed"));

    // Wire format: bearer auth, sandbox labels/lifecycle, sentinel, per-command timeout in seconds.
    assert!(reqs.iter().all(|r| r.authorization.as_deref() == Some("Bearer dtn_test")));
    let create: Value = serde_json::from_str(&reqs.iter().find(|r| r.path == "/sandbox").unwrap().body).unwrap();
    assert_eq!(create["labels"]["app"], "qwenodyssey-swarm");
    assert_eq!(create["autoStopInterval"], 15);
    assert_eq!(create["autoDeleteInterval"], 120);
    assert!(create["name"].as_str().unwrap_or("").starts_with("qod-swarm-"));
    let execs: Vec<Value> = reqs
        .iter()
        .filter(|r| r.path.ends_with("/process/execute"))
        .map(|r| serde_json::from_str(&r.body).unwrap())
        .collect();
    assert_eq!(execs[0]["command"], "touch /tmp/.qod_swarm_start");
    assert!(execs.iter().any(|b| b["command"] == "echo one" && b["timeout"] == 120));
}

#[tokio::test]
async fn daytona_surfaces_http_errors_as_failed_results_not_panics() {
    let server = MockServer::start(Arc::new(|_| (401, "nope".into()))).await;
    let ex = DaytonaExecutor::with_client(settings_for(&server), local_client());
    let r = ex.run("echo hi", RunOptions::default()).await;
    assert!(!r.ok);
    assert!(r.output.contains("HTTP 401"), "{r:?}");
    // Nothing was created, so artifacts/dispose are no-ops.
    assert!(ex.artifacts().await.files.is_empty());
    ex.dispose().await;
}

#[tokio::test]
async fn daytona_reports_a_sandbox_that_fails_to_start_and_retries_next_time() {
    let creates = Arc::new(AtomicUsize::new(0));
    let c2 = Arc::clone(&creates);
    let server = MockServer::start(Arc::new(move |req| match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/sandbox") => {
            c2.fetch_add(1, Ordering::SeqCst);
            (200, json!({ "id": "sb-9", "state": "creating" }).to_string())
        }
        ("GET", "/sandbox/sb-9") => (200, json!({ "state": "error", "errorReason": "no capacity" }).to_string()),
        _ => (404, "x".into()),
    }))
    .await;
    let ex = DaytonaExecutor::with_client(settings_for(&server), local_client()).with_poll_interval(Duration::from_millis(5));
    let r = ex.run("echo", RunOptions::default()).await;
    assert!(r.output.contains("failed to start: no capacity"), "{r:?}");
    let _ = ex.run("echo", RunOptions::default()).await;
    assert_eq!(creates.load(Ordering::SeqCst), 2, "a failed creation is retryable");
}

#[tokio::test]
async fn daytona_requests_are_cancellable() {
    let server = MockServer::start(Arc::new(|req| {
        if req.path == "/sandbox" {
            (200, json!({ "id": "sb-2", "state": "creating" }).to_string())
        } else {
            (200, json!({ "id": "sb-2", "state": "creating" }).to_string()) // never starts
        }
    }))
    .await;
    let ex = DaytonaExecutor::with_client(settings_for(&server), local_client()).with_poll_interval(Duration::from_secs(5));
    let cancel = qo_core::CancellationToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        c2.cancel();
    });
    let t0 = Instant::now();
    let r = ex.run("echo", RunOptions { cancel: Some(cancel), ..Default::default() }).await;
    assert!(!r.ok);
    assert!(r.output.contains("interrupted"), "{r:?}");
    assert!(t0.elapsed() < Duration::from_secs(3));
}

/* ── the agent tool loop (executor wired into the swarm) ── */

fn subtask_a() -> Vec<Subtask> {
    vec![Subtask { id: "a".into(), title: "A".into(), detail: "da".into(), depends_on: vec![] }]
}

#[tokio::test]
async fn lets_an_agent_call_run_shell_see_the_output_and_finish_with_text() {
    let w = Fake::new("w1")
        .on_stream_full(|messages, opts, idx| {
            if idx == 0 {
                // First turn: the agent decides to run a command (tools are offered).
                assert_eq!(opts.tools.as_ref().map(|t| t[0].name.as_str()), Some("run_shell"));
                return Ok(GenerateResult {
                    tool_calls: vec![ToolCall { id: "c1".into(), name: "run_shell".into(), arguments: json!({ "command": "node --version" }) }],
                    completion_tokens: Some(5),
                    ..Default::default()
                });
            }
            // Second turn: the tool result must be visible in the conversation.
            let tool_msg = messages.iter().find(|m| m.role == Role::Tool);
            let saw = tool_msg.is_some_and(|m| m.content.contains("OUT(node --version)") && m.content.starts_with("exit=0 ok=true"));
            Ok(GenerateResult {
                text: if saw { "verified via shell" } else { "never saw tool output" }.into(),
                completion_tokens: Some(7),
                ..Default::default()
            })
        })
        .shared();
    let (env, _) = env_with(|_| {});
    let swarm = CoordinatedSwarm::new(env, vec![worker(w)], Fake::new("lead").shared(), Some(200));
    let mut rx = swarm.subscribe();
    let exec = Arc::new(FakeExec::default());
    let run = swarm
        .run(
            "t",
            CoordinateOptions {
                synthesize: false,
                executor: Some(exec.clone()),
                subtasks: Some(subtask_a()),
                ..Default::default()
            },
        )
        .await;
    let cmds = exec.log();
    assert!(cmds.contains(&"node --version".to_string()));
    assert_eq!(cmds.last().map(String::as_str), Some("__disposed__"), "sandbox torn down after the run");
    assert!(run.results[0].ok);
    assert!(run.results[0].text.contains("verified via shell"));
    assert_eq!(run.results[0].completion_tokens, Some(12), "usage summed across steps");
    assert_eq!(run.artifacts.as_ref().map(|a| a.files.clone()), Some(vec!["a.py".to_string()]));

    let mut saw_exec_cmd = false;
    let mut pane_text = String::new();
    let mut plan_exec = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            SwarmEvent::ExecCmd(e) => saw_exec_cmd = e.command == "node --version" && e.ok,
            SwarmEvent::Delta(d) => pane_text.push_str(&d.delta),
            SwarmEvent::Plan(p) => plan_exec = p.exec,
            _ => {}
        }
    }
    assert!(saw_exec_cmd);
    assert!(pane_text.contains("$ node --version"), "commands are shown in the pane: {pane_text}");
    assert_eq!(plan_exec.as_deref(), Some("fake exec"));
}

#[tokio::test]
async fn kimi_special_token_calls_run_and_the_last_step_forces_a_text_answer() {
    let w = Fake::new("w1")
        .on_stream_full(|_m, opts, _idx| {
            if opts.tools.is_some() {
                // Keeps asking for a command via special tokens while tools are offered.
                return Ok(GenerateResult {
                    text: "<|tool_calls_section_begin|><|tool_call_begin|>functions.run_shell:1\
                           <|tool_call_argument_begin|>{\"command\":\"ls\"}<|tool_call_end|><|tool_calls_section_end|>"
                        .into(),
                    ..Default::default()
                });
            }
            Ok(GenerateResult { text: "final words".into(), ..Default::default() })
        })
        .shared();
    let (env, _) = env_with(|e| e.swarm.exec_max_steps = 2);
    let swarm = CoordinatedSwarm::new(env, vec![worker(w)], Fake::new("lead").shared(), Some(200));
    let exec = Arc::new(FakeExec::default());
    let run = swarm
        .run("t", CoordinateOptions { synthesize: false, executor: Some(exec.clone()), subtasks: Some(subtask_a()), ..Default::default() })
        .await;
    assert_eq!(exec.log().iter().filter(|c| *c == "ls").count(), 2, "one command per tool step");
    assert_eq!(run.results[0].text, "final words");
}

#[tokio::test]
async fn exec_mode_routes_through_choose_executor_and_notes_the_fallback() {
    let w = Fake::new("w1").on_stream(|_| Ok("text only".into())).shared();
    let (env, _) = env_with(|e| e.daytona = no_key());
    let swarm = CoordinatedSwarm::new(env, vec![worker(w)], Fake::new("lead").shared(), Some(200));
    let mut rx = swarm.subscribe();
    let dir = tempfile::tempdir().expect("tempdir");
    let run = swarm
        .run(
            "t",
            CoordinateOptions {
                synthesize: false,
                exec_mode: ExecMode::Daytona,
                bare: BareExecutorOptions { cwd: Some(dir.path().to_path_buf()), ..Default::default() },
                subtasks: Some(subtask_a()),
                ..Default::default()
            },
        )
        .await;
    assert!(run.results[0].ok);
    assert!(run.artifacts.as_ref().is_some_and(|a| a.location.starts_with("bare metal")));
    let mut plan = None;
    while let Ok(ev) = rx.try_recv() {
        if let SwarmEvent::Plan(p) = ev {
            plan = Some(p);
        }
    }
    let plan = plan.expect("plan");
    assert_eq!(plan.exec.as_deref(), Some("bare metal"));
    assert!(plan.note.as_deref().unwrap_or("").contains("no DAYTONA_API_KEY"));
}
