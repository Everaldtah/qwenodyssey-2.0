//! The `agent_swarm` tool (tools/swarmTools.ts) against fake providers.

mod common;

use std::sync::{Arc, Mutex};

use common::{env_with, user_of, Fake, FakeFactory};
use qo_core::{Tool, ToolContext};
use qo_swarm::{
    create_swarm_tools_with_env, lead_provider, swarm_ready_roster, swarm_tool_specs, AgentSwarmTool, SwarmEnv, WorkerKind,
};
use serde_json::{json, Value};

fn ctx_with_log() -> (ToolContext, Arc<Mutex<Vec<Value>>>) {
    let log: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let l2 = Arc::clone(&log);
    let mut ctx = ToolContext::new(std::env::temp_dir());
    ctx.log = Arc::new(move |v| l2.lock().unwrap().push(v));
    (ctx, log)
}

/// Two cloud workers + a lead, all canned.
fn cloud_env(configure: impl FnOnce(&mut SwarmEnv)) -> (Arc<SwarmEnv>, Arc<FakeFactory>) {
    let (env, factory) = env_with(|e| {
        e.have_nvidia = true;
        e.have_openrouter = true;
        e.fallback_models = vec!["nvidia:org/alpha".into(), "openrouter:moonshotai/kimi-k2.6".into()];
        configure(e);
    });
    let plan = r#"{"subtasks":[{"id":"one","title":"One","detail":"part one"},{"id":"two","title":"Two","detail":"part two","dependsOn":["one"]}]}"#;
    for m in ["org/alpha", "moonshotai/kimi-k2.6"] {
        let name = m.to_string();
        factory.register(
            Fake::new(m)
                .on_generate(move |msgs| {
                    let u = user_of(msgs);
                    if u.contains("Decompose this task") {
                        return Ok(plan.to_string());
                    }
                    if u.contains("Combine their answers") {
                        return Ok("ENSEMBLE-MERGED".into());
                    }
                    Ok(format!("answer from {name}"))
                })
                .on_stream(|msgs| {
                    let u = user_of(msgs);
                    if u.contains("Write the integrated Markdown report") {
                        return Ok("INTEGRATED REPORT".into());
                    }
                    Ok(format!("did {}", if u.contains("YOUR SUBTASK [one]") { "one" } else { "two" }))
                })
                .shared(),
        );
    }
    factory.register(Fake::new("primary-model").on_generate(|_| Ok("PRIMARY-MERGED".into())).shared());
    (env, factory)
}

#[test]
fn specs_advertise_the_documented_schema() {
    let specs = swarm_tool_specs();
    assert_eq!(specs.len(), 1);
    let s = &specs[0];
    assert_eq!(s.name, "agent_swarm");
    assert_eq!(s.parameters["required"], json!(["task"]));
    assert_eq!(s.parameters["properties"]["mode"]["enum"], json!(["coordinate", "ensemble", "divide"]));
    assert_eq!(s.parameters["properties"]["subtasks"]["items"]["type"], "string");
    let (env, _) = env_with(|_| {});
    let tools = create_swarm_tools_with_env(env);
    assert_eq!(tools[0].name(), "agent_swarm");
    assert!(!tools[0].mutating());
}

#[tokio::test]
async fn requires_a_task() {
    let (env, _) = env_with(|_| {});
    let (ctx, _) = ctx_with_log();
    let r = AgentSwarmTool::new(env).run(json!({ "task": "   " }), &ctx).await;
    assert!(!r.ok);
    assert!(r.output.contains("'task' is required"));
}

#[tokio::test]
async fn coordinate_mode_plans_shares_and_integrates() {
    let (env, _) = cloud_env(|_| {});
    let (ctx, log) = ctx_with_log();
    let r = AgentSwarmTool::new(env).run(json!({ "task": "build it" }), &ctx).await;
    assert!(r.ok, "{}", r.output);
    assert!(r.output.starts_with("🜂 coordinated swarm — 2 subtask(s), 2 ok"), "{}", r.output);
    assert!(r.output.contains("[two] Two — "));
    assert!(r.output.contains("(after one)"));
    assert!(r.output.contains("=== integrated answer (by moonshotai/kimi-k2.6) ===\nINTEGRATED REPORT"));
    let data = r.data.expect("structured run data");
    assert_eq!(data["results"].as_array().map(Vec::len), Some(2));
    let entry = log.lock().unwrap()[0].clone();
    assert_eq!(entry["mode"], "coordinate");
    assert_eq!(entry["exec"], "off");
    assert_eq!(entry["workers"], json!(["org/alpha", "moonshotai/kimi-k2.6"]));
}

#[tokio::test]
async fn divide_mode_uses_the_supplied_subtasks() {
    let (env, _) = cloud_env(|_| {});
    let (ctx, _) = ctx_with_log();
    let r = AgentSwarmTool::new(env)
        .run(json!({ "task": "t", "mode": "divide", "subtasks": ["first part", "", "second part"], "synthesize": false }), &ctx)
        .await;
    assert!(r.ok, "{}", r.output);
    assert!(r.output.contains("[s1] first part"));
    assert!(r.output.contains("[s2] second part"));
    assert!(!r.output.contains("integrated answer"));
}

#[tokio::test]
async fn exec_is_only_enabled_when_the_session_allows_shell() {
    let (env, _) = cloud_env(|e| e.swarm.exec = qo_swarm::ExecMode::Bare);
    let (mut ctx, log) = ctx_with_log();
    ctx.allow_shell = false;
    let r = AgentSwarmTool::new(Arc::clone(&env)).run(json!({ "task": "t", "synthesize": false }), &ctx).await;
    assert!(r.ok);
    assert_eq!(log.lock().unwrap()[0]["exec"], "off");

    let dir = tempfile::tempdir().expect("tempdir");
    let (mut ctx, log) = ctx_with_log();
    ctx.cwd = dir.path().to_path_buf();
    let r = AgentSwarmTool::new(env).run(json!({ "task": "t", "synthesize": false }), &ctx).await;
    assert!(r.ok);
    assert_eq!(log.lock().unwrap()[0]["exec"], "bare");
}

#[tokio::test]
async fn ensemble_mode_answers_in_parallel_then_merges_with_the_synthesizer() {
    // synthesizer "primary" (the config default) → the primary chat model merges.
    let (env, _) = cloud_env(|_| {});
    let (ctx, _) = ctx_with_log();
    let r = AgentSwarmTool::new(env).run(json!({ "task": "why?", "mode": "ensemble" }), &ctx).await;
    assert!(r.ok, "{}", r.output);
    assert!(r.output.starts_with("🜂 agent swarm — ensemble · 2 worker(s), 2 ok"), "{}", r.output);
    assert!(r.output.contains("answer from org/alpha"));
    assert!(r.output.contains("=== synthesized answer (by primary-model) ===\nPRIMARY-MERGED"));

    // "auto" → the best ranked frontier worker merges.
    let (env, _) = cloud_env(|e| e.swarm.synthesizer = "auto".into());
    let r = AgentSwarmTool::new(env).run(json!({ "task": "why?", "mode": "ensemble" }), &ctx).await;
    assert!(r.output.contains("(by moonshotai/kimi-k2.6) ===\nENSEMBLE-MERGED"), "{}", r.output);
}

#[tokio::test]
async fn reports_when_no_workers_exist() {
    let (env, _) = env_with(|e| e.swarm.coordinate = false);
    let (ctx, _) = ctx_with_log();
    let r = AgentSwarmTool::new(env).run(json!({ "task": "t" }), &ctx).await;
    assert!(!r.ok);
    assert!(r.output.contains("no frontier workers available"));
}

#[test]
fn lead_provider_honours_the_synthesizer_setting() {
    let (env, factory) = cloud_env(|e| e.swarm.synthesizer = "NVIDIA:Org/Custom".into());
    assert_eq!(lead_provider(&env).model(), "Org/Custom");
    assert!(factory.created().contains(&(WorkerKind::Nvidia, "Org/Custom".to_string())));
    let (env, factory) = cloud_env(|e| e.swarm.synthesizer = "lmstudio:qwen3-8b".into());
    lead_provider(&env);
    assert!(factory.created().contains(&(WorkerKind::LmStudio, "qwen3-8b".to_string())));
    let (env, factory) = cloud_env(|e| e.swarm.synthesizer = "llama3.2:3b".into());
    lead_provider(&env);
    assert!(factory.created().contains(&(WorkerKind::Ollama, "llama3.2:3b".to_string())));
    let (env, _) = env_with(|e| e.swarm.synthesizer = "auto".into());
    assert_eq!(lead_provider(&env).model(), "primary-model", "auto without frontier keys → primary");
}

#[tokio::test]
async fn an_interrupted_tool_run_says_so() {
    let (env, factory) = cloud_env(|_| {});
    factory.register(Fake::new("org/alpha").hang().shared());
    factory.register(Fake::new("moonshotai/kimi-k2.6").hang().shared());
    let (mut ctx, _) = ctx_with_log();
    let cancel = qo_core::CancellationToken::new();
    ctx.cancel = Some(cancel.clone());
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        c2.cancel();
    });
    let r = AgentSwarmTool::new(env)
        .run(json!({ "task": "t", "mode": "divide", "subtasks": ["a", "b"] }), &ctx)
        .await;
    assert!(!r.ok);
    assert!(r.output.contains("stopped by user"));
    assert!(r.output.contains("(interrupted — partial results)"));
}

#[test]
fn the_chat_gate_needs_two_frontier_workers_and_an_enabled_swarm() {
    let (env, _) = cloud_env(|_| {});
    let roster = swarm_ready_roster(&env);
    assert_eq!(roster.iter().map(|r| r.label.as_str()).collect::<Vec<_>>(), vec!["alpha", "kimi-k2.6"]);
    let (env, _) = cloud_env(|e| e.swarm.enabled = false);
    assert!(swarm_ready_roster(&env).is_empty());
    let (env, _) = cloud_env(|e| e.have_openrouter = false);
    assert!(swarm_ready_roster(&env).is_empty(), "one frontier worker is not a swarm");
}
