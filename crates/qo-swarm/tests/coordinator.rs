//! Port of tests/swarmCoordinator.test.ts plus stop/abort/event-order coverage.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{env_with, user_of, worker, Fake};
use qo_core::SharedProvider;
use qo_swarm::{
    build_roster, clean_agent_text, decompose, decompose_with, parse_plan, parse_special_tool_calls, pick_lead,
    preflight_workers, Blackboard, CoordinateOptions, CoordinatedSwarm, Subtask, SubtaskStatus, SwarmEvent, SwarmRun,
    WorkerKind, WorkerResult,
};
use serde_json::json;

fn st(id: &str, deps: &[&str]) -> Subtask {
    Subtask {
        id: id.into(),
        title: id.to_uppercase(),
        detail: format!("d{id}"),
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
    }
}

fn by_id(run: &SwarmRun) -> HashMap<String, WorkerResult> {
    run.results.iter().map(|r| (r.id.clone().unwrap_or_default(), r.clone())).collect()
}

fn swarm_of(workers: Vec<SharedProvider>, lead: SharedProvider) -> CoordinatedSwarm {
    let (env, _) = env_with(|_| {});
    CoordinatedSwarm::new(env, workers.into_iter().map(worker).collect(), lead, Some(500))
}

/* ── parsePlan ── */

#[test]
fn parse_plan_extracts_from_fenced_prose_and_drops_dangling_deps() {
    let text = format!(
        "Here is the plan:\n```json\n{}\n```\nGood luck.",
        json!({ "subtasks": [
            { "id": "a", "title": "A", "detail": "do a", "dependsOn": [] },
            { "id": "b", "title": "B", "detail": "do b", "dependsOn": ["a", "ghost"] }
        ]})
    );
    let plan = parse_plan(&text);
    assert_eq!(plan.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
    // "ghost" references no real subtask → removed so scheduling can't deadlock.
    assert_eq!(plan[1].depends_on, vec!["a"]);
}

#[test]
fn parse_plan_returns_empty_without_json() {
    assert!(parse_plan("no json here at all").is_empty());
    assert!(parse_plan("{ not valid json").is_empty());
}

#[test]
fn parse_plan_accepts_aliases_arrays_and_fixes_ids() {
    let text = json!([
        { "slug": "Set Up DB", "name": "Database", "description": "make the schema", "deps": ["API Layer"] },
        { "id": "api layer", "title": "API", "task": "routes" },
        { "id": "api layer", "title": "dup", "detail": "dup id", "dependsOn": ["api-layer"] },
        { "id": 7, "title": "numeric", "detail": "x" },
        "just a string"
    ])
    .to_string();
    let plan = parse_plan(&text);
    let ids: Vec<&str> = plan.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["set-up-db", "api-layer", "s3", "7", "s5"]);
    assert_eq!(plan[0].title, "Database");
    assert_eq!(plan[0].detail, "make the schema");
    assert_eq!(plan[0].depends_on, vec!["api-layer"]);
    assert_eq!(plan[1].detail, "routes");
    assert_eq!(plan[2].depends_on, vec!["api-layer"]);
    // A bare string element has no fields: id/title/detail all fall back.
    assert_eq!(plan[4].title, "s5");
    assert_eq!(plan[4].detail, "s5");
    // Self-dependencies are dropped.
    let selfdep = parse_plan(r#"{"subtasks":[{"id":"x","detail":"d","dependsOn":["x"]}]}"#);
    assert!(selfdep[0].depends_on.is_empty());
}

/* ── preflightWorkers ── */

#[tokio::test]
async fn preflight_reports_each_workers_key_as_functional_or_broken() {
    let workers = vec![
        worker(Fake::new("good").shared()),
        worker(Fake::new("bad").health(false, Some("HTTP 401")).shared()),
    ];
    let seen = Mutex::new(Vec::new());
    let on = |r: &qo_swarm::PreflightResult| seen.lock().unwrap().push(r.model.clone());
    let results = preflight_workers(&workers, Some(&on), 15_000, None).await;
    assert!(results.iter().find(|r| r.model == "good").unwrap().ok);
    let bad = results.iter().find(|r| r.model == "bad").unwrap();
    assert!(!bad.ok);
    assert_eq!(bad.detail.as_deref(), Some("HTTP 401"));
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn preflight_times_out_a_hung_health_check() {
    struct Hung;
    #[async_trait::async_trait]
    impl qo_core::Provider for Hung {
        fn name(&self) -> &str {
            "hung"
        }
        fn model(&self) -> String {
            "hung".into()
        }
        fn set_model(&self, _: &str) {}
        async fn generate(&self, _: &[qo_core::Message], _: &qo_core::ModelOptions) -> anyhow::Result<qo_core::GenerateResult> {
            anyhow::bail!("unused")
        }
        async fn stream(
            &self,
            _: &[qo_core::Message],
            _: &mut (dyn for<'a> FnMut(&'a str) + Send),
            _: &qo_core::ModelOptions,
        ) -> anyhow::Result<qo_core::GenerateResult> {
            anyhow::bail!("unused")
        }
        async fn health_check(&self) -> qo_core::HealthStatus {
            std::future::pending().await
        }
    }
    let r = preflight_workers(&[worker(Arc::new(Hung))], None, 50, None).await;
    assert!(!r[0].ok);
    assert!(r[0].detail.as_deref().unwrap_or("").contains("timed out after 50ms"));
}

#[tokio::test]
async fn coordinated_preflight_drops_broken_workers_but_keeps_the_good_ones() {
    let lead = Fake::new("good").shared();
    let swarm = swarm_of(vec![Fake::new("good").shared(), Fake::new("bad").health(false, None).shared()], lead);
    swarm.preflight(None).await;
    assert_eq!(swarm.roster().iter().map(|r| r.model.as_str()).collect::<Vec<_>>(), vec!["good"]);
}

#[tokio::test]
async fn a_run_where_every_agent_fails_preflight_reports_it_and_stops() {
    let w = Fake::new("w1").health(false, Some("HTTP 403")).on_stream(|_| Ok("never".into()));
    let w = Arc::new(w);
    let swarm = swarm_of(vec![w.clone() as SharedProvider], Fake::new("lead").shared());
    let mut rx = swarm.subscribe();
    let run = swarm.run("t", CoordinateOptions { preflight: Some(true), ..Default::default() }).await;
    assert!(run.results.is_empty());
    assert_eq!(w.call_count(), 0, "no model calls after a failed preflight");
    let mut note = None;
    while let Ok(ev) = rx.try_recv() {
        if let SwarmEvent::Plan(p) = ev {
            assert_eq!(p.planned_by.as_deref(), Some("(none)"));
            note = p.note;
        }
    }
    let note = note.expect("plan event");
    assert!(note.contains("all agents failed preflight"));
    assert!(note.contains("w1: HTTP 403"));
}

/* ── Blackboard ── */

#[test]
fn blackboard_treats_a_stopped_dependency_as_resolved() {
    let mut board = Blackboard::new(&[st("a", &[]), st("b", &["a"])]);
    board.get_mut("a").unwrap().status = SubtaskStatus::Stopped;
    assert_eq!(board.ready().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["b"]);
}

#[test]
fn blackboard_only_releases_a_subtask_once_its_dependencies_are_resolved() {
    let mut board = Blackboard::new(&[st("a", &[]), st("b", &["a"])]);
    assert_eq!(board.ready().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["a"]);
    let a = board.get_mut("a").unwrap();
    a.status = SubtaskStatus::Done;
    a.result = "ALPHA-RESULT".into();
    assert_eq!(board.ready().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["b"]);
}

#[test]
fn blackboard_injects_a_dependencys_result_into_the_dependents_digest() {
    let mut board = Blackboard::new(&[
        Subtask { id: "a".into(), title: "A".into(), detail: "da".into(), depends_on: vec![] },
        Subtask { id: "b".into(), title: "B".into(), detail: "db".into(), depends_on: vec!["a".into()] },
        Subtask { id: "c".into(), title: "C".into(), detail: "dc".into(), depends_on: vec![] },
        Subtask { id: "d".into(), title: "D".into(), detail: "dd".into(), depends_on: vec!["a".into(), "c".into()] },
    ]);
    {
        let a = board.get_mut("a").unwrap();
        a.status = SubtaskStatus::Done;
        a.result = "ALPHA-RESULT".into();
    }
    board.get_mut("c").unwrap().status = SubtaskStatus::Failed;
    let digest = board.digest_for("b", 1400);
    assert!(digest.contains("ALPHA-RESULT"), "the actual result is shared");
    assert!(digest.contains("[a] A"), "labeled by id/title");
    assert!(digest.contains("TEAM PLAN"), "peer awareness section");
    let d = board.digest_for("d", 5);
    assert!(d.contains("ALPHA\n…[truncated]"));
    assert!(d.contains("did not finish"));
    assert!(d.contains("- [c] C (failed)"));
    assert_eq!(board.digest_for("nope", 10), "");
}

/* ── CoordinatedSwarm.run ── */

#[tokio::test]
async fn runs_subtasks_in_dependency_order_shares_context_and_isolates_failures() {
    let lead = Fake::new("lead").on_stream(|_| Ok("INTEGRATED".into())).shared();
    // One worker so a→b are forced sequential; b must SEE a's result via the board.
    let w = Fake::new("w1")
        .on_stream(|msgs| {
            let u = user_of(msgs);
            if u.contains("YOUR SUBTASK [a]") {
                return Ok("RESULT-A".into());
            }
            if u.contains("YOUR SUBTASK [b]") {
                return Ok(if u.contains("RESULT-A") { "B-SAW-A" } else { "B-BLIND" }.into());
            }
            if u.contains("YOUR SUBTASK [c]") {
                anyhow::bail!("boom");
            }
            Ok("?".into())
        })
        .shared();
    let swarm = swarm_of(vec![w], lead);
    let mut rx = swarm.subscribe();
    let run = swarm
        .run(
            "the overall task",
            CoordinateOptions {
                subtasks: Some(vec![
                    Subtask { id: "a".into(), title: "A".into(), detail: "da".into(), depends_on: vec![] },
                    Subtask { id: "b".into(), title: "B".into(), detail: "db".into(), depends_on: vec!["a".into()] },
                    Subtask { id: "c".into(), title: "C".into(), detail: "dc".into(), depends_on: vec![] },
                ]),
                ..Default::default()
            },
        )
        .await;
    let r = by_id(&run);
    assert_eq!(r["a"].text, "RESULT-A");
    assert_eq!(r["b"].text, "B-SAW-A", "built on the dependency's output");
    assert!(!r["c"].ok, "failed, but did not block a/b");
    assert!(r["c"].error.as_deref().unwrap_or("").contains("boom"));
    assert_eq!(r["b"].depends_on, vec!["a"]);
    assert_eq!(run.synthesis.as_deref(), Some("INTEGRATED"));
    assert_eq!(run.synthesized_by.as_deref(), Some("lead"));

    let mut planned = 0;
    let mut done = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            SwarmEvent::Plan(_) => planned += 1,
            SwarmEvent::Done(_) => done = true,
            _ => {}
        }
    }
    assert_eq!(planned, 1);
    assert!(done);
}

#[tokio::test]
async fn passes_the_real_overall_task_into_each_agents_prompt() {
    let saw = Arc::new(Mutex::new(false));
    let saw2 = Arc::clone(&saw);
    let w = Fake::new("w1")
        .on_stream(move |msgs| {
            if user_of(msgs).contains("OVERALL TASK:\nbuild a parser") {
                *saw2.lock().unwrap() = true;
            }
            Ok("ok".into())
        })
        .shared();
    let swarm = swarm_of(vec![w], Fake::new("lead").on_stream(|_| Ok("X".into())).shared());
    swarm
        .run("build a parser", CoordinateOptions { synthesize: false, subtasks: Some(vec![st("a", &[])]), ..Default::default() })
        .await;
    assert!(*saw.lock().unwrap());
}

#[tokio::test]
async fn emits_the_full_event_sequence() {
    let lead = Fake::new("lead").on_stream(|_| Ok("SYNTH".into())).shared();
    let w1 = Fake::new("w1").on_stream(|_| Ok("one".into())).shared();
    let w2 = Fake::new("w2").on_stream(|_| Ok("two".into())).shared();
    let swarm = swarm_of(vec![w1, w2], lead);
    let mut rx = swarm.subscribe();
    let run = swarm
        .run("t", CoordinateOptions { subtasks: Some(vec![st("a", &[]), st("b", &[]), st("c", &["a", "b"])]), ..Default::default() })
        .await;
    assert_eq!(run.results.len(), 3);
    let mut kinds = Vec::new();
    let mut waves = Vec::new();
    let mut panes = Vec::new();
    let mut synth_text = String::new();
    while let Ok(ev) = rx.try_recv() {
        let k = match &ev {
            SwarmEvent::PreflightStart { .. } => "preflightStart",
            SwarmEvent::Preflight(_) => "preflight",
            SwarmEvent::PreflightDone { .. } => "preflightDone",
            SwarmEvent::Planner { .. } => "planner",
            SwarmEvent::Plan(_) => "plan",
            SwarmEvent::Wave(w) => {
                waves.push(w.subtask_ids.clone());
                "wave"
            }
            SwarmEvent::Assign(a) => {
                panes.push((a.subtask_id.clone(), a.pane));
                "assign"
            }
            SwarmEvent::Delta(_) => "delta",
            SwarmEvent::Status(_) => "status",
            SwarmEvent::Result(_) => "result",
            SwarmEvent::ExecCmd(_) => "execCmd",
            SwarmEvent::Synth { .. } => "synth",
            SwarmEvent::SynthDelta { delta } => {
                synth_text.push_str(delta);
                "synthDelta"
            }
            SwarmEvent::Artifacts(_) => "artifacts",
            SwarmEvent::Done(_) => "done",
        };
        if kinds.last() != Some(&k) {
            kinds.push(k);
        }
    }
    assert_eq!(kinds.first(), Some(&"preflightStart"), "preflight on by default: {kinds:?}");
    assert!(kinds.contains(&"plan") && kinds.contains(&"wave") && kinds.contains(&"delta"));
    assert_eq!(&kinds[kinds.len() - 3..], &["synth", "synthDelta", "done"]);
    assert_eq!(waves, vec![vec!["a".to_string(), "b".to_string()], vec!["c".to_string()]]);
    // a and b share wave 0 on panes 0 and 1; c runs alone on pane 0.
    assert_eq!(panes, vec![("a".into(), 0), ("b".into(), 1), ("c".into(), 0)]);
    assert_eq!(synth_text, "SYNTH");
}

#[tokio::test]
async fn a_dependency_cycle_is_broken_instead_of_deadlocking() {
    let w = Fake::new("w1").on_stream(|m| Ok(format!("did {}", if user_of(m).contains("[x]") { "x" } else { "y" }))).shared();
    let swarm = swarm_of(vec![w], Fake::new("lead").shared());
    let run = swarm
        .run(
            "t",
            CoordinateOptions {
                synthesize: false,
                preflight: Some(false),
                subtasks: Some(vec![st("x", &["y"]), st("y", &["x"])]),
                ..Default::default()
            },
        )
        .await;
    assert_eq!(run.results.len(), 2);
    assert!(run.results.iter().all(|r| r.ok));
}

#[tokio::test]
async fn stop_pane_stops_only_that_agent_and_dependents_still_run() {
    let slow = Fake::new("slow").hang().on_stream(|_| Ok("slow:c".into())).shared();
    let fast = Fake::new("fast").on_stream(|m| Ok(format!("fast:{}", if user_of(m).contains("YOUR SUBTASK [c]") { "c" } else { "b" }))).shared();
    let lead = Fake::new("lead").on_stream(|_| Ok("SYN".into())).shared();
    let swarm = Arc::new(swarm_of(vec![slow, fast], lead));
    let mut rx = swarm.subscribe();
    let s2 = Arc::clone(&swarm);
    let handle = tokio::spawn(async move {
        s2.run(
            "t",
            CoordinateOptions {
                preflight: Some(false),
                subtasks: Some(vec![st("a", &[]), st("b", &[]), st("c", &["a"])]),
                ..Default::default()
            },
        )
        .await
    });
    // Wait until pane 0 (the hanging agent) has streamed its first chunk, then stop it.
    loop {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(SwarmEvent::Delta(d))) if d.pane == 0 => break,
            Ok(Some(_)) => {}
            other => panic!("no delta from pane 0: {other:?}"),
        }
    }
    swarm.stop_pane(0);
    let run = tokio::time::timeout(Duration::from_secs(5), handle).await.expect("run finishes").expect("join");
    let r = by_id(&run);
    assert!(!r["a"].ok);
    assert_eq!(r["a"].error.as_deref(), Some("stopped by user"));
    assert_eq!(r["a"].text, "partial work", "partial stream text is kept");
    assert_eq!(r["b"].text, "fast:b");
    assert!(r["c"].ok, "a stopped dependency still releases its dependents");
    assert_eq!(run.synthesis.as_deref(), Some("SYN"));
}

#[tokio::test]
async fn global_cancel_stops_every_agent_and_skips_synthesis() {
    let w1 = Fake::new("w1").hang().shared();
    let w2 = Fake::new("w2").hang().shared();
    let lead = Arc::new(Fake::new("lead").on_stream(|_| Ok("never".into())));
    let swarm = swarm_of(vec![w1, w2], lead.clone());
    let cancel = qo_core::CancellationToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        c2.cancel();
    });
    let run = tokio::time::timeout(
        Duration::from_secs(5),
        swarm.run(
            "t",
            CoordinateOptions {
                cancel: Some(cancel),
                preflight: Some(false),
                subtasks: Some(vec![st("a", &[]), st("b", &[])]),
                ..Default::default()
            },
        ),
    )
    .await
    .expect("abort is prompt");
    assert_eq!(run.results.len(), 2);
    assert!(run.results.iter().all(|r| !r.ok && r.error.as_deref() == Some("stopped by user")));
    assert!(run.synthesis.is_none());
    assert_eq!(lead.call_count(), 0);
}

#[tokio::test]
async fn synthesis_falls_back_to_a_non_streaming_call_when_the_stream_fails() {
    let lead = Fake::new("lead")
        .on_stream(|_| anyhow::bail!("stream broke"))
        .on_generate(|_| Ok("FROM-GENERATE".into()))
        .shared();
    let w = Fake::new("w1").on_stream(|m| Ok(user_of(m).lines().find(|l| l.starts_with("YOUR SUBTASK")).unwrap_or("").into())).shared();
    let swarm = swarm_of(vec![w], lead);
    let run = swarm
        .run("t", CoordinateOptions { preflight: Some(false), subtasks: Some(vec![st("a", &[]), st("b", &[])]), ..Default::default() })
        .await;
    assert_eq!(run.synthesis.as_deref(), Some("FROM-GENERATE"));
}

/* ── decompose ── */

#[tokio::test]
async fn decompose_parses_a_valid_plan_from_the_lead() {
    let lead = Fake::new("lead")
        .on_generate(|_| Ok(json!({ "subtasks": [{ "id": "x", "title": "X", "detail": "dx", "dependsOn": [] }] }).to_string()))
        .shared();
    let subs = decompose(lead, "t", 2, 500, None).await;
    assert_eq!(subs.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["x"]);
}

#[tokio::test]
async fn decompose_degrades_to_a_single_whole_task_subtask_without_json() {
    let lead = Fake::new("lead").on_generate(|_| Ok("sorry, no plan".into())).shared();
    let subs = decompose(lead, "the big task", 3, 500, None).await;
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].detail, "the big task");
}

#[tokio::test]
async fn the_planner_prompt_sizes_the_plan_and_asks_for_json() {
    let seen = Arc::new(Mutex::new(String::new()));
    let s2 = Arc::clone(&seen);
    let lead = Fake::new("lead")
        .on_generate(move |m| {
            *s2.lock().unwrap() = user_of(m);
            Ok(r#"{"complexity":"complex","subtasks":[{"id":"x","detail":"d"}]}"#.into())
        })
        .shared();
    let res = decompose_with(&[lead], "T", 2, 500, None, &mut |_| {}).await;
    assert_eq!(res.complexity, qo_swarm::PlanComplexity::Complex);
    let p = seen.lock().unwrap().clone();
    assert!(p.contains("team of 2 expert agent(s)"));
    assert!(p.contains("Aim for 3-4 subtasks."));
    assert!(p.contains("Return ONLY JSON"));
}

const GOOD_PLAN: &str = r#"{"subtasks":[{"id":"x","title":"X","detail":"dx","dependsOn":[]}]}"#;

#[tokio::test]
async fn decompose_with_falls_through_to_the_next_candidate_when_the_first_times_out() {
    let slow = Fake::new("slow-lead").on_generate(|_| anyhow::bail!("nvidia request timed out after 90000ms")).shared();
    let fast = Fake::new("fast-lead").on_generate(|_| Ok(GOOD_PLAN.into())).shared();
    let mut attempts = Vec::new();
    let res = decompose_with(&[slow, fast], "t", 2, 500, None, &mut |m| attempts.push(m.to_string())).await;
    assert_eq!(res.subtasks.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["x"]);
    assert_eq!(res.planned_by, "fast-lead");
    assert!(res.note.as_deref().unwrap_or("").contains("slow-lead"));
    assert_eq!(attempts, vec!["slow-lead", "fast-lead"]);
}

#[tokio::test]
async fn decompose_with_skips_a_candidate_that_returns_unparseable_text() {
    let vague = Fake::new("vague").on_generate(|_| Ok("no json from me".into())).shared();
    let fast = Fake::new("fast").on_generate(|_| Ok(GOOD_PLAN.into())).shared();
    let res = decompose_with(&[vague, fast], "t", 2, 500, None, &mut |_| {}).await;
    assert_eq!(res.planned_by, "fast");
}

#[tokio::test]
async fn decompose_with_degrades_with_a_note_when_every_candidate_fails() {
    let bad1 = Fake::new("b1").on_generate(|_| anyhow::bail!("boom1")).shared();
    let bad2 = Fake::new("b2").on_generate(|_| Ok("nope".into())).shared();
    let res = decompose_with(&[bad1, bad2], "whole task", 2, 500, None, &mut |_| {}).await;
    assert_eq!(res.subtasks.len(), 1);
    assert_eq!(res.subtasks[0].detail, "whole task");
    assert_eq!(res.planned_by, "(fallback)");
    assert!(res.note.as_deref().unwrap_or("").contains("boom1"));
}

#[tokio::test]
async fn the_run_plans_with_roster_order_then_the_lead() {
    let w1 = Fake::new("w1").on_generate(|_| anyhow::bail!("w1 down")).on_stream(|_| Ok("r".into())).shared();
    let lead = Fake::new("lead")
        .on_generate(|_| Ok(r#"{"subtasks":[{"id":"p","detail":"d1"},{"id":"q","detail":"d2"}]}"#.into()))
        .on_stream(|_| Ok("S".into()))
        .shared();
    let swarm = swarm_of(vec![w1], lead);
    let mut rx = swarm.subscribe();
    let run = swarm.run("t", CoordinateOptions { preflight: Some(false), ..Default::default() }).await;
    assert_eq!(run.results.len(), 2);
    let mut planners = Vec::new();
    let mut plan = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            SwarmEvent::Planner { model } => planners.push(model),
            SwarmEvent::Plan(p) => plan = Some(p),
            _ => {}
        }
    }
    assert_eq!(planners, vec!["w1", "lead"]);
    let plan = plan.expect("plan");
    assert_eq!(plan.planned_by.as_deref(), Some("lead"));
    assert!(plan.note.as_deref().unwrap_or("").contains("planner fell back to lead"));
}

/* ── special tool calls / cleanup ── */

#[test]
fn parses_kimi_special_token_tool_calls() {
    let text = "Sure.<|tool_calls_section_begin|> <|tool_call_begin|> functions.run_shell:5 \
                <|tool_call_argument_begin|> {\"command\": \"echo hi\"} <|tool_call_end|> <|tool_calls_section_end|>";
    let calls = parse_special_tool_calls(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "run_shell");
    assert_eq!(calls[0].id, "special_5");
    assert_eq!(calls[0].arguments["command"], "echo hi");
}

#[test]
fn special_tool_calls_are_empty_for_normal_text_and_parse_several() {
    assert!(parse_special_tool_calls("just a normal answer").is_empty());
    let t = "<|tool_call_begin|>functions.a:1<|tool_call_argument_begin|>{\"x\":1}<|tool_call_end|>\
             <|tool_call_begin|>functions.b:2<|tool_call_argument_begin|>{\"y\":2}<|tool_call_end|>";
    let calls = parse_special_tool_calls(t);
    assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
    let bad = parse_special_tool_calls("<|tool_call_begin|>run<|tool_call_argument_begin|>{oops}<|tool_call_end|>");
    assert_eq!(bad[0].arguments["_raw"], "{oops}");
    assert_eq!(bad[0].id, "special_0");
}

#[test]
fn clean_agent_text_strips_control_tokens_and_think_blocks() {
    let dirty = "<think>planning</think>Hello <|tool_calls_section_begin|><|tool_call_begin|>functions.x:1\
                 <|tool_call_argument_begin|>{\"a\":1}<|tool_call_end|><|tool_calls_section_end|> world<|im_end|>";
    let clean = clean_agent_text(dirty);
    assert!(!clean.contains("<|"));
    assert!(!clean.contains("<think>"));
    assert!(clean.contains("Hello"));
    assert!(clean.contains("world"));
}

#[tokio::test]
async fn plan_event_reports_supplied_for_pre_supplied_subtasks() {
    let swarm = swarm_of(vec![Fake::new("w1").on_stream(|_| Ok("ok".into())).shared()], Fake::new("lead").shared());
    let mut rx = swarm.subscribe();
    swarm.run("t", CoordinateOptions { synthesize: false, subtasks: Some(vec![st("a", &[])]), ..Default::default() }).await;
    let mut planned_by = None;
    while let Ok(ev) = rx.try_recv() {
        if let SwarmEvent::Plan(p) = ev {
            planned_by = p.planned_by;
        }
    }
    assert_eq!(planned_by.as_deref(), Some("(supplied)"));
}

/* ── roster / lead ── */

#[test]
fn roster_is_frontier_first_then_local_then_primary() {
    let (env, _) = env_with(|e| {
        e.have_nvidia = true;
        e.fallback_models = vec!["qwen2.5:7b".into(), "nvidia:meta/llama-3.3-70b-instruct".into()];
    });
    let r = build_roster(&env, None);
    assert!(!r.local);
    assert_eq!(r.workers.iter().map(|w| w.kind).collect::<Vec<_>>(), vec![WorkerKind::Nvidia]);

    let (env, _) = env_with(|e| e.fallback_models = vec!["qwen2.5:7b".into()]);
    let r = build_roster(&env, None);
    assert!(r.local);
    assert_eq!(r.workers[0].model, "qwen2.5:7b");

    let (env, _) = env_with(|_| {});
    let r = build_roster(&env, None);
    assert!(r.local);
    assert_eq!(r.workers.len(), 1);
    assert_eq!(r.workers[0].model, "primary-model");
}

#[test]
fn pick_lead_prefers_the_ranked_frontier_model() {
    let (env, _) = env_with(|_| {});
    let ws = vec![
        worker(Fake::new("meta/llama-3.3-70b-instruct").shared()),
        worker(Fake::new("moonshotai/kimi-k2.6").shared()),
    ];
    assert_eq!(pick_lead(&env, &ws).model(), "moonshotai/kimi-k2.6");
    let plain = vec![worker(Fake::new("x/unknown").shared())];
    assert_eq!(pick_lead(&env, &plain).model(), "x/unknown");
    assert_eq!(pick_lead(&env, &[]).model(), "primary-model");
}
