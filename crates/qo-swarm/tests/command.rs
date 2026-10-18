//! The `swarm` subcommand logic (cli/commands/swarm.ts), plain/non-TTY paths only
//! (`live: false`, so a test never takes over a real terminal).

mod common;

use std::sync::Arc;

use common::{env_with, user_of, Fake};
use qo_swarm::{
    run_swarm_command_with_env, swarm::SwarmArtifacts, SwarmCommandOptions, SwarmMode, SwarmRun, WorkerResult,
};

fn opts(task: Option<&str>) -> SwarmCommandOptions {
    SwarmCommandOptions {
        task: task.map(str::to_string),
        live: false,
        color: false,
        cwd: std::env::temp_dir(),
        ..Default::default()
    }
}

async fn run(env: Arc<qo_swarm::SwarmEnv>, o: SwarmCommandOptions) -> (qo_swarm::SwarmCommandOutcome, String) {
    let mut out: Vec<u8> = Vec::new();
    let outcome = run_swarm_command_with_env(env, o, &mut out).await.expect("command runs");
    (outcome, String::from_utf8_lossy(&out).into_owned())
}

#[tokio::test]
async fn a_task_is_required_unless_listing() {
    let (env, _) = env_with(|_| {});
    let (o, text) = run(env, opts(None)).await;
    assert_eq!(o.exit_code, 1);
    assert!(text.contains("A <task> is required"));
}

#[tokio::test]
async fn list_shows_the_roster_without_calling_any_model() {
    let alpha = Arc::new(Fake::new("org/alpha"));
    let (env, factory) = env_with(|e| {
        e.have_nvidia = true;
        e.fallback_models = vec!["nvidia:org/alpha".into()];
    });
    factory.register(alpha.clone());
    let (o, text) = run(env, SwarmCommandOptions { list: true, ..opts(None) }).await;
    assert_eq!(o.exit_code, 0);
    assert!(text.contains("Coordinated swarm — 1 agent(s):"), "{text}");
    assert!(text.contains("• org/alpha (nvidia)"));
    assert!(text.contains("(--list: roster only; no models were called.)"));
    assert_eq!(alpha.call_count(), 0);
}

#[tokio::test]
async fn coordinated_plain_path_logs_plan_progress_and_summary() {
    let (env, factory) = env_with(|e| {
        e.have_nvidia = true;
        e.fallback_models = vec!["nvidia:org/alpha".into()];
    });
    factory.register(
        Fake::new("org/alpha")
            .on_generate(|_| Ok(r#"{"subtasks":[{"id":"a","title":"Alpha","detail":"x"},{"id":"b","title":"Beta","detail":"y","dependsOn":["a"]}]}"#.into()))
            .on_stream(|m| {
                let u = user_of(&m.to_vec());
                Ok(if u.contains("integrated Markdown") { "THE REPORT".into() } else { "piece".into() })
            })
            .shared(),
    );
    let (o, text) = run(env, opts(Some("make a thing"))).await;
    assert_eq!(o.exit_code, 0);
    assert!(text.contains("Checking 1 agent key(s)…"), "{text}");
    assert!(text.contains("Plan — 2 subtask(s) (planned by org/alpha):"), "{text}");
    assert!(text.contains("[b] Beta ← a"));
    assert!(text.contains("▸ org/alpha → [a] Alpha"));
    assert!(text.contains("✓ [a] ("));
    assert!(text.contains("Synthesizing with org/alpha…"));
    assert!(text.contains("═══ Integrated answer (by org/alpha) ═══\nTHE REPORT"));
    assert!(text.contains("(swarm finished in "));
    let run = o.run.expect("run");
    assert_eq!(run.results.len(), 2);
}

#[tokio::test]
async fn divide_flag_supplies_subtasks_and_bad_exec_is_rejected() {
    let (env, _) = env_with(|_| {});
    let (o, text) = run(
        Arc::clone(&env),
        SwarmCommandOptions { divide: vec!["first".into(), "second".into()], synth: false, ..opts(Some("t")) },
    )
    .await;
    assert_eq!(o.exit_code, 0);
    assert!(text.contains("[s1] first"), "{text}");
    assert!(!text.contains("planned by"), "supplied plans don't name a planner");
    assert!(text.contains("(local fallback — no frontier key configured)"));

    let (o, text) = run(env, SwarmCommandOptions { exec: Some("cloud".into()), ..opts(Some("t")) }).await;
    assert_eq!(o.exit_code, 1);
    assert!(text.contains("Unknown --exec mode 'cloud'"));
}

#[tokio::test]
async fn plain_mode_runs_the_classic_ensemble_and_synthesizes_with_the_primary() {
    let (env, factory) = env_with(|e| {
        e.have_nvidia = true;
        e.fallback_models = vec!["nvidia:org/alpha".into(), "nvidia:org/beta".into()];
    });
    factory.register(Fake::new("org/alpha").on_generate(|_| Ok("A says".into())).shared());
    factory.register(Fake::new("org/beta").on_generate(|_| anyhow::bail!("HTTP 429 rate limit")).shared());
    factory.register(Fake::new("primary-model").on_generate(|_| Ok("unused: single answer short-circuits".into())).shared());
    let (o, text) = run(env, SwarmCommandOptions { plain: true, ..opts(Some("question")) }).await;
    assert_eq!(o.exit_code, 0);
    assert!(text.contains("Agent swarm (plain) — 2 model(s) in parallel:"), "{text}");
    assert!(text.contains("✓ org/alpha"));
    assert!(text.contains("✗ org/beta"));
    assert!(text.contains("HTTP 429"));
    assert!(text.contains("═══ Synthesized answer ═══\nA says"));
    assert_eq!(o.run.expect("run").mode, SwarmMode::Ensemble);
}

#[tokio::test]
async fn plain_mode_without_frontier_workers_fails_cleanly() {
    let (env, _) = env_with(|_| {});
    let (o, text) = run(env, SwarmCommandOptions { plain: true, ..opts(Some("q")) }).await;
    assert_eq!(o.exit_code, 1);
    assert!(text.contains("No frontier workers available."));
}

#[tokio::test]
async fn demo_rehearses_the_real_coordinator_without_models() {
    let (env, _) = env_with(|_| {});
    let (o, text) = run(env, SwarmCommandOptions { demo: true, ..opts(None) }).await;
    assert_eq!(o.exit_code, 0);
    assert!(text.contains("Swarm demo — no models are called"));
    assert!(text.contains("[integrate] Integrate everything ← requirements, architecture, examples"), "{text}");
    assert!(text.contains("(no TTY here"));
    let run = o.run.expect("run");
    assert_eq!(run.results.len(), 4);
    assert!(run.results.iter().all(|r| r.ok));
    assert_eq!(run.synthesized_by.as_deref(), Some("demo/lead"));
}

#[test]
fn result_markdown_lists_answer_files_and_agents() {
    let run = SwarmRun {
        mode: SwarmMode::Divide,
        results: vec![
            WorkerResult {
                id: Some("a".into()),
                title: Some("Alpha".into()),
                model: "m1".into(),
                ok: true,
                text: "did a".into(),
                ms: 1500,
                ..Default::default()
            },
            WorkerResult { id: Some("b".into()), title: Some("Beta".into()), model: "m2".into(), error: Some("boom".into()), ..Default::default() },
        ],
        synthesis: Some("FINAL".into()),
        synthesized_by: Some("lead".into()),
        artifacts: Some(SwarmArtifacts { location: "bare metal: X".into(), files: vec!["X/a.py".into()] }),
    };
    let md = qo_swarm::command::render_result_markdown(&run, "the task");
    assert!(md.starts_with("# Swarm result\n\n**Task:** the task\n"));
    assert!(md.contains("## Integrated result\n\nFINAL"));
    assert!(md.contains("Location: bare metal: X"));
    assert!(md.contains("- X/a.py"));
    assert!(md.contains("### [a] Alpha — m1 1.5s\n\ndid a"));
    assert!(md.contains("### [b] Beta — m2 0.0s (FAILED)\n\nerror: boom"));

    let dir = tempfile::tempdir().expect("tempdir");
    let file = qo_swarm::command::save_result(&run, "the task", dir.path()).expect("saved");
    assert!(file.file_name().unwrap().to_string_lossy().starts_with("swarm-result-"));
    assert_eq!(std::fs::read_to_string(file).unwrap(), md);
}
