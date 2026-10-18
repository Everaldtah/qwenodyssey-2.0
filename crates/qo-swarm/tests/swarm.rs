//! Port of tests/swarm.test.ts plus ensemble/divide/synthesis coverage.

mod common;

use std::sync::Arc;

use common::{env_with, user_of, worker, Fake};
use qo_swarm::{
    frontier_workers, strip_thinking, synthesize, Swarm, SwarmEnv, SwarmMode, SwarmOptions, SwarmRun, WorkerKind,
    WorkerResult,
};

#[test]
fn includes_a_cloud_ref_only_when_its_api_key_is_present() {
    let fallbacks = vec!["nvidia:foo/bar".to_string(), "openrouter:baz/qux".to_string()];
    // No keys → no cloud workers.
    let (env, _) = env_with(|e| e.fallback_models = fallbacks.clone());
    assert!(frontier_workers(&env, &SwarmOptions::default()).is_empty());

    // NVIDIA key only → just the nvidia worker.
    let (env, _) = env_with(|e| {
        e.fallback_models = fallbacks.clone();
        e.have_nvidia = true;
    });
    let nv: Vec<WorkerKind> = frontier_workers(&env, &SwarmOptions::default()).iter().map(|w| w.kind).collect();
    assert_eq!(nv, vec![WorkerKind::Nvidia]);

    // Both keys → both workers.
    let (env, _) = env_with(|e| {
        e.fallback_models = fallbacks.clone();
        e.have_nvidia = true;
        e.have_openrouter = true;
    });
    let both: Vec<WorkerKind> = frontier_workers(&env, &SwarmOptions::default()).iter().map(|w| w.kind).collect();
    assert_eq!(both, vec![WorkerKind::Nvidia, WorkerKind::OpenRouter]);
}

#[test]
fn adds_the_primary_when_it_is_a_cloud_model_and_dedupes_against_fallbacks() {
    let (env, factory) = env_with(|e| {
        e.have_openrouter = true;
        e.primary_backend = "openrouter".into();
        e.primary_model = "moonshotai/kimi-k2.6".into();
        // Same model also listed as a fallback — must not appear twice.
        e.fallback_models = vec!["openrouter:moonshotai/kimi-k2.6".into()];
    });
    let workers = frontier_workers(&env, &SwarmOptions::default());
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0].model, "moonshotai/kimi-k2.6");
    assert_eq!(workers[0].label, "kimi-k2.6");
    assert_eq!(workers[0].model_ref, "openrouter:moonshotai/kimi-k2.6");
    assert_eq!(factory.created(), vec![(WorkerKind::OpenRouter, "moonshotai/kimi-k2.6".to_string())]);
}

#[test]
fn excludes_local_refs_unless_include_local_and_respects_max_workers() {
    let (env, _) = env_with(|e| {
        e.have_nvidia = true;
        e.fallback_models =
            vec!["nvidia:a/one".into(), "nvidia:a/two".into(), "qwen2.5:7b".into(), "deepseek-r1:7b".into()];
    });
    // Cloud-only by default.
    let kinds = |o: &SwarmOptions| frontier_workers(&env, o).iter().map(|w| w.kind).collect::<Vec<_>>();
    assert_eq!(kinds(&SwarmOptions::default()), vec![WorkerKind::Nvidia, WorkerKind::Nvidia]);

    // include_local pulls in the bare ollama tags too.
    let with_local = SwarmOptions { include_local: true, ..Default::default() };
    assert_eq!(kinds(&with_local), vec![WorkerKind::Nvidia, WorkerKind::Nvidia, WorkerKind::Ollama, WorkerKind::Ollama]);
    let local = frontier_workers(&env, &with_local);
    assert_eq!(local[2].model_ref, "qwen2.5:7b", "ollama refs stay bare tags");

    // max_workers caps the roster.
    let capped = SwarmOptions { include_local: true, max_workers: Some(2), ..Default::default() };
    assert_eq!(frontier_workers(&env, &capped).len(), 2);
}

#[test]
fn lmstudio_refs_are_local_and_blank_refs_are_skipped() {
    let (env, _) = env_with(|e| e.fallback_models = vec!["  ".into(), "lmstudio:qwen/qwen3-8b".into()]);
    assert!(frontier_workers(&env, &SwarmOptions::default()).is_empty());
    let w = frontier_workers(&env, &SwarmOptions { include_local: true, ..Default::default() });
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].kind, WorkerKind::LmStudio);
    assert_eq!(w[0].label, "qwen3-8b");
}

#[test]
fn env_from_real_config_resolves_keys_and_roster() {
    let mut cfg = qo_core::config::default_config();
    cfg.nvidia.api_key = "nvapi-test".into();
    // Host env vars may hold real keys, so "unavailable" is modelled by disabling the backend.
    cfg.openrouter.enabled = false;
    cfg.model.fallback_models = vec!["nvidia:foo/bar".into(), "openrouter:baz/qux".into()];
    cfg.swarm.max_workers = 3;
    let env = SwarmEnv::from_config(&cfg);
    assert!(env.have_nvidia);
    assert!(!env.have_openrouter);
    assert_eq!(env.swarm.max_workers, 3);
    assert_eq!(env.daytona.base_url, "https://app.daytona.io/api");
    let kinds: Vec<WorkerKind> = frontier_workers(&env, &SwarmOptions::default()).iter().map(|w| w.kind).collect();
    assert_eq!(kinds, vec![WorkerKind::Nvidia]);

    // A disabled backend contributes nothing even with a key.
    cfg.nvidia.enabled = false;
    assert!(!SwarmEnv::from_config(&cfg).have_nvidia);
}

#[tokio::test]
async fn ensemble_asks_every_worker_and_isolates_failures() {
    let good = Fake::new("org/good").on_generate(|m| {
        assert!(user_of(m).contains("--- Context ---\nCTX"));
        Ok("<think>hmm</think>  ANSWER  ".into())
    });
    let bad = Fake::new("org/bad").on_generate(|_| anyhow::bail!("HTTP 401 unauthorized"));
    let swarm =
        Swarm::with_system(vec![worker(good.shared()), worker(bad.shared())], SwarmOptions::default(), "SYSTEM");
    assert_eq!(swarm.size(), 2);
    let run = swarm.ensemble("the task", Some("CTX")).await;
    assert_eq!(run.mode, SwarmMode::Ensemble);
    assert_eq!(run.results.len(), 2);
    assert!(run.results[0].ok);
    assert_eq!(run.results[0].text, "ANSWER");
    assert_eq!(run.results[0].task, "the task");
    assert!(!run.results[1].ok);
    assert!(run.results[1].error.as_deref().unwrap_or("").contains("401"));
}

#[tokio::test]
async fn divide_shards_round_robin_and_keeps_order_per_worker() {
    let a = Fake::new("a").on_generate(|m| Ok(format!("A:{}", user_of(m).lines().next().unwrap_or(""))));
    let b = Fake::new("b").on_generate(|m| Ok(format!("B:{}", user_of(m).lines().next().unwrap_or(""))));
    let swarm = Swarm::with_system(vec![worker(a.shared()), worker(b.shared())], SwarmOptions::default(), "S");
    let subs = vec!["one".to_string(), " ".to_string(), "two".to_string(), "three".to_string()];
    let run = swarm.divide(&subs, None).await;
    assert_eq!(run.mode, SwarmMode::Divide);
    let texts: Vec<&str> = run.results.iter().map(|r| r.text.as_str()).collect();
    // Worker a takes subtasks 0 and 2 ("one", "three"), b takes 1 ("two").
    assert_eq!(texts, vec!["A:one", "A:three", "B:two"]);
    assert_eq!(run.results[1].task, "three");
}

#[tokio::test]
async fn divide_with_no_workers_is_empty() {
    let swarm = Swarm::with_system(vec![], SwarmOptions::default(), "S");
    let run = swarm.divide(&["x".to_string()], None).await;
    assert!(run.results.is_empty());
}

fn ok(model: &str, text: &str) -> WorkerResult {
    WorkerResult { model: model.into(), ok: true, text: text.into(), ..Default::default() }
}

#[tokio::test]
async fn synthesize_short_circuits_merges_and_falls_back() {
    let lead = Fake::new("lead").on_generate(|m| {
        let u = user_of(m);
        assert!(u.contains("### Answer 1 — model: m1\nfirst"));
        assert!(u.contains("Combine their answers"));
        Ok("<think>x</think>MERGED".into())
    });
    let lead = Arc::new(lead);

    let none = SwarmRun::default();
    assert_eq!(
        synthesize(lead.as_ref(), "t", &none, 100, None).await,
        "All swarm workers failed; no answer to synthesize."
    );
    let single = SwarmRun { results: vec![ok("m1", "only")], ..Default::default() };
    assert_eq!(synthesize(lead.as_ref(), "t", &single, 100, None).await, "only");
    assert_eq!(lead.call_count(), 0, "0 or 1 answers never call the lead");

    let two = SwarmRun { results: vec![ok("m1", "first"), ok("m2", "second")], ..Default::default() };
    assert_eq!(synthesize(lead.as_ref(), "t", &two, 100, None).await, "MERGED");

    let broken = Fake::new("lead").on_generate(|_| anyhow::bail!("down"));
    let merged = synthesize(&broken, "t", &two, 100, None).await;
    assert!(merged.contains("### Answer 1 — model: m1\nfirst"));
    assert!(merged.contains("### Answer 2 — model: m2\nsecond"));
}

#[test]
fn strip_thinking_handles_closed_and_close_only_blocks() {
    assert_eq!(strip_thinking("<think>a\nb</think>\n\nfinal"), "final");
    assert_eq!(strip_thinking("<THINKING>x</THINKING> ok "), "ok");
    assert_eq!(strip_thinking("pre-filled reasoning</think>answer"), "answer");
    assert_eq!(strip_thinking("plain"), "plain");
}
