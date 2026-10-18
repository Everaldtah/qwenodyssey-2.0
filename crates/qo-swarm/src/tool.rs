//! The `agent_swarm` tool: lets the model spin up MANY frontier agents at once,
//! each on a different cloud model/API key, and either (coordinate) have a lead
//! split the task into dependency-aware subtasks that share a blackboard, or
//! (ensemble) have them all answer the same task then synthesize a best answer.
//! Auto-built from the configured cloud keys.

use std::sync::Arc;

use async_trait::async_trait;
use qo_core::tool::SharedTool;
use qo_core::{SharedProvider, Tool, ToolContext, ToolResult, ToolSpec};
use serde_json::{json, Value};

use crate::coordinator::{create_coordinated_swarm, CoordinateOptions, Subtask, LEAD_RANK};
use crate::env::SwarmEnv;
use crate::exec::{BareExecutorOptions, ExecMode};
use crate::swarm::{frontier_workers, synthesize, Swarm, SwarmMode, SwarmOptions, SwarmRun, WorkerKind};

/// Ranked preference for the synthesizer/orchestrator: strong general+coding
/// reasoners first. We pick the best one that's actually available (has a key),
/// matched by substring against the worker roster.
const SYNTH_RANK: [&str; 6] = LEAD_RANK;

/// The strongest available FRONTIER model to lead synthesis, or `None` if none.
fn best_frontier(env: &SwarmEnv) -> Option<SharedProvider> {
    let workers = frontier_workers(env, &SwarmOptions::default());
    for pref in SYNTH_RANK {
        if let Some(w) = workers.iter().find(|x| x.model.to_lowercase().contains(pref)) {
            return Some(Arc::clone(&w.provider));
        }
    }
    // Any frontier beats a small local primary.
    workers.first().map(|w| Arc::clone(&w.provider))
}

/// Resolve the synthesizer/judge provider. "auto" ALWAYS uses the best available
/// frontier model so a small local primary (e.g. qwen2.5-coder:7b) never does the
/// final data synthesis — that's where swarm quality is won or lost. "primary"
/// forces the configured chat model; an explicit ref forces that model; "auto"
/// falls back to the primary only when no frontier worker is configured.
pub fn lead_provider(env: &SwarmEnv) -> SharedProvider {
    let raw = env.swarm.synthesizer.trim();
    let raw = if raw.is_empty() { "auto" } else { raw };
    let key = raw.to_lowercase();
    if key == "primary" {
        return env.factory.primary();
    }
    if key != "auto" {
        // Backend prefixes match case-insensitively; the model id keeps its case.
        // (lmstudio: is honoured too — the TS handed it to Ollama verbatim.)
        for (prefix, kind) in
            [("nvidia:", WorkerKind::Nvidia), ("openrouter:", WorkerKind::OpenRouter), ("lmstudio:", WorkerKind::LmStudio)]
        {
            if raw.get(..prefix.len()).is_some_and(|p| p.eq_ignore_ascii_case(prefix)) {
                return env.factory.create(kind, &raw[prefix.len()..]);
            }
        }
        return env.factory.create(WorkerKind::Ollama, raw);
    }
    best_frontier(env).unwrap_or_else(|| env.factory.primary())
}

fn secs(ms: u64) -> String {
    format!("{:.1}", ms as f64 / 1000.0)
}

/// Format a plain ensemble/divide run: each worker's answer plus the synthesis.
pub fn format_run(run: &SwarmRun) -> String {
    let mut lines: Vec<String> = Vec::new();
    let ok_count = run.results.iter().filter(|r| r.ok).count();
    lines.push(format!("🜂 agent swarm — {} · {} worker(s), {ok_count} ok", run.mode.as_str(), run.results.len()));
    for r in &run.results {
        lines.push(format!("\n[{}] {}s{}", r.model, secs(r.ms), if r.ok { "" } else { " — FAILED" }));
        if run.mode == SwarmMode::Divide {
            lines.push(format!("  · subtask: {}", r.task));
        }
        lines.push(if r.ok { r.text.clone() } else { format!("  error: {}", r.error.as_deref().unwrap_or("")) });
    }
    if let Some(s) = &run.synthesis {
        let by = run.synthesized_by.as_deref().map(|m| format!(" (by {m})")).unwrap_or_default();
        lines.push(format!("\n=== synthesized answer{by} ==="));
        lines.push(s.clone());
    }
    lines.join("\n")
}

/// Format a coordinated run: the plan/subtask results plus the integrated answer.
pub fn format_coordinated(run: &SwarmRun) -> String {
    let mut lines: Vec<String> = Vec::new();
    let ok_count = run.results.iter().filter(|r| r.ok).count();
    lines.push(format!("🜂 coordinated swarm — {} subtask(s), {ok_count} ok", run.results.len()));
    for r in &run.results {
        let deps = if r.depends_on.is_empty() { String::new() } else { format!(" (after {})", r.depends_on.join(", ")) };
        lines.push(format!(
            "\n[{}] {} — {} {}s{deps}{}",
            r.id.as_deref().unwrap_or("?"),
            r.title.as_deref().unwrap_or(""),
            r.model,
            secs(r.ms),
            if r.ok { "" } else { " — FAILED" }
        ));
        lines.push(if r.ok { r.text.clone() } else { format!("  error: {}", r.error.as_deref().unwrap_or("")) });
    }
    if let Some(s) = &run.synthesis {
        let by = run.synthesized_by.as_deref().map(|m| format!(" (by {m})")).unwrap_or_default();
        lines.push(format!("\n=== integrated answer{by} ==="));
        lines.push(s.clone());
    }
    lines.join("\n")
}

/// The model-facing schema for `agent_swarm` (SWARM_TOOL_SPECS in chatTools.ts).
/// Advertise it when ≥2 frontier workers (cloud keys) are available.
pub fn swarm_tool_specs() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: "agent_swarm".to_string(),
        description: "Spin up a SWARM of frontier models that run IN PARALLEL — each worker is a different cloud \
model on its own API key. Use for genuinely COMPLEX tasks. Modes: 'coordinate' (default) — a \
lead model splits the task into dependency-aware subtasks and the agents SHARE A BLACKBOARD so \
each sees its dependencies' results and builds on them (best for multi-part jobs that need to \
fit together); 'ensemble' — every model answers the SAME task and the answers are merged (best \
for hard reasoning/design where multiple perspectives help). For simple questions answer \
directly instead. Returns the plan, each subtask's result, and one integrated answer."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The overall task/question for the swarm (always required — it frames the whole run)."
                },
                "mode": {
                    "type": "string",
                    "enum": ["coordinate", "ensemble", "divide"],
                    "description": "'coordinate' (default): a lead model decomposes the task; agents share results via a blackboard and run in dependency order. \
'ensemble': every model answers the SAME task, then answers are merged. \
'divide': like coordinate but you supply `subtasks` yourself (skips auto-decomposition)."
                },
                "subtasks": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Your own subtasks (used by mode 'divide'; optional otherwise — the lead will decompose)."
                },
                "synthesize": {
                    "type": "boolean",
                    "description": "Merge the workers' outputs into one final answer (default true)."
                }
            },
            "required": ["task"]
        }),
    }]
}

/// The chat session's swarm gate (chat.ts `swarmReady`): the swarm is usable only
/// when `[swarm].enabled` and at least two frontier workers (cloud models with
/// keys) exist. Returns that roster (for the "agent swarm ready — N frontier
/// models: …" line), or an empty list when the tool should not be offered.
pub fn swarm_ready_roster(env: &SwarmEnv) -> Vec<crate::swarm::RosterEntry> {
    if !env.swarm.enabled {
        return Vec::new();
    }
    let workers = frontier_workers(
        env,
        &SwarmOptions { max_workers: Some(env.swarm.max_workers), include_local: env.swarm.include_local, ..Default::default() },
    );
    if workers.len() < 2 {
        return Vec::new();
    }
    workers
        .iter()
        .map(|w| crate::swarm::RosterEntry { label: w.label.clone(), model: w.model.clone(), backend: w.kind.as_str().to_string() })
        .collect()
}

/// The `agent_swarm` tool.
pub struct AgentSwarmTool {
    env: Arc<SwarmEnv>,
}

impl AgentSwarmTool {
    pub fn new(env: Arc<SwarmEnv>) -> Self {
        Self { env }
    }
}

/// The swarm tool set for a chat session's registry (see `qo_swarm::create_swarm_tools`
/// for the `&Config` entry point).
pub fn create_swarm_tools_with_env(env: Arc<SwarmEnv>) -> Vec<SharedTool> {
    vec![Arc::new(AgentSwarmTool::new(env))]
}

/// JavaScript `String(x)` for a tool argument (`null`/missing → "").
fn arg_string(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

#[async_trait]
impl Tool for AgentSwarmTool {
    fn name(&self) -> &str {
        "agent_swarm"
    }

    fn description(&self) -> &str {
        "Run a swarm of frontier models in parallel for a complex task. Each worker is a different \
cloud model on its own API key, all executing at the same time."
    }

    fn mutating(&self) -> bool {
        false
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let env = &self.env;
        let task = arg_string(args.get("task")).trim().to_string();
        if task.is_empty() {
            return ToolResult::err("agent_swarm: 'task' is required.");
        }
        let do_synth = args.get("synthesize") != Some(&Value::Bool(false)); // default true
        // Default mode honors [swarm].coordinate; an explicit mode overrides.
        let coordinate = match args.get("mode").and_then(Value::as_str) {
            Some("ensemble") => false,
            Some("divide") | Some("coordinate") => true,
            _ => env.swarm.coordinate,
        };
        let interrupted_note = |out: String| {
            if ctx.cancelled() {
                format!("{out}\n\n(interrupted — partial results)")
            } else {
                out
            }
        };

        // ── Coordinated: lead splits the task, agents share a blackboard ──
        if coordinate {
            let Some(built) = create_coordinated_swarm(Arc::clone(env), None) else {
                return ToolResult::err(
                    "agent_swarm: no workers available. Configure cloud models in model.fallback_models \
                     (e.g. nvidia:… / openrouter:…) and set the matching API key(s), or a local primary.",
                );
            };
            let presupplied: Option<Vec<Subtask>> = args.get("subtasks").and_then(Value::as_array).map(|arr| {
                arr.iter()
                    .map(|s| arg_string(Some(s)))
                    .filter(|s| !s.trim().is_empty())
                    .enumerate()
                    .map(|(i, s)| Subtask::supplied(i, &s))
                    .collect()
            });
            // Agents may run commands only when this chat session itself is allowed
            // to (allow_shell) and the config enables swarm exec.
            let exec_mode = if ctx.allow_shell { env.swarm.exec } else { ExecMode::Off };
            ctx.log(json!({
                "tool": "agent_swarm",
                "mode": "coordinate",
                "local": built.local,
                "exec": exec_mode.as_str(),
                "workers": built.swarm.roster().iter().map(|r| r.model.clone()).collect::<Vec<_>>(),
            }));
            let run = built
                .swarm
                .run(
                    &task,
                    CoordinateOptions {
                        synthesize: do_synth,
                        subtasks: presupplied,
                        cancel: ctx.cancel.clone(),
                        exec_mode,
                        bare: BareExecutorOptions {
                            cwd: Some(ctx.cwd.clone()),
                            allow_commands: ctx.allow_commands.clone(),
                            deny_commands: ctx.deny_commands.clone(),
                            default_timeout_ms: Some(env.swarm.exec_timeout_s * 1000),
                        },
                        ..Default::default()
                    },
                )
                .await;
            let any_ok = run.results.iter().any(|r| r.ok);
            let out = interrupted_note(format_coordinated(&run));
            return ToolResult { ok: any_ok, output: out, data: serde_json::to_value(&run).ok() };
        }

        // ── Ensemble: every model answers the same task, then merge ──
        let opts = SwarmOptions {
            max_workers: Some(env.swarm.max_workers),
            include_local: env.swarm.include_local,
            max_tokens: Some(env.swarm.max_tokens),
            cancel: ctx.cancel.clone(),
        };
        let workers = frontier_workers(env, &opts);
        if workers.is_empty() {
            return ToolResult::err(
                "agent_swarm: no frontier workers available. Configure cloud models in model.fallback_models \
                 (e.g. nvidia:… / openrouter:…) and set the matching API key(s).",
            );
        }
        let swarm = Swarm::new(workers, opts);
        ctx.log(json!({
            "tool": "agent_swarm",
            "mode": "ensemble",
            "workers": swarm.roster().iter().map(|r| r.model.clone()).collect::<Vec<_>>(),
        }));
        let mut run = swarm.ensemble(&task, None).await;
        if do_synth && run.results.iter().any(|r| r.ok) && !ctx.cancelled() {
            let lead = lead_provider(env);
            run.synthesis =
                Some(synthesize(lead.as_ref(), &task, &run, env.swarm.max_tokens, ctx.cancel.as_ref()).await);
            run.synthesized_by = Some(lead.model());
        }
        let any_ok = run.results.iter().any(|r| r.ok);
        let out = interrupted_note(format_run(&run));
        ToolResult { ok: any_ok, output: out, data: serde_json::to_value(&run).ok() }
    }
}
