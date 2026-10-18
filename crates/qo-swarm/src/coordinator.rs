//! Coordinated agent swarm: instead of fanning the SAME task out to many models
//! (ensemble) or blindly round-robining pre-split subtasks (divide), this drives a
//! TEAM of frontier agents that share context and build on each other's work:
//!
//!   1. PLAN     — a lead model decomposes the task into dependency-aware subtasks
//!                 (`{ id, title, detail, dependsOn[] }`). Accurate splitting, by a
//!                 model that sees the whole task, not a naive shard.
//!   2. WAVES    — subtasks run in topological order. Every subtask whose deps are
//!                 already done runs in PARALLEL (one per worker); when a wave
//!                 finishes, its results land on a shared BLACKBOARD.
//!   3. CONTEXT  — each agent's prompt is injected with its dependencies' results
//!                 plus a roster of what teammates are doing, so later agents build
//!                 on earlier ones instead of working blind (real coordination).
//!   4. INTEGRATE— a lead model synthesizes the whole blackboard into one result.
//!
//! Roster policy: FRONTIER-FIRST. Workers (and the lead) are frontier cloud models
//! when any API key is configured; only when NO frontier model is available does the
//! roster fall back to local models, and finally to the configured primary — so the
//! swarm degrades gracefully on a keyless machine instead of failing outright.
//!
//! The coordinator is UI-agnostic: it emits [`SwarmEvent`]s (plan/assign/delta/
//! result/…) that a TUI (or plain logger) subscribes to. No terminal code lives here.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use futures::future::join_all;
use qo_core::{CancellationToken, Message, ModelOptions, ProviderError, SharedProvider, ToolCall, ToolSpec};
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::env::SwarmEnv;
use crate::events::{
    AssignEvent, DeltaEvent, ExecCmdEvent, PlanEvent, PreflightResult, StatusEvent, SwarmEvent, SwarmEvents, WaveEvent,
};
use crate::exec::{choose_executor, BareExecutorOptions, ExecMode, RunOptions, SharedExecutor};
pub use crate::exec::PlanComplexity;
use crate::swarm::{
    elapsed_ms, frontier_workers, is_reasoning_model, short_label, strip_thinking, synthesize, RosterEntry, SwarmMode,
    SwarmOptions, SwarmRun, SwarmWorker, WorkerKind, WorkerResult,
};

/* ───────────────────────── Swarm-agent system prompts ────────────────────────
 * These are deliberately NOT the full Qwenodyssey coding-agent harness prompt: a
 * swarm agent has no tools and no edit loop, it just produces a deliverable. Using
 * the harness prompt makes models emit <tool> call syntax as text and behave like
 * they're in a file-editing session — so the swarm uses focused, tool-free prompts.
 *
 * The rigor below ADAPTS the orchestration doctrine from openclaw's AGENTS.md
 * (github.com/openclaw/openclaw) — evidence-driven, exhaustive, best-fix work;
 * complete production-grade output; verify-when-feasible; no memory-only guesses —
 * to push small models toward more accurate, fleshed-out code. (Design ideas only;
 * no source copied.)
 */

/// Shared standards every swarm agent is held to (openclaw-AGENTS.md-inspired).
const SWARM_STANDARDS: &str = "Standards:\n\
- Deliver COMPLETE, production-grade work: full implementations with edge cases, error handling, input \
validation, and sensible defaults. NO stubs, NO TODOs, NO placeholders, NO '...'.\n\
- Be the BEST solution, not merely a plausible one: choose the approach a senior engineer would defend.\n\
- Ground every decision in the task and the shared context, not assumptions or memory. If a fact is \
genuinely unknown, state the assumption explicitly instead of guessing silently.\n\
- Code must be runnable as written: correct imports, types, names, and signatures consistent with teammates' work.\n\
- Prefer correctness and completeness over brevity; include what's needed to actually use it.";

pub const SWARM_PLANNER_SYSTEM: &str = "You are the LEAD PLANNER of a team of expert AI engineers working in parallel. Break the task into a \
small set of concrete, separable subtasks whose UNION fully covers the task with NO gaps, then they \
integrate. Each subtask must name a clear, verifiable deliverable. Split along natural seams so subtasks \
run in parallel; use dependsOn only when one genuinely needs another's RESULT first; include an \
integration (and, for buildable work, a verification/test) subtask. You do NOT solve the task — you ONLY \
produce the plan as strict JSON. You have no tools.";

/// The text-only agent prompt (no execution backend attached).
pub fn swarm_agent_system() -> String {
    format!(
        "You are an expert AI engineer completing ONE part of a larger task as a member of a coordinated team. \
Produce the actual, finished deliverable for YOUR assigned subtask — the real content (design, code, \
prose, analysis), not a plan to do it later. You have NO tools and cannot run commands, browse, or edit \
files: never emit tool-call syntax (no <tool> tags, no function calls) — output only the finished result. \
When teammates' results are provided as shared context, build directly on them and do not redo their work.\n\n{SWARM_STANDARDS}"
    )
}

/// Variant used when an execution backend is attached to the run.
pub fn swarm_agent_exec_system(exec_label: &str) -> String {
    format!(
        "You are an expert AI engineer completing ONE part of a larger task as a member of a coordinated team. \
Produce the actual, finished deliverable for YOUR assigned subtask — real content, not a plan.\n\
You have exactly ONE tool: run_shell, which executes a shell command on {exec_label}. Use it to create \
the real files, install dependencies, build, and — critically — ACTUALLY RUN and VERIFY your work \
(execute the program, run the tests) before claiming success. Only state 'done' after verification passes. \
Workspace is SHARED with teammates: their files are already present; place yours alongside and don't clobber \
theirs. Keep commands non-interactive and non-destructive. Build on teammates' context; never redo their work. \
When done, answer with the final deliverable text and the paths of the files you created/verified.\n\n{SWARM_STANDARDS}"
    )
}

pub const SWARM_SYNTH_SYSTEM: &str = "You are the LEAD INTEGRATOR of a team of expert AI engineers. Merge the team's completed subtask results \
into ONE coherent, correct, COMPLETE deliverable that fully fulfills the overall task. Reconcile overlaps \
and contradictions, close any gaps between parts, and keep each part's substance. The result must be \
production-grade with no stubs, TODOs, or placeholders. Do not mention the team or the process. You have \
no tools; output only the final integrated result.";

/* ───────────────────────── Plan / blackboard types ───────────────────────── */

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Subtask {
    pub id: String,
    pub title: String,
    pub detail: String,
    /// ids of subtasks whose RESULTS this one needs before it can run.
    #[serde(rename = "dependsOn")]
    pub depends_on: Vec<String>,
}

impl Subtask {
    /// A dependency-free subtask `s{n}` from a user-supplied line (`--divide`,
    /// the tool's `subtasks`): title = first 60 chars, detail = the whole line.
    pub fn supplied(index: usize, text: &str) -> Self {
        Subtask {
            id: format!("s{}", index + 1),
            title: take_chars(text, 60),
            detail: text.to_string(),
            depends_on: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SubtaskStatus {
    #[default]
    Pending,
    Running,
    Done,
    Failed,
    Stopped,
}

impl SubtaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SubtaskStatus::Pending => "pending",
            SubtaskStatus::Running => "running",
            SubtaskStatus::Done => "done",
            SubtaskStatus::Failed => "failed",
            SubtaskStatus::Stopped => "stopped",
        }
    }

    /// Finished one way or another (dependents may run).
    pub fn resolved(&self) -> bool {
        matches!(self, SubtaskStatus::Done | SubtaskStatus::Failed | SubtaskStatus::Stopped)
    }
}

/// Live state of one subtask on the shared blackboard.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BoardEntry {
    pub id: String,
    pub title: String,
    pub detail: String,
    pub depends_on: Vec<String>,
    pub status: SubtaskStatus,
    /// Worker model that produced (or is producing) this result.
    pub assigned_to: Option<String>,
    pub result: String,
    pub error: Option<String>,
    pub ms: Option<u64>,
}

/* ──────────────────────────── Roster / lead ──────────────────────────────── */

/// Ranked preference for the lead (decomposer + synthesizer).
pub const LEAD_RANK: [&str; 6] =
    ["nemotron-3-ultra", "qwen3-coder-480b", "deepseek-v4", "kimi-k2", "llama-3.3-70b", "qwen3-coder"];

pub struct RosterResult {
    pub workers: Vec<SwarmWorker>,
    /// True when we fell back to local models (no frontier key available).
    pub local: bool,
}

/// Build the worker roster FRONTIER-FIRST. Frontier cloud workers when any key is
/// present; otherwise fall back to local refs from the fallback chain; otherwise a
/// single worker wrapping the configured primary provider (so a keyless, local-only
/// machine still runs the team, just on local models).
pub fn build_roster(env: &SwarmEnv, max_workers: Option<usize>) -> RosterResult {
    let cap = max_workers.unwrap_or(env.swarm.max_workers);
    let opts = |include_local| SwarmOptions {
        max_workers: Some(cap),
        include_local,
        max_tokens: Some(env.swarm.max_tokens),
        cancel: None,
    };
    let frontier = frontier_workers(env, &opts(false));
    if !frontier.is_empty() {
        return RosterResult { workers: frontier, local: false };
    }
    // No frontier key — fall back to local refs in the fallback chain.
    let local = frontier_workers(env, &opts(true));
    if !local.is_empty() {
        return RosterResult { workers: local, local: true };
    }
    // Nothing configured — wrap the active primary as a lone worker.
    let primary = env.factory.primary();
    let model = primary.model();
    RosterResult {
        workers: vec![SwarmWorker {
            model_ref: model.clone(),
            kind: WorkerKind::from_provider_name(primary.name()),
            label: short_label(&model),
            model,
            provider: primary,
        }],
        local: true,
    }
}

/// Pick the lead (decomposer + synthesizer): strongest frontier worker, else the primary.
pub fn pick_lead(env: &SwarmEnv, workers: &[SwarmWorker]) -> SharedProvider {
    for pref in LEAD_RANK {
        if let Some(w) = workers.iter().find(|x| x.model.to_lowercase().contains(pref)) {
            return Arc::clone(&w.provider);
        }
    }
    match workers.first() {
        Some(w) => Arc::clone(&w.provider),
        None => env.factory.primary(),
    }
}

/// Default per-worker health-check timeout.
pub const PREFLIGHT_TIMEOUT_MS: u64 = 15_000;

/// Health-check every worker's provider in PARALLEL before the run, so a broken or
/// expired API key / dead endpoint is caught up front instead of failing a subtask
/// mid-flight. Each provider's `health_check()` hits its /models (or equivalent)
/// WITH the configured key, so a 401/403 or network error surfaces here.
pub async fn preflight_workers(
    workers: &[SwarmWorker],
    on_result: Option<&(dyn Fn(&PreflightResult) + Send + Sync)>,
    timeout_ms: u64,
    cancel: Option<&CancellationToken>,
) -> Vec<PreflightResult> {
    let never = CancellationToken::new();
    let cancel = cancel.unwrap_or(&never);
    join_all(workers.iter().enumerate().map(|(pane, w)| async move {
        let start = Instant::now();
        let check = tokio::time::timeout(Duration::from_millis(timeout_ms), w.provider.health_check());
        let (ok, detail) = tokio::select! {
            r = check => match r {
                Ok(h) => (h.ok, h.detail),
                Err(_) => (false, Some(format!("health check timed out after {timeout_ms}ms"))),
            },
            _ = cancel.cancelled() => (false, Some("interrupted".to_string())),
        };
        let r = PreflightResult {
            pane,
            label: w.label.clone(),
            model: w.model.clone(),
            backend: w.kind.as_str().to_string(),
            ok,
            detail,
            ms: elapsed_ms(start),
        };
        if let Some(f) = on_result {
            f(&r);
        }
        r
    }))
    .await
}

/* ──────────────────────────── Decomposition ──────────────────────────────── */

const PLAN_SCHEMA_HINT: &str = "Return ONLY JSON of the form {\"complexity\":\"simple\"|\"complex\",\"subtasks\":[{\"id\":\"slug\",\"title\":\"…\",\"detail\":\"…\",\"dependsOn\":[\"otherId\"]}]}. \
\"complex\" = the work needs real project execution (installing dependencies, building/running a multi-file \
codebase, long-running processes or services); \"simple\" = analysis, writing, design, or single-file code.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecomposeResult {
    pub subtasks: Vec<Subtask>,
    pub planned_by: String,
    pub note: Option<String>,
    pub complexity: PlanComplexity,
}

static COMPLEXITY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)"complexity"\s*:\s*"(simple|complex)""#).expect("static regex"));

/// Read the planner's complexity judgment out of its JSON (defaults to "simple").
pub fn parse_complexity(text: &str) -> PlanComplexity {
    match COMPLEXITY_RE.captures(text).and_then(|c| c.get(1)) {
        Some(m) if m.as_str().eq_ignore_ascii_case("complex") => PlanComplexity::Complex,
        _ => PlanComplexity::Simple,
    }
}

/// Ask a lead model to split the task into dependency-aware subtasks sized for the
/// team. RESILIENT: tries each candidate planner in order (a slow/timing-out lead —
/// e.g. nemotron-3-ultra at 90s — must not silently collapse the whole run into one
/// subtask). Only when EVERY candidate fails does it degrade to a single subtask
/// that is the whole task, with a note saying so.
pub async fn decompose_with(
    candidates: &[SharedProvider],
    task: &str,
    worker_count: usize,
    max_tokens: u32,
    cancel: Option<&CancellationToken>,
    on_attempt: &mut (dyn FnMut(&str) + Send),
) -> DecomposeResult {
    let mut errors: Vec<String> = Vec::new();
    for lead in candidates {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            break;
        }
        let model = lead.model();
        on_attempt(&model);
        match decompose_once(lead, task, worker_count, max_tokens, cancel).await {
            Ok((subtasks, complexity)) if !subtasks.is_empty() => {
                let note = if errors.is_empty() {
                    None
                } else {
                    Some(format!("planner fell back to {model} ({})", errors.join("; ")))
                };
                return DecomposeResult { subtasks, complexity, planned_by: model, note };
            }
            Ok(_) => errors.push(format!("{model}: returned no parseable plan")),
            Err(e) => errors.push(format!("{model}: {e:#}")),
        }
    }
    DecomposeResult {
        subtasks: vec![Subtask {
            id: "task".to_string(),
            title: take_chars(task, 60),
            detail: task.to_string(),
            depends_on: Vec::new(),
        }],
        planned_by: "(fallback)".to_string(),
        complexity: PlanComplexity::Simple,
        note: Some(format!("decomposition failed — running as a single task ({})", errors.join("; "))),
    }
}

/// Single-lead wrapper around [`decompose_with`].
pub async fn decompose(
    lead: SharedProvider,
    task: &str,
    worker_count: usize,
    max_tokens: u32,
    cancel: Option<&CancellationToken>,
) -> Vec<Subtask> {
    decompose_with(&[lead], task, worker_count, max_tokens, cancel, &mut |_| {}).await.subtasks
}

async fn decompose_once(
    lead: &SharedProvider,
    task: &str,
    worker_count: usize,
    max_tokens: u32,
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<(Vec<Subtask>, PlanComplexity)> {
    let target = (worker_count * 2).clamp(3, 8);
    let prompt = format!(
        "Decompose this task for a team of {worker_count} expert agent(s) working in PARALLEL.\n\
Aim for {}-{target} subtasks.\n\n\
TASK:\n{task}\n\n\
Rules:\n\
- Split into the natural, separable parts of the task (e.g. distinct components, layers, files, or questions). If the task names several parts, make each its own subtask.\n\
- The subtasks' UNION must FULLY cover the task with no gaps. Include an integration subtask, and a verification/test subtask for buildable work.\n\
- Each subtask's \"detail\" names a CONCRETE, VERIFIABLE deliverable (what file/output proves it's done).\n\
- Return AT LEAST 2 subtasks unless the task is genuinely atomic and cannot be divided.\n\
- Each subtask is concrete, self-contained, and independently executable by one agent.\n\
- Use \"dependsOn\" to list the ids of subtasks whose RESULTS a subtask needs first (foundational work has no deps). This is how the team shares context.\n\
- Prefer breadth: subtasks that CAN run in parallel should have NO dependency between them.\n\
- ids are short kebab-case slugs, unique.\n\
- Do NOT solve the task here; only plan the split.\n\n{PLAN_SCHEMA_HINT}",
        target.min(3)
    );
    let messages = vec![Message::system(SWARM_PLANNER_SYSTEM), Message::user(prompt)];
    let opts = ModelOptions {
        temperature: Some(0.2),
        max_tokens: Some(max_tokens.min(1200)),
        json: true,
        cancel: cancel.cloned(),
        ..Default::default()
    };
    let res = lead.generate(&messages, &opts).await?;
    let clean = strip_thinking(&res.text);
    Ok((parse_plan(&clean), parse_complexity(&clean)))
}

static WS_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("static regex"));

/// JavaScript `String(v)` for the JSON shapes a planner may emit.
fn js_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => a.iter().map(|x| if x.is_null() { String::new() } else { js_string(x) }).collect::<Vec<_>>().join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

/// `a ?? b ?? …` over object fields: the first present, non-null one.
fn first_field<'a>(obj: Option<&'a Map<String, Value>>, keys: &[&str]) -> Option<&'a Value> {
    let obj = obj?;
    keys.iter().find_map(|k| obj.get(*k).filter(|v| !v.is_null()))
}

fn slug(s: &str) -> String {
    WS_RUN.replace_all(&s.trim().to_lowercase(), "-").into_owned()
}

/// Extract a subtasks array from a model response that may wrap JSON in prose/fences.
pub fn parse_plan(text: &str) -> Vec<Subtask> {
    let Some(raw) = extract_json(text) else {
        return Vec::new();
    };
    let Ok(obj) = serde_json::from_str::<Value>(raw) else {
        return Vec::new();
    };
    let arr: &[Value] = match &obj {
        Value::Array(a) => a,
        Value::Object(o) => match o.get("subtasks") {
            Some(Value::Array(a)) => a,
            _ => &[],
        },
        _ => &[],
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Subtask> = Vec::new();
    for (i, s) in arr.iter().enumerate() {
        let o = s.as_object();
        let fallback_id = format!("s{}", i + 1);
        let mut id = match first_field(o, &["id", "slug"]) {
            Some(v) => slug(&js_string(v)),
            None => fallback_id.clone(),
        };
        if id.is_empty() || seen.contains(&id) {
            id = fallback_id;
        }
        seen.insert(id.clone());
        let title = match first_field(o, &["title", "name"]) {
            Some(v) => take_chars(js_string(v).trim(), 80),
            None => take_chars(&id, 80),
        };
        let detail = match first_field(o, &["detail", "description", "task"]) {
            Some(v) => js_string(v).trim().to_string(),
            None => title.trim().to_string(),
        };
        let depends_on = match first_field(o, &["dependsOn", "deps"]) {
            Some(Value::Array(a)) => a.iter().map(|d| slug(&js_string(d))).filter(|d| !d.is_empty()).collect(),
            _ => Vec::new(),
        };
        if !detail.is_empty() {
            out.push(Subtask { id, title, detail, depends_on });
        }
    }
    // Drop dangling deps (referencing ids that don't exist) so scheduling can't deadlock.
    let ids: HashSet<String> = out.iter().map(|s| s.id.clone()).collect();
    for s in &mut out {
        let me = s.id.clone();
        s.depends_on.retain(|d| ids.contains(d) && *d != me);
    }
    out
}

/// Find the first balanced `{…}` or `[…]` block in a string.
fn extract_json(text: &str) -> Option<&str> {
    let start = text.find(['{', '['])?;
    let bytes = text.as_bytes();
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        if c == b'"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(&text[start..=i]);
            }
        }
    }
    None
}

/* ───────────── model-output cleanup / non-standard (Kimi) tool calls ────────── */

static SPECIAL_CALL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"<\|tool_call_begin\|>\s*(?:functions\.)?([A-Za-z0-9_.\-]+?)(?::(\d+))?\s*<\|tool_call_argument_begin\|>\s*(\{[\s\S]*?\})\s*<\|tool_call_end\|>",
    )
    .expect("static regex")
});

/// Some hosted models — notably Kimi K2.x on NVIDIA NIM — emit tool calls as SPECIAL
/// TOKENS inside the text instead of OpenAI-style `tool_calls`:
///
/// ```text
/// <|tool_calls_section_begin|><|tool_call_begin|>functions.run_shell:5
/// <|tool_call_argument_begin|>{"command":"..."}<|tool_call_end|><|tool_calls_section_end|>
/// ```
///
/// Without parsing, these leak as raw text AND the commands never run. Convert them
/// to real ToolCalls so the exec loop executes them.
pub fn parse_special_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls: Vec<ToolCall> = Vec::new();
    for m in SPECIAL_CALL.captures_iter(text) {
        let raw_name = m.get(1).map_or("", |x| x.as_str());
        let name = raw_name.rsplit('.').next().unwrap_or(raw_name).to_string();
        let raw_args = m.get(3).map_or("{}", |x| x.as_str());
        let arguments = match serde_json::from_str::<Value>(raw_args) {
            Ok(v @ Value::Object(_)) => v,
            _ => json!({ "_raw": raw_args }),
        };
        let id = match m.get(2) {
            Some(idx) => format!("special_{}", idx.as_str()),
            None => format!("special_{}", calls.len()),
        };
        calls.push(ToolCall { id, name, arguments });
    }
    calls
}

static TOOL_SECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<\|tool_calls_section_begin\|>[\s\S]*?<\|tool_calls_section_end\|>").expect("static regex")
});
static CONTROL_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\|[a-zA-Z0-9_]+\|>").expect("static regex"));

/// Strip a model's chain-of-thought and any special control/tool tokens so streamed
/// and stored agent text stays human-readable (no `<|tool_call…|>` noise in panes).
pub fn clean_agent_text(text: &str) -> String {
    let s = strip_thinking(text);
    let s = TOOL_SECTION.replace_all(&s, "");
    let s = CONTROL_TOKEN.replace_all(&s, "");
    s.trim_end().to_string()
}

/* ─────────────────────────── Shared blackboard ───────────────────────────── */

/// The shared store every agent reads context from and writes results to.
#[derive(Clone, Debug, Default)]
pub struct Blackboard {
    entries: Vec<BoardEntry>,
    index: HashMap<String, usize>,
}

impl Blackboard {
    pub fn new(subtasks: &[Subtask]) -> Self {
        let mut b = Blackboard::default();
        for s in subtasks {
            let entry = BoardEntry {
                id: s.id.clone(),
                title: s.title.clone(),
                detail: s.detail.clone(),
                depends_on: s.depends_on.clone(),
                ..Default::default()
            };
            match b.index.get(&s.id) {
                // Same id twice: the later one wins (Map.set semantics), keeping its slot.
                Some(&i) => b.entries[i] = entry,
                None => {
                    b.index.insert(s.id.clone(), b.entries.len());
                    b.entries.push(entry);
                }
            }
        }
        b
    }

    pub fn get(&self, id: &str) -> Option<&BoardEntry> {
        self.index.get(id).map(|&i| &self.entries[i])
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut BoardEntry> {
        match self.index.get(id) {
            Some(&i) => Some(&mut self.entries[i]),
            None => None,
        }
    }

    pub fn all(&self) -> &[BoardEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Subtasks whose deps are all resolved (done/failed/stopped) and which are still pending.
    pub fn ready(&self) -> Vec<&BoardEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.status == SubtaskStatus::Pending
                    && e.depends_on.iter().all(|d| self.get(d).map_or(true, |dep| dep.status.resolved()))
            })
            .collect()
    }

    pub fn pending(&self) -> Vec<&BoardEntry> {
        self.entries.iter().filter(|e| e.status == SubtaskStatus::Pending).collect()
    }

    /// Count a subtask's dependencies that aren't yet resolved.
    pub fn unmet(&self, e: &BoardEntry) -> usize {
        e.depends_on.iter().filter(|d| self.get(d).is_some_and(|dep| !dep.status.resolved())).count()
    }

    /// The context block injected into a worker's prompt: the results its dependencies
    /// already produced (so it builds on them), plus a compact roster of every other
    /// subtask's status (so it understands the whole plan and doesn't duplicate work).
    pub fn digest_for(&self, id: &str, per_dep_chars: usize) -> String {
        let Some(me) = self.get(id) else {
            return String::new();
        };
        let mut parts: Vec<String> = Vec::new();

        let deps: Vec<&BoardEntry> = me.depends_on.iter().filter_map(|d| self.get(d)).collect();
        let done: Vec<&&BoardEntry> =
            deps.iter().filter(|d| d.status == SubtaskStatus::Done && !d.result.is_empty()).collect();
        if !done.is_empty() {
            parts.push("RESULTS FROM YOUR DEPENDENCIES (build on these — do not redo them):".to_string());
            for d in done {
                parts.push(format!("\n### [{}] {}\n{}", d.id, d.title, truncate_marked(&d.result, per_dep_chars)));
            }
        }
        let failed: Vec<&&BoardEntry> =
            deps.iter().filter(|d| matches!(d.status, SubtaskStatus::Failed | SubtaskStatus::Stopped)).collect();
        if !failed.is_empty() {
            parts.push(format!(
                "\nNOTE — these dependencies did not finish, so their output is missing; compensate as best you can:\n{}",
                failed.iter().map(|d| format!("- [{}] {} ({})", d.id, d.title, d.status.as_str())).collect::<Vec<_>>().join("\n")
            ));
        }

        let others: Vec<&BoardEntry> = self.entries.iter().filter(|e| e.id != id).collect();
        if !others.is_empty() {
            parts.push(format!(
                "\nTEAM PLAN (other agents' subtasks — for awareness; do NOT do their work):\n{}",
                others.iter().map(|e| format!("- [{}] {} — {}", e.id, e.title, e.status.as_str())).collect::<Vec<_>>().join("\n")
            ));
        }
        parts.join("\n")
    }
}

/// Default per-dependency excerpt length in [`Blackboard::digest_for`].
pub const DIGEST_DEP_CHARS: usize = 1400;

fn truncate_marked(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}\n…[truncated]", &s[..i]),
        None => s.to_string(),
    }
}

pub(crate) fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/* ───────────────────────────── Coordinator ───────────────────────────────── */

/// Knobs for one [`CoordinatedSwarm::run`].
#[derive(Clone)]
pub struct CoordinateOptions {
    /// Run the final synthesis step (default true); false returns just the per-subtask results.
    pub synthesize: bool,
    /// Pre-supplied subtasks (e.g. from --divide) — skips the decompose call when non-empty.
    pub subtasks: Option<Vec<Subtask>>,
    /// Esc / Ctrl-C for the whole run (every pane's token is a child of it).
    pub cancel: Option<CancellationToken>,
    /// Where agents may run commands: off (default, text-only), auto, bare, daytona.
    pub exec_mode: ExecMode,
    /// Host-side settings for the bare executor (cwd, allow/deny lists).
    pub bare: BareExecutorOptions,
    /// Injectable executor (tests). Overrides `exec_mode` routing.
    pub executor: Option<SharedExecutor>,
    /// Health-check the roster's API keys before running (`None` = config default).
    pub preflight: Option<bool>,
}

impl Default for CoordinateOptions {
    fn default() -> Self {
        Self {
            synthesize: true,
            subtasks: None,
            cancel: None,
            exec_mode: ExecMode::Off,
            bare: BareExecutorOptions::default(),
            executor: None,
            preflight: None,
        }
    }
}

/// Disposes the run's executor even when the run future is dropped mid-flight
/// (the TS `finally`): if `dispose` wasn't awaited, it is spawned on drop.
struct DisposeGuard(Option<SharedExecutor>);

impl DisposeGuard {
    async fn dispose(mut self) {
        if let Some(e) = self.0.take() {
            e.dispose().await;
        }
    }
}

impl Drop for DisposeGuard {
    fn drop(&mut self) {
        if let Some(e) = self.0.take() {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move { e.dispose().await });
            }
        }
    }
}

/// Text accumulated for one pane, mirrored to the event stream as it arrives.
struct PaneSink<'a> {
    events: &'a SwarmEvents,
    pane: usize,
    subtask_id: String,
    text: String,
}

impl PaneSink<'_> {
    fn delta(&mut self, d: &str) {
        if d.is_empty() {
            return;
        }
        self.text.push_str(d);
        self.events.emit(SwarmEvent::Delta(DeltaEvent { pane: self.pane, subtask_id: self.subtask_id.clone(), delta: d.to_string() }));
    }
}

struct LoopOutput {
    text: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

/// A coordinated team run. Share it as `Arc<CoordinatedSwarm>`: `run` takes
/// `&self`, so the UI can call [`CoordinatedSwarm::stop_pane`] while it runs.
pub struct CoordinatedSwarm {
    env: Arc<SwarmEnv>,
    workers: Mutex<Vec<SwarmWorker>>,
    lead: SharedProvider,
    max_tokens: u32,
    events: SwarmEvents,
    /// Per-pane stop tokens so a single agent can be stopped (Esc) mid-run.
    pane_tokens: Mutex<HashMap<usize, (u64, CancellationToken)>>,
    next_token_id: AtomicU64,
}

impl CoordinatedSwarm {
    pub fn new(env: Arc<SwarmEnv>, roster: Vec<SwarmWorker>, lead: SharedProvider, max_tokens: Option<u32>) -> Self {
        let max_tokens = max_tokens.unwrap_or(env.swarm.max_tokens);
        Self {
            env,
            workers: Mutex::new(roster),
            lead,
            max_tokens,
            events: SwarmEvents::new(),
            pane_tokens: Mutex::new(HashMap::new()),
            next_token_id: AtomicU64::new(1),
        }
    }

    pub fn events(&self) -> &SwarmEvents {
        &self.events
    }

    /// Shorthand for `events().subscribe()`.
    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<SwarmEvent> {
        self.events.subscribe()
    }

    pub fn size(&self) -> usize {
        self.workers_snapshot().len()
    }

    pub fn lead_model(&self) -> String {
        self.lead.model()
    }

    pub fn roster(&self) -> Vec<RosterEntry> {
        self.workers_snapshot()
            .iter()
            .map(|w| RosterEntry { label: w.label.clone(), model: w.model.clone(), backend: w.kind.as_str().to_string() })
            .collect()
    }

    /// Stop a single agent's in-flight work (its stream aborts → subtask "stopped").
    pub fn stop_pane(&self, pane: usize) {
        if let Some((_, t)) = lock(&self.pane_tokens).get(&pane) {
            t.cancel();
        }
    }

    fn workers_snapshot(&self) -> Vec<SwarmWorker> {
        lock(&self.workers).clone()
    }

    /// Health-check the roster's keys; emits preflight events and DROPS broken workers
    /// (keeping all of them if every one fails, so the run can still surface the
    /// error). Returns the results. Safe to call before `run`.
    pub async fn preflight(&self, cancel: Option<&CancellationToken>) -> Vec<PreflightResult> {
        let workers = self.workers_snapshot();
        self.events.emit(SwarmEvent::PreflightStart { count: workers.len() });
        let events = &self.events;
        let emit = move |r: &PreflightResult| events.emit(SwarmEvent::Preflight(r.clone()));
        let results = preflight_workers(&workers, Some(&emit), PREFLIGHT_TIMEOUT_MS, cancel).await;
        let ok_idx: HashSet<usize> = results.iter().filter(|r| r.ok).map(|r| r.pane).collect();
        if !ok_idx.is_empty() && ok_idx.len() < workers.len() {
            let kept: Vec<SwarmWorker> =
                workers.into_iter().enumerate().filter(|(i, _)| ok_idx.contains(i)).map(|(_, w)| w).collect();
            *lock(&self.workers) = kept;
        }
        let ok = results.iter().filter(|r| r.ok).count();
        self.events.emit(SwarmEvent::PreflightDone { ok, broken: results.len() - ok });
        results
    }

    /// Run the full coordinated flow: preflight → plan → waves → synthesize.
    /// Never fails: per-subtask failures become failed results.
    pub async fn run(&self, task: &str, opts: CoordinateOptions) -> SwarmRun {
        let cancel = opts.cancel.clone().unwrap_or_default();

        // Validate each agent's API key/endpoint up front (drops broken workers).
        if opts.preflight.unwrap_or(self.env.swarm.preflight) {
            let pf = self.preflight(Some(&cancel)).await;
            if self.size() == 0 || pf.iter().all(|r| !r.ok) {
                let detail = pf
                    .iter()
                    .map(|r| format!("{}: {}", r.model, r.detail.as_deref().unwrap_or("unreachable")))
                    .collect::<Vec<_>>()
                    .join("; ");
                let run = SwarmRun { mode: SwarmMode::Divide, ..Default::default() };
                self.events.emit(SwarmEvent::Plan(PlanEvent {
                    task: task.to_string(),
                    subtasks: Vec::new(),
                    roster: self.roster(),
                    planned_by: Some("(none)".to_string()),
                    note: Some(format!("all agents failed preflight — {detail}")),
                    ..Default::default()
                }));
                self.events.emit(SwarmEvent::Done(run.clone()));
                return run;
            }
        }
        let workers = self.workers_snapshot();
        if workers.is_empty() {
            let run = SwarmRun { mode: SwarmMode::Divide, ..Default::default() };
            self.events.emit(SwarmEvent::Plan(PlanEvent {
                task: task.to_string(),
                planned_by: Some("(none)".to_string()),
                note: Some("no swarm workers available".to_string()),
                ..Default::default()
            }));
            self.events.emit(SwarmEvent::Done(run.clone()));
            return run;
        }

        // Planner candidates: roster order (primary first — fast models lead) with the
        // lead appended; a slow/timing-out lead then costs one attempt, not the plan.
        let mut plan_cands: Vec<SharedProvider> = Vec::new();
        let mut cand_models: Vec<String> = Vec::new();
        for p in workers.iter().map(|w| Arc::clone(&w.provider)).chain(std::iter::once(Arc::clone(&self.lead))) {
            let m = p.model();
            match cand_models.iter().position(|x| *x == m) {
                Some(i) => plan_cands[i] = p, // Map semantics: same key keeps its slot, last value wins
                None => {
                    cand_models.push(m);
                    plan_cands.push(p);
                }
            }
        }

        let (subtasks, planned_by, mut note, complexity) = match opts.subtasks.as_ref().filter(|s| !s.is_empty()) {
            Some(s) => (s.clone(), "(supplied)".to_string(), None, PlanComplexity::Simple),
            None => {
                let events = &self.events;
                let mut on_attempt = |model: &str| events.emit(SwarmEvent::Planner { model: model.to_string() });
                let plan =
                    decompose_with(&plan_cands, task, workers.len(), self.max_tokens, Some(&cancel), &mut on_attempt).await;
                (plan.subtasks, plan.planned_by, plan.note, plan.complexity)
            }
        };

        // Execution backend: bare metal for simple projects, an isolated Daytona
        // sandbox for complex ones (mode "auto"); shared by all agents in the run.
        let mut executor = opts.executor.clone();
        if executor.is_none() && opts.exec_mode != ExecMode::Off {
            let choice = choose_executor(&self.env, opts.exec_mode, complexity, opts.bare.clone());
            executor = choice.executor;
            if let Some(n) = choice.note {
                note = Some(match note {
                    Some(prev) => format!("{prev} · {n}"),
                    None => n,
                });
            }
        }
        let guard = DisposeGuard(executor.clone());

        self.events.emit(SwarmEvent::Plan(PlanEvent {
            task: task.to_string(),
            subtasks: subtasks.clone(),
            roster: self.roster(),
            planned_by: Some(planned_by),
            note,
            complexity: Some(complexity),
            exec: executor.as_ref().map(|e| e.label().to_string()),
        }));

        let board = Mutex::new(Blackboard::new(&subtasks));
        let sink: Mutex<Vec<WorkerResult>> = Mutex::new(Vec::new());

        let mut wave_index = 0usize;
        let mut guard_count = subtasks.len() + 2; // hard stop against any scheduling pathology
        loop {
            let wave: Vec<String> = {
                let b = lock(&board);
                if b.pending().is_empty() || guard_count == 0 {
                    break;
                }
                guard_count -= 1;
                let ready: Vec<String> = b.ready().iter().map(|e| e.id.clone()).collect();
                if ready.is_empty() {
                    // Dependency cycle or all-blocked: break it by forcing the pending subtask
                    // with the fewest unmet deps to run anyway (its digest will note the gap).
                    let mut pending = b.pending();
                    pending.sort_by_key(|e| b.unmet(e));
                    match pending.first() {
                        Some(stuck) => vec![stuck.id.clone()],
                        None => break,
                    }
                } else {
                    ready
                }
            };

            self.events.emit(SwarmEvent::Wave(WaveEvent { index: wave_index, subtask_ids: wave.clone() }));

            // Run the wave in batches no larger than the roster, so each active subtask
            // owns exactly one worker/pane at a time.
            for batch in wave.chunks(workers.len()) {
                join_all(batch.iter().enumerate().map(|(slot, id)| {
                    self.run_subtask(task, &board, id, slot, &workers, &cancel, &sink, executor.as_ref())
                }))
                .await;
            }
            wave_index += 1;
        }

        let results = std::mem::take(&mut *lock(&sink));
        let mut run = SwarmRun { mode: SwarmMode::Divide, results, ..Default::default() };

        if opts.synthesize && run.results.iter().any(|r| r.ok) {
            let lead_model = self.lead.model();
            self.events.emit(SwarmEvent::Synth { model: lead_model.clone() });
            run.synthesis = Some(self.synthesize_streamed(task, &run, &cancel).await);
            run.synthesized_by = Some(lead_model);
        }

        // Collect generated files (before the sandbox is torn down) for the results pane.
        if let Some(exec) = &executor {
            let arts = exec.artifacts().await;
            self.events.emit(SwarmEvent::Artifacts(arts.clone()));
            run.artifacts = Some(arts);
        }
        // Tear down the run's sandbox (also on drop, see DisposeGuard).
        guard.dispose().await;

        self.events.emit(SwarmEvent::Done(run.clone()));
        run
    }

    /// Execute one subtask on a worker, streaming its tokens and writing to the board.
    #[allow(clippy::too_many_arguments)]
    async fn run_subtask(
        &self,
        task: &str,
        board: &Mutex<Blackboard>,
        id: &str,
        pane: usize,
        workers: &[SwarmWorker],
        global: &CancellationToken,
        sink: &Mutex<Vec<WorkerResult>>,
        executor: Option<&SharedExecutor>,
    ) {
        let worker = &workers[pane % workers.len()];
        let (entry, context, solo) = {
            let mut b = lock(board);
            let Some(e) = b.get_mut(id) else { return };
            e.status = SubtaskStatus::Running;
            e.assigned_to = Some(worker.model.clone());
            let entry = e.clone();
            (entry, b.digest_for(id, DIGEST_DEP_CHARS), b.len() <= 1)
        };

        self.events.emit(SwarmEvent::Assign(AssignEvent {
            pane,
            subtask_id: entry.id.clone(),
            title: entry.title.clone(),
            worker_label: worker.label.clone(),
            model: worker.model.clone(),
        }));
        self.events.emit(SwarmEvent::Status(StatusEvent {
            pane,
            subtask_id: entry.id.clone(),
            status: SubtaskStatus::Running,
            ms: None,
            error: None,
        }));

        let mut prompt = format!("OVERALL TASK:\n{task}\n\n");
        if !solo {
            prompt.push_str(
                "YOU ARE ONE AGENT IN A COORDINATED TEAM. Focus ONLY on your assigned subtask below; teammates handle the rest.\n\n",
            );
        }
        prompt.push_str(&format!("YOUR SUBTASK [{}] {}:\n{}\n\n", entry.id, entry.title, entry.detail));
        if !context.is_empty() {
            prompt.push_str(&format!("--- SHARED CONTEXT FROM YOUR TEAM ---\n{context}\n\n"));
        }
        prompt.push_str(
            "Produce a focused, complete, final result for YOUR subtask. Reference teammates' results where relevant; don't repeat their work.",
        );
        let system = match executor {
            Some(e) => swarm_agent_exec_system(e.label()),
            None => swarm_agent_system(),
        };
        let messages = vec![Message::system(system), Message::user(prompt)];

        // Per-pane token so the user can stop THIS agent (Esc) without touching the
        // others; as a child of the run's token it also trips on a global abort.
        let token = global.child_token();
        let token_id = self.next_token_id.fetch_add(1, Ordering::Relaxed);
        lock(&self.pane_tokens).insert(pane, (token_id, token.clone()));

        let start = Instant::now();
        let mut pane_sink = PaneSink { events: &self.events, pane, subtask_id: entry.id.clone(), text: String::new() };
        let outcome = self.agent_loop(worker, messages, &mut pane_sink, &token, executor, pane, &entry.id).await;
        let ms = elapsed_ms(start);
        let base = WorkerResult {
            id: Some(entry.id.clone()),
            title: Some(entry.title.clone()),
            depends_on: entry.depends_on.clone(),
            label: worker.label.clone(),
            model: worker.model.clone(),
            backend: worker.kind,
            task: entry.detail.clone(),
            ms,
            ..Default::default()
        };

        let (result, status, error) = match outcome {
            Ok(out) => {
                let src = if out.text.is_empty() { pane_sink.text.as_str() } else { out.text.as_str() };
                let clean = clean_agent_text(src).trim().to_string();
                if let Some(e) = lock(board).get_mut(id) {
                    e.status = SubtaskStatus::Done;
                    e.result = clean.clone();
                    e.ms = Some(ms);
                }
                let r = WorkerResult {
                    ok: true,
                    text: clean,
                    prompt_tokens: out.prompt_tokens,
                    completion_tokens: out.completion_tokens,
                    ..base
                };
                (r, SubtaskStatus::Done, None)
            }
            Err(err) => {
                let stopped = token.is_cancelled() || qo_core::error::is_interrupt(&err);
                let status = if stopped { SubtaskStatus::Stopped } else { SubtaskStatus::Failed };
                let error = if stopped { "stopped by user".to_string() } else { format!("{err:#}") };
                if let Some(e) = lock(board).get_mut(id) {
                    e.status = status;
                    e.error = Some(error.clone());
                    e.ms = Some(ms);
                }
                let text = if stopped { clean_agent_text(&pane_sink.text).trim().to_string() } else { String::new() };
                let r = WorkerResult { ok: false, text, error: Some(error.clone()), ..base };
                (r, status, Some(error))
            }
        };
        lock(sink).push(result.clone());
        self.events.emit(SwarmEvent::Status(StatusEvent { pane, subtask_id: entry.id.clone(), status, ms: Some(ms), error }));
        self.events.emit(SwarmEvent::Result(result));

        let mut tokens = lock(&self.pane_tokens);
        if tokens.get(&pane).is_some_and(|(tid, _)| *tid == token_id) {
            tokens.remove(&pane);
        }
    }

    /// One agent turn — or, when an executor is attached, a bounded tool loop: the
    /// model may call run_shell (executed on the run's backend), sees each result,
    /// and finishes with a text deliverable. Commands and output tails are emitted
    /// as pane deltas so the TUI shows the agent *working*, not just talking.
    #[allow(clippy::too_many_arguments)]
    async fn agent_loop(
        &self,
        worker: &SwarmWorker,
        messages: Vec<Message>,
        sink: &mut PaneSink<'_>,
        cancel: &CancellationToken,
        executor: Option<&SharedExecutor>,
        pane: usize,
        subtask_id: &str,
    ) -> anyhow::Result<LoopOutput> {
        let base_opts = ModelOptions {
            temperature: Some(if is_reasoning_model(&worker.model) { 0.6 } else { 0.3 }),
            max_tokens: Some(self.max_tokens),
            cancel: Some(cancel.clone()),
            ..Default::default()
        };
        let Some(executor) = executor else {
            let res = worker.provider.stream(&messages, &mut |d: &str| sink.delta(d), &base_opts).await?;
            return Ok(LoopOutput { text: res.text, prompt_tokens: res.prompt_tokens, completion_tokens: res.completion_tokens });
        };

        let spec = ToolSpec {
            name: "run_shell".to_string(),
            description: format!("Run a non-interactive shell command on {}. Returns exit code and output.", executor.label()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." },
                    "cwd": { "type": "string", "description": "Working directory (optional)." }
                },
                "required": ["command"]
            }),
        };
        let max_steps = self.env.swarm.exec_max_steps.max(1);
        let timeout_ms = self.env.swarm.exec_timeout_s.max(1) * 1000;
        let mut convo = messages;
        let mut all_text = String::new();
        let mut prompt_tokens: Option<u64> = None;
        let mut completion_tokens: Option<u64> = None;

        for step in 0..=max_steps {
            if cancel.is_cancelled() {
                return Err(ProviderError::Interrupted.into());
            }
            // Tools are offered until the last step, which forces a final text answer.
            let opts = ModelOptions { tools: (step < max_steps).then(|| vec![spec.clone()]), ..base_opts.clone() };
            let res = worker.provider.stream(&convo, &mut |d: &str| sink.delta(d), &opts).await?;
            // Honor OpenAI tool_calls OR special-token tool calls (Kimi on NIM).
            let mut tool_calls = res.tool_calls.clone();
            if tool_calls.is_empty() {
                tool_calls = parse_special_tool_calls(&res.text);
            }
            let clean_text = clean_agent_text(&res.text);
            if !clean_text.is_empty() {
                if !all_text.is_empty() {
                    all_text.push('\n');
                }
                all_text.push_str(&clean_text);
            }
            prompt_tokens = res.prompt_tokens.or(prompt_tokens);
            if let Some(c) = res.completion_tokens {
                completion_tokens = Some(completion_tokens.unwrap_or(0) + c);
            }
            if tool_calls.is_empty() {
                break;
            }

            convo.push(Message::assistant_calls(clean_text, tool_calls.clone()));
            for tc in tool_calls {
                let cmd = match tc.arguments.get("command") {
                    None | Some(Value::Null) => String::new(),
                    Some(v) => js_string(v).trim().to_string(),
                };
                let cwd = match tc.arguments.get("cwd") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(js_string(v)).filter(|s| !s.is_empty()),
                };
                sink.delta(&format!("\n$ {cmd}\n"));
                let r = if cmd.is_empty() {
                    crate::exec::ExecResult { ok: false, exit_code: None, output: "run_shell: 'command' is required.".to_string() }
                } else {
                    executor.run(&cmd, RunOptions { cwd, timeout_ms: Some(timeout_ms), cancel: Some(cancel.clone()) }).await
                };
                let tail = tail_chars(&r.output, 1200);
                let exit = r.exit_code.map_or("?".to_string(), |c| c.to_string());
                sink.delta(&format!("{tail}{}", if r.ok { "\n".to_string() } else { format!("\n[exit {exit}]\n") }));
                self.events.emit(SwarmEvent::ExecCmd(ExecCmdEvent {
                    pane,
                    subtask_id: subtask_id.to_string(),
                    command: cmd.clone(),
                    ok: r.ok,
                }));
                let exit_field = r.exit_code.map_or("null".to_string(), |c| c.to_string());
                convo.push(Message::tool_result(tc.id.clone(), "run_shell", format!("exit={exit_field} ok={}\n{tail}", r.ok)));
            }
        }
        Ok(LoopOutput { text: all_text, prompt_tokens, completion_tokens })
    }

    /// Stream the synthesizer so the TUI/log can show the final answer forming.
    async fn synthesize_streamed(&self, task: &str, run: &SwarmRun, cancel: &CancellationToken) -> String {
        let ok: Vec<&WorkerResult> = run.results.iter().filter(|r| r.ok && !r.text.is_empty()).collect();
        if ok.is_empty() {
            return "All swarm agents failed; no answer to synthesize.".to_string();
        }
        if ok.len() == 1 {
            return ok[0].text.clone();
        }
        let blocks = ok
            .iter()
            .map(|r| {
                format!(
                    "### [{}] {}\n{}",
                    r.id.as_deref().unwrap_or("?"),
                    r.title.as_deref().unwrap_or(&r.model),
                    r.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let messages = vec![
            Message::system(SWARM_SYNTH_SYSTEM),
            Message::user(format!(
                "OVERALL TASK:\n{task}\n\n\
A team of agents completed the subtasks below, sharing context as they went. Write ONE clear, \
human-readable final report in Markdown that integrates their work:\n\
- a short summary of what was built/decided;\n\
- the files or artifacts produced, with their paths;\n\
- the key code in fenced ``` code blocks (real newlines — NEVER a raw JSON object of escaped file contents);\n\
- how to run or use it.\n\
Reconcile overlaps and contradictions; do not mention the team or the process.\n\n\
{blocks}\n\n--- Write the integrated Markdown report below. ---"
            )),
        ];
        let opts = ModelOptions {
            temperature: Some(0.3),
            max_tokens: Some(self.max_tokens),
            cancel: Some(cancel.clone()),
            ..Default::default()
        };
        let events = &self.events;
        let mut on_chunk = |d: &str| {
            if !d.is_empty() {
                events.emit(SwarmEvent::SynthDelta { delta: d.to_string() });
            }
        };
        match self.lead.stream(&messages, &mut on_chunk, &opts).await {
            Ok(res) => {
                let s = strip_thinking(&res.text);
                if s.is_empty() {
                    blocks
                } else {
                    s
                }
            }
            // Interrupted: don't start another (non-streaming) request behind the user's back.
            Err(e) if qo_core::error::is_interrupt(&e) || cancel.is_cancelled() => blocks,
            // Non-streaming fallback to the shared helper (which itself degrades to concatenation).
            Err(_) => synthesize(self.lead.as_ref(), task, run, self.max_tokens, Some(cancel)).await,
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The last `n` chars of `s` (char-boundary safe).
fn tail_chars(s: &str, n: usize) -> &str {
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    match s.char_indices().nth(count - n) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// A ready coordinator built from the environment.
pub struct BuiltSwarm {
    pub swarm: Arc<CoordinatedSwarm>,
    /// True when the roster fell back to local models.
    pub local: bool,
}

/// Convenience: build the roster + lead from the environment and return a ready
/// coordinator. Returns `None` when there is genuinely nothing to run with.
pub fn create_coordinated_swarm(env: Arc<SwarmEnv>, max_workers: Option<usize>) -> Option<BuiltSwarm> {
    let RosterResult { workers, local } = build_roster(&env, max_workers);
    if workers.is_empty() {
        return None;
    }
    let lead = pick_lead(&env, &workers);
    let max_tokens = env.swarm.max_tokens;
    Some(BuiltSwarm { swarm: Arc::new(CoordinatedSwarm::new(env, workers, lead, Some(max_tokens))), local })
}
