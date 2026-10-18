//! One user turn (port of `runAssistantTurn` + `executeToolCall` in
//! src/cli/commands/chat.ts): let the model call tools in a loop, feeding each
//! tool's real output back, until it produces a final text answer.
//!
//! Small models sometimes regress and "describe" a command in a ``` fence
//! instead of emitting a native tool call. We (1) run tool turns at low
//! temperature, (2) inject a one-time corrective nudge, and (3) if it still
//! regresses, execute the shell-tagged command ourselves.
//!
//! Esc: the turn owns ONE cancellation token. It rides on every request
//! (`ModelOptions.cancel`) and on the tool context, so Esc stops a request at
//! any stage, or a running shell command, and the turn ends cleanly.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use qo_core::error::is_interrupt;
use qo_core::{CancellationToken, Message, ModelOptions, ToolCall, ToolSpec};
use qo_net::evolution::TurnSignals;
use serde_json::{Map, Value};

use super::text::{
    echoes_tool_result, extract_shell_commands, last_tool_result, sanitize_answer, split_thinking, tool_result_budget,
    truncate_middle,
};
use super::Chat;
use crate::outln;
use crate::render::{format_tokens, indent, thinking_word};
use crate::ui::{style, InterjectOpts, Spinner};

/// Shared between the turn and its interjection-capture callbacks.
#[derive(Default)]
struct SideChannel {
    /// Asides typed while the model works, injected at the next step boundary.
    pending: Mutex<Vec<String>>,
    /// The spinner of the request in flight (paused while the user types).
    spinner: Mutex<Option<Spinner>>,
}

impl SideChannel {
    fn take_pending(&self) -> Vec<String> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()))
    }
    fn has_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }
    fn set_spinner(&self, s: Option<Spinner>) {
        *self.spinner.lock().unwrap_or_else(|e| e.into_inner()) = s;
    }
    fn with_spinner(&self, f: impl FnOnce(&Spinner)) {
        if let Some(s) = self.spinner.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            f(s);
        }
    }
}

/// Per-turn mutable bookkeeping.
#[derive(Default)]
struct TurnState {
    failures: Vec<String>,
    seen_calls: HashSet<String>,
    /// Repeated-identical-call detections. Small models that ignore the first
    /// "you already ran this" rarely recover; after the second one the turn
    /// continues WITHOUT tools so the model has to answer in words.
    loop_hits: u32,
    force_answer: bool,
}

/// One prepared call.
struct Item {
    call: ToolCall,
    name: String,
    args: Value,
    error: Option<String>,
    note: Option<String>,
    sig: String,
    mutating: bool,
}

const INTERRUPTED_NOTE: &str = "\n  ⏸ interrupted — type your next message.\n";

impl Chat {
    /// Run one assistant turn over `specs` (empty = no tools this turn).
    pub async fn run_assistant_turn(&mut self, specs: &[ToolSpec]) -> anyhow::Result<TurnSignals> {
        let cancel = CancellationToken::new();
        self.tools.context_mut().cancel = Some(cancel.clone());

        // ── /btw side-channel + Esc ─────────────────────────────────────────
        let side = Arc::new(SideChannel::default());
        let interject = {
            let (s1, s2, s3) = (side.clone(), side.clone(), side.clone());
            let c = cancel.clone();
            self.ui.start_interjector(InterjectOpts {
                label: style::yellow("btw › "),
                on_start_typing: Some(Box::new(move || s1.with_spinner(|s| s.pause()))),
                on_stop_typing: Some(Box::new(move || s2.with_spinner(|s| s.resume()))),
                on_interrupt: Some(Box::new(move || c.cancel())),
                on_submit: Box::new(move |text| {
                    let t = strip_btw_prefix(&text);
                    if t.is_empty() {
                        return;
                    }
                    s3.pending.lock().unwrap_or_else(|e| e.into_inner()).push(t.clone());
                    outln!("\n{}{}\n", style::yellow("  ↩ noted — I'll address that next step: "), style::gray(&t));
                }),
            })
        };
        let typing = {
            let flag = interject.as_ref().map(|ij| ij.typing_flag());
            move || flag.as_ref().is_some_and(|f| f.load(std::sync::atomic::Ordering::SeqCst))
        };

        let result = self.turn_loop(specs, &cancel, &side, interject.is_some(), &typing).await;

        // Asides that never got delivered (the turn ended first) and any
        // half-typed text go back to the prompt instead of being lost.
        let leftovers = side.take_pending();
        if let Some(ij) = interject {
            self.ui.stop_interjector(ij, leftovers).await;
        }
        self.tools.context_mut().cancel = None;
        result
    }

    async fn turn_loop(
        &mut self,
        specs: &[ToolSpec],
        cancel: &CancellationToken,
        side: &Arc<SideChannel>,
        interactive: bool,
        typing: &(dyn Fn() -> bool + Send + Sync),
    ) -> anyhow::Result<TurnSignals> {
        let mut nudged = false;
        let mut empty_nudges = 0u32;
        let mut tool_answer_nudges = 0u32;
        // `think_on` = should the model deliberate this turn (None = no
        // thinking mode). Named apart from the per-step thinking TEXT.
        let (profile, think_on) = self.turn_profile();
        let mut st = TurnState::default();

        // Budget only PRODUCTIVE steps (turns that ran tools). Corrective nudges
        // don't burn it — they count against a hard iteration cap instead.
        let step_budget = self.s.config.agent.max_tool_steps.max(1);
        let hard_cap = step_budget * 3;
        let mut productive = 0u32;
        // Thinking mode: `think` calls spend a separate budget, so deliberation
        // never starves real work.
        let known_names: Vec<String> = specs.iter().map(|t| t.name.clone()).collect();
        let thinking_budget = if self.s.config.agent.thinking_mode { self.s.config.agent.max_thinking_turns } else { 0 };
        let mut thinking_turns = 0u32;
        let mut thinking_nudged = false;

        let interrupted = |st: TurnState| {
            outln!("{}", style::yellow(INTERRUPTED_NOTE));
            TurnSignals { user_message: String::new(), failures: st.failures, final_answer: String::new(), step_limit_hit: false }
        };

        let mut step = 0u32;
        while productive < step_budget && step < hard_cap {
            // Inject any asides the user typed (/btw) since the last step.
            self.drain_pending(side);
            // Live status: elapsed time + the session's cumulative input tokens.
            // The hint matters: a slow or dead backend otherwise looks like a
            // frozen terminal.
            let spinner = Spinner::new(thinking_word(), self.meter.session_in, if interactive { "esc to interrupt" } else { "" });
            side.set_spinner(Some(spinner.clone()));
            if typing() {
                spinner.pause();
            }
            spinner.begin();
            let opts = ModelOptions {
                temperature: Some(self.turn_temperature(&profile, think_on)),
                tools: (!specs.is_empty() && !st.force_answer).then(|| specs.to_vec()),
                think: think_on,
                cancel: Some(cancel.clone()),
                ..Default::default()
            };
            let res = self.stream_with_fallback(&opts, &spinner).await;
            spinner.stop();
            side.set_spinner(None);
            let res = match res {
                // Esc: stop this turn cleanly and hand control back to the
                // prompt so the user can redirect — no fallback, no error spew.
                Err(e) if is_interrupt(&e) || cancel.is_cancelled() => return Ok(interrupted(st)),
                Err(e) => return Err(e),
                Ok(_) if cancel.is_cancelled() => return Ok(interrupted(st)),
                Ok(r) => r,
            };
            let est_in = self.history_tokens();
            let est_out = self.s.provider.count_tokens(&res.text) as u64;
            self.meter.record(&res, est_in, est_out);

            // Small models often emit the tool call as JSON TEXT instead of via
            // the structured field. Recover those so they execute.
            let mut calls = res.tool_calls.clone();
            let mut from_text = false;
            if calls.is_empty() {
                let text_calls = extract_text_tool_calls(&res.text, specs, step);
                if !text_calls.is_empty() {
                    calls = text_calls;
                    from_text = true;
                }
            }

            // Happy path: real tool calls. Store only the non-reasoning part of
            // the assistant message (a chain-of-thought left in history confuses
            // reasoning models later); text-emitted calls drop the raw JSON.
            if !calls.is_empty() {
                let content = if from_text { String::new() } else { split_thinking(&res.text).answer };
                self.history.push(Message::assistant_calls(content, calls.clone()));
                self.run_calls(calls.clone(), specs, &mut st).await;
                if cancel.is_cancelled() {
                    return Ok(interrupted(st));
                }
                // A step that ONLY thought spends a thinking turn, not a tool step.
                let all_think = calls
                    .iter()
                    .all(|c| qo_tools::resolve_tool_name(&c.name, &known_names).name.as_deref().unwrap_or(&c.name) == "think");
                if all_think {
                    if thinking_turns < thinking_budget {
                        thinking_turns += 1;
                        step += 1;
                        continue;
                    }
                    if !thinking_nudged {
                        thinking_nudged = true;
                        self.history.push(Message::user(
                            "[system] You've used all your thinking turns. Stop calling `think` — act now: \
make the tool call your plan needs, or give your final answer.",
                        ));
                    }
                }
                productive += 1;
                step += 1;
                continue;
            }

            // Reasoning: prefer the backend's separate thinking field, else
            // inline <think> tags. Fence detection runs on the answer only.
            let inline = split_thinking(&res.text);
            let thinking = res
                .thinking
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .unwrap_or(inline.thinking)
                .trim()
                .to_string();
            // Drop any chat-template tool tags the model leaked into its reply.
            let answer = sanitize_answer(&inline.answer);
            let fenced = extract_shell_commands(&answer);

            if fenced.is_empty() {
                // The model ran a tool and then parroted the raw result (or
                // produced no prose). Nudge it once to summarize. Bounded.
                let last_tool = last_tool_result(&self.history).to_string();
                let echoed = echoes_tool_result(&answer, &last_tool);
                if (answer.trim().is_empty() || echoed) && !last_tool.is_empty() && tool_answer_nudges < 2 {
                    tool_answer_nudges += 1;
                    st.failures.push("returned raw tool output / no prose answer after a tool call".into());
                    self.history.push(Message::user(
                        "[system] Do NOT repeat the tool output or wrap anything in <tool_response>/<tool_call> tags. \
Using the tool result above, answer my question directly in one or two plain sentences.",
                    ));
                    step += 1;
                    continue;
                }

                // Planned but produced NO tool call and NO answer: nudge it to
                // follow through. Bounded.
                if answer.trim().is_empty() && !thinking.is_empty() && empty_nudges < 2 {
                    empty_nudges += 1;
                    outln!("{}", style::magenta("qwen ⟂ thinking"));
                    outln!("{}\n", style::gray(&indent(&thinking)));
                    st.failures.push("reasoned without acting (no tool call or answer)".into());
                    self.history.push(Message::user(
                        "[system] You reasoned but didn't act. Now DO it in this reply: make the actual tool call \
(e.g. shell_help then run_shell), or give your final answer. Don't just describe it.",
                    ));
                    step += 1;
                    continue;
                }

                // Genuine final answer.
                if !thinking.is_empty() {
                    outln!("{}", style::magenta("qwen ⟂ thinking"));
                    outln!("{}\n", style::gray(&indent(&thinking)));
                }
                outln!("{}{}\n", style::green("qwen › "), if answer.is_empty() { "(no response)" } else { answer.as_str() });
                // Per-turn usage + the running session totals.
                let m = self.meter;
                if m.session_in > 0 || m.session_out > 0 {
                    outln!(
                        "{}",
                        style::gray(&format!(
                            "  this turn: ↑ {} ↓ {}  ·  session: ↑ {} ↓ {} tokens\n",
                            format_tokens(m.last_in),
                            format_tokens(m.last_out),
                            format_tokens(m.session_in),
                            format_tokens(m.session_out)
                        ))
                    );
                }
                // Store only the clean answer, NOT the raw chain-of-thought.
                self.history.push(Message::assistant(answer.clone()));
                // An aside arrived while finishing? Address it before ending.
                if side.has_pending() {
                    self.drain_pending(side);
                    step += 1;
                    continue;
                }
                return Ok(TurnSignals { user_message: String::new(), failures: st.failures, final_answer: answer, step_limit_hit: false });
            }

            if !nudged {
                // First regression: correct it without running anything yet.
                nudged = true;
                st.failures.push("printed a command in a ``` fence instead of calling the run_shell tool".into());
                self.history.push(Message::user(
                    "[system] Do not print commands inside ``` code fences — fenced code is NOT executed and the user \
sees no result. To actually run a command, call the run_shell tool. If you already have what you need, give your final \
answer with no code fences.",
                ));
                step += 1;
                continue;
            }

            // Still regressing after a nudge: honour the obvious intent and run it.
            outln!("{}", style::gray("  (model described a command instead of calling the tool — running it)"));
            let synthetic: Vec<ToolCall> = fenced
                .into_iter()
                .enumerate()
                .map(|(i, command)| ToolCall { id: format!("auto_{step}_{i}"), name: "run_shell".into(), arguments: serde_json::json!({ "command": command }) })
                .collect();
            self.history.push(Message::assistant_calls("", synthetic.clone()));
            self.run_calls(synthetic, specs, &mut st).await;
            if cancel.is_cancelled() {
                return Ok(interrupted(st));
            }
            productive += 1;
            step += 1;
        }

        outln!("{}\n", style::yellow(&format!("[stopped after {step_budget} tool steps]")));
        self.history.push(Message::assistant(format!("Reached the {step_budget}-step tool limit for this turn.")));
        Ok(TurnSignals { user_message: String::new(), failures: st.failures, final_answer: String::new(), step_limit_hit: true })
    }

    fn drain_pending(&mut self, side: &SideChannel) {
        for t in side.take_pending() {
            self.history.push(Message::user(format!(
                "[Side question from the user, sent while you were working — address it too, then carry on]: {t}"
            )));
        }
    }

    /// Run a batch of tool calls. Each is first hardened (fuzzy name
    /// resolution + schema coercion/validation). Consecutive read-only calls
    /// run CONCURRENTLY; any mutating call, validation error, or repeat is
    /// handled serially so ordering and the confirm gate stay intact.
    async fn run_calls(&mut self, calls: Vec<ToolCall>, specs: &[ToolSpec], st: &mut TurnState) {
        let items: Vec<Item> = calls
            .into_iter()
            .map(|call| {
                let prep = qo_tools::prepare_tool_call(&call.name, Some(&call.arguments), specs);
                let sig = format!("{} {}", prep.name, serde_json::to_string(&prep.arguments).unwrap_or_default());
                let tool = if prep.error.is_some() { None } else { self.tools.get(&prep.name) };
                let mutating = tool.map_or(true, |t| t.mutating());
                Item { call, name: prep.name, args: prep.arguments, error: prep.error, note: prep.note, sig, mutating }
            })
            .collect();

        let mut i = 0usize;
        while i < items.len() {
            if self.tools.context().cancelled() {
                return;
            }
            // Greedily gather a run of parallel-safe reads: validated,
            // non-mutating, and not already executed this turn.
            let mut batch: Vec<&Item> = Vec::new();
            while i < items.len() {
                let it = &items[i];
                if it.error.is_some() || it.mutating || st.seen_calls.contains(&it.sig) {
                    break;
                }
                st.seen_calls.insert(it.sig.clone());
                if let Some(n) = &it.note {
                    outln!("{}", style::gray(&format!("  ({n})")));
                }
                batch.push(it);
                i += 1;
            }
            if !batch.is_empty() {
                if batch.len() > 1 {
                    outln!("{}", style::gray(&format!("  ⚡ running {} read-only tools in parallel", batch.len())));
                }
                let this = &*self;
                let results = futures::future::join_all(batch.iter().map(|b| this.run_prepared(&b.call, &b.name, &b.args))).await;
                for (msg, failure) in results {
                    if let Some(f) = failure {
                        st.failures.push(f);
                    }
                    self.history.push(msg);
                }
                continue;
            }

            // Not parallel-safe — handle exactly one, then re-loop.
            let it = &items[i];
            i += 1;
            if let Some(n) = &it.note {
                outln!("{}", style::gray(&format!("  ({n})")));
            }
            if let Some(err) = &it.error {
                st.failures.push(format!("{} → {err}", it.call.name));
                self.history.push(Message::tool_result(&it.call.id, &it.call.name, err.clone()));
                continue;
            }
            // Loop guard: an identical (resolved) call already ran.
            if st.seen_calls.contains(&it.sig) {
                st.failures.push(format!("repeated identical call to {} (loop)", it.name));
                st.loop_hits += 1;
                if st.loop_hits >= 2 {
                    st.force_answer = true;
                }
                let content = if st.force_answer {
                    format!(
                        "You called {} with these same arguments again. Tools are now switched off for this turn. \
Using the results already above, answer the user in plain words now.",
                        it.name
                    )
                } else {
                    format!(
                        "You ALREADY called {} with these arguments and its result is above. Do NOT call it again. \
Take the NEXT step now: run the actual command with run_shell, or give your final answer.",
                        it.name
                    )
                };
                self.history.push(Message::tool_result(&it.call.id, &it.name, content));
                continue;
            }
            st.seen_calls.insert(it.sig.clone());
            let (msg, failure) = self.run_prepared(&it.call, &it.name, &it.args).await;
            if let Some(f) = failure {
                st.failures.push(f);
            }
            self.history.push(msg);
        }
    }

    /// Execute one validated call → its tool-result message (+ a failure note).
    async fn run_prepared(&self, call: &ToolCall, name: &str, args: &Value) -> (Message, Option<String>) {
        let effective = ToolCall { id: call.id.clone(), name: name.to_string(), arguments: args.clone() };
        let (content, ok) = self.execute_tool_call(&effective).await;
        let failure = (!ok).then(|| {
            let a: String = serde_json::to_string(args).unwrap_or_default().chars().take(120).collect();
            let c: String = content.chars().take(200).collect();
            format!("{name} {a} → {c}")
        });
        (Message::tool_result(&call.id, name, content), failure)
    }

    /// Run a single tool call against the registry, with a confirm gate for shell.
    pub async fn execute_tool_call(&self, call: &ToolCall) -> (String, bool) {
        let empty = Map::new();
        let args = call.arguments.as_object().unwrap_or(&empty);

        // Show what the model is doing.
        if call.name == "run_shell" || call.name == "shell_session" {
            let cmd = args.get("command").map(value_text).unwrap_or_default().trim().to_string();
            let tag = if call.name == "shell_session" { "  $⟳ " } else { "  $ " };
            outln!("{}{}", style::gray(tag), style::cyan(&cmd));

            let co = qo_tools::ClassifyOptions {
                allow: self.s.config.tools.allow_commands.clone(),
                deny: self.s.config.tools.deny_commands.clone(),
            };
            match qo_tools::classify_command(&cmd, Some(&co)) {
                qo_tools::CommandClass::Blocked => {
                    outln!("{}", style::red("  ⛔ refused (hard-blocked / deny-listed)\n"));
                    return (format!("Refused: \"{cmd}\" is hard-blocked or deny-listed and was not run."), false);
                }
                qo_tools::CommandClass::Destructive if self.s.config.tools.confirm_destructive && !self.s.auto_confirm => {
                    outln!("{}", style::yellow("  ⚠ looks destructive — type 'y' to run, anything else to skip:"));
                    let reply = self.ask().await;
                    if !crate::ui::is_yes(&reply) {
                        outln!("{}", style::gray("  (declined)\n"));
                        return ("Declined by user.".into(), false);
                    }
                }
                _ => {}
            }
        } else {
            let summary = args
                .iter()
                .map(|(k, v)| format!("{k}={}", serde_json::to_string(v).unwrap_or_default()))
                .collect::<Vec<_>>()
                .join(" ");
            outln!("{}", style::gray(&format!("  ⚙ {} {summary}", call.name)));
        }

        // Blind-overwrite guard: write_file replaces a file's ENTIRE contents.
        // Small models routinely "add a function" by writing a file that holds
        // only the new function. If the target exists and hasn't been read this
        // session, refuse and point at apply_edit / read_file, unless the model
        // explicitly passes overwrite=true.
        let args_value = Value::Object(args.clone());
        let refusal = self.seen_files.lock().unwrap_or_else(|e| e.into_inner()).check(&self.s.cwd, &call.name, &args_value);
        if let Some(refusal) = refusal {
            let shown = args.get("path").map(value_text).unwrap_or_default();
            outln!("{}", style::yellow(&format!("  ⚠ refused: {shown} exists and hasn't been read — use apply_edit or read_file first\n")));
            return (refusal, false);
        }

        let result = self.tools.run(&call.name, args_value.clone()).await;
        self.seen_files.lock().unwrap_or_else(|e| e.into_inner()).record(&self.s.cwd, &call.name, &args_value, result.ok);
        // The think tool returns its thought as data (library crates never
        // print): show it the way inline <think> reasoning is shown.
        if let Some(thought) = result.data.as_ref().and_then(|d| d.get("thought")).and_then(Value::as_str) {
            outln!("{}", style::magenta("qwen ⟂ thinking"));
            outln!("{}\n", style::gray(&format!("  {}", thought.split('\n').collect::<Vec<_>>().join("\n  "))));
        }
        let out = result.output;
        let shown = if out.chars().count() > 4000 { format!("{}\n…(truncated)", out.chars().take(4000).collect::<String>()) } else { out.clone() };
        if !shown.trim().is_empty() {
            outln!("{}\n", style::gray(&indent(&shown)));
        }
        // Feed back head+tail within a context-scaled budget: small models keep
        // the error/exit summary (usually at the END) without flooding context.
        let content = truncate_middle(&out, tool_result_budget(self.s.config.model.context_tokens));
        let content = if content.is_empty() {
            if result.ok { "(ok, no output)".to_string() } else { "(failed, no output)".to_string() }
        } else {
            content
        };
        (content, result.ok)
    }
}

/// `/btw` prefix stripped from an aside.
fn strip_btw_prefix(text: &str) -> String {
    let t = text.trim();
    let lower = t.to_lowercase();
    if lower.starts_with("/btw") && t[4..].chars().next().map_or(true, |c| !c.is_alphanumeric() && c != '_') {
        t[4..].trim().to_string()
    } else {
        t.to_string()
    }
}

/// A JSON value as display text (strings unquoted).
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Recover tool calls a small model emitted as TEXT — a JSON object/array with
/// {name, arguments} — instead of via the structured tool_calls field. Accepts
/// the common shape variants ({tool_call:…}, {function:…}, parameters/args
/// aliases), only for tools advertised this turn; hallucinated names go
/// through the same alias/fuzzy resolver as dispatched calls.
pub fn extract_text_tool_calls(text: &str, specs: &[ToolSpec], step: u32) -> Vec<ToolCall> {
    if text.is_empty() || specs.is_empty() {
        return Vec::new();
    }
    let known_names: Vec<String> = specs.iter().map(|t| t.name.clone()).collect();
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut seen = HashSet::new();

    let mut consider = |raw: &Value| {
        let Some(obj) = raw.as_object() else { return };
        let o = obj
            .get("tool_call")
            .or_else(|| obj.get("function"))
            .or_else(|| obj.get("tool"))
            .and_then(Value::as_object)
            .unwrap_or(obj);
        let name = o.get("name").and_then(Value::as_str).or_else(|| o.get("tool_name").and_then(Value::as_str));
        let Some(mut name) = name.map(str::to_string) else { return };
        if !known_names.contains(&name) {
            // A text-emitted call with a hallucinated name ("bash", "read")
            // used to be dropped here and the raw JSON printed as the answer.
            match qo_tools::resolve_tool_name(&name, &known_names).name {
                Some(n) => name = n,
                None => return, // genuinely unknown → leave as prose
            }
        }
        let mut args = o
            .get("arguments")
            .or_else(|| o.get("parameters"))
            .or_else(|| o.get("args"))
            .or_else(|| o.get("input"))
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        if let Value::String(s) = &args {
            args = serde_json::from_str(s).unwrap_or(Value::Object(Map::new()));
        }
        if !args.is_object() {
            args = Value::Object(Map::new());
        }
        let key = format!("{name}{}", serde_json::to_string(&args).unwrap_or_default());
        if !seen.insert(key) {
            return; // de-dupe a call the model repeated
        }
        calls.push(ToolCall { id: format!("text_{step}_{}", calls.len()), name, arguments: args });
    };

    for value in qo_core::parse::extract_all_json(text) {
        match &value {
            Value::Array(items) => items.iter().for_each(&mut consider),
            other => consider(other),
        }
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec { name: name.into(), description: String::new(), parameters: serde_json::json!({"type":"object"}) }
    }

    #[test]
    fn recovers_text_tool_calls() {
        let specs = vec![spec("run_shell"), spec("read_file")];
        let t = "```json\n{\"name\": \"run_shell\", \"arguments\": {\"command\": \"dir\"}}\n```";
        let c = extract_text_tool_calls(t, &specs, 2);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].id, "text_2_0");
        assert_eq!(c[0].arguments["command"], "dir");

        let t = "[{\"function\":{\"name\":\"read_file\",\"parameters\":\"{\\\"path\\\":\\\"a\\\"}\"}}, {\"function\":{\"name\":\"read_file\",\"parameters\":\"{\\\"path\\\":\\\"a\\\"}\"}}]";
        let c = extract_text_tool_calls(t, &specs, 0);
        assert_eq!(c.len(), 1, "repeated call de-duped");
        assert_eq!(c[0].arguments["path"], "a");

        assert!(extract_text_tool_calls("{\"name\":\"totally_unknown_zzz\"}", &specs, 0).is_empty());
        assert!(extract_text_tool_calls("no json here", &specs, 0).is_empty());
    }

    #[test]
    fn btw_prefix_on_asides() {
        assert_eq!(strip_btw_prefix("/btw also check X"), "also check X");
        assert_eq!(strip_btw_prefix("plain"), "plain");
        assert_eq!(strip_btw_prefix("/btwfoo"), "/btwfoo");
    }
}
