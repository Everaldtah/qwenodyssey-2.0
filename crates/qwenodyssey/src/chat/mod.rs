//! Interactive pair-coding chat (port of src/cli/commands/chat.ts).
//!
//! Streams responses, lets the model call tools in a loop, persists sessions,
//! and walks a fallback chain when a model is unavailable. Type @path to inline
//! a file's contents, /reset to clear history, /exit to quit.
//!
//! Terminal model (the 2.0 fix for "switched models and now I can't type"):
//! raw mode is entered ONCE here and restored by a guard; one input thread owns
//! stdin; the prompt, pickers, y/N gates and the turn-time Esc/"btw" capture
//! take turns on that stream ([`crate::ui::Ui`]). Every request carries the
//! turn's cancellation token, so Esc always works — including while a model
//! accepts a request and never answers.

pub mod models;
pub mod prompts;
pub mod text;
pub mod turn;
#[cfg(test)]
mod turn_tests;
pub mod views;

use std::io::IsTerminal;
use std::sync::{Arc, Mutex};

use qo_core::model_profile::{model_profile, resolve_thinking, sampling_for, ModelProfile};
use qo_core::{GenerateResult, Message, ModelInfo, Role, SharedProvider, ToolContext, ToolRegistry, ToolSpec};
use qo_net::{derive_title, ChatSession, ChatSessionMeta, KnowledgeBase, SessionStore};
use qo_providers::LmsModel;
use tokio::task::JoinHandle;

use crate::cli::{ChatArgs, GlobalOpts};
use crate::outln;
use crate::render::{banner, hrule, BannerInfo};
use crate::session::Session;
use crate::ui::{self, out, style, Input, SlashCommand, TermConfirmer, Ui};

use prompts::*;

/// Slash commands — shared by the live `/` palette and the static /help menu.
pub static SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand { name: "/help", aliases: &["/commands", "/?"], args: None, desc: "Show this list of commands" },
    SlashCommand { name: "/settings", aliases: &["/config"], args: None, desc: "Show the current model & runtime settings" },
    SlashCommand { name: "/models", aliases: &[], args: None, desc: "Pick a model with ↑/↓ + Enter (or list when piped)" },
    SlashCommand {
        name: "/model",
        aliases: &[],
        args: Some("[name|#]"),
        desc: "Open the model picker, or switch directly to name/number",
    },
    SlashCommand { name: "/context", aliases: &[], args: None, desc: "Show how much of the context window is used" },
    SlashCommand { name: "/compact", aliases: &[], args: None, desc: "Summarize older turns now to free up context" },
    SlashCommand { name: "/plan", aliases: &[], args: None, desc: "Show the agent's current step-by-step plan" },
    SlashCommand {
        name: "/memory",
        aliases: &["/knowledge"],
        args: None,
        desc: "Show the long-term knowledge vault (notes & path)",
    },
    SlashCommand {
        name: "/lessons",
        aliases: &["/evolution"],
        args: None,
        desc: "Show lessons the agent learned from past mistakes",
    },
    SlashCommand { name: "/sessions", aliases: &[], args: None, desc: "List saved chat sessions you can resume" },
    SlashCommand {
        name: "/resume",
        aliases: &[],
        args: Some("[id]"),
        desc: "Resume a saved session (pick from a list, or by id)",
    },
    SlashCommand {
        name: "/btw",
        aliases: &[],
        args: Some("<question>"),
        desc: "Aside — also type it while the model is working to queue it",
    },
    SlashCommand { name: "/reset", aliases: &[], args: None, desc: "Clear history and start a new session" },
    SlashCommand { name: "/exit", aliases: &["/quit"], args: None, desc: "Quit Qwenodyssey" },
];

/// Tool turns run deterministically; temp 0 markedly improves tool adherence.
pub const TOOL_TEMP: f64 = 0.0;

/// Cloud backends, where a hard 0 temperature risks repetition loops.
pub fn is_cloud_provider(name: &str) -> bool {
    matches!(name, "nvidia" | "openai" | "openrouter" | "anthropic")
}

/// Running tally of the tokens actually consumed this session, accumulated
/// from the EXACT usage the model reports after each response. Starts at 0 and
/// only grows — the spinner shows the cumulative input total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenMeter {
    /// Σ prompt tokens across all requests this session.
    pub session_in: u64,
    /// Σ completion tokens.
    pub session_out: u64,
    /// This turn's prompt tokens.
    pub last_in: u64,
    /// This turn's completion tokens.
    pub last_out: u64,
}

impl TokenMeter {
    /// Add a response's exact usage to the totals. If a backend omits usage,
    /// the heuristic `estimate` keeps a truthful-ish tally instead of zero.
    pub fn record(&mut self, res: &GenerateResult, est_in: u64, est_out: u64) {
        self.last_in = res.prompt_tokens.unwrap_or(est_in);
        self.last_out = res.completion_tokens.unwrap_or(est_out);
        self.session_in += self.last_in;
        self.session_out += self.last_out;
    }
}

/// Everything the chat loop owns.
pub struct Chat {
    pub s: Session,
    pub ui: Arc<Ui>,
    pub history: Vec<Message>,
    /// The chat's own tool registry (defaults + web/knowledge/github/swarm/
    /// MCP/shell-session/plan/think as configured).
    pub tools: ToolRegistry,
    /// Schemas advertised to the model.
    pub tool_specs: Vec<ToolSpec>,
    pub kb: Arc<KnowledgeBase>,
    pub memory_enabled: bool,
    pub evolution: Option<qo_net::EvolutionEngine>,
    /// The active provider, shared with background users that must follow
    /// model switches (evolution reflection).
    pub provider_cell: Arc<Mutex<SharedProvider>>,
    pub meter: TokenMeter,
    /// Files the model has read or written this session — the write_file
    /// blind-overwrite guard.
    pub seen_files: Mutex<qo_tools::BlindOverwriteGuard>,
    pub plan_state: qo_tools::PlanState,
    /// Cached result of the last model listing, so `/model <#>` resolves.
    pub last_models: Vec<ModelInfo>,
    /// The startup LM Studio inventory job (`/model` waits for it).
    pub lms_job: Option<JoinHandle<Vec<LmsModel>>>,
    /// The rest of the background startup (repo scan → PROJECT summary,
    /// symbol-index prewarm); awaited before the first real turn.
    pub startup_job: Option<JoinHandle<Option<String>>>,
    /// Whether startup model resolution has run.
    pub startup_resolved: bool,
    /// Cached "Anthropic credential available" (resolved in the background).
    pub anthropic_auth: Arc<std::sync::OnceLock<bool>>,
}

impl Chat {
    /// What the ACTIVE model is and whether it deliberates this turn
    /// (recomputed per turn: the model can change mid-session).
    pub fn turn_profile(&self) -> (ModelProfile, Option<bool>) {
        let profile = model_profile(&self.s.provider.model());
        let thinking = resolve_thinking(&profile, self.s.config.model.think);
        (profile, thinking)
    }

    /// Effective temperature for a turn: TOOL_TEMP raised only to the family's
    /// floor (Qwen3/R1/QwQ loop at 0) and a small cloud floor as
    /// anti-degeneration insurance for models the profiler doesn't recognize.
    pub fn turn_temperature(&self, profile: &ModelProfile, thinking: Option<bool>) -> f64 {
        let cloud_floor = if is_cloud_provider(self.s.provider.name()) { 0.2 } else { 0.0 };
        TOOL_TEMP.max(sampling_for(profile, thinking).min_turn_temp).max(cloud_floor)
    }

    /// The SELF-AWARENESS block for the active provider/model.
    pub fn self_awareness_text(&self) -> String {
        let vault = self.kb.dir().display().to_string();
        let root = self.s.self_root.display().to_string();
        let model = self.s.provider.model();
        self_awareness(&SelfAwareness {
            provider: self.s.provider.name(),
            model: &model,
            vault_dir: self.memory_enabled.then_some(vault.as_str()),
            self_root: self.s.config.tools.allow_self_edit.then_some(root.as_str()),
            evolution_on: self.evolution.is_some(),
        })
    }

    /// Keep the model's stated identity in sync with the active backend.
    pub fn refresh_identity(&mut self) {
        let text = self.self_awareness_text();
        text::replace_self_awareness(&mut self.history, &text);
    }

    /// Read a line at the main prompt, boxed in rules (a rule above, the `❯`
    /// input line with the palette floating beneath it, a rule below).
    pub async fn ask(&self) -> String {
        out::print(&format!("{}\n", hrule()));
        let answer = self.ui.read_line(&style::cyan("❯ "), SLASH_COMMANDS).await;
        out::print(&format!("{}\n\n", hrule()));
        answer
    }

    /// History tokens (heuristic) with the active provider's counter.
    pub fn history_tokens(&self) -> u64 {
        qo_net::history_tokens(&self.history, self.s.provider.as_ref()) as u64
    }

    /// Make `p` the active backend (everywhere that follows it).
    pub fn set_provider(&mut self, p: SharedProvider) {
        *self.provider_cell.lock().unwrap_or_else(|e| e.into_inner()) = p.clone();
        self.s.provider = p;
    }

    /// Wait for the LM Studio inventory (no-op once it is in).
    pub async fn await_lms(&mut self) {
        let Some(job) = self.lms_job.take() else { return };
        let list = job.await.unwrap_or_default();
        self.s.lms_model_keys = list.iter().map(|m| m.key.clone()).collect();
        self.s.lms_models = Some(list);
        // LM Studio is the PRIMARY backend: load its model with the harness's
        // context now, instead of letting the server JIT-load it at 4096 and
        // fail the first tool turn.
        if self.s.provider.name() == "lmstudio" && !self.s.model_pinned {
            let model = self.s.provider.model();
            self.lms_safe_load(&model, false).await;
        }
    }

    /// Make sure background startup (project scan, LM Studio, model
    /// resolution) has finished before the first real turn.
    pub async fn await_ready(&mut self) {
        self.await_lms().await;
        if let Some(job) = self.startup_job.take() {
            if let Ok(Some(summary)) = job.await {
                if !summary.trim().is_empty() {
                    if let Some(sys) = self.history.first_mut() {
                        sys.content.push_str("\n\nPROJECT:\n");
                        sys.content.push_str(&summary);
                    }
                }
            }
        }
        if !self.startup_resolved {
            self.startup_resolved = true;
            self.resolve_startup_model().await;
        }
    }
}

/// `qwenodyssey chat` / bare `qwenodyssey`. Returns the exit code.
pub async fn chat_command(opts: &GlobalOpts, args: &ChatArgs) -> anyhow::Result<i32> {
    let s = Session::create(opts, None)?;
    let health = s.provider.health_check().await;
    if !health.ok {
        outln!("{}", style::red(&format!("Model backend not reachable: {}", health.detail.unwrap_or_default())));
        return Ok(1);
    }
    let cfg = s.config.clone();

    // Long-term knowledge vault (RAG memory). Created on first use.
    let kb = Arc::new(KnowledgeBase::from_config(&cfg));
    let memory_enabled = cfg.knowledge.enabled;
    if memory_enabled {
        kb.ensure();
    }

    // Self-improvement: reflect on failed turns and bank lessons in the vault.
    // Reflection runs on whatever model is active at the time.
    let provider_cell = Arc::new(Mutex::new(s.provider.clone()));
    let evolution = (memory_enabled && cfg.evolution.enabled).then(|| {
        let cell = provider_cell.clone();
        let generate: qo_net::GenerateFn = Arc::new(move |messages, options| {
            let p = cell.lock().unwrap_or_else(|e| e.into_inner()).clone();
            Box::pin(async move { p.generate(&messages, &options).await })
        });
        qo_net::EvolutionEngine::new(
            kb.clone(),
            generate,
            qo_net::EvolutionOptions { enabled: true, reflect_on_failure: cfg.evolution.reflect_on_failure },
        )
    });

    // Agent swarm: usable only when ≥2 frontier workers (cloud models with
    // keys) exist. Computed up front so the system prompt and tool set agree.
    // (`swarm_chat_roster` applies the gate: swarm.enabled and ≥2 workers.)
    let swarm_workers = qo_swarm::swarm_chat_roster(&cfg);
    let swarm_ready = !swarm_workers.is_empty();

    // MCP servers (opt-in): connect + discover tools up front so the system
    // prompt and the advertised tool set agree. Kept ahead of SELF-AWARENESS so
    // a live /model switch (replace_self_awareness) never clips it.
    let mcp_specs = qo_net::mcp_servers_from_config(&cfg.mcp);
    let mcp = if mcp_specs.is_empty() {
        None
    } else {
        Some(qo_net::create_mcp_tools(mcp_specs, qo_net::McpOptions::from_config(&cfg.mcp)).await)
    };
    let mcp_servers = mcp.as_ref().map(|m| m.servers()).unwrap_or_default();

    // Base system prompt now; the PROJECT summary is appended once the repo
    // scan finishes in the background (so a slow scan doesn't delay the prompt).
    let mut sys = qo_core::prompts::load_prompt("system")?;
    for block in [CONVERSATION_GUARD, TOOL_SYSTEM, DEEP_THINK, PLAN_SYSTEM] {
        sys.push('\n');
        sys.push_str(block);
    }
    if cfg.agent.thinking_mode {
        sys.push('\n');
        sys.push_str(THINK_SYSTEM);
    }
    if memory_enabled || cfg.web.enabled {
        sys.push('\n');
        sys.push_str(MEMORY_SYSTEM);
    }
    if cfg.tools.shell_session && cfg.tools.allow_shell {
        sys.push('\n');
        sys.push_str(SHELL_SESSION_SYSTEM);
    }
    if swarm_ready {
        sys.push('\n');
        sys.push_str(SWARM_SYSTEM);
    }
    if !mcp_servers.is_empty() {
        sys.push('\n');
        sys.push_str(&mcp_system(&mcp_servers));
    }

    // Heavy startup work runs in the BACKGROUND so the banner/prompt appear
    // instantly; it is awaited (usually already finished) before the first turn.
    let lms_job = (cfg.lmstudio.enabled && s.lms.installed()).then(|| {
        let lms = s.lms.clone();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            if cfg.lmstudio.start_server {
                let port = text::lms_port(&cfg.lmstudio.base_url);
                let _ = lms.ensure_server(port, "0.0.0.0", &cfg.lmstudio.base_url).await;
            }
            // Tool-capable models first, then the rest, so a fallback lands on a
            // model that can drive the tools (embedding models excluded).
            let mut list: Vec<LmsModel> = lms.list().await.into_iter().filter(|m| !text::is_embedding_model(&m.key)).collect();
            list.sort_by_key(|m| !m.tool_use);
            list
        })
    });
    // Anthropic auth may need the `ant` CLI (up to 15 s): resolve it off the
    // prompt's critical path.
    let anthropic_auth = Arc::new(std::sync::OnceLock::new());
    {
        let cell = anthropic_auth.clone();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            let v = qo_providers::anthropic_auth_available_async(&cfg).await;
            let _ = cell.set(v);
        });
    }
    let startup_job = {
        let cwd = s.cwd.clone();
        let prewarm = cfg.agent.prebuild_symbol_index;
        Some(tokio::spawn(async move {
            let scan = async {
                let repo = qo_pipeline::scan_repo(&cwd).await;
                qo_pipeline::summarize_repo(&repo)
            };
            // Warm the symbol index so find_symbol / outline_file / read_symbol
            // are instant on first use (small models lean on them heavily).
            let warm = async {
                if prewarm {
                    qo_tools::prewarm_symbol_index(&cwd).await;
                }
            };
            let (summary, ()) = tokio::join!(scan, warm);
            Some(summary)
        }))
    };

    outln!(
        "{}",
        banner(&BannerInfo {
            version: crate::cli::VERSION,
            model: &s.provider.model(),
            provider: s.provider.name(),
            mode: s.mode.as_str(),
            cwd: &s.cwd.display().to_string(),
        })
    );

    // The ONE input owner. In a terminal: raw mode + bracketed paste for the
    // whole session (restored by the guard / panic hook / term::exit), and a
    // reader thread for key events. Piped: a line reader that queues every line.
    let interactive = std::io::stdin().is_terminal();
    let _guard = if interactive { Some(ui::term::TerminalGuard::enter()?) } else { None };
    let input = if interactive { Input::start_tty() } else { Input::start_lines() };
    let ui = Ui::new(input);

    // Dedicated tool registry for chat. autoConfirm is true because the chat
    // runs its OWN confirmation gate for shell commands (executeToolCall); the
    // hard-block list inside run_shell still applies regardless. The confirmer
    // is installed too, so any other tool asking goes through the same input.
    let mut tool_ctx = ToolContext::new(&s.cwd);
    tool_ctx.auto_confirm = true;
    tool_ctx.confirm_destructive = false;
    tool_ctx.allow_shell = cfg.tools.allow_shell;
    tool_ctx.sandbox = cfg.tools.sandbox;
    tool_ctx.allow_commands = cfg.tools.allow_commands.clone();
    tool_ctx.deny_commands = cfg.tools.deny_commands.clone();
    tool_ctx.shell_timeout_ms = Some(cfg.tools.shell_timeout_ms);
    // Writes outside the project are only allowed into the agent's own source
    // when the user opted in (tools.allow_self_edit). Reads work regardless.
    tool_ctx.self_root = cfg.tools.allow_self_edit.then(|| s.self_root.clone());
    tool_ctx.log = s.logger.log_fn();
    tool_ctx.confirmer = Some(Arc::new(TermConfirmer::new(ui.clone())));
    let mut tools = ToolRegistry::new(tool_ctx);
    qo_tools::register_defaults(&mut tools);

    // Compose the advertised tool set: shell/file/git always, plus internet and
    // long-term memory when enabled in config.
    let mut tool_specs: Vec<ToolSpec> = qo_tools::chat_tool_specs();
    tool_specs.extend(qo_tools::code_nav_tool_specs());
    if cfg.web.enabled {
        tools.register_all(qo_net::create_web_tools(qo_net::WebConfig::from_config(&cfg.web)));
        tool_specs.extend(qo_net::web_tool_specs());
    }
    if memory_enabled {
        tools.register_all(qo_net::create_knowledge_tools(kb.clone()));
        tool_specs.extend(qo_net::knowledge_tool_specs());
    }
    if swarm_ready {
        tools.register_all(qo_swarm::create_swarm_tools(&cfg));
        tool_specs.extend(qo_swarm::swarm_tool_specs());
        let labels: Vec<String> = swarm_workers.iter().map(|w| w.label.clone()).collect();
        outln!("{}", style::gray(&format!("  ✦ agent swarm ready — {} frontier models: {}\n", swarm_workers.len(), labels.join(", "))));
    }
    if cfg.github.enabled {
        tools.register_all(qo_net::create_github_tools(qo_net::GithubConfig::from_config(&cfg.github)));
        tool_specs.extend(qo_net::github_tool_specs());
        // Probe the login so the user sees who the agent acts as (best-effort,
        // in the background — never delays the prompt).
        if let Some(t) = tools.get("github_whoami") {
            let ctx = tools.context().clone();
            tokio::spawn(async move {
                let r = t.run(serde_json::json!({}), &ctx).await;
                if r.ok {
                    outln!("{}", style::gray(&format!("  ✦ GitHub ready — {}\n", r.output)));
                } else {
                    outln!("{}", style::gray("  ✦ GitHub tools loaded (not yet authenticated — run `gh auth login`)\n"));
                }
            });
        }
    }
    // MCP: register the discovered tools and advertise their schemas.
    if let Some(m) = &mcp {
        tools.register_all(m.tools.iter().cloned());
        tool_specs.extend(m.specs.iter().cloned());
        if !m.infos.is_empty() {
            outln!("{}", style::gray(&format!("  ✦ {}\n", m.infos.join("\n  ✦ "))));
        }
    }
    // Persistent shell session (opt-in): a real pty whose cwd/env/processes
    // survive across calls. Disposed when the chat exits.
    let shell_session = if cfg.tools.shell_session && cfg.tools.allow_shell {
        let session = Arc::new(qo_tools::ShellSession::new(&s.cwd));
        tools.register_all(qo_tools::create_shell_session_tools(session.clone()));
        tool_specs.extend(qo_tools::shell_session_tool_specs());
        outln!("{}", style::gray("  ✦ persistent shell session enabled (shell_session)\n"));
        Some(session)
    } else {
        None
    };
    // In-session plan / TODO tracking (always available); its spec is part of
    // the chat specs already.
    let plan_state = qo_tools::PlanState::new();
    tools.register(Arc::new(qo_tools::create_plan_tool(plan_state.clone())));
    // Thinking mode: a `think` scratchpad tool, budgeted separately.
    if cfg.agent.thinking_mode {
        tools.register(Arc::new(qo_tools::create_think_tool()));
        tool_specs.push(qo_tools::think_tool_spec());
    }

    let mut chat = Chat {
        s,
        ui,
        history: vec![Message::system(sys)],
        tools,
        tool_specs,
        kb,
        memory_enabled,
        evolution,
        provider_cell,
        meter: TokenMeter::default(),
        seen_files: Mutex::new(qo_tools::BlindOverwriteGuard::new()),
        plan_state,
        last_models: Vec::new(),
        lms_job,
        startup_job,
        startup_resolved: false,
        anthropic_auth,
    };
    let sa = chat.self_awareness_text();
    if let Some(sys) = chat.history.first_mut() {
        sys.content.push('\n');
        sys.content.push_str(&sa);
    }

    // Persisted sessions: saved after each turn so a conversation can be
    // resumed later with --continue / --resume / /resume.
    let store = SessionStore::new(None);
    let mut session_id = store.new_id();
    let mut session_created = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    // Resume a prior session if requested (replaces history after the system msg).
    if let Some((id, created, messages)) = resume_from_flags(&chat, &store, args).await {
        chat.history.extend(messages);
        session_id = id;
        session_created = created;
    }

    loop {
        let mut line = chat.ask().await.trim().to_string();
        if line.is_empty() {
            continue;
        }
        // `/btw` is mainly for typing WHILE the model works; at the prompt it's
        // a normal question, so strip the prefix and fall through.
        if line == "/btw" {
            outln!("{}", style::gray("  /btw <question> — ask an aside; also type it while the model is working to queue it\n"));
            continue;
        }
        if let Some(rest) = strip_btw(&line) {
            line = rest;
        }
        match line.as_str() {
            "/exit" | "/quit" => break,
            "/reset" => {
                chat.history.truncate(1);
                // Strip any compaction memo folded into the system prompt.
                if let Some(sys) = chat.history.first_mut() {
                    if let Some(i) = sys.content.find("\n\n[CONVERSATION SO FAR") {
                        sys.content.truncate(i);
                    }
                }
                session_id = store.new_id();
                session_created = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
                chat.plan_state.clear();
                outln!("{}", style::gray("(history cleared — new session)\n"));
                continue;
            }
            "/context" => {
                outln!("{}", views::render_context(&chat));
                continue;
            }
            "/compact" => {
                let opts = qo_net::CompactOptions { keep_user_turns: 2, cancel: None };
                let r = qo_net::compact_history(&chat.history, chat.s.provider.as_ref(), &opts).await;
                match r.new_messages.filter(|_| r.compacted) {
                    Some(new_messages) => {
                        chat.history = new_messages;
                        persist(&chat, &store, &session_id, &session_created);
                        outln!("{}", style::gray(&format!("(compacted {} earlier messages into a summary)\n", r.removed)));
                    }
                    None => outln!("{}", style::gray("(not enough history to compact yet)\n")),
                }
                continue;
            }
            "/plan" => {
                outln!(
                    "{}\n{}\n{}\n{}\n",
                    hrule(),
                    style::bold("  Current plan"),
                    crate::render::indent(&chat.plan_state.render()),
                    hrule()
                );
                continue;
            }
            "/sessions" => {
                outln!("{}", views::render_sessions(&store, &session_id));
                continue;
            }
            "/help" | "/?" | "/commands" | "/" => {
                outln!("{}", views::render_command_menu());
                continue;
            }
            "/settings" | "/config" => {
                outln!("{}", views::render_settings(&chat));
                continue;
            }
            "/memory" | "/knowledge" => {
                outln!("{}", views::render_memory(&chat));
                continue;
            }
            "/lessons" | "/evolution" => {
                outln!("{}", views::render_lessons(&chat));
                continue;
            }
            _ => {}
        }
        if line == "/resume" || line.starts_with("/resume ") {
            let arg = line.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
            match choose_session(&chat, &store, &arg).await {
                Some(picked) => {
                    if let Some(full) = store.load(&picked.id) {
                        chat.history.truncate(1);
                        let n = full.messages.len();
                        chat.history.extend(full.messages);
                        session_id = picked.id.clone();
                        session_created = picked.created_at.clone();
                        outln!(
                            "{}{}",
                            style::green(&format!("✓ resumed {}", picked.id)),
                            style::gray(&format!(" — {n} messages · \"{}\"\n", picked.title))
                        );
                    }
                }
                None => outln!("{}", style::gray("(no matching session)\n")),
            }
            continue;
        }
        if is_model_command(&line) {
            // LM Studio inventory may still be loading right after launch.
            chat.await_lms().await;
            chat.handle_models(&line).await;
            continue;
        }

        // Background startup must be done before the first real turn.
        chat.await_ready().await;

        // Auto-compact near the model's context budget, so we don't overflow
        // (or silently lose the system prompt).
        if qo_net::should_compact(
            &chat.history,
            chat.s.provider.as_ref(),
            chat.s.config.model.context_tokens as u64,
            chat.s.config.model.max_tokens as u64,
        ) {
            let opts = qo_net::CompactOptions { keep_user_turns: 3, cancel: None };
            let r = qo_net::compact_history(&chat.history, chat.s.provider.as_ref(), &opts).await;
            if let Some(new_messages) = r.new_messages.filter(|_| r.compacted) {
                chat.history = new_messages;
                outln!("{}", style::gray(&format!("  ✦ auto-compacted {} earlier messages to free up context\n", r.removed)));
            }
        }

        let mut user_content = expand_file_refs(&line, &chat.s.cwd);

        // Trivial greeting/smalltalk → answer with NO tools and NO memory cruft.
        let smalltalk = text::is_smalltalk(&line);

        // Auto-recall: prepend relevant long-term-memory notes to the USER
        // message (multiple system messages break some prompt templates).
        if chat.memory_enabled && chat.s.config.knowledge.auto_recall && !smalltalk {
            if let Some(recalled) =
                qo_net::recall_knowledge(&chat.kb, &line, chat.s.config.knowledge.recall_k as usize).await
            {
                user_content = format!("{recalled}\n\n———\n\n{user_content}");
            }
        }
        chat.history.push(Message::user(user_content));

        // Small local models get COMPACT tool specs (full descriptions across
        // 35+ tools cost thousands of tokens and hurt 7B-class selection).
        let specs: Vec<ToolSpec> = if smalltalk {
            Vec::new()
        } else if chat.s.config.agent.small_model_mode && !is_cloud_provider(chat.s.provider.name()) {
            qo_tools::compact_tool_specs(&chat.tool_specs, 180)
        } else {
            chat.tool_specs.clone()
        };
        match chat.run_assistant_turn(&specs).await {
            Ok(signals) => {
                // Evolution: reflect on rough turns and bank a lesson.
                if let Some(evo) = &chat.evolution {
                    let mut sig = signals;
                    sig.user_message = line.clone();
                    if let Some(learned) = evo.reflect(&sig).await {
                        outln!("{}", style::magenta(&format!("  ✦ learned a lesson: {learned}\n")));
                    }
                }
            }
            Err(e) => outln!("{}", style::red(&format!("\n[error: {e:#}]"))),
        }
        persist(&chat, &store, &session_id, &session_created);
    }
    persist(&chat, &store, &session_id, &session_created);
    if let Some(sh) = shell_session {
        sh.dispose();
    }
    if let Some(m) = &mcp {
        m.dispose_all();
    }
    Ok(0)
}

/// `/btw <text>` → `<text>`.
fn strip_btw(line: &str) -> Option<String> {
    let lower = line.to_lowercase();
    let rest = lower.strip_prefix("/btw")?;
    if rest.starts_with(char::is_whitespace) {
        Some(line[4..].trim().to_string())
    } else {
        None
    }
}

fn is_model_command(line: &str) -> bool {
    line == "/model" || line == "/models" || line.starts_with("/model ") || line.starts_with("/models ")
}

/// Checkpoint the session (nothing but the system prompt → nothing to save).
fn persist(chat: &Chat, store: &SessionStore, id: &str, created: &str) {
    if chat.history.len() <= 1 {
        return;
    }
    let messages: Vec<Message> = chat.history[1..].to_vec();
    let turns = messages.iter().filter(|m| m.role == Role::User).count() as u64;
    store.save(&ChatSession {
        meta: ChatSessionMeta {
            id: id.to_string(),
            cwd: chat.s.cwd.display().to_string(),
            title: derive_title(&messages),
            model: chat.s.provider.model(),
            provider: chat.s.provider.name().to_string(),
            created_at: created.to_string(),
            updated_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            turns,
        },
        messages,
    });
}

/// Resolve which saved session to resume from CLI flags (--continue / --resume[=id]).
async fn resume_from_flags(chat: &Chat, store: &SessionStore, args: &ChatArgs) -> Option<(String, String, Vec<Message>)> {
    if !args.continue_ && args.resume.is_none() {
        return None;
    }
    let meta = match args.resume.as_deref() {
        Some(id) if !id.is_empty() => match store.list().into_iter().find(|m| m.id == id) {
            Some(m) => m,
            None => {
                outln!("{}\n", style::red(&format!("No saved session \"{id}\".")));
                return None;
            }
        },
        Some(_) => choose_session(chat, store, "").await?,
        None => match store.latest_for_cwd(&chat.s.cwd) {
            Some(m) => m,
            None => {
                outln!("{}", style::gray("(no previous session in this directory — starting fresh)\n"));
                return None;
            }
        },
    };
    let full = store.load(&meta.id)?;
    let n = full.messages.len();
    outln!("{}{}", style::green(&format!("✓ resumed session {}", meta.id)), style::gray(&format!(" — {n} messages · \"{}\"\n", meta.title)));
    Some((meta.id, meta.created_at, full.messages))
}

/// Pick a saved session by id (exact/substring) or interactively when no arg.
async fn choose_session(chat: &Chat, store: &SessionStore, arg: &str) -> Option<ChatSessionMeta> {
    let all = store.list();
    if all.is_empty() {
        outln!("{}", style::gray("(no saved sessions yet)\n"));
        return None;
    }
    if !arg.is_empty() {
        return all.iter().find(|m| m.id == arg).or_else(|| all.iter().find(|m| m.id.contains(arg))).cloned();
    }
    if !chat.ui.is_tty() {
        return all.into_iter().next();
    }
    let items: Vec<ui::SelectItem> = all
        .iter()
        .take(30)
        .map(|m| ui::SelectItem {
            label: if m.title.is_empty() { m.id.clone() } else { m.title.clone() },
            hint: Some(format!(
                "{} turns · {} · {}",
                m.turns,
                m.updated_at.chars().take(16).collect::<String>().replace('T', " "),
                std::path::Path::new(&m.cwd).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()
            )),
            current: false,
        })
        .collect();
    let idx = chat.ui.select("Resume which session?", items, 0).await?;
    all.into_iter().nth(idx)
}

/// Inline `@path` file references (first 6000 chars) into the message.
pub fn expand_file_refs(line: &str, cwd: &std::path::Path) -> String {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"@(\S+)").expect("static regex"));
    RE.replace_all(line, |c: &regex::Captures| {
        let p = &c[1];
        if let Ok(abs) = qo_tools::resolve_inside(cwd, p) {
            if abs.is_file() {
                if let Ok(bytes) = std::fs::read(&abs) {
                    let content: String = String::from_utf8_lossy(&bytes).chars().take(6000).collect();
                    return format!("\n\n[file {p}]\n```\n{content}\n```\n");
                }
            }
        }
        c[0].to_string()
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btw_prefix() {
        assert_eq!(strip_btw("/btw what about x"), Some("what about x".into()));
        assert_eq!(strip_btw("/BTW  hi"), Some("hi".into()));
        assert_eq!(strip_btw("/btwx"), None);
        assert!(is_model_command("/model 3"));
        assert!(is_model_command("/models"));
        assert!(!is_model_command("/modelx"));
    }

    #[test]
    fn meter_prefers_exact_usage() {
        let mut m = TokenMeter::default();
        let r = GenerateResult { prompt_tokens: Some(100), completion_tokens: None, ..Default::default() };
        m.record(&r, 50, 7);
        assert_eq!((m.last_in, m.last_out, m.session_in, m.session_out), (100, 7, 100, 7));
        m.record(&r, 1, 1);
        assert_eq!(m.session_in, 200);
    }

    #[test]
    fn file_refs_inline_project_files_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let out = expand_file_refs("see @a.txt and @missing.txt", dir.path());
        assert!(out.contains("[file a.txt]\n```\nhello\n```"));
        assert!(out.contains("@missing.txt"));
    }

    #[test]
    fn palette_has_every_command() {
        let names: Vec<&str> = SLASH_COMMANDS.iter().map(|c| c.name).collect();
        for n in ["/help", "/settings", "/models", "/model", "/context", "/compact", "/plan", "/memory", "/lessons", "/sessions", "/resume", "/btw", "/reset", "/exit"] {
            assert!(names.contains(&n), "{n}");
        }
    }
}
