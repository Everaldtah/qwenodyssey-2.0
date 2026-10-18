//! Model management for the chat (ports of chat.ts `fallbackChain`,
//! `applyModelRef`, `lmsSafeLoad`, `probeModel`, `switchModel`,
//! `resolveStartupModel`, `streamWithFallback`, `generateWithFallback`,
//! `gatherModelEntries`, `handleModels`) — including the v0.4.2 freeze fixes:
//! cloud models are probed after a manual switch (15 s, Esc skips), a dead
//! model asks before being kept, the fallback chain never retries a model that
//! already failed this turn, an interrupt never falls back, and every fallback
//! notice says why.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qo_core::error::{is_interrupt, looks_context_overflow, looks_unavailable, no_tool_support};
use qo_core::{CancellationToken, GenerateResult, ModelInfo, ModelKind, ModelOptions, ModelRef, ProviderError, SharedProvider};
use qo_providers::{LmsModel, SafeLoadOpts};

use super::text::{clip, is_embedding_model, squash_ws};
use super::Chat;
use crate::outln;
use crate::render::format_tokens;
use crate::ui::{style, InterjectOpts, SelectItem, Spinner};

/// Backends reached over the internet; local ones load models on demand, so
/// they are not probed.
pub fn probed_kind(kind: ModelKind) -> bool {
    kind.is_cloud()
}
pub const PROBE_TIMEOUT: Duration = Duration::from_millis(15_000);

/// Outcome of [`Chat::probe_model`].
#[derive(Debug)]
pub struct ProbeOutcome {
    pub ok: bool,
    pub ms: u128,
    pub skipped: bool,
    pub detail: Option<String>,
}

/// One selectable model in the picker.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelEntry {
    /// "lmstudio:<key>", "nvidia:<model>", … or a bare Ollama tag.
    pub r#ref: String,
    pub label: String,
    pub hint: String,
    pub current: bool,
}

/// The Claude lineup offered whenever Anthropic auth is available.
const CLAUDE_MODELS: &[&str] = &["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-7", "claude-fable-5"];

impl Chat {
    pub fn nvidia_key_present(&self) -> bool {
        !qo_providers::resolve_nvidia_key(&self.s.config).trim().is_empty()
    }
    pub fn openrouter_key_present(&self) -> bool {
        !qo_providers::resolve_openrouter_key(&self.s.config).trim().is_empty()
    }
    /// Anthropic credential known to be available (cached; the check may run
    /// the `ant` CLI, so it is resolved in the background at startup and an
    /// unknown answer counts as "no" in synchronous paths).
    pub fn anthropic_auth_present(&self) -> bool {
        self.anthropic_auth.get().copied().unwrap_or(false)
    }

    /// Anthropic auth, resolving (and caching) it now if still unknown.
    pub async fn anthropic_auth_now(&self) -> bool {
        if let Some(v) = self.anthropic_auth.get() {
            return *v;
        }
        let v = qo_providers::anthropic_auth_available_async(&self.s.config).await;
        let _ = self.anthropic_auth.set(v);
        v
    }

    /// The fallback-chain ref of the active model.
    pub fn active_ref(&self) -> String {
        ModelRef::of(self.s.provider.name(), &self.s.provider.model())
    }

    /// Ordered fallback chain (refs), de-duped against the active model: the
    /// configured fallbacks plus every installed LM Studio model (tool-capable
    /// first) when include_as_fallback is on. Cloud refs need a key/auth.
    pub fn fallback_chain(&self) -> Vec<String> {
        let cfg = &self.s.config;
        let mut refs: Vec<String> = cfg.model.fallback_models.iter().map(|m| m.trim().to_string()).collect();
        if cfg.lmstudio.enabled && cfg.lmstudio.include_as_fallback {
            refs.extend(self.s.lms_model_keys.iter().map(|k| format!("lmstudio:{k}")));
        }
        let nvidia_ok = cfg.nvidia.enabled && cfg.nvidia.include_as_fallback && self.nvidia_key_present();
        let openrouter_ok = cfg.openrouter.enabled && cfg.openrouter.include_as_fallback && self.openrouter_key_present();
        let anthropic_ok = cfg.anthropic.enabled && cfg.anthropic.include_as_fallback && self.anthropic_auth_present();
        let active_kind = ModelKind::from_provider_name(self.s.provider.name()).unwrap_or(ModelKind::Ollama);
        let active_model = self.s.provider.model().to_lowercase();
        let mut seen = HashSet::new();
        refs.into_iter()
            .filter(|r| {
                if r.is_empty() {
                    return false;
                }
                let mr = ModelRef::parse(r);
                let ok = match mr.kind {
                    ModelKind::Nvidia => nvidia_ok,
                    ModelKind::OpenRouter => openrouter_ok,
                    ModelKind::Anthropic => anthropic_ok,
                    _ => true,
                };
                // Never "fall back" to the backend+model already active.
                ok && !(mr.kind == active_kind && mr.model.to_lowercase() == active_model) && seen.insert(r.clone())
            })
            .collect()
    }

    /// Switch the active backend+model. LM Studio models are safe-loaded first
    /// (capped context / partial GPU for big ones); Ollama always gets a real
    /// Ollama provider (the configured primary may be a cloud backend).
    pub async fn apply_model_ref(&mut self, r: &str) {
        let mr = ModelRef::parse(r);
        let cfg = &self.s.config;
        let next: SharedProvider = match mr.kind {
            ModelKind::Nvidia => qo_providers::create_nvidia_provider(cfg, &mr.model),
            ModelKind::OpenRouter => qo_providers::create_openrouter_provider(cfg, &mr.model),
            ModelKind::Anthropic => qo_providers::create_anthropic_provider(cfg, &mr.model),
            ModelKind::LmStudio => {
                self.lms_safe_load(&mr.model, false).await;
                qo_providers::create_lmstudio_provider(&self.s.config, &mr.model)
            }
            ModelKind::Ollama => {
                if self.s.provider.name() == "ollama" {
                    self.s.provider.set_model(&mr.model);
                    self.s.provider.clone()
                } else {
                    qo_providers::create_ollama_provider(cfg, &mr.model)
                }
            }
        };
        self.set_provider(next);
        self.refresh_identity();
    }

    fn safe_load_opts(&self, force: bool) -> SafeLoadOpts {
        let l = &self.s.config.lmstudio;
        SafeLoadOpts {
            big_params_b: l.big_params_b,
            big_size_gb: l.big_size_gb,
            big_context: l.big_context as u64,
            ttl_seconds: l.ttl_seconds,
            context_tokens: self.s.config.model.context_tokens as u64,
            force,
        }
    }

    /// Load an LM Studio model with the harness's context window (capped for
    /// big models) before pointing requests at it. Best-effort; true when a
    /// (re)load actually happened.
    pub async fn lms_safe_load(&mut self, model: &str, force: bool) -> bool {
        if !self.s.config.lmstudio.safe_load || !self.s.lms.installed() {
            return false;
        }
        let list: Vec<LmsModel> = match &self.s.lms_models {
            Some(l) => l.clone(),
            None => self.s.lms.list().await,
        };
        let Some(m) = list.into_iter().find(|x| x.key == model) else { return false };
        let opts = self.safe_load_opts(force);
        let Ok(r) = self.s.lms.safe_load(&m, &opts, None).await else { return false };
        if r.skipped {
            return false;
        }
        if r.ok {
            let gpu = self.s.lms.gpu_mem().await;
            let ctx = self.s.lms.context_for(&m, &opts, gpu);
            let how = if r.big { " (big for this GPU: capped context, partial offload, auto-unload)" } else { "" };
            let with = if ctx > 0 { format!(" with a {}-token context", format_tokens(ctx)) } else { String::new() };
            outln!("{}", style::gray(&format!("  ✦ LM Studio: {} {model}{with}{how}", if r.reloaded { "reloaded" } else { "loaded" })));
        } else if !r.detail.is_empty() {
            let last = r.detail.lines().last().unwrap_or("");
            outln!("{}", style::gray(&format!("  (LM Studio could not load {model}: {last})")));
        }
        r.ok
    }

    /// One tiny request to see whether the active model answers at all. Some
    /// hosted models accept a request and never send a byte, which made the
    /// first turn after a /model switch look like a frozen terminal. Esc skips.
    pub async fn probe_model(&self) -> ProbeOutcome {
        let cancel = CancellationToken::new();
        let skipped = Arc::new(AtomicBool::new(false));
        let typed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let esc = {
            let (c, sk, ty) = (cancel.clone(), skipped.clone(), typed.clone());
            self.ui.start_interjector(InterjectOpts {
                label: String::new(),
                // Type-ahead during the probe is kept for the prompt.
                on_submit: Box::new(move |t| ty.lock().unwrap_or_else(|e| e.into_inner()).push(t)),
                on_start_typing: None,
                on_stop_typing: None,
                on_interrupt: Some(Box::new(move || {
                    sk.store(true, Ordering::SeqCst);
                    c.cancel();
                })),
            })
        };
        let spinner = Spinner::new(&format!("checking {}", self.s.provider.model()), 0, "esc to skip");
        spinner.begin();
        let t0 = Instant::now();
        let provider = self.s.provider.clone();
        let res = tokio::time::timeout(PROBE_TIMEOUT, provider.probe(&cancel)).await;
        let ms = t0.elapsed().as_millis();
        spinner.stop();
        let leftovers = std::mem::take(&mut *typed.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(ij) = esc {
            self.ui.stop_interjector(ij, leftovers).await;
        }
        match res {
            Ok(Ok(())) => ProbeOutcome { ok: true, ms, skipped: false, detail: None },
            Ok(Err(_)) if skipped.load(Ordering::SeqCst) => ProbeOutcome { ok: true, ms, skipped: true, detail: None },
            Ok(Err(e)) => ProbeOutcome { ok: false, ms, skipped: false, detail: Some(clip(&squash_ws(&format!("{e:#}")), 160)) },
            Err(_) => {
                cancel.cancel();
                ProbeOutcome { ok: false, ms, skipped: false, detail: Some(format!("no reply within {}s", PROBE_TIMEOUT.as_secs())) }
            }
        }
    }

    /// Manual model switch (/model, picker). Cloud models get a liveness probe
    /// first; if the model doesn't answer the user decides whether to keep it,
    /// instead of discovering it as a "stuck" first turn.
    pub async fn switch_model(&mut self, r: &str) {
        let prev = self.s.provider.clone();
        self.s.model_pinned = true; // manual choice beats background startup resolution
        self.apply_model_ref(r).await;
        let switched = |c: &Chat| {
            format!("{}{}", style::green(&format!("✓ switched to {}", c.s.provider.model())), style::gray(&format!(" ({})", c.s.provider.name())))
        };
        if !probed_kind(ModelRef::parse(r).kind) || !self.ui.is_tty() {
            outln!("{}\n", switched(self));
            return;
        }
        let p = self.probe_model().await;
        if p.ok {
            let note = if p.skipped { "check skipped".to_string() } else { format!("answered in {:.1}s", p.ms as f64 / 1000.0) };
            outln!("{}{}", switched(self), style::gray(&format!(" · {note}\n")));
            return;
        }
        outln!("{}", style::yellow(&format!("⚠ {} is not answering: {}", self.s.provider.model(), p.detail.unwrap_or_default())));
        outln!("{}", style::gray(&format!("  Keep it anyway? Type y to keep, anything else to stay on {}.", prev.model())));
        let answer = self.ask().await;
        if crate::ui::is_yes(&answer) {
            outln!("{}{}", switched(self), style::gray(" · kept although it did not answer; Esc interrupts a stuck turn\n"));
            return;
        }
        self.set_provider(prev);
        self.refresh_identity();
        outln!("{}", style::gray(&format!("(stayed on {})\n", self.s.provider.model())));
    }

    /// At launch, make sure the active model is actually available; if not,
    /// switch to the first available model in the fallback chain. Best-effort.
    pub async fn resolve_startup_model(&mut self) {
        // The user already chose a model by hand while startup was running.
        if self.s.model_pinned {
            return;
        }
        let name = self.s.provider.name().to_string();
        let configured = self.s.config.model.model.clone();
        match name.as_str() {
            // LM Studio primary: fine when installed there. An empty key list
            // means `lms` couldn't tell us — assume ok rather than guess wrong.
            "lmstudio" => {
                let model = self.s.provider.model();
                if self.s.lms_model_keys.is_empty() || self.s.lms_model_keys.contains(&model) {
                    return;
                }
                outln!("{}", style::yellow(&format!("⚠ \"{model}\" is not installed in LM Studio. Trying fallbacks.")));
            }
            // Cloud primaries: usable as long as we have a key (assume reachable).
            "nvidia" => {
                if self.nvidia_key_present() {
                    return;
                }
                outln!(
                    "{}",
                    style::yellow(&format!(
                        "⚠ \"{configured}\" needs an NVIDIA API key (set NVIDIA_API_KEY or [nvidia].api_key) — not found. Trying local fallbacks."
                    ))
                );
            }
            "openrouter" => {
                if self.openrouter_key_present() {
                    return;
                }
                outln!(
                    "{}",
                    style::yellow(&format!(
                        "⚠ \"{configured}\" needs an OpenRouter API key (set OPENROUTER_API_KEY or [openrouter].api_key) — not found. Trying local fallbacks."
                    ))
                );
            }
            "anthropic" => {
                if self.anthropic_auth_now().await {
                    return;
                }
                outln!(
                    "{}",
                    style::yellow(&format!(
                        "⚠ \"{configured}\" needs Anthropic auth — log in with `ant auth login` (Claude subscription) or set ANTHROPIC_API_KEY. Trying local fallbacks."
                    ))
                );
            }
            _ => {}
        }

        let anthropic = self.anthropic_auth_now().await;
        let chain = self.fallback_chain();
        if chain.is_empty() {
            return;
        }
        // Ollama's installed tags, gathered independently of the active
        // provider so Ollama fallbacks can be judged under any primary.
        let ollama_tags = self.ollama_tags().await;
        let ollama_has = |n: &str| {
            let n = n.to_lowercase();
            ollama_tags.iter().any(|m| {
                let t = m.name.to_lowercase();
                t == n || t.starts_with(&n)
            })
        };
        if name == "ollama" && (ollama_has(&self.s.provider.model()) || ollama_tags.is_empty()) {
            return; // primary ok, or Ollama down — can't tell
        }
        let pick = chain.iter().find(|r| {
            let mr = ModelRef::parse(r);
            match mr.kind {
                ModelKind::LmStudio => self.s.lms_model_keys.contains(&mr.model),
                ModelKind::Nvidia => self.nvidia_key_present(),
                ModelKind::OpenRouter => self.openrouter_key_present(),
                ModelKind::Anthropic => anthropic,
                ModelKind::Ollama => ollama_has(&mr.model),
            }
        });
        match pick {
            Some(p) => {
                let p = p.clone();
                let was = self.s.provider.model();
                if self.s.model_pinned {
                    return;
                }
                self.apply_model_ref(&p).await;
                outln!("{}", style::yellow(&format!("⚠ \"{was}\" not available — using fallback \"{p}\".")));
            }
            None if !ollama_tags.is_empty() || name != "ollama" => {
                outln!("{}", style::yellow(&format!("⚠ \"{configured}\" and all fallbacks are unavailable. Pick one with /model.")));
            }
            None => {}
        }
    }

    /// Installed Ollama tags (empty when Ollama isn't running).
    async fn ollama_tags(&self) -> Vec<ModelInfo> {
        let op = if self.s.provider.name() == "ollama" {
            self.s.provider.clone()
        } else {
            qo_providers::create_ollama_provider(&self.s.config, &self.s.provider.model())
        };
        // Bounded: a wedged local server must not hold the prompt.
        match tokio::time::timeout(Duration::from_secs(5), op.list_models()).await {
            Ok(Ok(v)) => v,
            _ => Vec::new(),
        }
    }

    /// Blocking completion against the active model, retried WITHOUT tools if
    /// the model rejects tool-calling (e.g. deepseek-r1).
    async fn try_generate(&self, opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
        let p = self.s.provider.clone();
        match p.generate(&self.history, opts).await {
            Ok(r) => Ok(r),
            Err(e) if opts.tools.as_ref().is_some_and(|t| !t.is_empty()) && no_tool_support(&e) => {
                let mut o = opts.clone();
                o.tools = None;
                p.generate(&self.history, &o).await
            }
            Err(e) => Err(e),
        }
    }

    /// Generate, walking the fallback chain when the active model/backend is
    /// unavailable. Switches are sticky. Models in `tried` already failed this
    /// turn and are never retried.
    pub async fn generate_with_fallback(&mut self, opts: &ModelOptions, tried: &mut HashSet<String>) -> anyhow::Result<GenerateResult> {
        let err = match self.try_generate(opts).await {
            Ok(r) => return Ok(r),
            Err(e) => e,
        };
        if is_interrupt(&err) || !looks_unavailable(&err) {
            return Err(err);
        }
        let mut last = format!("{err:#}");
        tried.insert(self.active_ref());
        for fb in self.fallback_chain() {
            if tried.contains(&fb) {
                continue; // already failed this turn — never retry it
            }
            if opts.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(ProviderError::Interrupted.into());
            }
            tried.insert(fb.clone());
            outln!("{}", self.fallback_notice(&fb, &last));
            self.apply_model_ref(&fb).await;
            match self.try_generate(opts).await {
                Ok(r) => return Ok(r),
                Err(e2) => {
                    if is_interrupt(&e2) || !looks_unavailable(&e2) {
                        return Err(e2);
                    }
                    last = format!("{e2:#}");
                }
            }
        }
        Err(err) // nothing in the chain worked
    }

    /// "⚠ X unavailable — falling back to Y." plus the reason.
    fn fallback_notice(&self, next: &str, why: &str) -> String {
        format!(
            "{}{}",
            style::yellow(&format!("\n⚠ \"{}\" unavailable — falling back to \"{next}\".", self.s.provider.model())),
            style::gray(&format!("\n  ({})", clip(&squash_ws(why), 110)))
        )
    }

    /// One streaming attempt against the active model: drives the live ↓
    /// counter (re-tokenized every ~32 chars), reconciles ↑/↓ to exact usage,
    /// and retries once without tools if the model rejects tool-calling.
    async fn attempt_stream(&self, opts: &ModelOptions, spinner: &Spinner) -> anyhow::Result<GenerateResult> {
        let p = self.s.provider.clone();
        let run = |o: ModelOptions| {
            let p = p.clone();
            let spinner = spinner.clone();
            let history = &self.history;
            async move {
                let mut streamed = String::new();
                let mut last_counted = 0usize;
                let mut since = 0usize;
                let counter = p.clone();
                let mut on_chunk = |delta: &str| {
                    streamed.push_str(delta);
                    since += delta.len();
                    let boundary = delta.ends_with(|c: char| c.is_whitespace() || ".,;:})]".contains(c));
                    if since >= 32 || boundary {
                        let est = counter.count_tokens(&streamed);
                        if est > last_counted {
                            spinner.bump_out((est - last_counted) as u64);
                            last_counted = est;
                        }
                        since = 0;
                    }
                };
                p.stream(history, &mut on_chunk, &o).await
            }
        };
        let res = match run(opts.clone()).await {
            Ok(r) => r,
            Err(e) if opts.tools.as_ref().is_some_and(|t| !t.is_empty()) && no_tool_support(&e) => {
                let mut o = opts.clone();
                o.tools = None;
                run(o).await?
            }
            Err(e) => return Err(e),
        };
        if let Some(n) = res.completion_tokens {
            spinner.set_out(n);
        }
        if let Some(n) = res.prompt_tokens {
            spinner.set_up(n);
        }
        Ok(res)
    }

    /// Streaming completion with the fallback chain (the v0.4.2 rules: an
    /// interrupt never falls back; models that failed this turn are never
    /// retried; Esc during a fallback stream is still an interrupt).
    pub async fn stream_with_fallback(&mut self, opts: &ModelOptions, spinner: &Spinner) -> anyhow::Result<GenerateResult> {
        // Seed ↑ with the measured prompt size so the bar starts truthful.
        spinner.set_up(self.history_tokens());
        let err = match self.attempt_stream(opts, spinner).await {
            Ok(r) => return Ok(r),
            Err(e) => e,
        };
        if is_interrupt(&err) {
            return Err(err); // user interrupt — never fall back or retry
        }
        // Prompt outgrew the context LM Studio loaded the model with → reload
        // with the configured context and retry once.
        if self.s.provider.name() == "lmstudio" && looks_context_overflow(&err) {
            outln!(
                "{}",
                style::yellow(&format!(
                    "\n⚠ \"{}\" was loaded with too small a context — reloading with {} tokens.",
                    self.s.provider.model(),
                    format_tokens(self.s.config.model.context_tokens as u64)
                ))
            );
            let model = self.s.provider.model();
            if self.lms_safe_load(&model, true).await {
                return self.attempt_stream(opts, spinner).await;
            }
        }
        let mut tried = HashSet::new();
        if !looks_unavailable(&err) {
            // Streaming failed for a non-availability reason — fall back to the
            // blocking path so the turn still completes.
            return self.generate_with_fallback(opts, &mut tried).await;
        }
        let mut last = format!("{err:#}");
        tried.insert(self.active_ref());
        for fb in self.fallback_chain() {
            if tried.contains(&fb) {
                continue;
            }
            if opts.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(ProviderError::Interrupted.into());
            }
            tried.insert(fb.clone());
            outln!("{}", self.fallback_notice(&fb, &last));
            self.apply_model_ref(&fb).await;
            match self.attempt_stream(opts, spinner).await {
                Ok(r) => return Ok(r),
                Err(e2) => {
                    // Esc during a fallback stream used to drop into the
                    // blocking path, which ignored Esc.
                    if is_interrupt(&e2) {
                        return Err(e2);
                    }
                    if !looks_unavailable(&e2) {
                        return self.generate_with_fallback(opts, &mut tried).await;
                    }
                    last = format!("{e2:#}");
                }
            }
        }
        Err(err)
    }

    /// Gather selectable models: Ollama tags (whenever Ollama is reachable),
    /// LM Studio models, and the configured cloud models with keys.
    pub async fn gather_model_entries(&self) -> Vec<ModelEntry> {
        let cfg = &self.s.config;
        let name = self.s.provider.name().to_string();
        let model = self.s.provider.model();
        let mut entries = Vec::new();

        let mut ollama = self.ollama_tags().await;
        if ollama.is_empty() {
            ollama = self.last_models.clone(); // a previously fetched list
        }
        for m in ollama {
            if is_embedding_model(&m.name) {
                continue;
            }
            let hint = [m.size.clone().unwrap_or_default(), "ollama".into()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · ");
            entries.push(ModelEntry { current: name == "ollama" && m.name == model, r#ref: m.name.clone(), label: m.name, hint });
        }

        if cfg.lmstudio.enabled && self.s.lms.installed() {
            let lms = match &self.s.lms_models {
                Some(l) => l.clone(),
                None => self.s.lms.list().await,
            };
            for m in lms {
                let big = m.params_b >= cfg.lmstudio.big_params_b || m.size_gb >= cfg.lmstudio.big_size_gb;
                entries.push(ModelEntry {
                    r#ref: format!("lmstudio:{}", m.key),
                    label: m.key.clone(),
                    hint: format!("{:.1}GB · lms{}{}", m.size_gb, if m.tool_use { " · tools" } else { "" }, if big { " · big" } else { "" }),
                    current: name == "lmstudio" && m.key == model,
                });
            }
        }

        let cloud = |prefix: &str, primary: bool| -> Vec<String> {
            let mut set: Vec<String> = Vec::new();
            if primary {
                set.push(cfg.model.model.clone());
            }
            for r in &cfg.model.fallback_models {
                if let Some(m) = r.trim().strip_prefix(prefix) {
                    if !set.iter().any(|x| x == m) {
                        set.push(m.to_string());
                    }
                }
            }
            set
        };
        use qo_core::config::ProviderKind;
        if cfg.nvidia.enabled && self.nvidia_key_present() {
            for m in cloud("nvidia:", cfg.model.provider == ProviderKind::Nvidia) {
                entries.push(ModelEntry { r#ref: format!("nvidia:{m}"), current: name == "nvidia" && m == model, label: m, hint: "nvidia · cloud".into() });
            }
        }
        if cfg.openrouter.enabled && self.openrouter_key_present() {
            for m in cloud("openrouter:", cfg.model.provider == ProviderKind::OpenRouter) {
                entries.push(ModelEntry {
                    r#ref: format!("openrouter:{m}"),
                    current: name == "openrouter" && m == model,
                    label: m,
                    hint: "openrouter · cloud".into(),
                });
            }
        }
        if self.anthropic_auth_now().await {
            let mut set: Vec<String> = CLAUDE_MODELS.iter().map(|s| s.to_string()).collect();
            for m in cloud("anthropic:", cfg.model.provider == ProviderKind::Anthropic) {
                if !set.contains(&m) {
                    set.push(m);
                }
            }
            for m in set {
                entries.push(ModelEntry {
                    r#ref: format!("anthropic:{m}"),
                    current: name == "anthropic" && m == model,
                    label: m,
                    hint: "anthropic · claude".into(),
                });
            }
        }
        entries
    }

    /// `/models` opens a picker over every backend; `/model <name|#>` switches
    /// directly.
    pub async fn handle_models(&mut self, line: &str) {
        let arg = line.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
        let entries = self.gather_model_entries().await;
        // Cache only real Ollama tags (1.x cached every non-LM-Studio ref, so a
        // later listing with Ollama down showed the cloud models twice).
        let ollama_list: Vec<ModelInfo> = entries
            .iter()
            .filter(|e| ModelRef::parse(&e.r#ref).kind == ModelKind::Ollama)
            .map(|e| ModelInfo { name: e.r#ref.clone(), size: None })
            .collect();

        if entries.is_empty() {
            outln!("{}", style::gray("(no models found — pull one with `ollama pull <name>` or install via LM Studio)\n"));
            return;
        }

        if arg.is_empty() {
            if self.ui.is_tty() {
                let current = entries.iter().position(|e| e.current).unwrap_or(0);
                let items: Vec<SelectItem> =
                    entries.iter().map(|e| SelectItem { label: e.label.clone(), hint: Some(e.hint.clone()), current: e.current }).collect();
                match self.ui.select("Select a model  (● = current)", items, current).await {
                    None => outln!("{}", style::gray("(model unchanged)\n")),
                    Some(i) if entries[i].current => outln!("{}", style::gray(&format!("(already on {})\n", entries[i].label))),
                    Some(i) => {
                        let r = entries[i].r#ref.clone();
                        self.switch_model(&r).await;
                    }
                }
                self.last_models = ollama_list;
                return;
            }
            outln!("{}", style::bold("Models (Ollama + LM Studio):"));
            for (i, e) in entries.iter().enumerate() {
                let marker = if e.current { style::green("●") } else { style::gray("○") };
                outln!("  {marker} {}  {}  {}", style::gray(&format!("{:>2}", i + 1)), e.label, style::gray(&e.hint));
            }
            outln!("{}", style::gray("\nSwitch with /model <name> or /model <number>\n"));
            self.last_models = ollama_list;
            return;
        }

        // Argument → resolve by 1-based index or by name/key and switch.
        let target = match arg.parse::<usize>() {
            Ok(n) if n >= 1 && n <= entries.len() => Some(entries[n - 1].clone()),
            _ => {
                let a = arg.to_lowercase();
                entries
                    .iter()
                    .find(|e| e.label.to_lowercase() == a || e.r#ref.to_lowercase() == a)
                    .or_else(|| entries.iter().find(|e| e.label.to_lowercase().contains(&a)))
                    .cloned()
            }
        };
        match target {
            Some(t) => self.switch_model(&t.r#ref).await,
            None => outln!("{}{}", style::red(&format!("[no model matches \"{arg}\"]")), style::gray(" — run /model to see the list.\n")),
        }
        self.last_models = ollama_list;
    }
}
