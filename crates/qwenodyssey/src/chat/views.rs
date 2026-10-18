//! Static chat views: /help, /context, /sessions, /settings, /memory,
//! /lessons (ports of the render* helpers in src/cli/commands/chat.ts).

use qo_core::config::MODEL_DEFAULTS;
use qo_core::model_profile::sampling_for;
use qo_net::session::SessionStore;

use super::{Chat, SLASH_COMMANDS};
use crate::render::{format_tokens, hrule};
use crate::ui::{style, SlashCommand};

const CHAT_TOOLS_LINE: &str = "run_shell, read/write_file, grep, git, web_search, web_fetch, knowledge_search/save, update_plan";

fn invocation(c: &SlashCommand) -> String {
    let mut names = vec![c.name];
    names.extend(c.aliases.iter().copied());
    let mut s = names.join("  ");
    if let Some(a) = c.args {
        s.push(' ');
        s.push_str(a);
    }
    s
}

/// Slash commands as an aligned two-column menu, boxed in rules.
pub fn render_command_menu() -> String {
    let pad = SLASH_COMMANDS.iter().map(|c| invocation(c).chars().count()).max().unwrap_or(0) + 3;
    let mut lines = vec![hrule(), style::bold("  Commands")];
    for c in SLASH_COMMANDS {
        lines.push(format!("  {}{}", style::cyan(&style::pad_end(&invocation(c), pad)), style::gray(c.desc)));
    }
    lines.push(format!("  {}{}", style::cyan(&style::pad_end("@path", pad)), style::gray("Inline a file's contents into your message")));
    lines.push(String::new());
    lines.push(format!("{}{}{}", style::gray("  Tip: type "), style::cyan("/"), style::gray(" to filter commands as you type.")));
    lines.push(format!("{}{}", style::gray("  The model can also call tools on its own: "), style::dim(CHAT_TOOLS_LINE)));
    lines.push(format!("{}\n", hrule()));
    lines.join("\n")
}

/// Context-usage bar: estimated history tokens vs the model's budget.
pub fn render_context(chat: &Chat) -> String {
    let used = chat.history_tokens();
    let budget = chat.s.config.model.context_tokens as u64;
    let pct = ((used as f64 / budget.max(1) as f64) * 100.0).round().min(100.0) as u64;
    let width = 28u64;
    let filled = ((pct as f64 / 100.0) * width as f64).round().clamp(0.0, width as f64) as usize;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(width as usize - filled));
    let color: fn(&str) -> String = if pct >= 85 {
        style::red
    } else if pct >= 65 {
        style::yellow
    } else {
        style::green
    };
    let msgs = chat.history.len().saturating_sub(1);
    [
        hrule(),
        style::bold("  Context usage"),
        format!("  {}  {pct}%", color(&bar)),
        format!(
            "  {}{} / {} tokens{}",
            style::gray("estimated "),
            format_tokens(used),
            format_tokens(budget),
            style::gray(&format!("  ·  {msgs} messages"))
        ),
        format!("  {}{}", style::gray("auto-compacts near the limit; force now with "), style::cyan("/compact")),
        format!("{}\n", hrule()),
    ]
    .join("\n")
}

fn short_ts(ts: &str) -> String {
    ts.chars().take(16).collect::<String>().replace('T', " ")
}

/// The saved-session list for /sessions.
pub fn render_sessions(store: &SessionStore, current_id: &str) -> String {
    let all = store.list();
    let mut lines = vec![hrule(), style::bold(&format!("  Saved sessions  ({})", all.len())), String::new()];
    if all.is_empty() {
        lines.push(format!("  {}", style::gray("(none yet)")));
    }
    for m in all.iter().take(30) {
        lines.push(format!(
            "  {}{}  {}{}",
            if m.id == current_id { style::green("● ") } else { style::gray("○ ") },
            style::cyan(&m.id),
            m.title,
            style::dim(&format!("  · {} turns · {}", m.turns, short_ts(&m.updated_at)))
        ));
    }
    lines.push(String::new());
    lines.push(format!(
        "{}{}{}{}",
        style::gray("  Resume with "),
        style::cyan("/resume <id>"),
        style::gray(", or relaunch "),
        style::cyan("qwenodyssey chat --resume")
    ));
    lines.push(format!("{}\n", hrule()));
    lines.join("\n")
}

/// How one optional sampling knob is resolved: an explicit config value, the
/// model profile's recommendation, or off (left to the backend's default).
fn sampling_knob<T: PartialEq + std::fmt::Display>(configured: T, default: T, from_profile: Option<T>, auto_tune: bool) -> String {
    if configured != default {
        return configured.to_string();
    }
    match from_profile {
        Some(p) if auto_tune => format!("{p} (auto)"),
        _ => "off".into(),
    }
}

/// Human-readable GPU offload policy (Ollama num_gpu).
pub fn gpu_policy(chat: &Chat) -> String {
    if chat.s.provider.name() != "ollama" {
        return format!("managed by {}", chat.s.provider.name());
    }
    let n = chat.s.config.model.gpu_layers;
    let lv = if chat.s.config.model.low_vram { " · low-vram" } else { "" };
    if n < 0 {
        format!("auto · max layers on GPU, overflow → CPU/RAM{lv}")
    } else if n == 0 {
        format!("CPU only{lv}")
    } else {
        format!("{n} layers forced on GPU{lv}")
    }
}

/// The current model + runtime settings as an aligned table.
pub fn render_settings(chat: &Chat) -> String {
    let cfg = &chat.s.config;
    let m = &cfg.model;
    let (profile, thinking) = chat.turn_profile();
    let sampling = sampling_for(&profile, thinking);
    let active_model = chat.s.provider.model();
    let embed = match chat.kb.embeddings_active() {
        Some(true) => format!("semantic ({})", cfg.knowledge.embed_model),
        Some(false) => "keyword (embed model not pulled)".into(),
        None => format!("semantic if available ({})", cfg.knowledge.embed_model),
    };
    let fb: Vec<String> = m.fallback_models.iter().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
    let fb_display = if fb.is_empty() {
        "—".to_string()
    } else {
        fb.iter()
            .map(|f| if f.to_lowercase() == active_model.to_lowercase() { format!("{f} (active)") } else { f.clone() })
            .collect::<Vec<_>>()
            .join(" → ")
    };
    let think_mode = m.think.as_str();
    let thinking_row = if !profile.reasoning {
        "step-by-step scaffold".to_string()
    } else if thinking == Some(true) {
        format!("deep · native reasoning ({think_mode})")
    } else {
        format!("off for speed · scaffold only ({think_mode}{})", if profile.hybrid_thinking { ", hybrid" } else { "" })
    };
    let top_p = if m.auto_tune && m.top_p == MODEL_DEFAULTS.top_p { sampling.top_p } else { m.top_p };
    let rows: Vec<(&str, String)> = vec![
        ("model", format!("{active_model}  ({} @ {})", chat.s.provider.name(), m.base_url)),
        ("fallback chain", fb_display),
        ("model profile", format!("{} · {}", profile.family, if m.auto_tune { "auto-tuned" } else { "auto-tune off" })),
        ("thinking", thinking_row),
        (
            "temperature",
            format!(
                "{} active · {} base{}",
                chat.turn_temperature(&profile, thinking),
                m.temperature,
                if m.auto_tune { format!(" · profile {}", sampling.temperature) } else { String::new() }
            ),
        ),
        (
            "sampling",
            [
                format!("top_p {top_p}"),
                format!("top_k {}", sampling_knob(m.top_k, MODEL_DEFAULTS.top_k, sampling.top_k, m.auto_tune)),
                format!("repeat {}", sampling_knob(m.repeat_penalty, MODEL_DEFAULTS.repeat_penalty, sampling.repeat_penalty, m.auto_tune)),
                format!("presence {}", sampling_knob(m.presence_penalty, MODEL_DEFAULTS.presence_penalty, sampling.presence_penalty, m.auto_tune)),
            ]
            .join(" · "),
        ),
        ("max output tokens", m.max_tokens.to_string()),
        (
            "context budget",
            format!(
                "{}{}",
                m.context_tokens,
                if m.context_tokens < profile.context_suggestion {
                    style::gray(&format!("  ({} can use {})", profile.family, profile.context_suggestion))
                } else {
                    String::new()
                }
            ),
        ),
        ("gpu", gpu_policy(chat)),
        ("agent mode", chat.s.mode.as_str().to_string()),
        ("shell tools", if cfg.tools.allow_shell { "enabled" } else { "disabled" }.into()),
        ("confirm destructive", if cfg.tools.confirm_destructive { "on (asks before risky cmds)" } else { "off" }.into()),
        ("internet", if cfg.web.enabled { format!("enabled ({})", cfg.web.provider.as_str()) } else { "disabled".into() }),
        ("long-term memory", if chat.memory_enabled { format!("on · {} notes", chat.kb.list().len()) } else { "disabled".into() }),
        ("memory retrieval", if chat.memory_enabled { embed } else { "—".into() }),
        ("vault", if chat.memory_enabled { chat.kb.dir().display().to_string() } else { "—".into() }),
        ("project dir", chat.s.cwd.display().to_string()),
    ];
    let pad = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0) + 2;
    let mut lines = vec![hrule(), style::bold("  Settings")];
    for (k, v) in rows {
        lines.push(format!("  {}{v}", style::gray(&style::pad_end(k, pad))));
    }
    lines.push(String::new());
    lines.push(format!(
        "{}{}{}",
        style::gray("  Change with /model, or edit "),
        style::dim(".qwenodyssey/config.toml"),
        style::gray(" (qwenodyssey config set …)")
    ));
    lines.push(format!("{}\n", hrule()));
    lines.join("\n")
}

/// The knowledge vault contents and status.
pub fn render_memory(chat: &Chat) -> String {
    if !chat.memory_enabled {
        return format!("{}\n  {}\n{}\n", hrule(), style::gray("Long-term memory is disabled (knowledge.enabled=false)."), hrule());
    }
    let notes = chat.kb.list();
    let retrieval = if chat.kb.embeddings_active() == Some(false) {
        "keyword".to_string()
    } else {
        format!("semantic ({})", chat.s.config.knowledge.embed_model)
    };
    let mut lines = vec![
        hrule(),
        style::bold("  Long-term memory"),
        format!("  {}{}", style::gray("vault    "), chat.kb.dir().display()),
        format!("  {}{}", style::gray("notes    "), notes.len()),
        format!("  {}{retrieval}", style::gray("retrieval ")),
        String::new(),
    ];
    if notes.is_empty() {
        lines.push(format!("  {}", style::gray("(empty — the model saves notes as it learns)")));
    }
    for n in notes.iter().take(30) {
        let tags = if n.meta.tags.is_empty() { String::new() } else { style::dim(&format!("  #{}", n.meta.tags.join(" #"))) };
        lines.push(format!("  {} {}{tags}", style::cyan(&format!("[[{}]]", n.meta.slug)), n.meta.title));
    }
    lines.push(format!("{}\n", hrule()));
    lines.join("\n")
}

/// Lessons the agent learned from past mistakes.
pub fn render_lessons(chat: &Chat) -> String {
    let Some(evo) = &chat.evolution else {
        return format!("{}\n  {}\n{}\n", hrule(), style::gray("Evolution is off (needs knowledge + evolution enabled)."), hrule());
    };
    let lessons = evo.lessons();
    let mut lines = vec![hrule(), style::bold(&format!("  Lessons learned  ({})", lessons.len())), String::new()];
    if lessons.is_empty() {
        lines.push(format!("  {}", style::gray("(none yet — the agent banks a lesson when a turn goes wrong)")));
    }
    for n in lessons.iter().take(40) {
        let title = n.meta.title.strip_prefix("Lesson:").map(str::trim_start).unwrap_or(&n.meta.title);
        let body: String = n.body.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(160).collect();
        lines.push(format!("  {}{title}\n    {}", style::magenta("✦ "), style::gray(&body)));
    }
    lines.push(String::new());
    lines.push(style::gray("  These are auto-recalled in future turns so mistakes aren't repeated."));
    lines.push(format!("{}\n", hrule()));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_menu_lists_aliases_and_args() {
        style::set_enabled(false);
        let m = render_command_menu();
        assert!(m.contains("/help  /commands  /?"));
        assert!(m.contains("/model [name|#]"));
        assert!(m.contains("@path"));
    }

    #[test]
    fn knob_resolution() {
        assert_eq!(sampling_knob(40u32, 0, Some(20), true), "40");
        assert_eq!(sampling_knob(0u32, 0, Some(20), true), "20 (auto)");
        assert_eq!(sampling_knob(0u32, 0, Some(20), false), "off");
        assert_eq!(sampling_knob(0u32, 0, None, true), "off");
    }
}
