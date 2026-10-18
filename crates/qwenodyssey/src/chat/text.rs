//! Pure text helpers of the chat loop (ported from src/cli/commands/chat.ts):
//! smalltalk detection, reasoning/answer splitting, template-artifact
//! stripping, fenced-command extraction, echo detection, tool-output
//! truncation and the SELF-AWARENESS block swap.

use std::sync::LazyLock;

use qo_core::{Message, Role};
use regex::Regex;

/* ───────────────────────────── smalltalk ───────────────────────────── */

const GREET_FIRST: &[&str] = &[
    "hi", "hii", "hiya", "hey", "heya", "hello", "helo", "yo", "sup", "wsup", "wassup", "howdy", "gm", "gn", "morning",
    "evening", "greetings", "yt",
];

const SMALLTALK_EXACT: &[&str] = &[
    "hi", "hii", "hiya", "hey", "heya", "hello", "helo", "hello there", "hi there", "hey there", "yo", "sup", "wsup",
    "wassup", "whats up", "what's up", "howdy", "how are you", "how are u", "how r u", "hru", "gm", "good morning",
    "good afternoon", "good evening", "good night", "gn", "thanks", "thank you", "thx", "ty", "cheers", "ok", "okay",
    "k", "kk", "cool", "nice", "great", "lol", "lmao", "haha", "hehe", "morning", "evening", "greetings", "test",
    "ping",
];

static WORK_MARKERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\\/@`?]|https?:|\.\w{1,4}\b").expect("static regex"));
static TRAILING_PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[!.…,~\s]+$").expect("static regex"));

/// True for trivial conversational messages (greetings, thanks, acks) that need
/// NO tools. These turns run with tools DISABLED so an eager model physically
/// can't fire read_file/web_search/shell on a bare "hi" — a deterministic
/// backstop to the CONVERSATION_GUARD prompt, which some models still ignore.
/// Conservative on purpose: anything with a path, code, URL, "?", or real
/// length is NOT caught, so genuine requests always keep their tools.
pub fn is_smalltalk(raw: &str) -> bool {
    let t = raw.trim().to_lowercase();
    if t.is_empty() || t.encode_utf16().count() > 40 {
        return false;
    }
    // A path, file ref, code, URL, or question mark means real work.
    if WORK_MARKERS.is_match(&t) {
        return false;
    }
    let w = TRAILING_PUNCT.replace(&t, "").trim().to_string();
    if SMALLTALK_EXACT.contains(&w.as_str()) {
        return true;
    }
    // Greeting word + a short address ("sup buddy", "hey man").
    let words: Vec<&str> = w.split_whitespace().collect();
    words.len() <= 3 && words.first().is_some_and(|f| GREET_FIRST.contains(f))
}

/* ───────────────────────────── reasoning ───────────────────────────── */

static THINK_PAIRED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<think(?:ing)?>(.*?)</think(?:ing)?>").expect("static regex"));
static THINK_CLOSE_ONLY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)^(.*?)</think(?:ing)?>\s*(.*)$").expect("static regex"));
static THINK_OPEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<think(?:ing)?>(.*)$").expect("static regex"));

/// A reply split into chain-of-thought and final answer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub thinking: String,
    pub answer: String,
}

/// Split a reply into its chain-of-thought and final answer. Reasoning models
/// (DeepSeek-R1, QwQ, qwen3.5) wrap deliberation in `<think>…</think>`. Handles:
/// - paired `<think>…</think> answer`
/// - close-only `…reasoning…</think> answer` (LM Studio templates pre-fill the
///   opening tag, so the model only emits the closing one)
/// - open-only `<think>…` (truncated/streamed — the rest is all reasoning)
pub fn split_thinking(text: &str) -> Split {
    if let Some(c) = THINK_PAIRED.captures(text) {
        let whole = c.get(0).map_or("", |m| m.as_str());
        return Split { thinking: c[1].trim().to_string(), answer: text.replacen(whole, "", 1).trim().to_string() };
    }
    if let Some(c) = THINK_CLOSE_ONLY.captures(text) {
        return Split { thinking: c[1].trim().to_string(), answer: c[2].trim().to_string() };
    }
    if let Some(c) = THINK_OPEN.captures(text) {
        return Split { thinking: c[1].trim().to_string(), answer: String::new() };
    }
    Split { thinking: String::new(), answer: text.trim().to_string() }
}

static TOOL_RESPONSE_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<tool_response>.*?</tool_response>").expect("static regex"));
static TOOL_CALL_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<tool_call>.*?</tool_call>").expect("static regex"));
static TOOL_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</?tool_(?:response|call)\s*>").expect("static regex"));
static SPECIAL_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\|[^|>]*\|>").expect("static regex"));

/// Strip chat-template tool-call artifacts a model leaks into its visible
/// answer. Small/coder models sometimes parrot the tool result back wrapped in
/// `<tool_response>…</tool_response>` — or emit `<tool_call>` / special `<|…|>`
/// tokens — instead of writing a real reply. None of that is ever a legitimate
/// answer; an empty result then routes to the "no real answer" nudge.
pub fn sanitize_answer(text: &str) -> String {
    let t = TOOL_RESPONSE_BLOCK.replace_all(text, "");
    let t = TOOL_CALL_BLOCK.replace_all(&t, "");
    let t = TOOL_TAG.replace_all(&t, "");
    let t = SPECIAL_TOKEN.replace_all(&t, "");
    t.trim().to_string()
}

/// The content of the most recent tool result in history, if any.
pub fn last_tool_result(history: &[Message]) -> &str {
    history.iter().rev().find(|m| m.role == Role::Tool).map_or("", |m| m.content.as_str())
}

/// True when the model's "answer" is really just the tool output handed back
/// (verbatim or nearly so) rather than a synthesized reply. Compared on
/// collapsed whitespace so formatting differences don't hide an echo.
pub fn echoes_tool_result(answer: &str, tool_result: &str) -> bool {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let a = norm(answer);
    let t = norm(tool_result);
    if a.chars().count() < 40 || t.chars().count() < 40 {
        return false;
    }
    let head: String = t.chars().take(200).collect();
    t.contains(&a) || a.contains(&head)
}

/* ───────────────────────────── fenced commands ───────────────────────────── */

/// Languages whose fenced blocks we treat as runnable shell commands.
const SHELL_LANGS: &[&str] = &[
    "sh", "bash", "shell", "zsh", "console", "terminal", "shell-session", "powershell", "pwsh", "ps", "ps1", "posh", "cmd",
    "bat", "dos", "batch",
];

static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"```([\w-]*)[ \t]*\r?\n((?s:.*?))```").expect("static regex"));
static PROMPT_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(PS[^>]*>|\$|>|#)\s+").expect("static regex"));

/// Pull runnable commands out of shell-tagged ``` fences in a model reply,
/// stripping shell-prompt prefixes (`$ `, `PS C:\>`, `> `). Untagged or
/// non-shell fences are ignored so we never auto-run code the user wanted
/// written rather than executed.
pub fn extract_shell_commands(text: &str) -> Vec<String> {
    let mut cmds = Vec::new();
    for c in FENCE.captures_iter(text) {
        let lang = c.get(1).map_or("", |m| m.as_str()).to_lowercase();
        if !SHELL_LANGS.contains(&lang.as_str()) {
            continue;
        }
        let body = c[2]
            .split('\n')
            .map(|l| PROMPT_PREFIX.replace(l, "").trim_end().to_string())
            .filter(|l| !l.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        if !body.is_empty() {
            cmds.push(body);
        }
    }
    cmds
}

/* ───────────────────────────── tool output ───────────────────────────── */

/// Char budget for ONE tool result fed back into history, scaled to the
/// model's context window (~4 chars/token; tool output may take ~12% of the
/// window per call). The old fixed 8000-char cap flooded a 4–8K-context model
/// in two or three shell calls, evicting the system prompt's tool rules.
/// Clamped to [2000, 8000].
pub fn tool_result_budget(context_tokens: u32) -> usize {
    let ctx = if context_tokens == 0 { 16384 } else { context_tokens } as f64;
    ((ctx * 4.0 * 0.12).floor() as usize).clamp(2000, 8000)
}

/// Middle-out truncation: keep the head AND the tail. Tool output carries its
/// verdict at the end (exit status, test summary, the actual error).
/// `budget` counts characters.
pub fn truncate_middle(out: &str, budget: usize) -> String {
    let len = out.chars().count();
    if len <= budget {
        return out.to_string();
    }
    let head = (budget as f64 * 0.6).floor() as usize;
    let tail = budget - head;
    let omitted = len - head - tail;
    let h: String = out.chars().take(head).collect();
    let t: String = out.chars().skip(len - tail).collect();
    format!("{h}\n…[{omitted} chars omitted — middle of output]…\n{t}")
}

/// Cut `s` to at most `max` characters (char-boundary safe).
pub fn clip(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Collapse runs of whitespace to one space.
pub fn squash_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/* ───────────────────────────── self-awareness ───────────────────────────── */

/// First words of the SELF-AWARENESS block; [`replace_self_awareness`] anchors on it.
pub const SELF_AWARENESS_HEADER: &str = "SELF-AWARENESS — you are \"Qwenodyssey\"";

/// Replace the SELF-AWARENESS block inside the system prompt (history[0]) in
/// place, preserving anything appended after it (the PROJECT summary, a
/// /compact memo). Called whenever the active model changes so the model's
/// stated identity tracks the real provider/model.
pub fn replace_self_awareness(history: &mut [Message], new_text: &str) {
    let Some(sys) = history.first_mut() else { return };
    if sys.role != Role::System {
        return;
    }
    // Match the block HEADER, not the bare word: system.md mentions "the
    // SELF-AWARENESS section below" near its top, and matching that used to
    // wipe everything between it and the real block on every model switch.
    let start = sys.content.find(SELF_AWARENESS_HEADER).or_else(|| sys.content.rfind("SELF-AWARENESS"));
    let Some(start) = start else {
        sys.content.push('\n');
        sys.content.push_str(new_text);
        return;
    };
    // The block ends where the next appended section begins (or at the end).
    let after = &sys.content[start..];
    let end = ["\n\nPROJECT:", "\n\n[CONVERSATION SO FAR"].iter().filter_map(|m| after.find(m)).min();
    let tail = end.map_or(String::new(), |i| after[i..].to_string());
    sys.content = format!("{}{new_text}{tail}", &sys.content[..start]);
}

/* ───────────────────────────── misc ───────────────────────────── */

/// Embedding models can't chat — keep them out of the model picker/fallbacks.
pub fn is_embedding_model(name: &str) -> bool {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)(^|[-_/])(embed|embedding|bge|gte|e5|nomic-embed)").expect("static regex"));
    RE.is_match(name)
}

/// Port out of an http(s)://host:port base URL (default 1234).
pub fn lms_port(base_url: &str) -> u16 {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r":(\d+)").expect("static regex"));
    RE.captures(base_url).and_then(|c| c[1].parse().ok()).unwrap_or(1234)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smalltalk_is_conservative() {
        for s in ["hi", "Hello!", "sup buddy", "thanks", "how are you", "hey there man", "ok."] {
            assert!(is_smalltalk(s), "{s}");
        }
        for s in ["hi?", "read src/main.rs", "hello how do I fix this bug in my code", "what is @foo", "check https://x.y", "fix a.py", ""] {
            assert!(!is_smalltalk(s), "{s}");
        }
    }

    #[test]
    fn splits_thinking_shapes() {
        assert_eq!(split_thinking("<think>plan</think> answer"), Split { thinking: "plan".into(), answer: "answer".into() });
        assert_eq!(split_thinking("reasoning</think>\nanswer"), Split { thinking: "reasoning".into(), answer: "answer".into() });
        assert_eq!(split_thinking("<thinking>still going"), Split { thinking: "still going".into(), answer: String::new() });
        assert_eq!(split_thinking("  plain  "), Split { thinking: String::new(), answer: "plain".into() });
    }

    #[test]
    fn sanitize_answer_matches_ts() {
        let text = "<tool_response>\n[[qwenodyssey]] Qwenodyssey  (tags: project)\n  # Qwenodyssey ...\n</tool_response>";
        assert_eq!(sanitize_answer(text), "");
        assert_eq!(
            sanitize_answer("You are talking to qwen2.5-coder:7b. <tool_call>{\"name\":\"x\"}</tool_call>"),
            "You are talking to qwen2.5-coder:7b."
        );
        assert_eq!(sanitize_answer("hi </tool_response>"), "hi");
        assert_eq!(sanitize_answer("<tool_response>only opener\nresult"), "only opener\nresult");
        assert_eq!(sanitize_answer("the answer<|im_end|>"), "the answer");
        assert_eq!(sanitize_answer("It is 3pm in the UK."), "It is 3pm in the UK.");
    }

    #[test]
    fn extracts_only_shell_fences() {
        let t = "Run:\n```powershell\nPS C:\\> Get-Process\n```\nand\n```python\nprint(1)\n```\n```bash\n$ ls -la\n\n$ pwd\n```";
        assert_eq!(extract_shell_commands(t), vec!["Get-Process".to_string(), "ls -la\npwd".to_string()]);
        assert!(extract_shell_commands("```\nuntagged\n```").is_empty());
    }

    #[test]
    fn echo_detection() {
        let tool = "Directory listing: a.txt b.txt c.txt d.txt e.txt f.txt g.txt h.txt";
        assert!(echoes_tool_result(tool, tool));
        assert!(!echoes_tool_result("short", tool));
        assert!(!echoes_tool_result("There are eight text files here, named a through h, nothing else.", tool));
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        assert_eq!(tool_result_budget(16384), 7864);
        assert_eq!(tool_result_budget(2048), 2000);
        assert_eq!(tool_result_budget(200_000), 8000);
        let s = format!("{}{}", "a".repeat(100), "z".repeat(100));
        let t = truncate_middle(&s, 50);
        assert!(t.starts_with(&"a".repeat(30)));
        assert!(t.ends_with(&"z".repeat(20)));
        assert!(t.contains("[150 chars omitted"));
        assert_eq!(truncate_middle("short", 50), "short");
    }

    fn sys(content: &str) -> Vec<Message> {
        vec![Message::system(content)]
    }

    const OLD: &str = "SELF-AWARENESS — you are \"Qwenodyssey\", powered by the model \"moonshotai/kimi-k2.6\" served through a cloud model via OpenRouter (provider id: openrouter).";
    const NEW: &str = "SELF-AWARENESS — you are \"Qwenodyssey\", powered by the model \"qwen2.5-coder:7b\" served through a local Ollama model (provider id: ollama).";

    #[test]
    fn self_awareness_swap() {
        let mut h = sys(&format!("BASE RULES\n{OLD}"));
        replace_self_awareness(&mut h, NEW);
        assert!(h[0].content.contains("qwen2.5-coder:7b") && !h[0].content.contains("kimi-k2.6"));
        assert!(h[0].content.contains("BASE RULES"));

        let mut h = sys(&format!("BASE RULES\n{OLD}\n\nPROJECT:\nfoo summary here"));
        replace_self_awareness(&mut h, NEW);
        assert!(h[0].content.contains("\n\nPROJECT:\nfoo summary here") && !h[0].content.contains("kimi"));

        let mut h = sys(&format!("BASE\n{OLD}\n\nPROJECT:\nsum\n\n[CONVERSATION SO FAR]\nmemo text"));
        replace_self_awareness(&mut h, NEW);
        assert!(h[0].content.contains("\n\nPROJECT:\nsum"));
        assert!(h[0].content.contains("[CONVERSATION SO FAR]\nmemo text"));

        let mut h = sys("BASE ONLY");
        replace_self_awareness(&mut h, NEW);
        assert!(h[0].content.contains("BASE ONLY") && h[0].content.contains("qwen2.5-coder:7b"));

        let mut h = vec![Message::user("hi")];
        replace_self_awareness(&mut h, NEW);
        assert_eq!(h[0].content, "hi");
    }

    #[test]
    fn self_awareness_does_not_clip_on_an_earlier_mention() {
        let s = "You are a coding agent. Your identity is in the SELF-AWARENESS section below.\n\n\
                 TOOL RULES: always call tools.\n\nPLAN RULES: keep a plan.\n\n\
                 SELF-AWARENESS — you are \"Qwenodyssey\", running on old-model.\n• MODEL: old-model\n\nPROJECT:\nfiles: a.py";
        let mut h = sys(s);
        replace_self_awareness(&mut h, "SELF-AWARENESS — you are \"Qwenodyssey\", running on new-model.\n• MODEL: new-model");
        let out = &h[0].content;
        assert!(out.contains("TOOL RULES: always call tools."));
        assert!(out.contains("PLAN RULES: keep a plan."));
        assert!(out.contains("new-model") && !out.contains("old-model"));
        assert!(out.contains("PROJECT:\nfiles: a.py"));
        assert!(out.contains("SELF-AWARENESS section below"));
    }

    #[test]
    fn misc_helpers() {
        assert!(is_embedding_model("nomic-embed-text"));
        assert!(is_embedding_model("text-embedding-bge-m3"));
        assert!(!is_embedding_model("qwen2.5-coder:7b"));
        assert_eq!(lms_port("http://100.109.193.63:1234"), 1234);
        assert_eq!(lms_port("http://localhost"), 1234);
    }
}
