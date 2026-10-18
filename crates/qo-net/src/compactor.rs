//! Context-window management: estimate how much of the model's context the
//! chat history is using and, when it gets close to the budget, COMPACT the
//! oldest turns into a dense summary so long sessions don't overflow (or
//! silently drop the system prompt). Critical for small local models with
//! modest context.

use qo_core::{CancellationToken, Message, ModelOptions, Provider, Role};

use crate::util::truncate_chars;

/// Rough token count of the whole history using the provider's tokenizer.
pub fn history_tokens(messages: &[Message], provider: &dyn Provider) -> usize {
    let mut n = 0;
    for m in messages {
        n += provider.count_tokens(&m.content);
        if let Some(calls) = &m.tool_calls {
            for c in calls {
                let args = if c.arguments.is_null() { "{}".to_string() } else { c.arguments.to_string() };
                n += provider.count_tokens(&args) + 6;
            }
        }
        n += 4; // per-message framing overhead
    }
    n
}

/// Should we auto-compact before the next turn? True once the history is
/// within headroom of the usable budget (context minus the answer allowance):
/// at 85% of `max(2000, context_tokens - max_tokens - 512)`.
pub fn should_compact(messages: &[Message], provider: &dyn Provider, context_tokens: u64, max_tokens: u64) -> bool {
    let usable = (context_tokens as i64 - max_tokens as i64 - 512).max(2000);
    history_tokens(messages, provider) as f64 >= (usable as f64 * 0.85).floor()
}

const MARK: &str = "\n\n[CONVERSATION SO FAR — earlier turns were compacted to save context]\n";
const MARK_PREFIX: &str = "\n\n[CONVERSATION SO FAR";

const COMPACT_SYSTEM: &str = "You compress a coding-assistant conversation into a dense memo so it can continue with less \
context. PRESERVE: the user's goals and decisions, important facts, exact file paths/commands, \
what was already done, tool results that still matter, and any unfinished tasks. Drop chit-chat \
and resolved detail. Output plain text, under 25 lines, no preamble, no code fences.";

/// Index (into messages) of the start of the Kth-from-last user turn.
fn recent_cut_index(messages: &[Message], keep_user_turns: usize) -> Option<usize> {
    let user_idx: Vec<usize> = (1..messages.len()).filter(|&i| messages[i].role == Role::User).collect();
    if user_idx.len() <= keep_user_turns {
        return None;
    }
    Some(user_idx[user_idx.len() - keep_user_turns])
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactResult {
    pub compacted: bool,
    /// How many messages were folded into the summary.
    pub removed: usize,
    pub summary: Option<String>,
    /// New message array to replace the old one with (when compacted).
    pub new_messages: Option<Vec<Message>>,
}

#[derive(Clone, Debug, Default)]
pub struct CompactOptions {
    /// User turns kept verbatim at the end (the chat uses 3 for auto, 2 for `/compact`).
    pub keep_user_turns: usize,
    /// Esc interrupt for the summarisation request.
    pub cancel: Option<CancellationToken>,
}

/// Summarize everything between the system prompt and the last
/// `keep_user_turns` user turns into a memo folded into the system message.
/// Cutting on a user-turn boundary guarantees we never split an
/// assistant(tool_calls)→tool-result group, which would otherwise produce an
/// invalid request. Any failure (or an empty summary) leaves history untouched.
pub async fn compact_history(messages: &[Message], provider: &dyn Provider, opts: &CompactOptions) -> CompactResult {
    let nothing = CompactResult::default();
    if messages.first().map(|m| m.role) != Some(Role::System) {
        return nothing;
    }
    let Some(cut) = recent_cut_index(messages, opts.keep_user_turns) else { return nothing };
    let system = &messages[0];
    let middle = &messages[1..cut];
    let recent = &messages[cut..];
    if middle.is_empty() {
        return nothing;
    }

    let transcript = middle
        .iter()
        .map(|m| {
            if m.role == Role::Tool {
                return format!("TOOL({}) → {}", m.name.as_deref().unwrap_or("tool"), truncate_chars(&m.content, 400, "…"));
            }
            if let Some(calls) = m.tool_calls.as_ref().filter(|c| !c.is_empty()) {
                let names = calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ");
                let extra = if m.content.is_empty() { String::new() } else { format!(" — {}", truncate_chars(&m.content, 200, "…")) };
                return format!("ASSISTANT called {names}{extra}");
            }
            let role = match m.role {
                Role::System => "SYSTEM",
                Role::User => "USER",
                Role::Assistant => "ASSISTANT",
                Role::Tool => "TOOL",
            };
            format!("{role}: {}", truncate_chars(&m.content, 900, "…"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = [
        Message::system(COMPACT_SYSTEM),
        Message::user(format!("Compress this earlier conversation:\n\n{transcript}")),
    ];
    let model_opts = ModelOptions { temperature: Some(0.2), max_tokens: Some(700), cancel: opts.cancel.clone(), ..Default::default() };
    let summary = match provider.generate(&prompt, &model_opts).await {
        Ok(res) => res.text.trim().to_string(),
        Err(_) => return nothing,
    };
    if summary.is_empty() {
        return nothing;
    }

    // Fold the memo into the system message (idempotent across repeated compactions).
    let base_sys = system.content.split(MARK_PREFIX).next().unwrap_or("");
    let mut new_messages = Vec::with_capacity(recent.len() + 1);
    new_messages.push(Message::system(format!("{base_sys}{MARK}{summary}")));
    new_messages.extend_from_slice(recent);
    CompactResult { compacted: true, removed: middle.len(), summary: Some(summary), new_messages: Some(new_messages) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use qo_core::{GenerateResult, HealthStatus};

    /// Token estimate = len/4, canned summary (the TS test's fakeProvider).
    struct Stub {
        summary: &'static str,
    }

    #[async_trait]
    impl Provider for Stub {
        fn name(&self) -> &str {
            "stub"
        }
        fn model(&self) -> String {
            "stub".into()
        }
        fn set_model(&self, _model: &str) {}
        async fn generate(&self, _messages: &[Message], _opts: &ModelOptions) -> anyhow::Result<GenerateResult> {
            Ok(GenerateResult { text: self.summary.into(), model: "stub".into(), ..Default::default() })
        }
        async fn stream(
            &self,
            _messages: &[Message],
            _on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
            _opts: &ModelOptions,
        ) -> anyhow::Result<GenerateResult> {
            Ok(GenerateResult { model: "stub".into(), ..Default::default() })
        }
        fn count_tokens(&self, text: &str) -> usize {
            text.len().div_ceil(4)
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus { ok: true, detail: None }
        }
    }

    const STUB: Stub = Stub { summary: "COMPACT SUMMARY" };

    fn big(label: &str) -> Vec<Message> {
        let text = format!("{label} ").repeat(200);
        vec![Message::user(text.clone()), Message::assistant(text)]
    }

    #[test]
    fn history_tokens_grow_and_should_compact_respects_the_budget() {
        let small = vec![Message::system("SYS"), Message::user("hi")];
        assert!(history_tokens(&small, &STUB) > 0);
        assert!(!should_compact(&small, &STUB, 16384, 4096));
        // ~16 messages × ~400 tokens ≈ 6400 tokens, comfortably over the ~5.5k
        // threshold for an 8k context with 1k answer headroom (usable≈6488, compact at 85%).
        let mut huge = vec![Message::system("SYS")];
        for _ in 0..8 {
            huge.extend(big(&"x".repeat(1600)));
        }
        assert!(history_tokens(&huge, &STUB) > 5514);
        assert!(should_compact(&huge, &STUB, 8000, 1000));
    }

    #[tokio::test]
    async fn compacts_older_turns_into_the_system_message() {
        let messages = vec![
            Message::system("SYS"),
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
            Message::user("u3"),
            Message::assistant("a3"),
            Message::user("u4"),
        ];
        let r = compact_history(&messages, &STUB, &CompactOptions { keep_user_turns: 2, cancel: None }).await;
        assert!(r.compacted);
        assert_eq!(r.removed, 4); // u1,a1,u2,a2
        let nm = r.new_messages.unwrap();
        assert_eq!(nm[0].role, Role::System);
        assert!(nm[0].content.contains("COMPACT SUMMARY"));
        // First non-system message is a user turn boundary (u3), so no dangling tool group.
        assert_eq!((nm[1].role, nm[1].content.as_str()), (Role::User, "u3"));
        assert_eq!(nm.last().map(|m| m.content.as_str()), Some("u4"));

        // Compacting again replaces the old memo instead of stacking a second one.
        let mut again = nm.clone();
        again.extend([Message::assistant("a4"), Message::user("u5"), Message::assistant("a5"), Message::user("u6")]);
        let r2 = compact_history(&again, &STUB, &CompactOptions { keep_user_turns: 2, cancel: None }).await;
        let sys = &r2.new_messages.unwrap()[0].content;
        assert_eq!(sys.matches("[CONVERSATION SO FAR").count(), 1);
        assert!(sys.starts_with("SYS\n\n[CONVERSATION SO FAR"));
    }

    #[tokio::test]
    async fn does_nothing_with_too_few_turns_or_an_empty_summary() {
        let msgs = vec![Message::system("SYS"), Message::user("only")];
        assert!(!compact_history(&msgs, &STUB, &CompactOptions { keep_user_turns: 3, cancel: None }).await.compacted);
        let msgs = vec![Message::system("S"), Message::user("a"), Message::assistant("b"), Message::user("c")];
        let empty = Stub { summary: "   " };
        assert!(!compact_history(&msgs, &empty, &CompactOptions { keep_user_turns: 1, cancel: None }).await.compacted);
        assert!(!compact_history(&msgs[1..], &STUB, &CompactOptions { keep_user_turns: 1, cancel: None }).await.compacted);
    }
}
