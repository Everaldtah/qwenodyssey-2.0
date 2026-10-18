//! Provider errors and the classifiers the chat layer branches on.
//!
//! The TypeScript version matched error *strings* everywhere ("__interrupted__",
//! "timed out", "HTTP 404" …). Here the provider returns a typed
//! [`ProviderError`] inside `anyhow::Error`, and the classifiers below look at
//! the variant first and only fall back to message heuristics for opaque
//! server text.

use std::sync::LazyLock;

use regex::Regex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    /// The caller cancelled (Esc). Never fall back or retry on this.
    #[error("interrupted")]
    Interrupted,
    /// No response headers within the request timeout.
    #[error("{provider} request timed out after {ms}ms")]
    Timeout { provider: String, ms: u64 },
    /// Headers arrived but the body went silent for `ms`.
    #[error("{provider} stream stalled — no data for {ms}ms")]
    Stalled { provider: String, ms: u64 },
    /// Non-2xx HTTP status with the (truncated) response body.
    #[error("{provider} HTTP {status}: {body}")]
    Http { provider: String, status: u16, body: String },
    /// An error envelope inside a 200 response or an in-band stream error frame.
    #[error("{provider} error: {message}")]
    Api { provider: String, message: String },
    /// Connection refused / DNS / TLS / reset.
    #[error("{provider} network error: {message}")]
    Network { provider: String, message: String },
    /// The backend can't do this (e.g. listing models).
    #[error("{0}")]
    Unsupported(String),
}

impl ProviderError {
    pub fn http(provider: &str, status: u16, body: impl Into<String>) -> Self {
        let mut body: String = body.into();
        if body.len() > 600 {
            let cut = (0..=600).rev().find(|i| body.is_char_boundary(*i)).unwrap_or(0);
            body.truncate(cut);
        }
        ProviderError::Http { provider: provider.to_string(), status, body }
    }
}

fn find(err: &anyhow::Error) -> Option<&ProviderError> {
    err.chain().find_map(|e| e.downcast_ref::<ProviderError>())
}

/// Esc interrupt — stop the turn quietly, no fallback, no error spew.
pub fn is_interrupt(err: &anyhow::Error) -> bool {
    matches!(find(err), Some(ProviderError::Interrupted))
}

static UNAVAILABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)not found|no such model|unknown model|failed to load|unable to load|cannot load|404|model .* does not exist|try pulling|connection refused|fetch failed|ECONNREFUSED|ECONNRESET|HTTP 50\d|\b50[234]\b|overloaded|service.?unavailable|out of memory|insufficient (memory|vram)|unsupported|no space|401|402|403|410|429|unauthorized|forbidden|invalid api key|api key|rate.?limit|quota|credits|timed out|timeout|stalled|aborted",
    )
    .unwrap()
});

/// Does this error mean the active model/backend is unavailable (so the chat
/// layer should walk the fallback chain)?
pub fn looks_unavailable(err: &anyhow::Error) -> bool {
    match find(err) {
        Some(ProviderError::Interrupted) => false,
        Some(ProviderError::Timeout { .. })
        | Some(ProviderError::Stalled { .. })
        | Some(ProviderError::Network { .. }) => true,
        Some(ProviderError::Http { status, body, .. }) => {
            matches!(status, 401 | 402 | 403 | 404 | 408 | 410 | 429 | 500..=599) || UNAVAILABLE.is_match(body)
        }
        _ => UNAVAILABLE.is_match(&format!("{err:#}")),
    }
}

static OVERFLOW: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)n_keep.*n_ctx|greater than the context length|larger context length|exceeds? the (model'?s? )?context").unwrap()
});

/// LM Studio / llama.cpp rejected a prompt longer than the loaded context.
pub fn looks_context_overflow(err: &anyhow::Error) -> bool {
    OVERFLOW.is_match(&format!("{err:#}"))
}

static NO_TOOLS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)does not support tools|tools.*not supported|tool (use|calling).*not|unsupported.*tool").unwrap()
});

/// The model rejects tool-calling (retry the request without tools).
pub fn no_tool_support(err: &anyhow::Error) -> bool {
    NO_TOOLS.is_match(&format!("{err:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_typed_errors() {
        let e: anyhow::Error = ProviderError::Interrupted.into();
        assert!(is_interrupt(&e));
        assert!(!looks_unavailable(&e));
        let e: anyhow::Error = ProviderError::Timeout { provider: "nvidia".into(), ms: 90_000 }.into();
        assert!(looks_unavailable(&e));
        let e: anyhow::Error = ProviderError::http("nvidia", 400, "thinking_token_budget is not yet supported").into();
        assert!(!looks_unavailable(&e));
        let e: anyhow::Error = ProviderError::http("openrouter", 402, "can only afford 3436").into();
        assert!(looks_unavailable(&e));
        let e = anyhow::anyhow!("n_keep: 6441 >= n_ctx: 4096");
        assert!(looks_context_overflow(&e));
        let e = anyhow::anyhow!("registry.ollama.ai/library/deepseek-r1 does not support tools");
        assert!(no_tool_support(&e));
    }
}
