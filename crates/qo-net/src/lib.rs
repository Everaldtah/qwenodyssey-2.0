//! qo-net: the network + knowledge layer of Qwenodyssey 2.0.
//!
//! - [`web`]: `web_search` / `web_fetch` / `web_research` / `deep_research`
//!   (DuckDuckGo HTML + Lite fallback ladder, Tavily, Brave, SearXNG; SSRF guard,
//!   size caps, deadlines, in-session page cache).
//! - [`github`]: the ten `github_*` tools over the REST API, token from config →
//!   env → `gh auth token`.
//! - [`mcp`]: stdio JSON-RPC MCP client + `mcp__<server>__<tool>` tools.
//! - [`knowledge`]: the Obsidian-style vault (RAG over Ollama embeddings with a
//!   keyword fallback) and its tools; same on-disk format as 1.x.
//! - [`session`]: persisted chat sessions (`~/.qwenodyssey/sessions`, 1.x format).
//! - [`evolution`]: lesson-learning from troubled turns.
//! - [`compactor`]: history token accounting + compaction.
//! - [`specs`]: the tool specs advertised to the model for these tools.
//! - [`wiring`]: option structs built from `qo_core::config::Config`.
//!
//! Every tool honours `ToolContext::cancel` (Esc): the in-flight request is
//! dropped and the tool reports "interrupted". Nothing here writes to the
//! terminal.

pub mod compactor;
pub mod evolution;
pub mod github;
pub mod knowledge;
pub mod mcp;
pub mod session;
pub mod specs;
pub mod web;
pub mod wiring;

#[cfg(test)]
mod testutil;
mod util;

pub use compactor::{compact_history, history_tokens, should_compact, CompactOptions, CompactResult};
pub use evolution::{EvolutionEngine, EvolutionOptions, GenerateFn, TurnSignals};
pub use github::{create_github_tools, GithubClient, GithubConfig};
pub use knowledge::{create_knowledge_tools, recall_knowledge, KnowledgeBase, KnowledgeOptions, Note, NoteHit, NoteInput, NoteMeta};
pub use mcp::{create_mcp_tools, mcp_tool_name, McpClient, McpOptions, McpServerSpec, McpSetup, McpToolDef};
pub use session::{derive_title, ChatSession, ChatSessionMeta, SessionStore};
pub use specs::{github_tool_specs, knowledge_tool_specs, web_tool_specs};
pub use web::{create_web_tools, SearchProvider, SearchResult, WebClient, WebConfig, WebEndpoints};
pub use wiring::mcp_servers_from_config;
