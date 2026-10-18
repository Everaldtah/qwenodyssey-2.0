//! Qwenodyssey 2.0 pipeline: scan → plan → context → code → patch → review →
//! test → fix → summary (port of the TS `core/orchestrator`, `core/*` edit
//! engine pieces, `agents/*` and the pipeline CLI commands' logic).
//!
//! - [`orchestrator`]: [`Orchestrator`] — `run_code` / `run_autofix`.
//! - [`commands`]: the `code`/`plan`/`edit`/`apply`/`review`/`test`/`memory`/`init`
//!   command logic as library functions returning outcomes for the binary to render.
//! - [`edit_pipeline`]: best-of-N SEARCH/REPLACE edits + verify + self-repair
//!   (the block engine itself is [`qo_tools::edit_block`]).
//! - [`candidate_select`], [`verify_loop`]: best-of-N selection and check runs.
//! - [`patch_engine`] / [`diff`]: unified-diff edits with a rollback journal (jsdiff dialect).
//! - [`repo_scanner`]: [`scan_repo`] / [`summarize_repo`] (also used by the chat layer).
//! - [`context_builder`]: token-budgeted prompt context.
//! - [`agents`]: planner, coders, reviewer, tester, error fixer.
//! - [`memory`]: the project [`MemoryStore`].
//! - [`ui`] / [`render`]: the [`PipelineUi`] trait the binary implements, and
//!   pure ANSI renderers (`colorize_diff`, plan/review blocks).
//!
//! Library code never prints: progress, rendered blocks and yes/no questions
//! go through [`PipelineUi`]; structured events go to the JSONL session logger.

pub mod agents;
pub mod candidate_select;
pub mod commands;
pub mod context_builder;
mod deps;
pub mod diff;
pub mod edit_pipeline;
pub mod memory;
pub mod orchestrator;
pub mod patch_engine;
pub mod process;
pub mod render;
pub mod repo_scanner;
pub mod state_machine;
pub mod ui;
pub mod verify_loop;

#[cfg(test)]
pub(crate) mod test_support;

pub use memory::{MemoryCategory, MemoryEntry, MemoryStore};
pub use orchestrator::{Orchestrator, RunOptions, RunResult};
pub use render::colorize_diff;
pub use repo_scanner::{scan_repo, summarize_repo};
pub use ui::{Level, PipelineUi, RecordingUi, SilentUi};
