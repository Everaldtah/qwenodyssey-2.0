//! qo-tools: the local tools Qwenodyssey's agents call — files, shell (with the
//! hard-block / destructive-gate safety layer), git, search, code navigation,
//! SEARCH/REPLACE edit blocks, a persistent shell session on a real pty, plan
//! and think tools — plus the tool specs advertised to the model and the
//! tool-call hardening (`prepare_tool_call`) small models need.
//!
//! Library code never writes to the terminal: results, thoughts and progress
//! come back in [`qo_core::ToolResult`]s; confirmations go through
//! `ToolContext::confirm`; Esc interrupts through `ToolContext::cancel`.
//!
//! Entry points the chat layer uses:
//! - [`register_defaults`] / [`default_registry`] — the default tool set.
//! - [`create_plan_tool`] + [`PlanState`], [`create_think_tool`],
//!   [`create_shell_session_tools`] + [`ShellSession`] — session-scoped tools.
//! - [`chat_tool_specs`], [`code_nav_tool_specs`], [`shell_session_tool_specs`],
//!   [`think_tool_spec`], [`compact_tool_specs`] — what is advertised.
//! - [`prepare_tool_call`] / [`resolve_tool_name`] — pre-dispatch hardening.
//! - [`classify_command`] — the shell safety classifier.
//! - [`BlindOverwriteGuard`] — refuse blind write_file over unread files.
//! - [`prewarm_symbol_index`] — warm the code-nav index at startup.

pub mod code_tools;
pub mod edit_block;
pub mod file_tools;
pub mod git_tools;
pub mod plan_tool;
pub mod ps_compat;
pub mod registry;
pub mod search_tools;
pub mod shell_encyclopedia;
pub mod shell_session;
pub mod shell_session_tools;
pub mod shell_tools;
pub mod specs;
pub mod symbol_index;
pub mod think_tool;
pub mod tool_call_prep;
mod util;

pub use code_tools::{code_nav_tools, invalidate_index, prewarm_symbol_index};
pub use edit_block::{
    apply_block_to_content, apply_edit_blocks, explain_failure, locate_block, parse_edit_blocks, similarity,
    BlockApplyResult, BlockLocation, ContentApplyResult, EditBlock, MatchStrategy,
};
pub use file_tools::{
    expand_home, resolve_inside, resolve_readable, resolve_readable_smart, resolve_writable, resolve_writable_in,
    strip_redundant_cwd_prefix, strip_redundant_cwd_prefix_in_command, BlindOverwriteGuard, IGNORE,
};
pub use git_tools::{git_status, is_dirty, is_repo, GitFileStatus, GitStatus};
pub use plan_tool::{create_plan_tool, render_plan, PlanItem, PlanState, PlanStatus};
pub use ps_compat::{adapt_chains_for_powershell, uses_windows_powershell};
pub use registry::{default_registry, default_tools, register_defaults};
pub use shell_encyclopedia::{lookup_shell_help, SHELL_ENCYCLOPEDIA};
pub use shell_session::{strip_ansi, RunResult, RunStatus, ShellSession};
pub use shell_session_tools::{create_shell_session_tools, shell_session_tool_specs};
pub use shell_tools::{classify_command, shell_command, ClassifyOptions, CommandClass, PS_PREAMBLE};
pub use specs::{chat_tool_names, chat_tool_specs, code_nav_tool_specs, compact_tool_specs, compact_tool_specs_default, plan_tool_spec};
pub use symbol_index::{Reference, SymbolDef, SymbolIndex, SymbolKind};
pub use think_tool::{create_think_tool, think_tool_spec, THINK_ACK};
pub use tool_call_prep::{levenshtein, prepare_tool_call, resolve_tool_name, PreparedCall, ToolNameResolution, ARG_ALIASES, TOOL_ALIASES};
