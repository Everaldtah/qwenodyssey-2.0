//! The single place that names other crates' helper APIs (prompt loading,
//! JSON extraction, path sandboxing, workspace layout).
//!
//! Everything else in the crate calls these wrappers, so a signature change
//! in `qo-core` / `qo-tools` only ever touches this file.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// `.qwenodyssey` under the project root (config.ts `workspacePath`).
pub(crate) fn workspace_path(cwd: &Path) -> PathBuf {
    qo_core::config::workspace_path(cwd)
}

/// A shipped prompt template by name ("planner", "coder_blocks", …).
pub(crate) fn load_prompt(name: &str) -> anyhow::Result<String> {
    qo_core::prompts::load_prompt(name)
}

/// Fill `{{placeholders}}` in a template; unknown keys become "" (promptLoader.ts `render`).
pub(crate) fn render(template: &str, vars: &[(&str, &str)]) -> String {
    qo_core::prompts::render(template, vars)
}

/// First JSON value in a model reply (fenced, bare, or repaired), if any (parse.ts `extractJson`).
pub(crate) fn extract_json(text: &str) -> Option<Value> {
    qo_core::parse::extract_json(text)
}

/// Resolve a path for writes that must stay inside `cwd` (fileTools.ts `resolveInside`).
pub(crate) fn resolve_inside(cwd: &Path, p: &str) -> anyhow::Result<PathBuf> {
    qo_tools::resolve_inside(cwd, p)
}

/// Resolve a write path allowed under `cwd` or the agent's own source root
/// (fileTools.ts `resolveWritable`).
pub(crate) fn resolve_writable(cwd: &Path, self_root: Option<&Path>, p: &str) -> anyhow::Result<PathBuf> {
    qo_tools::resolve_writable_in(cwd, self_root, p)
}
