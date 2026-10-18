//! Default tool set for a [`ToolRegistry`] (ported from tools/registry.ts,
//! whose constructor registered exactly these).

use std::sync::Arc;

use qo_core::tool::SharedTool;
use qo_core::ToolRegistry;

use crate::code_tools::code_nav_tools;
use crate::file_tools::{CreateFileTool, DeleteFileTool, ListFilesTool, ReadFileTool, TreeTool, WriteFileTool};
use crate::git_tools::{GitDiffTool, GitStatusTool};
use crate::search_tools::{GrepTool, SearchDocsTool};
use crate::shell_encyclopedia::ShellHelpTool;
use crate::shell_tools::RunShellTool;

/// The default local tools: files, shell, shell_help, git, search, and the
/// code-navigation + apply_edit tools.
pub fn default_tools() -> Vec<SharedTool> {
    let mut tools: Vec<SharedTool> = vec![
        Arc::new(ReadFileTool),
        Arc::new(WriteFileTool),
        Arc::new(CreateFileTool),
        Arc::new(DeleteFileTool),
        Arc::new(ListFilesTool),
        Arc::new(TreeTool),
        Arc::new(RunShellTool),
        Arc::new(ShellHelpTool),
        Arc::new(GitStatusTool),
        Arc::new(GitDiffTool),
        Arc::new(GrepTool),
        Arc::new(SearchDocsTool),
    ];
    tools.extend(code_nav_tools());
    tools
}

/// Register [`default_tools`] on `reg` (later registrations with the same name
/// replace earlier ones, so callers can override any of them afterwards).
pub fn register_defaults(reg: &mut ToolRegistry) {
    reg.register_all(default_tools());
}

/// A registry for `ctx` with the default tools already registered (what the TS
/// `new ToolRegistry(ctx)` produced).
pub fn default_registry(ctx: qo_core::ToolContext) -> ToolRegistry {
    let mut reg = ToolRegistry::new(ctx);
    register_defaults(&mut reg);
    reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn registers_the_ts_default_set() {
        let reg = default_registry(qo_core::ToolContext::new("."));
        let names = reg.names();
        for n in [
            "read_file", "write_file", "create_file", "delete_file", "list_files", "tree", "run_shell", "shell_help",
            "git_status", "git_diff", "grep", "search_docs", "find_symbol", "search_symbols", "outline_file",
            "find_references", "read_symbol", "apply_edit",
        ] {
            assert!(names.iter().any(|x| x == n), "{n}");
        }
        assert_eq!(names.len(), 18);
        let r = reg.run("nope", json!({})).await;
        assert_eq!(r.output, "Unknown tool: nope");
        // Mutating flags: reads may run in parallel, writes serially.
        assert!(!reg.get("read_file").unwrap().mutating());
        assert!(reg.get("apply_edit").unwrap().mutating());
    }
}
