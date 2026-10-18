//! Tester (port of `src/agents/tester.ts`): run the project's detected test
//! command through the registry's `run_shell` tool (which owns the shell
//! safety rules, timeout and Esc cancellation).

use serde_json::json;

use qo_core::{RepoInfo, TestResult, ToolRegistry};

/// Run the project's test command. Never fakes success: with no command
/// detected it returns `ran: false` so the caller can decide what to do.
pub async fn run_tests(tools: &ToolRegistry, repo: &RepoInfo, explicit_command: Option<&str>) -> TestResult {
    let command = explicit_command.filter(|c| !c.is_empty()).or(repo.test_command.as_deref().filter(|c| !c.is_empty()));
    let Some(command) = command else {
        return TestResult {
            ran: false,
            passed: false,
            command: None,
            output: "No test command detected for this project.".into(),
        };
    };
    let res = tools.run("run_shell", json!({ "command": command })).await;
    TestResult { ran: true, passed: res.ok, command: Some(command.to_string()), output: res.output }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{registry_with, FakeTool};
    use qo_core::ToolResult;

    #[tokio::test]
    async fn no_command_means_not_ran() {
        let reg = registry_with(std::env::temp_dir(), vec![]);
        let r = run_tests(&reg, &RepoInfo::default(), None).await;
        assert!(!r.ran && !r.passed);
    }

    #[tokio::test]
    async fn runs_through_run_shell() {
        let tool = FakeTool::new("run_shell", ToolResult::ok("3 passed"));
        let reg = registry_with(std::env::temp_dir(), vec![tool.clone()]);
        let repo = RepoInfo { test_command: Some("cargo test".into()), ..Default::default() };
        let r = run_tests(&reg, &repo, None).await;
        assert!(r.ran && r.passed);
        assert_eq!(r.command.as_deref(), Some("cargo test"));
        assert_eq!(tool.calls()[0]["command"], "cargo test");
        let r = run_tests(&reg, &repo, Some("npm test")).await;
        assert_eq!(r.command.as_deref(), Some("npm test"));
    }
}
