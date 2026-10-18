//! Tools exposing a persistent shell session (see [`crate::shell_session`]).
//! These give a small model a terminal whose STATE PERSISTS across calls — so
//! `cd`, venv activation, exported env vars, REPLs, and long-running servers all
//! survive between tool calls, and a long command can be started then polled
//! instead of blocking the whole turn or being killed at the timeout.
//!
//! Registered only when tools.shell_session is enabled (it weakens the
//! per-command isolation of the default run_shell, so it's opt-in).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use qo_core::tool::SharedTool;
use qo_core::{Tool, ToolContext, ToolResult, ToolSpec};
use serde_json::{json, Value};

use crate::file_tools::strip_redundant_cwd_prefix_in_command;
use crate::shell_session::{RunResult, RunStatus, ShellSession};
use crate::shell_tools::{classify_command, ClassifyOptions, CommandClass};
use crate::util::{js_number_or, js_string, js_truthy};

fn format(r: &RunResult) -> ToolResult {
    let body = r.output.trim();
    match r.status {
        RunStatus::Done => {
            let code = r.exit_code.map_or_else(|| "?".to_string(), |c| c.to_string());
            let ok = r.exit_code.map_or(true, |c| c == 0);
            let out = format!("{}\n[finished, exit code {code}]", if body.is_empty() { "(no output)" } else { body });
            ToolResult { ok, output: out, data: Some(json!({ "status": "done", "exitCode": r.exit_code })) }
        }
        RunStatus::Running => ToolResult::ok(format!(
            "{}\n[still running — call shell_session_read to get more output once it progresses]",
            if body.is_empty() { "(no output yet)" } else { body }
        ))
        .with_data(json!({ "status": "running" })),
        RunStatus::Interrupted => ToolResult::err(format!(
            "{}\n[interrupted — Ctrl+C was sent to the persistent shell]",
            if body.is_empty() { "(no output)" } else { body }
        ))
        .with_data(json!({ "status": "interrupted" })),
        RunStatus::Busy | RunStatus::Idle | RunStatus::Error => ToolResult {
            ok: r.status != RunStatus::Error,
            output: body.to_string(),
            data: Some(json!({ "status": r.status.as_str() })),
        },
    }
}

/// `Math.min(Math.max(Number(x) || dflt, lo), hi)` in milliseconds.
fn clamp_ms(v: Option<&Value>, dflt: f64, lo: f64, hi: f64) -> Duration {
    Duration::from_millis(js_number_or(v, dflt).clamp(lo, hi) as u64)
}

pub struct ShellSessionTool {
    session: Arc<ShellSession>,
}

#[async_trait]
impl Tool for ShellSessionTool {
    fn name(&self) -> &str {
        "shell_session"
    }
    fn description(&self) -> &str {
        "Run a command in a PERSISTENT shell whose state (cwd, env vars, activated venv, running \
         processes) survives between calls — unlike run_shell, which is fresh each time. Use it when \
         commands depend on each other (cd then build), or to start a long-running process (server, \
         watcher, REPL) and poll it with shell_session_read."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        if !ctx.allow_shell {
            return ToolResult::err("Shell execution is disabled (tools.allow_shell=false).");
        }
        let raw = if js_truthy(args.get("command")) { js_string(args.get("command")) } else { String::new() };
        let raw = raw.trim();
        if raw.is_empty() {
            return ToolResult::err("No command given");
        }
        let command = strip_redundant_cwd_prefix_in_command(&ctx.cwd, raw);
        let class = classify_command(&command, Some(&ClassifyOptions::from_ctx(ctx)));
        if class == CommandClass::Blocked {
            ctx.log(json!({ "tool": "shell_session", "command": command, "blocked": true }));
            return ToolResult::err(format!("Refused: \"{command}\" matches a hard-blocked/deny-listed pattern."));
        }
        // Same destructive gate as run_shell (the TS chat loop applied it to both).
        if class == CommandClass::Destructive && ctx.confirm_destructive && !ctx.auto_confirm {
            let go = ctx.confirm(&format!("Run potentially destructive command?\n  {command}")).await;
            if !go {
                ctx.log(json!({ "tool": "shell_session", "command": command, "declined": true }));
                return ToolResult::err("Declined by user.");
            }
        }
        let timeout = clamp_ms(args.get("timeout_ms"), 15_000.0, 1_000.0, 120_000.0);
        let r = self.session.run(&command, timeout, ctx.cancel.as_ref()).await;
        ctx.log(json!({ "tool": "shell_session", "command": command, "status": r.status.as_str(), "exitCode": r.exit_code }));
        format(&r)
    }
}

pub struct ShellSessionReadTool {
    session: Arc<ShellSession>,
}

#[async_trait]
impl Tool for ShellSessionReadTool {
    fn name(&self) -> &str {
        "shell_session_read"
    }
    fn description(&self) -> &str {
        "Read more output from a command still running in the persistent shell (started by \
         shell_session). Returns new output and whether the command has finished."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let timeout = clamp_ms(args.get("timeout_ms"), 10_000.0, 500.0, 120_000.0);
        let r = self.session.read(timeout, ctx.cancel.as_ref()).await;
        ctx.log(json!({ "tool": "shell_session_read", "status": r.status.as_str() }));
        format(&r)
    }
}

pub struct ShellSessionResetTool {
    session: Arc<ShellSession>,
}

#[async_trait]
impl Tool for ShellSessionResetTool {
    fn name(&self) -> &str {
        "shell_session_reset"
    }
    fn description(&self) -> &str {
        "Kill and restart the persistent shell (clears cwd/env and stops any running process in it). \
         Use if the session is stuck or you want a clean environment."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
        self.session.reset();
        ctx.log(json!({ "tool": "shell_session_reset" }));
        ToolResult::ok("Persistent shell reset — a fresh shell will start on the next command.")
    }
}

/// The three shell-session tools sharing one `session`.
pub fn create_shell_session_tools(session: Arc<ShellSession>) -> Vec<SharedTool> {
    vec![
        Arc::new(ShellSessionTool { session: Arc::clone(&session) }),
        Arc::new(ShellSessionReadTool { session: Arc::clone(&session) }),
        Arc::new(ShellSessionResetTool { session }),
    ]
}

/// Specs advertised to the model when the shell session is enabled.
pub fn shell_session_tool_specs() -> Vec<ToolSpec> {
    let spec = |name: &str, description: &str, parameters: Value| ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
    };
    vec![
        spec(
            "shell_session",
            "Run a command in a PERSISTENT shell whose state (current directory, environment variables, \
             activated virtualenv, background processes) survives across calls — unlike run_shell, which \
             starts fresh every time. Use when commands build on each other (e.g. `cd build` then `cmake ..`), \
             or to launch a long-running process (dev server, file watcher, REPL) and then poll it with \
             shell_session_read. On Windows the shell is PowerShell.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command line to run in the persistent shell." },
                    "timeout_ms": { "type": "integer", "description": "How long to wait for it to finish before returning partial output (default 15000)." }
                },
                "required": ["command"]
            }),
        ),
        spec(
            "shell_session_read",
            "Get more output from a command that was still running in the persistent shell (after \
             shell_session reported it hadn't finished). Returns the new output and whether it's done now.",
            json!({
                "type": "object",
                "properties": {
                    "timeout_ms": { "type": "integer", "description": "How long to wait for more output (default 10000)." }
                }
            }),
        ),
        spec(
            "shell_session_reset",
            "Restart the persistent shell from scratch — clears the working directory, environment, and any \
             process running inside it. Use if the session is wedged or you need a clean slate.",
            json!({ "type": "object", "properties": {} }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_each_status() {
        let done = format(&RunResult { status: RunStatus::Done, output: "hi\n".into(), exit_code: Some(0) });
        assert!(done.ok);
        assert_eq!(done.output, "hi\n[finished, exit code 0]");
        let failed = format(&RunResult { status: RunStatus::Done, output: String::new(), exit_code: Some(2) });
        assert!(!failed.ok);
        assert_eq!(failed.output, "(no output)\n[finished, exit code 2]");
        let unknown = format(&RunResult { status: RunStatus::Done, output: "x".into(), exit_code: None });
        assert!(unknown.ok && unknown.output.ends_with("exit code ?]"));
        let running = format(&RunResult { status: RunStatus::Running, output: String::new(), exit_code: None });
        assert!(running.ok && running.output.starts_with("(no output yet)"));
        let err = format(&RunResult { status: RunStatus::Error, output: "boom".into(), exit_code: None });
        assert!(!err.ok);
    }

    #[tokio::test]
    async fn refuses_blocked_and_disabled() {
        let session = Arc::new(ShellSession::new(std::env::temp_dir()));
        let tools = create_shell_session_tools(Arc::clone(&session));
        let mut ctx = ToolContext::new(std::env::temp_dir());
        let r = tools[0].run(json!({ "command": "shutdown now" }), &ctx).await;
        assert!(r.output.starts_with("Refused"));
        ctx.allow_shell = false;
        let r = tools[0].run(json!({ "command": "echo x" }), &ctx).await;
        assert!(r.output.contains("disabled"));
        let r = tools[1].run(json!({}), &ToolContext::new(".")).await;
        assert!(r.ok && r.output.starts_with("No command has been run"));
        let r = tools[2].run(json!({}), &ToolContext::new(".")).await;
        assert!(r.ok);
        assert!(!session.is_alive());
    }

    #[test]
    fn specs_names_match_tools() {
        let names: Vec<String> = shell_session_tool_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["shell_session", "shell_session_read", "shell_session_reset"]);
    }
}
