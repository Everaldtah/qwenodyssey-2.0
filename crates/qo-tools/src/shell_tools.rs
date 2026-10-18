//! Shell execution with a hard blocklist for catastrophic commands and an
//! optional confirmation gate for anything that looks destructive.
//!
//! The gate asks through `ctx.confirm(...)` (the chat layer's single input
//! owner) — a tool never reads stdin itself. On Windows commands run through
//! Windows PowerShell 5.1 exactly like the TS (`-NoProfile -NonInteractive
//! -InputFormat None -ExecutionPolicy Bypass -Command <preamble + command>`),
//! with the child's stdin null, no console window, a timeout that kills the
//! whole process tree, and Esc cancellation via `ctx.cancel`.

use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};

use crate::file_tools::strip_redundant_cwd_prefix_in_command;
use crate::ps_compat::adapt_chains_for_powershell;
use crate::util::{command, js_number, js_string, run_captured, tool_error};

fn ci(pattern: &str) -> Regex {
    RegexBuilder::new(pattern).case_insensitive(true).build().expect("static shell-safety regex")
}

/// Patterns that are NEVER run, regardless of confirmation.
static HARD_BLOCK: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        ci(r"\brm\s+-rf?\s+(/|~|\$HOME|\.\.)(\s|$)"),
        ci(r"\bmkfs(\.\w+)?\b"),
        ci(r"\bdd\s+if="),
        ci(r"\bshutdown\b"),
        ci(r"\breboot\b"),
        ci(r"\bhalt\b"),
        ci(r"\bpoweroff\b"),
        // classic fork bomb :(){ :|:& };:
        Regex::new(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;").expect("static regex"),
        ci(r"\bformat\s+[a-z]:"), // Windows format C:
        ci(r">\s*/dev/sda"),
        ci(r"\bchmod\s+-R\s+777\s+/"),
        // ── Windows / PowerShell catastrophic operations ──
        ci(r"\bFormat-Volume\b"),
        ci(r"\b(Clear|Initialize|Remove)-Disk\b"),
        ci(r"\b(Stop|Restart)-Computer\b"),
        // Recursive force-delete of a drive root, the user profile, or the Windows dir
        // (either -Recurse/-Force order). Defense-in-depth for the PowerShell shell.
        ci(r#"\bRemove-Item\b[^|\n]*-(Recurse|Force)\b[^|\n]*-(Recurse|Force)\b[^|\n]*(\b[A-Za-z]:\\?(?:\s|$|["'])|\$HOME\b|\$env:USERPROFILE\b|\$env:SystemRoot\b|[\\/]Windows\b)"#),
        // Deleting a whole registry hive
        ci(r"\breg\s+delete\s+HK(LM|EY_LOCAL_MACHINE|CR|EY_CLASSES_ROOT)\b"),
        ci(r"\bRemove-Item\b[^|\n]*\bHKLM:"),
        // cmd recursive wipe of a drive root: rd /s /q C:\
        ci(r"\b(rd|rmdir)\s+/s\s+/q\s+[A-Za-z]:\\?(\s|$)"),
        ci(r"\bdel\s+/[sq]\s+.*[A-Za-z]:\\?(\s|$)"),
    ]
});

/// Patterns that require confirmation when confirm_destructive is on.
static DESTRUCTIVE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\brm\b",
        r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+.*--force)",
        r"\bdel\b",
        r"\brmdir\b",
        r"\bRemove-Item\b",
        r"\bRemove-Item\b.*\bHK(CU|LM|CR):",
        r"\bmv\b",
        r"\bMove-Item\b",
        r"\bnpm\s+uninstall\b",
        r"\bdrop\s+(table|database)\b",
        // ── Windows / PowerShell state-changing operations (confirm first) ──
        r"\breg\s+(delete|add)\b",
        r"\b(takeown|icacls)\b",
        r"\bClear-Content\b",
        r"\bSet-ItemProperty\b",
        r"\bStop-Process\b",
        r"\bUninstall-\w+",
        r"\bnet\s+user\b.*/(delete|add)",
    ]
    .iter()
    .map(|p| ci(p))
    .collect()
});

/// Compile config-supplied patterns; fall back to a literal-substring match
/// when a pattern is not a valid regex (e.g. JS-only lookaround syntax).
fn compile_patterns(patterns: &[String]) -> Vec<Regex> {
    patterns
        .iter()
        .filter_map(|p| {
            RegexBuilder::new(p)
                .case_insensitive(true)
                .build()
                .or_else(|_| RegexBuilder::new(&regex::escape(p)).case_insensitive(true).build())
                .ok()
        })
        .collect()
}

/// Safety class of a shell command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandClass {
    Blocked,
    Destructive,
    Safe,
}

impl CommandClass {
    /// The TS string form ("blocked" | "destructive" | "safe").
    pub fn as_str(&self) -> &'static str {
        match self {
            CommandClass::Blocked => "blocked",
            CommandClass::Destructive => "destructive",
            CommandClass::Safe => "safe",
        }
    }
}

impl std::fmt::Display for CommandClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Config-driven overrides for [`classify_command`].
#[derive(Clone, Debug, Default)]
pub struct ClassifyOptions {
    /// Config tools.allow_commands — matching commands skip the destructive gate.
    pub allow: Vec<String>,
    /// Config tools.deny_commands — matching commands are hard-blocked.
    pub deny: Vec<String>,
}

impl ClassifyOptions {
    /// The allow/deny lists carried by a tool context.
    pub fn from_ctx(ctx: &ToolContext) -> Self {
        Self { allow: ctx.allow_commands.clone(), deny: ctx.deny_commands.clone() }
    }
}

/// Classify a command for the safety gate. Precedence:
///   deny-list  →  built-in hard-block  →  allow-list  →  built-in destructive.
/// The allow-list can downgrade a *destructive* command to safe (auto-approve),
/// but it can NEVER override a hard-block — you cannot allowlist `rm -rf /`.
pub fn classify_command(cmd: &str, opts: Option<&ClassifyOptions>) -> CommandClass {
    if let Some(o) = opts {
        if compile_patterns(&o.deny).iter().any(|re| re.is_match(cmd)) {
            return CommandClass::Blocked;
        }
    }
    if HARD_BLOCK.iter().any(|re| re.is_match(cmd)) {
        return CommandClass::Blocked;
    }
    if let Some(o) = opts {
        if compile_patterns(&o.allow).iter().any(|re| re.is_match(cmd)) {
            return CommandClass::Safe;
        }
    }
    if DESTRUCTIVE.iter().any(|re| re.is_match(cmd)) {
        return CommandClass::Destructive;
    }
    CommandClass::Safe
}

/// Default run_shell timeout when neither the call nor the config sets one.
pub const DEFAULT_SHELL_TIMEOUT_MS: u64 = 240_000;

/// Clamp a requested timeout into a sane band (1s … 10min).
fn resolve_timeout(args: &Value, ctx: &ToolContext) -> u64 {
    let requested = js_number(args.get("timeout_ms"));
    let requested = if requested.is_nan() || requested == 0.0 {
        match ctx.shell_timeout_ms {
            Some(ms) if ms > 0 => ms as f64,
            _ => DEFAULT_SHELL_TIMEOUT_MS as f64,
        }
    } else {
        requested
    };
    requested.clamp(1_000.0, 600_000.0) as u64
}

/// Preamble prepended to every Windows PowerShell command so common web cmdlets
/// work in our `-NonInteractive` shell. In Windows PowerShell 5.1, an
/// `Invoke-WebRequest`/`Invoke-RestMethod` WITHOUT `-UseBasicParsing` falls back to
/// the legacy Internet Explorer DOM engine, which tries to prompt and then dies
/// with "Windows PowerShell is in NonInteractive mode. Read and Prompt
/// functionality is not available." Defaulting `-UseBasicParsing` (and silencing
/// the progress bar) fixes that for any command the model writes, without it
/// having to remember the flag.
pub const PS_PREAMBLE: &str = "$ProgressPreference='SilentlyContinue'; \
$PSDefaultParameterValues['Invoke-WebRequest:UseBasicParsing']=$true; \
$PSDefaultParameterValues['Invoke-RestMethod:UseBasicParsing']=$true; ";

/// Build the platform shell invocation for `cmd`. On Windows: PowerShell 5.1,
/// passing the command as a single `-Command` argument (not -EncodedCommand:
/// base64-encoded commands are flagged/stalled by some antivirus products);
/// `-InputFormat None` + a null stdin stop PowerShell hanging without a console.
/// PowerShell 5.1 has no `&&`/`||`, so bash-style chains are rewritten first.
/// Elsewhere: `sh -c`.
pub fn shell_command(cmd: &str) -> tokio::process::Command {
    if cfg!(windows) {
        let mut c = command("powershell.exe");
        let ps_cmd = adapt_chains_for_powershell(cmd);
        c.args([
            "-NoProfile",
            "-NonInteractive",
            "-InputFormat",
            "None",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &format!("{PS_PREAMBLE}{ps_cmd}"),
        ]);
        c
    } else {
        let mut c = command("sh");
        c.args(["-c", cmd]);
        c
    }
}

async fn execute(cmd: &str, ctx: &ToolContext, timeout_ms: u64) -> ToolResult {
    let mut proc = shell_command(cmd);
    proc.current_dir(&ctx.cwd);
    let res = match run_captured(proc, Some(Duration::from_millis(timeout_ms)), ctx.cancel.as_ref()).await {
        Ok(r) => r,
        Err(e) => return tool_error(ctx, "run_shell", e),
    };
    ctx.log(json!({
        "tool": "run_shell",
        "command": cmd,
        "exitCode": res.exit_code,
        "timedOut": res.timed_out,
        "interrupted": res.interrupted,
    }));
    // On timeout, return the partial output captured so far PLUS a clear pointer to
    // the right tool — so the model can recover instead of treating it as a dead end.
    let note = if res.timed_out {
        format!(
            "\n[timed out after {}s — partial output above. \
             For a slow command, retry run_shell with a larger timeout_ms (up to 600000). \
             For a genuinely long-running or interactive job (clone of a big repo, install, build, \
             server), use shell_session (persistent) and poll it with shell_session_read instead.]",
            (timeout_ms as f64 / 1000.0).round() as u64
        )
    } else if res.interrupted {
        "\n[interrupted — the command was stopped by the user (Esc)]".to_string()
    } else {
        String::new()
    };
    let ok = res.exit_code == Some(0) && !res.timed_out && !res.interrupted;
    ToolResult {
        ok,
        output: format!("{}{note}", res.all),
        data: Some(json!({ "exitCode": res.exit_code, "timedOut": res.timed_out, "interrupted": res.interrupted })),
    }
}

pub struct RunShellTool;

#[async_trait]
impl Tool for RunShellTool {
    fn name(&self) -> &str {
        "run_shell"
    }
    fn description(&self) -> &str {
        "Run a shell command in the project directory."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = if crate::util::js_truthy(args.get("command")) { js_string(args.get("command")) } else { String::new() };
        let raw = raw.trim();
        if raw.is_empty() {
            return ToolResult::err("No command given");
        }
        if !ctx.allow_shell {
            return ToolResult::err("Shell execution is disabled (tools.allow_shell=false)");
        }
        // "tree projA/" while already inside projA → "tree ." (small-model habit).
        let cmd = strip_redundant_cwd_prefix_in_command(&ctx.cwd, raw);
        if cmd != raw {
            ctx.log(json!({ "tool": "run_shell", "command": raw, "rewritten": cmd }));
        }

        let class = classify_command(&cmd, Some(&ClassifyOptions::from_ctx(ctx)));
        if class == CommandClass::Blocked {
            ctx.log(json!({ "tool": "run_shell", "command": cmd, "blocked": true }));
            return ToolResult::err(format!("Refused: \"{cmd}\" matches a hard-blocked dangerous pattern."));
        }
        if class == CommandClass::Destructive && ctx.confirm_destructive && !ctx.auto_confirm {
            let go = ctx.confirm(&format!("Run potentially destructive command?\n  {cmd}")).await;
            if !go {
                ctx.log(json!({ "tool": "run_shell", "command": cmd, "declined": true }));
                return ToolResult::err("Declined by user.");
            }
        }
        if ctx.cancelled() {
            return ToolResult::err("Interrupted before the command started.");
        }
        execute(&cmd, ctx, resolve_timeout(&args, ctx)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn c(cmd: &str) -> CommandClass {
        classify_command(cmd, None)
    }
    fn opts(allow: &[&str], deny: &[&str]) -> ClassifyOptions {
        ClassifyOptions {
            allow: allow.iter().map(|s| s.to_string()).collect(),
            deny: deny.iter().map(|s| s.to_string()).collect(),
        }
    }

    // shellSafety.test.ts
    #[test]
    fn hard_blocks_catastrophic_commands() {
        for cmd in ["rm -rf /", "sudo rm -rf /", "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda", "shutdown now", ":(){ :|:& };:"] {
            assert_eq!(c(cmd), CommandClass::Blocked, "{cmd}");
        }
    }

    #[test]
    fn flags_destructive_but_allowed() {
        for cmd in ["rm file.txt", "git reset --hard HEAD~1", "git push origin main --force"] {
            assert_eq!(c(cmd), CommandClass::Destructive, "{cmd}");
        }
    }

    #[test]
    fn passes_ordinary_commands() {
        for cmd in ["ls -la", "npm test", "git status"] {
            assert_eq!(c(cmd), CommandClass::Safe, "{cmd}");
        }
    }

    // harnessUpgrades.test.ts: Windows + allow/deny
    #[test]
    fn hard_blocks_powershell_catastrophes() {
        for cmd in ["Format-Volume -DriveLetter D", "Clear-Disk -Number 0", "Restart-Computer", "reg delete HKLM\\SOFTWARE\\Foo /f"] {
            assert_eq!(c(cmd), CommandClass::Blocked, "{cmd}");
        }
        assert_eq!(c("Remove-Item -Recurse -Force C:\\"), CommandClass::Blocked);
        assert_eq!(c("rd /s /q C:\\"), CommandClass::Blocked);
    }

    #[test]
    fn flags_windows_state_changes_as_destructive() {
        for cmd in ["Remove-Item foo.txt", "takeown /f C:\\x", "reg add HKCU\\Foo /v Bar"] {
            assert_eq!(c(cmd), CommandClass::Destructive, "{cmd}");
        }
    }

    #[test]
    fn deny_list_blocks_safe_command() {
        assert_eq!(classify_command("npm publish", Some(&opts(&[], &["^npm publish"]))), CommandClass::Blocked);
    }

    #[test]
    fn allow_list_downgrades_destructive() {
        assert_eq!(classify_command("rm scratch.txt", Some(&opts(&["^rm scratch"], &[]))), CommandClass::Safe);
    }

    #[test]
    fn allow_list_never_overrides_hard_block() {
        assert_eq!(classify_command("rm -rf /", Some(&opts(&["rm -rf"], &[]))), CommandClass::Blocked);
    }

    #[test]
    fn invalid_regex_matches_literally() {
        assert_eq!(classify_command("weird(cmd", Some(&opts(&[], &["weird(cmd"]))), CommandClass::Blocked);
    }

    #[test]
    fn preamble_fixes_noninteractive_web_cmdlets() {
        assert!(PS_PREAMBLE.contains("Invoke-WebRequest:UseBasicParsing']=$true"));
        assert!(PS_PREAMBLE.contains("Invoke-RestMethod:UseBasicParsing']=$true"));
        assert!(PS_PREAMBLE.contains("$ProgressPreference='SilentlyContinue'"));
    }

    fn ctx(dir: &std::path::Path) -> ToolContext {
        let mut c = ToolContext::new(dir);
        c.auto_confirm = false;
        c.confirm_destructive = true;
        c
    }

    #[tokio::test]
    async fn runs_a_harmless_command() {
        let dir = tempfile::tempdir().unwrap();
        let r = RunShellTool.run(json!({ "command": "echo qo-hello" }), &ctx(dir.path())).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains("qo-hello"));
        assert_eq!(r.data.unwrap()["exitCode"], json!(0));
    }

    #[tokio::test]
    async fn chains_work_on_every_platform() {
        let dir = tempfile::tempdir().unwrap();
        let r = RunShellTool.run(json!({ "command": "echo one && echo two" }), &ctx(dir.path())).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains("one") && r.output.contains("two"));
    }

    #[tokio::test]
    async fn refuses_blocked_and_declines_destructive_without_confirmer() {
        let dir = tempfile::tempdir().unwrap();
        let r = RunShellTool.run(json!({ "command": "shutdown now" }), &ctx(dir.path())).await;
        assert!(!r.ok);
        assert!(r.output.starts_with("Refused"));
        let r = RunShellTool.run(json!({ "command": "rm nothing-here.txt" }), &ctx(dir.path())).await;
        assert_eq!(r.output, "Declined by user.");
    }

    struct Yes;
    #[async_trait]
    impl qo_core::Confirmer for Yes {
        async fn confirm(&self, _q: &str) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn confirmer_approves_destructive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("gone.txt"), "x").unwrap();
        let mut cx = ctx(dir.path());
        cx.confirmer = Some(Arc::new(Yes));
        let cmd = if cfg!(windows) { "Remove-Item gone.txt" } else { "rm gone.txt" };
        let r = RunShellTool.run(json!({ "command": cmd }), &cx).await;
        assert!(r.ok, "{}", r.output);
        assert!(!dir.path().join("gone.txt").exists());
    }

    #[tokio::test]
    async fn times_out_with_partial_output_note() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = if cfg!(windows) { "Write-Output early; Start-Sleep -Seconds 30" } else { "echo early; sleep 30" };
        let r = RunShellTool.run(json!({ "command": cmd, "timeout_ms": 4000 }), &ctx(dir.path())).await;
        assert!(!r.ok);
        assert!(r.output.contains("[timed out after 4s"), "{}", r.output);
        assert_eq!(r.data.unwrap()["timedOut"], json!(true));
    }

    #[tokio::test]
    async fn cancellation_stops_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let mut cx = ctx(dir.path());
        let token = qo_core::CancellationToken::new();
        cx.cancel = Some(token.clone());
        let t2 = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            t2.cancel();
        });
        let cmd = if cfg!(windows) { "Start-Sleep -Seconds 30" } else { "sleep 30" };
        let started = std::time::Instant::now();
        let r = RunShellTool.run(json!({ "command": cmd }), &cx).await;
        assert!(!r.ok);
        assert!(r.output.contains("interrupted"));
        assert!(started.elapsed() < Duration::from_secs(20));
    }

    #[test]
    fn timeout_is_clamped() {
        let cx = ToolContext::new(".");
        assert_eq!(resolve_timeout(&json!({ "timeout_ms": 5 }), &cx), 1_000);
        assert_eq!(resolve_timeout(&json!({ "timeout_ms": "9999999" }), &cx), 600_000);
        assert_eq!(resolve_timeout(&json!({}), &cx), DEFAULT_SHELL_TIMEOUT_MS);
    }
}
