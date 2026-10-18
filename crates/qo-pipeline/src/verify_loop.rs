//! Verify-in-the-loop (port of `src/core/verifyLoop.ts`).
//!
//! The reliable way to lift a small model to strong coding output is to
//! *close the loop*: after an edit, actually run the type checker / linter /
//! tests and feed the structured failures back so the next attempt is
//! grounded in reality rather than the model's imagination. Frontier models
//! can sometimes one-shot; small models need the feedback loop, and with it
//! they reach comparable correctness on well-scoped tasks.
//!
//! This module auto-detects the project's check commands and runs them with a
//! timeout (stdin null, Esc-cancellable, whole process tree killed), returning
//! a compact, de-duplicated, model-friendly error digest (truncated,
//! most-relevant-first) rather than a wall of raw output.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::process::run_captured;

/// Default per-check timeout (ms).
pub const DEFAULT_CHECK_TIMEOUT_MS: u64 = 120_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckKind {
    Typecheck,
    Lint,
    Test,
    Build,
}

impl CheckKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckKind::Typecheck => "typecheck",
            CheckKind::Lint => "lint",
            CheckKind::Test => "test",
            CheckKind::Build => "build",
        }
    }
}

impl std::fmt::Display for CheckKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckCommand {
    pub kind: CheckKind,
    pub cmd: String,
    pub args: Vec<String>,
    /// Working directory (defaults to the project root).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
}

impl CheckCommand {
    fn new(kind: CheckKind, cmd: &str, args: &[&str]) -> Self {
        Self { kind, cmd: cmd.to_string(), args: args.iter().map(|a| a.to_string()).collect(), cwd: None }
    }

    /// `cmd args…` as shown to the user / model.
    pub fn display(&self) -> String {
        format!("{} {}", self.cmd, self.args.join(" "))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckResult {
    pub kind: CheckKind,
    pub cmd: String,
    pub passed: bool,
    pub duration_ms: u64,
    /// Compact, model-friendly digest of failures (empty when passed).
    pub digest: String,
    /// Raw tail, for debugging.
    pub raw_tail: String,
    pub timed_out: bool,
    /// Stopped by the Esc interrupt.
    #[serde(default)]
    pub interrupted: bool,
}

fn read_pkg(root: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(root.join("package.json")).ok()?).ok()
}

fn has(root: &Path, f: &str) -> bool {
    root.join(f).exists()
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(_) => true,
    }
}

/// Detect the check commands available in a project. Order matters:
/// typecheck and lint are fast feedback; tests are the ground truth; build last.
pub fn detect_checks(root: &Path) -> Vec<CheckCommand> {
    let mut checks = Vec::new();

    // ── Node / TypeScript ──
    if let Some(pkg) = read_pkg(root) {
        let scripts = pkg.get("scripts");
        let script = |name: &str| truthy(scripts.and_then(|s| s.get(name)));
        let dep = |name: &str| {
            truthy(pkg.get("dependencies").and_then(|d| d.get(name)))
                || truthy(pkg.get("devDependencies").and_then(|d| d.get(name)))
        };
        let runner = if has(root, "pnpm-lock.yaml") {
            "pnpm"
        } else if has(root, "yarn.lock") {
            "yarn"
        } else {
            "npm"
        };
        let run_script = |kind: CheckKind, name: &str| {
            if runner == "npm" {
                CheckCommand::new(kind, runner, &["run", name])
            } else {
                CheckCommand::new(kind, runner, &[name])
            }
        };
        if script("typecheck") {
            checks.push(run_script(CheckKind::Typecheck, "typecheck"));
        } else if dep("typescript") && has(root, "tsconfig.json") {
            checks.push(CheckCommand::new(CheckKind::Typecheck, "npx", &["tsc", "--noEmit"]));
        }
        if script("lint") {
            checks.push(run_script(CheckKind::Lint, "lint"));
        }
        if script("test") {
            checks.push(run_script(CheckKind::Test, "test"));
        }
        if script("build") {
            checks.push(run_script(CheckKind::Build, "build"));
        }
        return checks;
    }

    // ── Python ──
    if has(root, "pyproject.toml") || has(root, "setup.py") || has(root, "requirements.txt") {
        if has(root, "mypy.ini") || has(root, "pyproject.toml") {
            checks.push(CheckCommand::new(CheckKind::Typecheck, "python3", &["-m", "mypy", "."]));
        }
        checks.push(CheckCommand::new(CheckKind::Lint, "python3", &["-m", "ruff", "check", "."]));
        checks.push(CheckCommand::new(CheckKind::Test, "python3", &["-m", "pytest", "-q"]));
        return checks;
    }

    // ── Go ──
    if has(root, "go.mod") {
        checks.push(CheckCommand::new(CheckKind::Build, "go", &["build", "./..."]));
        checks.push(CheckCommand::new(CheckKind::Test, "go", &["test", "./..."]));
        return checks;
    }

    // ── Rust ──
    if has(root, "Cargo.toml") {
        checks.push(CheckCommand::new(CheckKind::Typecheck, "cargo", &["check"]));
        checks.push(CheckCommand::new(CheckKind::Test, "cargo", &["test"]));
        return checks;
    }

    checks
}

// The TS regexes ran without the `u` flag, so `\b` and `\w` are ASCII.
static ERROR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?-u:\b)(error|fail(ed|ure)?|exception|cannot find|not assignable|undefined|expected|Traceback|panic:|FAILED|✗|×)(?-u:\b)",
    )
    .expect("static regex")
});
static FILE_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)([A-Za-z0-9_./\\-]+\.(ts|tsx|js|jsx|py|go|rs|java|kt|c|cpp|rb|php)):(\d+)").expect("static regex")
});

/// Compress raw tool output into a model-friendly error digest.
pub fn digest_errors(kind: CheckKind, output: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = output.split('\n').collect();
    // Heuristic: lines that look like errors/warnings/failures get priority.
    let mut priority: Vec<&str> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for l in &lines {
        let t = l.trim_end();
        if !t.trim().is_empty() && (ERROR_RE.is_match(t) || FILE_LINE_RE.is_match(t)) {
            let key = t.trim();
            if seen.insert(key) {
                priority.push(t);
            }
        }
        if priority.len() >= max_lines {
            break;
        }
    }

    if !priority.is_empty() {
        let more = if priority.len() >= max_lines { format!("\n… ({kind} output truncated)") } else { String::new() };
        return priority.join("\n") + &more;
    }
    // Nothing matched the error heuristic — return the tail.
    let start = lines.len().saturating_sub(max_lines);
    let tail = lines[start..].join("\n");
    let tail = tail.trim();
    if tail.is_empty() {
        "(no output)".to_string()
    } else {
        tail.to_string()
    }
}

fn last_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.split('\n').collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Run a single check with a timeout; never fails (spawn errors become a
/// failed result with a "Could not run" digest).
pub async fn run_check(
    root: &Path,
    check: &CheckCommand,
    timeout_ms: u64,
    cancel: Option<&CancellationToken>,
) -> CheckResult {
    let t0 = Instant::now();
    let cwd = check.cwd.clone().unwrap_or_else(|| root.to_path_buf());
    let env = [("CI", "1"), ("FORCE_COLOR", "0")];
    match run_captured(&check.cmd, &check.args, &cwd, &env, Duration::from_millis(timeout_ms), cancel).await {
        Ok(res) => {
            // execa strips one final newline from the captured output.
            let out = res.output.strip_suffix('\n').unwrap_or(&res.output);
            let passed = res.success();
            let digest = if passed {
                String::new()
            } else if res.interrupted {
                format!("{} interrupted", check.kind)
            } else {
                digest_errors(check.kind, out, 40)
            };
            CheckResult {
                kind: check.kind,
                cmd: check.display(),
                passed,
                duration_ms: t0.elapsed().as_millis() as u64,
                digest,
                raw_tail: last_lines(out, 60),
                timed_out: res.timed_out,
                interrupted: res.interrupted,
            }
        }
        Err(e) => CheckResult {
            kind: check.kind,
            cmd: check.display(),
            passed: false,
            duration_ms: t0.elapsed().as_millis() as u64,
            digest: format!("Could not run {}: {e}", check.kind),
            raw_tail: e.to_string(),
            timed_out: false,
            interrupted: false,
        },
    }
}

#[derive(Clone, Debug, Default)]
pub struct VerifyOptions {
    /// Restrict to these kinds (e.g. only fast checks during an inner loop).
    pub only: Option<Vec<CheckKind>>,
    /// Stop at the first failing check (fast-fail) instead of running all.
    pub stop_on_first_failure: bool,
    /// Per-check timeout; default [`DEFAULT_CHECK_TIMEOUT_MS`].
    pub timeout_ms: Option<u64>,
    /// Esc interrupt: kills the running check and stops the loop.
    pub cancel: Option<CancellationToken>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyOutcome {
    pub ran: bool,
    pub all_passed: bool,
    pub results: Vec<CheckResult>,
    /// Combined model-friendly feedback for the failing checks.
    pub feedback: String,
    /// The loop was stopped by the Esc interrupt.
    #[serde(default)]
    pub interrupted: bool,
}

/// Run the detected checks and assemble a single feedback string for the
/// model. The orchestrator / error fixer can paste `feedback` straight into
/// the next turn.
pub async fn verify(root: &Path, opts: &VerifyOptions) -> VerifyOutcome {
    let mut checks = detect_checks(root);
    if let Some(only) = &opts.only {
        checks.retain(|c| only.contains(&c.kind));
    }
    verify_checks(root, &checks, opts).await
}

/// [`verify`] over an explicit check list.
pub async fn verify_checks(root: &Path, checks: &[CheckCommand], opts: &VerifyOptions) -> VerifyOutcome {
    if checks.is_empty() {
        return VerifyOutcome { ran: false, all_passed: true, ..Default::default() };
    }
    let timeout = opts.timeout_ms.unwrap_or(DEFAULT_CHECK_TIMEOUT_MS);
    let mut results = Vec::new();
    let mut interrupted = false;
    for c in checks {
        if opts.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            interrupted = true;
            break;
        }
        let r = run_check(root, c, timeout, opts.cancel.as_ref()).await;
        let failed = !r.passed;
        interrupted |= r.interrupted;
        results.push(r);
        if interrupted || (failed && opts.stop_on_first_failure) {
            break;
        }
    }

    let failing: Vec<&CheckResult> = results.iter().filter(|r| !r.passed).collect();
    let all_passed = failing.is_empty() && !interrupted;
    let feedback =
        failing.iter().map(|r| format!("### {} FAILED ({})\n{}", r.kind, r.cmd, r.digest)).collect::<Vec<_>>().join("\n\n");
    VerifyOutcome { ran: true, all_passed, results, feedback, interrupted }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // Ported from tests/smallModelTools.test.ts ("verifyLoop detection + digest").

    #[test]
    fn detects_npm_test_and_typecheck_scripts() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"scripts":{"test":"vitest run","typecheck":"tsc --noEmit"},"devDependencies":{"typescript":"5"}}"#,
        )
        .unwrap();
        let kinds: Vec<CheckKind> = detect_checks(dir.path()).iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&CheckKind::Test));
        assert!(kinds.contains(&CheckKind::Typecheck));
    }

    #[test]
    fn detects_cargo_for_a_rust_project() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        assert!(detect_checks(dir.path()).iter().any(|c| c.cmd == "cargo"));
    }

    #[test]
    fn digests_errors_prioritizing_error_lines_with_file_line() {
        let raw = [
            "Compiling...",
            "ok ok ok",
            "src/foo.ts:42:10 - error TS2322: Type 'string' is not assignable to type 'number'.",
            "noise noise",
            "FAILED src/bar.test.ts > does a thing",
        ]
        .join("\n");
        let digest = digest_errors(CheckKind::Typecheck, &raw, 40);
        assert!(digest.contains("src/foo.ts:42"));
        assert!(digest.contains("FAILED"));
        assert!(!digest.contains("noise"));
    }

    // Additional coverage.

    #[test]
    fn detects_node_checks_in_order() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"scripts":{"build":"b","test":"t","lint":"l"},"devDependencies":{"typescript":"5"}}"#,
        )
        .unwrap();
        fs::write(dir.path().join("tsconfig.json"), "{}").unwrap();
        let c = detect_checks(dir.path());
        let shown: Vec<String> = c.iter().map(|c| format!("{}:{}", c.kind, c.display())).collect();
        assert_eq!(
            shown,
            vec!["typecheck:npx tsc --noEmit", "lint:npm run lint", "test:npm run test", "build:npm run build"]
        );
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        assert_eq!(detect_checks(dir.path())[1].display(), "yarn lint");
    }

    #[test]
    fn detects_other_ecosystems() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detect_checks(dir.path()).is_empty());
        fs::write(dir.path().join("Cargo.toml"), "").unwrap();
        let kinds: Vec<CheckKind> = detect_checks(dir.path()).iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![CheckKind::Typecheck, CheckKind::Test]);
        fs::write(dir.path().join("go.mod"), "").unwrap();
        assert_eq!(detect_checks(dir.path())[0].display(), "go build ./...");
        fs::write(dir.path().join("requirements.txt"), "").unwrap();
        let shown: Vec<String> = detect_checks(dir.path()).iter().map(|c| c.display()).collect();
        assert_eq!(shown, vec!["python3 -m ruff check .", "python3 -m pytest -q"]);
    }

    #[test]
    fn digest_prioritises_error_lines_and_dedupes() {
        let out = "compiling\nsrc/a.ts:3:1 - error TS2322: not assignable\nok line\nsrc/a.ts:3:1 - error TS2322: not assignable\nTests FAILED\n";
        let d = digest_errors(CheckKind::Typecheck, out, 40);
        assert_eq!(d, "src/a.ts:3:1 - error TS2322: not assignable\nTests FAILED");
        assert_eq!(digest_errors(CheckKind::Test, "one\ntwo\nthree", 2), "two\nthree");
        // Like the TS: the tail is taken before trimming, so trailing blank
        // lines can leave nothing.
        assert_eq!(digest_errors(CheckKind::Test, "one\ntwo\nthree\n\n", 2), "(no output)");
        assert_eq!(digest_errors(CheckKind::Test, "\n\n", 40), "(no output)");
        let many: String = (0..50).map(|i| format!("error {i}\n")).collect();
        let d = digest_errors(CheckKind::Lint, &many, 40);
        assert!(d.ends_with("\n… (lint output truncated)"));
        assert_eq!(d.lines().count(), 41);
    }

    #[test]
    fn ascii_word_boundaries_like_js() {
        // `\bpanic:\b` needs a word char right after the colon (JS semantics).
        assert!(!ERROR_RE.is_match("panic: oh no"));
        assert!(ERROR_RE.is_match("panic:oh"));
        assert!(ERROR_RE.is_match("Error: x"));
        assert!(!ERROR_RE.is_match("errors"));
    }

    #[tokio::test]
    async fn verify_reports_failure_digest_and_passes_after_fix() {
        if which::which("node").is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("package.json"), r#"{"scripts":{"typecheck":"node check.js"}}"#).unwrap();
        fs::write(
            dir.path().join("check.js"),
            "const fs=require('fs');const t=fs.readFileSync('t.txt','utf8');if(!t.includes('FIXED')){console.error('error: t.txt is broken');process.exit(1)}",
        )
        .unwrap();
        fs::write(dir.path().join("t.txt"), "BROKEN").unwrap();
        let opts = VerifyOptions { stop_on_first_failure: true, ..Default::default() };
        let v = verify(dir.path(), &opts).await;
        assert!(v.ran && !v.all_passed, "{v:?}");
        assert!(v.feedback.starts_with("### typecheck FAILED (npm run typecheck)"), "{}", v.feedback);
        assert!(v.feedback.contains("error: t.txt is broken"), "{}", v.feedback);
        fs::write(dir.path().join("t.txt"), "FIXED").unwrap();
        let v = verify(dir.path(), &opts).await;
        assert!(v.all_passed, "{v:?}");
        assert!(v.feedback.is_empty());
    }

    #[tokio::test]
    async fn no_checks_means_not_ran_but_passed() {
        let dir = tempfile::tempdir().unwrap();
        let v = verify(dir.path(), &VerifyOptions::default()).await;
        assert!(!v.ran && v.all_passed);
    }

    #[tokio::test]
    async fn missing_tool_becomes_a_failed_check() {
        let dir = tempfile::tempdir().unwrap();
        let checks = [CheckCommand::new(CheckKind::Lint, "definitely-not-a-linter-xyz", &["."])];
        let v = verify_checks(dir.path(), &checks, &VerifyOptions::default()).await;
        assert!(v.ran && !v.all_passed);
        assert!(v.results[0].digest.starts_with("Could not run lint:"));
    }

    #[tokio::test]
    async fn cancelled_verify_stops_and_reports_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let checks = [CheckCommand::new(CheckKind::Test, "git", &["--version"])];
        let v = verify_checks(dir.path(), &checks, &VerifyOptions { cancel: Some(cancel), ..Default::default() }).await;
        assert!(v.interrupted && !v.all_passed);
        assert!(v.results.is_empty());
    }
}
