//! Git tools. The TS used simple-git; this shells out to the `git` binary with
//! the same commands simple-git runs and renders the same output.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::util::{command, js_string, js_truthy, run_captured, tool_error};

/// Generous ceiling for a git invocation (a huge diff on a slow disk).
const GIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Run `git <args>` in `cwd`; `Ok((exit_code, stdout, combined))`.
async fn git(cwd: &Path, args: &[&str], cancel: Option<&CancellationToken>) -> anyhow::Result<(Option<i32>, String, String)> {
    let mut cmd = command("git");
    // Never page, never prompt for credentials, stable machine-readable output.
    cmd.args(["-c", "core.quotepath=false", "--no-pager"]).args(args).current_dir(cwd);
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    let r = run_captured(cmd, Some(GIT_TIMEOUT), cancel).await?;
    if r.interrupted {
        anyhow::bail!("interrupted");
    }
    if r.timed_out {
        anyhow::bail!("git {} timed out", args.join(" "));
    }
    Ok((r.exit_code, r.stdout, r.all))
}

/// Is `cwd` inside a git work tree? (simple-git `checkIsRepo()`.)
pub async fn is_repo(cwd: &Path) -> bool {
    match git(cwd, &["rev-parse", "--is-inside-work-tree"], None).await {
        Ok((Some(0), out, _)) => out.trim() == "true",
        _ => false,
    }
}

/// Does the work tree have uncommitted changes (incl. untracked files)?
pub async fn is_dirty(cwd: &Path) -> bool {
    match git_status(cwd, None).await {
        Ok(s) => !s.is_clean(),
        Err(_) => false,
    }
}

/// One changed path, as simple-git's `FileStatusResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitFileStatus {
    pub path: String,
    /// Index (staged) status letter, e.g. "M", "A", "?" (untracked), " ".
    pub index: String,
    /// Work-tree status letter.
    pub working_dir: String,
    /// Original path of a rename/copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
}

/// Parsed `git status`, mirroring the simple-git `StatusResult` fields the
/// tools and repo scanner use.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatus {
    pub current: Option<String>,
    pub tracking: Option<String>,
    pub ahead: u32,
    pub behind: u32,
    pub detached: bool,
    pub files: Vec<GitFileStatus>,
}

impl GitStatus {
    pub fn is_clean(&self) -> bool {
        self.files.is_empty()
    }
}

/// Parse `git status --porcelain -b -u --null` output (what simple-git runs).
pub fn parse_porcelain_status(raw: &str) -> GitStatus {
    let mut st = GitStatus::default();
    let mut fields = raw.split('\0').filter(|f| !f.is_empty()).peekable();
    while let Some(entry) = fields.next() {
        if let Some(branch) = entry.strip_prefix("## ") {
            parse_branch_line(branch, &mut st);
            continue;
        }
        if entry.len() < 3 {
            continue;
        }
        let mut chars = entry.chars();
        let x = chars.next().unwrap_or(' ').to_string();
        let y = chars.next().unwrap_or(' ').to_string();
        let path = entry.get(3..).unwrap_or("").to_string();
        // Renames/copies carry the original path as the next NUL field.
        let from = if x == "R" || x == "C" || y == "R" || y == "C" { fields.next().map(str::to_string) } else { None };
        st.files.push(GitFileStatus { path, index: x, working_dir: y, from });
    }
    st
}

fn parse_branch_line(line: &str, st: &mut GitStatus) {
    // Forms: "main", "main...origin/main [ahead 1, behind 2]", "No commits yet on main",
    // "HEAD (no branch)".
    let line = line.trim();
    if let Some(rest) = line.strip_prefix("No commits yet on ").or_else(|| line.strip_prefix("Initial commit on ")) {
        st.current = Some(rest.trim().to_string());
        return;
    }
    if line.starts_with("HEAD (no branch)") {
        st.current = Some("HEAD".to_string());
        st.detached = true;
        return;
    }
    let (names, counts) = match line.find(" [") {
        Some(i) => (&line[..i], Some(line[i + 2..].trim_end_matches(']'))),
        None => (line, None),
    };
    match names.split_once("...") {
        Some((cur, track)) => {
            st.current = Some(cur.to_string());
            st.tracking = Some(track.to_string());
        }
        None => st.current = Some(names.to_string()),
    }
    if let Some(counts) = counts {
        for part in counts.split(',') {
            let part = part.trim();
            if let Some(n) = part.strip_prefix("ahead ") {
                st.ahead = n.trim().parse().unwrap_or(0);
            } else if let Some(n) = part.strip_prefix("behind ") {
                st.behind = n.trim().parse().unwrap_or(0);
            }
        }
    }
}

/// `git status` for `cwd`, parsed.
pub async fn git_status(cwd: &Path, cancel: Option<&CancellationToken>) -> anyhow::Result<GitStatus> {
    let (code, out, all) = git(cwd, &["status", "--porcelain", "-b", "-u", "--null"], cancel).await?;
    if code != Some(0) {
        anyhow::bail!("{}", all.trim());
    }
    Ok(parse_porcelain_status(&out))
}

pub struct GitStatusTool;

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }
    fn description(&self) -> &str {
        "Show git working-tree status."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
        if !is_repo(&ctx.cwd).await {
            return ToolResult::err("Not a git repository.");
        }
        let status = match git_status(&ctx.cwd, ctx.cancel.as_ref()).await {
            Ok(s) => s,
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        let mut lines = vec![
            format!("branch: {}", status.current.as_deref().unwrap_or("null")),
            format!("clean: {}", status.is_clean()),
        ];
        lines.extend(status.files.iter().map(|f| format!("  {}{} {}", f.index, f.working_dir, f.path)));
        let data = serde_json::to_value(&status).unwrap_or(Value::Null);
        ToolResult::ok(lines.join("\n")).with_data(data)
    }
}

pub struct GitDiffTool;

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }
    fn description(&self) -> &str {
        "Show the current git diff (optionally for one path)."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        if !is_repo(&ctx.cwd).await {
            return ToolResult::err("Not a git repository.");
        }
        let path = js_truthy(args.get("path")).then(|| js_string(args.get("path")));
        let mut git_args = vec!["diff"];
        if let Some(p) = path.as_deref() {
            git_args.extend(["--", p]);
        }
        let diff = match git(&ctx.cwd, &git_args, ctx.cancel.as_ref()).await {
            Ok((Some(0), out, _)) => out,
            Ok((_, _, all)) => return tool_error(ctx, self.name(), all.trim()),
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        let shown = if diff.is_empty() { "(no changes)".to_string() } else { diff.clone() };
        ToolResult::ok(shown).with_data(json!(diff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_porcelain_with_branch_and_rename() {
        let raw = "## main...origin/main [ahead 2, behind 1]\0 M src/a.rs\0?? new.txt\0R  b.rs\0old_b.rs\0A  c.rs\0";
        let st = parse_porcelain_status(raw);
        assert_eq!(st.current.as_deref(), Some("main"));
        assert_eq!(st.tracking.as_deref(), Some("origin/main"));
        assert_eq!((st.ahead, st.behind), (2, 1));
        assert_eq!(st.files.len(), 4);
        assert_eq!(st.files[0], GitFileStatus { path: "src/a.rs".into(), index: " ".into(), working_dir: "M".into(), from: None });
        assert_eq!(st.files[1].index, "?");
        assert_eq!(st.files[2].path, "b.rs");
        assert_eq!(st.files[2].from.as_deref(), Some("old_b.rs"));
        assert!(!st.is_clean());
    }

    #[test]
    fn parses_fresh_repo_branch() {
        let st = parse_porcelain_status("## No commits yet on main\0");
        assert_eq!(st.current.as_deref(), Some("main"));
        assert!(st.is_clean());
    }

    #[tokio::test]
    async fn status_and_diff_on_a_temp_repo() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        // Make it a repo first: the temp dir may itself sit inside a bigger work
        // tree (e.g. a home directory under git), which must not be scanned.
        // Skip silently if git is unavailable on this machine.
        let Ok(Ok((Some(0), _, _))) =
            tokio::time::timeout(Duration::from_secs(30), git(dir.path(), &["init", "-q"], None)).await
        else {
            return;
        };
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        assert!(is_repo(dir.path()).await);
        assert!(is_dirty(dir.path()).await);
        let r = GitStatusTool.run(json!({}), &ctx).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains("clean: false"));
        assert!(r.output.contains("  ?? a.txt"));
        let d = GitDiffTool.run(json!({}), &ctx).await;
        assert_eq!(d.output, "(no changes)");
    }
}
