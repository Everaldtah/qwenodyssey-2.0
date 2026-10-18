//! Filesystem tools. All paths are resolved relative to `ctx.cwd`; writes and
//! deletes are prevented from escaping the project root (or the agent's own
//! source root when `tools.allow_self_edit` put it in `ctx.self_root`).
//!
//! Also home to the public path helpers every other crate uses
//! ([`resolve_inside`], [`resolve_writable`], [`resolve_readable`], …) and the
//! chat layer's [`BlindOverwriteGuard`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::Regex;
use serde_json::{json, Value};

use crate::util::{
    glob_walk, home_dir, js_is_absolute, js_number, js_number_or, js_string, js_string_or_empty, js_truthy,
    lexical_resolve, read_text_lossy, tool_error, GlobOpts,
};

/// Glob patterns every listing/search skips (fast-glob `ignore`).
pub const IGNORE: &[&str] = &[
    "**/node_modules/**",
    "**/.git/**",
    "**/dist/**",
    "**/build/**",
    "**/.qwenodyssey/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/target/**",
    "**/*.lock",
];

/// Expand a leading "~" (or "~/", "~\") to the user's home directory.
pub fn expand_home(p: &str) -> String {
    if p == "~" {
        return home_dir().to_string_lossy().into_owned();
    }
    if let Some(rest) = p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")) {
        return lexical_resolve(&home_dir(), rest).to_string_lossy().into_owned();
    }
    p.to_string()
}

/// The last path segment of `cwd` once resolved (Node `path.basename(path.resolve(cwd))`).
fn cwd_basename(cwd: &Path) -> String {
    lexical_resolve(cwd, "").file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Small models often address files as "<project folder>/file" because the tool
/// output and PROJECT summary show the absolute cwd, e.g. `projA/calc.py` while
/// already inside projA. When no such sub-folder exists, drop the redundant
/// prefix so the call hits the file the model meant. Absolute and ~ paths, and
/// genuine sub-folders of the same name, are left untouched.
pub fn strip_redundant_cwd_prefix(cwd: &Path, p: &str) -> String {
    if p.is_empty() || js_is_absolute(p) || p.starts_with('~') {
        return p.to_string();
    }
    let base = cwd_basename(cwd);
    if base.is_empty() {
        return p.to_string();
    }
    let norm = p.replace('\\', "/");
    let norm = norm.strip_prefix("./").unwrap_or(&norm);
    let (head, rest) = match norm.find('/') {
        Some(i) => (&norm[..i], &norm[i + 1..]),
        None => (norm, ""),
    };
    if head.to_lowercase() != base.to_lowercase() {
        return p.to_string();
    }
    if lexical_resolve(cwd, head).exists() {
        return p.to_string(); // a real sub-folder with that name
    }
    if rest.is_empty() {
        ".".to_string()
    } else {
        rest.to_string()
    }
}

/// Shell form of [`strip_redundant_cwd_prefix`]: rewrite `projA/x`, `.\projA\x`,
/// `projA\` tokens inside a command when cwd is `.../projA` and no `projA`
/// sub-folder exists.
pub fn strip_redundant_cwd_prefix_in_command(cwd: &Path, cmd: &str) -> String {
    let base = cwd_basename(cwd);
    if base.is_empty() || base.chars().count() < 2 || cwd.join(&base).exists() {
        return cmd.to_string();
    }
    let pattern = format!(r#"(?i)(^|[\s"'=(])(?:\.[\\/])?{}[\\/]"#, regex::escape(&base));
    let Ok(re) = Regex::new(&pattern) else {
        return cmd.to_string();
    };
    static AFTER_STOP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"^[\s"'|;&)]"#).expect("static regex"));
    re.replace_all(cmd, |caps: &regex::Captures<'_>| {
        let pre = caps.get(1).map_or("", |m| m.as_str());
        let end = caps.get(0).map_or(0, |m| m.end());
        let after = &cmd[end..];
        // `cd projA/` with nothing after → stay put (`.`), otherwise drop the prefix.
        if after.is_empty() || AFTER_STOP.is_match(after) {
            format!("{pre}.")
        } else {
            pre.to_string()
        }
    })
    .into_owned()
}

/// Resolve a path for WRITES/DELETES: must stay inside the project root.
pub fn resolve_inside(cwd: &Path, p: &str) -> anyhow::Result<PathBuf> {
    let p = strip_redundant_cwd_prefix(cwd, p);
    let abs = lexical_resolve(cwd, &p);
    let root = lexical_resolve(cwd, "");
    if abs != root && !abs.starts_with(&root) {
        anyhow::bail!("Path escapes project root: {p}");
    }
    Ok(abs)
}

/// Resolve a path for READ-ONLY access. Relative paths resolve against the
/// project dir; an absolute path (e.g. "C:\Projects\Overstory") is allowed
/// anywhere on the machine, so the assistant can inspect a directory the user
/// names even when it's outside the launch dir. Use [`resolve_inside`] for writes.
pub fn resolve_readable(cwd: &Path, p: Option<&str>) -> PathBuf {
    let raw = match p {
        Some(s) if !s.trim().is_empty() => s,
        _ => ".",
    };
    lexical_resolve(cwd, &expand_home(&strip_redundant_cwd_prefix(cwd, raw)))
}

/// [`resolve_readable`] plus a HOME fallback. Small models frequently emit
/// home-relative paths in shapes that don't resolve where they meant: "~/.foo",
/// a leading-slash dotfile like "/.qwenodyssey/config.toml" (which on Windows
/// resolves to the drive root C:\.qwenodyssey, not home), or a bare ".foo" when
/// the cwd isn't home. When the primary resolution doesn't exist, retry the same
/// tail rooted at the user's home dir and use that if it exists. Only ever
/// redirects to an EXISTING file, so a correct absolute path is never overridden.
pub fn resolve_readable_smart(cwd: &Path, p: Option<&str>) -> PathBuf {
    let primary = resolve_readable(cwd, p);
    let Some(p) = p.filter(|s| !s.trim().is_empty()) else {
        return primary;
    };
    if primary.exists() {
        return primary;
    }
    let t = p.trim();
    let t = t.strip_prefix('~').unwrap_or(t);
    let tail = t.trim_start_matches(['\\', '/']);
    if !tail.is_empty() {
        let home_try = lexical_resolve(&home_dir(), tail);
        if home_try != primary && home_try.exists() {
            return home_try;
        }
    }
    primary
}

/// Resolve a path for WRITES, allowing either the project root OR the agent's
/// own source root (`self_root`) — so Qwenodyssey can modify itself when the
/// user opted in. Anything else is rejected to avoid scribbling across the
/// whole machine.
pub fn resolve_writable_in(cwd: &Path, self_root: Option<&Path>, p: &str) -> anyhow::Result<PathBuf> {
    let p = strip_redundant_cwd_prefix(cwd, p);
    let abs = lexical_resolve(cwd, &p);
    let mut roots = vec![lexical_resolve(cwd, "")];
    if let Some(sr) = self_root {
        roots.push(lexical_resolve(sr, ""));
    }
    for root in &roots {
        if abs == *root || abs.starts_with(root) {
            return Ok(abs);
        }
    }
    anyhow::bail!("Path escapes the allowed roots (project or agent source): {p}")
}

/// [`resolve_writable_in`] with the roots taken from a tool context.
pub fn resolve_writable(ctx: &ToolContext, p: &str) -> anyhow::Result<PathBuf> {
    resolve_writable_in(&ctx.cwd, ctx.self_root.as_deref(), p)
}

/* ───────────────────────── blind-overwrite guard ───────────────────────── */

/// Blind-overwrite guard (chat layer): `write_file` replaces a file's ENTIRE
/// contents. Small models routinely "add a function" by writing a file that
/// contains only the new function (a 9B did exactly that to calc.py). If the
/// target exists and the model has not read it this session, the write is
/// refused with a pointer to `apply_edit` / `read_file`, unless the model
/// explicitly passes `overwrite: true`.
///
/// The chat loop calls [`BlindOverwriteGuard::check`] before dispatching a
/// tool call and [`BlindOverwriteGuard::record`] after it ran.
#[derive(Clone, Debug, Default)]
pub struct BlindOverwriteGuard {
    seen: HashSet<String>,
}

impl BlindOverwriteGuard {
    pub fn new() -> Self {
        Self::default()
    }

    fn resolve(cwd: &Path, p: Option<&Value>) -> Option<PathBuf> {
        let p = p?.as_str()?;
        if p.trim().is_empty() {
            return None;
        }
        Some(lexical_resolve(cwd, &strip_redundant_cwd_prefix(cwd, &expand_home(p))))
    }

    /// Normalised, lower-cased key for a path argument (`None` when unusable).
    pub fn key(cwd: &Path, p: Option<&Value>) -> Option<String> {
        Self::resolve(cwd, p).map(|r| r.to_string_lossy().to_lowercase())
    }

    /// `Some(refusal)` when this call would blindly overwrite an unread file.
    pub fn check(&self, cwd: &Path, tool: &str, args: &Value) -> Option<String> {
        if tool != "write_file" || args.get("overwrite") == Some(&Value::Bool(true)) {
            return None;
        }
        let path = Self::resolve(cwd, args.get("path"))?;
        let key = path.to_string_lossy().to_lowercase();
        if path.is_file() && !self.seen.contains(&key) {
            let shown = js_string(args.get("path"));
            return Some(format!(
                "Refused: \"{shown}\" already exists and you have not read it this session, so replacing it \
                 would destroy its current contents. Either (a) call apply_edit to change part of it, or (b) call \
                 read_file first and then write_file with the COMPLETE new contents (existing code + your change). \
                 Only if you truly mean to discard the current file, call write_file again with \"overwrite\": true."
            ));
        }
        None
    }

    /// Remember files the model has now seen (successful reads/edits/writes).
    pub fn record(&mut self, cwd: &Path, tool: &str, args: &Value, ok: bool) {
        const SEEN_BY: &[&str] = &["read_file", "read_symbol", "outline_file", "apply_edit", "write_file", "edit_file"];
        if !ok || !SEEN_BY.contains(&tool) {
            return;
        }
        if let Some(key) = Self::key(cwd, args.get("path")) {
            self.seen.insert(key);
        }
    }

    /// Has this path been seen this session?
    pub fn has_seen(&self, cwd: &Path, p: &str) -> bool {
        Self::key(cwd, Some(&Value::String(p.to_string()))).is_some_and(|k| self.seen.contains(&k))
    }
}

/* ───────────────────────── tools ───────────────────────── */

/// Auto-paginate files longer than this even without offset/limit.
const AUTO_PAGE: usize = 800;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "Read the contents of a file, optionally a line range (offset/limit) for large files."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let shown = js_string(args.get("path"));
        let abs = resolve_readable_smart(&ctx.cwd, Some(&shown));
        if !abs.exists() {
            return ToolResult::err(format!("Not found: {shown}"));
        }
        if abs.is_dir() {
            return ToolResult::err(format!("{shown} is a directory — use tree or list_files for it."));
        }
        let content = match read_text_lossy(&abs) {
            Ok(c) => c,
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        ctx.log(json!({ "tool": "read_file", "path": args.get("path"), "bytes": content.encode_utf16().count() }));

        // Pagination: when offset/limit are given (or the file is very large) return
        // a window of lines plus a header telling the model how to fetch the next
        // page, instead of dumping a huge file into the context.
        let has_window = args.get("offset").is_some() || args.get("limit").is_some();
        let lines: Vec<&str> = content.split('\n').collect();
        let total = lines.len();
        if has_window || total > AUTO_PAGE {
            let start = js_number_or(args.get("offset"), 1.0).max(1.0).trunc() as usize; // 1-based
            let count = js_number_or(args.get("limit"), AUTO_PAGE as f64).max(1.0).trunc() as usize;
            let end = total.min(start.saturating_add(count) - 1);
            let slice = if start <= end { lines[start - 1..end].join("\n") } else { String::new() };
            let more = if end < total {
                format!("\n…({} more lines — call read_file again with offset={})", total - end, end + 1)
            } else {
                String::new()
            };
            let header = format!("{shown} — lines {start}-{end} of {total}\n");
            return ToolResult::ok(format!("{header}{slice}{more}")).with_data(Value::String(content));
        }
        ToolResult::ok(content.clone()).with_data(Value::String(content))
    }
}

fn write_text(abs: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(abs, content.as_bytes())
}

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> &str {
        "Overwrite (or create) a file with the given content."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let shown = js_string(args.get("path"));
        let abs = match resolve_writable(ctx, &shown) {
            Ok(a) => a,
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        if let Err(e) = write_text(&abs, &js_string_or_empty(args.get("content"))) {
            return tool_error(ctx, self.name(), e);
        }
        ctx.log(json!({ "tool": "write_file", "path": args.get("path") }));
        ToolResult::ok(format!("Wrote {shown}"))
    }
}

pub struct CreateFileTool;

#[async_trait]
impl Tool for CreateFileTool {
    fn name(&self) -> &str {
        "create_file"
    }
    fn description(&self) -> &str {
        "Create a new file. Fails if it already exists."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let shown = js_string(args.get("path"));
        let abs = match resolve_writable(ctx, &shown) {
            Ok(a) => a,
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        if abs.exists() {
            return ToolResult::err(format!("Already exists: {shown}"));
        }
        if let Err(e) = write_text(&abs, &js_string_or_empty(args.get("content"))) {
            return tool_error(ctx, self.name(), e);
        }
        ctx.log(json!({ "tool": "create_file", "path": args.get("path") }));
        ToolResult::ok(format!("Created {shown}"))
    }
}

pub struct DeleteFileTool;

#[async_trait]
impl Tool for DeleteFileTool {
    fn name(&self) -> &str {
        "delete_file"
    }
    fn description(&self) -> &str {
        "Delete a file."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let shown = js_string(args.get("path"));
        let abs = match resolve_writable(ctx, &shown) {
            Ok(a) => a,
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        if !abs.exists() {
            return ToolResult::err(format!("Not found: {shown}"));
        }
        if let Err(e) = std::fs::remove_file(&abs) {
            return tool_error(ctx, self.name(), e);
        }
        ctx.log(json!({ "tool": "delete_file", "path": args.get("path") }));
        ToolResult::ok(format!("Deleted {shown}"))
    }
}

/// `args.path ? String(args.path) : "."` as the TS listing tools do.
fn dir_arg(args: &Value) -> String {
    if js_truthy(args.get("path")) {
        js_string(args.get("path"))
    } else {
        ".".to_string()
    }
}

pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }
    fn description(&self) -> &str {
        "List files matching an optional glob (default: all tracked-ish files)."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let base = resolve_readable_smart(&ctx.cwd, Some(&dir_arg(&args)));
        if !base.exists() {
            return ToolResult::err(format!("Directory not found: {}", base.display()));
        }
        let pattern = if js_truthy(args.get("pattern")) { js_string(args.get("pattern")) } else { "**/*".to_string() };
        let walk_base = base.clone();
        let files = match tokio::task::spawn_blocking(move || {
            glob_walk(&walk_base, &[pattern.as_str()], IGNORE, &GlobOpts::default())
        })
        .await
        {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => return tool_error(ctx, self.name(), e),
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        const CAP: usize = 500;
        let shown: Vec<&str> = files.iter().take(CAP).map(String::as_str).collect();
        let note = if files.len() > CAP {
            format!("\n…({} more; narrow with a pattern or subdir path)", files.len() - CAP)
        } else {
            String::new()
        };
        let header = format!("{} — {} file(s)\n", base.display(), files.len());
        ToolResult::ok(format!("{header}{}{note}", shown.join("\n"))).with_data(json!(files))
    }
}

pub struct TreeTool;

#[async_trait]
impl Tool for TreeTool {
    fn name(&self) -> &str {
        "tree"
    }
    fn description(&self) -> &str {
        "Show a compact directory tree (depth-limited)."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let base = resolve_readable_smart(&ctx.cwd, Some(&dir_arg(&args)));
        if !base.exists() {
            return ToolResult::err(format!("Directory not found: {}", base.display()));
        }
        if !base.is_dir() {
            return ToolResult::err(format!("{} is a file, not a directory — use read_file.", base.display()));
        }
        // Number(args.depth ?? 2)
        let depth_raw = match args.get("depth") {
            None | Some(Value::Null) => 2.0,
            v => js_number(v),
        };
        let max_depth = if depth_raw.is_finite() && depth_raw >= 1.0 { Some(depth_raw.trunc() as usize) } else { None };
        let depth_label = if depth_raw.is_nan() {
            "NaN".to_string()
        } else {
            crate::util::js_string(Some(&json!(depth_raw)))
        };
        let walk_base = base.clone();
        let mut files = match tokio::task::spawn_blocking(move || {
            glob_walk(
                &walk_base,
                &["**/*"],
                IGNORE,
                &GlobOpts { only_files: false, mark_dirs: true, max_depth, ..Default::default() },
            )
        })
        .await
        {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => return tool_error(ctx, self.name(), e),
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        files.sort();
        const CAP: usize = 400;
        let shown: Vec<&str> = files.iter().take(CAP).map(String::as_str).collect();
        let note = if files.len() > CAP {
            format!("\n…({} more entries; tree a subdirectory or raise depth to see more)", files.len() - CAP)
        } else {
            String::new()
        };
        let header = format!("{}  (depth {depth_label}, {} entries)\n", base.display(), files.len());
        ToolResult::ok(format!("{header}{}{note}", shown.join("\n"))).with_data(json!(files))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(dir: &Path) -> ToolContext {
        let mut c = ToolContext::new(dir);
        c.auto_confirm = true;
        c.confirm_destructive = false;
        c
    }

    // pathResolve.test.ts
    #[test]
    fn expand_home_bare_and_slash() {
        assert_eq!(expand_home("~"), home_dir().to_string_lossy());
        assert_eq!(
            PathBuf::from(expand_home("~/.qwenodyssey/config.toml")),
            home_dir().join(".qwenodyssey").join("config.toml")
        );
    }

    #[test]
    fn expand_home_leaves_non_tilde_paths() {
        assert_eq!(expand_home("src/index.ts"), "src/index.ts");
        assert_eq!(expand_home("/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn resolve_readable_tilde_goes_home() {
        let r = resolve_readable(Path::new("C:/some/where"), Some("~/.qwenodyssey"));
        assert_eq!(r, lexical_resolve(&home_dir(), ".qwenodyssey"));
    }

    struct HomeMarker(PathBuf, String);
    impl HomeMarker {
        fn new(tag: &str) -> Self {
            let name = format!(".qod-test-{}-{}-{tag}", std::process::id(), rand::random::<u32>());
            let abs = home_dir().join(&name);
            std::fs::write(&abs, "x").unwrap();
            Self(abs, name)
        }
    }
    impl Drop for HomeMarker {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn smart_never_overrides_existing_absolute_path() {
        let me = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let s = me.to_string_lossy().into_owned();
        assert_eq!(resolve_readable_smart(Path::new("C:/anything"), Some(&s)), normalize(&me));
    }

    fn normalize(p: &Path) -> PathBuf {
        crate::util::normalize_path(p)
    }

    #[test]
    fn smart_redirects_leading_slash_dotfile_to_home() {
        let m = HomeMarker::new("slash");
        let r = resolve_readable_smart(Path::new("C:/some/other/cwd"), Some(&format!("/{}", m.1)));
        assert_eq!(r, m.0);
    }

    #[test]
    fn smart_redirects_bare_home_relative_path() {
        let m = HomeMarker::new("bare");
        let r = resolve_readable_smart(&std::env::temp_dir().join("not-home"), Some(&m.1));
        assert_eq!(r, m.0);
    }

    #[test]
    fn smart_returns_primary_when_no_home_match() {
        let p = "/definitely/not/here-qod-xyz";
        let r = resolve_readable_smart(Path::new("C:/some/cwd"), Some(p));
        assert_eq!(r, lexical_resolve(Path::new("C:/some/cwd"), p));
    }

    // bugfixes2026_09: redundant '<project folder>/' prefix repair
    fn proj() -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let cwd = root.path().join("projA");
        std::fs::create_dir(&cwd).unwrap();
        std::fs::write(cwd.join("calc.py"), "x = 1\n").unwrap();
        (root, cwd)
    }

    #[test]
    fn strips_cwd_folder_name_when_no_subfolder() {
        let (_root, cwd) = proj();
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "projA/calc.py"), "calc.py");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "./projA\\calc.py"), "calc.py");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "projA/"), ".");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "projA"), ".");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "PROJA/calc.py"), "calc.py");
    }

    #[test]
    fn leaves_other_absolute_and_tilde_paths() {
        let (_root, cwd) = proj();
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "calc.py"), "calc.py");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "src/projA/x.py"), "src/projA/x.py");
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "~/projA/x"), "~/projA/x");
        let abs = cwd.join("projA").join("x").to_string_lossy().into_owned();
        assert_eq!(strip_redundant_cwd_prefix(&cwd, &abs), abs);
    }

    #[test]
    fn keeps_prefix_when_real_subfolder_exists() {
        let (_root, cwd) = proj();
        std::fs::create_dir(cwd.join("projA")).unwrap();
        assert_eq!(strip_redundant_cwd_prefix(&cwd, "projA/calc.py"), "projA/calc.py");
    }

    #[test]
    fn resolve_readable_lands_on_meant_file() {
        let (_root, cwd) = proj();
        assert_eq!(resolve_readable(&cwd, Some("projA/calc.py")), normalize(&cwd.join("calc.py")));
    }

    #[test]
    fn rewrites_shell_commands_the_same_way() {
        let (_root, cwd) = proj();
        let f = |c: &str| strip_redundant_cwd_prefix_in_command(&cwd, c);
        assert_eq!(f("cat projA/calc.py"), "cat calc.py");
        assert_eq!(f("tree projA/"), "tree .");
        assert_eq!(f("tree projA/ | findstr calc"), "tree . | findstr calc");
        assert_eq!(f("python \"projA\\calc.py\""), "python \"calc.py\"");
        assert_eq!(f("cat .\\projA\\calc.py"), "cat calc.py");
        assert_eq!(f("cat other/projA/x"), "cat other/projA/x");
        assert_eq!(f("echo projA"), "echo projA");
    }

    #[test]
    fn resolve_inside_and_writable_enforce_roots() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_inside(dir.path(), "a/b.txt").is_ok());
        assert!(resolve_inside(dir.path(), "../escape.txt").is_err());
        let me = Path::new(env!("CARGO_MANIFEST_DIR"));
        let outside = me.join("x.txt").to_string_lossy().into_owned();
        assert!(resolve_writable_in(dir.path(), None, &outside).is_err());
        assert!(resolve_writable_in(dir.path(), Some(me), &outside).is_ok());
    }

    // harnessUpgrades: read_file pagination
    #[tokio::test]
    async fn read_file_paginates_with_offset_limit() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("big.txt");
        let body: Vec<String> = (1..=100).map(|i| format!("line {i}")).collect();
        std::fs::write(&file, body.join("\n")).unwrap();
        let r = ReadFileTool
            .run(json!({ "path": file.to_string_lossy(), "offset": 10, "limit": 5 }), &ctx(dir.path()))
            .await;
        assert!(r.ok);
        assert!(r.output.contains("lines 10-14 of 100"));
        assert!(r.output.contains("line 10"));
        assert!(r.output.contains("line 14"));
        assert!(!r.output.contains("line 15"));
        assert!(r.output.contains("offset=15"));
    }

    #[tokio::test]
    async fn read_file_small_unpaginated() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("small.txt"), "a\nb\nc").unwrap();
        let r = ReadFileTool.run(json!({ "path": "small.txt" }), &ctx(dir.path())).await;
        assert_eq!(r.output, "a\nb\nc");
    }

    #[tokio::test]
    async fn file_tools_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let c = ctx(dir.path());
        assert!(WriteFileTool.run(json!({ "path": "src/a.txt", "content": "hi" }), &c).await.ok);
        assert!(!CreateFileTool.run(json!({ "path": "src/a.txt", "content": "x" }), &c).await.ok);
        assert!(CreateFileTool.run(json!({ "path": "b.txt" }), &c).await.ok);
        let w = WriteFileTool.run(json!({ "path": "../out.txt", "content": "x" }), &c).await;
        assert!(!w.ok);
        assert!(w.output.contains("escapes the allowed roots"));
        let l = ListFilesTool.run(json!({}), &c).await;
        assert!(l.output.contains("2 file(s)"));
        assert!(l.output.contains("src/a.txt"));
        let t = TreeTool.run(json!({}), &c).await;
        assert!(t.output.contains("src/"));
        assert!(t.output.contains("(depth 2, 3 entries)"));
        assert!(DeleteFileTool.run(json!({ "path": "b.txt" }), &c).await.ok);
        assert!(!DeleteFileTool.run(json!({ "path": "b.txt" }), &c).await.ok);
        let dirread = ReadFileTool.run(json!({ "path": "src" }), &c).await;
        assert!(dirread.output.contains("is a directory"));
    }

    /// Machine-dependent (walks the real home dir): on Windows the profile holds
    /// deny-listed junctions such as "Application Data" that must be skipped,
    /// not followed or reported as errors. Run with `--ignored`.
    #[tokio::test]
    #[ignore]
    async fn tree_on_home_survives_junctions() {
        let home = home_dir();
        let r = TreeTool.run(json!({ "path": home.to_string_lossy(), "depth": 2 }), &ctx(&home)).await;
        assert!(r.ok, "{}", r.output);
        assert!(!r.output.contains("Application Data/"), "junction must not be listed as a directory");
        assert!(!r.output.contains("Application Data/Microsoft"), "junction must not be descended");
    }

    #[test]
    fn blind_overwrite_guard() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("calc.py"), "def add(): pass\n").unwrap();
        let mut g = BlindOverwriteGuard::new();
        let args = json!({ "path": "calc.py", "content": "def mul(): pass\n" });
        assert!(g.check(dir.path(), "write_file", &args).unwrap().starts_with("Refused"));
        assert!(g.check(dir.path(), "write_file", &json!({ "path": "new.py", "content": "" })).is_none());
        assert!(g
            .check(dir.path(), "write_file", &json!({ "path": "calc.py", "content": "", "overwrite": true }))
            .is_none());
        g.record(dir.path(), "read_file", &json!({ "path": "calc.py" }), true);
        assert!(g.check(dir.path(), "write_file", &args).is_none());
        assert!(g.has_seen(dir.path(), "CALC.py"));
    }
}
