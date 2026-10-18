//! Small shared helpers: JS-style argument coercion (the TS tools lean on
//! `String(x)` / `Number(x)` semantics, and small models send every JSON type
//! imaginable), Node-style lexical path resolution, a fast-glob work-alike
//! directory walker, and child-process helpers (no console window, stdin null,
//! tree kill, cancellation).

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf, MAIN_SEPARATOR_STR};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use globset::{Glob, GlobBuilder, GlobSet, GlobSetBuilder};
use qo_core::ToolResult;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

/* ───────────────────────── JS-style coercion ───────────────────────── */

/// JS `String(v)` for a JSON argument (`None` = `undefined`).
pub(crate) fn js_string(v: Option<&Value>) -> String {
    match v {
        None => "undefined".to_string(),
        Some(Value::Null) => "null".to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => js_number_to_string(n),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|x| match x {
                Value::Null => String::new(),
                other => js_string(Some(other)),
            })
            .collect::<Vec<_>>()
            .join(","),
        Some(Value::Object(_)) => "[object Object]".to_string(),
    }
}

/// JS `String(v ?? "")`: `undefined`/`null` become the empty string.
pub(crate) fn js_string_or_empty(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        other => js_string(other),
    }
}

fn js_number_to_string(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(f64::NAN);
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e21 {
        format!("{f:.0}")
    } else {
        f.to_string()
    }
}

/// JS truthiness of a JSON argument.
pub(crate) fn js_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(_)) | Some(Value::Object(_)) => true,
    }
}

/// JS `Number(v)` for a JSON argument (`None` = `undefined` → NaN).
pub(crate) fn js_number(v: Option<&Value>) -> f64 {
    match v {
        None => f64::NAN,
        Some(Value::Null) => 0.0,
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => parse_js_number(s),
        Some(Value::Array(items)) => match items.len() {
            0 => 0.0,
            1 => parse_js_number(&js_string(items.first())),
            _ => f64::NAN,
        },
        Some(Value::Object(_)) => f64::NAN,
    }
}

/// JS `Number(string)`: trims, "" → 0, accepts hex/octal/binary prefixes and
/// `Infinity`, rejects anything else Rust's float parser would allow (`inf`, `nan`).
pub(crate) fn parse_js_number(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    let lower = t.to_ascii_lowercase();
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            return u64::from_str_radix(rest, radix).map(|v| v as f64).unwrap_or(f64::NAN);
        }
    }
    if lower.contains("inf") || lower.contains("nan") || lower.contains('_') {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// JS `Number(v) || fallback` (NaN and 0 fall back).
pub(crate) fn js_number_or(v: Option<&Value>, fallback: f64) -> f64 {
    let n = js_number(v);
    if n.is_nan() || n == 0.0 {
        fallback
    } else {
        n
    }
}

/// `s.slice(0, n)` by characters (JS slices UTF-16 units; chars are the closest
/// safe Rust equivalent and never split a code point).
pub(crate) fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Standard error result for a tool whose body failed — the TS registry caught
/// the exception and returned exactly this text.
pub(crate) fn tool_error(ctx: &qo_core::ToolContext, name: &str, err: impl std::fmt::Display) -> ToolResult {
    let msg = err.to_string();
    ctx.log(serde_json::json!({ "tool": name, "error": msg }));
    ToolResult::err(format!("Tool \"{name}\" error: {msg}"))
}

/* ───────────────────────── Node-style paths ───────────────────────── */

/// Node's `path.isAbsolute` for the current platform. On Windows a path with a
/// root but no drive ("/foo", "\\foo") counts as absolute, unlike Rust's
/// `Path::is_absolute`.
pub(crate) fn js_is_absolute(p: &str) -> bool {
    if cfg!(windows) {
        let b = p.as_bytes();
        if b.first().is_some_and(|c| *c == b'/' || *c == b'\\') {
            return true;
        }
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'/' || b[2] == b'\\')
    } else {
        p.starts_with('/')
    }
}

/// Lexically normalise a path: drop `.`, fold `..`, unify separators, strip a
/// trailing separator. Never touches the filesystem (like `path.normalize`).
pub(crate) fn normalize_path(p: &Path) -> PathBuf {
    let mut prefix: Option<OsString> = None;
    let mut has_root = false;
    let mut parts: Vec<OsString> = Vec::new();
    for comp in p.components() {
        match comp {
            Component::Prefix(pre) => prefix = Some(pre.as_os_str().to_owned()),
            Component::RootDir => has_root = true,
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.last().is_some_and(|l| l != "..") {
                    parts.pop();
                } else if !has_root {
                    parts.push(OsString::from(".."));
                }
            }
            Component::Normal(s) => parts.push(s.to_owned()),
        }
    }
    let mut out = OsString::new();
    if let Some(pre) = prefix {
        out.push(pre);
    }
    if has_root {
        out.push(MAIN_SEPARATOR_STR);
    }
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push(MAIN_SEPARATOR_STR);
        }
        out.push(part);
    }
    if out.is_empty() {
        out.push(".");
    }
    PathBuf::from(out)
}

/// Node's `path.resolve(base, p)`: an absolute `p` wins (on Windows a rooted
/// "/foo" keeps `base`'s drive), relative parts are joined and normalised, and
/// a relative result is anchored at the process cwd.
pub(crate) fn lexical_resolve(base: &Path, p: &str) -> PathBuf {
    let cwd = || std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let base_abs = if base.is_absolute() { base.to_path_buf() } else { cwd().join(base) };
    let joined = base_abs.join(p);
    let joined = if joined.is_absolute() { joined } else { cwd().join(joined) };
    normalize_path(&joined)
}

/// The user's home directory (Node's `os.homedir()`).
pub(crate) fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Relative path of `entry` under `base` with `/` separators (fast-glob style).
fn rel_slash(base: &Path, entry: &Path) -> Option<String> {
    let rel = entry.strip_prefix(base).ok()?;
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("/"))
    }
}

/* ───────────────────────── fast-glob work-alike ───────────────────────── */

/// Options mirroring the fast-glob options the TS tools pass.
#[derive(Clone, Debug)]
pub(crate) struct GlobOpts {
    /// Match entries whose name starts with "." (fast-glob `dot`).
    pub dot: bool,
    pub only_files: bool,
    /// Append "/" to directories (fast-glob `markDirectories`).
    pub mark_dirs: bool,
    /// fast-glob `deep`: max path segments below the base.
    pub max_depth: Option<usize>,
}

impl Default for GlobOpts {
    fn default() -> Self {
        Self { dot: false, only_files: true, mark_dirs: false, max_depth: None }
    }
}

fn build_globset(patterns: &[&str]) -> Result<GlobSet, globset::Error> {
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        b.add(compile_glob(p)?);
    }
    b.build()
}

fn compile_glob(p: &str) -> Result<Glob, globset::Error> {
    GlobBuilder::new(&clean_pattern(p)).literal_separator(true).backslash_escape(false).build()
}

/// Normalise a user glob: forward slashes (models on Windows write `src\**`),
/// no leading "./".
fn clean_pattern(p: &str) -> String {
    let mut s = p.replace('\\', "/");
    while let Some(rest) = s.strip_prefix("./") {
        s = rest.to_string();
    }
    s
}

/// Dot-prefixed path segments a pattern names literally (".github/**" may see
/// inside .github even with `dot: false`, as in fast-glob).
fn literal_dot_segments(patterns: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for p in patterns {
        for seg in clean_pattern(p).split('/') {
            if seg.starts_with('.') && seg != "." && seg != ".." && !seg.contains(['*', '?', '[', '{']) {
                out.push(seg.to_string());
            }
        }
    }
    out
}

/// Walk `base` and return the relative (`/`-separated) paths that match any of
/// `patterns` and none of `ignore`, like `fg(patterns, { cwd: base, ignore, ... })`
/// with `suppressErrors: true` and `followSymbolicLinks: false`. Unreadable
/// directories (Windows junctions such as "Application Data", protected
/// %TEMP% folders) are skipped silently instead of aborting the walk; junctions
/// and symlinks are never descended into.
pub(crate) fn glob_walk(base: &Path, patterns: &[&str], ignore: &[&str], opts: &GlobOpts) -> anyhow::Result<Vec<String>> {
    let include = build_globset(patterns)?;
    let ignore_set = build_globset(ignore)?;
    let dot_ok = literal_dot_segments(patterns);
    let mut walker = walkdir::WalkDir::new(base).follow_links(false).min_depth(1).sort_by_file_name();
    if let Some(d) = opts.max_depth {
        walker = walker.max_depth(d.max(1));
    }
    let mut out = Vec::new();
    let mut it = walker.into_iter();
    while let Some(entry) = it.next() {
        let Ok(entry) = entry else { continue };
        let is_dir = entry.file_type().is_dir();
        let name = entry.file_name().to_string_lossy();
        if !opts.dot && name.starts_with('.') && !dot_ok.iter().any(|d| d.as_str() == name) {
            if is_dir {
                it.skip_current_dir();
            }
            continue;
        }
        let Some(rel) = rel_slash(base, entry.path()) else { continue };
        if is_dir {
            // `**/node_modules/**` style ignores prune the directory itself too.
            if ignore_set.is_match(&rel) || ignore_set.is_match(format!("{rel}/_")) {
                it.skip_current_dir();
                continue;
            }
        } else if ignore_set.is_match(&rel) {
            continue;
        }
        if opts.only_files && !entry.file_type().is_file() {
            continue;
        }
        if include.is_match(&rel) {
            if is_dir && opts.mark_dirs {
                out.push(format!("{rel}/"));
            } else {
                out.push(rel);
            }
        }
    }
    Ok(out)
}

/// Read a file as text the way Node's `readFileSync(p, "utf-8")` does
/// (invalid UTF-8 becomes U+FFFD instead of failing).
pub(crate) fn read_text_lossy(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    })
}

/* ───────────────────────── child processes ───────────────────────── */

/// `CREATE_NO_WINDOW`: execa's default `windowsHide: true` — a child must never
/// flash a console window on the user's desktop.
#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A `tokio::process::Command` with the house rules applied: stdin null (no
/// child may steal the keyboard), no console window, killed if dropped.
pub(crate) fn command(program: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(program);
    cmd.stdin(Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// Kill a child and, on Windows, its whole process tree (PowerShell's own
/// children otherwise survive and keep the output pipes open).
pub(crate) async fn kill_tree(child: &mut tokio::process::Child) {
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        let mut tk = command("taskkill");
        tk.args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null());
        if let Ok(mut c) = tk.spawn() {
            let _ = tokio::time::timeout(Duration::from_secs(5), c.wait()).await;
        }
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

/// How a captured child run ended.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Captured {
    /// stdout and stderr interleaved in arrival order (execa `all: true`).
    pub all: String,
    pub stdout: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub interrupted: bool,
}

/// Run a prepared command to completion, capturing stdout+stderr interleaved,
/// with an optional timeout and cancellation token. On timeout/cancel the
/// process tree is killed and the output captured so far is returned.
pub(crate) async fn run_captured(
    mut cmd: tokio::process::Command,
    timeout: Option<Duration>,
    cancel: Option<&CancellationToken>,
) -> std::io::Result<Captured> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let all = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_only = Arc::new(Mutex::new(Vec::<u8>::new()));

    let mut readers = Vec::new();
    if let Some(mut so) = child.stdout.take() {
        let all = Arc::clone(&all);
        let out_only = Arc::clone(&out_only);
        readers.push(tokio::spawn(async move {
            let mut buf = [0u8; 8192];
            while let Ok(n) = so.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                if let Ok(mut a) = all.lock() {
                    a.extend_from_slice(&buf[..n]);
                }
                if let Ok(mut o) = out_only.lock() {
                    o.extend_from_slice(&buf[..n]);
                }
            }
        }));
    }
    if let Some(mut se) = child.stderr.take() {
        let all = Arc::clone(&all);
        readers.push(tokio::spawn(async move {
            let mut buf = [0u8; 8192];
            while let Ok(n) = se.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                if let Ok(mut a) = all.lock() {
                    a.extend_from_slice(&buf[..n]);
                }
            }
        }));
    }

    let never = CancellationToken::new();
    let cancel = cancel.unwrap_or(&never);
    let sleep = async {
        match timeout {
            Some(t) => tokio::time::sleep(t).await,
            None => std::future::pending::<()>().await,
        }
    };

    let mut result = Captured::default();
    tokio::select! {
        status = child.wait() => {
            result.exit_code = status.ok().and_then(|s| s.code());
        }
        _ = sleep => {
            result.timed_out = true;
            kill_tree(&mut child).await;
        }
        _ = cancel.cancelled() => {
            result.interrupted = true;
            kill_tree(&mut child).await;
        }
    }

    // Drain what the pipes still hold, but never hang on a grandchild that
    // inherited them and keeps them open.
    for r in readers {
        let _ = tokio::time::timeout(Duration::from_secs(2), r).await;
    }
    let take = |m: &Arc<Mutex<Vec<u8>>>| -> String {
        m.lock().map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
    };
    result.all = take(&all);
    result.stdout = take(&out_only);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn js_number_semantics() {
        assert_eq!(js_number(Some(&json!("5"))), 5.0);
        assert_eq!(js_number(Some(&json!(" 12 "))), 12.0);
        assert_eq!(js_number(Some(&json!(""))), 0.0);
        assert!(js_number(Some(&json!("abc"))).is_nan());
        assert!(js_number(Some(&json!("inf"))).is_nan());
        assert_eq!(js_number(Some(&json!("0x10"))), 16.0);
        assert!(js_number(None).is_nan());
        assert_eq!(js_number(Some(&json!(null))), 0.0);
        assert_eq!(js_number_or(Some(&json!("x")), 7.0), 7.0);
    }

    #[test]
    fn js_string_semantics() {
        assert_eq!(js_string(None), "undefined");
        assert_eq!(js_string(Some(&json!(3))), "3");
        assert_eq!(js_string(Some(&json!(2.5))), "2.5");
        assert_eq!(js_string(Some(&json!([1, "a", null]))), "1,a,");
        assert_eq!(js_string(Some(&json!({"a": 1}))), "[object Object]");
        assert_eq!(js_string_or_empty(Some(&json!(null))), "");
    }

    #[test]
    fn normalizes_paths() {
        let p = normalize_path(Path::new("a/./b/../c/"));
        assert_eq!(p, PathBuf::from("a").join("c"));
        let base = std::env::temp_dir();
        assert_eq!(lexical_resolve(&base, "x/../y"), normalize_path(&base.join("y")));
        assert_eq!(lexical_resolve(&base, ""), normalize_path(&base));
    }

    #[cfg(windows)]
    #[test]
    fn rooted_path_keeps_drive_on_windows() {
        assert_eq!(lexical_resolve(Path::new("C:/some/where"), "/foo"), PathBuf::from(r"C:\foo"));
        assert!(js_is_absolute("/foo"));
        assert!(js_is_absolute(r"C:\x"));
        assert!(!js_is_absolute("C:x"));
    }

    #[test]
    fn glob_walk_respects_dot_ignore_and_depth() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("src/deep/er")).unwrap();
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("src/a.ts"), "").unwrap();
        std::fs::write(root.join("src/deep/er/b.ts"), "").unwrap();
        std::fs::write(root.join("node_modules/pkg/i.js"), "").unwrap();
        std::fs::write(root.join(".git/HEAD"), "").unwrap();
        std::fs::write(root.join("Cargo.lock"), "").unwrap();
        std::fs::write(root.join("readme.md"), "").unwrap();

        let all = glob_walk(root, &["**/*"], &["**/node_modules/**", "**/.git/**", "**/*.lock"], &GlobOpts::default()).unwrap();
        assert_eq!(all, vec!["readme.md", "src/a.ts", "src/deep/er/b.ts"]);

        let ts = glob_walk(root, &["src/**/*.ts"], &[], &GlobOpts::default()).unwrap();
        assert_eq!(ts, vec!["src/a.ts", "src/deep/er/b.ts"]);

        let shallow = glob_walk(
            root,
            &["**/*"],
            &["**/node_modules/**"],
            &GlobOpts { only_files: false, mark_dirs: true, max_depth: Some(2), ..Default::default() },
        )
        .unwrap();
        assert!(shallow.contains(&"src/".to_string()));
        assert!(shallow.contains(&"src/deep/".to_string()));
        assert!(!shallow.iter().any(|e| e.starts_with("src/deep/er")));
        assert!(!shallow.iter().any(|e| e.starts_with("node_modules")));
    }

    #[tokio::test]
    async fn run_captured_times_out_and_returns() {
        let mut cmd = if cfg!(windows) {
            let mut c = command("powershell.exe");
            c.args(["-NoProfile", "-NonInteractive", "-Command", "Write-Output start; Start-Sleep -Seconds 20"]);
            c
        } else {
            let mut c = command("sh");
            c.args(["-c", "echo start; sleep 20"]);
            c
        };
        cmd.current_dir(std::env::temp_dir());
        let started = std::time::Instant::now();
        let r = run_captured(cmd, Some(Duration::from_millis(3000)), None).await.unwrap();
        assert!(r.timed_out);
        assert!(started.elapsed() < Duration::from_secs(15));
    }
}
