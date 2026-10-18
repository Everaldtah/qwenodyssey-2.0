//! Detect language, framework, package manager, and the test/lint/build
//! commands for the project at `cwd` (port of `src/core/repoScanner.ts`).
//!
//! The file walk is EPERM/junction-safe: symlinks and Windows junctions
//! (e.g. `Application Data`) are never followed, unreadable directories are
//! skipped instead of aborting, dotfiles are skipped (fast-glob `dot: false`),
//! and the walk is depth-bounded and hard-capped so a huge tree (a git-init'ed
//! home directory) cannot stall the run.

use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde_json::Value;
use walkdir::WalkDir;

use crate::process::run_captured;
use qo_core::RepoInfo;

/// Directory names never descended into: qo-tools `IGNORE` (fileTools.ts).
pub const IGNORE_DIRS: &[&str] =
    &["node_modules", ".git", "dist", "build", ".qwenodyssey", "__pycache__", ".venv", "venv", "target"];

/// Extra directories skipped by the project scan: heavy/system dirs found
/// under a home directory (repoScanner.ts `SCAN_IGNORE`).
pub const SCAN_IGNORE_DIRS: &[&str] = &[
    "AppData",
    "Application Data",
    "Downloads",
    "OneDrive",
    ".cache",
    ".lmstudio",
    ".ollama",
    ".cargo",
    ".rustup",
    ".nuget",
    "Library",
];

/// Hard cap on files collected by the deep scan.
pub const FILE_CAP: usize = 4000;
/// fast-glob `deep: 6` for the project scan.
const SCAN_DEPTH: usize = 6;

fn exists(cwd: &Path, f: &str) -> bool {
    cwd.join(f).exists()
}

fn read_pkg(cwd: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(cwd.join("package.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

fn safe_read(cwd: &Path, f: &str) -> String {
    std::fs::read(cwd.join(f)).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

/// JS truthiness for a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Walk files under `root` like fast-glob `**/*` with the ignore list,
/// `onlyFiles`, `followSymbolicLinks: false`, `suppressErrors: true` and
/// `dot: false`. Paths are relative with forward slashes; `keep` filters
/// files; at most `cap` are returned.
pub fn walk_files(
    root: &Path,
    max_depth: usize,
    extra_ignore: &[&str],
    cap: usize,
    mut keep: impl FnMut(&str) -> bool,
) -> Vec<String> {
    let mut out = Vec::new();
    if cap == 0 {
        return out;
    }
    let walker = WalkDir::new(root)
        .min_depth(1)
        .max_depth(max_depth)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            if e.depth() == 0 {
                return true;
            }
            let name = e.file_name().to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
            if e.file_type().is_dir() {
                return !IGNORE_DIRS.contains(&name.as_ref()) && !extra_ignore.contains(&name.as_ref());
            }
            true
        });
    // Errors (EPERM, vanished entries, junction loops) are skipped, never fatal.
    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if name.ends_with(".lock") {
            continue; // IGNORE "**/*.lock"
        }
        let Ok(rel) = entry.path().strip_prefix(root) else { continue };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if keep(&rel) {
            out.push(rel);
            if out.len() >= cap {
                break;
            }
        }
    }
    out
}

/// Resolve the node test/lint/build script into a runnable command.
fn node_script(cwd: &Path, pm: &str, script: &str) -> Option<String> {
    let pkg = read_pkg(cwd)?;
    if !pkg.get("scripts").and_then(|s| s.get(script)).is_some_and(truthy) {
        return None;
    }
    Some(if pm == "npm" { format!("npm run {script}") } else { format!("{pm} {script}") })
}

/// One language detector (repoScanner.ts `DETECTORS`, same order).
struct Detector {
    language: &'static str,
    marker: fn(&Path) -> bool,
    pm: fn(&Path) -> Option<String>,
    test_command: Option<&'static str>,
    lint_command: Option<&'static str>,
    build_command: Option<&'static str>,
    framework: Option<fn(&Path) -> Option<String>>,
}

fn js_framework(c: &Path) -> Option<String> {
    let pkg = read_pkg(c)?;
    let mut deps = serde_json::Map::new();
    for key in ["dependencies", "devDependencies"] {
        if let Some(Value::Object(m)) = pkg.get(key) {
            deps.extend(m.clone());
        }
    }
    let has = |k: &str| deps.get(k).is_some_and(truthy);
    [("next", "Next.js"), ("react", "React"), ("vue", "Vue"), ("express", "Express"), ("fastify", "Fastify")]
        .into_iter()
        .find(|(dep, _)| has(dep))
        .map(|(_, name)| name.to_string())
}

static FASTAPI: LazyLock<Regex> = LazyLock::new(|| Regex::new("(?i)fastapi").expect("static regex"));
static FLASK: LazyLock<Regex> = LazyLock::new(|| Regex::new("(?i)flask").expect("static regex"));

fn py_framework(c: &Path) -> Option<String> {
    if exists(c, "manage.py") {
        return Some("Django".into());
    }
    let req = safe_read(c, "requirements.txt") + &safe_read(c, "pyproject.toml");
    if FASTAPI.is_match(&req) {
        return Some("FastAPI".into());
    }
    if FLASK.is_match(&req) {
        return Some("Flask".into());
    }
    None
}

/// Bounded: only a couple of levels deep, so this never walks a huge tree
/// (e.g. when launched from a home directory).
fn has_csproj(c: &Path) -> bool {
    !walk_files(c, 2, &[], 1, |f| f.ends_with(".csproj")).is_empty()
}

const DETECTORS: &[Detector] = &[
    Detector {
        language: "JavaScript/TypeScript",
        marker: |c| exists(c, "package.json"),
        pm: |c| {
            Some(
                if exists(c, "pnpm-lock.yaml") {
                    "pnpm"
                } else if exists(c, "yarn.lock") {
                    "yarn"
                } else if exists(c, "bun.lockb") {
                    "bun"
                } else {
                    "npm"
                }
                .into(),
            )
        },
        test_command: Some("test"),
        lint_command: Some("lint"),
        build_command: Some("build"),
        framework: Some(js_framework),
    },
    Detector {
        language: "Python",
        marker: |c| exists(c, "pyproject.toml") || exists(c, "requirements.txt") || exists(c, "setup.py"),
        pm: |c| {
            Some(if exists(c, "poetry.lock") { "poetry" } else if exists(c, "uv.lock") { "uv" } else { "pip" }.into())
        },
        test_command: Some("pytest"),
        lint_command: None,
        build_command: None,
        framework: Some(py_framework),
    },
    Detector {
        language: "Rust",
        marker: |c| exists(c, "Cargo.toml"),
        pm: |_| Some("cargo".into()),
        test_command: Some("cargo test"),
        lint_command: Some("cargo clippy"),
        build_command: Some("cargo build"),
        framework: None,
    },
    Detector {
        language: "Go",
        marker: |c| exists(c, "go.mod"),
        pm: |_| Some("go".into()),
        test_command: Some("go test ./..."),
        lint_command: Some("go vet ./..."),
        build_command: Some("go build ./..."),
        framework: None,
    },
    Detector {
        language: "Java",
        marker: |c| exists(c, "pom.xml") || exists(c, "build.gradle") || exists(c, "build.gradle.kts"),
        pm: |c| Some(if exists(c, "pom.xml") { "maven" } else { "gradle" }.into()),
        test_command: None,
        lint_command: None,
        build_command: None,
        framework: Some(|c| Some(if exists(c, "pom.xml") { "Maven" } else { "Gradle" }.into())),
    },
    Detector {
        language: "C#",
        marker: has_csproj,
        pm: |_| Some("dotnet".into()),
        test_command: Some("dotnet test"),
        lint_command: None,
        build_command: Some("dotnet build"),
        framework: None,
    },
    Detector {
        language: "PHP",
        marker: |c| exists(c, "composer.json"),
        pm: |_| Some("composer".into()),
        test_command: Some("vendor/bin/phpunit"),
        lint_command: None,
        build_command: None,
        framework: None,
    },
    Detector {
        language: "Ruby",
        marker: |c| exists(c, "Gemfile"),
        pm: |_| Some("bundler".into()),
        test_command: Some("bundle exec rspec"),
        lint_command: None,
        build_command: None,
        framework: None,
    },
];

/// Git probes never prompt and never take optional locks (a scan must not
/// contend with the user's own git commands).
const GIT_ENV: &[(&str, &str)] = &[("GIT_TERMINAL_PROMPT", "0"), ("GIT_OPTIONAL_LOCKS", "0")];

/// Is `cwd` inside a git work tree? (simple-git `checkIsRepo`.)
pub async fn is_git_repo(cwd: &Path) -> bool {
    let args = ["rev-parse".to_string(), "--is-inside-work-tree".to_string()];
    match run_captured("git", &args, cwd, GIT_ENV, Duration::from_secs(15), None).await {
        Ok(out) => out.success() && out.stdout.trim() == "true",
        Err(_) => false,
    }
}

/// Does the work tree have uncommitted changes (incl. untracked files)?
/// Errors and timeouts count as clean, as in the TS.
pub async fn is_git_dirty(cwd: &Path) -> bool {
    let args = ["status".to_string(), "--porcelain".to_string()];
    match run_captured("git", &args, cwd, GIT_ENV, Duration::from_secs(60), None).await {
        Ok(out) => out.success() && !out.stdout.trim().is_empty(),
        Err(_) => false,
    }
}

static ENTRYPOINT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(^|/)(index|main|app|cli|server)\.(ts|js|py|go|rs)$").expect("static regex"));
static KEY_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(package\.json|pyproject\.toml|Cargo\.toml|go\.mod|README|tsconfig\.json)").expect("static regex")
});

/// Detected project facts, without the git probes (sync; used by [`scan_repo`]).
fn detect(cwd: &Path) -> (Vec<String>, RepoInfo) {
    let mut languages = Vec::new();
    let mut info = RepoInfo { root: cwd.to_string_lossy().to_string(), ..Default::default() };
    for d in DETECTORS {
        if !(d.marker)(cwd) {
            continue;
        }
        languages.push(d.language.to_string());
        if info.framework.is_none() {
            if let Some(f) = d.framework {
                info.framework = f(cwd);
            }
        }
        let pm = (d.pm)(cwd);
        if info.package_manager.is_none() {
            info.package_manager.clone_from(&pm);
        }
        match (d.language.starts_with("JavaScript"), pm.as_deref()) {
            (true, Some(pm)) => {
                let script = |s: Option<&str>| s.and_then(|s| node_script(cwd, pm, s));
                info.test_command = info.test_command.take().or_else(|| script(d.test_command));
                info.lint_command = info.lint_command.take().or_else(|| script(d.lint_command));
                info.build_command = info.build_command.take().or_else(|| script(d.build_command));
            }
            _ => {
                let own = |s: Option<&str>| s.map(str::to_string);
                info.test_command = info.test_command.take().or_else(|| own(d.test_command));
                info.lint_command = info.lint_command.take().or_else(|| own(d.lint_command));
                info.build_command = info.build_command.take().or_else(|| own(d.build_command));
            }
        }
    }
    (languages, info)
}

/// Scan the project at `cwd`.
pub async fn scan_repo(cwd: &Path) -> RepoInfo {
    let root = cwd.to_path_buf();
    // Filesystem work runs off the async threads (a large tree takes a while).
    let walked = tokio::task::spawn_blocking(move || {
        let (languages, info) = detect(&root);
        // Only deep-scan when a real project MARKER is present (package.json,
        // Cargo.toml, …). A bare git repo with no markers — e.g. a home
        // directory the user happens to have `git init`-ed — is NOT
        // deep-scanned, since walking it can take 20s+.
        let files = if languages.is_empty() {
            Vec::new()
        } else {
            walk_files(&root, SCAN_DEPTH, SCAN_IGNORE_DIRS, FILE_CAP, |_| true)
        };
        (languages, info, files)
    })
    .await;
    let (languages, mut info, all_files) = match walked {
        Ok(v) => v,
        Err(_) => (Vec::new(), RepoInfo { root: cwd.to_string_lossy().to_string(), ..Default::default() }, Vec::new()),
    };

    info.has_git = is_git_repo(cwd).await;
    info.entrypoints = all_files.iter().filter(|f| ENTRYPOINT.is_match(f)).take(10).cloned().collect();
    info.key_files = all_files.iter().filter(|f| KEY_FILE.is_match(f)).take(20).cloned().collect();
    info.file_count = all_files.len();
    // `git status` can be slow on a huge tree (e.g. a git-init'ed home dir),
    // so only check dirtiness for an actual project.
    info.dirty = !languages.is_empty() && info.has_git && is_git_dirty(cwd).await;
    info.languages = if languages.is_empty() { vec!["unknown".to_string()] } else { languages };
    info
}

/// A compact text summary of the repo for prompt context.
pub fn summarize_repo(info: &RepoInfo) -> String {
    let mut lines = vec![format!("Languages: {}", info.languages.join(", "))];
    if let Some(f) = info.framework.as_deref().filter(|s| !s.is_empty()) {
        lines.push(format!("Framework: {f}"));
    }
    if let Some(pm) = info.package_manager.as_deref().filter(|s| !s.is_empty()) {
        lines.push(format!("Package manager: {pm}"));
    }
    match info.test_command.as_deref().filter(|s| !s.is_empty()) {
        Some(t) => lines.push(format!("Test command: {t}")),
        None => lines.push("Test command: (none detected)".to_string()),
    }
    lines.push(format!("Files: {}", info.file_count));
    if !info.entrypoints.is_empty() {
        lines.push(format!("Entrypoints: {}", info.entrypoints.join(", ")));
    }
    let git = if info.has_git {
        if info.dirty {
            "dirty"
        } else {
            "clean"
        }
    } else {
        "not a repo"
    };
    lines.push(format!("Git: {git}"));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    async fn scratch() -> tempfile::TempDir {
        crate::test_support::scratch_repo()
    }

    // Ported from tests/repoScanner.test.ts.

    #[tokio::test]
    async fn detects_a_node_project_and_its_package_manager_and_test_script() {
        let dir = scratch().await;
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"x","scripts":{"test":"vitest"},"dependencies":{"express":"^4"}}"#,
        )
        .unwrap();
        fs::write(dir.path().join("package-lock.json"), "{}").unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src").join("index.js"), "console.log(1)").unwrap();

        let info = scan_repo(dir.path()).await;
        assert!(info.languages.join(",").contains("JavaScript"));
        assert_eq!(info.package_manager.as_deref(), Some("npm"));
        assert_eq!(info.test_command.as_deref(), Some("npm run test"));
        assert_eq!(info.framework.as_deref(), Some("Express"));
        assert!(info.entrypoints.contains(&"src/index.js".to_string()), "{:?}", info.entrypoints);
    }

    #[tokio::test]
    async fn detects_a_python_project() {
        let dir = scratch().await;
        fs::write(dir.path().join("requirements.txt"), "fastapi\nuvicorn\n").unwrap();
        let info = scan_repo(dir.path()).await;
        assert!(info.languages.join(",").contains("Python"));
        assert_eq!(info.test_command.as_deref(), Some("pytest"));
        assert_eq!(info.framework.as_deref(), Some("FastAPI"));
    }

    #[tokio::test]
    async fn detects_a_rust_project() {
        let dir = scratch().await;
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        let info = scan_repo(dir.path()).await;
        assert!(info.languages.join(",").contains("Rust"));
        assert_eq!(info.test_command.as_deref(), Some("cargo test"));
    }

    // Additional coverage.

    #[tokio::test]
    async fn no_marker_means_no_deep_scan() {
        let dir = scratch().await;
        fs::write(dir.path().join("notes.txt"), "x").unwrap();
        let info = scan_repo(dir.path()).await;
        assert_eq!(info.languages, vec!["unknown".to_string()]);
        assert_eq!(info.file_count, 0);
        assert!(!info.dirty);
    }

    #[tokio::test]
    async fn pnpm_scripts_and_dirty_tree() {
        let dir = scratch().await;
        fs::write(
            dir.path().join("package.json"),
            r#"{"scripts":{"test":"vitest","build":"tsc"},"devDependencies":{"next":"14"}}"#,
        )
        .unwrap();
        fs::write(dir.path().join("pnpm-lock.yaml"), "").unwrap();
        let info = scan_repo(dir.path()).await;
        assert_eq!(info.package_manager.as_deref(), Some("pnpm"));
        assert_eq!(info.test_command.as_deref(), Some("pnpm test"));
        assert_eq!(info.build_command.as_deref(), Some("pnpm build"));
        assert_eq!(info.lint_command, None);
        assert_eq!(info.framework.as_deref(), Some("Next.js"));
        if info.has_git {
            assert!(info.dirty, "untracked files make the tree dirty");
        }
    }

    #[test]
    fn walk_skips_ignored_dot_and_lock_entries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        for d in ["node_modules/x", "src/deep", ".hidden", "AppData/Local", "target"] {
            fs::create_dir_all(p.join(d)).unwrap();
        }
        for f in [
            "node_modules/x/a.js",
            "src/main.rs",
            "src/deep/lib.rs",
            ".hidden/h.txt",
            ".env",
            "Cargo.lock",
            "AppData/Local/z.txt",
            "target/t.o",
            "README.md",
        ] {
            fs::write(p.join(f), "x").unwrap();
        }
        let files = walk_files(p, 6, SCAN_IGNORE_DIRS, 100, |_| true);
        assert_eq!(files, vec!["README.md", "src/deep/lib.rs", "src/main.rs"]);
        // Depth bound: depth 1 = direct children only.
        assert_eq!(walk_files(p, 1, SCAN_IGNORE_DIRS, 100, |_| true), vec!["README.md"]);
        // Cap.
        assert_eq!(walk_files(p, 6, SCAN_IGNORE_DIRS, 2, |_| true).len(), 2);
    }

    #[test]
    fn csproj_detection_is_depth_bounded() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("a/b")).unwrap();
        fs::write(dir.path().join("a/b/deep.csproj"), "").unwrap();
        assert!(!has_csproj(dir.path()));
        fs::write(dir.path().join("a/app.csproj"), "").unwrap();
        assert!(has_csproj(dir.path()));
    }

    #[test]
    fn summary_lines() {
        let info = RepoInfo {
            root: "/x".into(),
            languages: vec!["Rust".into()],
            package_manager: Some("cargo".into()),
            test_command: None,
            entrypoints: vec!["src/main.rs".into()],
            file_count: 3,
            has_git: true,
            dirty: false,
            ..Default::default()
        };
        assert_eq!(
            summarize_repo(&info),
            "Languages: Rust\nPackage manager: cargo\nTest command: (none detected)\nFiles: 3\nEntrypoints: src/main.rs\nGit: clean"
        );
    }
}
