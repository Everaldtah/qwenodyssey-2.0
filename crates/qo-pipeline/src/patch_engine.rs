//! Patch engine: apply file edits as unified diffs / creates / rewrites /
//! deletes, with a rollback journal under `.qwenodyssey/patches`
//! (port of `src/core/patchEngine.ts`).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::deps::{resolve_inside, workspace_path};
use crate::diff::{apply_patch, create_two_files_patch};
use qo_core::{EditMode, FileEdit};

/// One journaled edit: what it was and the file's pre-image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedEdit {
    pub path: String,
    /// "patch" | "create" | "rewrite" | "delete" (or a block match strategy).
    pub mode: String,
    /// Content before the edit; `None` = the file did not exist.
    pub before: Option<String>,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyOutcome {
    pub applied: Vec<AppliedEdit>,
    pub journal_file: Option<PathBuf>,
    pub all_ok: bool,
}

#[derive(Serialize, Deserialize)]
struct Journal {
    ts: String,
    applied: Vec<AppliedEdit>,
}

/// Lowercase mode name as the TS wrote it ("patch", "create", …).
pub fn mode_name(mode: EditMode) -> &'static str {
    match mode {
        EditMode::Patch => "patch",
        EditMode::Create => "create",
        EditMode::Rewrite => "rewrite",
        EditMode::Delete => "delete",
    }
}

/// Read a file as UTF-8, lossily (Node `readFileSync(p, "utf-8")` semantics).
pub(crate) fn read_lossy(path: &Path) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    })
}

fn write_creating_dirs(abs: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = abs.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(abs, content)
}

/// Build a colourisable unified diff preview for a single edit (no writes).
/// Errs only when the path escapes the project root.
pub fn preview_edit(cwd: &Path, edit: &FileEdit) -> anyhow::Result<String> {
    let abs = resolve_inside(cwd, &edit.path)?;
    let before = if abs.is_file() { read_lossy(&abs).unwrap_or_default() } else { String::new() };
    if edit.mode == EditMode::Delete {
        return Ok(create_two_files_patch(&edit.path, "/dev/null", &before, "", Some(""), Some("")));
    }
    if edit.mode == EditMode::Patch {
        if let Some(d) = edit.diff.as_deref().filter(|d| !d.is_empty()) {
            return Ok(d.to_string());
        }
    }
    let after = edit.content.as_deref().unwrap_or("");
    Ok(create_two_files_patch(&edit.path, &edit.path, &before, after, Some(""), Some("")))
}

/// [`preview_edit`], with an escaping path rendered as a one-line note instead
/// of an error (for display paths that must not abort the run).
pub fn preview_edit_or_note(cwd: &Path, edit: &FileEdit) -> String {
    preview_edit(cwd, edit).unwrap_or_else(|e| format!("(cannot preview {}: {e})", edit.path))
}

fn apply_single(cwd: &Path, edit: &FileEdit) -> AppliedEdit {
    let mode = mode_name(edit.mode).to_string();
    // The TS resolved outside its try/catch, so one escaping path aborted the
    // whole batch half-applied and unjournaled; here it fails just this edit.
    let abs = match resolve_inside(cwd, &edit.path) {
        Ok(a) => a,
        Err(e) => {
            return AppliedEdit { path: edit.path.clone(), mode, before: None, ok: false, error: Some(e.to_string()) }
        }
    };
    let existed = abs.exists();
    let before = if existed { read_lossy(&abs).ok() } else { None };
    let fail = |before: Option<String>, msg: String| AppliedEdit {
        path: edit.path.clone(),
        mode: mode.clone(),
        before,
        ok: false,
        error: Some(msg),
    };
    let ok = |before: Option<String>| AppliedEdit { path: edit.path.clone(), mode: mode.clone(), before, ok: true, error: None };

    match edit.mode {
        EditMode::Delete => {
            if existed {
                if let Err(e) = fs::remove_file(&abs) {
                    return fail(before, e.to_string());
                }
            }
            ok(before)
        }
        EditMode::Patch => {
            let Some(diff) = edit.diff.as_deref().filter(|d| !d.is_empty()) else {
                return fail(before, "patch edit missing diff".to_string());
            };
            match apply_patch(before.as_deref().unwrap_or(""), diff) {
                Ok(Some(patched)) => match write_creating_dirs(&abs, &patched) {
                    Ok(()) => ok(before),
                    Err(e) => fail(before, e.to_string()),
                },
                Ok(None) => fail(before, "unified diff did not apply cleanly".to_string()),
                Err(e) => fail(before, e.to_string()),
            }
        }
        EditMode::Create | EditMode::Rewrite => {
            if edit.mode == EditMode::Create && existed {
                return fail(before, "create target already exists".to_string());
            }
            match write_creating_dirs(&abs, edit.content.as_deref().unwrap_or("")) {
                Ok(()) => ok(before),
                Err(e) => fail(before, e.to_string()),
            }
        }
    }
}

fn patches_dir(cwd: &Path) -> PathBuf {
    workspace_path(cwd).join("patches")
}

/// Apply a list of edits, journaling pre-images so the batch can be rolled back.
/// Errs only when the journal itself cannot be written.
pub fn apply_edits(cwd: &Path, edits: &[FileEdit]) -> anyhow::Result<ApplyOutcome> {
    let applied: Vec<AppliedEdit> = edits.iter().map(|e| apply_single(cwd, e)).collect();
    let all_ok = applied.iter().all(|a| a.ok);
    let journal_file = write_journal(cwd, &applied)?;
    Ok(ApplyOutcome { applied, journal_file: Some(journal_file), all_ok })
}

/// Write `patch-<YYYYMMDDTHHMMSS>.json` (the TS stamp). A second batch in the
/// same second gets `_1`, `_2`, … instead of overwriting the first (those
/// names sort after the plain one, so [`latest_journal`] still finds the newest).
pub(crate) fn write_journal(cwd: &Path, applied: &[AppliedEdit]) -> anyhow::Result<PathBuf> {
    let dir = patches_dir(cwd);
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let now = chrono::Utc::now();
    let stamp = now.format("%Y%m%dT%H%M%S").to_string();
    let mut file = dir.join(format!("patch-{stamp}.json"));
    let mut n = 1;
    while file.exists() {
        file = dir.join(format!("patch-{stamp}_{n}.json"));
        n += 1;
    }
    let journal = Journal {
        ts: qo_core::logger::iso_timestamp(now),
        applied: applied.to_vec(),
    };
    let text = serde_json::to_string_pretty(&journal)?;
    fs::write(&file, text).with_context(|| format!("writing {}", file.display()))?;
    Ok(file)
}

/// Restore files to their pre-image from a journal file. Returns the paths
/// restored (entries whose path no longer resolves inside `cwd` are skipped).
pub fn rollback(cwd: &Path, journal_file: &Path) -> anyhow::Result<Vec<String>> {
    let raw = fs::read_to_string(journal_file).with_context(|| format!("reading {}", journal_file.display()))?;
    let journal: Journal =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", journal_file.display()))?;
    let mut restored = Vec::new();
    for a in journal.applied {
        let Ok(abs) = resolve_inside(cwd, &a.path) else { continue };
        match &a.before {
            None => {
                if abs.exists() {
                    fs::remove_file(&abs).with_context(|| format!("removing {}", abs.display()))?;
                }
            }
            Some(before) => {
                write_creating_dirs(&abs, before).with_context(|| format!("restoring {}", abs.display()))?;
            }
        }
        restored.push(a.path);
    }
    Ok(restored)
}

/// The most recent patch journal, if any.
pub fn latest_journal(cwd: &Path) -> Option<PathBuf> {
    let dir = patches_dir(cwd);
    let mut names: Vec<String> = fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|f| f.starts_with("patch-") && f.ends_with(".json"))
        .collect();
    names.sort();
    names.pop().map(|f| dir.join(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(path: &str, mode: EditMode, content: Option<&str>, diff: Option<&str>) -> FileEdit {
        FileEdit {
            path: path.into(),
            mode,
            diff: diff.map(str::to_string),
            content: content.map(str::to_string),
            rationale: None,
        }
    }

    // Ported from tests/patchEngine.test.ts.

    #[test]
    fn creates_a_file_and_can_roll_it_back() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = apply_edits(dir.path(), &[edit("hello.txt", EditMode::Create, Some("hi\n"), None)]).unwrap();
        assert!(outcome.all_ok);
        assert_eq!(fs::read_to_string(dir.path().join("hello.txt")).unwrap(), "hi\n");
        rollback(dir.path(), outcome.journal_file.as_ref().unwrap()).unwrap();
        assert!(!dir.path().join("hello.txt").exists());
    }

    #[test]
    fn rewrites_a_file_and_restores_on_rollback() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "original\n").unwrap();
        let outcome = apply_edits(dir.path(), &[edit("a.txt", EditMode::Rewrite, Some("changed\n"), None)]).unwrap();
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "changed\n");
        rollback(dir.path(), outcome.journal_file.as_ref().unwrap()).unwrap();
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "original\n");
    }

    #[test]
    fn preview_produces_a_unified_diff_for_a_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("b.txt"), "one\n").unwrap();
        let diff = preview_edit(dir.path(), &edit("b.txt", EditMode::Rewrite, Some("two\n"), None)).unwrap();
        assert!(diff.contains("-one"));
        assert!(diff.contains("+two"));
    }

    #[test]
    fn reports_failure_when_a_unified_diff_does_not_apply() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("c.txt"), "real content\n").unwrap();
        let bad = "Index: c.txt\n===\n--- c.txt\n+++ c.txt\n@@ -1,1 +1,1 @@\n-nonexistent line\n+replacement\n";
        let outcome = apply_edits(dir.path(), &[edit("c.txt", EditMode::Patch, None, Some(bad))]).unwrap();
        assert!(!outcome.all_ok);
        assert!(!outcome.applied[0].ok);
        assert_eq!(fs::read_to_string(dir.path().join("c.txt")).unwrap(), "real content\n");
    }

    // Additional coverage.

    #[test]
    fn applies_a_clean_patch_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("p.txt"), "a\nb\nc\n").unwrap();
        fs::write(dir.path().join("gone.txt"), "x").unwrap();
        let diff = "--- p.txt\n+++ p.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n";
        let outcome = apply_edits(
            dir.path(),
            &[edit("p.txt", EditMode::Patch, None, Some(diff)), edit("gone.txt", EditMode::Delete, None, None)],
        )
        .unwrap();
        assert!(outcome.all_ok, "{:?}", outcome.applied);
        assert_eq!(fs::read_to_string(dir.path().join("p.txt")).unwrap(), "a\nB\nc\n");
        assert!(!dir.path().join("gone.txt").exists());
        let restored = rollback(dir.path(), &latest_journal(dir.path()).unwrap()).unwrap();
        assert_eq!(restored, vec!["p.txt".to_string(), "gone.txt".to_string()]);
        assert_eq!(fs::read_to_string(dir.path().join("p.txt")).unwrap(), "a\nb\nc\n");
        assert_eq!(fs::read_to_string(dir.path().join("gone.txt")).unwrap(), "x");
    }

    #[test]
    fn create_refuses_existing_and_escape_fails_only_that_edit() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("e.txt"), "keep").unwrap();
        let outcome = apply_edits(
            dir.path(),
            &[
                edit("e.txt", EditMode::Create, Some("new"), None),
                edit("../outside.txt", EditMode::Create, Some("x"), None),
                edit("sub/dir/n.txt", EditMode::Create, Some("n"), None),
            ],
        )
        .unwrap();
        assert!(!outcome.all_ok);
        assert_eq!(outcome.applied[0].error.as_deref(), Some("create target already exists"));
        assert!(outcome.applied[1].error.as_deref().unwrap_or("").contains("escapes"));
        assert!(outcome.applied[2].ok);
        assert_eq!(fs::read_to_string(dir.path().join("e.txt")).unwrap(), "keep");
        assert_eq!(fs::read_to_string(dir.path().join("sub/dir/n.txt")).unwrap(), "n");
    }

    #[test]
    fn journals_in_the_same_second_do_not_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_journal(dir.path(), &[]).unwrap();
        let b = write_journal(dir.path(), &[]).unwrap();
        assert_ne!(a, b);
        let latest = latest_journal(dir.path()).unwrap();
        // Whichever was written last must be the one latest_journal picks.
        assert_eq!(latest, b);
        let text = fs::read_to_string(&a).unwrap();
        assert!(text.contains("\"ts\""));
        assert!(text.contains("\"applied\""));
    }

    #[test]
    fn journal_keeps_null_before_and_omits_missing_error() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = apply_edits(dir.path(), &[edit("n.txt", EditMode::Create, Some("x"), None)]).unwrap();
        let text = fs::read_to_string(outcome.journal_file.unwrap()).unwrap();
        assert!(text.contains("\"before\": null"));
        assert!(!text.contains("\"error\""));
    }

    #[test]
    fn delete_preview_targets_dev_null() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("d.txt"), "bye\n").unwrap();
        let p = preview_edit(dir.path(), &edit("d.txt", EditMode::Delete, None, None)).unwrap();
        assert!(p.contains("+++ /dev/null"));
        assert!(p.contains("-bye"));
        assert!(preview_edit(dir.path(), &edit("../x", EditMode::Rewrite, Some(""), None)).is_err());
    }
}
