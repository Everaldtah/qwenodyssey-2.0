//! Reliable edit pipeline — the heart of the small-model coding uplift
//! (port of `src/core/editPipeline.ts`; the SEARCH/REPLACE engine itself is
//! [`qo_tools::edit_block`]).
//!
//! Composes the building blocks into one robust flow:
//!
//! ```text
//!   generate (best-of-N edit-block candidates)
//!     → dry-run apply each candidate in memory (reject non-applying)
//!     → apply the best to the real tree with a rollback snapshot
//!     → verify (typecheck / lint / test)
//!     → if verify fails, feed the digest back and self-repair (bounded)
//!     → on unrecoverable failure, roll back cleanly
//! ```
//!
//! Each layer targets a specific small-model failure mode:
//!   - edit blocks        → unreliable unified diffs
//!   - best-of-N + apply  → single-sample variance
//!   - verify-in-the-loop → models that don't check their work
//!   - self-repair        → grounding the retry in real compiler/test output
//!   - rollback snapshot  → never leave the tree half-edited

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::candidate_select::{best_of_n, score, Drawn, Score};
use crate::patch_engine::read_lossy;
use crate::verify_loop::{verify, CheckKind, VerifyOptions, VerifyOutcome};
use qo_tools::edit_block::{apply_block_to_content, apply_edit_blocks, parse_edit_blocks, BlockApplyResult, EditBlock};

/// Maps a relative path to an absolute one, enforcing the write sandbox.
pub type Resolver = Arc<dyn Fn(&str) -> anyhow::Result<PathBuf> + Send + Sync>;

/// Progress hook for the UI.
pub type EventSink = Arc<dyn Fn(&PipelineEvent) + Send + Sync>;

#[derive(Clone)]
pub struct EditPipelineOptions {
    pub root: PathBuf,
    pub resolve: Resolver,
    /// How many candidates to sample per attempt (best-of-N). 1 disables it.
    pub candidates: usize,
    /// Max self-repair rounds after a failing verify. 0 disables repair.
    pub max_repair_rounds: usize,
    /// Run verification (typecheck/lint/test) after applying.
    pub run_verify: bool,
    /// Only run these check kinds during the inner loop (e.g. fast checks).
    pub verify_only: Option<Vec<CheckKind>>,
    /// Esc interrupt for the verify runs (model calls take it via the generator).
    pub cancel: Option<CancellationToken>,
    pub on_event: Option<EventSink>,
}

impl EditPipelineOptions {
    /// Defaults: one candidate, no repair, no verify.
    pub fn new(root: impl Into<PathBuf>, resolve: Resolver) -> Self {
        Self {
            root: root.into(),
            resolve,
            candidates: 1,
            max_repair_rounds: 0,
            run_verify: false,
            verify_only: None,
            cancel: None,
            on_event: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum PipelineEvent {
    Sampling { attempt: usize, candidate: usize },
    CandidateScored { candidate: usize, score: f64, reason: String },
    Applied { results: Vec<BlockApplyResult> },
    Verifying,
    VerifyResult { passed: bool, feedback: String },
    Repairing { round: usize },
    RolledBack { reason: String },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditPipelineResult {
    pub applied: bool,
    pub verified: bool,
    pub blocks: Vec<EditBlock>,
    pub apply_results: Vec<BlockApplyResult>,
    pub verify: Option<VerifyOutcome>,
    pub rounds: usize,
    pub message: String,
}

/// Pre-images of the files a set of blocks will touch (`None` = absent).
type Snapshot = HashMap<String, Option<String>>;

/// Snapshot the files a set of blocks will touch, for clean rollback.
fn snapshot(blocks: &[EditBlock], resolve: &Resolver) -> Snapshot {
    let mut snap = Snapshot::new();
    for b in blocks {
        if snap.contains_key(&b.path) {
            continue;
        }
        let content = resolve(&b.path).ok().filter(|abs| abs.exists()).and_then(|abs| read_lossy(&abs).ok());
        snap.insert(b.path.clone(), content);
    }
    snap
}

/// Best-effort restore of a snapshot.
fn restore(snap: &Snapshot, resolve: &Resolver) {
    for (rel, content) in snap {
        let Ok(abs) = resolve(rel) else { continue };
        match content {
            None => {
                if abs.exists() {
                    let _ = std::fs::remove_file(&abs);
                }
            }
            Some(c) => {
                if let Some(dir) = abs.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let _ = std::fs::write(&abs, c);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DryRun {
    pub ok: bool,
    pub first_error: Option<String>,
}

/// Dry-run: would these blocks all apply to the current on-disk content?
/// Pure — no writes. Sequential blocks on one file compose in memory.
pub fn dry_run_apply(blocks: &[EditBlock], resolve: &Resolver) -> DryRun {
    let mut live: HashMap<&str, String> = HashMap::new();
    for b in blocks {
        let content = match live.get(b.path.as_str()) {
            Some(c) => c.clone(),
            None => {
                let abs = match resolve(&b.path) {
                    Ok(a) => a,
                    Err(e) => return DryRun { ok: false, first_error: Some(e.to_string()) },
                };
                if abs.exists() {
                    match read_lossy(&abs) {
                        Ok(c) => c,
                        Err(e) => return DryRun { ok: false, first_error: Some(e.to_string()) },
                    }
                } else {
                    String::new()
                }
            }
        };
        let r = apply_block_to_content(&content, b);
        match (r.ok, r.result) {
            (true, Some(next)) => {
                live.insert(b.path.as_str(), next);
            }
            _ => {
                let err = r.error.unwrap_or_else(|| "did not apply".to_string());
                return DryRun { ok: false, first_error: Some(format!("{}: {err}", b.path)) };
            }
        }
    }
    DryRun { ok: true, first_error: None }
}

fn apply_blocks(blocks: &[EditBlock], resolve: &Resolver) -> Vec<BlockApplyResult> {
    apply_edit_blocks(blocks, |rel: &str| resolve(rel))
}

/// Run the full reliable edit pipeline. `generate(attempt, feedback)` produces
/// raw model output (edit blocks); on repair rounds it receives the verify
/// feedback to incorporate.
///
/// Errors: an Esc interrupt during sampling, or any model error during a
/// repair round (as the TS let it throw). Failed sampling attempts in phase 1
/// are otherwise skipped.
pub async fn run_edit_pipeline<G, Fut>(mut generate: G, opts: &EditPipelineOptions) -> anyhow::Result<EditPipelineResult>
where
    G: FnMut(usize, Option<String>) -> Fut,
    Fut: Future<Output = anyhow::Result<String>>,
{
    let emit = |e: PipelineEvent| {
        if let Some(sink) = &opts.on_event {
            sink(&e);
        }
    };

    // ── Phase 1: best-of-N candidate generation, scored by "does it apply?" ──
    let resolve = opts.resolve.clone();
    let selection = best_of_n(
        opts.candidates.max(1),
        |attempt| {
            emit(PipelineEvent::Sampling { attempt: 0, candidate: attempt });
            let fut = generate(attempt, None);
            async move {
                let raw = fut.await?;
                let blocks = parse_edit_blocks(&raw);
                Ok(Drawn { value: if blocks.is_empty() { None } else { Some(blocks) }, raw })
            }
        },
        |blocks: &Vec<EditBlock>| {
            let dry = dry_run_apply(blocks, &resolve);
            let n = blocks.len();
            async move {
                Ok(if dry.ok {
                    // More blocks that apply cleanly is mild positive signal.
                    Score { score: score::APPLIES + n.min(10) as f64, reason: format!("{n} block(s) apply") }
                } else {
                    Score { score: score::REJECT, reason: dry.first_error.unwrap_or_else(|| "did not apply".into()) }
                })
            }
        },
        // The first applying candidate is good enough to try.
        Some(score::APPLIES),
        |candidate, scored| {
            if let Some(s) = scored {
                emit(PipelineEvent::CandidateScored { candidate, score: s.score, reason: s.reason.clone() });
            }
        },
    )
    .await?;

    let Some(best) = selection.best else {
        return Ok(EditPipelineResult {
            message: "No candidate produced applicable edits. The model may need clearer instructions or the SEARCH text didn't match.".into(),
            ..Default::default()
        });
    };
    let mut blocks = best.value;

    // ── Phase 2: apply with a rollback snapshot ──
    let mut snap = snapshot(&blocks, &opts.resolve);
    let mut apply_results = apply_blocks(&blocks, &opts.resolve);
    emit(PipelineEvent::Applied { results: apply_results.clone() });

    if !apply_results.iter().all(|r| r.ok) {
        restore(&snap, &opts.resolve);
        emit(PipelineEvent::RolledBack { reason: "apply failed".into() });
        let errs = apply_results
            .iter()
            .filter(|r| !r.ok)
            .map(|r| format!("{}: {}", r.path, r.error.as_deref().unwrap_or("")))
            .collect::<Vec<_>>()
            .join("; ");
        return Ok(EditPipelineResult {
            blocks,
            apply_results,
            message: format!("Edits failed to apply and were rolled back: {errs}"),
            ..Default::default()
        });
    }

    // ── Phase 3: verify + bounded self-repair ──
    if !opts.run_verify {
        return Ok(EditPipelineResult {
            applied: true,
            blocks,
            apply_results,
            message: "Edits applied (verification disabled).".into(),
            ..Default::default()
        });
    }

    let vopts = VerifyOptions {
        only: opts.verify_only.clone(),
        stop_on_first_failure: true,
        timeout_ms: None,
        cancel: opts.cancel.clone(),
    };
    let mut rounds = 0usize;
    let mut last_verify: Option<VerifyOutcome> = None;
    for round in 0..=opts.max_repair_rounds {
        emit(PipelineEvent::Verifying);
        let v = verify(&opts.root, &vopts).await;
        emit(PipelineEvent::VerifyResult { passed: v.all_passed, feedback: v.feedback.clone() });

        if v.interrupted {
            // Esc during the checks: keep the edits, report honestly.
            return Ok(EditPipelineResult {
                applied: true,
                verified: false,
                blocks,
                apply_results,
                verify: Some(v),
                rounds,
                message: "Edits applied; verification was interrupted.".into(),
            });
        }
        if v.all_passed || !v.ran {
            let message = if v.ran { "Edits applied and verified." } else { "Edits applied (no checks detected)." };
            return Ok(EditPipelineResult {
                applied: true,
                verified: v.all_passed,
                blocks,
                apply_results,
                verify: Some(v),
                rounds,
                message: message.into(),
            });
        }
        let feedback = v.feedback.clone();
        last_verify = Some(v);
        if round >= opts.max_repair_rounds {
            break;
        }

        // Self-repair: regenerate with the failure digest as feedback.
        rounds += 1;
        emit(PipelineEvent::Repairing { round: rounds });
        let raw = generate(round + 1, Some(feedback)).await?;
        let repair_blocks = parse_edit_blocks(&raw);
        if repair_blocks.is_empty() {
            break;
        }
        if !dry_run_apply(&repair_blocks, &opts.resolve).ok {
            continue; // bad repair; try another round if budget remains
        }

        // Extend the snapshot to cover any newly-touched files, then apply.
        for (rel, content) in snapshot(&repair_blocks, &opts.resolve) {
            snap.entry(rel).or_insert(content);
        }
        let repair_results = apply_blocks(&repair_blocks, &opts.resolve);
        if repair_results.iter().all(|r| r.ok) {
            blocks.extend(repair_blocks);
            apply_results.extend(repair_results);
        }
    }

    // Verification never passed within budget. Keep the edits but report
    // honestly; the caller decides whether to roll back (partial progress is
    // often useful, and the user has the snapshot via the patch journal).
    let feedback = last_verify.as_ref().map(|v| v.feedback.clone()).unwrap_or_default();
    Ok(EditPipelineResult {
        applied: true,
        verified: false,
        blocks,
        apply_results,
        verify: last_verify,
        rounds,
        message: format!("Edits applied but verification still fails after {rounds} repair round(s):\n{feedback}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scratch() -> (tempfile::TempDir, Resolver) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let resolve: Resolver = Arc::new(move |rel: &str| {
            let abs = root.join(rel);
            if rel.contains("..") || Path::new(rel).is_absolute() {
                anyhow::bail!("escape");
            }
            Ok(abs)
        });
        (dir, resolve)
    }

    fn block_text(p: &str, search: &str, replace: &str) -> String {
        [p, "<<<<<<< SEARCH", search, "=======", replace, ">>>>>>> REPLACE"].join("\n")
    }

    fn eb(p: &str, s: &str, r: &str) -> EditBlock {
        EditBlock::new(p, s, r)
    }

    // Ported from tests/editPipeline.test.ts.

    #[test]
    fn dry_run_ok_when_blocks_apply_error_when_not() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        assert!(dry_run_apply(&[eb("a.txt", "hello", "hi")], &resolve).ok);
        let bad = dry_run_apply(&[eb("a.txt", "nope", "x")], &resolve);
        assert!(!bad.ok);
        assert!(bad.first_error.unwrap().starts_with("a.txt: "));
    }

    #[test]
    fn dry_run_composes_sequential_blocks_in_memory() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("a.txt"), "x\ny").unwrap();
        assert!(dry_run_apply(&[eb("a.txt", "x", "X"), eb("a.txt", "y", "Y")], &resolve).ok);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "x\ny", "dry run never writes");
    }

    #[tokio::test]
    async fn applies_the_best_applying_candidate_verify_disabled() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("f.ts"), "const v = 1;\n").unwrap();
        let opts = EditPipelineOptions::new(dir.path(), resolve);
        let res =
            run_edit_pipeline(|_, _| async { Ok(block_text("f.ts", "const v = 1;", "const v = 2;")) }, &opts).await.unwrap();
        assert!(res.applied);
        assert!(fs::read_to_string(dir.path().join("f.ts")).unwrap().contains("const v = 2;"));
        assert_eq!(res.message, "Edits applied (verification disabled).");
    }

    #[tokio::test]
    async fn rejects_non_applying_candidates_and_reports_when_none_apply() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("f.ts"), "const v = 1;\n").unwrap();
        let mut opts = EditPipelineOptions::new(dir.path(), resolve);
        opts.candidates = 2;
        let calls = AtomicUsize::new(0);
        let res = run_edit_pipeline(
            |_, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(block_text("f.ts", "DOES NOT EXIST", "x")) }
            },
            &opts,
        )
        .await
        .unwrap();
        assert!(!res.applied);
        assert!(res.message.to_lowercase().contains("no candidate"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(fs::read_to_string(dir.path().join("f.ts")).unwrap(), "const v = 1;\n");
    }

    #[tokio::test]
    async fn prefers_a_candidate_that_applies_over_one_that_does_not() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("f.ts"), "let count = 0;\n").unwrap();
        let mut opts = EditPipelineOptions::new(dir.path(), resolve);
        opts.candidates = 3;
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ev = events.clone();
        opts.on_event = Some(Arc::new(move |e: &PipelineEvent| ev.lock().unwrap().push(e.clone())));
        let attempt = AtomicUsize::new(0);
        let res = run_edit_pipeline(
            |_, _| {
                // First candidate is garbage, second is valid.
                let n = attempt.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    Ok(if n == 1 {
                        block_text("f.ts", "nonexistent line", "x")
                    } else {
                        block_text("f.ts", "let count = 0;", "let count = 99;")
                    })
                }
            },
            &opts,
        )
        .await
        .unwrap();
        assert!(res.applied);
        assert!(fs::read_to_string(dir.path().join("f.ts")).unwrap().contains("let count = 99;"));
        assert_eq!(attempt.load(Ordering::SeqCst), 2, "early accept after the first applying candidate");
        let ev = events.lock().unwrap();
        assert!(matches!(ev[0], PipelineEvent::Sampling { attempt: 0, candidate: 0 }));
        assert!(ev.iter().any(|e| matches!(e, PipelineEvent::CandidateScored { score, .. } if score.is_infinite())));
        assert!(ev.iter().any(|e| matches!(e, PipelineEvent::Applied { .. })));
    }

    #[tokio::test]
    async fn runs_a_verify_pass_and_self_repairs_using_feedback() {
        if which::which("node").is_err() {
            return;
        }
        let (dir, resolve) = scratch();
        // A package.json with a 'typecheck' that fails until a marker is present.
        fs::write(dir.path().join("package.json"), r#"{"scripts":{"typecheck":"node check.js"}}"#).unwrap();
        fs::write(
            dir.path().join("check.js"),
            "const fs=require('fs');const t=fs.existsSync('target.txt')?fs.readFileSync('target.txt','utf8'):'';process.exit(t.includes('FIXED')?0:1);",
        )
        .unwrap();
        fs::write(dir.path().join("target.txt"), "BROKEN").unwrap();

        let mut opts = EditPipelineOptions::new(dir.path(), resolve);
        opts.max_repair_rounds = 2;
        opts.run_verify = true;
        opts.verify_only = Some(vec![CheckKind::Typecheck]);
        let res = run_edit_pipeline(
            |_attempt, feedback| async move {
                // Round 1: an edit that applies but won't satisfy the check.
                // Repair round: once we see verify feedback, fix it properly.
                Ok(if feedback.is_some() {
                    block_text("target.txt", "STILL_BROKEN", "FIXED")
                } else {
                    block_text("target.txt", "BROKEN", "STILL_BROKEN")
                })
            },
            &opts,
        )
        .await
        .unwrap();
        assert!(res.applied);
        assert!(res.verified, "{}", res.message);
        assert!(res.rounds >= 1);
        assert!(fs::read_to_string(dir.path().join("target.txt")).unwrap().contains("FIXED"));
        assert_eq!(res.blocks.len(), 2);
    }

    // Additional coverage.

    #[tokio::test]
    async fn no_checks_detected_counts_as_applied_unverified() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("n.txt"), "a").unwrap();
        let mut opts = EditPipelineOptions::new(dir.path(), resolve);
        opts.run_verify = true;
        let res = run_edit_pipeline(|_, _| async { Ok(block_text("n.txt", "a", "b")) }, &opts).await.unwrap();
        // As in the TS, `verified` mirrors `allPassed`, which is true when no
        // checks exist; the message tells the two cases apart.
        assert!(res.applied && res.verified);
        assert_eq!(res.message, "Edits applied (no checks detected).");
        assert!(!res.verify.unwrap().ran);
    }

    #[tokio::test]
    async fn interrupt_during_sampling_is_an_error() {
        let (dir, resolve) = scratch();
        let mut opts = EditPipelineOptions::new(dir.path(), resolve);
        opts.candidates = 3;
        let r = run_edit_pipeline(
            |_, _| async { Err::<String, _>(anyhow::Error::from(qo_core::ProviderError::Interrupted)) },
            &opts,
        )
        .await;
        assert!(r.is_err_and(|e| qo_core::error::is_interrupt(&e)));
    }

    #[tokio::test]
    async fn creates_new_files_from_empty_search() {
        let (dir, resolve) = scratch();
        let opts = EditPipelineOptions::new(dir.path(), resolve);
        let res = run_edit_pipeline(|_, _| async { Ok(block_text("new/mod.rs", "", "pub fn x() {}")) }, &opts)
            .await
            .unwrap();
        assert!(res.applied, "{}", res.message);
        assert_eq!(fs::read_to_string(dir.path().join("new/mod.rs")).unwrap(), "pub fn x() {}");
    }

    #[test]
    fn snapshot_restore_round_trip() {
        let (dir, resolve) = scratch();
        fs::write(dir.path().join("a.txt"), "A").unwrap();
        let blocks = [eb("a.txt", "A", "B"), eb("b.txt", "", "new")];
        let snap = snapshot(&blocks, &resolve);
        fs::write(dir.path().join("a.txt"), "changed").unwrap();
        fs::write(dir.path().join("b.txt"), "created").unwrap();
        restore(&snap, &resolve);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "A");
        assert!(!dir.path().join("b.txt").exists());
    }

    #[test]
    fn events_serialize_with_ts_type_tags() {
        let j = serde_json::to_value(PipelineEvent::Repairing { round: 1 }).unwrap();
        assert_eq!(j, serde_json::json!({"type": "repairing", "round": 1}));
        let j = serde_json::to_value(PipelineEvent::CandidateScored { candidate: 0, score: 61.0, reason: "r".into() })
            .unwrap();
        assert_eq!(j["type"], "candidate-scored");
    }
}
