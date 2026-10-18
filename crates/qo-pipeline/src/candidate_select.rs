//! Best-of-N candidate selection (port of `src/core/candidateSelect.ts`).
//!
//! A single sample from a 7B–20B model on a coding task has a meaningful
//! failure rate. Sampling several candidates (at modest temperature) and
//! selecting the best by an OBJECTIVE signal — does it parse? does it apply?
//! does it typecheck? do tests pass? — recovers a large fraction of that gap,
//! because the failures are largely independent. This is the cheapest
//! reliability win after structured editing, and it's why this harness can
//! make a small model feel far steadier.
//!
//! The scorer is pluggable so callers grade candidates with whatever signal
//! they have (a pure parse check, an apply-dry-run, or a full verify pass).

use std::future::Future;

/// A standard scoring scale so different scorers compose consistently:
///   100  all checks pass (tests + typecheck)
///   80   typechecks/builds, no tests run or tests partially pass
///   60   applies cleanly, parses, but checks fail
///   10   parses only
///   -inf rejected (didn't parse / didn't apply)
pub mod score {
    pub const ALL_PASS: f64 = 100.0;
    pub const COMPILES: f64 = 80.0;
    pub const APPLIES: f64 = 60.0;
    pub const PARSES: f64 = 10.0;
    pub const REJECT: f64 = f64::NEG_INFINITY;
}

/// One drawn sample: the parsed value (`None` = unusable) and the raw model text.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn<T> {
    pub value: Option<T>,
    pub raw: String,
}

/// A scorer's verdict. Higher is better; [`score::REJECT`] rejects outright.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub score: f64,
    /// Human-readable reason (e.g. "typechecks; 2 tests fail").
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredCandidate<T> {
    pub value: T,
    /// Raw model text the candidate was parsed from (for debugging).
    pub raw: String,
    pub index: usize,
    pub score: f64,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BestOfNResult<T> {
    pub best: Option<ScoredCandidate<T>>,
    /// Every finite-scored candidate, best first.
    pub all: Vec<ScoredCandidate<T>>,
    /// True if we stopped early on an early-accept hit.
    pub early_accepted: bool,
}

/// Draw up to `n` candidates, score each, and return the best. Sampling is
/// sequential (so a scorer that mutates the workspace and rolls back is safe).
/// `early_accept_score` short-circuits as soon as a clearly-good candidate
/// appears, so the common case stays cheap.
///
/// A sample that errors is skipped like an unusable one — except an Esc
/// interrupt ([`qo_core::error::is_interrupt`]), which stops the whole
/// selection and is returned as the error. A scorer error rejects that
/// candidate with the error as its reason.
pub async fn best_of_n<T, S, SFut, C, CFut, P>(
    n: usize,
    mut sample: S,
    mut score_fn: C,
    early_accept_score: Option<f64>,
    mut on_progress: P,
) -> anyhow::Result<BestOfNResult<T>>
where
    T: Clone,
    S: FnMut(usize) -> SFut,
    SFut: Future<Output = anyhow::Result<Drawn<T>>>,
    C: FnMut(&T) -> CFut,
    CFut: Future<Output = anyhow::Result<Score>>,
    P: FnMut(usize, Option<&ScoredCandidate<T>>),
{
    let mut scored: Vec<ScoredCandidate<T>> = Vec::new();
    let mut early_accepted = false;

    for i in 0..n {
        let drawn = match sample(i).await {
            Ok(d) => d,
            Err(e) if qo_core::error::is_interrupt(&e) => return Err(e),
            Err(_) => {
                on_progress(i, None);
                continue;
            }
        };
        let Some(value) = drawn.value else {
            on_progress(i, None);
            continue;
        };

        let sc = match score_fn(&value).await {
            Ok(s) => s,
            Err(e) => Score { score: score::REJECT, reason: format!("scorer error: {e}") },
        };

        let cand = ScoredCandidate { value, raw: drawn.raw, index: i, score: sc.score, reason: sc.reason };
        on_progress(i, Some(&cand));
        let accept = early_accept_score.is_some_and(|t| cand.score >= t);
        if cand.score.is_finite() {
            scored.push(cand);
        }
        if accept {
            early_accepted = true;
            break;
        }
    }

    // Stable: equal scores keep sampling order.
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(BestOfNResult { best: scored.first().cloned(), all: scored, early_accepted })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qo_core::ProviderError;

    // Ported from tests/smallModelTools.test.ts ("bestOfN").

    async fn sample_index(i: usize) -> anyhow::Result<Drawn<i64>> {
        Ok(Drawn { value: Some(i as i64), raw: i.to_string() })
    }

    #[tokio::test]
    async fn selects_the_highest_scoring_candidate() {
        let res = best_of_n(
            4,
            sample_index,
            |v: &i64| {
                let v = *v;
                async move { Ok(Score { score: v as f64, reason: format!("val {v}") }) }
            },
            None,
            |_, _| {},
        )
        .await
        .unwrap();
        assert_eq!(res.best.unwrap().value, 3);
        assert_eq!(res.all.len(), 4);
    }

    #[tokio::test]
    async fn rejects_neg_infinity_candidates() {
        let res = best_of_n(
            3,
            sample_index,
            |v: &i64| {
                let v = *v;
                async move {
                    Ok(if v == 1 {
                        Score { score: score::REJECT, reason: "bad".into() }
                    } else {
                        Score { score: v as f64, reason: "ok".into() }
                    })
                }
            },
            None,
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(!res.all.iter().any(|c| c.value == 1));
    }

    #[tokio::test]
    async fn short_circuits_on_early_accept() {
        let mut calls = 0;
        let res = best_of_n(
            10,
            |i| {
                calls += 1;
                sample_index(i)
            },
            |v: &i64| {
                let s = if *v >= 0 { score::ALL_PASS } else { 0.0 };
                async move { Ok(Score { score: s, reason: String::new() }) }
            },
            Some(score::ALL_PASS),
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(res.early_accepted);
        assert_eq!(calls, 1, "stopped after the first good one");
    }

    #[tokio::test]
    async fn handles_a_sampler_that_returns_null() {
        let res = best_of_n(
            2,
            |_| async { Ok(Drawn::<i64> { value: None, raw: String::new() }) },
            |_v: &i64| async { Ok(Score { score: 1.0, reason: String::new() }) },
            None,
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(res.best.is_none());
    }

    // Additional coverage.

    #[tokio::test]
    async fn picks_highest_and_skips_rejects() {
        let vals = [Some(1), None, Some(3), Some(2)];
        let mut progress = Vec::new();
        let r = best_of_n(
            4,
            |i| {
                let v = vals[i];
                async move { Ok(Drawn { value: v, raw: format!("raw{i}") }) }
            },
            |v: &i32| {
                let v = *v;
                async move {
                    Ok(Score { score: if v == 2 { score::REJECT } else { v as f64 }, reason: format!("v{v}") })
                }
            },
            None,
            |i, s| progress.push((i, s.map(|c| c.score))),
        )
        .await
        .unwrap();
        let best = r.best.unwrap();
        assert_eq!((best.value, best.index, best.raw.as_str()), (3, 2, "raw2"));
        assert_eq!(r.all.len(), 2);
        assert!(!r.early_accepted);
        assert_eq!(progress.len(), 4);
        assert_eq!(progress[1], (1, None));
        assert_eq!(progress[3].1, Some(f64::NEG_INFINITY));
    }

    #[tokio::test]
    async fn early_accept_stops_sampling() {
        let mut drawn = 0;
        let r = best_of_n(
            5,
            |i| {
                drawn += 1;
                async move { Ok(Drawn { value: Some(i), raw: String::new() }) }
            },
            |_v: &usize| async { Ok(Score { score: score::APPLIES, reason: "applies".into() }) },
            Some(score::APPLIES),
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(r.early_accepted);
        assert_eq!(drawn, 1);
        assert_eq!(r.best.unwrap().index, 0);
    }

    #[tokio::test]
    async fn sample_errors_are_skipped_but_interrupts_stop() {
        let r = best_of_n(
            2,
            |i| async move {
                if i == 0 {
                    Err(anyhow::anyhow!("model hiccup"))
                } else {
                    Ok(Drawn { value: Some(7), raw: String::new() })
                }
            },
            |_v: &i32| async { Err(anyhow::anyhow!("scorer broke")) },
            None,
            |_, _| {},
        )
        .await
        .unwrap();
        assert!(r.best.is_none(), "scorer error rejects the candidate");

        let e = best_of_n(
            3,
            |_| async { Err::<Drawn<i32>, _>(anyhow::Error::from(ProviderError::Interrupted)) },
            |_v: &i32| async { Ok(Score { score: 1.0, reason: String::new() }) },
            None,
            |_, _| {},
        )
        .await;
        assert!(e.is_err_and(|e| qo_core::error::is_interrupt(&e)));
    }
}
