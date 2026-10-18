//! Pure terminal renderers for pipeline output (ported from `src/cli/render.ts`
//! `colorizeDiff` plus the orchestrator's plan / review printers).
//!
//! Library crates never print: these return ANSI-coloured strings and the
//! binary decides where (and whether, e.g. no-colour terminals) they go.

use qo_core::{FileEdit, Plan, ReviewResult, Severity};

/// Minimal ANSI SGR helpers matching chalk's open/close codes, so output looks
/// the same as the TypeScript CLI.
pub mod ansi {
    pub fn bold(s: &str) -> String {
        format!("\x1b[1m{s}\x1b[22m")
    }
    pub fn red(s: &str) -> String {
        format!("\x1b[31m{s}\x1b[39m")
    }
    pub fn green(s: &str) -> String {
        format!("\x1b[32m{s}\x1b[39m")
    }
    pub fn yellow(s: &str) -> String {
        format!("\x1b[33m{s}\x1b[39m")
    }
    pub fn cyan(s: &str) -> String {
        format!("\x1b[36m{s}\x1b[39m")
    }
    /// chalk.gray (bright black).
    pub fn gray(s: &str) -> String {
        format!("\x1b[90m{s}\x1b[39m")
    }
}

/// Colour a unified diff: headers bold, hunk headers cyan, additions green,
/// removals red, context gray.
pub fn colorize_diff(diff: &str) -> String {
    diff.split('\n')
        .map(|line| {
            if line.starts_with("+++") || line.starts_with("---") {
                ansi::bold(line)
            } else if line.starts_with("@@") {
                ansi::cyan(line)
            } else if line.starts_with('+') {
                ansi::green(line)
            } else if line.starts_with('-') {
                ansi::red(line)
            } else {
                ansi::gray(line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Upper-case edit mode label ("PATCH", "CREATE", …).
pub fn edit_mode_label(edit: &FileEdit) -> String {
    format!("{:?}", edit.mode).to_uppercase()
}

/// Header line shown above an edit's diff: `\nREWRITE src/a.ts — rationale`.
pub fn render_edit_header(edit: &FileEdit) -> String {
    let mut s = ansi::bold(&format!("\n{} {}", edit_mode_label(edit), edit.path));
    if let Some(r) = edit.rationale.as_deref().filter(|r| !r.is_empty()) {
        s.push_str(&ansi::gray(&format!(" — {r}")));
    }
    s
}

/// The orchestrator's "Proposed plan" block (printPlan). Ends with a blank line.
pub fn render_plan(plan: &Plan) -> String {
    let mut out = Vec::new();
    out.push(format!("{}{}", ansi::bold("\nProposed plan: "), ansi::gray(&plan.goal)));
    if !plan.files.is_empty() {
        out.push(ansi::bold("Files likely needed:"));
    }
    for f in &plan.files {
        out.push(format!("  - {f}"));
    }
    out.push(ansi::bold("Steps:"));
    for s in &plan.steps {
        out.push(render_step(s.index, &s.title, s.detail.as_deref()));
    }
    if !plan.assumptions.is_empty() {
        out.push(ansi::yellow("Assumptions:"));
        for a in &plan.assumptions {
            out.push(format!("  · {a}"));
        }
    }
    out.push(String::new());
    out.join("\n")
}

/// The `plan` command's layout ("Goal:" / "Files:" / "Steps:" / "Assumptions:").
pub fn render_plan_command(plan: &Plan) -> String {
    let mut out = Vec::new();
    out.push(format!("{}{}", ansi::bold("\nGoal: "), plan.goal));
    if !plan.files.is_empty() {
        out.push(ansi::bold("Files:"));
        for f in &plan.files {
            out.push(format!("  - {f}"));
        }
    }
    out.push(ansi::bold("Steps:"));
    for s in &plan.steps {
        out.push(render_step(s.index, &s.title, s.detail.as_deref()));
    }
    if !plan.assumptions.is_empty() {
        out.push(ansi::yellow("Assumptions:"));
        for a in &plan.assumptions {
            out.push(format!("  · {a}"));
        }
    }
    out.join("\n")
}

fn render_step(index: u32, title: &str, detail: Option<&str>) -> String {
    let mut line = format!("  {index}. {title}");
    if let Some(d) = detail.filter(|d| !d.is_empty()) {
        line.push_str(&ansi::gray(&format!(" — {d}")));
    }
    line
}

/// Reviewer verdict + issues (printReview / the `review` command).
pub fn render_review(review: &ReviewResult) -> String {
    let verdict = if review.approve { ansi::green("approved") } else { ansi::red("changes suggested") };
    let mut head = format!("{}{}", ansi::bold("\nReview: "), verdict);
    if let Some(s) = review.summary.as_deref().filter(|s| !s.is_empty()) {
        head.push_str(&ansi::gray(&format!(" — {s}")));
    }
    let mut out = vec![head];
    for i in &review.issues {
        let tag = match i.severity {
            Severity::Error => ansi::red("error"),
            Severity::Warning => ansi::yellow("warn"),
            Severity::Info => ansi::gray("info"),
        };
        let file = i.file.as_deref().filter(|f| !f.is_empty()).map(|f| format!("{f}: ")).unwrap_or_default();
        out.push(format!("  [{tag}] {file}{}", i.message));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use qo_core::{EditMode, PlanStep, ReviewIssue};

    #[test]
    fn colorizes_each_line_kind() {
        let d = "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n ctx";
        let c = colorize_diff(d);
        let lines: Vec<&str> = c.split('\n').collect();
        assert_eq!(lines[0], "\x1b[1m--- a\x1b[22m");
        assert_eq!(lines[1], "\x1b[1m+++ b\x1b[22m");
        assert_eq!(lines[2], "\x1b[36m@@ -1 +1 @@\x1b[39m");
        assert_eq!(lines[3], "\x1b[31m-old\x1b[39m");
        assert_eq!(lines[4], "\x1b[32m+new\x1b[39m");
        assert_eq!(lines[5], "\x1b[90m ctx\x1b[39m");
    }

    #[test]
    fn renders_plan_and_review() {
        let plan = Plan {
            goal: "g".into(),
            steps: vec![PlanStep { index: 1, title: "t".into(), detail: Some("d".into()), files: vec![] }],
            files: vec!["a.rs".into()],
            assumptions: vec!["x".into()],
        };
        let p = render_plan(&plan);
        assert!(p.contains("Proposed plan: "));
        assert!(p.contains("  - a.rs"));
        assert!(p.contains("  1. t"));
        assert!(p.contains("  · x"));
        let r = render_review(&ReviewResult {
            approve: false,
            issues: vec![ReviewIssue { severity: Severity::Error, file: Some("f".into()), message: "m".into() }],
            summary: Some("s".into()),
        });
        assert!(r.contains("changes suggested"));
        assert!(r.contains("f: m"));
        let e = FileEdit { path: "p".into(), mode: EditMode::Rewrite, diff: None, content: None, rationale: None };
        assert!(render_edit_header(&e).contains("REWRITE p"));
    }
}
