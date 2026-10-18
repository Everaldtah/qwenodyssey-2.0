//! Pack the most useful context into a small token budget. Designed for 7B
//! models: relevant files first, then errors, then structure, then memory
//! (port of `src/core/contextBuilder.ts`).

use std::path::Path;

use crate::deps::resolve_inside;
use crate::patch_engine::read_lossy;
use crate::repo_scanner::summarize_repo;
use qo_core::{Provider, RepoInfo};

/// Default number of files [`guess_relevant_files`] returns.
pub const DEFAULT_RELEVANT_LIMIT: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct ContextInput<'a> {
    pub user_request: &'a str,
    pub repo: &'a RepoInfo,
    /// Candidate relevant files (paths relative to `repo.root`).
    pub files: &'a [String],
    pub errors: Option<&'a str>,
    pub git_diff: Option<&'a str>,
    pub memory: Option<&'a str>,
    pub budget_tokens: usize,
}

struct Section {
    label: String,
    body: String,
    /// Lower = more important; priority 1 is never dropped.
    priority: u8,
}

fn non_empty(s: Option<&str>) -> Option<&str> {
    s.filter(|s| !s.is_empty())
}

pub fn build_context(provider: &dyn Provider, input: &ContextInput<'_>) -> String {
    let mut sections: Vec<Section> = Vec::new();

    // 1. User request (always first / never dropped).
    sections.push(Section { label: "USER REQUEST".into(), body: input.user_request.to_string(), priority: 1 });

    // 2. Relevant files (unreadable / escaping / directory entries are skipped).
    let root = Path::new(&input.repo.root);
    for rel in input.files {
        let Ok(abs) = resolve_inside(root, rel) else { continue };
        if !abs.is_file() {
            continue;
        }
        let Ok(raw) = read_lossy(&abs) else { continue };
        sections.push(Section {
            label: format!("FILE: {rel}"),
            body: format!("```\n{}\n```", clip(&raw, 8000)),
            priority: 2,
        });
    }

    // 3. Current errors.
    if let Some(e) = non_empty(input.errors) {
        sections.push(Section { label: "FAILING OUTPUT".into(), body: clip(e, 4000), priority: 3 });
    }

    // 4. Repo structure.
    sections.push(Section { label: "REPO".into(), body: summarize_repo(input.repo), priority: 4 });

    // 5. Git diff.
    if let Some(d) = non_empty(input.git_diff) {
        sections.push(Section { label: "CURRENT GIT DIFF".into(), body: clip(d, 3000), priority: 5 });
    }

    // 6. Memory.
    if let Some(m) = non_empty(input.memory) {
        sections.push(Section { label: "PROJECT MEMORY".into(), body: clip(m, 2000), priority: 6 });
    }

    // Greedily include by priority until the budget is exhausted.
    sections.sort_by_key(|s| s.priority);
    let mut out: Vec<String> = Vec::new();
    let mut used = 0usize;
    for s in sections {
        let block = format!("## {}\n{}\n", s.label, s.body);
        let cost = provider.count_tokens(&block);
        if used + cost > input.budget_tokens && s.priority > 1 {
            continue;
        }
        out.push(block);
        used += cost;
    }
    out.join("\n")
}

/// Keep the head (70%) and tail (20%) of an over-long text, noting the cut.
/// Lengths are in characters (the TS counted UTF-16 units).
pub fn clip(text: &str, max_chars: usize) -> String {
    let len = text.chars().count();
    if len <= max_chars {
        return text.to_string();
    }
    let head_n = (max_chars as f64 * 0.7).floor() as usize;
    let tail_n = (max_chars as f64 * 0.2).floor() as usize;
    let head: String = text.chars().take(head_n).collect();
    let tail: String = text.chars().skip(len - tail_n).collect();
    format!("{head}\n… [{} chars omitted] …\n{tail}", len - max_chars)
}

/// Heuristic relevant-file picker when the planner doesn't name files.
pub fn guess_relevant_files(repo: &RepoInfo, request: &str, limit: usize) -> Vec<String> {
    let lower = request.to_lowercase();
    let terms: Vec<&str> = lower
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'))
        .filter(|t| t.chars().count() > 3)
        .collect();
    let mut scored: Vec<(&String, usize)> = repo
        .key_files
        .iter()
        .chain(repo.entrypoints.iter())
        .map(|f| {
            let name = Path::new(f).file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            let score = terms.iter().filter(|t| name.contains(**t)).count();
            (f, score)
        })
        .collect();
    // Stable sort, highest score first (Array.prototype.sort is stable).
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    let mut uniq: Vec<String> = Vec::new();
    for (f, _) in scored {
        if !uniq.contains(f) {
            uniq.push(f.clone());
        }
    }
    uniq.truncate(limit);
    uniq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeProvider;
    use std::fs;

    fn repo(root: &Path) -> RepoInfo {
        RepoInfo {
            root: root.to_string_lossy().to_string(),
            languages: vec!["Rust".into()],
            key_files: vec!["Cargo.toml".into(), "README.md".into()],
            entrypoints: vec!["src/main.rs".into(), "src/parser/lib.rs".into()],
            ..Default::default()
        }
    }

    #[test]
    fn packs_sections_in_priority_order() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        let r = repo(dir.path());
        let files = vec!["a.rs".to_string(), "missing.rs".to_string(), "../escape.rs".to_string()];
        let p = FakeProvider::new(vec![]);
        let ctx = build_context(
            &p,
            &ContextInput {
                user_request: "do it",
                repo: &r,
                files: &files,
                errors: Some("boom"),
                git_diff: Some(""),
                memory: Some("remember"),
                budget_tokens: 10_000,
            },
        );
        let order: Vec<usize> = ["## USER REQUEST", "## FILE: a.rs", "## FAILING OUTPUT", "## REPO", "## PROJECT MEMORY"]
            .iter()
            .map(|h| ctx.find(h).unwrap_or_else(|| panic!("missing {h} in {ctx}")))
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]));
        assert!(ctx.contains("```\nfn a() {}\n```"));
        assert!(!ctx.contains("CURRENT GIT DIFF"), "empty diff is skipped");
        assert!(!ctx.contains("missing.rs"));
    }

    #[test]
    fn budget_drops_low_priority_but_never_the_request() {
        let dir = tempfile::tempdir().unwrap();
        let r = repo(dir.path());
        let p = FakeProvider::new(vec![]);
        let ctx = build_context(
            &p,
            &ContextInput {
                user_request: "a long request that costs tokens",
                repo: &r,
                files: &[],
                errors: None,
                git_diff: None,
                memory: Some("memory"),
                budget_tokens: 1,
            },
        );
        assert!(ctx.starts_with("## USER REQUEST\n"));
        assert!(!ctx.contains("## REPO"));
        assert!(!ctx.contains("PROJECT MEMORY"));
    }

    #[test]
    fn clip_keeps_head_and_tail() {
        let s: String = "x".repeat(100) + &"y".repeat(100);
        let c = clip(&s, 100);
        assert!(c.starts_with(&"x".repeat(70)));
        assert!(c.ends_with(&"y".repeat(20)));
        assert!(c.contains("… [100 chars omitted] …"));
        assert_eq!(clip("short", 100), "short");
    }

    #[test]
    fn guesses_files_by_basename_terms() {
        let r = repo(Path::new("/x"));
        let got = guess_relevant_files(&r, "Fix the parser bug in lib", 6);
        // "parser" isn't in any basename; no term > 3 chars matches "lib.rs"
        // ("lib" is too short), so original order is kept.
        assert_eq!(got, vec!["Cargo.toml", "README.md", "src/main.rs", "src/parser/lib.rs"]);
        let got = guess_relevant_files(&r, "update the readme", 2);
        assert_eq!(got, vec!["README.md", "Cargo.toml"]);
    }
}
