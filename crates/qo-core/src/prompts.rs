//! Prompt templates (ported from `src/core/promptLoader.ts`).
//!
//! The TS loader read `src/prompts/<name>.md` from disk, resolving both the
//! in-repo (dev) and installed-package layouts. The Rust binary is a single
//! file, so the templates are embedded at compile time instead: they live in
//! `crates/qo-core/prompts/` as byte-identical copies of the TS originals (a
//! `.gitattributes` there stops git from normalizing their line endings). The
//! TS had no user-level override directory, so there is none here either.

use anyhow::{anyhow, Result};

/// Every shipped template, by name (the file stem of `prompts/<name>.md`).
pub const PROMPTS: &[(&str, &str)] = &[
    ("coder", include_str!("../prompts/coder.md")),
    ("coder_blocks", include_str!("../prompts/coder_blocks.md")),
    ("error_fix", include_str!("../prompts/error_fix.md")),
    ("memory", include_str!("../prompts/memory.md")),
    ("planner", include_str!("../prompts/planner.md")),
    ("reviewer", include_str!("../prompts/reviewer.md")),
    ("summarizer", include_str!("../prompts/summarizer.md")),
    ("system", include_str!("../prompts/system.md")),
];

/// The embedded template `name`, if it exists.
pub fn prompt(name: &str) -> Option<&'static str> {
    PROMPTS.iter().find(|(n, _)| *n == name).map(|(_, text)| *text)
}

/// Load the template `name` (e.g. "system", "planner"). Errors exactly like the
/// TS when there is no such template: `Prompt template not found: <name>.md`.
pub fn load_prompt(name: &str) -> Result<String> {
    prompt(name)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Prompt template not found: {name}.md"))
}

/// Fill `{{placeholders}}` in a template. A placeholder name is ASCII
/// `[A-Za-z0-9_]+` (JS `\w+`); unknown placeholders become "" (the TS `?? ""`).
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    render_with(template, |key| {
        vars.iter().find(|(k, _)| *k == key).map(|(_, v)| (*v).to_string())
    })
}

/// [`render`] with a lookup function instead of a list of pairs.
pub fn render_with(template: &str, lookup: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let name_len = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if name_len > 0 && after[name_len..].starts_with("}}") {
            out.push_str(&rest[..start]);
            out.push_str(&lookup(&after[..name_len]).unwrap_or_default());
            rest = &after[name_len + 2..];
        } else {
            // Not a placeholder: keep the first '{' and rescan from the next char,
            // so "{{{x}}}" still finds "{{x}}" like the global JS regex does.
            out.push_str(&rest[..start + 1]);
            rest = &rest[start + 1..];
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_prompt_is_embedded_and_non_empty() {
        for name in [
            "coder",
            "coder_blocks",
            "error_fix",
            "memory",
            "planner",
            "reviewer",
            "summarizer",
            "system",
        ] {
            let text = load_prompt(name).unwrap();
            assert!(!text.trim().is_empty(), "{name} is empty");
        }
        assert_eq!(PROMPTS.len(), 8);
    }

    #[test]
    fn embedded_prompts_are_byte_identical_to_the_files() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("prompts");
        for (name, text) in PROMPTS {
            let on_disk = std::fs::read(dir.join(format!("{name}.md"))).unwrap();
            assert_eq!(on_disk, text.as_bytes(), "{name}.md");
        }
    }

    #[test]
    fn unknown_prompt_errors_like_the_ts() {
        let err = load_prompt("nope").unwrap_err().to_string();
        assert_eq!(err, "Prompt template not found: nope.md");
        assert!(prompt("nope").is_none());
    }

    #[test]
    fn render_fills_placeholders() {
        assert_eq!(render("Task: {{task}}\n{{context}}", &[("task", "T"), ("context", "C")]), "Task: T\nC");
        assert_eq!(render("{{missing}}!", &[]), "!");
        assert_eq!(render("{{a}}{{a}}", &[("a", "x")]), "xx");
        assert_eq!(render("{{ a }} {{a-b}} {x}", &[("a", "x")]), "{{ a }} {{a-b}} {x}");
        assert_eq!(render("{{{a}}}", &[("a", "x")]), "{x}");
        assert_eq!(render("unterminated {{a", &[("a", "x")]), "unterminated {{a");
        assert_eq!(render("ünï {{a}} çødé", &[("a", "x")]), "ünï x çødé");
    }

    #[test]
    fn render_planner_template_leaves_no_placeholders() {
        let t = load_prompt("planner").unwrap();
        let out = render(&t, &[("task", "do it"), ("repo", "r"), ("files", "f")]);
        assert!(!out.contains("{{"), "{out}");
        assert!(out.contains("do it"));
    }
}
