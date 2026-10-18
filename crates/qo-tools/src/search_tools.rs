//! Search tools: grep across the repo and a simple docs lookup.

use async_trait::async_trait;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::RegexBuilder;
use serde_json::{json, Value};

use crate::file_tools::{resolve_readable_smart, IGNORE};
use crate::util::{glob_walk, js_number, js_string, js_truthy, read_text_lossy, take_chars, tool_error, GlobOpts};

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search file contents for a regex pattern."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let pattern = if js_truthy(args.get("pattern")) { js_string(args.get("pattern")) } else { String::new() };
        if pattern.is_empty() {
            return ToolResult::err("No pattern given");
        }
        let dir = if js_truthy(args.get("path")) { js_string(args.get("path")) } else { ".".to_string() };
        let base = resolve_readable_smart(&ctx.cwd, Some(&dir));
        if !base.exists() {
            return ToolResult::err(format!("Directory not found: {}", base.display()));
        }
        let glob = if js_truthy(args.get("glob")) { js_string(args.get("glob")) } else { "**/*".to_string() };
        let re = match RegexBuilder::new(&pattern).case_insensitive(js_truthy(args.get("ignoreCase"))).build() {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("Invalid regex: {e}")),
        };
        // Number(args.limit ?? 200); NaN never trips the `>=` cap in JS → unlimited.
        let max_hits = match args.get("limit") {
            None | Some(Value::Null) => 200.0,
            v => js_number(v),
        };
        let max_hits = if max_hits.is_nan() { usize::MAX } else { max_hits.max(0.0) as usize };

        let job = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<String>> {
            let files = glob_walk(&base, &[glob.as_str()], IGNORE, &GlobOpts::default())?;
            let mut hits = Vec::new();
            'files: for rel in files {
                let Ok(content) = read_text_lossy(&base.join(&rel)) else { continue };
                for (i, line) in content.split('\n').enumerate() {
                    if re.is_match(line) {
                        hits.push(format!("{rel}:{}: {}", i + 1, take_chars(line.trim(), 200)));
                        if hits.len() >= max_hits {
                            break 'files;
                        }
                    }
                }
            }
            Ok(hits)
        });
        let hits = match job.await {
            Ok(Ok(h)) => h,
            Ok(Err(e)) => return tool_error(ctx, self.name(), e),
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        let out = if hits.is_empty() { "(no matches)".to_string() } else { hits.join("\n") };
        ToolResult::ok(out).with_data(json!(hits))
    }
}

pub struct SearchDocsTool;

#[async_trait]
impl Tool for SearchDocsTool {
    fn name(&self) -> &str {
        "search_docs"
    }
    fn description(&self) -> &str {
        "Search README and docs/ for a keyword."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let q = if js_truthy(args.get("query")) { js_string(args.get("query")).to_lowercase() } else { String::new() };
        let cwd = ctx.cwd.clone();
        let job = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<String>> {
            let files = glob_walk(&cwd, &["README*", "docs/**/*.md", "**/*.md"], IGNORE, &GlobOpts::default())?;
            let mut hits = Vec::new();
            for rel in files.iter().take(100) {
                let Ok(content) = read_text_lossy(&cwd.join(rel)) else { continue };
                let lower = content.to_lowercase();
                if let Some(byte_idx) = lower.find(&q) {
                    let ci = lower[..byte_idx].chars().count();
                    let start = ci.saturating_sub(60);
                    let snippet: String = content.chars().skip(start).take(ci + 120 - start).collect();
                    hits.push(format!("{rel}: …{}…", snippet.replace('\n', " ")));
                }
            }
            Ok(hits)
        });
        let hits = match job.await {
            Ok(Ok(h)) => h,
            Ok(Err(e)) => return tool_error(ctx, self.name(), e),
            Err(e) => return tool_error(ctx, self.name(), e),
        };
        let out = if hits.is_empty() { "(no matches)".to_string() } else { hits.join("\n") };
        ToolResult::ok(out).with_data(json!(hits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/x")).unwrap();
        std::fs::write(dir.path().join("src/a.ts"), "// TODO: fix\nconst x = 1;\n").unwrap();
        std::fs::write(dir.path().join("node_modules/x/i.js"), "// TODO: hidden\n").unwrap();
        std::fs::write(dir.path().join("README.md"), "# Demo\nThis project explains the Frobnicator in depth.\n").unwrap();
        dir
    }

    #[tokio::test]
    async fn grep_finds_hits_and_skips_ignored() {
        let dir = fixture();
        let ctx = ToolContext::new(dir.path());
        let r = GrepTool.run(json!({ "pattern": "todo", "ignoreCase": true }), &ctx).await;
        assert!(r.ok);
        assert_eq!(r.output, "src/a.ts:1: // TODO: fix");
        let r = GrepTool.run(json!({ "pattern": "todo" }), &ctx).await;
        assert_eq!(r.output, "(no matches)");
        let r = GrepTool.run(json!({ "pattern": "(" }), &ctx).await;
        assert!(!r.ok && r.output.starts_with("Invalid regex"));
        let r = GrepTool.run(json!({ "pattern": "const", "glob": "**/*.py" }), &ctx).await;
        assert_eq!(r.output, "(no matches)");
    }

    #[tokio::test]
    async fn search_docs_snippets() {
        let dir = fixture();
        let ctx = ToolContext::new(dir.path());
        let r = SearchDocsTool.run(json!({ "query": "frobnicator" }), &ctx).await;
        assert!(r.output.starts_with("README.md: …"), "{}", r.output);
        assert!(r.output.contains("Frobnicator"));
    }
}
