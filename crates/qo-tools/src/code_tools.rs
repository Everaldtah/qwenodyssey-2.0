//! Code-navigation tools — the grounding layer that lets small models behave
//! like they have an IDE. Backed by [`crate::symbol_index`]. The index is built
//! lazily on first use and cached per project root for the whole process.
//!
//! Tools:
//!   find_symbol      — locate a definition by name (file:line + signature)
//!   search_symbols   — fuzzy symbol search by substring
//!   outline_file     — list every definition in a file, in order
//!   find_references  — every whole-word use of a name across the repo
//!   read_symbol      — read just one function/class body, not the whole file
//!   apply_edit       — apply a SEARCH/REPLACE block (reliable small-model edits)

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use async_trait::async_trait;
use qo_core::tool::SharedTool;
use qo_core::{Tool, ToolContext, ToolResult};
use serde_json::{json, Value};

use crate::edit_block::{apply_edit_blocks, explain_failure, EditBlock, MatchStrategy};
use crate::file_tools::resolve_writable;
use crate::symbol_index::{SymbolDef, SymbolIndex, SymbolKind, DEFAULT_MAX_FILES};
use crate::util::{js_number_or, js_string, js_string_or_empty, js_truthy, lexical_resolve, read_text_lossy};

type SharedIndex = Arc<RwLock<SymbolIndex>>;

/// Process-wide cached index per project root, so we build once per session.
static INDEX_CACHE: LazyLock<Mutex<HashMap<PathBuf, SharedIndex>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn cache_key(cwd: &Path) -> PathBuf {
    lexical_resolve(cwd, "")
}

async fn build_index(root: PathBuf) -> SymbolIndex {
    tokio::task::spawn_blocking(move || {
        let mut idx = SymbolIndex::new(root);
        idx.build(DEFAULT_MAX_FILES);
        idx
    })
    .await
    .unwrap_or_default()
}

async fn get_index(cwd: &Path) -> SharedIndex {
    let key = cache_key(cwd);
    if let Some(idx) = INDEX_CACHE.lock().ok().and_then(|m| m.get(&key).cloned()) {
        return idx;
    }
    let idx = Arc::new(RwLock::new(build_index(key.clone()).await));
    if let Ok(mut m) = INDEX_CACHE.lock() {
        // Another caller may have finished first; keep whichever landed.
        return Arc::clone(m.entry(key).or_insert(idx));
    }
    idx
}

/// Force a rebuild (e.g. after a batch of edits changed declarations). A no-op
/// when no index has been built for this root yet.
pub async fn invalidate_index(cwd: &Path) {
    let key = cache_key(cwd);
    let Some(shared) = INDEX_CACHE.lock().ok().and_then(|m| m.get(&key).cloned()) else {
        return;
    };
    let fresh = build_index(key).await;
    let mut guard = match shared.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = fresh;
}

/// Build + cache the index ahead of first use (called at session startup so
/// find_symbol / outline_file / read_symbol are instant on first use).
pub async fn prewarm_symbol_index(cwd: &Path) {
    let _ = get_index(cwd).await;
}

fn fmt_def(d: &SymbolDef) -> String {
    format!("{}:{}  [{}]  {}", d.file, d.line, d.kind, d.signature)
}

/// `String(args.x || "").trim()`
fn arg_trimmed(args: &Value, key: &str) -> String {
    if js_truthy(args.get(key)) {
        js_string(args.get(key)).trim().to_string()
    } else {
        String::new()
    }
}

/// Read access to a shared index, tolerating a poisoned lock.
fn with_index<T>(idx: &SharedIndex, f: impl FnOnce(&SymbolIndex) -> T) -> T {
    match idx.read() {
        Ok(g) => f(&g),
        Err(poisoned) => f(&poisoned.into_inner()),
    }
}

pub struct FindSymbolTool;

#[async_trait]
impl Tool for FindSymbolTool {
    fn name(&self) -> &str {
        "find_symbol"
    }
    fn description(&self) -> &str {
        "Find where a function, class, interface, type, or constant is DEFINED. \
         Returns file:line and the declaration. Use this before editing so you edit the right place."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let name = arg_trimmed(&args, "name");
        if name.is_empty() {
            return ToolResult::err("Provide a symbol 'name'.");
        }
        let idx = get_index(&ctx.cwd).await;
        // An unknown kind string filters everything out, as in the TS.
        let kind_filter = js_truthy(args.get("kind")).then(|| js_string(args.get("kind")));
        let (hits, near) = with_index(&idx, |i| {
            let hits = match &kind_filter {
                Some(k) => match SymbolKind::parse(k) {
                    Some(kind) => i.find(&name, Some(kind)),
                    None => Vec::new(),
                },
                None => i.find(&name, None),
            };
            let near = if hits.is_empty() { i.search(&name, 6) } else { Vec::new() };
            (hits, near)
        });
        if hits.is_empty() {
            // Offer fuzzy suggestions so the model can correct a typo.
            let mut uniq: Vec<String> = Vec::new();
            for d in near {
                if !uniq.contains(&d.name) {
                    uniq.push(d.name);
                }
            }
            let hint = if uniq.is_empty() { String::new() } else { format!(" Did you mean: {}?", uniq.join(", ")) };
            return ToolResult::err(format!("No definition named \"{name}\".{hint}"));
        }
        ctx.log(json!({ "tool": "find_symbol", "name": name, "hits": hits.len() }));
        let out = hits.iter().map(fmt_def).collect::<Vec<_>>().join("\n");
        ToolResult::ok(out).with_data(json!(hits))
    }
}

pub struct SearchSymbolsTool;

#[async_trait]
impl Tool for SearchSymbolsTool {
    fn name(&self) -> &str {
        "search_symbols"
    }
    fn description(&self) -> &str {
        "Fuzzy-search symbol names across the project (substring match). Use when you \
         don't know the exact name. Returns ranked file:line declarations."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let q = arg_trimmed(&args, "query");
        if q.is_empty() {
            return ToolResult::err("Provide a 'query'.");
        }
        let idx = get_index(&ctx.cwd).await;
        let limit = js_number_or(args.get("limit"), 25.0).max(0.0) as usize;
        let hits = with_index(&idx, |i| i.search(&q, limit));
        if hits.is_empty() {
            return ToolResult::ok(format!("No symbols matching \"{q}\"."));
        }
        ctx.log(json!({ "tool": "search_symbols", "query": q, "hits": hits.len() }));
        ToolResult::ok(hits.iter().map(fmt_def).collect::<Vec<_>>().join("\n")).with_data(json!(hits))
    }
}

pub struct OutlineFileTool;

#[async_trait]
impl Tool for OutlineFileTool {
    fn name(&self) -> &str {
        "outline_file"
    }
    fn description(&self) -> &str {
        "List every definition (functions, classes, methods, types) in a file, in order, \
         with line numbers. A cheap structural map — read this before reading a whole large file."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let rel = arg_trimmed(&args, "path");
        if rel.is_empty() {
            return ToolResult::err("Provide a file 'path'.");
        }
        let idx = get_index(&ctx.cwd).await;
        let defs = with_index(&idx, |i| i.outline(&rel));
        if defs.is_empty() {
            return ToolResult::ok(format!("No indexed symbols in {rel} (or file not found)."));
        }
        ctx.log(json!({ "tool": "outline_file", "path": rel, "defs": defs.len() }));
        let out = defs.iter().map(|d| format!("L{}\t[{}]\t{}", d.line, d.kind, d.name)).collect::<Vec<_>>().join("\n");
        ToolResult::ok(out).with_data(json!(defs))
    }
}

pub struct FindReferencesTool;

#[async_trait]
impl Tool for FindReferencesTool {
    fn name(&self) -> &str {
        "find_references"
    }
    fn description(&self) -> &str {
        "Find every place a name is USED across the codebase (whole-word). Definitions are \
         marked. Use before renaming or changing a signature to see the blast radius."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let name = arg_trimmed(&args, "name");
        if name.is_empty() {
            return ToolResult::err("Provide a 'name'.");
        }
        let idx = get_index(&ctx.cwd).await;
        let limit = js_number_or(args.get("limit"), 80.0).max(0.0) as usize;
        let idx2 = Arc::clone(&idx);
        let name2 = name.clone();
        let refs = tokio::task::spawn_blocking(move || with_index(&idx2, |i| i.references(&name2, limit)))
            .await
            .unwrap_or_default();
        if refs.is_empty() {
            return ToolResult::ok(format!("No references to \"{name}\"."));
        }
        ctx.log(json!({ "tool": "find_references", "name": name, "refs": refs.len() }));
        let body = refs
            .iter()
            .map(|r| format!("{}:{}{}\t{}", r.file, r.line, if r.is_definition { " (def)" } else { "" }, r.text))
            .collect::<Vec<_>>()
            .join("\n");
        ToolResult::ok(body).with_data(json!(refs))
    }
}

pub struct ReadSymbolTool;

#[async_trait]
impl Tool for ReadSymbolTool {
    fn name(&self) -> &str {
        "read_symbol"
    }
    fn description(&self) -> &str {
        "Read the body of ONE symbol (function/class/method) by name, instead of the whole file. \
         Keeps context small. If the name is ambiguous, the first definition is returned."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let name = arg_trimmed(&args, "name");
        if name.is_empty() {
            return ToolResult::err("Provide a symbol 'name'.");
        }
        let idx = get_index(&ctx.cwd).await;
        let max_lines = js_number_or(args.get("max_lines"), 200.0).max(1.0) as usize;
        let file = js_truthy(args.get("file")).then(|| js_string(args.get("file")));
        let found = with_index(&idx, |i| {
            let hits = i.find(&name, None);
            let first = hits.first()?.clone();
            let def = match &file {
                Some(f) => hits.iter().find(|h| &h.file == f).cloned().unwrap_or(first),
                None => first,
            };
            let body = i.read_symbol(&def, max_lines);
            Some((def, body))
        });
        let Some((def, body)) = found else {
            return ToolResult::err(format!("No definition named \"{name}\"."));
        };
        let Some(body) = body else {
            return ToolResult::err(format!("Could not read {}.", def.file));
        };
        ctx.log(json!({ "tool": "read_symbol", "name": name, "file": def.file }));
        ToolResult::ok(format!("// {}:{}\n{body}", def.file, def.line)).with_data(json!({ "def": def, "body": body }))
    }
}

pub struct ApplyEditTool;

#[async_trait]
impl Tool for ApplyEditTool {
    fn name(&self) -> &str {
        "apply_edit"
    }
    fn description(&self) -> &str {
        "Apply a precise code edit using a SEARCH/REPLACE block — the RELIABLE way to edit files \
         (preferred over write_file for changes to existing files). Provide 'path', the exact \
         'search' text to find (copy it verbatim from the file), and the 'replace' text. \
         Leave 'search' empty to create a new file or append. Whitespace-tolerant with auto-repair."
    }
    fn mutating(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let rel = arg_trimmed(&args, "path");
        if rel.is_empty() {
            return ToolResult::err("Provide a file 'path'.");
        }
        let block = EditBlock {
            path: rel.clone(),
            search: js_string_or_empty(args.get("search")),
            replace: js_string_or_empty(args.get("replace")),
        };
        let results = apply_edit_blocks(std::slice::from_ref(&block), |p| resolve_writable(ctx, p));
        let Some(r) = results.into_iter().next() else {
            return ToolResult::err("No edit applied.");
        };
        if !r.ok {
            // Give the model targeted feedback to fix its SEARCH on the next try.
            let mut detail = r.error.clone().unwrap_or_else(|| "edit failed".to_string());
            if let Ok(abs) = resolve_writable(ctx, &rel) {
                if abs.exists() {
                    if let Ok(content) = read_text_lossy(&abs) {
                        detail = explain_failure(&content, &block);
                    }
                }
            }
            ctx.log(json!({ "tool": "apply_edit", "path": rel, "ok": false }));
            return ToolResult::err(detail);
        }
        // Declarations may have changed — rebuild the index in the background.
        let cwd = ctx.cwd.clone();
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move { invalidate_index(&cwd).await });
        }
        let note = match r.strategy {
            Some(s) if s != MatchStrategy::Exact => {
                let conf = match r.confidence {
                    Some(c) if c < 1.0 => format!(", confidence {:.0}%", c * 100.0),
                    _ => String::new(),
                };
                format!(" (matched via {s}{conf})")
            }
            _ => String::new(),
        };
        ctx.log(json!({ "tool": "apply_edit", "path": rel, "ok": true, "strategy": r.strategy }));
        ToolResult::ok(format!("Edited {rel}{note}."))
    }
}

/// The code-navigation + reliable-edit tools, in registration order.
pub fn code_nav_tools() -> Vec<SharedTool> {
    vec![
        Arc::new(FindSymbolTool),
        Arc::new(SearchSymbolsTool),
        Arc::new(OutlineFileTool),
        Arc::new(FindReferencesTool),
        Arc::new(ReadSymbolTool),
        Arc::new(ApplyEditTool),
    ]
}

/// Names of [`code_nav_tools`].
pub fn code_nav_tool_names() -> BTreeSet<&'static str> {
    ["find_symbol", "search_symbols", "outline_file", "find_references", "read_symbol", "apply_edit"].into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("calc.py"),
            "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("main.py"), "from calc import add\nprint(add(1, 2))\n").unwrap();
        dir
    }

    #[tokio::test]
    async fn navigation_tools_end_to_end() {
        let dir = fixture();
        let ctx = ToolContext::new(dir.path());
        prewarm_symbol_index(dir.path()).await;

        let r = FindSymbolTool.run(json!({ "name": "add" }), &ctx).await;
        assert!(r.ok);
        assert_eq!(r.output, "calc.py:1  [function]  def add(a, b):");

        let r = FindSymbolTool.run(json!({ "name": "ad" }), &ctx).await;
        assert!(!r.ok);
        assert!(r.output.contains("Did you mean: add?"), "{}", r.output);

        let r = FindSymbolTool.run(json!({ "name": "add", "kind": "class" }), &ctx).await;
        assert!(!r.ok);

        let r = OutlineFileTool.run(json!({ "path": "calc.py" }), &ctx).await;
        assert_eq!(r.output, "L1\t[function]\tadd\nL5\t[function]\tsub");

        let r = FindReferencesTool.run(json!({ "name": "add" }), &ctx).await;
        assert!(r.output.contains("calc.py:1 (def)"));
        assert!(r.output.contains("main.py:2\tprint(add(1, 2))"));

        let r = ReadSymbolTool.run(json!({ "name": "sub" }), &ctx).await;
        assert_eq!(r.output, "// calc.py:5\ndef sub(a, b):\n    return a - b\n");

        let r = SearchSymbolsTool.run(json!({ "query": "zz" }), &ctx).await;
        assert_eq!(r.output, "No symbols matching \"zz\".");
    }

    #[tokio::test]
    async fn apply_edit_reports_strategy_and_failure_hint() {
        let dir = fixture();
        let ctx = ToolContext::new(dir.path());
        let r = ApplyEditTool
            .run(json!({ "path": "calc.py", "search": "return a - b", "replace": "return b - a" }), &ctx)
            .await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.starts_with("Edited calc.py (matched via reindented"), "{}", r.output);
        let body = std::fs::read_to_string(dir.path().join("calc.py")).unwrap();
        assert!(body.contains("    return b - a"));

        let r = ApplyEditTool.run(json!({ "path": "calc.py", "search": "def add(x, y):", "replace": "" }), &ctx).await;
        assert!(!r.ok);
        assert!(r.output.contains("Closest line in the file is L1"), "{}", r.output);

        let r = ApplyEditTool.run(json!({ "path": "../out.py", "search": "", "replace": "x" }), &ctx).await;
        assert!(!r.ok);
        assert!(r.output.contains("escapes"));

        let r = ApplyEditTool.run(json!({ "path": "new.py", "replace": "x = 1\n" }), &ctx).await;
        assert_eq!(r.output, "Edited new.py (matched via created).");
    }
}
