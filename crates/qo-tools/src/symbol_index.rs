//! Lightweight, dependency-free symbol index.
//!
//! Frontier models get "go to definition" / "find references" for free from the
//! IDE context they were trained on; small local models don't. Giving them cheap
//! grounding tools — "where is function X defined?", "what calls it?", "outline
//! this file" — dramatically cuts the rate at which they edit the wrong place or
//! hallucinate APIs.
//!
//! This is a regex-based extractor (not a full parser) covering the common
//! declaration shapes across TS/JS, Python, Go, Rust, Java/Kotlin, C/C++, Ruby,
//! and PHP. It is intentionally approximate: fast, no native deps, good enough to
//! point a model at the right file:line. Definitions are indexed precisely;
//! "references" are a whole-word text search (with a definition flag), which is
//! the right precision/recall tradeoff for a 7B's needs.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::util::{glob_walk, read_text_lossy, take_chars, GlobOpts};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Function,
    Class,
    Method,
    Interface,
    Type,
    Const,
    Enum,
    Struct,
    Trait,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Class => "class",
            SymbolKind::Method => "method",
            SymbolKind::Interface => "interface",
            SymbolKind::Type => "type",
            SymbolKind::Const => "const",
            SymbolKind::Enum => "enum",
            SymbolKind::Struct => "struct",
            SymbolKind::Trait => "trait",
        }
    }

    /// Parse the TS string form; unknown kinds yield `None`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "function" => SymbolKind::Function,
            "class" => SymbolKind::Class,
            "method" => SymbolKind::Method,
            "interface" => SymbolKind::Interface,
            "type" => SymbolKind::Type,
            "const" => SymbolKind::Const,
            "enum" => SymbolKind::Enum,
            "struct" => SymbolKind::Struct,
            "trait" => SymbolKind::Trait,
            _ => return None,
        })
    }
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolDef {
    pub name: String,
    pub kind: SymbolKind,
    /// Relative path (`/` separators).
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// The trimmed declaration line.
    pub signature: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Reference {
    pub file: String,
    pub line: usize,
    pub text: String,
    pub is_definition: bool,
}

struct Pattern {
    re: Regex,
    kind: SymbolKind,
    name_group: usize,
}

struct LangRule {
    /// File extension match.
    test: Regex,
    patterns: Vec<Pattern>,
}

fn re(s: &str) -> Regex {
    Regex::new(s).expect("static symbol-index regex")
}

fn p(r: &str, kind: SymbolKind) -> Pattern {
    Pattern { re: re(r), kind, name_group: 1 }
}

static RULES: LazyLock<Vec<LangRule>> = LazyLock::new(|| {
    use SymbolKind::*;
    vec![
        LangRule {
            test: re(r"\.(ts|tsx|js|jsx|mjs|cjs)$"),
            patterns: vec![
                p(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", Function),
                p(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", Class),
                p(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)", Interface),
                p(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=", Type),
                p(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][\w$]*)", Enum),
                p(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", Function),
                p(r"^\s*(?:export\s+)?const\s+([A-Z][\w$]*)\s*[:=]", Const),
                p(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*[:{]", Method),
            ],
        },
        LangRule {
            test: re(r"\.py$"),
            patterns: vec![p(r"^\s*def\s+([A-Za-z_]\w*)", Function), p(r"^\s*class\s+([A-Za-z_]\w*)", Class)],
        },
        LangRule {
            test: re(r"\.go$"),
            patterns: vec![
                p(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", Function),
                p(r"^\s*type\s+([A-Za-z_]\w*)\s+struct", Struct),
                p(r"^\s*type\s+([A-Za-z_]\w*)\s+interface", Interface),
            ],
        },
        LangRule {
            test: re(r"\.rs$"),
            patterns: vec![
                p(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", Function),
                p(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)", Struct),
                p(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_]\w*)", Trait),
                p(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_]\w*)", Enum),
            ],
        },
        LangRule {
            test: re(r"\.(java|kt)$"),
            patterns: vec![
                p(r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+([A-Za-z_]\w*)", Class),
                p(r"^\s*(?:public\s+|private\s+|protected\s+)?interface\s+([A-Za-z_]\w*)", Interface),
                p(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*[\w<>\[\].]+\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*\{", Method),
                p(r"^\s*fun\s+([A-Za-z_]\w*)", Function),
            ],
        },
        LangRule {
            test: re(r"\.(c|h|cpp|cc|hpp|cxx)$"),
            patterns: vec![
                p(r"^\s*(?:[\w*&<>:]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$", Function),
                p(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)", Class),
            ],
        },
        LangRule {
            test: re(r"\.rb$"),
            patterns: vec![p(r"^\s*def\s+([A-Za-z_]\w*[!?]?)", Function), p(r"^\s*(?:class|module)\s+([A-Za-z_]\w*)", Class)],
        },
        LangRule {
            test: re(r"\.php$"),
            patterns: vec![
                p(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*function\s+([A-Za-z_]\w*)", Function),
                p(r"^\s*(?:abstract\s+|final\s+)?class\s+([A-Za-z_]\w*)", Class),
                p(r"^\s*interface\s+([A-Za-z_]\w*)", Interface),
            ],
        },
    ]
});

const IGNORE_DIRS: &[&str] = &["node_modules", ".git", "dist", "build", ".next", "out", "target", "__pycache__", ".venv", "venv", "vendor"];

fn rule_for(rel: &str) -> Option<&'static LangRule> {
    RULES.iter().find(|r| r.test.is_match(rel))
}

/// Default cap on scanned files per build.
pub const DEFAULT_MAX_FILES: usize = 4000;

#[derive(Debug, Default)]
pub struct SymbolIndex {
    root: PathBuf,
    defs: Vec<SymbolDef>,
    by_name: HashMap<String, Vec<usize>>,
    file_list: Vec<String>,
    built: bool,
}

impl SymbolIndex {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), ..Default::default() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Build (or rebuild) the index by scanning the project tree. Blocking IO —
    /// call from `spawn_blocking` in async code.
    pub fn build(&mut self, max_files: usize) {
        self.defs.clear();
        self.by_name.clear();
        let ignore: Vec<String> = IGNORE_DIRS.iter().map(|d| format!("**/{d}/**")).collect();
        let ignore_refs: Vec<&str> = ignore.iter().map(String::as_str).collect();
        let entries = glob_walk(&self.root, &["**/*"], &ignore_refs, &GlobOpts::default()).unwrap_or_default();
        self.file_list = entries.into_iter().take(max_files).collect();
        let files = self.file_list.clone();
        for rel in &files {
            let Some(rule) = rule_for(rel) else { continue };
            let abs = self.root.join(rel);
            match std::fs::metadata(&abs) {
                Ok(m) if m.len() > 1_500_000 => continue, // skip huge/minified
                Ok(_) => {}
                Err(_) => continue,
            }
            let Ok(content) = read_text_lossy(&abs) else { continue };
            self.index_file(rel, &content, rule);
        }
        self.built = true;
    }

    fn index_file(&mut self, rel: &str, content: &str, rule: &LangRule) {
        let mut seen: HashSet<(String, usize)> = HashSet::new();
        for (i, line) in content.split('\n').enumerate() {
            if line.encode_utf16().count() > 400 {
                continue; // likely minified
            }
            for pat in &rule.patterns {
                let Some(caps) = pat.re.captures(line) else { continue };
                let Some(m) = caps.get(pat.name_group) else { continue };
                let name = m.as_str().to_string();
                if name.is_empty() || !seen.insert((name.clone(), i)) {
                    continue;
                }
                let def = SymbolDef {
                    name: name.clone(),
                    kind: pat.kind,
                    file: rel.to_string(),
                    line: i + 1,
                    signature: take_chars(line.trim(), 200),
                };
                self.by_name.entry(name).or_default().push(self.defs.len());
                self.defs.push(def);
                break; // one declaration per line
            }
        }
    }

    pub fn is_built(&self) -> bool {
        self.built
    }

    pub fn size(&self) -> usize {
        self.defs.len()
    }

    /// Exact-name symbol lookup, optionally filtered by kind.
    pub fn find(&self, name: &str, kind: Option<SymbolKind>) -> Vec<SymbolDef> {
        self.by_name
            .get(name)
            .map(|ids| {
                ids.iter()
                    .map(|&i| &self.defs[i])
                    .filter(|d| kind.map_or(true, |k| d.kind == k))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Fuzzy symbol search by substring / case-insensitive contains.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SymbolDef> {
        let q = query.to_lowercase();
        let mut scored: Vec<(&SymbolDef, u32)> = self
            .defs
            .iter()
            .map(|d| {
                let n = d.name.to_lowercase();
                let score = if n == q {
                    100
                } else if n.starts_with(&q) {
                    60
                } else if n.contains(&q) {
                    30
                } else {
                    0
                };
                (d, score)
            })
            .filter(|(_, s)| *s > 0)
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().take(limit).map(|(d, _)| d.clone()).collect()
    }

    /// All definitions in a file, in declaration order — a structural outline.
    pub fn outline(&self, rel_path: &str) -> Vec<SymbolDef> {
        let mut out: Vec<SymbolDef> = self.defs.iter().filter(|d| d.file == rel_path).cloned().collect();
        out.sort_by_key(|d| d.line);
        out
    }

    /// Whole-word references to `name` across the codebase. Flags the line as a
    /// definition when it coincides with an indexed declaration. Capped for sanity.
    pub fn references(&self, name: &str, limit: usize) -> Vec<Reference> {
        let mut out = Vec::new();
        let Ok(word_re) = Regex::new(&format!(r"\b{}\b", regex::escape(name))) else {
            return out;
        };
        let def_lines: HashSet<(String, usize)> =
            self.find(name, None).into_iter().map(|d| (d.file, d.line)).collect();
        for rel in &self.file_list {
            if rule_for(rel).is_none() {
                continue;
            }
            let Ok(content) = read_text_lossy(&self.root.join(rel)) else { continue };
            if !content.contains(name) {
                continue;
            }
            for (i, line) in content.split('\n').enumerate() {
                if word_re.is_match(line) {
                    out.push(Reference {
                        file: rel.clone(),
                        line: i + 1,
                        text: take_chars(line.trim(), 200),
                        is_definition: def_lines.contains(&(rel.clone(), i + 1)),
                    });
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    }

    /// Read the body of a symbol (its declaration plus a brace/indent-balanced
    /// span), so a model can inspect just the relevant function rather than the
    /// whole file. Best-effort: brace counting for C-family, indent for Python.
    pub fn read_symbol(&self, def: &SymbolDef, max_lines: usize) -> Option<String> {
        let content = read_text_lossy(&self.root.join(&def.file)).ok()?;
        let lines: Vec<&str> = content.split('\n').collect();
        let start = def.line.checked_sub(1)?;
        if start >= lines.len() {
            return None;
        }
        let leading_ws = |l: &str| l.chars().take_while(|c| c.is_whitespace()).count();

        let mut end = start;
        if def.file.ends_with(".py") {
            let base_indent = leading_ws(lines[start]);
            end = start + 1;
            while end < lines.len() {
                let l = lines[end];
                if !l.trim().is_empty() && leading_ws(l) <= base_indent {
                    break;
                }
                end += 1;
                if end - start > max_lines {
                    break;
                }
            }
        } else {
            let mut depth: i64 = 0;
            let mut opened = false;
            let mut i = start;
            while i < lines.len() && i - start <= max_lines {
                for ch in lines[i].chars() {
                    if ch == '{' {
                        depth += 1;
                        opened = true;
                    } else if ch == '}' {
                        depth -= 1;
                    }
                }
                end = i + 1;
                if opened && depth <= 0 {
                    break;
                }
                if !opened && lines[i].contains(';') {
                    break; // one-liner / prototype
                }
                i += 1;
            }
        }
        let stop = end.min(start + max_lines).min(lines.len());
        Some(lines[start..stop].join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // smallModelTools.test.ts: SymbolIndex
    fn fixture() -> (tempfile::TempDir, SymbolIndex) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("math.ts"),
            [
                "export function add(a: number, b: number) {",
                "  return a + b;",
                "}",
                "",
                "export class Calculator {",
                "  multiply(a: number, b: number) {",
                "    return a * b;",
                "  }",
                "}",
                "",
                "export interface Op { name: string; }",
            ]
            .join("\n"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("app.py"),
            ["def main():", "    print(add(1, 2))", "", "class Service:", "    def run(self):", "        return main()"].join("\n"),
        )
        .unwrap();
        let mut idx = SymbolIndex::new(dir.path());
        idx.build(DEFAULT_MAX_FILES);
        (dir, idx)
    }

    #[test]
    fn indexes_ts_functions_classes_interfaces() {
        let (_d, idx) = fixture();
        assert_eq!(idx.find("add", Some(SymbolKind::Function)).len(), 1);
        assert_eq!(idx.find("Calculator", Some(SymbolKind::Class)).len(), 1);
        assert_eq!(idx.find("Op", Some(SymbolKind::Interface)).len(), 1);
    }

    #[test]
    fn indexes_python_defs_and_classes() {
        let (_d, idx) = fixture();
        assert_eq!(idx.find("main", Some(SymbolKind::Function)).len(), 1);
        assert_eq!(idx.find("Service", Some(SymbolKind::Class)).len(), 1);
    }

    #[test]
    fn outline_in_line_order() {
        let (_d, idx) = fixture();
        let names: Vec<String> = idx.outline("math.ts").into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["add", "Calculator", "multiply", "Op"]);
    }

    #[test]
    fn references_flag_definitions() {
        let (_d, idx) = fixture();
        let refs = idx.references("add", 80);
        assert!(refs.len() >= 2);
        assert!(refs.iter().any(|r| r.is_definition));
    }

    #[test]
    fn reads_symbol_body_by_brace_balancing() {
        let (_d, idx) = fixture();
        let defs = idx.find("add", Some(SymbolKind::Function));
        let body = idx.read_symbol(&defs[0], 200).unwrap();
        assert!(body.contains("return a + b;"));
        assert!(!body.contains("Calculator"));
    }

    #[test]
    fn reads_python_symbol_by_indent() {
        let (_d, idx) = fixture();
        let defs = idx.find("Service", Some(SymbolKind::Class));
        let body = idx.read_symbol(&defs[0], 200).unwrap();
        assert!(body.contains("def run(self):"));
        assert!(!body.contains("def main"));
    }

    #[test]
    fn fuzzy_search_by_substring() {
        let (_d, idx) = fixture();
        assert!(idx.search("calc", 25).iter().any(|h| h.name == "Calculator"));
    }
}
