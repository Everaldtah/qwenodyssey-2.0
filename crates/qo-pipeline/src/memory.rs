//! Project memory: simple markdown-backed notes under `.qwenodyssey/memory`
//! (port of `src/agents/memory.ts`).

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryCategory {
    #[default]
    Project,
    User,
    Decisions,
    Commands,
}

impl MemoryCategory {
    /// All categories, in listing order.
    pub const ALL: [MemoryCategory; 4] =
        [MemoryCategory::Project, MemoryCategory::User, MemoryCategory::Decisions, MemoryCategory::Commands];

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryCategory::Project => "project",
            MemoryCategory::User => "user",
            MemoryCategory::Decisions => "decisions",
            MemoryCategory::Commands => "commands",
        }
    }

    /// The markdown file backing this category.
    pub fn file_name(&self) -> &'static str {
        match self {
            MemoryCategory::Project => "project.md",
            MemoryCategory::User => "user.md",
            MemoryCategory::Decisions => "decisions.md",
            MemoryCategory::Commands => "commands.md",
        }
    }
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryCategory {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "project" => Ok(MemoryCategory::Project),
            "user" => Ok(MemoryCategory::User),
            "decisions" => Ok(MemoryCategory::Decisions),
            "commands" => Ok(MemoryCategory::Commands),
            other => anyhow::bail!("Unknown memory category: {other}. Use project|user|decisions|commands."),
        }
    }
}

/// One non-empty category file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEntry {
    pub category: MemoryCategory,
    /// Trimmed file content.
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct MemoryStore {
    dir: PathBuf,
}

impl MemoryStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The store for a project: `memory_path` (config `memory.path`, default
    /// `.qwenodyssey/memory`) resolved against `cwd`.
    pub fn for_project(cwd: &Path, memory_path: &str) -> Self {
        Self::new(cwd.join(memory_path))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn file(&self, cat: MemoryCategory) -> PathBuf {
        self.dir.join(cat.file_name())
    }

    /// Append a dated bullet: `- (YYYY-MM-DD) text`.
    pub fn add(&self, text: &str, cat: MemoryCategory) -> anyhow::Result<()> {
        fs::create_dir_all(&self.dir).with_context(|| format!("creating {}", self.dir.display()))?;
        let stamp = chrono::Utc::now().format("%Y-%m-%d");
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.file(cat))
            .with_context(|| format!("opening {}", self.file(cat).display()))?;
        f.write_all(format!("- ({stamp}) {}\n", text.trim()).as_bytes())?;
        Ok(())
    }

    /// Non-empty categories with their (trimmed) content.
    pub fn list(&self) -> Vec<MemoryEntry> {
        let mut out = Vec::new();
        for cat in MemoryCategory::ALL {
            let Ok(raw) = fs::read(self.file(cat)) else { continue };
            let content = String::from_utf8_lossy(&raw).trim().to_string();
            if !content.is_empty() {
                out.push(MemoryEntry { category: cat, content });
            }
        }
        out
    }

    /// Case-insensitive line search: `[category] line` (bullet marker dropped).
    pub fn search(&self, query: &str) -> Vec<String> {
        let q = query.to_lowercase();
        let mut hits = Vec::new();
        for MemoryEntry { category, content } in self.list() {
            for line in content.split('\n') {
                if line.to_lowercase().contains(&q) {
                    hits.push(format!("[{category}] {}", line.strip_prefix("- ").unwrap_or(line)));
                }
            }
        }
        hits
    }

    /// Concatenated memory for prompt context (bounded by the caller).
    pub fn digest(&self) -> String {
        self.list()
            .into_iter()
            .map(|e| format!("### {}\n{}", e.category, e.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn clear(&self) -> anyhow::Result<()> {
        for cat in MemoryCategory::ALL {
            let f = self.file(cat);
            if f.exists() {
                fs::remove_file(&f).with_context(|| format!("removing {}", f.display()))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_list_search_digest_clear() {
        let dir = tempfile::tempdir().unwrap();
        let m = MemoryStore::new(dir.path().join("memory"));
        assert!(m.list().is_empty());
        m.add("  uses pnpm  ", MemoryCategory::Commands).unwrap();
        m.add("Prefers tabs", MemoryCategory::User).unwrap();
        let all = m.list();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].category, MemoryCategory::User); // listing order: project, user, decisions, commands
        assert!(all[1].content.ends_with(") uses pnpm"));
        let hits = m.search("PNPM");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].starts_with("[commands] ("));
        assert!(m.digest().contains("### user\n- ("));
        m.clear().unwrap();
        assert!(m.list().is_empty());
        assert_eq!("decisions".parse::<MemoryCategory>().unwrap(), MemoryCategory::Decisions);
        assert!("nope".parse::<MemoryCategory>().is_err());
    }
}
