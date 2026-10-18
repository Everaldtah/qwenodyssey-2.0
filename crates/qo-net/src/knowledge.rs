//! Obsidian-style long-term knowledge vault with RAG retrieval, plus the tools
//! that let the model use it (`knowledge_search` / `_read` / `_save` / `_list`,
//! `record_lesson`).
//!
//! Each note is a markdown file with YAML-ish frontmatter (title, tags, sources,
//! timestamps) and an [[wikilink]]-friendly body. Retrieval is semantic when an
//! Ollama embedding model is available (cosine over cached vectors) and falls
//! back to keyword scoring otherwise. This gives the agent a persistent,
//! improvable memory it can search, read, and update across sessions.
//!
//! On-disk format is byte-compatible with Qwenodyssey 1.x: the same note
//! layout, the same slugs, and the same `.index.json`
//! (`{"<slug>": {"updated": "<iso>", "vector": [..]}}`), so an existing vault
//! and its cached embeddings keep working.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use indexmap::IndexMap;
use qo_core::tool::SharedTool;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::util::{cancellable, collapse_ws, is_js_space, iso_now, js_trim, num_or, str_arg, take_chars, truncate_chars, value_to_string, INTERRUPTED};

/// Embedding backend settings (`model.base_url` + `knowledge.embed_model`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnowledgeOptions {
    /// Ollama base URL; a trailing `/v1` (OpenAI-compatible form) is stripped.
    pub base_url: String,
    pub embed_model: String,
}

impl Default for KnowledgeOptions {
    fn default() -> Self {
        Self { base_url: "http://localhost:11434".into(), embed_model: "nomic-embed-text".into() }
    }
}

/// What to write with [`KnowledgeBase::write`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoteInput {
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub sources: Vec<String>,
    pub links: Vec<String>,
    /// Explicit slug (default: slugified title).
    pub slug: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoteMeta {
    pub slug: String,
    pub title: String,
    pub tags: Vec<String>,
    pub sources: Vec<String>,
    pub links: Vec<String>,
    /// ISO timestamps as written in the frontmatter ("" when missing).
    pub created: String,
    pub updated: String,
    pub file: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Note {
    pub meta: NoteMeta,
    pub body: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NoteHit {
    pub meta: NoteMeta,
    pub score: f64,
    pub snippet: String,
}

/// One `.index.json` entry: when the vector was computed and the vector itself.
#[derive(Clone, Debug, Default, PartialEq)]
struct IndexEntry {
    updated: String,
    vector: Option<Vec<f64>>,
}

impl IndexEntry {
    /// Lenient read: a malformed entry just loses its vector (re-embedded later).
    fn from_value(v: &Value) -> Self {
        let updated = v.get("updated").and_then(Value::as_str).unwrap_or("").to_string();
        let vector = v
            .get("vector")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<f64>>())
            .filter(|v| !v.is_empty());
        Self { updated, vector }
    }

    fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("updated".into(), json!(self.updated));
        if let Some(v) = &self.vector {
            m.insert("vector".into(), json!(v));
        }
        Value::Object(m)
    }
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "with", "as", "at", "by", "be", "this",
    "that", "it", "from", "your", "you", "its", "into", "how", "what", "when",
];

static STOPWORD_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| STOPWORDS.iter().copied().collect());

/// Per-request ceiling for the embedding call (the TS had none; a wedged
/// Ollama must not hang a turn). Generous: the first call loads the model.
const EMBED_TIMEOUT: Duration = Duration::from_secs(60);

/// The vault. Cheap to share behind an `Arc`; all methods take `&self`.
pub struct KnowledgeBase {
    dir: PathBuf,
    index_path: PathBuf,
    index: Mutex<IndexMap<String, IndexEntry>>,
    /// `None` = untested, `Some(false)` = embeddings unavailable this session.
    embed_ok: Mutex<Option<bool>>,
    opts: KnowledgeOptions,
    http: reqwest::Client,
}

impl KnowledgeBase {
    /// `configured_path` empty → `~/.qwenodyssey/knowledge`.
    pub fn new(configured_path: &str, opts: KnowledgeOptions) -> Self {
        let dir = if configured_path.trim().is_empty() {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".qwenodyssey").join("knowledge")
        } else {
            absolute(Path::new(configured_path))
        };
        let index_path = dir.join(".index.json");
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(EMBED_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self { dir, index_path, index: Mutex::new(IndexMap::new()), embed_ok: Mutex::new(None), opts, http }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Create the vault directory and (re)load the retrieval index from disk.
    pub fn ensure(&self) {
        let _ = std::fs::create_dir_all(&self.dir);
        if !self.index_path.exists() {
            return;
        }
        let loaded: IndexMap<String, IndexEntry> = std::fs::read_to_string(&self.index_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Map<String, Value>>(&raw).ok())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), IndexEntry::from_value(v))).collect())
            .unwrap_or_default();
        if let Ok(mut idx) = self.index.lock() {
            *idx = loaded;
        }
    }

    /// Best-effort write of the index (compact JSON, like `JSON.stringify`).
    fn save_index_locked(&self, idx: &IndexMap<String, IndexEntry>) {
        let obj: Map<String, Value> = idx.iter().map(|(k, v)| (k.clone(), v.to_value())).collect();
        let _ = std::fs::write(&self.index_path, Value::Object(obj).to_string());
    }

    /// All notes (parsed), most recently updated first.
    pub fn list(&self) -> Vec<Note> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let mut files: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".md"))
            .map(|e| e.path())
            .collect();
        files.sort();
        let mut notes: Vec<Note> = files.iter().filter_map(|f| parse_note(f)).collect();
        notes.sort_by(|a, b| b.meta.updated.cmp(&a.meta.updated));
        notes
    }

    pub fn read(&self, slug: &str) -> Option<Note> {
        let file = self.dir.join(format!("{}.md", slugify(slug)));
        if file.exists() {
            parse_note(&file)
        } else {
            None
        }
    }

    /// Create a new note or update an existing one (same slug). Updating
    /// merges tags/sources/links and replaces the body, bumping `updated`.
    pub async fn write(&self, input: NoteInput) -> anyhow::Result<Note> {
        self.ensure();
        let slug_src = input.slug.as_deref().filter(|s| !s.is_empty()).unwrap_or(&input.title);
        let slug = slugify(slug_src);
        let file = self.dir.join(format!("{slug}.md"));
        let now = iso_now();
        let existing = if file.exists() { parse_note(&file) } else { None };

        let title = if !input.title.is_empty() {
            input.title.clone()
        } else {
            existing.as_ref().map(|n| n.meta.title.clone()).filter(|t| !t.is_empty()).unwrap_or_else(|| slug.clone())
        };
        let merged = |old: Option<&Vec<String>>, new: &[String]| {
            uniq(old.map(|v| v.as_slice()).unwrap_or_default().iter().chain(new.iter()))
        };
        let meta = NoteMeta {
            slug: slug.clone(),
            title,
            tags: merged(existing.as_ref().map(|n| &n.meta.tags), &input.tags),
            sources: merged(existing.as_ref().map(|n| &n.meta.sources), &input.sources),
            links: merged(existing.as_ref().map(|n| &n.meta.links), &input.links),
            // `existing?.created ?? now`: an existing note keeps its value even when blank.
            created: existing.as_ref().map(|n| n.meta.created.clone()).unwrap_or_else(|| now.clone()),
            updated: now.clone(),
            file: file.clone(),
        };
        let body = js_trim(&input.content).to_string();
        std::fs::write(&file, render(&meta, &body))?;

        // Update the retrieval index (embedding is best-effort).
        let vector = self.embed(&format!("{}\n{}", meta.title, input.content)).await;
        if let Ok(mut idx) = self.index.lock() {
            idx.insert(slug, IndexEntry { updated: now, vector });
            self.save_index_locked(&idx);
        }
        Ok(Note { meta, body })
    }

    /// Rank notes against a query (top `k`). Semantic when embeddings are
    /// available, keyword-scored otherwise.
    pub async fn search(&self, query: &str, k: usize) -> Vec<NoteHit> {
        let notes = self.list();
        if notes.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(Note, f64)> = Vec::with_capacity(notes.len());
        if let Some(qvec) = self.embed(query).await {
            // Ensure every note has a cached vector, then cosine-rank.
            for note in notes {
                let cached = self.index.lock().ok().and_then(|idx| idx.get(&note.meta.slug).cloned());
                let entry = match cached {
                    Some(e) if e.vector.is_some() && e.updated == note.meta.updated => e,
                    _ => {
                        let v = self.embed(&format!("{}\n{}", note.meta.title, note.body)).await;
                        let e = IndexEntry { updated: note.meta.updated.clone(), vector: v };
                        if let Ok(mut idx) = self.index.lock() {
                            idx.insert(note.meta.slug.clone(), e.clone());
                        }
                        e
                    }
                };
                let score = entry.vector.as_deref().map_or(0.0, |v| cosine(&qvec, v));
                scored.push((note, score));
            }
            if let Ok(idx) = self.index.lock() {
                self.save_index_locked(&idx);
            }
        } else {
            scored = notes.into_iter().map(|n| {
                let s = keyword_score(query, &n);
                (n, s)
            }).collect();
        }
        scored.retain(|(_, s)| *s > 0.0);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(note, score)| NoteHit { snippet: snippet(&note.body, 240), meta: note.meta, score })
            .collect()
    }

    /// Embed text via Ollama; returns `None` (and disables embeddings for the
    /// session) if unavailable.
    async fn embed(&self, text: &str) -> Option<Vec<f64>> {
        if self.embed_ok.lock().ok().and_then(|g| *g) == Some(false) {
            return None;
        }
        let root = self.opts.base_url.trim_end_matches('/');
        let root = root.strip_suffix("/v1").unwrap_or(root);
        let res = self
            .http
            .post(format!("{root}/api/embeddings"))
            .json(&json!({"model": self.opts.embed_model, "prompt": take_chars(text, 8000)}))
            .send()
            .await;
        let vec = match res {
            Ok(r) if r.status().is_success() => r
                .json::<Value>()
                .await
                .ok()
                .and_then(|j| j.get("embedding").and_then(Value::as_array).cloned())
                .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<f64>>())
                .filter(|v| !v.is_empty()),
            _ => None,
        };
        if let Ok(mut g) = self.embed_ok.lock() {
            *g = Some(vec.is_some());
        }
        vec
    }

    /// `Some(true)` semantic retrieval works, `Some(false)` keyword fallback
    /// in use, `None` not tried yet (for `/settings` and `/memory`).
    pub fn embeddings_active(&self) -> Option<bool> {
        self.embed_ok.lock().ok().and_then(|g| *g)
    }
}

/// Retrieve the top notes for a user message and format them as the context
/// block the chat layer prepends to the turn (`recallKnowledge` in chat.ts).
pub async fn recall_knowledge(kb: &KnowledgeBase, query: &str, k: usize) -> Option<String> {
    let hits = kb.search(query, k).await;
    if hits.is_empty() {
        return None;
    }
    let blocks: Vec<String> = hits.iter().map(|h| format!("### {} [[{}]]\n{}", h.meta.title, h.meta.slug, h.snippet)).collect();
    Some(format!(
        "RELEVANT KNOWLEDGE recalled from your long-term memory (use it; read full notes \
         with knowledge_read; update them with knowledge_save if you learn more):\n\n{}",
        blocks.join("\n\n")
    ))
}

/* ── helpers ── */

/// `path.resolve()` without touching the filesystem.
fn absolute(p: &Path) -> PathBuf {
    let joined = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(p) };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Lowercase; keep `[A-Za-z0-9_]`, whitespace and `-`; trim; whitespace runs →
/// `-`; max 80 chars; `"note"` when nothing is left. Must stay identical to the
/// TS slugify — slugs are the note file names.
pub fn slugify(s: &str) -> String {
    let lowered = s.to_lowercase();
    let kept: String =
        lowered.chars().filter(|&c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || is_js_space(c)).collect();
    let mut out = String::new();
    let mut in_ws = false;
    for c in js_trim(&kept).chars() {
        if is_js_space(c) {
            if !in_ws {
                out.push('-');
                in_ws = true;
            }
        } else {
            out.push(c);
            in_ws = false;
        }
    }
    let out = take_chars(&out, 80).to_string();
    if out.is_empty() {
        "note".into()
    } else {
        out
    }
}

/// `[...new Set(a.filter(Boolean))]`
fn uniq<'a>(items: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut seen = HashSet::new();
    items.filter(|s| !s.is_empty() && seen.insert(s.as_str().to_string())).cloned().collect()
}

fn snippet(body: &str, len: usize) -> String {
    let clean = collapse_ws(body);
    truncate_chars(js_trim(&clean), len, "…")
}

fn render(meta: &NoteMeta, body: &str) -> String {
    let links: Vec<String> = meta.links.iter().map(|l| format!("[[{}]]", slugify(l))).collect();
    let fm = [
        "---".to_string(),
        format!("title: {}", meta.title),
        format!("slug: {}", meta.slug),
        format!("tags: [{}]", meta.tags.join(", ")),
        format!("sources: [{}]", meta.sources.join(", ")),
        format!("links: [{}]", links.join(", ")),
        format!("created: {}", meta.created),
        format!("updated: {}", meta.updated),
        "---".to_string(),
        String::new(),
    ]
    .join("\n");
    format!("{fm}{body}\n")
}

static FRONTMATTER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\A---\n(.*?)\n---\n?(.*)\z").expect("static regex"));
static FM_LINE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([A-Za-z0-9_]+):(.*)$").expect("static regex"));

fn parse_note(file: &Path) -> Option<Note> {
    let raw = std::fs::read_to_string(file).ok()?;
    let slug = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(m) = FRONTMATTER_RE.captures(&raw) else {
        return Some(Note {
            meta: NoteMeta { title: slug.clone(), slug, file: file.to_path_buf(), ..Default::default() },
            body: js_trim(&raw).to_string(),
        });
    };
    let fm = parse_frontmatter(m.get(1).map_or("", |x| x.as_str()));
    let list = |k: &str| match fm.get(k) {
        Some(FmValue::List(v)) => v.clone(),
        _ => Vec::new(),
    };
    let scalar = |k: &str| match fm.get(k) {
        Some(FmValue::Scalar(s)) => s.clone(),
        _ => String::new(),
    };
    let title = scalar("title");
    Some(Note {
        meta: NoteMeta {
            title: if title.is_empty() { slug.clone() } else { title },
            slug,
            tags: list("tags"),
            sources: list("sources"),
            links: list("links"),
            created: scalar("created"),
            updated: scalar("updated"),
            file: file.to_path_buf(),
        },
        body: js_trim(m.get(2).map_or("", |x| x.as_str())).to_string(),
    })
}

enum FmValue {
    Scalar(String),
    List(Vec<String>),
}

/// `key: value` lines; `tags`/`sources`/`links` are `[a, b]` lists (links as
/// `[[slug]]`). Later duplicate keys win, like the TS object assignment.
fn parse_frontmatter(text: &str) -> HashMap<String, FmValue> {
    let mut out = HashMap::new();
    for line in text.split('\n') {
        let Some(c) = FM_LINE_RE.captures(line) else { continue };
        // JS `.` stops at \r (and U+2028/9) as well as \n, so such a line never
        // matched `^(\w+):\s*(.*)$` in the TS — skip it the same way.
        if c[2].contains(['\r', '\u{2028}', '\u{2029}']) {
            continue;
        }
        let key = c[1].to_string();
        let val = c[2].trim_start_matches(is_js_space);
        if matches!(key.as_str(), "tags" | "sources" | "links") {
            let inner = val.strip_prefix('[').unwrap_or(val);
            let inner = inner.strip_suffix(']').unwrap_or(inner);
            let items = inner
                .split(',')
                .map(|s| {
                    let t = js_trim(s);
                    let t = t.strip_prefix("[[").unwrap_or(t);
                    t.strip_suffix("]]").unwrap_or(t).to_string()
                })
                .filter(|s| !s.is_empty())
                .collect();
            out.insert(key, FmValue::List(items));
        } else {
            out.insert(key, FmValue::Scalar(js_trim(val).to_string()));
        }
    }
    out
}

static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9_]+").expect("static regex"));

fn tokenize(s: &str) -> Vec<String> {
    let lower = s.to_lowercase();
    TOKEN_RE
        .find_iter(&lower)
        .map(|m| m.as_str())
        .filter(|t| t.len() > 1 && !STOPWORD_SET.contains(t))
        .map(str::to_string)
        .collect()
}

/// Lightweight keyword relevance: term overlap with title/tag boosts.
fn keyword_score(query: &str, note: &Note) -> f64 {
    let q = tokenize(query);
    if q.is_empty() {
        return 0.0;
    }
    let title_terms: HashSet<String> = tokenize(&format!("{} {}", note.meta.title, note.meta.tags.join(" "))).into_iter().collect();
    let mut body_freq: HashMap<String, usize> = HashMap::new();
    for t in tokenize(&note.body) {
        *body_freq.entry(t).or_insert(0) += 1;
    }
    let mut score = 0.0;
    for term in &q {
        if title_terms.contains(term) {
            score += 3.0;
        }
        score += body_freq.get(term).copied().unwrap_or(0).min(4) as f64;
    }
    score / q.len() as f64
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na != 0.0 && nb != 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// `toList()`: an array of strings, or a comma-separated string.
fn to_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a.iter().map(|x| js_trim(&value_to_string(x)).to_string()).filter(|s| !s.is_empty()).collect(),
        Some(Value::String(s)) => s.split(',').map(|x| js_trim(x).to_string()).filter(|s| !s.is_empty()).collect(),
        _ => Vec::new(),
    }
}

/* ── tools ── */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Search,
    Read,
    Save,
    List,
    Lesson,
}

/// Build the five knowledge tools around a shared vault.
pub fn create_knowledge_tools(kb: Arc<KnowledgeBase>) -> Vec<SharedTool> {
    [Kind::Search, Kind::Read, Kind::Save, Kind::List, Kind::Lesson]
        .into_iter()
        .map(|kind| Arc::new(KnowledgeTool { kind, kb: kb.clone() }) as SharedTool)
        .collect()
}

struct KnowledgeTool {
    kind: Kind,
    kb: Arc<KnowledgeBase>,
}

#[async_trait]
impl Tool for KnowledgeTool {
    fn name(&self) -> &str {
        match self.kind {
            Kind::Search => "knowledge_search",
            Kind::Read => "knowledge_read",
            Kind::Save => "knowledge_save",
            Kind::List => "knowledge_list",
            Kind::Lesson => "record_lesson",
        }
    }

    fn description(&self) -> &str {
        match self.kind {
            Kind::Search => {
                "Search your long-term memory (the knowledge vault) for what you've already learned \
                 about a topic, codebase, API, or concept. Check here BEFORE searching the web."
            }
            Kind::Read => "Read a full note from your knowledge vault by its slug (e.g. from knowledge_search).",
            Kind::Save => {
                "Save or update a note in your long-term memory so you remember it permanently. Use this \
                 whenever you learn something worth keeping: how a codebase works, an API's usage, a fix, \
                 a fact you looked up. Saving with an existing title UPDATES that note (improving knowledge). \
                 Write durable, distilled knowledge — not chit-chat. Cite sources (URLs) when you have them."
            }
            Kind::List => "List the titles of everything currently in your knowledge vault.",
            Kind::Lesson => {
                "Record a durable LESSON learned from a mistake or correction so you don't repeat it (e.g. \
                 'use Get-WinEvent, not eventquery, for Windows logs'). Lessons are auto-recalled in future \
                 turns. Use whenever a tool/command failed and you found the right way, or the user corrects you."
            }
        }
    }

    fn mutating(&self) -> bool {
        matches!(self.kind, Kind::Save | Kind::Lesson)
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Search may embed every stale note (one HTTP call each); Esc stops it.
        cancellable(ctx, self.exec(&args, ctx)).await.unwrap_or_else(|| ToolResult::err(INTERRUPTED))
    }
}

impl KnowledgeTool {
    async fn exec(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let kb = &self.kb;
        match self.kind {
            Kind::Search => {
                let query = js_trim(&str_arg(args, &["query"])).to_string();
                if query.is_empty() {
                    return ToolResult::err("No query given");
                }
                let k = num_or(args, "k", 4.0).max(1.0) as usize;
                let hits = kb.search(&query, k).await;
                ctx.log(json!({"tool": "knowledge_search", "query": query, "hits": hits.len()}));
                if hits.is_empty() {
                    return ToolResult::ok(format!("No notes found for \"{query}\"."));
                }
                let text = hits
                    .iter()
                    .map(|h| {
                        let tags = if h.meta.tags.is_empty() { String::new() } else { format!("  (tags: {})", h.meta.tags.join(", ")) };
                        format!("[[{}]] {}{tags}\n   {}", h.meta.slug, h.meta.title, h.snippet)
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                ToolResult::ok(text)
            }
            Kind::Read => {
                let slug = js_trim(&str_arg(args, &["slug"])).to_string();
                let note = kb.read(&slug);
                ctx.log(json!({"tool": "knowledge_read", "slug": slug, "found": note.is_some()}));
                let Some(note) = note else {
                    return ToolResult::err(format!("No note with slug \"{slug}\"."));
                };
                let mut out = format!("# {}\n", note.meta.title);
                if !note.meta.tags.is_empty() {
                    out.push_str(&format!("tags: {}\n", note.meta.tags.join(", ")));
                }
                if !note.meta.sources.is_empty() {
                    out.push_str(&format!("sources: {}\n", note.meta.sources.join(", ")));
                }
                out.push_str(&format!("\n{}", note.body));
                ToolResult::ok(out)
            }
            Kind::Save => {
                let title = js_trim(&str_arg(args, &["title"])).to_string();
                let content = js_trim(&str_arg(args, &["content"])).to_string();
                if title.is_empty() || content.is_empty() {
                    return ToolResult::err("Need both title and content.");
                }
                let input = NoteInput {
                    title,
                    content,
                    tags: to_list(args.get("tags")),
                    sources: to_list(args.get("sources")),
                    links: to_list(args.get("links")),
                    slug: None,
                };
                match kb.write(input).await {
                    Ok(note) => {
                        ctx.log(json!({"tool": "knowledge_save", "slug": note.meta.slug}));
                        ToolResult::ok(format!("Saved note [[{}]] \"{}\".", note.meta.slug, note.meta.title))
                    }
                    Err(e) => write_error(self.name(), &e, ctx),
                }
            }
            Kind::List => {
                let notes = kb.list();
                ctx.log(json!({"tool": "knowledge_list", "count": notes.len()}));
                if notes.is_empty() {
                    return ToolResult::ok("Knowledge vault is empty.");
                }
                ToolResult::ok(notes.iter().map(|n| format!("[[{}]] {}", n.meta.slug, n.meta.title)).collect::<Vec<_>>().join("\n"))
            }
            Kind::Lesson => {
                let lesson = js_trim(&str_arg(args, &["lesson", "content"])).to_string();
                if lesson.is_empty() {
                    return ToolResult::err("Provide the lesson text.");
                }
                let title = match str_arg(args, &["title"]) {
                    t if t.is_empty() => take_chars(&lesson, 60).to_string(),
                    t => t,
                };
                let mut tags = vec!["lesson".to_string(), "evolution".to_string()];
                tags.extend(to_list(args.get("tags")));
                let input = NoteInput {
                    title: format!("Lesson: {}", js_trim(&title)),
                    content: lesson,
                    tags,
                    ..Default::default()
                };
                match kb.write(input).await {
                    Ok(note) => {
                        ctx.log(json!({"tool": "record_lesson", "slug": note.meta.slug}));
                        ToolResult::ok(format!("Lesson saved: {}", note.meta.title))
                    }
                    Err(e) => write_error(self.name(), &e, ctx),
                }
            }
        }
    }
}

/// A failed vault write, reported the way the TS registry reported a throw.
fn write_error(name: &str, e: &anyhow::Error, ctx: &ToolContext) -> ToolResult {
    ctx.log(json!({"tool": name, "error": format!("{e:#}")}));
    ToolResult::err(format!("Tool \"{name}\" error: {e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{serve, Reply};

    /// An "Ollama" without the embedding model (HTTP 404) → keyword fallback.
    /// (A closed port works too, but costs ~2 s per connect on Windows.)
    async fn offline_opts() -> (crate::testutil::MockServer, KnowledgeOptions) {
        let server = serve(Arc::new(|_| Reply::json(404, json!({"error": "model not found"})))).await;
        let opts = KnowledgeOptions { base_url: format!("{}/v1", server.base), embed_model: "nomic-embed-text".into() };
        (server, opts)
    }

    fn kb_in(dir: &Path, opts: KnowledgeOptions) -> KnowledgeBase {
        let kb = KnowledgeBase::new(&dir.to_string_lossy(), opts);
        kb.ensure();
        kb
    }

    #[test]
    fn slugify_matches_ts() {
        assert_eq!(slugify("Lesson: Shell Command Timeout"), "lesson-shell-command-timeout");
        assert_eq!(slugify("  Hello,   World! _x-y "), "hello-world-_x-y");
        assert_eq!(slugify("Café über naïve"), "caf-ber-nave");
        assert_eq!(slugify("!!!"), "note");
        assert_eq!(slugify(&"a".repeat(100)).len(), 80);
        assert_eq!(slugify("../../etc/passwd"), "etcpasswd");
    }

    #[test]
    fn frontmatter_round_trip() {
        let meta = NoteMeta {
            slug: "x".into(),
            title: "Lesson: X".into(),
            tags: vec!["lesson".into(), "evolution".into()],
            sources: vec!["https://a.example".into()],
            links: vec!["Other Note".into()],
            created: "2026-06-06T16:12:51.722Z".into(),
            updated: "2026-06-06T16:12:51.722Z".into(),
            file: PathBuf::from("x.md"),
        };
        let text = render(&meta, "Body [[other-note]]");
        assert!(text.starts_with("---\ntitle: Lesson: X\nslug: x\ntags: [lesson, evolution]\n"), "{text}");
        assert!(text.contains("links: [[[other-note]]]\n"));
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("x.md");
        std::fs::write(&f, &text).unwrap();
        let n = parse_note(&f).unwrap();
        assert_eq!(n.meta.title, "Lesson: X");
        assert_eq!(n.meta.tags, ["lesson", "evolution"]);
        assert_eq!(n.meta.links, ["other-note"]);
        assert_eq!(n.meta.sources, ["https://a.example"]);
        assert_eq!(n.meta.created, "2026-06-06T16:12:51.722Z");
        assert_eq!(n.body, "Body [[other-note]]");

        // No frontmatter (e.g. a hand-written Obsidian note) → title = slug, whole text is the body.
        std::fs::write(dir.path().join("plain.md"), "  just text \n").unwrap();
        let n = parse_note(&dir.path().join("plain.md")).unwrap();
        assert_eq!((n.meta.title.as_str(), n.body.as_str(), n.meta.updated.as_str()), ("plain", "just text", ""));
    }

    #[tokio::test]
    async fn keyword_fallback_search_on_temp_vault() {
        let dir = tempfile::tempdir().unwrap();
        let (_ollama, opts) = offline_opts().await;
        let kb = kb_in(dir.path(), opts);
        kb.write(NoteInput {
            title: "Rust borrow checker".into(),
            content: "Lifetimes and borrowing rules in Rust. The borrow checker rejects aliasing.".into(),
            tags: vec!["rust".into()],
            ..Default::default()
        })
        .await
        .unwrap();
        kb.write(NoteInput { title: "Pasta recipe".into(), content: "Boil water, add salt, cook pasta.".into(), ..Default::default() })
            .await
            .unwrap();
        assert_eq!(kb.embeddings_active(), Some(false));

        let hits = kb.search("how does the rust borrow checker work", 4).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meta.slug, "rust-borrow-checker");
        assert!(hits[0].score > 0.0);
        assert!(kb.search("the and of", 4).await.is_empty(), "stopwords only → no hits");

        // Updating by title merges tags and keeps `created`.
        let first = kb.read("Rust borrow checker").unwrap();
        let updated = kb
            .write(NoteInput { title: "Rust borrow checker".into(), content: "v2".into(), tags: vec!["lang".into(), "rust".into()], ..Default::default() })
            .await
            .unwrap();
        assert_eq!(updated.meta.tags, ["rust", "lang"]);
        assert_eq!(updated.meta.created, first.meta.created);
        assert_eq!(kb.read("rust-borrow-checker").unwrap().body, "v2");
        assert_eq!(kb.list().len(), 2);

        // The index got an entry without a vector (embeddings were down).
        let idx: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join(".index.json")).unwrap()).unwrap();
        assert_eq!(idx["rust-borrow-checker"]["updated"], json!(updated.meta.updated));
        assert!(idx["rust-borrow-checker"].get("vector").is_none());
    }

    #[tokio::test]
    async fn reads_a_ts_shaped_index_and_vault() {
        // Fixture shaped exactly like a 1.x vault: note + compact `.index.json`
        // with an up-to-date cached vector.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lesson-shell-command-timeout.md"),
            "---\ntitle: Lesson: Shell Command Timeout\nslug: lesson-shell-command-timeout\ntags: [lesson, evolution]\nsources: []\nlinks: []\ncreated: 2026-06-06T16:12:51.722Z\nupdated: 2026-06-06T16:12:51.722Z\n---\nUse Start-Process with -NoNewWindow to prevent timeouts.\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join(".index.json"),
            r#"{"lesson-shell-command-timeout":{"updated":"2026-06-06T16:12:51.722Z","vector":[1,0,0]},"gone-note":{"updated":"x"}}"#,
        )
        .unwrap();

        // Mock Ollama: the query embeds to [1,0,0]; any other text would be [0,1,0].
        let server = serve(Arc::new(|req| {
            assert_eq!(req.path, "/api/embeddings");
            let body: Value = serde_json::from_str(&req.body).unwrap_or_default();
            assert_eq!(body["model"], "nomic-embed-text");
            let v = if body["prompt"] == "timeouts" { json!([1.0, 0.0, 0.0]) } else { json!([0.0, 1.0, 0.0]) };
            Reply::json(200, json!({"embedding": v}))
        }))
        .await;
        let kb = kb_in(dir.path(), KnowledgeOptions { base_url: format!("{}/v1/", server.base), ..Default::default() });

        let hits = kb.search("timeouts", 3).await;
        assert_eq!(kb.embeddings_active(), Some(true));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].meta.title, "Lesson: Shell Command Timeout");
        assert!((hits[0].score - 1.0).abs() < 1e-9);
        // Only the query was embedded — the cached vector was reused.
        assert_eq!(server.requests().len(), 1);

        // Index rewritten in the same shape, unknown entries preserved in order.
        let raw = std::fs::read_to_string(dir.path().join(".index.json")).unwrap();
        assert!(raw.starts_with(r#"{"lesson-shell-command-timeout":{"updated":"2026-06-06T16:12:51.722Z","vector":[1.0,0.0,0.0]}"#), "{raw}");
        assert!(raw.contains(r#""gone-note":{"updated":"x"}"#));

        let block = recall_knowledge(&kb, "timeouts", 2).await.unwrap();
        assert!(block.starts_with("RELEVANT KNOWLEDGE recalled from your long-term memory"));
        assert!(block.contains("### Lesson: Shell Command Timeout [[lesson-shell-command-timeout]]"));
    }

    #[tokio::test]
    async fn tools_save_read_list_and_lessons() {
        let dir = tempfile::tempdir().unwrap();
        let (_ollama, opts) = offline_opts().await;
        let kb = Arc::new(kb_in(dir.path(), opts));
        let tools = create_knowledge_tools(kb.clone());
        let tool = |n: &str| tools.iter().find(|t| t.name() == n).cloned().unwrap();
        let ctx = ToolContext::new(".");

        assert_eq!(tool("knowledge_list").run(json!({}), &ctx).await.output, "Knowledge vault is empty.");
        let r = tool("knowledge_save")
            .run(json!({"title": "Tokio select", "content": "biased; polls in order", "tags": "rust, async", "sources": ["https://docs.rs"]}), &ctx)
            .await;
        assert_eq!(r.output, "Saved note [[tokio-select]] \"Tokio select\".");
        assert_eq!(tool("knowledge_save").run(json!({"title": "x"}), &ctx).await.output, "Need both title and content.");

        let r = tool("knowledge_read").run(json!({"slug": "tokio-select"}), &ctx).await;
        assert_eq!(r.output, "# Tokio select\ntags: rust, async\nsources: https://docs.rs\n\nbiased; polls in order");
        assert!(!tool("knowledge_read").run(json!({"slug": "nope"}), &ctx).await.ok);

        let r = tool("knowledge_search").run(json!({"query": "tokio select"}), &ctx).await;
        assert!(r.output.starts_with("[[tokio-select]] Tokio select  (tags: rust, async)\n   biased;"), "{}", r.output);

        let r = tool("record_lesson").run(json!({"lesson": "Use Get-WinEvent for Windows logs."}), &ctx).await;
        assert_eq!(r.output, "Lesson saved: Lesson: Use Get-WinEvent for Windows logs.");
        let lesson = kb.read("lesson-use-get-winevent-for-windows-logs").unwrap();
        assert_eq!(lesson.meta.tags, ["lesson", "evolution"]);
        assert_eq!(tool("knowledge_list").run(json!({}), &ctx).await.output.lines().count(), 2);

        let mutating: Vec<&str> = tools.iter().filter(|t| t.mutating()).map(|t| t.name()).collect();
        assert_eq!(mutating, ["knowledge_save", "record_lesson"]);
        let specs: Vec<String> = crate::specs::knowledge_tool_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(tools.iter().map(|t| t.name().to_string()).collect::<Vec<_>>(), specs);
    }
}
