//! Persisted chat sessions so a conversation can be resumed later
//! (`--continue`, `--resume`, `/resume`). Each session is one JSON file under
//! `~/.qwenodyssey/sessions/`. We store the message history WITHOUT the system
//! prompt (it's rebuilt fresh on resume so prompt/tool/self-awareness changes
//! take effect), plus lightweight metadata for the picker.
//!
//! Same directory and JSON shape as Qwenodyssey 1.x
//! (`{id, cwd, title, model, provider, createdAt, updatedAt, turns, messages}`),
//! so 1.x sessions resume in 2.0 and vice versa.

use std::path::{Component, Path, PathBuf};

use qo_core::{Message, Role, ToolCall};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::util::{collapse_ws, js_trim, take_chars, value_to_string};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSessionMeta {
    pub id: String,
    pub cwd: String,
    pub title: String,
    pub model: String,
    pub provider: String,
    /// ISO timestamps (`new Date().toISOString()` in 1.x).
    pub created_at: String,
    pub updated_at: String,
    pub turns: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatSession {
    #[serde(flatten)]
    pub meta: ChatSessionMeta,
    /// History excluding the leading system message.
    pub messages: Vec<Message>,
}

pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// `dir = None` → `~/.qwenodyssey/sessions`.
    pub fn new(dir: Option<PathBuf>) -> Self {
        let dir = dir.unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".qwenodyssey").join("sessions"));
        Self { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)
    }

    /// `None` for ids that would escape the sessions directory.
    fn file(&self, id: &str) -> Option<PathBuf> {
        if id.is_empty() || id.contains(['/', '\\', ':']) || id == "." || id == ".." {
            return None;
        }
        Some(self.dir.join(format!("{id}.json")))
    }

    /// Time-sortable id like `20260608-143501-a1b2` (local time + 4 random base-36 chars).
    pub fn new_id(&self) -> String {
        const ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let mut rng = rand::thread_rng();
        let suffix: String = (0..4).map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char).collect();
        format!("{stamp}-{suffix}")
    }

    /// Best effort — never crash chat over a failed save. Returns whether the
    /// session was written. Writes a temp file then renames, so a crash
    /// mid-write can't leave a truncated session behind.
    pub fn save(&self, session: &ChatSession) -> bool {
        let Some(file) = self.file(&session.meta.id) else { return false };
        if self.ensure().is_err() {
            return false;
        }
        let Ok(text) = serde_json::to_string(session) else { return false };
        let tmp = file.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_err() {
            return false;
        }
        if std::fs::rename(&tmp, &file).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        true
    }

    pub fn load(&self, id: &str) -> Option<ChatSession> {
        let raw = std::fs::read_to_string(self.file(id)?).ok()?;
        let v: Value = serde_json::from_str(&raw).ok()?;
        let obj = v.as_object()?;
        let messages = obj
            .get("messages")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(message_from_value).collect())
            .unwrap_or_default();
        Some(ChatSession { meta: meta_from_obj(obj), messages })
    }

    /// All sessions (metadata only), newest first.
    pub fn list(&self) -> Vec<ChatSessionMeta> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let mut files: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
            .map(|e| e.path())
            .collect();
        files.sort();
        let mut metas: Vec<ChatSessionMeta> = files
            .iter()
            .filter_map(|f| {
                let v: Value = serde_json::from_str(&std::fs::read_to_string(f).ok()?).ok()?;
                let m = meta_from_obj(v.as_object()?);
                (!m.id.is_empty()).then_some(m)
            })
            .collect();
        metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        metas
    }

    /// Most recent session started in this directory (for `--continue`).
    pub fn latest_for_cwd(&self, cwd: &Path) -> Option<ChatSessionMeta> {
        let norm = resolve_lower(cwd);
        self.list().into_iter().find(|m| resolve_lower(Path::new(&m.cwd)) == norm)
    }
}

/// `path.resolve(p).toLowerCase()` — absolute, `.`/`..` folded, separators
/// normalised, case-folded (Windows paths are case-insensitive).
fn resolve_lower(p: &Path) -> String {
    let joined = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(p) };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out.to_string_lossy().to_lowercase()
}

fn meta_from_obj(obj: &Map<String, Value>) -> ChatSessionMeta {
    let s = |k: &str| obj.get(k).filter(|v| !v.is_null()).map(value_to_string).unwrap_or_default();
    ChatSessionMeta {
        id: s("id"),
        cwd: s("cwd"),
        title: s("title"),
        model: s("model"),
        provider: s("provider"),
        created_at: s("createdAt"),
        updated_at: s("updatedAt"),
        turns: obj.get("turns").and_then(|t| t.as_u64().or_else(|| t.as_f64().map(|f| f.max(0.0) as u64))).unwrap_or(0),
    }
}

/// Lenient message read: a 1.x file with a missing `content`, a tool call
/// without an id or with string-encoded arguments still loads.
fn message_from_value(v: &Value) -> Option<Message> {
    if let Ok(m) = serde_json::from_value::<Message>(v.clone()) {
        return Some(m);
    }
    let role: Role = serde_json::from_value(v.get("role")?.clone()).ok()?;
    let opt_str = |k: &str| v.get(k).filter(|x| !x.is_null()).map(value_to_string);
    let tool_calls = v.get("tool_calls").and_then(Value::as_array).map(|calls| {
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let arguments = match c.get("arguments") {
                    Some(a @ Value::Object(_)) => a.clone(),
                    Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok().filter(Value::is_object).unwrap_or_else(|| Value::Object(Map::new())),
                    _ => Value::Object(Map::new()),
                };
                ToolCall {
                    id: c.get("id").filter(|x| !x.is_null()).map(value_to_string).unwrap_or_else(|| format!("call_{i}")),
                    name: c.get("name").map(value_to_string).unwrap_or_default(),
                    arguments,
                }
            })
            .collect()
    });
    Some(Message {
        role,
        content: opt_str("content").unwrap_or_default(),
        tool_calls,
        tool_call_id: opt_str("tool_call_id"),
        name: opt_str("name"),
    })
}

/// A short, human-readable title from the first real user message.
pub fn derive_title(messages: &[Message]) -> String {
    let Some(u) = messages.iter().find(|m| m.role == Role::User) else {
        return "(empty session)".into();
    };
    let collapsed = collapse_ws(&u.content);
    let mut t = js_trim(&collapsed);
    // Strip a recalled-knowledge prefix ("…———…") if present so the title is the question.
    if let Some(sep) = t.rfind("———") {
        t = js_trim(&t[sep + "———".len()..]);
    }
    let t = take_chars(t, 70);
    if t.is_empty() {
        "(session)".into()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session(id: &str, cwd: &str, updated: &str, messages: Vec<Message>) -> ChatSession {
        ChatSession {
            meta: ChatSessionMeta {
                id: id.into(),
                cwd: cwd.into(),
                title: derive_title(&messages),
                model: "m".into(),
                provider: "p".into(),
                created_at: "t0".into(),
                updated_at: updated.into(),
                turns: 1,
            },
            messages,
        }
    }

    #[test]
    fn round_trips_a_session_and_lists_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(Some(dir.path().to_path_buf()));
        let id = store.new_id();
        let messages = vec![Message::user("fix the build"), Message::assistant("done")];
        assert!(store.save(&session(&id, "C:/proj", "t1", messages)));
        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.meta.title, "fix the build");
        assert!(store.list().iter().any(|m| m.id == id));
        assert!(store.load("../escape").is_none());
    }

    #[test]
    fn derives_a_title_stripping_a_recall_prefix() {
        assert_eq!(derive_title(&[Message::user("recall blah ——— actual question here")]), "actual question here");
        assert_eq!(derive_title(&[Message::assistant("x")]), "(empty session)");
        assert_eq!(derive_title(&[Message::user("   ")]), "(session)");
        assert_eq!(derive_title(&[Message::user(&"é".repeat(100))]).chars().count(), 70);
    }

    #[test]
    fn new_id_is_time_sortable_and_unique() {
        let store = SessionStore::new(None);
        let a = store.new_id();
        let b = store.new_id();
        assert_ne!(a, b);
        let re = regex::Regex::new(r"^\d{8}-\d{6}-[a-z0-9]{4}$").unwrap();
        assert!(re.is_match(&a), "{a}");
    }

    #[test]
    fn reads_ts_written_sessions_and_writes_the_same_shape() {
        let dir = tempfile::tempdir().unwrap();
        // Exactly what 1.x wrote (JSON.stringify, key order included).
        let ts = r#"{"id":"20260608-152954-gmgt","cwd":"C:\\Users\\me","title":"list files","model":"qwen3:8b","provider":"ollama","createdAt":"2026-06-08T14:29:54.000Z","updatedAt":"2026-06-08T14:31:00.000Z","turns":1,"messages":[{"role":"user","content":"list files"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","name":"list_files","arguments":{"path":"."}}]},{"role":"tool","content":"a.txt","tool_call_id":"call_1","name":"list_files"},{"role":"assistant","content":"One file: a.txt"}]}"#;
        std::fs::write(dir.path().join("20260608-152954-gmgt.json"), ts).unwrap();
        // A sloppy one: no content, string arguments, no tool-call id.
        std::fs::write(
            dir.path().join("20260101-000000-zzzz.json"),
            r#"{"id":"20260101-000000-zzzz","cwd":".","updatedAt":"2026-01-01","messages":[{"role":"assistant","tool_calls":[{"name":"x","arguments":"{\"a\":1}"}]}]}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("broken.json"), "{not json").unwrap();
        std::fs::write(dir.path().join("noid.json"), r#"{"title":"x"}"#).unwrap();

        let store = SessionStore::new(Some(dir.path().to_path_buf()));
        let list = store.list();
        assert_eq!(list.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["20260608-152954-gmgt", "20260101-000000-zzzz"]);

        let s = store.load("20260608-152954-gmgt").unwrap();
        assert_eq!(s.messages.len(), 4);
        assert_eq!(s.messages[1].tool_calls.as_ref().unwrap()[0].arguments, json!({"path": "."}));
        assert_eq!(s.messages[2].role, Role::Tool);
        // Re-saving produces byte-identical JSON.
        assert!(store.save(&s));
        assert_eq!(std::fs::read_to_string(dir.path().join("20260608-152954-gmgt.json")).unwrap(), ts);

        let sloppy = store.load("20260101-000000-zzzz").unwrap();
        let call = &sloppy.messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!((call.id.as_str(), call.arguments.clone()), ("call_0", json!({"a": 1})));
        assert_eq!(sloppy.messages[0].content, "");

        // --continue: newest session whose cwd resolves to the same directory.
        let here = std::env::current_dir().unwrap();
        assert_eq!(store.latest_for_cwd(&here).map(|m| m.id), Some("20260101-000000-zzzz".to_string()));
        let upper = PathBuf::from(here.to_string_lossy().to_uppercase());
        if cfg!(windows) {
            assert!(store.latest_for_cwd(&upper).is_some(), "case-insensitive match");
        }
        assert_eq!(store.latest_for_cwd(Path::new("C:\\Users\\me\\")).map(|m| m.id).as_deref(), if cfg!(windows) { Some("20260608-152954-gmgt") } else { None });
    }
}
