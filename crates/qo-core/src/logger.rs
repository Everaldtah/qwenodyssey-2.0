//! Structured JSONL session logging (ported from `src/core/logger.ts`).
//!
//! Each session appends one JSON object per line to
//! `<cwd>/.qwenodyssey/logs/session-YYYYMMDD-HHMMSS.jsonl`, every entry shaped
//! `{"ts": "<ISO-8601 UTC, ms>", ...fields}` — identical to the TS logger, so
//! existing log tooling keeps working.
//!
//! The TS class also had console helpers (`info`/`step`/`success`/`warn`/
//! `error`/`debug`). Library crates never print (the terminal belongs to the
//! binary's UI layer) and nothing in the TS called those helpers, so they are
//! not ported; the `verbose` flag is kept for the UI to consult.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

use crate::config::workspace_path;
use crate::tool::LogFn;

/// `Date.prototype.toISOString()` format: `2026-10-18T15:10:00.123Z`.
pub fn iso_timestamp(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// The current time as [`iso_timestamp`].
pub fn now_iso() -> String {
    iso_timestamp(Utc::now())
}

/// Session log file name for a start time: `session-20261018-151000.jsonl`
/// (the ISO stamp with `-`/`:` and the fraction removed, `T` → `-`).
pub fn session_file_name(t: DateTime<Utc>) -> String {
    format!("session-{}.jsonl", t.format("%Y%m%d-%H%M%S"))
}

/// JSONL session logger. Cheap to clone; clones share the same log file, so one
/// logger can be handed to the tool registry, the pipeline and the chat loop.
#[derive(Clone, Debug)]
pub struct Logger {
    cwd: PathBuf,
    file: Arc<Mutex<Option<PathBuf>>>,
    /// Verbose output requested (`--verbose`); the UI layer decides what to show.
    pub verbose: bool,
}

impl Logger {
    /// A logger for the project at `cwd`. Nothing is written until [`Logger::start`].
    pub fn new(cwd: impl Into<PathBuf>, verbose: bool) -> Self {
        Self { cwd: cwd.into(), file: Arc::new(Mutex::new(None)), verbose }
    }

    /// Begin a session log file under `.qwenodyssey/logs` and record
    /// `{"type": "session_start", "task": task}`. Returns the log file path.
    pub fn start(&self, task: &str) -> Result<PathBuf> {
        let dir = workspace_path(&self.cwd).join("logs");
        fs::create_dir_all(&dir).with_context(|| format!("Failed to create {}", dir.display()))?;
        let file = dir.join(session_file_name(Utc::now()));
        if let Ok(mut slot) = self.file.lock() {
            *slot = Some(file.clone());
        }
        self.event(serde_json::json!({ "type": "session_start", "task": task }));
        Ok(file)
    }

    /// Append a structured event to the session log (if started). `entry` should
    /// be a JSON object; its fields follow `ts` (a field named `ts` overrides the
    /// timestamp, like the TS spread). A non-object is stored under `"data"`.
    /// Never fails: logging must never crash the run.
    pub fn event(&self, entry: Value) {
        let Ok(guard) = self.file.lock() else { return };
        let Some(file) = guard.as_ref() else { return };
        let line = event_line(&now_iso(), entry);
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(file) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// The current session log file, once started.
    pub fn log_file(&self) -> Option<PathBuf> {
        self.file.lock().ok().and_then(|g| g.clone())
    }

    /// A [`LogFn`] for `ToolContext.log` that appends to this logger.
    pub fn log_fn(&self) -> LogFn {
        let me = self.clone();
        Arc::new(move |entry| me.event(entry))
    }
}

/// One JSONL line (with trailing newline) for `entry` stamped at `ts`.
fn event_line(ts: &str, entry: Value) -> String {
    let mut obj = Map::new();
    obj.insert("ts".into(), Value::String(ts.to_string()));
    match entry {
        Value::Object(fields) => {
            for (k, v) in fields {
                obj.insert(k, v);
            }
        }
        Value::Null => {}
        other => {
            obj.insert("data".into(), other);
        }
    }
    let mut line = Value::Object(obj).to_string();
    line.push('\n');
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    #[test]
    fn timestamp_formats_match_js() {
        let t = Utc.with_ymd_and_hms(2026, 10, 18, 15, 4, 5).unwrap()
            + chrono::Duration::milliseconds(7);
        assert_eq!(iso_timestamp(t), "2026-10-18T15:04:05.007Z");
        assert_eq!(session_file_name(t), "session-20261018-150405.jsonl");
    }

    #[test]
    fn event_before_start_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let log = Logger::new(dir.path(), false);
        log.event(json!({"type": "x"}));
        assert!(log.log_file().is_none());
        assert!(!dir.path().join(".qwenodyssey").exists());
    }

    #[test]
    fn start_writes_session_start_then_events() {
        let dir = tempfile::tempdir().unwrap();
        let log = Logger::new(dir.path(), true);
        let file = log.start("fix the bug").unwrap();
        assert!(file.starts_with(dir.path().join(".qwenodyssey").join("logs")));
        let name = file.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("session-") && name.ends_with(".jsonl"), "{name}");

        let shared = log.clone();
        shared.event(json!({"type": "tool", "name": "read_file", "ok": true}));
        (log.log_fn())(json!({"type": "via_log_fn"}));

        let text = fs::read_to_string(&file).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["type"], "session_start");
        assert_eq!(lines[0]["task"], "fix the bug");
        assert_eq!(lines[1]["name"], "read_file");
        assert_eq!(lines[2]["type"], "via_log_fn");
        // ts comes first, ISO with millis + Z
        let first_key = text.lines().next().unwrap();
        assert!(first_key.starts_with("{\"ts\":\""), "{first_key}");
        let ts = lines[0]["ts"].as_str().unwrap();
        assert!(ts.ends_with('Z') && ts.len() == 24, "{ts}");
        assert_eq!(log.log_file(), Some(file));
    }

    #[test]
    fn event_line_shapes() {
        assert_eq!(event_line("T", json!({"a": 1})), "{\"ts\":\"T\",\"a\":1}\n");
        assert_eq!(event_line("T", json!({"ts": "override"})), "{\"ts\":\"override\"}\n");
        assert_eq!(event_line("T", json!("s")), "{\"ts\":\"T\",\"data\":\"s\"}\n");
    }
}
