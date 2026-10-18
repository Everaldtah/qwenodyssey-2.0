//! Minimal pipeline state machine + trace. Mirrors the Odysseus-style
//! step-by-step task execution so runs are inspectable.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Intent,
    Scan,
    Plan,
    Context,
    Code,
    Patch,
    Review,
    Test,
    Fix,
    Summary,
    Done,
}

/// The stages in execution order.
pub const PIPELINE: [Stage; 11] = [
    Stage::Intent,
    Stage::Scan,
    Stage::Plan,
    Stage::Context,
    Stage::Code,
    Stage::Patch,
    Stage::Review,
    Stage::Test,
    Stage::Fix,
    Stage::Summary,
    Stage::Done,
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub stage: Stage,
    /// ISO-8601 timestamp.
    pub at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug)]
pub struct StateMachine {
    stage: Stage,
    pub trace: Vec<TraceEntry>,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    pub fn new() -> Self {
        Self { stage: Stage::Intent, trace: Vec::new() }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn enter(&mut self, stage: Stage, note: Option<&str>) {
        self.stage = stage;
        self.trace.push(TraceEntry {
            stage,
            at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            note: note.map(str::to_string),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_trace() {
        let mut sm = StateMachine::new();
        assert_eq!(sm.stage(), Stage::Intent);
        sm.enter(Stage::Scan, Some("x"));
        sm.enter(Stage::Plan, None);
        assert_eq!(sm.stage(), Stage::Plan);
        assert_eq!(sm.trace.len(), 2);
        assert_eq!(sm.trace[0].note.as_deref(), Some("x"));
        assert_eq!(PIPELINE.first(), Some(&Stage::Intent));
        assert_eq!(PIPELINE.last(), Some(&Stage::Done));
    }
}
