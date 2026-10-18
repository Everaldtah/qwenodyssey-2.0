//! The pipeline's only window to the terminal.
//!
//! The TypeScript orchestrator printed straight to the console (chalk) and
//! asked questions with `prompts`. Library crates must not touch the terminal
//! (the binary's input owner would fight them for the keyboard), so every
//! status line, rendered block and yes/no question goes through
//! [`PipelineUi`], which the binary implements on top of its UI layer.

use async_trait::async_trait;

use crate::render::{colorize_diff, render_edit_header};
use qo_core::FileEdit;

/// Kind of one-line status message. The binary adds the glyph + colour the
/// TS logger used: Step `→ ` (cyan), Success `✓ ` (green), Warn `! ` (yellow),
/// Error `✗ ` (red), Debug `· ` (gray, verbose only), Info plain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Step,
    Info,
    Success,
    Warn,
    Error,
    Debug,
}

#[async_trait]
pub trait PipelineUi: Send + Sync {
    /// One status line.
    fn line(&self, level: Level, msg: &str);

    /// A pre-rendered multi-line block (plan, review, colourised diff). May
    /// contain ANSI escapes.
    fn block(&self, text: &str);

    /// Yes/no question (`initial` is the default answer the TS prompt offered).
    /// Non-interactive implementations should answer `false` for anything
    /// destructive; the pipeline only asks when `--yes`/auto-apply is off.
    async fn ask_yes_no(&self, question: &str, initial: bool) -> bool;

    /// Show one edit's header and colourised diff preview.
    fn show_edit_diff(&self, edit: &FileEdit, diff: &str) {
        self.block(&format!("{}\n{}", render_edit_header(edit), colorize_diff(diff)));
    }

    fn step(&self, msg: &str) {
        self.line(Level::Step, msg)
    }
    fn info(&self, msg: &str) {
        self.line(Level::Info, msg)
    }
    fn success(&self, msg: &str) {
        self.line(Level::Success, msg)
    }
    fn warn(&self, msg: &str) {
        self.line(Level::Warn, msg)
    }
    fn error(&self, msg: &str) {
        self.line(Level::Error, msg)
    }
    fn debug(&self, msg: &str) {
        self.line(Level::Debug, msg)
    }
}

/// A UI that shows nothing and answers every question with a fixed value.
/// Useful for headless runs (`--yes`) and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct SilentUi {
    pub answer: bool,
}

impl SilentUi {
    pub fn answering(answer: bool) -> Self {
        Self { answer }
    }
}

#[async_trait]
impl PipelineUi for SilentUi {
    fn line(&self, _level: Level, _msg: &str) {}
    fn block(&self, _text: &str) {}
    async fn ask_yes_no(&self, _question: &str, _initial: bool) -> bool {
        self.answer
    }
}

/// A UI that records everything it is shown (tests, or a binary that wants to
/// buffer output). Questions are answered from `answers` in order, then with
/// `default_answer`.
#[derive(Debug, Default)]
pub struct RecordingUi {
    pub lines: std::sync::Mutex<Vec<(Level, String)>>,
    pub blocks: std::sync::Mutex<Vec<String>>,
    pub questions: std::sync::Mutex<Vec<String>>,
    pub answers: std::sync::Mutex<std::collections::VecDeque<bool>>,
    pub default_answer: bool,
}

impl RecordingUi {
    pub fn new(default_answer: bool) -> Self {
        Self { default_answer, ..Default::default() }
    }
    /// Queue answers for the next questions.
    pub fn with_answers(self, answers: impl IntoIterator<Item = bool>) -> Self {
        if let Ok(mut q) = self.answers.lock() {
            q.extend(answers);
        }
        self
    }
    pub fn lines(&self) -> Vec<(Level, String)> {
        self.lines.lock().map(|l| l.clone()).unwrap_or_default()
    }
    pub fn blocks(&self) -> Vec<String> {
        self.blocks.lock().map(|b| b.clone()).unwrap_or_default()
    }
    pub fn questions(&self) -> Vec<String> {
        self.questions.lock().map(|q| q.clone()).unwrap_or_default()
    }
    /// Every line and block joined, for substring assertions.
    pub fn transcript(&self) -> String {
        let mut s = String::new();
        for (_, l) in self.lines() {
            s.push_str(&l);
            s.push('\n');
        }
        for b in self.blocks() {
            s.push_str(&b);
            s.push('\n');
        }
        s
    }
}

#[async_trait]
impl PipelineUi for RecordingUi {
    fn line(&self, level: Level, msg: &str) {
        if let Ok(mut l) = self.lines.lock() {
            l.push((level, msg.to_string()));
        }
    }
    fn block(&self, text: &str) {
        if let Ok(mut b) = self.blocks.lock() {
            b.push(text.to_string());
        }
    }
    async fn ask_yes_no(&self, question: &str, _initial: bool) -> bool {
        if let Ok(mut q) = self.questions.lock() {
            q.push(question.to_string());
        }
        self.answers.lock().ok().and_then(|mut a| a.pop_front()).unwrap_or(self.default_answer)
    }
}
