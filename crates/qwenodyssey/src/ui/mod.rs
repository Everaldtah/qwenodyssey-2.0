//! The interactive terminal layer.
//!
//! - [`term`]: raw mode + bracketed paste, switched on ONCE per chat session
//!   and restored by a guard / panic hook / `term::exit`.
//! - [`input`]: the ONE input owner (a thread reading terminal events into a
//!   channel); consumers take turns through a lease.
//! - [`out`]: the single UI writer ("\n" → "\r\n" in raw mode, overlay line).
//! - [`editor`] (prompt + `/` palette), [`picker`] (arrow-key list),
//!   [`interject`] (Esc / "btw" capture during a turn), [`spinner`].
//! - [`Ui`]: the hub the chat loop uses; it serializes those consumers and
//!   implements the qo-core [`Confirmer`] for y/N gates.

pub mod editor;
pub mod input;
pub mod interject;
pub mod out;
pub mod picker;
pub mod spinner;
pub mod style;
pub mod term;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use qo_core::Confirmer;

pub use editor::SlashCommand;
pub use input::Input;
#[cfg(test)]
use input::InputEvent;
pub use interject::{InterjectOpts, Interjector};
pub use picker::SelectItem;
pub use spinner::Spinner;

/// `println!` through the UI writer.
#[macro_export]
macro_rules! outln {
    () => { $crate::ui::out::line("") };
    ($($arg:tt)*) => { $crate::ui::out::line(&format!($($arg)*)) };
}

/// The interactive session's input hub.
pub struct Ui {
    input: Arc<Input>,
    /// Control channel of the interjection capture running right now (if any),
    /// so a prompt that needs the keyboard mid-turn can suspend it first.
    active: Mutex<Option<interject::CtlTx>>,
}

impl Ui {
    pub fn new(input: Arc<Input>) -> Arc<Self> {
        Arc::new(Self { input, active: Mutex::new(None) })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn input(&self) -> &Arc<Input> {
        &self.input
    }

    /// Interactive terminal (raw keys), as opposed to piped lines.
    pub fn is_tty(&self) -> bool {
        self.input.is_tty()
    }

    fn active_ctl(&self) -> Option<interject::CtlTx> {
        self.active.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Read one line with the slash palette. If a turn's interjection capture
    /// is running it is suspended for the duration (the TS `askGuarded`).
    pub async fn read_line(&self, label: &str, commands: &'static [SlashCommand]) -> String {
        let ctl = self.active_ctl();
        if let Some(c) = &ctl {
            interject::suspend_ctl(c).await;
        }
        let line = {
            let mut lease = self.input.lease().await;
            editor::read_line(&mut lease, &self.input, label, commands).await
        };
        if let Some(c) = &ctl {
            let _ = c.send(interject::Ctl::Resume);
        }
        line
    }

    /// Arrow-key picker (`None` = cancelled / not a TTY).
    pub async fn select(&self, title: &str, items: Vec<SelectItem>, initial: usize) -> Option<usize> {
        picker::select_from_list(&self.input, title, items, initial).await
    }

    /// Start a turn's interjection capture and register it as the active one.
    pub fn start_interjector(&self, opts: InterjectOpts) -> Option<Interjector> {
        let ij = Interjector::start(self.input.clone(), opts)?;
        *self.active.lock().unwrap_or_else(|e| e.into_inner()) = Some(ij.ctl());
        Some(ij)
    }

    /// Stop a capture started with [`Ui::start_interjector`].
    pub async fn stop_interjector(&self, ij: Interjector, leftovers: Vec<String>) {
        *self.active.lock().unwrap_or_else(|e| e.into_inner()) = None;
        ij.stop(leftovers).await;
    }
}

/// y/N gate on top of the shared input: prints the question and reads one
/// answer line through the same input owner as everything else, so a
/// confirmation during a turn can never fight the prompt for the terminal.
pub struct TermConfirmer {
    ui: Arc<Ui>,
}

impl TermConfirmer {
    pub fn new(ui: Arc<Ui>) -> Self {
        Self { ui }
    }
}

/// Is a typed answer a "yes"?
pub fn is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
}

#[async_trait]
impl Confirmer for TermConfirmer {
    async fn confirm(&self, question: &str) -> bool {
        out::line(&style::yellow(&format!("  ⚠ {question} — type 'y' to confirm, anything else to skip:")));
        let answer = self.ui.read_line(&style::yellow("  y/N › "), &[]).await;
        is_yes(&answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyCode;

    #[tokio::test]
    async fn confirmer_reads_from_shared_input_while_capture_runs() {
        let (input, tx) = Input::from_channel(true);
        let ui = Ui::new(input);
        let ij = ui
            .start_interjector(InterjectOpts {
                label: "btw › ".into(),
                on_submit: Box::new(|_| {}),
                on_start_typing: None,
                on_stop_typing: None,
                on_interrupt: Some(Box::new(|| {})),
            })
            .unwrap();
        let c = TermConfirmer::new(ui.clone());
        tx.send(InputEvent::Paste("y".into())).unwrap();
        tx.send(InputEvent::key(KeyCode::Enter)).unwrap();
        // The capture may have consumed the scripted "y" as an aside before the
        // confirm suspended it; either way nothing deadlocks. Feed a second
        // answer that only the confirm can see.
        let answer = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            let fut = c.confirm("delete it?");
            tokio::pin!(fut);
            tokio::select! {
                r = &mut fut => r,
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                    tx.send(InputEvent::Paste("yes".into())).unwrap();
                    tx.send(InputEvent::key(KeyCode::Enter)).unwrap();
                    fut.await
                }
            }
        })
        .await
        .expect("confirm must not deadlock");
        assert!(answer);
        ui.stop_interjector(ij, Vec::new()).await;
    }

    #[test]
    fn yes_answers() {
        assert!(is_yes("y"));
        assert!(is_yes(" YES "));
        assert!(!is_yes("n"));
        assert!(!is_yes(""));
    }
}
