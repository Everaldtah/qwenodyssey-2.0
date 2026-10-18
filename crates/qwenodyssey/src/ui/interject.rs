//! Turn-time input capture (port of src/cli/prompt.ts `captureInterjections`).
//!
//! While the assistant works, this consumer holds the input lease:
//! - **Esc** interrupts the turn (the caller cancels the turn's
//!   `CancellationToken`, which aborts the request at any stage — connect,
//!   headers, body — so a model that never answers can always be escaped);
//! - **typing** starts a "btw ›" aside on the overlay line (the spinner pauses
//!   so the text isn't clobbered); Enter queues it via `on_submit` and the chat
//!   layer injects it at the next step boundary;
//! - **Ctrl-C** restores the terminal and exits 130, as in 1.x.
//!
//! A y/N confirmation during the turn [`Interjector::suspend`]s the capture
//! (the lease is released and re-acquired afterwards) — the two can never fight
//! over the terminal because they take turns on the same input stream. On
//! [`Interjector::stop`], a half-typed aside is handed back to the next
//! consumer (the prompt editor) instead of being lost.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyModifiers};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use super::input::{is_text_key, Input, InputEvent, Lease};
use super::out::{self, Overlay};
use super::{style, term};

type Callback = Box<dyn Fn() + Send + Sync>;

/// Callbacks for an interjection capture.
pub struct InterjectOpts {
    /// Prompt shown on the aside line, e.g. a yellow "btw › ".
    pub label: String,
    /// A completed aside (trimmed, non-empty).
    pub on_submit: Box<dyn Fn(String) + Send + Sync>,
    pub on_start_typing: Option<Callback>,
    pub on_stop_typing: Option<Callback>,
    /// Esc pressed while the model is working → interrupt the current request.
    pub on_interrupt: Option<Callback>,
}

pub(crate) enum Ctl {
    Suspend(oneshot::Sender<()>),
    Resume,
    Stop(Vec<String>, oneshot::Sender<()>),
}

/// Control channel to a running capture (shared with the UI hub so a confirm
/// prompt can suspend it).
pub(crate) type CtlTx = mpsc::UnboundedSender<Ctl>;

/// Handle to a running interjection capture.
pub struct Interjector {
    ctl: CtlTx,
    typing: Arc<AtomicBool>,
    task: Option<JoinHandle<()>>,
}

impl Interjector {
    /// Start capturing. `None` when stdin is not a TTY (you can't type
    /// concurrently into a pipe anyway).
    pub fn start(input: Arc<Input>, opts: InterjectOpts) -> Option<Self> {
        if !input.is_tty() {
            return None;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let typing = Arc::new(AtomicBool::new(false));
        let cap = Capture { opts, buf: String::new(), lines: Vec::new(), typing: typing.clone() };
        let task = tokio::spawn(run(input, cap, rx));
        Some(Self { ctl: tx, typing, task: Some(task) })
    }

    pub(crate) fn ctl(&self) -> CtlTx {
        self.ctl.clone()
    }

    /// Is the user mid-way through typing an aside?
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn typing(&self) -> bool {
        self.typing.load(Ordering::SeqCst)
    }

    /// The shared "typing an aside" flag (outlives a borrow of the handle).
    pub fn typing_flag(&self) -> Arc<AtomicBool> {
        self.typing.clone()
    }

    /// Release the input (e.g. while a y/N confirm prompt runs). Resolves once
    /// the lease is free.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn suspend(&self) {
        suspend_ctl(&self.ctl).await;
    }

    /// Re-attach after `suspend`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn resume(&self) {
        let _ = self.ctl.send(Ctl::Resume);
    }

    /// Stop for good. `leftovers` (asides queued but never delivered) and any
    /// half-typed aside are handed to the next input consumer as one paste.
    pub async fn stop(mut self, leftovers: Vec<String>) {
        let (ack, done) = oneshot::channel();
        if self.ctl.send(Ctl::Stop(leftovers, ack)).is_ok() {
            let _ = done.await;
        }
        if let Some(t) = self.task.take() {
            let _ = t.await;
        }
    }
}

impl Drop for Interjector {
    fn drop(&mut self) {
        // Dropped without stop() (e.g. a panic unwinding the turn): make sure
        // the capture task ends and releases the lease.
        if let Some(t) = self.task.take() {
            let (ack, _) = oneshot::channel();
            let _ = self.ctl.send(Ctl::Stop(Vec::new(), ack));
            drop(t);
        }
    }
}

pub(crate) async fn suspend_ctl(ctl: &CtlTx) {
    let (ack, done) = oneshot::channel();
    if ctl.send(Ctl::Suspend(ack)).is_ok() {
        let _ = done.await;
    }
}

struct Capture {
    opts: InterjectOpts,
    /// The line being typed.
    buf: String,
    /// Completed lines of a multi-line aside (a paste), flushed as ONE aside.
    lines: Vec<String>,
    typing: Arc<AtomicBool>,
}

impl Capture {
    fn is_typing(&self) -> bool {
        self.typing.load(Ordering::SeqCst)
    }

    fn redraw(&self, cols: usize) {
        // Show the tail that fits: the overlay line must never wrap, or the
        // "clear line" before the next draw would leave the wrapped rows behind.
        let label_w = style::visible_width(&self.opts.label);
        let room = cols.saturating_sub(label_w + 1).max(1);
        let w = style::visible_width(&self.buf);
        let shown: String = if w <= room {
            self.buf.clone()
        } else {
            let mut tail: Vec<char> = Vec::new();
            let mut tw = 0usize;
            for ch in self.buf.chars().rev() {
                let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
                if tw + cw + 1 > room {
                    break;
                }
                tw += cw;
                tail.push(ch);
            }
            tail.reverse();
            format!("…{}", tail.into_iter().collect::<String>())
        };
        out::set_overlay(Overlay::Aside, &format!("{}{shown}", self.opts.label));
    }

    fn begin_typing(&mut self) {
        if self.is_typing() {
            return;
        }
        self.typing.store(true, Ordering::SeqCst);
        if let Some(f) = &self.opts.on_start_typing {
            f();
        }
    }

    fn end_typing(&mut self) {
        self.typing.store(false, Ordering::SeqCst);
        self.buf.clear();
        self.lines.clear();
        if let Some(f) = &self.opts.on_stop_typing {
            f();
        }
    }

    fn joined(&self) -> String {
        let mut all = self.lines.clone();
        all.push(self.buf.clone());
        all.join("\n")
    }

    /// Submit the aside being typed.
    fn flush(&mut self) {
        if !self.is_typing() {
            return;
        }
        let text = self.joined().trim().to_string();
        out::clear_overlay(Overlay::Aside);
        self.end_typing();
        if !text.is_empty() {
            (self.opts.on_submit)(text);
        }
    }

    /// A pasted / scripted chunk: lines inside it are gathered, and a chunk
    /// that contains a newline is submitted as ONE aside (1.x flooded the
    /// screen with one aside per pasted line).
    fn commit_paste(&mut self, raw: &str, cols: usize) {
        let text = raw.replace("\r\n", "\n").replace('\r', "\n");
        if text.is_empty() {
            return;
        }
        self.begin_typing();
        let parts: Vec<&str> = text.split('\n').collect();
        for p in &parts[..parts.len() - 1] {
            let line = format!("{}{p}", self.buf);
            self.lines.push(line);
            self.buf.clear();
        }
        self.buf.push_str(parts[parts.len() - 1]);
        self.redraw(cols);
        if text.contains('\n') {
            self.flush();
        }
    }

    fn handle(&mut self, ev: InputEvent, input: &Input) {
        let cols = input.cols() as usize;
        match ev {
            InputEvent::Paste(t) | InputEvent::Line(t) => self.commit_paste(&t, cols),
            InputEvent::Resize(c, r) => {
                input.note_resize(c, r);
                if self.is_typing() {
                    self.redraw(c as usize);
                }
            }
            InputEvent::Eof => {}
            InputEvent::Key(k) => {
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL) && !k.modifiers.contains(KeyModifiers::ALT);
                if ctrl && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('C')) {
                    // Preserve Ctrl-C = quit during a turn (raw mode turns it
                    // into a key, so do what the signal used to do).
                    out::clear_overlay(Overlay::Aside);
                    out::clear_overlay(Overlay::Spinner);
                    out::raw("\n");
                    term::exit(130);
                }
                if let Some(c) = is_text_key(&k) {
                    self.begin_typing();
                    self.buf.push(c);
                    self.redraw(cols);
                    return;
                }
                match k.code {
                    KeyCode::Esc => {
                        if self.opts.on_interrupt.is_none() {
                            return;
                        }
                        // Clear a half-typed aside first so it isn't left
                        // dangling on screen.
                        if self.is_typing() {
                            out::clear_overlay(Overlay::Aside);
                            self.end_typing();
                        }
                        if let Some(f) = &self.opts.on_interrupt {
                            f();
                        }
                    }
                    KeyCode::Enter => {
                        if !self.is_typing() {
                            return;
                        }
                        let line = std::mem::take(&mut self.buf);
                        self.lines.push(line);
                        self.flush();
                    }
                    KeyCode::Backspace => {
                        if !self.is_typing() {
                            return;
                        }
                        self.buf.pop();
                        self.redraw(cols);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Hide the aside line while suspended (the typed text is kept).
    fn hide(&self) {
        if self.is_typing() {
            out::clear_overlay(Overlay::Aside);
        }
    }

    fn show(&self, cols: usize) {
        if self.is_typing() {
            self.redraw(cols);
        }
    }

    /// Final stop: hand leftovers + a half-typed aside to the next consumer.
    fn stop(&mut self, lease: &mut Lease, leftovers: Vec<String>) {
        let mut parts: Vec<String> = leftovers.into_iter().filter(|s| !s.trim().is_empty()).collect();
        if self.is_typing() {
            out::clear_overlay(Overlay::Aside);
            let partial = self.joined();
            if !partial.trim().is_empty() {
                parts.push(partial);
            }
            self.typing.store(false, Ordering::SeqCst);
            self.buf.clear();
            self.lines.clear();
        }
        if !parts.is_empty() {
            lease.unread_front(InputEvent::Paste(parts.join("\n")));
        }
    }
}

async fn run(input: Arc<Input>, mut cap: Capture, mut ctl: mpsc::UnboundedReceiver<Ctl>) {
    'lease: loop {
        // Wait for the lease, but stay responsive to control messages.
        let mut lease = tokio::select! {
            l = input.lease() => l,
            c = ctl.recv() => match c {
                Some(Ctl::Suspend(ack)) => { let _ = ack.send(()); continue 'lease; }
                Some(Ctl::Resume) => continue 'lease,
                Some(Ctl::Stop(left, ack)) => {
                    // Not holding the lease: deliver leftovers on a fresh one.
                    let mut l = input.lease().await;
                    cap.stop(&mut l, left);
                    let _ = ack.send(());
                    return;
                }
                None => return,
            },
        };
        cap.show(input.cols() as usize);
        let mut eof = false;
        loop {
            tokio::select! {
                biased;
                c = ctl.recv() => match c {
                    Some(Ctl::Suspend(ack)) => {
                        cap.hide();
                        drop(lease);
                        let _ = ack.send(());
                        // Wait for resume (or stop) without holding the input.
                        loop {
                            match ctl.recv().await {
                                Some(Ctl::Resume) => continue 'lease,
                                Some(Ctl::Suspend(a)) => { let _ = a.send(()); }
                                Some(Ctl::Stop(left, ack)) => {
                                    let mut l = input.lease().await;
                                    cap.stop(&mut l, left);
                                    let _ = ack.send(());
                                    return;
                                }
                                None => return,
                            }
                        }
                    }
                    Some(Ctl::Resume) => {}
                    Some(Ctl::Stop(left, ack)) => {
                        cap.stop(&mut lease, left);
                        drop(lease);
                        let _ = ack.send(());
                        return;
                    }
                    None => {
                        cap.stop(&mut lease, Vec::new());
                        return;
                    }
                },
                ev = lease.next(), if !eof => {
                    if ev == InputEvent::Eof {
                        // Input is gone: keep the Eof for the next consumer and
                        // only wait for control messages from now on.
                        lease.unread(InputEvent::Eof);
                        eof = true;
                        continue;
                    }
                    cap.handle(ev, &input);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    struct Rec {
        submitted: Arc<Mutex<Vec<String>>>,
        interrupts: Arc<Mutex<u32>>,
        starts: Arc<Mutex<u32>>,
    }

    fn opts() -> (InterjectOpts, Rec) {
        let submitted = Arc::new(Mutex::new(Vec::new()));
        let interrupts = Arc::new(Mutex::new(0));
        let starts = Arc::new(Mutex::new(0));
        let (s, i, st) = (submitted.clone(), interrupts.clone(), starts.clone());
        (
            InterjectOpts {
                label: "btw › ".into(),
                on_submit: Box::new(move |t| s.lock().unwrap().push(t)),
                on_start_typing: Some(Box::new(move || *st.lock().unwrap() += 1)),
                on_stop_typing: None,
                on_interrupt: Some(Box::new(move || *i.lock().unwrap() += 1)),
            },
            Rec { submitted, interrupts, starts },
        )
    }

    async fn settle() {
        tokio::time::sleep(Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn esc_interrupts_and_typing_queues_an_aside() {
        let (input, tx) = Input::from_channel(true);
        let (o, rec) = opts();
        let ij = Interjector::start(input.clone(), o).unwrap();
        tx.send(InputEvent::key(KeyCode::Char('h'))).unwrap();
        tx.send(InputEvent::key(KeyCode::Char('i'))).unwrap();
        settle().await;
        assert!(ij.typing());
        tx.send(InputEvent::key(KeyCode::Enter)).unwrap();
        tx.send(InputEvent::key(KeyCode::Esc)).unwrap();
        settle().await;
        assert_eq!(*rec.submitted.lock().unwrap(), vec!["hi".to_string()]);
        assert_eq!(*rec.interrupts.lock().unwrap(), 1);
        assert_eq!(*rec.starts.lock().unwrap(), 1);
        ij.stop(Vec::new()).await;
    }

    #[tokio::test]
    async fn multiline_paste_is_one_aside() {
        let (input, tx) = Input::from_channel(true);
        let (o, rec) = opts();
        let ij = Interjector::start(input.clone(), o).unwrap();
        tx.send(InputEvent::Paste("one\ntwo\nthree".into())).unwrap();
        settle().await;
        assert_eq!(*rec.submitted.lock().unwrap(), vec!["one\ntwo\nthree".to_string()]);
        ij.stop(Vec::new()).await;
    }

    #[tokio::test]
    async fn suspend_releases_the_input_for_a_confirm() {
        let (input, tx) = Input::from_channel(true);
        let (o, rec) = opts();
        let ij = Interjector::start(input.clone(), o).unwrap();
        settle().await;
        ij.suspend().await;
        tx.send(InputEvent::Line("y".into())).unwrap();
        {
            let mut l = input.lease().await; // must not deadlock
            assert_eq!(l.next().await, InputEvent::Line("y".into()));
        }
        ij.resume();
        tx.send(InputEvent::key(KeyCode::Esc)).unwrap();
        settle().await;
        assert_eq!(*rec.interrupts.lock().unwrap(), 1);
        ij.stop(Vec::new()).await;
    }

    #[tokio::test]
    async fn stop_hands_half_typed_text_to_the_prompt() {
        let (input, tx) = Input::from_channel(true);
        let (o, _rec) = opts();
        let ij = Interjector::start(input.clone(), o).unwrap();
        tx.send(InputEvent::Paste("next q".into())).unwrap();
        settle().await;
        ij.stop(vec!["queued aside".into()]).await;
        let mut l = input.lease().await;
        assert_eq!(l.next().await, InputEvent::Paste("queued aside\nnext q".into()));
    }

    #[tokio::test]
    async fn eof_does_not_spin_and_is_kept() {
        let (input, tx) = Input::from_channel(true);
        let (o, _rec) = opts();
        let ij = Interjector::start(input.clone(), o).unwrap();
        drop(tx);
        settle().await;
        ij.stop(Vec::new()).await;
        let mut l = input.lease().await;
        assert_eq!(l.next().await, InputEvent::Eof);
    }

    #[test]
    fn not_started_without_a_tty() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let (input, _tx) = Input::from_channel(false);
            let (o, _rec) = opts();
            assert!(Interjector::start(input, o).is_none());
        });
    }
}
