//! `qwenodyssey live` — a hands-free, continuous voice+vision conversation.
//!
//! Pipeline (senses local, brain remote, voice local):
//! ```text
//!   mic ─▶ whisper.cpp stream (local STT, voice-activity detection) ─┐
//!                                                                     ├▶ frontier model ─▶ reply ─▶ Piper TTS ─▶ speakers
//!   camera frame ─▶ vision model (NIM / Ollama, e.g. moondream) ──────┘
//! ```
//! You just talk — whisper-stream segments your speech with VAD (no
//! push-to-talk). On each utterance a fresh camera frame is described and
//! bundled with your words for the frontier model; its reply is spoken aloud.
//! Listening is muted while it speaks (so it doesn't transcribe its own voice).
//!
//! With `[omni].enabled` a single multimodal model does everything instead:
//! one ffmpeg mic reader feeds our own VAD ([`crate::micstream`]) and each
//! utterance's audio (+ a fresh frame) goes straight to the omni model.
//!
//! Keys: `c` toggle camera · `m` mute/unmute mic · `q` / Esc / Ctrl-C quit.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Result;
use crossterm::style::Stylize;
use qo_core::{CancellationToken, Message, ModelOptions, SharedProvider};
use regex::Regex;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::bins::{file_label, resolve_ffmpeg, resolve_whisper_stream, whisper_model_path};
use crate::media::{capture_frame, collapse_ws, err_msg, fmt_num, pick_audio_device, pick_video_device, LineSplitter};
use crate::micstream::{start_mic_vad, VadEvent, VadOptions};
use crate::omni::{ask_omni, OmniInput};
use crate::proc::{self, head_chars};
use crate::settings::MediaSettings;
use crate::term::{gray, say, thinking_word, Key, Keys, Spinner};
use crate::tts::{speak, tts_available, TtsStatus};
use crate::vision::describe_image;

pub const LIVE_SYSTEM: &str = "You are Qwenodyssey in LIVE mode: a hands-free spoken conversation. The user talks to you \
(their speech is transcribed, so expect minor errors) and shares what their webcam sees. Lines \
prefixed with [Camera view: …] are an automatic description of the CURRENT camera frame — treat \
them as your own eyes, not the user's words. Your replies are SPOKEN ALOUD, so be brief, natural, \
and conversational; avoid code blocks, lists, and long output unless asked.";

pub const OMNI_INSTRUCTION: &str = "You are Qwenodyssey, a live voice assistant. The user just spoke to you (their audio is \
attached); a webcam frame may also be attached as your eyes. Reply briefly and naturally — it \
is read aloud. No code blocks or lists unless asked.";

/// Ignore the tail of our own speech for this long after a reply.
const SELF_HEARING_COOLDOWN: Duration = Duration::from_millis(1200);
/// Omni mode: resume listening this long after a reply finished.
const OMNI_RESUME_DELAY: Duration = Duration::from_millis(400);
/// Omni mode: don't repeat "hearing you…" more often than this.
const HEARING_NOTICE_GAP: Duration = Duration::from_millis(1500);

/* ── whisper-stream output ── */

/// `whisper-stream -m <model> -l <lang> --step 0 --length 30000 -vth <vad>`
/// (step 0 = VAD segment mode: one transcription block per utterance).
pub fn whisper_stream_args(model: &Path, language: &str, vad_threshold: f64) -> Vec<String> {
    vec![
        "-m".into(),
        model.to_string_lossy().into_owned(),
        "-l".into(),
        language.to_string(),
        "--step".into(),
        "0".into(),
        "--length".into(),
        "30000".into(),
        "-vth".into(),
        fmt_num(vad_threshold),
    ]
}

static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").expect("valid ansi regex"));
static RE_START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^###\s*Transcription\s+\d+\s+START").expect("valid start regex"));
static RE_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^###\s*Transcription\s+\d+\s+END").expect("valid end regex"));
static NON_SPEECH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(?:blank_audio|inaudible|music|silence|noise|sound)\]").expect("valid tag regex")
});
static LEADING_JUNK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\s.\-]+").expect("valid junk regex"));

/// Clean a captured segment: drop ANSI, bracketed non-speech tags, leading
/// dots/dashes. Needs at least two word characters to count as a real
/// utterance (else "").
pub fn extract_speech(seg: &str) -> String {
    let t = ANSI.replace_all(seg, " ");
    let t = NON_SPEECH.replace_all(&t, " ");
    let t = collapse_ws(&t);
    let t = LEADING_JUNK.replace(&t, "").trim().to_string();
    if t.chars().filter(|c| c.is_ascii_alphanumeric()).count() >= 2 {
        t
    } else {
        String::new()
    }
}

/// whisper-stream (`--step 0` / VAD) prints each utterance as a block:
/// ```text
///   ### Transcription N START | t0 = … | t1 = …
///   <the transcribed text, possibly several lines>
///   ### Transcription N END
/// ```
/// We capture the text BETWEEN the markers — never the marker line itself.
/// Some builds print text inline without markers; stray lines count too.
#[derive(Debug, Default)]
pub struct TranscriptParser {
    capturing: bool,
    seg: String,
}

impl TranscriptParser {
    /// Feed one raw stdout line; returns the cleaned utterance when one completes.
    pub fn push_line(&mut self, raw: &str) -> Option<String> {
        let stripped = ANSI.replace_all(raw, "");
        let line = stripped.trim();
        if RE_START.is_match(line) {
            self.capturing = true;
            self.seg.clear();
            return None;
        }
        if RE_END.is_match(line) {
            self.capturing = false;
            let seg = std::mem::take(&mut self.seg);
            return non_empty(extract_speech(&seg));
        }
        if self.capturing {
            self.seg.push(' ');
            self.seg.push_str(line);
            return None;
        }
        if !line.is_empty() && !line.starts_with("###") {
            return non_empty(extract_speech(line));
        }
        None
    }
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

/* ── shared turn plumbing ── */

/// A response in flight, polled by the command's select loop.
type Inflight<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Everything one non-omni turn needs.
pub(crate) struct LiveCtx<'a> {
    pub settings: &'a MediaSettings,
    pub provider: SharedProvider,
    pub ffmpeg: PathBuf,
    /// The camera to describe, when the camera is ON for this turn.
    pub camera: Option<String>,
    pub tts: TtsStatus,
    pub cancel: CancellationToken,
}

/// One live turn: optional camera description, a streamed reply, spoken
/// aloud. Takes the history by value and hands it back (the select loop owns
/// it between turns). Errors are printed, never returned.
pub(crate) async fn live_turn(mut history: Vec<Message>, user_text: String, ctx: LiveCtx<'_>) -> Vec<Message> {
    say(format!("{}{user_text}", gray("  you said: ")));
    let mut content = user_text.clone();
    if let Some(cam) = &ctx.camera {
        let described = async {
            let frame = capture_frame(cam, &ctx.ffmpeg, &ctx.cancel).await?;
            describe_image(&frame, &user_text, ctx.settings, &ctx.cancel).await
        }
        .await;
        match described {
            Ok(desc) if !desc.is_empty() => {
                say(gray(&format!("  👁  {}", head_chars(&collapse_ws(&desc), 180))));
                content = format!("[Camera view: {desc}]\n\n{user_text}");
            }
            Ok(_) => {}
            Err(e) => say(format!("  (vision skipped: {})", err_msg(&e)).yellow()),
        }
    }
    history.push(Message::user(content));

    let result = async {
        let mut spinner = Spinner::begin(thinking_word());
        let counter = spinner.counter();
        let provider = &ctx.provider;
        // Stream so the token meter ticks live; count output tokens as they
        // arrive and reconcile to exact usage at the end.
        let mut streamed = String::new();
        let mut last_counted = 0usize;
        let mut on_chunk = |d: &str| {
            streamed.push_str(d);
            let est = provider.count_tokens(&streamed);
            if est > last_counted {
                counter.bump((est - last_counted) as u64);
                last_counted = est;
            }
        };
        let opts = ModelOptions {
            temperature: Some(0.5),
            max_tokens: Some(ctx.settings.model_max_tokens),
            cancel: Some(ctx.cancel.clone()),
            ..Default::default()
        };
        let res = provider.stream(&history, &mut on_chunk, &opts).await;
        if let Ok(r) = &res {
            if let Some(n) = r.completion_tokens {
                spinner.set_out(n);
            }
        }
        spinner.stop();
        res
    }
    .await;

    match result {
        Ok(res) => {
            let answer = res.text.trim().to_string();
            say(format!("{}{}\r\n", "  model › ".dark_green(), if answer.is_empty() { "(no response)" } else { &answer }));
            history.push(Message::assistant(answer.clone()));
            if !answer.is_empty() && ctx.tts.ok {
                speak(&answer, &ctx.settings.tts, &ctx.cancel).await;
            }
        }
        Err(e) => say(format!("  [error: {}]\r\n", err_msg(&e)).dark_red()),
    }
    history
}

/// One omni turn: optional fresh frame, audio + frame to the omni model, the
/// reply printed and spoken. Errors are printed, never returned.
pub(crate) async fn omni_turn(
    wav: PathBuf,
    camera: Option<String>,
    settings: &MediaSettings,
    ffmpeg: &Path,
    tts: &TtsStatus,
    cancel: &CancellationToken,
) {
    let mut image_path = None;
    if let Some(cam) = &camera {
        match capture_frame(cam, ffmpeg, cancel).await {
            Ok(p) => image_path = Some(p),
            Err(e) => say(format!("  (no camera frame: {})", err_msg(&e)).yellow()),
        }
    }
    let input = OmniInput {
        model: settings.omni.model.clone(),
        text: OMNI_INSTRUCTION.into(),
        audio_path: Some(wav),
        image_path,
        max_tokens: Some(settings.model_max_tokens),
        ..Default::default()
    };
    let mut spinner = Spinner::begin(thinking_word());
    let reply = ask_omni(&input, settings, cancel).await;
    spinner.stop();
    match reply {
        Ok(reply) => {
            say(format!("{}{}\r\n", "  model › ".dark_green(), if reply.is_empty() { "(no response)" } else { &reply }));
            if !reply.is_empty() && tts.ok {
                speak(&reply, &settings.tts, cancel).await;
            }
        }
        Err(e) => say(format!("  [error: {}]\r\n", err_msg(&e)).dark_red()),
    }
}

/// Await the in-flight response, or pend forever when there is none (for use
/// as a `select!` branch).
async fn poll_inflight<T>(f: &mut Option<Inflight<'_, T>>) -> T {
    match f.as_mut() {
        Some(f) => f.await,
        None => std::future::pending().await,
    }
}

async fn sleep_until_opt(t: Option<Instant>) {
    match t {
        Some(t) => tokio::time::sleep_until(t).await,
        None => std::future::pending().await,
    }
}

fn voice_line(tts: &TtsStatus) -> String {
    if tts.ok {
        tts.detail.clone()
    } else {
        format!("off ({})", tts.detail).yellow().to_string()
    }
}

/* ── live (whisper-stream) ── */

/// Run `qwenodyssey live`. `provider` is the chat model (from the provider
/// factory); it is only used when omni mode is off. Owns the terminal until
/// the user quits (`q` / Esc / Ctrl-C) or the listener exits.
pub async fn run_live(settings: &MediaSettings, provider: SharedProvider) -> Result<()> {
    if settings.omni.enabled {
        return run_omni_live(settings).await;
    }
    let ffmpeg = resolve_ffmpeg();
    let cancel = CancellationToken::new();
    let cam = if settings.vision.enabled {
        pick_video_device(&settings.vision.device, &ffmpeg, &cancel).await.ok().flatten()
    } else {
        None
    };
    let whisper_stream = resolve_whisper_stream();
    let model = whisper_model_path(&settings.audio.whisper_model);
    let tts = tts_available(&settings.tts);

    say("\r\n  Qwenodyssey — live voice + vision  (continuous)\r\n".bold());
    say(format!("{}{} ({})", gray("  brain   : "), provider.model(), provider.name()));
    say(format!(
        "{}whisper-stream · {} (local, VAD {})",
        gray("  hearing : "),
        file_label(&model.to_string_lossy()),
        fmt_num(settings.audio.vad_threshold)
    ));
    let vision_line = match &cam {
        Some(c) => format!("{c} → {} ({})", settings.vision.model, settings.vision.provider.as_str()),
        None => "none".yellow().to_string(),
    };
    say(format!("{}{vision_line}", gray("  vision  : ")));
    say(format!("{}{}", gray("  voice   : "), voice_line(&tts)));
    say(format!(
        "\r\n  {}{}{}{}{}{}{}\r\n",
        gray("Just talk. "),
        "c".dark_cyan(),
        gray(" toggle camera · "),
        "m".dark_cyan(),
        gray(" mute/unmute mic · "),
        "q".dark_cyan(),
        gray(" quit")
    ));

    // ── Continuous listener: whisper-stream in VAD segment mode (--step 0). ──
    if !settings.audio.enabled {
        say("  [audio] is disabled — nothing to listen with.\r\n".dark_red());
        return Ok(());
    }
    let args = whisper_stream_args(&model, &settings.audio.language, settings.audio.vad_threshold);
    let mut cmd = proc::command(&whisper_stream, &args);
    // whisper-stream captures audio via SDL, not stdin; its stderr is never
    // read, so don't let it fill a pipe and stall the process.
    cmd.stderr(Stdio::null());
    let mut ws = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            say(format!("  failed to start whisper-stream: {e}\r\n").dark_red());
            return Ok(());
        }
    };
    let (line_tx, mut lines) = mpsc::unbounded_channel::<String>();
    let reader = ws.stdout.take().map(|mut out| {
        tokio::spawn(async move {
            let mut splitter = LineSplitter::default();
            let mut buf = [0u8; 4096];
            loop {
                match out.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => splitter.push(&buf[..n], |l| {
                        let _ = line_tx.send(l.to_string());
                    }),
                }
            }
            splitter.finish(|l| {
                let _ = line_tx.send(l.to_string());
            });
        })
    });

    let mut history = vec![Message::system(LIVE_SYSTEM)];
    let mut camera_on = settings.vision.enabled && cam.is_some();
    let mut muted = false;
    let mut muted_until: Option<Instant> = None; // brief cooldown after speaking
    let mut parser = TranscriptParser::default();
    let mut inflight: Option<Inflight<'_, Vec<Message>>> = None;
    let mut keys = Keys::start();

    loop {
        tokio::select! {
            key = keys.next() => {
                if key.is_quit() {
                    say(gray("\r\n  ending live session…\r\n"));
                    break;
                }
                match key {
                    Key::Char('c') => {
                        if cam.is_none() {
                            say("  (no camera)\r\n".yellow());
                        } else {
                            camera_on = !camera_on;
                            say(gray(&format!("  camera {}", if camera_on { "ON" } else { "off" })));
                        }
                    }
                    Key::Char('m') => {
                        muted = !muted;
                        say(gray(&format!("  mic {}", if muted { "MUTED" } else { "live" })));
                    }
                    _ => {}
                }
            }
            line = lines.recv() => {
                let Some(line) = line else { break }; // whisper-stream exited
                let Some(text) = parser.push_line(&line) else { continue };
                let cooling = muted_until.map_or(false, |t| Instant::now() < t);
                if muted || inflight.is_some() || cooling {
                    continue; // don't hear our own voice
                }
                let ctx = LiveCtx {
                    settings,
                    provider: provider.clone(),
                    ffmpeg: ffmpeg.clone(),
                    camera: if camera_on { cam.clone() } else { None },
                    tts: tts.clone(),
                    cancel: cancel.clone(),
                };
                inflight = Some(Box::pin(live_turn(std::mem::take(&mut history), text, ctx)));
            }
            h = poll_inflight(&mut inflight) => {
                history = h;
                inflight = None;
                muted_until = Some(Instant::now() + SELF_HEARING_COOLDOWN);
            }
        }
    }

    // Cleanup: abort any reply in flight (kills TTS/ffmpeg children), stop the
    // listener, give the keyboard back.
    cancel.cancel();
    drop(inflight);
    keys.restore();
    if let Some(r) = reader {
        r.abort();
    }
    let _ = ws.start_kill();
    let _ = ws.wait().await;
    Ok(())
}

/* ── omni live (one model does everything) ── */

/// Omni live: a single multimodal model does everything. One ffmpeg mic reader
/// feeds our VAD; each detected utterance's audio (plus a fresh camera frame)
/// is sent straight to the omni model, whose reply is spoken. Listening is
/// paused while it thinks/speaks so it never hears its own voice.
async fn run_omni_live(settings: &MediaSettings) -> Result<()> {
    let ffmpeg = resolve_ffmpeg();
    let cancel = CancellationToken::new();
    let cam = if settings.vision.enabled {
        pick_video_device(&settings.vision.device, &ffmpeg, &cancel).await.ok().flatten()
    } else {
        None
    };
    let mic = pick_audio_device(&settings.audio.device, cam.as_deref(), &ffmpeg, &cancel).await.ok().flatten();
    let tts = tts_available(&settings.tts);

    say("\r\n  Qwenodyssey — live (omni: one model does everything)\r\n".bold());
    say(format!(
        "{}{} ({}) — audio + vision + reasoning",
        gray("  brain : "),
        settings.omni.model,
        settings.omni.provider.as_str()
    ));
    say(format!("{}{}", gray("  mic   : "), mic.clone().unwrap_or_else(|| "none".dark_red().to_string())));
    say(format!(
        "{}{}{}",
        gray("  camera: "),
        cam.clone().unwrap_or_else(|| "none".yellow().to_string()),
        if settings.omni.send_image { String::new() } else { gray(" (image off)") }
    ));
    say(format!("{}{}", gray("  voice : "), voice_line(&tts)));
    say(format!(
        "\r\n  {}{}{}{}{}{}{}\r\n",
        gray("Just talk. "),
        "c".dark_cyan(),
        gray(" camera · "),
        "m".dark_cyan(),
        gray(" mute · "),
        "q".dark_cyan(),
        gray(" quit")
    ));

    let Some(mic) = mic else {
        say("  No microphone found. Set audio.device or check Windows mic privacy.\r\n".dark_red());
        return Ok(());
    };

    let (mut vad, mut events) = match start_mic_vad(&mic, VadOptions { speak_db: -38.0, ..Default::default() }, &ffmpeg) {
        Ok(v) => v,
        Err(e) => {
            say(format!("  [error: {}]\r\n", err_msg(&e)).dark_red());
            return Ok(());
        }
    };

    let mut camera_on = settings.vision.enabled && cam.is_some();
    let mut muted = false;
    let mut last_heard: Option<Instant> = None;
    let mut inflight: Option<Inflight<'_, ()>> = None;
    let mut resume_at: Option<Instant> = None;
    let mut keys = Keys::start();

    loop {
        let busy = muted || inflight.is_some() || resume_at.is_some();
        tokio::select! {
            key = keys.next() => {
                if key.is_quit() {
                    break;
                }
                match key {
                    Key::Char('c') => {
                        if cam.is_none() {
                            say("  (no camera)".yellow());
                        } else {
                            camera_on = !camera_on;
                            say(gray(&format!("  camera {}", if camera_on { "ON" } else { "off" })));
                        }
                    }
                    Key::Char('m') => {
                        // Toggle listening.
                        muted = !muted;
                        if muted {
                            vad.pause();
                        } else if inflight.is_none() && resume_at.is_none() {
                            vad.resume();
                        }
                        say(gray(&format!("  mic {}", if muted { "MUTED" } else { "live" })));
                    }
                    _ => {}
                }
            }
            ev = events.recv() => match ev {
                Some(VadEvent::SpeechStart) => {
                    if busy {
                        continue;
                    }
                    let now = Instant::now();
                    if last_heard.map_or(true, |t| now.duration_since(t) > HEARING_NOTICE_GAP) {
                        say(gray("  🎤 hearing you…"));
                        last_heard = Some(now);
                    }
                }
                Some(VadEvent::Utterance(wav)) => {
                    if busy {
                        continue; // ignore speech captured while we're replying
                    }
                    vad.pause(); // don't hear our own TTS
                    let camera = if camera_on && settings.omni.send_image { cam.clone() } else { None };
                    let (ffmpeg, tts, cancel) = (ffmpeg.clone(), tts.clone(), cancel.clone());
                    inflight = Some(Box::pin(async move {
                        omni_turn(wav, camera, settings, &ffmpeg, &tts, &cancel).await;
                    }));
                }
                // Levels are available for a future inline meter; ffmpeg stderr is noise.
                Some(VadEvent::Level(_)) | Some(VadEvent::Error(_)) => {}
                Some(VadEvent::Ended) | None => {
                    say("  (microphone stream ended)".yellow());
                    break;
                }
            },
            _ = poll_inflight(&mut inflight) => {
                inflight = None;
                resume_at = Some(Instant::now() + OMNI_RESUME_DELAY);
            }
            _ = sleep_until_opt(resume_at) => {
                resume_at = None;
                if !muted {
                    vad.resume();
                }
            }
        }
    }

    cancel.cancel();
    drop(inflight);
    vad.stop();
    keys.restore();
    say(gray("\r\n  live session ended.\r\n"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use qo_core::{GenerateResult, HealthStatus, Provider, Role};
    use std::sync::{Arc, Mutex};

    #[test]
    fn whisper_args() {
        assert_eq!(
            whisper_stream_args(Path::new("C:/m/ggml-base.en.bin"), "en", 0.6),
            vec!["-m", "C:/m/ggml-base.en.bin", "-l", "en", "--step", "0", "--length", "30000", "-vth", "0.6"]
        );
        assert_eq!(whisper_stream_args(Path::new("m"), "de", 1.0)[9], "1");
    }

    #[test]
    fn speech_extraction() {
        assert_eq!(extract_speech("  [BLANK_AUDIO]  "), "");
        assert_eq!(extract_speech(" ... hello there"), "hello there");
        assert_eq!(extract_speech("\x1b[2K\x1b[0m what's   up [Music]"), "what's up");
        assert_eq!(extract_speech("- - a"), "");
        assert_eq!(extract_speech("ok"), "ok");
        assert_eq!(extract_speech("[inaudible] [noise] (sighs)"), "(sighs)");
    }

    #[test]
    fn transcript_blocks_and_stray_lines() {
        let mut p = TranscriptParser::default();
        assert_eq!(p.push_line("### Transcription 0 START | t0 = 0 ms | t1 = 2310 ms"), None);
        assert_eq!(p.push_line(""), None);
        assert_eq!(p.push_line(" Hello, can you"), None);
        assert_eq!(p.push_line("\x1b[1m see me? \x1b[0m"), None);
        assert_eq!(p.push_line("### Transcription 0 END"), Some("Hello, can you see me?".into()));
        // A block with only non-speech yields nothing.
        assert_eq!(p.push_line("### transcription 1 start"), None);
        assert_eq!(p.push_line("[BLANK_AUDIO]"), None);
        assert_eq!(p.push_line("### Transcription 1 END"), None);
        // Builds without markers: stray lines are utterances; other ### lines are not.
        assert_eq!(p.push_line("  what time is it  "), Some("what time is it".into()));
        assert_eq!(p.push_line("### init: found 1 capture device"), None);
        assert_eq!(p.push_line("."), None);
    }

    /// A provider that records the request and streams a canned reply.
    struct FakeProvider {
        reply: Result<String, String>,
        seen: Mutex<Vec<(Vec<Message>, Option<f64>, Option<u32>, bool)>>,
    }

    #[async_trait]
    impl Provider for FakeProvider {
        fn name(&self) -> &str {
            "fake"
        }
        fn model(&self) -> String {
            "fake-model".into()
        }
        fn set_model(&self, _model: &str) {}
        async fn generate(&self, _m: &[Message], _o: &ModelOptions) -> anyhow::Result<GenerateResult> {
            anyhow::bail!("unused")
        }
        async fn stream(
            &self,
            messages: &[Message],
            on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
            opts: &ModelOptions,
        ) -> anyhow::Result<GenerateResult> {
            self.seen.lock().unwrap().push((messages.to_vec(), opts.temperature, opts.max_tokens, opts.cancel.is_some()));
            match &self.reply {
                Ok(text) => {
                    for w in text.split_inclusive(' ') {
                        on_chunk(w);
                    }
                    Ok(GenerateResult { text: format!(" {text} \n"), completion_tokens: Some(7), ..Default::default() })
                }
                Err(e) => Err(anyhow::anyhow!("{e}")),
            }
        }
        async fn health_check(&self) -> HealthStatus {
            HealthStatus { ok: true, detail: None }
        }
    }

    fn ctx<'a>(settings: &'a MediaSettings, provider: SharedProvider) -> LiveCtx<'a> {
        LiveCtx {
            settings,
            provider,
            ffmpeg: PathBuf::from("ffmpeg"),
            camera: None, // never touch a real camera in tests
            tts: TtsStatus { ok: false, detail: "disabled".into() },
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn live_turn_streams_and_records_history() {
        let fake = Arc::new(FakeProvider { reply: Ok("Hi there!".into()), seen: Mutex::new(vec![]) });
        let settings = MediaSettings { model_max_tokens: 1234, ..Default::default() };
        let history = vec![Message::system(LIVE_SYSTEM)];
        let out = live_turn(history, "hello".into(), ctx(&settings, fake.clone())).await;
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], Message::user("hello"));
        assert_eq!(out[2], Message::assistant("Hi there!"));
        let seen = fake.seen.lock().unwrap();
        let (msgs, temp, max, has_cancel) = &seen[0];
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(*temp, Some(0.5));
        assert_eq!(*max, Some(1234));
        assert!(has_cancel);
    }

    #[tokio::test]
    async fn live_turn_error_keeps_user_message_only() {
        let fake = Arc::new(FakeProvider { reply: Err("nvidia HTTP 500: boom".into()), seen: Mutex::new(vec![]) });
        let settings = MediaSettings::default();
        let out = live_turn(vec![Message::system(LIVE_SYSTEM)], "hey".into(), ctx(&settings, fake)).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].role, Role::User);
    }

    #[tokio::test]
    async fn omni_turn_against_mock_server() {
        let (base, rx) = crate::http::mock::serve(200, r#"{"choices":[{"message":{"content":"Hello!"}}]}"#).await;
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("utt.wav");
        std::fs::write(&wav, crate::micstream::wav_bytes(&[0u8; 64])).unwrap();
        let settings = MediaSettings { nvidia_base_url: base, model_max_tokens: 777, ..Default::default() };
        let tts = TtsStatus { ok: false, detail: "disabled".into() };
        omni_turn(wav, None, &settings, Path::new("ffmpeg"), &tts, &CancellationToken::new()).await;
        let req = rx.await.unwrap();
        assert_eq!(req.body["model"], "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning");
        assert_eq!(req.body["max_tokens"], 777);
        assert_eq!(req.body["messages"][0]["content"][0]["text"], OMNI_INSTRUCTION);
        assert_eq!(req.body["messages"][0]["content"][1]["type"], "input_audio");
    }

    #[test]
    fn command_futures_are_send() {
        fn assert_send<T: Send>(_: &T) {}
        let settings = MediaSettings::default();
        let provider: SharedProvider = Arc::new(FakeProvider { reply: Ok(String::new()), seen: Mutex::new(vec![]) });
        // Constructed but never polled: nothing is spawned or opened.
        assert_send(&run_live(&settings, provider));
        assert_send(&crate::mictest::run_mictest(&settings));
        assert_send(&crate::miclevel::run_miclevel(&settings));
    }
}
