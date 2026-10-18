//! Continuous microphone VAD: one ffmpeg process streams raw 16 kHz mono PCM,
//! and we detect speech segments (voice-activity detection) frame-by-frame.
//! Each completed utterance is written to a WAV and handed back — no
//! push-to-talk, and a single mic reader (so it never fights a second capture).
//! Also reports a live loudness level for a VU/speaking indicator.
//!
//! The detector itself ([`VadState`]) is pure: it eats PCM bytes and emits
//! [`VadSignal`]s, so it is tested on synthetic buffers without any device.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{anyhow, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Child;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::bins::{capture_dir, now_millis};
use crate::media::fmt_num;
use crate::proc::{self, head_chars};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_MS: u32 = 20;
/// 16-bit mono → 640 bytes per 20 ms frame.
pub const FRAME_BYTES: usize = (SAMPLE_RATE as usize * FRAME_MS as usize) / 1000 * 2;

/// Detector tuning (all optional in the TS; these are its defaults).
#[derive(Clone, Debug, PartialEq)]
pub struct VadOptions {
    /// dB above which a frame counts as speech.
    pub speak_db: f64,
    /// Trailing silence that ends an utterance.
    pub hangover_ms: u32,
    /// Ignore blips shorter than this.
    pub min_utterance_ms: u32,
    /// Force-flush a very long utterance.
    pub max_utterance_ms: u32,
    /// Audio kept before speech onset (so the first syllable isn't clipped).
    pub preroll_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self { speak_db: -38.0, hangover_ms: 700, min_utterance_ms: 300, max_utterance_ms: 30_000, preroll_ms: 200 }
    }
}

/// RMS loudness of one s16le frame in dBFS (−90 for digital silence).
pub fn frame_db(frame: &[u8]) -> f64 {
    let mut sum = 0f64;
    // JS: buf.length / 2 (fractional for an odd length), floored at 1.
    let n = (frame.len() as f64 / 2.0).max(1.0);
    for pair in frame.chunks_exact(2) {
        let s = f64::from(i16::from_le_bytes([pair[0], pair[1]]));
        sum += s * s;
    }
    let rms = (sum / n).sqrt();
    if rms > 0.0 {
        20.0 * (rms / 32768.0).log10()
    } else {
        -90.0
    }
}

/// A canonical 44-byte RIFF/WAVE header + the PCM (16 kHz, mono, 16-bit).
pub fn wav_bytes(pcm: &[u8]) -> Vec<u8> {
    let len = u32::try_from(pcm.len()).unwrap_or(u32::MAX - 36);
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(len)).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Write an utterance to `<tmp>/qwenodyssey-media/utt-<ms>.wav`.
pub fn write_wav(pcm: &[u8]) -> Result<PathBuf> {
    let out = capture_dir()?.join(format!("utt-{}.wav", now_millis()));
    write_wav_to(&out, pcm)?;
    Ok(out)
}

pub(crate) fn write_wav_to(out: &Path, pcm: &[u8]) -> Result<()> {
    std::fs::write(out, wav_bytes(pcm)).map_err(|e| anyhow!("cannot write {}: {e}", out.display()))
}

/// What the detector reports while consuming audio.
#[derive(Clone, Debug, PartialEq)]
pub enum VadSignal {
    /// Loudness of a frame (~50×/sec), reported even while paused.
    Level(f64),
    /// Speech onset.
    SpeechStart,
    /// A finished speech segment's raw PCM (pre-roll included).
    Utterance(Vec<u8>),
}

/// The frame-by-frame speech segmenter (port of the closure state in
/// `startMicVad`). Feed it arbitrary-sized chunks with [`VadState::push`].
#[derive(Debug)]
pub struct VadState {
    opts: VadOptions,
    pre_frames: usize,
    leftover: Vec<u8>,
    speaking: bool,
    paused: bool,
    silence_ms: u32,
    utt_ms: u32,
    preroll: VecDeque<Vec<u8>>,
    utt: Vec<u8>,
}

impl VadState {
    pub fn new(opts: VadOptions) -> Self {
        let pre_frames = (f64::from(opts.preroll_ms) / f64::from(FRAME_MS)).round().max(0.0) as usize;
        Self {
            opts,
            pre_frames,
            leftover: Vec::new(),
            speaking: false,
            paused: false,
            silence_ms: 0,
            utt_ms: 0,
            preroll: VecDeque::new(),
            utt: Vec::new(),
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Drop audio (e.g. while the assistant is speaking) and forget any
    /// half-captured utterance.
    pub fn pause(&mut self) {
        self.paused = true;
        self.speaking = false;
        self.utt.clear();
        self.utt_ms = 0;
        self.silence_ms = 0;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Consume a chunk of s16le PCM; complete 20 ms frames are processed and a
    /// partial tail is kept for the next chunk.
    pub fn push(&mut self, chunk: &[u8], mut sink: impl FnMut(VadSignal)) {
        self.leftover.extend_from_slice(chunk);
        let mut off = 0;
        while self.leftover.len() - off >= FRAME_BYTES {
            let frame = self.leftover[off..off + FRAME_BYTES].to_vec();
            self.process_frame(frame, &mut sink);
            off += FRAME_BYTES;
        }
        if off > 0 {
            self.leftover.drain(..off);
        }
    }

    fn end_utterance(&mut self, sink: &mut impl FnMut(VadSignal)) {
        let had = self.speaking;
        self.speaking = false;
        let frames = std::mem::take(&mut self.utt);
        self.silence_ms = 0;
        let ms = self.utt_ms;
        self.utt_ms = 0;
        if had && ms >= self.opts.min_utterance_ms {
            sink(VadSignal::Utterance(frames));
        }
    }

    fn process_frame(&mut self, frame: Vec<u8>, sink: &mut impl FnMut(VadSignal)) {
        let db = frame_db(&frame);
        sink(VadSignal::Level(db));
        if self.paused {
            return;
        }

        // Maintain a short pre-roll ring buffer so we don't clip the first syllable.
        self.preroll.push_back(frame.clone());
        while self.preroll.len() > self.pre_frames {
            self.preroll.pop_front();
        }

        if db > self.opts.speak_db {
            if !self.speaking {
                self.speaking = true;
                self.utt_ms = 0;
                // Include audio just before onset (`preroll.slice(0, -1)`: the
                // newest pre-roll entry is the current frame, pushed below).
                self.utt.clear();
                let keep = self.preroll.len().saturating_sub(1);
                for f in self.preroll.iter().take(keep) {
                    self.utt.extend_from_slice(f);
                }
                sink(VadSignal::SpeechStart);
            }
            self.utt.extend_from_slice(&frame);
            self.utt_ms += FRAME_MS;
            self.silence_ms = 0;
            if self.utt_ms >= self.opts.max_utterance_ms {
                self.end_utterance(sink);
            }
        } else if self.speaking {
            self.utt.extend_from_slice(&frame);
            self.utt_ms += FRAME_MS;
            self.silence_ms += FRAME_MS;
            if self.silence_ms >= self.opts.hangover_ms {
                self.end_utterance(sink);
            }
        }
    }
}

/// ffmpeg args that stream `audio=<device>` as raw 16 kHz mono s16le to stdout.
pub fn mic_vad_args(device: &str) -> Vec<String> {
    let mut a: Vec<String> = ["-hide_banner", "-loglevel", "error", "-f", "dshow", "-i"].map(String::from).to_vec();
    a.push(format!("audio={device}"));
    a.push("-ar".into());
    a.push(fmt_num(f64::from(SAMPLE_RATE)));
    a.extend(["-ac", "1", "-f", "s16le", "-"].map(String::from));
    a
}

/// Events from a running [`MicVad`].
#[derive(Clone, Debug, PartialEq)]
pub enum VadEvent {
    /// ~50×/sec loudness for a meter.
    Level(f64),
    SpeechStart,
    /// A finished speech segment, already written to a WAV.
    Utterance(PathBuf),
    /// ffmpeg stderr noise (first 200 chars of a chunk) or a WAV write failure.
    Error(String),
    /// The PCM stream ended (ffmpeg exited — device lost, privacy block, …).
    Ended,
}

/// A running mic VAD. Dropping it (or [`MicVad::stop`]) kills ffmpeg and
/// releases the microphone.
pub struct MicVad {
    child: Option<Child>,
    state: Arc<Mutex<VadState>>,
    tasks: Vec<JoinHandle<()>>,
}

fn lock(state: &Mutex<VadState>) -> std::sync::MutexGuard<'_, VadState> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MicVad {
    /// Drop audio (e.g. while the assistant is speaking).
    pub fn pause(&self) {
        lock(&self.state).pause();
    }

    pub fn resume(&self) {
        lock(&self.state).resume();
    }

    pub fn is_paused(&self) -> bool {
        lock(&self.state).is_paused()
    }

    /// Kill ffmpeg and stop the reader tasks.
    pub fn stop(&mut self) {
        for t in self.tasks.drain(..) {
            t.abort();
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.start_kill();
        }
    }
}

impl Drop for MicVad {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start continuous listening on `device`. Events arrive on the returned
/// channel until the stream ends ([`VadEvent::Ended`]) or the handle is stopped.
pub fn start_mic_vad(
    device: &str,
    options: VadOptions,
    ffmpeg: &Path,
) -> Result<(MicVad, mpsc::UnboundedReceiver<VadEvent>)> {
    let mut cmd = proc::command(ffmpeg, &mic_vad_args(device));
    let mut child = cmd.spawn().map_err(|e| anyhow!("failed to start {}: {e}", ffmpeg.display()))?;
    let (tx, rx) = mpsc::unbounded_channel();
    let state = Arc::new(Mutex::new(VadState::new(options)));
    let mut tasks = Vec::new();

    if let Some(mut stderr) = child.stderr.take() {
        let tx = tx.clone();
        tasks.push(tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            loop {
                match stderr.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let msg = head_chars(&String::from_utf8_lossy(&buf[..n]), 200);
                        if tx.send(VadEvent::Error(msg)).is_err() {
                            break;
                        }
                    }
                }
            }
        }));
    }

    if let Some(mut stdout) = child.stdout.take() {
        let state = Arc::clone(&state);
        tasks.push(tokio::spawn(async move {
            let mut buf = vec![0u8; 16 * 1024];
            loop {
                let n = match stdout.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                let mut out = Vec::new();
                lock(&state).push(&buf[..n], |s| out.push(s));
                for s in out {
                    let ev = match s {
                        VadSignal::Level(db) => VadEvent::Level(db),
                        VadSignal::SpeechStart => VadEvent::SpeechStart,
                        VadSignal::Utterance(pcm) => match write_wav(&pcm) {
                            Ok(p) => VadEvent::Utterance(p),
                            Err(e) => VadEvent::Error(format!("{e:#}")),
                        },
                    };
                    if tx.send(ev).is_err() {
                        return;
                    }
                }
            }
            let _ = tx.send(VadEvent::Ended);
        }));
    }

    Ok((MicVad { child: Some(child), state, tasks }, rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 20 ms frame of a constant-amplitude square-ish signal.
    fn frame(amplitude: i16) -> Vec<u8> {
        let mut v = Vec::with_capacity(FRAME_BYTES);
        for i in 0..FRAME_BYTES / 2 {
            let s = if i % 2 == 0 { amplitude } else { amplitude.saturating_neg() };
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    fn frames(amplitude: i16, n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| frame(amplitude)).collect()
    }

    const LOUD: i16 = 8000; // ≈ −12 dBFS
    const QUIET: i16 = 100; // ≈ −50 dBFS

    fn run(state: &mut VadState, pcm: &[u8]) -> Vec<VadSignal> {
        let mut got = Vec::new();
        state.push(pcm, |s| got.push(s));
        got
    }

    fn non_level(sigs: &[VadSignal]) -> Vec<&VadSignal> {
        sigs.iter().filter(|s| !matches!(s, VadSignal::Level(_))).collect()
    }

    #[test]
    fn constants_match_ts() {
        assert_eq!(FRAME_BYTES, 640);
        assert_eq!(VadOptions::default().speak_db, -38.0);
        assert_eq!(VadOptions::default().hangover_ms, 700);
    }

    #[test]
    fn frame_db_levels() {
        assert_eq!(frame_db(&vec![0u8; FRAME_BYTES]), -90.0);
        assert!((frame_db(&frame(i16::MAX)) - 0.0).abs() < 0.01);
        let loud = frame_db(&frame(LOUD));
        assert!((loud - (-12.25)).abs() < 0.1, "{loud}");
        assert!(frame_db(&frame(QUIET)) < -38.0);
        assert_eq!(frame_db(&[]), -90.0);
    }

    #[test]
    fn wav_header_layout() {
        let pcm = vec![1u8, 2, 3, 4];
        let w = wav_bytes(&pcm);
        assert_eq!(w.len(), 48);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([w[4], w[5], w[6], w[7]]), 40);
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([w[20], w[21]]), 1);
        assert_eq!(u16::from_le_bytes([w[22], w[23]]), 1);
        assert_eq!(u32::from_le_bytes([w[24], w[25], w[26], w[27]]), 16_000);
        assert_eq!(u32::from_le_bytes([w[28], w[29], w[30], w[31]]), 32_000);
        assert_eq!(u16::from_le_bytes([w[34], w[35]]), 16);
        assert_eq!(&w[36..40], b"data");
        assert_eq!(u32::from_le_bytes([w[40], w[41], w[42], w[43]]), 4);
        assert_eq!(&w[44..], &pcm[..]);

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("u.wav");
        write_wav_to(&p, &pcm).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), w);
    }

    #[test]
    fn segments_one_utterance_with_preroll_and_hangover() {
        let mut v = VadState::new(VadOptions::default());
        // 15 quiet frames (fills the 10-frame pre-roll), 25 loud (500 ms), then 40 quiet.
        let mut pcm = frames(QUIET, 15);
        pcm.extend(frames(LOUD, 25));
        pcm.extend(frames(QUIET, 40));
        let sigs = run(&mut v, &pcm);
        let levels = sigs.iter().filter(|s| matches!(s, VadSignal::Level(_))).count();
        assert_eq!(levels, 80, "one level per frame");
        let events = non_level(&sigs);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], &VadSignal::SpeechStart);
        let VadSignal::Utterance(pcm) = events[1] else { panic!("expected utterance") };
        // 9 pre-roll frames + 25 speech + 35 hangover frames (700 ms of silence ends it).
        assert_eq!(pcm.len(), (9 + 25 + 35) * FRAME_BYTES);
        assert_eq!(&pcm[..FRAME_BYTES], &frame(QUIET)[..]);
        assert_eq!(&pcm[9 * FRAME_BYTES..10 * FRAME_BYTES], &frame(LOUD)[..]);
        assert!(!v.is_speaking());
    }

    #[test]
    fn ignores_blips_shorter_than_min_utterance() {
        let mut v = VadState::new(VadOptions { hangover_ms: 100, ..Default::default() });
        // 3 loud frames (60 ms) + 5 quiet (100 ms hangover) = 160 ms < 300 ms.
        let mut pcm = frames(LOUD, 3);
        pcm.extend(frames(QUIET, 10));
        let sigs = run(&mut v, &pcm);
        assert_eq!(non_level(&sigs), vec![&VadSignal::SpeechStart]);
    }

    #[test]
    fn force_flushes_long_utterances() {
        let mut v = VadState::new(VadOptions {
            max_utterance_ms: 200,
            min_utterance_ms: 100,
            preroll_ms: 0,
            ..Default::default()
        });
        let sigs = run(&mut v, &frames(LOUD, 25));
        let events = non_level(&sigs);
        // 10 frames → flush, restart → 10 frames → flush, 5 frames still open.
        assert_eq!(events.len(), 5);
        assert!(matches!(events[1], VadSignal::Utterance(p) if p.len() == 10 * FRAME_BYTES));
        assert_eq!(events[2], &VadSignal::SpeechStart);
        assert!(v.is_speaking());
    }

    #[test]
    fn handles_partial_frames_across_chunks() {
        let mut v = VadState::new(VadOptions::default());
        let pcm = frames(QUIET, 3);
        let mut levels = 0;
        for chunk in pcm.chunks(333) {
            levels += run(&mut v, chunk).len();
        }
        assert_eq!(levels, 3);
        // A 100-byte tail stays buffered.
        assert!(run(&mut v, &[0u8; 100]).is_empty());
        assert_eq!(run(&mut v, &[0u8; FRAME_BYTES - 100]).len(), 1);
    }

    #[test]
    fn pause_drops_audio_but_keeps_levels() {
        let mut v = VadState::new(VadOptions::default());
        let sigs = run(&mut v, &frames(LOUD, 5));
        assert_eq!(non_level(&sigs), vec![&VadSignal::SpeechStart]);
        v.pause();
        assert!(v.is_paused() && !v.is_speaking());
        let sigs = run(&mut v, &frames(LOUD, 50));
        assert_eq!(sigs.len(), 50);
        assert!(non_level(&sigs).is_empty());
        v.resume();
        let mut pcm = frames(LOUD, 20);
        pcm.extend(frames(QUIET, 35));
        let sigs = run(&mut v, &pcm);
        let events = non_level(&sigs);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1], VadSignal::Utterance(_)));
    }

    #[test]
    fn mic_args() {
        assert_eq!(
            mic_vad_args("Microphone (NewEye 60s)"),
            vec![
                "-hide_banner", "-loglevel", "error", "-f", "dshow", "-i", "audio=Microphone (NewEye 60s)",
                "-ar", "16000", "-ac", "1", "-f", "s16le", "-"
            ]
        );
    }
}
