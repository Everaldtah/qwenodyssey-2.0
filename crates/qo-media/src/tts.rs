//! Text-to-speech for spoken replies. Default engine is Piper (local neural
//! TTS): pipe text → WAV → play with ffplay (`-nodisp`, no window). Falls back
//! to the built-in Windows SAPI voice, or no speech. Fully local either way.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use qo_core::CancellationToken;
use regex::Regex;

use crate::bins::{now_millis, piper_bin, piper_voice, resolve_ffplay};
use crate::proc;
use crate::settings::{TtsEngine, TtsSettings};

/// Every TTS child gets this long before it is killed (TS `timeout: 120000`).
const TTS_TIMEOUT: Duration = Duration::from_secs(120);

/// Is speech actually available for the configured engine?
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsStatus {
    pub ok: bool,
    /// "disabled", "Windows SAPI", "piper · `<voice>.onnx`", or what is missing.
    pub detail: String,
}

/// Check the configured engine (files only — nothing is spawned).
pub fn tts_available(t: &TtsSettings) -> TtsStatus {
    tts_available_with(t, &piper_bin(&t.piper_bin), &piper_voice(&t.piper_voice))
}

fn tts_available_with(t: &TtsSettings, bin: &Path, voice: &Path) -> TtsStatus {
    if !t.enabled || t.engine == TtsEngine::None {
        return TtsStatus { ok: false, detail: "disabled".into() };
    }
    if t.engine == TtsEngine::Sapi {
        return TtsStatus { ok: true, detail: "Windows SAPI".into() };
    }
    if !bin.exists() {
        return TtsStatus { ok: false, detail: format!("piper not found at {}", bin.display()) };
    }
    if !voice.exists() {
        return TtsStatus { ok: false, detail: format!("voice not found at {}", voice.display()) };
    }
    let name = voice.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    TtsStatus { ok: true, detail: format!("piper · {name}") }
}

static CODE_FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").expect("valid fence regex"));

/// What actually gets spoken: fenced code collapsed to "(code block)", all
/// whitespace runs collapsed, trimmed.
pub fn clean_speech_text(text: &str) -> String {
    let t = CODE_FENCE.replace_all(text, " (code block) ");
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `piper -m <voice> -f <wav>` (text on stdin).
pub fn piper_args(voice: &Path, wav: &Path) -> Vec<String> {
    vec!["-m".into(), voice.to_string_lossy().into_owned(), "-f".into(), wav.to_string_lossy().into_owned()]
}

/// `ffplay -nodisp -autoexit -loglevel quiet <wav>` — audio only, no window.
pub fn ffplay_args(wav: &Path) -> Vec<String> {
    ["-nodisp", "-autoexit", "-loglevel", "quiet"].iter().map(|s| s.to_string()).chain([wav.to_string_lossy().into_owned()]).collect()
}

/// Synchronous SAPI speak in a child PowerShell (text on stdin).
pub fn sapi_args() -> Vec<String> {
    vec![
        "-NoProfile".into(),
        "-Command".into(),
        "Add-Type -AssemblyName System.Speech; \
         $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
         $s.Speak([Console]::In.ReadToEnd())"
            .into(),
    ]
}

/// Where Piper writes the reply audio (`<tmp>/qwenodyssey-tts-<ms>.wav`).
fn tts_wav_path() -> PathBuf {
    std::env::temp_dir().join(format!("qwenodyssey-tts-{}.wav", now_millis()))
}

/// Speak `text` aloud, returning once playback finishes. Best-effort: never
/// fails (speech is non-essential — never break the conversation over it).
/// Cancelling `cancel` kills piper/ffplay/PowerShell at once.
pub async fn speak(text: &str, t: &TtsSettings, cancel: &CancellationToken) {
    let clean = clean_speech_text(text);
    if clean.is_empty() || !t.enabled || t.engine == TtsEngine::None {
        return;
    }

    if t.engine == TtsEngine::Sapi {
        let _ = proc::run(Path::new("powershell.exe"), &sapi_args(), Some(clean.as_bytes()), Some(TTS_TIMEOUT), cancel)
            .await;
        return;
    }

    // Piper → WAV → ffplay.
    let bin = piper_bin(&t.piper_bin);
    let voice = piper_voice(&t.piper_voice);
    if !bin.exists() || !voice.exists() {
        return;
    }
    let wav = tts_wav_path();
    let _ = proc::run(&bin, &piper_args(&voice, &wav), Some(clean.as_bytes()), Some(TTS_TIMEOUT), cancel).await;
    if wav.exists() {
        let _ = proc::run(&resolve_ffplay(), &ffplay_args(&wav), None, Some(TTS_TIMEOUT), cancel).await;
        // Temp cleanup is best-effort.
        let _ = std::fs::remove_file(&wav);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("piper.exe");
        let voice = dir.path().join("en_US-amy-medium.onnx");
        let mut t = TtsSettings::default();

        let s = tts_available_with(&t, &bin, &voice);
        assert!(!s.ok);
        assert_eq!(s.detail, format!("piper not found at {}", bin.display()));
        std::fs::write(&bin, b"").unwrap();
        assert_eq!(tts_available_with(&t, &bin, &voice).detail, format!("voice not found at {}", voice.display()));
        std::fs::write(&voice, b"").unwrap();
        assert_eq!(
            tts_available_with(&t, &bin, &voice),
            TtsStatus { ok: true, detail: "piper · en_US-amy-medium.onnx".into() }
        );

        t.engine = TtsEngine::Sapi;
        assert_eq!(tts_available_with(&t, &bin, &voice), TtsStatus { ok: true, detail: "Windows SAPI".into() });
        t.engine = TtsEngine::None;
        assert_eq!(tts_available_with(&t, &bin, &voice).detail, "disabled");
        t.engine = TtsEngine::Piper;
        t.enabled = false;
        assert!(!tts_available_with(&t, &bin, &voice).ok);
    }

    #[test]
    fn cleans_text_for_speech() {
        assert_eq!(clean_speech_text("Here:\n```rust\nfn x() {}\n```\nDone.  "), "Here: (code block) Done.");
        assert_eq!(clean_speech_text("a ```x``` b ```y``` c"), "a (code block) b (code block) c");
        assert_eq!(clean_speech_text("   \n "), "");
    }

    #[test]
    fn child_args() {
        assert_eq!(piper_args(Path::new("v.onnx"), Path::new("o.wav")), vec!["-m", "v.onnx", "-f", "o.wav"]);
        assert_eq!(ffplay_args(Path::new("o.wav")), vec!["-nodisp", "-autoexit", "-loglevel", "quiet", "o.wav"]);
        let s = sapi_args();
        assert_eq!(&s[..2], &["-NoProfile", "-Command"]);
        assert!(s[2].contains("System.Speech") && s[2].ends_with("$s.Speak([Console]::In.ReadToEnd())"));
    }

    #[tokio::test]
    async fn speak_is_a_noop_when_disabled_or_empty() {
        // None of these may spawn anything (no audio is ever played in tests).
        let cancel = CancellationToken::new();
        let mut t = TtsSettings { engine: TtsEngine::None, ..Default::default() };
        speak("hello", &t, &cancel).await;
        t.engine = TtsEngine::Sapi;
        speak("  \n ", &t, &cancel).await; // nothing left to say
        t.enabled = false;
        speak("hello", &t, &cancel).await;
        // Piper with missing files returns without spawning.
        let t = TtsSettings {
            piper_bin: "C:/definitely/missing/piper.exe".into(),
            piper_voice: "C:/definitely/missing/v.onnx".into(),
            ..Default::default()
        };
        speak("hello", &t, &cancel).await;
    }
}
