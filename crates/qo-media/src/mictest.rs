//! `qwenodyssey mictest` — record your voice and verify two things at once:
//!   1. local speech-to-text (whisper.cpp) transcribes it, and
//!   2. the omni model (e.g. phi-4-multimodal) accepts the raw audio and replies.
//!
//! Interactive push-to-talk so we can test together: press Enter (or space),
//! speak, Enter again to stop. Optionally attaches a camera frame to confirm
//! audio+vision in one call. `q` / Esc / Ctrl-C quits at any point (a running
//! transcription or omni request is aborted).

use std::future::Future;

use anyhow::Result;
use crossterm::style::Stylize;
use qo_core::CancellationToken;

use crate::bins::{file_label, resolve_ffmpeg, resolve_whisper_bin, whisper_model_path};
use crate::media::{capture_frame, err_msg, pick_audio_device, pick_video_device, start_recording, transcribe, WhisperOpts};
use crate::omni::{ask_omni, OmniInput};
use crate::settings::MediaSettings;
use crate::term::{gray, put, say, Key, Keys, CLEAR_LINE};

/// The instruction sent with the recorded audio.
pub fn mictest_prompt(with_image: bool) -> String {
    format!(
        "The user spoke to you (audio attached{}). First repeat what you heard, then respond.",
        if with_image { " along with their camera view" } else { "" }
    )
}

fn is_go(k: Key) -> bool {
    matches!(k, Key::Enter | Key::Char(' '))
}

/// Run `fut` unless the user quits first (then cancel and return `None`).
async fn unless_quit<T>(keys: &mut Keys, cancel: &CancellationToken, fut: impl Future<Output = T>) -> Option<T> {
    tokio::pin!(fut);
    loop {
        tokio::select! {
            r = &mut fut => return Some(r),
            k = keys.next() => {
                if k.is_quit() {
                    cancel.cancel();
                    return None;
                }
            }
        }
    }
}

/// Run `qwenodyssey mictest`. Owns the terminal until the user quits.
pub async fn run_mictest(settings: &MediaSettings) -> Result<()> {
    let ffmpeg = resolve_ffmpeg();
    let cancel = CancellationToken::new();
    let cam = if settings.vision.enabled {
        pick_video_device(&settings.vision.device, &ffmpeg, &cancel).await.ok().flatten()
    } else {
        None
    };
    let mic = pick_audio_device(&settings.audio.device, cam.as_deref(), &ffmpeg, &cancel).await.ok().flatten();
    let whisper = WhisperOpts {
        bin: resolve_whisper_bin(&settings.audio.whisper_bin),
        model: whisper_model_path(&settings.audio.whisper_model),
        language: settings.audio.language.clone(),
    };
    let omni_model = &settings.omni.model;

    say("\r\n  Qwenodyssey — microphone + omni test\r\n".bold());
    say(format!("{}{}", gray("  mic        : "), mic.clone().unwrap_or_else(|| "none detected".dark_red().to_string())));
    say(format!(
        "{}{} · {}",
        gray("  whisper    : "),
        whisper.bin.display(),
        file_label(&whisper.model.to_string_lossy())
    ));
    say(format!("{}{omni_model} ({})", gray("  omni model : "), settings.omni.provider.as_str()));
    say(format!("{}{}\r\n", gray("  camera     : "), cam.as_deref().unwrap_or("none")));

    let Some(mic) = mic else {
        say("  No microphone found via DirectShow. Set audio.device or check Windows mic privacy.\r\n".dark_red());
        return Ok(());
    };

    let mut keys = Keys::start();
    if !keys.interactive() {
        say("  mictest needs an interactive terminal (push-to-talk).\r\n".yellow());
        return Ok(());
    }

    'session: loop {
        put(format!("\r\n{}", "  Press Enter to record (type q to quit): ".dark_cyan()));
        loop {
            let k = keys.next().await;
            if k.is_quit() {
                say("");
                break 'session;
            }
            if is_go(k) {
                break;
            }
        }
        say("");

        say(format!("{}{}", "  ● recording…".dark_magenta(), gray("  speak now, then press Enter to stop")));
        let mut rec = match start_recording(&mic, settings.audio.max_seconds as f64, &ffmpeg) {
            Ok(r) => r,
            Err(e) => {
                say(format!("  recording failed: {}", err_msg(&e)).dark_red());
                continue;
            }
        };
        let mut quit = false;
        loop {
            tokio::select! {
                k = keys.next() => {
                    if k.is_quit() {
                        quit = true;
                        break;
                    }
                    if is_go(k) {
                        break;
                    }
                }
                // The max_seconds cap stopped ffmpeg on its own.
                _ = rec.wait_finished() => break,
            }
        }
        let wav = match rec.stop().await {
            Ok(w) => w,
            Err(e) => {
                say(format!("  recording failed: {}", err_msg(&e)).dark_red());
                if quit {
                    break;
                }
                continue;
            }
        };
        if quit {
            break;
        }

        // 1) Local whisper transcription.
        put(gray("  transcribing locally… "));
        let Some(heard) = unless_quit(&mut keys, &cancel, transcribe(&wav, &whisper, &cancel)).await else {
            put(CLEAR_LINE);
            break;
        };
        put(CLEAR_LINE);
        match heard {
            Ok(text) => say(format!(
                "{}{}",
                gray("  whisper heard : "),
                if text.is_empty() { "(nothing)".yellow().to_string() } else { text.white().to_string() }
            )),
            Err(e) => say(format!("  whisper failed: {}", err_msg(&e)).dark_red()),
        }

        // 2) Omni model — send the raw audio (and optionally a camera frame).
        let mut image_path = None;
        if let (Some(cam), true) = (&cam, settings.omni.send_image) {
            match unless_quit(&mut keys, &cancel, capture_frame(cam, &ffmpeg, &cancel)).await {
                None => break,
                Some(Ok(p)) => {
                    say(gray("  (attached a camera frame)"));
                    image_path = Some(p);
                }
                Some(Err(e)) => say(format!("  (no camera frame: {})", err_msg(&e)).yellow()),
            }
        }
        let input = OmniInput {
            model: omni_model.clone(),
            text: mictest_prompt(image_path.is_some()),
            audio_path: Some(wav),
            image_path,
            ..Default::default()
        };
        put(gray(&format!("  asking {omni_model}… ")));
        let Some(reply) = unless_quit(&mut keys, &cancel, ask_omni(&input, settings, &cancel)).await else {
            put(CLEAR_LINE);
            break;
        };
        put(CLEAR_LINE);
        match reply {
            Ok(r) => say(format!(
                "{}{}",
                "  omni reply    : ".dark_green(),
                if r.is_empty() { "(empty)".yellow().to_string() } else { r }
            )),
            Err(e) => {
                say(format!("  omni failed   : {}", err_msg(&e)).dark_red());
                say(gray("  → if this is an audio-format error, this model can't take audio input on NIM."));
            }
        }
    }

    keys.restore();
    say(gray("\r\n  mic test ended.\r\n"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_text_matches_ts() {
        assert_eq!(
            mictest_prompt(false),
            "The user spoke to you (audio attached). First repeat what you heard, then respond."
        );
        assert_eq!(
            mictest_prompt(true),
            "The user spoke to you (audio attached along with their camera view). First repeat what you heard, then respond."
        );
    }

    #[test]
    fn go_keys() {
        assert!(is_go(Key::Enter) && is_go(Key::Char(' ')));
        assert!(!is_go(Key::Char('q')) && !is_go(Key::Esc));
    }
}
