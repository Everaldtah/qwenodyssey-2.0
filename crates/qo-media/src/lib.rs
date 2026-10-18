//! Qwenodyssey 2.0 voice + vision.
//!
//! - [`media`]: webcam frame + push-to-talk mic capture (ffmpeg / DirectShow),
//!   device listing, live loudness meter, whisper.cpp transcription.
//! - [`micstream`]: continuous voice-activity detection over one ffmpeg PCM stream.
//! - [`tts`]: Piper (→ ffplay, no window) or Windows SAPI spoken replies.
//! - [`vision`]: describe an image with a VLM (NIM / OpenAI-compatible / Ollama).
//! - [`omni`]: one multimodal call with text + audio + image.
//! - [`run_live`], [`run_mictest`], [`run_miclevel`]: the `live`, `mictest` and
//!   `miclevel` commands. They own the terminal while running (raw-mode keys,
//!   restored on exit and on panic) and print directly — the documented
//!   exception to "library crates never print".
//! - [`settings`]: the `[vision]` / `[audio]` / `[omni]` / `[tts]` config slice
//!   ([`MediaSettings`]); the binary builds it from the full config.
//!
//! Every external tool (ffmpeg, whisper, piper, ffplay, PowerShell) runs with
//! stdin null unless it needs piped input, is killed when its future is
//! dropped, and honours a cancellation token.

pub mod bins;
mod http;
pub mod live;
pub mod media;
pub mod miclevel;
pub mod micstream;
pub mod mictest;
pub mod omni;
mod proc;
pub mod settings;
mod term;
pub mod tts;
pub mod vision;

pub use http::api_base;
pub use live::run_live;
pub use media::{
    capture_frame, list_dshow_devices, pick_audio_device, pick_video_device, start_level_meter, start_recording,
    transcribe, DshowDevices, LevelMeter, Recording, WhisperOpts,
};
pub use miclevel::run_miclevel;
pub use micstream::{start_mic_vad, MicVad, VadEvent, VadOptions};
pub use mictest::run_mictest;
pub use omni::{ask_omni, OmniInput};
pub use settings::{
    AudioSettings, Endpoint, MediaSettings, OmniProvider, OmniSettings, TtsEngine, TtsSettings, VisionProvider,
    VisionSettings,
};
pub use tts::{speak, tts_available, TtsStatus};
pub use vision::describe_image;
