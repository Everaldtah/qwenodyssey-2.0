//! Webcam + microphone capture for live voice/vision, built on ffmpeg (DirectShow
//! on Windows) and a local whisper.cpp binary for speech-to-text. Everything here
//! is local: frames/audio are written to a temp dir; only the captured *frame* is
//! later sent to a vision model (see [`crate::vision`]), and transcription stays
//! on-device.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{anyhow, Result};
use qo_core::{CancellationToken, ProviderError};
use regex::Regex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin};
use tokio::task::JoinHandle;

use crate::bins::{capture_dir, now_millis};
use crate::proc::{self, check};

/// DirectShow devices as enumerated by ffmpeg.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DshowDevices {
    pub video: Vec<String>,
    pub audio: Vec<String>,
}

/// `ffmpeg -list_devices true -f dshow -i dummy`.
pub fn list_devices_args() -> Vec<String> {
    ["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"].map(String::from).to_vec()
}

static DEVICE_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)"([^"]+)"\s*\((video|audio)\)"#).expect("valid device regex"));

/// Parse ffmpeg's device listing: lines like
/// `[dshow @ 0x…] "NewEye 60s" (video)` / `… "Microphone (NewEye 60s)" (audio)`.
/// "Alternative name" lines carry no `(video|audio)` tag and are skipped.
pub fn parse_dshow_devices(output: &str) -> DshowDevices {
    let mut d = DshowDevices::default();
    for line in output.lines() {
        let Some(m) = DEVICE_LINE.captures(line) else { continue };
        let name = m[1].to_string();
        if m[2].eq_ignore_ascii_case("video") {
            d.video.push(name);
        } else {
            d.audio.push(name);
        }
    }
    d
}

/// Enumerate DirectShow video + audio devices (Windows). Listing does NOT
/// activate the camera/mic — ffmpeg just queries the device graph and exits
/// (with a non-zero code, which is expected and ignored).
pub async fn list_dshow_devices(ffmpeg: &Path, cancel: &CancellationToken) -> Result<DshowDevices> {
    let out = proc::run(ffmpeg, &list_devices_args(), None, Some(Duration::from_secs(20)), cancel).await?;
    Ok(parse_dshow_devices(&out.all()))
}

/// Pick the camera device: configured name, else the first enumerated video device.
pub async fn pick_video_device(configured: &str, ffmpeg: &Path, cancel: &CancellationToken) -> Result<Option<String>> {
    if !configured.is_empty() {
        return Ok(Some(configured.to_string()));
    }
    let d = list_dshow_devices(ffmpeg, cancel).await?;
    Ok(d.video.into_iter().next())
}

static VIRTUAL_MIC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)virtual|loopback|stereo mix|cable").expect("valid mic regex"));

/// The mic choice, given the enumerated audio devices: configured name, else
/// an audio device whose name references the camera (e.g. "Microphone (NewEye
/// 60s)" for camera "NewEye 60s"), else the first real mic that isn't an
/// obvious virtual/loopback device, else the first device at all.
pub fn choose_audio_device(configured: &str, camera: Option<&str>, audio: &[String]) -> Option<String> {
    if !configured.is_empty() {
        return Some(configured.to_string());
    }
    if let Some(cam) = camera {
        // JS: cameraName.split(/\s+/)[0] — "" (falsy) when it starts with whitespace.
        let token = cam.split(char::is_whitespace).next().unwrap_or("").to_lowercase();
        if !token.is_empty() {
            if let Some(m) = audio.iter().find(|a| a.to_lowercase().contains(&token)) {
                return Some(m.clone());
            }
        }
    }
    audio.iter().find(|a| !VIRTUAL_MIC.is_match(a)).or_else(|| audio.first()).cloned()
}

/// Pick the mic (see [`choose_audio_device`]); only lists devices when nothing
/// is configured.
pub async fn pick_audio_device(
    configured: &str,
    camera: Option<&str>,
    ffmpeg: &Path,
    cancel: &CancellationToken,
) -> Result<Option<String>> {
    if !configured.is_empty() {
        return Ok(Some(configured.to_string()));
    }
    let d = list_dshow_devices(ffmpeg, cancel).await?;
    Ok(choose_audio_device("", camera, &d.audio))
}

/// ffmpeg args that grab one still frame from `video=<device>` into `out` (JPEG q2).
pub fn capture_frame_args(device: &str, out: &Path) -> Vec<String> {
    let mut a: Vec<String> =
        ["-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-i"].map(String::from).to_vec();
    a.push(format!("video={device}"));
    a.extend(["-frames:v", "1", "-q:v", "2"].map(String::from));
    a.push(out.to_string_lossy().into_owned());
    a
}

/// Capture a single still frame from the camera to a JPEG. Returns the path.
/// 20 s timeout; a non-zero ffmpeg exit is an error.
pub async fn capture_frame(device: &str, ffmpeg: &Path, cancel: &CancellationToken) -> Result<PathBuf> {
    let out = capture_dir()?.join(format!("frame-{}.jpg", now_millis()));
    let res = proc::run(ffmpeg, &capture_frame_args(device, &out), None, Some(Duration::from_secs(20)), cancel).await?;
    check(res, ffmpeg)?;
    if !out.exists() {
        return Err(anyhow!("camera capture produced no file"));
    }
    Ok(out)
}

/// `-t` value: JS `String(maxSeconds)` (30 → "30", 2.5 → "2.5").
pub(crate) fn fmt_num(n: f64) -> String {
    format!("{n}")
}

/// ffmpeg args that record `audio=<device>` to a 16 kHz mono WAV (whisper's
/// preferred format), capped at `max_seconds`.
pub fn record_args(device: &str, max_seconds: f64, out: &Path) -> Vec<String> {
    let mut a: Vec<String> =
        ["-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-i"].map(String::from).to_vec();
    a.push(format!("audio={device}"));
    a.extend(["-ar", "16000", "-ac", "1", "-t"].map(String::from));
    a.push(fmt_num(max_seconds));
    a.push(out.to_string_lossy().into_owned());
    a
}

/// A running push-to-talk recording. Dropping it kills ffmpeg (the WAV may
/// then be incomplete) — call [`Recording::stop`] to finalize it.
pub struct Recording {
    child: Child,
    stdin: Option<ChildStdin>,
    out: PathBuf,
}

/// How long `stop()` waits for ffmpeg to flush the WAV after `q` before killing it.
const STOP_GRACE: Duration = Duration::from_secs(10);

impl Recording {
    /// The WAV being written.
    pub fn path(&self) -> &Path {
        &self.out
    }

    /// Has ffmpeg already exited (e.g. the `max_seconds` cap was reached)?
    pub fn finished(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(Some(_)))
    }

    /// Resolves when ffmpeg exits on its own (cap reached / device lost).
    pub async fn wait_finished(&mut self) {
        let _ = self.child.wait().await;
    }

    /// Stop recording cleanly: send `q` to ffmpeg so the WAV is finalized, wait
    /// for it to exit, and return the wav path.
    pub async fn stop(mut self) -> Result<PathBuf> {
        let mut graceful = false;
        if let Some(mut stdin) = self.stdin.take() {
            // ffmpeg: graceful stop, flush + close the WAV.
            graceful = stdin.write_all(b"q").await.is_ok() && stdin.flush().await.is_ok();
            drop(stdin);
        }
        if !graceful {
            let _ = self.child.start_kill();
        }
        if tokio::time::timeout(STOP_GRACE, self.child.wait()).await.is_err() {
            let _ = self.child.start_kill();
            let _ = self.child.wait().await;
        }
        if !self.out.exists() {
            return Err(anyhow!("microphone capture produced no file"));
        }
        Ok(self.out)
    }
}

/// Start recording the mic to a 16 kHz mono WAV. Returns a handle whose
/// [`Recording::stop`] finalizes the file; the `max_seconds` cap auto-stops it.
/// stdin is piped (that is how ffmpeg is told to stop gracefully).
pub fn start_recording(device: &str, max_seconds: f64, ffmpeg: &Path) -> Result<Recording> {
    let out = capture_dir()?.join(format!("audio-{}.wav", now_millis()));
    let mut cmd = proc::command(ffmpeg, &record_args(device, max_seconds, &out));
    cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| anyhow!("failed to start {}: {e}", ffmpeg.display()))?;
    let stdin = child.stdin.take();
    Ok(Recording { child, stdin, out })
}

/// ffmpeg args for the live loudness meter: ~0.1 s windows → ~10 updates/sec,
/// RMS level printed to stderr by `ametadata=print`.
pub fn level_meter_args(device: &str) -> Vec<String> {
    let mut a: Vec<String> = ["-hide_banner", "-f", "dshow", "-i"].map(String::from).to_vec();
    a.push(format!("audio={device}"));
    a.push("-af".into());
    a.push(
        "aresample=16000,asetnsamples=n=1600:p=0,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level"
            .into(),
    );
    a.extend(["-f", "null", "-"].map(String::from));
    a
}

static RMS_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)RMS_level=(-?inf|-?\d+(?:\.\d+)?)").expect("valid rms regex"));

/// Parse one ffmpeg stderr line into an RMS dB value (`-inf` → −90).
pub fn parse_rms_level(line: &str) -> Option<f64> {
    let m = RMS_LINE.captures(line)?;
    let v = &m[1];
    if v.to_ascii_lowercase().contains("inf") {
        return Some(-90.0);
    }
    v.parse::<f64>().ok().filter(|d| !d.is_nan())
}

/// Splits a byte stream into lines on `\n`, `\r\n` or a lone `\r` (ffmpeg's
/// progress lines end in `\r`), like Node's readline.
#[derive(Default)]
pub(crate) struct LineSplitter {
    buf: Vec<u8>,
}

impl LineSplitter {
    pub fn push(&mut self, chunk: &[u8], mut on_line: impl FnMut(&str)) {
        for &b in chunk {
            if b == b'\n' || b == b'\r' {
                if !self.buf.is_empty() {
                    on_line(&String::from_utf8_lossy(&self.buf));
                    self.buf.clear();
                }
            } else {
                self.buf.push(b);
            }
        }
    }

    pub fn finish(&mut self, mut on_line: impl FnMut(&str)) {
        if !self.buf.is_empty() {
            on_line(&String::from_utf8_lossy(&self.buf));
            self.buf.clear();
        }
    }
}

/// A running loudness meter; stop it (or drop it) to release the mic.
pub struct LevelMeter {
    child: Option<Child>,
    reader: Option<JoinHandle<()>>,
}

impl LevelMeter {
    /// Stop the meter: end the reader and kill ffmpeg.
    pub fn stop(&mut self) {
        if let Some(r) = self.reader.take() {
            r.abort();
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.start_kill();
        }
    }
}

impl Drop for LevelMeter {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Stream live microphone loudness (RMS dB) from ffmpeg's astats filter. Calls
/// `on_level` ~10×/sec with a value roughly in [-90, 0] dB (−90 ≈ silence).
/// Used to draw a VU meter so the user can see their mic is actually picking
/// up sound.
pub fn start_level_meter(
    device: &str,
    mut on_level: impl FnMut(f64) + Send + 'static,
    ffmpeg: &Path,
) -> Result<LevelMeter> {
    let mut cmd = proc::command(ffmpeg, &level_meter_args(device));
    cmd.stdout(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| anyhow!("failed to start {}: {e}", ffmpeg.display()))?;
    let stderr = child.stderr.take();
    let reader = stderr.map(|mut stderr| {
        tokio::spawn(async move {
            let mut lines = LineSplitter::default();
            let mut buf = [0u8; 4096];
            loop {
                match stderr.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => lines.push(&buf[..n], |l| {
                        if let Some(db) = parse_rms_level(l) {
                            on_level(db);
                        }
                    }),
                }
            }
            lines.finish(|l| {
                if let Some(db) = parse_rms_level(l) {
                    on_level(db);
                }
            });
        })
    });
    Ok(LevelMeter { child: Some(child), reader })
}

/// Where whisper.cpp should write its `.txt`: the wav path minus a `.wav`
/// suffix (any case) plus `-stt` (whisper appends `.txt` itself).
pub fn stt_base(wav: &Path) -> PathBuf {
    let s = wav.to_string_lossy();
    let stem = if s.len() >= 4 && s.is_char_boundary(s.len() - 4) && s[s.len() - 4..].eq_ignore_ascii_case(".wav") {
        &s[..s.len() - 4]
    } else {
        &s[..]
    };
    PathBuf::from(format!("{stem}-stt"))
}

/// whisper.cpp args: `-m model -f wav -l lang -nt -otxt -of base`.
pub fn transcribe_args(model: &Path, wav: &Path, language: &str, base: &Path) -> Vec<String> {
    vec![
        "-m".into(),
        model.to_string_lossy().into_owned(),
        "-f".into(),
        wav.to_string_lossy().into_owned(),
        "-l".into(),
        language.to_string(),
        "-nt".into(),
        "-otxt".into(),
        "-of".into(),
        base.to_string_lossy().into_owned(),
    ]
}

/// Collapse whitespace runs to single spaces and trim.
pub fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// whisper.cpp invocation settings.
#[derive(Clone, Debug)]
pub struct WhisperOpts {
    pub bin: PathBuf,
    pub model: PathBuf,
    pub language: String,
}

/// Transcribe a WAV with whisper.cpp. Writes a `.txt` next to a temp base and
/// returns the whitespace-collapsed text. Errors if the model is missing or
/// whisper fails (120 s timeout).
pub async fn transcribe(wav: &Path, opts: &WhisperOpts, cancel: &CancellationToken) -> Result<String> {
    if !opts.model.exists() {
        return Err(anyhow!(
            "whisper model not found at {}. Download one, e.g. ggml-base.en.bin, or set audio.whisper_model.",
            opts.model.display()
        ));
    }
    let base = stt_base(wav);
    let args = transcribe_args(&opts.model, wav, &opts.language, &base);
    let out = proc::run(&opts.bin, &args, None, Some(Duration::from_secs(120)), cancel).await?;
    check(out, &opts.bin)?;
    let txt = PathBuf::from(format!("{}.txt", base.to_string_lossy()));
    let text = tokio::fs::read_to_string(&txt).await.unwrap_or_default();
    Ok(collapse_ws(&text))
}

/// Map an error to the message the commands print (`(e as Error).message`).
pub(crate) fn err_msg(e: &anyhow::Error) -> String {
    if qo_core::error::is_interrupt(e) {
        return "interrupted".into();
    }
    match e.downcast_ref::<ProviderError>() {
        Some(p) => p.to_string(),
        None => format!("{e:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LISTING: &str = r#"[dshow @ 000001] "NewEye 60s" (video)
[dshow @ 000001]   Alternative name "@device_pnp_\\?\usb#vid_1234"
[dshow @ 000001] "OBS Virtual Camera" (video)
[dshow @ 000001] "CABLE Output (VB-Audio Virtual Cable)" (audio)
[dshow @ 000001]   Alternative name "@device_cm_{33D9A762}\wave_{X}"
[dshow @ 000001] "Microphone (NewEye 60s)" (audio)
[dshow @ 000001] "Headset Mic (Realtek)" (AUDIO)
dummy: Immediate exit requested"#;

    #[test]
    fn parses_device_listing() {
        let d = parse_dshow_devices(LISTING);
        assert_eq!(d.video, vec!["NewEye 60s", "OBS Virtual Camera"]);
        assert_eq!(d.audio, vec!["CABLE Output (VB-Audio Virtual Cable)", "Microphone (NewEye 60s)", "Headset Mic (Realtek)"]);
        assert_eq!(parse_dshow_devices("nothing here"), DshowDevices::default());
    }

    #[test]
    fn audio_device_choice() {
        let audio: Vec<String> = parse_dshow_devices(LISTING).audio;
        // Configured wins outright.
        assert_eq!(choose_audio_device("My Mic", Some("NewEye 60s"), &audio).as_deref(), Some("My Mic"));
        // Camera-name token match ("neweye").
        assert_eq!(choose_audio_device("", Some("NewEye 60s"), &audio).as_deref(), Some("Microphone (NewEye 60s)"));
        // No camera: first non-virtual device (skips the CABLE one).
        assert_eq!(choose_audio_device("", None, &audio).as_deref(), Some("Microphone (NewEye 60s)"));
        // Camera token with no match falls through to the first real mic.
        assert_eq!(choose_audio_device("", Some("Logitech C920"), &audio).as_deref(), Some("Microphone (NewEye 60s)"));
        // Leading whitespace → empty token → no camera match.
        assert_eq!(
            choose_audio_device("", Some(" Headset"), &["CABLE x".to_string(), "Stereo Mix".into()]).as_deref(),
            Some("CABLE x")
        );
        // Only virtual devices: the first one.
        assert_eq!(choose_audio_device("", None, &["Stereo Mix (Realtek)".to_string()]).as_deref(), Some("Stereo Mix (Realtek)"));
        assert_eq!(choose_audio_device("", None, &[]), None);
    }

    #[test]
    fn capture_and_record_args() {
        let out = Path::new("C:/tmp/frame-1.jpg");
        assert_eq!(
            capture_frame_args("NewEye 60s", out),
            vec!["-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-i", "video=NewEye 60s", "-frames:v", "1", "-q:v", "2", "C:/tmp/frame-1.jpg"]
        );
        let wav = Path::new("C:/tmp/audio-1.wav");
        assert_eq!(
            record_args("Microphone (NewEye 60s)", 30.0, wav),
            vec!["-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-i", "audio=Microphone (NewEye 60s)", "-ar", "16000", "-ac", "1", "-t", "30", "C:/tmp/audio-1.wav"]
        );
        assert_eq!(record_args("m", 2.5, wav)[13], "2.5");
        assert_eq!(list_devices_args(), vec!["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]);
    }

    #[test]
    fn level_meter_args_and_parsing() {
        let a = level_meter_args("Mic");
        assert_eq!(&a[..5], &["-hide_banner", "-f", "dshow", "-i", "audio=Mic"]);
        assert_eq!(a[5], "-af");
        assert!(a[6].contains("astats=metadata=1:reset=1") && a[6].ends_with("key=lavfi.astats.Overall.RMS_level"));
        assert_eq!(&a[7..], &["-f", "null", "-"]);

        assert_eq!(parse_rms_level("lavfi.astats.Overall.RMS_level=-42.123456"), Some(-42.123456));
        assert_eq!(parse_rms_level("[Parsed_ametadata_3 @ 0x1] lavfi.astats.Overall.RMS_level=-inf"), Some(-90.0));
        assert_eq!(parse_rms_level("rms_level=-7"), Some(-7.0));
        assert_eq!(parse_rms_level("frame:3 pts:4800 pts_time:0.3"), None);
    }

    #[test]
    fn line_splitter_handles_cr_and_partial_chunks() {
        let mut s = LineSplitter::default();
        let mut got = Vec::new();
        s.push(b"size=N/A time=00:00:01\rlavfi.astats.Overall.RMS", |l| got.push(l.to_string()));
        s.push(b"_level=-30.5\r\nnext", |l| got.push(l.to_string()));
        s.finish(|l| got.push(l.to_string()));
        assert_eq!(got, vec!["size=N/A time=00:00:01", "lavfi.astats.Overall.RMS_level=-30.5", "next"]);
        let levels: Vec<f64> = got.iter().filter_map(|l| parse_rms_level(l)).collect();
        assert_eq!(levels, vec![-30.5]);
    }

    #[test]
    fn whisper_args_and_paths() {
        assert_eq!(stt_base(Path::new("C:/t/audio-5.wav")), PathBuf::from("C:/t/audio-5-stt"));
        assert_eq!(stt_base(Path::new("C:/t/AUDIO.WAV")), PathBuf::from("C:/t/AUDIO-stt"));
        assert_eq!(stt_base(Path::new("C:/t/clip.mp3")), PathBuf::from("C:/t/clip.mp3-stt"));
        let a = transcribe_args(Path::new("m.bin"), Path::new("a.wav"), "en", Path::new("a-stt"));
        assert_eq!(a, vec!["-m", "m.bin", "-f", "a.wav", "-l", "en", "-nt", "-otxt", "-of", "a-stt"]);
        assert_eq!(collapse_ws("  hello \n  world\t "), "hello world");
    }

    #[tokio::test]
    async fn transcribe_reports_missing_model_without_running_anything() {
        let dir = tempfile::tempdir().unwrap();
        let opts = WhisperOpts {
            bin: PathBuf::from("whisper-cli"),
            model: dir.path().join("nope.bin"),
            language: "en".into(),
        };
        let e = transcribe(Path::new("x.wav"), &opts, &CancellationToken::new()).await.unwrap_err();
        assert!(e.to_string().starts_with("whisper model not found at "));
        assert!(e.to_string().ends_with("or set audio.whisper_model."));
    }

    #[test]
    fn num_formatting_matches_js_string() {
        assert_eq!(fmt_num(30.0), "30");
        assert_eq!(fmt_num(0.6), "0.6");
        assert_eq!(fmt_num(2.5), "2.5");
    }
}
