//! Locating the external tools and the files the media layer reads/writes.
//!
//! Same search order as the TS: scoop shims under the user's home first (that
//! is how ffmpeg / whisper.cpp are installed on this machine), else the bare
//! name on PATH. Each resolver has an `_in(home)` variant so tests can point it
//! at a temp dir.

use std::path::{Path, PathBuf};

/// The user's home directory ("" if it can't be determined, as `os.homedir()`
/// never fails in Node).
pub fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default()
}

fn shims(home: &Path) -> PathBuf {
    home.join("scoop").join("shims")
}

/// Locate the ffmpeg binary: the scoop shim, else `ffmpeg` on PATH.
pub fn resolve_ffmpeg() -> PathBuf {
    resolve_ffmpeg_in(&home_dir())
}

pub fn resolve_ffmpeg_in(home: &Path) -> PathBuf {
    for c in ["ffmpeg.exe", "ffmpeg"] {
        let p = shims(home).join(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("ffmpeg") // assume on PATH
}

/// Locate the whisper.cpp CLI. Honors an explicit path (or a name trusted to
/// be on PATH), else searches the scoop shims for the known binary names
/// (newer builds renamed `main` → `whisper-cli`).
pub fn resolve_whisper_bin(explicit: &str) -> PathBuf {
    resolve_whisper_bin_in(explicit, &home_dir())
}

pub fn resolve_whisper_bin_in(explicit: &str, home: &Path) -> PathBuf {
    if !explicit.is_empty() {
        // An existing path, or an explicit name trusted to be on PATH.
        return PathBuf::from(explicit);
    }
    for n in ["whisper-cli", "whisper-cpp", "whisper", "main"] {
        for ext in [".exe", ".cmd", ""] {
            let p = shims(home).join(format!("{n}{ext}"));
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from("whisper-cli") // last resort: assume on PATH
}

/// Locate the whisper.cpp streaming binary (continuous mic transcription).
pub fn resolve_whisper_stream() -> PathBuf {
    resolve_whisper_stream_in(&home_dir())
}

pub fn resolve_whisper_stream_in(home: &Path) -> PathBuf {
    for ext in [".exe", ".cmd", ""] {
        let p = shims(home).join(format!("whisper-stream{ext}"));
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("whisper-stream")
}

/// Default whisper model path under the user's Qwenodyssey home.
pub fn default_whisper_model() -> PathBuf {
    default_whisper_model_in(&home_dir())
}

pub fn default_whisper_model_in(home: &Path) -> PathBuf {
    home.join(".qwenodyssey").join("models").join("ggml-base.en.bin")
}

/// `audio.whisper_model`, else the default model path.
pub fn whisper_model_path(configured: &str) -> PathBuf {
    if configured.is_empty() {
        default_whisper_model()
    } else {
        PathBuf::from(configured)
    }
}

/// `tts.piper_bin`, else `~/.qwenodyssey/piper/piper/piper.exe`.
pub fn piper_bin(explicit: &str) -> PathBuf {
    piper_bin_in(explicit, &home_dir())
}

pub fn piper_bin_in(explicit: &str, home: &Path) -> PathBuf {
    if !explicit.is_empty() {
        return PathBuf::from(explicit);
    }
    home.join(".qwenodyssey").join("piper").join("piper").join("piper.exe")
}

/// `tts.piper_voice`, else `~/.qwenodyssey/piper/voices/en_US-amy-medium.onnx`.
pub fn piper_voice(explicit: &str) -> PathBuf {
    piper_voice_in(explicit, &home_dir())
}

pub fn piper_voice_in(explicit: &str, home: &Path) -> PathBuf {
    if !explicit.is_empty() {
        return PathBuf::from(explicit);
    }
    home.join(".qwenodyssey").join("piper").join("voices").join("en_US-amy-medium.onnx")
}

/// ffplay: the scoop shim, else `ffplay` on PATH.
pub fn resolve_ffplay() -> PathBuf {
    resolve_ffplay_in(&home_dir())
}

pub fn resolve_ffplay_in(home: &Path) -> PathBuf {
    let shim = shims(home).join("ffplay.exe");
    if shim.exists() {
        shim
    } else {
        PathBuf::from("ffplay")
    }
}

/// Temp dir for capture artifacts (`<tmp>/qwenodyssey-media`), created on demand.
pub fn capture_dir() -> std::io::Result<PathBuf> {
    let d = std::env::temp_dir().join("qwenodyssey-media");
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Milliseconds since the epoch (`Date.now()`), used to name capture files.
pub(crate) fn now_millis() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// Last path component of a `/`- or `\`-separated path (`split(/[\\/]/).pop()`).
pub fn file_label(path: &str) -> String {
    path.rsplit(['\\', '/']).next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    #[test]
    fn ffmpeg_prefers_scoop_shim() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(resolve_ffmpeg_in(home.path()), PathBuf::from("ffmpeg"));
        let shim = home.path().join("scoop").join("shims").join("ffmpeg.exe");
        touch(&shim);
        assert_eq!(resolve_ffmpeg_in(home.path()), shim);
    }

    #[test]
    fn whisper_bin_resolution_order() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(resolve_whisper_bin_in("", home.path()), PathBuf::from("whisper-cli"));
        assert_eq!(resolve_whisper_bin_in("my-whisper", home.path()), PathBuf::from("my-whisper"));
        let main = home.path().join("scoop").join("shims").join("main.exe");
        touch(&main);
        assert_eq!(resolve_whisper_bin_in("", home.path()), main);
        // whisper-cpp (.cmd) outranks main because names are searched first.
        let cpp = home.path().join("scoop").join("shims").join("whisper-cpp.cmd");
        touch(&cpp);
        assert_eq!(resolve_whisper_bin_in("", home.path()), cpp);
        let stream = home.path().join("scoop").join("shims").join("whisper-stream.exe");
        assert_eq!(resolve_whisper_stream_in(home.path()), PathBuf::from("whisper-stream"));
        touch(&stream);
        assert_eq!(resolve_whisper_stream_in(home.path()), stream);
    }

    #[test]
    fn default_paths() {
        let home = Path::new("C:/Users/u");
        assert_eq!(
            default_whisper_model_in(home),
            home.join(".qwenodyssey").join("models").join("ggml-base.en.bin")
        );
        assert_eq!(piper_bin_in("", home), home.join(".qwenodyssey/piper/piper/piper.exe"));
        assert_eq!(piper_bin_in("D:/p.exe", home), PathBuf::from("D:/p.exe"));
        assert_eq!(piper_voice_in("", home), home.join(".qwenodyssey/piper/voices/en_US-amy-medium.onnx"));
        assert_eq!(resolve_ffplay_in(home), PathBuf::from("ffplay"));
    }

    #[test]
    fn labels() {
        assert_eq!(file_label("C:\\models\\ggml-base.en.bin"), "ggml-base.en.bin");
        assert_eq!(file_label("/a/b/c.onnx"), "c.onnx");
        assert_eq!(file_label("plain"), "plain");
    }
}
