//! `qwenodyssey miclevel` — a live, animated VU meter that reacts to your voice
//! so you can confirm the microphone is actually working before a live
//! session. Reads real RMS levels from the mic via ffmpeg and flags when
//! you're speaking. `q` / Esc / Ctrl-C stops it.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::Result;
use crossterm::style::Stylize;
use qo_core::CancellationToken;

use crate::bins::resolve_ffmpeg;
use crate::media::{err_msg, pick_audio_device, pick_video_device, start_level_meter};
use crate::settings::MediaSettings;
use crate::term::{gray, put, say, Keys, CLEAR_LINE};

/// Bar width in cells.
pub const WIDTH: usize = 36;
/// dB mapped to an empty bar.
pub const FLOOR: f64 = -70.0;
/// Above this ≈ speaking.
pub const SPEAK_DB: f64 = -38.0;

/// Smoothed display value + a slowly-falling peak marker for a lively animation.
#[derive(Clone, Debug, PartialEq)]
pub struct VuMeter {
    /// 0..1
    pub level: f64,
    pub peak: f64,
    pub last_db: f64,
    pub speaking: bool,
}

impl Default for VuMeter {
    fn default() -> Self {
        Self { level: 0.0, peak: 0.0, last_db: -90.0, speaking: false }
    }
}

/// One cell of the bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Peak,
    Green,
    Yellow,
    Red,
    Empty,
}

impl VuMeter {
    /// Feed a new RMS reading. Attack fast, release slow → natural VU feel.
    pub fn on_level(&mut self, db: f64) {
        self.last_db = db;
        let target = ((db - FLOOR) / (0.0 - FLOOR)).clamp(0.0, 1.0);
        self.level += (target - self.level) * if target > self.level { 0.6 } else { 0.25 };
        if self.level > self.peak {
            self.peak = self.level;
        }
        self.speaking = db > SPEAK_DB;
    }

    /// Per-frame decay: the peak slowly falls.
    pub fn tick(&mut self) {
        self.peak = (self.peak - 0.02).max(0.0);
    }

    pub fn cells(&self) -> Vec<Cell> {
        let filled = (self.level * WIDTH as f64).round() as usize;
        let peak_cell = ((self.peak * WIDTH as f64).round() as usize).min(WIDTH - 1);
        (0..WIDTH)
            .map(|i| {
                if i == peak_cell && peak_cell >= filled {
                    Cell::Peak
                } else if i < filled {
                    let frac = i as f64 / WIDTH as f64;
                    if frac > 0.85 {
                        Cell::Red
                    } else if frac > 0.6 {
                        Cell::Yellow
                    } else {
                        Cell::Green
                    }
                } else {
                    Cell::Empty
                }
            })
            .collect()
    }

    /// The dB readout: " -inf" at the floor, else the rounded value padded to 4.
    pub fn db_text(&self) -> String {
        if self.last_db <= -90.0 {
            " -inf".into()
        } else {
            // JS toFixed(0) rounds half away from zero, like f64::round.
            format!("{:>4}", format!("{}", self.last_db.round()))
        }
    }

    /// The whole meter line (without the leading `\r`).
    pub fn render(&self, color: bool) -> String {
        let bar: String = self
            .cells()
            .into_iter()
            .map(|c| {
                let (ch, styled) = match c {
                    Cell::Peak => ("│", "│".white().to_string()),
                    Cell::Green => ("█", "█".dark_green().to_string()),
                    Cell::Yellow => ("█", "█".dark_yellow().to_string()),
                    Cell::Red => ("█", "█".dark_red().to_string()),
                    Cell::Empty => ("─", "─".dark_grey().to_string()),
                };
                if color {
                    styled
                } else {
                    ch.to_string()
                }
            })
            .collect();
        let tag = if self.speaking { "● SPEAKING" } else { "· quiet   " };
        let db = format!("{} dB", self.db_text());
        if color {
            let tag = if self.speaking { tag.green().to_string() } else { gray(tag) };
            format!("  🎤 [{bar}] {}  {tag}", gray(&db))
        } else {
            format!("  🎤 [{bar}] {db}  {tag}")
        }
    }
}

/// Run `qwenodyssey miclevel`. Owns the terminal until the user quits.
pub async fn run_miclevel(settings: &MediaSettings) -> Result<()> {
    let ffmpeg = resolve_ffmpeg();
    let cancel = CancellationToken::new();
    let cam = if settings.vision.enabled {
        pick_video_device(&settings.vision.device, &ffmpeg, &cancel).await.ok().flatten()
    } else {
        None
    };
    let mic = pick_audio_device(&settings.audio.device, cam.as_deref(), &ffmpeg, &cancel).await.ok().flatten();

    say("\r\n  Qwenodyssey — microphone level meter\r\n".bold());
    let Some(mic) = mic else {
        say("  No microphone found. Set audio.device or check Windows mic privacy.\r\n".dark_red());
        return Ok(());
    };
    say(format!("{}{mic}", gray("  mic: ")));
    say(gray("  Speak — the bar should move. Press q or Ctrl-C to stop.\r\n"));

    let state = Arc::new(Mutex::new(VuMeter::default()));
    let sink = Arc::clone(&state);
    let mut meter = match start_level_meter(
        &mic,
        move |db| sink.lock().unwrap_or_else(PoisonError::into_inner).on_level(db),
        &ffmpeg,
    ) {
        Ok(m) => m,
        Err(e) => {
            say(format!("  [error: {}]\r\n", err_msg(&e)).dark_red());
            return Ok(());
        }
    };

    let mut keys = Keys::start();
    let mut frame = tokio::time::interval(Duration::from_millis(60));
    loop {
        tokio::select! {
            _ = frame.tick() => {
                let line = {
                    let mut m = state.lock().unwrap_or_else(PoisonError::into_inner);
                    m.tick();
                    m.render(true)
                };
                put(format!("\r{line}"));
            }
            k = keys.next() => {
                if k.is_quit() {
                    break;
                }
            }
        }
    }

    meter.stop();
    keys.restore();
    put(CLEAR_LINE);
    say(gray("\r\n  meter stopped.\r\n"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attack_fast_release_slow() {
        let mut m = VuMeter::default();
        m.on_level(-35.0); // target 0.5
        assert!((m.level - 0.3).abs() < 1e-9);
        assert!(m.speaking);
        assert!((m.peak - 0.3).abs() < 1e-9);
        m.on_level(-70.0); // target 0 → release at 0.25
        assert!((m.level - 0.225).abs() < 1e-9);
        assert!(!m.speaking);
        assert!((m.peak - 0.3).abs() < 1e-9, "peak holds");
        m.tick();
        assert!((m.peak - 0.28).abs() < 1e-9);
        for _ in 0..100 {
            m.tick();
        }
        assert_eq!(m.peak, 0.0);
        m.on_level(10.0); // clamps to 1
        assert!((m.level - (0.225 + 0.775 * 0.6)).abs() < 1e-9);
    }

    #[test]
    fn bar_cells_and_text() {
        let m = VuMeter::default();
        assert_eq!(m.render(false), format!("  🎤 [│{}]  -inf dB  · quiet   ", "─".repeat(WIDTH - 1)));

        let m = VuMeter { level: 1.0, peak: 1.0, last_db: -3.4, speaking: true };
        let cells = m.cells();
        assert_eq!(cells.len(), WIDTH);
        assert_eq!(cells[0], Cell::Green);
        assert_eq!(cells[21], Cell::Green); // 21/36 = 0.58
        assert_eq!(cells[22], Cell::Yellow); // 22/36 = 0.61 > 0.6
        assert_eq!(cells[31], Cell::Red); // 31/36 = 0.86 > 0.85
        assert_eq!(cells[35], Cell::Red); // the peak cell is under the fill
        assert!(m.render(false).ends_with("  -3 dB  ● SPEAKING"));

        let m = VuMeter { level: 0.5, peak: 0.8, last_db: -42.5, speaking: false };
        let cells = m.cells();
        assert_eq!(cells[17], Cell::Green);
        assert_eq!(cells[18], Cell::Empty);
        assert_eq!(cells[29], Cell::Peak);
        assert_eq!(m.db_text(), " -43");
    }
}
