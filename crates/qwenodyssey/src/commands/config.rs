//! `qwenodyssey config [get|set|list|path] [key] [value]` (port of
//! src/cli/commands/config.ts).

use qo_core::config::{config_path, get_by_path, load_config, save_config, set_by_path};

use crate::cli::GlobalOpts;
use crate::outln;
use crate::ui::style;

/// Returns the process exit code.
pub fn config_command(action: Option<&str>, key: Option<&str>, value: Option<&str>, opts: &GlobalOpts) -> anyhow::Result<i32> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    let config = load_config(&cwd)?;

    match action {
        Some("get") => {
            let Some(key) = key else { return Ok(fail("Usage: qwenodyssey config get <key>")) };
            // JSON.stringify(undefined) prints nothing useful in the TS; print
            // `undefined` so a missing key is visible.
            match get_by_path(&config, key) {
                Some(v) => outln!("{}", serde_json::to_string(&v)?),
                None => outln!("undefined"),
            }
            Ok(0)
        }
        Some("set") => {
            let (Some(key), Some(value)) = (key, value) else {
                return Ok(fail("Usage: qwenodyssey config set <key> <value>"));
            };
            match set_by_path(&config, key, value).and_then(|next| save_config(&cwd, &next)) {
                Ok(()) => {
                    outln!("{}", style::green(&format!("✓ {key} = {value}")));
                    Ok(0)
                }
                Err(e) => Ok(fail(&format!("{e:#}"))),
            }
        }
        Some("path") => {
            outln!("{}", config_path(&cwd).display());
            Ok(0)
        }
        Some("list") | None => {
            outln!("{}", style::bold(&format!("Config ({}):", config_path(&cwd).display())));
            outln!("{}", serde_json::to_string_pretty(&config.to_json())?);
            Ok(0)
        }
        Some(other) => Ok(fail(&format!("Unknown config action: {other}. Use get|set|list|path."))),
    }
}

fn fail(msg: &str) -> i32 {
    outln!("{}", style::red(msg));
    1
}
