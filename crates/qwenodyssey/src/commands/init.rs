//! `qwenodyssey init [--force]` (port of src/cli/commands/init.ts): create the
//! project workspace layout, a default config and the seed memory files. The
//! work itself is `qo_pipeline::commands::init_workspace`; this only renders it.

use qo_pipeline::commands::{init_workspace, InitOutcome, INIT_NEXT_STEPS};

use crate::cli::GlobalOpts;
use crate::outln;
use crate::ui::style;

pub fn init_command(opts: &GlobalOpts, force: bool) -> anyhow::Result<i32> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    match init_workspace(&cwd, force)? {
        InitOutcome::AlreadyExists(cfg_file) => {
            outln!("{}", style::yellow(&format!("{} already exists. Use --force to overwrite.", cfg_file.display())));
        }
        InitOutcome::Initialized(cfg_file) => {
            outln!("{}", style::green("✓ Initialized Qwenodyssey workspace"));
            outln!("  Config: {}", style::cyan(&cfg_file.display().to_string()));
            outln!("\nNext steps:");
            for hint in INIT_NEXT_STEPS {
                outln!("{}", style::gray(hint));
            }
        }
    }
    Ok(0)
}
