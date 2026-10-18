//! Commands that own the screen themselves: `swarm` (live split-pane TUI) and
//! `live` / `mictest` / `miclevel` (voice + vision). They must NOT start the
//! chat's input owner or raw mode — each manages its own terminal state and
//! restores it on exit and on panic.

use crate::cli::{GlobalOpts, SwarmArgs};

/// `qwenodyssey swarm [task] …`.
pub async fn swarm(opts: &GlobalOpts, a: &SwarmArgs) -> anyhow::Result<i32> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    let config = qo_core::config::load_config(&cwd)?;
    let o = qo_swarm::SwarmCommandOptions {
        task: a.task.clone(),
        divide: a.divide.clone().unwrap_or_default(),
        synth: !a.no_synth,
        local: a.local,
        list: a.list,
        live: !a.no_live,
        plain: a.plain,
        demo: a.demo,
        exec: a.exec.clone(),
        cwd,
        color: crate::ui::style::enabled(),
    };
    let mut stdout = std::io::stdout();
    let outcome = qo_swarm::run_swarm_command(&config, o, &mut stdout).await?;
    Ok(outcome.exit_code)
}

fn media_settings(opts: &GlobalOpts) -> anyhow::Result<(qo_core::config::Config, qo_media::MediaSettings)> {
    let cwd = crate::session::resolve_cwd(opts.cwd.as_deref());
    let config = qo_core::config::load_config(&cwd)?;
    let settings = qo_media::MediaSettings::from_config(&config);
    Ok((config, settings))
}

/// `qwenodyssey live`.
pub async fn live(opts: &GlobalOpts) -> anyhow::Result<i32> {
    let (config, settings) = media_settings(opts)?;
    let provider = qo_providers::create_provider(&config);
    qo_media::run_live(&settings, provider).await?;
    Ok(0)
}

/// `qwenodyssey mictest`.
pub async fn mictest(opts: &GlobalOpts) -> anyhow::Result<i32> {
    let (_, settings) = media_settings(opts)?;
    qo_media::run_mictest(&settings).await?;
    Ok(0)
}

/// `qwenodyssey miclevel`.
pub async fn miclevel(opts: &GlobalOpts) -> anyhow::Result<i32> {
    let (_, settings) = media_settings(opts)?;
    qo_media::run_miclevel(&settings).await?;
    Ok(0)
}
