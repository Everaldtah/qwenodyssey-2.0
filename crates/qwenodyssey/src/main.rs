//! Qwenodyssey 2.0 — a coding agent CLI for local and cloud models.
//!
//! `qwenodyssey` (bare) and `qwenodyssey chat` run the interactive chat; the
//! other subcommands run the plan→code→review pipeline, the agent swarm, or
//! the voice/vision modes. Only the chat takes over stdin (raw mode + the one
//! input-owner thread); every other command leaves the terminal alone or, for
//! swarm/live/mictest/miclevel, manages it itself.

mod chat;
mod cli;
mod commands;
mod render;
mod session;
mod ui;

use clap::Parser;

use cli::{Cli, Command};

fn main() {
    ui::style::init();
    ui::term::install_panic_hook();
    let args = Cli::parse();
    if args.version {
        println!("{}", cli::VERSION);
        return;
    }
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Fatal: {e}");
            std::process::exit(1);
        }
    };
    let code = rt.block_on(run(args)).unwrap_or_else(|e| {
        ui::term::restore();
        outln!("{}", ui::style::red(&format!("Fatal: {e:#}")));
        1
    });
    // Exit promptly even if a best-effort background job (LM Studio start,
    // model list, GitHub probe) is still pending — nothing is left to wait for.
    ui::term::exit(code);
}

async fn run(args: Cli) -> anyhow::Result<i32> {
    let g = args.global.clone();
    match args.command {
        None => chat::chat_command(&g, &cli::ChatArgs::default()).await,
        Some(Command::Chat(a)) => chat::chat_command(&g, &a).await,
        Some(Command::Init { force }) => commands::init::init_command(&g, force),
        Some(Command::Config { action, key, value }) => {
            commands::config::config_command(action.as_deref(), key.as_deref(), value.as_deref(), &g)
        }
        Some(Command::Code { task, fast, safe, deep, autofix }) => {
            commands::pipeline::code(&g, &task, cli::resolve_mode(fast, safe, deep, autofix)).await
        }
        Some(Command::Edit { file, instruction }) => commands::pipeline::edit(&g, &file, &instruction).await,
        Some(Command::Plan { goal }) => commands::pipeline::plan(&g, &goal).await,
        Some(Command::Review) => commands::pipeline::review(&g).await,
        Some(Command::Test) => commands::pipeline::test(&g).await,
        Some(Command::Apply { rollback }) => commands::pipeline::apply(&g, rollback),
        Some(Command::Memory { action, text, category }) => {
            commands::pipeline::memory(&g, action.as_deref(), text.as_deref(), &category)
        }
        Some(Command::Swarm(a)) => commands::external::swarm(&g, &a).await,
        Some(Command::Live) => commands::external::live(&g).await,
        Some(Command::Mictest) => commands::external::mictest(&g).await,
        Some(Command::Miclevel) => commands::external::miclevel(&g).await,
    }
}
