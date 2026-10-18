//! Command-line interface (port of src/cli/index.ts — commander → clap derive).
//! Same subcommands and flags; bare `qwenodyssey` drops straight into chat.

use clap::{Args, Parser, Subcommand};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "qwenodyssey",
    about = "Qwenodyssey 2.0 - a coding agent CLI for local and cloud models, in Rust.",
    disable_version_flag = true
)]
pub struct Cli {
    /// output the version number
    #[arg(short = 'V', long = "version", global = true)]
    pub version: bool,

    #[command(flatten)]
    pub global: GlobalOpts,

    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Root options shared by every subcommand (commander's `program.opts()`).
#[derive(Args, Debug, Clone, Default)]
pub struct GlobalOpts {
    /// project directory (default: current)
    #[arg(long, global = true, value_name = "dir")]
    pub cwd: Option<String>,
    /// verbose debug output
    #[arg(long, global = true)]
    pub verbose: bool,
    /// auto-confirm all prompts
    #[arg(short = 'y', long, global = true)]
    pub yes: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a .qwenodyssey workspace in this project
    Init {
        /// overwrite an existing config
        #[arg(long)]
        force: bool,
    },
    /// Get/set/list configuration (config set model.model qwen2.5:7b)
    Config {
        /// get | set | list | path
        action: Option<String>,
        /// dotted key, e.g. model.provider
        key: Option<String>,
        /// value (for set)
        value: Option<String>,
    },
    /// Plan, edit, review, and test toward a task
    Code {
        /// what you want done
        task: String,
        /// minimal planning, direct edit
        #[arg(long)]
        fast: bool,
        /// plan + review + confirm (default for small models)
        #[arg(long)]
        safe: bool,
        /// plan + code + review + test + retry loop
        #[arg(long)]
        deep: bool,
        /// run tests, diagnose failures, patch, repeat
        #[arg(long)]
        autofix: bool,
    },
    /// Edit a single file with an instruction
    Edit {
        /// file to edit
        file: String,
        /// what to change
        instruction: String,
    },
    /// Produce a step-by-step plan without editing
    Plan {
        /// the goal to plan for
        goal: String,
    },
    /// Review the current uncommitted git diff
    Review,
    /// Detect and run the project's test command
    Test,
    /// Apply previously generated (pending) edits, or roll back the last patch
    Apply {
        /// roll back the most recent applied patch
        #[arg(long)]
        rollback: bool,
    },
    /// Project memory: add | search | list | clear
    Memory {
        /// add | search | list | clear
        action: Option<String>,
        /// fact or query
        text: Option<String>,
        /// project | user | decisions | commands
        #[arg(long, value_name = "cat", default_value = "project")]
        category: String,
    },
    /// Interactive pair-coding chat (streaming)
    Chat(ChatArgs),
    /// Run a complex task across a COORDINATED team of frontier agents (live split-pane TUI)
    Swarm(SwarmArgs),
    /// Talk to the model out loud and let it see your webcam (continuous voice + vision)
    Live,
    /// Record your voice and test local transcription + the omni model's audio support
    Mictest,
    /// Live animated mic VU meter — confirm the microphone hears you
    Miclevel,
}

#[derive(Args, Debug, Clone, Default)]
pub struct ChatArgs {
    /// resume the most recent chat session in this directory
    #[arg(short = 'c', long = "continue")]
    pub continue_: bool,
    /// resume a saved chat session (omit id to pick from a list)
    #[arg(long, value_name = "id", num_args = 0..=1, default_missing_value = "")]
    pub resume: Option<String>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SwarmArgs {
    /// the task/question for the swarm (omit only with --list)
    pub task: Option<String>,
    /// supply the subtasks yourself instead of auto-decomposing
    #[arg(long, value_name = "subtasks", num_args = 1..)]
    pub divide: Option<Vec<String>>,
    /// don't integrate a final answer; just show each agent's output
    #[arg(long = "no-synth")]
    pub no_synth: bool,
    /// coordinated, but log plain lines instead of the live TUI
    #[arg(long = "no-live")]
    pub no_live: bool,
    /// classic uncoordinated mode (ensemble, or --divide shards), no shared context
    #[arg(long)]
    pub plain: bool,
    /// (plain mode) also enlist local fallback models, not just cloud ones
    #[arg(long)]
    pub local: bool,
    /// show the swarm roster and exit (no models are called)
    #[arg(long)]
    pub list: bool,
    /// rehearse the live TUI with fake streaming agents (no models called)
    #[arg(long)]
    pub demo: bool,
    /// let agents run commands: off | auto (bare metal for simple plans, daytona sandbox for complex) | bare | daytona
    #[arg(long, value_name = "mode")]
    pub exec: Option<String>,
}

/// `--fast/--deep/--autofix/--safe` → agent mode (first flag wins, as in the TS).
pub fn resolve_mode(fast: bool, safe: bool, deep: bool, autofix: bool) -> Option<qo_core::AgentMode> {
    use qo_core::AgentMode;
    if fast {
        Some(AgentMode::Fast)
    } else if deep {
        Some(AgentMode::Deep)
    } else if autofix {
        Some(AgentMode::Autofix)
    } else if safe {
        Some(AgentMode::Safe)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("qwenodyssey").chain(args.iter().copied())).expect("parses")
    }

    #[test]
    fn bare_is_chat_and_version_flag() {
        let c = parse(&[]);
        assert!(c.command.is_none());
        assert!(parse(&["--version"]).version);
        assert!(parse(&["-V"]).version);
        assert_eq!(VERSION, "2.0.0");
    }

    #[test]
    fn globals_anywhere() {
        let c = parse(&["--cwd", "x", "chat", "-y", "--verbose"]);
        assert_eq!(c.global.cwd.as_deref(), Some("x"));
        assert!(c.global.yes && c.global.verbose);
    }

    #[test]
    fn chat_resume_forms() {
        let Some(Command::Chat(a)) = parse(&["chat", "--resume"]).command else { panic!() };
        assert_eq!(a.resume.as_deref(), Some(""));
        let Some(Command::Chat(a)) = parse(&["chat", "--resume", "abc"]).command else { panic!() };
        assert_eq!(a.resume.as_deref(), Some("abc"));
        let Some(Command::Chat(a)) = parse(&["chat", "-c"]).command else { panic!() };
        assert!(a.continue_ && a.resume.is_none());
    }

    #[test]
    fn code_modes_and_memory_category() {
        let Some(Command::Code { task, deep, .. }) = parse(&["code", "do it", "--deep"]).command else { panic!() };
        assert_eq!(task, "do it");
        assert!(deep);
        assert_eq!(resolve_mode(false, true, true, false), Some(qo_core::AgentMode::Deep));
        let Some(Command::Memory { category, .. }) = parse(&["memory", "list"]).command else { panic!() };
        assert_eq!(category, "project");
    }

    #[test]
    fn swarm_flags() {
        let Some(Command::Swarm(s)) = parse(&["swarm", "task", "--no-synth", "--divide", "a", "b", "--exec", "auto"]).command
        else {
            panic!()
        };
        assert_eq!(s.task.as_deref(), Some("task"));
        assert!(s.no_synth);
        assert_eq!(s.divide, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(s.exec.as_deref(), Some("auto"));
    }
}
