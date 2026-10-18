//! qo-swarm: Qwenodyssey's agent swarm.
//!
//! - [`swarm`]: frontier fan-out — the roster from the configured cloud keys,
//!   ensemble / divide runs, synthesis (core/swarm.ts).
//! - [`coordinator`]: the coordinated team — lead-planned dependency-aware
//!   subtasks, a shared blackboard, parallel waves, per-agent stop, streamed
//!   synthesis (core/swarmCoordinator.ts).
//! - [`events`]: the coordinator's event stream (tokio channels, lossless fan-out).
//! - [`exec`]: where agents may run commands — bare metal behind the run_shell
//!   safety gate, or a Daytona sandbox over REST; `auto` routing (core/agentExec.ts).
//! - [`tool`]: the `agent_swarm` tool + its model-facing spec (tools/swarmTools.ts).
//! - [`tui`]: the ratatui live split-pane dashboard (cli/swarmTui.ts).
//! - [`command`]: the `qwenodyssey swarm` subcommand logic (cli/commands/swarm.ts).
//! - [`env`](mod@env): the settings + provider factory everything codes against; built
//!   from the real config by [`SwarmEnv::from_config`] (the only upstream-API seam).

mod compat;

pub mod command;
pub mod coordinator;
pub mod env;
pub mod events;
pub mod exec;
pub mod swarm;
pub mod tool;
pub mod tui;

use std::io::Write;
use std::sync::Arc;

use qo_core::config::Config;
use qo_core::tool::SharedTool;

pub use command::{run_swarm_command_with_env, DemoProvider, SwarmCommandOptions, SwarmCommandOutcome};
pub use compat::{daytona_settings, swarm_settings, ConfigProviderFactory};
pub use coordinator::{
    build_roster, clean_agent_text, create_coordinated_swarm, decompose, decompose_with, parse_complexity,
    parse_plan, parse_special_tool_calls, pick_lead, preflight_workers, Blackboard, BoardEntry, BuiltSwarm,
    CoordinateOptions, CoordinatedSwarm, DecomposeResult, PlanComplexity, RosterResult, Subtask, SubtaskStatus,
};
pub use env::{DaytonaSettings, ProviderFactory, SwarmEnv, SwarmSettings};
pub use events::{
    AssignEvent, DeltaEvent, ExecCmdEvent, PlanEvent, PreflightResult, StatusEvent, SwarmEvent, SwarmEvents, WaveEvent,
};
pub use exec::{
    choose_executor, daytona_available, resolve_daytona_key, AgentExecutor, BareExecutor, BareExecutorOptions,
    DaytonaExecutor, ExecKind, ExecMode, ExecResult, ExecutorChoice, RunOptions, SharedExecutor,
};
pub use swarm::{
    frontier_workers, is_reasoning_model, parse_ref, short_label, strip_thinking, synthesize, RosterEntry, Swarm,
    SwarmArtifacts, SwarmMode, SwarmOptions, SwarmRun, SwarmWorker, WorkerKind, WorkerResult,
};
pub use tool::{
    create_swarm_tools_with_env, format_coordinated, format_run, lead_provider, swarm_ready_roster, swarm_tool_specs,
    AgentSwarmTool,
};
pub use tui::{SwarmTui, SwarmTuiOptions, TuiControls, TuiOutcome};

/// The swarm environment for a loaded config (settings + the real provider factory).
pub fn swarm_env(config: &Config) -> Arc<SwarmEnv> {
    Arc::new(SwarmEnv::from_config(config))
}

/// The `agent_swarm` tool for a chat session's registry. Register it (and
/// advertise [`swarm_tool_specs`]) only when [`swarm_chat_roster`] is non-empty.
pub fn create_swarm_tools(config: &Config) -> Vec<SharedTool> {
    create_swarm_tools_with_env(swarm_env(config))
}

/// Chat gate: the frontier roster when `[swarm].enabled` and ≥2 frontier workers
/// exist (offer the agent_swarm tool + its system-prompt section), else empty.
pub fn swarm_chat_roster(config: &Config) -> Vec<RosterEntry> {
    swarm_ready_roster(&SwarmEnv::from_config(config))
}

/// `qwenodyssey swarm …`: run the subcommand for `config`, writing its line
/// output to `out` (the live TUI draws on the terminal directly).
pub async fn run_swarm_command(
    config: &Config,
    opts: SwarmCommandOptions,
    out: &mut (dyn Write + Send),
) -> anyhow::Result<SwarmCommandOutcome> {
    run_swarm_command_with_env(swarm_env(config), opts, out).await
}
