# Qwenodyssey 2.0

A coding-agent CLI for local and cloud models, rewritten in Rust. It drives
small local models (Ollama, LM Studio, vLLM, llama.cpp) and hosted ones
(NVIDIA NIM, OpenRouter, Anthropic, any OpenAI-compatible endpoint) through one
harness: an interactive pair-coding chat with real tools (shell, files, git,
code navigation, web, a long-term knowledge vault, MCP servers, an agent
swarm), and a plan → code → review → test pipeline for one-shot tasks.

2.0 is a port of the 1.x TypeScript CLI (v0.4.2). It keeps the behaviour, the
prompts, the config file and the saved sessions. The terminal layer is a new
design.

## What changed from 1.x

### The "switched models and now I can't type" freeze

In 1.x, switching models with `/model` could leave the terminal looking dead.
A hosted model that accepted the request and never answered (for example
`z-ai/glm-5.3` on NVIDIA NIM) held the turn for the full 90 s request timeout.
Nothing said that Esc would help, and anything typed went to the `btw ›`
side-channel instead of the prompt. The blocking request path ignored Esc, and
the fallback chain re-tried models that had already failed, each one costing
another timeout. v0.4.2 patched those paths. 2.0 keeps every one of those fixes:

- After a manual `/model` switch, cloud models get a 15 s liveness probe (Esc
  skips it). If the model doesn't answer, you are asked whether to keep it.
  Otherwise the session stays on the previous model.
- Every request (streaming and blocking, on every backend, including fallback
  attempts) carries the turn's cancellation token. Esc aborts it at once,
  whether it is still connecting, waiting for headers, or reading the body.
- The fallback chain never retries a model that already failed in the same turn.
  An interrupt never falls back. Each fallback notice says why the switch
  happened.
- A stream that goes silent after its headers is aborted as stalled.
- The spinner shows `esc to interrupt` (`esc to skip` while probing).

### One input owner (why the freeze cannot come back)

Underneath, 1.x had every prompt, picker, interjection capture and y/N gate
toggle raw mode and attach and detach its own stdin listener. A component that
forgot to undo that, or an error between the two calls, left the keyboard
going nowhere. 2.0 removes that whole class of bug by construction:

- **One input owner.** A single thread reads terminal events (keys, bracketed
  paste, resize) into a channel for the whole chat session. Nothing else reads
  stdin.
- **Raw mode is switched on once.** Raw mode and bracketed paste are enabled
  when the chat starts. A guard restores the terminal on exit, on Ctrl-C and on
  panic (through a panic hook). No component ever toggles the terminal mode.
- **Consumers take turns.** The prompt editor, the model picker, the turn-time
  capture and the y/N confirmer each borrow the input stream (a lease) while
  they are active. The turn-time capture handles Esc (interrupt) and typing (a
  queued `btw` aside, with the spinner paused while you type). The confirmer
  suspends the turn-time capture instead of fighting it. Text typed ahead
  during a probe, or half-typed when a turn ends, is handed to the next prompt
  instead of being lost.
- **One writer.** All output goes through a single UI writer. It turns `\n`
  into `\r\n` in raw mode and keeps the spinner line, the aside line and the
  prompt from overwriting each other. Background messages that arrive while
  you are typing are printed above the prompt, and the prompt is redrawn below
  them.
- **Piped stdin.** When stdin is not a terminal, chat reads plain lines and
  queues every one of them (a piped script runs every line, not just the
  first). Raw mode is never used in that case.

Non-chat commands never start the input owner. `swarm`, `live`, `mictest` and
`miclevel` manage the screen themselves.

## Install

```bash
cargo install --path crates/qwenodyssey
qwenodyssey --version   # 2.0.0
```

Or build in place: `cargo build --release -p qwenodyssey` produces
`target/release/qwenodyssey(.exe)`.

### Build notes

- Rust 1.80 or newer (the workspace's `rust-version`).
- **Windows (default): MSVC.** Install Visual Studio Build Tools 2022 with the
  "Desktop development with C++" workload, then build with the default
  `stable-x86_64-pc-windows-msvc` toolchain.
- **Windows alternative: GNU.** Without the Visual Studio Build Tools, install
  MinGW-w64 (for example `scoop install mingw`), then run
  `rustup toolchain install stable-x86_64-pc-windows-gnu` and, in the repo,
  `rustup override set stable-x86_64-pc-windows-gnu`. Keep MinGW's `bin`
  directory on `PATH` when you run cargo.
- HTTP goes through rustls, so no OpenSSL is needed.
- Optional external tools, the same as 1.x: `ollama`, LM Studio's `lms`, `git`,
  `gh` (for GitHub login), `ffmpeg`, `whisper`, `piper` (voice and vision),
  and `npx`/`uvx` for MCP servers.

## Configuration and compatibility with 1.x

- 2.0 reads the same files: `~/.qwenodyssey/config.toml` (global), layered
  under `<project>/.qwenodyssey/config.toml`. Same sections, keys and defaults.
  `qwenodyssey config list` shows the merged result.
- API keys come from the config or from the environment, as before
  (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` or `ant auth
  login`, `GITHUB_TOKEN`/`gh auth token`, `DAYTONA_API_KEY`).
- Saved chat sessions live in `~/.qwenodyssey/sessions/` in the 1.x JSON
  format. Sessions saved by 1.x resume in 2.0 with `chat -c`, `chat --resume`
  or `/resume`.
- The knowledge vault keeps the same Markdown, frontmatter and `.index.json`
  format, so an existing Obsidian vault keeps working.

## Commands

| Command | Description |
|---------|-------------|
| `qwenodyssey` / `qwenodyssey chat` | Interactive pair-coding chat (streaming). `-c/--continue` resumes the last session in this directory; `--resume [id]` picks one |
| `qwenodyssey init [--force]` | Create the `.qwenodyssey` workspace |
| `qwenodyssey config [get\|set\|list\|path] [key] [value]` | Configuration (`config set model.model qwen2.5:7b`) |
| `qwenodyssey code "task" [--fast\|--safe\|--deep\|--autofix]` | Full pipeline: plan → edit → review → test |
| `qwenodyssey edit <file> "instruction"` | Edit a single file |
| `qwenodyssey plan "goal"` | Produce a plan without editing |
| `qwenodyssey review` | Review the current git diff |
| `qwenodyssey test` | Detect and run the project's tests |
| `qwenodyssey apply [--rollback]` | Apply pending edits, or roll back the last patch |
| `qwenodyssey memory [add\|search\|list\|clear] [text] [--category c]` | Project memory |
| `qwenodyssey swarm "task"` | Coordinated multi-model team with a live split-pane TUI (`--divide`, `--no-synth`, `--no-live`, `--plain`, `--local`, `--list`, `--demo`, `--exec`) |
| `qwenodyssey live` | Talk to the model out loud; it can see your webcam |
| `qwenodyssey mictest` / `miclevel` | Test the microphone and transcription / live mic meter |

Global flags: `--cwd <dir>`, `--verbose`, `-y/--yes` (auto-confirm).

### In chat

Type a message, or `@path` to inline a file. Type `/` to open the command
palette. It filters as you type; ↑/↓ moves through it, and Tab or Enter
accepts. A multi-line paste collapses into a `[+N lines]` chip and is expanded
when you submit.

| Slash command | |
|---|---|
| `/help` `/commands` `/?` | List commands |
| `/settings` `/config` | Current model and runtime settings |
| `/models`, `/model [name\|#]` | Model picker, or switch directly (cloud models are probed) |
| `/context`, `/compact` | Context usage bar / summarize older turns now |
| `/plan` | The agent's current step plan |
| `/memory` `/knowledge` | The long-term knowledge vault |
| `/lessons` `/evolution` | Lessons learned from past mistakes |
| `/sessions`, `/resume [id]` | Saved sessions |
| `/btw <question>` | An aside. You can also just type while the model works |
| `/reset`, `/exit` `/quit` | New session / quit |

While the model works: **Esc** interrupts. Typing queues an aside, which the
model addresses at its next step. **Ctrl-C** quits.

## Safety

- Dangerous shell commands are hard-blocked. Other destructive commands ask
  first, through the same input owner as everything else. `tools.allow_commands`
  and `tools.deny_commands` tune this.
- `write_file` refuses to blindly overwrite a file the model has not read in
  this session.
- Edits from the pipeline are journaled to `.qwenodyssey/patches/` and can be
  rolled back with `qwenodyssey apply --rollback`.
- Child processes get a null stdin unless they really need input, so nothing
  can take over the keyboard.

## Workspace layout

| Crate | Contents |
|---|---|
| `qo-core` | Shared types and traits, config, model profiles, prompts, JSON repair, logger |
| `qo-providers` | Every model backend, the factory, LM Studio `lms` control |
| `qo-tools` | Local tools, tool specs, tool-call hardening, persistent shell session |
| `qo-net` | Web, GitHub, MCP, knowledge vault, sessions, evolution, compaction |
| `qo-pipeline` | The plan → code → patch → review → test pipeline, repo scanner, project memory |
| `qo-swarm` | Coordinated agent swarm, execution backends, swarm TUI |
| `qo-media` | Voice and vision: mic, VAD, TTS, vision, omni |
| `qwenodyssey` | The binary: CLI, chat loop, terminal UI |

## Tests

`cargo test --workspace` runs about 650 tests. They are headless and never call a paid API:
providers, web, GitHub, MCP, vision and Daytona are exercised against local mock servers on
127.0.0.1. A few machine-specific checks (the real NIM endpoint, your real config, your home
directory) are `#[ignore]` and run with `cargo test -- --ignored`.

## License

MIT. See [LICENSE](LICENSE).
