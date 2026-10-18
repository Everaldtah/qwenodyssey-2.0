//! Shared runtime objects a command needs, built from the global CLI flags
//! (port of src/cli/session.ts).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use qo_core::config::{load_config, Config};
use qo_core::logger::Logger;
use qo_core::{AgentMode, SharedProvider, ToolContext, ToolRegistry};
use qo_providers::{LmStudioControl, LmsModel};

use crate::cli::GlobalOpts;

pub struct Session {
    pub cwd: PathBuf,
    pub config: Config,
    pub logger: Logger,
    /// The active backend. Replaced (not mutated) when `/model` or the
    /// fallback chain switches to another backend.
    pub provider: SharedProvider,
    pub auto_confirm: bool,
    pub mode: AgentMode,
    /// Qwenodyssey's own source root (the cargo workspace this binary was
    /// built from), for self-inspection/modification when allowed.
    pub self_root: PathBuf,
    /// Headless LM Studio control (`lms` CLI).
    pub lms: Arc<LmStudioControl>,
    /// LM Studio model keys from the startup inventory (tool-capable first).
    pub lms_model_keys: Vec<String>,
    /// Full LM Studio inventory from the same startup `lms ls`.
    pub lms_models: Option<Vec<LmsModel>>,
    /// Set once the user picks a model by hand (/model, /models). Startup model
    /// resolution must then leave the choice alone — a slow startup used to
    /// finish AFTER a manual switch and silently move the session elsewhere.
    pub model_pinned: bool,
}

/// The workspace root this binary was built from (crates/qwenodyssey/../..).
pub fn self_root() -> PathBuf {
    normalize(&Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

impl Session {
    pub fn create(opts: &GlobalOpts, mode: Option<AgentMode>) -> anyhow::Result<Self> {
        let cwd = resolve_cwd(opts.cwd.as_deref());
        let config = load_config(&cwd)?;
        let logger = Logger::new(&cwd, opts.verbose);
        let provider = qo_providers::create_provider(&config);
        let cli_path = config.lmstudio.cli_path.trim().to_string();
        let lms = Arc::new(LmStudioControl::new(if cli_path.is_empty() { None } else { Some(cli_path.as_str()) }));
        let mode = mode.unwrap_or(if config.agent.small_model_mode { AgentMode::Safe } else { AgentMode::Deep });
        Ok(Self {
            cwd,
            config,
            logger,
            provider,
            auto_confirm: opts.yes,
            mode,
            self_root: self_root(),
            lms,
            lms_model_keys: Vec::new(),
            lms_models: None,
            model_pinned: false,
        })
    }

    /// The tool context for the non-chat commands (the TS `createSession`).
    pub fn tool_context(&self) -> ToolContext {
        let mut ctx = ToolContext::new(&self.cwd);
        ctx.auto_confirm = self.auto_confirm;
        ctx.confirm_destructive = self.config.tools.confirm_destructive;
        ctx.allow_shell = self.config.tools.allow_shell;
        ctx.sandbox = self.config.tools.sandbox;
        ctx.self_root = Some(self.self_root.clone());
        ctx.log = self.logger.log_fn();
        ctx
    }

    /// A registry with the default tool set (TS `new ToolRegistry(ctx)`).
    pub fn default_tools(&self) -> ToolRegistry {
        let mut reg = ToolRegistry::new(self.tool_context());
        qo_tools::register_defaults(&mut reg);
        reg
    }

    /// Project memory store (`memory.path`, relative to the project).
    pub fn memory(&self) -> qo_pipeline::memory::MemoryStore {
        qo_pipeline::memory::MemoryStore::for_project(&self.cwd, &self.config.memory.path)
    }
}

/// `path.resolve(opts.cwd || process.cwd())`.
pub fn resolve_cwd(cwd: Option<&str>) -> PathBuf {
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let p = match cwd {
        Some(c) if !c.trim().is_empty() => {
            let p = Path::new(c);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base.join(p)
            }
        }
        _ => base,
    };
    normalize(&p)
}

/// Lexically normalize `.` / `..` (like `path.resolve`) without touching the
/// filesystem (no `\\?\` prefixes from `canonicalize` on Windows).
pub fn normalize(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_and_normalizes() {
        let here = std::env::current_dir().unwrap();
        assert_eq!(resolve_cwd(None), normalize(&here));
        assert_eq!(resolve_cwd(Some("a/../b")), normalize(&here.join("b")));
        assert!(self_root().join("Cargo.toml").exists());
    }
}
