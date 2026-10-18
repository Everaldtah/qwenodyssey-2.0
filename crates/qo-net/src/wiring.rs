//! Building this crate's option structs from the loaded `qo_core::config::Config`
//! (the TS chat did these mappings inline in `commands/chat.ts`).

use qo_core::config;

use crate::evolution::EvolutionOptions;
use crate::github::GithubConfig;
use crate::knowledge::{KnowledgeBase, KnowledgeOptions};
use crate::mcp::{McpOptions, McpServerSpec};
use crate::web::{SearchProvider, WebConfig};

impl WebConfig {
    /// `[web]` → web tool options.
    pub fn from_config(c: &config::WebConfig) -> Self {
        Self {
            provider: SearchProvider::parse(c.provider.as_str()),
            api_key: c.api_key.clone(),
            searxng_url: c.searxng_url.clone(),
            max_results: c.max_results as usize,
            fetch_chars: c.fetch_chars as usize,
            fetch_timeout_ms: c.fetch_timeout_ms,
            ..Self::default()
        }
    }
}

impl GithubConfig {
    /// `[github]` → GitHub tool options.
    pub fn from_config(c: &config::GithubConfig) -> Self {
        Self {
            token: c.token.clone(),
            token_env: c.token_env.clone(),
            default_owner: c.default_owner.clone(),
            api_base: c.api_base.clone(),
        }
    }
}

impl McpOptions {
    /// `mcp.init_timeout_ms` / `mcp.call_timeout_ms`.
    pub fn from_config(c: &config::McpConfig) -> Self {
        Self { init_timeout_ms: c.init_timeout_ms, call_timeout_ms: c.call_timeout_ms }
    }
}

/// The servers to connect: none unless `mcp.enabled`; each `[mcp.servers.<name>]`
/// that is enabled and has a command, in config order.
pub fn mcp_servers_from_config(c: &config::McpConfig) -> Vec<McpServerSpec> {
    if !c.enabled {
        return Vec::new();
    }
    c.servers
        .iter()
        .filter(|(_, def)| def.enabled && !def.command.is_empty())
        .map(|(name, def)| McpServerSpec {
            name: name.clone(),
            command: def.command.clone(),
            args: def.args.clone(),
            env: def.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            cwd: (!def.cwd.is_empty()).then(|| def.cwd.clone().into()),
        })
        .collect()
}

impl KnowledgeBase {
    /// The vault at `knowledge.path` (default `~/.qwenodyssey/knowledge`),
    /// embedding through `model.base_url` with `knowledge.embed_model`. Call
    /// [`KnowledgeBase::ensure`] when `knowledge.enabled`.
    pub fn from_config(c: &config::Config) -> Self {
        KnowledgeBase::new(
            &c.knowledge.path,
            KnowledgeOptions { base_url: c.model.base_url.clone(), embed_model: c.knowledge.embed_model.clone() },
        )
    }
}

impl EvolutionOptions {
    /// `[evolution]` (the chat only builds an engine when knowledge is enabled too).
    pub fn from_config(c: &config::EvolutionConfig) -> Self {
        Self { enabled: c.enabled, reflect_on_failure: c.reflect_on_failure }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn maps_defaults() {
        let cfg = config::Config::default();
        let web = WebConfig::from_config(&cfg.web);
        assert_eq!(web.provider, SearchProvider::DuckDuckGo);
        assert_eq!((web.max_results, web.fetch_chars, web.fetch_timeout_ms), (5, 8000, 15_000));
        assert!(!web.allow_private_hosts);
        let gh = GithubConfig::from_config(&cfg.github);
        assert_eq!(gh, GithubConfig::default());
        // MCP is opt-in: disabled with no servers by default.
        assert!(!cfg.mcp.enabled);
        assert!(cfg.mcp.servers.is_empty());
        assert!(cfg.mcp.call_timeout_ms > 0);
        assert!(mcp_servers_from_config(&cfg.mcp).is_empty());
        assert_eq!(McpOptions::from_config(&cfg.mcp), McpOptions::default());
        let kb = KnowledgeBase::from_config(&cfg);
        assert!(kb.dir().ends_with(Path::new(".qwenodyssey").join("knowledge")));
    }

    #[test]
    fn mcp_servers_filtered_in_config_order() {
        let toml_src = r#"
            [mcp]
            enabled = true
            [mcp.servers.zeta]
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
            [mcp.servers.alpha]
            command = "uvx"
            cwd = "C:/work"
            env = { TOKEN = "x" }
            [mcp.servers.off]
            command = "node"
            enabled = false
            [mcp.servers.nocmd]
        "#;
        let cfg: config::Config = toml::from_str(toml_src).unwrap();
        let specs = mcp_servers_from_config(&cfg.mcp);
        assert_eq!(specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["zeta", "alpha"]);
        assert_eq!(specs[0].args.len(), 3);
        assert_eq!(specs[0].cwd, None);
        assert_eq!(specs[1].cwd.as_deref(), Some(Path::new("C:/work")));
        assert_eq!(specs[1].env.get("TOKEN").map(String::as_str), Some("x"));
    }
}
