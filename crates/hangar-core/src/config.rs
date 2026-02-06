//! `hangar.toml` - the single source of truth for the daemon.

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct HangarConfig {
    pub runtime: RuntimeConfig,
    pub server: ServerConfig,
    pub memory: MemoryConfig,
    pub routing: BTreeMap<String, RouteProfile>,
    pub agent: AgentConfig,
    pub cluster: ClusterConfig,
}

impl HangarConfig {
    pub fn load_or_default(path: &str) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(s) => Ok(toml::from_str(&s)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClusterConfig {
    pub role: String,
    pub coordinator_url: String,
    pub node_name: String,
    pub advertise_url: String,
    pub heartbeat_secs: u64,
    pub node_timeout_secs: u64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            role: "standalone".into(),
            coordinator_url: String::new(),
            node_name: String::new(),
            advertise_url: String::new(),
            heartbeat_secs: 5,
            node_timeout_secs: 20,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub backend: String,
    pub offline: bool,
    pub ollama_url: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: "ollama".into(),
            offline: true,
            ollama_url: "http://127.0.0.1:11434".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11435,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub ceiling_gb: Option<f64>,
    pub warm_pool: Vec<String>,
    /// Unload the models we loaded when the daemon stops. On by default because
    /// we load with keep_alive = -1, so without this they would stay resident in
    /// the runtime after hangard exits.
    pub unload_on_exit: bool,
    pub idle_unload_minutes: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            ceiling_gb: None,
            warm_pool: Vec::new(),
            unload_on_exit: true,
            idle_unload_minutes: 30,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteProfile {
    pub model: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    pub tools: Vec<String>,
    pub confirm_before: Vec<String>,
    pub auto_edit: bool,
}

#[cfg(test)]
mod tests {
    use super::HangarConfig;

    #[test]
    fn empty_config_uses_defaults() {
        let cfg: HangarConfig = toml::from_str("").unwrap();
        assert_eq!(cfg.runtime.backend, "ollama");
        assert!(cfg.runtime.offline);
        assert_eq!(cfg.server.port, 11435);
        assert!(!cfg.agent.auto_edit);
    }

    #[test]
    fn sample_config_round_trips() {
        let toml = r#"
            [runtime]
            backend = "ollama"
            offline = true

            [memory]
            ceiling_gb = 40
            warm_pool = ["qwen2.5-coder:3b"]

            [routing.autocomplete]
            model = "qwen2.5-coder:3b"

            [agent]
            tools = ["read_file", "edit_file", "bash", "grep", "glob"]
            confirm_before = ["edit_file", "bash"]
        "#;
        let cfg: HangarConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.memory.ceiling_gb, Some(40.0));
        assert_eq!(cfg.routing["autocomplete"].model, "qwen2.5-coder:3b");
        assert_eq!(cfg.agent.confirm_before, vec!["edit_file", "bash"]);
    }
}
