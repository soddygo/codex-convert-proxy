//! CLI argument parsing module.
//!
//! This module provides command-line argument parsing using clap.

use crate::config::{BackendConfig, MatchRules, ProxyConfig};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Codex Convert Proxy CLI.
#[derive(Parser, Debug)]
#[command(name = "codex-convert-proxy")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Proxy for converting between OpenAI Responses API and Chat API")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// CLI commands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the proxy server.
    Start(StartArgs),
    /// Generate a config file template.
    Init(InitArgs),
}

/// Start the proxy server.
#[derive(Args, Debug)]
pub struct StartArgs {
    /// Config file path.
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Provider name (glm, kimi, deepseek, minimax).
    #[arg(short, long)]
    pub provider: Option<String>,

    /// Upstream URL.
    #[arg(short, long)]
    pub upstream_url: Option<String>,

    /// API key for the upstream provider.
    #[arg(short, long)]
    pub api_key: Option<String>,

    /// Listen address.
    #[arg(short, long)]
    pub listen: Option<String>,

    /// Log directory.
    #[arg(long, default_value = "./logs")]
    pub log_dir: PathBuf,

    /// Log request/response bodies.
    #[arg(long, default_value = "false")]
    pub log_body: bool,
}

/// Initialize/generate config file.
#[derive(Args, Debug)]
pub struct InitArgs {
    /// Output path for config file.
    #[arg(default_value = "config.json")]
    pub output: PathBuf,
}

impl Cli {
    /// Parse command line arguments.
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

impl StartArgs {
    /// Build ProxyConfig from arguments.
    pub fn to_proxy_config(&self) -> anyhow::Result<ProxyConfig> {
        let mut config = if let Some(config_path) = &self.config {
            if config_path.exists() {
                let content = std::fs::read_to_string(config_path)?;
                serde_json::from_str(&content)?
            } else {
                ProxyConfig::default()
            }
        } else {
            ProxyConfig::default()
        };

        // Override with CLI args
        if let Some(provider) = &self.provider {
            let default_upstream = "https://api.example.com".to_string();
            let default_api_key = String::new();

            let upstream_url = self.upstream_url.as_ref().unwrap_or(&default_upstream);
            let api_key = self.api_key.as_ref().unwrap_or(&default_api_key);

            config.backends = vec![BackendConfig {
                name: provider.clone(),
                url: upstream_url.clone(),
                api_key: api_key.clone(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules {
                    default: true,
                    ..Default::default()
                },
            }];
        }

        if let Some(listen) = &self.listen {
            config.listen = listen.clone();
        }
        config.log_dir = self.log_dir.to_string_lossy().to_string();
        config.log_body = self.log_body;

        Ok(config)
    }
}

/// Resolved configuration that combines file and CLI args.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub proxy: ProxyConfig,
    pub backend: BackendConfig,
    pub provider_name: String,
}

impl ResolvedConfig {
    /// Create from CLI args.
    pub fn from_args(args: &StartArgs) -> anyhow::Result<Self> {
        let proxy = args.to_proxy_config()?;

        if proxy.backends.is_empty() {
            return Err(anyhow::anyhow!("No backend configured"));
        }

        let backend = proxy.backends[0].clone();
        let provider_name = backend.name.clone();

        Ok(Self {
            proxy,
            backend,
            provider_name,
        })
    }

    /// Get listen address and port.
    pub fn listen_addr(&self) -> (String, u16) {
        let parts: Vec<&str> = self.proxy.listen.split(':').collect();
        if parts.len() == 2 {
            let port: u16 = parts[1].parse().unwrap_or(8080);
            (parts[0].to_string(), port)
        } else {
            ("0.0.0.0".to_string(), 8080)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listen_addr_parsing() {
        let config = ResolvedConfig {
            proxy: ProxyConfig {
                listen: "0.0.0.0:8080".to_string(),
                ..Default::default()
            },
            backend: BackendConfig {
                name: "test".to_string(),
                url: "https://api.example.com".to_string(),
                api_key: "xxx".to_string(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules::default(),
            },
            provider_name: "test".to_string(),
        };

        let (host, port) = config.listen_addr();
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8080);
    }
}
