//! Configuration module for proxy settings and backend routing.
//!
//! This module provides configuration structures for the proxy server,
//! including backend definitions and routing rules.

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Single backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend name (for logging and stats).
    pub name: String,
    /// Backend URL (e.g., https://api.anthropic.com).
    pub url: String,
    /// API Key for authentication.
    pub api_key: String,
    /// API protocol: "openai" or "anthropic".
    #[serde(default = "default_protocol")]
    pub protocol: String,
    /// Model to use for this backend (overrides request model).
    #[serde(default)]
    pub model: Option<String>,
    /// Match rules for routing.
    #[serde(default)]
    pub match_rules: MatchRules,
}

fn default_protocol() -> String {
    "openai".to_string()
}

/// Backend matching rules.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MatchRules {
    /// Path prefix for matching.
    pub path_prefix: Option<String>,
    /// Header match rule.
    pub header: Option<HeaderMatch>,
    /// Whether this is the default backend.
    #[serde(default)]
    pub default: bool,
}

/// Header matching rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderMatch {
    pub name: String,
    pub value: String,
}

/// Parsed backend connection information.
#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub use_tls: bool,
    pub base_path: String,
    pub api_key: String,
    pub protocol: String,
    pub model: Option<String>,
}

impl BackendInfo {
    /// Parse connection info from backend config.
    pub fn from_config(config: &BackendConfig) -> anyhow::Result<Self> {
        let parsed = url::Url::parse(&config.url)
            .map_err(|e| anyhow::anyhow!("Invalid backend URL '{}': {}", config.name, e))?;

        let host = parsed
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("Backend '{}' missing host", config.name))?
            .to_string();

        let use_tls = parsed.scheme() == "https";
        let port = parsed.port().unwrap_or(if use_tls { 443 } else { 80 });
        let base_path = parsed.path().trim_end_matches('/').to_string();

        Ok(Self {
            name: config.name.clone(),
            host,
            port,
            use_tls,
            base_path,
            api_key: config.api_key.clone(),
            protocol: config.protocol.clone(),
            model: config.model.clone(),
        })
    }

    /// Check if using Anthropic-style auth (x-api-key header).
    pub fn use_anthropic_auth(&self) -> bool {
        self.protocol.to_lowercase() != "openai"
    }
}

/// Backend router for selecting backends based on request characteristics.
#[derive(Debug, Clone)]
pub struct BackendRouter {
    backends: Vec<(BackendConfig, BackendInfo)>,
    default_index: Option<usize>,
}

impl BackendRouter {
    fn path_matches_prefix(path: &str, prefix: &str) -> bool {
        let normalized = if prefix != "/" {
            prefix.trim_end_matches('/')
        } else {
            prefix
        };
        if normalized.is_empty() {
            return false;
        }
        if path == normalized {
            return true;
        }
        let with_slash = format!("{}/", normalized);
        path.starts_with(&with_slash)
    }

    /// Create a new backend router from configs.
    pub fn new(configs: Vec<BackendConfig>) -> anyhow::Result<Self> {
        if configs.is_empty() {
            return Err(anyhow::anyhow!("At least one backend must be configured"));
        }

        let mut backends = Vec::new();
        let mut default_index = None;

        for (i, config) in configs.into_iter().enumerate() {
            let default_marker = if config.match_rules.default { " [default]" } else { "" };
            info!(
                "Loading backend [{}]: {} -> {}{}",
                config.name, config.url, config.protocol, default_marker
            );

            if config.match_rules.default {
                if default_index.is_some() {
                    warn!("Multiple default backends configured, using last one");
                }
                default_index = Some(i);
            }

            let info = BackendInfo::from_config(&config)?;
            backends.push((config, info));
        }

        // Use first backend as default if none specified
        let default_index = default_index.or(Some(0));

        Ok(Self {
            backends,
            default_index,
        })
    }

    /// Select a backend based on request path and headers.
    pub fn select(&self, path: &str, headers: &[(String, String)]) -> Option<&BackendInfo> {
        for (config, info) in &self.backends {
            // Check path prefix match
            if let Some(ref prefix) = config.match_rules.path_prefix
                && Self::path_matches_prefix(path, prefix) {
                    debug!(
                        "Path '{}' matched backend '{}' (prefix: {})",
                        path, config.name, prefix
                    );
                    return Some(info);
                }

            // Check header match
            if let Some(ref header_match) = config.match_rules.header {
                for (name, value) in headers {
                    if name.eq_ignore_ascii_case(&header_match.name)
                        && value == &header_match.value
                    {
                        debug!(
                            "Header '{}: {}' matched backend '{}'",
                            header_match.name, header_match.value, config.name
                        );
                        return Some(info);
                    }
                }
            }
        }

        // Fall back to default backend
        if let Some(index) = self.default_index {
            debug!("Using default backend '{}'", self.backends[index].1.name);
            return Some(&self.backends[index].1);
        }

        None
    }

    /// Select backend and compute rewritten path.
    pub fn select_and_rewrite(
        &self,
        path: &str,
        headers: &[(String, String)],
    ) -> Option<(&BackendInfo, String)> {
        let (config, info) = self.select_with_config(path, headers)?;

        // Remove path prefix if present
        let new_path = if let Some(ref prefix) = config.match_rules.path_prefix {
            path.strip_prefix(prefix).unwrap_or(path).to_string()
        } else {
            path.to_string()
        };

        // Add backend's base_path
        let new_path = if !info.base_path.is_empty() {
            format!("{}{}", info.base_path, new_path)
        } else {
            new_path
        };

        Some((info, new_path))
    }

    /// Select backend with config.
    pub fn select_with_config(
        &self,
        path: &str,
        headers: &[(String, String)],
    ) -> Option<(&BackendConfig, &BackendInfo)> {
        for (config, info) in &self.backends {
            // Check path prefix
            if let Some(ref prefix) = config.match_rules.path_prefix
                && Self::path_matches_prefix(path, prefix) {
                    return Some((config, info));
                }

            // Check header match
            if let Some(ref header_match) = config.match_rules.header {
                for (name, value) in headers {
                    if name.eq_ignore_ascii_case(&header_match.name)
                        && value == &header_match.value
                    {
                        return Some((config, info));
                    }
                }
            }
        }

        // Fall back to default
        self.default_index.map(|i| (&self.backends[i].0, &self.backends[i].1))
    }

    /// Get all backend names.
    pub fn backend_names(&self) -> Vec<&str> {
        self.backends.iter().map(|(c, _)| c.name.as_str()).collect()
    }

    /// Get the default backend.
    pub fn default_backend(&self) -> Option<&BackendInfo> {
        self.default_index.map(|i| &self.backends[i].1)
    }
}

/// Proxy configuration for the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// Listen address.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Log directory.
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
    /// Whether to log request/response bodies.
    #[serde(default = "default_log_body")]
    pub log_body: bool,
    /// Backends configuration.
    pub backends: Vec<BackendConfig>,
}

fn default_listen() -> String {
    "0.0.0.0:8080".to_string()
}

fn default_log_dir() -> String {
    "./logs".to_string()
}

fn default_log_body() -> bool {
    false
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            log_dir: default_log_dir(),
            log_body: default_log_body(),
            backends: Vec::new(),
        }
    }
}

impl BackendConfig {
    /// Convert config to BackendInfo.
    pub fn to_backend_info(&self) -> anyhow::Result<BackendInfo> {
        BackendInfo::from_config(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_router() {
        let configs = vec![
            BackendConfig {
                name: "anthropic".to_string(),
                url: "https://api.anthropic.com".to_string(),
                api_key: "sk-ant-xxx".to_string(),
                protocol: "anthropic".to_string(),
                model: None,
                match_rules: MatchRules {
                    path_prefix: Some("/anthropic".to_string()),
                    ..Default::default()
                },
            },
            BackendConfig {
                name: "openai".to_string(),
                url: "https://api.openai.com/v1".to_string(),
                api_key: "sk-xxx".to_string(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules {
                    path_prefix: Some("/openai".to_string()),
                    ..Default::default()
                },
            },
            BackendConfig {
                name: "default".to_string(),
                url: "https://api.example.com".to_string(),
                api_key: "xxx".to_string(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules {
                    default: true,
                    ..Default::default()
                },
            },
        ];

        let router = BackendRouter::new(configs).unwrap();

        // Test path matching
        let (info, path) = router.select_and_rewrite("/anthropic/v1/messages", &[]).unwrap();
        assert_eq!(info.name, "anthropic");
        assert_eq!(path, "/v1/messages");

        let (info, path) = router.select_and_rewrite("/openai/chat/completions", &[]).unwrap();
        assert_eq!(info.name, "openai");
        assert_eq!(path, "/v1/chat/completions");

        // Test default fallback
        let (info, path) = router.select_and_rewrite("/other/path", &[]).unwrap();
        assert_eq!(info.name, "default");
        assert_eq!(path, "/other/path");
    }

    #[test]
    fn test_anthropic_auth() {
        let info = BackendInfo {
            name: "test".to_string(),
            host: "localhost".to_string(),
            port: 443,
            use_tls: true,
            base_path: String::new(),
            api_key: "test".to_string(),
            protocol: "anthropic".to_string(),
            model: None,
        };
        assert!(info.use_anthropic_auth());

        let info = BackendInfo {
            protocol: "openai".to_string(),
            ..info
        };
        assert!(!info.use_anthropic_auth());
    }

    #[test]
    fn test_select_and_rewrite_with_responses_prefix() {
        let configs = vec![
            BackendConfig {
                name: "kimi".to_string(),
                url: "https://api.moonshot.cn/v1".to_string(),
                api_key: "sk-kimi".to_string(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules {
                    path_prefix: Some("/kimi".to_string()),
                    ..Default::default()
                },
            },
            BackendConfig {
                name: "default".to_string(),
                url: "https://api.example.com".to_string(),
                api_key: "sk-default".to_string(),
                protocol: "openai".to_string(),
                model: None,
                match_rules: MatchRules {
                    default: true,
                    ..Default::default()
                },
            },
        ];

        let router = BackendRouter::new(configs).unwrap();
        let (info, rewritten_path) = router.select_and_rewrite("/kimi/responses", &[]).unwrap();
        assert_eq!(info.name, "kimi");
        assert_eq!(rewritten_path, "/v1/responses");
    }
}
