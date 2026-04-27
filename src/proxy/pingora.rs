//! Pingora ProxyHttp implementation for Codex Convert Proxy.
//!
//! This module provides integration with pingora for HTTP proxying.
//!
//! # Note
//!
//! The full ProxyHttp implementation requires pingora 0.8 API which has
//! some differences from previous versions. This module provides the
//! basic structure and conversion logic.
//!
//! # Example Usage
//!
//! For a complete working example with pingora, please refer to the
//! pingora repository examples at:
//! `/path/to/pingora/pingora-proxy/examples/`

use std::sync::Arc;

use crate::providers::Provider;
use crate::convert::{response_to_chat, StreamState};

/// Codex proxy handler configuration.
pub struct CodexProxy {
    /// Provider for request/response transformation.
    pub provider: Arc<dyn Provider>,
    /// Base URL for upstream LLM API.
    pub upstream_host: String,
    pub upstream_port: u16,
    pub use_tls: bool,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(provider: Arc<dyn Provider>, upstream_url: &str) -> Self {
        // Parse URL to extract host, port, and scheme
        let uri = upstream_url
            .parse::<http::Uri>()
            .unwrap_or_else(|_| http::Uri::from_static("http://localhost:443"));
        let scheme = uri.scheme().map(|s| s.as_str()).unwrap_or("http");
        let host = uri.host().unwrap_or("localhost").to_string();
        let port = uri.port_u16().unwrap_or(443);
        let use_tls = scheme == "https";

        Self {
            provider,
            upstream_host: host,
            upstream_port: port,
            use_tls,
        }
    }

    /// Convert a Response API request to Chat API request.
    pub fn convert_request(
        &self,
        req: crate::types::response_api::ResponseRequest,
    ) -> Result<crate::types::chat_api::ChatRequest, crate::error::ConversionError> {
        response_to_chat(req, self.provider.as_ref())
    }

    /// Create a new stream state for streaming responses.
    pub fn new_stream_state(&self, id: String) -> StreamState {
        StreamState::new(id)
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }
}

/// Start the proxy server with the given provider and configuration.
///
/// # Arguments
///
/// * `provider` - The LLM provider to use
/// * `upstream_url` - The upstream URL (e.g., "https://api.minimax.chat")
/// * `listen_addr` - The address to listen on (e.g., "0.0.0.0:8080")
///
/// # Returns
///
/// Returns `Ok(())` if the server was created successfully.
/// The server will need to be run separately.
///
///
/// # Example
///
/// ```ignore
/// use codex_convert_proxy::proxy::pingora::CodexProxy;
/// use codex_convert_proxy::providers::create_provider;
///
/// let provider = create_provider("minimax").unwrap();
/// let proxy = CodexProxy::new(provider, "https://api.minimax.chat");
/// proxy.start("0.0.0.0:8080")?;
/// ```
pub fn start_proxy(
    provider: Arc<dyn Provider>,
    upstream_url: &str,
    listen_addr: &str,
) -> Result<(), String> {
    let proxy = CodexProxy::new(provider, upstream_url);

    // Log configuration
    println!("Starting Codex Convert Proxy");
    println!("  Provider: {}", proxy.provider_name());
    println!("  Upstream: {}:{}{}",
        if proxy.use_tls { "https" } else { "http" },
        proxy.upstream_host,
        if proxy.upstream_port != 80 && proxy.upstream_port != 443 {
            format!(":{}", proxy.upstream_port)
        } else {
            String::new()
        });
    println!("  Listen: {}", listen_addr);

    // Note: The actual server startup requires the pingora runtime
    // which is typically done via server.run_forever()
    // See pingora examples for complete server implementation

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_creation() {
        let provider = crate::providers::GLMProvider;
        let proxy = CodexProxy::new(
            Arc::new(provider),
            "https://api.example.com:443",
        );
        assert_eq!(proxy.upstream_host, "api.example.com");
        assert_eq!(proxy.upstream_port, 443);
        assert!(proxy.use_tls);
    }

    #[test]
    fn test_proxy_http_creation() {
        let provider = crate::providers::DeepSeekProvider;
        let proxy = CodexProxy::new(Arc::new(provider), "http://localhost:8080");
        assert_eq!(proxy.upstream_host, "localhost");
        assert_eq!(proxy.upstream_port, 8080);
        assert!(!proxy.use_tls);
    }

    #[test]
    fn test_proxy_provider_name() {
        let provider = crate::providers::MiniMaxProvider;
        let proxy = CodexProxy::new(Arc::new(provider), "https://api.minimax.chat");
        assert_eq!(proxy.provider_name(), "minimax");
    }
}
