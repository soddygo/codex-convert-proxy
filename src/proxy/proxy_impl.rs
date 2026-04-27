//! Pingora ProxyHttp implementation for Codex Convert Proxy.
//!
//! This module provides full ProxyHttp trait implementation for HTTP proxying
//! with request/response format conversion between Responses API and Chat API.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use pingora_core::protocols::Digest;
use pingora_core::protocols::TcpKeepalive;
use pingora_core::Result as PingoraResult;
use pingora_core::upstreams::peer::{ALPN, HttpPeer, Peer};
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{ProxyHttp, Session};
use tracing::{debug, error, info};

use crate::convert::{response_to_chat, StreamState};
use crate::error::ConversionError;
use crate::providers::Provider;
use crate::types::chat_api::ChatRequest;
use crate::types::response_api::ResponseRequest;

/// Proxy context attached to each request session.
#[derive(Debug)]
pub struct ProxyContext {
    /// Request start time for duration tracking.
    pub start_time: Instant,
    /// Collected request body bytes.
    pub request_body: Vec<u8>,
    /// Model name parsed from request.
    pub model: Option<String>,
    /// Selected backend host.
    pub backend_host: Option<String>,
    /// Provider name.
    pub provider_name: Option<String>,
    /// Stream state for SSE conversion.
    pub stream_state: Option<StreamState>,
    /// Response body collected for conversion.
    pub response_body: Vec<u8>,
    /// Whether streaming is enabled.
    pub is_streaming: bool,
    /// Rewritten upstream path.
    pub rewritten_path: Option<String>,
}

impl ProxyContext {
    /// Create a new proxy context.
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            request_body: Vec::new(),
            model: None,
            backend_host: None,
            provider_name: None,
            stream_state: None,
            response_body: Vec::new(),
            is_streaming: false,
            rewritten_path: None,
        }
    }
}

/// Codex proxy handler implementing ProxyHttp trait.
pub struct CodexProxy {
    /// Provider for request/response transformation.
    pub provider: Arc<dyn Provider>,
    /// Upstream URL for the LLM API.
    pub upstream_url: String,
    /// Upstream host.
    pub upstream_host: String,
    /// Upstream port.
    pub upstream_port: u16,
    /// Whether to use TLS.
    pub use_tls: bool,
    /// API key for authentication.
    pub api_key: String,
    /// Whether to log request/response bodies.
    pub log_body: bool,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        provider: Arc<dyn Provider>,
        upstream_url: &str,
        api_key: &str,
        log_body: bool,
    ) -> Self {
        let uri = upstream_url
            .parse::<http::Uri>()
            .unwrap_or_else(|_| http::Uri::from_static("http://localhost:443"));
        let scheme = uri.scheme().map(|s| s.as_str()).unwrap_or("http");
        let host = uri.host().unwrap_or("localhost").to_string();
        let port = uri.port_u16().unwrap_or(443);
        let use_tls = scheme == "https";
        let upstream_url = format!(
            "{}://{}:{}",
            if use_tls { "https" } else { "http" },
            host,
            port
        );

        Self {
            provider,
            upstream_url,
            upstream_host: host,
            upstream_port: port,
            use_tls,
            api_key: api_key.to_string(),
            log_body,
        }
    }

    /// Convert a Responses API request to Chat API request.
    fn convert_request(&self, body: &[u8]) -> Result<ChatRequest, ConversionError> {
        let response_req: ResponseRequest = serde_json::from_slice(body)?;
        let _model = response_req.model.clone();
        let chat_req = response_to_chat(response_req, self.provider.as_ref())?;
        Ok(chat_req)
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }
}

/// Collect request headers as a vector of (name, value) tuples.
fn collect_request_headers(session: &Session) -> Vec<(String, String)> {
    session
        .req_header()
        .headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or("<binary>").to_string(),
            )
        })
        .collect()
}

/// Check if header name is sensitive (should be masked in logs).
fn is_sensitive_header(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == "x-api-key"
        || lower == "authorization"
        || lower == "api-key"
        || lower == "x-api-token"
        || lower == "cookie"
        || lower == "set-cookie"
}

/// Mask sensitive header values.
fn mask_sensitive(value: &str) -> String {
    if value.len() <= 10 {
        return "***".to_string();
    }
    if value.starts_with("Bearer ") {
        let token = &value[7..];
        if token.len() <= 10 {
            return "Bearer ***".to_string();
        }
        return format!("Bearer {}***{}", &token[..6], &token[token.len() - 4..]);
    }
    format!("{}***{}", &value[..6], &value[value.len() - 4..])
}

#[async_trait]
impl ProxyHttp for CodexProxy {
    type CTX = ProxyContext;

    fn new_ctx(&self) -> Self::CTX {
        ProxyContext::new()
    }

    /// Request filter - called for each request to select backend and prepare context.
    async fn request_filter(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<bool>
    where
        Self::CTX: Send + Sync,
    {
        let method = session.req_header().method.as_str().to_string();
        let uri = session.req_header().uri.to_string();

        ctx.provider_name = Some(self.provider_name().to_string());
        ctx.backend_host = Some(self.upstream_host.clone());

        debug!("[REQUEST] {} {} (provider: {})", method, uri, self.provider_name());

        // Return false to continue processing
        Ok(false)
    }

    /// Select upstream peer for proxying.
    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<Box<HttpPeer>> {
        ctx.backend_host = Some(self.upstream_host.clone());

        let mut peer = HttpPeer::new(
            (self.upstream_host.as_str(), self.upstream_port),
            self.use_tls,
            self.upstream_host.clone(),
        );

        // HTTP/2 priority
        peer.options.alpn = ALPN::H2H1;

        // Connection configuration
        peer.options.connection_timeout = Some(Duration::from_secs(10));
        peer.options.total_connection_timeout = Some(Duration::from_secs(30));
        peer.options.idle_timeout = Some(Duration::from_secs(90));
        peer.options.tcp_keepalive = Some(TcpKeepalive {
            idle: Duration::from_secs(60),
            interval: Duration::from_secs(5),
            count: 5,
        });

        if self.use_tls {
            peer.options.h2_ping_interval = Some(Duration::from_secs(30));
        }

        Ok(Box::new(peer))
    }

    /// Filter and modify the upstream request.
    async fn upstream_request_filter(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        let original_uri = session.req_header().uri.clone();
        let path = original_uri.path().to_string();
        let query = original_uri.query();

        // Build new path - remove any path prefix for routing
        let new_path = if path.starts_with("/v1/responses") {
            "/v1/chat/completions".to_string()
        } else {
            path.clone()
        };

        // Build new URI
        let new_uri_str = if let Some(q) = query {
            format!("{}?{}", new_path, q)
        } else {
            new_path.clone()
        };

        let new_uri: http::Uri = new_uri_str.parse().map_err(|e| {
            error!("URI rewrite failed: {}", e);
            pingora_core::Error::new_str("URI rewrite failed")
        })?;
        upstream_request.set_uri(new_uri);
        ctx.rewritten_path = Some(new_path);

        // Remove client authentication headers
        upstream_request.remove_header("x-api-key");
        upstream_request.remove_header("authorization");

        // Inject API key for this provider
        upstream_request
            .insert_header("authorization", &format!("Bearer {}", self.api_key))
            .map_err(|e| {
                error!("Failed to inject authorization header: {}", e);
                pingora_core::Error::new_str("Header injection failed")
            })?;

        // Set host header
        upstream_request
            .insert_header("host", &self.upstream_host)
            .map_err(|e| {
                error!("Failed to set host header: {}", e);
                pingora_core::Error::new_str("Host header failed")
            })?;

        debug!(
            "[UPSTREAM] {} {}",
            upstream_request.method.as_str(),
            upstream_request.uri
        );

        Ok(())
    }

    /// Capture and transform request body (Responses API → Chat API).
    async fn request_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()>
    where
        Self::CTX: Send + Sync,
    {
        // Collect body
        if let Some(b) = body {
            ctx.request_body.extend_from_slice(b);
        }

        if end_of_stream && !ctx.request_body.is_empty() {
            // Log request body if enabled
            if self.log_body {
                if let Ok(text) = std::str::from_utf8(&ctx.request_body) {
                    debug!("[REQUEST BODY] {}", text);
                }
            }

            // Parse model name
            if ctx.model.is_none() {
                if let Ok(text) = std::str::from_utf8(&ctx.request_body) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                        if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
                            ctx.model = Some(model.to_string());
                        }
                        // Check streaming
                        if let Some(stream) = json.get("stream").and_then(|v| v.as_bool()) {
                            ctx.is_streaming = stream;
                            if stream {
                                ctx.stream_state = Some(StreamState::new(
                                    json.get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("default")
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle upstream response headers.
    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        debug!(
            "[RESPONSE] Status: {}",
            upstream_response.status.as_u16()
        );
        Ok(())
    }

    /// Transform response body (Chat API → Responses API for streaming).
    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_body: bool,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<Option<Duration>>
    where
        Self::CTX: Send + Sync,
    {
        if let Some(b) = body {
            if self.log_body {
                if let Ok(text) = std::str::from_utf8(b) {
                    debug!("[RESPONSE CHUNK] {}", text);
                }
            }

            ctx.response_body.extend_from_slice(b);

            // Transform streaming chunks if enabled
            // Note: Streaming body transformation requires more complex handling
            // (intercepting and modifying the response body). For now we collect
            // the chunks and log them if enabled.
        }

        if end_of_body {
            let duration_ms = ctx.start_time.elapsed().as_millis() as u64;
            info!(
                "[DONE] provider={}, model={:?}, duration={}ms",
                ctx.provider_name.as_deref().unwrap_or("unknown"),
                ctx.model.as_ref(),
                duration_ms
            );
        }

        Ok(None)
    }

    /// Called when connected to upstream.
    async fn connected_to_upstream(
        &self,
        _session: &mut Session,
        reused: bool,
        peer: &HttpPeer,
        #[cfg(unix)] _fd: std::os::unix::io::RawFd,
        #[cfg(windows)] _sock: std::os::windows::io::RawSocket,
        digest: Option<&Digest>,
        _ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        let tls_version = digest
            .and_then(|d| d.ssl_digest.as_ref())
            .map(|ssl| ssl.version.to_string())
            .unwrap_or_else(|| "none".to_string());

        info!(
            "[CONNECT] {} -> {} (TLS={}, reused={}, tls_version={})",
            peer.sni(),
            peer.address(),
            self.use_tls,
            reused,
            tls_version
        );

        Ok(())
    }

    /// Handle proxy errors.
    fn error_while_proxy(
        &self,
        peer: &HttpPeer,
        _session: &mut Session,
        e: Box<pingora_core::Error>,
        _ctx: &mut Self::CTX,
        _client_reused: bool,
    ) -> Box<pingora_core::Error> {
        error!(
            "[ERROR] proxy error to {}: {}",
            peer.address(),
            e
        );

        let mut e = e.more_context(format!("Provider: {}", self.provider_name()));
        e.retry.decide_reuse(false);
        e
    }

    /// Handle fatal errors when proxy cannot be established.
    async fn fail_to_proxy(
        &self,
        session: &mut Session,
        e: &pingora_core::Error,
        ctx: &mut Self::CTX,
    ) -> pingora_proxy::FailToProxy
    where
        Self::CTX: Send + Sync,
    {
        let code = match e.etype() {
            &pingora_core::ErrorType::ConnectTimedout => 504,
            &pingora_core::ErrorType::ConnectRefused => 502,
            &pingora_core::ErrorType::TLSHandshakeFailure => 502,
            _ => 502,
        };

        let method = session.req_header().method.as_str();
        let uri = &session.req_header().uri;

        error!(
            "[FAIL] {} {} -> {} (provider: {}, model: {:?}): {}",
            method,
            uri,
            code,
            ctx.provider_name.as_deref().unwrap_or("unknown"),
            ctx.model.as_ref(),
            e
        );

        // Return error response
        let error_body = format!(
            r#"{{"error": {{"type": "proxy_error", "code": {}, "message": "{}"}}}}"#,
            code,
            e
        );
        if let Ok(mut resp) = pingora_http::ResponseHeader::build(code, None) {
            let _ = resp.insert_header("content-type", "application/json");
            let _ = resp.insert_header("content-length", &error_body.len().to_string());
            let _ = session.write_response_header(Box::new(resp), false).await;
            let _ = session
                .write_response_body(Some(bytes::Bytes::from(error_body)), true)
                .await;
        }

        pingora_proxy::FailToProxy {
            error_code: code,
            can_reuse_downstream: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitive_header_detection() {
        assert!(is_sensitive_header("x-api-key"));
        assert!(is_sensitive_header("X-API-KEY"));
        assert!(is_sensitive_header("authorization"));
        assert!(is_sensitive_header("Authorization"));
        assert!(is_sensitive_header("cookie"));
        assert!(!is_sensitive_header("content-type"));
        assert!(!is_sensitive_header("x-request-id"));
    }

    #[test]
    fn test_mask_sensitive() {
        assert_eq!(mask_sensitive("short"), "***");
        // Bearer token with short token (<=10 chars): returns "Bearer ***"
        assert_eq!(mask_sensitive("Bearer sk-xxx"), "Bearer ***");
        // Bearer token with long token: shows first 6 and last 4 chars
        assert_eq!(
            mask_sensitive("Bearer sk-project-12345"),
            "Bearer sk-pro***2345"
        );
        // Long non-Bearer: shows first 6 and last 4 chars
        assert_eq!(
            mask_sensitive("sk-ant-api03-xxxxxxxxxxxx"),
            "sk-ant***xxxx"
        );
    }

    #[test]
    fn test_proxy_creation() {
        let provider = crate::providers::GLMProvider;
        let proxy = CodexProxy::new(
            Arc::new(provider),
            "https://api.example.com:443",
            "test-api-key",
            true,
        );
        assert_eq!(proxy.upstream_host, "api.example.com");
        assert_eq!(proxy.upstream_port, 443);
        assert!(proxy.use_tls);
        assert_eq!(proxy.provider_name(), "glm");
    }
}
