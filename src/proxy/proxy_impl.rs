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
use tracing::{debug, error, info, warn};

use crate::config::BackendRouter;
use crate::config::BackendInfo;
use crate::convert::StreamState;

/// Proxy context attached to each request session.
#[derive(Debug)]
pub struct ProxyContext {
    /// Request start time for duration tracking.
    pub start_time: Instant,
    /// Collected request body bytes.
    pub request_body: Vec<u8>,
    /// Model name parsed from request.
    pub model: Option<String>,
    /// Selected backend for this request.
    pub selected_backend: Option<BackendInfo>,
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
            selected_backend: None,
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
    /// Backend router for multi-backend routing.
    pub router: Arc<BackendRouter>,
    /// Whether to log request/response bodies.
    pub log_body: bool,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        router: Arc<BackendRouter>,
        log_body: bool,
    ) -> Self {
        Self {
            router,
            log_body,
        }
    }
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
        let path = session.req_header().uri.path().to_string();

        // Collect headers for routing
        let headers: Vec<(String, String)> = session
            .req_header()
            .headers
            .iter()
            .map(|(name, value)| {
                (
                    name.as_str().to_string(),
                    value.to_str().unwrap_or("<binary>").to_string(),
                )
            })
            .collect();

        // Select backend using router
        let backend = match self.router.select(&path, &headers) {
            Some(b) => b,
            None => {
                warn!("[REQUEST] No matching backend for path: {}", path);
                return Err(pingora_core::Error::new_str("No matching backend"));
            }
        };

        ctx.selected_backend = Some(backend.clone());
        ctx.provider_name = Some(backend.name.clone());

        debug!("[REQUEST] {} {} -> {}", method, path, backend.name);

        // Return false to continue processing
        Ok(false)
    }

    /// Select upstream peer for proxying.
    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<Box<HttpPeer>> {
        let backend = ctx.selected_backend.as_ref().ok_or_else(|| {
            error!("No backend selected");
            pingora_core::Error::new_str("No backend selected")
        })?;

        let mut peer = HttpPeer::new(
            (backend.host.as_str(), backend.port),
            backend.use_tls,
            backend.host.clone(),
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

        if backend.use_tls {
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
        let backend = ctx.selected_backend.as_ref().ok_or_else(|| {
            error!("No backend selected");
            pingora_core::Error::new_str("No backend selected")
        })?;

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

        // Inject API key for this backend
        upstream_request
            .insert_header("authorization", &format!("Bearer {}", backend.api_key))
            .map_err(|e| {
                error!("Failed to inject authorization header: {}", e);
                pingora_core::Error::new_str("Header injection failed")
            })?;

        // Set host header
        upstream_request
            .insert_header("host", &backend.host)
            .map_err(|e| {
                error!("Failed to set host header: {}", e);
                pingora_core::Error::new_str("Host header failed")
            })?;

        debug!(
            "[UPSTREAM] {} {} -> {}",
            upstream_request.method.as_str(),
            upstream_request.uri,
            backend.name
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
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        let tls_version = digest
            .and_then(|d| d.ssl_digest.as_ref())
            .map(|ssl| ssl.version.to_string())
            .unwrap_or_else(|| "none".to_string());

        let use_tls = ctx.selected_backend.as_ref().map(|b| b.use_tls).unwrap_or(false);
        let backend_name = ctx.provider_name.as_deref().unwrap_or("unknown");

        info!(
            "[CONNECT] {} -> {} (backend={}, TLS={}, reused={}, tls_version={})",
            peer.sni(),
            peer.address(),
            backend_name,
            use_tls,
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
        ctx: &mut Self::CTX,
        _client_reused: bool,
    ) -> Box<pingora_core::Error> {
        error!(
            "[ERROR] proxy error to {}: {}",
            peer.address(),
            e
        );

        let mut e = e.more_context(format!("Provider: {}", ctx.provider_name.as_deref().unwrap_or("unknown")));
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
    use crate::config::{BackendConfig, MatchRules};

    #[test]
    fn test_proxy_creation() {
        let configs = vec![BackendConfig {
            name: "glm".to_string(),
            url: "https://api.example.com".to_string(),
            api_key: "test-key".to_string(),
            protocol: "openai".to_string(),
            match_rules: MatchRules {
                default: true,
                ..Default::default()
            },
        }];
        let router = Arc::new(BackendRouter::new(configs).unwrap());
        let proxy = CodexProxy::new(router, true);
        assert!(proxy.log_body);
    }
}
