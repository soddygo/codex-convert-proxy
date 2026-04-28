//! Pingora ProxyHttp implementation for Codex Convert Proxy.
//!
//! This module provides full ProxyHttp trait implementation for HTTP proxying
//! with request/response format conversion between Responses API and Chat API.

use std::collections::HashMap;
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
use crate::convert::{StreamState, response_to_chat};
use crate::providers::Provider;
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
    /// Whether this is a streaming response (for conversion tracking).
    pub is_stream_response: bool,
    /// Collected Chat API response for conversion at end of stream.
    pub chat_response_body: Vec<u8>,
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
            is_stream_response: false,
            chat_response_body: Vec::new(),
        }
    }
}

/// Codex proxy handler implementing ProxyHttp trait.
pub struct CodexProxy {
    /// Backend router for multi-backend routing.
    pub router: Arc<BackendRouter>,
    /// Providers for each backend.
    pub providers: HashMap<String, Box<dyn Provider + Send + Sync>>,
    /// Whether to log request/response bodies.
    pub log_body: bool,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        router: Arc<BackendRouter>,
        providers: HashMap<String, Box<dyn Provider + Send + Sync>>,
        log_body: bool,
    ) -> Self {
        Self {
            router,
            providers,
            log_body,
        }
    }

    /// Get cloned provider for a backend.
    fn get_provider(&self, name: &str) -> Option<Box<dyn Provider + Send + Sync>> {
        self.providers.get(name).map(|p| p.clone_box())
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

        // Check if this is a conversion request (Responses API → Chat API)
        let is_conversion_request = path.starts_with("/v1/responses") || path.starts_with("/responses");

        // Build new path - Responses API → Chat API conversion
        // Handle both /v1/responses and /responses paths
        let new_path = if path.starts_with("/v1/responses") {
            "/v1/chat/completions".to_string()
        } else if path.starts_with("/responses") {
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

        // For conversion requests, remove content-length header since body size will change
        // The upstream will use HTTP/2 DATA frame lengths instead of content-length validation
        if is_conversion_request {
            upstream_request.remove_header("content-length");
            debug!("[UPSTREAM] Removed content-length for body transformation");
        }

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
        // Collect body bytes
        if let Some(b) = body {
            ctx.request_body.extend_from_slice(b);
        }

        // Only process when we have the complete body
        if end_of_stream && !ctx.request_body.is_empty() {
            let backend_name = ctx.selected_backend.as_ref().map(|b| b.name.clone());

            // Parse and convert request
            if let (Some(b_name), Some(_backend)) = (backend_name.as_ref(), ctx.selected_backend.as_ref()) {
                if let Some(mut provider) = self.get_provider(b_name) {
                    // Parse as ResponseRequest
                    match serde_json::from_slice::<ResponseRequest>(&ctx.request_body) {
                        Ok(response_req) => {
                            // Convert using provider
                            match response_to_chat(response_req, provider.as_mut()) {
                                Ok(chat_req) => {
                                    // Serialize ChatRequest as JSON
                                    match serde_json::to_vec(&chat_req) {
                                        Ok(converted) => {
                                            // Log conversion if enabled
                                            if self.log_body {
                                                debug!("[CONVERTED REQUEST] {}", String::from_utf8_lossy(&converted));
                                            }
                                            // Save converted request for debugging to logs directory
                                            let log_dir = std::path::Path::new("logs");
                                            let path = log_dir.join("converted_request.json");
                                            if std::fs::write(&path, &converted).is_ok() {
                                                debug!("[CONVERTED REQUEST SAVED] to {}", path.display());
                                            }
                                            // Replace body with converted request
                                            *body = Some(Bytes::from(converted));
                                        }
                                        Err(e) => {
                                            error!("Failed to serialize ChatRequest: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to convert request: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            // If parsing fails, keep original body
                            debug!("Failed to parse as ResponseRequest, keeping original: {}", e);
                            // Save full request body to file for debugging
                            let path = "logs/codex_request_body.json";
                            if let Ok(_) = std::fs::write(path, &ctx.request_body) {
                                debug!("[REQUEST BODY SAVED] to {}", path);
                            }
                        }
                    }
                }
            }

            // Parse model name from original request for logging
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
                                ctx.is_stream_response = true;
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
        // Clone the body bytes for processing
        let body_clone = body.clone();

        if let Some(b) = body_clone.as_ref() {
            ctx.response_body.extend_from_slice(b);

            // For streaming responses, convert each SSE chunk
            if ctx.is_streaming {
                let text = std::str::from_utf8(b).unwrap_or("");
                let mut converted_chunks: Vec<String> = Vec::new();

                for line in text.lines() {
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            // Stream ended marker - emit done event
                            converted_chunks.push("data: [DONE]\n\n".to_string());
                            continue;
                        }

                        // Parse as ChatStreamChunk
                        if let Ok(chunk) = serde_json::from_str::<crate::types::chat_api::ChatStreamChunk>(data) {
                            let mut chunk = chunk;

                            // Apply provider transformation if available
                            if let Some(b_name) = ctx.provider_name.as_ref() {
                                if let Some(mut provider) = self.get_provider(b_name) {
                                    provider.transform_stream_chunk(&mut chunk);
                                }
                            }

                            // Convert to Response API events
                            if let Some(ref mut state) = ctx.stream_state {
                                match crate::convert::chat_chunk_to_response_events(&chunk, state) {
                                    Ok(events) => {
                                        let sse_data: String = events
                                            .iter()
                                            .map(crate::convert::event_to_sse)
                                            .collect();
                                        if !sse_data.is_empty() {
                                            if self.log_body {
                                                debug!("[STREAM CHUNK] {}", sse_data);
                                            }
                                            converted_chunks.push(sse_data);
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to convert stream chunk: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }

                if !converted_chunks.is_empty() {
                    *body = Some(Bytes::from(converted_chunks.join("")));
                }
            }
        }

        if end_of_body {
            let duration_ms = ctx.start_time.elapsed().as_millis() as u64;

            // For non-streaming responses, convert the full body
            if !ctx.is_streaming && !ctx.response_body.is_empty() {
                if let Ok(text) = std::str::from_utf8(&ctx.response_body) {
                    if let Ok(chat_resp) = serde_json::from_str::<crate::types::chat_api::ChatResponse>(text) {
                        match crate::convert::chat_to_response(chat_resp) {
                            Ok(response_obj) => {
                                if let Ok(converted) = serde_json::to_vec(&response_obj) {
                                    if self.log_body {
                                        debug!("[CONVERTED RESPONSE] {}", String::from_utf8_lossy(&converted));
                                    }
                                    *body = Some(Bytes::from(converted));
                                }
                            }
                            Err(e) => {
                                error!("Failed to convert response: {}", e);
                            }
                        }
                    }
                }
            }

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
    use crate::providers::GLMProvider;
    use crate::providers::Provider;

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
        let mut providers = HashMap::new();
        providers.insert("glm".to_string(), Box::new(GLMProvider) as Box<dyn Provider + Send + Sync>);
        let proxy = CodexProxy::new(router, providers, true);
        assert!(proxy.log_body);
    }
}
