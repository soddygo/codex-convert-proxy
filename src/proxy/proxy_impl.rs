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
use crate::constants::*;
use crate::convert::{ResponseRequestContext, StreamState, response_to_chat};
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
    /// Whether this is a conversion request (Responses API -> Chat API).
    pub is_conversion_request: bool,
    /// Offset in response_body that has been parsed (to avoid re-parsing events).
    pub stream_body_parsed_offset: usize,
    /// Request path after optional routing prefix stripping.
    pub normalized_path: Option<String>,
    /// Parsed original Responses request for protocol-aligned streaming events.
    pub response_request_context: Option<ResponseRequestContext>,
    /// Whether current upstream response should be converted as SSE stream.
    pub should_convert_stream_response: bool,
    /// Upstream status code captured in response_filter for diagnostics.
    pub upstream_status: Option<u16>,
    /// Upstream content-type captured in response_filter for diagnostics.
    pub upstream_content_type: Option<String>,
    /// Number of valid upstream chat stream chunks parsed.
    pub stream_chunks_parsed: usize,
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
            is_conversion_request: false,
            stream_body_parsed_offset: 0,
            normalized_path: None,
            response_request_context: None,
            should_convert_stream_response: false,
            upstream_status: None,
            upstream_content_type: None,
            stream_chunks_parsed: 0,
        }
    }

    /// Parse model name and stream flag from request body, initialize StreamState if streaming.
    fn init_from_request_body(&mut self) {
        if self.model.is_some() {
            return;
        }
        if let Ok(text) = std::str::from_utf8(&self.request_body) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
                    self.model = Some(model.to_string());
                }
                if let Some(stream) = json.get("stream").and_then(|v| v.as_bool()) {
                    self.is_streaming = stream;
                    if stream {
                        self.is_stream_response = true;
                        let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
                        self.stream_state = Some(StreamState::new(
                            format!("resp_{}", uuid::Uuid::new_v4()),
                            model,
                            self.response_request_context.clone(),
                        ));
                    }
                }
            }
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
    /// Directory for debug log files.
    pub log_dir: std::path::PathBuf,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        router: Arc<BackendRouter>,
        providers: HashMap<String, Box<dyn Provider + Send + Sync>>,
        log_body: bool,
        log_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            router,
            providers,
            log_body,
            log_dir,
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

        // Select backend with config to support path_prefix stripping.
        let (backend_config, backend) = match self.router.select_with_config(&path, &headers) {
            Some(pair) => pair,
            None => {
                warn!("[REQUEST] No matching backend for path: {}", path);
                return Err(pingora_core::Error::new_str("No matching backend"));
            }
        };

        let normalized_path = if let Some(prefix) = backend_config.match_rules.path_prefix.as_deref() {
            let stripped = path.strip_prefix(prefix).unwrap_or(path.as_str());
            if stripped.is_empty() {
                "/".to_string()
            } else if stripped.starts_with('/') {
                stripped.to_string()
            } else {
                format!("/{}", stripped)
            }
        } else {
            path.clone()
        };
        ctx.normalized_path = Some(normalized_path.clone());

        // Check if this is a conversion request (Responses API -> Chat API)
        let is_conversion = normalized_path.starts_with("/v1/responses") || normalized_path.starts_with("/responses");
        ctx.is_conversion_request = is_conversion;

        if is_conversion {
            debug!("[REQUEST] {} {} -> {} (CONVERSION)", method, normalized_path, "conversion");
        }

        ctx.selected_backend = Some(backend.clone());
        ctx.provider_name = Some(backend.name.clone());

        debug!("[REQUEST] {} {} -> {}", method, normalized_path, backend.name);

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
        let normalized_path = ctx.normalized_path.as_deref().unwrap_or(path.as_str());

        // Check if this is a conversion request (Responses API → Chat API)
        let is_conversion_request = normalized_path.starts_with("/v1/responses")
            || normalized_path.starts_with("/responses");

        // Get the provider's chat completions path (may differ per provider)
        // Handle Responses API paths (/v1/responses, /responses) and also Chat API paths (/v1/chat/completions)
        let chat_api_path = if normalized_path.starts_with("/v1/responses")
            || normalized_path.starts_with("/responses")
            || normalized_path.starts_with("/v1/chat/completions")
        {
            // Get provider and use its chat_completions_path
            if let Some(provider) = self.get_provider(&backend.name) {
                provider.chat_completions_path()
            } else {
                // Fallback to standard path
                "/v1/chat/completions".to_string()
            }
        } else {
            normalized_path.to_string()
        };

        // Prepend backend's base_path (e.g., /api/coding/paas/v4 for GLM)
        let new_path = if !backend.base_path.is_empty() {
            format!("{}{}", backend.base_path, chat_api_path)
        } else {
            chat_api_path
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
            .insert_header("authorization", format!("Bearer {}", backend.api_key))
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
    ///
    /// For conversion requests, we need to buffer ALL chunks and send the
    /// converted body at end_of_stream. This is because HTTP/2 DATA frames
    /// arrive incrementally and we can't convert partial bodies.
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
        // For conversion requests, buffer all chunks and suppress forwarding
        // until we have the complete body for conversion.
        if ctx.is_conversion_request {
            // Buffer the chunk with size limit check
            if let Some(b) = body {
                if ctx.request_body.len() + b.len() > MAX_REQUEST_BODY_SIZE {
                    error!("[BODY] Request body exceeds maximum size limit of {} bytes", MAX_REQUEST_BODY_SIZE);
                    return Err(pingora_core::Error::new_str("Request body too large"));
                }
                ctx.request_body.extend_from_slice(b);
                debug!("[BODY] Buffered {} bytes (total: {})", b.len(), ctx.request_body.len());
            }

            // For conversion requests, suppress forwarding by returning empty body
            // At end_of_stream, we'll send the converted body
            *body = Some(Bytes::new());

            // Only process when we have the complete body
            if end_of_stream {
                debug!("[BODY] Conversion request complete, {} bytes buffered", ctx.request_body.len());

                let backend_name = ctx.selected_backend.as_ref().map(|b| b.name.clone());

                // Parse and convert request
                if let (Some(b_name), Some(_backend)) = (backend_name.as_ref(), ctx.selected_backend.as_ref()) {
                    if let Some(mut provider) = self.get_provider(b_name) {
                        // Get model override from backend config
                        let model_override = ctx.selected_backend.as_ref().and_then(|b| b.model.as_deref());
                        // Parse as ResponseRequest
                        match serde_json::from_slice::<ResponseRequest>(&ctx.request_body) {
                            Ok(response_req) => {
                                ctx.response_request_context =
                                    Some(ResponseRequestContext::from(&response_req));
                                // Convert using provider
                                match response_to_chat(response_req, provider.as_mut(), model_override) {
                                    Ok(chat_req) => {
                                        // Serialize ChatRequest as JSON
                                        match serde_json::to_vec(&chat_req) {
                                            Ok(converted) => {
                                                let converted_len = converted.len();
                                                // Log conversion if enabled
                                                if self.log_body {
                                                    debug!("[CONVERTED REQUEST] {}", String::from_utf8_lossy(&converted));
                                                }
                                                // Save converted request for debugging to logs directory
                                                let path = self.log_dir.join("converted_request.json");
                                                if std::fs::write(&path, &converted).is_ok() {
                                                    debug!("[CONVERTED REQUEST SAVED] to {}", path.display());
                                                }
                                                // Replace body with converted request - this is what will be sent
                                                *body = Some(Bytes::from(converted));
                                                debug!("[BODY] Sending converted body: {} bytes", converted_len);
                                            }
                                            Err(e) => {
                                                error!("Failed to serialize ChatRequest: {}", e);
                                                // Restore original body to let upstream handle the error
                                                *body = Some(Bytes::from(ctx.request_body.clone()));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to convert request: {}", e);
                                        // Restore original body to let upstream handle the error
                                        *body = Some(Bytes::from(ctx.request_body.clone()));
                                    }
                                }
                            }
                            Err(e) => {
                                // If parsing fails, keep original body
                                debug!("Failed to parse as ResponseRequest, keeping original: {}", e);
                                // Restore original body to let upstream handle the error
                                *body = Some(Bytes::from(ctx.request_body.clone()));
                                // Save full request body to file for debugging
                                let path = self.log_dir.join("codex_request_body.json");
                                if std::fs::write(&path, &ctx.request_body).is_ok() {
                                    debug!("[REQUEST BODY SAVED] to {}", path.display());
                                }
                            }
                        }
                    }
                }

                // Parse model name from original request for logging
                ctx.init_from_request_body();
            }

            return Ok(());
        }

        // Non-conversion requests: pass through normally
        if let Some(b) = body {
            ctx.request_body.extend_from_slice(b);
        }

        // Only process when we have the complete body
        if end_of_stream && !ctx.request_body.is_empty() {
            // Parse model name from original request for logging
            ctx.init_from_request_body();
        }

        Ok(())
    }

    /// Handle upstream response headers.
    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        let status = upstream_response.status.as_u16();
        let content_type = upstream_response
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let is_sse = content_type.to_ascii_lowercase().contains("text/event-stream");
        let is_success = (200..300).contains(&status);

        ctx.upstream_status = Some(status);
        ctx.upstream_content_type = Some(content_type.clone());
        ctx.should_convert_stream_response =
            ctx.is_streaming && ctx.is_conversion_request && is_success && is_sse;

        if ctx.is_conversion_request {
            upstream_response.remove_header("content-length");
            debug!(
                "[RESPONSE] removed content-length for conversion response (status={}, content_type={})",
                status,
                content_type
            );
        }

        if ctx.is_streaming && ctx.is_conversion_request && !ctx.should_convert_stream_response {
            warn!(
                "[RESPONSE] bypass stream conversion: status={}, content_type='{}', reason={}",
                status,
                content_type,
                if !is_success {
                    "upstream_non_2xx"
                } else if !is_sse {
                    "upstream_not_sse"
                } else {
                    "unknown"
                }
            );
        }

        debug!(
            "[RESPONSE] status={}, is_streaming={}, is_conversion={}, should_convert_stream={}",
            status,
            ctx.is_streaming,
            ctx.is_conversion_request,
            ctx.should_convert_stream_response
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

        debug!(
            "[RESPONSE_BODY] len={:?}, end={}, is_streaming={}, is_conversion={}",
            body_clone.as_ref().map(|b| b.len()),
            end_of_body,
            ctx.is_streaming,
            ctx.is_conversion_request
        );

        if let Some(b) = body_clone.as_ref() {
            // Check response body size limit for non-streaming conversion only.
            // For streaming conversion we compact parsed bytes to avoid unbounded growth
            // and must not switch protocol mid-stream.
            if !ctx.is_streaming
                && ctx.is_conversion_request
                && ctx.response_body.len() + b.len() > MAX_RESPONSE_BODY_SIZE
            {
                warn!(
                    "[RESPONSE_BODY] Response body exceeds maximum size limit of {} bytes",
                    MAX_RESPONSE_BODY_SIZE
                );
            } else {
                ctx.response_body.extend_from_slice(b);
            }

            // Suppress intermediate chunks for non-streaming conversion requests
            // (the converted body will be sent at end_of_body)
            if !ctx.is_streaming && ctx.is_conversion_request && !end_of_body {
                *body = Some(Bytes::new());
            }

            // For streaming conversion responses, convert each SSE chunk
            if ctx.should_convert_stream_response {
                // Suppress raw Chat API body immediately - it will be replaced
                // with converted Responses API events below (or empty if no events)
                *body = Some(Bytes::new());

                // Use accumulated body for SSE parsing (events may span multiple frames)
                // Only parse from the last parsed offset to avoid re-processing events
                let text = std::str::from_utf8(&ctx.response_body).unwrap_or("");
                let unparsed_text = &text[ctx.stream_body_parsed_offset..];
                debug!("[STREAM_RAW] is_streaming=true, body={}", String::from_utf8_lossy(&ctx.response_body).chars().take(200).collect::<String>());
                let mut converted_chunks: Vec<String> = Vec::new();

                // Use SSE utility module to parse only new events
                let (events, parse_end_pos) = crate::util::parse_sse(unparsed_text);
                let new_events_count = events.len();
                debug!("[STREAM_PARSE] Found {} new SSE events (offset={}, parse_end={})", new_events_count, ctx.stream_body_parsed_offset, parse_end_pos);

                for event in events {
                    // Skip [DONE] marker events - they don't contain JSON
                    if event.data == "[DONE]" {
                        continue;
                    }

                    // Parse as ChatStreamChunk
                    match serde_json::from_str::<crate::types::chat_api::ChatStreamChunk>(&event.data) {
                        Ok(chunk) => {
                            ctx.stream_chunks_parsed += 1;
                            let mut chunk = chunk;

                            // Apply provider transformation
                            if let Some(b_name) = ctx.provider_name.as_ref() {
                                if let Some(mut provider) = self.get_provider(b_name) {
                                    provider.transform_stream_chunk(&mut chunk);
                                }
                            }

                            // Convert to Response API events
                            if let Some(ref mut state) = ctx.stream_state {
                                // Update usage from this chunk
                                state.update_usage(&chunk);

                                match crate::convert::chat_chunk_to_response_events(&chunk, state) {
                                    Ok(events) => {
                                        let sse_data: String = events
                                            .iter()
                                            .map(crate::convert::event_to_sse)
                                            .collect();
                                        if !sse_data.is_empty() {
                                            debug!("[STREAM_CHUNK] {}", sse_data);
                                            converted_chunks.push(sse_data);
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to convert stream chunk: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            debug!("[STREAM_PARSE] Failed to parse JSON: {}", e);
                        }
                    }
                }

                // Update the parse offset to avoid re-parsing on next frame
                // Use parse_end_pos (relative to unparsed_text) to calculate absolute position
                if new_events_count > 0 {
                    ctx.stream_body_parsed_offset += parse_end_pos;
                }

                // Compact parsed prefix periodically to keep streaming memory bounded.
                if ctx.stream_body_parsed_offset >= STREAM_PARSE_COMPACT_THRESHOLD {
                    ctx.response_body.drain(..ctx.stream_body_parsed_offset);
                    debug!(
                        "[STREAM_PARSE] compacted parsed prefix bytes={}",
                        ctx.stream_body_parsed_offset
                    );
                    ctx.stream_body_parsed_offset = 0;
                }

                // For streaming responses, append response.completed event at end_of_body
                if end_of_body {
                    if let Some(ref mut state) = ctx.stream_state {
                        if !state.is_completed {
                            if ctx.stream_chunks_parsed == 0 {
                                warn!(
                                    "[STREAM_COMPLETE_SKIP] skip response.completed because no valid upstream chunks were parsed (status={:?}, content_type={:?})",
                                    ctx.upstream_status,
                                    ctx.upstream_content_type
                                );
                                state.is_completed = true;
                            } else {
                            let response_obj = state.build_response_object();
                            debug!(
                                "[STREAM_COMPLETE] response_id={}, output_count={}, has_reasoning={}, has_text={}, tool_calls={}, parsed_chunks={}",
                                response_obj.id,
                                response_obj.output.len(),
                                state.is_reasoning_added,
                                state.is_output_item_added,
                                state.completed_tool_calls.len(),
                                ctx.stream_chunks_parsed
                            );
                            if self.log_body {
                                if let Ok(json) = serde_json::to_string(&response_obj) {
                                    debug!("[STREAM_COMPLETE_JSON] {}", json);
                                }
                            }
                            let completed_event = crate::convert::ResponseStreamEvent::Completed {
                                response: response_obj,
                            };
                            let sse_data = crate::convert::event_to_sse(&completed_event);
                            converted_chunks.push(sse_data);
                            // Append SSE [DONE] marker to signal stream end
                            converted_chunks.push("data: [DONE]\n\n".to_string());
                            state.is_completed = true;
                            }
                        }
                    }
                }

                if !converted_chunks.is_empty() {
                    *body = Some(Bytes::from(converted_chunks.join("")));
                }
                // If converted_chunks is empty, *body remains as Bytes::new()
                // (set at the top of this block), suppressing the raw Chat API body.
            }
        }

        if end_of_body {
            let duration_ms = ctx.start_time.elapsed().as_millis() as u64;

            // For non-streaming conversion responses, convert the full body
            if !ctx.is_streaming && ctx.is_conversion_request && !ctx.response_body.is_empty() {
                if let Ok(text) = std::str::from_utf8(&ctx.response_body) {
                    if let Ok(chat_resp) = serde_json::from_str::<crate::types::chat_api::ChatResponse>(text) {
                        match crate::convert::chat_to_response_with_context(
                            chat_resp,
                            ctx.response_request_context.as_ref(),
                        ) {
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
        let code = match *e.etype() {
            pingora_core::ErrorType::ConnectTimedout => 504,
            pingora_core::ErrorType::ConnectRefused => 502,
            pingora_core::ErrorType::TLSHandshakeFailure => 502,
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

        // Return error response (use serde_json for proper escaping)
        let error_body = serde_json::json!({
            "error": {
                "type": "proxy_error",
                "code": code,
                "message": e.to_string()
            }
        })
        .to_string();
        if let Ok(mut resp) = pingora_http::ResponseHeader::build(code, None) {
            let _ = resp.insert_header("content-type", "application/json");
            let _ = resp.insert_header("content-length", error_body.len().to_string());
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
            model: None,
            match_rules: MatchRules {
                default: true,
                ..Default::default()
            },
        }];
        let router = Arc::new(BackendRouter::new(configs).unwrap());
        let mut providers = HashMap::new();
        providers.insert("glm".to_string(), Box::new(GLMProvider) as Box<dyn Provider + Send + Sync>);
        let proxy = CodexProxy::new(router, providers, true, std::path::PathBuf::from("logs"));
        assert!(proxy.log_body);
    }
}
