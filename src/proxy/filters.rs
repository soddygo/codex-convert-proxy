//! Pingora ProxyHttp trait implementation.

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pingora_core::protocols::Digest;
use pingora_core::protocols::TcpKeepalive;
use pingora_core::Result as PingoraResult;
use pingora_core::upstreams::peer::{ALPN, HttpPeer, Peer};
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{ProxyHttp, Session};
use tracing::{debug, error, info, warn};

use crate::constants::*;
use crate::convert::{ResponseRequestContext, response_to_chat, ToolPriority};
use crate::proxy::context_store::ConversationSnapshot;
use crate::types::chat_api::{ChatMessage, MessageRole};
use crate::types::response_api::ResponseRequest;

use super::context::ProxyContext;
use super::core::CodexProxy;
use crate::proxy::streaming_handler::StreamingResponseHandler;

impl CodexProxy {
    /// Convert a buffered Responses-API request body to a Chat-API body.
    ///
    /// Returns `Err(ConversionError)` if any of:
    /// - no provider is configured for the selected backend
    /// - body cannot be parsed as `ResponseRequest`
    /// - protocol conversion fails
    /// - the converted `ChatRequest` cannot be serialised
    ///
    /// Callers should propagate the error to the client as a 4xx response
    /// (Fail-Fast — earlier behaviour silently passed the original
    /// Responses-API body to a Chat endpoint and let the upstream return 400,
    /// which obscured proxy bugs).
    fn try_convert_request_body(
        &self,
        ctx: &mut ProxyContext,
    ) -> Result<Vec<u8>, crate::error::ConversionError> {
        use crate::error::ConversionError;

        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
            ConversionError::ProviderError("no backend selected".to_string())
        })?;
        let model_override = backend.model.clone();
        let provider = self.get_provider(&backend.name).ok_or_else(|| {
            ConversionError::ProviderError(format!(
                "no provider registered for backend '{}'",
                backend.name
            ))
        })?;

        let mut response_req: ResponseRequest =
            serde_json::from_slice(&ctx.buffers.request_body)?;
        ctx.init_from_response_request(&response_req);

        // Resolve previous-turn history if requested.
        let mut previous_messages: Option<Vec<ChatMessage>> = None;
        if let Some(prev_id) = response_req.previous_response_id.clone() {
            if let Some(snapshot) = self.get_conversation(&prev_id) {
                if matches!(
                    &response_req.input,
                    crate::types::response_api::InputItemOrString::Array(_)
                ) {
                    debug!(
                        "[REQUEST_CONVERT] previous_response_id + input[] detected, applying prefer-previous merge policy"
                    );
                }
                if response_req.instructions.is_none() {
                    response_req.instructions = snapshot.instructions.clone();
                }
                previous_messages = Some(snapshot.messages);
            } else {
                warn!(
                    "[REQUEST_CONVERT] previous_response_id not found in context store: {}",
                    prev_id
                );
            }
        }

        let context = ResponseRequestContext::from(&response_req);
        ctx.set_response_request_context(context);

        let mut chat_req = response_to_chat(
            response_req,
            provider.as_ref(),
            model_override.as_deref(),
            ToolPriority::Merge,
        )?;

        if let Some(history) = previous_messages {
            chat_req.messages = merge_history_messages(history, chat_req.messages);
        }

        ctx.follow_up.pending_instructions = chat_req
            .messages
            .iter()
            .find(|m| m.role == MessageRole::System)
            .map(|m| m.content.as_text());
        ctx.follow_up.pending_conversation_messages = Some(chat_req.messages.clone());

        serde_json::to_vec(&chat_req).map_err(ConversionError::from)
    }
}

fn merge_history_messages(
    mut history: Vec<ChatMessage>,
    current_turn_messages: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
    // prefer-previous strategy:
    // history from previous_response_id is authoritative; only append incremental suffix.
    let mut overlap = 0usize;
    while overlap < history.len() && overlap < current_turn_messages.len() {
        let same = serde_json::to_value(&history[overlap]).ok()
            == serde_json::to_value(&current_turn_messages[overlap]).ok();
        if !same {
            break;
        }
        overlap += 1;
    }

    if overlap > 0 {
        debug!(
            "[REQUEST_CONVERT] detected {} overlapping history messages, appending incremental suffix only",
            overlap
        );
    } else if !current_turn_messages.is_empty() {
        debug!(
            "[REQUEST_CONVERT] no overlap with cached history, appending all current messages as incremental"
        );
    }

    history.extend(current_turn_messages.into_iter().skip(overlap));
    history
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
        let query = session.req_header().uri.query().unwrap_or("");

        debug!("[REQUEST_FILTER] ★★★ RECEIVED REQUEST ★★★");
        debug!("[REQUEST_FILTER] Method: {}, Path: {}, Query: {}", method, path, query);

        // Log ALL headers for debugging
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

        for (name, value) in &headers {
            debug!("[REQUEST_FILTER]   Header: {}: {}", name, value);
        }

        debug!("[REQUEST_FILTER] ★★★ END REQUEST HEADER ★★★");

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
        ctx.route.normalized_path = Some(normalized_path.clone());

        // Check if this is a conversion request (Responses API -> Chat API)
        let is_conversion = (normalized_path.starts_with("/v1/responses") || normalized_path.starts_with("/responses")) && method == "POST";
        ctx.flags.is_conversion_request = is_conversion;

        if is_conversion {
            debug!("[REQUEST] {} {} -> {} (CONVERSION)", method, normalized_path, "conversion");
        }

        ctx.route.selected_backend = Some(backend.clone());
        ctx.route.provider_name = Some(backend.name.clone());

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
        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
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
        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
            error!("No backend selected");
            pingora_core::Error::new_str("No backend selected")
        })?;

        let original_uri = session.req_header().uri.clone();
        let path = original_uri.path().to_string();
        let query = original_uri.query();
        let normalized_path = ctx.route.normalized_path.as_deref().unwrap_or(path.as_str());

        // Check if this is a conversion request (Responses API → Chat API)
        let is_conversion_request = (normalized_path.starts_with("/v1/responses")
            || normalized_path.starts_with("/responses"))
            && upstream_request.method.as_str() == "POST";

        // Get the provider's chat completions path (may differ per provider)
        // Handle Responses API paths (/v1/responses, /responses) and also Chat API paths (/v1/chat/completions)
        let chat_api_path = if is_conversion_request || normalized_path.starts_with("/v1/chat/completions") {
            // Get provider and use its chat_completions_path
            if let Some(provider) = self.get_provider(&backend.name) {
                provider.chat_completions_path()
            } else {
                // Fallback to standard path (without /v1 prefix, which comes from base_path)
                "/chat/completions".to_string()
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
        ctx.route.rewritten_path = Some(new_path);

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
        if ctx.flags.is_conversion_request {
            // Buffer the chunk with size limit check
            if let Some(b) = body {
                if ctx.buffers.request_body.len() + b.len() > MAX_REQUEST_BODY_SIZE {
                    error!("[BODY] Request body exceeds maximum size limit of {} bytes", MAX_REQUEST_BODY_SIZE);
                    return Err(pingora_core::Error::new_str("Request body too large"));
                }
                ctx.buffers.request_body.extend_from_slice(b);
                debug!("[BODY] Buffered {} bytes (total: {})", b.len(), ctx.buffers.request_body.len());
            }

            // For conversion requests, suppress forwarding by returning empty body
            // At end_of_stream, we'll send the converted body
            *body = Some(Bytes::new());

            // Only process when we have the complete body
            if end_of_stream {
                debug!("[BODY] Conversion request complete, {} bytes buffered", ctx.buffers.request_body.len());

                match self.try_convert_request_body(ctx) {
                    Ok(converted) => {
                        if self.log_body {
                            debug!(
                                "[CONVERTED REQUEST] {}",
                                String::from_utf8_lossy(&converted)
                            );
                        }
                        let path = self.log_dir.join("converted_request.json");
                        if std::fs::write(&path, &converted).is_ok() {
                            debug!("[CONVERTED REQUEST SAVED] to {}", path.display());
                        }
                        debug!("[BODY] Sending converted body: {} bytes", converted.len());
                        *body = Some(Bytes::from(converted));
                    }
                    Err(e) => {
                        error!("[BODY] Conversion failed; aborting upstream: {}", e);
                        let path = self.log_dir.join("codex_request_body.json");
                        let _ = std::fs::write(&path, &ctx.buffers.request_body);
                        return Err(pingora_core::Error::explain(
                            pingora_core::ErrorType::HTTPStatus(400),
                            format!("proxy conversion failed: {e}"),
                        ));
                    }
                }
            }

            return Ok(());
        }

        // Non-conversion requests: pass through normally
        if let Some(b) = body {
            ctx.buffers.request_body.extend_from_slice(b);
        }

        // Only process when we have the complete body
        if end_of_stream
            && !ctx.buffers.request_body.is_empty()
            && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&ctx.buffers.request_body)
        {
            // Non-conversion path: extract just model/stream for diagnostics
            ctx.init_from_passthrough_json(&json);
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

        ctx.diagnostics.upstream_status = Some(status);
        ctx.diagnostics.upstream_content_type = Some(content_type.clone());
        ctx.flags.should_convert_stream_response =
            ctx.flags.is_streaming && ctx.flags.is_conversion_request && is_success && is_sse;

        if ctx.flags.is_conversion_request {
            upstream_response.remove_header("content-length");
            debug!(
                "[RESPONSE] removed content-length for conversion response (status={}, content_type={})",
                status,
                content_type
            );
        }

        if ctx.flags.is_streaming && ctx.flags.is_conversion_request && !ctx.flags.should_convert_stream_response {
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
            ctx.flags.is_streaming,
            ctx.flags.is_conversion_request,
            ctx.flags.should_convert_stream_response
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
            ctx.flags.is_streaming,
            ctx.flags.is_conversion_request
        );

        if let Some(b) = body_clone.as_ref() {
            // Check response body size limit for non-streaming conversion only.
            // For streaming conversion we compact parsed bytes to avoid unbounded growth
            // and must not switch protocol mid-stream.
            if !ctx.flags.is_streaming
                && ctx.flags.is_conversion_request
                && ctx.buffers.response_body.len() + b.len() > MAX_RESPONSE_BODY_SIZE
            {
                warn!(
                    "[RESPONSE_BODY] Response body exceeds maximum size limit of {} bytes",
                    MAX_RESPONSE_BODY_SIZE
                );
            } else {
                ctx.buffers.response_body.extend_from_slice(b);
            }

            // Suppress intermediate chunks for non-streaming conversion requests
            // (the converted body will be sent at end_of_body)
            if !ctx.flags.is_streaming && ctx.flags.is_conversion_request && !end_of_body {
                *body = Some(Bytes::new());
            }

            // For streaming conversion responses, convert each SSE chunk
            if ctx.flags.should_convert_stream_response {
                // Suppress raw Chat API body immediately - it will be replaced
                // with converted Responses API events below (or empty if no events)
                *body = Some(Bytes::new());

                // Get provider clone before mutable borrow to avoid borrow conflict
                let provider = ctx.route.provider_name.as_ref()
                    .and_then(|name| self.get_provider(name));

                // Delegate to StreamingResponseHandler for chunk processing
                let mut handler = StreamingResponseHandler::new(
                    ctx,
                    provider,
                    self.conversation_store.clone(),
                );

                // Process streaming frames until end_of_body
                if let Some(converted) = handler.process_stream_frame() {
                    *body = Some(Bytes::from(converted));
                }

                // For streaming responses, append response.completed event at end_of_body
                if end_of_body {
                    let completed_events = handler.finalize_stream();
                    if !completed_events.is_empty() {
                        let existing = std::str::from_utf8(body.as_ref().unwrap_or(&Bytes::new()))
                            .unwrap_or("")
                            .to_string();
                        let combined = format!("{}{}", existing, completed_events.join(""));
                        *body = Some(Bytes::from(combined));
                    }
                }
            }
        }

        if end_of_body {
            let duration_ms = ctx.start_time.elapsed().as_millis() as u64;

            // For non-streaming conversion responses, convert the full body.
            // Fail-Fast: a parse/convert error here means the proxy cannot
            // honour its protocol contract, so surface an error rather than
            // silently delivering the raw Chat body (which the client expects
            // in Responses-API shape).
            if !ctx.flags.is_streaming && ctx.flags.is_conversion_request && !ctx.buffers.response_body.is_empty() {
                let text = std::str::from_utf8(&ctx.buffers.response_body).map_err(|e| {
                    error!("[RESPONSE_BODY] upstream body not valid UTF-8: {}", e);
                    pingora_core::Error::explain(
                        pingora_core::ErrorType::HTTPStatus(502),
                        format!("upstream response is not valid UTF-8: {e}"),
                    )
                })?;
                let chat_resp: crate::types::chat_api::ChatResponse =
                    serde_json::from_str(text).map_err(|e| {
                        error!("[RESPONSE_BODY] failed to parse upstream ChatResponse: {}", e);
                        pingora_core::Error::explain(
                            pingora_core::ErrorType::HTTPStatus(502),
                            format!("upstream response not a valid Chat completion: {e}"),
                        )
                    })?;
                let assistant_message = chat_resp.choices.first().map(|c| c.message.clone());
                let request_context = ctx
                    .stream_state
                    .as_ref()
                    .and_then(|s| s.request_context.as_ref());
                let response_obj =
                    crate::convert::chat_to_response_with_context(chat_resp, request_context)
                        .map_err(|e| {
                            error!("[RESPONSE_BODY] failed to convert response: {}", e);
                            pingora_core::Error::explain(
                                pingora_core::ErrorType::HTTPStatus(500),
                                format!("proxy response conversion failed: {e}"),
                            )
                        })?;
                let converted = serde_json::to_vec(&response_obj).map_err(|e| {
                    error!("[RESPONSE_BODY] failed to serialize converted response: {}", e);
                    pingora_core::Error::explain(
                        pingora_core::ErrorType::HTTPStatus(500),
                        format!("proxy response serialization failed: {e}"),
                    )
                })?;
                if self.log_body {
                    debug!(
                        "[CONVERTED RESPONSE] {}",
                        String::from_utf8_lossy(&converted)
                    );
                }
                *body = Some(Bytes::from(converted));
                if let (Some(mut messages), Some(assistant_message)) = (
                    ctx.follow_up.pending_conversation_messages.clone(),
                    assistant_message,
                ) {
                    messages.push(assistant_message);
                    self.store_conversation(
                        response_obj.id.clone(),
                        ConversationSnapshot {
                            instructions: ctx.follow_up.pending_instructions.clone(),
                            messages,
                        },
                    );
                }
            }

            info!(
                "[DONE] provider={}, model={:?}, duration={}ms",
                ctx.route.provider_name.as_deref().unwrap_or("unknown"),
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

        let use_tls = ctx.route.selected_backend.as_ref().map(|b| b.use_tls).unwrap_or(false);
        let backend_name = ctx.route.provider_name.as_deref().unwrap_or("unknown");

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

        let mut e = e.more_context(format!("Provider: {}", ctx.route.provider_name.as_deref().unwrap_or("unknown")));
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
            ctx.route.provider_name.as_deref().unwrap_or("unknown"),
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
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::config::{BackendConfig, MatchRules};
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};
    use crate::providers::GLMProvider;
    use crate::providers::Provider;

    use super::CodexProxy;

    fn make_test_proxy() -> CodexProxy {
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
        let router = Arc::new(crate::config::BackendRouter::new(configs).unwrap());
        let mut providers = HashMap::new();
        providers.insert("glm".to_string(), Arc::new(GLMProvider) as Arc<dyn Provider>);
        CodexProxy::new(router, providers, false, std::path::PathBuf::from("logs"))
    }

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
        let router = Arc::new(crate::config::BackendRouter::new(configs).unwrap());
        let mut providers = HashMap::new();
        providers.insert("glm".to_string(), Arc::new(GLMProvider) as Arc<dyn Provider>);
        let proxy = CodexProxy::new(router, providers, true, std::path::PathBuf::from("logs"));
        assert!(proxy.log_body);
    }

    #[test]
    fn test_try_convert_request_body_rejects_malformed_json() {
        // Fail-Fast: a body that doesn't deserialize as ResponseRequest must
        // produce a ConversionError rather than silently being passed upstream.
        let proxy = make_test_proxy();
        let mut ctx = super::ProxyContext::new();
        ctx.flags.is_conversion_request = true;
        ctx.route.selected_backend = proxy.router.select_with_config("/", &[]).map(|(_, info)| info.clone());
        ctx.buffers.request_body = b"{not valid json".to_vec();
        let err = proxy
            .try_convert_request_body(&mut ctx)
            .expect_err("must fail on invalid JSON");
        assert!(matches!(err, crate::error::ConversionError::JsonError(_)));
    }

    #[test]
    fn test_try_convert_request_body_succeeds_on_minimal_request() {
        let proxy = make_test_proxy();
        let mut ctx = super::ProxyContext::new();
        ctx.flags.is_conversion_request = true;
        ctx.route.selected_backend = proxy.router.select_with_config("/", &[]).map(|(_, info)| info.clone());
        ctx.buffers.request_body = br#"{"model":"glm-4","input":"hi"}"#.to_vec();
        let bytes = proxy
            .try_convert_request_body(&mut ctx)
            .expect("conversion should succeed");
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "glm-4");
        assert!(json["messages"].is_array());
    }

    fn msg(role: MessageRole, text: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: Content::String(text.to_string()),
            name: None,
            annotations: None,
            tool_calls: None,
            function_call: None,
            tool_call_id: None,
            refusal: None,
        }
    }

    #[test]
    fn test_merge_history_messages_prefers_previous_and_appends_incremental() {
        let history = vec![
            msg(MessageRole::System, "You are helpful"),
            msg(MessageRole::User, "hello"),
            msg(MessageRole::Assistant, "hi"),
        ];
        let current = vec![
            msg(MessageRole::System, "You are helpful"),
            msg(MessageRole::User, "hello"),
            msg(MessageRole::Assistant, "hi"),
            msg(MessageRole::User, "next question"),
        ];

        let merged = super::merge_history_messages(history, current);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[3].content.as_text(), "next question");
    }

    #[test]
    fn test_merge_history_messages_when_no_overlap_appends_all_current() {
        let history = vec![msg(MessageRole::System, "system"), msg(MessageRole::Assistant, "a1")];
        let current = vec![msg(MessageRole::User, "new question")];

        let merged = super::merge_history_messages(history, current);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[2].content.as_text(), "new question");
    }
}
