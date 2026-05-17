//! Pingora ProxyHttp trait implementation.

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use pingora_core::protocols::Digest;
use pingora_core::Result as PingoraResult;
use pingora_core::upstreams::peer::{HttpPeer, Peer};
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{ProxyHttp, Session};
use tracing::{debug, error};

use crate::constants::MAX_REQUEST_BODY_SIZE;
use crate::proxy::error_response::write_fail_response;

use super::context::ProxyContext;
use super::core::CodexProxy;

#[async_trait]
impl ProxyHttp for CodexProxy {
    type CTX = ProxyContext;

    fn new_ctx(&self) -> Self::CTX {
        ProxyContext::new()
    }

    async fn request_filter(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<bool>
    where
        Self::CTX: Send + Sync,
    {
        self.handle_request_filter(session, ctx).await
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<Box<HttpPeer>> {
        self.select_upstream_peer(ctx).await
    }

    async fn upstream_request_filter(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        self.rewrite_upstream_request(session, upstream_request, ctx)
            .await
    }

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
        if ctx.flags.is_conversion_request {
            if let Some(b) = body {
                if ctx.buffers.request_body.len() + b.len() > MAX_REQUEST_BODY_SIZE {
                    error!(
                        "[BODY] Request body exceeds maximum size limit of {} bytes",
                        MAX_REQUEST_BODY_SIZE
                    );
                    return Err(pingora_core::Error::new_str("Request body too large"));
                }
                ctx.buffers.request_body.extend_from_slice(b);
                debug!(
                    "[BODY] Buffered {} bytes (total: {})",
                    b.len(),
                    ctx.buffers.request_body.len()
                );
            }

            *body = Some(Bytes::new());

            if end_of_stream {
                debug!(
                    "[BODY] Conversion request complete, {} bytes buffered",
                    ctx.buffers.request_body.len()
                );

                match self.try_convert_request_body(ctx) {
                    Ok(converted) => {
                        if self.log_body {
                            debug!(
                                "[CONVERTED REQUEST] {}",
                                String::from_utf8_lossy(&converted)
                            );
                            let path = self
                                .log_dir
                                .join(format!("{}.converted_request.json", ctx.request_id));
                            if std::fs::write(&path, &converted).is_ok() {
                                debug!("[CONVERTED REQUEST SAVED] to {}", path.display());
                            }
                        }
                        debug!("[BODY] Sending converted body: {} bytes", converted.len());
                        *body = Some(Bytes::from(converted));
                    }
                    Err(e) => {
                        error!("[BODY] Conversion failed; aborting upstream: {}", e);
                        if self.log_body {
                            let path = self
                                .log_dir
                                .join(format!("{}.codex_request_body.json", ctx.request_id));
                            let _ = std::fs::write(&path, &ctx.buffers.request_body);
                        }
                        return Err(pingora_core::Error::explain(
                            pingora_core::ErrorType::HTTPStatus(400),
                            format!("proxy conversion failed: {e}"),
                        ));
                    }
                }
            }

            return Ok(());
        }

        if let Some(b) = body {
            ctx.buffers.request_body.extend_from_slice(b);
        }

        if end_of_stream
            && !ctx.buffers.request_body.is_empty()
            && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&ctx.buffers.request_body)
        {
            ctx.init_from_passthrough_json(&json);
        }

        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> PingoraResult<()> {
        self.handle_response_filter(upstream_response, ctx);
        Ok(())
    }

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
        self.handle_response_body(body, end_of_body, ctx)
    }

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
        self.log_connected_to_upstream(reused, peer, digest, ctx)
            .await
    }

    fn error_while_proxy(
        &self,
        peer: &HttpPeer,
        _session: &mut Session,
        e: Box<pingora_core::Error>,
        ctx: &mut Self::CTX,
        _client_reused: bool,
    ) -> Box<pingora_core::Error> {
        error!("[ERROR] proxy error to {}: {}", peer.address(), e);

        let mut e = e.more_context(format!(
            "Provider: {}",
            ctx.route.provider_name.as_deref().unwrap_or("unknown")
        ));
        e.retry.decide_reuse(false);
        e
    }

    async fn fail_to_proxy(
        &self,
        session: &mut Session,
        e: &pingora_core::Error,
        ctx: &mut Self::CTX,
    ) -> pingora_proxy::FailToProxy
    where
        Self::CTX: Send + Sync,
    {
        write_fail_response(session, e, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::config::{BackendConfig, MatchRules};
    use crate::providers::GLMProvider;
    use crate::providers::Provider;
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};

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
        providers.insert("glm".to_string(), Arc::new(GLMProvider::new()) as Arc<dyn Provider>);
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
        providers.insert("glm".to_string(), Arc::new(GLMProvider::new()) as Arc<dyn Provider>);
        let proxy = CodexProxy::new(router, providers, true, std::path::PathBuf::from("logs"));
        assert!(proxy.log_body);
    }

    #[test]
    fn test_try_convert_request_body_rejects_malformed_json() {
        let proxy = make_test_proxy();
        let mut ctx = crate::proxy::context::ProxyContext::new();
        ctx.flags.is_conversion_request = true;
        ctx.route.selected_backend = proxy
            .router
            .select_with_config("/", &[])
            .map(|(_, info)| info.clone());
        ctx.buffers.request_body = b"{not valid json".to_vec();
        let err = proxy
            .try_convert_request_body(&mut ctx)
            .expect_err("must fail on invalid JSON");
        assert!(matches!(err, crate::error::ConversionError::JsonError(_)));
    }

    #[test]
    fn test_try_convert_request_body_succeeds_on_minimal_request() {
        let proxy = make_test_proxy();
        let mut ctx = crate::proxy::context::ProxyContext::new();
        ctx.flags.is_conversion_request = true;
        ctx.route.selected_backend = proxy
            .router
            .select_with_config("/", &[])
            .map(|(_, info)| info.clone());
        ctx.buffers.request_body = br#"{"model":"glm-4","input":"hi"}"#.to_vec();
        let bytes = proxy
            .try_convert_request_body(&mut ctx)
            .expect("conversion should succeed");
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "glm-4");
        assert!(json["messages"].is_array());
        assert!(ctx.response_request_context.is_some());
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

        let merged = crate::proxy::request_body::merge_history_messages(history, current);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[3].content.as_text(), "next question");
    }

    #[test]
    fn test_merge_history_messages_when_no_overlap_appends_all_current() {
        let history = vec![msg(MessageRole::System, "system"), msg(MessageRole::Assistant, "a1")];
        let current = vec![msg(MessageRole::User, "new question")];

        let merged = crate::proxy::request_body::merge_history_messages(history, current);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[2].content.as_text(), "new question");
    }
}
