//! Response header/body conversion helpers.

use bytes::Bytes;
use pingora_core::Result as PingoraResult;
use pingora_http::ResponseHeader;
use tracing::{debug, error, info, warn};

use std::time::Duration;

use crate::constants::MAX_RESPONSE_BODY_SIZE;
use crate::proxy::context::ProxyContext;
use crate::proxy::context_store::ConversationSnapshot;
use crate::proxy::core::CodexProxy;
use crate::proxy::streaming_handler::StreamingResponseHandler;

impl CodexProxy {
    pub(crate) fn handle_response_filter(
        &self,
        upstream_response: &mut ResponseHeader,
        ctx: &mut ProxyContext,
    ) {
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
                status, content_type
            );
        }

        if ctx.flags.is_streaming
            && ctx.flags.is_conversion_request
            && !ctx.flags.should_convert_stream_response
        {
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
    }

    pub(crate) fn handle_response_body(
        &self,
        body: &mut Option<Bytes>,
        end_of_body: bool,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<Option<Duration>> {
        let body_clone = body.clone();

        debug!(
            "[RESPONSE_BODY] len={:?}, end={}, is_streaming={}, is_conversion={}",
            body_clone.as_ref().map(|b| b.len()),
            end_of_body,
            ctx.flags.is_streaming,
            ctx.flags.is_conversion_request
        );

        if let Some(b) = body_clone.as_ref() {
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

            if !ctx.flags.is_streaming && ctx.flags.is_conversion_request && !end_of_body {
                *body = Some(Bytes::new());
            }

            if ctx.flags.should_convert_stream_response {
                *body = Some(Bytes::new());

                let provider = ctx
                    .route
                    .provider_name
                    .as_ref()
                    .and_then(|name| self.get_provider(name));

                let adapter = provider.as_ref().map(|p| p.protocol_adapter());
                let config = provider.as_ref().map(|p| p.config());

                let mut handler =
                    StreamingResponseHandler::new(ctx, adapter, config, self.conversation_store.clone());

                if let Some(converted) = handler.process_stream_frame() {
                    *body = Some(Bytes::from(converted));
                }

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

            if !ctx.flags.is_streaming
                && ctx.flags.is_conversion_request
                && !ctx.buffers.response_body.is_empty()
            {
                self.convert_non_streaming_response_body(body, ctx)?;
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

    fn convert_non_streaming_response_body(
        &self,
        body: &mut Option<Bytes>,
        ctx: &mut ProxyContext,
    ) -> PingoraResult<()> {
        let text = std::str::from_utf8(&ctx.buffers.response_body).map_err(|e| {
            error!("[RESPONSE_BODY] upstream body not valid UTF-8: {}", e);
            pingora_core::Error::explain(
                pingora_core::ErrorType::HTTPStatus(502),
                format!("upstream response is not valid UTF-8: {e}"),
            )
        })?;
        let raw_body: serde_json::Value = serde_json::from_str(text).map_err(|e| {
            error!(
                "[RESPONSE_BODY] failed to parse upstream response body: {}",
                e
            );
            pingora_core::Error::explain(
                pingora_core::ErrorType::HTTPStatus(502),
                format!("upstream response not valid JSON: {e}"),
            )
        })?;
        let provider = ctx
            .route
            .provider_name
            .as_ref()
            .and_then(|name| self.get_provider(name));
        let chat_resp = match provider.as_ref() {
            Some(p) => {
                let adapter = p.protocol_adapter();
                adapter.parse_response(&raw_body, p.config()).map_err(|e| {
                    error!("[RESPONSE_BODY] adapter parse_response failed: {}", e);
                    pingora_core::Error::explain(
                        pingora_core::ErrorType::HTTPStatus(502),
                        format!("response parse failed: {e}"),
                    )
                })?
            }
            None => {
                serde_json::from_value(raw_body).map_err(|e| {
                    error!("[RESPONSE_BODY] failed to parse as ChatResponse: {}", e);
                    pingora_core::Error::explain(
                        pingora_core::ErrorType::HTTPStatus(502),
                        format!("upstream response not a valid Chat completion: {e}"),
                    )
                })?
            }
        };
        let assistant_message = chat_resp.choices.first().map(|c| c.message.clone());
        let response_obj =
            crate::convert::chat_to_response_with_context(
                chat_resp,
                ctx.response_request_context.as_ref(),
            )
            .map_err(|e| {
                error!("[RESPONSE_BODY] failed to convert response: {}", e);
                pingora_core::Error::explain(
                    pingora_core::ErrorType::HTTPStatus(500),
                    format!("proxy response conversion failed: {e}"),
                )
            })?;
        let converted = serde_json::to_vec(&response_obj).map_err(|e| {
            error!(
                "[RESPONSE_BODY] failed to serialize converted response: {}",
                e
            );
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
                ctx.route.provider_name.as_deref().unwrap_or("unknown"),
                response_obj.id.clone(),
                ConversationSnapshot {
                    instructions: ctx.follow_up.pending_instructions.clone(),
                    messages,
                },
            );
        }

        Ok(())
    }
}
