//! Request body conversion helpers.

use tracing::{debug, warn};

use crate::convert::{response_to_chat, ResponseRequestContext, ToolPriority};
use crate::error::ConversionError;
use crate::proxy::context::ProxyContext;
use crate::proxy::context_store::ConversationSnapshot;
use crate::proxy::core::CodexProxy;
use crate::types::chat_api::{ChatMessage, MessageRole};
use crate::types::response_api::ResponseRequest;

impl CodexProxy {
    /// Convert a buffered Responses-API request body to a Chat-API body.
    pub(crate) fn try_convert_request_body(
        &self,
        ctx: &mut ProxyContext,
    ) -> Result<Vec<u8>, ConversionError> {
        let backend = ctx.route.selected_backend.as_ref().ok_or_else(|| {
            ConversionError::ProviderError("no backend selected".to_string())
        })?.clone();
        let model_override = backend.model.clone();
        let provider = self.get_provider(&backend.name).ok_or_else(|| {
            ConversionError::ProviderError(format!(
                "no provider registered for backend '{}'",
                    backend.name
            ))
        })?;

        let mut response_req: ResponseRequest = serde_json::from_slice(&ctx.buffers.request_body)?;
        ctx.init_from_response_request(&response_req);

        let mut previous_messages: Option<Vec<ChatMessage>> = None;
        if let Some(prev_id) = response_req.previous_response_id.clone() {
            if let Some(snapshot) = self.get_conversation(&backend.name, &prev_id) {
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

        let adapter = provider.protocol_adapter();
        let body_value = adapter.build_request_body(&chat_req, provider.config())?;
        serde_json::to_vec(&body_value).map_err(ConversionError::from)
    }
}

pub(crate) fn merge_history_messages(
    mut history: Vec<ChatMessage>,
    current_turn_messages: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
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

#[allow(dead_code)]
fn _keep_snapshot_import(_: Option<ConversationSnapshot>) {}
