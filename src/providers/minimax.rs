//! MiniMax provider implementation.

use crate::providers::adapter::{
    ChatApiQuirks, OpenAiChatAdapter, ProtocolAdapter, ProviderConfig, ToolChoiceSupport,
    TokenLimitField,
};
use crate::providers::Provider;
use crate::types::chat_api::{ChatMessage, ChatRequest, Content, MessageRole};

/// MiniMax provider.
///
/// MiniMax specific handling:
/// - tool_choice: unsupported, must be removed
/// - token_limit: must use `max_completion_tokens`
/// - stream_options: unsupported
/// - parallel_tool_calls: unsupported
/// - tool_strict: unsupported
/// - Multiple system messages must be merged into one
/// - Reasoning: injects `{"reasoning_split": true}`
/// - Content arrays must be flattened to strings (request + response)
pub struct MiniMaxProvider {
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for MiniMaxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiChatAdapter,
            config: ProviderConfig {
                name: "minimax",
                endpoint_url: "https://api.minimax.chat/v1/",
                chat_path: "chat/completions",
                auth_header_name: "Authorization",
                auth_header_value_prefix: "Bearer ",
                protocol: crate::providers::adapter::ProtocolType::ChatApi,
                default_model: None,
                chat_quirks: ChatApiQuirks {
                    tool_choice_mode: ToolChoiceSupport::Unsupported,
                    token_limit_field: TokenLimitField::MaxCompletionTokens,
                    flatten_content: true,
                    merge_system_messages: true,
                    supports_tool_strict: false,
                    supports_stream_options: false,
                    supports_parallel_tool_calls: false,
                    reasoning_extension: Some(("reasoning_split", serde_json::json!(true))),
                    flatten_response: true,
                    ..ChatApiQuirks::default()
                },
            },
        }
    }
}

impl Provider for MiniMaxProvider {
    fn name(&self) -> &'static str {
        "minimax"
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &self.adapter
    }

    fn sanitize_request(&self, request: &mut ChatRequest) {
        merge_system_messages(request);
    }
}

fn merge_system_messages(request: &mut ChatRequest) {
    let mut system_texts = Vec::new();
    let mut non_system_messages = Vec::with_capacity(request.messages.len());

    for message in request.messages.drain(..) {
        if message.role == MessageRole::System {
            system_texts.push(message.content.as_text());
        } else {
            non_system_messages.push(message);
        }
    }

    if system_texts.is_empty() {
        request.messages = non_system_messages;
        return;
    }

    let mut messages = Vec::with_capacity(non_system_messages.len() + 1);
    messages.push(ChatMessage {
        role: MessageRole::System,
        content: Content::String(system_texts.join("\n\n")),
        name: None,
        annotations: None,
        tool_calls: None,
        function_call: None,
        tool_call_id: None,
        refusal: None,
    });
    messages.extend(non_system_messages);
    request.messages = messages;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatMessage, ChatRequest, Content, MessageRole};

    #[test]
    fn test_minimax_merges_multiple_system_messages() {
        let mut request = ChatRequest {
            model: "MiniMax-M2.7-highspeed".to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: Content::String("First instruction".to_string()),
                    name: None,
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                    refusal: None,
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: Content::String("Hello".to_string()),
                    name: None,
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                    refusal: None,
                },
                ChatMessage {
                    role: MessageRole::System,
                    content: Content::String("Second instruction".to_string()),
                    name: None,
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                    refusal: None,
                },
            ],
            tools: None,
            tool_choice: None,
            stream: Some(true),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            user: None,
            stream_options: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            stop: None,
            response_format: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            seed: None,
            service_tier: None,
            web_search_options: None,
            modalities: None,
            prediction: None,
            audio: None,
            extra: Default::default(),
        };

        MiniMaxProvider::new().sanitize_request(&mut request);

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(
            request.messages[0].content.as_text(),
            "First instruction\n\nSecond instruction"
        );
        assert_eq!(request.messages[1].role, MessageRole::User);
        assert_eq!(request.messages[1].content.as_text(), "Hello");
    }
}
