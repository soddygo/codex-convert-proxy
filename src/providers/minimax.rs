//! MiniMax provider implementation.

use crate::convert::ResponseRequestContext;
use crate::error::ConversionError;
use crate::providers::{
    Provider, ProviderCapabilities, ProviderExtensions, TokenLimitField, ToolChoiceSupport,
};
use crate::types::chat_api::{ChatResponse, ChatStreamChunk, Content};

/// MiniMax provider.
///
/// MiniMax specific handling:
/// - OpenAI-compatible endpoint supports tools
/// - Does not support 'developer' role, convert to 'system'
pub struct MiniMaxProvider;

impl Default for MiniMaxProvider {
    fn default() -> Self {
        Self
    }
}

impl MiniMaxProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for MiniMaxProvider {
    fn name(&self) -> &'static str {
        "minimax"
    }

    fn chat_completions_path(&self) -> String {
        // MiniMax base_path already includes version prefix (/v1),
        // so we only need the endpoint suffix.
        "/chat/completions".to_string()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            tool_choice: ToolChoiceSupport::AutoAndNone,
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            supports_developer_role: false,
            flatten_request_content: false,
        }
    }

    fn provider_extensions(
        &self,
        context: &ResponseRequestContext,
    ) -> Result<ProviderExtensions, ConversionError> {
        let mut extensions = ProviderExtensions::new();
        if context.reasoning.is_some() {
            extensions.insert("reasoning_split", serde_json::json!(true))?;
        }
        Ok(extensions)
    }

    fn transform_response(&self, response: &mut ChatResponse) {
        // Ensure content is string
        for choice in &mut response.choices {
            if matches!(choice.message.content, Content::Array(_)) {
                let text = choice.message.content.as_text();
                choice.message.content = Content::String(text);
            }
        }
    }

    fn transform_stream_chunk(&self, chunk: &mut ChatStreamChunk) {
        // Ensure delta content is string
        for choice in &mut chunk.choices {
            if let Some(delta) = &mut choice.delta
                && matches!(delta.content, Some(Content::Array(_)))
                    && let Some(content) = delta.content.take() {
                        let text = content.as_text();
                        if !text.is_empty() {
                            delta.content = Some(Content::String(text));
                        }
                    }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatMessage, ChatRequest, Content, MessageRole};

    #[test]
    fn test_minimax_capabilities_keep_array_content() {
        let mut request = ChatRequest {
            model: "ab-01".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::Developer,
                content: Content::Array(vec![crate::types::chat_api::ContentBlock {
                    block_type: "text".to_string(),
                    text: Some("Hello".to_string()),
                    image_url: None,
                    input_audio: None,
                    file: None,
                    refusal: None,
                }]),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
                refusal: None,
            }],
            tools: None,
            tool_choice: None,
            stream: Some(false),
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

        let provider = MiniMaxProvider;
        provider.capabilities().sanitize_request(&mut request);

        let msg = request.messages.first().unwrap();
        assert_eq!(msg.role, MessageRole::System);
        assert!(matches!(msg.content, Content::Array(_)));
        assert_eq!(msg.content.as_text(), "Hello");
    }
}
