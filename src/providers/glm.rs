//! GLM provider implementation.

use crate::convert::ResponseRequestContext;
use crate::error::ConversionError;
use crate::providers::{
    Provider, ProviderCapabilities, ProviderExtensions, TokenLimitField, ToolChoiceSupport,
};
use crate::types::chat_api::{ChatResponse, ChatStreamChunk};

/// GLM (Zhipu AI) provider.
///
/// GLM has some specific requirements:
/// - Supports function calling tools on current chat-completions API
/// - Only `auto` tool_choice is known-safe
/// - Messages should be flattened to simple text format
/// - API path is /chat/completions (not /v1/chat/completions)
pub struct GLMProvider;

impl Default for GLMProvider {
    fn default() -> Self {
        Self
    }
}

impl GLMProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for GLMProvider {
    fn name(&self) -> &'static str {
        "glm"
    }

    fn chat_completions_path(&self) -> String {
        // GLM base_path already includes version prefix (/api/paas/v4),
        // so we only need the endpoint suffix.
        "/chat/completions".to_string()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            tool_choice: ToolChoiceSupport::AutoOnly,
            token_limit_field: TokenLimitField::MaxTokens,
            supports_developer_role: false,
            flatten_request_content: true,
        }
    }

    fn provider_extensions(
        &self,
        context: &ResponseRequestContext,
    ) -> Result<ProviderExtensions, ConversionError> {
        let mut extensions = ProviderExtensions::new();
        if context.reasoning.is_some() {
            extensions.insert("thinking", serde_json::json!({"type": "enabled"}))?;
        }
        Ok(extensions)
    }

    fn transform_response(&self, response: &mut ChatResponse) {
        // Ensure content is string format
        for choice in &mut response.choices {
            let text = choice.message.content.as_text();
            choice.message.content = crate::types::chat_api::Content::String(text);
        }
    }

    fn transform_stream_chunk(&self, chunk: &mut ChatStreamChunk) {
        // Ensure delta content is string format
        for choice in &mut chunk.choices {
            if let Some(delta) = &mut choice.delta
                && let Some(content) = &delta.content {
                    let text = content.as_text();
                    if !text.is_empty() {
                        delta.content = Some(crate::types::chat_api::Content::String(text));
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
    fn test_glm_capabilities_keep_tools_and_flatten_content() {
        let mut request = ChatRequest {
            model: "glm-4".to_string(),
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
            tools: Some(vec![]),
            tool_choice: Some(crate::types::chat_api::ChatToolChoice::Mode(
                crate::types::chat_api::ChatToolChoiceMode::Required,
            )),
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

        let provider = GLMProvider;
        provider.capabilities().sanitize_request(&mut request);

        let msg = request.messages.first().unwrap();
        assert_eq!(msg.role, MessageRole::System);
        assert!(matches!(msg.content, Content::String(_)));
        assert_eq!(msg.content.as_text(), "Hello");
        assert!(request.tools.is_some());
        assert!(matches!(
            request.tool_choice,
            Some(crate::types::chat_api::ChatToolChoice::Mode(
                crate::types::chat_api::ChatToolChoiceMode::Auto
            ))
        ));
    }
}
