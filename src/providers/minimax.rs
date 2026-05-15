//! MiniMax provider implementation.

use crate::convert::ResponseRequestContext;
use crate::error::ConversionError;
use crate::providers::{
    Provider, ProviderCapabilities, ProviderExtensions, TokenLimitField, ToolChoiceSupport,
};
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk, Content};

/// MiniMax provider.
///
/// MiniMax specific handling:
/// - OpenAI-compatible endpoint supports tools
/// - Does not support 'developer' role, convert to 'system'
/// - Rejects multiple system messages, so they are merged into one
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
            tool_choice: ToolChoiceSupport::Unsupported,
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            supports_developer_role: false,
            flatten_request_content: true,
            supports_tool_strict: false,
            supports_stream_options: false,
            supports_parallel_tool_calls: false,
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

    fn sanitize_request(&self, request: &mut ChatRequest) {
        merge_system_messages(request);
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
                && let Some(content) = delta.content.take()
            {
                let text = content.as_text();
                if !text.is_empty() {
                    delta.content = Some(Content::String(text));
                }
            }
        }
    }
}

fn merge_system_messages(request: &mut ChatRequest) {
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};

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
    fn test_minimax_capabilities_flatten_array_content() {
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
        assert!(matches!(msg.content, Content::String(_)));
        assert_eq!(msg.content.as_text(), "Hello");
    }

    #[test]
    fn test_minimax_removes_unsupported_chat_settings() {
        let mut request = ChatRequest {
            model: "MiniMax-M2.7-highspeed".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::String("Hello".to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
                refusal: None,
            }],
            tools: None,
            tool_choice: None,
            stream: Some(true),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            user: None,
            stream_options: Some(crate::types::chat_api::StreamOptions {
                include_usage: Some(true),
            }),
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            stop: None,
            response_format: None,
            reasoning_effort: None,
            parallel_tool_calls: Some(false),
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

        assert_eq!(request.stream, Some(true));
        assert!(request.stream_options.is_none());
        assert!(request.parallel_tool_calls.is_none());
    }

    #[test]
    fn test_minimax_removes_tool_choice() {
        let mut request = ChatRequest {
            model: "MiniMax-M2.7-highspeed".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::String("Hello".to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
                refusal: None,
            }],
            tools: Some(vec![crate::types::chat_api::ChatTool {
                tool_type: "function".to_string(),
                function: crate::types::chat_api::FunctionDefinition {
                    name: "lookup".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {}
                    })),
                    strict: None,
                },
            }]),
            tool_choice: Some(crate::types::chat_api::ChatToolChoice::Mode(
                crate::types::chat_api::ChatToolChoiceMode::Auto,
            )),
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

        let provider = MiniMaxProvider;
        provider.capabilities().sanitize_request(&mut request);

        assert!(request.tools.is_some());
        assert!(request.tool_choice.is_none());
    }

    #[test]
    fn test_minimax_removes_tool_strict() {
        let mut request = ChatRequest {
            model: "MiniMax-M2.7-highspeed".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::String("Hello".to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
                refusal: None,
            }],
            tools: Some(vec![crate::types::chat_api::ChatTool {
                tool_type: "function".to_string(),
                function: crate::types::chat_api::FunctionDefinition {
                    name: "lookup".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {}
                    })),
                    strict: Some(false),
                },
            }]),
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

        let provider = MiniMaxProvider;
        provider.capabilities().sanitize_request(&mut request);

        let tool = request
            .tools
            .as_ref()
            .and_then(|tools| tools.first())
            .expect("tool should be kept");
        assert!(tool.function.strict.is_none());
    }

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

        MiniMaxProvider.sanitize_request(&mut request);

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
