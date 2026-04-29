//! GLM provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};
use std::any::Any;

#[derive(Clone)]
/// GLM (Zhipu AI) provider.
///
/// GLM has some specific requirements:
/// - Does not support function calling tools
/// - Messages should be flattened to simple text format
/// - API path is /chat/completions (not /v1/chat/completions)
pub struct GLMProvider;

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
        // GLM uses /chat/completions not /v1/chat/completions
        "/chat/completions".to_string()
    }

    fn transform_request(&mut self, request: &mut ChatRequest) {
        // GLM doesn't support tools - remove them
        request.tools = None;
        request.tool_choice = None;

        // Flatten message content to simple strings
        for message in &mut request.messages {
            // GLM doesn't support developer role - convert to user
            if message.role == crate::types::chat_api::MessageRole::Developer {
                message.role = crate::types::chat_api::MessageRole::User;
            }
            let text = message.content.as_text();
            message.content = crate::types::chat_api::Content::String(text);
        }
    }

    fn transform_response(&mut self, response: &mut ChatResponse) {
        // Ensure content is string format
        for choice in &mut response.choices {
            let text = choice.message.content.as_text();
            choice.message.content = crate::types::chat_api::Content::String(text);
        }
    }

    fn transform_stream_chunk(&mut self, chunk: &mut ChatStreamChunk) {
        // Ensure delta content is string format
        for choice in &mut chunk.choices {
            if let Some(delta) = &mut choice.delta {
                if let Some(content) = &delta.content {
                    let text = content.as_text();
                    if !text.is_empty() {
                        delta.content = Some(crate::types::chat_api::Content::String(text));
                    }
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};

    #[test]
    fn test_glm_removes_tools() {
        let mut request = ChatRequest {
            model: "glm-4".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::String("Hello".to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Some(vec![]),
            tool_choice: None,
            stream: Some(false),
            temperature: None,
            max_tokens: None,
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
            seed: None,
            service_tier: None,
        };

        let mut provider = GLMProvider;
        provider.transform_request(&mut request);

        assert!(request.tools.is_none());
        assert!(request.tool_choice.is_none());
    }

    #[test]
    fn test_glm_flattens_content() {
        let mut request = ChatRequest {
            model: "glm-4".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::Array(vec![crate::types::chat_api::ContentBlock {
                    block_type: "text".to_string(),
                    text: Some("Hello".to_string()),
                    image_url: None,
                }]),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            tool_choice: None,
            stream: Some(false),
            temperature: None,
            max_tokens: None,
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
            seed: None,
            service_tier: None,
        };

        let mut provider = GLMProvider;
        provider.transform_request(&mut request);

        let msg = request.messages.first().unwrap();
        assert!(matches!(msg.content, Content::String(_)));
        assert_eq!(msg.content.as_text(), "Hello");
    }
}
