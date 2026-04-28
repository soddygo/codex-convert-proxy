//! GLM provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};

/// GLM (Zhipu AI) provider.
///
/// GLM has some specific requirements:
/// - Does not support function calling tools
/// - Messages should be flattened to simple text format
/// - May need model name normalization
pub struct GLMProvider;

impl Provider for GLMProvider {
    fn name(&self) -> &'static str {
        "glm"
    }

    fn normalize_model(&self, model: String) -> String {
        // GLM model naming: glm-4, glm-4-flash, glm-4-plus, glm-3
        // No transformation needed as Responses API model names should match
        model
    }

    fn transform_request(&mut self, request: &mut ChatRequest) {
        // GLM doesn't support tools - remove them
        request.tools = None;
        request.tool_choice = None;

        // Flatten message content to simple strings
        for message in &mut request.messages {
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

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(GLMProvider)
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
        };

        let mut provider = GLMProvider;
        provider.transform_request(&mut request);

        let msg = request.messages.first().unwrap();
        assert!(matches!(msg.content, Content::String(_)));
        assert_eq!(msg.content.as_text(), "Hello");
    }
}
