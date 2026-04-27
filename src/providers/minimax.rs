//! MiniMax provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk, Content};

/// MiniMax provider.
///
/// MiniMax specific handling:
/// - Content must be string, not array
/// - Model name may need normalization
pub struct MiniMaxProvider;

impl Provider for MiniMaxProvider {
    fn name(&self) -> &'static str {
        "minimax"
    }

    fn normalize_model(&self, model: String) -> String {
        // MiniMax usesab-01,ab-02 for their models
        // No specific normalization needed
        model
    }

    fn transform_request(&mut self, request: &mut ChatRequest) {
        // MiniMax requires content to be a string, not an array
        for message in &mut request.messages {
            if matches!(message.content, Content::Array(_)) {
                let text = message.content.as_text();
                message.content = Content::String(text);
            }
        }
    }

    fn transform_response(&mut self, response: &mut ChatResponse) {
        // Ensure content is string
        for choice in &mut response.choices {
            if matches!(choice.message.content, Content::Array(_)) {
                let text = choice.message.content.as_text();
                choice.message.content = Content::String(text);
            }
        }
    }

    fn transform_stream_chunk(&mut self, chunk: &mut ChatStreamChunk) {
        // Ensure delta content is string
        for choice in &mut chunk.choices {
            if let Some(delta) = &mut choice.delta {
                if matches!(delta.content, Some(Content::Array(_))) {
                    if let Some(content) = delta.content.take() {
                        let text = content.as_text();
                        if !text.is_empty() {
                            delta.content = Some(Content::String(text));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};

    #[test]
    fn test_minimax_flattens_content() {
        let mut request = ChatRequest {
            model: "ab-01".to_string(),
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

        let mut provider = MiniMaxProvider;
        provider.transform_request(&mut request);

        let msg = request.messages.first().unwrap();
        assert!(matches!(msg.content, Content::String(_)));
        assert_eq!(msg.content.as_text(), "Hello");
    }
}
