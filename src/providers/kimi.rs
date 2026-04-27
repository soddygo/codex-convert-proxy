//! Kimi (Moonshot) provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk, Content};

/// Kimi (Moonshot AI) provider.
///
/// Kimi specific handling:
/// - Model name may need normalization (kimi- → moonshot-v1-)
/// - Generally compatible with standard Chat API
pub struct KimiProvider;

impl Provider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn normalize_model(&self, model: String) -> String {
        // Normalize model names: kimi-* -> moonshot-v1-*
        if model.starts_with("kimi-") {
            model.replace("kimi-", "moonshot-v1-")
        } else if model.starts_with("moonshot-") {
            // Already normalized
            model
        } else {
            // Pass through unknown models unchanged
            model
        }
    }

    fn transform_request(&self, request: &mut ChatRequest) {
        // Kimi is mostly standard, but ensure content format is correct
        for message in &mut request.messages {
            let text = message.content.as_text();
            // Keep content as array for rich content, string for simple text
            if text.is_empty() && matches!(message.content, Content::Array(_)) {
                // Keep array if it has content
            } else if matches!(message.content, Content::Array(ref arr) if arr.len() == 1 && arr[0].text.is_some()) {
                // Simplify single-text array to string
                message.content = Content::String(text);
            }
        }
    }

    fn transform_response(&self, _response: &mut ChatResponse) {
        // Standard handling
    }

    fn transform_stream_chunk(&self, _chunk: &mut ChatStreamChunk) {
        // Standard handling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kimi_normalizes_model_name() {
        let provider = KimiProvider;

        assert_eq!(
            provider.normalize_model("kimi-llm".to_string()),
            "moonshot-v1-llm"
        );
        assert_eq!(
            provider.normalize_model("moonshot-v1-8k".to_string()),
            "moonshot-v1-8k"
        );
    }
}
