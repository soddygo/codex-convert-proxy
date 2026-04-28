//! Provider trait definition.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};

/// Provider trait for LLM provider-specific transformations.
///
/// Each Chinese LLM provider may have slightly different API requirements
/// or model name formats that need to be normalized.
pub trait Provider: Send + Sync {
    /// Get provider name.
    fn name(&self) -> &'static str;

    /// Normalize model name from Responses API to provider's format.
    fn normalize_model(&self, model: String) -> String {
        model
    }

    /// Transform request before sending to provider.
    ///
    /// This is called after the standard conversion but before sending
    /// to the upstream provider. Providers can modify the request to
    /// handle API differences.
    fn transform_request(&mut self, _request: &mut ChatRequest) {}

    /// Transform response after receiving from provider.
    ///
    /// This is called after receiving the response but before converting
    /// to Responses API format. Providers can normalize response format.
    fn transform_response(&mut self, _response: &mut ChatResponse) {}

    /// Transform streaming chunk in real-time.
    ///
    /// This is called for each SSE chunk received from the provider.
    /// Providers can modify chunk content before event conversion.
    fn transform_stream_chunk(&mut self, _chunk: &mut ChatStreamChunk) {}

    /// Clone the provider as a boxed trait object.
    fn clone_box(&self) -> Box<dyn Provider + Send + Sync>;
}

/// Create a provider by name.
pub fn create_provider(name: &str) -> Result<Box<dyn Provider>, ConversionError> {
    match name.to_lowercase().as_str() {
        "glm" => Ok(Box::new(super::glm::GLMProvider)),
        "kimi" => Ok(Box::new(super::kimi::KimiProvider)),
        "moonshot" => Ok(Box::new(super::kimi::KimiProvider)),
        "deepseek" => Ok(Box::new(super::deepseek::DeepSeekProvider)),
        "minimax" => Ok(Box::new(super::minimax::MiniMaxProvider)),
        _ => Err(ConversionError::ProviderError(format!(
            "Unknown provider: {}",
            name
        ))),
    }
}
