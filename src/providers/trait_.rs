//! Provider trait definition.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};
use std::collections::HashMap;
use std::sync::OnceLock;

// ============================================================================
// Provider Factory Registry
// ============================================================================

/// Factory function type for creating providers (type-erased function pointer).
type ProviderFactory = unsafe fn() -> Box<dyn Provider + Send + Sync>;

/// Static registry of provider factories.
fn get_registry() -> &'static HashMap<&'static str, ProviderFactory> {
    static REGISTRY: OnceLock<HashMap<&'static str, ProviderFactory>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("glm", glm_factory as ProviderFactory);
        m.insert("kimi", kimi_factory as ProviderFactory);
        m.insert("deepseek", deepseek_factory as ProviderFactory);
        m.insert("minimax", minimax_factory as ProviderFactory);
        m
    })
}

/// Get all registered provider names.
pub fn registered_provider_names() -> Vec<&'static str> {
    get_registry().keys().copied().collect()
}

// Factory functions (must be in separate functions to get unique addresses)
unsafe fn glm_factory() -> Box<dyn Provider + Send + Sync> {
    Box::new(super::glm::GLMProvider::new())
}
unsafe fn kimi_factory() -> Box<dyn Provider + Send + Sync> {
    Box::new(super::kimi::KimiProvider::new())
}
unsafe fn deepseek_factory() -> Box<dyn Provider + Send + Sync> {
    Box::new(super::deepseek::DeepSeekProvider::new())
}
unsafe fn minimax_factory() -> Box<dyn Provider + Send + Sync> {
    Box::new(super::minimax::MiniMaxProvider::new())
}

// ============================================================================
// Provider Trait
// ============================================================================

/// Provider trait for LLM provider-specific transformations.
///
/// Each Chinese LLM provider may have slightly different API requirements
/// or model name formats that need to be normalized.
pub trait Provider: Send + Sync + 'static {
    /// Get provider name.
    fn name(&self) -> &'static str;

    /// Normalize model name from Responses API to provider's format.
    fn normalize_model(&self, model: String) -> String {
        model
    }

    /// Get the chat completions path for this provider.
    /// Only returns the endpoint path, e.g., "/chat/completions".
    /// The version prefix (e.g., "/v1") should come from the backend URL's base_path.
    fn chat_completions_path(&self) -> String {
        "/v1/chat/completions".to_string()
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

    /// Convert self to Any for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

// ============================================================================
// Clone Implementation
// ============================================================================

/// Clone for Box<dyn Provider> uses downcasting (Rust object safety limitation).
impl Clone for Box<dyn Provider + Send + Sync> {
    fn clone(&self) -> Self {
        self.as_ref().clone_box()
    }
}

// ============================================================================
// Factory Function
// ============================================================================

/// Create a provider by name using the static registry.
///
/// Supports both exact names and aliases (e.g., "moonshot" -> "kimi").
pub fn create_provider(name: &str) -> Result<Box<dyn Provider + Send + Sync>, ConversionError> {
    let name_lower = name.to_lowercase();

    // Handle aliases
    let normalized_name = match name_lower.as_str() {
        "moonshot" => "kimi",
        other => other,
    };

    // Try to get from registry
    if let Some(factory) = get_registry().get(normalized_name) {
        // SAFETY: factory functions are guaranteed to return valid Box<dyn Provider>
        return Ok(unsafe { factory() });
    }

    // Return error with available provider names
    let available = registered_provider_names();
    Err(ConversionError::ProviderError(format!(
        "Unknown provider: {}. Available: {:?}",
        name, available
    )))
}
