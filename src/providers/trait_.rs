//! Provider trait definition.

use tracing::info;

use crate::error::ConversionError;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

// ============================================================================
// Provider Factory Registry
// ============================================================================

/// Factory function type for creating providers (type-erased function pointer).
type ProviderFactory = fn() -> Arc<dyn Provider>;

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
fn glm_factory() -> Arc<dyn Provider> {
    Arc::new(super::glm::GLMProvider::new())
}
fn kimi_factory() -> Arc<dyn Provider> {
    Arc::new(super::kimi::KimiProvider::new())
}
fn deepseek_factory() -> Arc<dyn Provider> {
    Arc::new(super::deepseek::DeepSeekProvider::new())
}
fn minimax_factory() -> Arc<dyn Provider> {
    Arc::new(super::minimax::MiniMaxProvider::new())
}
// ============================================================================
// Provider Trait
// ============================================================================

/// Provider trait for LLM provider-specific transformations.
///
/// Each Chinese LLM provider may have slightly different API requirements
/// or model name formats that need to be normalized.
///
/// Implementations are expected to be **stateless** so a single instance can
/// be shared across all requests via `Arc<dyn Provider>`.
pub trait Provider: Send + Sync + 'static {
    /// Get provider type identifier.
    ///
    /// Returns a static string identifying the provider type, e.g. `"glm"`,
    /// `"kimi"`, `"default"`. This is used for programmatic dispatch
    /// (e.g. `provider.name() == "minimax"`) and should **not** be used
    /// for user-facing logs.
    fn name(&self) -> &'static str;

    /// Get the display name for logging and diagnostics.
    ///
    /// For named providers (GLM, Kimi, etc.) this returns the same value as
    /// [`name()`](Self::name). For [`DefaultProvider`] this returns the
    /// original backend name from config (e.g. `"qwen"`, `"yi-lightning"`),
    /// making it easy to identify which backend a request was routed to.
    ///
    /// Use this in log messages, metrics labels, and diagnostics.
    fn display_name(&self) -> &str {
        self.name()
    }

    /// Normalize model name from Responses API to provider's format.
    fn normalize_model(&self, model: String) -> String {
        model
    }

    /// Get the chat completions path for this provider.
    ///
    /// Returns the endpoint path **without** the version prefix, e.g.
    /// `"/chat/completions"`. The version prefix (e.g., `"/v1"`) comes
    /// from the backend URL's `base_path` configured in `config.json` and
    /// is prepended automatically during path rewriting in
    /// `upstream_request_filter`.
    ///
    /// # Example
    ///
    /// Config URL `https://api.moonshot.cn/v1` → `base_path = "/v1"`
    /// `chat_completions_path() = "/chat/completions"`
    /// → final path: `/v1/chat/completions`
    fn chat_completions_path(&self) -> String {
        "/chat/completions".to_string()
    }

    /// Transform request before sending to provider.
    ///
    /// This is called after the standard conversion but before sending
    /// to the upstream provider. Providers can modify the request to
    /// handle API differences.
    fn transform_request(&self, _request: &mut ChatRequest) {}

    /// Transform response after receiving from provider.
    ///
    /// This is called after receiving the response but before converting
    /// to Responses API format. Providers can normalize response format.
    fn transform_response(&self, _response: &mut ChatResponse) {}

    /// Transform streaming chunk in real-time.
    ///
    /// This is called for each SSE chunk received from the provider.
    /// Providers can modify chunk content before event conversion.
    fn transform_stream_chunk(&self, _chunk: &mut ChatStreamChunk) {}
}

// ============================================================================
// Factory Function
// ============================================================================

/// Create a provider by name using the static registry.
///
/// Supports both exact names and aliases (e.g., "moonshot" -> "kimi").
///
/// When the provider name is not found in the registry, falls back to
/// [`DefaultProvider`] which makes minimal assumptions about the provider
/// (standard OpenAI-compatible Chat API format). This allows any Chinese LLM
/// provider not explicitly registered to work out of the box.
pub fn create_provider(name: &str) -> Result<Arc<dyn Provider>, ConversionError> {
    let name_lower = name.to_lowercase();

    // Handle aliases
    let normalized_name = match name_lower.as_str() {
        "moonshot" => "kimi",
        other => other,
    };

    // Try to get from registry
    if let Some(factory) = get_registry().get(normalized_name) {
        return Ok(factory());
    }

    // Fall back to DefaultProvider (OpenAI compatible) for unknown providers.
    // This is not in the registry to keep "default" as a reserved fallback
    // concept, separate from user-facing provider names.
    info!(
        "[PROVIDER] Unknown provider '{}', falling back to DefaultProvider (OpenAI compatible)",
        name
    );
    Ok(Arc::new(super::default::DefaultProvider::new(name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_provider_known() {
        // Known provider should return the correct provider
        let provider = create_provider("glm").unwrap();
        assert_eq!(provider.name(), "glm");
        assert_eq!(provider.display_name(), "glm");

        let provider = create_provider("kimi").unwrap();
        assert_eq!(provider.name(), "kimi");
        assert_eq!(provider.display_name(), "kimi");

        let provider = create_provider("deepseek").unwrap();
        assert_eq!(provider.name(), "deepseek");
        assert_eq!(provider.display_name(), "deepseek");

        let provider = create_provider("minimax").unwrap();
        assert_eq!(provider.name(), "minimax");
        assert_eq!(provider.display_name(), "minimax");
    }

    #[test]
    fn test_create_provider_unknown_fallback_to_default() {
        // Unknown provider should fall back to DefaultProvider
        let provider = create_provider("qwen").unwrap();
        assert_eq!(provider.name(), "default");

        let provider = create_provider("some-unknown-provider").unwrap();
        assert_eq!(provider.name(), "default");

        let provider = create_provider("abc").unwrap();
        assert_eq!(provider.name(), "default");
    }

    #[test]
    fn test_default_provider_display_name_preserves_backend_name() {
        // DefaultProvider.display_name() should return the original backend name
        let provider = create_provider("qwen").unwrap();
        assert_eq!(provider.name(), "default");
        assert_eq!(provider.display_name(), "qwen");

        // Case insensitive
        let provider = create_provider("Yi-Lightning").unwrap();
        assert_eq!(provider.name(), "default");
        assert_eq!(provider.display_name(), "yi-lightning");

        let provider = create_provider("some-unknown-provider").unwrap();
        assert_eq!(provider.name(), "default");
        assert_eq!(provider.display_name(), "some-unknown-provider");
    }

    #[test]
    fn test_create_provider_alias() {
        // Aliases should work
        let provider = create_provider("moonshot").unwrap();
        assert_eq!(provider.name(), "kimi");
        assert_eq!(provider.display_name(), "kimi");
    }

    #[test]
    fn test_registered_provider_names_excludes_default() {
        // "default" should NOT appear in registered names — it's a fallback, not a named provider
        let names = registered_provider_names();
        assert!(!names.contains(&"default"), "default should not be in registered_provider_names");
        assert!(names.contains(&"glm"));
        assert!(names.contains(&"kimi"));
        assert!(names.contains(&"deepseek"));
        assert!(names.contains(&"minimax"));
    }
}
