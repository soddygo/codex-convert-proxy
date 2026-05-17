//! Provider trait definition.

use tracing::info;

use crate::error::ConversionError;
use crate::types::chat_api::ChatRequest;
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
        m.insert("openai", openai_factory as ProviderFactory);
        m.insert("groq", groq_factory as ProviderFactory);
        m.insert("together", together_factory as ProviderFactory);
        m.insert("fireworks", fireworks_factory as ProviderFactory);
        m.insert("nebius", nebius_factory as ProviderFactory);
        m.insert("xai", xai_factory as ProviderFactory);
        m.insert("aliyun", aliyun_factory as ProviderFactory);
        m.insert("baidu", baidu_factory as ProviderFactory);
        m.insert("mimo", mimo_factory as ProviderFactory);
        m.insert("ollama", ollama_factory as ProviderFactory);
        m.insert("ollama_cloud", ollama_cloud_factory as ProviderFactory);
        m.insert("github_copilot", github_copilot_factory as ProviderFactory);
        m.insert("opencode_go", opencode_go_factory as ProviderFactory);
        m.insert("bigmodel", bigmodel_factory as ProviderFactory);
        m.insert("openai_resp", openai_resp_factory as ProviderFactory);
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
fn openai_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::OpenAiProvider::new())
}
fn groq_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::GroqProvider::new())
}
fn together_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::TogetherProvider::new())
}
fn fireworks_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::FireworksProvider::new())
}
fn nebius_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::NebiusProvider::new())
}
fn xai_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::XaiProvider::new())
}
fn aliyun_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::AliyunProvider::new())
}
fn baidu_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::BaiduProvider::new())
}
fn mimo_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::MimoProvider::new())
}
fn ollama_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::OllamaProvider::new())
}
fn ollama_cloud_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::OllamaCloudProvider::new())
}
fn github_copilot_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::GitHubCopilotProvider::new())
}
fn opencode_go_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::OpenCodeGoProvider::new())
}
fn bigmodel_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::BigModelProvider::new())
}
fn openai_resp_factory() -> Arc<dyn Provider> {
    Arc::new(super::openai_compat::OpenAiResponsesProvider::new())
}
// ============================================================================
// Provider Trait
// ============================================================================

use crate::providers::adapter::{OpenAiChatAdapter, ProtocolAdapter, ProviderConfig};

/// Provider trait for LLM provider-specific transformations.
///
/// Each LLM provider may have slightly different API requirements
/// that are expressed declaratively via [`ProviderConfig`] and
/// [`ProtocolAdapter`] (the two-layer design).
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

    /// Provider configuration — the single source of truth for endpoint,
    /// auth, protocol, and quirks.
    fn config(&self) -> &ProviderConfig;

    /// The protocol adapter that builds requests and parses responses
    /// for this provider's wire protocol.
    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &OpenAiChatAdapter
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
    fn chat_completions_path(&self) -> String {
        self.config().chat_path.to_string()
    }

    /// Final provider-specific request normalization (e.g. MiniMax system message merging).
    fn sanitize_request(&self, _request: &mut ChatRequest) {}
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

        let provider = create_provider("kimi").unwrap();
        assert_eq!(provider.name(), "kimi");

        let provider = create_provider("deepseek").unwrap();
        assert_eq!(provider.name(), "deepseek");

        let provider = create_provider("minimax").unwrap();
        assert_eq!(provider.name(), "minimax");

        let provider = create_provider("openai").unwrap();
        assert_eq!(provider.name(), "openai");

        let provider = create_provider("groq").unwrap();
        assert_eq!(provider.name(), "groq");

        let provider = create_provider("together").unwrap();
        assert_eq!(provider.name(), "together");

        let provider = create_provider("mimo").unwrap();
        assert_eq!(provider.name(), "mimo");
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
        let names = registered_provider_names();
        assert!(!names.contains(&"default"));
        assert!(names.contains(&"glm"));
        assert!(names.contains(&"kimi"));
        assert!(names.contains(&"deepseek"));
        assert!(names.contains(&"minimax"));
        assert!(names.contains(&"openai"));
        assert!(names.contains(&"groq"));
        assert!(names.contains(&"xai"));
        assert!(names.contains(&"ollama"));
    }
}
