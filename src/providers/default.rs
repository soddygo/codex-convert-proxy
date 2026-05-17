//! Default generic provider implementation.
//!
//! This provider is used as a fallback when the provider name is not found
//! in the registry. It makes minimal assumptions about the provider,
//! assuming most providers follow the OpenAI-compatible Chat API format.
//!
//! Unlike named providers (GLM, Kimi, etc.), `DefaultProvider` is **not**
//! registered in the static factory registry. Instead, [`create_provider`]
//! constructs it directly when no matching registry entry is found. This
//! keeps the fallback mechanism separate from the user-facing provider
//! namespace.

use crate::providers::adapter::{OpenAiChatAdapter, ProtocolAdapter, ProviderConfig};
use crate::providers::Provider;

/// Default provider that makes minimal assumptions.
///
/// This provider assumes:
/// - Standard OpenAI-compatible API path: `/chat/completions`
///   (the `/v1` prefix comes from the backend URL's base_path)
/// - Model names are passed through as-is
/// - No special request/response transformations needed
///
/// Use this as a fallback for providers not explicitly supported.
///
/// The `backend_name` field preserves the backend name from the config
/// (e.g., `"qwen"`, `"yi"`) so that logging and diagnostics can identify
/// which backend a request was routed to, even though the provider
/// implementation is the generic default.
pub struct DefaultProvider {
    /// The original backend name from config, preserved for diagnostics.
    backend_name: String,
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for DefaultProvider {
    fn default() -> Self {
        Self {
            backend_name: "default".to_string(),
            config: ProviderConfig::default(),
            adapter: OpenAiChatAdapter,
        }
    }
}

impl DefaultProvider {
    /// Create a new DefaultProvider with the original backend name.
    ///
    /// The `backend_name` is the name the user specified in `config.json`
    /// (e.g., `"qwen"`, `"yi-lightning"`). It is preserved for logging
    /// and diagnostics; the provider type is always `"default"`.
    pub fn new(backend_name: &str) -> Self {
        Self {
            backend_name: backend_name.to_lowercase(),
            config: ProviderConfig {
                name: "default",
                ..ProviderConfig::default()
            },
            adapter: OpenAiChatAdapter,
        }
    }
}

impl Provider for DefaultProvider {
    /// Returns `"default"` to indicate this is the generic fallback provider.
    fn name(&self) -> &'static str {
        "default"
    }

    /// Returns the original backend name from config (e.g., `"qwen"`,
    /// `"yi-lightning"`), making it easy to identify which backend a
    /// request was routed to in logs and diagnostics.
    fn display_name(&self) -> &str {
        &self.backend_name
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &self.adapter
    }
}
