//! Kimi (Moonshot AI) provider implementation.

use crate::providers::{Provider, ProviderCapabilities, TokenLimitField, ToolChoiceSupport};

/// Kimi (Moonshot AI) provider.
///
/// Kimi API accepts both "kimi-" and "moonshot-v1-" model name prefixes natively.
/// No model name normalization needed - all trait methods use default implementations.
pub struct KimiProvider;

impl Default for KimiProvider {
    fn default() -> Self {
        Self
    }
}

impl KimiProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            tool_choice: ToolChoiceSupport::AutoOnly,
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            supports_developer_role: false,
            flatten_request_content: false,
            supports_tool_strict: true,
            supports_stream_options: true,
            supports_parallel_tool_calls: true,
        }
    }
}
