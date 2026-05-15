//! DeepSeek provider implementation.

use crate::providers::{
    Provider, ProviderCapabilities, TokenLimitField, ToolChoiceSupport,
};

/// DeepSeek provider.
///
/// DeepSeek is mostly compatible with standard Chat API.
/// Minimal transformation needed - all trait methods use default implementations.
pub struct DeepSeekProvider;

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self
    }
}

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for DeepSeekProvider {
    fn name(&self) -> &'static str {
        "deepseek"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            tool_choice: ToolChoiceSupport::Full,
            token_limit_field: TokenLimitField::MaxTokens,
            supports_developer_role: false,
            flatten_request_content: true,
        }
    }

    fn normalize_reasoning_effort(&self, effort: Option<String>) -> Option<String> {
        effort.map(|effort| {
            match effort.as_str() {
                "xhigh" | "max" => "max".to_string(),
                "low" | "medium" | "high" => "high".to_string(),
                _ => "high".to_string(),
            }
        })
    }
}
