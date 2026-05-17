//! Kimi (Moonshot AI) provider implementation.

use crate::providers::adapter::{
    OpenAiChatAdapter, ProtocolAdapter, ProviderConfig, ToolChoiceSupport, TokenLimitField,
};
use crate::providers::Provider;

/// Kimi (Moonshot AI) provider.
///
/// Kimi API accepts both "kimi-" and "moonshot-v1-" model name prefixes natively.
/// Specific quirks:
/// - tool_choice: `auto` only
/// - token_limit_field: `max_completion_tokens`
pub struct KimiProvider {
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for KimiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KimiProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiChatAdapter,
            config: ProviderConfig {
                name: "kimi",
                endpoint_url: "https://api.moonshot.cn/v1/",
                chat_path: "chat/completions",
                auth_header_name: "Authorization",
                auth_header_value_prefix: "Bearer ",
                protocol: crate::providers::adapter::ProtocolType::ChatApi,
                default_model: None,
                chat_quirks: crate::providers::adapter::ChatApiQuirks {
                    tool_choice_mode: ToolChoiceSupport::AutoOnly,
                    token_limit_field: TokenLimitField::MaxCompletionTokens,
                    ..crate::providers::adapter::ChatApiQuirks::default()
                },
            },
        }
    }
}

impl Provider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &self.adapter
    }
}
