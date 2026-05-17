//! GLM provider implementation.

use crate::providers::adapter::{
    ChatApiQuirks, OpenAiChatAdapter, ProtocolAdapter, ProviderConfig, ToolChoiceSupport,
};
use crate::providers::Provider;

/// GLM (Zhipu AI) provider.
///
/// GLM has some specific requirements:
/// - tool_choice: only `auto` is known-safe
/// - flatten_content: messages should be flattened to simple text format
/// - reasoning: injects `{"thinking": {"type": "enabled"}}`
/// - flatten_response: response content arrays → strings
pub struct GLMProvider {
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for GLMProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GLMProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiChatAdapter,
            config: ProviderConfig {
                name: "glm",
                endpoint_url: "https://api.z.ai/api/paas/v4/",
                chat_path: "chat/completions",
                auth_header_name: "Authorization",
                auth_header_value_prefix: "Bearer ",
                protocol: crate::providers::adapter::ProtocolType::ChatApi,
                default_model: None,
                chat_quirks: ChatApiQuirks {
                    tool_choice_mode: ToolChoiceSupport::AutoOnly,
                    flatten_content: true,
                    reasoning_extension: Some((
                        "thinking",
                        serde_json::json!({"type": "enabled"}),
                    )),
                    flatten_response: true,
                    ..ChatApiQuirks::default()
                },
            },
        }
    }
}

impl Provider for GLMProvider {
    fn name(&self) -> &'static str {
        "glm"
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &self.adapter
    }
}
