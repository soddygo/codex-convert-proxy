//! DeepSeek provider implementation.

use crate::providers::adapter::{
    ChatApiQuirks, OpenAiChatAdapter, ProtocolAdapter, ProviderConfig,
};
use crate::providers::Provider;

/// DeepSeek provider.
///
/// DeepSeek has some specific requirements:
/// - flatten_content: content arrays should be converted to plain strings
/// - reasoning_effort_map: `xhigh`/`max` → `max`, others → `high`
pub struct DeepSeekProvider {
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiChatAdapter,
            config: ProviderConfig {
                name: "deepseek",
                endpoint_url: "https://api.deepseek.com/v1/",
                chat_path: "chat/completions",
                auth_header_name: "Authorization",
                auth_header_value_prefix: "Bearer ",
                protocol: crate::providers::adapter::ProtocolType::ChatApi,
                default_model: None,
                chat_quirks: ChatApiQuirks {
                    flatten_content: true,
                    reasoning_effort_map: Some(map_reasoning_effort),
                    ..ChatApiQuirks::default()
                },
            },
        }
    }
}

fn map_reasoning_effort(effort: Option<String>) -> Option<String> {
    effort.map(|e| match e.as_str() {
        "xhigh" | "max" => "max".to_string(),
        _ => "high".to_string(),
    })
}

impl Provider for DeepSeekProvider {
    fn name(&self) -> &'static str {
        "deepseek"
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn protocol_adapter(&self) -> &dyn ProtocolAdapter {
        &self.adapter
    }
}
