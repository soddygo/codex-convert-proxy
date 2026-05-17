//! OpenAI-compatible Chat API providers.
//!
//! Each provider is a ~15-line configuration struct that specifies:
//! - endpoint URL and auth
//! - chat path (always "chat/completions")
//! - protocol (always ChatApi)
//! - provider-specific quirks (ChatApiQuirks)
//!
//! All providers share the same `OpenAiChatAdapter` for serialization.

use crate::providers::adapter::{
    ChatApiQuirks, OpenAiChatAdapter, OpenAiResponsesAdapter, ProtocolAdapter, ProviderConfig,
    ToolChoiceSupport,
};
use crate::providers::Provider;

/// Macro to DRY simple Chat API provider definitions.
///
/// Providers with custom quirks or non-standard chat paths should be defined
/// manually for clarity.
macro_rules! chat_provider {
    (
        $struct_name:ident,
        $name:literal,
        $endpoint_url:literal,
        $auth_header_name:literal,
        $auth_header_value_prefix:literal
    ) => {
        pub struct $struct_name {
            config: ProviderConfig,
            adapter: OpenAiChatAdapter,
        }

        impl Default for $struct_name {
            fn default() -> Self { Self::new() }
        }

        impl $struct_name {
            pub fn new() -> Self {
                Self {
                    adapter: OpenAiChatAdapter,
                    config: ProviderConfig {
                        name: $name,
                        endpoint_url: $endpoint_url,
                        chat_path: "chat/completions",
                        auth_header_name: $auth_header_name,
                        auth_header_value_prefix: $auth_header_value_prefix,
                        protocol: crate::providers::adapter::ProtocolType::ChatApi,
                        default_model: None,
                        chat_quirks: ChatApiQuirks::default(),
                    },
                }
            }
        }

        impl Provider for $struct_name {
            fn name(&self) -> &'static str { self.config.name }
            fn config(&self) -> &ProviderConfig { &self.config }
            fn protocol_adapter(&self) -> &dyn ProtocolAdapter { &self.adapter }
        }
    };
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

chat_provider!(OpenAiProvider, "openai", "https://api.openai.com/v1/", "Authorization", "Bearer ");

// ── Groq ──────────────────────────────────────────────────────────────────────

chat_provider!(GroqProvider, "groq", "https://api.groq.com/openai/v1/", "Authorization", "Bearer ");

// ── Together AI ───────────────────────────────────────────────────────────────

chat_provider!(TogetherProvider, "together", "https://api.together.xyz/v1/", "Authorization", "Bearer ");

// ── Fireworks AI ──────────────────────────────────────────────────────────────

chat_provider!(FireworksProvider, "fireworks", "https://api.fireworks.ai/inference/v1/", "Authorization", "Bearer ");

// ── Nebius AI Studio ──────────────────────────────────────────────────────────

chat_provider!(NebiusProvider, "nebius", "https://api.studio.nebius.ai/v1/", "Authorization", "Bearer ");

// ── xAI (Grok) ────────────────────────────────────────────────────────────────

chat_provider!(XaiProvider, "xai", "https://api.x.ai/v1/", "Authorization", "Bearer ");

// ── Aliyun (DashScope) ────────────────────────────────────────────────────────

chat_provider!(AliyunProvider, "aliyun", "https://dashscope.aliyuncs.com/compatible-mode/v1/", "Authorization", "Bearer ");

// ── Baidu (Qianfan) ───────────────────────────────────────────────────────────

chat_provider!(BaiduProvider, "baidu", "https://qianfan.baidubce.com/v2/", "Authorization", "Bearer ");

// ── Mimo ─────────────────────────────────────────────────────────────────────

chat_provider!(MimoProvider, "mimo", "https://api.xiaomimimo.com/v1/", "Authorization", "Bearer ");

// ── Ollama (local) ────────────────────────────────────────────────────────────

chat_provider!(OllamaProvider, "ollama", "http://localhost:11434/v1/", "Authorization", "Bearer ");

// ── Ollama Cloud ──────────────────────────────────────────────────────────────

chat_provider!(OllamaCloudProvider, "ollama_cloud", "https://api.ollama.com/v1/", "Authorization", "Bearer ");

// ── GitHub Copilot ────────────────────────────────────────────────────────────
// Note: GitHub Copilot uses namespace-based routing with multi-namespace support
// (openai/, anthropic/, google/). This provider covers the OpenAI namespace.

chat_provider!(GitHubCopilotProvider, "github_copilot", "https://api.githubcopilot.com/", "Authorization", "Bearer ");

// ── OpenCode Go ───────────────────────────────────────────────────────────────
// Note: OpenCode Go routes based on model namespace. MiniMax models use x-api-key
// auth; this default config uses Bearer auth. MiniMax routing is handled
// separately via the minimax provider.

chat_provider!(OpenCodeGoProvider, "opencode_go", "https://api.opencode.ai/v1/", "Authorization", "Bearer ");

// ── BigModel (Zhipu BigModel) ─────────────────────────────────────────────────
// Same API quirks as GLM: AutoOnly tool_choice, flatten_content, thinking extension

pub struct BigModelProvider {
    config: ProviderConfig,
    adapter: OpenAiChatAdapter,
}

impl Default for BigModelProvider {
    fn default() -> Self { Self::new() }
}

impl BigModelProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiChatAdapter,
            config: ProviderConfig {
                name: "bigmodel",
                endpoint_url: "https://open.bigmodel.cn/api/paas/v4/",
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

impl Provider for BigModelProvider {
    fn name(&self) -> &'static str { self.config.name }
    fn config(&self) -> &ProviderConfig { &self.config }
    fn protocol_adapter(&self) -> &dyn ProtocolAdapter { &self.adapter }
}

// ── OpenAI Responses API (native pass-through) ─────────────────────────────────
// Uses OpenAiResponsesAdapter — the proxy forwards the original Responses API
// JSON without Chat API conversion.

pub struct OpenAiResponsesProvider {
    config: ProviderConfig,
    adapter: OpenAiResponsesAdapter,
}

impl Default for OpenAiResponsesProvider {
    fn default() -> Self { Self::new() }
}

impl OpenAiResponsesProvider {
    pub fn new() -> Self {
        Self {
            adapter: OpenAiResponsesAdapter,
            config: ProviderConfig {
                name: "openai_resp",
                endpoint_url: "https://api.openai.com/v1/",
                chat_path: "responses",
                auth_header_name: "Authorization",
                auth_header_value_prefix: "Bearer ",
                protocol: crate::providers::adapter::ProtocolType::ResponsesApi,
                default_model: None,
                chat_quirks: ChatApiQuirks::default(), // unused for ResponsesApi
            },
        }
    }
}

impl Provider for OpenAiResponsesProvider {
    fn name(&self) -> &'static str { self.config.name }
    fn config(&self) -> &ProviderConfig { &self.config }
    fn protocol_adapter(&self) -> &dyn ProtocolAdapter { &self.adapter }
}
