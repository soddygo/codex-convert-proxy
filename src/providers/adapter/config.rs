use super::traits::ProtocolType;

/// Token-limit field preference for Chat API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLimitField {
    /// Standard `max_tokens` field.
    MaxTokens,
    /// Newer `max_completion_tokens` field (used by Kimi, MiniMax, o-series).
    MaxCompletionTokens,
}

/// How strictly a provider supports `tool_choice` in Chat API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoiceSupport {
    /// All modes and named function choices.
    Full,
    /// Omit `tool_choice` entirely.
    Unsupported,
    /// Only `auto` is safe.
    AutoOnly,
    /// Only `none` and `auto` are safe.
    AutoAndNone,
}

/// Provider-specific deviations from vanilla OpenAI Chat API.
///
/// When `protocol != ChatApi`, this struct is ignored.
#[derive(Debug, Clone)]
pub struct ChatApiQuirks {
    /// Whether the provider accepts function-calling tools.
    pub supports_tools: bool,
    /// Tool choice support level.
    pub tool_choice_mode: ToolChoiceSupport,
    /// Which token limit field to use: `max_tokens` or `max_completion_tokens`.
    pub token_limit_field: TokenLimitField,
    /// Convert array content to plain string (GLM, DeepSeek, MiniMax).
    pub flatten_content: bool,
    /// Merge multiple system messages into one (MiniMax-specific).
    pub merge_system_messages: bool,
    /// Whether the provider accepts `strict` on function definitions.
    pub supports_tool_strict: bool,
    /// Whether the provider accepts `stream_options`.
    pub supports_stream_options: bool,
    /// Whether the provider accepts `parallel_tool_calls`.
    pub supports_parallel_tool_calls: bool,
    /// If set, applies before inserting `reasoning_effort` (DeepSeek maps xhigh→max).
    pub reasoning_effort_map: Option<fn(Option<String>) -> Option<String>>,
    /// Extra JSON key-value to inject when reasoning is active.
    /// (GLM injects `{"thinking": {"type": "enabled"}}`;
    ///  MiniMax injects `{"reasoning_split": true}`).
    pub reasoning_extension: Option<(&'static str, serde_json::Value)>,
    /// Whether to flatten response content array → string after parsing.
    pub flatten_response: bool,
}

/// Sensible defaults for a vanilla OpenAI-compatible Chat API provider.
pub const DEFAULT_CHAT_QUIRKS: ChatApiQuirks = ChatApiQuirks {
    supports_tools: true,
    tool_choice_mode: ToolChoiceSupport::Full,
    token_limit_field: TokenLimitField::MaxTokens,
    flatten_content: false,
    merge_system_messages: false,
    supports_tool_strict: true,
    supports_stream_options: true,
    supports_parallel_tool_calls: true,
    reasoning_effort_map: None,
    reasoning_extension: None,
    flatten_response: false,
};

impl Default for ChatApiQuirks {
    fn default() -> Self {
        DEFAULT_CHAT_QUIRKS
    }
}

/// Static provider configuration — the single source of truth for a provider's
/// identity, endpoint, auth, protocol, and quirks.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Short name for programmatic dispatch, e.g. `"deepseek"`.
    pub name: &'static str,

    /// Base URL of the provider's API, e.g. `"https://api.deepseek.com/v1/"`.
    pub endpoint_url: &'static str,

    /// Path suffix for chat/responses endpoint, e.g. `"chat/completions"`.
    pub chat_path: &'static str,

    /// HTTP header name for authentication, e.g. `"Authorization"`.
    pub auth_header_name: &'static str,

    /// Value prefix before the API key, e.g. `"Bearer "`.
    /// Empty string for headers like `x-api-key: <key>`.
    pub auth_header_value_prefix: &'static str,

    /// Which protocol family this provider speaks.
    pub protocol: ProtocolType,

    /// Default model to use when the request doesn't specify one.
    pub default_model: Option<&'static str>,

    /// Chat API specific quirks (ignored for non-ChatApi protocols).
    pub chat_quirks: ChatApiQuirks,
}

impl ProviderConfig {
    /// Build the full chat endpoint URL from `endpoint_url` + `chat_path`,
    /// avoiding double slashes.
    pub fn build_chat_url(&self) -> String {
        let base = self.endpoint_url.trim_end_matches('/');
        let path = self.chat_path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// Build the `Authorization` (or equivalent) header value.
    pub fn build_auth_header_value(&self, api_key: &str) -> String {
        format!("{}{api_key}", self.auth_header_value_prefix)
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            name: "default",
            endpoint_url: "https://api.openai.com/v1/",
            chat_path: "chat/completions",
            auth_header_name: "Authorization",
            auth_header_value_prefix: "Bearer ",
            protocol: ProtocolType::ChatApi,
            default_model: None,
            chat_quirks: DEFAULT_CHAT_QUIRKS,
        }
    }
}
