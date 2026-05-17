use crate::error::ConversionError;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};

use super::config::{ChatApiQuirks, ProviderConfig, DEFAULT_CHAT_QUIRKS};

/// The wire-protocol family used to communicate with the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolType {
    /// OpenAI Chat Completions API (`POST /v1/chat/completions`).
    /// Used by ~80% of providers.
    ChatApi,
    /// OpenAI Responses API (`POST /v1/responses`).
    /// Used by newer OpenAI models (gpt-5, codex). Mostly pass-through.
    ResponsesApi,
}

impl ProtocolType {
    pub fn default_chat_path(&self) -> &'static str {
        match self {
            ProtocolType::ChatApi => "chat/completions",
            ProtocolType::ResponsesApi => "responses",
        }
    }

    pub fn default_quirks(&self) -> ChatApiQuirks {
        match self {
            ProtocolType::ChatApi => DEFAULT_CHAT_QUIRKS,
            // ResponsesApi doesn't use ChatApiQuirks
            ProtocolType::ResponsesApi => ChatApiQuirks::default(),
        }
    }
}

/// A protocol adapter knows how to convert a provider-neutral `ChatRequest`
/// into a specific wire-protocol JSON body, and how to parse responses back.
///
/// Implementations are **stateless** singletons shared across all providers
/// that use the same protocol (e.g. all Chat API providers share one
/// `OpenAiChatAdapter` instance).
pub trait ProtocolAdapter: Send + Sync + 'static {
    /// Human-readable protocol name for logging.
    fn protocol_name(&self) -> &'static str;

    /// The [`ProtocolType`] this adapter handles.
    fn protocol_type(&self) -> ProtocolType;

    /// Build the request body JSON to send to the upstream provider.
    ///
    /// This **owns** serialization — it only includes fields the provider
    /// actually supports, determined by `config.chat_quirks`.
    fn build_request_body(
        &self,
        chat_req: &ChatRequest,
        config: &ProviderConfig,
    ) -> Result<serde_json::Value, ConversionError>;

    /// Parse a non-streaming response body into a unified `ChatResponse`.
    fn parse_response(
        &self,
        body: &serde_json::Value,
        config: &ProviderConfig,
    ) -> Result<ChatResponse, ConversionError>;

    /// Parse a single streaming SSE delta event from raw JSON into a
    /// `ChatStreamChunk`, applying config-level quirks like content flattening.
    fn parse_stream_chunk(
        &self,
        chunk_json: &serde_json::Value,
        config: &ProviderConfig,
    ) -> Result<ChatStreamChunk, ConversionError>;
}
