//! OpenAiResponsesAdapter — pass-through adapter for providers that natively
//! speak the OpenAI Responses API.
//!
//! For these providers, the proxy skips request body conversion entirely and
//! forwards the original Responses API JSON from Codex. This adapter exists
//! primarily for type-system consistency and to provide `parse_response` for
//! the response direction.

use serde_json::Value;

use crate::error::ConversionError;
use crate::providers::adapter::traits::{ProtocolAdapter, ProtocolType};
use crate::providers::adapter::config::ProviderConfig;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};

/// Stateless singleton adapter for OpenAI Responses API.
///
/// For Responses API providers, the proxy does NOT convert the request body;
/// it passes through the Codex-originated Responses JSON as-is. The
/// `build_request_body` method is therefore a thin pass-through that
/// re-serializes the ChatRequest (which should already be in Responses format
/// at that point, or we bypass conversion entirely in the proxy layer).
#[derive(Debug, Clone, Copy)]
pub struct OpenAiResponsesAdapter;

impl ProtocolAdapter for OpenAiResponsesAdapter {
    fn protocol_name(&self) -> &'static str {
        "openai-responses"
    }

    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::ResponsesApi
    }

    fn build_request_body(
        &self,
        _chat_req: &ChatRequest,
        _config: &ProviderConfig,
    ) -> Result<Value, ConversionError> {
        // Responses API providers bypass the ChatRequest conversion in the
        // proxy layer — the original ResponseRequest JSON is forwarded as-is.
        // This method should not be called in normal operation.
        Ok(Value::Null)
    }

    fn parse_response(
        &self,
        body: &Value,
        _config: &ProviderConfig,
    ) -> Result<ChatResponse, ConversionError> {
        // Responses API responses are not converted to ChatResponse;
        // they are forwarded as-is to the client.
        // Return an empty ChatResponse as a placeholder.
        Ok(ChatResponse {
            id: body
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            object_name: "response".to_string(),
            created: body
                .get("created_at")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as u64,
            model: body
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            choices: vec![],
            usage: None,
            service_tier: None,
            system_fingerprint: None,
        })
    }

    fn parse_stream_chunk(
        &self,
        _chunk_json: &Value,
        _config: &ProviderConfig,
    ) -> Result<ChatStreamChunk, ConversionError> {
        // Responses API streaming uses a different event format (not SSE
        // ChatStreamChunks). The proxy handles Responses API streaming
        // as a pass-through in the response body filter.
        Err(ConversionError::ProviderError(
            "Responses API streaming is handled as pass-through; parse_stream_chunk should not be called".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::adapter::config::ProviderConfig;

    fn make_config() -> ProviderConfig {
        ProviderConfig {
            name: "openai_resp",
            endpoint_url: "https://api.openai.com/v1/",
            chat_path: "responses",
            protocol: ProtocolType::ResponsesApi,
            ..ProviderConfig::default()
        }
    }

    #[test]
    fn test_responses_adapter_parse_basic_response() {
        let adapter = OpenAiResponsesAdapter;
        let config = make_config();
        let body = serde_json::json!({
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-5-mini",
            "status": "completed",
            "output": []
        });

        let response = adapter.parse_response(&body, &config).unwrap();
        assert_eq!(response.id, "resp_abc123");
        assert_eq!(response.model, "gpt-5-mini");
    }
}
