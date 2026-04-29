//! Proxy context data structures.

use std::time::Instant;

use crate::convert::{ResponseRequestContext, StreamState};
use crate::config::BackendInfo;

/// Proxy context attached to each request session.
#[derive(Debug)]
pub struct ProxyContext {
    /// Request start time for duration tracking.
    pub start_time: Instant,
    /// Collected request body bytes.
    pub request_body: Vec<u8>,
    /// Model name parsed from request.
    pub model: Option<String>,
    /// Selected backend for this request.
    pub selected_backend: Option<BackendInfo>,
    /// Provider name.
    pub provider_name: Option<String>,
    /// Stream state for SSE conversion (also used for non-streaming conversion context).
    pub stream_state: Option<StreamState>,
    /// Response body collected for conversion.
    pub response_body: Vec<u8>,
    /// Whether streaming is enabled.
    pub is_streaming: bool,
    /// Rewritten upstream path.
    pub rewritten_path: Option<String>,
    /// Whether this is a streaming response (for conversion tracking).
    pub is_stream_response: bool,
    /// Whether this is a conversion request (Responses API -> Chat API).
    pub is_conversion_request: bool,
    /// Offset in response_body that has been parsed (to avoid re-parsing events).
    pub stream_body_parsed_offset: usize,
    /// Request path after optional routing prefix stripping.
    pub normalized_path: Option<String>,
    /// Whether current upstream response should be converted as SSE stream.
    pub should_convert_stream_response: bool,
    /// Upstream status code captured in response_filter for diagnostics.
    pub upstream_status: Option<u16>,
    /// Upstream content-type captured in response_filter for diagnostics.
    pub upstream_content_type: Option<String>,
    /// Number of valid upstream chat stream chunks parsed.
    pub stream_chunks_parsed: usize,
}

impl ProxyContext {
    /// Create a new proxy context.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            request_body: Vec::new(),
            model: None,
            selected_backend: None,
            provider_name: None,
            stream_state: None,
            response_body: Vec::new(),
            is_streaming: false,
            rewritten_path: None,
            is_stream_response: false,
            is_conversion_request: false,
            stream_body_parsed_offset: 0,
            normalized_path: None,
            should_convert_stream_response: false,
            upstream_status: None,
            upstream_content_type: None,
            stream_chunks_parsed: 0,
        }
    }

    /// Parse model name and stream flag from request body.
    /// Initializes StreamState for ALL conversion requests (both streaming and non-streaming).
    /// StreamState holds ResponseRequestContext for protocol-aligned response generation.
    pub fn init_from_request_body(&mut self) {
        if self.model.is_some() {
            return;
        }
        if let Ok(text) = std::str::from_utf8(&self.request_body) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
                    self.model = Some(model.to_string());
                }
                if let Some(stream) = json.get("stream").and_then(|v| v.as_bool()) {
                    self.is_streaming = stream;
                    if stream {
                        self.is_stream_response = true;
                    }
                }

                // Always initialize StreamState for conversion requests to hold context
                // This is used for both streaming and non-streaming conversion
                if self.is_conversion_request {
                    let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
                    // Get response_request_context from the stored request if available
                    let context = self.stream_state
                        .as_ref()
                        .and_then(|s| s.request_context.clone());
                    self.stream_state = Some(StreamState::new(
                        format!("resp_{}", uuid::Uuid::new_v4()),
                        model,
                        context,
                    ));
                }
            }
        }
    }

    /// Set the response request context from a parsed ResponseRequest.
    /// This should be called during request_body_filter processing.
    pub fn set_response_request_context(&mut self, context: ResponseRequestContext) {
        // If stream_state already exists, update its request_context
        if let Some(ref mut state) = self.stream_state {
            state.request_context = Some(context);
        } else {
            // Create stream_state with the context
            let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
            self.stream_state = Some(StreamState::new(
                format!("resp_{}", uuid::Uuid::new_v4()),
                model,
                Some(context),
            ));
        }
    }
}

impl Default for ProxyContext {
    fn default() -> Self {
        Self::new()
    }
}
