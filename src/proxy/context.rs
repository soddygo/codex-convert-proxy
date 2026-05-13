//! Proxy context data structures.
//!
//! The per-request `ProxyContext` composes several single-responsibility
//! sub-structs so each cluster of related fields can be reasoned about (and
//! initialised) independently:
//!
//! - [`RouteInfo`] — backend selection + path rewriting (populated in
//!   `request_filter`).
//! - [`ConversionFlags`] — boolean state machine that drives whether and how
//!   bodies are converted.
//! - [`ConversionBuffers`] — buffered request/response bodies + stream parse
//!   cursor.
//! - [`UpstreamDiagnostics`] — observed upstream status / content-type / chunk
//!   count, kept for logging and bypass decisions.
//! - [`FollowUpContext`] — conversation messages we'll persist after the
//!   upstream completes (for `previous_response_id` expansion).
//!
//! The root keeps only fields that don't fit any cluster: `start_time`,
//! `model`, and the streaming `StreamState`.

use std::time::Instant;

use crate::convert::{ResponseRequestContext, StreamState};
use crate::config::BackendInfo;
use crate::types::chat_api::ChatMessage;
use crate::types::response_api::ResponseRequest;

/// Backend selection and request-path information.
#[derive(Debug, Default)]
pub struct RouteInfo {
    /// Selected backend connection info.
    pub selected_backend: Option<BackendInfo>,
    /// Backend / provider name (cached for diagnostics after the backend
    /// reference is no longer convenient to hold).
    pub provider_name: Option<String>,
    /// Request path after optional `path_prefix` stripping.
    pub normalized_path: Option<String>,
    /// Rewritten upstream path (includes backend's base_path).
    pub rewritten_path: Option<String>,
}

/// Boolean state machine that records what kind of conversion is in flight.
#[derive(Debug, Default)]
pub struct ConversionFlags {
    /// True if the original request targets `/v1/responses` (or `/responses`)
    /// and must be converted to a Chat-API request.
    pub is_conversion_request: bool,
    /// True if the client requested streaming (`stream: true`).
    pub is_streaming: bool,
    /// True if the upstream is producing a streaming response. Usually equal
    /// to `is_streaming` but retained as a separate flag for clarity around
    /// upstream content-type detection.
    pub is_stream_response: bool,
    /// True only after `response_filter` confirmed the upstream is producing
    /// SSE we should convert. Drives `response_body_filter`'s streaming path.
    pub should_convert_stream_response: bool,
}

/// Buffered request/response bodies and the stream-parse cursor.
#[derive(Debug, Default)]
pub struct ConversionBuffers {
    /// Collected request body bytes (cleared after conversion).
    pub request_body: Vec<u8>,
    /// Collected response body bytes (for non-streaming conversion + stream
    /// frame accumulation).
    pub response_body: Vec<u8>,
    /// Offset in `response_body` that has already been parsed, so SSE event
    /// reparsing across chunks is bounded.
    pub stream_body_parsed_offset: usize,
}

/// Upstream observations for diagnostics + bypass decisions.
#[derive(Debug, Default)]
pub struct UpstreamDiagnostics {
    /// Status code captured in `response_filter`.
    pub upstream_status: Option<u16>,
    /// `content-type` captured in `response_filter`.
    pub upstream_content_type: Option<String>,
    /// Count of valid upstream stream chunks parsed (drives "skip
    /// response.completed" bypass when zero).
    pub stream_chunks_parsed: usize,
}

/// Information needed to persist a completed turn for subsequent
/// `previous_response_id` lookups.
#[derive(Debug, Default)]
pub struct FollowUpContext {
    /// Conversation messages collected before the upstream response (so the
    /// assistant reply can be appended to them and stored).
    pub pending_conversation_messages: Option<Vec<ChatMessage>>,
    /// Effective instructions used for this turn (after history expansion).
    pub pending_instructions: Option<String>,
}

/// Proxy context attached to each request session.
#[derive(Debug)]
pub struct ProxyContext {
    /// Request start time for duration tracking.
    pub start_time: Instant,
    /// Model name parsed from request.
    pub model: Option<String>,
    /// Stream state for SSE conversion (also used as a carrier for
    /// `ResponseRequestContext` in non-streaming conversions; this dual use is
    /// scheduled to be decoupled in a follow-up commit).
    pub stream_state: Option<StreamState>,

    pub route: RouteInfo,
    pub flags: ConversionFlags,
    pub buffers: ConversionBuffers,
    pub diagnostics: UpstreamDiagnostics,
    pub follow_up: FollowUpContext,
}

impl ProxyContext {
    /// Create a new proxy context.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            model: None,
            stream_state: None,
            route: RouteInfo::default(),
            flags: ConversionFlags::default(),
            buffers: ConversionBuffers::default(),
            diagnostics: UpstreamDiagnostics::default(),
            follow_up: FollowUpContext::default(),
        }
    }

    /// Populate `model` + streaming flags from a parsed `ResponseRequest`.
    ///
    /// Conversion path: callers already deserialize the body as `ResponseRequest`
    /// to perform the conversion, so we accept the parsed struct directly to
    /// avoid re-parsing the JSON.
    pub fn init_from_response_request(&mut self, req: &ResponseRequest) {
        if self.model.is_none() {
            self.model = Some(req.model.clone());
        }
        self.flags.is_streaming = req.stream;
        if req.stream {
            self.flags.is_stream_response = true;
        }

        if self.flags.is_conversion_request {
            let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
            let context = self
                .stream_state
                .as_ref()
                .and_then(|s| s.request_context.clone());
            self.stream_state = Some(StreamState::new(
                format!("resp_{}", uuid::Uuid::new_v4()),
                model,
                context,
            ));
        }
    }

    /// Pass-through path: extract just `model` + `stream` from a serde_json::Value.
    ///
    /// Used when the body is not a Responses API request (e.g., direct Chat
    /// Completions pass-through), so a full typed parse would be wrong.
    pub fn init_from_passthrough_json(&mut self, json: &serde_json::Value) {
        if self.model.is_none()
            && let Some(model) = json.get("model").and_then(|v| v.as_str())
        {
            self.model = Some(model.to_string());
        }
        if let Some(stream) = json.get("stream").and_then(|v| v.as_bool()) {
            self.flags.is_streaming = stream;
            if stream {
                self.flags.is_stream_response = true;
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
