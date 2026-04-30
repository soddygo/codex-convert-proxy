//! Streaming response handler for SSE conversion.
//!
//! Extracts the streaming response conversion logic from `response_body_filter`
//! to improve code organization and single responsibility principle.

use tracing::{debug, error, warn};

use crate::convert::{chat_chunk_to_response_events, event_to_sse, ResponseStreamEvent};
use crate::providers::Provider;
use crate::types::chat_api::ChatStreamChunk;
use crate::util::parse_sse;

use super::context::ProxyContext;

/// Handler for streaming SSE response conversion.
/// Processes Chat API SSE chunks and converts them to Responses API SSE events.
pub struct StreamingResponseHandler<'a> {
    /// Reference to proxy context for state access.
    ctx: &'a mut ProxyContext,
    /// Provider clone for transformations (owned to avoid lifetime complexity).
    provider: Option<Box<dyn Provider + Send + Sync>>,
    /// Whether to log bodies for debugging.
    log_body: bool,
}

impl<'a> StreamingResponseHandler<'a> {
    /// Create a new streaming handler.
    pub fn new(
        ctx: &'a mut ProxyContext,
        provider: Option<Box<dyn Provider + Send + Sync>>,
        log_body: bool,
    ) -> Self {
        Self {
            ctx,
            provider,
            log_body,
        }
    }

    /// Process a single streaming frame/body chunk.
    ///
    /// Parses SSE events from the accumulated response body (starting from last
    /// parsed offset), converts each ChatStreamChunk to Responses API events,
    /// and returns the combined SSE output.
    ///
    /// Returns `None` if no conversion events were generated.
    pub fn process_stream_frame(&mut self) -> Option<String> {
        // Use accumulated body for SSE parsing (events may span multiple frames)
        // Only parse from the last parsed offset to avoid re-processing events
        let text = std::str::from_utf8(&self.ctx.response_body).unwrap_or("");
        let unparsed_text = &text[self.ctx.stream_body_parsed_offset..];

        debug!(
            "[STREAM_RAW] is_streaming=true, body={}",
            String::from_utf8_lossy(&self.ctx.response_body)
                .chars()
                .take(200)
                .collect::<String>()
        );

        let mut converted_chunks: Vec<String> = Vec::new();

        // Use SSE utility module to parse only new events
        let (events, parse_end_pos) = parse_sse(unparsed_text);
        let new_events_count = events.len();
        debug!(
            "[STREAM_PARSE] Found {} new SSE events (offset={}, parse_end={})",
            new_events_count,
            self.ctx.stream_body_parsed_offset,
            parse_end_pos
        );

        for event in events {
            // Skip [DONE] marker events - they don't contain JSON
            if event.data == "[DONE]" {
                continue;
            }

            // Parse as ChatStreamChunk
            match serde_json::from_str::<ChatStreamChunk>(&event.data) {
                Ok(chunk) => {
                    self.ctx.stream_chunks_parsed += 1;
                    let mut chunk = chunk;

                    // Apply provider transformation
                    self.apply_provider_transform(&mut chunk);

                    // Convert to Response API events
                    self.convert_chunk_to_events(&mut chunk, &mut converted_chunks);
                }
                Err(e) => {
                    debug!("[STREAM_PARSE] Failed to parse JSON: {}", e);
                }
            }
        }

        // Update the parse offset to avoid re-parsing on next frame
        // Use parse_end_pos (relative to unparsed_text) to calculate absolute position
        if new_events_count > 0 {
            self.ctx.stream_body_parsed_offset += parse_end_pos;
        }

        // Compact parsed prefix periodically to keep streaming memory bounded.
        if self.ctx.stream_body_parsed_offset >= crate::constants::STREAM_PARSE_COMPACT_THRESHOLD {
            self.ctx.response_body.drain(..self.ctx.stream_body_parsed_offset);
            debug!(
                "[STREAM_PARSE] compacted parsed prefix bytes={}",
                self.ctx.stream_body_parsed_offset
            );
            self.ctx.stream_body_parsed_offset = 0;
        }

        if !converted_chunks.is_empty() {
            Some(converted_chunks.join(""))
        } else {
            None
        }
    }

    /// Finalize the stream by appending response.completed event.
    ///
    /// Should be called at end_of_body when streaming conversion is enabled.
    /// Returns SSE events for the completed response, including [DONE] marker.
    pub fn finalize_stream(&mut self) -> Vec<String> {
        let mut converted_chunks: Vec<String> = Vec::new();

        if let Some(ref mut state) = self.ctx.stream_state {
            if !state.is_completed {
                if self.ctx.stream_chunks_parsed == 0 {
                    warn!(
                        "[STREAM_COMPLETE_SKIP] skip response.completed because no valid upstream chunks were parsed (status={:?}, content_type={:?})",
                        self.ctx.upstream_status,
                        self.ctx.upstream_content_type
                    );
                    state.is_completed = true;
                } else {
                    let response_obj = state.build_response_object();
                    debug!(
                        "[STREAM_COMPLETE] response_id={}, output_count={}, has_reasoning={}, has_text={}, tool_calls={}, parsed_chunks={}",
                        response_obj.id,
                        response_obj.output.len(),
                        state.is_reasoning_added,
                        state.is_output_item_added,
                        state.completed_tool_calls.len(),
                        self.ctx.stream_chunks_parsed
                    );
                    if self.log_body {
                        if let Ok(json) = serde_json::to_string(&response_obj) {
                            debug!("[STREAM_COMPLETE_JSON] {}", json);
                        }
                    }
                    let completed_event = ResponseStreamEvent::Completed {
                        response: response_obj,
                    };
                    let sse_data = event_to_sse(&completed_event);
                    converted_chunks.push(sse_data);
                    // Append SSE [DONE] marker to signal stream end
                    converted_chunks.push("data: [DONE]\n\n".to_string());
                    state.is_completed = true;
                }
            }
        }

        converted_chunks
    }

    /// Apply provider-specific transformation to stream chunk.
    fn apply_provider_transform(&mut self, chunk: &mut ChatStreamChunk) {
        if let Some(ref mut provider) = self.provider {
            provider.transform_stream_chunk(chunk);
        }
    }

    /// Convert a ChatStreamChunk to Response API events.
    fn convert_chunk_to_events(
        &mut self,
        chunk: &mut ChatStreamChunk,
        converted_chunks: &mut Vec<String>,
    ) {
        if let Some(ref mut state) = self.ctx.stream_state {
            // Update usage from this chunk
            state.update_usage(chunk);

            match chat_chunk_to_response_events(chunk, state) {
                Ok(events) => {
                    let sse_data: String = events
                        .iter()
                        .map(event_to_sse)
                        .collect();
                    if !sse_data.is_empty() {
                        debug!("[STREAM_CHUNK] {}", sse_data);
                        converted_chunks.push(sse_data);
                    }
                }
                Err(e) => {
                    error!("Failed to convert stream chunk: {}", e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration tests for StreamingResponseHandler behavior
    // are covered by proxy::filters tests
}