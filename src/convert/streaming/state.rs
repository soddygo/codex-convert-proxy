//! Streaming state types: StreamState and ToolCallState.
//!
//! `StreamState` was historically a single struct with ~30 fields covering
//! seven distinct concerns. It is now decomposed into single-responsibility
//! sub-structs (still `pub` for convenience at the call sites — the grouping
//! itself documents intent and lifecycle):
//!
//! - [`TextAccumulator`] — text/refusal/reasoning content + the thinking-tag
//!   buffer used by the streaming parser.
//! - [`IndexAllocator`] — `output_index` / `content_index` allocator state and
//!   the spec-required `sequence_number` counter.
//! - [`UsageMetrics`] — token counts received from upstream.
//! - [`ToolCallTracker`] — in-flight (`current`) vs finalised (`completed`)
//!   function-call state.
//! - [`EmitState`] — boolean flags that drive the streaming emit state
//!   machine + the final response status / incomplete reason.
//!
//! `request_context` no longer lives on `StreamState`; callers pass it as a
//! parameter to [`StreamState::build_response_object`] so that non-streaming
//! flows don't have to construct a `StreamState` just to carry context.

use crate::types::chat_api::ChatStreamChunk;
use crate::convert::context::ResponseRequestContext;

use super::super::util::{extract_queries_from_arguments, map_tool_name_to_output_type};

/// Accumulated text content emitted across stream chunks.
#[derive(Debug, Clone, Default)]
pub struct TextAccumulator {
    /// Assistant text seen so far (with thinking tags stripped).
    pub full_text: String,
    /// Refusal text emitted via `delta.refusal`.
    pub refusal_text: String,
    /// Reasoning content captured from `delta.reasoning_content` or thinking
    /// tags.
    pub reasoning_text: String,
    /// Partial buffer for unterminated `<think>` / `<thought>` tags split
    /// across chunks.
    pub thinking_buffer: String,
    /// True if the next text byte belongs to a thinking tag's interior.
    pub is_thinking: bool,
}

/// Output-index / content-index / sequence-number allocators for the
/// stream's lifetime.
#[derive(Debug, Clone, Default)]
pub struct IndexAllocator {
    pub content_index: u32,
    pub next_output_index: u32,
    pub text_output_index: Option<u32>,
    pub reasoning_output_index: Option<u32>,
    /// Monotonic sequence counter for SSE events (spec-required
    /// `sequence_number`).
    pub next_sequence_number: u64,
}

impl IndexAllocator {
    /// Allocate and return the next sequence number, then advance the counter.
    pub fn take_sequence_number(&mut self) -> u64 {
        let n = self.next_sequence_number;
        self.next_sequence_number += 1;
        n
    }
}

/// Token usage observed from upstream chunks.
#[derive(Debug, Clone, Default)]
pub struct UsageMetrics {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
}

impl UsageMetrics {
    /// Absorb token counts from a streaming chunk's usage block, if present.
    pub fn update_from_chunk(&mut self, chunk: &ChatStreamChunk) {
        if let Some(usage) = &chunk.usage {
            self.input_tokens = usage.prompt_tokens.map(|v| v as i64);
            self.output_tokens = usage.completion_tokens.map(|v| v as i64);
            self.total_tokens = usage.total_tokens.map(|v| v as i64);
            self.cached_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .map(|v| v as i64);
            self.reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .map(|v| v as i64);
        }
    }
}

/// In-flight + finalised function-call state.
#[derive(Debug, Clone, Default)]
pub struct ToolCallTracker {
    pub current: Vec<ToolCallState>,
    pub completed: Vec<ToolCallState>,
}

/// Booleans driving the streaming emit state machine.
#[derive(Debug, Clone)]
pub struct EmitState {
    pub is_first_chunk: bool,
    pub is_output_item_added: bool,
    pub is_content_part_added: bool,
    pub is_reasoning_added: bool,
    pub is_function_call_item_added: bool,
    pub is_completed: bool,
    /// Final response status derived from `finish_reason`.
    pub final_status: String,
    /// Optional incomplete reason when `final_status == "incomplete"`.
    pub incomplete_reason: Option<String>,
}

impl Default for EmitState {
    fn default() -> Self {
        Self {
            is_first_chunk: true,
            is_output_item_added: false,
            is_content_part_added: false,
            is_reasoning_added: false,
            is_function_call_item_added: false,
            is_completed: false,
            final_status: "completed".to_string(),
            incomplete_reason: None,
        }
    }
}

/// Streaming converter state for tracking incremental changes.
#[derive(Debug, Clone)]
pub struct StreamState {
    pub response_id: String,
    pub output_id: String,
    pub model: String,

    pub text: TextAccumulator,
    pub indices: IndexAllocator,
    pub usage: UsageMetrics,
    pub tool_calls: ToolCallTracker,
    pub emit: EmitState,

}

#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub upstream_id: Option<String>,
    pub id: String,
    pub call_id: String,
    pub item_type: String,
    pub name: String,
    pub arguments: String,
    pub output_index: u32,
    pub chat_api_index: u32,
}

impl StreamState {
    /// Create a new stream state.
    pub fn new(response_id: String, model: String) -> Self {
        Self {
            output_id: format!("msg_{}", response_id),
            response_id,
            model,
            text: TextAccumulator::default(),
            indices: IndexAllocator::default(),
            usage: UsageMetrics::default(),
            tool_calls: ToolCallTracker::default(),
            emit: EmitState::default(),
        }
    }

    /// Allocate and return the next sequence number, then advance the counter.
    #[inline]
    pub fn take_sequence_number(&mut self) -> u64 {
        self.indices.take_sequence_number()
    }

    /// Update usage from a ChatStreamChunk.
    #[inline]
    pub fn update_usage(&mut self, chunk: &ChatStreamChunk) {
        self.usage.update_from_chunk(chunk);
    }

    /// Build the final ResponseObject with all accumulated outputs.
    ///
    /// The `request_context` parameter supplies request-level fields
    /// (instructions, tools, sampling params, etc.).
    pub fn build_response_object(
        &self,
        request_context: Option<&ResponseRequestContext>,
    ) -> Box<crate::types::response_api::ResponseObject> {
        use crate::types::response_api::{
            InputTokensDetails, OutputItemType, OutputTokensDetails, ResponseContentPart,
            ResponseObject, ResponseOutputItem, Usage,
        };
        use chrono::Utc;

        let mut output = Vec::new();

        // Add reasoning output if present
        if self.emit.is_reasoning_added && !self.text.reasoning_text.is_empty() {
            output.push(ResponseOutputItem {
                id: format!("reasoning_{}", self.response_id),
                item_type: OutputItemType::Reasoning,
                status: None,
                content: Some(vec![]),
                summary: Some(vec![
                    crate::types::response_api::ReasoningSummaryPart::SummaryText {
                        text: self.text.reasoning_text.clone(),
                    },
                ]),
                role: None,
                name: None,
                arguments: None,
                call_id: None,
                queries: None,
                results: None,
                namespace: None,
            });
        }

        // Add assistant message output (text and/or refusal)
        if self.emit.is_output_item_added
            && (!self.text.full_text.is_empty() || !self.text.refusal_text.is_empty())
        {
            let mut content_parts = Vec::new();
            if !self.text.full_text.is_empty() {
                content_parts.push(ResponseContentPart::OutputText {
                    text: self.text.full_text.clone(),
                    annotations: vec![],
                    logprobs: vec![],
                });
            }
            if !self.text.refusal_text.is_empty() {
                content_parts.push(ResponseContentPart::Refusal {
                    refusal: self.text.refusal_text.clone(),
                });
            }
            output.push(ResponseOutputItem {
                id: self.output_id.clone(),
                item_type: OutputItemType::Message,
                status: Some("completed".to_string()),
                content: Some(content_parts),
                role: Some("assistant".to_string()),
                name: None,
                arguments: None,
                call_id: None,
                queries: None,
                results: None,
                summary: None,
                namespace: None,
            });
        }

        // Add function call outputs
        for tc in &self.tool_calls.completed {
            let item_type =
                map_tool_name_to_output_type(&tc.name, request_context.map(|ctx| &ctx.tools));
            let (queries, results) = if item_type != OutputItemType::FunctionCall {
                (
                    extract_queries_from_arguments(&tc.arguments),
                    Some(serde_json::Value::Null),
                )
            } else {
                (None, None)
            };
            output.push(ResponseOutputItem {
                id: tc.id.clone(),
                item_type,
                status: Some("completed".to_string()),
                content: None,
                role: None,
                name: Some(tc.name.clone()),
                arguments: Some(tc.arguments.clone()),
                call_id: Some(tc.call_id.clone()),
                queries,
                results,
                summary: None,
                namespace: None,
            });
        }

        // Start from the typed stub so request-level fields and spec-required
        // defaults are populated consistently, then layer in the streamed
        // output + final status + usage.
        let mut response = ResponseObject::stub(
            self.response_id.clone(),
            self.model.clone(),
            self.emit.final_status.clone(),
            Utc::now().timestamp(),
            request_context,
        );
        response.completed_at = Some(Utc::now().timestamp());
        response.incomplete_details = self
            .emit
            .incomplete_reason
            .as_ref()
            .map(|reason| serde_json::json!({ "reason": reason }));
        response.output = output;
        response.usage = if self.usage.input_tokens.is_some()
            || self.usage.output_tokens.is_some()
            || self.usage.total_tokens.is_some()
        {
            Some(Usage {
                input_tokens: self.usage.input_tokens,
                input_tokens_details: Some(InputTokensDetails {
                    cached_tokens: self.usage.cached_tokens.unwrap_or(0),
                }),
                output_tokens: self.usage.output_tokens,
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: self.usage.reasoning_tokens.unwrap_or(0),
                }),
                total_tokens: self.usage.total_tokens,
            })
        } else {
            None
        };
        Box::new(response)
    }
}
