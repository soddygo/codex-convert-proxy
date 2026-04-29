//! Response stream event types for the Responses API streaming protocol.

use crate::types::response_api::ResponseObject;

use super::state::ResponseRequestContext;

/// SSE event types for Responses API streaming.
#[derive(Debug, Clone)]
pub enum ResponseStreamEvent {
    /// Initial response created event.
    Created {
        id: String,
        model: String,
        status: String,
        created_at: i64,
        request_context: Option<ResponseRequestContext>,
    },
    /// Response is in progress.
    InProgress {
        id: String,
        model: String,
        status: String,
        created_at: i64,
        request_context: Option<ResponseRequestContext>,
    },
    /// Output item was added.
    OutputItemAdded {
        output_index: u32,
        item_id: String,
        item_type: String,
        role: Option<String>,
        call_id: Option<String>,
    },
    /// Content part was added.
    ContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    /// Output text delta (content chunk).
    OutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Output text done.
    OutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    /// Content part done.
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    /// Output item done.
    OutputItemDone {
        output_index: u32,
        item_id: String,
        item_type: String,
        role: Option<String>,
        call_id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
        text: Option<String>,
    },
    /// Reasoning output item added.
    ReasoningAdded {
        output_index: u32,
        item_id: String,
    },
    /// Reasoning text delta.
    ReasoningDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Reasoning text done.
    ReasoningTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    /// Reasoning summary text delta.
    ReasoningSummaryTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Reasoning summary text done.
    ReasoningSummaryTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    /// Function call arguments delta.
    FunctionCallArgumentsDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// Function call arguments done.
    FunctionCallArgumentsDone {
        output_index: u32,
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// Response completed with final object.
    Completed {
        response: Box<ResponseObject>,
    },
    /// Response error event.
    Error {
        id: Option<String>,
        error_type: String,
        message: String,
        code: Option<String>,
    },
    /// Response failed event.
    Failed {
        id: String,
        model: String,
        status: String,
        created_at: i64,
    },
    /// Response incomplete event.
    Incomplete {
        id: String,
        model: String,
        status: String,
        created_at: i64,
        reason: Option<String>,
    },
    /// Refusal content delta.
    RefusalDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Refusal content done.
    RefusalDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        refusal: String,
    },
}
