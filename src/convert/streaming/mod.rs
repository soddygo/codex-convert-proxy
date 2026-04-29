//! Streaming conversion: Chat SSE chunks → Responses API event sequence.
//!
//! Submodules:
//! - `events` — ResponseStreamEvent enum
//! - `state` — StreamState, ToolCallState, ResponseRequestContext
//! - `converter` — chat_chunk_to_response_events(), finalize_output()
//! - `sse_ser` — event_to_sse(), sse_event(), response_stub_json()

pub mod events;
pub mod state;
pub mod converter;
pub mod sse_ser;

pub use events::*;
pub use state::*;
pub use converter::*;
pub use sse_ser::*;
