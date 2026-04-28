//! Utility modules.

pub mod sse;

pub use sse::{parse_sse, serialize_sse, collect_frames, SseEvent};
