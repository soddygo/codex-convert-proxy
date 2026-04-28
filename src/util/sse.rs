//! SSE (Server-Sent Events) parsing and serialization utilities.
//!
//! This module provides utilities for parsing SSE format which is used
//! in streaming responses.

use bytes::{Bytes, BytesMut};

/// SSE event with optional event type and data.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type (e.g., "response.created", "response.output_text.delta")
    pub event_type: Option<String>,
    /// Event data (JSON string)
    pub data: String,
}

/// SSE parse error.
#[derive(Debug, Clone)]
pub enum SseParseError {
    UnterminatedJson,
    MissingDelimiter,
    InvalidUtf8,
}

impl std::fmt::Display for SseParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SseParseError::UnterminatedJson => write!(f, "unterminated JSON object"),
            SseParseError::MissingDelimiter => write!(f, "missing SSE event delimiter"),
            SseParseError::InvalidUtf8 => write!(f, "invalid UTF-8"),
        }
    }
}

/// Iterator over SSE events in a text stream.
pub struct SseEventIterator<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> SseEventIterator<'a> {
    /// Create a new iterator over SSE events.
    pub fn new(text: &'a str) -> Self {
        Self { text, position: 0 }
    }

    /// Get the current position in the text.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Skip past an incomplete event at current position.
    /// Returns the new position, or text.len() if already at end.
    pub fn skip_incomplete_event(&mut self) -> usize {
        // Try to find \n\n after current position to skip to
        if let Some(next_delim) = self.text[self.position..].find("\n\n") {
            self.position += next_delim + 2;
        } else {
            // No \n\n found, skip to end of text
            self.position = self.text.len();
        }
        self.position
    }

    /// Parse the next SSE event from the text.
    /// Returns None if no more events, or Some(Err) on parse error.
    pub fn next_event(&mut self) -> Option<Result<SseEvent, SseParseError>> {
        let base_pos = self.position;
        let text = &self.text[base_pos..];

        // Find "data:" prefix (with or without trailing space)
        let data_start = match text.find("data:") {
            Some(pos) => pos,
            None => return None,
        };

        // Look for "event:" line before this "data:" line
        let event_type = if data_start > 0 {
            // Get the text before "data:"
            let before_data = &text[..data_start];
            // Look for "event:" in the preceding lines (search backwards)
            let mut result = None;
            for line in before_data.lines().rev() {
                let line = line.trim();
                if let Some(stripped) = line.strip_prefix("event:") {
                    result = Some(stripped.trim().to_string());
                    break;
                }
                // If we hit a non-empty, non-event line, stop looking
                if !line.is_empty() && !line.starts_with("event:") {
                    break;
                }
            }
            result
        } else {
            None
        };

        let after_prefix = data_start + 5; // after "data:"
        if after_prefix >= text.len() {
            return None;
        }

        let mut value_start = after_prefix;
        while value_start < text.len() {
            let c = text.as_bytes()[value_start];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                value_start += 1;
            } else {
                break;
            }
        }
        if value_start >= text.len() {
            return None;
        }

        let rest = &text[value_start..];

        // Handle "[DONE]"
        if rest.starts_with("[DONE]") {
            self.position = base_pos + value_start + 6;
            // Skip \n\n if present
            if self.position + 2 <= self.text.len()
                && &self.text[self.position..self.position + 2] == "\n\n"
            {
                self.position += 2;
            }
            return Some(Ok(SseEvent {
                event_type: None, // [DONE] doesn't have an event type
                data: "[DONE]".to_string(),
            }));
        }

        // Find JSON end by brace matching
        let json_end = match find_json_end(rest) {
            Some(pos) => pos,
            None => return Some(Err(SseParseError::UnterminatedJson)),
        };

        // Find \n\n SSE delimiter after JSON
        let after_json = &rest[json_end..];
        match after_json.find("\n\n") {
            Some(delimiter_pos) => {
                let json_content = &rest[..json_end];
                self.position = base_pos + value_start + json_end + delimiter_pos + 2;
                Some(Ok(SseEvent {
                    event_type,
                    data: json_content.to_string(),
                }))
            }
            None => Some(Err(SseParseError::MissingDelimiter)),
        }
    }
}

/// Parse SSE format text into events.
///
/// SSE format example:
/// - `event: response.created\ndata: {"id": "resp_123"}`
/// - `event: response.output_text.delta\ndata: {"delta": "hello"}`
///
/// Events are separated by double newlines (`\n\n`).
///
/// Returns (events, parse_end_position) where parse_end_position is the byte
/// offset in `text` where parsing stopped (either at end of text, or after
/// skipping an incomplete event).
pub fn parse_sse(text: &str) -> (Vec<SseEvent>, usize) {
    let mut events = Vec::new();
    let mut iter = SseEventIterator::new(text);

    while let Some(result) = iter.next_event() {
        match result {
            Ok(event) => events.push(event),
            Err(e) => {
                tracing::debug!("SSE parse error: {}, skipping incomplete event", e);
                // Skip past the incomplete event so we don't re-parse it
                iter.skip_incomplete_event();
                break;
            }
        }
    }

    // If we parsed all events without error, position will be at end of text
    let end_pos = if events.len() > 0 && iter.position() >= text.len() {
        text.len()
    } else {
        iter.position()
    };

    (events, end_pos)
}

/// Find the end of a JSON object/array in text, handling nested braces
/// and escaped characters in strings.
fn find_json_end(text: &str) -> Option<usize> {
    let mut brace_depth = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, c) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_string => {
                escaped = true;
            }
            '"' => {
                in_string = !in_string;
            }
            '{' | '[' if !in_string => {
                brace_depth += 1;
            }
            '}' | ']' if !in_string => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    return Some(i + c.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

/// Serialize a single SSE event to string format.
pub fn serialize_sse(event: &SseEvent) -> String {
    let mut result = String::new();

    if let Some(ref et) = event.event_type {
        result.push_str("event: ");
        result.push_str(et);
        result.push('\n');
    }

    result.push_str("data: ");
    result.push_str(&event.data);
    result.push_str("\n\n");

    result
}

/// Collect body frames into a single Bytes buffer.
///
/// This is a helper function to aggregate multiple body chunks
/// into one continuous buffer for easier SSE parsing.
pub fn collect_frames(frames: &[Bytes]) -> Bytes {
    if frames.is_empty() {
        return Bytes::new();
    }

    if frames.len() == 1 {
        return frames[0].clone();
    }

    // Calculate total length
    let total_len: usize = frames.iter().map(|f| f.len()).sum();

    let mut result = BytesMut::with_capacity(total_len);
    for frame in frames {
        result.extend_from_slice(frame);
    }

    result.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_event() {
        let text = "data: {\"type\": \"response.created\", \"id\": \"resp_123\"}\n\n";
        let (events, _) = parse_sse(text);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, None);
        assert_eq!(
            events[0].data,
            "{\"type\": \"response.created\", \"id\": \"resp_123\"}"
        );
    }

    #[test]
    fn test_parse_event_with_type() {
        let text = "event: response.created\ndata: {\"id\": \"resp_123\"}\n\n";
        let (events, _) = parse_sse(text);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].event_type,
            Some("response.created".to_string())
        );
        assert_eq!(events[0].data, "{\"id\": \"resp_123\"}");
    }

    #[test]
    fn test_parse_event_without_space_after_data_colon() {
        let text = "event: response.created\ndata:{\"id\":\"resp_123\"}\n\n";
        let (events, _) = parse_sse(text);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, Some("response.created".to_string()));
        assert_eq!(events[0].data, "{\"id\":\"resp_123\"}");
    }

    #[test]
    fn test_parse_event_with_newline_after_data_colon() {
        let text = "event: response.created\ndata:\n{\"id\":\"resp_123\"}\n\n";
        let (events, _) = parse_sse(text);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, Some("response.created".to_string()));
        assert_eq!(events[0].data, "{\"id\":\"resp_123\"}");
    }

    #[test]
    fn test_parse_done_event() {
        // [DONE] event has no event type, just data: [DONE]
        // The [DONE] string goes into the data field, event_type is None
        let text = "data: [DONE]\n\n";
        let (events, _) = parse_sse(text);

        assert_eq!(events.len(), 1);
        // event_type is None because there's no "event:" line in SSE
        assert_eq!(events[0].event_type, None);
        // data contains "[DONE]"
        assert_eq!(events[0].data, "[DONE]");
    }

    #[test]
    fn test_parse_multiple_events() {
        let text = "event: response.created\ndata: {\"id\": \"1\"}\n\nevent: response.output_text.delta\ndata: {\"delta\": \"hello\"}\n\n";
        let (events, _) = parse_sse(text);

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0].event_type,
            Some("response.created".to_string())
        );
        assert_eq!(
            events[1].event_type,
            Some("response.output_text.delta".to_string())
        );
    }

    #[test]
    fn test_parse_empty_data() {
        let text = "event: done\ndata: \n\n";
        let (events, _) = parse_sse(text);

        // Empty data should not create an event
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_serialize_sse() {
        let event = SseEvent {
            event_type: Some("response.created".to_string()),
            data: "{\"id\": \"resp_123\"}".to_string(),
        };

        let result = serialize_sse(&event);
        assert!(result.contains("event: response.created\n"));
        assert!(result.contains("data: {\"id\": \"resp_123\"}\n\n"));
    }

    #[test]
    fn test_collect_frames_empty() {
        let frames: [Bytes; 0] = [];
        let result = collect_frames(&frames);
        assert!(result.is_empty());
    }

    #[test]
    fn test_collect_frames_single() {
        let frames = vec![Bytes::from("hello")];
        let result = collect_frames(&frames);
        assert_eq!(&result[..], b"hello");
    }

    #[test]
    fn test_collect_frames_multiple() {
        let frames = vec![
            Bytes::from("hello"),
            Bytes::from(" world"),
            Bytes::from("!"),
        ];
        let result = collect_frames(&frames);
        assert_eq!(&result[..], b"hello world!");
    }

    #[test]
    fn test_find_json_end() {
        // Simple object
        assert_eq!(find_json_end(r#"{"key": "value"}"#), Some(16));

        // Nested object
        assert_eq!(
            find_json_end(r#"{"outer": {"inner": "value"}}"#),
            Some(29)
        );

        // Array - "[1, 2, 3]" is 9 chars
        assert_eq!(find_json_end(r#"[1, 2, 3]"#), Some(9));

        // Empty
        assert_eq!(find_json_end(""), None);

        // Unterminated
        assert_eq!(find_json_end(r#"{"key": "value"#), None);
    }
}
