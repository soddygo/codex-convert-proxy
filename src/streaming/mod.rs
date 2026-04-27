//! SSE parsing and generation utilities.

use crate::error::ConversionError;

/// Parse SSE data into individual chunks.
pub fn parse_sse_chunks(data: &[u8]) -> Result<Vec<SseChunk>, ConversionError> {
    let text = String::from_utf8_lossy(data);
    let mut chunks = Vec::new();

    for line_group in text.split("\n\n") {
        let line_group = line_group.trim();
        if line_group.is_empty() {
            continue;
        }

        let mut event_type = None;
        let mut data = String::new();

        for line in line_group.lines() {
            if line.starts_with("event: ") {
                event_type = Some(line[7..].to_string());
            } else if line.starts_with("data: ") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(&line[6..]);
            }
        }

        if !data.is_empty() || event_type.is_some() {
            chunks.push(SseChunk {
                event_type,
                data,
            });
        }
    }

    Ok(chunks)
}

/// A parsed SSE chunk.
#[derive(Debug, Clone)]
pub struct SseChunk {
    pub event_type: Option<String>,
    pub data: String,
}

/// Check if data represents the SSE "done" marker.
pub fn is_sse_done(data: &str) -> bool {
    data.trim() == "[DONE]" || data.trim() == "data: [DONE]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_chunks() {
        let data = b"event: response.created\ndata: {\"id\":\"123\"}\n\ndata: {\"id\":\"456\"}\n\n";
        let chunks = parse_sse_chunks(data).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].event_type.as_deref(), Some("response.created"));
        assert_eq!(chunks[0].data, "{\"id\":\"123\"}");
        assert_eq!(chunks[1].event_type, None);
        assert_eq!(chunks[1].data, "{\"id\":\"456\"}");
    }

    #[test]
    fn test_is_sse_done() {
        assert!(is_sse_done("[DONE]"));
        assert!(is_sse_done("data: [DONE]"));
        assert!(is_sse_done(" [DONE] "));
        assert!(!is_sse_done("{\"id\":\"123\"}"));
    }
}
