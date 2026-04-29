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
            // SSE standard: "event:" and "data:" can have optional space after colon
            if let Some(rest) = line.strip_prefix("event:") {
                event_type = Some(rest.trim_start().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest.trim_start());
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
    let trimmed = data.trim();
    trimmed == "[DONE]" || trimmed == "data: [DONE]" || trimmed.starts_with("data:[DONE]")
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

    #[test]
    fn test_parse_sse_without_space() {
        // Test parsing with no space after colon
        let data = b"event:test\ndata:{\"id\":\"123\"}\n\ndata:{\"id\":\"456\"}\n\n";
        let chunks = parse_sse_chunks(data).unwrap();
        
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].event_type.as_deref(), Some("test"));
        assert_eq!(chunks[0].data, "{\"id\":\"123\"}");
        assert_eq!(chunks[1].event_type, None);
        assert_eq!(chunks[1].data, "{\"id\":\"456\"}");
    }
    
    #[test]
    fn test_is_sse_done_no_space() {
        assert!(is_sse_done("data:[DONE]"));  // no space
        assert!(is_sse_done("data: [DONE] "));
    }
