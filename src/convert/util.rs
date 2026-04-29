//! Shared conversion utilities used by both streaming and non-streaming paths.

use crate::constants::MAX_THINKING_BUFFER_SIZE;
use crate::types::response_api::{OutputItemType, Tool, ToolType};

use super::streaming::ResponseRequestContext;

/// Map a tool name to its `OutputItemType` using the original tools list.
pub fn map_tool_name_to_output_type(
    tool_name: &str,
    original_tools: Option<&Vec<Tool>>,
) -> OutputItemType {
    if let Some(tools) = original_tools {
        for t in tools {
            if t.name.as_deref() == Some(tool_name) {
                return match t.tool_type {
                    ToolType::WebSearchPreview => OutputItemType::WebSearchCall,
                    ToolType::FileSearch => OutputItemType::FileSearchCall,
                    _ => OutputItemType::FunctionCall,
                };
            }
        }
    }
    match tool_name {
        "web_search_preview" | "web_search" => OutputItemType::WebSearchCall,
        "file_search" => OutputItemType::FileSearchCall,
        _ => OutputItemType::FunctionCall,
    }
}

/// Map a tool name to its stream item type string using request context.
pub fn map_tool_name_to_stream_item_type(
    tool_name: &str,
    request_context: Option<&ResponseRequestContext>,
) -> String {
    let tools = request_context.map(|ctx| &ctx.tools);
    match map_tool_name_to_output_type(tool_name, tools) {
        OutputItemType::WebSearchCall => "web_search_call".to_string(),
        OutputItemType::FileSearchCall => "file_search_call".to_string(),
        _ => "function_call".to_string(),
    }
}

/// Extract query/queries from JSON arguments string.
pub fn extract_queries_from_arguments(arguments: &str) -> Option<Vec<String>> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) {
        if let Some(query) = value.get("query").and_then(|v| v.as_str()) {
            return Some(vec![query.to_string()]);
        }
        if let Some(queries) = value.get("queries").and_then(|v| v.as_array()) {
            let qs: Vec<String> = queries
                .iter()
                .filter_map(|q| q.as_str().map(|s| s.to_string()))
                .collect();
            if !qs.is_empty() {
                return Some(qs);
            }
        }
    }
    None
}

/// Parse thinking tags from a complete text string (non-streaming).
///
/// Supports both `<thought>...</thought>` and `<think>...</think>` tags.
/// Returns (actual_content, reasoning_text).
pub fn parse_thought_tags(content: &str) -> (String, Option<String>) {
    let mut actual_content = String::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut remaining = content;

    loop {
        let thought_start = remaining.find("<thought>");
        let think_start = remaining.find("<think>");

        let (start_idx, open_tag, close_tag) = match (thought_start, think_start) {
            (Some(t), Some(k)) => {
                if t <= k {
                    (t, "<thought>", "</thought>")
                } else {
                    (k, "<think>", "</think>")
                }
            }
            (Some(t), None) => (t, "<thought>", "</thought>"),
            (None, Some(k)) => (k, "<think>", "</think>"),
            (None, None) => break,
        };

        actual_content.push_str(&remaining[..start_idx]);

        let after_start = &remaining[start_idx + open_tag.len()..];
        if let Some(end_idx) = after_start.find(close_tag) {
            let reasoning_content = &after_start[..end_idx];
            if !reasoning_content.is_empty() {
                reasoning_parts.push(reasoning_content.to_string());
            }
            remaining = &after_start[end_idx + close_tag.len()..];
        } else {
            actual_content.push_str(&remaining[start_idx..]);
            remaining = "";
            break;
        }
    }

    actual_content.push_str(remaining);

    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n\n"))
    };

    (actual_content.trim().to_string(), reasoning)
}

/// Parse thinking tags from streaming content.
///
/// Handles `<think>...</think>` and `<thought>...</thought>` tags that may be
/// split across multiple chunks. Returns (actual_text, reasoning_delta, new_is_thinking).
pub fn parse_streaming_thinking(
    text: &str,
    is_thinking: bool,
    buffer: &mut String,
) -> (String, Option<String>, bool) {
    let mut actual_text = String::new();
    let mut reasoning = String::new();
    let mut current_is_thinking = is_thinking;

    buffer.push_str(text);

    if buffer.len() > MAX_THINKING_BUFFER_SIZE {
        let flushed = buffer.clone();
        buffer.clear();
        return (String::new(), Some(flushed), false);
    }

    let full_content = buffer.clone();
    buffer.clear();

    let mut pos = 0;
    let chars: Vec<char> = full_content.chars().collect();

    while pos < chars.len() {
        if current_is_thinking {
            let think_close = find_pattern(&chars, pos, &['<', '/', 't', 'h', 'i', 'n', 'k', '>']);
            let thought_close = find_pattern(&chars, pos, &['<', '/', 't', 'h', 'o', 'u', 'g', 'h', 't', '>']);

            match (think_close, thought_close) {
                (Some(close_pos), Some(thought_close_pos)) => {
                    if close_pos <= thought_close_pos {
                        let content: String = chars[pos..close_pos].iter().collect();
                        reasoning.push_str(&content);
                        pos = close_pos + 8;
                        current_is_thinking = false;
                    } else {
                        let content: String = chars[pos..thought_close_pos].iter().collect();
                        reasoning.push_str(&content);
                        pos = thought_close_pos + 10;
                        current_is_thinking = false;
                    }
                }
                (Some(close_pos), None) => {
                    let content: String = chars[pos..close_pos].iter().collect();
                    reasoning.push_str(&content);
                    pos = close_pos + 8;
                    current_is_thinking = false;
                }
                (None, Some(thought_close_pos)) => {
                    let content: String = chars[pos..thought_close_pos].iter().collect();
                    reasoning.push_str(&content);
                    pos = thought_close_pos + 10;
                    current_is_thinking = false;
                }
                (None, None) => {
                    let remaining: String = chars[pos..].iter().collect();
                    buffer.push_str(&remaining);
                    break;
                }
            }
        } else {
            let think_open = find_pattern(&chars, pos, &['<', 't', 'h', 'i', 'n', 'k', '>']);
            let thought_open = find_pattern(&chars, pos, &['<', 't', 'h', 'o', 'u', 'g', 'h', 't', '>']);

            match (think_open, thought_open) {
                (Some(open_pos), Some(thought_open_pos)) => {
                    if open_pos <= thought_open_pos {
                        let content: String = chars[pos..open_pos].iter().collect();
                        actual_text.push_str(&content);
                        pos = open_pos + 7;
                        current_is_thinking = true;
                    } else {
                        let content: String = chars[pos..thought_open_pos].iter().collect();
                        actual_text.push_str(&content);
                        pos = thought_open_pos + 9;
                        current_is_thinking = true;
                    }
                }
                (Some(open_pos), None) => {
                    let content: String = chars[pos..open_pos].iter().collect();
                    actual_text.push_str(&content);
                    pos = open_pos + 7;
                    current_is_thinking = true;
                }
                (None, Some(thought_open_pos)) => {
                    let content: String = chars[pos..thought_open_pos].iter().collect();
                    actual_text.push_str(&content);
                    pos = thought_open_pos + 9;
                    current_is_thinking = true;
                }
                (None, None) => {
                    let remaining: String = chars[pos..].iter().collect();
                    actual_text.push_str(&remaining);
                    break;
                }
            }
        }
    }

    let reasoning_delta = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };

    (actual_text, reasoning_delta, current_is_thinking)
}

/// Find a pattern in char array starting from pos.
pub fn find_pattern(chars: &[char], start: usize, pattern: &[char]) -> Option<usize> {
    if start + pattern.len() > chars.len() {
        return None;
    }
    for i in start..=chars.len() - pattern.len() {
        if chars[i..i + pattern.len()] == *pattern {
            return Some(i);
        }
    }
    None
}
/// Escape pseudo XML tool tags that some upstream models emit as plain text.
pub fn sanitize_pseudo_tool_markup(text: &str) -> String {
    text.replace("<request_user_input", "&lt;request_user_input")
        .replace("</request_user_input>", "&lt;/request_user_input&gt;")
        .replace("<parameter ", "&lt;parameter ")
        .replace("</parameter>", "&lt;/parameter&gt;")
}
