//! Response conversion: Chat API → Responses API.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatResponse, Content};
use crate::types::response_api::{
    OutputItemType, ResponseContentPart, ResponseObject, ResponseOutputItem, Usage,
};

/// Convert a Chat API response to a Responses API ResponseObject.
pub fn chat_to_response(chat_resp: ChatResponse) -> Result<ResponseObject, ConversionError> {
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| ConversionError::MissingField("choices".to_string()))?;

    let mut outputs = Vec::new();

    // Convert message content (strip thinking tags)
    if let Some(content) = extract_content(&choice.message.content) {
        let (actual_content, reasoning) = parse_thought_tags(&content);

        // Add reasoning output if present
        if let Some(ref reasoning_text) = reasoning {
            if !reasoning_text.is_empty() {
                outputs.push(ResponseOutputItem {
                    id: format!("reasoning_{}", chat_resp.id),
                    item_type: OutputItemType::Reasoning,
                    status: Some("completed".to_string()),
                    content: Some(vec![ResponseContentPart::OutputText {
                        text: reasoning_text.clone(),
                    }]),
                    name: None,
                    arguments: None,
                    call_id: None,
                });
            }
        }

        // Add text output if present (after stripping thinking tags)
        if !actual_content.is_empty() {
            outputs.push(ResponseOutputItem {
                id: format!("msg_{}", chat_resp.id),
                item_type: OutputItemType::Message,
                status: Some("completed".to_string()),
                content: Some(vec![ResponseContentPart::OutputText { text: actual_content }]),
                name: None,
                arguments: None,
                call_id: None,
            });
        }
    }

    // Convert tool calls
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            outputs.push(ResponseOutputItem {
                id: tc.id.clone(),
                item_type: OutputItemType::FunctionCall,
                status: Some("completed".to_string()),
                content: None,
                name: Some(tc.function.name.clone()),
                arguments: Some(tc.function.arguments.clone()),
                call_id: Some(tc.id.clone()),
            });
        }
    }

    let usage = chat_resp.usage.map(|u| Usage {
        input_tokens: u.prompt_tokens.map(|t| t as i64),
        output_tokens: u.completion_tokens.map(|t| t as i64),
        total_tokens: u.total_tokens.map(|t| t as i64),
    });

    Ok(ResponseObject {
        id: format!("resp_{}", chat_resp.id),
        object: "response".to_string(),
        status: "completed".to_string(),
        model: chat_resp.model,
        created_at: chat_resp.created as i64,
        completed_at: Some(chrono::Utc::now().timestamp()),
        input: None,  // Input not available in non-streaming context
        output: outputs,
        usage,
    })
}

/// Parse thinking tags from content and extract reasoning text.
///
/// Supports both `<thought>...</thought>` and `<think>...</think>` tags
/// (MiniMax uses `<think>`, OpenAI-compatible models use `<thought>`).
///
/// Returns (actual_content, reasoning_text) where reasoning_text is the content
/// inside thinking tags, and actual_content is everything else.
pub fn parse_thought_tags(content: &str) -> (String, Option<String>) {
    let mut actual_content = String::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut remaining = content;

    // Try both tag formats
    loop {
        // Find the next opening tag (either <thought> or <think>)
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

        // Add content before the tag to actual_content
        actual_content.push_str(&remaining[..start_idx]);

        // Find the closing tag
        let after_start = &remaining[start_idx + open_tag.len()..];
        if let Some(end_idx) = after_start.find(close_tag) {
            // Extract reasoning content
            let reasoning_content = &after_start[..end_idx];
            if !reasoning_content.is_empty() {
                reasoning_parts.push(reasoning_content.to_string());
            }
            // Continue with content after closing tag
            remaining = &after_start[end_idx + close_tag.len()..];
        } else {
            // No closing tag found, treat rest as actual content
            actual_content.push_str(&remaining[start_idx..]);
            remaining = "";
            break;
        }
    }

    // Add any remaining content
    actual_content.push_str(remaining);

    let reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n\n"))
    };

    (actual_content.trim().to_string(), reasoning)
}

/// Extract text content from a ChatMessage.
fn extract_content(content: &Content) -> Option<String> {
    let text = match content {
        Content::String(s) => {
            if s.is_empty() {
                return None;
            }
            s.clone()
        }
        Content::Array(arr) => {
            let text: String = arr
                .iter()
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                return None;
            }
            text
        }
    };
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatChoice, ChatMessage, Content, MessageRole};

    #[test]
    fn test_basic_response_conversion() {
        let chat_resp = ChatResponse {
            id: "chat_123".to_string(),
            object_name: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: MessageRole::Assistant,
                    content: Content::String("Hello, how can I help you?".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(crate::types::chat_api::ChatUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
            }),
        };

        let response = chat_to_response(chat_resp).unwrap();

        assert_eq!(response.status, "completed");
        assert!(!response.output.is_empty());

        let msg_output = response.output.first().unwrap();
        assert_eq!(msg_output.item_type, OutputItemType::Message);

        let text = msg_output.content.as_ref().and_then(|c| c.first());
        match text {
            Some(ResponseContentPart::OutputText { text }) => {
                assert_eq!(text, "Hello, how can I help you?");
            }
            _ => panic!("Expected output text"),
        }

        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
    }

    #[test]
    fn test_tool_call_conversion() {
        let chat_resp = ChatResponse {
            id: "chat_123".to_string(),
            object_name: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: MessageRole::Assistant,
                    content: Content::String(String::new()),
                    name: None,
                    tool_calls: Some(vec![crate::types::chat_api::ToolCall {
                        id: "call_abc".to_string(),
                        tool_type: "function".to_string(),
                        function: crate::types::chat_api::FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"city":"Beijing"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let response = chat_to_response(chat_resp).unwrap();

        // Should have function call output
        let func_output = response
            .output
            .iter()
            .find(|o| o.item_type == OutputItemType::FunctionCall);
        assert!(func_output.is_some());

        let func = func_output.unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some(r#"{"city":"Beijing"}"#));
    }

    #[test]
    fn test_parse_thought_tags() {
        // No thought tags - should return original content
        let (content, reasoning) = parse_thought_tags("Hello world");
        assert_eq!(content, "Hello world");
        assert!(reasoning.is_none());

        // Single thought tag
        let (content, reasoning) = parse_thought_tags("<thought>I should search</thought>Hello world");
        assert_eq!(content, "Hello world");
        assert_eq!(reasoning, Some("I should search".to_string()));

        // Multiple thought tags - reasoning parts are joined with newlines
        let (content, reasoning) = parse_thought_tags(
            "<thought>Step 1: analyze</thought>Result1<thought>Step 2: conclude</thought>Final answer"
        );
        assert_eq!(content, "Result1Final answer");
        assert_eq!(reasoning, Some("Step 1: analyze\n\nStep 2: conclude".to_string()));

        // Unclosed thought tag
        let (content, reasoning) = parse_thought_tags("<thought>unclosed Hello");
        assert_eq!(content, "<thought>unclosed Hello");
        assert!(reasoning.is_none());

        // Content before and after thought
        let (content, reasoning) = parse_thought_tags("Hello<thought>reasoning</thought>World");
        assert_eq!(content, "HelloWorld");
        assert_eq!(reasoning, Some("reasoning".to_string()));
    }

    #[test]
    fn test_parse_think_tags() {
        // MiniMax uses <think> tags instead of <thought>
        let (content, reasoning) = parse_thought_tags("<think>\n分析当前目录\n</think>\n\n让我看看项目");
        assert_eq!(content, "让我看看项目");
        assert_eq!(reasoning, Some("\n分析当前目录\n".to_string()));

        // Multiple think tags
        let (content, reasoning) = parse_thought_tags(
            "<think>Step 1</think>Result1<think>Step 2</think>Final"
        );
        assert_eq!(content, "Result1Final");
        assert_eq!(reasoning, Some("Step 1\n\nStep 2".to_string()));

        // Mixed tags (shouldn't happen but test robustness)
        let (content, reasoning) = parse_thought_tags("<thought>A</thought>B<think>C</think>D");
        assert_eq!(content, "BD");
        assert_eq!(reasoning, Some("A\n\nC".to_string()));

        // Empty think tag
        let (content, reasoning) = parse_thought_tags("<think>Hello");
        assert_eq!(content, "<think>Hello");
        assert!(reasoning.is_none());
    }
}
