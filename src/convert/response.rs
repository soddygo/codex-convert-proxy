//! Response conversion: Chat API → Responses API.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatMessage, ChatResponse, Content};
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

    // Handle reasoning content (GLM extension)
    let _reasoning_text = extract_reasoning_content(&choice.message);

    // Convert message content
    if let Some(content) = extract_content(&choice.message.content) {
        outputs.push(ResponseOutputItem {
            id: format!("msg_{}", chat_resp.id),
            item_type: OutputItemType::Message,
            status: Some("completed".to_string()),
            content: Some(vec![ResponseContentPart::OutputText { text: content }]),
            name: None,
            arguments: None,
            call_id: None,
        });
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
        output: outputs,
        usage,
    })
}

/// Extract reasoning content from message (GLM extension).
fn extract_reasoning_content(_message: &ChatMessage) -> Option<String> {
    // GLM uses reasoning_content field which is not part of standard Chat API
    // We need to check if the message has this via reflection
    None
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
}
