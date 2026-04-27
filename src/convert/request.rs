//! Request conversion: Responses API → Chat API.

use crate::error::ConversionError;
use crate::providers::Provider;
use crate::types::chat_api::{
    ChatMessage, ChatRequest, ChatTool, ChatToolChoice, ChatToolChoiceMode, Content, ContentBlock,
    FunctionCall, FunctionChoice, FunctionDefinition, MessageRole, ToolCall,
};
use crate::types::response_api::{
    Content as ResponseContent, ContentPart, InputItemOrString,
    ResponseRequest, Tool as ResponseTool, ToolChoice as ResponseToolChoice,
    ToolType as ResponseToolType,
};

/// Convert a Responses API request to a Chat API request.
pub fn response_to_chat(
    response_req: ResponseRequest,
    provider: &dyn Provider,
) -> Result<ChatRequest, ConversionError> {
    let messages = convert_input_to_messages(response_req.input, response_req.instructions)?;
    let tools = convert_tools(response_req.tools);
    let tool_choice = convert_tool_choice(response_req.tool_choice);

    // Apply provider-specific transformations
    let mut chat_req = ChatRequest {
        model: provider.normalize_model(response_req.model),
        messages,
        tools: Some(tools).filter(|t| !t.is_empty()),
        tool_choice: Some(tool_choice).filter(|tc| !tc.is_none()),
        stream: Some(response_req.stream),
        temperature: response_req.temperature,
        max_tokens: response_req.max_tokens,
        top_p: response_req.top_p,
        user: response_req.user,
        stream_options: None,
    };

    // Apply min_tokens floor validation (some providers reject max_tokens < 16)
    if let Some(max_tokens) = chat_req.max_tokens {
        if max_tokens < 16 {
            chat_req.max_tokens = Some(16);
        }
    }

    provider.transform_request(&mut chat_req);

    Ok(chat_req)
}

/// Convert input (with optional instructions) to Chat messages.
fn convert_input_to_messages(
    input: InputItemOrString,
    instructions: Option<String>,
) -> Result<Vec<ChatMessage>, ConversionError> {
    let mut messages = Vec::new();

    // Add system message from instructions
    if let Some(inst) = instructions {
        messages.push(ChatMessage {
            role: MessageRole::System,
            content: Content::String(inst),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert input items
    match input {
        InputItemOrString::String(s) => {
            messages.push(ChatMessage {
                role: MessageRole::User,
                content: Content::String(s),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        InputItemOrString::Array(items) => {
            let mut pending_tool_calls: Option<Vec<ToolCall>> = None;

            for item in items {
                // If previous item had tool_calls, attach them to an assistant message
                if let Some(tool_calls) = pending_tool_calls.take() {
                    messages.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: Content::String(String::new()),
                        name: None,
                        tool_calls: Some(tool_calls),
                        tool_call_id: None,
                    });
                }

                match item.item_type {
                    crate::types::response_api::InputItemType::Message => {
                        let role = match item.role.as_deref() {
                            Some("developer") => MessageRole::Developer,
                            Some("system") => MessageRole::System,
                            Some("assistant") => MessageRole::Assistant,
                            Some("tool") => MessageRole::Tool,
                            _ => MessageRole::User,
                        };

                        let content = extract_content(&item.content)?;
                        messages.push(ChatMessage {
                            role,
                            content,
                            name: item.name,
                            tool_calls: None,
                            tool_call_id: item.call_id,
                        });
                    }
                    crate::types::response_api::InputItemType::FunctionCall => {
                        let arguments = item.arguments.unwrap_or_default();
                        let name = item
                            .name
                            .ok_or_else(|| ConversionError::MissingField("name".to_string()))?;
                        let id = item.id.unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

                        let tool_call = ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCall { name, arguments },
                        };

                        pending_tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    }
                    crate::types::response_api::InputItemType::FunctionCallOutput => {
                        messages.push(ChatMessage {
                            role: MessageRole::Tool,
                            content: Content::String(item.output.unwrap_or_default()),
                            name: item.name,
                            tool_calls: None,
                            tool_call_id: item.call_id,
                        });
                    }
                }
            }

            // Handle any remaining tool calls
            if let Some(tool_calls) = pending_tool_calls {
                messages.push(ChatMessage {
                    role: MessageRole::Assistant,
                    content: Content::String(String::new()),
                    name: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            }
        }
    }

    Ok(messages)
}

/// Extract text content from Response API content.
fn extract_content(content: &Option<ResponseContent>) -> Result<Content, ConversionError> {
    match content {
        Some(ResponseContent::String(s)) => Ok(Content::String(s.clone())),
        Some(ResponseContent::Array(parts)) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::InputText { text } => Some(text.clone()),
                    ContentPart::InputImage { .. } => None,
                    ContentPart::OutputText { text } => Some(text.clone()),
                })
                .collect();

            if texts.len() == 1 {
                Ok(Content::String(texts[0].clone()))
            } else {
                Ok(Content::Array(
                    texts.into_iter()
                        .map(|text| ContentBlock {
                            block_type: "text".to_string(),
                            text: Some(text),
                            image_url: None,
                        })
                        .collect(),
                ))
            }
        }
        None => Ok(Content::String(String::new())),
    }
}

/// Convert Responses API tools to Chat API tools.
fn convert_tools(tools: Vec<ResponseTool>) -> Vec<ChatTool> {
    tools
        .into_iter()
        .map(|t| match t.tool_type {
            ResponseToolType::Function => ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_default(),
                    description: t.description,
                    parameters: t.parameters,
                },
            },
            // Convert built-in tools to function tools with default parameters
            ResponseToolType::WebSearch => ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_else(|| "web_search".to_string()),
                    description: t.description.or_else(|| Some("Web search tool".to_string())),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    })),
                },
            },
            ResponseToolType::CodeInterpreter => ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_else(|| "code_interpreter".to_string()),
                    description: t
                        .description
                        .or_else(|| Some("Code interpreter tool".to_string())),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code to execute"
                            }
                        },
                        "required": ["code"]
                    })),
                },
            },
            ResponseToolType::FileSearch => ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_else(|| "file_search".to_string()),
                    description: t.description.or_else(|| Some("File search tool".to_string())),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    })),
                },
            },
        })
        .collect()
}

/// Convert Responses API tool choice to Chat API tool choice.
fn convert_tool_choice(choice: ResponseToolChoice) -> ChatToolChoice {
    match choice {
        ResponseToolChoice::Auto => ChatToolChoice::Mode(ChatToolChoiceMode::Auto),
        ResponseToolChoice::None => ChatToolChoice::Mode(ChatToolChoiceMode::None),
        ResponseToolChoice::Required => ChatToolChoice::Mode(ChatToolChoiceMode::Required),
        ResponseToolChoice::Function(f) => ChatToolChoice::Function(FunctionChoice { name: f.name }),
    }
}

impl ChatToolChoice {
    /// Check if the tool choice is "none" (no tools).
    pub fn is_none(&self) -> bool {
        match self {
            ChatToolChoice::Mode(mode) => *mode == ChatToolChoiceMode::None,
            ChatToolChoice::Function(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::glm::GLMProvider;
    use crate::types::response_api::{InputItem, InputItemType};

    #[test]
    fn test_instructions_to_system_message() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::String("Hello".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_tokens: None,
            top_p: None,
            user: None,
        };

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider).unwrap();

        // First message should be system
        let first = chat_req.messages.first().unwrap();
        assert_eq!(first.role, MessageRole::System);
        assert_eq!(first.content.as_text(), "You are a helpful assistant.");

        // Second should be user
        let second = chat_req.messages.get(1).unwrap();
        assert_eq!(second.role, MessageRole::User);
        assert_eq!(second.content.as_text(), "Hello");
    }

    #[test]
    fn test_function_call_conversion() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::Array(vec![InputItem {
                id: Some("call_123".to_string()),
                item_type: InputItemType::FunctionCall,
                role: None,
                content: None,
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                call_id: None,
                output: None,
            }]),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_tokens: None,
            top_p: None,
            user: None,
        };

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider).unwrap();

        // Should have an assistant message with tool_calls
        let msg = chat_req.messages.first().unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
        assert!(msg.tool_calls.is_some());

        let tc = msg.tool_calls.as_ref().unwrap().first().unwrap();
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn test_function_call_output() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::Array(vec![
                InputItem {
                    id: Some("call_123".to_string()),
                    item_type: InputItemType::FunctionCall,
                    role: None,
                    content: None,
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                    call_id: None,
                    output: None,
                },
                InputItem {
                    id: None,
                    item_type: InputItemType::FunctionCallOutput,
                    role: None,
                    content: None,
                    name: Some("get_weather".to_string()),
                    arguments: None,
                    call_id: Some("call_123".to_string()),
                    output: Some("25 degrees, sunny".to_string()),
                },
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_tokens: None,
            top_p: None,
            user: None,
        };

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider).unwrap();

        // Should have assistant message with tool_calls and tool message
        assert_eq!(chat_req.messages.len(), 2);

        let assistant = &chat_req.messages[0];
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert!(assistant.tool_calls.is_some());

        let tool_msg = &chat_req.messages[1];
        assert_eq!(tool_msg.role, MessageRole::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(tool_msg.content.as_text(), "25 degrees, sunny");
    }
}
