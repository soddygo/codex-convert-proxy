//! Request conversion: Responses API → Chat API.

use crate::error::ConversionError;
use crate::providers::Provider;
use crate::types::chat_api::{
    ChatMessage, ChatRequest, ChatTool, ChatToolChoice, ChatToolChoiceMode, Content, ContentBlock,
    FunctionCall, FunctionChoice, FunctionDefinition, MessageRole, StreamOptions, ToolCall,
};
use crate::types::response_api::{
    Content as ResponseContent, ContentPart, InputItemOrString,
    ResponseRequest, Tool as ResponseTool, ToolChoice as ResponseToolChoice,
    ToolType as ResponseToolType,
};
use tracing::{debug, warn};

/// Convert a Responses API request to a Chat API request.
pub fn response_to_chat(
    response_req: ResponseRequest,
    provider: &mut dyn Provider,
    model_override: Option<&str>,
) -> Result<ChatRequest, ConversionError> {
    let messages = convert_input_to_messages(response_req.input, response_req.instructions)?;
    let tools = convert_tools(response_req.tools);
    let tool_choice = convert_tool_choice(response_req.tool_choice);

    // Use model from config if specified, otherwise use provider's model normalization
    let model = model_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| provider.normalize_model(response_req.model));

    // Apply provider-specific transformations
    let mut chat_req = ChatRequest {
        model,
        messages,
        tools: Some(tools).filter(|t| !t.is_empty()),
        tool_choice: Some(tool_choice).filter(|tc| !tc.is_none()),
        stream: Some(response_req.stream),
        temperature: response_req.temperature,
        max_tokens: response_req.max_output_tokens.or(response_req.max_tokens),
        top_p: response_req.top_p,
        user: response_req.user,
        stream_options: if response_req.stream {
            Some(StreamOptions { include_usage: Some(true) })
        } else {
            None
        },
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
            annotations: None,
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
                annotations: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        InputItemOrString::Array(items) => {
            let mut pending_tool_calls: Option<Vec<ToolCall>> = None;
            let mut emitted_tool_call_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            for item in items {
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

                        // If we have pending tool calls and now receive an assistant message,
                        // merge them into ONE assistant message. Some providers require
                        // tool outputs to immediately follow the assistant tool_calls message.
                        if role == MessageRole::Assistant {
                            if let Some(tool_calls) = pending_tool_calls.take() {
                                for tc in &tool_calls {
                                    emitted_tool_call_ids.insert(tc.id.clone());
                                }
                                messages.push(ChatMessage {
                                    role,
                                    content,
                                    name: item.name,
                                    annotations: None,
                                    tool_calls: Some(tool_calls),
                                    tool_call_id: item.call_id,
                                });
                                debug!(
                                    "[REQUEST_CONVERT] merged assistant message with pending tool_calls to keep tool result adjacency"
                                );
                                continue;
                            }
                        } else if let Some(tool_calls) = pending_tool_calls.take() {
                            // Flush pending tool calls before non-assistant message items
                            for tc in &tool_calls {
                                emitted_tool_call_ids.insert(tc.id.clone());
                            }
                            messages.push(ChatMessage {
                                role: MessageRole::Assistant,
                                content: Content::String(String::new()),
                                name: None,
                                annotations: None,
                                tool_calls: Some(tool_calls),
                                tool_call_id: None,
                            });
                        }

                        messages.push(ChatMessage {
                            role,
                            content,
                            name: item.name,
                            annotations: None,
                            tool_calls: None,
                            tool_call_id: item.call_id,
                        });
                    }
                    crate::types::response_api::InputItemType::FunctionCall => {
                        // Accumulate FunctionCall items into pending_tool_calls
                        let arguments = item.arguments.unwrap_or_default();
                        let name = item
                            .name
                            .ok_or_else(|| ConversionError::MissingField("name".to_string()))?;
                        // Use call_id to match FunctionCallOutput's call_id reference
                        let id = item.call_id.or(item.id).unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

                        let tool_call = ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCall { name, arguments },
                        };

                        pending_tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    }
                    crate::types::response_api::InputItemType::FunctionCallOutput => {
                        let call_id = item
                            .call_id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));
                        let output_name = item
                            .name
                            .clone()
                            .unwrap_or_else(|| "tool_call".to_string());

                        // Flush pending tool calls before FunctionCallOutput
                        if let Some(tool_calls) = pending_tool_calls.take() {
                            for tc in &tool_calls {
                                emitted_tool_call_ids.insert(tc.id.clone());
                            }
                            messages.push(ChatMessage {
                                role: MessageRole::Assistant,
                                content: Content::String(String::new()),
                                name: None,
                                annotations: None,
                                tool_calls: Some(tool_calls),
                                tool_call_id: None,
                            });
                        }

                        // Some providers require a preceding assistant.tool_calls message
                        // before each tool result. If missing in the current input window,
                        // synthesize a minimal one to preserve protocol validity.
                        if !emitted_tool_call_ids.contains(&call_id) {
                            warn!(
                                "[REQUEST_CONVERT] function_call_output without preceding function_call, synthesizing assistant tool_call (call_id={}, name={})",
                                call_id,
                                output_name
                            );
                            let synthetic_tool_call = ToolCall {
                                id: call_id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCall {
                                    name: output_name.clone(),
                                    arguments: "{}".to_string(),
                                },
                            };
                            messages.push(ChatMessage {
                                role: MessageRole::Assistant,
                                content: Content::String(String::new()),
                                name: None,
                                annotations: None,
                                tool_calls: Some(vec![synthetic_tool_call]),
                                tool_call_id: None,
                            });
                            emitted_tool_call_ids.insert(call_id.clone());
                        }

                        messages.push(ChatMessage {
                            role: MessageRole::Tool,
                            content: Content::String(item.output.unwrap_or_default()),
                            name: item.name,
                            annotations: None,
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                        debug!(
                            "[REQUEST_CONVERT] emitted tool result message (call_id={}, name={})",
                            call_id,
                            output_name
                        );
                    }
                }
            }

            // Handle any remaining tool calls
            if let Some(tool_calls) = pending_tool_calls {
                for tc in &tool_calls {
                    emitted_tool_call_ids.insert(tc.id.clone());
                }
                messages.push(ChatMessage {
                    role: MessageRole::Assistant,
                    content: Content::String(String::new()),
                    name: None,
                    annotations: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            }
            debug!(
                "[REQUEST_CONVERT] input array converted: messages={}, emitted_tool_calls={}",
                messages.len(),
                emitted_tool_call_ids.len()
            );
        }
    }

    Ok(messages)
}

/// Extract text content from Response API content.
fn extract_content(content: &Option<ResponseContent>) -> Result<Content, ConversionError> {
    match content {
        Some(ResponseContent::String(s)) => Ok(Content::String(s.clone())),
        Some(ResponseContent::Array(parts)) => {
            let mut blocks: Vec<ContentBlock> = Vec::new();
            for part in parts {
                match part {
                    ContentPart::InputText { text } => blocks.push(ContentBlock {
                        block_type: "text".to_string(),
                        text: Some(text.clone()),
                        image_url: None,
                    }),
                    ContentPart::OutputText { text, .. } => blocks.push(ContentBlock {
                        block_type: "text".to_string(),
                        text: Some(text.clone()),
                        image_url: None,
                    }),
                    ContentPart::InputImage { image_url } => blocks.push(ContentBlock {
                        block_type: "image_url".to_string(),
                        text: None,
                        image_url: Some(image_url.clone().into()),
                    }),
                    ContentPart::InputFile { file_url, file_id } => {
                        let file_ref = file_url
                            .as_ref()
                            .or(file_id.as_ref())
                            .cloned()
                            .unwrap_or_else(|| "unknown_file".to_string());
                        blocks.push(ContentBlock {
                            block_type: "text".to_string(),
                            text: Some(format!("[input_file] {}", file_ref)),
                            image_url: None,
                        });
                    }
                }
            }

            if blocks.is_empty() {
                Ok(Content::String(String::new()))
            } else if blocks.len() == 1 && blocks[0].block_type == "text" {
                Ok(Content::String(blocks[0].text.clone().unwrap_or_default()))
            } else {
                Ok(Content::Array(blocks))
            }
        }
        None => Ok(Content::String(String::new())),
    }
}

/// Convert Responses API tools to Chat API tools.
fn convert_tools(tools: Vec<ResponseTool>) -> Vec<ChatTool> {
    tools
        .into_iter()
        .filter_map(|t| match t.tool_type {
            ResponseToolType::Function => Some(ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_default(),
                    description: t.description,
                    parameters: t.parameters,
                },
            }),
            // Convert built-in tools to function tools with default parameters
            ResponseToolType::WebSearchPreview => Some(ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name.unwrap_or_else(|| "web_search_preview".to_string()),
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
            }),
            ResponseToolType::CodeInterpreter => Some(ChatTool {
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
            }),
            ResponseToolType::FileSearch => Some(ChatTool {
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
            }),
            // Namespace tools are not convertible to Chat API format - skip them
            ResponseToolType::Namespace => None,
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
    use std::collections::HashMap;

    use super::*;
    use crate::providers::glm::GLMProvider;
    use crate::types::response_api::{InputItem, InputItemType, Tool, ToolType};

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
            max_output_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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
            max_output_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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
            max_output_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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

    #[test]
    fn test_orphan_function_call_output_synthesizes_preceding_tool_call() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::Array(vec![InputItem {
                id: None,
                item_type: InputItemType::FunctionCallOutput,
                role: None,
                content: None,
                name: Some("get_weather".to_string()),
                arguments: None,
                call_id: Some("call_orphan".to_string()),
                output: Some("sunny".to_string()),
            }]),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert_eq!(chat_req.messages.len(), 2);

        let assistant = &chat_req.messages[0];
        assert_eq!(assistant.role, MessageRole::Assistant);
        let tc = assistant
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .expect("synthetic tool call should exist");
        assert_eq!(tc.id, "call_orphan");
        assert_eq!(tc.function.name, "get_weather");

        let tool_msg = &chat_req.messages[1];
        assert_eq!(tool_msg.role, MessageRole::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_orphan"));
        assert_eq!(tool_msg.content.as_text(), "sunny");
    }

    #[test]
    fn test_assistant_message_merges_with_pending_tool_calls() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::Array(vec![
                InputItem {
                    id: Some("fc_1".to_string()),
                    item_type: InputItemType::FunctionCall,
                    role: None,
                    content: None,
                    name: Some("exec_command".to_string()),
                    arguments: Some(r#"{"cmd":"ls"}"#.to_string()),
                    call_id: Some("call_1".to_string()),
                    output: None,
                },
                InputItem {
                    id: Some("msg_1".to_string()),
                    item_type: InputItemType::Message,
                    role: Some("assistant".to_string()),
                    content: Some(ResponseContent::String("我先看下目录".to_string())),
                    name: None,
                    arguments: None,
                    call_id: None,
                    output: None,
                },
                InputItem {
                    id: Some("fco_1".to_string()),
                    item_type: InputItemType::FunctionCallOutput,
                    role: None,
                    content: None,
                    name: Some("exec_command".to_string()),
                    arguments: None,
                    call_id: Some("call_1".to_string()),
                    output: Some("ok".to_string()),
                },
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

        assert_eq!(chat_req.messages.len(), 2);
        let assistant = &chat_req.messages[0];
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert_eq!(assistant.content.as_text(), "我先看下目录");
        let tc = assistant
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .expect("assistant should carry merged tool call");
        assert_eq!(tc.id, "call_1");

        let tool = &chat_req.messages[1];
        assert_eq!(tool.role, MessageRole::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.content.as_text(), "ok");
    }

    #[test]
    fn test_max_output_tokens_maps_to_chat_max_tokens() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::String("Hello".to_string()),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_output_tokens: Some(8),
            max_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert_eq!(chat_req.max_tokens, Some(16));
    }

    #[test]
    fn test_web_search_preview_tool_degrades_to_function() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::String("Hello".to_string()),
            instructions: None,
            tools: vec![Tool {
                tool_type: ToolType::WebSearchPreview,
                name: None,
                description: None,
                parameters: None,
                strict: None,
                extra: HashMap::new(),
            }],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_output_tokens: None,
            max_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = crate::providers::kimi::KimiProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        let tools = chat_req.tools.unwrap_or_default();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "web_search_preview");
    }

    #[test]
    fn test_input_file_is_not_dropped() {
        let request = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::Array(vec![InputItem {
                id: None,
                item_type: InputItemType::Message,
                role: Some("user".to_string()),
                content: Some(ResponseContent::Array(vec![
                    ContentPart::InputText {
                        text: "Analyze file".to_string(),
                    },
                    ContentPart::InputFile {
                        file_url: Some("https://example.com/file.pdf".to_string()),
                        file_id: None,
                    },
                ])),
                name: None,
                arguments: None,
                call_id: None,
                output: None,
            }]),
            instructions: None,
            tools: vec![],
            tool_choice: ResponseToolChoice::Auto,
            stream: false,
            temperature: None,
            max_output_tokens: None,
            max_tokens: None,
            top_p: None,
            user: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            parallel_tool_calls: None,
        };

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert!(!chat_req.messages.is_empty());
        let body = chat_req.messages[0].content.as_text();
        assert!(body.contains("[input_file]"));
        assert!(body.contains("file.pdf"));
    }
}
