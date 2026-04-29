//! Message conversion utilities for Responses API → Chat API.

use crate::error::ConversionError;
use crate::types::chat_api::{
    ChatMessage, Content, ContentBlock, FunctionCall, MessageRole, ToolCall,
};
use crate::types::response_api::{
    Content as ResponseContent, ContentPart, InputItemOrString,
};

/// Convert input (with optional instructions) to Chat messages.
pub fn convert_input_to_messages(
    input: InputItemOrString,
    instructions: Option<String>,
    enforce_tool_result_adjacency: bool,
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
            let mut emitted_tool_call_names: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();

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
                        if enforce_tool_result_adjacency && role == MessageRole::Assistant {
                            if let Some(tool_calls) = pending_tool_calls.take() {
                                for tc in &tool_calls {
                                    emitted_tool_call_ids.insert(tc.id.clone());
                                    emitted_tool_call_names.insert(tc.id.clone(), tc.function.name.clone());
                                }
                                messages.push(ChatMessage {
                                    role,
                                    content,
                                    name: item.name,
                                    annotations: None,
                                    tool_calls: Some(tool_calls),
                                    tool_call_id: item.call_id,
                                });
                                tracing::debug!(
                                    "[REQUEST_CONVERT] merged assistant message with pending tool_calls to keep tool result adjacency"
                                );
                                continue;
                            }
                        } else if let Some(tool_calls) = pending_tool_calls.take() {
                            // Flush pending tool calls before non-assistant message items
                            for tc in &tool_calls {
                                emitted_tool_call_ids.insert(tc.id.clone());
                                emitted_tool_call_names.insert(tc.id.clone(), tc.function.name.clone());
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
                            .or_else(|| emitted_tool_call_names.get(&call_id).cloned())
                            .unwrap_or_else(|| "unknown_tool".to_string());

                        // Flush pending tool calls before FunctionCallOutput
                        if let Some(tool_calls) = pending_tool_calls.take() {
                            for tc in &tool_calls {
                                emitted_tool_call_ids.insert(tc.id.clone());
                                emitted_tool_call_names.insert(tc.id.clone(), tc.function.name.clone());
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
                        if enforce_tool_result_adjacency && !emitted_tool_call_ids.contains(&call_id) {
                            tracing::warn!(
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
                        tracing::debug!(
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
                    emitted_tool_call_names.insert(tc.id.clone(), tc.function.name.clone());
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
            tracing::debug!(
                "[REQUEST_CONVERT] input array converted: messages={}, emitted_tool_calls={}",
                messages.len(),
                emitted_tool_call_ids.len()
            );
        }
    }

    Ok(messages)
}

/// Extract text content from Response API content.
pub fn extract_content(content: &Option<ResponseContent>) -> Result<Content, ConversionError> {
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
