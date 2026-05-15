//! Message conversion utilities for Responses API → Chat API.

use crate::error::ConversionError;
use crate::types::chat_api::{
    ChatMessage, Content, ContentBlock, FunctionCall, ImageUrlField, ImageUrlObject, MessageRole,
    ToolCall,
};
use crate::types::response_api::{
    Content as ResponseContent, ContentPart, InputItemOrString, Tool,
};

/// Convert input (with optional instructions) to Chat messages.
///
/// Returns:
/// - `messages`: The converted chat messages
/// - `extracted_tools`: Tools extracted from `tool_search_output` items
pub fn convert_input_to_messages(
    input: InputItemOrString,
    instructions: Option<String>,
    enforce_tool_result_adjacency: bool,
) -> Result<(Vec<ChatMessage>, Vec<Tool>), ConversionError> {
    let mut messages = Vec::new();
    #[allow(unused_mut)]
    let mut extracted_tools: Vec<Tool> = Vec::new();
    let mut pending_tool_calls: Option<Vec<ToolCall>> = None;
    let mut emitted_tool_call_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut emitted_tool_call_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Add system message from instructions
    if let Some(inst) = instructions {
        messages.push(ChatMessage {
            role: MessageRole::System,
            content: Content::String(inst),
            name: None,
            annotations: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
            refusal: None,
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
                function_call: None,
                refusal: None,
            });
        }
        InputItemOrString::Array(items) => {
            for mut item in items {
                match item.item_type {
                    crate::types::response_api::InputItemType::Message => {
                        let role = match item.role.as_deref() {
                            Some("developer") => MessageRole::Developer,
                            Some("system") => MessageRole::System,
                            Some("assistant") => MessageRole::Assistant,
                            Some("tool") => MessageRole::Tool,
                            Some("user") | None => MessageRole::User,
                            Some("unknown") => MessageRole::Unknown,
                            Some("critic") => MessageRole::Critic,
                            Some("discriminator") => MessageRole::Discriminator,
                            Some(other) => {
                                return Err(ConversionError::InvalidFormat(format!(
                                    "unsupported message role: {other}"
                                )));
                            }
                        };

                        // tool_call_id is only valid on role=tool per Chat API spec
                        // (ChatCompletionRequestToolMessage). Avoid leaking it to other roles.
                        let tool_call_id_for_msg = if matches!(role, MessageRole::Tool) {
                            item.call_id.clone()
                        } else {
                            None
                        };

                        let content = extract_content(&item.content)?;

                        // If we have pending tool calls and now receive an assistant message,
                        // merge them into ONE assistant message. Some providers require
                        // tool outputs to immediately follow the assistant tool_calls message.
                        if enforce_tool_result_adjacency && role == MessageRole::Assistant {
                            if let Some(tool_calls) = pending_tool_calls.take() {
                                for tc in &tool_calls {
                                    emitted_tool_call_ids.insert(tc.id.clone());
                                    emitted_tool_call_names
                                        .insert(tc.id.clone(), tc.function.name.clone());
                                }
                                messages.push(ChatMessage {
                                    role,
                                    content,
                                    name: item.name,
                                    annotations: None,
                                    tool_calls: Some(tool_calls),
                                    tool_call_id: tool_call_id_for_msg.clone(),
                                    function_call: None,
                                    refusal: None,
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
                                emitted_tool_call_names
                                    .insert(tc.id.clone(), tc.function.name.clone());
                            }
                            messages.push(ChatMessage {
                                role: MessageRole::Assistant,
                                content: Content::String(String::new()),
                                name: None,
                                annotations: None,
                                tool_calls: Some(tool_calls),
                                tool_call_id: None,
                                function_call: None,
                                refusal: None,
                            });
                        }

                        messages.push(ChatMessage {
                            role,
                            content,
                            name: item.name,
                            annotations: None,
                            tool_calls: None,
                            tool_call_id: tool_call_id_for_msg,
                            function_call: None,
                            refusal: None,
                        });
                    }
                    crate::types::response_api::InputItemType::FunctionCall => {
                        // Accumulate FunctionCall items into pending_tool_calls
                        let arguments = item.arguments.unwrap_or_default();
                        let name = item
                            .name
                            .ok_or_else(|| ConversionError::MissingField("name".to_string()))?;
                        // Use call_id to match FunctionCallOutput's call_id reference
                        let id = item
                            .call_id
                            .or(item.id)
                            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

                        let tool_call = ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCall { name, arguments },
                        };

                        pending_tool_calls
                            .get_or_insert_with(Vec::new)
                            .push(tool_call);
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
                                emitted_tool_call_names
                                    .insert(tc.id.clone(), tc.function.name.clone());
                            }
                            messages.push(ChatMessage {
                                role: MessageRole::Assistant,
                                content: Content::String(String::new()),
                                name: None,
                                annotations: None,
                                tool_calls: Some(tool_calls),
                                tool_call_id: None,
                                function_call: None,
                                refusal: None,
                            });
                        }

                        // Some providers require a preceding assistant.tool_calls message
                        // before each tool result. If missing in the current input window,
                        // synthesize a minimal one to preserve protocol validity.
                        if enforce_tool_result_adjacency
                            && !emitted_tool_call_ids.contains(&call_id)
                        {
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
                                function_call: None,
                                refusal: None,
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
                            function_call: None,
                            refusal: None,
                        });
                        tracing::debug!(
                            "[REQUEST_CONVERT] emitted tool result message (call_id={}, name={})",
                            call_id,
                            output_name
                        );
                    }
                    // --- Unsupported InputItemType variants ---
                    // These are valid Response API types but not supported for Chat API conversion.
                    // Log warning with item id if available for debugging.
                    // --- Unsupported InputItemType variants ---
                    // These are valid Response API types but not supported for Chat API conversion.
                    // We skip them with a warning so the request can continue processing.
                    // These items typically represent internal/server-side features that don't
                    // affect the actual conversation flow when skipped.
                    crate::types::response_api::InputItemType::ComputerCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping computer_call input item (id={:?}), \
                            computer use feature not supported in Chat API",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ComputerCallOutput => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping computer_call_output input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::FileSearchCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping file_search_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::WebSearchCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping web_search_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::CodeInterpreterCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping code_interpreter_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::Reasoning => {
                        tracing::debug!(
                            "[REQUEST_CONVERT] skipping reasoning input item (id={:?}), \
                            reasoning items are for context but cannot be converted to Chat API format",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ToolSearchCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping tool_search_call input item (id={:?}, call_id={:?}), \
                            tool_search_call is an output item type",
                            item.id,
                            item.call_id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ToolSearchOutput => {
                        // Extract tools from tool_search_output and merge them
                        if let Some(tools) = item.tools.take() {
                            let count = tools.len();
                            extracted_tools.extend(tools);
                            tracing::debug!(
                                "[REQUEST_CONVERT] extracted {} tools from tool_search_output (id={:?})",
                                count,
                                item.id
                            );
                        } else {
                            tracing::debug!(
                                "[REQUEST_CONVERT] tool_search_output has no tools (id={:?})",
                                item.id
                            );
                        }
                        // tool_search_output is just a tool carrier - no message emitted
                        continue;
                    }
                    crate::types::response_api::InputItemType::ImageGenerationCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping image_generation_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::LocalShellCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping local_shell_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::LocalShellCallOutput => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping local_shell_call_output input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ShellCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping shell_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ShellCallOutput => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping shell_call_output input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::McpListTools => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping mcp_list_tools input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::McpApprovalRequest => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping mcp_approval_request input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::McpApprovalResponse => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping mcp_approval_response input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::McpCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping mcp_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::CustomToolCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping custom_tool_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::CustomToolCallOutput => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping custom_tool_call_output input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ApplyPatchCall => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping apply_patch_call input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::ApplyPatchCallOutput => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping apply_patch_call_output input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::Compaction => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping compaction input item (id={:?})",
                            item.id
                        );
                        continue;
                    }
                    crate::types::response_api::InputItemType::Unknown => {
                        tracing::warn!(
                            "[REQUEST_CONVERT] skipping unknown input item type (id={:?}), \
                            this may be a new type not yet supported by the proxy",
                            item.id
                        );
                        continue;
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
                    function_call: None,
                    refusal: None,
                });
            }
            tracing::debug!(
                "[REQUEST_CONVERT] input array converted: messages={}, emitted_tool_calls={}",
                messages.len(),
                emitted_tool_call_ids.len()
            );
        }
    }

    Ok((messages, extracted_tools))
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
                        input_audio: None,
                        file: None,
                        refusal: None,
                    }),
                    ContentPart::OutputText { text, .. } => blocks.push(ContentBlock {
                        block_type: "text".to_string(),
                        text: Some(text.clone()),
                        image_url: None,
                        input_audio: None,
                        file: None,
                        refusal: None,
                    }),
                    ContentPart::InputImage { image_url } => blocks.push(ContentBlock {
                        block_type: "image_url".to_string(),
                        text: None,
                        image_url: Some(ImageUrlField::Object(ImageUrlObject {
                            url: image_url.clone(),
                            detail: None,
                        })),
                        input_audio: None,
                        file: None,
                        refusal: None,
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
                            input_audio: None,
                            file: None,
                            refusal: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::response_api::{Content as ResponseContent, ContentPart};

    #[test]
    fn test_extract_content_image_url_serializes_as_object() {
        // OpenAI `ChatCompletionRequestMessageContentPartImage.image_url` is a
        // required object `{url, detail?}`, not a string.
        let content = ResponseContent::Array(vec![
            ContentPart::InputText {
                text: "see this:".into(),
            },
            ContentPart::InputImage {
                image_url: "https://example.com/x.png".into(),
            },
        ]);
        let chat_content = extract_content(&Some(content)).unwrap();
        let json = serde_json::to_value(&chat_content).unwrap();
        let arr = json.as_array().expect("array content");
        let image_block = arr
            .iter()
            .find(|b| b["type"] == "image_url")
            .expect("image_url block present");
        assert!(
            image_block["image_url"].is_object(),
            "image_url must be object: {image_block}"
        );
        assert_eq!(image_block["image_url"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn test_unknown_role_returns_error() {
        let input = InputItemOrString::Array(vec![crate::types::response_api::InputItem {
            id: None,
            item_type: crate::types::response_api::InputItemType::Message,
            role: Some("alien".to_string()),
            content: Some(ResponseContent::String("hi".into())),
            name: None,
            arguments: None,
            call_id: None,
            output: None,
            namespace: None,
            tools: None,
        }]);
        let err =
            convert_input_to_messages(input, None, false).expect_err("unknown role must fail");
        assert!(matches!(err, ConversionError::InvalidFormat(_)));
    }
}
