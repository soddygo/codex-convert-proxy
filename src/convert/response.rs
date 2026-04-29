//! Response conversion: Chat API → Responses API.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatMessageAnnotation, ChatResponse, Content};
use crate::types::response_api::{
    InputTokensDetails, OutputItemType, OutputTokensDetails, ResponseAnnotation, ResponseContentPart, ResponseObject,
    ResponseOutputItem, ResponseTextConfig, ResponseTextFormat, Usage,
};
use crate::convert::streaming::ResponseRequestContext;
use super::util::{extract_queries_from_arguments, map_tool_name_to_output_type, parse_thought_tags};

/// Convert a Chat API response to a Responses API ResponseObject.
pub fn chat_to_response(chat_resp: ChatResponse) -> Result<ResponseObject, ConversionError> {
    chat_to_response_with_context(chat_resp, None)
}

/// Convert a Chat API response to a Responses API ResponseObject with optional request context.
pub fn chat_to_response_with_context(
    chat_resp: ChatResponse,
    request_context: Option<&ResponseRequestContext>,
) -> Result<ResponseObject, ConversionError> {
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| ConversionError::MissingField("choices".to_string()))?;
    let mapped_annotations = choice
        .message
        .annotations
        .as_ref()
        .map(|annotations| {
            annotations
                .iter()
                .filter_map(|anno| match anno {
                    ChatMessageAnnotation::UrlCitation {
                        start_index,
                        end_index,
                        url,
                        title,
                    } => Some(ResponseAnnotation::UrlCitation {
                        start_index: *start_index,
                        end_index: *end_index,
                        url: url.clone(),
                        title: title.clone(),
                    }),
                    ChatMessageAnnotation::FileCitation {
                        index,
                        file_id,
                        filename,
                    } => Some(ResponseAnnotation::FileCitation {
                        index: *index,
                        file_id: file_id.clone(),
                        filename: filename.clone(),
                    }),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();


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
                        annotations: mapped_annotations.clone(),
                    }]),
                    role: None,
                    name: None,
                    arguments: None,
                    call_id: None,
                    queries: None,
                    results: None,
                });
            }
        }

        // Add text output if present (after stripping thinking tags)
        if !actual_content.is_empty() {
            outputs.push(ResponseOutputItem {
                id: format!("msg_{}", chat_resp.id),
                item_type: OutputItemType::Message,
                status: Some("completed".to_string()),
                content: Some(vec![ResponseContentPart::OutputText {
                    text: actual_content,
                    annotations: mapped_annotations.clone(),
                }]),
                role: Some("assistant".to_string()),
                name: None,
                arguments: None,
                call_id: None,
                queries: None,
                results: None,
            });
        }
    }

    // Convert tool calls
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tc in tool_calls {
            let mapped_type = map_tool_name_to_output_type(
                &tc.function.name,
                request_context.map(|ctx| &ctx.tools),
            );
            let (queries, results) = if mapped_type != OutputItemType::FunctionCall {
                (extract_queries_from_arguments(&tc.function.arguments), Some(serde_json::Value::Null))
            } else {
                (None, None)
            };

            outputs.push(ResponseOutputItem {
                id: format!("fc_{}", tc.id),
                item_type: mapped_type,
                status: Some("completed".to_string()),
                content: None,
                role: None,
                name: Some(tc.function.name.clone()),
                arguments: Some(tc.function.arguments.clone()),
                call_id: Some(tc.id.clone()),
                queries,
                results,
            });
        }
    }

    let usage = chat_resp.usage.map(|u| Usage {
        input_tokens: u.prompt_tokens.map(|t| t as i64),
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .map(|v| v as i64)
                .unwrap_or(0),
        }),
        output_tokens: u.completion_tokens.map(|t| t as i64),
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .map(|v| v as i64)
                .unwrap_or(0),
        }),
        total_tokens: u.total_tokens.map(|t| t as i64),
    });

    let default_text = Some(ResponseTextConfig {
        format: Some(ResponseTextFormat {
            format_type: "text".to_string(),
        }),
    });

    Ok(ResponseObject {
        id: format!("resp_{}", chat_resp.id),
        object: "response".to_string(),
        status: "completed".to_string(),
        model: chat_resp.model,
        created_at: chat_resp.created as i64,
        completed_at: Some(chrono::Utc::now().timestamp()),
        error: None,
        incomplete_details: None,
        instructions: request_context.and_then(|ctx| ctx.instructions.clone()),
        max_output_tokens: request_context.and_then(|ctx| ctx.max_output_tokens),
        max_tool_calls: None,
        input: None,  // Input not available in non-streaming context
        output: outputs,
        parallel_tool_calls: request_context.and_then(|ctx| ctx.parallel_tool_calls),
        previous_response_id: request_context.and_then(|ctx| ctx.previous_response_id.clone()),
        reasoning: request_context.and_then(|ctx| ctx.reasoning.clone()),
        store: request_context.and_then(|ctx| ctx.store),
        temperature: request_context.and_then(|ctx| ctx.temperature),
        text: request_context.and_then(|ctx| ctx.text.clone()).or(default_text),
        tool_choice: request_context.map(|ctx| ctx.tool_choice.clone()),
        tools: request_context.map(|ctx| ctx.tools.clone()),
        top_p: request_context.and_then(|ctx| ctx.top_p),
        truncation: request_context.and_then(|ctx| ctx.truncation.clone()),
        user: request_context.and_then(|ctx| ctx.user.clone()),
        metadata: request_context.and_then(|ctx| ctx.metadata.clone()),
        usage,
    })
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
    use crate::types::chat_api::{
        ChatChoice, ChatMessage, ChatMessageAnnotation, CompletionTokensDetails, Content, MessageRole,
        PromptTokensDetails,
    };
    use crate::types::response_api::{InputItemOrString, ResponseRequest, Tool, ToolChoice, ToolType};
    use std::collections::HashMap;

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
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(crate::types::chat_api::ChatUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            service_tier: None,
            system_fingerprint: None,
        };

        let response = chat_to_response(chat_resp).unwrap();

        assert_eq!(response.status, "completed");
        assert!(!response.output.is_empty());

        let msg_output = response.output.first().unwrap();
        assert_eq!(msg_output.item_type, OutputItemType::Message);

        let text = msg_output.content.as_ref().and_then(|c| c.first());
        match text {
            Some(ResponseContentPart::OutputText { text, .. }) => {
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
    fn test_annotation_and_usage_details_mapping() {
        let chat_resp = ChatResponse {
            id: "chat_anno".to_string(),
            object_name: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: MessageRole::Assistant,
                    content: Content::String("参考来源".to_string()),
                    name: None,
                    annotations: Some(vec![ChatMessageAnnotation::UrlCitation {
                        start_index: 0,
                        end_index: 4,
                        url: "https://example.com".to_string(),
                        title: "Example".to_string(),
                    }]),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(crate::types::chat_api::ChatUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: Some(3),
                }),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: Some(7),
                }),
            }),
            service_tier: None,
            system_fingerprint: None,
        };

        let response = chat_to_response(chat_resp).unwrap();
        let content = response.output[0].content.as_ref().unwrap();
        match &content[0] {
            ResponseContentPart::OutputText { annotations, .. } => {
                assert!(!annotations.is_empty());
            }
            _ => panic!("expected output text"),
        }
        let usage = response.usage.unwrap();
        assert_eq!(
            usage.input_tokens_details.unwrap().cached_tokens,
            3
        );
        assert_eq!(
            usage.output_tokens_details.unwrap().reasoning_tokens,
            7
        );
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
                    annotations: None,
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
            service_tier: None,
            system_fingerprint: None,
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
    fn test_builtin_tool_call_roundtrip_type_mapping() {
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
                    annotations: None,
                    tool_calls: Some(vec![crate::types::chat_api::ToolCall {
                        id: "call_web".to_string(),
                        tool_type: "function".to_string(),
                        function: crate::types::chat_api::FunctionCall {
                            name: "web_search_preview".to_string(),
                            arguments: r#"{"query":"news"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
            service_tier: None,
            system_fingerprint: None,
        };

        let req = ResponseRequest {
            model: "gpt-4o".to_string(),
            input: InputItemOrString::String("hi".to_string()),
            instructions: None,
            tools: vec![Tool {
                tool_type: ToolType::WebSearchPreview,
                name: Some("web_search_preview".to_string()),
                description: None,
                parameters: None,
                strict: None,
                extra: HashMap::new(),
            }],
            tool_choice: ToolChoice::Auto,
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
        let ctx = crate::convert::streaming::ResponseRequestContext::from(&req);
        let response = chat_to_response_with_context(chat_resp, Some(&ctx)).unwrap();

        let web = response
            .output
            .iter()
            .find(|o| o.item_type == OutputItemType::WebSearchCall)
            .expect("should map to web_search_call");
        assert_eq!(web.call_id.as_deref(), Some("call_web"));
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
