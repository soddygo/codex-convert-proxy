//! Request conversion: Responses API → Chat API.

mod context;
mod messages;
mod tools;

pub use context::{ToolPriority, ToolSearchContext};
pub use messages::{convert_input_to_messages, extract_content};
pub use tools::{convert_tools, convert_tool_choice};

use crate::constants::MIN_MAX_TOKENS;
use crate::error::ConversionError;
use crate::convert::ResponseRequestContext;
use crate::providers::{Provider, TokenLimitField};
use crate::types::chat_api::{ChatRequest, StreamOptions};
use crate::types::response_api::{ResponseRequest, ResponseTextConfig};
use tracing::{debug, warn};

fn to_chat_response_format(
    text: Option<&ResponseTextConfig>,
) -> Result<Option<serde_json::Value>, ConversionError> {
    let Some(format) = text.and_then(|t| t.format.as_ref()) else {
        return Ok(None);
    };
    match format.format_type.as_str() {
        "json_schema" => {
            let mut json_schema = serde_json::json!({
                "name": format.name.clone().unwrap_or_else(|| "response_schema".to_string()),
                "schema": format.schema.clone().unwrap_or_else(|| serde_json::json!({})),
            });
            if let Some(strict) = format.strict {
                json_schema["strict"] = serde_json::json!(strict);
            }
            Ok(Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": json_schema
            })))
        }
        "json_object" => Ok(Some(serde_json::json!({
            "type": "json_object"
        }))),
        "text" => Ok(Some(serde_json::json!({
            "type": "text"
        }))),
        other => Err(ConversionError::InvalidFormat(format!(
            "unsupported text.format.type: {other}"
        ))),
    }
}

/// Convert a Responses API request to a Chat API request.
pub fn response_to_chat(
    response_req: ResponseRequest,
    provider: &dyn Provider,
    model_override: Option<&str>,
    _tool_priority: ToolPriority,
) -> Result<ChatRequest, ConversionError> {
    let request_context = ResponseRequestContext::from(&response_req);
    let capabilities = provider.capabilities();
    let enforce_tool_result_adjacency = capabilities.supports_tools;
    let (messages, extracted_tools) = convert_input_to_messages(
        response_req.input,
        response_req.instructions,
        enforce_tool_result_adjacency,
    )?;

    // Merge predefined tools with dynamically discovered tools
    let merged_tools = if extracted_tools.is_empty() {
        response_req.tools
    } else if response_req.tools.is_empty() {
        extracted_tools
    } else {
        // Both have tools - use the context's merge strategy
        use crate::convert::request::context::merge_tools_map;
        merge_tools_map(&response_req.tools, &extracted_tools)
    };

    let tools = convert_tools(merged_tools);
    let tool_choice = convert_tool_choice(response_req.tool_choice);

    // Use model from config if specified, otherwise use provider's model normalization
    let model = model_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| provider.normalize_model(response_req.model));

    let response_format = to_chat_response_format(response_req.text.as_ref())?;

    // Apply provider-specific transformations.
    // Tools: pass through even when empty `vec![]` so we don't violate provider expectations on
    //   schema; serializer skips when None. We filter empty to None to mirror Chat API idiom.
    // tool_choice: pass through "none" honoring user intent (spec allows "none" mode).
    let token_limit = response_req.max_output_tokens.or(response_req.max_tokens);
    let (max_tokens, max_completion_tokens) = match capabilities.token_limit_field {
        TokenLimitField::MaxTokens => (token_limit, None),
        TokenLimitField::MaxCompletionTokens => (None, token_limit),
    };

    let mut chat_req = ChatRequest {
        model,
        messages,
        tools: Some(tools).filter(|t| !t.is_empty()),
        tool_choice: Some(tool_choice),
        stream: Some(response_req.stream),
        temperature: response_req.temperature,
        max_tokens,
        max_completion_tokens,
        top_p: response_req.top_p,
        user: response_req.user,
        stream_options: if response_req.stream {
            Some(StreamOptions { include_usage: Some(true) })
        } else {
            None
        },
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        n: None,
        stop: None,
        response_format,
        reasoning_effort: provider
            .normalize_reasoning_effort(response_req.reasoning.as_ref().and_then(|r| r.effort.clone())),
        parallel_tool_calls: response_req.parallel_tool_calls,
        seed: None,
        service_tier: None,
        web_search_options: None,
        modalities: None,
        prediction: None,
        audio: None,
        extra: Default::default(),
    };

    // Apply min_tokens floor validation (some providers reject max_tokens < MIN_MAX_TOKENS).
    // Surface as a warn so the silent mutation is observable in logs.
    if let Some(max_tokens) = chat_req.max_tokens
        && max_tokens < MIN_MAX_TOKENS {
            warn!(
                "[REQUEST_CONVERT] max_tokens {} below floor {}; raising to floor",
                max_tokens, MIN_MAX_TOKENS
            );
            chat_req.max_tokens = Some(MIN_MAX_TOKENS);
        }

    if let Some(max_completion_tokens) = chat_req.max_completion_tokens
        && max_completion_tokens < MIN_MAX_TOKENS {
            warn!(
                "[REQUEST_CONVERT] max_completion_tokens {} below floor {}; raising to floor",
                max_completion_tokens, MIN_MAX_TOKENS
            );
            chat_req.max_completion_tokens = Some(MIN_MAX_TOKENS);
        }

    capabilities.sanitize_request(&mut chat_req);
    let extensions = provider.provider_extensions(&request_context)?;
    chat_req.apply_provider_extensions(extensions);

    debug!(
        "[REQUEST_CONVERT] converted request: model={}, messages={}, tools={}",
        chat_req.model,
        chat_req.messages.len(),
        chat_req.tools.as_ref().map_or(0, |t| t.len())
    );

    Ok(chat_req)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::providers::glm::GLMProvider;
    use crate::types::chat_api::MessageRole;
    use crate::types::response_api::{
        Content as ResponseContent, InputItem, InputItemOrString, InputItemType, ResponseReasoning,
        ResponseRequest, ResponseTextConfig, ResponseTextFormat,
        Tool, ToolChoice as ResponseToolChoice, ToolType,
    };

    fn make_request(input: InputItemOrString) -> ResponseRequest {
        ResponseRequest {
            model: "gpt-4o".to_string(),
            input,
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
            background: None,
        }
    }

    #[test]
    fn test_instructions_to_system_message() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.instructions = Some("You are a helpful assistant.".to_string());

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

        let first = chat_req.messages.first().unwrap();
        assert_eq!(first.role, MessageRole::System);
        assert_eq!(first.content.as_text(), "You are a helpful assistant.");

        let second = chat_req.messages.get(1).unwrap();
        assert_eq!(second.role, MessageRole::User);
        assert_eq!(second.content.as_text(), "Hello");
    }

    #[test]
    fn test_function_call_conversion() {
        let request = make_request(InputItemOrString::Array(vec![InputItem {
            id: Some("call_123".to_string()),
            item_type: InputItemType::FunctionCall,
            role: None,
            content: None,
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
            call_id: None,
            output: None,
            namespace: None,
            tools: None,
        }]));

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

        let msg = chat_req.messages.first().unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
        assert!(msg.tool_calls.is_some());

        let tc = msg.tool_calls.as_ref().unwrap().first().unwrap();
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn test_function_call_output() {
        let request = make_request(InputItemOrString::Array(vec![
            InputItem {
                id: Some("call_123".to_string()),
                item_type: InputItemType::FunctionCall,
                role: None,
                content: None,
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                call_id: None,
                output: None,
                namespace: None,
                tools: None,
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
                namespace: None,
                tools: None,
            },
        ]));

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

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
        let request = make_request(InputItemOrString::Array(vec![InputItem {
            id: None,
            item_type: InputItemType::FunctionCallOutput,
            role: None,
            content: None,
            name: Some("get_weather".to_string()),
            arguments: None,
            call_id: Some("call_orphan".to_string()),
            output: Some("sunny".to_string()),
            namespace: None,
            tools: None,
        }]));

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
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
        let request = make_request(InputItemOrString::Array(vec![
            InputItem {
                id: Some("fc_1".to_string()),
                item_type: InputItemType::FunctionCall,
                role: None,
                content: None,
                name: Some("exec_command".to_string()),
                arguments: Some(r#"{"cmd":"ls"}"#.to_string()),
                call_id: Some("call_1".to_string()),
                output: None,
                namespace: None,
                tools: None,
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
                namespace: None,
                tools: None,
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
                namespace: None,
                tools: None,
            },
        ]));

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

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
    fn test_tool_providers_merge_assistant_with_pending_tool_calls() {
        let request = make_request(InputItemOrString::Array(vec![
            InputItem {
                id: Some("fc_1".to_string()),
                item_type: InputItemType::FunctionCall,
                role: None,
                content: None,
                name: Some("exec_command".to_string()),
                arguments: Some(r#"{"cmd":"ls"}"#.to_string()),
                call_id: Some("call_1".to_string()),
                output: None,
                namespace: None,
                tools: None,
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
                namespace: None,
                tools: None,
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
                namespace: None,
                tools: None,
            },
        ]));

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert_eq!(chat_req.messages.len(), 2);
        assert_eq!(chat_req.messages[0].role, MessageRole::Assistant);
        assert!(chat_req.messages[0].tool_calls.is_some());
        assert_eq!(chat_req.messages[0].content.as_text(), "我先看下目录");
        assert_eq!(chat_req.messages[1].role, MessageRole::Tool);
    }

    #[test]
    fn test_max_output_tokens_maps_to_chat_max_tokens() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.max_output_tokens = Some(8);

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert_eq!(chat_req.max_tokens, Some(16));
        assert_eq!(chat_req.max_completion_tokens, None);
    }

    #[test]
    fn test_kimi_and_minimax_use_max_completion_tokens() {
        for provider in [
            Box::new(crate::providers::kimi::KimiProvider) as Box<dyn Provider>,
            Box::new(crate::providers::minimax::MiniMaxProvider) as Box<dyn Provider>,
        ] {
            let mut request = make_request(InputItemOrString::String("Hello".to_string()));
            request.max_output_tokens = Some(8);
            let chat_req =
                response_to_chat(request, provider.as_ref(), None, ToolPriority::Merge).unwrap();
            assert_eq!(chat_req.max_tokens, None);
            assert_eq!(chat_req.max_completion_tokens, Some(16));
        }
    }

    #[test]
    fn test_domestic_providers_convert_developer_role_to_system() {
        for provider in [
            Box::new(crate::providers::glm::GLMProvider) as Box<dyn Provider>,
            Box::new(crate::providers::kimi::KimiProvider) as Box<dyn Provider>,
            Box::new(crate::providers::deepseek::DeepSeekProvider) as Box<dyn Provider>,
            Box::new(crate::providers::minimax::MiniMaxProvider) as Box<dyn Provider>,
        ] {
            let request = make_request(InputItemOrString::Array(vec![InputItem {
                id: None,
                item_type: InputItemType::Message,
                role: Some("developer".to_string()),
                content: Some(ResponseContent::String("rules".to_string())),
                name: None,
                arguments: None,
                call_id: None,
                output: None,
                namespace: None,
                tools: None,
            }]));
            let chat_req =
                response_to_chat(request, provider.as_ref(), None, ToolPriority::Merge).unwrap();
            assert_eq!(chat_req.messages[0].role, MessageRole::System);
        }
    }

    #[test]
    fn test_glm_keeps_tools_and_downgrades_tool_choice_to_auto() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.tools = vec![Tool {
            tool_type: ToolType::Function,
            name: Some("lookup".to_string()),
            description: None,
            parameters: None,
            strict: None,
            extra: HashMap::new(),
        }];
        request.tool_choice = ResponseToolChoice::Required;

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert_eq!(chat_req.tools.as_ref().map(|t| t.len()), Some(1));
        let choice = serde_json::to_value(chat_req.tool_choice.unwrap()).unwrap();
        assert_eq!(choice, serde_json::json!("auto"));
    }

    #[test]
    fn test_deepseek_reasoning_effort_maps_to_supported_values() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.reasoning = Some(ResponseReasoning {
            effort: Some("xhigh".to_string()),
            summary: None,
        });

        let provider = crate::providers::deepseek::DeepSeekProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert_eq!(chat_req.reasoning_effort.as_deref(), Some("max"));
    }

    #[test]
    fn test_web_search_preview_tool_degrades_to_function() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.tools = vec![Tool {
            tool_type: ToolType::WebSearchPreview,
            name: None,
            description: None,
            parameters: None,
            strict: None,
            extra: HashMap::new(),
        }];

        let provider = crate::providers::kimi::KimiProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        let tools = chat_req.tools.unwrap_or_default();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "web_search_preview");
    }

    #[test]
    fn test_tool_search_output_extracts_tools() {
        let request = make_request(InputItemOrString::Array(vec![
            InputItem {
                id: Some("tsc_1".to_string()),
                item_type: InputItemType::Message,
                role: Some("user".to_string()),
                content: Some(ResponseContent::String("Find tools".to_string())),
                name: None,
                arguments: None,
                call_id: None,
                output: None,
                namespace: None,
                tools: None,
            },
            InputItem {
                id: Some("tso_1".to_string()),
                item_type: InputItemType::ToolSearchOutput,
                role: None,
                content: None,
                name: None,
                arguments: None,
                call_id: Some("tsc_call_1".to_string()),
                output: None,
                namespace: None,
                tools: Some(vec![
                    Tool {
                        tool_type: ToolType::Function,
                        name: Some("search_tool".to_string()),
                        description: Some("A search tool".to_string()),
                        parameters: Some(serde_json::json!({
                            "type": "object",
                            "properties": {}
                        })),
                        strict: Some(false),
                        extra: HashMap::new(),
                    },
                    Tool {
                        tool_type: ToolType::Function,
                        name: Some("calc_tool".to_string()),
                        description: Some("A calculator".to_string()),
                        parameters: Some(serde_json::json!({
                            "type": "object",
                            "properties": {}
                        })),
                        strict: Some(false),
                        extra: HashMap::new(),
                    },
                ]),
            },
        ]));

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

        // Should have 2 messages: user message + tool_search_output (no message emitted for tool_search)
        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(chat_req.messages[0].role, MessageRole::User);

        // Tools should be extracted from tool_search_output
        let tools = chat_req.tools.unwrap_or_default();
        assert_eq!(tools.len(), 2);
        let tool_names: Vec<_> = tools.iter().map(|t| t.function.name.clone()).collect();
        assert!(tool_names.contains(&"search_tool".to_string()));
        assert!(tool_names.contains(&"calc_tool".to_string()));
    }

    #[test]
    fn test_tool_search_output_merges_with_predefined_tools() {
        let mut request = make_request(InputItemOrString::Array(vec![
            InputItem {
                id: Some("tso_1".to_string()),
                item_type: InputItemType::ToolSearchOutput,
                role: None,
                content: None,
                name: None,
                arguments: None,
                call_id: Some("tsc_call_1".to_string()),
                output: None,
                namespace: None,
                tools: Some(vec![Tool {
                    tool_type: ToolType::Function,
                    name: Some("search_tool".to_string()),
                    description: None,
                    parameters: None,
                    strict: None,
                    extra: HashMap::new(),
                }]),
            },
        ]));
        // Predefined tool with same name - searched tool should override
        request.tools = vec![Tool {
            tool_type: ToolType::Function,
            name: Some("search_tool".to_string()),
            description: Some("Predefined search".to_string()),
            parameters: None,
            strict: None,
            extra: HashMap::new(),
        }];

        let provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();

        let tools = chat_req.tools.unwrap_or_default();
        assert_eq!(tools.len(), 1);
        // Searched tool should override predefined (merge_tools_map semantics)
        assert_eq!(tools[0].function.name, "search_tool");
    }

    #[test]
    fn test_input_file_is_not_dropped() {
        let request = make_request(InputItemOrString::Array(vec![InputItem {
            id: None,
            item_type: InputItemType::Message,
            role: Some("user".to_string()),
            content: Some(ResponseContent::Array(vec![
                crate::types::response_api::ContentPart::InputText {
                    text: "Analyze file".to_string(),
                },
                crate::types::response_api::ContentPart::InputFile {
                    file_url: Some("https://example.com/file.pdf".to_string()),
                    file_id: None,
                },
            ])),
            name: None,
            arguments: None,
            call_id: None,
            output: None,
            namespace: None,
            tools: None,
        }]));

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert!(!chat_req.messages.is_empty());
        let body = chat_req.messages[0].content.as_text();
        assert!(body.contains("[input_file]"));
        assert!(body.contains("file.pdf"));
    }

    #[test]
    fn test_text_format_maps_to_chat_response_format() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.text = Some(ResponseTextConfig {
            format: Some(ResponseTextFormat {
                format_type: "json_schema".to_string(),
                name: Some("AnswerSchema".to_string()),
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    }
                })),
                strict: Some(true),
            }),
        });

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        let response_format = chat_req.response_format.expect("response_format should be mapped");
        assert_eq!(response_format["type"], "json_schema");
        assert_eq!(response_format["json_schema"]["name"], "AnswerSchema");
        assert_eq!(response_format["json_schema"]["strict"], true);
    }

    #[test]
    fn test_reasoning_effort_and_parallel_tool_calls_mapped() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.reasoning = Some(ResponseReasoning {
            effort: Some("high".to_string()),
            summary: None,
        });
        request.parallel_tool_calls = Some(false);

        let provider = GLMProvider;
        let chat_req = response_to_chat(request, &provider, None, ToolPriority::Merge).unwrap();
        assert_eq!(chat_req.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(chat_req.parallel_tool_calls, Some(false));
    }
}
