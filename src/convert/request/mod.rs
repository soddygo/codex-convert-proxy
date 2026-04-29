//! Request conversion: Responses API → Chat API.

mod messages;
mod tools;

pub use messages::{convert_input_to_messages, extract_content};
pub use tools::{convert_tools, convert_tool_choice, is_tool_choice_none};

use crate::constants::MIN_MAX_TOKENS;
use crate::error::ConversionError;
use crate::providers::Provider;
use crate::types::chat_api::{ChatRequest, StreamOptions};
use crate::types::response_api::ResponseRequest;
use tracing::debug;

/// Convert a Responses API request to a Chat API request.
pub fn response_to_chat(
    response_req: ResponseRequest,
    provider: &mut dyn Provider,
    model_override: Option<&str>,
) -> Result<ChatRequest, ConversionError> {
    let enforce_tool_result_adjacency = provider.name() == "minimax";
    let messages = convert_input_to_messages(
        response_req.input,
        response_req.instructions,
        enforce_tool_result_adjacency,
    )?;
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
        tool_choice: Some(tool_choice).filter(|tc| !is_tool_choice_none(tc)),
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
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        n: None,
        stop: None,
        response_format: None,
        seed: None,
        service_tier: None,
    };

    // Apply min_tokens floor validation (some providers reject max_tokens < MIN_MAX_TOKENS)
    if let Some(max_tokens) = chat_req.max_tokens {
        if max_tokens < MIN_MAX_TOKENS {
            chat_req.max_tokens = Some(MIN_MAX_TOKENS);
        }
    }

    provider.transform_request(&mut chat_req);

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
        Content as ResponseContent, InputItem, InputItemOrString, InputItemType, ResponseRequest,
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
        }
    }

    #[test]
    fn test_instructions_to_system_message() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.instructions = Some("You are a helpful assistant.".to_string());

        let mut provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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
        }]));

        let mut provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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
        ]));

        let mut provider = crate::providers::minimax::MiniMaxProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();

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
        }]));

        let mut provider = crate::providers::minimax::MiniMaxProvider;
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
        ]));

        let mut provider = crate::providers::minimax::MiniMaxProvider;
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
    fn test_non_minimax_keeps_assistant_and_tool_call_split() {
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
        ]));

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert_eq!(chat_req.messages.len(), 3);
        assert_eq!(chat_req.messages[0].role, MessageRole::Assistant);
        assert!(chat_req.messages[0].tool_calls.is_some());
        assert_eq!(chat_req.messages[1].role, MessageRole::Assistant);
        assert_eq!(chat_req.messages[2].role, MessageRole::Tool);
    }

    #[test]
    fn test_max_output_tokens_maps_to_chat_max_tokens() {
        let mut request = make_request(InputItemOrString::String("Hello".to_string()));
        request.max_output_tokens = Some(8);

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert_eq!(chat_req.max_tokens, Some(16));
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

        let mut provider = crate::providers::kimi::KimiProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        let tools = chat_req.tools.unwrap_or_default();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "web_search_preview");
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
        }]));

        let mut provider = GLMProvider;
        let chat_req = response_to_chat(request, &mut provider, None).unwrap();
        assert!(!chat_req.messages.is_empty());
        let body = chat_req.messages[0].content.as_text();
        assert!(body.contains("[input_file]"));
        assert!(body.contains("file.pdf"));
    }
}
