//! OpenAiChatAdapter — converts provider-neutral `ChatRequest` into
//! OpenAI Chat Completions API JSON and parses responses back.
//!
//! This adapter **owns** serialization: it builds the JSON body field-by-field,
//! only including fields the provider supports (as declared in `ChatApiQuirks`).
//! This replaces the previous approach of `serde_json::to_vec(&chat_req)` followed
//! by post-hoc field stripping via `ProviderCapabilities::sanitize_request()`.

use serde_json::{Value, json};

use crate::error::ConversionError;
use crate::providers::adapter::traits::{ProtocolAdapter, ProtocolType};
use crate::providers::adapter::config::{ChatApiQuirks, ProviderConfig, ToolChoiceSupport, TokenLimitField};
use crate::types::chat_api::{
    ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatTool, ChatToolChoice,
    ChatToolChoiceMode, Content, MessageRole,
};

/// Stateless singleton adapter for OpenAI Chat Completions API.
///
/// A single instance is shared across all providers that speak this protocol.
#[derive(Debug, Clone, Copy)]
pub struct OpenAiChatAdapter;

impl ProtocolAdapter for OpenAiChatAdapter {
    fn protocol_name(&self) -> &'static str {
        "openai-chat"
    }

    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::ChatApi
    }

    fn build_request_body(
        &self,
        chat_req: &ChatRequest,
        config: &ProviderConfig,
    ) -> Result<Value, ConversionError> {
        let quirks = &config.chat_quirks;
        let messages = build_messages(chat_req, quirks);
        let mut root = serde_json::Map::new();

        root.insert("model".into(), json!(chat_req.model));
        root.insert("messages".into(), Value::Array(messages));

        // -- Stream
        if let Some(ref stream) = chat_req.stream {
            root.insert("stream".into(), json!(stream));
        }

        // -- Tools (only if provider supports them AND they're present)
        if quirks.supports_tools {
            if let Some(ref tools) = chat_req.tools {
                root.insert("tools".into(), serialize_tools(tools, quirks));
            }
        }

        // -- tool_choice (only if provider supports it, and per support mode)
        if quirks.supports_tools {
            if let Some(ref tc) = chat_req.tool_choice {
                if let Some(sanitized) = sanitize_tool_choice(tc, quirks.tool_choice_mode) {
                    root.insert("tool_choice".into(), sanitized);
                }
            }
        }

        // -- Token limit: use provider's preferred field name
        let token_limit = chat_req.max_tokens.or(chat_req.max_completion_tokens);
        if let Some(tl) = token_limit {
            match quirks.token_limit_field {
                TokenLimitField::MaxTokens => {
                    root.insert("max_tokens".into(), json!(tl));
                }
                TokenLimitField::MaxCompletionTokens => {
                    root.insert("max_completion_tokens".into(), json!(tl));
                }
            }
        }

        // -- Temperature
        if let Some(temp) = chat_req.temperature {
            root.insert("temperature".into(), json!(temp));
        }

        // -- top_p
        if let Some(top_p) = chat_req.top_p {
            root.insert("top_p".into(), json!(top_p));
        }

        // -- user
        if let Some(ref user) = chat_req.user {
            root.insert("user".into(), json!(user));
        }

        // -- response_format
        if let Some(ref rf) = chat_req.response_format {
            root.insert("response_format".into(), rf.clone());
        }

        // -- reasoning_effort (with optional provider mapping)
        let reasoning_effort = if let Some(map_fn) = quirks.reasoning_effort_map {
            map_fn(chat_req.reasoning_effort.clone())
        } else {
            chat_req.reasoning_effort.clone()
        };
        if let Some(ref re) = reasoning_effort {
            root.insert("reasoning_effort".into(), json!(re));
        }

        // -- stream_options (only if provider supports it)
        if quirks.supports_stream_options {
            if let Some(ref so) = chat_req.stream_options {
                root.insert("stream_options".into(), serde_json::to_value(so)?);
            }
        }

        // -- parallel_tool_calls (only if provider supports it)
        if quirks.supports_parallel_tool_calls {
            if let Some(ptc) = chat_req.parallel_tool_calls {
                root.insert("parallel_tool_calls".into(), json!(ptc));
            }
        }

        // -- stop
        if let Some(ref stop) = chat_req.stop {
            root.insert("stop".into(), json!(stop));
        }

        // -- seed
        if let Some(seed) = chat_req.seed {
            root.insert("seed".into(), json!(seed));
        }

        // -- frequency_penalty, presence_penalty
        if let Some(fp) = chat_req.frequency_penalty {
            root.insert("frequency_penalty".into(), json!(fp));
        }
        if let Some(pp) = chat_req.presence_penalty {
            root.insert("presence_penalty".into(), json!(pp));
        }

        // -- logit_bias, logprobs, top_logprobs
        if let Some(ref lb) = chat_req.logit_bias {
            root.insert("logit_bias".into(), serde_json::to_value(lb)?);
        }
        if let Some(lp) = chat_req.logprobs {
            root.insert("logprobs".into(), json!(lp));
        }
        if let Some(tl) = chat_req.top_logprobs {
            root.insert("top_logprobs".into(), json!(tl));
        }

        // -- n
        if let Some(n) = chat_req.n {
            root.insert("n".into(), json!(n));
        }

        // -- service_tier
        if let Some(ref st) = chat_req.service_tier {
            root.insert("service_tier".into(), json!(st));
        }

        // -- reasoning extension (provider-specific extra field when reasoning active)
        if reasoning_effort.is_some() || chat_req.reasoning_effort.is_some() {
            if let Some((key, ref val)) = quirks.reasoning_extension {
                root.insert(key.to_string(), val.clone());
            }
        }

        // -- Provider extensions (from extra fields)
        for (key, value) in &chat_req.extra {
            root.insert(key.clone(), value.clone());
        }

        Ok(Value::Object(root))
    }

    fn parse_response(
        &self,
        body: &Value,
        config: &ProviderConfig,
    ) -> Result<ChatResponse, ConversionError> {
        let mut response: ChatResponse = serde_json::from_value(body.clone())?;

        // Apply content flattening if provider needs it
        if config.chat_quirks.flatten_response {
            for choice in &mut response.choices {
                if matches!(choice.message.content, Content::Array(_)) {
                    let text = choice.message.content.as_text();
                    choice.message.content = Content::String(text);
                }
            }
        }

        Ok(response)
    }

    fn parse_stream_chunk(
        &self,
        chunk_json: &Value,
        config: &ProviderConfig,
    ) -> Result<ChatStreamChunk, ConversionError> {
        let mut chunk: ChatStreamChunk = serde_json::from_value(chunk_json.clone())?;

        if config.chat_quirks.flatten_response {
            for choice in &mut chunk.choices {
                if let Some(ref mut delta) = choice.delta {
                    if let Some(ref content) = delta.content {
                        let text = content.as_text();
                        if !text.is_empty() {
                            delta.content = Some(Content::String(text));
                        }
                    }
                }
            }
        }

        Ok(chunk)
    }
}

// ── Message building ──────────────────────────────────────────────────────────

fn build_messages(chat_req: &ChatRequest, quirks: &ChatApiQuirks) -> Vec<Value> {
    chat_req
        .messages
        .iter()
        .map(|msg| build_message(msg, quirks))
        .collect()
}

fn build_message(msg: &ChatMessage, quirks: &ChatApiQuirks) -> Value {
    let role = role_to_string(msg.role, quirks);
    let content = build_message_content(msg, quirks);

    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), json!(role));

    // Content: use the manually-built content value
    obj.insert("content".into(), content);

    // Tool calls (assistant messages only)
    if let Some(ref tool_calls) = msg.tool_calls {
        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": tc.tool_type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(calls));
    }

    // Function call (legacy, non-streaming)
    if let Some(ref fc) = msg.function_call {
        obj.insert(
            "function_call".into(),
            json!({
                "name": fc.name,
                "arguments": fc.arguments,
            }),
        );
    }

    // tool_call_id (tool messages)
    if let Some(ref tcid) = msg.tool_call_id {
        obj.insert("tool_call_id".into(), json!(tcid));
    }

    // name (optional, used by some providers)
    if let Some(ref name) = msg.name {
        obj.insert("name".into(), json!(name));
    }

    // refusal
    if let Some(ref refusal) = msg.refusal {
        obj.insert("refusal".into(), json!(refusal));
    }

    Value::Object(obj)
}

fn role_to_string(role: MessageRole, _quirks: &ChatApiQuirks) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Developer => {
            // Developer role → system for providers that don't support it
            "system"
        }
        MessageRole::Tool => "tool",
        // Unknown / forward-compat roles fall back to "user"
        MessageRole::Unknown | MessageRole::Critic | MessageRole::Discriminator => "user",
    }
}

fn build_message_content(msg: &ChatMessage, quirks: &ChatApiQuirks) -> Value {
    match &msg.content {
        Content::String(s) => {
            json!(s)
        }
        Content::Array(blocks) => {
            if quirks.flatten_content {
                // Flatten array content to a single string
                let text = blocks
                    .iter()
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("");
                json!(text)
            } else {
                // Preserve array structure
                let parts: Vec<Value> = blocks
                    .iter()
                    .map(|block| {
                        let mut obj = serde_json::Map::new();
                        obj.insert("type".into(), json!(block.block_type));

                        if let Some(ref text) = block.text {
                            obj.insert("text".into(), json!(text));
                        }
                        if let Some(ref image_url) = block.image_url {
                            match image_url {
                                crate::types::chat_api::ImageUrlField::String(url) => {
                                    obj.insert(
                                        "image_url".into(),
                                        json!({"url": url}),
                                    );
                                }
                                crate::types::chat_api::ImageUrlField::Object(img_obj) => {
                                    obj.insert(
                                        "image_url".into(),
                                        serde_json::to_value(img_obj).unwrap_or_default(),
                                    );
                                }
                            }
                        }
                        if let Some(ref input_audio) = block.input_audio {
                            obj.insert(
                                "input_audio".into(),
                                json!({
                                    "data": input_audio.data,
                                    "format": input_audio.format,
                                }),
                            );
                        }
                        if let Some(ref file) = block.file {
                            obj.insert(
                                "file".into(),
                                json!({"file_id": file.file_id}),
                            );
                        }
                        if let Some(ref refusal) = block.refusal {
                            obj.insert("refusal".into(), json!(refusal));
                        }

                        Value::Object(obj)
                    })
                    .collect();

                Value::Array(parts)
            }
        }
    }
}

// ── Tool serialization ────────────────────────────────────────────────────────

fn serialize_tools(tools: &[ChatTool], quirks: &ChatApiQuirks) -> Value {
    let tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let mut func = serde_json::Map::new();
            func.insert("name".into(), json!(tool.function.name));
            if let Some(ref desc) = tool.function.description {
                func.insert("description".into(), json!(desc));
            }
            if let Some(ref params) = tool.function.parameters {
                func.insert("parameters".into(), params.clone());
            }
            if quirks.supports_tool_strict {
                if let Some(strict) = tool.function.strict {
                    func.insert("strict".into(), json!(strict));
                }
            }

            json!({
                "type": tool.tool_type,
                "function": Value::Object(func),
            })
        })
        .collect();

    Value::Array(tools)
}

fn sanitize_tool_choice(
    choice: &ChatToolChoice,
    support: ToolChoiceSupport,
) -> Option<Value> {
    match support {
        ToolChoiceSupport::Full => {
            // Pass through as-is
            serde_json::to_value(choice).ok()
        }
        ToolChoiceSupport::Unsupported => None,
        ToolChoiceSupport::AutoOnly => {
            // Always emit "auto"
            Some(json!("auto"))
        }
        ToolChoiceSupport::AutoAndNone => match choice {
            ChatToolChoice::Mode(ChatToolChoiceMode::None) => Some(json!("none")),
            _ => Some(json!("auto")),
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{
        ChatMessage, ChatRequest, ChatToolChoiceMode, Content, MessageRole, StreamOptions,
        ChatTool, FunctionDefinition,
    };

    fn make_config(quirks: ChatApiQuirks) -> ProviderConfig {
        ProviderConfig {
            chat_quirks: quirks,
            ..ProviderConfig::default()
        }
    }

    fn make_request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: Content::String("You are helpful.".to_string()),
                    name: None,
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                    refusal: None,
                },
                ChatMessage {
                    role: MessageRole::User,
                    content: Content::String("Hello".to_string()),
                    name: None,
                    annotations: None,
                    tool_calls: None,
                    tool_call_id: None,
                    function_call: None,
                    refusal: None,
                },
            ],
            tools: Some(vec![ChatTool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
                    strict: Some(true),
                },
            }]),
            tool_choice: Some(ChatToolChoice::Mode(ChatToolChoiceMode::Auto)),
            stream: Some(true),
            temperature: None,
            max_tokens: Some(1024),
            max_completion_tokens: None,
            top_p: None,
            user: None,
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            stop: None,
            response_format: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            seed: None,
            service_tier: None,
            web_search_options: None,
            modalities: None,
            prediction: None,
            audio: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn test_basic_chat_request_json() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks::default());
        let req = make_request("gpt-4o");

        let body = adapter.build_request_body(&req, &config).unwrap();

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("messages").unwrap().as_array().unwrap().len() == 2);
        assert!(body.get("tools").is_some());
        assert!(body.get("tool_choice").is_some());
        assert!(body.get("stream_options").is_some());
    }

    #[test]
    fn test_quirk_tool_choice_unsupported_omits_field() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            tool_choice_mode: ToolChoiceSupport::Unsupported,
            ..ChatApiQuirks::default()
        });
        let req = make_request("minimax-model");

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn test_quirk_tool_choice_auto_only_downgrades_to_auto() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            tool_choice_mode: ToolChoiceSupport::AutoOnly,
            ..ChatApiQuirks::default()
        });
        let mut req = make_request("glm-4");
        req.tool_choice = Some(ChatToolChoice::Mode(ChatToolChoiceMode::Required));

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_quirk_max_completion_tokens() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            token_limit_field: TokenLimitField::MaxCompletionTokens,
            ..ChatApiQuirks::default()
        });
        let req = make_request("kimi-model");

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 1024);
    }

    #[test]
    fn test_quirk_no_stream_options() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            supports_stream_options: false,
            ..ChatApiQuirks::default()
        });
        let req = make_request("minimax-model");

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn test_quirk_no_parallel_tool_calls() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            supports_parallel_tool_calls: false,
            ..ChatApiQuirks::default()
        });
        let mut req = make_request("minimax-model");
        req.parallel_tool_calls = Some(false);

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn test_quirk_no_tool_strict() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            supports_tool_strict: false,
            ..ChatApiQuirks::default()
        });
        let req = make_request("minimax-model");

        let body = adapter.build_request_body(&req, &config).unwrap();
        let tools = body["tools"].as_array().unwrap();
        let func = &tools[0]["function"];
        assert!(func.get("strict").is_none());
    }

    #[test]
    fn test_quirk_flatten_content() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            flatten_content: true,
            ..ChatApiQuirks::default()
        });
        let mut req = make_request("glm-4");
        req.messages[1].content = Content::Array(vec![
            crate::types::chat_api::ContentBlock {
                block_type: "text".to_string(),
                text: Some("Part A".to_string()),
                image_url: None,
                input_audio: None,
                file: None,
                refusal: None,
            },
            crate::types::chat_api::ContentBlock {
                block_type: "text".to_string(),
                text: Some("Part B".to_string()),
                image_url: None,
                input_audio: None,
                file: None,
                refusal: None,
            },
        ]);

        let body = adapter.build_request_body(&req, &config).unwrap();
        let user_msg = &body["messages"][1];
        assert_eq!(user_msg["content"], "Part APart B");
    }

    #[test]
    fn test_quirk_reasoning_extension() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            reasoning_extension: Some(("thinking", json!({"type": "enabled"}))),
            ..ChatApiQuirks::default()
        });
        let mut req = make_request("glm-4");
        req.reasoning_effort = Some("high".to_string());

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["thinking"], json!({"type": "enabled"}));
    }

    #[test]
    fn test_quirk_reasoning_effort_map() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            reasoning_effort_map: Some(|effort| {
                effort.map(|e| match e.as_str() {
                    "xhigh" | "max" => "max".to_string(),
                    _ => "high".to_string(),
                })
            }),
            ..ChatApiQuirks::default()
        });
        let mut req = make_request("deepseek-model");
        req.reasoning_effort = Some("xhigh".to_string());

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert_eq!(body["reasoning_effort"], "max");
    }

    #[test]
    fn test_quirk_supports_tools_false_omits_tools_and_tool_choice() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks {
            supports_tools: false,
            ..ChatApiQuirks::default()
        });
        let req = make_request("no-tools-model");

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn test_developer_role_maps_to_system() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks::default());
        let mut req = make_request("gpt-4o");
        req.messages[0].role = MessageRole::Developer;

        let body = adapter.build_request_body(&req, &config).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks::default());
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::Assistant,
                content: Content::String("Let me check.".to_string()),
                name: None,
                annotations: None,
                tool_calls: Some(vec![crate::types::chat_api::ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::types::chat_api::FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Beijing"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                function_call: None,
                refusal: None,
            }],
            tools: None,
            tool_choice: None,
            stream: Some(false),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            user: None,
            stream_options: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            stop: None,
            response_format: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            seed: None,
            service_tier: None,
            web_search_options: None,
            modalities: None,
            prediction: None,
            audio: None,
            extra: Default::default(),
        };

        let body = adapter.build_request_body(&req, &config).unwrap();
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "Let me check.");
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "get_weather");
    }

    #[test]
    fn test_tool_message_with_tool_call_id() {
        let adapter = OpenAiChatAdapter;
        let config = make_config(ChatApiQuirks::default());
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::Tool,
                content: Content::String("25 degrees".to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                function_call: None,
                refusal: None,
            }],
            tools: None,
            tool_choice: None,
            stream: Some(false),
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            user: None,
            stream_options: None,
            frequency_penalty: None,
            presence_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            stop: None,
            response_format: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            seed: None,
            service_tier: None,
            web_search_options: None,
            modalities: None,
            prediction: None,
            audio: None,
            extra: Default::default(),
        };

        let body = adapter.build_request_body(&req, &config).unwrap();
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_1");
        assert_eq!(msg["content"], "25 degrees");
    }
}
