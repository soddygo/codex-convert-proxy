use std::fs;

use codex_convert_proxy::config::ProxyConfig;
use codex_convert_proxy::convert::response_to_chat;
use codex_convert_proxy::providers::GLMProvider;
use codex_convert_proxy::types::response_api::{
    InputItemOrString, ResponseReasoning, ResponseRequest, Tool, ToolChoice, ToolType,
};
use codex_convert_proxy::types::ChatRequest;
use codex_convert_proxy::util::parse_sse;
use codex_convert_proxy::{BackendConfig, Provider};
use codex_convert_proxy::convert::ToolPriority;
use serde_json::json;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ── Fast tests: quirk verification via adapter output ─────────────────────────

#[test]
fn glm_adapter_injects_thinking_when_reasoning_effort_set() -> TestResult {
    let provider = GLMProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "glm-5".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        stream: Some(false),
        reasoning_effort: Some("high".to_string()),
        ..empty_chat_request("glm-5")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["thinking"], json!({"type": "enabled"}),
        "GLM should inject thinking extension when reasoning_effort is set");
    Ok(())
}

#[test]
fn glm_adapter_downgrades_tool_choice_required_to_auto() -> TestResult {
    let provider = GLMProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "glm-5".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        tools: Some(vec![serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up info",
                "parameters": {"type": "object", "properties": {}}
            }
        }))?]),
        tool_choice: Some(serde_json::from_value(json!("required"))?),
        stream: Some(false),
        ..empty_chat_request("glm-5")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert_eq!(body["tool_choice"], "auto",
        "GLM should downgrade tool_choice required→auto");
    Ok(())
}

#[test]
fn glm_adapter_flattens_array_content_to_string() -> TestResult {
    let provider = GLMProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "glm-5".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Part A"},
                {"type": "text", "text": "Part B"}
            ]
        }))?],
        stream: Some(false),
        ..empty_chat_request("glm-5")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    let msg_content = &body["messages"][0]["content"];
    assert_eq!(msg_content, "Part APart B",
        "GLM should flatten content array to concatenated string");
    Ok(())
}

#[test]
fn glm_adapter_flattens_response_content() -> TestResult {
    let provider = GLMProvider::new();
    let adapter = provider.protocol_adapter();

    let response_json = json!({
        "id": "resp_1",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": "glm-5",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
            },
            "finish_reason": "stop"
        }]
    });

    let parsed = adapter.parse_response(&response_json, provider.config())?;
    let content = &parsed.choices[0].message.content;
    assert!(
        matches!(content, codex_convert_proxy::types::chat_api::Content::String(s) if s == "Hello world"),
        "GLM should flatten response content array to string, got: {:?}", content
    );
    Ok(())
}

#[test]
fn glm_adapter_does_not_inject_thinking_without_reasoning() -> TestResult {
    let provider = GLMProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "glm-5".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        stream: Some(false),
        ..empty_chat_request("glm-5")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert!(body.get("thinking").is_none(),
        "GLM should NOT inject thinking when reasoning_effort is absent");
    Ok(())
}

// ── Live tests (ignored) ──────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "calls the live GLM API using local config.json"]
async fn glm_live_simple_message_succeeds() -> TestResult {
    let backend = load_glm_backend()?;
    let provider = GLMProvider::new();
    let request = build_response_request(&backend, "只回复两个字：你好");
    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_body(&body, "glm");
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live GLM API using local config.json"]
async fn glm_live_with_reasoning_effort_succeeds() -> TestResult {
    let backend = load_glm_backend()?;
    let provider = GLMProvider::new();
    let mut request = build_response_request(&backend, "解释一下量子计算的基本原理");
    request.reasoning = Some(ResponseReasoning {
        effort: Some("high".to_string()),
        summary: None,
    });

    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_body(&body, "glm");
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live GLM API using local config.json"]
async fn glm_live_with_tool_succeeds() -> TestResult {
    let backend = load_glm_backend()?;
    let provider = GLMProvider::new();
    let mut request = build_response_request(&backend, "What is the weather in Beijing?");
    request.tools.push(simple_tool());

    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_body(&body, "glm");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn load_glm_backend() -> TestResult<BackendConfig> {
    let text = fs::read_to_string("config.json")?;
    let config: ProxyConfig = serde_json::from_str(&text)?;
    config
        .backends
        .into_iter()
        .find(|backend| backend.name.eq_ignore_ascii_case("glm"))
        .ok_or_else(|| "config.json does not contain a glm backend".into())
}

fn empty_chat_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![],
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
    }
}

fn build_response_request(backend: &BackendConfig, user_message: &str) -> ResponseRequest {
    let model = backend.model.clone().unwrap_or_else(|| "glm-5".to_string());
    ResponseRequest {
        model,
        input: InputItemOrString::String(user_message.to_string()),
        instructions: Some("You are a helpful assistant. Reply in Chinese.".to_string()),
        tools: Vec::new(),
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
        background: None,
    }
}

fn simple_tool() -> Tool {
    Tool {
        tool_type: ToolType::Function,
        name: Some("get_weather".to_string()),
        description: Some("Get the current weather for a city.".to_string()),
        parameters: Some(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        })),
        strict: None,
        extra: Default::default(),
    }
}

async fn send_chat_request(
    backend: &BackendConfig,
    provider: &dyn Provider,
    chat: &ChatRequest,
) -> TestResult<String> {
    let url = format!(
        "{}/{}",
        backend.url.trim_end_matches('/'),
        provider.chat_completions_path().trim_start_matches('/')
    );
    let adapter = provider.protocol_adapter();
    let body = adapter.build_request_body(chat, provider.config())?;
    let response = reqwest::Client::new()
        .post(url)
        .bearer_auth(&backend.api_key)
        .json(&body)
        .send()
        .await?;
    Ok(response.text().await?)
}

fn assert_successful_body(body: &str, provider_name: &str) {
    if body.contains("\"error\"") {
        panic!("{provider_name} returned error body: {body}");
    }

    let has_content = body.contains("\"choices\"");
    if has_content {
        // Non-streaming response with choices — good
        return;
    }

    // Try parsing as SSE stream
    let (chunks, _) = parse_sse(body);
    let has_valid_chunk = chunks.iter().any(|chunk| {
        chunk.data == "[DONE]" || chunk.data.contains("\"choices\"")
    });

    assert!(
        has_valid_chunk,
        "unexpected {provider_name} body (len={}): {body}",
        body.len()
    );
}
