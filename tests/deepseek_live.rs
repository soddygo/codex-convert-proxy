use std::fs;

use codex_convert_proxy::config::ProxyConfig;
use codex_convert_proxy::convert::response_to_chat;
use codex_convert_proxy::providers::DeepSeekProvider;
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
fn deepseek_adapter_maps_xhigh_to_max_reasoning_effort() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        reasoning_effort: Some("xhigh".to_string()),
        stream: Some(false),
        ..empty_chat_request("deepseek-v4-flash")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert_eq!(body["reasoning_effort"], "max",
        "DeepSeek should map xhigh→max");
    Ok(())
}

#[test]
fn deepseek_adapter_maps_max_to_max_reasoning_effort() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        reasoning_effort: Some("max".to_string()),
        stream: Some(false),
        ..empty_chat_request("deepseek-v4-flash")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert_eq!(body["reasoning_effort"], "max",
        "DeepSeek should keep max unchanged");
    Ok(())
}

#[test]
fn deepseek_adapter_maps_low_medium_high_to_high() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    for effort in ["low", "medium", "high"] {
        let chat = ChatRequest {
            model: "deepseek-v4-flash".to_string(),
            messages: vec![serde_json::from_value(json!({
                "role": "user",
                "content": "Hello"
            }))?],
            reasoning_effort: Some(effort.to_string()),
            stream: Some(false),
            ..empty_chat_request("deepseek-v4-flash")
        };

        let body = adapter.build_request_body(&chat, provider.config())?;
        assert_eq!(body["reasoning_effort"], "high",
            "DeepSeek should map {effort}→high");
    }
    Ok(())
}

#[test]
fn deepseek_adapter_preserves_none_reasoning_effort() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        stream: Some(false),
        ..empty_chat_request("deepseek-v4-flash")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert!(body.get("reasoning_effort").is_none(),
        "DeepSeek should not emit reasoning_effort when None");
    Ok(())
}

#[test]
fn deepseek_adapter_flattens_array_content_to_string() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Part A"},
                {"type": "text", "text": "Part B"}
            ]
        }))?],
        stream: Some(false),
        ..empty_chat_request("deepseek-v4-flash")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    let msg_content = &body["messages"][0]["content"];
    assert_eq!(msg_content, "Part APart B",
        "DeepSeek should flatten content array to concatenated string");
    Ok(())
}

#[test]
fn deepseek_adapter_uses_max_tokens_field() -> TestResult {
    let provider = DeepSeekProvider::new();
    let adapter = provider.protocol_adapter();

    let chat = ChatRequest {
        model: "deepseek-v4-flash".to_string(),
        messages: vec![serde_json::from_value(json!({
            "role": "user",
            "content": "Hello"
        }))?],
        max_tokens: Some(1024),
        stream: Some(false),
        ..empty_chat_request("deepseek-v4-flash")
    };

    let body = adapter.build_request_body(&chat, provider.config())?;
    assert_eq!(body["max_tokens"], 1024,
        "DeepSeek should use max_tokens field");
    assert!(body.get("max_completion_tokens").is_none(),
        "DeepSeek should not emit max_completion_tokens");
    Ok(())
}

// ── Live tests (ignored) ──────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "calls the live DeepSeek API using local config.json"]
async fn deepseek_live_simple_message_succeeds() -> TestResult {
    let backend = load_deepseek_backend()?;
    let provider = DeepSeekProvider::new();
    let request = build_response_request(&backend, "只回复两个字：你好");
    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_body(&body, "deepseek");
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live DeepSeek API using local config.json"]
async fn deepseek_live_with_reasoning_succeeds() -> TestResult {
    let backend = load_deepseek_backend()?;
    let provider = DeepSeekProvider::new();
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
    assert_successful_body(&body, "deepseek");
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live DeepSeek API using local config.json"]
async fn deepseek_live_with_tool_succeeds() -> TestResult {
    let backend = load_deepseek_backend()?;
    let provider = DeepSeekProvider::new();
    let mut request = build_response_request(&backend, "What is the weather in Beijing?");
    request.tools.push(simple_tool());

    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_body(&body, "deepseek");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn load_deepseek_backend() -> TestResult<BackendConfig> {
    let text = fs::read_to_string("config.json")?;
    let config: ProxyConfig = serde_json::from_str(&text)?;
    config
        .backends
        .into_iter()
        .find(|backend| backend.name.eq_ignore_ascii_case("deepseek"))
        .ok_or_else(|| "config.json does not contain a deepseek backend".into())
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
    let model = backend.model.clone().unwrap_or_else(|| "deepseek-v4-flash".to_string());
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
        return;
    }

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
