use std::fs;

use codex_convert_proxy::config::ProxyConfig;
use codex_convert_proxy::convert::response_to_chat;
use codex_convert_proxy::providers::MiniMaxProvider;
use codex_convert_proxy::types::response_api::{
    InputItem, InputItemOrString, InputItemType, ResponseRequest, Tool, ToolChoice, ToolType,
};
use codex_convert_proxy::types::{ChatRequest, MessageRole};
use codex_convert_proxy::util::parse_sse;
use codex_convert_proxy::{BackendConfig, Provider};
use codex_convert_proxy::convert::ToolPriority;
use serde_json::json;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[tokio::test]
#[ignore = "calls the live MiniMax API using local config.json"]
async fn minimax_live_single_system_without_tools_succeeds() -> TestResult {
    let backend = load_minimax_backend()?;
    let provider = MiniMaxProvider::new();
    let request = build_response_request(&backend, false);
    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    assert_eq!(chat.messages.iter().filter(|m| m.role == MessageRole::System).count(), 1);
    assert!(chat.tools.is_none());

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_minimax_body(&body);
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live MiniMax API using local config.json"]
async fn minimax_live_single_system_with_simple_tool_reports_current_behavior() -> TestResult {
    let backend = load_minimax_backend()?;
    let provider = MiniMaxProvider::new();
    let mut request = build_response_request(&backend, false);
    request.tools.push(simple_tool());
    let chat = response_to_chat(
        request,
        &provider,
        backend.model.as_deref(),
        ToolPriority::default(),
    )?;

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert_successful_minimax_body(&body);
    Ok(())
}

#[tokio::test]
#[ignore = "calls the live MiniMax API using local config.json"]
async fn minimax_live_raw_multi_system_reproduces_2013() -> TestResult {
    let backend = load_minimax_backend()?;
    let provider = MiniMaxProvider::new();
    let mut chat = build_raw_chat_request(&backend, vec![
        ("system", "You are a concise assistant."),
        ("system", "Reply in Chinese."),
        ("user", "只回复两个字：你好"),
    ]);
    provider.capabilities().sanitize_request(&mut chat);

    let body = send_chat_request(&backend, &provider, &chat).await?;
    assert!(
        body.contains("invalid chat setting (2013)") || body.contains("\"http_code\":\"400\""),
        "expected MiniMax to reject multi-system chat setting, got: {body}"
    );
    Ok(())
}

fn load_minimax_backend() -> TestResult<BackendConfig> {
    let text = fs::read_to_string("config.json")?;
    let config: ProxyConfig = serde_json::from_str(&text)?;
    config
        .backends
        .into_iter()
        .find(|backend| backend.name.eq_ignore_ascii_case("minimax"))
        .ok_or_else(|| "config.json does not contain a minimax backend".into())
}

fn build_response_request(backend: &BackendConfig, with_split_system: bool) -> ResponseRequest {
    let model = backend
        .model
        .clone()
        .unwrap_or_else(|| "MiniMax-M2.7-highspeed".to_string());
    let input = if with_split_system {
        InputItemOrString::Array(vec![
            message_item("system", "Reply in Chinese."),
            message_item("user", "只回复两个字：你好"),
        ])
    } else {
        InputItemOrString::String("只回复两个字：你好".to_string())
    };

    ResponseRequest {
        model,
        input,
        instructions: Some("You are a concise assistant.".to_string()),
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        stream: true,
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

fn message_item(role: &str, content: &str) -> InputItem {
    InputItem {
        id: None,
        item_type: InputItemType::Message,
        role: Some(role.to_string()),
        content: Some(codex_convert_proxy::types::response_api::Content::String(
            content.to_string(),
        )),
        name: None,
        arguments: None,
        call_id: None,
        output: None,
        namespace: None,
        tools: None,
    }
}

fn simple_tool() -> Tool {
    Tool {
        tool_type: ToolType::Function,
        name: Some("get_project_name".to_string()),
        description: Some("Get the current project name.".to_string()),
        parameters: Some(json!({
            "type": "object",
            "properties": {},
            "required": []
        })),
        strict: None,
        extra: Default::default(),
    }
}

fn build_raw_chat_request(backend: &BackendConfig, messages: Vec<(&str, &str)>) -> ChatRequest {
    serde_json::from_value(json!({
        "model": backend
            .model
            .clone()
            .unwrap_or_else(|| "MiniMax-M2.7-highspeed".to_string()),
        "messages": messages
            .into_iter()
            .map(|(role, content)| {
                json!({
                    "role": role,
                    "content": content
                })
            })
            .collect::<Vec<_>>(),
        "stream": true,
    }))
    .expect("valid raw chat request")
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
    let response = reqwest::Client::new()
        .post(url)
        .bearer_auth(&backend.api_key)
        .json(chat)
        .send()
        .await?;
    Ok(response.text().await?)
}

fn assert_successful_minimax_body(body: &str) {
    if body.contains("invalid chat setting (2013)") || body.contains("\"error\"") {
        panic!("MiniMax returned error body: {body}");
    }

    let (chunks, _) = parse_sse(body);
    assert!(
        chunks.iter().any(|chunk| chunk.data == "[DONE]")
            || chunks.iter().any(|chunk| chunk.data.contains("\"choices\""))
            || body.contains("\"choices\""),
        "unexpected MiniMax body: {body}"
    );
}
