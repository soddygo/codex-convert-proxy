//! SSE serialization: ResponseStreamEvent to SSE string format.

use super::events::ResponseStreamEvent;
use super::state::ResponseRequestContext;

/// Generate SSE string from Response stream event.
pub fn event_to_sse(event: &ResponseStreamEvent) -> String {
    match event {
        ResponseStreamEvent::Created {
            id,
            model,
            status,
            created_at,
            request_context,
        } => {
            sse_event(
                "response.created",
                serde_json::json!({
                    "type": "response.created",
                    "response": response_stub_json(id, model, status, *created_at, request_context.as_ref()),
                }),
            )
        }
        ResponseStreamEvent::InProgress {
            id,
            model,
            status,
            created_at,
            request_context,
        } => {
            sse_event(
                "response.in_progress",
                serde_json::json!({
                    "type": "response.in_progress",
                    "response": response_stub_json(id, model, status, *created_at, request_context.as_ref()),
                }),
            )
        }
        ResponseStreamEvent::OutputItemAdded { output_index, item_id, item_type, role, call_id } => {
            let mut item = serde_json::Map::new();
            item.insert("id".to_string(), serde_json::json!(item_id));
            item.insert("type".to_string(), serde_json::json!(item_type));
            item.insert("status".to_string(), serde_json::json!("in_progress"));
            if let Some(r) = role {
                item.insert("role".to_string(), serde_json::json!(r));
            }
            if let Some(cid) = call_id {
                item.insert("call_id".to_string(), serde_json::json!(cid));
            }
            if item_type == "message" || item_type == "reasoning" {
                item.insert("content".to_string(), serde_json::json!([]));
            }
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": serde_json::Value::Object(item),
                }),
            )
        }
        ResponseStreamEvent::ContentPartAdded { item_id, output_index, content_index } => {
            sse_event(
                "response.content_part.added",
                serde_json::json!({
                    "type": "response.content_part.added",
                    "output_index": output_index,
                    "item_id": item_id,
                    "content_index": content_index,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    }
                }),
            )
        }
        ResponseStreamEvent::OutputTextDelta { item_id, output_index, content_index, delta } => {
            sse_event(
                "response.output_text.delta",
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            )
        }
        ResponseStreamEvent::OutputTextDone {
            item_id,
            output_index,
            content_index,
            text,
        } => {
            sse_event(
                "response.output_text.done",
                serde_json::json!({
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                }),
            )
        }
        ResponseStreamEvent::ContentPartDone {
            item_id,
            output_index,
            content_index,
            text,
        } => {
            sse_event(
                "response.content_part.done",
                serde_json::json!({
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                }),
            )
        }
        ResponseStreamEvent::OutputItemDone {
            output_index,
            item_id,
            item_type,
            role,
            call_id,
            name,
            arguments,
            text,
        } => {
            let mut item = serde_json::Map::new();
            item.insert("id".to_string(), serde_json::json!(item_id));
            item.insert("type".to_string(), serde_json::json!(item_type));
            item.insert("status".to_string(), serde_json::json!("completed"));
            if let Some(r) = role {
                item.insert("role".to_string(), serde_json::json!(r));
            }
            if let Some(cid) = call_id {
                item.insert("call_id".to_string(), serde_json::json!(cid));
            }
            if let Some(n) = name {
                item.insert("name".to_string(), serde_json::json!(n));
            }
            if let Some(args) = arguments {
                item.insert("arguments".to_string(), serde_json::json!(args));
            }
            if let Some(body_text) = text {
                item.insert(
                    "content".to_string(),
                    serde_json::json!([{
                        "type": "output_text",
                        "text": body_text,
                        "annotations": [],
                    }]),
                );
            }
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": serde_json::Value::Object(item),
                }),
            )
        }
        ResponseStreamEvent::ReasoningAdded { output_index, item_id } => {
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "content": [],
                    },
                }),
            )
        }
        ResponseStreamEvent::ReasoningDelta { item_id, output_index, content_index, delta } => {
            sse_event(
                "response.reasoning_text.delta",
                serde_json::json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            )
        }
        ResponseStreamEvent::ReasoningTextDone { item_id, output_index, content_index, text } => {
            sse_event(
                "response.reasoning_text.done",
                serde_json::json!({
                    "type": "response.reasoning_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                }),
            )
        }
        ResponseStreamEvent::ReasoningSummaryTextDelta { item_id, output_index, content_index, delta } => {
            sse_event(
                "response.reasoning_summary_text.delta",
                serde_json::json!({
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            )
        }
        ResponseStreamEvent::ReasoningSummaryTextDone { item_id, output_index, content_index, text } => {
            sse_event(
                "response.reasoning_summary_text.done",
                serde_json::json!({
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                }),
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDelta { output_index, item_id, delta } => {
            sse_event(
                "response.function_call_arguments.delta",
                serde_json::json!({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_index,
                    "item_id": item_id,
                    "delta": delta,
                }),
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDone { output_index, item_id, call_id, name, arguments } => {
            sse_event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "item_id": item_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }),
            )
        }
        ResponseStreamEvent::Completed { response } => {
            sse_event(
                "response.completed",
                serde_json::json!({
                    "type": "response.completed",
                    "response": response,
                }),
            )
        }
        ResponseStreamEvent::Error { id, error_type, message, code } => {
            let mut payload = serde_json::json!({
                "type": "response.error",
                "error": {
                    "type": error_type,
                    "message": message,
                }
            });
            if let Some(id) = id {
                payload["id"] = serde_json::json!(id);
            }
            if let Some(code) = code {
                payload["error"]["code"] = serde_json::json!(code);
            }
            sse_event("response.error", payload)
        }
        ResponseStreamEvent::Failed { id, model, status, created_at } => {
            sse_event(
                "response.failed",
                serde_json::json!({
                    "type": "response.failed",
                    "response": response_stub_json(id, model, status, *created_at, None),
                }),
            )
        }
        ResponseStreamEvent::Incomplete { id, model, status, created_at, reason } => {
            let mut resp = response_stub_json(id, model, status, *created_at, None);
            if let Some(r) = reason {
                resp["incomplete_details"] = serde_json::json!({ "reason": r });
            }
            sse_event(
                "response.incomplete",
                serde_json::json!({
                    "type": "response.incomplete",
                    "response": resp,
                }),
            )
        }
        ResponseStreamEvent::RefusalDelta { item_id, output_index, content_index, delta } => {
            sse_event(
                "response.refusal.delta",
                serde_json::json!({
                    "type": "response.refusal.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            )
        }
        ResponseStreamEvent::RefusalDone { item_id, output_index, content_index, refusal } => {
            sse_event(
                "response.refusal.done",
                serde_json::json!({
                    "type": "response.refusal.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": refusal,
                }),
            )
        }
    }
}

fn sse_event(event_name: &str, payload: serde_json::Value) -> String {
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    format!("event: {event_name}
data: {data}

")
}

fn response_stub_json(
    id: &str,
    model: &str,
    status: &str,
    created_at: i64,
    request_context: Option<&ResponseRequestContext>,
) -> serde_json::Value {
    let mut resp = if let Some(ctx) = request_context {
        serde_json::to_value(ctx).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    resp["id"] = serde_json::json!(id);
    resp["object"] = serde_json::json!("response");
    resp["created_at"] = serde_json::json!(created_at);
    resp["status"] = serde_json::json!(status);
    resp["error"] = serde_json::Value::Null;
    resp["incomplete_details"] = serde_json::Value::Null;
    resp["model"] = serde_json::json!(model);
    resp["output"] = serde_json::json!([]);
    resp["usage"] = serde_json::Value::Null;

    if resp.get("text").is_none_or(|v| v.is_null()) {
        resp["text"] = serde_json::json!({"format":{"type":"text"}});
    }
    if resp.get("tools").is_none_or(|v| v.is_null()) {
        resp["tools"] = serde_json::json!([]);
    }

    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::types::response_api::{InputItemOrString, ResponseRequest, Tool, ToolChoice, ToolType};

    #[test]
    fn test_content_part_added_includes_part_payload() {
        let event = ResponseStreamEvent::ContentPartAdded {
            item_id: "msg_test".to_string(),
            output_index: 0,
            content_index: 0,
        };
        let sse = event_to_sse(&event);
        assert!(sse.contains("event: response.content_part.added"));
        assert!(sse.contains(r#""part":{"#));
        assert!(sse.contains(r#""type":"output_text""#));
        assert!(sse.contains(r#""annotations":[]"#));
    }

    #[test]
    fn test_output_text_done_includes_text_payload() {
        let event = ResponseStreamEvent::OutputTextDone {
            item_id: "msg_test".to_string(),
            output_index: 0,
            content_index: 0,
            text: "hello".to_string(),
        };
        let sse = event_to_sse(&event);
        assert!(sse.contains("event: response.output_text.done"));
        assert!(sse.contains(r#""text":"hello""#));
    }

    #[test]
    fn test_response_stub_json_defaults_text_when_missing() {
        let value = response_stub_json("resp_1", "gpt-x", "in_progress", 123, None);
        assert_eq!(
            value.get("text"),
            Some(&serde_json::json!({"format":{"type":"text"}}))
        );
    }

    #[test]
    fn test_request_context_includes_proxy_tool_map() {
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
        };
        let ctx = ResponseRequestContext::from(&req);
        let metadata = ctx.metadata.unwrap_or_default();
        assert!(metadata.contains_key("x_proxy_tool_map"));
    }

    #[test]
    fn test_sanitize_pseudo_tool_markup() {
        use crate::convert::util::sanitize_pseudo_tool_markup;
        let lt = "<";
        let text = format!(r#"<request_user_input">
{lt}parameter name="questions">x{lt}/parameter>
{lt}/request_user_input>"#);
        let sanitized = sanitize_pseudo_tool_markup(&text);
        assert!(sanitized.contains(r#"&lt;request_user_input"#));
        assert!(sanitized.contains(r#"&lt;parameter name="questions">"#));
        assert!(sanitized.contains(r#"&lt;/parameter&gt;"#));
        assert!(sanitized.contains(r#"&lt;/request_user_input&gt;"#));
    }
}
