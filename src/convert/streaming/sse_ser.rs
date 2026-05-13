//! SSE serialization: ResponseStreamEvent to SSE string format.

use super::events::ResponseStreamEvent;
use crate::convert::context::ResponseRequestContext;

/// Generate SSE string from a Response stream event.
///
/// `sequence_number` is the spec-required monotonic counter; callers should
/// allocate it from `StreamState::take_sequence_number()` immediately before
/// invoking this function so events come out in stream order.
pub fn event_to_sse(event: &ResponseStreamEvent, seq: u64) -> String {
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
                seq,
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
                seq,
            )
        }
        ResponseStreamEvent::OutputItemAdded { output_index, item_id, item_type, role, call_id, name } => {
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
            if let Some(n) = name {
                item.insert("name".to_string(), serde_json::json!(n));
            }
            if item_type == "message" {
                item.insert("content".to_string(), serde_json::json!([]));
            }
            if item_type == "reasoning" {
                item.insert("summary".to_string(), serde_json::json!([]));
                item.insert("content".to_string(), serde_json::json!([]));
            }
            if item_type == "function_call" {
                item.insert("arguments".to_string(), serde_json::json!(""));
            }
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": serde_json::Value::Object(item),
                }),
                seq,
            )
        }
        ResponseStreamEvent::ContentPartAdded { item_id, output_index, content_index, part_type } => {
            let part: serde_json::Value = if part_type == "refusal" {
                serde_json::json!({
                    "type": "refusal",
                    "refusal": "",
                })
            } else {
                serde_json::json!({
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": [],
                })
            };
            sse_event(
                "response.content_part.added",
                serde_json::json!({
                    "type": "response.content_part.added",
                    "output_index": output_index,
                    "item_id": item_id,
                    "content_index": content_index,
                    "part": part,
                }),
                seq,
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
                    "logprobs": [],
                }),
                seq,
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
                    "logprobs": [],
                }),
                seq,
            )
        }
        ResponseStreamEvent::ContentPartDone {
            item_id,
            output_index,
            content_index,
            part_type,
            text,
        } => {
            let part: serde_json::Value = if part_type == "refusal" {
                serde_json::json!({
                    "type": "refusal",
                    "refusal": text,
                })
            } else {
                serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                    "logprobs": [],
                })
            };
            sse_event(
                "response.content_part.done",
                serde_json::json!({
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
                seq,
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
            refusal,
            summary,
        } => {
            let mut item = serde_json::Map::new();
            item.insert("id".to_string(), serde_json::json!(item_id));
            item.insert("type".to_string(), serde_json::json!(item_type));
            if item_type != "reasoning" {
                item.insert("status".to_string(), serde_json::json!("completed"));
            }
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
            let mut content_parts = Vec::new();
            if let Some(body_text) = text {
                content_parts.push(serde_json::json!({
                    "type": "output_text",
                    "text": body_text,
                    "annotations": [],
                    "logprobs": [],
                }));
            }
            if let Some(refusal_text) = refusal {
                content_parts.push(serde_json::json!({
                    "type": "refusal",
                    "refusal": refusal_text,
                }));
            }
            if !content_parts.is_empty() {
                item.insert("content".to_string(), serde_json::Value::Array(content_parts));
            }
            if let Some(summary_parts) = summary {
                let parts: Vec<serde_json::Value> = summary_parts
                    .iter()
                    .map(|p| serde_json::json!(p))
                    .collect();
                item.insert("summary".to_string(), serde_json::Value::Array(parts));
            }
            sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": serde_json::Value::Object(item),
                }),
                seq,
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
                        "summary": [],
                        "content": [],
                    },
                }),
                seq,
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
                seq,
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
                seq,
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
                seq,
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
                seq,
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
                seq,
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDone { output_index, item_id, name, arguments } => {
            sse_event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "item_id": item_id,
                    "name": name,
                    "arguments": arguments,
                }),
                seq,
            )
        }
        ResponseStreamEvent::Completed { response } => {
            sse_event(
                "response.completed",
                serde_json::json!({
                    "type": "response.completed",
                    "response": response,
                }),
                seq,
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
            sse_event("response.error", payload, seq)
        }
        ResponseStreamEvent::Failed { id, model, status, created_at } => {
            sse_event(
                "response.failed",
                serde_json::json!({
                    "type": "response.failed",
                    "response": response_stub_json(id, model, status, *created_at, None),
                }),
                seq,
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
                seq,
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
                seq,
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
                seq,
            )
        }
    }
}

fn sse_event(event_name: &str, mut payload: serde_json::Value, sequence_number: u64) -> String {
    if let Some(obj) = payload.as_object_mut() {
        obj.insert(
            "sequence_number".to_string(),
            serde_json::json!(sequence_number),
        );
    }
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    format!("event: {event_name}
data: {data}

")
}

/// Build the stub `response` payload embedded in `response.created` /
/// `response.in_progress` / `response.failed` / `response.incomplete` events.
///
/// Uses the typed `ResponseObject::stub` constructor so the streaming stub
/// and the final `response.completed` payload share a single schema source.
fn response_stub_json(
    id: &str,
    model: &str,
    status: &str,
    created_at: i64,
    request_context: Option<&ResponseRequestContext>,
) -> serde_json::Value {
    let stub = crate::types::response_api::ResponseObject::stub(
        id.to_string(),
        model.to_string(),
        status.to_string(),
        created_at,
        request_context,
    );
    serde_json::to_value(&stub).unwrap_or(serde_json::json!({}))
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
            part_type: "output_text".to_string(),
        };
        let sse = event_to_sse(&event, 1);
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
        let sse = event_to_sse(&event, 1);
        assert!(sse.contains("event: response.output_text.done"));
        assert!(sse.contains(r#""text":"hello""#));
    }

    #[test]
    fn test_output_item_done_includes_refusal_content_part() {
        let event = ResponseStreamEvent::OutputItemDone {
            output_index: 0,
            item_id: "msg_ref".to_string(),
            item_type: "message".to_string(),
            role: Some("assistant".to_string()),
            call_id: None,
            name: None,
            arguments: None,
            text: None,
            refusal: Some("Not allowed".to_string()),
            summary: None,
        };
        let sse = event_to_sse(&event, 1);
        assert!(sse.contains("event: response.output_item.done"));
        assert!(sse.contains(r#""type":"refusal""#));
        assert!(sse.contains(r#""refusal":"Not allowed""#));
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
            background: None,
        };
        let ctx = ResponseRequestContext::from(&req);
        let metadata = ctx.metadata.unwrap_or_default();
        assert!(metadata.contains_key("x_proxy_tool_map"));
    }

    #[test]
    fn test_every_event_carries_sequence_number() {
        // Spec requires `sequence_number` on every Responses streaming event.
        let cases: Vec<(ResponseStreamEvent, &str)> = vec![
            (
                ResponseStreamEvent::OutputTextDelta {
                    item_id: "msg_1".into(),
                    output_index: 0,
                    content_index: 0,
                    delta: "hi".into(),
                },
                "response.output_text.delta",
            ),
            (
                ResponseStreamEvent::FunctionCallArgumentsDone {
                    output_index: 0,
                    item_id: "fc_1".into(),
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                },
                "response.function_call_arguments.done",
            ),
            (
                ResponseStreamEvent::RefusalDelta {
                    item_id: "msg_1".into(),
                    output_index: 0,
                    content_index: 0,
                    delta: "no".into(),
                },
                "response.refusal.delta",
            ),
        ];
        for (event, event_name) in cases {
            let sse = event_to_sse(&event, 42);
            assert!(sse.contains(&format!("event: {event_name}")));
            assert!(
                sse.contains(r#""sequence_number":42"#),
                "missing sequence_number for {event_name}: {sse}"
            );
        }
    }

    #[test]
    fn test_output_text_events_include_logprobs() {
        let delta = ResponseStreamEvent::OutputTextDelta {
            item_id: "msg".into(),
            output_index: 0,
            content_index: 0,
            delta: "hi".into(),
        };
        let sse = event_to_sse(&delta, 1);
        assert!(sse.contains(r#""logprobs":[]"#), "delta missing logprobs: {sse}");

        let done = ResponseStreamEvent::OutputTextDone {
            item_id: "msg".into(),
            output_index: 0,
            content_index: 0,
            text: "hi".into(),
        };
        let sse = event_to_sse(&done, 2);
        assert!(sse.contains(r#""logprobs":[]"#), "done missing logprobs: {sse}");
    }

    #[test]
    fn test_function_call_arguments_done_uses_name_not_call_id() {
        let event = ResponseStreamEvent::FunctionCallArgumentsDone {
            output_index: 0,
            item_id: "fc_1".into(),
            name: "lookup".into(),
            arguments: r#"{"q":"x"}"#.into(),
        };
        let sse = event_to_sse(&event, 1);
        assert!(sse.contains(r#""name":"lookup""#), "missing name: {sse}");
        assert!(!sse.contains(r#""call_id""#), "stray call_id: {sse}");
    }

    #[test]
    fn test_output_item_added_function_call_has_arguments() {
        let event = ResponseStreamEvent::OutputItemAdded {
            output_index: 0,
            item_id: "fc_1".into(),
            item_type: "function_call".into(),
            role: None,
            call_id: Some("call_x".into()),
            name: Some("lookup".into()),
        };
        let sse = event_to_sse(&event, 1);
        assert!(sse.contains(r#""arguments":"""#), "missing empty arguments: {sse}");
    }

    #[test]
    fn test_reasoning_added_has_summary_array() {
        let event = ResponseStreamEvent::ReasoningAdded {
            output_index: 0,
            item_id: "rs_1".into(),
        };
        let sse = event_to_sse(&event, 1);
        assert!(sse.contains(r#""summary":[]"#), "missing summary: {sse}");
    }

    #[test]
    fn test_response_stub_json_backfills_required_fields_when_no_context() {
        // Failed/Incomplete events pass None for request_context; the stub must still
        // satisfy Response.required (tools, parallel_tool_calls, metadata, tool_choice,
        // instructions, temperature, top_p).
        let value = response_stub_json("resp_1", "gpt-x", "failed", 0, None);
        let obj = value.as_object().expect("stub is object");
        for key in [
            "tools",
            "parallel_tool_calls",
            "metadata",
            "tool_choice",
            "instructions",
            "temperature",
            "top_p",
            "error",
            "incomplete_details",
        ] {
            assert!(obj.contains_key(key), "stub missing required key {key}");
        }
        assert_eq!(value.get("parallel_tool_calls"), Some(&serde_json::json!(true)));
        assert_eq!(value.get("tool_choice"), Some(&serde_json::json!("auto")));
        assert_eq!(value.get("metadata"), Some(&serde_json::json!({})));
        assert_eq!(value.get("tools"), Some(&serde_json::json!([])));
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
