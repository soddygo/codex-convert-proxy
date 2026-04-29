//! Core conversion logic: Chat SSE chunks to Responses API stream events.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatStreamChunk, Content};

use super::events::ResponseStreamEvent;
use super::state::{StreamState, ToolCallState};
use super::super::util::{
    map_tool_name_to_stream_item_type, parse_streaming_thinking, sanitize_pseudo_tool_markup,
};

/// Convert a Chat API SSE chunk to Responses API SSE events.
pub fn chat_chunk_to_response_events(
    chunk: &ChatStreamChunk,
    state: &mut StreamState,
) -> Result<Vec<ResponseStreamEvent>, ConversionError> {
    let mut events = Vec::new();
    let id = state.response_id.clone();
    let model = chunk.model.as_deref().unwrap_or("unknown");

    // On first chunk, emit created and in_progress
    if state.is_first_chunk {
        let created_at = chrono::Utc::now().timestamp();
        events.push(ResponseStreamEvent::Created {
            id: id.to_string(),
            model: model.to_string(),
            status: "in_progress".to_string(),
            created_at,
            request_context: state.request_context.clone(),
        });
        events.push(ResponseStreamEvent::InProgress {
            id: id.to_string(),
            model: model.to_string(),
            status: "in_progress".to_string(),
            created_at,
            request_context: state.request_context.clone(),
        });
        state.is_first_chunk = false;
    }

    // Process each choice
    for choice in &chunk.choices {
        if let Some(delta) = &choice.delta {
            tracing::debug!("[DELTA] content={:?}, tool_calls={:?}, reasoning_content={:?}",
                delta.content.is_some(),
                delta.tool_calls.as_ref().map(|tc| tc.len()),
                delta.reasoning_content.as_ref().map(|r| r.len()));
            // Handle reasoning content (GLM extension)
            if let Some(reasoning) = &delta.reasoning_content {
                if !reasoning.is_empty() {
                    if !state.is_reasoning_added {
                        let reasoning_id = format!("reasoning_{}", id);
                        let reasoning_idx = state.next_output_index;
                        state.next_output_index += 1;
                        state.reasoning_output_index = Some(reasoning_idx);
                        events.push(ResponseStreamEvent::ReasoningAdded {
                            output_index: reasoning_idx,
                            item_id: reasoning_id.clone(),
                        });
                        state.is_reasoning_added = true;
                    }
                    let reasoning_idx = state.reasoning_output_index.unwrap_or(0);
                    events.push(ResponseStreamEvent::ReasoningDelta {
                        item_id: format!("reasoning_{}", id),
                        output_index: reasoning_idx,
                        content_index: 0,
                        delta: reasoning.clone(),
                    });
                    state.reasoning_text.push_str(reasoning);
                }
            }

            // Handle text content
            if let Some(content) = &delta.content {
                let text = match content {
                    Content::String(s) => s.clone(),
                    Content::Array(arr) => arr
                        .iter()
                        .filter_map(|b| b.text.clone())
                        .collect::<Vec<_>>()
                        .join(""),
                };

                if !text.is_empty() {
                    // Parse thinking tags (<think> or <thought>) from content
                    let (actual_text, reasoning_delta, new_is_thinking) =
                        parse_streaming_thinking(&text, state.is_thinking, &mut state.thinking_buffer);
                    let sanitized_actual_text = sanitize_pseudo_tool_markup(&actual_text);

                    state.is_thinking = new_is_thinking;

                    // Emit reasoning events if we have reasoning content
                    if let Some(reasoning) = reasoning_delta {
                        if !reasoning.is_empty() {
                            if !state.is_reasoning_added {
                                let reasoning_id = format!("reasoning_{}", id);
                                let reasoning_idx = state.next_output_index;
                                state.next_output_index += 1;
                                state.reasoning_output_index = Some(reasoning_idx);
                                events.push(ResponseStreamEvent::ReasoningAdded {
                                    output_index: reasoning_idx,
                                    item_id: reasoning_id.clone(),
                                });
                                state.is_reasoning_added = true;
                            }
                            let reasoning_idx = state.reasoning_output_index.unwrap_or(0);
                            events.push(ResponseStreamEvent::ReasoningDelta {
                                item_id: format!("reasoning_{}", id),
                                output_index: reasoning_idx,
                                content_index: 0,
                                delta: reasoning.clone(),
                            });
                            state.reasoning_text.push_str(&reasoning);
                        }
                    }

                    // Emit text events if we have actual content
                    if !sanitized_actual_text.is_empty() {
                        if !state.is_output_item_added {
                            let text_idx = state.next_output_index;
                            state.next_output_index += 1;
                            state.text_output_index = Some(text_idx);
                            events.push(ResponseStreamEvent::OutputItemAdded {
                                output_index: text_idx,
                                item_id: state.output_id.clone(),
                                item_type: "message".to_string(),
                                role: Some("assistant".to_string()),
                                call_id: None,
                            });
                            events.push(ResponseStreamEvent::ContentPartAdded {
                                item_id: state.output_id.clone(),
                                output_index: text_idx,
                                content_index: 0,
                            });
                            state.is_output_item_added = true;
                            state.is_content_part_added = true;
                        }

                        let text_idx = state.text_output_index.unwrap_or(0);
                        events.push(ResponseStreamEvent::OutputTextDelta {
                            item_id: state.output_id.clone(),
                            output_index: text_idx,
                            content_index: 0,
                            delta: sanitized_actual_text.clone(),
                        });
                        state.full_text.push_str(&sanitized_actual_text);
                    }
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = &delta.tool_calls {
                tracing::debug!("[TOOL_CALL] Processing {} tool calls in chunk", tool_calls.len());
                for tc in tool_calls {
                    tracing::debug!("[TOOL_CALL] Tool call: id={:?}, index={}, name={:?}, args_len={}",
                        tc.id, tc.index, tc.function.name, tc.function.arguments.as_ref().map(|a| a.len()).unwrap_or(0));

                    let existing_idx = if let Some(tc_id) = tc.id.as_ref() {
                        state.current_tool_calls.iter().position(|t| t.upstream_id.as_ref() == Some(tc_id))
                    } else {
                        state.current_tool_calls.iter().position(|t| t.chat_api_index == tc.index)
                    };
                    tracing::debug!("[TOOL_CALL] existing_idx={:?}, tc.index={}", existing_idx, tc.index);

                    if existing_idx.is_none() {
                        if let Some(tc_id) = tc.id.clone() {
                            let func_output_index = state.next_output_index;
                            state.next_output_index += 1;
                            let func_id = format!("func_{}_{}", func_output_index, state.response_id);
                            let initial_name = tc.function.name.clone().unwrap_or_default();
                            let item_type = map_tool_name_to_stream_item_type(&initial_name, state.request_context.as_ref());
                            tracing::debug!("[TOOL_CALL] Creating new tool call: func_id={}, output_index={}", func_id, func_output_index);
                            events.push(ResponseStreamEvent::OutputItemAdded {
                                output_index: func_output_index,
                                item_id: func_id.clone(),
                                item_type: item_type.clone(),
                                role: None,
                                call_id: Some(tc_id.clone()),
                            });
                            state.is_function_call_item_added = true;

                            let initial_args = tc.function.arguments.clone().unwrap_or_default();
                            let tc_state = ToolCallState {
                                upstream_id: Some(tc_id.clone()),
                                id: func_id.clone(),
                                call_id: tc_id,
                                item_type,
                                name: initial_name,
                                arguments: initial_args.clone(),
                                output_index: func_output_index,
                                chat_api_index: tc.index,
                                last_args_len: initial_args.len(),
                            };

                            state.current_tool_calls.push(tc_state);

                            events.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                                output_index: func_output_index,
                                item_id: func_id,
                                delta: initial_args,
                            });
                            tracing::debug!("[TOOL_CALL] Emitted OutputItemAdded and FunctionCallArgumentsDelta, total events now: {}", events.len());
                        } else {
                            tracing::debug!("[TOOL_CALL] Skipping tool call - no id and no matching index {}", tc.index);
                        }
                    } else if let Some(idx) = existing_idx {
                        let tc_state = &mut state.current_tool_calls[idx];
                        if let Some(args) = &tc.function.arguments {
                            let prev_len = tc_state.last_args_len;
                            let new_delta = if args.len() > prev_len && args.starts_with(&tc_state.arguments) {
                                let delta = args[prev_len..].to_string();
                                tc_state.arguments = args.clone();
                                tc_state.last_args_len = args.len();
                                delta
                            } else {
                                let delta = args.clone();
                                tc_state.arguments.push_str(args);
                                tc_state.last_args_len = tc_state.arguments.len();
                                delta
                            };

                            if !new_delta.is_empty() {
                                events.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                                    output_index: tc_state.output_index,
                                    item_id: tc_state.id.clone(),
                                    delta: new_delta,
                                });
                            }
                        }
                        if let Some(name) = &tc.function.name {
                            if !name.is_empty() && tc_state.name.is_empty() {
                                tc_state.name = name.clone();
                            }
                        }
                    }
                }
            }

            // Handle finish
            tracing::debug!("[FINISH_REASON] choice.finish_reason={:?}, current_tool_calls_len={}", choice.finish_reason, state.current_tool_calls.len());
            if let Some(reason) = &choice.finish_reason {
                tracing::debug!("[FINISH_REASON] reason={}", reason);
                if reason == "stop" || reason == "length" {
                    events.extend(finalize_output(state, &id));
                } else if reason == "tool_calls" {
                    for tc_state in state.current_tool_calls.drain(..) {
                        events.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                            output_index: tc_state.output_index,
                            item_id: tc_state.id.clone(),
                            call_id: tc_state.call_id.clone(),
                            name: tc_state.name.clone(),
                            arguments: tc_state.arguments.clone(),
                        });
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: tc_state.output_index,
                            item_id: tc_state.id.clone(),
                            item_type: tc_state.item_type.clone(),
                            role: None,
                            call_id: Some(tc_state.call_id.clone()),
                            name: Some(tc_state.name.clone()),
                            arguments: Some(tc_state.arguments.clone()),
                            text: None,
                        });
                        state.completed_tool_calls.push(tc_state);
                    }
                    if state.is_output_item_added {
                        let text_idx = state.text_output_index.unwrap_or(0);
                        events.push(ResponseStreamEvent::OutputTextDone {
                            item_id: state.output_id.clone(),
                            output_index: text_idx,
                            content_index: 0,
                            text: state.full_text.clone(),
                        });
                        events.push(ResponseStreamEvent::ContentPartDone {
                            item_id: state.output_id.clone(),
                            output_index: text_idx,
                            content_index: 0,
                            text: state.full_text.clone(),
                        });
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: text_idx,
                            item_id: state.output_id.clone(),
                            item_type: "message".to_string(),
                            role: Some("assistant".to_string()),
                            call_id: None,
                            name: None,
                            arguments: None,
                            text: Some(state.full_text.clone()),
                        });
                    }
                    if state.is_reasoning_added {
                        let reasoning_idx = state.reasoning_output_index.unwrap_or(0);
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: reasoning_idx,
                            item_id: format!("reasoning_{}", id),
                            item_type: "reasoning".to_string(),
                            role: None,
                            call_id: None,
                            name: None,
                            arguments: None,
                            text: Some(state.reasoning_text.clone()),
                        });
                    }
                }
            }
        }
    }

    tracing::debug!("[CHUNK_EVENTS] Generated {} events: {:?}", events.len(),
        events.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>());
    Ok(events)
}

/// Finalize output items when stream ends.
fn finalize_output(state: &mut StreamState, id: &str) -> Vec<ResponseStreamEvent> {
    let mut events = Vec::new();

    tracing::debug!("[FINALIZE] is_output_item_added={}, is_reasoning_added={}, current_tool_calls={}",
        state.is_output_item_added, state.is_reasoning_added, state.current_tool_calls.len());

    // Finalize any pending tool calls
    for tc_state in state.current_tool_calls.drain(..) {
        events.push(ResponseStreamEvent::FunctionCallArgumentsDone {
            output_index: tc_state.output_index,
            item_id: tc_state.id.clone(),
            call_id: tc_state.call_id.clone(),
            name: tc_state.name.clone(),
            arguments: tc_state.arguments.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: tc_state.output_index,
            item_id: tc_state.id.clone(),
            item_type: tc_state.item_type.clone(),
            role: None,
            call_id: Some(tc_state.call_id.clone()),
            name: Some(tc_state.name.clone()),
            arguments: Some(tc_state.arguments.clone()),
            text: None,
        });
        state.completed_tool_calls.push(tc_state);
    }

    if state.is_output_item_added {
        let text_idx = state.text_output_index.unwrap_or(0);
        events.push(ResponseStreamEvent::OutputTextDone {
            item_id: state.output_id.clone(),
            output_index: text_idx,
            content_index: 0,
            text: state.full_text.clone(),
        });
        events.push(ResponseStreamEvent::ContentPartDone {
            item_id: state.output_id.clone(),
            output_index: text_idx,
            content_index: 0,
            text: state.full_text.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: text_idx,
            item_id: state.output_id.clone(),
            item_type: "message".to_string(),
            role: Some("assistant".to_string()),
            call_id: None,
            name: None,
            arguments: None,
            text: Some(state.full_text.clone()),
        });
    }

    if state.is_reasoning_added {
        let reasoning_idx = state.reasoning_output_index.unwrap_or(0);
        let reasoning_id = format!("reasoning_{}", id);
        events.push(ResponseStreamEvent::ReasoningTextDone {
            item_id: reasoning_id.clone(),
            output_index: reasoning_idx,
            content_index: 0,
            text: state.reasoning_text.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: reasoning_idx,
            item_id: reasoning_id,
            item_type: "reasoning".to_string(),
            role: None,
            call_id: None,
            name: None,
            arguments: None,
            text: Some(state.reasoning_text.clone()),
        });
    }

    tracing::debug!("[FINALIZE] Produced {} events", events.len());
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatDelta, ChatStreamChoice, Content, ToolCallDelta, FunctionCallDelta};

    #[test]
    fn test_first_chunk_generates_created_event() {
        let chunk = ChatStreamChunk {
            id: Some("chat_123".to_string()),
            object: Some("chat.completion.chunk".to_string()),
            created: Some(1234567890),
            model: Some("gpt-4o".to_string()),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: Some(ChatDelta {
                    role: Some("assistant".to_string()),
                    content: Some(Content::String("Hello".to_string())),
                    tool_calls: None,
                    reasoning_content: None,
                }),
                finish_reason: None,
            }],
            usage: None,
        };

        let mut state = StreamState::new("chat_123".to_string(), "gpt-4o".to_string(), None);
        let events = chat_chunk_to_response_events(&chunk, &mut state).unwrap();

        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::Created { .. })));
        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::InProgress { .. })));
        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::OutputTextDelta { delta, .. } if delta == "Hello")));
    }

    #[test]
    fn test_tool_call_generates_function_call_events() {
        let chunk = ChatStreamChunk {
            id: Some("chat_123".to_string()),
            object: Some("chat.completion.chunk".to_string()),
            created: Some(1234567890),
            model: Some("gpt-4o".to_string()),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: Some(ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index: 0,
                        id: Some("call_abc".to_string()),
                        tool_type: Some("function".to_string()),
                        function: FunctionCallDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                        },
                    }]),
                    reasoning_content: None,
                }),
                finish_reason: None,
            }],
            usage: None,
        };

        let mut state = StreamState::new("chat_123".to_string(), "gpt-4o".to_string(), None);
        let _ = chat_chunk_to_response_events(&chunk, &mut state);

        assert!(!state.current_tool_calls.is_empty());
        let tc = state.current_tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
    }

    #[test]
    fn test_parse_streaming_thinking_basic() {
        use crate::convert::util::parse_streaming_thinking;
        let mut buffer = String::new();
        let (actual, reasoning, is_thinking) = parse_streaming_thinking("Hello world", false, &mut buffer);
        assert_eq!(actual, "Hello world");
        assert!(reasoning.is_none());
        assert!(!is_thinking);
    }

    #[test]
    fn test_parse_streaming_thinking_with_think_tag() {
        use crate::convert::util::parse_streaming_thinking;
        let mut buffer = String::new();
        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            "<think>\nreasoning\n</think>\n\nactual text",
            false,
            &mut buffer,
        );
        assert_eq!(actual, "\n\nactual text");
        assert_eq!(reasoning, Some("\nreasoning\n".to_string()));
        assert!(!is_thinking);
    }

    #[test]
    fn test_parse_streaming_thinking_chunked() {
        use crate::convert::util::parse_streaming_thinking;
        let mut buffer = String::new();

        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            "<think>\npartial",
            false,
            &mut buffer,
        );
        assert_eq!(actual, "");
        assert!(reasoning.is_none());
        assert!(is_thinking);

        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            " content\n</think>\n\nfinal",
            is_thinking,
            &mut buffer,
        );
        assert_eq!(actual, "\n\nfinal");
        assert_eq!(reasoning, Some("\npartial content\n".to_string()));
        assert!(!is_thinking);
    }

    #[test]
    fn test_parse_streaming_thought_tag() {
        use crate::convert::util::parse_streaming_thinking;
        let mut buffer = String::new();
        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            "<thought>reasoning</thought>actual",
            false,
            &mut buffer,
        );
        assert_eq!(actual, "actual");
        assert_eq!(reasoning, Some("reasoning".to_string()));
        assert!(!is_thinking);
    }
}
