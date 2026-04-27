//! Streaming conversion: Chat SSE chunks → Responses API event sequence.

use crate::error::ConversionError;
use crate::types::chat_api::{ChatStreamChunk, Content};
use crate::types::response_api::ResponseObject;

/// SSE event types for Responses API streaming.
#[derive(Debug, Clone)]
pub enum ResponseStreamEvent {
    /// Initial response created event.
    Created {
        id: String,
        model: String,
        status: String,
    },
    /// Response is in progress.
    InProgress {
        id: String,
    },
    /// Output item was added.
    OutputItemAdded {
        id: String,
        output_id: String,
        item_type: String,
    },
    /// Content part was added.
    ContentPartAdded {
        output_id: String,
        part_index: u32,
    },
    /// Output text delta (content chunk).
    OutputTextDelta {
        output_id: String,
        content: String,
    },
    /// Output text done.
    OutputTextDone {
        output_id: String,
    },
    /// Content part done.
    ContentPartDone {
        output_id: String,
    },
    /// Output item done.
    OutputItemDone {
        id: String,
    },
    /// Reasoning output item added.
    ReasoningAdded {
        id: String,
        output_id: String,
    },
    /// Reasoning text delta.
    ReasoningDelta {
        output_id: String,
        content: String,
    },
    /// Function call arguments delta.
    FunctionCallArgumentsDelta {
        output_id: String,
        name: String,
        arguments: String,
    },
    /// Function call arguments done.
    FunctionCallArgumentsDone {
        output_id: String,
        name: String,
        arguments: String,
    },
    /// Response completed with final object.
    Completed {
        response: ResponseObject,
    },
}

/// Streaming converter state for tracking incremental changes.
#[derive(Debug, Clone)]
pub struct StreamState {
    pub response_id: String,
    pub output_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub full_text: String,
    pub reasoning_text: String,
    pub is_first_chunk: bool,
    pub is_output_item_added: bool,
    pub is_content_part_added: bool,
    pub is_reasoning_added: bool,
    pub is_completed: bool,
    pub current_tool_call: Option<ToolCallState>,
    pub completed_tool_calls: Vec<ToolCallState>,
    pub tool_call_output_index: u32,
}

#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl StreamState {
    /// Create a new stream state.
    pub fn new(response_id: String) -> Self {
        Self {
            response_id: response_id.clone(),
            output_id: format!("msg_{}", response_id),
            output_index: 0,
            content_index: 0,
            full_text: String::new(),
            reasoning_text: String::new(),
            is_first_chunk: true,
            is_output_item_added: false,
            is_content_part_added: false,
            is_reasoning_added: false,
            is_completed: false,
            current_tool_call: None,
            completed_tool_calls: Vec::new(),
            tool_call_output_index: 1,
        }
    }
}

/// Convert a Chat API SSE chunk to Responses API SSE events.
pub fn chat_chunk_to_response_events(
    chunk: &ChatStreamChunk,
    state: &mut StreamState,
) -> Result<Vec<ResponseStreamEvent>, ConversionError> {
    let mut events = Vec::new();
    let id = chunk.id.clone().unwrap_or_else(|| state.response_id.clone());
    let model = chunk.model.as_deref().unwrap_or("unknown");

    // On first chunk, emit created and in_progress
    if state.is_first_chunk {
        events.push(ResponseStreamEvent::Created {
            id: id.to_string(),
            model: model.to_string(),
            status: "in_progress".to_string(),
        });
        events.push(ResponseStreamEvent::InProgress {
            id: id.to_string(),
        });
        state.is_first_chunk = false;
    }

    // Process each choice
    for choice in &chunk.choices {
        if let Some(delta) = &choice.delta {
            // Handle reasoning content (GLM extension)
            if let Some(reasoning) = &delta.reasoning_content {
                if !reasoning.is_empty() {
                    if !state.is_reasoning_added {
                        events.push(ResponseStreamEvent::ReasoningAdded {
                            id: id.to_string(),
                            output_id: format!("reasoning_{}", id),
                        });
                        state.is_reasoning_added = true;
                    }
                    events.push(ResponseStreamEvent::ReasoningDelta {
                        output_id: format!("reasoning_{}", id),
                        content: reasoning.clone(),
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
                    if !state.is_output_item_added {
                        events.push(ResponseStreamEvent::OutputItemAdded {
                            id: id.to_string(),
                            output_id: state.output_id.clone(),
                            item_type: "message".to_string(),
                        });
                        events.push(ResponseStreamEvent::ContentPartAdded {
                            output_id: state.output_id.clone(),
                            part_index: state.content_index,
                        });
                        state.is_output_item_added = true;
                        state.is_content_part_added = true;
                    }

                    events.push(ResponseStreamEvent::OutputTextDelta {
                        output_id: state.output_id.clone(),
                        content: text.clone(),
                    });
                    state.full_text.push_str(&text);
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = &delta.tool_calls {
                for tc in tool_calls {
                    // New tool call started
                    if state.current_tool_call.is_none() {
                        let output_id = format!("func_{}_{}", state.tool_call_output_index, id);
                        events.push(ResponseStreamEvent::OutputItemAdded {
                            id: id.to_string(),
                            output_id: output_id.clone(),
                            item_type: "function_call".to_string(),
                        });

                        state.current_tool_call = Some(ToolCallState {
                            id: tc.id.clone(),
                            name: tc.function.name.clone().unwrap_or_default(),
                            arguments: tc.function.arguments.clone().unwrap_or_default(),
                        });
                    }

                    // Accumulate function call arguments
                    if let Some(tc_state) = &mut state.current_tool_call {
                        if let Some(name) = &tc.function.name {
                            if !tc_state.name.ends_with("()") && !name.is_empty() {
                                tc_state.name = format!("{}(", name);
                            }
                        }
                        if let Some(args) = &tc.function.arguments {
                            tc_state.arguments.push_str(args);

                            events.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                                output_id: format!("func_{}_{}", state.tool_call_output_index, id),
                                name: tc_state.name.clone(),
                                arguments: tc_state.arguments.clone(),
                            });
                        }
                    }
                }
            }

            // Handle finish
            if let Some(reason) = &choice.finish_reason {
                if reason == "stop" || reason == "length" {
                    events.extend(finalize_output(state, &id));
                } else if reason == "tool_calls" {
                    // Tool calls finished, finalize them
                    if let Some(tc_state) = state.current_tool_call.take() {
                        let output_id = format!("func_{}_{}", state.tool_call_output_index, id);
                        events.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                            output_id: output_id.clone(),
                            name: tc_state.name.clone(),
                            arguments: tc_state.arguments.clone(),
                        });
                        events.push(ResponseStreamEvent::OutputItemDone {
                            id: output_id,
                        });
                        state.completed_tool_calls.push(tc_state);
                        state.tool_call_output_index += 1;
                    }
                }
            }
        }
    }

    Ok(events)
}

/// Finalize output items when stream ends.
fn finalize_output(state: &mut StreamState, id: &str) -> Vec<ResponseStreamEvent> {
    let mut events = Vec::new();

    if state.is_output_item_added {
        events.push(ResponseStreamEvent::OutputTextDone {
            output_id: state.output_id.clone(),
        });
        events.push(ResponseStreamEvent::ContentPartDone {
            output_id: state.output_id.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            id: state.output_id.clone(),
        });
    }

    if state.is_reasoning_added {
        events.push(ResponseStreamEvent::OutputItemDone {
            id: format!("reasoning_{}", id),
        });
    }

    state.is_completed = true;
    events
}

/// Generate SSE string from Response stream event.
pub fn event_to_sse(event: &ResponseStreamEvent) -> String {
    match event {
        ResponseStreamEvent::Created { id, model, status } => {
            format!(
                "event: response.created\ndata: {{\"id\":\"{}\",\"model\":\"{}\",\"status\":\"{}\"}}\n\n",
                escape_sse(id),
                escape_sse(model),
                status
            )
        }
        ResponseStreamEvent::InProgress { id } => {
            format!(
                "event: response.in_progress\ndata: {{\"id\":\"{}\"}}\n\n",
                escape_sse(id)
            )
        }
        ResponseStreamEvent::OutputItemAdded { id, output_id, item_type } => {
            format!(
                "event: response.output_item.added\ndata: {{\"id\":\"{}\",\"output_id\":\"{}\",\"type\":\"{}\"}}\n\n",
                escape_sse(id),
                escape_sse(output_id),
                item_type
            )
        }
        ResponseStreamEvent::ContentPartAdded { output_id, part_index } => {
            format!(
                "event: response.content_part.added\ndata: {{\"output_id\":\"{}\",\"index\":{}}}\n\n",
                escape_sse(output_id),
                part_index
            )
        }
        ResponseStreamEvent::OutputTextDelta { output_id, content } => {
            format!(
                "event: response.output_text.delta\ndata: {{\"output_id\":\"{}\",\"content\":{{\"text\":\"{}\"}}}}\n\n",
                escape_sse(output_id),
                escape_json_string(content)
            )
        }
        ResponseStreamEvent::OutputTextDone { output_id } => {
            format!(
                "event: response.output_text.done\ndata: {{\"output_id\":\"{}\"}}\n\n",
                escape_sse(output_id)
            )
        }
        ResponseStreamEvent::ContentPartDone { output_id } => {
            format!(
                "event: response.content_part.done\ndata: {{\"output_id\":\"{}\"}}\n\n",
                escape_sse(output_id)
            )
        }
        ResponseStreamEvent::OutputItemDone { id } => {
            format!(
                "event: response.output_item.done\ndata: {{\"id\":\"{}\"}}\n\n",
                escape_sse(id)
            )
        }
        ResponseStreamEvent::ReasoningAdded { id, output_id } => {
            format!(
                "event: response.output_item.added\ndata: {{\"id\":\"{}\",\"output_id\":\"{}\",\"type\":\"reasoning\"}}\n\n",
                escape_sse(id),
                escape_sse(output_id)
            )
        }
        ResponseStreamEvent::ReasoningDelta { output_id, content } => {
            format!(
                "event: response.output_text.delta\ndata: {{\"output_id\":\"{}\",\"content\":{{\"text\":\"{}\"}}}}\n\n",
                escape_sse(output_id),
                escape_json_string(content)
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDelta { output_id, name, arguments } => {
            format!(
                "event: response.function_call_arguments.delta\ndata: {{\"output_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}\n\n",
                escape_sse(output_id),
                escape_json_string(name),
                escape_json_string(arguments)
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDone { output_id, name, arguments } => {
            format!(
                "event: response.function_call_arguments.done\ndata: {{\"output_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}\n\n",
                escape_sse(output_id),
                escape_json_string(name),
                escape_json_string(arguments)
            )
        }
        ResponseStreamEvent::Completed { response } => {
            format!(
                "event: response.completed\ndata: {}\n\n",
                serde_json::to_string(response).unwrap_or_default()
            )
        }
    }
}

fn escape_sse(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n").replace('\r', "\\r")
}

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
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
        };

        let mut state = StreamState::new("chat_123".to_string());
        let events = chat_chunk_to_response_events(&chunk, &mut state).unwrap();

        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::Created { .. })));
        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::InProgress { .. })));
        assert!(events.iter().any(|e| matches!(e, ResponseStreamEvent::OutputTextDelta { content, .. } if content == "Hello")));
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
                        id: "call_abc".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCallDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"city":"Beijing"}"#.to_string()),
                        },
                    }]),
                    reasoning_content: None,
                }),
                finish_reason: None,
            }],
        };

        let mut state = StreamState::new("chat_123".to_string());
        // First chunk to establish state
        let _ = chat_chunk_to_response_events(&chunk, &mut state);

        assert!(state.current_tool_call.is_some());
        let tc = state.current_tool_call.as_ref().unwrap();
        assert_eq!(tc.name, "get_weather(");
    }
}
