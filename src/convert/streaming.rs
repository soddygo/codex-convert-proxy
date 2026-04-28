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
        created_at: i64,
    },
    /// Response is in progress.
    InProgress {
        id: String,
    },
    /// Output item was added.
    OutputItemAdded {
        output_index: u32,
        item_id: String,
        item_type: String,
        role: Option<String>,
        call_id: Option<String>,
    },
    /// Content part was added.
    ContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    /// Output text delta (content chunk).
    OutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Output text done.
    OutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    /// Content part done.
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    /// Output item done.
    OutputItemDone {
        output_index: u32,
        item_id: String,
        item_type: String,
        role: Option<String>,
        call_id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    /// Reasoning output item added.
    ReasoningAdded {
        output_index: u32,
        item_id: String,
    },
    /// Reasoning text delta.
    ReasoningDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Function call arguments delta.
    FunctionCallArgumentsDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// Function call arguments done.
    FunctionCallArgumentsDone {
        output_index: u32,
        item_id: String,
        call_id: String,
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
    pub is_function_call_item_added: bool,
    pub is_completed: bool,
    pub current_tool_calls: Vec<ToolCallState>,
    pub completed_tool_calls: Vec<ToolCallState>,
    pub model: String,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub upstream_id: Option<String>,  // Original ID from upstream provider (None if malformed)
    pub id: String,           // Internal ID we generate
    pub name: String,
    pub arguments: String,
    pub output_index: u32,     // Each tool call stores its own output_index
    pub last_args_len: usize,  // Track for delta calculation
}

impl StreamState {
    /// Create a new stream state.
    pub fn new(response_id: String, model: String) -> Self {
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
            is_function_call_item_added: false,
            is_completed: false,
            current_tool_calls: Vec::new(),
            completed_tool_calls: Vec::new(),
            model,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        }
    }

    /// Update usage from a ChatStreamChunk.
    pub fn update_usage(&mut self, chunk: &ChatStreamChunk) {
        if let Some(usage) = &chunk.usage {
            self.input_tokens = usage.prompt_tokens.map(|v| v as i64);
            self.output_tokens = usage.completion_tokens.map(|v| v as i64);
            self.total_tokens = usage.total_tokens.map(|v| v as i64);
        }
    }

    /// Build the final ResponseObject with all accumulated outputs.
    pub fn build_response_object(&self) -> crate::types::response_api::ResponseObject {
        use crate::types::response_api::{
            ResponseContentPart, ResponseObject, ResponseOutputItem,
            OutputItemType, Usage,
        };
        use chrono::Utc;

        let mut output = Vec::new();

        // Add reasoning output if present
        if self.is_reasoning_added && !self.reasoning_text.is_empty() {
            output.push(ResponseOutputItem {
                id: format!("reasoning_{}", self.response_id),
                item_type: OutputItemType::Reasoning,
                status: Some("completed".to_string()),
                content: Some(vec![ResponseContentPart::OutputText {
                    text: self.reasoning_text.clone(),
                }]),
                name: None,
                arguments: None,
                call_id: None,
            });
        }

        // Add text output if present
        if self.is_output_item_added && !self.full_text.is_empty() {
            output.push(ResponseOutputItem {
                id: self.output_id.clone(),
                item_type: OutputItemType::Message,
                status: Some("completed".to_string()),
                content: Some(vec![ResponseContentPart::OutputText {
                    text: self.full_text.clone(),
                }]),
                name: Some("assistant".to_string()),
                arguments: None,
                call_id: None,
            });
        }

        // Add function call outputs
        for tc in &self.completed_tool_calls {
            output.push(ResponseOutputItem {
                id: tc.id.clone(),
                item_type: OutputItemType::FunctionCall,
                status: Some("completed".to_string()),
                content: None,
                name: Some(tc.name.clone()),
                arguments: Some(tc.arguments.clone()),
                call_id: Some(tc.id.clone()),
            });
        }

        let usage = if self.input_tokens.is_some() || self.output_tokens.is_some() || self.total_tokens.is_some() {
            Some(Usage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                total_tokens: self.total_tokens,
            })
        } else {
            None
        };

        ResponseObject {
            id: self.response_id.clone(),
            object: "response".to_string(),
            status: "completed".to_string(),
            model: self.model.clone(),
            created_at: Utc::now().timestamp(),
            completed_at: Some(Utc::now().timestamp()),
            input: None,  // Input not available in streaming context
            output,
            usage,
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
        let created_at = chrono::Utc::now().timestamp();
        events.push(ResponseStreamEvent::Created {
            id: id.to_string(),
            model: model.to_string(),
            status: "in_progress".to_string(),
            created_at,
        });
        events.push(ResponseStreamEvent::InProgress {
            id: id.to_string(),
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
                        events.push(ResponseStreamEvent::ReasoningAdded {
                            output_index: 0,
                            item_id: reasoning_id.clone(),
                        });
                        state.is_reasoning_added = true;
                    }
                    events.push(ResponseStreamEvent::ReasoningDelta {
                        item_id: format!("reasoning_{}", id),
                        output_index: 0,
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
                    if !state.is_output_item_added {
                        events.push(ResponseStreamEvent::OutputItemAdded {
                            output_index: 0,
                            item_id: state.output_id.clone(),
                            item_type: "message".to_string(),
                            role: Some("assistant".to_string()),
                            call_id: None,
                        });
                        events.push(ResponseStreamEvent::ContentPartAdded {
                            item_id: state.output_id.clone(),
                            output_index: 0,
                            content_index: 0,
                        });
                        state.is_output_item_added = true;
                        state.is_content_part_added = true;
                    }

                    events.push(ResponseStreamEvent::OutputTextDelta {
                        item_id: state.output_id.clone(),
                        output_index: 0,
                        content_index: 0,
                        delta: text.clone(),
                    });
                    state.full_text.push_str(&text);
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = &delta.tool_calls {
                tracing::debug!("[TOOL_CALL] Processing {} tool calls in chunk", tool_calls.len());
                for tc in tool_calls {
                    tracing::debug!("[TOOL_CALL] Tool call: id={:?}, index={}, name={:?}, args_len={}",
                        tc.id, tc.index, tc.function.name, tc.function.arguments.as_ref().map(|a| a.len()).unwrap_or(0));

                    // Try to find existing tool call by upstream_id or by index
                    let existing_idx = if let Some(tc_id) = tc.id.as_ref() {
                        // Has ID -> match by upstream_id
                        state.current_tool_calls.iter().position(|t| t.upstream_id.as_ref() == Some(tc_id))
                    } else {
                        // No ID (incremental arguments chunk) -> match by index
                        state.current_tool_calls.iter().position(|t| t.output_index == tc.index + 1)
                    };
                    tracing::debug!("[TOOL_CALL] existing_idx={:?}, tc.index={}", existing_idx, tc.index);

                    if existing_idx.is_none() && tc.id.is_some() {
                        // New tool call (has id) - use tc.index + 1 as output_index (text=0, reasoning=0, first func_call=1)
                        let tc_id = tc.id.as_ref().unwrap().clone();
                        let func_output_index = tc.index + 1;
                        let func_id = format!("func_{}_{}", func_output_index, id);
                        tracing::debug!("[TOOL_CALL] Creating new tool call: func_id={}, output_index={}", func_id, func_output_index);
                        events.push(ResponseStreamEvent::OutputItemAdded {
                            output_index: func_output_index,
                            item_id: func_id.clone(),
                            item_type: "function_call".to_string(),
                            role: None,
                            call_id: Some(func_id.clone()),
                        });
                        state.is_function_call_item_added = true;

                        let initial_name = tc.function.name.clone().unwrap_or_default();
                        let initial_args = tc.function.arguments.clone().unwrap_or_default();
                        let tc_state = ToolCallState {
                            upstream_id: Some(tc_id),
                            id: func_id.clone(),
                            name: initial_name,
                            arguments: initial_args.clone(),
                            output_index: func_output_index,
                            last_args_len: initial_args.len(),
                        };

                        state.current_tool_calls.push(tc_state);

                        events.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                            output_index: func_output_index,
                            item_id: func_id,
                            delta: initial_args,
                        });
                        tracing::debug!("[TOOL_CALL] Emitted OutputItemAdded and FunctionCallArgumentsDelta, total events now: {}", events.len());
                    } else if let Some(idx) = existing_idx {
                        // Existing tool call - accumulate arguments and emit delta
                        let tc_state = &mut state.current_tool_calls[idx];
                        if let Some(args) = &tc.function.arguments {
                            let prev_len = tc_state.last_args_len;
                            // OpenAI-style: arguments are cumulative (growing string)
                            // MiniMax-style: arguments are incremental (only new bytes)
                            // Detect by checking if args starts with tc_state.arguments
                            let new_delta = if args.len() > prev_len && args.starts_with(&tc_state.arguments) {
                                // Cumulative: extract the new suffix
                                let delta = args[prev_len..].to_string();
                                tc_state.arguments = args.clone();
                                tc_state.last_args_len = args.len();
                                delta
                            } else {
                                // Incremental: args is just the new chunk
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
                        // Also update name if provided (some providers send name in subsequent chunks)
                        if let Some(name) = &tc.function.name {
                            if !name.is_empty() && tc_state.name.is_empty() {
                                tc_state.name = name.clone();
                            }
                        }
                    } else {
                        // No id and no matching index - this is truly malformed
                        tracing::debug!("[TOOL_CALL] Skipping tool call - no id and no matching index {}", tc.index);
                    }
                }
            }

            // Handle finish
            tracing::debug!("[FINISH_REASON] choice.finish_reason={:?}, current_tool_calls_len={}", choice.finish_reason, state.current_tool_calls.len());
            if let Some(reason) = &choice.finish_reason {
                tracing::debug!("[FINISH_REASON] reason={}", reason);
                if reason == "stop" || reason == "length" {
                    // When stop/length, finalize everything (text + any pending tool calls)
                    events.extend(finalize_output(state, &id));
                } else if reason == "tool_calls" {
                    // When tool_calls, finalize all tracked tool calls
                    for tc_state in state.current_tool_calls.drain(..) {
                        events.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                            output_index: tc_state.output_index,
                            item_id: tc_state.id.clone(),
                            call_id: tc_state.id.clone(),
                            name: tc_state.name.clone(),
                            arguments: tc_state.arguments.clone(),
                        });
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: tc_state.output_index,
                            item_id: tc_state.id.clone(),
                            item_type: "function_call".to_string(),
                            role: None,
                            call_id: Some(tc_state.id.clone()),
                            name: Some(tc_state.name.clone()),
                            arguments: Some(tc_state.arguments.clone()),
                        });
                        state.completed_tool_calls.push(tc_state);
                    }
                    // Finalize text output (if any)
                    if state.is_output_item_added {
                        events.push(ResponseStreamEvent::OutputTextDone {
                            item_id: state.output_id.clone(),
                            output_index: 0,
                            content_index: 0,
                        });
                        events.push(ResponseStreamEvent::ContentPartDone {
                            item_id: state.output_id.clone(),
                            output_index: 0,
                            content_index: 0,
                        });
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: 0,
                            item_id: state.output_id.clone(),
                            item_type: "message".to_string(),
                            role: Some("assistant".to_string()),
                            call_id: None,
                            name: None,
                            arguments: None,
                        });
                    }
                    if state.is_reasoning_added {
                        events.push(ResponseStreamEvent::OutputItemDone {
                            output_index: 0,
                            item_id: format!("reasoning_{}", id),
                            item_type: "reasoning".to_string(),
                            role: None,
                            call_id: None,
                            name: None,
                            arguments: None,
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
            call_id: tc_state.id.clone(),
            name: tc_state.name.clone(),
            arguments: tc_state.arguments.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: tc_state.output_index,
            item_id: tc_state.id.clone(),
            item_type: "function_call".to_string(),
            role: None,
            call_id: Some(tc_state.id.clone()),
            name: Some(tc_state.name.clone()),
            arguments: Some(tc_state.arguments.clone()),
        });
        state.completed_tool_calls.push(tc_state);
    }

    if state.is_output_item_added {
        events.push(ResponseStreamEvent::OutputTextDone {
            item_id: state.output_id.clone(),
            output_index: 0,
            content_index: 0,
        });
        events.push(ResponseStreamEvent::ContentPartDone {
            item_id: state.output_id.clone(),
            output_index: 0,
            content_index: 0,
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: 0,
            item_id: state.output_id.clone(),
            item_type: "message".to_string(),
            role: Some("assistant".to_string()),
            call_id: None,
            name: None,
            arguments: None,
        });
    }

    if state.is_reasoning_added {
        events.push(ResponseStreamEvent::OutputItemDone {
            output_index: 0,
            item_id: format!("reasoning_{}", id),
            item_type: "reasoning".to_string(),
            role: None,
            call_id: None,
            name: None,
            arguments: None,
        });
    }

    // Note: is_completed should NOT be set here - it should only be set
    // when the response.completed event is actually sent
    tracing::debug!("[FINALIZE] Produced {} events", events.len());
    events
}

/// Generate SSE string from Response stream event.
pub fn event_to_sse(event: &ResponseStreamEvent) -> String {
    match event {
        ResponseStreamEvent::Created { id, model, status, created_at } => {
            format!(
                "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{},\"model\":\"{}\",\"status\":\"{}\"}}}}\n\n",
                escape_sse(id),
                created_at,
                escape_sse(model),
                status
            )
        }
        ResponseStreamEvent::InProgress { id } => {
            format!(
                "event: response.in_progress\ndata: {{\"type\":\"response.in_progress\",\"response\":{{\"id\":\"{}\"}}}}\n\n",
                escape_sse(id)
            )
        }
        ResponseStreamEvent::OutputItemAdded { output_index, item_id, item_type, role, call_id } => {
            let role_str = match role {
                Some(r) => format!(",\"role\":\"{}\"", escape_sse(r)),
                None => String::new(),
            };
            let call_id_str = match call_id {
                Some(cid) => format!(",\"call_id\":\"{}\"", escape_sse(cid)),
                None => String::new(),
            };
            format!(
                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"{}\"{}{}}}}}\n\n",
                output_index,
                escape_sse(item_id),
                item_type,
                role_str,
                call_id_str
            )
        }
        ResponseStreamEvent::ContentPartAdded { item_id, output_index, content_index } => {
            format!(
                "event: response.content_part.added\ndata: {{\"type\":\"response.content_part.added\",\"output_index\":{},\"item_id\":\"{}\",\"content_index\":{}}}\n\n",
                output_index,
                escape_sse(item_id),
                content_index
            )
        }
        ResponseStreamEvent::OutputTextDelta { item_id, output_index, content_index, delta } => {
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{},\"delta\":\"{}\"}}\n\n",
                escape_sse(item_id),
                output_index,
                content_index,
                escape_json_string(delta)
            )
        }
        ResponseStreamEvent::OutputTextDone { item_id, output_index, content_index } => {
            format!(
                "event: response.output_text.done\ndata: {{\"type\":\"response.output_text.done\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{}}}\n\n",
                escape_sse(item_id),
                output_index,
                content_index
            )
        }
        ResponseStreamEvent::ContentPartDone { item_id, output_index, content_index } => {
            format!(
                "event: response.content_part.done\ndata: {{\"type\":\"response.content_part.done\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{}}}\n\n",
                escape_sse(item_id),
                output_index,
                content_index
            )
        }
        ResponseStreamEvent::OutputItemDone { output_index, item_id, item_type, role, call_id, name, arguments } => {
            let role_str = match role {
                Some(r) => format!(",\"role\":\"{}\"", escape_sse(r)),
                None => String::new(),
            };
            let call_id_str = match call_id {
                Some(cid) => format!(",\"call_id\":\"{}\"", escape_sse(cid)),
                None => String::new(),
            };
            let name_str = match name {
                Some(n) => format!(",\"name\":\"{}\"", escape_json_string(n)),
                None => String::new(),
            };
            let args_str = match arguments {
                Some(a) => format!(",\"arguments\":\"{}\"", escape_json_string(a)),
                None => String::new(),
            };
            let status_str = ",\"status\":\"completed\"";
            format!(
                "event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"{}\"{}{}{}{}{}}}}}\n\n",
                output_index,
                escape_sse(item_id),
                item_type,
                role_str,
                call_id_str,
                name_str,
                args_str,
                status_str
            )
        }
        ResponseStreamEvent::ReasoningAdded { output_index, item_id } => {
            format!(
                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":{},\"item\":{{\"id\":\"{}\",\"type\":\"reasoning\"}}}}\n\n",
                output_index,
                escape_sse(item_id)
            )
        }
        ResponseStreamEvent::ReasoningDelta { item_id, output_index, content_index, delta } => {
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{}\",\"output_index\":{},\"content_index\":{},\"delta\":\"{}\"}}\n\n",
                escape_sse(item_id),
                output_index,
                content_index,
                escape_json_string(delta)
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDelta { output_index, item_id, delta } => {
            format!(
                "event: response.function_call_arguments.delta\ndata: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":{},\"item_id\":\"{}\",\"delta\":\"{}\"}}\n\n",
                output_index,
                escape_sse(item_id),
                escape_json_string(delta)
            )
        }
        ResponseStreamEvent::FunctionCallArgumentsDone { output_index, item_id, call_id, name, arguments } => {
            format!(
                "event: response.function_call_arguments.done\ndata: {{\"type\":\"response.function_call_arguments.done\",\"output_index\":{},\"item_id\":\"{}\",\"call_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}\n\n",
                output_index,
                escape_sse(item_id),
                escape_sse(call_id),
                escape_json_string(name),
                escape_json_string(arguments)
            )
        }
        ResponseStreamEvent::Completed { response } => {
            format!(
                "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{}}}\n\n",
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
            usage: None,
        };

        let mut state = StreamState::new("chat_123".to_string(), "gpt-4o".to_string());
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

        let mut state = StreamState::new("chat_123".to_string(), "gpt-4o".to_string());
        // First chunk to establish state
        let _ = chat_chunk_to_response_events(&chunk, &mut state);

        assert!(!state.current_tool_calls.is_empty());
        let tc = state.current_tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
    }
}
