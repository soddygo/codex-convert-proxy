//! Streaming conversion: Chat SSE chunks → Responses API event sequence.

use std::collections::HashMap;

use crate::error::ConversionError;
use crate::types::chat_api::{ChatStreamChunk, Content};
use crate::types::response_api::{
    ResponseObject, ResponseReasoning, ResponseRequest, ResponseTextConfig, Tool, ToolChoice,
};

/// SSE event types for Responses API streaming.
#[derive(Debug, Clone)]
pub enum ResponseStreamEvent {
    /// Initial response created event.
    Created {
        id: String,
        model: String,
        status: String,
        created_at: i64,
        request_context: Option<ResponseRequestContext>,
    },
    /// Response is in progress.
    InProgress {
        id: String,
        model: String,
        status: String,
        created_at: i64,
        request_context: Option<ResponseRequestContext>,
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
        text: String,
    },
    /// Content part done.
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
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
        text: Option<String>,
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
    pub cached_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    /// Buffer for incomplete think/thought tags during streaming
    pub thinking_buffer: String,
    /// Whether we're currently inside a thinking tag
    pub is_thinking: bool,
    /// Next available output_index for sequential assignment
    pub next_output_index: u32,
    /// Stored output_index for text message items
    pub text_output_index: Option<u32>,
    /// Stored output_index for reasoning items
    pub reasoning_output_index: Option<u32>,
    /// Original Responses request fields for protocol-consistent events.
    pub request_context: Option<ResponseRequestContext>,
}

#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub upstream_id: Option<String>,  // Original ID from upstream provider (None if malformed)
    pub id: String,           // Internal ID we generate
    pub call_id: String,      // Stable call_id exposed in Responses API
    pub item_type: String,    // Response output item type: function_call/web_search_call/file_search_call
    pub name: String,
    pub arguments: String,
    pub output_index: u32,     // Each tool call stores its own output_index
    pub chat_api_index: u32,   // Original index from Chat API chunk (for matching)
    pub last_args_len: usize,  // Track for delta calculation
}

#[derive(Debug, Clone)]
pub struct ResponseRequestContext {
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: Option<bool>,
    pub previous_response_id: Option<String>,
    pub reasoning: Option<ResponseReasoning>,
    pub store: Option<bool>,
    pub temperature: Option<f32>,
    pub text: Option<ResponseTextConfig>,
    pub tool_choice: ToolChoice,
    pub tools: Vec<Tool>,
    pub top_p: Option<f32>,
    pub truncation: Option<String>,
    pub user: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl From<&ResponseRequest> for ResponseRequestContext {
    fn from(req: &ResponseRequest) -> Self {
        let mut metadata = req.metadata.clone().unwrap_or_default();
        let tool_map: serde_json::Map<String, serde_json::Value> = req
            .tools
            .iter()
            .filter_map(|t| {
                t.name.as_ref().map(|name| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "type": t.tool_type,
                            "strict": t.strict,
                            "extra": t.extra,
                        }),
                    )
                })
            })
            .collect();
        if !tool_map.is_empty() {
            metadata.insert(
                "x_proxy_tool_map".to_string(),
                serde_json::Value::Object(tool_map),
            );
        }

        Self {
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens.or(req.max_tokens),
            parallel_tool_calls: req.parallel_tool_calls,
            previous_response_id: req.previous_response_id.clone(),
            reasoning: req.reasoning.clone(),
            store: req.store,
            temperature: req.temperature,
            text: req.text.clone(),
            tool_choice: req.tool_choice.clone(),
            tools: req.tools.clone(),
            top_p: req.top_p,
            truncation: req.truncation.clone(),
            user: req.user.clone(),
            metadata: if metadata.is_empty() {
                None
            } else {
                Some(metadata)
            },
        }
    }
}

impl StreamState {
    /// Create a new stream state.
    pub fn new(
        response_id: String,
        model: String,
        request_context: Option<ResponseRequestContext>,
    ) -> Self {
        Self {
            response_id: response_id.clone(),
            output_id: format!("msg_{}", response_id),
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
            cached_tokens: None,
            reasoning_tokens: None,
            thinking_buffer: String::new(),
            is_thinking: false,
            next_output_index: 0,
            text_output_index: None,
            reasoning_output_index: None,
            request_context,
        }
    }

    /// Update usage from a ChatStreamChunk.
    pub fn update_usage(&mut self, chunk: &ChatStreamChunk) {
        if let Some(usage) = &chunk.usage {
            self.input_tokens = usage.prompt_tokens.map(|v| v as i64);
            self.output_tokens = usage.completion_tokens.map(|v| v as i64);
            self.total_tokens = usage.total_tokens.map(|v| v as i64);
            self.cached_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .map(|v| v as i64);
            self.reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens)
                .map(|v| v as i64);
        }
    }

    /// Build the final ResponseObject with all accumulated outputs.
    pub fn build_response_object(&self) -> crate::types::response_api::ResponseObject {
        use crate::types::response_api::{
            InputTokensDetails, OutputItemType, OutputTokensDetails, ResponseContentPart, ResponseObject,
            ResponseOutputItem, ResponseTextConfig, ResponseTextFormat, Usage,
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
                    annotations: vec![],
                }]),
                role: None,
                name: None,
                arguments: None,
                call_id: None,
                queries: None,
                results: None,
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
                    annotations: vec![],
                }]),
                role: Some("assistant".to_string()),
                name: None,
                arguments: None,
                call_id: None,
                queries: None,
                results: None,
            });
        }

        // Add function call outputs
        for tc in &self.completed_tool_calls {
            let item_type = map_tool_name_to_output_type(&tc.name, self.request_context.as_ref());
            let (queries, results) = if item_type != OutputItemType::FunctionCall {
                (extract_queries_from_arguments(&tc.arguments), Some(serde_json::Value::Null))
            } else {
                (None, None)
            };
            output.push(ResponseOutputItem {
                id: tc.id.clone(),
                item_type,
                status: Some("completed".to_string()),
                content: None,
                role: None,
                name: Some(tc.name.clone()),
                arguments: Some(tc.arguments.clone()),
                call_id: Some(tc.call_id.clone()),
                queries,
                results,
            });
        }

        let usage = if self.input_tokens.is_some() || self.output_tokens.is_some() || self.total_tokens.is_some() {
            Some(Usage {
                input_tokens: self.input_tokens,
                input_tokens_details: Some(InputTokensDetails {
                    cached_tokens: self.cached_tokens,
                }),
                output_tokens: self.output_tokens,
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: self.reasoning_tokens,
                }),
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
            error: None,
            incomplete_details: None,
            instructions: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.instructions.clone()),
            max_output_tokens: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.max_output_tokens),
            max_tool_calls: None,
            input: None,  // Input not available in streaming context
            output,
            parallel_tool_calls: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.parallel_tool_calls),
            previous_response_id: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.previous_response_id.clone()),
            reasoning: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.reasoning.clone())
                .or_else(|| {
                    Some(crate::types::response_api::ResponseReasoning {
                        effort: None,
                        summary: None,
                    })
                }),
            store: self.request_context.as_ref().and_then(|ctx| ctx.store),
            temperature: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.temperature),
            text: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.text.clone())
                .or_else(|| {
                    Some(ResponseTextConfig {
                        format: Some(ResponseTextFormat {
                            format_type: "text".to_string(),
                        }),
                    })
                }),
            tool_choice: self
                .request_context
                .as_ref()
                .map(|ctx| ctx.tool_choice.clone()),
            tools: self
                .request_context
                .as_ref()
                .map(|ctx| ctx.tools.clone()),
            top_p: self.request_context.as_ref().and_then(|ctx| ctx.top_p),
            truncation: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.truncation.clone()),
            user: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.user.clone()),
            metadata: self
                .request_context
                .as_ref()
                .and_then(|ctx| ctx.metadata.clone()),
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
    // Keep a stable response id across the whole stream lifecycle.
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
                    if !actual_text.is_empty() {
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
                            delta: actual_text.clone(),
                        });
                        state.full_text.push_str(&actual_text);
                    }
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
                        // No ID (incremental arguments chunk) -> match by chat_api_index
                        state.current_tool_calls.iter().position(|t| t.chat_api_index == tc.index)
                    };
                    tracing::debug!("[TOOL_CALL] existing_idx={:?}, tc.index={}", existing_idx, tc.index);

                    if existing_idx.is_none() {
                        if let Some(tc_id) = tc.id.clone() {
                            // New tool call (has id) - assign sequential output_index
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
                    // Finalize text output (if any)
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

    // Note: is_completed should NOT be set here - it should only be set
    // when the response.completed event is actually sent
    tracing::debug!("[FINALIZE] Produced {} events", events.len());
    events
}

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
    }
}

fn sse_event(event_name: &str, payload: serde_json::Value) -> String {
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    format!("event: {event_name}\ndata: {data}\n\n")
}

fn response_stub_json(
    id: &str,
    model: &str,
    status: &str,
    created_at: i64,
    request_context: Option<&ResponseRequestContext>,
) -> serde_json::Value {
    let (
        instructions,
        max_output_tokens,
        parallel_tool_calls,
        previous_response_id,
        reasoning,
        store,
        temperature,
        text,
        tool_choice,
        tools,
        top_p,
        truncation,
        user,
        metadata,
    ) = if let Some(ctx) = request_context {
        (
            serde_json::to_value(&ctx.instructions).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(ctx.max_output_tokens).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(ctx.parallel_tool_calls).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.previous_response_id).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.reasoning).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(ctx.store).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(ctx.temperature).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.text).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.tool_choice).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.tools).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(ctx.top_p).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.truncation).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.user).unwrap_or(serde_json::Value::Null),
            serde_json::to_value(&ctx.metadata).unwrap_or(serde_json::Value::Null),
        )
    } else {
        (
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::json!([]),
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
            serde_json::Value::Null,
        )
    };

    serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": null,
        "incomplete_details": null,
        "instructions": instructions,
        "max_output_tokens": max_output_tokens,
        "model": model,
        "output": [],
        "parallel_tool_calls": parallel_tool_calls,
        "previous_response_id": previous_response_id,
        "reasoning": reasoning,
        "store": store,
        "temperature": temperature,
        "text": if text.is_null() {
            serde_json::json!({"format":{"type":"text"}})
        } else {
            text
        },
        "tool_choice": tool_choice,
        "tools": tools,
        "top_p": top_p,
        "truncation": truncation,
        "usage": null,
        "user": user,
        "metadata": metadata,
    })
}

fn map_tool_name_to_output_type(
    tool_name: &str,
    request_context: Option<&ResponseRequestContext>,
) -> crate::types::response_api::OutputItemType {
    use crate::types::response_api::{OutputItemType, ToolType};
    if let Some(ctx) = request_context {
        for t in &ctx.tools {
            if t.name.as_deref() == Some(tool_name) {
                return match t.tool_type {
                    ToolType::WebSearchPreview => OutputItemType::WebSearchCall,
                    ToolType::FileSearch => OutputItemType::FileSearchCall,
                    _ => OutputItemType::FunctionCall,
                };
            }
        }
    }
    match tool_name {
        "web_search_preview" | "web_search" => OutputItemType::WebSearchCall,
        "file_search" => OutputItemType::FileSearchCall,
        _ => OutputItemType::FunctionCall,
    }
}

fn map_tool_name_to_stream_item_type(
    tool_name: &str,
    request_context: Option<&ResponseRequestContext>,
) -> String {
    use crate::types::response_api::OutputItemType;
    match map_tool_name_to_output_type(tool_name, request_context) {
        OutputItemType::WebSearchCall => "web_search_call".to_string(),
        OutputItemType::FileSearchCall => "file_search_call".to_string(),
        _ => "function_call".to_string(),
    }
}

fn extract_queries_from_arguments(arguments: &str) -> Option<Vec<String>> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) {
        if let Some(query) = value.get("query").and_then(|v| v.as_str()) {
            return Some(vec![query.to_string()]);
        }
        if let Some(queries) = value.get("queries").and_then(|v| v.as_array()) {
            let qs: Vec<String> = queries
                .iter()
                .filter_map(|q| q.as_str().map(|s| s.to_string()))
                .collect();
            if !qs.is_empty() {
                return Some(qs);
            }
        }
    }
    None
}

/// Maximum size for the thinking buffer (1MB) to prevent unbounded growth.
const MAX_THINKING_BUFFER_SIZE: usize = 1024 * 1024;

/// Parse thinking tags from streaming content.
///
/// Handles `<think>...</think>` and `<thought>...</thought>` tags that may be
/// split across multiple chunks. Returns (actual_text, reasoning_delta, new_is_thinking).
///
/// The `buffer` is used to accumulate incomplete tag content across chunks.
fn parse_streaming_thinking(
    text: &str,
    is_thinking: bool,
    buffer: &mut String,
) -> (String, Option<String>, bool) {
    let mut actual_text = String::new();
    let mut reasoning = String::new();
    let mut current_is_thinking = is_thinking;

    // Append to buffer for tag detection
    buffer.push_str(text);

    // Prevent unbounded buffer growth: if buffer exceeds limit, flush as reasoning
    if buffer.len() > MAX_THINKING_BUFFER_SIZE {
        let flushed = buffer.clone();
        buffer.clear();
        return (String::new(), Some(flushed), false);
    }

    let full_content = buffer.clone();
    buffer.clear();

    let mut pos = 0;
    let chars: Vec<char> = full_content.chars().collect();

    while pos < chars.len() {
        if current_is_thinking {
            // Look for closing tags: <think> or </thought>
            let think_close = find_pattern(&chars, pos, &['<', '/', 't', 'h', 'i', 'n', 'k', '>']);
            let thought_close = find_pattern(&chars, pos, &['<', '/', 't', 'h', 'o', 'u', 'g', 'h', 't', '>']);

            match (think_close, thought_close) {
                (Some(close_pos), Some(thought_close_pos)) => {
                    if close_pos <= thought_close_pos {
                        // Found <think> first
                        let content: String = chars[pos..close_pos].iter().collect();
                        reasoning.push_str(&content);
                        pos = close_pos + 8; // len("</think>")
                        current_is_thinking = false;
                    } else {
                        // Found </thought> first
                        let content: String = chars[pos..thought_close_pos].iter().collect();
                        reasoning.push_str(&content);
                        pos = thought_close_pos + 10; // len("</thought>")
                        current_is_thinking = false;
                    }
                }
                (Some(close_pos), None) => {
                    let content: String = chars[pos..close_pos].iter().collect();
                    reasoning.push_str(&content);
                    pos = close_pos + 8;
                    current_is_thinking = false;
                }
                (None, Some(thought_close_pos)) => {
                    let content: String = chars[pos..thought_close_pos].iter().collect();
                    reasoning.push_str(&content);
                    pos = thought_close_pos + 10;
                    current_is_thinking = false;
                }
                (None, None) => {
                    // No closing tag found, keep in buffer for next chunk
                    let remaining: String = chars[pos..].iter().collect();
                    buffer.push_str(&remaining);
                    break;
                }
            }
        } else {
            // Look for opening tags: <think> or <thought>
            let think_open = find_pattern(&chars, pos, &['<', 't', 'h', 'i', 'n', 'k', '>']);
            let thought_open = find_pattern(&chars, pos, &['<', 't', 'h', 'o', 'u', 'g', 'h', 't', '>']);

            match (think_open, thought_open) {
                (Some(open_pos), Some(thought_open_pos)) => {
                    if open_pos <= thought_open_pos {
                        // Found <think> first
                        let content: String = chars[pos..open_pos].iter().collect();
                        actual_text.push_str(&content);
                        pos = open_pos + 7; // len("<think>")
                        current_is_thinking = true;
                    } else {
                        // Found <thought> first
                        let content: String = chars[pos..thought_open_pos].iter().collect();
                        actual_text.push_str(&content);
                        pos = thought_open_pos + 9; // len("<thought>")
                        current_is_thinking = true;
                    }
                }
                (Some(open_pos), None) => {
                    let content: String = chars[pos..open_pos].iter().collect();
                    actual_text.push_str(&content);
                    pos = open_pos + 7;
                    current_is_thinking = true;
                }
                (None, Some(thought_open_pos)) => {
                    let content: String = chars[pos..thought_open_pos].iter().collect();
                    actual_text.push_str(&content);
                    pos = thought_open_pos + 9;
                    current_is_thinking = true;
                }
                (None, None) => {
                    // No opening tag found, rest is actual text
                    let remaining: String = chars[pos..].iter().collect();
                    actual_text.push_str(&remaining);
                    break;
                }
            }
        }
    }

    let reasoning_delta = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };

    (actual_text, reasoning_delta, current_is_thinking)
}

/// Find a pattern in char array starting from pos.
fn find_pattern(chars: &[char], start: usize, pattern: &[char]) -> Option<usize> {
    if start + pattern.len() > chars.len() {
        return None;
    }
    for i in start..=chars.len() - pattern.len() {
        if chars[i..i + pattern.len()] == *pattern {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::types::chat_api::{ChatDelta, ChatStreamChoice, Content, ToolCallDelta, FunctionCallDelta};
    use crate::types::response_api::{InputItemOrString, ResponseRequest, Tool, ToolChoice, ToolType};

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
        // First chunk to establish state
        let _ = chat_chunk_to_response_events(&chunk, &mut state);

        assert!(!state.current_tool_calls.is_empty());
        let tc = state.current_tool_calls.first().unwrap();
        assert_eq!(tc.name, "get_weather");
    }

    #[test]
    fn test_parse_streaming_thinking_basic() {
        let mut buffer = String::new();

        // No thinking tags
        let (actual, reasoning, is_thinking) = parse_streaming_thinking("Hello world", false, &mut buffer);
        assert_eq!(actual, "Hello world");
        assert!(reasoning.is_none());
        assert!(!is_thinking);
    }

    #[test]
    fn test_parse_streaming_thinking_with_think_tag() {
        let mut buffer = String::new();

        // Complete <think> tag
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
        let mut buffer = String::new();

        // First chunk: partial thinking tag
        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            "<think>\npartial",
            false,
            &mut buffer,
        );
        assert_eq!(actual, "");
        assert!(reasoning.is_none());
        assert!(is_thinking);

        // Second chunk: rest of thinking and closing tag
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
        let mut buffer = String::new();

        // <thought> tag
        let (actual, reasoning, is_thinking) = parse_streaming_thinking(
            "<thought>reasoning</thought>actual",
            false,
            &mut buffer,
        );
        assert_eq!(actual, "actual");
        assert_eq!(reasoning, Some("reasoning".to_string()));
        assert!(!is_thinking);
    }

    #[test]
    fn test_content_part_added_includes_part_payload() {
        let event = ResponseStreamEvent::ContentPartAdded {
            item_id: "msg_test".to_string(),
            output_index: 0,
            content_index: 0,
        };
        let sse = event_to_sse(&event);
        assert!(sse.contains("event: response.content_part.added"));
        assert!(sse.contains("\"part\":{"));
        assert!(sse.contains("\"type\":\"output_text\""));
        assert!(sse.contains("\"annotations\":[]"));
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
        assert!(sse.contains("\"text\":\"hello\""));
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
}
