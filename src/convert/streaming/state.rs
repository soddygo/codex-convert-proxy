//! Streaming state types: StreamState, ToolCallState, and ResponseRequestContext.

use std::collections::HashMap;

use serde::Serialize;
use crate::types::chat_api::ChatStreamChunk;
use crate::types::response_api::{
    ResponseReasoning, ResponseRequest, ResponseTextConfig, Tool, ToolChoice,
};

use super::super::util::{extract_queries_from_arguments, map_tool_name_to_output_type};

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
    pub upstream_id: Option<String>,
    pub id: String,
    pub call_id: String,
    pub item_type: String,
    pub name: String,
    pub arguments: String,
    pub output_index: u32,
    pub chat_api_index: u32,
    pub last_args_len: usize,
}

#[derive(Debug, Clone, Serialize)]
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
    pub fn build_response_object(&self) -> Box<crate::types::response_api::ResponseObject> {
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
            let item_type = map_tool_name_to_output_type(&tc.name, self.request_context.as_ref().map(|ctx| &ctx.tools));
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
                    cached_tokens: self.cached_tokens.unwrap_or(0),
                }),
                output_tokens: self.output_tokens,
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: self.reasoning_tokens.unwrap_or(0),
                }),
                total_tokens: self.total_tokens,
            })
        } else {
            None
        };

        Box::new(ResponseObject {
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
            input: None,
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
                .or({
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
        })
    }
}
