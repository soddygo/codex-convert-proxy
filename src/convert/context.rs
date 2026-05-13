//! Request-level context shared between conversion paths.
//!
//! `ResponseRequestContext` captures the original Responses API request fields
//! needed to reconstruct a spec-compliant response (instructions, tools, sampling
//! params, etc.). It is consumed by both streaming and non-streaming flows, so
//! it lives at the `convert` root rather than under `streaming::state`.

use std::collections::HashMap;

use serde::Serialize;

use crate::types::response_api::{
    ResponseReasoning, ResponseRequest, ResponseTextConfig, Tool, ToolChoice,
};

/// Fields from the originating Responses API request that the proxy must
/// echo back on the synthesized Response payload (and intermediate stream
/// stubs). Kept separate from `StreamState` so non-streaming flows can use
/// it without dragging streaming-specific bookkeeping.
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
