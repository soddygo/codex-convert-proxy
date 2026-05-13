//! Responses API types (Codex input format).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

/// Root request type for Responses API (Codex → Proxy)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseRequest {
    /// Model identifier
    pub model: String,

    /// Conversation history as input
    pub input: InputItemOrString,

    /// System prompt/instructions
    #[serde(default)]
    pub instructions: Option<String>,

    /// Built-in tools (web_search, code_interpreter, file_search)
    #[serde(default)]
    pub tools: Vec<Tool>,

    /// Tool choice policy
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Streaming indicator
    #[serde(default)]
    pub stream: bool,

    /// Additional parameters
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Official Responses field name.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// Compatibility alias for older payloads.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub user: Option<String>,

    #[serde(default, alias = "thinking")]
    pub reasoning: Option<ResponseReasoning>,

    #[serde(default)]
    pub text: Option<ResponseTextConfig>,

    #[serde(default)]
    pub truncation: Option<String>,

    #[serde(default)]
    pub store: Option<bool>,

    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub previous_response_id: Option<String>,

    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Whether to run the response in background.
    #[serde(default)]
    pub background: Option<bool>,
}

/// Input can be either a string or an array of InputItems.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItemOrString {
    String(String),
    Array(Vec<InputItem>),
}

/// An item in the input array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InputItem {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub item_type: InputItemType,
    pub role: Option<String>,
    pub content: Option<Content>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub call_id: Option<String>,
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Tools from tool_search_output items.
    /// These are dynamically discovered tools that can be merged into the request.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tools: Option<Vec<Tool>>,
}

/// Content can be a string or array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    String(String),
    Array(Vec<ContentPart>),
}

/// A content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(default)]
        file_url: Option<String>,
        #[serde(default)]
        file_id: Option<String>,
    },
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<ResponseAnnotation>,
    },
}

/// Type of input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputItemType {
    Message,
    FunctionCall,
    FunctionCallOutput,
    /// Computer use tool call (Codex computer use feature)
    #[serde(rename = "computer_call")]
    ComputerCall,
    /// Computer use output
    #[serde(rename = "computer_call_output")]
    ComputerCallOutput,
    /// File search tool call
    #[serde(rename = "file_search_call")]
    FileSearchCall,
    /// Web search tool call
    #[serde(rename = "web_search_call")]
    WebSearchCall,
    /// Code interpreter tool call
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall,
    /// Reasoning item
    Reasoning,
    /// Tool search call
    #[serde(rename = "tool_search_call")]
    ToolSearchCall,
    /// Tool search output
    #[serde(rename = "tool_search_output")]
    ToolSearchOutput,
    /// Image generation call
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall,
    /// Local shell call (OpenAI internal)
    #[serde(rename = "local_shell_call")]
    LocalShellCall,
    /// Local shell call output (OpenAI internal)
    #[serde(rename = "local_shell_call_output")]
    LocalShellCallOutput,
    /// Shell call (different from local_shell_call)
    #[serde(rename = "shell_call")]
    ShellCall,
    /// Shell call output
    #[serde(rename = "shell_call_output")]
    ShellCallOutput,
    /// MCP list tools
    #[serde(rename = "mcp_list_tools")]
    McpListTools,
    /// MCP approval request
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest,
    /// MCP approval response
    #[serde(rename = "mcp_approval_response")]
    McpApprovalResponse,
    /// MCP tool call
    #[serde(rename = "mcp_call")]
    McpCall,
    /// Custom tool call
    #[serde(rename = "custom_tool_call")]
    CustomToolCall,
    /// Custom tool call output
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput,
    /// Apply patch call (OpenAI internal)
    #[serde(rename = "apply_patch_call")]
    ApplyPatchCall,
    /// Apply patch call output
    #[serde(rename = "apply_patch_call_output")]
    ApplyPatchCallOutput,
    /// Compaction item (for response compaction API)
    Compaction,
    /// Catch-all for unknown input item types.
    /// Note: serde(other) captures unknown variants but discards the type name.
    /// The item will be logged as an unsupported type during conversion.
    #[serde(other)]
    Unknown,
}

/// A tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Type of tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    #[serde(rename = "web_search_preview", alias = "web_search")]
    WebSearchPreview,
    CodeInterpreter,
    FileSearch,
    Namespace,
    Custom,
    /// Catch-all for unknown tool types (forward compatibility).
    #[serde(other)]
    Other,
}

/// Tool choice policy.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(FunctionToolChoice),
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ToolChoiceHelper {
            String(String),
            Object(serde_json::Value),
        }

        let helper = ToolChoiceHelper::deserialize(deserializer)?;
        match helper {
            ToolChoiceHelper::String(s) => match s.as_str() {
                "auto" => Ok(ToolChoice::Auto),
                "none" => Ok(ToolChoice::None),
                "required" => Ok(ToolChoice::Required),
                _ => Ok(ToolChoice::Auto),
            },
            ToolChoiceHelper::Object(value) => {
                // Accept both {name:"..."} and OpenAI-style {type:"function",function:{name:"..."}}
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        value
                            .get("function")
                            .and_then(|v| v.get("name"))
                            .and_then(|v| v.as_str())
                    });
                if let Some(name) = name {
                    Ok(ToolChoice::Function(FunctionToolChoice {
                        name: name.to_string(),
                    }))
                } else {
                    Ok(ToolChoice::Auto)
                }
            }
        }
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ToolChoice::Auto => serializer.serialize_str("auto"),
            ToolChoice::None => serializer.serialize_str("none"),
            ToolChoice::Required => serializer.serialize_str("required"),
            ToolChoice::Function(f) => {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": f.name
                    }
                })
                .serialize(serializer)
            }
        }
    }
}

/// Function tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolChoice {
    pub name: String,
}

/// A part within a reasoning summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummaryPart {
    #[serde(rename = "summary_text")]
    SummaryText { text: String },
}

/// Response output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: OutputItemType,
    pub status: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    pub content: Option<Vec<ResponseContentPart>>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<ReasoningSummaryPart>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Type of output item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemType {
    Message,
    #[serde(rename = "function_call")]
    FunctionCall,
    Reasoning,
    #[serde(rename = "web_search_call")]
    WebSearchCall,
    #[serde(rename = "file_search_call")]
    FileSearchCall,
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall,
    #[serde(rename = "computer_call")]
    ComputerCall,
    #[serde(rename = "computer_call_output")]
    ComputerCallOutput,
    #[serde(rename = "tool_search_call")]
    ToolSearchCall,
    #[serde(rename = "tool_search_output")]
    ToolSearchOutput,
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall,
    #[serde(rename = "local_shell_call")]
    LocalShellCall,
    #[serde(rename = "local_shell_call_output")]
    LocalShellCallOutput,
    #[serde(rename = "shell_call")]
    ShellCall,
    #[serde(rename = "shell_call_output")]
    ShellCallOutput,
    #[serde(rename = "mcp_list_tools")]
    McpListTools,
    #[serde(rename = "mcp_approval_request")]
    McpApprovalRequest,
    #[serde(rename = "mcp_approval_response")]
    McpApprovalResponse,
    #[serde(rename = "mcp_call")]
    McpCall,
    #[serde(rename = "custom_tool_call")]
    CustomToolCall,
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput,
    #[serde(rename = "apply_patch_call")]
    ApplyPatchCall,
    #[serde(rename = "apply_patch_call_output")]
    ApplyPatchCallOutput,
    Compaction,
    /// Catch-all for unknown output item types.
    #[serde(other)]
    Other,
}

/// Content part in response output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<ResponseAnnotation>,
        #[serde(default)]
        logprobs: Vec<serde_json::Value>,
    },
    Refusal { refusal: String },
    InputSummary { text: String },
}

/// Output text annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseAnnotation {
    UrlCitation {
        start_index: usize,
        end_index: usize,
        url: String,
        title: String,
    },
    FileCitation {
        index: usize,
        file_id: String,
        filename: String,
    },
}

/// Response object (Responses API format).
///
/// Per the official OpenAPI spec (`Response.required`), the following fields must
/// be present in the serialized JSON even when their value is null/empty:
/// `error, incomplete_details, instructions, tools, parallel_tool_calls, metadata,
/// tool_choice, temperature, top_p`. We therefore avoid `skip_serializing_if` on
/// those fields (or use concrete non-Option types with sane defaults).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub status: String,
    pub model: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub error: Option<serde_json::Value>,
    pub incomplete_details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<serde_json::Value>>,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)]
    pub tools: Vec<Tool>,
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    pub usage: Option<Usage>,
}

impl ResponseObject {
    /// Build a "stub" ResponseObject for in-flight streaming events
    /// (`response.created`, `response.in_progress`, `response.failed`,
    /// `response.incomplete`).
    ///
    /// The stub carries the request-level context (instructions, tools,
    /// sampling params) so the emitted JSON satisfies `Response.required` even
    /// before any output items exist. This is the **single source of truth** —
    /// both Created/InProgress (with context) and Failed/Incomplete (without
    /// context) flow through this constructor so the schema cannot drift the
    /// way the prior hand-built JSON Maps did.
    pub fn stub(
        id: String,
        model: String,
        status: String,
        created_at: i64,
        ctx: Option<&crate::convert::context::ResponseRequestContext>,
    ) -> Self {
        let default_text = Some(ResponseTextConfig {
            format: Some(ResponseTextFormat {
                format_type: "text".to_string(),
                name: None,
                schema: None,
                strict: None,
            }),
        });
        Self {
            id,
            object: "response".to_string(),
            status,
            model,
            created_at,
            completed_at: None,
            error: None,
            incomplete_details: None,
            background: None,
            instructions: ctx.and_then(|c| c.instructions.clone()),
            max_output_tokens: ctx.and_then(|c| c.max_output_tokens),
            max_tool_calls: None,
            input: None,
            output: Vec::new(),
            parallel_tool_calls: ctx.and_then(|c| c.parallel_tool_calls).unwrap_or(true),
            previous_response_id: ctx.and_then(|c| c.previous_response_id.clone()),
            reasoning: ctx.and_then(|c| c.reasoning.clone()),
            store: ctx.and_then(|c| c.store),
            temperature: ctx.and_then(|c| c.temperature),
            text: ctx.and_then(|c| c.text.clone()).or(default_text),
            tool_choice: ctx
                .map(|c| c.tool_choice.clone())
                .unwrap_or_default(),
            tools: ctx.map(|c| c.tools.clone()).unwrap_or_default(),
            top_p: ctx.and_then(|c| c.top_p),
            truncation: ctx.and_then(|c| c.truncation.clone()),
            user: ctx.and_then(|c| c.user.clone()),
            metadata: ctx
                .and_then(|c| c.metadata.clone())
                .unwrap_or_default(),
            service_tier: None,
            top_logprobs: None,
            usage: None,
        }
    }
}

/// Usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    pub input_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    pub total_tokens: Option<i64>,
}

/// Input token details.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: i64,
}

/// Output token details.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i64,
}

/// Reasoning fields on Responses request/response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Text output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponseTextFormat>,
}

/// Text format configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseTextFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
