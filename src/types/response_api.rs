//! Responses API types (Codex input format).

use serde::{Deserialize, Serialize};

/// Root request type for Responses API (Codex → Proxy)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
    #[serde(rename = "tool_choice", default)]
    pub tool_choice: ToolChoice,

    /// Streaming indicator
    #[serde(default)]
    pub stream: bool,

    /// Additional parameters
    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub user: Option<String>,
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
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    InputText { text: String },
    InputImage { image_url: String },
    OutputText { text: String },
}

/// Type of input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputItemType {
    Message,
    FunctionCall,
    FunctionCallOutput,
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
}

/// Type of tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    WebSearch,
    CodeInterpreter,
    FileSearch,
}

/// Tool choice policy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(FunctionToolChoice),
}

/// Function tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolChoice {
    pub name: String,
}

/// Response output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: OutputItemType,
    pub status: Option<String>,
    pub content: Option<Vec<ResponseContentPart>>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub call_id: Option<String>,
}

/// Type of output item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemType {
    Message,
    FunctionCall,
    Reasoning,
}

/// Content part in response output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    OutputText { text: String },
    Refusal { text: String },
    InputSummary { text: String },
}

/// Response object (Responses API format).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub status: String,
    pub model: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub output: Vec<ResponseOutputItem>,
    pub usage: Option<Usage>,
}

/// Usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
}
