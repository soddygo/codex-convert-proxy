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
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
    #[serde(rename = "output_text")]
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
    Namespace,
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
            Object(FunctionToolChoice),
        }

        let helper = ToolChoiceHelper::deserialize(deserializer)?;
        match helper {
            ToolChoiceHelper::String(s) => match s.as_str() {
                "auto" => Ok(ToolChoice::Auto),
                "none" => Ok(ToolChoice::None),
                "required" => Ok(ToolChoice::Required),
                _ => Ok(ToolChoice::Auto),
            },
            ToolChoiceHelper::Object(f) => Ok(ToolChoice::Function(f)),
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
                let helper = FunctionToolChoice { name: f.name.clone() };
                helper.serialize(serializer)
            }
        }
    }
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
