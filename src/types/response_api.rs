//! Responses API types (Codex input format).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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

    #[serde(default)]
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
}

/// Type of output item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemType {
    Message,
    FunctionCall,
    Reasoning,
    WebSearchCall,
    FileSearchCall,
}

/// Content part in response output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentPart {
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<ResponseAnnotation>,
    },
    Refusal { text: String },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub status: String,
    pub model: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<serde_json::Value>>,
    pub output: Vec<ResponseOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub usage: Option<Usage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<i64>,
}

/// Output token details.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OutputTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
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
}
