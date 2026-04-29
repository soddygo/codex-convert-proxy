//! Tool conversion utilities for Responses API → Chat API.

use crate::types::chat_api::{ChatTool, ChatToolChoice, ChatToolChoiceMode, FunctionChoice, FunctionDefinition};
use crate::types::response_api::{
    Tool as ResponseTool, ToolChoice as ResponseToolChoice,
    ToolType as ResponseToolType,
};

/// Convert Responses API tools to Chat API tools.
pub fn convert_tools(tools: Vec<ResponseTool>) -> Vec<ChatTool> {
    tools
        .into_iter()
        .filter_map(|t| match t.tool_type {
            ResponseToolType::Function | ResponseToolType::Custom | ResponseToolType::Other => {
                Some(passthrough_tool(t))
            }
            ResponseToolType::WebSearchPreview => Some(builtin_tool(
                t,
                "web_search_preview",
                "Web search tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }),
            )),
            ResponseToolType::CodeInterpreter => Some(builtin_tool(
                t,
                "code_interpreter",
                "Code interpreter tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        }
                    },
                    "required": ["code"]
                }),
            )),
            ResponseToolType::FileSearch => Some(builtin_tool(
                t,
                "file_search",
                "File search tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }),
            )),
            ResponseToolType::Namespace => None,
        })
        .collect()
}

/// Pass through a tool as-is as a function tool.
fn passthrough_tool(t: ResponseTool) -> ChatTool {
    ChatTool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: t.name.unwrap_or_default(),
            description: t.description,
            parameters: t.parameters,
        },
    }
}

/// Convert a built-in tool to a function tool with default parameters.
fn builtin_tool(
    t: ResponseTool,
    default_name: &str,
    default_desc: &str,
    param_schema: serde_json::Value,
) -> ChatTool {
    ChatTool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: t.name.unwrap_or_else(|| default_name.to_string()),
            description: t.description.or_else(|| Some(default_desc.to_string())),
            parameters: Some(param_schema),
        },
    }
}

/// Convert Responses API tool choice to Chat API tool choice.
pub fn convert_tool_choice(choice: ResponseToolChoice) -> ChatToolChoice {
    match choice {
        ResponseToolChoice::Auto => ChatToolChoice::Mode(ChatToolChoiceMode::Auto),
        ResponseToolChoice::None => ChatToolChoice::Mode(ChatToolChoiceMode::None),
        ResponseToolChoice::Required => ChatToolChoice::Mode(ChatToolChoiceMode::Required),
        ResponseToolChoice::Function(f) => ChatToolChoice::Function(FunctionChoice { name: f.name }),
    }
}

/// Check if the tool choice is "none" (no tools).
pub fn is_tool_choice_none(choice: &ChatToolChoice) -> bool {
    match choice {
        ChatToolChoice::Mode(mode) => *mode == ChatToolChoiceMode::None,
        ChatToolChoice::Function(_) => false,
    }
}
