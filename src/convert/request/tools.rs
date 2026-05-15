//! Tool conversion utilities for Responses API → Chat API.

use crate::types::chat_api::{
    ChatTool, ChatToolChoice, ChatToolChoiceMode, FunctionChoice, FunctionDefinition,
    NamedToolChoice,
};
use crate::types::response_api::{
    Tool as ResponseTool, ToolChoice as ResponseToolChoice, ToolType as ResponseToolType,
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
            strict: t.strict,
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
            strict: t.strict,
        },
    }
}

/// Convert Responses API tool choice to Chat API tool choice.
pub fn convert_tool_choice(choice: ResponseToolChoice) -> ChatToolChoice {
    match choice {
        ResponseToolChoice::Auto => ChatToolChoice::Mode(ChatToolChoiceMode::Auto),
        ResponseToolChoice::None => ChatToolChoice::Mode(ChatToolChoiceMode::None),
        ResponseToolChoice::Required => ChatToolChoice::Mode(ChatToolChoiceMode::Required),
        ResponseToolChoice::Function(f) => ChatToolChoice::Named(NamedToolChoice {
            tool_type: "function".to_string(),
            function: FunctionChoice { name: f.name },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::response_api::FunctionToolChoice;

    #[test]
    fn test_named_tool_choice_serializes_to_openai_shape() {
        // OpenAI `ChatCompletionNamedToolChoice` requires
        // `{type:"function", function:{name:"…"}}`.
        let choice = convert_tool_choice(ResponseToolChoice::Function(FunctionToolChoice {
            name: "get_weather".to_string(),
        }));
        let json = serde_json::to_value(&choice).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "get_weather");
    }

    #[test]
    fn test_tool_choice_mode_serializes_as_string() {
        let json = serde_json::to_value(convert_tool_choice(ResponseToolChoice::None)).unwrap();
        assert_eq!(json, serde_json::json!("none"));
        let json = serde_json::to_value(convert_tool_choice(ResponseToolChoice::Required)).unwrap();
        assert_eq!(json, serde_json::json!("required"));
    }

    #[test]
    fn test_function_definition_carries_strict() {
        let tool = ResponseTool {
            tool_type: ResponseToolType::Function,
            name: Some("f".to_string()),
            description: None,
            parameters: None,
            strict: Some(true),
            extra: Default::default(),
        };
        let chat_tools = convert_tools(vec![tool]);
        assert_eq!(chat_tools[0].function.strict, Some(true));
    }
}
