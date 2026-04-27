//! Type definitions for both API formats.

pub mod chat_api;
pub mod response_api;

// Re-export Chat API types
pub use chat_api::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatTool, ChatToolChoice,
    Content, ContentBlock, FunctionCall, FunctionChoice, FunctionDefinition, MessageRole,
    StreamOptions, ToolCall, ChatDelta, ChatStreamChoice, ChatToolChoiceMode, ChatUsage,
    FunctionCallDelta, ToolCallDelta,
};

// Re-export Response API types
pub use response_api::{
    Content as ResponseContent, ContentPart, InputItem, InputItemOrString, InputItemType,
    OutputItemType, ResponseOutputItem, ResponseRequest, Tool, ToolChoice, ToolType, Usage,
    ResponseContentPart, ResponseObject,
};
