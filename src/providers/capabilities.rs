//! Provider capability matrix and controlled extension fields.

use std::collections::HashMap;

use crate::error::ConversionError;
use crate::types::chat_api::{
    ChatRequest, ChatToolChoice, ChatToolChoiceMode, Content, MessageRole,
};

/// Chat token-limit field preferred by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenLimitField {
    /// Legacy / widely supported Chat Completions field.
    MaxTokens,
    /// Newer OpenAI-compatible field used by some providers.
    MaxCompletionTokens,
}

/// How strictly a provider supports `tool_choice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoiceSupport {
    /// Preserve all OpenAI Chat API tool_choice modes and named choices.
    Full,
    /// Omit `tool_choice` entirely.
    Unsupported,
    /// Only `auto` is known-safe.
    AutoOnly,
    /// Only `none` and `auto` are known-safe.
    AutoAndNone,
}

/// Static provider capability matrix used by the generic conversion layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub tool_choice: ToolChoiceSupport,
    pub token_limit_field: TokenLimitField,
    pub supports_developer_role: bool,
    pub flatten_request_content: bool,
    pub supports_tool_strict: bool,
    pub supports_stream_options: bool,
    pub supports_parallel_tool_calls: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            supports_tools: true,
            tool_choice: ToolChoiceSupport::Full,
            token_limit_field: TokenLimitField::MaxTokens,
            supports_developer_role: false,
            flatten_request_content: false,
            supports_tool_strict: true,
            supports_stream_options: true,
            supports_parallel_tool_calls: true,
        }
    }
}

impl ProviderCapabilities {
    pub fn sanitize_request(&self, request: &mut ChatRequest) {
        for message in &mut request.messages {
            if !self.supports_developer_role && message.role == MessageRole::Developer {
                message.role = MessageRole::System;
            }
            if self.flatten_request_content && matches!(message.content, Content::Array(_)) {
                let text = message.content.as_text();
                message.content = Content::String(text);
            }
        }

        if !self.supports_tools {
            request.tools = None;
            request.tool_choice = None;
        } else {
            if !self.supports_tool_strict
                && let Some(tools) = request.tools.as_mut()
            {
                for tool in tools {
                    tool.function.strict = None;
                }
            }
            request.tool_choice =
                sanitize_tool_choice(request.tool_choice.take(), self.tool_choice);
        }

        if !self.supports_stream_options {
            request.stream_options = None;
        }
        if !self.supports_parallel_tool_calls {
            request.parallel_tool_calls = None;
        }
    }
}

fn sanitize_tool_choice(
    choice: Option<ChatToolChoice>,
    support: ToolChoiceSupport,
) -> Option<ChatToolChoice> {
    match support {
        ToolChoiceSupport::Full => choice,
        ToolChoiceSupport::Unsupported => None,
        ToolChoiceSupport::AutoOnly => {
            choice.map(|_| ChatToolChoice::Mode(ChatToolChoiceMode::Auto))
        }
        ToolChoiceSupport::AutoAndNone => match choice {
            Some(ChatToolChoice::Mode(ChatToolChoiceMode::None)) => {
                Some(ChatToolChoice::Mode(ChatToolChoiceMode::None))
            }
            Some(_) => Some(ChatToolChoice::Mode(ChatToolChoiceMode::Auto)),
            None => None,
        },
    }
}

/// Provider-specific extension fields to flatten into a Chat request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderExtensions {
    fields: HashMap<String, serde_json::Value>,
}

impl ProviderExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<(), ConversionError> {
        let key = key.into();
        if is_standard_chat_request_field(&key) {
            return Err(ConversionError::ProviderError(format!(
                "provider extension field '{key}' conflicts with standard ChatRequest field"
            )));
        }
        self.fields.insert(key, value);
        Ok(())
    }

    pub fn into_fields(self) -> HashMap<String, serde_json::Value> {
        self.fields
    }
}

fn is_standard_chat_request_field(key: &str) -> bool {
    matches!(
        key,
        "model"
            | "messages"
            | "tools"
            | "tool_choice"
            | "stream"
            | "temperature"
            | "max_tokens"
            | "max_completion_tokens"
            | "top_p"
            | "user"
            | "stream_options"
            | "frequency_penalty"
            | "presence_penalty"
            | "logit_bias"
            | "logprobs"
            | "top_logprobs"
            | "n"
            | "stop"
            | "response_format"
            | "reasoning_effort"
            | "parallel_tool_calls"
            | "seed"
            | "service_tier"
            | "web_search_options"
            | "modalities"
            | "prediction"
            | "audio"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_rejects_standard_fields() {
        let mut extensions = ProviderExtensions::new();
        let err = extensions
            .insert("model", serde_json::json!("bad"))
            .expect_err("standard field must be rejected");
        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn extension_accepts_provider_fields() {
        let mut extensions = ProviderExtensions::new();
        extensions
            .insert("thinking", serde_json::json!({"type":"enabled"}))
            .unwrap();
        assert_eq!(
            extensions.into_fields().get("thinking"),
            Some(&serde_json::json!({"type":"enabled"}))
        );
    }
}
