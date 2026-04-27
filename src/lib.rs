//! Codex Convert Proxy Library
//!
//! A library for converting between OpenAI Responses API (used by Codex)
//! and Chat Completions API (used by Chinese LLM providers).
//!
//! # Overview
//!
//! Codex 0.118+ only supports the Responses API, but Chinese LLM providers
//! (GLM, Kimi, DeepSeek, MiniMax) only support the Chat Completions API.
//! This library provides bidirectional conversion between these formats.
//!
//! # Usage
//!
//! ```ignore
//! use codex_convert_proxy::{response_to_chat, chat_to_response};
//! use codex_convert_proxy::providers::{GLMProvider, Provider};
//! use codex_convert_proxy::types::response_api::{ResponseRequest, InputItemOrString};
//!
//! // Create a Responses API request
//! let response_req = ResponseRequest {
//!     model: "glm-4".to_string(),
//!     input: InputItemOrString::String("Hello".to_string()),
//!     instructions: Some("You are a helpful assistant.".to_string()),
//!     tools: vec![],
//!     tool_choice: Default::default(),
//!     stream: false,
//!     temperature: None,
//!     max_tokens: None,
//!     top_p: None,
//!     user: None,
//! };
//!
//! // Convert Responses API request to Chat API request
//! let chat_req = response_to_chat(response_req, &GLMProvider).unwrap();
//! ```
//!
//! # Providers
//!
//! Each Chinese LLM provider may have slightly different API requirements.
//! Use the appropriate provider when converting:
//!
//! - [`GLMProvider`] - For Zhipu AI GLM models
//! - [`KimiProvider`] - For Moonshot AI Kimi models
//! - [`DeepSeekProvider`] - For DeepSeek models
//! - [`MiniMaxProvider`] - For MiniMax models

pub mod cli;
pub mod config;
pub mod error;
pub mod logger;
pub mod proxy;
pub mod providers;
pub mod stats;
pub mod streaming;
pub mod telemetry;
pub mod types;
pub mod convert;

// Re-export main types
pub use error::{ConversionError, ProxyError};
pub use types::*;

// Re-export convert functions
pub use convert::{response_to_chat, chat_to_response, chat_chunk_to_response_events, event_to_sse};

// Re-export providers
pub use providers::{Provider, GLMProvider, KimiProvider, DeepSeekProvider, MiniMaxProvider, create_provider};

// Re-export proxy
pub use proxy::CodexProxy;

// Re-export config
pub use config::{BackendConfig, BackendInfo, BackendRouter, ProxyConfig};

// Re-export stats
pub use stats::{RequestRecord, RequestStats, StatsSummary, TokenUsage};

// Re-export CLI
pub use cli::{Cli, Commands, StartArgs};

// Re-export telemetry
pub use telemetry::TelemetryConfig;
