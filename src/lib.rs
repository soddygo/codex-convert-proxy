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
//! ```rust
//! use codex_convert_proxy::{response_to_chat, chat_to_response};
//! use codex_convert_proxy::providers::{GLMProvider, Provider};
//!
//! // Convert Responses API request to Chat API request
//! let chat_req = response_to_chat(response_req, &GLMProvider)?;
//!
//! // After getting Chat API response, convert back to Responses API format
//! let response_obj = chat_to_response(chat_resp)?;
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

pub mod error;
pub mod types;
pub mod convert;
pub mod providers;
pub mod streaming;
pub mod proxy;

// Re-export main types
pub use error::{ConversionError, ProxyError};
pub use types::*;

// Re-export convert functions
pub use convert::{response_to_chat, chat_to_response, chat_chunk_to_response_events, event_to_sse};

// Re-export providers
pub use providers::{Provider, GLMProvider, KimiProvider, DeepSeekProvider, MiniMaxProvider, create_provider};

// Re-export proxy
pub use proxy::CodexProxy;
