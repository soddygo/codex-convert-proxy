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
//! # Features
//!
//! - `providers-lib`: Provider trait + implementations + conversion functions (minimal embedding)
//! - `config-lib`: BackendRouter and config types (optional, for standalone proxy)
//! - `lib`: Alias for `providers-lib`
//! - `full` (default): Full library including server functionality and CLI
//! - `server`: Pingora-based proxy server (implies `lib` + `config-lib`)
//! - `binary`: CLI binary support (alias for `full`)
//!
//! # Usage (as library)
//!
//! ```ignore
//! use codex_convert_proxy::{
//!     response_to_chat, chat_to_response, chat_to_response_with_context,
//!     chat_chunk_to_response_events, event_to_sse,
//!     Provider, GLMProvider, create_provider,
//!     StreamState, ResponseRequestContext, ResponseStreamEvent,
//!     types::response_api::{ResponseRequest, InputItemOrString},
//!     util::parse_sse,
//! };
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

// Core modules (always available)
pub mod constants;
pub mod error;
pub mod stats;
pub mod types;
pub mod util;

// Conversion modules (require providers-lib for Provider trait)
#[cfg(feature = "providers-lib")]
pub mod convert;

// Re-export main types
pub use error::{ConversionError, ProxyError};
pub use types::*;

// Re-export convert functions (requires providers-lib)
#[cfg(feature = "providers-lib")]
pub use convert::{
    chat_chunk_to_response_events, chat_to_response, chat_to_response_with_context,
    event_to_sse, response_to_chat,
};

// Re-export streaming types for library consumers
#[cfg(feature = "providers-lib")]
pub use convert::streaming::{ResponseRequestContext, ResponseStreamEvent, StreamState};

// Re-export stats
pub use stats::{RequestRecord, RequestStats, StatsSummary, TokenUsage};

// Providers module (providers-lib feature)
#[cfg(feature = "providers-lib")]
pub mod providers;

#[cfg(feature = "providers-lib")]
pub use providers::{create_provider, DeepSeekProvider, GLMProvider, KimiProvider, MiniMaxProvider, Provider};

// Config module (config-lib feature - optional for standalone proxy)
#[cfg(feature = "config-lib")]
pub mod config;

#[cfg(feature = "config-lib")]
pub use config::{BackendConfig, BackendInfo, BackendRouter, ProxyConfig};

// Server functionality (server feature - Pingora proxy)
#[cfg(feature = "server")]
pub mod proxy;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "server")]
pub use proxy::CodexProxy;

// CLI binary support (full feature implies binary)
#[cfg(feature = "full")]
pub mod cli;

#[cfg(feature = "full")]
pub use cli::{Cli, Commands, StartArgs};

// Logging (server/binary feature - requires tracing-subscriber and tracing-appender)
#[cfg(feature = "server")]
pub mod logger;

// Telemetry (telemetry-lib feature)
#[cfg(feature = "telemetry-lib")]
pub mod telemetry;

#[cfg(feature = "telemetry-lib")]
pub use telemetry::TelemetryConfig;
