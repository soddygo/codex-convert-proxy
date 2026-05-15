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
//! - `lib`: Provider trait + implementations + conversion + types (for embedding
//!   the conversion logic in other Rust applications)
//! - `server` (default): Full Pingora-based proxy binary (implies `lib`, adds
//!   config parsing, the HTTP server, CLI, and logging)
//! - `telemetry`: OpenTelemetry tracing/metrics exporters
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
//! - [`DefaultProvider`] - Generic fallback for any OpenAI-compatible provider
//!   (used automatically by [`create_provider`] when the provider name is
//!   not in the built-in registry)

// Core modules (always available)
pub mod constants;
pub mod error;
pub mod stats;
pub mod types;
pub mod util;

// Conversion modules (require the `lib` feature for the Provider trait)
#[cfg(feature = "lib")]
pub mod convert;

// Re-export main types
pub use error::{ConversionError, ProxyError};
pub use types::*;

// Re-export convert functions (requires the `lib` feature)
#[cfg(feature = "lib")]
pub use convert::{
    chat_chunk_to_response_events, chat_to_response, chat_to_response_with_context, event_to_sse,
    response_to_chat,
};

// Re-export streaming types for library consumers
#[cfg(feature = "lib")]
pub use convert::ResponseRequestContext;
#[cfg(feature = "lib")]
pub use convert::streaming::{ResponseStreamEvent, StreamState};

// Re-export stats
pub use stats::{RequestRecord, RequestStats, StatsSummary, TokenUsage};

// Providers module (`lib` feature)
#[cfg(feature = "lib")]
pub mod providers;

#[cfg(feature = "lib")]
pub use providers::{
    DeepSeekProvider, DefaultProvider, GLMProvider, KimiProvider, MiniMaxProvider, Provider,
    create_provider,
};

// Config module (`server` feature - parsing backend definitions)
#[cfg(feature = "server")]
pub mod config;

#[cfg(feature = "server")]
pub use config::{BackendConfig, BackendInfo, BackendRouter, ProxyConfig};

// Server functionality (server feature - Pingora proxy)
#[cfg(feature = "server")]
pub mod proxy;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "server")]
pub use proxy::CodexProxy;

// CLI binary support (full feature implies binary)
#[cfg(feature = "server")]
pub mod cli;

#[cfg(feature = "acp")]
pub mod acp;

#[cfg(feature = "server")]
pub use cli::{Cli, Commands, ServerArgs};

// Logging (server/binary feature - requires tracing-subscriber and tracing-appender)
#[cfg(feature = "server")]
pub mod logger;

// Telemetry (`telemetry` feature)
#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(feature = "telemetry")]
pub use telemetry::TelemetryConfig;
