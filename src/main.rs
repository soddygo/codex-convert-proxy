//! Codex Convert Proxy
//!
//! A library for converting between OpenAI Responses API (used by Codex)
//! and Chat Completions API (used by Chinese LLM providers).
//!
//! # Example
//!
//! ```rust
//! use codex_convert_proxy::{response_to_chat, chat_to_response};
//! use codex_convert_proxy::providers::GLMProvider;
//!
//! // Convert Responses API request to Chat API request
//! let chat_req = response_to_chat(response_req, &GLMProvider);
//! ```
//!
//! # Running as a Proxy Server
//!
//! For a full proxy server implementation using pingora, see the `examples/proxy_server.rs`.
//! The proxy integration requires adaptation to the specific pingora version API.

fn main() {
    println!("codex-convert-proxy v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("This is a library for converting between OpenAI Responses API and Chat API.");
    println!("See examples/proxy_server.rs for a full proxy server implementation.");
}
