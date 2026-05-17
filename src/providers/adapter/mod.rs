//! Protocol adapter layer — converts provider-neutral ChatRequest into
//! protocol-specific JSON and parses responses back.
//!
//! Inspired by rust-genai's adapter pattern: each protocol adapter owns
//! the serialization, so provider-specific quirks are expressed
//! declaratively in `ChatApiQuirks` rather than patched post-hoc.

pub mod traits;
pub mod config;
pub mod openai_chat;
pub mod openai_responses;

pub use traits::{ProtocolAdapter, ProtocolType};
pub use config::{ChatApiQuirks, ProviderConfig, TokenLimitField, ToolChoiceSupport};
pub use openai_chat::OpenAiChatAdapter;
pub use openai_responses::OpenAiResponsesAdapter;
