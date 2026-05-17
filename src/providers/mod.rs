//! Provider implementations for Chinese LLM services.

pub mod glm;
pub mod kimi;
pub mod deepseek;
pub mod minimax;
pub mod default;
pub mod trait_;
pub mod adapter;
pub mod openai_compat;

pub use trait_::*;
pub use glm::GLMProvider;
pub use kimi::KimiProvider;
pub use deepseek::DeepSeekProvider;
pub use minimax::MiniMaxProvider;
pub use default::DefaultProvider;
pub use adapter::{OpenAiChatAdapter, OpenAiResponsesAdapter, ProtocolAdapter, ProtocolType};
pub use adapter::{ChatApiQuirks, ProviderConfig};
