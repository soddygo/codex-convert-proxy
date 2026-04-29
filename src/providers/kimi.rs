//! Kimi (Moonshot AI) provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};
use std::any::Any;

#[derive(Clone)]
/// Kimi (Moonshot AI) provider.
///
/// Kimi API accepts both "kimi-" and "moonshot-v1-" model name prefixes natively.
/// No model name normalization is needed.
pub struct KimiProvider;

impl Provider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn transform_request(&mut self, _request: &mut ChatRequest) {
        // Standard Chat API compatible, no transformation needed
    }

    fn transform_response(&mut self, _response: &mut ChatResponse) {
        // Standard handling
    }

    fn transform_stream_chunk(&mut self, _chunk: &mut ChatStreamChunk) {
        // Standard handling
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}
