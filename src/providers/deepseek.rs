//! DeepSeek provider implementation.

use crate::providers::trait_::Provider;
use crate::types::chat_api::{ChatRequest, ChatResponse, ChatStreamChunk};
use std::any::Any;

#[derive(Clone)]
/// DeepSeek provider.
///
/// DeepSeek is mostly compatible with standard Chat API.
/// Minimal transformation needed.
pub struct DeepSeekProvider;

impl Provider for DeepSeekProvider {
    fn name(&self) -> &'static str {
        "deepseek"
    }

    fn transform_request(&mut self, _request: &mut ChatRequest) {
        // DeepSeek has excellent Chat API compatibility
        // No modifications needed
    }

    fn transform_response(&mut self, _response: &mut ChatResponse) {
        // No modifications needed
    }

    fn transform_stream_chunk(&mut self, _chunk: &mut ChatStreamChunk) {
        // No modifications needed
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}
