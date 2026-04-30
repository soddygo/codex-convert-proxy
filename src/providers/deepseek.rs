//! DeepSeek provider implementation.

use crate::providers::trait_::Provider;
use std::any::Any;

#[derive(Clone)]
/// DeepSeek provider.
///
/// DeepSeek is mostly compatible with standard Chat API.
/// Minimal transformation needed - all trait methods use default implementations.
pub struct DeepSeekProvider;

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self
    }
}

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for DeepSeekProvider {
    fn name(&self) -> &'static str {
        "deepseek"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}
