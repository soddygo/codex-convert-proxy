//! Kimi (Moonshot AI) provider implementation.

use crate::providers::trait_::Provider;
use std::any::Any;

#[derive(Clone)]
/// Kimi (Moonshot AI) provider.
///
/// Kimi API accepts both "kimi-" and "moonshot-v1-" model name prefixes natively.
/// No model name normalization needed - all trait methods use default implementations.
pub struct KimiProvider;

impl Default for KimiProvider {
    fn default() -> Self {
        Self
    }
}

impl KimiProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Provider for KimiProvider {
    fn name(&self) -> &'static str {
        "kimi"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Provider + Send + Sync> {
        Box::new(self.clone())
    }
}
