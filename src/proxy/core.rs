//! Codex proxy core structures.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::BackendRouter;
use crate::providers::Provider;

/// Codex proxy handler implementing ProxyHttp trait.
pub struct CodexProxy {
    /// Backend router for multi-backend routing.
    pub router: Arc<BackendRouter>,
    /// Providers for each backend.
    pub providers: HashMap<String, Box<dyn Provider + Send + Sync>>,
    /// Whether to log request/response bodies.
    pub log_body: bool,
    /// Directory for debug log files.
    pub log_dir: std::path::PathBuf,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        router: Arc<BackendRouter>,
        providers: HashMap<String, Box<dyn Provider + Send + Sync>>,
        log_body: bool,
        log_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            router,
            providers,
            log_body,
            log_dir,
        }
    }

    /// Get cloned provider for a backend.
    pub fn get_provider(&self, name: &str) -> Option<Box<dyn Provider + Send + Sync>> {
        self.providers.get(name).map(|p| p.clone_box())
    }
}
