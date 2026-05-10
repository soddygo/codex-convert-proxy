//! Codex proxy core structures.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::BackendRouter;
use crate::providers::Provider;
use crate::proxy::context_store::{ConversationSnapshot, ConversationStore};

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
    /// In-memory conversation store for previous_response_id expansion.
    pub conversation_store: Arc<ConversationStore>,
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
            conversation_store: Arc::new(ConversationStore::new()),
        }
    }

    /// Get cloned provider for a backend.
    pub fn get_provider(&self, name: &str) -> Option<Box<dyn Provider + Send + Sync>> {
        self.providers.get(name).map(|p| p.clone_box())
    }

    /// Lookup conversation snapshot by response id.
    pub fn get_conversation(&self, response_id: &str) -> Option<ConversationSnapshot> {
        self.conversation_store.get(response_id)
    }

    /// Persist conversation snapshot for follow-up turns.
    pub fn store_conversation(&self, response_id: String, snapshot: ConversationSnapshot) {
        self.conversation_store.insert(response_id, snapshot);
    }
}
