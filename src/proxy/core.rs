//! Codex proxy core structures.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::config::BackendRouter;
use crate::providers::Provider;
use crate::proxy::context_store::{ConversationSnapshot, ConversationStore};

/// Codex proxy handler implementing ProxyHttp trait.
pub struct CodexProxy {
    /// Backend router for multi-backend routing.
    pub router: Arc<BackendRouter>,
    /// Providers for each backend (shared, stateless).
    pub providers: HashMap<String, Arc<dyn Provider>>,
    /// Whether to log request/response bodies.
    pub log_body: bool,
    /// Whether to skip Responses→Chat API body conversion (pass-through).
    pub no_convert: bool,
    /// Directory for debug log files.
    pub log_dir: std::path::PathBuf,
    /// In-memory conversation store for previous_response_id expansion.
    pub conversation_store: Arc<ConversationStore>,
}

impl CodexProxy {
    /// Create a new CodexProxy instance.
    pub fn new(
        router: Arc<BackendRouter>,
        providers: HashMap<String, Arc<dyn Provider>>,
        log_body: bool,
        log_dir: std::path::PathBuf,
    ) -> Self {
        Self::with_conversation_ttl(
            router,
            providers,
            log_body,
            false,
            log_dir,
            ConversationStore::DEFAULT_TTL,
        )
    }

    pub fn with_conversation_ttl(
        router: Arc<BackendRouter>,
        providers: HashMap<String, Arc<dyn Provider>>,
        log_body: bool,
        no_convert: bool,
        log_dir: std::path::PathBuf,
        conversation_ttl: Duration,
    ) -> Self {
        Self {
            router,
            providers,
            log_body,
            no_convert,
            log_dir,
            conversation_store: Arc::new(ConversationStore::with_ttl(conversation_ttl)),
        }
    }

    /// Get a shared handle to the provider for a backend.
    pub fn get_provider(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).map(Arc::clone)
    }

    /// Lookup conversation snapshot by response id.
    pub fn get_conversation(
        &self,
        backend_name: &str,
        response_id: &str,
    ) -> Option<ConversationSnapshot> {
        self.conversation_store.get(backend_name, response_id)
    }

    /// Persist conversation snapshot for follow-up turns.
    pub fn store_conversation(
        &self,
        backend_name: &str,
        response_id: String,
        snapshot: ConversationSnapshot,
    ) {
        self.conversation_store
            .insert(backend_name, response_id, snapshot);
    }
}
