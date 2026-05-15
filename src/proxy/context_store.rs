//! In-memory conversation context store for previous_response_id expansion.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::types::chat_api::ChatMessage;

/// Conversation snapshot used to reconstruct chat history.
#[derive(Debug, Clone)]
pub struct ConversationSnapshot {
    pub instructions: Option<String>,
    pub messages: Vec<ChatMessage>,
}

const MAX_CONVERSATION_ENTRIES: usize = 1024;

#[derive(Debug, Default)]
struct StoreInner {
    map: HashMap<String, StoreEntry>,
    lru_order: VecDeque<String>,
}

#[derive(Debug, Clone)]
struct StoreEntry {
    snapshot: ConversationSnapshot,
    inserted_at: Instant,
}

/// Thread-safe in-memory store keyed by response id.
///
/// Uses `Mutex` rather than `RwLock`: every `get` mutates the LRU order, so
/// there is no read-only path that could benefit from shared access.
pub struct ConversationStore {
    inner: Mutex<StoreInner>,
    ttl: Duration,
}

impl ConversationStore {
    pub const DEFAULT_TTL: Duration = Duration::from_secs(2 * 60 * 60);

    pub fn new() -> Self {
        Self::with_ttl(Self::DEFAULT_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(StoreInner::default()),
            ttl,
        }
    }

    pub fn get(&self, backend_name: &str, response_id: &str) -> Option<ConversationSnapshot> {
        let key = store_key(backend_name, response_id);
        let mut guard = self.inner.lock().ok()?;
        let entry = guard.map.get(&key).cloned()?;
        if entry.inserted_at.elapsed() >= self.ttl {
            guard.map.remove(&key);
            if let Some(pos) = guard.lru_order.iter().position(|k| k == &key) {
                guard.lru_order.remove(pos);
            }
            return None;
        }
        if let Some(pos) = guard.lru_order.iter().position(|k| k == &key) {
            guard.lru_order.remove(pos);
        }
        guard.lru_order.push_back(key);
        Some(entry.snapshot)
    }

    pub fn insert(&self, backend_name: &str, response_id: String, snapshot: ConversationSnapshot) {
        if let Ok(mut guard) = self.inner.lock() {
            let key = store_key(backend_name, &response_id);
            if let Some(pos) = guard.lru_order.iter().position(|k| k == &key) {
                guard.lru_order.remove(pos);
            }
            guard.lru_order.push_back(key.clone());
            guard.map.insert(
                key,
                StoreEntry {
                    snapshot,
                    inserted_at: Instant::now(),
                },
            );

            while guard.map.len() > MAX_CONVERSATION_ENTRIES {
                if let Some(oldest_key) = guard.lru_order.pop_front() {
                    guard.map.remove(&oldest_key);
                } else {
                    break;
                }
            }
        }
    }
}

impl Default for ConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

fn store_key(backend_name: &str, response_id: &str) -> String {
    format!("{backend_name}:{response_id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat_api::{ChatMessage, Content, MessageRole};

    fn snapshot(text: &str) -> ConversationSnapshot {
        ConversationSnapshot {
            instructions: Some("test".to_string()),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: Content::String(text.to_string()),
                name: None,
                annotations: None,
                tool_calls: None,
                function_call: None,
                tool_call_id: None,
                refusal: None,
            }],
        }
    }

    #[test]
    fn test_lru_eviction_keeps_recent_entries() {
        let store = ConversationStore::new();
        for i in 0..=MAX_CONVERSATION_ENTRIES {
            store.insert("glm", format!("resp_{i}"), snapshot("x"));
        }
        assert!(store.get("glm", "resp_0").is_none());
        assert!(store.get("glm", &format!("resp_{}", MAX_CONVERSATION_ENTRIES)).is_some());
    }

    #[test]
    fn test_get_refreshes_lru_order() {
        let store = ConversationStore::new();
        for i in 0..MAX_CONVERSATION_ENTRIES {
            store.insert("glm", format!("resp_{i}"), snapshot("x"));
        }
        // Touch oldest so it becomes recent.
        assert!(store.get("glm", "resp_0").is_some());
        // Insert one more, now resp_1 should be evicted first.
        store.insert("glm", "resp_new".to_string(), snapshot("y"));
        assert!(store.get("glm", "resp_0").is_some());
        assert!(store.get("glm", "resp_1").is_none());
    }

    #[test]
    fn test_backend_namespace_separates_same_response_id() {
        let store = ConversationStore::new();
        store.insert("glm", "resp_same".to_string(), snapshot("glm"));
        store.insert("kimi", "resp_same".to_string(), snapshot("kimi"));

        assert_eq!(
            store.get("glm", "resp_same").unwrap().messages[0].content.as_text(),
            "glm"
        );
        assert_eq!(
            store.get("kimi", "resp_same").unwrap().messages[0].content.as_text(),
            "kimi"
        );
    }

    #[test]
    fn test_ttl_expiration() {
        let store = ConversationStore::with_ttl(Duration::ZERO);
        store.insert("glm", "resp_old".to_string(), snapshot("x"));
        assert!(store.get("glm", "resp_old").is_none());
    }
}
