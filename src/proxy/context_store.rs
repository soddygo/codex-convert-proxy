//! In-memory conversation context store for previous_response_id expansion.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

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
    map: HashMap<String, ConversationSnapshot>,
    lru_order: VecDeque<String>,
}

/// Thread-safe in-memory store keyed by response id.
///
/// Uses `Mutex` rather than `RwLock`: every `get` mutates the LRU order, so
/// there is no read-only path that could benefit from shared access.
#[derive(Debug, Default)]
pub struct ConversationStore {
    inner: Mutex<StoreInner>,
}

impl ConversationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, response_id: &str) -> Option<ConversationSnapshot> {
        let mut guard = self.inner.lock().ok()?;
        let snapshot = guard.map.get(response_id).cloned()?;
        if let Some(pos) = guard.lru_order.iter().position(|k| k == response_id) {
            guard.lru_order.remove(pos);
        }
        guard.lru_order.push_back(response_id.to_string());
        Some(snapshot)
    }

    pub fn insert(&self, response_id: String, snapshot: ConversationSnapshot) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(pos) = guard.lru_order.iter().position(|k| k == &response_id) {
                guard.lru_order.remove(pos);
            }
            guard.lru_order.push_back(response_id.clone());
            guard.map.insert(response_id, snapshot);

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
            store.insert(format!("resp_{i}"), snapshot("x"));
        }
        assert!(store.get("resp_0").is_none());
        assert!(store.get(&format!("resp_{}", MAX_CONVERSATION_ENTRIES)).is_some());
    }

    #[test]
    fn test_get_refreshes_lru_order() {
        let store = ConversationStore::new();
        for i in 0..MAX_CONVERSATION_ENTRIES {
            store.insert(format!("resp_{i}"), snapshot("x"));
        }
        // Touch oldest so it becomes recent.
        assert!(store.get("resp_0").is_some());
        // Insert one more, now resp_1 should be evicted first.
        store.insert("resp_new".to_string(), snapshot("y"));
        assert!(store.get("resp_0").is_some());
        assert!(store.get("resp_1").is_none());
    }
}
