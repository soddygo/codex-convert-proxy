//! Statistics and metrics module.
//!
//! This module provides request tracking and token usage statistics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tracing::info;

/// Single request record for statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub timestamp: DateTime<Utc>,
    pub method: String,
    pub uri: String,
    pub backend: String,
    pub model: Option<String>,
    pub status: u16,
    pub duration_ms: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
    pub error: Option<String>,
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
}

impl TokenUsage {
    /// Parse token usage from SSE response.
    pub fn parse_from_sse(chunk: &str) -> Option<Self> {
        for line in chunk.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    // Check Anthropic message_start format
                    if let Some(usage) = json.get("message").and_then(|m| m.get("usage")) {
                        return Some(Self {
                            input_tokens: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                            output_tokens: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                            cache_read_tokens: usage.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                            cache_creation_tokens: usage.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                        });
                    }

                    // Check usage object directly
                    if let Some(usage) = json.get("usage") {
                        return Some(Self {
                            input_tokens: usage.get("input_tokens").or(usage.get("prompt_tokens")).and_then(|v| v.as_u64()).unwrap_or(0),
                            output_tokens: usage.get("output_tokens").or(usage.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0),
                            cache_read_tokens: usage.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                            cache_creation_tokens: usage.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                        });
                    }
                }
            }
        }
        None
    }
}

/// Global request statistics.
#[derive(Debug, Default)]
pub struct RequestStats {
    total_requests: AtomicU64,
    success_count: AtomicU64,
    client_error_count: AtomicU64,
    server_error_count: AtomicU64,
    total_duration_ms: AtomicU64,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
    total_cache_read_tokens: AtomicU64,
    total_cache_creation_tokens: AtomicU64,
    model_counts: Mutex<HashMap<String, AtomicU64>>,
    backend_counts: Mutex<HashMap<String, AtomicU64>>,
    recent_requests: Mutex<VecDeque<RequestRecord>>,
    max_recent: usize,
}

impl RequestStats {
    /// Create new statistics instance.
    pub fn new(max_recent: usize) -> Self {
        Self {
            max_recent,
            ..Default::default()
        }
    }

    /// Record a request.
    pub fn record_request(&self, record: RequestRecord) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_duration_ms.fetch_add(record.duration_ms, Ordering::Relaxed);

        // Count by status
        if record.status >= 200 && record.status < 300 {
            self.success_count.fetch_add(1, Ordering::Relaxed);
        } else if record.status >= 400 && record.status < 500 {
            self.client_error_count.fetch_add(1, Ordering::Relaxed);
        } else if record.status >= 500 {
            self.server_error_count.fetch_add(1, Ordering::Relaxed);
        }

        // Token counts
        if let Some(t) = record.input_tokens {
            self.total_input_tokens.fetch_add(t, Ordering::Relaxed);
        }
        if let Some(t) = record.output_tokens {
            self.total_output_tokens.fetch_add(t, Ordering::Relaxed);
        }
        if let Some(t) = record.cache_read_tokens {
            self.total_cache_read_tokens.fetch_add(t, Ordering::Relaxed);
        }
        if let Some(t) = record.cache_creation_tokens {
            self.total_cache_creation_tokens.fetch_add(t, Ordering::Relaxed);
        }

        // Per-model counts
        if let Some(ref model) = record.model {
            if let Ok(mut counts) = self.model_counts.lock() {
                counts
                    .entry(model.clone())
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        // Per-backend counts
        if !record.backend.is_empty() {
            if let Ok(mut counts) = self.backend_counts.lock() {
                counts
                    .entry(record.backend.clone())
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        // Recent requests
        if let Ok(mut recent) = self.recent_requests.lock() {
            if recent.len() >= self.max_recent {
                recent.pop_front();
            }
            recent.push_back(record);
        }
    }

    /// Get statistics summary.
    pub fn summary(&self) -> StatsSummary {
        let total = self.total_requests.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        let client_err = self.client_error_count.load(Ordering::Relaxed);
        let server_err = self.server_error_count.load(Ordering::Relaxed);
        let total_duration = self.total_duration_ms.load(Ordering::Relaxed);

        StatsSummary {
            total_requests: total,
            success_count: success,
            client_error_count: client_err,
            server_error_count: server_err,
            success_rate: if total > 0 { success as f64 / total as f64 * 100.0 } else { 0.0 },
            avg_duration_ms: if total > 0 { total_duration as f64 / total as f64 } else { 0.0 },
            total_input_tokens: self.total_input_tokens.load(Ordering::Relaxed),
            total_output_tokens: self.total_output_tokens.load(Ordering::Relaxed),
            total_cache_read_tokens: self.total_cache_read_tokens.load(Ordering::Relaxed),
            total_cache_creation_tokens: self.total_cache_creation_tokens.load(Ordering::Relaxed),
        }
    }

    /// Export statistics as JSON.
    pub fn export_json(&self) -> serde_json::Value {
        let summary = self.summary();

        let model_counts: HashMap<String, u64> = self.model_counts.lock()
            .map(|counts| counts.iter().map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed))).collect())
            .unwrap_or_default();

        let backend_counts: HashMap<String, u64> = self.backend_counts.lock()
            .map(|counts| counts.iter().map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed))).collect())
            .unwrap_or_default();

        serde_json::json!({
            "summary": {
                "total_requests": summary.total_requests,
                "success_count": summary.success_count,
                "client_error_count": summary.client_error_count,
                "server_error_count": summary.server_error_count,
                "success_rate_percent": format!("{:.2}", summary.success_rate),
                "avg_duration_ms": format!("{:.2}", summary.avg_duration_ms),
                "tokens": {
                    "input": summary.total_input_tokens,
                    "output": summary.total_output_tokens,
                    "cache_read": summary.total_cache_read_tokens,
                    "cache_creation": summary.total_cache_creation_tokens,
                }
            },
            "model_counts": model_counts,
            "backend_counts": backend_counts,
        })
    }

    /// Print statistics summary.
    pub fn print_summary(&self) {
        let summary = self.summary();
        info!(
            "Statistics: total={}, success={}, client_err={}, server_err={}, rate={:.1}%, avg={:.0}ms",
            summary.total_requests,
            summary.success_count,
            summary.client_error_count,
            summary.server_error_count,
            summary.success_rate,
            summary.avg_duration_ms
        );
        info!(
            "Tokens: input={}, output={}, cache_read={}, cache_creation={}",
            summary.total_input_tokens,
            summary.total_output_tokens,
            summary.total_cache_read_tokens,
            summary.total_cache_creation_tokens
        );
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.client_error_count.store(0, Ordering::Relaxed);
        self.server_error_count.store(0, Ordering::Relaxed);
        self.total_duration_ms.store(0, Ordering::Relaxed);
        self.total_input_tokens.store(0, Ordering::Relaxed);
        self.total_output_tokens.store(0, Ordering::Relaxed);
        self.total_cache_read_tokens.store(0, Ordering::Relaxed);
        self.total_cache_creation_tokens.store(0, Ordering::Relaxed);

        if let Ok(mut counts) = self.model_counts.lock() {
            counts.clear();
        }
        if let Ok(mut counts) = self.backend_counts.lock() {
            counts.clear();
        }
        if let Ok(mut recent) = self.recent_requests.lock() {
            recent.clear();
        }
    }
}

/// Statistics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    pub total_requests: u64,
    pub success_count: u64,
    pub client_error_count: u64,
    pub server_error_count: u64,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_cache_creation_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_parse() {
        let chunk = r#"data: {"type":"message_start","message":{"id":"msg_xxx","usage":{"input_tokens":100,"output_tokens":0}}}"#;
        let usage = TokenUsage::parse_from_sse(chunk);
        assert!(usage.is_some());
        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn test_stats_record() {
        let stats = RequestStats::new(100);
        stats.record_request(RequestRecord {
            timestamp: Utc::now(),
            method: "POST".to_string(),
            uri: "/v1/messages".to_string(),
            backend: "anthropic".to_string(),
            model: Some("claude-3-opus".to_string()),
            status: 200,
            duration_ms: 1000,
            input_tokens: Some(100),
            output_tokens: Some(50),
            cache_read_tokens: None,
            cache_creation_tokens: None,
            error: None,
        });

        let summary = stats.summary();
        assert_eq!(summary.total_requests, 1);
        assert_eq!(summary.success_count, 1);
        assert_eq!(summary.total_input_tokens, 100);
        assert_eq!(summary.total_output_tokens, 50);
    }
}
