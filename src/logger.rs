//! Logging module based on tracing ecosystem.
//!
//! This module provides structured logging with:
//! - Multi-output (stdout + file)
//! - Sensitive data masking
//! - Request lifecycle tracking

use std::path::Path;
use tracing::{debug, error, info};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

/// Global log configuration.
static LOG_CONFIG: std::sync::OnceLock<LogConfig> = std::sync::OnceLock::new();

struct LogConfig {
    log_body: bool,
    log_headers: bool,
}

/// Initialize the logging system.
///
/// Configures tracing with multiple outputs:
/// - stdout: terminal output (timeline level)
/// - file: detailed log file (all levels)
pub fn init_logging(log_dir: &Path, log_body: bool, log_headers: bool) -> anyhow::Result<()> {
    std::fs::create_dir_all(log_dir)?;

    // Create file appender
    let file_appender = tracing_appender::rolling::daily(log_dir, "proxy.log");
    let file_layer = fmt::layer()
        .with_writer(file_appender)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .with_ansi(false)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")));

    // Terminal output
    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_thread_ids(false)
        .with_ansi(true)
        .with_filter(create_timeline_filter());

    // Initialize subscriber
    Registry::default()
        .with(file_layer)
        .with(stdout_layer)
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;

    // Store global config
    LOG_CONFIG.get_or_init(|| LogConfig {
        log_body,
        log_headers,
    });

    info!("Logging initialized: {}", log_dir.display());
    Ok(())
}

/// Create timeline filter (info level for timeline events).
fn create_timeline_filter() -> EnvFilter {
    EnvFilter::builder()
        .parse("info")
        .unwrap_or_else(|_| EnvFilter::new("info"))
}

/// Check if header is sensitive (should be masked in logs).
pub fn is_sensitive_header(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == "x-api-key"
        || lower == "authorization"
        || lower == "api-key"
        || lower == "x-api-token"
        || lower == "cookie"
        || lower == "set-cookie"
}

/// Mask sensitive values for display.
pub fn mask_sensitive(value: &str) -> String {
    if value.len() <= 10 {
        return "***".to_string();
    }
    if value.starts_with("Bearer ") {
        let token = &value[7..];
        if token.len() <= 10 {
            return "Bearer ***".to_string();
        }
        return format!("Bearer {}***{}", &token[..6], &token[token.len() - 4..]);
    }
    format!("{}***{}", &value[..6], &value[value.len() - 4..])
}

/// Request logger for tracking individual request lifecycle.
pub struct RequestLogger {
    method: String,
    uri: String,
    backend: String,
    span: tracing::Span,
}

impl RequestLogger {
    /// Create a new request logger.
    pub fn new(method: &str, uri: &str, backend: &str) -> Self {
        let span = tracing::info_span!(
            "request",
            method = %method,
            uri = %uri,
            backend = %backend,
        );

        span.in_scope(|| {
            info!("[REQUEST] {} {}", method, uri);
        });

        Self {
            method: method.to_string(),
            uri: uri.to_string(),
            backend: backend.to_string(),
            span,
        }
    }

    /// Log request headers.
    pub fn log_request_headers(&mut self, headers: &[(String, String)]) {
        let config = LOG_CONFIG.get().unwrap();
        if !config.log_headers {
            return;
        }

        self.span.in_scope(|| {
            debug!("Request Headers:");
            for (name, value) in headers {
                let display_value = if is_sensitive_header(name) {
                    mask_sensitive(value)
                } else {
                    value.clone()
                };
                debug!("  {}: {}", name, display_value);
            }
        });
    }

    /// Log upstream request.
    pub fn log_upstream_request(&self, method: &str, uri: &str, headers: &[(String, String)]) {
        self.span.in_scope(|| {
            info!("[UPSTREAM] {} {}", method, uri);

            let config = LOG_CONFIG.get().unwrap();
            if config.log_headers {
                debug!("Upstream Headers:");
                for (name, value) in headers {
                    let display_value = if is_sensitive_header(name) {
                        mask_sensitive(value)
                    } else {
                        value.clone()
                    };
                    debug!("  {}: {}", name, display_value);
                }
            }
        });
    }

    /// Log connection info.
    pub fn log_connection(
        &self,
        sni: &str,
        address: &str,
        use_tls: bool,
        reused: bool,
        tls_version: &str,
    ) {
        self.span.in_scope(|| {
            info!(
                "[CONNECT] {} -> {} (TLS={}, reused={}, tls_version={})",
                sni, address, use_tls, reused, tls_version
            );
        });
    }

    /// Log response start.
    pub fn log_response_start(&self, status: u16, headers: &[(String, String)]) {
        self.span.in_scope(|| {
            info!("[RESPONSE] Status: {}", status);

            let config = LOG_CONFIG.get().unwrap();
            if config.log_headers {
                debug!("Response Headers:");
                for (name, value) in headers {
                    debug!("  {}: {}", name, value);
                }
            }
        });
    }

    /// Log response chunk.
    pub fn log_response_chunk(&self, chunk: &[u8]) {
        let config = LOG_CONFIG.get().unwrap();
        if !config.log_body {
            return;
        }

        if let Ok(text) = std::str::from_utf8(chunk) {
            for line in text.lines() {
                if !line.is_empty() {
                    debug!("  {}", line);
                }
            }
        }
    }

    /// Log request completion.
    pub fn log_request_end(&self, duration_ms: u64, status: u16) {
        self.span.in_scope(|| {
            info!("[DONE] duration: {}ms, status: {}", duration_ms, status);
        });
    }

    /// Log error.
    pub fn log_error(&self, message: &str) {
        self.span.in_scope(|| {
            error!("[ERROR] {}", message);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_sensitive() {
        assert_eq!(mask_sensitive("short"), "***");
        assert_eq!(mask_sensitive("Bearer sk-xxx"), "Bearer ***");
        assert_eq!(
            mask_sensitive("Bearer sk-project-12345"),
            "Bearer sk-pro***2345"
        );
        assert_eq!(
            mask_sensitive("sk-ant-api03-xxxxxxxxxxxx"),
            "sk-ant***xxxx"
        );
    }

    #[test]
    fn test_is_sensitive_header() {
        assert!(is_sensitive_header("x-api-key"));
        assert!(is_sensitive_header("X-API-KEY"));
        assert!(is_sensitive_header("authorization"));
        assert!(is_sensitive_header("Authorization"));
        assert!(!is_sensitive_header("content-type"));
        assert!(!is_sensitive_header("x-request-id"));
    }
}
