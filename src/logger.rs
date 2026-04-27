//! Logging module based on tracing ecosystem.
//!
//! This module provides structured logging with:
//! - Multi-output (stdout + file)
//! - Sensitive data masking
//! - Request lifecycle tracking

use std::path::Path;
use tracing::info;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

/// Global log configuration.
static LOG_CONFIG: std::sync::OnceLock<LogConfig> = std::sync::OnceLock::new();

struct LogConfig {
    #[allow(dead_code)]
    log_body: bool,
    #[allow(dead_code)]
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
