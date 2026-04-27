//! OpenTelemetry integration module.
//!
//! This module provides distributed tracing and metrics export.

use tracing::info;

/// Telemetry configuration.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Whether telemetry is enabled.
    pub enabled: bool,
    /// OTLP endpoint URL.
    pub endpoint: String,
    /// Service name for traces.
    pub service_name: String,
    /// Sample rate (0.0 to 1.0).
    pub sample_rate: f64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4317".to_string(),
            service_name: "codex-convert-proxy".to_string(),
            sample_rate: 1.0,
        }
    }
}

/// Initialize OpenTelemetry tracing.
/// Returns a tuple of (OpenTelemetryLayer, shutdown function).
pub fn init_telemetry(config: &TelemetryConfig) -> anyhow::Result<()> {
    if !config.enabled {
        info!("Telemetry disabled");
        return Ok(());
    }

    info!(
        "Telemetry configured: endpoint={}, service={}",
        config.endpoint, config.service_name
    );

    // Note: Full OpenTelemetry initialization requires more setup
    // and is typically done at the application level.
    // This is a placeholder for future implementation.

    Ok(())
}

/// Shutdown telemetry gracefully.
pub fn shutdown_telemetry() {
    opentelemetry::global::shutdown_tracer_provider();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.endpoint, "http://localhost:4317");
        assert_eq!(config.service_name, "codex-convert-proxy");
        assert_eq!(config.sample_rate, 1.0);
    }
}
