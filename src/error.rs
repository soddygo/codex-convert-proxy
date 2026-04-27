//! Error types for the library.

use thiserror::Error;

/// Error types for conversion operations.
#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid input format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("SSE parsing error: {0}")]
    SseParseError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Error types for proxy operations.
#[derive(Error, Debug)]
pub enum ProxyError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Upstream error: {0}")]
    UpstreamError(String),

    #[error("Request error: {0}")]
    RequestError(String),

    #[error("Response error: {0}")]
    ResponseError(String),

    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
}
