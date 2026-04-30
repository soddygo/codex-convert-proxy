//! 项目级配置常量

/// 最大请求体大小 (10 MB)
pub const MAX_REQUEST_BODY_SIZE: usize = 10 * 1024 * 1024;

/// 最大响应体大小 (50 MB)
pub const MAX_RESPONSE_BODY_SIZE: usize = 50 * 1024 * 1024;

/// 流式解析缓冲区压缩阈值 (64 KB)
pub const STREAM_PARSE_COMPACT_THRESHOLD: usize = 64 * 1024;

/// 思考标签缓冲区最大大小 (100 MB)
pub const MAX_THINKING_BUFFER_SIZE: usize = 100 * 1024 * 1024;

/// 部分提供商允许的最小 max_tokens 值（低于此值可能被拒绝）
pub const MIN_MAX_TOKENS: u32 = 16;
