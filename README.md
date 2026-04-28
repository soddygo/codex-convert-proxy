# Codex Convert Proxy

A high-performance proxy service that converts between OpenAI **Responses API** (used by Codex) and **Chat Completions API** (used by Chinese LLM providers).

## Background

Codex 0.118+ uses the Responses API, but most Chinese LLM providers (GLM, Kimi, DeepSeek, MiniMax) only support the Chat Completions API. This proxy bridges the gap by performing bidirectional format conversion.

## Architecture

```
┌─────────────┐    Responses API    ┌──────────────────┐   Chat API   ┌─────────────────┐
│    Codex    │ ──────────────────▶ │ codex-convert-   │ ──────────▶ │  LLM Provider   │
│   Client    │ ◀────────────────── │     proxy        │ ◀────────── │  (GLM/Kimi/...) │
└─────────────┘    Responses API     └──────────────────┘   SSE/JSON    └─────────────────┘
```

## Features

- **Multi-backend routing**: Route requests to different LLM providers based on path prefix
- **Protocol conversion**: Responses API ↔ Chat API bidirectional conversion
- **Streaming SSE support**: Real-time conversion of Server-Sent Events
- **Provider-specific handling**: Per-provider request/response transformation
- **OpenTelemetry tracing**: Built-in observability support
- **TLS support**: Secure upstream connections via rustls

## Supported Providers

| Provider | Model | API Path | Notes |
|----------|-------|----------|-------|
| GLM (Zhipu AI) | glm-4, glm-5 | `/chat/completions` | Removes tools, flattens content |
| Kimi (Moonshot) | moonshot-v1-* | `/v1/chat/completions` | OpenAI compatible |
| DeepSeek | deepseek-chat | `/v1/chat/completions` | OpenAI compatible |
| MiniMax | ab-01, ab-02 | `/v1/chat/completions` | Flattens content, converts developer role |

## Quick Start

### 1. Configure

Edit `config.json`:

```json
{
  "listen": "0.0.0.0:8280",
  "log_dir": "./logs",
  "log_body": false,
  "backends": [
    {
      "name": "glm",
      "url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key": "YOUR_API_KEY",
      "model": "glm-5",
      "protocol": "openai",
      "match_rules": {
        "default": true
      }
    },
    {
      "name": "kimi",
      "url": "https://api.moonshot.cn/v1",
      "api_key": "YOUR_API_KEY",
      "model": "moonshot-v1-8k",
      "protocol": "openai",
      "match_rules": {
        "path_prefix": "/kimi"
      }
    },
    {
      "name": "deepseek",
      "url": "https://api.deepseek.com/v1",
      "api_key": "YOUR_API_KEY",
      "model": "deepseek-chat",
      "protocol": "openai",
      "match_rules": {
        "path_prefix": "/deepseek"
      }
    }
  ]
}
```

### 2. Build & Run

```bash
# Development
cargo run --bin server -- --config config.json

# Production
cargo build --release
./target/release/codex-convert-proxy --config config.json
```

Or use Make:

```bash
make run
```

### 3. Test with Codex

```bash
# Set Codex endpoint
export CODEX_API_URL=http://localhost:8280

# Or use codex exec with proxy
codex exec --provider openai-compatible --api-key dummy --base-url http://localhost:8280/v1 --model glm-5 "Hello, write a hello world in Python"
```

## Configuration Reference

### Backend Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Backend identifier |
| `url` | string | Yes | Upstream API base URL |
| `api_key` | string | Yes | API authentication key |
| `model` | string | Yes | Model name to use with this backend |
| `protocol` | string | Yes | Currently only `"openai"` is supported |
| `match_rules.path_prefix` | string | No | Route requests by path prefix |
| `match_rules.default` | boolean | No | Set as default backend |

### Match Rules

Requests are routed based on `match_rules`:

1. **path_prefix**: Matches if request path starts with the prefix (e.g., `/kimi` → `kimi` backend)
2. **default**: The fallback backend when no prefix matches

## API Endpoints

The proxy handles the Responses API format at the root path:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Convert and forward Responses API request |
| GET | `/health` | Health check |

## Provider-Specific Handling

### GLM

- API path: `/chat/completions` (not `/v1/chat/completions`)
- Removes `tools` and `tool_choice` from requests (not supported)
- Converts `developer` role to `user`
- Flattens content array to string

### MiniMax

- Converts `developer` role to `user`
- Flattens content array to string
- Ensures response content is string format

### Kimi / DeepSeek

- Fully OpenAI Chat API compatible
- No special transformation needed

## Monitoring

### Logs

Logs are written to the configured `log_dir`:

```
./logs/
├── codex-convert-proxy.log
└── error.log
```

### OpenTelemetry

Configure OTLP endpoint for tracing:

```json
{
  "telemetry": {
    "enabled": true,
    "service_name": "codex-convert-proxy",
    "otlp_endpoint": "http://localhost:4317"
  }
}
```

## Development

```bash
# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run --bin server -- --config config.json

# Build release
cargo build --release
```

## Project Structure

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library exports
├── config.rs            # Configuration loading
├── proxy/
│   └── proxy_impl.rs    # Pingora ProxyHttp implementation
├── providers/
│   ├── trait_.rs        # Provider trait definition
│   ├── glm.rs           # GLM provider
│   ├── kimi.rs          # Kimi/Moonshot provider
│   ├── deepseek.rs      # DeepSeek provider
│   └── minimax.rs       # MiniMax provider
├── convert/
│   ├── request.rs       # Responses → Chat API conversion
│   ├── response.rs      # Chat → Responses API conversion
│   └── streaming.rs    # SSE streaming conversion
├── types/
│   ├── chat_api.rs      # Chat API types
│   └── response_api.rs  # Responses API types
├── streaming.rs         # SSE parsing utilities
├── server.rs            # Pingora server setup
├── stats.rs             # Request statistics
├── telemetry.rs         # OpenTelemetry setup
├── logger.rs            # Logging configuration
└── error.rs             # Error types
```

## License

MIT
