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
| GLM (Zhipu AI) | glm-4, glm-5 | `/chat/completions` | Keeps function tools, flattens content, `tool_choice` → `auto` |
| Kimi (Moonshot) | moonshot-v1-* | `/v1/chat/completions` | Keeps tools/content arrays, uses `max_completion_tokens` |
| DeepSeek | deepseek-chat | `/v1/chat/completions` | Keeps tools/tool_choice, maps reasoning effort |
| MiniMax | ab-01, ab-02 | `/v1/chat/completions` | Keeps tools/content arrays, uses `max_completion_tokens` |

## Quick Start

### 1. Configure

Edit `config.json`:

```json
{
  "listen": "0.0.0.0:8280",
  "log_dir": "./logs",
  "log_body": false,
  "conversation_ttl_seconds": 7200,
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

Top-level optional fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_body` | boolean | `false` | Write per-request converted/debug bodies |
| `conversation_ttl_seconds` | integer | `7200` | Expiry for in-memory `previous_response_id` history |

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
- Keeps function `tools`
- Downgrades `tool_choice` to `auto`
- Converts `developer` role to `system`
- Flattens content array to string

### MiniMax

- Converts `developer` role to `system`
- Keeps content arrays for the OpenAI-compatible endpoint
- Uses `max_completion_tokens`
- Ensures response content is string format

### Kimi

- Converts `developer` role to `system`
- Keeps content arrays
- Uses `max_completion_tokens`
- Downgrades `tool_choice` to `auto`

### DeepSeek

- Converts `developer` role to `system`
- Keeps function tools and `tool_choice`
- Uses `max_tokens`
- Maps Responses reasoning effort to DeepSeek-supported values

### Built-in Responses tools

Responses built-ins such as `web_search_preview`, `file_search`, and
`code_interpreter` are degraded to Chat API function tools. This preserves the
wire shape for Codex/tool-call loops, but it is not equivalent to OpenAI's
server-side built-in tool execution.

## Monitoring

### Logs

Logs are written to the configured `log_dir`:

```
./logs/
├── codex-convert-proxy.log
└── error.log
```

Request/response body dumps are disabled by default. Set `log_body: true` only
for local debugging; dumps use per-request filenames such as
`{request_id}.converted_request.json` to avoid concurrent overwrite, and failed
conversion request bodies are written only under the same flag.

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
│   ├── filters.rs       # Thin Pingora ProxyHttp orchestration
│   ├── routing.rs       # Backend selection and request rewriting
│   ├── request_body.rs  # Responses request buffering/conversion
│   ├── response_body.rs # Chat response conversion
│   ├── error_response.rs # JSON/SSE error helpers
│   └── context_store.rs # In-memory previous_response_id store
├── providers/
│   ├── capabilities.rs  # Provider capability/extension policy
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

The proxy layer owns HTTP concerns only. Protocol conversion lives under
`convert/`, provider modules declare capabilities plus small extensions, and
streaming state stores only incremental stream accumulation. The conversation
store is intentionally in-memory, backend-namespaced, capped by LRU, and expires
entries after two hours by default; restart or multi-process deployments do not
share history.

## License

MIT
