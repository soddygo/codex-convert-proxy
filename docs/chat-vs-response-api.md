# Chat API vs Response API 协议转换文档

> 基于 OpenAI 官方文档 (https://platform.openai.com/docs) 整理

## 1. 背景与目标

本项目实现一个 **双向协议转换网关**，用于：

- **Inbound（Codex → 国内Provider）**：将 Codex 的 Response API 请求转换为国内大模型兼容的 Chat API 格式
- **Outbound（Provider → Codex）**：将国内大模型的 Chat API 响应转换回 Response API 格式

这样 Codex Agent 可以通过网关使用国内大模型（如 GLM、Kimi、DeepSeek、Minimax）。

---

## 2. 两种 API 概述对比

| 特性 | Chat Completions API | Responses API |
|------|----------------------|---------------|
| **Endpoint** | `POST /v1/chat/completions` | `POST /v1/responses` |
| **设计理念** | 消息历史会话 | 分离式指令+输入 |
| **系统提示** | `messages` 数组首条 system 消息 | 独立 `instructions` 字段 |
| **用户输入** | `messages` 数组最后 user 消息 | 独立 `input` 字段 |
| **输出格式** | `choices[].message.content` | `output_text` + `output[]` 数组 |
| **工具调用** | `tool_calls` 嵌套在 message 内 | 独立 `output[]` 项（function_call 类型） |
| **缓存优化** | 普通 | 更好（40-80% 提升），支持 `previous_response_id` |
| **适用场景** | 传统对话、聊天应用 | Agent、多轮推理、工具调用 |

---

## 3. Endpoint 与基础格式

### 3.1 Chat Completions API

```
POST https://api.openai.com/v1/chat/completions
```

**简单请求示例：**
```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ]
}
```

### 3.2 Responses API

```
POST https://api.openai.com/v1/responses
```

**简单请求示例：**
```json
{
  "model": "gpt-4o",
  "instructions": "You are a helpful assistant.",
  "input": "Hello!"
}
```

---

## 4. Chat Completions API 详细规格

### 4.1 请求参数 (POST /v1/chat/completions)

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | ✅ | 模型 ID，如 "gpt-4o", "gpt-4o-mini" |
| `messages` | array | ✅ | 消息对象数组 |
| `temperature` | number | | 采样温度 0-2，默认 1 |
| `top_p` | number | | 核采样参数 |
| `n` | integer | | 生成多少个 completion choices（Response API 只支持 1） |
| `stream` | boolean | | true 启用流式响应 |
| `stop` | string/array | | 最多 4 个停止序列 |
| `max_tokens` | integer | | 生成最大 token 数 |
| `presence_penalty` | number | | 基于是否已出现惩罚 |
| `frequency_penalty` | number | | 基于频率惩罚 |
| `logit_bias` | object | | 修改特定 token 概率 |
| `user` | string | | 最终用户标识 |
| `tools` | array | | 模型可调用的工具列表 |
| `tool_choice` | string/object | | 控制工具调用策略 |
| `response_format` | object | | 输出格式 (json_object/text) |
| `seed` | integer | | 确定性输出种子 |
| `modalities` | array | | 输出模态 (text, audio) |
| `audio` | object | | 音频配置 |
| `stream_options` | object | | 流式响应选项 `{ include_usage: boolean }` |
| `reasoning_effort` | string | | 推理模型思考预算 (low/medium/high) |
| `max_output_tokens` | integer | | **输出**最大 token 数（部分 provider 不支持） |
| `store` | boolean | | 是否存储完成结果用于蒸馏/微调（不是对话历史存储） |
| `parallel_tool_calls` | boolean | | 是否启用并行工具调用（默认 true） |
| `prediction` | object | | 预测参数，用于减少延迟 |

### 4.2 Message 对象结构

```json
{
  "role": "system | user | assistant | developer | tool",
  "content": "消息内容",
  "name": "可选 - 发送者名称",
  "tool_calls": [
    {
      "id": "tool_call_id",
      "type": "function",
      "function": {
        "name": "function_name",
        "arguments": "{\"arg\": \"value\"}"
      }
    }
  ],
  "tool_call_id": "工具调用 ID（role=tool 时必填）"
}
```

**注意**：`developer` role 是为系统级指令保留的，功能上等同于 `system`。

### 4.3 响应格式 (非流式)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1694268190,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助你的吗？",
        "tool_calls": [...]
      },
      "finish_reason": "stop | length | tool_calls | content_filter | refuse | function_call",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30,
    "prompt_tokens_details": {...},
    "completion_tokens_details": {...}
  },
  "service_tier": "scale | premium | default"
}
```

### 4.4 流式响应格式 (Streaming)

每个 SSE chunk 是一个独立的 JSON 行：

```json
{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
```

**ChatDelta 结构：**
```json
{
  "content": "可选 - 部分文本内容",
  "role": "可选 - assistant",
  "tool_calls": [
    {
      "index": 0,
      "id": "call_xxx",
      "type": "function",
      "function": {
        "name": "function_name",      // 可能为空，后续chunk继续传输
        "arguments": "部分 JSON 参数" // 可能为空，后续chunk继续传输
      }
    }
  ],
  "function_call": {...},  // 已废弃，用 tool_calls
  "refusal": "可选 - 拒绝消息（对应 refusal 类型 output）"
}
```

**⚠️ 重要：tool_calls 分块传输**

流式传输时，一个 tool_call 的 `id`、`name`、`arguments` 可能分布在**不同的 chunk** 中到达：

```json
// Chunk 1: 只有 id
{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"","arguments":""}}]}}]}

// Chunk 2: 只有 name
{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"get_weather","arguments":""}}]}}]}

// Chunk 3: 部分 arguments
{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"","arguments":"{\"location\":"}}]}}]}

// Chunk 4: 剩余 arguments + finish_reason
{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"","arguments":"\"Beijing\"}"}}]},"finish_reason":"tool_calls"}]}
```

**转换实现要点**：
1. 维护 `StreamState` 状态机，跟踪正在进行的 tool_call
2. 当收到新的 `tool_calls` 时，**追加**而非覆盖 `id`、`name`、`arguments`
3. 只有收到 `finish_reason="tool_calls"` 才认为 tool_call 完成

---

## 5. Responses API 详细规格

### 5.1 请求参数 (POST /v1/responses)

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | ✅ | 模型 ID |
| `input` | string/array | ✅ | 用户输入，可以是字符串或结构化内容数组 |
| `instructions` | string | | 系统指令，类似于 system prompt |
| `stream` | boolean | | true 启用流式 |
| `tools` | array | | 工具列表（web_search, code_interpreter, file_search, function） |
| `tool_choice` | string/object | | 工具选择策略 |
| `temperature` | number | | 采样温度 |
| `top_p` | number | | 核采样 |
| `max_tokens` | integer | | 最大生成 token |
| `metadata` | object | | 附加元数据（键值对） |
| `previous_response_id` | string | | 多轮对话上下文（重要！Chat API 无此功能） |
| `thinking` | object | | 推理配置 `{ effort: "low/medium/high", summary: "auto" }` |
| `text` | object | | 文本格式配置（Zod schema 等） |
| `store` | boolean | | 是否存储对话到历史（默认 false，用于多轮对话持久化） |

### 5.2 Input 结构

`input` 可以是简单字符串：

```json
"input": "Hello!"
```

或结构化数组（支持多模态和多角色）：

```json
"input": [
  {
    "role": "user",
    "content": [
      { "type": "input_text", "text": "描述一下这张图" },
      { "type": "input_image", "image_url": "https://example.com/image.jpg" }
    ]
  },
  {
    "role": "assistant",
    "content": [{ "type": "output_text", "text": "这是助手的回复" }]
  },
  {
    "role": "tool",
    "content": [{ "type": "input_text", "text": "工具返回的结果: 25°C" }],
    "call_id": "call_abc123"
  }
]
```

**⚠️ 重要：`input` 数组中的特殊 role**：

| role | 说明 | 转换到 Chat API |
|------|------|----------------|
| `user` | 用户消息 | `role: "user"` |
| `assistant` | 助手消息 | `role: "assistant"` |
| `system` | 系统指令 | 转为 Chat API system 消息 |
| `tool` | 工具结果 | `role: "tool"`, `tool_call_id: call_id` |

### 5.3 响应格式 (非流式)

**⚠️ 重要：`output_text` vs `output[]` 的关系**

```json
{
  "id": "resp_xxx",
  "output_text": "这是简洁的文本输出",  // SDK 便捷属性，等同于 output[message].content[output_text].text
  "output": [
    {
      "id": "msg_xxx",
      "type": "message",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "这是完整的文本输出"
        }
      ],
      "role": "assistant"
    }
  ]
}
```

**转换到 Chat API 时**：
- `output_text` 是 SDK 的便捷属性，核心数据在 `output[]` 数组
- 需遍历 `output[]` 提取文本内容：`output[].content[].text`
- 工具调用在 `output[]` 中是独立的 `function_call` 类型项

**流式响应中的 usage**：

Chat API 开启 `stream_options: { include_usage: true }` 时，最后一个 chunk 包含 usage：

```json
// 最后一个 chunk
{
  "choices": [],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

Responses API 的 usage 包含在最终的 `response.completed` 事件中。

```json
{
  "id": "resp_68af4030592c81938ec0a5fbab4a3e9f05438e46b5f69a3b",
  "object": "response",
  "created_at": 1756315696,
  "model": "gpt-5-2025-08-07",
  "output": [
    {
      "id": "rs_68af4030baa48193b0b43b4c2a176a1a05438e46b5f69a3b",
      "type": "reasoning",
      "content": [],
      "summary": [
        {
          "type": "summary_text",
          "text": "**思考过程摘要**..."
        }
      ]
    },
    {
      "id": "msg_68af40337e58819392e935fb404414d005438e46b5f69a3b",
      "type": "message",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "这是模型的回复内容",
          "annotations": [],
          "logprobs": []
        }
      ],
      "role": "assistant"
    },
    {
      "id": "fc_6888f6d86e28819aaaa1ba69cca766b70e683233d39188e7",
      "type": "function_call",
      "status": "completed",
      "name": "get_weather",
      "arguments": "{\"location\":\"San Francisco, CA\",\"unit\":\"f\"}",
      "call_id": "call_XOnF4B9DvB8EJVB3JvWnGg83"
    }
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 50,
    "total_tokens": 65
  },
  "reasoning": {
    "steps": [...]
  }
}
```

**Output Item 类型：**

| type | 说明 | 是否有 status |
|------|------|--------------|
| `reasoning` | 推理过程（包含 summary 摘要） | ❌ 无 |
| `message` | 消息输出 | ✅ `completed` |
| `function_call` | 函数调用 | ✅ `completed` |
| `refusal` | 拒绝输出（是 message.content 内的类型） | ❌ 无（包含在 message 内） |
| `not_sensitive_image` | 图片内容 | ❌ 无 |
| `transcribed_audio` | 转录音频 | ❌ 无 |
| `code_interpreter_call` | Code interpreter 调用 | ✅ 有多种状态 |
| `file_search_call` | File search 调用 | ✅ 有多种状态 |

**⚠️ 重要**：`refusal` 不是独立的 output item type，而是在 `message` output 的 `content` 数组中的一个元素，类型为 `{ type: "refusal", refusal: "..." }`。

### 5.4 流式响应事件

Responses API 流式返回 SSE 事件，TypeScript 类型定义：

```typescript
type StreamingEvent =
  | ResponseCreatedEvent
  | ResponseInProgressEvent
  | ResponseFailedEvent
  | ResponseCompletedEvent
  | ResponseOutputItemAdded
  | ResponseOutputItemDone
  | ResponseContentPartAdded
  | ResponseContentPartDone
  | ResponseOutputTextDelta
  | ResponseOutputTextAnnotationAdded
  | ResponseTextDone
  | ResponseRefusalDelta
  | ResponseRefusalDone
  | ResponseFunctionCallArgumentsDelta
  | ResponseFunctionCallArgumentsDone
  | ResponseFileSearchCallInProgress
  | ResponseFileSearchCallSearching
  | ResponseFileSearchCallCompleted
  | ResponseCodeInterpreterInProgress
  | ResponseCodeInterpreterCallCodeDelta
  | ResponseCodeInterpreterCallCodeDone
  | ResponseCodeInterpreterCallInterpreting
  | ResponseCodeInterpreterCallCompleted
  | Error
```

**核心事件详解：**

| 事件 | 说明 | 典型 payload |
|------|------|-------------|
| `response.created` | 响应创建 | `{ id, object, created_at, model, status }` |
| `response.in_progress` | 开始处理 | `{ id }` |
| `response.output_item.added` | 输出项添加 | `{ output_index, item: { id, type, call_id?, name?, arguments? } }` |
| `response.content_part.added` | 内容部分添加 | `{ output_index, item_id, content_index }` |
| `response.output_text.delta` | 文本增量 | `{ output_index, item_id, content_index, delta }` |
| `response.output_text.done` | 文本完成 | `{ output_index, item_id, content_index }` |
| `response.refusal.delta` | 拒绝增量 | `{ output_index, item_id, content_index, delta }` |
| `response.refusal.done` | 拒绝完成 | `{ output_index, item_id, content_index }` |
| `response.function_call_arguments.delta` | 函数参数增量 | `{ output_index, item_id, call_id, delta }` |
| `response.function_call_arguments.done` | 函数参数完成 | `{ event_id, response_id, output_index, item_id, call_id, arguments }`（name 在 output_item.added 已提供） |
| `response.output_item.done` | 输出项完成 | `{ event_id, output_index, item_id, response_id, item }` |
| `response.completed` | 整个响应完成 | 包含最终 ResponseObject，status 可能为 completed/cancelled/failed/incomplete |
| `error` | 错误 | `{ type, message, code }` |

**SSE 格式：**
```
event: response.created
data: {"type":"response.created","response":{"id":"resp_xxx","object":"response","created_at":1234567890,"model":"gpt-4o","status":"in_progress"}}

event: response.in_progress
data: {"type":"response.in_progress","response":{"id":"resp_xxx"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_xxx","output_index":0,"content_index":0,"delta":"Hello"}

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_xxx","output_index":0,"content_index":0}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item_id":"msg_xxx"}

event: response.completed
data: {"type":"response.completed","response":{...}}
```

---

## 6. 工具 (Tools) 格式差异

### 6.1 Chat API 工具格式

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定位置的天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  }
}
```

### 6.2 Responses API 工具格式

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "获取指定位置的天气",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "城市名称"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

**关键差异：**

| 差异点 | Chat API | Responses API |
|--------|----------|---------------|
| 函数名位置 | `function.name` | `name`（根级别） |
| 描述位置 | `function.description` | `description` |
| 参数位置 | `function.parameters` | `parameters` |
| 结构 | 嵌套 `function` 对象 | 扁平结构 |

### 6.3 Responses API 内置工具

Responses API 支持内置工具，Chat API 需要转换为 function 格式：

| 内置工具 | 说明 | Chat API 转换 |
|---------|------|--------------|
| `web_search` | 网络搜索 | 转换为 `{ type: "function", name: "web_search", parameters: { query: string } }` |
| `code_interpreter` | 代码解释器 | 转换为 `{ type: "function", name: "code_interpreter", parameters: { code: string } }` |
| `file_search` | 文件搜索 | 转换为 `{ type: "function", name: "file_search", parameters: { query: string } }` |

### 6.4 Tool Choice 格式

**Chat API：**
```json
"tool_choice": "auto" | "none" | "required"
```
或对象形式：
```json
"tool_choice": {
  "type": "function",
  "function": { "name": "get_weather" }
}
```

**Responses API：**
```json
"tool_choice": "auto" | "none" | "required" | { "type": "function", "name": "get_weather" }
```

格式基本兼容。

---

## 7. 字段映射关系

### 7.1 请求字段映射 (Response → Chat)

| Responses API | Chat API | 转换说明 |
|--------------|----------|---------|
| `model` | `model` | 直接传递 |
| `instructions` | `messages[0]` (role: system) | 转为 system 消息 |
| `input` (string) | `messages[last]` (role: user) | 作为最后一条 user 消息 |
| `input` (array) | `messages[]` | 遍历转换 |
| `input[].role` | `messages[].role` | user/assistant/system 直接映射 |
| `input[].content[]` | `messages[].content` | input_text → text, input_image → image_url |
| `tools` | `tools` | 扁平 function 结构 → 嵌套 function 结构 |
| `tool_choice` | `tool_choice` | 格式兼容 |
| `stream` | `stream` | 直接传递 |
| `temperature` | `temperature` | 直接传递 |
| `max_tokens` | `max_tokens` | 直接传递 |
| `top_p` | `top_p` | 直接传递 |
| `thinking.budget_tokens` | `reasoning_effort` | 概念对应，但不是完全等价 |
| `previous_response_id` | ❌ | Chat API 无此概念，需自行维护历史 |
| `response_format` | `text.format` | 结构不同，需重新组织 |
| `store` | ❌ | Chat API 无此概念 |
| `presence_penalty` | ❌ | Responses API 不支持 |
| `frequency_penalty` | ❌ | Responses API 不支持 |
| `logit_bias` | ❌ | Responses API 不支持 |

### 7.2 响应字段映射 (Chat → Response)

| Chat API | Responses API | 转换说明 |
|----------|--------------|---------|
| `id` (chatcmpl_xxx) | `id` (resp_xxx) | 前缀替换 |
| `model` | `model` | 直接传递 |
| `created` (unix timestamp) | `created_at` (unix timestamp) | 都是 unix timestamp，字段名不同 |
| `choices[0].message` | `output[]` (type: message) | 转为 OutputItem，content 在 `output[].content[].text` |
| `choices[0].finish_reason` | `output[].status` | stop/tool_calls → completed；length → incomplete；content_filter → content_filtered；refuse → completed（内容为 refusal） |
| `choices[0].message.refusal` | `output[].content[]` (type: refusal) | refusal 是 message content 数组内的类型，不是独立 output item |
| `choices[0].message.tool_calls` | `output[]` (type: function_call) | 拆分为独立输出项 |
| `usage.prompt_tokens` | `usage.input_tokens` | 字段重命名（u32 → i64） |
| `usage.completion_tokens` | `usage.output_tokens` | 字段重命名（u32 → i64） |
| `usage.total_tokens` | `usage.total_tokens` | 直接传递 |
| `service_tier` | ❌ | Response API 无此字段 |
| `choices[].logprobs` | ❌ | Response API 无此字段 |
| `choices[].message.annotations` | `output[].content[].annotations` | 输出文本的注解（仅在 Response API 中） |

### 7.3 流式事件映射 (Chat → Response)

| Chat Delta | Response Event | 备注 |
|-----------|---------------|------|
| 首 chunk | `response.created` + `response.in_progress` | 初始化流 |
| `delta.content` | `response.output_text.delta` | 文本增量 |
| `delta.tool_calls[].id` | 触发 `response.output_item.added` (type: function_call) | 包含 name="" 和 arguments="" |
| `delta.tool_calls[].function.name` | 累积在 ToolCallState.name 中 | 后续 chunk 继续到达 |
| `delta.tool_calls[].function.arguments` | `response.function_call_arguments.delta` | arguments 分块到达 |
| finish_reason=stop | `response.output_text.done` → `response.output_item.done` → `response.completed` | 完整结束流程 |
| finish_reason=tool_calls | `response.function_call_arguments.done` → `response.output_item.done` → `response.completed` | tool_call 结束 |
| finish_reason=function_call | 同 tool_calls | 已废弃，旧版 Provider 可能用 |
| finish_reason=length | `response.output_text.done` + `response.completed` (status=incomplete) | 输出被截断 |
| finish_reason=content_filter | `response.output_text.done` + `response.completed` | 内容被过滤 |
| finish_reason=refuse | `response.refusal.done` → `response.completed` | 模型拒绝 |
| `delta.refusal` | `response.refusal.delta` | 拒绝内容增量 |
| `delta.reasoning_content` (部分Provider扩展) | `response.output_text.delta` (在 reasoning item 中) | 需跟踪 reasoning 状态 |
| usage（在最后 chunk） | 包含在 `response.completed` 的 response 对象中 | 需开启 `include_usage: true` |

**⚠️ output_index 和 content_index 递增规则：**
- `output_index`：每当有新的 output item 开始（如 text、function_call、reasoning）时 +1
- `content_index`：每当同一个 item 内有新的 content part 时 +1

---

## 8. 错误响应格式

### 8.1 OpenAI 标准错误格式

```json
{
  "error": {
    "type": "server_error | rate_limit_exceeded | invalid_request_error | authentication_error | etc",
    "message": "错误描述信息",
    "code": "optional_error_code",
    "param": "optional_related_param",
    "request_id": "req_xxx"
  }
}
```

**常见错误类型：**

| type | HTTP Status | 说明 |
|------|-------------|------|
| `invalid_request_error` | 400 | 请求参数错误 |
| `authentication_error` | 401 | API Key 无效 |
| `rate_limit_exceeded` | 429 | 速率限制 |
| `server_error` | 500 | 服务器内部错误 |
| `internal_server_error` | 500 | 服务器内部错误 |

### 8.2 转换错误处理

网关层错误应保持格式一致，建议：

```json
{
  "error": {
    "type": "proxy_error",
    "message": "Provider timeout: glm-4",
    "code": "PROVIDER_ERROR",
    "request_id": "local_xxx"
  }
}
```

---

## 9. 多轮对话处理

### 9.1 Chat API 多轮

Chat API 每次请求需携带完整历史 messages：

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are helpful" },
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi!" },
    { "role": "user", "content": "How are you?" }
  ]
}
```

**⚠️ `n` 参数（多 choice）的处理**：

Chat API 支持 `n > 1` 生成多个回复，但 Responses API 只支持 `n = 1`。

**转换策略**：
- Response → Chat：如果 Codex 传了 `n > 1`，选择 `choices[0]` 转换，其余丢弃（需记录）
- Chat → Response：`output[]` 数组中只会有一个 message 项

### 9.2 Responses API 多轮

Responses API 支持 `previous_response_id` 自动维护历史：

```json
{
  "model": "gpt-4o",
  "instructions": "You are helpful",
  "input": "How are you?",
  "previous_response_id": "resp_abc123"
}
```

**优势**：无需每次传递完整历史，缓存效率更高（40-80%）。

### 9.3 转换策略

**Response → Chat 时**：
1. 如果有 `previous_response_id`，需展开其历史 messages 到当前请求
2. `instructions` 每次都转为 `messages[0]` (system)
3. 最新的 `input` 追加到 `messages` 末尾

**Chat → Response 时**：
1. 可选择维护 `previous_response_id`（如 Provider 支持多轮）
2. 每次需重新传 `instructions`（Response API 不自动持久化 system prompt）

**⚠️ 关键挑战**：`instructions` 在 Response API 中是请求级参数，不随 `previous_response_id` 自动保持。如果 Codex 只传了 `previous_response_id` 而没有重复传 `instructions`，网关需要自行缓存并补全。

---

## 10. 特殊字段差异

### 10.1 推理/思考内容

**Responses API（原生支持）：**
```json
{
  "output": [
    {
      "id": "rs_xxx",
      "type": "reasoning",
      "summary": [{ "type": "summary_text", "text": "**思考摘要**" }],
      "content": [{ "type": "reasoning_text", "text": "完整思考过程" }]
    }
  ]
}
```

**Chat API（部分 Provider 扩展）：**
```json
{
  "choices": [{
    "delta": {
      "reasoning_content": "推理内容..."
    }
  }]
}
```

**转换**：将 `delta.reasoning_content` 转为 `output[].type: reasoning` 项。

### 10.2 结构化输出

**Chat API：**
```json
"response_format": { "type": "json_object" }
// 或指定 schema
"response_format": {
  "type": "json_schema",
  "json_schema": { "name": "SchemaName", "schema": {...} }
}
```

**Responses API：**
```json
"text": {
  "format": {
    "type": "json_schema",
    "name": "schema_name",
    "schema": {...},
    "strict": true   // 是否严格遵循 schema
  }
}
```

**转换注意**：`response_format` 与 `text.format` 结构不同，转换时需重新组织格式。

### 10.3 `max_tokens` 语义差异

| API | 语义 | 说明 |
|-----|------|------|
| Chat API | 输出 token 上限 | `max_tokens` 限制模型生成的回答长度 |
| Chat API | 另有 `max_output_tokens` | 部分 provider 支持，显式限制输出 |
| Responses API | **整体** token 上限 | 包括输入+输出+推理+工具调用的总 token 限制 |

**⚠️ 重要**：`max_tokens` 在 Responses API 中是**整个响应**的 token 上限，包括：
1. 输入 token
2. 推理过程 token（如果有）
3. 工具调用 token（如果有）
4. 输出文本 token

**转换注意**：如果 Chat API 请求同时有 `max_tokens` 和 `max_output_tokens`，转换为 Responses API 时取较小值。但注意语义不同，Responses API 的 `max_tokens` 可能比 Chat API 的 `max_tokens` 更早触发截断。

### 10.4 `truncation` 参数差异

| API | 参数 | 说明 |
|-----|------|------|
| Chat API | `truncation` | 控制截断策略，可为 `auto`（默认）或 `last_k` |
| Responses API | ❌ | 无对应参数 |

### 10.5 `function_call` 废弃字段

Chat API 早期使用 `function_call` 字段，已废弃但部分旧 Provider 可能还在用：

```json
// 旧格式（已废弃）
{
  "choices": [{
    "message": {
      "function_call": {
        "name": "get_weather",
        "arguments": "{\"location\":\"Beijing\"}"
      }
    }
  }]
}
```

**转换**：需转换为 `tool_calls` 格式：

```json
// 转换为 tool_calls
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Beijing\"}"
        }
      }]
    }
  }]
}
```

### 10.6 音频输出

**Chat API（需指定 modalities）：**
```json
"modalities": ["text", "audio"],
"audio": { "voice": "alloy", "format": "mp3" }
```

**Responses API**：暂不支持直接音频输出。

---

## 11. 国内 Provider 适配要点

| Provider | 特殊处理 |
|----------|---------|
| **GLM** | 不支持 tools/tool_choice；content 需扁平为 string；developer role → user |
| **Kimi/Moonshot** | 标准 Chat API 兼容 |
| **DeepSeek** | 标准 Chat API 兼容 |
| **MiniMax** | developer role → user；content 需扁平为 string |

---

## 12. 转换示例

### 12.1 简单请求：Response → Chat

**Input (Response API):**
```json
{
  "model": "gpt-4o",
  "instructions": "You are a helpful assistant",
  "input": "Hello!"
}
```

**Output (Chat API):**
```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant" },
    { "role": "user", "content": "Hello!" }
  ]
}
```

### 12.2 工具调用：Response → Chat

**Input (Response API):**
```json
{
  "model": "gpt-4o",
  "instructions": "You can use tools",
  "input": "What's the weather in Beijing?",
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  ]
}
```

**Output (Chat API):**
```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You can use tools" },
    { "role": "user", "content": "What's the weather in Beijing?" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {...}
      }
    }
  ]
}
```

### 12.3 工具调用响应：Chat → Response

**Input (Chat API):**
```json
{
  "id": "chatcmpl_123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'll check the weather for you.",
      "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Beijing\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Output (Response API):**
```json
{
  "id": "resp_chatcmpl_123",
  "output": [
    {
      "id": "msg_resp_123",
      "type": "message",
      "content": [{ "type": "output_text", "text": "I'll check the weather for you." }],
      "role": "assistant"
    },
    {
      "id": "call_abc",
      "type": "function_call",
      "name": "get_weather",
      "arguments": "{\"location\":\"Beijing\"}",
      "call_id": "call_abc"
    }
  ]
}
```

### 12.4 流式响应：Chat → Response

**Input (Chat Streaming Chunk):**
```json
{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}
{"choices":[{"index":0,"delta":{"content":"Hello"}}]}
{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
```

**Output (Response Streaming Events):**
```
event: response.created
data: {"type":"response.created","response":{"id":"resp_xxx","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_xxx","type":"message"}}

event: response.content_part.added
data: {"type":"response.content_part.added","output_index":0,"item_id":"msg_xxx","content_index":0}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_xxx","output_index":0,"content_index":0,"delta":"Hello"}

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_xxx","output_index":0,"content_index":0}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item_id":"msg_xxx"}

event: response.completed
data: {"type":"response.completed","response":{...}}
```

---

## 13. 参考资料

- [OpenAI Responses API 文档](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI Chat Completions API 文档](https://platform.openai.com/docs/api-reference/chat)
- [Responses vs Chat Completions 迁移指南](https://platform.openai.com/docs/guides/migrate-to-responses)
- [Streaming Responses 指南](https://platform.openai.com/docs/guides/streaming-responses)
- [Function Calling 指南](https://platform.openai.com/docs/guides/function-calling)
- [Responses API vs Chat Completions 对比](https://platform.openai.com/learn/agents)
