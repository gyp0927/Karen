# gpt-agents 技术文档

## 项目概述

**gpt-agents** 是一个功能完善的多 Agent AI 聊天系统，采用模块化架构设计，支持国内外 20+ 家大语言模型（LLM）提供商。系统通过 LangGraph 实现多 Agent 协作，使用 Flask + SocketIO 提供实时 Web 交互，并支持多种部署方式（Web/桌面/控制台）。

### 核心功能

| 功能 | 说明 |
|------|------|
| 多 Agent 协作 | Coordinator → Researcher → Responder → Reviewer 协作流程 |
| 多 LLM 支持 | 20+ 家国内外厂商，OpenAI 兼容 API |
| RAG 知识库 | 基于 Embedding 的文档检索（PDF/Word/文本） |
| 联网搜索 | DuckDuckGo 搜索集成 |
| 代码执行 | Python 代码沙箱（AST 安全检查） |
| 会话管理 | 多会话切换、SQLite 持久化 |
| 聊天记录导出 | Markdown/JSON/HTML/PDF 格式 |
| 流式输出 | Token 级实时响应 |
| API 统计 | Token 用量、费用估算 |
| 主题切换 | 亮色/暗色/系统模式 |

---

## 技术栈

### 编程语言

| 语言 | 用途 |
|------|------|
| Python 3.13 | 后端核心、Agent 逻辑、数据处理 |
| JavaScript | 前端交互（原生，无框架） |
| HTML/CSS | Web 界面 |
| Batch | Windows 启动脚本 |

### 后端依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| flask | >=3.0.0 | Web 框架，提供 HTTP API |
| flask-socketio | >=5.3.0 | WebSocket 实时双向通信 |
| flask-cors | >=4.0.0 | 跨域资源共享支持 |
| langchain-openai | >=0.2.0 | OpenAI 兼容 API 的 LLM 调用 |
| langgraph | >=0.2.0 | 多 Agent 工作流图编排 |
| python-dotenv | >=1.0.0 | 环境变量管理 |
| eventlet | >=0.35.0 | SocketIO 异步驱动 |
| pywebview | >=5.0 | 桌面客户端 WebView2 渲染 |

### 前端依赖（CDN）

| 包名 | 版本 | CDN | 用途 |
|------|------|-----|------|
| Socket.IO Client | 4.7.4 | cdn.socket.io | WebSocket 通信 |
| Marked.js | 9.1.6 | cdnjs | Markdown 渲染 |
| Highlight.js | 11.9.0 | cdnjs | 代码语法高亮 |

### 可选依赖

| 包名 | 用途 |
|------|------|
| numpy | RAG 向量存储和相似度计算 |
| PyPDF2 | PDF 文档解析 |
| python-docx | Word 文档解析 |
| requests | HTTP 请求（Embedding API） |
| weasyprint | PDF 导出（HTML → PDF 转换） |

### 数据存储

| 技术 | 用途 |
|------|------|
| SQLite | 聊天历史、会话管理、消息持久化 |
| JSON 文件 | 模型配置持久化（`state/model_configs.json`） |
| 内存字典 | LLM 实例缓存、向量存储运行时缓存 |

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                               │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│  Web 浏览器  │  桌面客户端  │  桌面应用   │     控制台        │
│  (Flask)    │ (WebView2)  │ (浏览器)    │   (命令行)        │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────┬─────────┘
       │             │             │                │
       └─────────────┴──────┬──────┴────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │     Flask + SocketIO      │
              │      (web/app.py)         │
              └─────────────┬─────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
┌──────┴──────┐    ┌────────┴────────┐   ┌─────┴──────┐
│  状态管理    │    │    核心功能      │   │   工具模块  │
│  State      │    │    Core         │   │   Tools    │
├─────────────┤    ├─────────────────┤   ├────────────┤
│ SessionManager│   │ config.py      │   │ code_executor│
│ persistence │    │ document_parser│   │ search.py   │
│ model_config│    │ export.py      │   └─────────────┘
│ stats.py    │    │ rag.py         │
└─────────────┘    └────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │      Agent 编排层          │
              │    (graph/orchestrator)    │
              └─────────────┬─────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
┌──────┴──────┐    ┌────────┴────────┐   ┌─────┴──────┐
│  Agent 工厂  │    │    提示词模板    │   │   外部接口  │
│  Factory    │    │   Prompts       │   │  Providers │
├─────────────┤    ├─────────────────┤   ├────────────┤
│ coordinator │    │ coordinator_   │   │ DeepSeek   │
│ researcher  │    │   prompt.py    │   │ OpenAI     │
│ responder   │    │ reviewer_      │   │ Anthropic  │
│ reviewer    │    │   prompt.py    │   │ ... 20+    │
│ planner     │    └─────────────────┘   └─────────────┘
└─────────────┘
```

### 多 Agent 协作流程

系统采用 LangGraph 构建工作流图，支持三种运行模式：

#### 1. 协调模式（Coordination Mode）

```
用户输入 → Coordinator → [判断]
                            │
              ┌─────────────┴─────────────┐
              │                           │
        [需要研究]                    [直接回答]
              │                           │
        Researcher ──────────────────→ Responder
                                              │
                                          Reviewer
                                              │
                                    [通过] ←→ [不通过]
                                              │
                                           最终输出
```

- **Coordinator（调度器）**：分析用户输入，输出 `[route: researcher]` 或 `[route: responder]` 路由标记
- **Researcher（研究员）**：对需要深入调研的问题提供详细信息
- **Responder（响应器）**：生成最终回答，支持流式输出到前端
- **Reviewer（审查者）**：审查回答质量，输出 `[通过]`/`[不通过]` 标记，不通过则返回 Responder 重新生成

#### 2. 快速模式（Fast Mode）

```
用户输入 → Responder → 输出
```

跳过 Coordinator 和 Researcher，直接调用 Responder，适合简单对话场景。

#### 3. 计划模式（Planning Mode）

```
用户输入 → Planner → 生成 JSON 计划
                          │
                    { title, steps[] }
                          │
              逐步执行每个步骤 → 汇总结果
```

Planner 生成结构化任务计划（3-8 个步骤），系统逐步执行并汇总。

---

## 核心模块详解

### 1. Web 应用层 (`web/`)

#### `web/app.py` (1745 行)

Flask 主应用，是整个系统的 HTTP/WebSocket 入口。

**核心功能：**
- Flask 应用实例创建，CORS 配置
- SocketIO 初始化（`async_mode="threading"`）
- 配置路由本地保护（仅允许 127.0.0.1/localhost 访问 `/config` 相关路由）
- 聊天记录导出 API（`/api/export`）
- 模型配置管理 API（CRUD 操作）
- RAG 知识库 API（文档上传、检索、清空）
- 统计信息 API
- SocketIO 事件处理：
  - `send_message`：接收用户消息，调用 Agent 图执行
  - `stop_generation`：停止当前生成
  - `new_session`/`switch_session`/`delete_session`：会话管理
  - `upload_file`：文件上传（PDF/DOCX/TXT）
  - `execute_code`：代码执行
  - `export_chat`：导出聊天记录
  - `get_stats`：获取 API 调用统计

**关键设计：**
- 使用 `socket_states` 字典按 Socket ID (sid) 隔离会话状态
- 流式输出通过 SocketIO `emit("stream_chunk")` 实时推送
- 单实例运行保护（通过端口检测）

#### `web/static/script.js` (1374 行)

前端交互逻辑，纯原生 JavaScript，无框架依赖。

**核心功能：**
- Socket.IO 客户端连接管理
- 主题切换（亮色/暗色/系统，持久化到 localStorage）
- 3D Agent 状态面板动画（CSS 3D 翻转效果）
- 消息渲染（Markdown → HTML，代码高亮）
- 流式消息追加显示
- 会话列表管理（新建/切换/删除）
- 模型切换器（模态框，从服务器获取配置列表）
- 文件上传（拖拽 + 点击）
- 代码预览面板（iframe sandbox）
- 输入框自动Resize
- 移动端侧边栏适配

**外部 CDN 依赖：**
```html
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
```

#### `web/static/style.css` (1780 行)

ChatGPT 风格的响应式界面样式。

**设计特点：**
- CSS 变量定义主题色（`data-theme="light"/"dark"` 切换）
- 3D Agent 状态卡片（`transform-style: preserve-3d`，翻转动画）
- 流式光标闪烁动画
- 代码块复制按钮
- 响应式布局（移动端侧边栏全屏覆盖）
- 消息气泡样式（用户/AI 区分）

#### `web/templates/index.html` (288 行)

聊天主界面模板。

**结构：**
- Sidebar（左侧）：新建对话按钮、3D Agent 状态面板、历史会话列表、底部工具栏
- Chat Area（中间）：消息区域、输入框
- Preview Panel（右侧）：代码预览 iframe（sandbox="allow-scripts allow-same-origin"）
- Model Switcher Modal：模型切换弹窗
- User Config Dialog：局域网用户本地 API Key 配置

---

### 2. Agent 层 (`agents/`)

#### `agents/factory.py` (254 行)

Agent 工厂，负责创建和管理 LLM 实例及 Agent 节点函数。

**核心组件：**

**LLM 实例缓存：**
```python
_llm_cache: dict[str, ChatOpenAI] = {}
```
- 使用 JSON 序列化参数作为缓存键
- 线程安全（`_llm_cache_lock`）
- 配置切换时调用 `clear_llm_cache()` 清除

**按 SID 隔离的配置：**
```python
_llm_configs: dict[str, dict | None] = {}
_token_callbacks: dict[str, Callable] = {}
```
- 支持每个 Socket 连接使用独立的 LLM 配置
- 线程安全（`_callbacks_lock`）

**Agent 节点函数：**
| 函数 | 职责 | 系统提示词 |
|------|------|-----------|
| `coordinator_node` | 分析需求并决定路由 | `COORDINATOR_PROMPT`（外部导入） |
| `researcher_node` | 提供深入信息 | 内联定义的研究专家提示词 |
| `responder_node` | 生成最终回答 | 内联定义的助手提示词 |
| `reviewer_node` | 审查回答质量 | `get_reviewer_prompt(language)`（动态生成） |
| `planner_node` | 生成结构化计划 | 内联定义的 JSON 格式计划提示词 |

**流式输出机制：**
- 仅 `responder` 节点触发流式回调
- 使用 `llm.astream(messages)` 异步流式生成
- 通过 `on_token` 回调或全局 `get_streaming_callback(sid)` 推送 token

**计划解析：**
- `parse_plan_from_response()`：从 Agent 输出中提取 JSON 计划
- 支持直接解析、Markdown 代码块提取、正则匹配三种方式

---

### 3. 工作流编排层 (`graph/`)

#### `graph/orchestrator.py` (134 行)

LangGraph 工作流定义，构建多 Agent 协作图。

**三种图构建函数：**

**`create_multi_agent_graph()`** - 带审查的完整图：
```
coordinator → [conditional] → researcher → reviewer → [conditional]
                              ↓              ↓
                           responder ←── [不通过]
                              │
                           [通过] → END
```

**`create_coordination_graph()`** - 协调+研究+响应：
```
coordinator → [conditional] → researcher → responder → END
                              ↓
                           responder
```

**`create_fast_graph()`** - 快速模式：
```
responder → END
```

**路由逻辑：**
- `route_from_coordinator()`：优先检查 `[route: researcher]`/`[route: responder]` 标记，回退到关键词匹配
- `route_from_reviewer()`：检查 `[approved]`/`[通过]` / `[rejected]`/`[不通过]` 标记，支持循环审查

---

### 4. 接口层 (`interface/`)

#### `interface/human_interface.py` (102 行)

人类用户接口，连接用户输入与多 Agent 系统。

**HumanInterface 类：**
- 管理消息历史（通过 `SessionManager`）
- 根据 `fast_mode` 选择对应的 LangGraph 工作流
- `send_message()`：发送消息 → 调用图执行 → 获取响应 → 可选审查
- `_do_review()`：执行审查流程（独立于主图）
- `get_history()`：获取完整消息历史

---

### 5. 核心功能层 (`core/`)

#### `core/config.py` (255 行)

LLM 提供商配置管理，支持 20+ 家厂商。

**配置来源优先级（从高到低）：**
1. `state/model_configs.json` 中的活跃配置（带文件修改时间缓存）
2. 环境变量（`.env` 文件）

**支持的提供商（23 家）：**

| 类型 | 提供商 | base_url | 默认模型 |
|------|--------|----------|----------|
| 国内 | deepseek | api.deepseek.com/v1 | deepseek-chat |
| 国内 | qwen | dashscope.aliyuncs.com | qwen-plus |
| 国内 | minimax | api.minimax.chat/v1 | MiniMax-Text-01 |
| 国内 | doubao | ark.cn-beijing.volces.com | doubao-pro-32k |
| 国内 | glm | open.bigmodel.cn | glm-4-flash |
| 国内 | ernie | qianfan.baidubce.com | ernie-4.0-8k-latest |
| 国内 | hunyuan | hunyuan.tencentcloudapi.com | hunyuan-pro |
| 国内 | spark | spark-api.xf-yun.com | spark-4.0 |
| 国内 | kimi | api.moonshot.cn | kimi-k2.6 |
| 国内 | kimi-code | api.kimi.com/coding/v1 | kimi-for-coding |
| 国内 | yi | api.lingyiwanwu.com | yi-large |
| 国内 | baichuan | api.baichuan-ai.com | baichuan4 |
| 国外 | openai | api.openai.com | gpt-4o-mini |
| 国外 | anthropic | api.anthropic.com | claude-sonnet-4-20250514 |
| 国外 | gemini | generativelanguage.googleapis.com | gemini-2.0-flash |
| 国外 | grok | api.x.ai | grok-3 |
| 国外 | mistral | api.mistral.ai | mistral-large-latest |
| 国外 | cohere | api.cohere.com | command-r-plus |
| 国外 | perplexity | api.perplexity.ai | sonar-pro |
| 国外 | groq | api.groq.com | llama-3.3-70b-versatile |
| 国外 | together | api.together.xyz | meta-llama/Llama-3.3-70B |
| 国外 | azure | (自定义) | gpt-4o |
| 本地 | ollama | localhost:11434/v1 | llama3.2 |

**特殊处理：**
- `kimi-code` 提供商添加特殊 HTTP Header（`User-Agent: claude-code/1.0`）
- Ollama 本地部署不需要 API Key
- 各提供商 API Key 完全隔离（环境变量命名：`LLM_API_KEY_{PROVIDER}`）

#### `core/rag.py` (277 行)

RAG（检索增强生成）模块，基于 OpenAI 兼容 Embedding API 的简单向量检索。

**设计特点：**
- 使用 numpy 进行向量存储和余弦相似度计算，无需额外向量数据库
- 单例模式 `SimpleVectorStore`（线程安全）
- JSON 文件持久化（`data/rag_store.json`）
- 文档分块（默认 500 字符，50 字符重叠，优先在句子边界截断）

**核心函数：**
| 函数 | 说明 |
|------|------|
| `add_document(text, source)` | 添加文档到知识库，自动分块和 Embedding |
| `search_knowledge(query, top_k)` | 检索相关知识并返回格式化文本 |
| `get_embedding(text, model)` | 调用 Embedding API 获取向量 |
| `get_knowledge_stats()` | 获取知识库统计 |
| `clear_knowledge()` | 清空知识库 |

#### `core/export.py` (138 行)

聊天记录导出功能。

**支持格式：**
| 格式 | 函数 | 说明 |
|------|------|------|
| Markdown | `export_markdown()` | 带元信息的 Markdown 文档 |
| JSON | `export_json()` | 结构化 JSON 数据 |
| HTML | `export_html()` | 带样式的 HTML 页面 |
| PDF | `export_pdf()` | 基于 weasyprint（HTML → PDF） |

#### `core/document_parser.py`

文档解析模块，支持 PDF、DOCX、TXT 等格式。
- PDF：使用 PyPDF2 提取文本
- DOCX：使用 python-docx 提取段落
- TXT/MD/PY 等：直接读取文本

---

### 6. 工具层 (`tools/`)

#### `tools/code_executor.py` (235 行)

Python 代码执行沙箱，安全运行 Agent 生成的代码。

**安全机制（双层防护）：**

**第一层 - AST 静态检查：**
```python
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "importlib", "ctypes", "socket",
    "urllib", "http", "ftplib", "smtplib", "pickle", "marshal",
    ...
}
_FORBIDDEN_CALLS_GLOBAL = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "system", "popen", "call", "run", ...
}
```
- 禁止导入危险模块
- 禁止调用危险函数
- 禁止 `getattr` 动态属性访问
- 检查 Import、ImportFrom、Call 等 AST 节点

**第二层 - 子进程隔离：**
- 使用 `multiprocessing.get_context("spawn")` 创建独立进程
- 自定义受限的 `__builtins__`（移除 `open`, `input`, `exec`, `eval` 等）
- 超时保护（默认 30 秒）
- 超时后先 `terminate()`，2 秒后仍存活则 `kill()`

**返回结果：**
```python
{
    "success": bool,
    "stdout": str,
    "stderr": str,
    "error": str | None,
    "traceback": str | None,
    "duration_ms": int,
}
```

#### `tools/search.py` (107 行)

联网搜索工具，为 Researcher Agent 提供搜索能力。

**DuckDuckGo 搜索：**
- 无需 API Key
- 请求 `html.duckduckgo.com/html/`（HTML 版本）
- 多模式正则匹配提取搜索结果
- 提取重定向链接中的实际 URL（`uddg=` 参数）

**网页内容获取：**
- 移除 `<script>`/`<style>` 标签
- 移除所有 HTML 标签
- 压缩空白字符
- 截断到最大字符数（默认 3000）

---

### 7. 状态管理层 (`state/`)

#### `state/types.py`

Agent 状态类型定义（TypedDict）：
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]  # 消息历史（reducer: add）
    active_agent: str | None                          # 当前活跃 Agent
    task_context: dict | None                         # 任务上下文
    human_input_required: bool                        # 是否需要人类输入
    base_model_response: str | None                   # 基础模型响应
    review_result: str | None                         # 审查结果
    awaiting_review: bool                             # 是否等待审查
```

#### `state/manager.py` (192 行)

多会话消息管理器（线程安全 + SQLite 持久化）。

**SessionManager 类：**
- 构造函数自动从数据库加载历史会话
- 线程安全（`_lock = threading.RLock()`）
- 自动管理会话标题（取第一条用户消息前 20 字）
- 消息数量限制：`get_messages_for_model(max_turns=10)` 保留最近 10 轮对话

**数据库表结构：**
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '新对话',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'human' | 'assistant'
    content TEXT NOT NULL,
    agent_name TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

#### `state/persistence.py` (184 行)

SQLite 持久化层。

**设计特点：**
- 线程本地连接缓存（`threading.local()`），每个线程一个 SQLite 连接
- 启用外键约束（`PRAGMA foreign_keys = ON`）
- 启动时自动清理孤立消息（外键未启用时遗留的数据）

#### `state/model_config_manager.py`

模型配置管理器，管理 `state/model_configs.json`：
- 支持多配置存储（每个配置包含 provider, model, apiKey, baseUrl, name）
- 活跃配置切换
- 配置同步到环境变量

#### `state/stats.py`

API 调用统计：
- 记录每次调用的模型、token 用量、费用
- 支持费用估算（按模型预设单价）
- 日统计汇总

#### `state/stop_flag.py`

停止标志管理（按 Socket ID 隔离）：
```python
set_stop(sid)     # 设置停止标志
clear_stop(sid)   # 清除停止标志
is_stopped(sid)   # 检查是否已停止
cleanup_sid(sid)  # 清理 SID 相关资源
```

---

### 8. 提示词层 (`prompts/`)

#### `prompts/coordinator_prompt.py`

Coordinator Agent 的系统提示词，定义其角色和路由规则。

#### `prompts/reviewer_prompt.py`

Reviewer Agent 的提示词生成器：
- `get_reviewer_prompt(language)`：获取审查者系统提示词
- `build_review_prompt(user_msg, response, language)`：构建具体审查任务的提示词
- 支持中英文切换

---

### 9. 入口文件

| 文件 | 用途 |
|------|------|
| `main.py` | 控制台入口，命令行交互 |
| `desktop_app.py` | 桌面应用入口（无控制台窗口 + 自动打开浏览器） |
| `desktop_client.py` | 桌面客户端入口（独立窗口，WebView2 渲染，自动检查安装依赖） |
| `config_gui.py` | Tkinter 配置 GUI |
| `create_icon.py` | 图标生成工具 |

---

## 数据流

### 消息处理流程（Web 模式）

```
1. 用户在浏览器输入消息
        ↓
2. script.js: socket.emit("send_message", { message, sessionId, config })
        ↓
3. app.py: @socketio.on("send_message")
   - 验证消息
   - 获取/创建 SocketState
   - 设置 LLM 配置（set_current_llm_config）
   - 设置流式回调（set_streaming_callback）
        ↓
4. HumanInterface.send_message(content)
   - 添加用户消息到 SessionManager
   - 构建初始状态（AgentState）
   - 调用 graph.ainvoke(initial_state)
        ↓
5. LangGraph 执行工作流
   - coordinator_node → 路由判断
   - researcher_node（可选）→ 调研
   - responder_node → 流式生成回答
   - reviewer_node（可选）→ 审查
        ↓
6. 流式输出
   - agents/factory.py: llm.astream() 逐 token 生成
   - 调用 on_token 回调 → socket.emit("stream_chunk")
        ↓
7. script.js: socket.on("stream_chunk")
   - 追加 token 到当前消息 DOM
   - 滚动到底部
        ↓
8. 生成完成
   - app.py: emit("message_complete")
   - 保存消息到 SQLite
   - 清理回调
```

---

## 安全机制

### 1. 代码执行安全
- AST 静态分析检查危险导入和函数调用
- 子进程隔离执行
- 自定义受限的 `__builtins__`
- 超时保护 + 强制终止

### 2. API Key 安全
- 配置路由仅允许本机访问（`LOCAL_ONLY_PREFIXES`）
- 局域网用户可使用浏览器本地存储的独立 API Key
- 各提供商 API Key 完全隔离

### 3. 文件路径安全
- 导出/下载功能防止目录遍历攻击（`os.path.basename` + 安全字符过滤）

### 4. 代码预览安全
- iframe 使用 `sandbox="allow-scripts allow-same-origin"`

---

## 部署方式

### 1. Web 应用
```bash
python web/app.py
# 或
python main.py
```
- 访问 http://localhost:5000

### 2. 桌面应用（浏览器版）
```bash
python desktop_app.py
```
- 无控制台窗口
- 自动打开系统默认浏览器

### 3. 桌面客户端（独立窗口）
```bash
python desktop_client.py
```
- 使用系统 WebView2 渲染
- 不需要额外安装浏览器
- 自动检查并安装缺失依赖

### 4. 打包为可执行文件
```bash
# 桌面客户端
pyinstaller desktop_client.spec

# 控制台版本
pyinstaller multi_agent_console.spec
```

### 5. Windows 批处理启动
- `启动网页.bat` - 启动 Web 版
- `启动客户端.bat` - 启动桌面客户端

---

## 配置文件

### `.env` / `.env.example`
```
LLM_PROVIDER=minimax          # 默认提供商
LLM_API_KEY=your_api_key      # API Key
```

### `state/model_configs.json`
```json
{
  "configs": [
    {
      "id": "uuid",
      "name": "配置名称",
      "provider": "deepseek",
      "model": "deepseek-chat",
      "apiKey": "sk-...",
      "baseUrl": "https://..."
    }
  ],
  "activeConfigId": "uuid"
}
```

---

## 项目文件结构

```
gpt-agents/
├── .claude/                    # Claude Code 配置
│   └── settings.local.json
├── .env                        # 环境变量（当前配置）
├── .env.example                # 环境变量模板
├── agents/                     # Agent 工厂
│   ├── factory.py              # 创建和管理 AI Agent
│   └── __pycache__/
├── core/                       # 核心功能模块
│   ├── config.py               # LLM 提供商配置管理
│   ├── document_parser.py      # 文档解析（PDF/DOCX/TXT）
│   ├── export.py               # 聊天记录导出
│   └── rag.py                  # RAG 检索增强生成
├── data/                       # 数据存储
│   ├── chat_history.db         # SQLite 聊天历史
│   ├── stats.db                # API 调用统计
│   └── rag_store.json          # RAG 向量存储
├── graph/                      # LangGraph 工作流
│   └── orchestrator.py         # 多 Agent 编排逻辑
├── interface/                  # 用户接口
│   └── human_interface.py      # 人类交互接口
├── prompts/                    # 提示词模板
│   ├── coordinator_prompt.py   # 协调者提示词
│   └── reviewer_prompt.py      # 审查者提示词
├── state/                      # 状态管理
│   ├── manager.py              # 会话管理器
│   ├── model_configs.json      # 模型配置文件
│   ├── model_config_manager.py # 配置管理
│   ├── persistence.py          # SQLite 持久化
│   ├── stats.py                # 统计记录
│   ├── stop_flag.py            # 停止标志（按 Socket 隔离）
│   └── types.py                # 类型定义
├── tools/                      # 工具模块
│   ├── code_executor.py        # Python 代码执行沙箱
│   └── search.py               # 联网搜索（DuckDuckGo）
├── web/                        # Web 应用
│   ├── app.py                  # Flask 主应用
│   ├── static/
│   │   ├── script.js           # 前端 JS
│   │   └── style.css           # 样式表
│   └── templates/
│       ├── index.html          # 聊天界面
│       └── config.html         # 配置页面
├── main.py                     # 控制台入口
├── desktop_app.py              # 桌面应用入口（浏览器版）
├── desktop_client.py           # 桌面客户端（WebView2 版）
├── desktop_client.spec         # PyInstaller 打包配置
├── multi_agent_chat.spec       # 另一打包配置
├── multi_agent_console.spec    # 控制台打包配置
├── config_gui.py               # Tkinter 配置 GUI
├── create_icon.py              # 图标生成工具
├── requirements.txt            # Python 依赖
├── icon.ico                    # 应用图标
├── 启动客户端.bat              # 启动桌面客户端
└── 启动网页.bat                # 启动 Web 版
```

---

## 关键技术决策

### 为什么选择 LangGraph？
- 声明式工作流定义，代码可读性强
- 内置状态管理（StateGraph + TypedDict）
- 支持条件边和循环（审查不通过时返回修改）
- 与 LangChain 生态无缝集成

### 为什么选择 Flask + SocketIO？
- 轻量级，适合中小型项目
- SocketIO 提供可靠的实时双向通信（自动降级到轮询）
- threading 异步模式兼容性好

### 为什么使用 numpy 而非专用向量数据库？
- 项目定位轻量级，避免额外依赖
- 向量数量不大时（<10000），numpy 性能足够
- JSON 持久化简单直接

### 为什么前端不使用框架？
- 项目复杂度可控，原生 JS 足够
- 减少构建步骤和依赖
- 直接使用 CDN 资源，部署简单

---

## 扩展建议

1. **向量数据库升级**：当知识库规模扩大时，可迁移到 Milvus/Pinecone/Weaviate
2. **缓存层**：添加 Redis 缓存 LLM 响应，减少 API 调用成本
3. **用户认证**：添加多用户支持和权限管理
4. **插件系统**：设计插件接口，支持自定义 Tool
5. **模型路由**：根据问题复杂度自动选择模型（简单问题用轻量模型）
6. **对话摘要**：长对话自动摘要，减少 token 消耗

---

*文档生成时间：2026-04-24*
*项目版本：基于当前代码库状态*
