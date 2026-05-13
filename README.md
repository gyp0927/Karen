# 凯伦 — 多 Agent AI 聊天系统

一个多 Agent AI 聊天系统，采用 LangGraph 实现智能体协作编排，支持 24 家国内外大语言模型，提供 Web、桌面客户端、控制台三种使用方式。

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 多 Agent 协作 | Coordinator → Researcher → Responder 智能协作流程 |
| 多模型支持 | 支持 24 家国内外 LLM 厂商（OpenAI 兼容 API） |
| RAG 知识库 | 基于 Embedding 的文档检索（PDF / Word / 文本） |
| 联网搜索 | 多源搜索：中文优先 360，英文优先 DuckDuckGo，Bing 兜底 |
| 代码执行 | Python 代码沙箱（AST 安全检查，默认关闭） |
| 多会话管理 | 多会话切换、SQLite 持久化存储 |
| 记录导出 | Markdown / JSON / HTML / PDF 格式 |
| 流式输出 | Token 级实时响应 |
| 用量统计 | Token 用量统计与费用估算 |
| 自适应记忆 | 基于语义的热/冷分层记忆系统，跨会话长期记忆 |
| 插件系统 | 可扩展插件机制 |
| MCP 支持 | Model Context Protocol 服务器接入 |

---

## 快速开始

### 环境要求

- Python 3.13+
- Windows / macOS / Linux

### 安装

```bash
git clone https://github.com/gyp0927/Karen.git
cd Karen
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`，填写 API Key：

```env
LLM_PROVIDER=minimax
LLM_API_KEY=your-api-key-here
```

支持的提供商：`deepseek`、`qwen`、`minimax`、`doubao`、`glm`、`ernie`、`hunyuan`、`spark`、`kimi`、`siliconflow`、`kimi-code`、`yi`、`baichuan`、`openai`、`anthropic`、`gemini`、`grok`、`mistral`、`cohere`、`perplexity`、`groq`、`together`、`azure`、`ollama`

也可通过 Web 配置页面 `/config` 添加和管理多个模型配置，随时切换。

#### 记忆系统配置（可选）

```env
EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
METADATA_DB_URL=sqlite+aiosqlite:///./data/adaptive_memory.db
```

#### 安全配置（可选）

```env
ENABLE_AUTH=false              # 设为 true 后 RAG/execute/plugin 端点需要 X-API-Key
TRUST_PROXY=false              # 反向代理后置时设为 true
CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
MAX_UPLOAD_BYTES=52428800      # 50MB 上传上限
MCP_ALLOWED_COMMANDS=npx,uvx   # 留空=允许任意，生产环境建议限定
ENABLE_CODE_EXECUTION=false    # 代码执行开关
```

### 启动

**方式一：Web 界面（推荐）**

```bash
python desktop_app.py
```

自动打开浏览器访问 `http://127.0.0.1:5000/`，局域网内其他设备可通过 `http://你的IP:5000` 访问。

**方式二：桌面客户端**

```bash
python desktop_client.py
```

使用 WebView2 渲染，体验更贴近原生应用。

**方式三：控制台**

```bash
python main.py
```

纯命令行交互，适合无 GUI 环境。

---

## 系统架构

```
用户交互层
├─ Web 浏览器  (Flask + SocketIO)
├─ 桌面客户端  (WebView2)
├─ 桌面应用    (浏览器)
└─ 控制台      (命令行)
       │
       ▼
   Flask 后端
       │
   ┌──┴──┐
   ▼     ▼
Agent 编排 (LangGraph)    核心功能
├─ Coordinator 调度器      ├─ 配置管理
├─ Researcher  研究员      ├─ RAG 知识库
├─ Responder   响应器      ├─ 自适应记忆
└─ Reviewer    审查者      ├─ 联网搜索
                            ├─ 代码执行
                            └─ 聊天记录导出
       │
       ▼
   LLM 提供商 (24 家)
```

### Agent 协作流程

**快速模式（默认）**

```
用户输入 → Responder → 直接输出
```

适合日常对话，响应最快。Responder 内部会根据需要自动触发搜索或工具调用。

**协调模式**

```
用户输入 → Coordinator（分析路由）
              │
    ┌─────────┴─────────┐
    ▼                   ▼
需要研究            直接回答
    │                   │
Researcher ──────→ Responder
                        │
                    Reviewer（质检）
                        │
                    最终输出
```

适合复杂问题，经过完整的多 Agent 协作流程。

---

## 项目结构

```
.
├── agents/                  # Agent 节点定义
│   ├── llm.py              # LLM 基础设施（HTTP 客户端、配置隔离）
│   ├── nodes.py            # Agent 节点（Coordinator/Researcher/Responder/Reviewer）
│   ├── prompts.py          # 系统提示词
│   ├── search.py           # 搜索 Agent
│   └── tools.py            # 工具调用 Agent
├── cognition/               # 认知模块
│   ├── engine.py           # 认知引擎
│   ├── human_mind.py       # 类人思维
│   ├── tool_engine.py      # 工具执行引擎
│   └── types.py            # 认知类型定义
├── core/                    # 核心功能模块
│   ├── config.py           # 配置管理（24 家提供商配置）
│   ├── auth.py             # 认证与权限
│   ├── cache.py            # 响应缓存（SQLite）
│   ├── rag.py              # RAG 知识库
│   ├── export.py           # 聊天记录导出
│   ├── plugin_system.py    # 插件系统
│   ├── mcp_manager.py      # MCP 服务器管理
│   ├── model_router.py     # 模型路由
│   └── memory_client.py    # 自适应记忆客户端
├── graph/                   # LangGraph 编排
│   └── orchestrator.py     # 多 Agent 图定义
├── hot_and_cold_memory/     # 热/冷分层记忆系统
│   ├── ingestion/          # 记忆摄取与嵌入
│   ├── tiers/              # 热层/冷层/压缩引擎
│   ├── migration/          # 跨层迁移引擎
│   └── storage/            # 元数据存储与向量存储
├── interface/               # 用户接口
│   └── human_interface.py  # 统一交互接口
├── state/                   # 状态管理
│   ├── manager.py          # 会话管理
│   ├── persistence.py      # 数据持久化
│   ├── model_config_manager.py  # 模型配置管理
│   └── stats.py            # 用量统计
├── tools/                   # 工具模块
│   ├── search.py           # 联网搜索实现
│   └── code_executor.py    # 代码执行沙箱
├── web/                     # Web 应用
│   ├── app.py              # Flask 后端 + SocketIO
│   ├── api.py              # REST API
│   ├── templates/          # HTML 模板
│   └── static/             # CSS / JS / 图片
├── plugins/                 # 插件目录
├── main.py                  # 控制台入口
├── desktop_app.py           # 桌面应用入口
├── desktop_client.py        # 桌面客户端（WebView）
├── test_all.py              # 测试套件
└── requirements.txt
```

---

## 使用指南

### 多模型配置

访问 `http://127.0.0.1:5000/config` 进入配置页面，可添加多个模型配置并随时切换。配置保存在 `state/model_configs.json`。

### 知识库

1. 访问 `/knowledge` 页面
2. 上传 PDF、Word 或文本文件
3. 聊天时勾选"知识库"开关即可使用 RAG 检索

### 自适应记忆

系统内置热/冷分层记忆，自动完成以下流程：

- **记忆检索**：发送消息时自动检索相关历史记忆，注入 LLM 上下文
- **记忆保存**：对话结束后自动保存关键信息到长期记忆
- **语义搜索**：基于向量相似度检索，无关关键词也能找到相关记忆
- **跨会话**：切换会话后仍可检索之前对话中的重要信息
- **分层存储**：高频记忆保留在热层（完整内容），低频记忆迁移到冷层（压缩摘要），节省存储成本

记忆数据存储在本地：
- `data/adaptive_memory.db` — 记忆元数据（SQLite）
- `data/qdrant_storage/` — 向量数据库
- `data/memories/` — 原始记忆文本

### 联网搜索

聊天时根据意图自动触发（查询含"什么是 / 为什么 / 介绍 / 最新 / 价格 / 天气"等事实/时效关键词时启用）。中文查询优先走 360 搜索，英文优先 DuckDuckGo，Bing 作为 fallback。无需手动开关。

### 代码执行

消息中包含 Python 代码块时，系统会自动检测并提供运行按钮。代码在 AST 沙箱中安全执行。需在 `.env` 中设置 `ENABLE_CODE_EXECUTION=true` 开启。

### 控制台命令

```
/review    开启/关闭回答审查
/fast      切换快速/协调模式
/clear     清空当前会话
/history   查看历史消息
exit       退出程序
```

---

## 性能优化

| 优化项 | 说明 |
|--------|------|
| 连接池 | HTTP 连接池（max_connections=100, keepalive=20） |
| 连接预热 | 启动时预建立 LLM 连接，减少首 token 延迟 |
| 并发控制 | LLM 流式请求信号量（max 8），防止限流 |
| Token 批处理 | 服务端凑齐 12 字符再 emit，平衡实时性与网络开销 |
| 响应缓存 | SQLite 缓存，相同请求直接返回 |
| 问候语模板 | 常见问候语走模板，不走 LLM |

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.13 | 后端核心 |
| Flask + SocketIO | Web 服务与实时通信 |
| LangGraph | 多 Agent 工作流编排 |
| LangChain | LLM 调用封装 |
| SQLite | 数据持久化与缓存 |
| Qdrant | 向量数据库（本地模式） |
| sentence-transformers | 本地 Embedding 模型 |
| WebView2 | 桌面客户端渲染 |

---

## 许可证

[MIT](LICENSE)

---

<p align="center">Made with ❤️ by 凯伦 Team</p>
