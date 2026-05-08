# 凯伦 — 多 Agent AI 聊天系统

一个功能完善的多 Agent AI 聊天系统，采用 LangGraph 实现智能体协作编排，支持 20+ 家国内外大语言模型，提供 Web、桌面客户端、控制台三种使用方式。

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🤖 多 Agent 协作 | Coordinator → Researcher → Responder → Reviewer 智能协作流程 |
| 🌐 多模型支持 | 支持 20+ 家国内外 LLM 厂商（OpenAI 兼容 API） |
| 📚 RAG 知识库 | 基于 Embedding 的文档检索（支持 PDF / Word / 文本） |
| 🔍 联网搜索 | 多源搜索:中文 query 优先 360,英文优先 DuckDuckGo,Bing 兜底 |
| 🐍 代码执行 | Python 代码沙箱（AST 安全检查） |
| 💬 多会话管理 | 多会话切换、SQLite 持久化存储 |
| 📤 记录导出 | Markdown / JSON / HTML / PDF 格式 |
| ⚡ 流式输出 | Token 级实时响应 |
| 📊 用量统计 | Token 用量统计与费用估算 |
| 🌙 主题切换 | 亮色 / 暗色 / 跟随系统 |
| 🧠 自适应记忆 | 基于语义的热/冷分层记忆系统，跨会话长期记忆 |
| 🔌 插件系统 | 可扩展插件机制 |
| 🔗 MCP 支持 | Model Context Protocol 服务器接入 |

---

## 🚀 快速开始

### 环境要求

- Python 3.13+
- Windows / macOS / Linux

### 安装

```bash
git clone https://github.com/gyp0927/guodong-ai.git
cd guodong-ai
pip install -r requirements.txt
```

### 配置 API Key

复制 `.env.example` 为 `.env`，填写你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# 选择默认提供商和模型
PROVIDER=deepseek
MODEL=deepseek-chat
API_KEY=your-api-key-here
BASE_URL=https://api.deepseek.com/v1
```

#### 记忆系统配置（可选）

系统默认启用本地自适应记忆，使用 sentence-transformers 进行本地 Embedding（无需额外 API Key）：

```env
EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
METADATA_DB_URL=sqlite+aiosqlite:///./data/adaptive_memory.db
```

支持提供商：OpenAI、Anthropic、Google Gemini、DeepSeek、阿里通义千问、月之暗面 Kimi、智谱 GLM、xAI Grok、Mistral、Groq、Azure OpenAI、Ollama（本地）等。

#### 安全相关 env(可选)

```env
ENABLE_AUTH=false                                  # 设 true 后 RAG/execute/plugin 端点要 X-API-Key
TRUST_PROXY=false                                  # 反代后置时设 true,否则任何客户端可伪造 127.0.0.1 绕 LOCAL_ONLY
CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
MAX_UPLOAD_BYTES=52428800                          # 50MB 上传上限
MCP_ALLOWED_COMMANDS=                              # 留空=允许任意,生产建议 "npx,uvx,python"
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

## 🏗️ 系统架构

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
├─ Reviewer    审查者      ├─ 联网搜索
└─ Planner     规划器      ├─ 代码执行
                            └─ 聊天记录导出
       │
       ▼
   LLM 提供商 (20+ 家)
```

### 多 Agent 协作流程

**协调模式（默认）**

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

**快速模式**

```
用户输入 → Responder → 直接输出
```

适合简单对话，跳过路由和调研环节，响应更快。

---

## 📂 项目结构

```
karen/
├── agents/                  # Agent 工厂与定义
│   └── factory.py
├── core/                    # 核心功能模块
│   ├── config.py           # 配置管理
│   ├── memory_client.py    # 自适应记忆系统客户端
│   ├── rag.py              # RAG 知识库
│   ├── export.py           # 聊天记录导出
│   ├── cache.py            # 响应缓存
│   ├── plugin_system.py    # 插件系统
│   ├── mcp_manager.py      # MCP 服务器管理
│   ├── model_router.py     # 模型路由
│   └── vector_store/       # 向量存储后端
├── graph/                   # LangGraph 编排
│   └── orchestrator.py
├── interface/               # 用户接口
│   └── human_interface.py
├── prompts/                 # 系统提示词
│   ├── coordinator_prompt.py
│   └── reviewer_prompt.py
├── state/                   # 状态管理
│   ├── manager.py          # 会话管理
│   ├── persistence.py      # 数据持久化
│   └── model_config_manager.py
├── tools/                   # 工具模块
│   ├── search.py           # 联网搜索
│   └── code_executor.py    # 代码执行
├── web/                     # Web 应用
│   ├── app.py              # Flask 后端
│   ├── templates/          # HTML 模板
│   └── static/             # CSS / JS
├── main.py                  # 控制台入口
├── desktop_app.py           # 桌面应用入口
├── desktop_client.py        # 桌面客户端（WebView）
├── config_gui.py            # 配置 GUI
└── requirements.txt
```

---

## 📖 使用指南

### 多模型配置

访问 `http://127.0.0.1:5000/config` 进入配置页面，可添加多个模型配置并随时切换。

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

记忆数据存储在本地：
- `data/adaptive_memory.db` — 记忆元数据（SQLite）
- `data/qdrant_storage/` — 向量数据库
- `data/memories/` — 原始记忆文本

### 联网搜索

聊天时根据意图自动触发(查询含"什么是 / 为什么 / 介绍 / 最新 / 价格 / 天气"等事实/时效关键词时启用)。中文查询优先走 360 搜索,英文优先 DuckDuckGo,Bing 作为 fallback。无需手动开关。

### 代码执行

消息中包含 Python 代码块时，系统会自动检测并提供运行按钮，代码在 AST 沙箱中安全执行。

### 控制台命令

```
/review    开启/关闭回答审查
/fast      切换快速模式
/clear     清空当前会话
/history   查看历史消息
exit       退出程序
```

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.13 | 后端核心 |
| Flask + SocketIO | Web 服务与实时通信 |
| LangGraph | 多 Agent 工作流编排 |
| LangChain | LLM 调用封装 |
| SQLite | 数据持久化 |
| Qdrant | 向量数据库（本地模式） |
| sentence-transformers | 本地 Embedding 模型 |
| WebView2 | 桌面客户端渲染 |

---

## 🤝 贡献

欢迎 Issue 和 PR！

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证

[MIT](LICENSE)

---

<p align="center">Made by 凯伦 Team</p>
