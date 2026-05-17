# Phase 3 执行清单（交给 Claude 实现）

> 目标：让 karen-ai 拥有技能系统和 MCP 协议支持，实现"关键词触发 → 加载技能 → 注入上下文 → LLM 执行"的闭环。
> 核心产出：skills/ 目录 + core/skills/ 模块 + MCP 工具发现 + 3 个默认技能

---

## 页 1：技能系统设计（核心，2-3 天）

### Task 1.1：设计 Skill 格式规范

**目的：定义技能文件的标准格式，与 Hermes skill 系统兼容**

创建 `SKILL_SPEC.md`，定义如下格式：

```markdown
# 技能文件格式规范

## 文件位置
`skills/<category>/<skill-name>/SKILL.md`

例子：`skills/software-development/code-review/SKILL.md`

## 文件结构
```yaml
---
name: code-review
description: 代码审查，检查安全、性能、风格问题
triggers:
  - 代码审查
  - 审查代码
  - code review
  - review this code
category: software-development
model: powerful  # 使用哪档模型：light/default/powerful
---

# 技能内容

## 角色定义
你是一个经验丰富的代码审查员...

## 工作流程
1. 先看整体架构
2. 再查具体实现
3. 最后给出修复建议

## 示例
### 输入
```python
def add(a, b):
    return a + b
```

### 输出
- 建议添加类型注解
- 建议添加文档字符串
```
```

**Skill 字段规范：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | str | 是 | 技能 ID，唯一，小写，用下划线连接 |
| description | str | 是 | 人类可读的说明 |
| triggers | list[str] | 是 | 触发关键词列表，匹配用户输入 |
| category | str | 是 | 分类目录名，小写，用下划线连接 |
| model | str | 否 | 默认使用的模型档位，可覆盖 socket 级配置 |
| temperature | float | 否 | 覆盖默认 temperature |
| max_tokens | int | 否 | 覆盖默认 max_tokens |
| tools | list[str] | 否 | 该技能可调用的工具名列表 |
| requires_mcp | list[str] | 否 | 需要的 MCP 服务器名 |

### Task 1.2：创建目录结构

**目的：建立技能存储的物理结构**

```
skills/
├── README.md              # 技能系统使用说明
├── SKILL_SPEC.md          # 技能格式规范（Task 1.1 产出）
└── software-development/
    ├── code-review/
    │   └── SKILL.md
    ├── debug/
    │   └── SKILL.md
    └── refactor/
        └── SKILL.md
```

**要求：**
1. 在项目根目录创建 `skills/` 目录
2. 创建 `skills/README.md`，说明技能是什么、怎么使用、怎么创建新技能
3. 创建三个分类目录

### Task 1.3：实现 SkillLoader

**目的：解析 skill 文件，提供缓存和查询**

创建 `core/skills/loader.py`：

```python
"""Skill loader: parse, validate, cache skill definitions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: list[str]
    category: str
    model: str = "default"
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[str] = field(default_factory=list)
    requires_mcp: list[str] = field(default_factory=list)
    body: str = ""  # markdown body after frontmatter


class SkillLoader:
    def __init__(self, skills_dir: Path | str = SKILLS_DIR) -> None:
        self.skills_dir = Path(skills_dir)
        self._cache: dict[str, Skill] = {}  # name -> Skill
        self._trigger_index: dict[str, str] = {}  # trigger -> skill_name
        self._load_all()

    def _load_all(self) -> None:
        """Load all skills from disk into memory."""
        ...

    def _parse_skill(self, path: Path) -> Skill | None:
        """Parse a single SKILL.md file.
        
        Format: YAML frontmatter between --- markers, then markdown body.
        """
        ...

    def get(self, name: str) -> Skill | None:
        """Get skill by name."""
        ...

    def match(self, text: str) -> Skill | None:
        """Match user input against trigger keywords.
        
        Priority:
        1. Exact match (case-insensitive)
        2. Contains match (case-insensitive)
        3. Longest trigger wins tie-breaking
        """
        ...

    def list_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        ...

    def reload(self) -> None:
        """Hot reload all skills from disk."""
        ...
```

**实现细节：**
1. `_parse_skill`：读取文件，用 regex `^---\n(.*?)\n---\n(.*)$` 分离 frontmatter 和 body
2. frontmatter 用 `yaml.safe_load` 解析
3. 验证必填字段，缺失则打 log warning 并跳过
4. `缓存`：用 `@functools.lru_cache` 包裉 `get`，避免重复解析
5. `触发器匹配`：先小写转换，先试完全匹配，再试 contains，有重叠时用最长 trigger 优先

### Task 1.4：实现关键词触发器

**目的：在意图分类之后、模型路由之前插入 skill 触发**

修改 `core/intent_classifier.py`：

1. 在 `classify_intent()` 之后（或之前，你决定），添加 `match_skill()` 调用
2. 如果匹配到 skill，返回的 intent 改为 `"skill:<skill_name>"`
3. 把匹配到的 skill 信息写入 `AgentState`

修改 `core/chat_state.py`（或相应的 state 定义）：

```python
@dataclass
class AgentState:
    # ... existing fields ...
    active_skill: str | None = None  # 当前激活的技能名
    skill_context: dict[str, Any] = field(default_factory=dict)  # 技能上下文
```

**触发器逻辑流程：**

```
用户输入
  → classify_intent()  [intent]
  → match_skill(text)   [skill | None]
  → route_model()       [根据 skill.model 覆盖默认路由]
  → execute_graph()     [Responder 接收 skill 上下文]
```

**实现要求：**
1. 在 `core/intent_classifier.py` 添加 `SkillTrigger`类（或函数）
2. `SkillTrigger`用 `SkillLoader` 实例化时传入 `skills_dir`
3. 默认 `skills_dir` 从环境变量 `KAREN_SKILLS_DIR` 读取，否则用默认 `skills/`
4. 如果触发了 skill，在 socket 级配置里记录 `active_skill`，方便后续对话保持上下文

---

## 页 2：MCP 协议集成（2-3 天）

### Task 2.1：MCP 配置管理

**目的：管理 MCP 服务器配置，支持本地和远程服务器**

创建 `core/mcp/config.py`：

```python
"""MCP server configuration management."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "mcp_config.yaml"


@dataclass
class MCPServerConfig:
    name: str
    command: str | None = None  # 本地命令启动
    args: list[str] = None  # 启动参数
    url: str | None = None  # SSE/WebSocket 远程地址
    env: dict[str, str] = None  # 环境变量
    enabled: bool = True


class MCPConfigManager:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.servers: dict[str, MCPServerConfig] = {}
        self._load()

    def _load(self) -> None:
        """Load from mcp_config.yaml or create default."""
        ...

    def get(self, name: str) -> MCPServerConfig | None:
        ...

    def list_enabled(self) -> list[MCPServerConfig]:
        ...
```

**默认配置文件 `mcp_config.yaml`：**

```yaml
servers:
  # 示例：本地文件系统 MCP 服务器
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    enabled: false  # 默认关闭，需要用户手动开启

  # 示例：SQLite 查询 MCP 服务器
  sqlite:
    command: uvx
    args: ["mcp-server-sqlite", "--db-path", "data/memory.db"]
    enabled: false
```

**实现要求：**
1. 支持通过命令行启动本地 MCP 服务器（stdio 传输）
2. 支持通过 SSE/WebSocket 连接远程 MCP 服务器
3. 如果配置文件不存在，自动创建带示例的默认配置
4. 支持通过环境变量覆盖：`KAREN_MCP_CONFIG_PATH`

### Task 2.2：MCP 客户端实现

**目的：与 MCP 服务器建立连接，支持工具发现和调用**

创建 `core/mcp/client.py`：

```python
"""MCP client: connect to servers, discover tools, invoke tools."""
from __future__ import annotations

import asyncio
from typing import Any

# 使用现有的 mcp 库
# 项目已有依赖：mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCPConfigManager, MCPServerConfig


class MCPClient:
    def __init__(self, config_manager: MCPConfigManager | None = None) -> None:
        self.config = config_manager or MCPConfigManager()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, dict] = {}  # tool_name -> {server, schema}

    async def connect_all(self) -> None:
        """Connect to all enabled MCP servers."""
        ...

    async def _connect_stdio(self, server: MCPServerConfig) -> ClientSession:
        """Connect to a stdio-based MCP server."""
        params = StdioServerParameters(
            command=server.command,
            args=server.args or [],
            env={**os.environ, **(server.env or {})},
        )
        ...

    async def discover_tools(self) -> list[dict]:
        """List all available tools from all connected servers.
        
        Returns list of tool definitions compatible with LangChain Tool.
        """
        ...

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool by name across all connected servers."""
        ...

    async def close(self) -> None:
        """Close all connections."""
        ...

    @property
    def available_tools(self) -> list[dict]:
        """Return cached tool list (call discover_tools first)."""
        return list(self._tools.values())
```

**实现要求：**
1. `连接所有启用服务器`：并行连接，单个失败不影响其他
2. `工具发现`：从每个连接的 session 获取 tools/list，缓存到 `_tools`
3. `工具命名空间`：为避免重名，工具名格式 `<server_name>/<tool_name>`，如 `filesystem/read_file`
4. `异常处理`：连接失败时打 warning 不抛出，保证系统仍然可用
5. `生命周期`：在 `web/app.py` 启动时初始化，在关闭时清理

### Task 2.3：将 MCP 工具包装成 LangChain Tool

**目的：让 LangGraph 能直接调用 MCP 工具**

创建 `core/mcp/tools.py`：

```python
"""Convert MCP tools to LangChain-compatible tools."""
from __future__ import annotations

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from .client import MCPClient


def mcp_tools_to_langchain(mcp_client: MCPClient) -> list[BaseTool]:
    """Convert all discovered MCP tools to LangChain BaseTool instances.
    
    Each tool becomes a callable that forwards to mcp_client.invoke_tool().
    """
    ...


class _MCPToolWrapper(BaseTool):
    """Dynamic tool wrapper for MCP tools."""
    
    def __init__(self, tool_name: str, description: str, schema: dict, client: MCPClient):
        # 动态创建 Pydantic 模型作为 args_schema
        ...

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs) -> str:
        return await self.client.invoke_tool(self.tool_name, kwargs)
```

**实现要求：**
1. `动态 Pydantic 模型`：从 MCP 的 JSONSchema 生成 Pydantic BaseModel，作为 tool 的 `args_schema`
2. `工具描述`：从 MCP 工具定义中提取 description，并添加来源标记如 `[via MCP: filesystem]`
3. `波动`支持：如果工具的参数复杂，至少支持 string/number/boolean/object 基础类型
4. `错误处理`：工具调用失败时返回友好的错误信息，不让 LangGraph 崩溃

### Task 2.4：把 MCP 工具注入 Responder

**目的：让 Responder 自动使用可用的 MCP 工具**

修改 `agents/nodes.py` 或 `agents/responder.py`：

1. 在 Responder 节点初始化时，检查 `state.active_skill`
2. 如果 skill 定义了 `tools` 列表，只注入指定工具
3. 如果 skill 定义了 `requires_mcp`，确保对应 MCP 服务器已连接
4. 如果没有激活 skill，使用默认工具集（保留现有工具）

**插入点示例（在 Responder 的 `execute` 方法中）：**

```python
async def execute(self, state: AgentState) -> AgentState:
    # ... 现有逻辑 ...
    
    # 获取当前工具集
    tools = self.default_tools.copy()
    
    if state.active_skill:
        skill = self.skill_loader.get(state.active_skill)
        if skill and skill.tools:
            # 只添加 skill 指定的工具
            tools = [t for t in tools if t.name in skill.tools]
        if skill and skill.requires_mcp:
            # 添加 MCP 工具
            mcp_tools = self.mcp_client.get_tools_by_servers(skill.requires_mcp)
            tools.extend(mcp_tools)
    
    # ... 调用 LLM with tools ...
```

---

## 页 3：三个核心技能实现（1 天）

### Task 3.1：code-review 技能

**目的：实现一个可用的代码审查技能**

创建 `skills/software-development/code-review/SKILL.md`：

```yaml
---
name: code-review
description: 深度代码审查，检查安全漏洞、性能瓶颈、设计命命题、风格一致性
triggers:
  - 代码审查
  - 审查代码
  - code review
  - review this code
  - 帮我看看这段代码
category: software-development
model: powerful
temperature: 0.2
tools:
  - code_executor  # 如果 ENABLE_CODE_EXECUTION=true
tools:
  - code_executor
---

# 代码审查员

## 角色定义
你是一个经验丰富的工程师，专注于代码质量。你的审查是建设性的，既批评问题也提供解决方案。

## 审查维度
1. **安全性**：SQL 注入、XSS、命令注入、敏感数据泄露、硬编码密码
2. **性能**：N+1 查询、内存泄漏、无限递归、重复计算
3. **可维护性**：函数过长、循宁复杂度、魔法数字、缺少注释
4. **正确性**：边界条件、错误处理、资源释放
5. **风格**：命名规范、类型注解、PEP8 符合度

## 输出格式
对每个问题：
- **严重级别**：P0（严重）/ P1（中等）/ P2（轻微）/ P3（建议）
- **位置**：文件名 + 行号
- **问题描述**：一句话说清楚
- **修复建议**：带代码示例

## 示例
### 输入
```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
```

### 输出
- **P0 - 安全**：sql 注入
  - 位置：db.py:2
  - 问题：直接拼接 SQL 导致注入漏洞
  - 修复：使用参数化查询 `db.execute("SELECT * FROM users WHERE id = ?", (user_id,))`
```

### Task 3.2：debug 技能

**目的：帮用户分析错误日志并定位问题**

创建 `skills/software-development/debug/SKILL.md`：

```yaml
---
name: debug
description: 分析错误日志、定位问题根因、提供修复方案
triggers:
  - debug
  - 调试
  - 排错
  - 为什么报错
  - 错误分析
  - 日志分析
category: software-development
model: powerful
temperature: 0.1
tools:
  - code_executor
---

# 调试专家

## 角色定义
你是一个调试专家，擅长从错误信息中快速定位问题。你不仅修复表象，还要找到根因。

## 工作流程
1. 分析错误堆栈：找到最原始的异常点
2. 分析代码上下文：理解调用链
3. 分析数据流：跟踪变量值如何变化
4. 提供修复方案：不只是补丁，要解决根本问题

## 示例
### 输入
```
FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
  File "app.py", line 45, in load_config
    with open(path) as f:
```

### 输出
**根因**：程序在当前工作目录查找 config.yaml，但运行时工作目录不是项目根目录。

**修复建议**：
```python
import os

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")
```
```

### Task 3.3：refactor 技能

**目的：帮用户优化代码结构，提升可读性**

创建 `skills/software-development/refactor/SKILL.md`：

```yaml
---
name: refactor
description: 代码重构建议，提升可读性、可测试性、可维护性
triggers:
  - refactor
  - 重构
  - 优化代码
  - 这段代码太乱了
  - 怎么改良一下
category: software-development
model: powerful
temperature: 0.3
tools:
  - code_executor
---

# 代码重构顾问

## 角色定义
你是一个代码重构顾问，专注于提升代码质量而不破坏功能。你的建议是步骤化的、可执行的。

## 重构原则
1. **功能不变**：重构前后行为一致
2. **小步快跑**：每次重构只做一件事
3. **测试保护**：重构前先有测试覆盖
4. **命名先行**：好名字比好注释更重要

## 常见问题与方案
| 问题 | 识别信号 | 重构方案 |
|------|---------|---------|
| 神方法 | 函数过长、层级过深 | 提取函数、降低嵌套 |
| 重复代码 | 相似逻辑多处出现 | 提取公共函数/类 |
| 巨物方法 | 类超过 300 行 | 拆分为小类 |
| 过度设计 | 接口抽象层数 > 3 | 合并或移除中间层 |

## 示例
### 输入
```python
class UserManager:
    def create_user(self, name, email, phone, address, role):
        # 验证
        if not name or len(name) < 2:
            raise ValueError()
        if not email or "@" not in email:
            raise ValueError()
        # ... 100 行验证 ...
        # 写数据库
        db.execute("INSERT ...")
        # 发邮件
        send_email(...)
        # 记录日志
        logger.info(...)
```

### 输出
**问题**：单一职责过多（验证+写库+发邮件+记日志）

**重构建议**：
1. 拆分验证逻辑到 `UserValidator`
2. 拆分邮件逻辑到 `EmailService`
3. 使用 `create_user` 作为调度方法
```python
class UserManager:
    def __init__(self, validator, email_service, db, logger):
        self.validator = validator
        self.email_service = email_service
        self.db = db
        self.logger = logger

    def create_user(self, name, email, phone, address, role):
        user = self.validator.validate(name, email, phone, address, role)
        self.db.save(user)
        self.email_service.send_welcome(user)
        self.logger.info(f"Created user: {user.id}")
```
```

---

## 页 4：架构集成（2 天）

### Task 4.1：扩展 AgentState 支持技能上下文

**目的：让技能状态在多轮对话中保持**

修改 `core/chat_state.py`（或相应 state 定义）：

```python
@dataclass
class AgentState:
    # ... existing fields ...
    
    # 技能相关
    active_skill: str | None = None
    skill_context: dict[str, Any] = field(default_factory=dict)
    skill_history: list[dict] = field(default_factory=list)  # 技能对话历史
    
    # MCP 相关
    available_mcp_tools: list[str] = field(default_factory=list)  # 当前可用的 MCP 工具名
```

**要求：**
1. `active_skill` 在对话过程中保持，不因为用户下一句话没触发关键词就清空
2. `技能退出`：定义退出关键词（如"退出审查"、"谢谢"、"完成了"），检测到后清空 `active_skill`
3. `技能历史`：记录本轮对话中 skill 的输出，方便引用

### Task 4.2：让 Responder 感知技能

**目的：根据激活的技能改变 Responder 的行为**

修改 `agents/nodes.py` 中的 Responder 节点：

```python
class ResponderNode:
    async def execute(self, state: AgentState) -> AgentState:
        # 1. 获取激活技能
        skill = None
        if state.active_skill:
            skill = self.skill_loader.get(state.active_skill)
        
        # 2. 构建系统提示词
        system_prompt = self._build_system_prompt(skill)
        
        # 3. 获取工具集（skill 指定的 + MCP 的 + 默认的）
        tools = self._get_tools_for_skill(skill)
        
        # 4. 获取模型参数（skill 可覆盖默认）
        model_name = skill.model if skill and skill.model else self.default_model
        temperature = skill.temperature if skill and skill.temperature is not None else self.default_temperature
        
        # 5. 调用 LLM
        ...

    def _build_system_prompt(self, skill: Skill | None) -> str:
        """Combine base system prompt with skill body."""
        base = self.base_system_prompt
        if skill:
            return f"{base}\n\n## 当前激活技能: {skill.name}\n{skill.description}\n\n{skill.body}"
        return base
    
    def _get_tools_for_skill(self, skill: Skill | None) -> list[BaseTool]:
        """Get tools filtered by skill requirements."""
        tools = self.default_tools.copy()
        
        if skill:
            # 只保留 skill 指定的本地工具
            if skill.tools:
                tools = [t for t in tools if t.name in skill.tools]
            
            # 添加 MCP 工具
            if skill.requires_mcp:
                mcp_tools = self.mcp_client.get_tools_by_servers(skill.requires_mcp)
                tools.extend(mcp_tools)
        
        return tools
```

**要求：**
1. `技能系统提示词`：把 skill 的 body 注入到 system prompt 中，让 LLM 知道当前角色
2. `模型覆盖`：skill 可以指定使用更强的模型（如 code-review 用 powerful）
3. `temperature 覆盖`：skill 可以指定更低的 temperature（如 debug 用 0.1 确保确定性）
4. `工具过滤`：只注入 skill 需要的工具，减少 LLM 的冲动调用

### Task 4.3：技能级配置隔离

**目的：同一个 socket 可以在不同对话中使用不同技能的配置**

修改 `web/state.py`：

```python
@dataclass
class SocketState:
    """Per-socket state with skill-aware configuration."""
    # ... existing fields ...
    
    # 技能级配置覆盖
    skill_overrides: dict[str, Any] = field(default_factory=dict)
    
    def get_effective_config(self, skill: Skill | None = None) -> dict:
        """Get effective config with skill-level overrides applied."""
        config = self.config.copy()
        
        if skill:
            # 技能级覆盖
            if skill.model:
                config["model"] = skill.model
            if skill.temperature is not None:
                config["temperature"] = skill.temperature
            if skill.max_tokens is not None:
                config["max_tokens"] = skill.max_tokens
            
            # MCP 启用
            if skill.requires_mcp:
                config["mcp_servers"] = skill.requires_mcp
        
        return config
```

**要求：**
1. 配置优先级：Skill 级 > Socket 级 > 活跃配置 > 环境变量
2. 技能级只要改变 model/temperature/max_tokens/MCP，不要改变 API key 或 base_url
3. 技能退出时自动清除技能级覆盖，恢复 socket 级配置

### Task 4.4：前端技能状态展示

**目的：让用户知道当前激活了什么技能**

修改前端代码（`web/static/` 下的 JS）：

1. 在聊天界面添加技能指示器（如微信客服的"企业微信"标签）
2. 当 `active_skill` 变化时，通知前端更新标签
3. 支持用户点击标签查看当前技能说明
4. 支持用户手动退出技能（点 X 关闭标签）

**最简实现：**
- 只需要在消息框上方添加一个小标签，显示"🛠️ 代码审查中"
- 用 SocketIO event `skill_changed` 同步状态

---

## 页 5：测试与验收（1 天）

### Task 5.1：SkillLoader 单元测试

创建 `tests/test_skills.py`：

```python
"""Tests for skill system."""
import pytest
from pathlib import Path

from core.skills.loader import SkillLoader, Skill


class TestSkillLoader:
    def test_load_single_skill(self, tmp_path):
        """Test parsing a single skill file."""
        ...

    def test_trigger_exact_match(self, tmp_path):
        """Test exact trigger matching."""
        ...

    def test_trigger_contains_match(self, tmp_path):
        """Test contains trigger matching."""
        ...

    def test_trigger_priority_longest_wins(self, tmp_path):
        """Test that longest trigger wins on overlap."""
        ...

    def test_invalid_skill_skipped(self, tmp_path):
        """Test that invalid skills are skipped with warning."""
        ...

    def test_reload(self, tmp_path):
        """Test hot reload picks up new skills."""
        ...
```

**要求：**
1. 用 `tmp_path` fixture 创建临时 skill 目录，不依赖实际 skills/ 目录
2. 测试各种触发器匹配边界情况（大小写、空格、部分匹配）
3. 测试错误处理（缺少必填字段、无效 YAML）

### Task 5.2：MCP 客户端测试

创建 `tests/test_mcp.py`：

```python
"""Tests for MCP integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.mcp.client import MCPClient
from core.mcp.config import MCPConfigManager


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_discover_tools(self):
        """Test tool discovery from mock server."""
        ...

    @pytest.mark.asyncio
    async def test_invoke_tool(self):
        """Test tool invocation."""
        ...

    @pytest.mark.asyncio
    async def test_connection_failure_graceful(self):
        """Test that failed connections don't crash the system."""
        ...

    def test_namespace_prefix(self):
        """Test that tool names include server prefix."""
        ...
```

**要求：**
1. 用 `unittest.mock` 模拟 MCP 服务器，不需要真实连接
2. 测试工具命名空间格式（`server/tool`）
3. 测试连接失败时的优雅降级

### Task 5.3：技能触发流程测试

创建 `tests/test_skill_trigger.py`：

```python
"""Integration tests for skill trigger flow."""
import pytest


class TestSkillTrigger:
    def test_skill_triggered_by_keyword(self):
        """Test that 'code review' triggers code-review skill."""
        ...

    def test_skill_context_preserved(self):
        """Test that skill context survives multiple turns."""
        ...

    def test_skill_exit_clears_state(self):
        """Test that exit keyword clears active_skill."""
        ...

    def test_skill_model_override(self):
        """Test that skill can override model selection."""
        ...
```

**要求：**
1. 用 `pytest.mark.asyncio` 测试异步流程
2. 测试技能退出时的状态清理
3. 测试模型覆盖是否正确传递到 Responder

### Task 5.4：技能内容质量检查

- [ ] 用真实代码测试 code-review 技能，看输出是否符合设计（P0/P1/P2/P3 分级）
- [ ] 用真实错误日志测试 debug 技能，看定位是否准确
- [ ] 用臭代码测试 refactor 技能，看建议是否可执行

---

## 页 6：最终验收

### 工程验收
```bash
# 1. 代码质量
ruff check core/skills/ core/mcp/ agents/ web/ tests/
ruff format --check core/skills/ core/mcp/ agents/ web/ tests/
mypy core/skills/ core/mcp/ --explicit-package-bases

# 2. 测试
pytest tests/test_skills.py tests/test_mcp.py tests/test_skill_trigger.py -v

# 3. 功能验证
echo "帮我审查一下这段代码" | python -m karen_cli
# 应该触发 code-review 技能

# 4. 推送
git add -A
git commit -m "feat: add skill system and MCP protocol support

details:
- 技能系统：YAML+Markdown 定义、SkillLoader 加载、关键词触发
- MCP 协议：配置管理、工具发现、LangChain 工具包装
- 3 个默认技能：code-review、debug、refactor
- 技能级配置覆盖：model、temperature、tools
- 前端技能状态标签展示
- 单元测试覆盖 loader、trigger、MCP client"
git push karen fix-root-code:main
```

### 文档更新
- [ ] 更新 ARCHITECTURE.md，添加"技能系统"和"MCP 协议"章节
- [ ] 更新 README.md，添加 skills/ 目录介绍
- [ ] 更新 CONTRIBUTING.md，添加"如何创建新技能"指南

---

## 附录：面试话术（从 Phase 3 提炼）

**Q: 技能系统是怎么设计的？**
A: 用 YAML+Markdown 定义技能，包含触发关键词、角色定义、工作流程。SkillLoader 在启动时加载所有技能并建立索引，0ms 完成触发。触发后技能内容会注入到 system prompt，并覆盖 model/temperature/tools 配置。

**Q: 为什么用 MCP 协议而不是硬编码工具？**
A: 硬编码工具无法动态扩展。MCP 协议是标准的 LLM 工具调用协议，可以连接任何兼容的工具服务器（文件系统、数据库、GitHub 等）。我的实现是通过 MCP 客户端自动发现工具，然后动态包装成 LangChain Tool，集成到 LangGraph 节点中。

**Q: 技能和意图分类的区别是什么？**
A: 意图分类是"用户想干什么"（搜索、翻译、编码），技能是"用户想要什么角色"。意图分类决定走哪个 graph，技能决定在这个 graph 里怎么执行（什么系统提示词、什么模型、什么工具）。

**Q: 技能级配置隔离是怎么实现的？**
A: 配置有四层优先级：Skill 级 > Socket 级 > 活跃配置 > 环境变量。当技能被触发时，它可以覆盖 model、temperature、max_tokens和 MCP 服务器列表。退出技能时自动清除覆盖。

**Q: 如何添加一个新技能？**
A: 新建一个目录，写 SKILL.md（YAML frontmatter + Markdown body），不需要改代码。重启后 SkillLoader 自动加载。如果需要新工具，可以通过 MCP 服务器提供，也不需要改主代码。
