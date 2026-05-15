# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发规范

### 代码修改后流程

每次修改代码后必须执行：

1. **运行测试** — `python test_all.py`
2. **测试通过 → 提交并推送**：
   - `git add -A`
   - `git commit -m "描述"`
   - `git push karen fix-root-code:main`
3. **向用户汇报** — 报告测试结果和推送状态
4. **测试失败 → 先修复再推**

### 例外情况

- 用户明确说"先不推"或"不要推送"时跳过推送
- 仅修改文档（README、注释）时可跳过测试，但仍需推送

### 推送目标

- Remote: `karen`
- 本地分支: `fix-root-code`
- 远程分支: `main`

---

## 常用命令

### 测试

```bash
# 运行全部测试
python test_all.py

# 运行单个测试（test_all.py 使用自定义装饰器，无 pytest）
# 方法：在 test_all.py 中注释掉 tests 列表中不需要的项
```

### 启动应用

```bash
# Web 界面（推荐，自动打开浏览器）
python desktop_app.py

# 纯 Web 服务端（不自动打开浏览器）
python web/app.py

# 桌面客户端（WebView2 渲染）
python desktop_client.py

# 控制台模式
python main.py
```

Web 服务默认绑定 `127.0.0.1:5000`，局域网访问需设置 `BIND_HOST=0.0.0.0`。

### 其他

```bash
# 安装依赖
pip install -r requirements.txt

# 无配置好的 linter/black，项目未引入代码格式化工具
```

---

## 高层架构

### 三种执行模式

系统有三种运行模式，由 `HumanInterface` 的 `fast_mode` 参数和 `graph/orchestrator.py` 中的图定义决定：

- **快速模式（默认）**：单 `Responder` 节点。搜索（web + memory）在 `responder_node` 内部异步并行执行，最多等待 1.2 秒。这是 Web 端的默认模式。
- **协调模式**：`Coordinator → Researcher → Responder`。Coordinator 输出 `[route: researcher]` 或 `[route: responder]` 标签决定分支。
- **审查模式**：不在主图内，由用户点击前端"检查"按钮后单独触发 `_do_review()`，直接调用 reviewer LLM。

`graph/orchestrator.py` 中的 `create_multi_agent_graph`（带 reviewer 循环的完整图）已标记为 DEPRECATED，生产环境使用 `create_coordination_graph` 或 `create_fast_graph`。

### LLM 配置隔离机制

配置存在三层，优先级从高到低：

1. **Socket 级配置**（Web 多用户场景）：`web/state.py` 中的 `socket_configs[sid]`，用户在 Web 界面输入自己的 API Key 后存储于此。
2. **活跃配置**：`state/model_config_manager.py` 管理的 `state/model_configs.json`，通过配置页面切换。
3. **环境变量**：`.env` 文件，向后兼容。

`agents/llm.py` 中的 `_llm_configs[sid]` 和 `_token_callbacks[sid]` 按 socket ID 隔离，使多个并发连接可使用不同模型和 API Key。`get_llm(sid)` 会根据 sid 查找对应的配置创建 LLM 实例。

### LangGraph 消息顺序问题

`agents/nodes.py` 中的 `_normalize_message_order()` 和 `_reorder_system_first()` 解决两个顺序问题：

- **并行 searcher 非确定性**：fast 模式下 web_search、memory_search、knowledge_search 并行执行，LangGraph 的 `add` reducer 按完成时序追加消息，导致 SystemMessage 位置不确定。`_normalize_message_order()` 将连续的具名 SystemMessage 按 `name` 排序。
- **SystemMessage 后置**：某些模型（如 V3.2-Pro）在 SystemMessage 出现在 HumanMessage 之后时输出异常。`_reorder_system_first()` 将所有 SystemMessage 提到非 SystemMessage 之前。

### 模型路由的编码意图过滤

`core/model_router.py` 根据关键词权重分析查询复杂度，分 `light/default/powerful` 三档。关键细节：

- `powerful` 档目前指向 `kimi-for-coding`（编码专用模型），该模型对非编码问题返回 HTTP 200 + 空响应。
- 因此 `_is_coding_intent()` 有严格过滤：必须包含代码块标记 ```、报错关键词、或明确编码动作词（写/实现/调试/重构等）。仅提到 "python" 而不含编码动作不算编码意图。
- 即使复杂度评分达到 powerful，非编码意图也会降级到 default。

### 意图分类决定搜索跳过

`core/intent.py` 使用纯正则规则（无 LLM 调用）分类意图，覆盖约 80% 场景。分类结果决定 responder 是否跳过搜索：

- **跳过全部搜索**：问候、告别、感谢、简单数学、代码请求、创意写作、闲聊
- **跳过联网，保留记忆**：翻译
- **全部启用**：事实查询（含"什么是/为什么/最新/天气/股价"等时效关键词）
- **未知**：默认启用全部搜索

这直接影响响应延迟：跳过的查询可节省 1-5 秒搜索时间。

### 记忆系统 vs 聊天历史

两者完全独立：

- **聊天历史**：`state/persistence.py` 使用 SQLite，存储完整对话消息，按会话隔离，用于界面显示和 LLM 上下文。`SessionManager` 默认保留最近 10 轮对话传给 LLM。
- **自适应记忆**：`hot_and_cold_memory/` 使用本地 sentence-transformers 生成 embedding，存储在 Qdrant 本地向量库 + SQLite 元数据。自动从对话中提取关键信息（用户偏好、重要事实），跨会话长期保留。热层保存完整内容，冷层保存压缩摘要。

### 缓存键归一化

`core/cache.py` 仅缓存 `responder` 和 `coordinator`。为保证缓存命中：

- 缓存键计算前，将动态的 `enhanced_prompt`（含 emotion/intuition 等认知状态）替换回静态 `system_prompt`
- 过滤掉 `[route:]` 标签和搜索结果 SystemMessage
- 单条消息最多取前 500 字符参与哈希，避免超长消息拖慢计算
- 命中后通过 `_replay_stream()` 按 50 字符 chunk 模拟流式效果

### Web 层职责拆分

经过重构后：

- **`web/state.py`**：SocketState 类、socket_states/socket_configs 全局状态、`get_socket_state()`、`cleanup_socket()`、`init_agents()`（预编译 fast_graph）、`has_valid_config()`
- **`web/utils.py`**：`run_async_in_thread()`、`_GENERATED_DIR`
- **`web/app.py`**：Flask/SocketIO 初始化、事件处理器（`@socketio.on`）、`_async_handle_message()`（核心消息处理逻辑）
- **`web/api.py`**：Flask Blueprint，HTTP REST API 路由（配置管理、插件、MCP、RAG、缓存等）

### 安全要点

- **代码执行**：`ENABLE_CODE_EXECUTION=false` 默认关闭。开启后通过 AST 扫描（禁危险 import/call）+ 子进程隔离（自定义受限 `__builtins__`）+ 30 秒超时执行。
- **插件上传**：`web/api.py` 的 `_scan_plugin_content()` 进行 AST 级安全检查，禁止 eval/exec/getattr/__import__ 等，限制文件大小 256KB。
- **本地访问限制**：`LOCAL_ONLY_PREFIXES` 中的路由（`/config`、`/api/auth/users` 等）仅允许 `127.0.0.1/localhost` 访问，防止局域网用户窃取 API Key。
- **API Key 脱敏**：`get_all_tiers()`、`web/api.py` 的 config API 中，API Key 只显示前 4 位和后 4 位，中间用 `****` 替换。
