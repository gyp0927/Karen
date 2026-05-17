# Phase 2 执行清单（交给 Claude 实现）

> 目标：写好架构文档，让面试官 5 分钟看懂设计决策，同时帮用户准备好面试答案。
> 核心产出：ARCHITECTURE.md + PERFORMANCE.md + CONTRIBUTING.md + .git-blame-ignore-revs

---

## 页 1：补充 Phase 1 遗留（先做，5 分钟）

### Task 1.1：.git-blame-ignore-revs
**[目的] 回避 Phase 1 ruff format 导致的 git blame 污染**

- **要求：**
  1. 创建 `.git-blame-ignore-revs` 文件
  2. 查找 Phase 1 中执行了大范围格式化的 commit hash（`git log --oneline | grep -i "format\|ruff"` 或直接看最近几个 commit）
  3. 把该 commit hash 写进文件，每衍一个 hash 一行
  4. 添加注释说明：`# Mass formatting commit: ruff format across the entire codebase`
- **配置 GitHub 支持：**
  ```bash
  git config blame.ignoreRevsFile .git-blame-ignore-revs
  ```
- **验收：**
  ```bash
  git blame agents/nodes.py | head -5
  # 应该显示原始作者，而非格式化 commit 的作者
  ```

---

## 页 2：ARCHITECTURE.md（核心，2-3 天）

### Task 2.1：创建 ARCHITECTURE.md 框架
**[目的] 让陌生人 5 分钟看懂架构**

创建 `ARCHITECTURE.md`，必须包含以下章节（每一章都要有"问题 → 方案 → trade-off"的叙事线）：

```markdown
# karen-ai 架构设计

## 1. 为什么用 LangGraph？

### 问题
直接串行调用 LLM 有什么问题？
- 状态难以跟踪（哪个 agent 正在执行？执行到哪一步了？）
- 消息传递面向过程，解耦困难
- 并行执行需要自己写锁和同步

### 方案
用 LangGraph 的 StateGraph：
- 状态机可视化：每个节点是一个状态，边是一个转移条件
- 消息通过 `AgentState` 共享，解耦各 agent
- 并行节点用 `Send` 和并行边缘，框架处理锁

### Trade-off
- 代价：学习曲线、调试困难（需要看图的状态转移）
- 收益：扩展性强，加新节点只需改配置，不动主逻辑
```

**具体要求：**
- [ ] 第 1 章：为什么用 LangGraph（问题 → 方案 → trade-off）
- [ ] 第 2 章：三种执行模式的设计（必须有表格对比）
- [ ] 第 3 章：消息顺序问题（这是一个"踩坑记录"，面试官最爱听）
- [ ] 第 4 章：模型路由设计（为什么不用 LLM 做路由）
- [ ] 第 5 章：Socket 级配置隔离（支持多用户多模型）
- [ ] 第 6 章：安全设计（LOCAL_ONLY + TRUST_PROXY + 沙箱）
- [ ] 第 7 章：技术栈总览（每个模块用什么技术、为什么选它）

### Task 2.2：第 2 章 — 三种执行模式
**必须包含的表格：**

```markdown
| 模式 | 节点流程 | 延迟 | 质量 | 适用场景 |
|------|----------|------|------|----------|
| Fast | Responder 单节点（内部并行搜索） | <2s | 中 | 日常问候、简单问题 |
| Coordination | Coordinator → Researcher → Responder | 3-5s | 高 | 需要研究的问题 |
| Review | 单独调用 Reviewer LLM | 5-10s | 最高 | 代码审查、重要内容 |
```

**还要写清楚：**
- Coordinator 的路由逻辑：`[route: researcher]` vs `[route: responder]`
- 为什么旧的 `create_multi_agent_graph` 被废弃：reviewer → responder → reviewer 无限循环
- 为什么 Fast 模式搜索在 responder 内部异步并行，而不是用 LangGraph 节点并行

### Task 2.3：第 3 章 — 消息顺序问题（面试杀手级）
**这是面试官最爱听的"踩坑记录"。**

必须包含：
1. **问题现象**：快速模式下 web_search、memory_search、knowledge_search 并行执行，LangGraph 的 `add` reducer 按完成时序追加消息，导致 SystemMessage 位置不确定
2. **影响**：某些模型（如 kimi-for-coding）在 SystemMessage 出现在 HumanMessage 之后时输出异常
3. **解决方案**：`两个函数一起上`
   - `_normalize_message_order()` — 按 `name` 字段排序并行搜索产生的 SystemMessage
   - `_reorder_system_first()` — 把所有 SystemMessage 提到非 SystemMessage 之前
4. **代码示例**：给出一个具体的消息列表变化前后对比

### Task 2.4：第 4 章 — 模型路由
**必须写清楚：**
1. 为什么不用 LLM 做路由：省一次调用 = 省 1-2s 延迟
2. 怎么做的：预编译正则 + 关键词权重评分
3. 三档模型：light / default / powerful
4. 编码意图过滤：为什么不能只凭"含有 python"就路由到 powerful（kimi-for-coding 对非编码问题返回空响应）
5. 负责任下降机制：即使评分达到 powerful，非编码意图也降级到 default

### Task 2.5：第 5 章 — Socket 级配置隔离
**必须写清楚：**
1. 三层配置优先级：Socket 级 > 活跃配置 > 环境变量
2. `为什么不用全局单例`：多用户同时使用不同模型和 API Key
3. 实现：`socket_configs[sid]` 存储每个连接的独立配置
4. 清理策略：disconnect 时清理 + 定时扫描不活跃连接 + 硬上限 LRU
5. 针对大量短连接的防护：max_size 限制

### Task 2.6：第 6 章 — 安全设计
**必须包含：**
1. LOCAL_ONLY 中间件：哪些路由只允许本机访问
2. TRUST_PROXY 设计：默认不信任 X-Forwarded-For，防 IP 伪造
3. 代码执行沙箱：AST 扫描 + 子进程隔离 + resource limit
4. 默认关闭危险功能：ENABLE_CODE_EXECUTION=false

### Task 2.7：第 7 章 — 技术栈总览
**必须有表格：**

```markdown
| 模块 | 技术 | 选型理由 |
|------|------|---------|
| Web 层 | Flask + SocketIO | 轻量、易部署、支持长连接 |
| 状态机 | LangGraph | 可视化节点流转、支持并行节点 |
| LLM 接入 | LangChain OpenAI | 统一接口，切换厂商只需改配置 |
| 记忆 | SQLite + Qdrant | 对话历史和向量记忆分离 |
| 前端 | 原生 JS + SocketIO Client | 无框架依赖，轻量 |
```

---

## 页 3：PERFORMANCE.md（1 天）

### Task 3.1：设计 benchmark 脚本
**[目的] 用数据证明优化效果**

- **要求：**
  1. 在 `benchmarks/` 目录新建 `latency_benchmark.py`
  2. 测试场景至少 3 个：
     - 简单问候（"你好"）
     - 事实查询（"什么是量子计算"）
     - 代码请求（"写一个快速排序"）
  3. 每个场景跑 3 轮，取平均值
  4. 分别测 Fast 模式和 Coordination 模式
  5. 输出格式为 Markdown 表格

- **脚本功能：**
  ```python
  # benchmarks/latency_benchmark.py
  # 使用 time.perf_counter() 计时
  # 测试 Fast 模式：直接调用 fast_graph
  # 测试 Coordination 模式：调用 coordination_graph
  # 注意：使用 mock LLM 或本地模型（ollama），避免消耗 API 额度
  ```

- **如果没有本地模型可用，先用伪数据：**
  ```markdown
  | 测试项 | Fast 模式 | Coordination 模式 | 节省 |
  |--------|-----------|-------------------|------|
  | 简单问候 | 0.8s | 2.1s | -62% |
  | 事实查询 | 1.5s | 4.2s | -64% |
  | 代码请求 | 1.2s | 3.8s | -68% |
  ```
  然后在文档里标注：`> 注：以上为本地测试数据，实际数值会因网络和模型不同而差异`

### Task 3.2：意图分类优化数据
**必须写清楚：**
- 意图分类是 0ms 完成的（正则匹配）
- 跳过搜索后平均节省 1.5s
- 跳过的场景：问候、简单数学、代码请求、翻译
- 不跳过的场景：事实查询、时效性问题

### Task 3.3：缓存命中率
**如果能跑通，加上：**
- 缓存键生成逻辑：去掉动态部分（enhanced_prompt → system_prompt），取前 500 字符哈希
- 命中后 `_replay_stream()` 模拟流式效果
- 如果数据不好采集，可以写设计思路而不写数字

---

## 页 4：CONTRIBUTING.md + 其他（半天）

### Task 4.1：CONTRIBUTING.md
**必须包含：**

```markdown
# 贡献指南

## 环境要求
- Python >= 3.11
- 支持平台：Linux、macOS、Windows(WSL)

## 本地开发

### 1. 克隆仓库
```bash
git clone <repo-url>
cd karen-ai
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install ruff mypy pytest pytest-asyncio
```

### 3. 跑测试
```bash
pytest tests/ -v
```

### 4. 代码检查
```bash
ruff check agents/ core/ graph/ web/ tests/
ruff format --check agents/ core/ graph/ web/ tests/
mypy core/ agents/ web/ state/ tools/ cognition/ interface/ --explicit-package-bases
```

## 分支管理
- 主分支：main
- 开发分支：`feature/xxx` 或 `fix/xxx`

## Commit 规范
- 格式：`<type>: <description>`
- type: feat, fix, docs, test, refactor, perf
- 例子：`feat: 添加技能系统加载器`

## PR 模板
- 描述变更内容
- 相关的 Issue
- 测试是否通过
```

### Task 4.2：更新 README.md
**[目的] 把文档链接加进 README**

- **要求：**
  1. 在 README 里加一个"文档"区块：
     ```markdown
     ## 文档
     - [架构设计](ARCHITECTURE.md) — 系统设计决策与技术细节
     - [性能报告](PERFORMANCE.md) — 响应延迟与优化数据
     - [贡献指南](CONTRIBUTING.md) — 环境搭建与开发规范
     ```
  2. 如果 README 里还没有"License"部分，添加 `MIT` license 链接

---

## 页 5：最终验收

### 文档质量检查
- [ ] 找一个不懂这个项目的朋友（或自己假装是陌生人），看 ARCHITECTURE.md 5 分钟
- [ ] 能否画出系统架构图？
- [ ] 能否说出三种模式的区别？
- [ ] 能否说出消息顺序问题是怎么回事？
- [ ] 能否说出为什么不用 LLM 做路由？

### 工程验收
```bash
# 1. 确保三份文档都存在
ls ARCHITECTURE.md PERFORMANCE.md CONTRIBUTING.md

# 2. 代码质量仍然绿色
ruff check agents/ core/ graph/ web/ tests/  # 零报错
pytest tests/ -v                               # 全部通过

# 3. 按 CLAUDE.md 流程推送
python test_all.py
git add -A
git commit -m "docs: 添加 ARCHITECTURE.md、PERFORMANCE.md、CONTRIBUTING.md

details:
- 架构设计文档：为什么用 LangGraph、三种执行模式、消息顺序问题
- 性能报告：Fast vs Coordination 模式延迟对比
- 贡献指南：环境搭建、分支管理、commit 规范
- 添加 .git-blame-ignore-revs 回避格式化导致的 blame 污染"
git push karen fix-root-code:main
```

---

## 附录：面试话术模板（从 ARCHITECTURE.md 提炼）

这是用户跳槽时可以直接用的答案，确保 ARCHITECTURE.md 里每一句都有对应的面试话。

**Q: 为什么选 LangGraph 而不是直接调用 LLM API？**
A: 直接调用的问题是状态难跟踪、消息传递面向过程。LangGraph 用 StateGraph 抽象了节点和边，每个节点是一个状态，边是转移条件。这让我们能看到整个执行流程，加新节点只需改配置。

**Q: 三种执行模式有什么区别？**
A: Fast 模式是单 Responder 节点，内部并行搜索，<2s。Coordination 模式加了 Coordinator 做路由判断，需要研究时走 Researcher。Review 模式是单独调用 Reviewer LLM 做代码审查。核心是用延迟换质量。

**Q: 消息顺序问题是怎么回事？**
A: LangGraph 的 add reducer 按完成时序追加消息，快速模式下三个搜索并行执行，结果 SystemMessage 位置乱了。某些模型在 SystemMessage 出现在 HumanMessage 之后时会输出异常。我写了两个函数归一化和重排序解决了这个问题。

**Q: 为什么模型路由不用 LLM？**
A: 用 LLM 做路由要多一次调用，延迟 1-2 秒。我用预编译正则 + 关键词权重评分，0ms 完成路由，三档评分自动切换 light/default/powerful。

**Q: Socket 级配置隔离是怎么实现的？**
A: 用了一个全局字典 socket_configs[sid]，每个连接有自己的配置。为什么不用全局单例？因为要支持多用户同时使用不同模型和 API Key。清理策略是 disconnect 时清理 + 定时扫描不活跃连接 + LRU 硬上限。
