# Phase 4 执行清单（交给 Claude 实现）

> 目标：实现子 Agent 委派系统，让 karen-ai 能并行派发多个子任务，用 asyncio.gather 聚合结果。这是"杀手级差异化能力"。
> 核心产出：core/subagent/ 模块 + 并行执行引擎 + 3 个并行场景 + 结果聚合策略

---

## 页 1：子 Agent 架构设计（核心，2 天）

### Task 1.1：设计 SubAgent 抽象层

**目的：定义子 Agent 的统一接口，支持不同执行策略**

创建 `core/subagent/base.py`：

```python
"""Sub-agent abstraction: define the interface for parallel task execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AggregationStrategy(Enum):
    """How to combine results from multiple sub-agents."""
    MERGE = "merge"      # 拼接所有结果
    VOTE = "vote"        # 多数表决（用于判断题）
    RANK = "rank"        # 按分数排序
    BEST = "best"        # 选最好的一个
    CUSTOM = "custom"    # 自定义聚合函数


@dataclass
class SubTask:
    """A single sub-task dispatched to a sub-agent."""
    id: str
    name: str                    # 子任务名称，如 "security-check"
    prompt: str                  # 给子 agent 的完整 prompt
    model: str = "default"       # 子 agent 使用的模型
    temperature: float = 0.2
    max_tokens: int | None = None
    tools: list[str] = field(default_factory=list)  # 允许使用的工具
    timeout: float = 30.0        # 单个子任务超时（秒）
    retry: int = 1               # 失败重试次数


@dataclass
class SubTaskResult:
    """Result from a single sub-agent execution."""
    task_id: str
    success: bool
    output: str = ""             # 文本输出
    metadata: dict[str, Any] = field(default_factory=dict)  # 结构化数据
    error: str | None = None
    latency_ms: float = 0.0      # 执行耗时
    tokens_used: int = 0


@dataclass
class ParallelResult:
    """Aggregated result from multiple sub-agents."""
    strategy: AggregationStrategy
    results: list[SubTaskResult]
    final_output: str = ""       # 聚合后的最终输出
    confidence: float = 0.0      # 聚合置信度（0-1）
    metadata: dict[str, Any] = field(default_factory=dict)


class SubAgent(ABC):
    """Abstract base class for sub-agents."""
    
    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, task: SubTask) -> SubTaskResult:
        """Execute a single sub-task and return result."""
        ...


class LLMSubAgent(SubAgent):
    """Default implementation: sub-agent backed by an LLM call."""
    
    async def execute(self, task: SubTask) -> SubTaskResult:
        """
        1. Build messages from task.prompt
        2. Call LLM with specified model/temperature/tools
        3. Parse output and return SubTaskResult
        """
        ...
```

**设计要求：**
1. `SubTask` 必须自包含：prompt、模型参数、工具、超时，全部内聚在一个对象里
2. `SubAgent` 是抽象层，目前只有 `LLMSubAgent` 实现，但预留了扩展（比如 `CodeSubAgent` 走代码沙箱，`SearchSubAgent` 走搜索 API）
3. `AggregationStrategy` 覆盖常见场景，支持自定义函数

### Task 1.2：实现并行执行引擎

**目的：调度、执行、限流、超时、错误隔离**

创建 `core/subagent/scheduler.py`：

```python
"""Parallel task scheduler: dispatch sub-agents with concurrency control."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from .base import AggregationStrategy, ParallelResult, SubAgent, SubTask, SubTaskResult


class TaskScheduler:
    """Schedule and execute sub-tasks in parallel with controlled concurrency."""
    
    def __init__(
        self,
        max_concurrency: int = 3,
        global_timeout: float = 60.0,
        default_agent_factory: Callable[[], SubAgent] | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency      # 最大并行数
        self.global_timeout = global_timeout        # 全局超时
        self.default_agent_factory = default_agent_factory or LLMSubAgent
        self._semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_parallel(
        self,
        tasks: list[SubTask],
        strategy: AggregationStrategy = AggregationStrategy.MERGE,
        custom_aggregator: Callable[[list[SubTaskResult]], str] | None = None,
    ) -> ParallelResult:
        """
        Execute all tasks in parallel with concurrency limit.
        
        1. Create semaphore-gathered coroutines
        2. Apply per-task timeout via asyncio.wait_for
        3. Catch exceptions, mark as failed but don't crash others
        4. Aggregate results based on strategy
        """
        ...
    
    async def _execute_single(
        self,
        task: SubTask,
        agent: SubAgent | None = None,
    ) -> SubTaskResult:
        """Execute one task with semaphore + timeout + retry."""
        agent = agent or self.default_agent_factory()
        
        async with self._semaphore:
            for attempt in range(task.retry + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.execute(task),
                        timeout=task.timeout,
                    )
                    return result
                except asyncio.TimeoutError:
                    if attempt == task.retry:
                        return SubTaskResult(
                            task_id=task.id,
                            success=False,
                            error=f"Timeout after {task.timeout}s",
                        )
                except Exception as e:
                    if attempt == task.retry:
                        return SubTaskResult(
                            task_id=task.id,
                            success=False,
                            error=str(e),
                        )
                    await asyncio.sleep(1 * (attempt + 1))  # exponential backoff
        
    def _aggregate(
        self,
        results: list[SubTaskResult],
        strategy: AggregationStrategy,
        custom_aggregator: Callable[[list[SubTaskResult]], str] | None = None,
    ) -> tuple[str, float]:
        """Aggregate results into final output.
        
        Returns: (final_output, confidence)
        """
        if strategy == AggregationStrategy.MERGE:
            # 按 task 顺序拼接，加分割线
            ...
        elif strategy == AggregationStrategy.VOTE:
            # 统计 success 比例作为 confidence
            ...
        elif strategy == AggregationStrategy.RANK:
            # 按 metadata.score 排序
            ...
        elif strategy == AggregationStrategy.BEST:
            # 选 tokens_used 最少且 success 的结果（效率优先）
            ...
        elif strategy == AggregationStrategy.CUSTOM and custom_aggregator:
            return custom_aggregator(results), 1.0
        else:
            return self._default_merge(results), 0.5
```

**实现细节：**
1. `max_concurrency=3`：防止同时开太多 LLM 调用导致 API 限流或内存爆炸
2. `semaphore`：用 asyncio.Semaphore 实现并发控制，不是线程锁
3. `错误隔离`：每个子任务用 try/except 包裹，一个失败不影响其他
4. `重试策略`：指数退避，最多 `task.retry` 次
5. `超时层级`：task.timeout（单任务）+ global_timeout（整体）
6. `聚合默认`：MERGE 策略按任务定义顺序拼接，用 `---` 分割线分隔

### Task 1.3：实现结果聚合器

**目的：针对不同场景提供专业的聚合逻辑**

创建 `core/subagent/aggregator.py`：

```python
"""Result aggregators for different parallel scenarios."""
from __future__ import annotations

from .base import SubTaskResult


def merge_code_reviews(results: list[SubTaskResult]) -> str:
    """Aggregate code review results from multiple sub-agents.
    
    Deduplicate by file:line, sort by severity, keep highest severity per issue.
    """
    ...


def merge_research(results: list[SubTaskResult]) -> str:
    """Aggregate research results from multiple sources.
    
    Remove duplicate facts, merge complementary information.
    """
    ...


def vote_boolean(results: list[SubTaskResult]) -> str:
    """Vote on a yes/no question.
    
    Returns majority decision with confidence score.
    """
    ...


def rank_by_confidence(results: list[SubTaskResult]) -> str:
    """Rank results by confidence score in metadata.
    
    Returns top result with explanation.
    """
    ...
```

**merge_code_reviews 的具体逻辑：**
1. 从每个子结果中提取问题列表（按 P0/P1/P2/P3 分级）
2. 按 `file:line` 去重，保留最高严重级别
3. 按严重级别排序：P0 → P1 → P2 → P3
4. 输出格式：先汇总统计（"共发现 3 个 P0, 5 个 P1..."），再列详情

---

## 页 2：与现有系统集成（2 天）

### Task 2.1：扩展 AgentState 支持子 Agent

**目的：让主流程能感知和调度子 Agent**

修改 `core/chat_state.py`：

```python
@dataclass
class AgentState:
    # ... existing fields ...
    
    # 子 agent 相关
    subagent_mode: bool = False           # 是否处于子 agent 委派模式
    pending_subtasks: list[dict] = field(default_factory=list)  # 待执行子任务
    subtask_results: list[dict] = field(default_factory=list)   # 子任务结果
    parallel_strategy: str = "merge"      # 当前使用的聚合策略
```

### Task 2.2：新增 SubAgentNode

**目的：在 LangGraph 中增加一个"子 Agent 委派"节点**

创建 `graph/nodes/subagent.py`：

```python
"""Sub-agent delegation node for LangGraph."""
from __future__ import annotations

from core.chat_state import AgentState
from core.subagent.scheduler import TaskScheduler
from core.subagent.aggregator import merge_code_reviews, merge_research
from core.subagent.base import AggregationStrategy, SubTask


class SubAgentNode:
    """LangGraph node that dispatches parallel sub-agents and aggregates results."""
    
    def __init__(self) -> None:
        self.scheduler = TaskScheduler(max_concurrency=3)
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        1. Parse state to determine parallel scenario
        2. Generate sub-tasks based on scenario
        3. Dispatch via TaskScheduler
        4. Aggregate results
        5. Append aggregated result to messages
        """
        scenario = self._detect_scenario(state)
        tasks = self._generate_tasks(state, scenario)
        
        result = await self.scheduler.run_parallel(
            tasks=tasks,
            strategy=self._get_strategy(scenario),
            custom_aggregator=self._get_aggregator(scenario),
        )
        
        # 把聚合结果写入 state
        state.subagent_mode = False
        state.subtask_results = [r.__dict__ for r in result.results]
        
        # 生成一条 SystemMessage 作为子 agent 的"总结报告"
        summary = self._format_summary(result, scenario)
        state.messages.append(SystemMessage(content=summary))
        
        return state
    
    def _detect_scenario(self, state: AgentState) -> str:
        """Detect which parallel scenario to use.
        
        Scenarios:
        - code_review: 用户要求代码审查
        - research: 用户要求深度研究
        - debug: 用户要求调试分析
        - default: 通用并行（简单分片）
        """
        ...
    
    def _generate_tasks(self, state: AgentState, scenario: str) -> list[SubTask]:
        """Generate sub-tasks for the detected scenario."""
        ...
    
    def _get_strategy(self, scenario: str) -> AggregationStrategy:
        strategies = {
            "code_review": AggregationStrategy.CUSTOM,
            "research": AggregationStrategy.CUSTOM,
            "debug": AggregationStrategy.MERGE,
            "default": AggregationStrategy.MERGE,
        }
        return strategies.get(scenario, AggregationStrategy.MERGE)
    
    def _get_aggregator(self, scenario: str) -> Callable | None:
        aggregators = {
            "code_review": merge_code_reviews,
            "research": merge_research,
        }
        return aggregators.get(scenario)
    
    def _format_summary(self, result: ParallelResult, scenario: str) -> str:
        """Format aggregated result into a summary message."""
        ...
```

### Task 2.3：修改 Coordinator 路由

**目的：让 Coordinator 能决定什么时候走子 Agent 模式**

修改 `graph/nodes/coordinator.py`：

```python
class CoordinatorNode:
    async def execute(self, state: AgentState) -> AgentState:
        # ... existing logic ...
        
        # 新增：判断是否需要子 agent 委派
        if self._should_delegate_to_subagents(state):
            state.subagent_mode = True
            state.pending_subtasks = self._build_subtasks(state)
            return state
        
        # ... existing routing logic ...
    
    def _should_delegate_to_subagents(self, state: AgentState) -> bool:
        """Determine if current task benefits from parallel sub-agents.
        
        Triggers:
        1. 用户明确要求代码审查（且代码量 > 100 行）
        2. 用户要求"深入研究"或"多角度分析"
        3. 用户要求"检查所有问题"
        4. 当前是 review 模式且代码块 > 50 行
        """
        ...
    
    def _build_subtasks(self, state: AgentState) -> list[dict]:
        """Build sub-task definitions for the scheduler."""
        ...
```

**路由修改：**
1. 在 `graph/orchestrator.py` 的 StateGraph 中，新增 `subagent` 节点
2. Coordinator 的输出边增加条件：`if state.subagent_mode → subagent_node`
3. SubAgentNode 执行完后 → ResponderNode（用聚合结果回答用户）

**新的 Graph 结构：**
```
Coordinator
  ├── researcher → responder
  ├── responder (fast)
  ├── reviewer → responder
  └── subagent → responder   # 新增
```

### Task 2.4：前端并行进度展示

**目的：让用户看到子 Agent 正在并行工作**

修改前端 SocketIO 处理：

1. 新增事件 `subagent_started`：发送 `{total: 3, tasks: [{id, name}]}`
2. 新增事件 `subagent_progress`：发送 `{completed: 1, total: 3, current: "security-check"}`
3. 新增事件 `subagent_completed`：发送 `{duration: 4.2, results: [...]}`

前端 UI：
- 显示并行进度条（如"🔍 安全检查 1/3"）
- 每个子任务一个状态点（等待中 → 执行中 → 完成/失败）
- 完成后显示"3 个子任务并行完成，耗时 4.2s"

**最简实现：**
- 在消息框上方显示一个临时进度条组件
- 子 agent 完成后进度条消失，结果以正常消息形式展示

---

## 页 3：三个并行场景实现（3 天）

### Task 3.1：代码审查并行（核心场景）

**目的：把代码审查拆成 3 个专家并行审查，提升深度和速度**

实现 `_generate_code_review_tasks()`：

```python
def _generate_code_review_tasks(self, state: AgentState) -> list[SubTask]:
    """Generate parallel code review sub-tasks.
    
    3 sub-agents:
    1. Security Agent: SQL injection, XSS, command injection, secrets
    2. Performance Agent: N+1 queries, memory leaks, algorithm complexity
    3. Style Agent: Naming, comments, type hints, PEP8, design patterns
    """
    code = self._extract_code_from_state(state)
    
    return [
        SubTask(
            id="security-check",
            name="安全审查",
            prompt=f"""你是一个安全专家。审查以下代码的安全问题：

{code}

关注：SQL 注入、XSS、命令注入、敏感数据硬编码、权限绕过。
用中文输出，格式：
- [严重级别] 位置 - 问题描述 - 修复建议
""",
            model="powerful",
            temperature=0.1,
            timeout=45.0,
        ),
        SubTask(
            id="performance-check",
            name="性能审查",
            prompt=f"""你是一个性能优化专家。审查以下代码的性能问题：

{code}

关注：时间复杂度、空间复杂度、N+1 查询、内存泄漏、不必要的循环。
用中文输出，格式同上。
""",
            model="powerful",
            temperature=0.1,
            timeout=45.0,
        ),
        SubTask(
            id="style-check",
            name="风格审查",
            prompt=f"""你是一个代码风格专家。审查以下代码的可读性和设计：

{code}

关注：命名规范、函数长度、注释质量、类型注解、设计模式、SOLID 原则。
用中文输出，格式同上。
""",
            model="default",  # 风格审查不需要最强模型
            temperature=0.2,
            timeout=30.0,
        ),
    ]
```

**聚合逻辑（merge_code_reviews）：**
1. 提取每个子结果中的问题列表（用正则匹配 `P[0-3]` 或 `\[严重\]` 等标记）
2. 按 `file:line` 去重
3. 严重级别映射：security 的 P0 = 最高，style 的 P0 = 次高
4. 最终输出：
   ```
   ## 代码审查报告（3 个专家并行审查，耗时 3.8s）
   
   ### 安全（1 个问题）
   - [P0] db.py:15 - SQL 注入...
   
   ### 性能（2 个问题）
   - [P1] views.py:42 - N+1 查询...
   
   ### 风格（5 个问题）
   - [P2] utils.py:88 - 函数过长...
   ```

### Task 3.2：深度研究并行

**目的：多源并行搜索，交叉验证信息**

实现 `_generate_research_tasks()`：

```python
def _generate_research_tasks(self, state: AgentState) -> list[SubTask]:
    """Generate parallel research sub-tasks.
    
    3 sub-agents with different angles:
    1. Technical: 技术实现细节、官方文档
    2. Practical: 实际应用案例、最佳实践
    3. Critical: 局限性、替代方案、常见坑
    """
    query = state.messages[-1].content
    
    return [
        SubTask(
            id="tech-research",
            name="技术分析",
            prompt=f"""研究以下问题的技术细节：{query}

从实现原理、核心概念、技术栈角度分析。
要求：引用具体的技术文档或标准。
""",
            model="powerful",
            tools=["web_search"],
            timeout=30.0,
        ),
        SubTask(
            id="practical-research",
            name="实践分析",
            prompt=f"""研究以下问题的实际应用：{query}

从实际案例、最佳实践、 industry adoption 角度分析。
要求：引用真实的应用案例或公司实践。
""",
            model="powerful",
            tools=["web_search"],
            timeout=30.0,
        ),
        SubTask(
            id="critical-research",
            name="批判分析",
            prompt=f"""批判性地分析以下问题：{query}

从局限性、替代方案、常见陷阱、未来趋势角度分析。
要求：指出主流观点可能忽略的问题。
""",
            model="powerful",
            tools=["web_search"],
            timeout=30.0,
        ),
    ]
```

**聚合逻辑（merge_research）：**
1. 把三个角度按"技术 → 实践 → 批判"组织
2. 标记信息冲突点（如 tech 说 A 好，critical 说 A 有坑）
3. 最终输出一个结构化的研究报告

### Task 3.3：调试分析并行

**目的：并行分析日志、代码、测试，快速定位根因**

实现 `_generate_debug_tasks()`：

```python
def _generate_debug_tasks(self, state: AgentState) -> list[SubTask]:
    """Generate parallel debug sub-tasks.
    
    3 sub-agents:
    1. Log Analyzer: 从日志中提取关键错误模式
    2. Code Tracer: 追踪代码执行路径，找异常点
    3. Test Validator: 验证假设，提出测试验证方案
    """
    error_info = self._extract_error_from_state(state)
    code_context = self._extract_code_from_state(state)
    
    return [
        SubTask(
            id="log-analysis",
            name="日志分析",
            prompt=f"""分析以下错误日志，提取关键信息：

{error_info}

输出：
1. 错误类型和位置
2. 错误发生前的上下文
3. 可能的触发条件
""",
            model="default",
            timeout=20.0,
        ),
        SubTask(
            id="code-trace",
            name="代码追踪",
            prompt=f"""追踪以下代码的执行路径，找出可能导致错误的位置：

错误信息：{error_info}
代码上下文：{code_context}

输出：
1. 可疑的代码行
2. 变量值变化路径
3. 边界条件分析
""",
            model="powerful",
            timeout=30.0,
        ),
        SubTask(
            id="test-proposal",
            name="测试验证",
            prompt=f"""基于以下错误，设计测试用例来验证修复方案：

错误信息：{error_info}
代码上下文：{code_context}

输出：
1. 最小复现步骤
2. 单元测试代码
3. 边界测试建议
""",
            model="default",
            tools=["code_executor"],
            timeout=30.0,
        ),
    ]
```

**聚合逻辑：**
- 用 MERGE 策略拼接，但按"日志分析 → 代码追踪 → 测试验证"逻辑组织
- 如果 log-analysis 和 code-trace 定位到同一行，高亮显示"多个角度一致指向 db.py:15"

---

## 页 4：高级功能（2 天）

### Task 4.1：动态任务拆分

**目的：根据输入自动决定拆分成几个子任务**

创建 `core/subagent/decomposer.py`：

```python
"""Dynamic task decomposition: let LLM decide how to split a task."""
from __future__ import annotations

from core.subagent.base import SubTask


class TaskDecomposer:
    """Use a lightweight LLM call to decide how to decompose a task."""
    
    async def decompose(self, task_description: str) -> list[SubTask]:
        """
        Ask a lightweight model to break down the task into sub-tasks.
        
        Prompt template:
        "把以下任务拆成 2-4 个并行的子任务。每个子任务要有明确的名称和职责。
        输出 JSON 格式：[{name, prompt, priority}]"
        
        Returns parsed sub-tasks.
        """
        ...
```

**使用场景：**
- 用户问了一个复杂问题，系统不确定用哪个预设场景
- 先让轻量模型（如 light 档）拆分任务，再并行执行
- 默认关闭，需要时手动开启（防止增加不必要的延迟）

### Task 4.2：子 Agent 结果缓存

**目的：相同的子任务不重复执行**

创建 `core/subagent/cache.py`：

```python
"""Sub-agent result cache: avoid redundant LLM calls."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from .base import SubTask, SubTaskResult


class SubAgentCache:
    """Cache sub-agent results by task content hash."""
    
    def __init__(self, max_size: int = 100) -> None:
        self._cache: dict[str, SubTaskResult] = {}
        self._max_size = max_size
    
    def _hash_task(self, task: SubTask) -> str:
        """Generate hash from task prompt + model + temperature."""
        content = f"{task.prompt}:{task.model}:{task.temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, task: SubTask) -> SubTaskResult | None:
        key = self._hash_task(task)
        return self._cache.get(key)
    
    def set(self, task: SubTask, result: SubTaskResult) -> None:
        key = self._hash_task(task)
        if len(self._cache) >= self._max_size:
            # LRU eviction
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result
```

### Task 4.3：人机协作委派

**目的：子 Agent 执行完后，让人类确认或修正**

在 `SubAgentNode.execute()` 中增加可选的"人工审核"模式：

```python
# 在聚合结果后、写入 state 前
if state.config.get("subagent_human_review", False):
    # 发送结果给前端等待确认
    await self._emit_for_review(state, result)
    # 前端返回确认或修改后的结果
    reviewed = await self._wait_for_review(state)
    result.final_output = reviewed
```

**前端实现：**
- 显示子 agent 的原始结果
- 提供"确认"、"重新执行某个子任务"、"修改后使用"三个按钮
- 默认关闭，通过配置开启

---

## 页 5：测试与验收（1 天）

### Task 5.1：调度器单元测试

创建 `tests/test_subagent_scheduler.py`：

```python
"""Tests for sub-agent scheduler."""
import pytest
import asyncio

from core.subagent.scheduler import TaskScheduler
from core.subagent.base import SubTask, AggregationStrategy


class TestTaskScheduler:
    @pytest.mark.asyncio
    async def test_sequential_tasks(self):
        """Test that tasks are executed within concurrency limit."""
        ...

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that slow tasks are timed out gracefully."""
        ...

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Test that one failing task doesn't affect others."""
        ...

    @pytest.mark.asyncio
    async def test_merge_aggregation(self):
        """Test MERGE strategy output."""
        ...

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry mechanism."""
        ...
```

### Task 5.2：聚合器测试

创建 `tests/test_subagent_aggregator.py`：

```python
"""Tests for result aggregators."""
from core.subagent.aggregator import merge_code_reviews, merge_research
from core.subagent.base import SubTaskResult


class TestAggregators:
    def test_merge_code_reviews_dedup(self):
        """Test that duplicate issues are deduplicated."""
        ...

    def test_merge_code_reviews_severity_priority(self):
        """Test that highest severity wins on duplicate."""
        ...

    def test_merge_research_complementary(self):
        """Test that complementary info is merged."""
        ...
```

### Task 5.3：集成测试

创建 `tests/test_subagent_integration.py`：

```python
"""Integration tests for sub-agent flow."""
import pytest


class TestSubAgentIntegration:
    @pytest.mark.asyncio
    async def test_code_review_parallel_flow(self):
        """End-to-end test: user asks for code review -> 3 parallel agents -> aggregated report."""
        ...

    @pytest.mark.asyncio
    async def test_research_parallel_flow(self):
        """End-to-end test: user asks for research -> 3 parallel agents -> merged report."""
        ...
```

### Task 5.4：性能基准

更新 `benchmarks/latency_benchmark.py`，增加子 Agent 模式测试：

```python
# 新增测试项
| 测试项 | 串行审查 | 3 子 Agent 并行 | 提升 |
|--------|---------|----------------|------|
| 代码审查(100行) | 12s | 5s | -58% |
| 深度研究 | 15s | 6s | -60% |
| 调试分析 | 10s | 4s | -60% |
```

**注意：** 这是理论值（受 API 并发限制），实际提升取决于网络延迟和模型速度。

---

## 页 6：最终验收

### 工程验收
```bash
# 1. 代码质量
ruff check core/subagent/ graph/nodes/ web/ tests/
ruff format --check core/subagent/ graph/nodes/ web/ tests/
mypy core/subagent/ --explicit-package-bases

# 2. 测试
pytest tests/test_subagent_*.py -v

# 3. 功能验证
# 发送代码审查请求，观察是否触发 3 个并行子 agent
# 检查输出是否有"安全/性能/风格"三个维度

# 4. 推送
git add -A
git commit -m "feat: add sub-agent delegation with parallel execution

details:
- SubAgent 抽象层：LLMSubAgent 实现
- TaskScheduler：并发控制、超时、重试、错误隔离
- 3 个并行场景：code_review、research、debug
- 结果聚合：MERGE/VOTE/RANK/BEST/CUSTOM 策略
- 聚合器：merge_code_reviews 去重+排序
- 动态任务拆分：TaskDecomposer（可选）
- 子 Agent 结果缓存：避免重复调用
- 前端并行进度展示
- Coordinator 路由：自动检测并行场景
- LangGraph 集成：subagent → responder 节点
- 完整单元测试和集成测试"
git push karen fix-root-code:main
```

### 文档更新
- [ ] 更新 ARCHITECTURE.md，添加"子 Agent 委派"章节
- [ ] 更新 PERFORMANCE.md，添加并行执行基准数据
- [ ] 更新 README.md，添加"并行代码审查"作为 killer feature 展示

---

## 附录：面试话术（从 Phase 4 提炼）

**Q: 子 Agent 系统是怎么设计的？**
A: 核心是 TaskScheduler + SubAgent 抽象。TaskScheduler 用 asyncio.Semaphore 控制并发（默认 3 个），每个子任务有独立的超时和重试。SubAgent 是抽象层，目前用 LLMSubAgent 实现，但预留了扩展（比如 CodeSubAgent 走沙箱执行）。关键是错误隔离：一个子 agent 失败不会拖累其他。

**Q: 怎么决定什么时候用子 Agent？**
A: Coordinator 节点里加了 `_should_delegate_to_subagents()` 判断。触发条件：用户明确要求代码审查且代码量 > 100 行、用户要求"深入研究"或"多角度分析"、当前是 review 模式且代码块较大。不满足条件就走普通模式，不增加延迟。

**Q: 并行执行结果怎么聚合？**
A: 定义了 5 种策略：MERGE（拼接）、VOTE（表决）、RANK（排序）、BEST（选最优）、CUSTOM（自定义函数）。代码审查用自定义聚合器，按 file:line 去重、按严重级别排序。研究任务按"技术→实践→批判"组织，并标记信息冲突点。

**Q: 并发控制怎么做的？**
A: 两层控制：全局 Semaphore 限制同时运行的子 agent 数量（默认 3），防止 API 限流；每个子任务有独立 timeout（默认 30s），超时不影响其他任务。还有指数退避重试机制。

**Q: 子 Agent 和技能系统怎么配合？**
A: 技能系统决定"角色和工具"，子 Agent 系统决定"怎么拆分和并行"。比如用户触发了 code-review 技能，Coordinator 检测到代码量大，决定走子 Agent 模式。3 个子 agent 都继承 code-review 技能的上下文，但各自聚焦不同维度（安全/性能/风格）。

**Q: 动态任务拆分是什么？**
A: 预设场景覆盖不了所有情况时，用轻量模型（light 档）先分析用户输入，自动拆成 2-4 个并行子任务。这是一个可选功能，默认关闭，防止增加不必要的延迟。
