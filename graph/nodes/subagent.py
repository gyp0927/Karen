"""Sub-agent delegation node for LangGraph.

并行派发多个子 Agent 执行任务，聚合结果后返回。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from core.subagent.aggregator import merge_code_reviews, merge_research
from core.subagent.base import AggregationStrategy, SubTask
from core.subagent.scheduler import TaskScheduler

logger = logging.getLogger(__name__)

# 子 Agent 场景检测关键词
_CODE_REVIEW_TRIGGERS = [
    "审查",
    "review",
    "检查代码",
    "code review",
    "看看这段代码",
    "审查一下",
]

_RESEARCH_TRIGGERS = [
    "深入研究",
    "多角度分析",
    "全面分析",
    "深度研究",
    "详细调研",
]

_DEBUG_TRIGGERS = [
    "调试",
    "debug",
    "排错",
    "定位问题",
    "分析错误",
]


def _extract_code_from_messages(messages: list) -> str:
    """从消息历史中提取代码块内容。"""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if not content:
            continue
        # 匹配 markdown 代码块
        matches = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
        if matches:
            return "\n\n".join(matches)
        # 也匹配单行代码
        if "def " in content or "class " in content or "import " in content:
            return content
    return ""


def _extract_error_from_messages(messages: list) -> str:
    """从消息历史中提取错误信息。"""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if not content:
            continue
        # 匹配常见错误模式
        if any(kw in content for kw in ("Error", "Exception", "Traceback", "错误", "异常")):
            return content
    return ""


def _detect_scenario(query: str, messages: list) -> str:
    """Detect which parallel scenario to use.

    Scenarios:
    - code_review: 用户要求代码审查（含代码块）
    - research: 用户要求深度研究
    - debug: 用户要求调试分析（含错误信息）
    - default: 通用并行
    """
    q = query.lower()

    # 代码审查：有明确关键词且消息中包含代码
    code = _extract_code_from_messages(messages)
    if code and any(t in q for t in _CODE_REVIEW_TRIGGERS):
        # 代码量判断（简单按行数）
        line_count = len(code.splitlines())
        if line_count >= 20:
            return "code_review"

    # 调试分析：有错误信息且含调试关键词
    error = _extract_error_from_messages(messages)
    if error and any(t in q for t in _DEBUG_TRIGGERS):
        return "debug"

    # 深度研究：明确的关键词
    if any(t in q for t in _RESEARCH_TRIGGERS):
        return "research"

    return "default"


def _generate_code_review_tasks(query: str, code: str) -> list[SubTask]:
    """Generate parallel code review sub-tasks.

    3 sub-agents:
    1. Security Agent
    2. Performance Agent
    3. Style Agent
    """
    code_block = f"```\n{code}\n```"
    return [
        SubTask(
            id="security-check",
            name="安全审查",
            prompt=f"""你是一个安全专家。审查以下代码的安全问题：

{code_block}

关注：SQL 注入、XSS、命令注入、敏感数据硬编码、权限绕过、不安全的反序列化。
用中文输出。对每个发现的问题，格式如下：
- [P0/P1/P2/P3] 文件:行号 - 问题描述 - 修复建议

如果没有发现问题，回答"未发现安全问题"。""",
            model="powerful",
            temperature=0.1,
            timeout=45.0,
            retry=1,
        ),
        SubTask(
            id="performance-check",
            name="性能审查",
            prompt=f"""你是一个性能优化专家。审查以下代码的性能问题：

{code_block}

关注：时间复杂度、空间复杂度、N+1 查询、内存泄漏、不必要的循环、I/O 阻塞。
用中文输出。对每个发现的问题，格式如下：
- [P0/P1/P2/P3] 文件:行号 - 问题描述 - 修复建议

如果没有发现问题，回答"未发现性能问题"。""",
            model="powerful",
            temperature=0.1,
            timeout=45.0,
            retry=1,
        ),
        SubTask(
            id="style-check",
            name="风格审查",
            prompt=f"""你是一个代码风格专家。审查以下代码的可读性和设计：

{code_block}

关注：命名规范、函数长度、注释质量、类型注解、设计模式、SOLID 原则、异常处理。
用中文输出。对每个发现的问题，格式如下：
- [P0/P1/P2/P3] 文件:行号 - 问题描述 - 修复建议

如果没有发现问题，回答"未发现风格问题"。""",
            model="default",
            temperature=0.2,
            timeout=30.0,
            retry=1,
        ),
    ]


def _generate_research_tasks(query: str) -> list[SubTask]:
    """Generate parallel research sub-tasks.

    3 sub-agents with different angles:
    1. Technical
    2. Practical
    3. Critical
    """
    return [
        SubTask(
            id="tech-research",
            name="技术分析",
            prompt=f"""研究以下问题的技术细节：{query}

从实现原理、核心概念、技术栈角度分析。
要求：引用具体的技术文档或标准（如果知道）。
用中文输出，200-400字。""",
            model="powerful",
            temperature=0.2,
            timeout=30.0,
            retry=1,
        ),
        SubTask(
            id="practical-research",
            name="实践分析",
            prompt=f"""研究以下问题的实际应用：{query}

从实际案例、最佳实践、行业采用角度分析。
要求：引用真实的应用案例或公司实践（如果知道）。
用中文输出，200-400字。""",
            model="powerful",
            temperature=0.2,
            timeout=30.0,
            retry=1,
        ),
        SubTask(
            id="critical-research",
            name="批判分析",
            prompt=f"""批判性地分析以下问题：{query}

从局限性、替代方案、常见陷阱、未来趋势角度分析。
要求：指出主流观点可能忽略的问题。
用中文输出，200-400字。""",
            model="powerful",
            temperature=0.2,
            timeout=30.0,
            retry=1,
        ),
    ]


def _generate_debug_tasks(query: str, error_info: str, code_context: str) -> list[SubTask]:
    """Generate parallel debug sub-tasks.

    3 sub-agents:
    1. Log Analyzer
    2. Code Tracer
    3. Test Validator
    """
    return [
        SubTask(
            id="log-analysis",
            name="日志分析",
            prompt=f"""分析以下错误日志，提取关键信息：

```
{error_info}
```

输出：
1. 错误类型和位置
2. 错误发生前的上下文
3. 可能的触发条件

用中文输出，简洁明了。""",
            model="default",
            temperature=0.1,
            timeout=20.0,
            retry=1,
        ),
        SubTask(
            id="code-trace",
            name="代码追踪",
            prompt=f"""追踪以下代码的执行路径，找出可能导致错误的位置：

错误信息：
```
{error_info}
```

代码上下文：
```
{code_context}
```

输出：
1. 可疑的代码行
2. 变量值变化路径
3. 边界条件分析

用中文输出。""",
            model="powerful",
            temperature=0.1,
            timeout=30.0,
            retry=1,
        ),
        SubTask(
            id="test-proposal",
            name="测试验证",
            prompt=f"""基于以下错误，设计测试用例来验证修复方案：

错误信息：
```
{error_info}
```

代码上下文：
```
{code_context}
```

输出：
1. 最小复现步骤
2. 单元测试代码（Python 示例）
3. 边界测试建议

用中文输出。""",
            model="default",
            temperature=0.2,
            timeout=30.0,
            retry=1,
        ),
    ]


def _get_strategy(scenario: str) -> AggregationStrategy:
    strategies: dict[str, AggregationStrategy] = {
        "code_review": AggregationStrategy.CUSTOM,
        "research": AggregationStrategy.CUSTOM,
        "debug": AggregationStrategy.MERGE,
        "default": AggregationStrategy.MERGE,
    }
    return strategies.get(scenario, AggregationStrategy.MERGE)


def _get_aggregator(scenario: str) -> Callable[[list], str] | None:
    aggregators: dict[str, Callable[[list], str]] = {
        "code_review": merge_code_reviews,
        "research": merge_research,
    }
    return aggregators.get(scenario)


def _format_summary(result, scenario: str) -> str:
    """Format aggregated result into a summary SystemMessage."""
    duration = sum(r.latency_ms for r in result.results) / 1000.0
    success_count = sum(1 for r in result.results if r.success)
    total = len(result.results)

    header = f"【子 Agent 并行分析结果】\n（{success_count}/{total} 个子任务成功，总耗时 {duration:.1f}s）\n\n"
    return header + result.final_output


async def subagent_node(state: dict, sid: str | None = None) -> dict:
    """子 Agent 委派节点 - 并行执行多个专家子任务。

    1. 检测场景（代码审查/深度研究/调试分析）
    2. 生成子任务
    3. 通过 TaskScheduler 并行调度
    4. 聚合结果并返回 SystemMessage
    """
    messages = state.get("messages", [])

    # 提取最新用户 query
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content or ""
            break

    if not query:
        return {"messages": [SystemMessage(content="【子 Agent 分析】未找到用户输入。")]}

    # 检测场景
    scenario = _detect_scenario(query, messages)
    logger.info(f"[subagent_node] Detected scenario: {scenario}")

    if scenario == "default":
        # 不触发子 agent，直接返回空
        return {"messages": []}

    # 生成子任务
    tasks: list[SubTask] = []
    if scenario == "code_review":
        code = _extract_code_from_messages(messages)
        tasks = _generate_code_review_tasks(query, code)
    elif scenario == "research":
        tasks = _generate_research_tasks(query)
    elif scenario == "debug":
        error = _extract_error_from_messages(messages)
        code = _extract_code_from_messages(messages)
        tasks = _generate_debug_tasks(query, error, code)

    if not tasks:
        return {"messages": []}

    # 执行并行子任务
    scheduler = TaskScheduler(max_concurrency=3)
    result = await scheduler.run_parallel(
        tasks=tasks,
        strategy=_get_strategy(scenario),
        custom_aggregator=_get_aggregator(scenario),
    )

    # 格式化聚合结果
    summary = _format_summary(result, scenario)

    return {"messages": [SystemMessage(content=summary, name="subagent")]}
