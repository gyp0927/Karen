"""Sub-agent abstraction: define the interface for parallel task execution."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage


class AggregationStrategy(Enum):
    """How to combine results from multiple sub-agents."""

    MERGE = "merge"  # 拼接所有结果
    VOTE = "vote"  # 多数表决（用于判断题）
    RANK = "rank"  # 按分数排序
    BEST = "best"  # 选最好的一个
    CUSTOM = "custom"  # 自定义聚合函数


@dataclass
class SubTask:
    """A single sub-task dispatched to a sub-agent."""

    id: str
    name: str  # 子任务名称，如 "security-check"
    prompt: str  # 给子 agent 的完整 prompt
    model: str = "default"  # 子 agent 使用的模型
    temperature: float = 0.2
    max_tokens: int | None = None
    tools: list[str] = field(default_factory=list)  # 允许使用的工具
    timeout: float = 30.0  # 单个子任务超时（秒）
    retry: int = 1  # 失败重试次数


@dataclass
class SubTaskResult:
    """Result from a single sub-agent execution."""

    task_id: str
    success: bool
    output: str = ""  # 文本输出
    metadata: dict[str, Any] = field(default_factory=dict)  # 结构化数据
    error: str | None = None
    latency_ms: float = 0.0  # 执行耗时
    tokens_used: int = 0


@dataclass
class ParallelResult:
    """Aggregated result from multiple sub-agents."""

    strategy: AggregationStrategy
    results: list[SubTaskResult]
    final_output: str = ""  # 聚合后的最终输出
    confidence: float = 0.0  # 聚合置信度（0-1）
    metadata: dict[str, Any] = field(default_factory=dict)


class SubAgent(ABC):
    """Abstract base class for sub-agents."""

    def __init__(self, name: str = "", config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def execute(self, task: SubTask) -> SubTaskResult:
        """Execute a single sub-task and return result."""
        ...


class LLMSubAgent(SubAgent):
    """Default implementation: sub-agent backed by an LLM call."""

    async def execute(self, task: SubTask) -> SubTaskResult:
        """Execute sub-task via LLM call.

        1. Build messages from task.prompt
        2. Call LLM with specified temperature/max_tokens
        3. Parse output and return SubTaskResult
        """

        from agents.llm import get_llm

        t0 = time.time()
        try:
            llm = get_llm("")
            # 用 bind 覆盖子任务参数
            if task.max_tokens is not None:
                llm = cast(Any, llm.bind(max_tokens=task.max_tokens, temperature=task.temperature))
            else:
                llm = cast(Any, llm.bind(temperature=task.temperature))

            messages = [
                SystemMessage(
                    content=(
                        "你是凯伦团队的专家子代理。"
                        "专注完成分配的任务，输出简洁专业。"
                        "不要问候，不要解释你的角色，直接输出结果。"
                    )
                ),
                HumanMessage(content=task.prompt),
            ]

            response_parts: list[str] = []
            async for chunk in llm.astream(messages):
                if chunk.content:
                    response_parts.append(cast(str, chunk.content))

            output = "".join(response_parts)
            latency_ms = (time.time() - t0) * 1000

            return SubTaskResult(
                task_id=task.id,
                success=bool(output.strip()),
                output=output,
                latency_ms=latency_ms,
            )
        except TimeoutError:
            return SubTaskResult(
                task_id=task.id,
                success=False,
                error=f"Timeout after {task.timeout}s",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SubTaskResult(
                task_id=task.id,
                success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )
