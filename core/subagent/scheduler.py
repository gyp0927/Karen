"""Parallel task scheduler: dispatch sub-agents with concurrency control."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from .base import (
    AggregationStrategy,
    LLMSubAgent,
    ParallelResult,
    SubAgent,
    SubTask,
    SubTaskResult,
)

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Schedule and execute sub-tasks in parallel with controlled concurrency."""

    def __init__(
        self,
        max_concurrency: int = 3,
        global_timeout: float = 60.0,
        default_agent_factory: Callable[[], SubAgent] | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.global_timeout = global_timeout
        self.default_agent_factory = default_agent_factory or LLMSubAgent
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_parallel(
        self,
        tasks: list[SubTask],
        strategy: AggregationStrategy = AggregationStrategy.MERGE,
        custom_aggregator: Callable[[list[SubTaskResult]], str] | None = None,
    ) -> ParallelResult:
        """Execute all tasks in parallel with concurrency limit.

        1. Create semaphore-gathered coroutines
        2. Apply per-task timeout via asyncio.wait_for
        3. Catch exceptions, mark as failed but don't crash others
        4. Aggregate results based on strategy
        """
        if not tasks:
            return ParallelResult(strategy=strategy, results=[], final_output="", confidence=0.0)

        logger.info(f"[TaskScheduler] Dispatching {len(tasks)} tasks with concurrency={self.max_concurrency}")

        coros = [self._execute_single(t) for t in tasks]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=self.global_timeout,
            )
        except TimeoutError:
            logger.warning(f"[TaskScheduler] Global timeout after {self.global_timeout}s")
            # 超时后，已完成的任务保留结果，未完成的标记为失败
            results = []
            for _ in tasks:
                results.append(
                    SubTaskResult(
                        task_id="unknown",
                        success=False,
                        error=f"Global timeout after {self.global_timeout}s",
                    )
                )

        # 处理异常（转换为 SubTaskResult）
        normalized: list[SubTaskResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                normalized.append(
                    SubTaskResult(
                        task_id=tasks[i].id if i < len(tasks) else "unknown",
                        success=False,
                        error=f"{type(r).__name__}: {r}",
                    )
                )
            elif isinstance(r, SubTaskResult):
                normalized.append(r)
            else:
                normalized.append(
                    SubTaskResult(
                        task_id=tasks[i].id if i < len(tasks) else "unknown",
                        success=False,
                        error=f"Unexpected result type: {type(r)}",
                    )
                )

        final_output, confidence = self._aggregate(normalized, strategy, custom_aggregator)

        success_count = sum(1 for r in normalized if r.success)
        logger.info(
            f"[TaskScheduler] Completed {success_count}/{len(tasks)} tasks, "
            f"strategy={strategy.value}, confidence={confidence:.2f}"
        )

        return ParallelResult(
            strategy=strategy,
            results=normalized,
            final_output=final_output,
            confidence=confidence,
        )

    async def _execute_single(self, task: SubTask) -> SubTaskResult:
        """Execute one task with semaphore + timeout + retry."""
        agent = self.default_agent_factory()

        async with self._semaphore:
            for attempt in range(task.retry + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.execute(task),
                        timeout=task.timeout,
                    )
                    if attempt > 0:
                        logger.info(f"[TaskScheduler] Task {task.id} succeeded after {attempt + 1} attempts")
                    return result
                except TimeoutError:
                    if attempt == task.retry:
                        logger.warning(f"[TaskScheduler] Task {task.id} timeout after {task.timeout}s")
                        return SubTaskResult(
                            task_id=task.id,
                            success=False,
                            error=f"Timeout after {task.timeout}s",
                        )
                    logger.info(f"[TaskScheduler] Task {task.id} timeout, retrying ({attempt + 1}/{task.retry})")
                except Exception as e:
                    if attempt == task.retry:
                        logger.warning(f"[TaskScheduler] Task {task.id} failed: {e}")
                        return SubTaskResult(
                            task_id=task.id,
                            success=False,
                            error=f"{type(e).__name__}: {e}",
                        )
                    wait_time = 1 * (attempt + 1)
                    logger.info(
                        f"[TaskScheduler] Task {task.id} error, retrying in {wait_time}s ({attempt + 1}/{task.retry})"
                    )
                    await asyncio.sleep(wait_time)

        # 不可达，但类型检查需要
        return SubTaskResult(task_id=task.id, success=False, error="Unknown error")

    def _aggregate(
        self,
        results: list[SubTaskResult],
        strategy: AggregationStrategy,
        custom_aggregator: Callable[[list[SubTaskResult]], str] | None = None,
    ) -> tuple[str, float]:
        """Aggregate results into final output.

        Returns: (final_output, confidence)
        """
        if not results:
            return "", 0.0

        if strategy == AggregationStrategy.MERGE:
            return self._merge_aggregate(results)
        if strategy == AggregationStrategy.VOTE:
            return self._vote_aggregate(results)
        if strategy == AggregationStrategy.RANK:
            return self._rank_aggregate(results)
        if strategy == AggregationStrategy.BEST:
            return self._best_aggregate(results)
        if strategy == AggregationStrategy.CUSTOM and custom_aggregator:
            return custom_aggregator(results), 1.0

        return self._merge_aggregate(results)

    def _merge_aggregate(self, results: list[SubTaskResult]) -> tuple[str, float]:
        """按 task 顺序拼接，加分割线。"""
        parts: list[str] = []
        for r in results:
            if r.success:
                header = f"## {r.task_id}\n\n"
                parts.append(header + r.output)
            else:
                parts.append(f"## {r.task_id}\n\n[执行失败: {r.error}]")

        output = "\n\n---\n\n".join(parts)
        success_rate = sum(1 for r in results if r.success) / len(results)
        return output, success_rate

    def _vote_aggregate(self, results: list[SubTaskResult]) -> tuple[str, float]:
        """统计 success 比例作为 confidence，输出多数意见。"""
        success_count = sum(1 for r in results if r.success)
        total = len(results)
        confidence = success_count / total if total > 0 else 0.0

        # 取第一个成功的输出作为代表
        for r in results:
            if r.success:
                return r.output, confidence

        return "所有子任务均失败", 0.0

    def _rank_aggregate(self, results: list[SubTaskResult]) -> tuple[str, float]:
        """按 metadata.score 排序，返回最高分的。"""
        scored = [(r, r.metadata.get("score", 0.0)) for r in results if r.success]
        if not scored:
            return "所有子任务均失败", 0.0

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        confidence = scored[0][1]
        return best.output, confidence

    def _best_aggregate(self, results: list[SubTaskResult]) -> tuple[str, float]:
        """选 tokens_used 最少且 success 的结果（效率优先）。"""
        successful = [r for r in results if r.success]
        if not successful:
            return "所有子任务均失败", 0.0

        # 优先 tokens_used 少的，其次 latency_ms 短的
        best = min(successful, key=lambda r: (r.tokens_used or 0, r.latency_ms or 0))
        return best.output, 1.0
