"""Tests for sub-agent scheduler."""

import asyncio

import pytest

from core.subagent.base import AggregationStrategy, SubAgent, SubTask, SubTaskResult
from core.subagent.scheduler import TaskScheduler


class MockSubAgent(SubAgent):
    """Mock sub-agent for testing."""

    def __init__(self, delay: float = 0.0, fail: bool = False, output: str = "mock result") -> None:
        super().__init__("mock")
        self.delay = delay
        self.fail = fail
        self.output = output

    async def execute(self, task: SubTask) -> SubTaskResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return SubTaskResult(
                task_id=task.id,
                success=False,
                error="mock failure",
            )
        return SubTaskResult(
            task_id=task.id,
            success=True,
            output=f"{self.output} for {task.id}",
            latency_ms=self.delay * 1000,
        )


class TestTaskScheduler:
    @pytest.mark.asyncio
    async def test_run_single_task(self):
        """Single task should execute successfully."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(),
        )
        tasks = [SubTask(id="t1", name="test", prompt="hello")]
        result = await scheduler.run_parallel(tasks)

        assert len(result.results) == 1
        assert result.results[0].success
        assert result.results[0].task_id == "t1"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_run_multiple_tasks(self):
        """Multiple tasks should all execute."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(),
        )
        tasks = [SubTask(id=f"t{i}", name=f"task{i}", prompt=f"prompt{i}") for i in range(3)]
        result = await scheduler.run_parallel(tasks)

        assert len(result.results) == 3
        assert all(r.success for r in result.results)
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Concurrency limit should be respected."""
        running = 0
        max_running = 0

        class CountingAgent(SubAgent):
            async def execute(self, task: SubTask) -> SubTaskResult:
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0.1)
                running -= 1
                return SubTaskResult(task_id=task.id, success=True, output="ok")

        scheduler = TaskScheduler(
            max_concurrency=2,
            default_agent_factory=CountingAgent,
        )
        tasks = [SubTask(id=f"t{i}", name="t", prompt="p") for i in range(5)]
        await scheduler.run_parallel(tasks)

        assert max_running <= 2

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """One failing task should not affect others."""
        call_count = 0

        class ConditionalAgent(SubAgent):
            async def execute(self, task: SubTask) -> SubTaskResult:
                nonlocal call_count
                call_count += 1
                if task.id == "t2":
                    return SubTaskResult(task_id=task.id, success=False, error="fail")
                return SubTaskResult(task_id=task.id, success=True, output="ok")

        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=ConditionalAgent,
        )
        tasks = [
            SubTask(id="t1", name="t", prompt="p"),
            SubTask(id="t2", name="t", prompt="p"),
            SubTask(id="t3", name="t", prompt="p"),
        ]
        result = await scheduler.run_parallel(tasks)

        assert result.results[0].success
        assert not result.results[1].success
        assert result.results[2].success
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_merge_aggregation(self):
        """MERGE strategy should concatenate results."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(output="result"),
        )
        tasks = [
            SubTask(id="t1", name="t", prompt="p"),
            SubTask(id="t2", name="t", prompt="p"),
        ]
        result = await scheduler.run_parallel(tasks, strategy=AggregationStrategy.MERGE)

        assert "## t1" in result.final_output
        assert "## t2" in result.final_output
        assert "---" in result.final_output

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Slow tasks should be timed out."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(delay=10.0),
        )
        tasks = [SubTask(id="t1", name="t", prompt="p", timeout=0.1)]
        result = await scheduler.run_parallel(tasks)

        assert not result.results[0].success
        assert "Timeout" in result.results[0].error

    @pytest.mark.asyncio
    async def test_global_timeout(self):
        """Global timeout should mark remaining tasks as failed."""
        scheduler = TaskScheduler(
            max_concurrency=1,
            global_timeout=0.1,
            default_agent_factory=lambda: MockSubAgent(delay=1.0),
        )
        tasks = [
            SubTask(id="t1", name="t", prompt="p"),
            SubTask(id="t2", name="t", prompt="p"),
        ]
        result = await scheduler.run_parallel(tasks)

        # With concurrency=1 and global_timeout=0.1, at least one task should fail
        failed_count = sum(1 for r in result.results if not r.success)
        assert failed_count >= 1

    @pytest.mark.asyncio
    async def test_vote_aggregation(self):
        """VOTE strategy should pick majority."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(output="yes推荐"),
        )
        tasks = [SubTask(id="t1", name="t", prompt="p") for _ in range(3)]
        result = await scheduler.run_parallel(tasks, strategy=AggregationStrategy.VOTE)

        assert "是" in result.final_output or "yes" in result.final_output

    @pytest.mark.asyncio
    async def test_custom_aggregator(self):
        """CUSTOM strategy should use provided aggregator."""
        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: MockSubAgent(output="data"),
        )
        tasks = [SubTask(id="t1", name="t", prompt="p")]

        def custom_agg(results: list) -> str:
            return f"CUSTOM:{len(results)}"

        result = await scheduler.run_parallel(
            tasks,
            strategy=AggregationStrategy.CUSTOM,
            custom_aggregator=custom_agg,
        )

        assert result.final_output == "CUSTOM:1"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        """Empty task list should return empty result."""
        scheduler = TaskScheduler()
        result = await scheduler.run_parallel([])

        assert result.final_output == ""
        assert result.confidence == 0.0
        assert len(result.results) == 0
