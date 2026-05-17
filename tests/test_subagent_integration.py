"""Integration tests for sub-agent delegation system.

Tests the full flow: scenario detection -> task generation -> scheduling -> aggregation.
"""

import pytest
from langchain_core.messages import HumanMessage

from core.subagent.base import AggregationStrategy, SubAgent, SubTask, SubTaskResult
from core.subagent.scheduler import TaskScheduler


class FakeLLMSubAgent(SubAgent):
    """Fake sub-agent that returns canned responses for testing."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        super().__init__("fake")
        self.responses = responses or {}
        self.call_log: list[SubTask] = []

    async def execute(self, task: SubTask) -> SubTaskResult:
        self.call_log.append(task)
        output = self.responses.get(task.id, f"result for {task.id}")
        return SubTaskResult(
            task_id=task.id,
            success=True,
            output=output,
        )


class TestScenarioDetection:
    def test_code_review_scenario(self):
        """含代码块和审查关键词应触发代码审查场景。"""
        from graph.nodes.subagent import _detect_scenario

        messages = [
            HumanMessage(content="请帮我审查这段代码\n```python\ndef foo():\n    pass\n" + "\n" * 20 + "```"),
        ]
        scenario = _detect_scenario("审查代码", messages)
        assert scenario == "code_review"

    def test_research_scenario(self):
        """深度研究关键词应触发研究场景。"""
        from graph.nodes.subagent import _detect_scenario

        messages = [HumanMessage(content="深入研究一下量子计算")]
        scenario = _detect_scenario("深入研究一下量子计算", messages)
        assert scenario == "research"

    def test_debug_scenario(self):
        """含错误信息和调试关键词应触发调试场景。"""
        from graph.nodes.subagent import _detect_scenario

        messages = [
            HumanMessage(content="帮我调试这个错误：Traceback: NameError"),
        ]
        scenario = _detect_scenario("debug", messages)
        assert scenario == "debug"

    def test_default_scenario(self):
        """普通查询不应触发子 agent。"""
        from graph.nodes.subagent import _detect_scenario

        messages = [HumanMessage(content="今天天气怎么样")]
        scenario = _detect_scenario("今天天气怎么样", messages)
        assert scenario == "default"


class TestCodeReviewFlow:
    @pytest.mark.asyncio
    async def test_full_code_review_pipeline(self):
        """完整代码审查流水线：生成3个子任务并并行执行。"""
        from graph.nodes.subagent import _generate_code_review_tasks

        code = "def hello():\n    print('hi')\n" + "\n" * 20
        tasks = _generate_code_review_tasks("审查代码", code)

        assert len(tasks) == 3
        assert tasks[0].id == "security-check"
        assert tasks[1].id == "performance-check"
        assert tasks[2].id == "style-check"

        # 用 fake agent 执行
        agent = FakeLLMSubAgent(
            {
                "security-check": "- P0 app.py:1 - 硬编码密钥",
                "performance-check": "- P1 app.py:2 - 低效循环",
                "style-check": "- P2 app.py:3 - 命名不规范",
            }
        )

        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: agent,
        )
        result = await scheduler.run_parallel(
            tasks,
            strategy=AggregationStrategy.CUSTOM,
            custom_aggregator=lambda rs: f"FOUND:{len(rs)}",
        )

        assert len(result.results) == 3
        assert all(r.success for r in result.results)
        assert result.final_output == "FOUND:3"


class TestResearchFlow:
    @pytest.mark.asyncio
    async def test_full_research_pipeline(self):
        """完整研究流水线：生成3个研究角度子任务。"""
        from graph.nodes.subagent import _generate_research_tasks

        tasks = _generate_research_tasks("分析 asyncio")

        assert len(tasks) == 3
        assert tasks[0].id == "tech-research"
        assert tasks[1].id == "practical-research"
        assert tasks[2].id == "critical-research"

        agent = FakeLLMSubAgent(
            {
                "tech-research": "asyncio 使用事件循环。",
                "practical-research": "FastAPI 大量使用 asyncio。",
                "critical-research": "在 CPU 密集场景表现不佳。",
            }
        )

        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: agent,
        )
        result = await scheduler.run_parallel(
            tasks,
            strategy=AggregationStrategy.CUSTOM,
            custom_aggregator=lambda rs: f"RESEARCH:{len(rs)}",
        )

        assert len(result.results) == 3
        assert all(r.success for r in result.results)


class TestDebugFlow:
    @pytest.mark.asyncio
    async def test_full_debug_pipeline(self):
        """完整调试流水线：生成3个调试子任务。"""
        from graph.nodes.subagent import _generate_debug_tasks

        tasks = _generate_debug_tasks("调试", "NameError: name 'x' is not defined", "x = 1")

        assert len(tasks) == 3
        assert tasks[0].id == "log-analysis"
        assert tasks[1].id == "code-trace"
        assert tasks[2].id == "test-proposal"

        agent = FakeLLMSubAgent(
            {
                "log-analysis": "错误类型: NameError",
                "code-trace": "变量 x 未定义",
                "test-proposal": "添加单元测试覆盖",
            }
        )

        scheduler = TaskScheduler(
            max_concurrency=3,
            default_agent_factory=lambda: agent,
        )
        result = await scheduler.run_parallel(tasks)

        assert len(result.results) == 3
        assert all(r.success for r in result.results)


class TestSubagentNode:
    @pytest.mark.asyncio
    async def test_no_user_query(self):
        """没有用户输入时应返回空提示。"""
        from graph.nodes.subagent import subagent_node

        state = {"messages": []}
        result = await subagent_node(state)
        assert len(result["messages"]) == 1
        assert "未找到用户输入" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_default_scenario_returns_empty(self):
        """default 场景不应触发子 agent。"""
        from graph.nodes.subagent import subagent_node

        state = {
            "messages": [HumanMessage(content="你好")],
        }
        result = await subagent_node(state)
        assert result["messages"] == []
