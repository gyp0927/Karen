"""Tests for sub-agent aggregators."""

from core.subagent.aggregator import (
    merge_code_reviews,
    merge_research,
    rank_by_confidence,
    vote_boolean,
)
from core.subagent.base import SubTaskResult


class TestMergeCodeReviewsBoundary:
    """merge_code_reviews 边界条件测试"""

    def test_empty_results(self):
        """空结果列表应返回友好提示"""
        output = merge_code_reviews([])
        assert "未发现明显问题" in output

    def test_all_failed_results(self):
        """全部失败的子任务应返回提示"""
        results = [
            SubTaskResult(task_id="t1", success=False, error="fail"),
            SubTaskResult(task_id="t2", success=False, error="timeout"),
        ]
        output = merge_code_reviews(results)
        assert "未发现明显问题" in output

    def test_no_location_in_output(self):
        """输出不含 severity 标记时应返回未发现明显问题"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="总体设计良好"),
        ]
        output = merge_code_reviews(results)
        # 不含 P0/P1/P2/P3/严重/中等/轻微/建议标记的内容会被忽略
        assert "未发现明显问题" in output


class TestMergeCodeReviews:
    def test_deduplicate_by_location(self):
        """同一位置的多条问题应去重，保留最高严重级别。"""
        results = [
            SubTaskResult(
                task_id="security",
                success=True,
                output="- P0 app.py:42 - SQL注入风险 - 使用参数化查询",
            ),
            SubTaskResult(
                task_id="style",
                success=True,
                output="- P2 app.py:42 - 变量命名不规范 - 改为 snake_case",
            ),
        ]
        output = merge_code_reviews(results)
        # 应该只保留 P0
        assert "P0" in output
        assert "SQL注入" in output
        # P2 的同一位置应该被去重
        assert output.count("app.py:42") == 1

    def test_severity_sorting(self):
        """问题应按严重级别排序（P0 在前）。"""
        results = [
            SubTaskResult(
                task_id="style",
                success=True,
                output="- P3 utils.py:10 - 缺少类型注解",
            ),
            SubTaskResult(
                task_id="security",
                success=True,
                output="- P0 main.py:5 - 硬编码密钥",
            ),
        ]
        output = merge_code_reviews(results)
        # P0 应该在输出中排在前面
        p0_pos = output.find("P0")
        p3_pos = output.find("P3")
        assert p0_pos < p3_pos

    def test_failed_task_ignored(self):
        """失败的子任务应被忽略。"""
        results = [
            SubTaskResult(
                task_id="security",
                success=False,
                error="timeout",
            ),
            SubTaskResult(
                task_id="style",
                success=True,
                output="- P2 main.py:1 - 缺少模块 docstring",
            ),
        ]
        output = merge_code_reviews(results)
        assert "P2" in output
        assert "timeout" not in output

    def test_no_issues_found(self):
        """没有发现问题时应返回友好提示。"""
        results = [
            SubTaskResult(
                task_id="security",
                success=True,
                output="未发现安全问题。",
            ),
            SubTaskResult(
                task_id="style",
                success=True,
                output="未发现风格问题。",
            ),
        ]
        output = merge_code_reviews(results)
        assert "未发现明显问题" in output

    def test_chinese_severity_labels(self):
        """支持中文严重级别标签。"""
        results = [
            SubTaskResult(
                task_id="security",
                success=True,
                output="- 严重 main.py:1 - 权限绕过",
            ),
        ]
        output = merge_code_reviews(results)
        assert "P0" in output or "严重" in output


class TestMergeResearchBoundary:
    """merge_research 边界条件测试"""

    def test_empty_results(self):
        """空结果列表应返回提示"""
        output = merge_research([])
        assert "所有研究子任务均失败" in output

    def test_single_result(self):
        """单条结果应正常输出"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="唯一结果"),
        ]
        output = merge_research(results)
        assert "唯一结果" in output


class TestMergeResearch:
    def test_merge_multiple_angles(self):
        """多个研究角度的结果应被组织在一起。"""
        results = [
            SubTaskResult(
                task_id="tech-research",
                success=True,
                output="Python 使用 GIL 管理线程。",
            ),
            SubTaskResult(
                task_id="practical-research",
                success=True,
                output="Instagram 使用 Python 处理大量请求。",
            ),
        ]
        output = merge_research(results)
        assert "技术分析" in output
        assert "实践分析" in output
        assert "GIL" in output
        assert "Instagram" in output

    def test_conflict_detection(self):
        """互斥关键词应触发冲突检测。"""
        results = [
            SubTaskResult(
                task_id="tech-research",
                success=True,
                output="推荐使用 asyncio。",
            ),
            SubTaskResult(
                task_id="critical-research",
                success=True,
                output="不推荐在 CPU 密集场景使用 asyncio。",
            ),
        ]
        output = merge_research(results)
        assert "冲突" in output or "注意" in output

    def test_all_failed(self):
        """所有子任务失败时应返回提示。"""
        results = [
            SubTaskResult(task_id="t1", success=False, error="fail"),
            SubTaskResult(task_id="t2", success=False, error="fail"),
        ]
        output = merge_research(results)
        assert "所有研究子任务均失败" in output


class TestVoteBooleanBoundary:
    """vote_boolean 边界条件测试"""

    def test_empty_results(self):
        """空结果列表应返回提示"""
        output = vote_boolean([])
        assert "无法达成有效表决" in output

    def test_tie_votes(self):
        """赞成反对各半时应合理处理"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="是的，推荐"),
            SubTaskResult(task_id="t2", success=True, output="不，不推荐"),
        ]
        output = vote_boolean(results)
        # 平局时应有明确提示
        assert "是" in output or "否" in output or "分歧" in output or "无法" in output

    def test_mixed_success(self):
        """部分成功部分失败时应基于成功结果表决"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="yes"),
            SubTaskResult(task_id="t2", success=False, error="fail"),
            SubTaskResult(task_id="t3", success=True, output="yes"),
        ]
        output = vote_boolean(results)
        assert "是" in output or "yes" in output


class TestVoteBoolean:
    def test_majority_yes(self):
        """多数赞成时应返回是。"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="是的，推荐使用。"),
            SubTaskResult(task_id="t2", success=True, output="yes, 这个方案很好。"),
            SubTaskResult(task_id="t3", success=True, output="不太确定。"),
        ]
        output = vote_boolean(results)
        assert "是" in output

    def test_majority_no(self):
        """多数反对时应返回否。"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="不推荐这个方案。"),
            SubTaskResult(task_id="t2", success=True, output="no, 有风险。"),
            SubTaskResult(task_id="t3", success=True, output="否，不可行。"),
        ]
        output = vote_boolean(results)
        assert "否" in output

    def test_all_failed(self):
        """所有子任务失败时应返回提示。"""
        results = [
            SubTaskResult(task_id="t1", success=False, error="fail"),
        ]
        output = vote_boolean(results)
        assert "无法达成有效表决" in output


class TestRankByConfidenceBoundary:
    """rank_by_confidence 边界条件测试"""

    def test_empty_results(self):
        """空结果列表应返回提示"""
        output = rank_by_confidence([])
        assert "所有子任务均失败" in output

    def test_no_score_metadata(self):
        """缺少 score metadata 时应基于成功状态处理"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="结果A"),
            SubTaskResult(task_id="t2", success=True, output="结果B"),
        ]
        output = rank_by_confidence(results)
        assert "结果A" in output or "结果B" in output

    def test_same_score(self):
        """相同 score 时应保留所有结果"""
        results = [
            SubTaskResult(task_id="t1", success=True, output="结果A", metadata={"score": 0.5}),
            SubTaskResult(task_id="t2", success=True, output="结果B", metadata={"score": 0.5}),
        ]
        output = rank_by_confidence(results)
        # 最优结果中包含第一个结果的 output
        assert "结果A" in output
        # 其他候选只显示 task_id，不显示 output
        assert "t2" in output


class TestRankByConfidence:
    def test_rank_by_score(self):
        """应按 metadata score 排序，返回最高分的。"""
        results = [
            SubTaskResult(
                task_id="low",
                success=True,
                output="低置信度结果",
                metadata={"score": 0.3},
            ),
            SubTaskResult(
                task_id="high",
                success=True,
                output="高置信度结果",
                metadata={"score": 0.9},
            ),
        ]
        output = rank_by_confidence(results)
        assert "高置信度结果" in output
        assert "0.9" in output
        assert "其他候选" in output

    def test_all_failed(self):
        """所有子任务失败时应返回提示。"""
        results = [
            SubTaskResult(task_id="t1", success=False, error="fail"),
        ]
        output = rank_by_confidence(results)
        assert "所有子任务均失败" in output
