"""子 Agent 委派系统 - 并行任务执行与结果聚合."""

from .aggregator import merge_code_reviews, merge_research, rank_by_confidence, vote_boolean
from .base import (
    AggregationStrategy,
    LLMSubAgent,
    ParallelResult,
    SubAgent,
    SubTask,
    SubTaskResult,
)
from .scheduler import TaskScheduler

__all__ = [
    "AggregationStrategy",
    "SubAgent",
    "LLMSubAgent",
    "SubTask",
    "SubTaskResult",
    "ParallelResult",
    "TaskScheduler",
    "merge_code_reviews",
    "merge_research",
    "vote_boolean",
    "rank_by_confidence",
]
