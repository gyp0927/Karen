import re
from collections.abc import Sequence
from operator import add
from typing import Annotated, TypedDict, cast

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """多Agent共享状态（加入认知状态）

    NOTE: ``messages`` 用 LangGraph 默认的 ``add`` reducer(list concat)。
    当前 fast_graph 为单 Responder 节点，搜索在 responder 内部异步并行处理，
    不再通过外部 LangGraph 节点并行写入，因此不存在排序问题。
    """

    messages: Annotated[Sequence[BaseMessage], add]
    active_agent: str | None
    task_context: dict | None
    human_input_required: bool
    base_model_response: str | None
    review_result: str | None
    awaiting_review: bool
    cognitive_state: dict | None  # 认知状态序列化后的字典


# 辅助函数：从state中提取和保存cognitive_state
def _get_cognitive_state_from_agent_state(state: AgentState) -> dict:
    return state.get("cognitive_state") or {}


def _make_result_with_cognitive_state(state: AgentState, result: dict) -> dict:
    """确保返回结果中包含cognitive_state"""
    cog = _get_cognitive_state_from_agent_state(state)
    if cog:
        result["cognitive_state"] = cog
    return result


def create_coordination_graph(coordinator_agent, researcher_agent, tool_caller, responder_agent):
    """协调模式：Coordinator → (Researcher → ToolCaller) → Responder

    Researcher 并行搜索 web + memory + knowledge，
    ToolCaller 按需执行非搜索工具，
    Responder 只负责生成最终回答。

    优化：Coordinator 判断为简单问候/闲聊时，直接路由到 Responder，
    跳过搜索和工具调用，减少 ~5-10s 网络等待。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("tool_caller", tool_caller)
    workflow.add_node("responder", responder_agent)

    workflow.set_entry_point("coordinator")

    _route_pattern = re.compile(r"\[route:\s*(subagent|responder|researcher)\s*\]")

    def _route_after_coordinator(state: AgentState) -> str:
        """解析 Coordinator 的输出，决定是否需要搜索或子 Agent 委派。"""
        for msg in reversed(state.get("messages", [])):
            content = getattr(msg, "content", "")
            match = _route_pattern.search(content)
            if match:
                return cast(str, match.group(1))
        # 默认走 researcher（保守策略）
        return "researcher"

    workflow.add_conditional_edges(
        "coordinator",
        _route_after_coordinator,
        {
            "researcher": "researcher",
            "responder": "responder",
            "subagent": "responder",
        },
    )

    workflow.add_edge("researcher", "tool_caller")
    workflow.add_edge("tool_caller", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


def create_fast_graph(responder_agent):
    """快速模式：单 Responder 节点。

    搜索和工具调用在 responder 内部异步处理，
    不再需要外部 web_searcher / memory_searcher / tool_caller 节点，
    减少 LangGraph 调度开销和并行等待时间。
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("responder", responder_agent)
    workflow.set_entry_point("responder")
    workflow.add_edge("responder", END)
    return workflow.compile()
