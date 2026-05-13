from typing import TypedDict, Annotated, Sequence, Optional
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from cognition.types import ThinkingMode


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
    cognitive_state: Optional[dict]  # 认知状态序列化后的字典


# 辅助函数：从state中提取和保存cognitive_state
def _get_cognitive_state_from_agent_state(state: AgentState) -> dict:
    return state.get("cognitive_state") or {}


def _make_result_with_cognitive_state(state: AgentState, result: dict) -> dict:
    """确保返回结果中包含cognitive_state"""
    cog = _get_cognitive_state_from_agent_state(state)
    if cog:
        result["cognitive_state"] = cog
    return result


def create_multi_agent_graph(
    coordinator_agent,
    researcher_agent,
    responder_agent,
    reviewer_agent
):
    """构建多Agent协作图（带检查者）

    DEPRECATED: 该图存在 reviewer -> responder -> reviewer 无限循环风险。
    生产环境请使用 create_coordination_graph 或 create_fast_graph。
    保留此函数仅作兼容，不再维护。
    """

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("responder", responder_agent)
    workflow.add_node("reviewer", reviewer_agent)

    # 设置入口点
    workflow.set_entry_point("coordinator")

    # Coordinator 的分析结果作为上下文传给 Researcher，
    # 由 Researcher 内部根据 mode 决定启用哪些搜索子 Agent。
    workflow.add_edge("coordinator", "researcher")

    # researcher/responder -> reviewer 审查
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("responder", "reviewer")

    # reviewer -> 再次到responder处理审查意见 -> reviewer循环直到通过
    def route_from_reviewer(state: AgentState) -> str:
        # 从 reviewer 输出的消息中提取审查结果
        review_result = ""
        for msg in reversed(state["messages"]):
            if getattr(msg, "name", None) == "reviewer":
                review_result = msg.content.lower().strip()
                break

        # 使用明确的批准/拒绝标记
        approved_markers = ("[approved]", "[通过]", "✓")
        rejected_markers = ("[rejected]", "[不通过]", "needs revision", "需要修改")

        if any(m in review_result for m in rejected_markers):
            return "responder"
        if any(m in review_result for m in approved_markers):
            return END
        # 内容较长且有实际审查意见时，认为需要修改
        if len(review_result) > 50:
            return "responder"
        return END

    workflow.add_conditional_edges(
        "reviewer",
        route_from_reviewer,
        {
            "responder": "responder",
            END: END
        }
    )

    return workflow.compile()


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

    def _route_after_coordinator(state: AgentState) -> str:
        """解析 Coordinator 的输出，决定是否需要搜索。"""
        for msg in reversed(state.get("messages", [])):
            content = getattr(msg, "content", "")
            if "[route: responder]" in content:
                return "responder"
            if "[route: researcher]" in content:
                return "researcher"
        # 默认走 researcher（保守策略）
        return "researcher"

    workflow.add_conditional_edges(
        "coordinator",
        _route_after_coordinator,
        {
            "researcher": "researcher",
            "responder": "responder",
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


