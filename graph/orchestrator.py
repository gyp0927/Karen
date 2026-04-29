from langgraph.graph import StateGraph, END

from state.types import AgentState


_RESEARCH_KEYWORDS = [
    "search", "find", "research", "look up",
    "what is", "who is", "how do", "why do",
    "介绍一下", "什么是", "怎么", "如何", "为什么",
    "查询", "搜索", "调研", "解释", "说明",
    "compare", "difference", "区别", "对比", "vs",
    "history", "历史", "background", "背景",
    "latest", "最新", "news", "新闻", "趋势", "trend"
]


def route_from_coordinator(state: AgentState) -> str:
    """Coordinator 条件路由：优先根据 coordinator 输出的路由标记判断。"""
    last_msg = state["messages"][-1].content
    lower = last_msg.lower()

    # 优先检查 coordinator 明确输出的路由标记
    if "[route: researcher]" in lower:
        return "researcher"
    if "[route: responder]" in lower:
        return "responder"

    # 回退：关键词匹配（兼容旧行为）
    if any(kw in lower for kw in _RESEARCH_KEYWORDS):
        return "researcher"
    return "responder"


def create_multi_agent_graph(
    coordinator_agent,
    researcher_agent,
    responder_agent,
    reviewer_agent
):
    """构建多Agent协作图（带检查者）"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("responder", responder_agent)
    workflow.add_node("reviewer", reviewer_agent)

    # 设置入口点
    workflow.set_entry_point("coordinator")

    # coordinator -> researcher 或 responder
    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "researcher": "researcher",
            "responder": "responder"
        }
    )

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


def create_coordination_graph(coordinator_agent, researcher_agent, responder_agent):
    """创建协调+研究+响应的协作图（Coordinator判断是否需要Researcher）"""
    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("responder", responder_agent)

    workflow.set_entry_point("coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {"researcher": "researcher", "responder": "responder"}
    )

    workflow.add_edge("researcher", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


def create_fast_graph(coordinator_agent, researcher_agent, responder_agent):
    """快速/计划模式：Coordinator → Researcher(2搜索) → Responder

    Researcher 下并行 2 个搜索子 Agent（联网 + 记忆）。
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("responder", responder_agent)
    workflow.set_entry_point("coordinator")
    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {"researcher": "researcher", "responder": "responder"}
    )
    workflow.add_edge("researcher", "responder")
    workflow.add_edge("responder", END)
    return workflow.compile()


# 兼容旧代码的别名
create_simple_responder_graph = create_fast_graph
