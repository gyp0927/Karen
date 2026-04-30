from langgraph.graph import StateGraph, END
from langgraph.types import Send

from state.types import AgentState

# 认知系统导入
from cognition.intuition import get_intuition_engine
from cognition.types import ThinkingMode


# 辅助函数：从state中提取和保存cognitive_state
def _get_cognitive_state_from_agent_state(state: AgentState) -> dict:
    return state.get("cognitive_state") or {}


def _make_result_with_cognitive_state(state: AgentState, result: dict) -> dict:
    """确保返回结果中包含cognitive_state"""
    cog = _get_cognitive_state_from_agent_state(state)
    if cog:
        result["cognitive_state"] = cog
    return result


def route_from_coordinator(state: AgentState) -> str:
    """Coordinator 条件路由。

    所有模式都路由到 Researcher，由 Researcher 内部根据 mode 决定启用哪些搜索子 Agent。
    Coordinator 的分析结果仍作为上下文传给 Researcher。
    """
    # 总是路由到 Researcher，确保搜索子 Agent 被执行
    return "researcher"


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


def create_fast_graph(web_searcher, memory_searcher, responder_agent):
    """快速/计划模式：并行 WebSearcher + MemorySearcher → Responder

    无 Coordinator、无 Researcher LLM 调用，直接并行搜索后生成回答。
    基于三层意图识别（规则+上下文+LLM）智能决定是否跳过搜索。
    """

    async def _search_node(
        state: AgentState,
        search_fn: callable,
        skip_key: str,
        node_name: str,
    ) -> dict:
        """通用搜索节点——消除 web_searcher_node 和 memory_searcher_node 的重复。"""
        query = state["messages"][-1].content
        intent = state.get("task_context", {}).get("intent_result")
        if intent and intent.get(skip_key):
            return _make_result_with_cognitive_state(state, {"messages": []})

        user_id = state.get("task_context", {}).get("user_id", "")
        result = await search_fn(query, user_id) if node_name == "memory" else await search_fn(query)
        if result:
            from langchain_core.messages import SystemMessage
            return _make_result_with_cognitive_state(state, {
                "messages": [SystemMessage(content=result)]
            })
        return _make_result_with_cognitive_state(state, {"messages": []})

    async def web_searcher_node(state: AgentState) -> dict:
        return await _search_node(state, web_searcher, "skip_search", "web")

    async def memory_searcher_node(state: AgentState) -> dict:
        return await _search_node(state, memory_searcher, "skip_memory", "memory")

    def _route_from_intent(
        state: AgentState,
        skip_search: bool,
        skip_memory: bool,
        source: str,
        confidence: float = 0.0,
        intent: str = "",
    ):
        """根据意图结果执行统一路由——消除重复的路由逻辑。"""
        state["task_context"]["intent_result"] = {
            "intent": intent,
            "confidence": confidence,
            "skip_search": skip_search,
            "skip_memory": skip_memory,
            "skip_knowledge": skip_search,
            "source": source,
        }

        if skip_search and skip_memory:
            return Send("responder", state)

        sends = []
        if not skip_search:
            sends.append(Send("web_searcher", state))
        if not skip_memory:
            sends.append(Send("memory_searcher", state))
        if not sends:
            return Send("responder", state)
        return sends

    def start_parallel_search(state: AgentState):
        """基于直觉引擎+意图识别的双层智能路由。

        1. 直觉引擎高信心判断 → 直接路由（0ms）
        2. 现有意图分类器 → 辅助决策
        3. 不确定的 → 并行搜索
        """
        from core.intent import classify_intent_sync

        query = state["messages"][-1].content
        history = state.get("messages", [])
        history_turns = len(history) // 2

        # 第一层：直觉引擎（系统1——快速、经验驱动）
        intuition = get_intuition_engine()
        intuition_result = intuition.route_decision(query, history_turns)

        state.setdefault("task_context", {})
        state["task_context"]["intuition_result"] = intuition_result
        state["task_context"]["thinking_mode"] = intuition_result["thinking_mode"].value

        # 直觉高信心时，直接用直觉决策
        if intuition_result["intuition_confidence"] > 0.7:
            return _route_from_intent(
                state,
                skip_search=intuition_result["skip_search"],
                skip_memory=intuition_result["skip_memory"],
                source="intuition",
                confidence=intuition_result["intuition_confidence"],
                intent=intuition_result["route"],
            )

        # 第二层：回退到现有意图分类器（系统2——理性分析）
        result = classify_intent_sync(query, history=history)

        return _route_from_intent(
            state,
            skip_search=result.skip_search,
            skip_memory=result.skip_memory,
            source=result.source,
            confidence=result.confidence,
            intent=result.intent,
        )

    workflow = StateGraph(AgentState)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("memory_searcher", memory_searcher_node)
    workflow.add_node("responder", responder_agent)

    workflow.add_conditional_edges(
        "__start__",
        start_parallel_search,
        {"web_searcher": "web_searcher", "memory_searcher": "memory_searcher", "responder": "responder"}
    )
    workflow.add_edge("web_searcher", "responder")
    workflow.add_edge("memory_searcher", "responder")
    workflow.add_edge("responder", END)
    return workflow.compile()


# 兼容旧代码的别名
create_simple_responder_graph = create_fast_graph
