from typing import TypedDict, Annotated, Sequence, Optional
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from agents.tools import _need_tool_call

# 认知系统导入
from cognition.intuition import get_intuition_engine
from cognition.types import ThinkingMode


class AgentState(TypedDict):
    """多Agent共享状态（加入认知状态）

    NOTE: ``messages`` 用 LangGraph 默认的 ``add`` reducer(list concat)。
    fast_graph 多个 searcher 通过 Send 并行写时,完成时序不确定;但每个 searcher
    输出的 SystemMessage 已带 ``name``(search_web / search_memory / tool_caller),
    ``_run_agent`` 在传给 LLM 前会把相邻具名 SystemMessage 按 name 排序,
    保证 LLM 看到的上下文顺序确定可复现。
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


async def _search_node(
    state: AgentState,
    search_fn: callable,
    skip_key: str,
    name: str = "",
) -> dict:
    """通用搜索节点。

    Args:
        name: 给输出的 SystemMessage 打上 name 标签,供 _run_agent 在传给 LLM
              前按 name 排序,保证多个并行 searcher 结果顺序确定。
    """
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""
    intent = state.get("task_context", {}).get("intent_result")
    if intent and intent.get(skip_key):
        return _make_result_with_cognitive_state(state, {"messages": []})

    user_id = state.get("task_context", {}).get("user_id", "")
    session_id = state.get("task_context", {}).get("session_id", "")
    result = await search_fn(query, user_id, session_id=session_id)
    if result:
        from langchain_core.messages import SystemMessage
        # SystemMessage 必须显式指令"基于以上结果"，否则 LLM 会忽略检索文本回退到训练知识。
        # 与 coordination 模式 researcher_node 的 wrapping 保持一致。
        msg = SystemMessage(
            content=(
                f"{result}\n\n"
                "请基于以上搜索结果生成最终回答，"
                "涉及实时信息、最新数据、价格、天气等时效内容时，"
                "优先以此处的搜索结果为准，而非你的训练知识。"
            ),
        )
        if name:
            msg.name = name
        return _make_result_with_cognitive_state(state, {"messages": [msg]})
    return _make_result_with_cognitive_state(state, {"messages": []})


def _route_from_intent(
    state: AgentState,
    skip_search: bool,
    skip_memory: bool,
    need_tools: bool,
    source: str,
    confidence: float = 0.0,
    intent: str = "",
):
    """根据意图结果执行统一路由。

    NOTE: LangGraph 要求状态不可变。此处对 state 的修改发生在
    条件边路由函数内部，属于历史遗留。实际运行时通过 Send 传递的
    state 是各节点 reducer 后的新副本，因此不会造成并发问题。
    为遵循约定，仍使用浅拷贝避免原地修改共享引用。
    """
    task_context = dict(state.get("task_context", {}))
    task_context["intent_result"] = {
        "intent": intent,
        "confidence": confidence,
        "skip_search": skip_search,
        "skip_memory": skip_memory,
        "skip_knowledge": skip_search,
        "source": source,
    }
    state = dict(state)
    state["task_context"] = task_context
    # 深拷贝 messages 列表，避免共享引用导致 LangGraph reducer 异常
    import copy
    state["messages"] = copy.deepcopy(state.get("messages", []))

    if skip_search and skip_memory and not need_tools:
        return Send("responder", state)

    sends = []
    if not skip_search:
        sends.append(Send("web_searcher", state))
    if not skip_memory:
        sends.append(Send("memory_searcher", state))
    if need_tools:
        sends.append(Send("tool_caller", state))
    if not sends:
        return Send("responder", state)
    return sends


def create_fast_graph(web_searcher, memory_searcher, tool_caller, responder_agent):
    """快速/计划模式：并行 WebSearcher + MemorySearcher + ToolCaller → Responder

    无 Coordinator，三个子 Agent 按需并行执行，Responder 只负责生成。
    """

    async def web_searcher_node(state: AgentState) -> dict:
        return await _search_node(state, web_searcher, "skip_search", name="search_web")

    async def memory_searcher_node(state: AgentState) -> dict:
        return await _search_node(state, memory_searcher, "skip_memory", name="search_memory")

    async def tool_caller_node(state: AgentState) -> dict:
        return await tool_caller(state)

    def start_parallel_search(state: AgentState):
        """快速模式路由：按需并行启动子 Agent。"""
        from core.intent import classify_intent_sync

        query = state["messages"][-1].content
        history = state.get("messages", [])
        history_turns = len(history) // 2
        need_tools = _need_tool_call(query)

        intuition = get_intuition_engine()
        intuition_result = intuition.route_decision(query, history_turns)

        import copy
        state = copy.deepcopy(state)
        task_ctx = dict(state.get("task_context", {}))
        task_ctx["intuition_result"] = intuition_result
        task_ctx["thinking_mode"] = intuition_result["thinking_mode"].value
        state["task_context"] = task_ctx

        if intuition_result["intuition_confidence"] > 0.7:
            return _route_from_intent(
                state,
                skip_search=intuition_result["skip_search"],
                skip_memory=intuition_result["skip_memory"],
                need_tools=need_tools,
                source="intuition",
                confidence=intuition_result["intuition_confidence"],
                intent=intuition_result["route"],
            )

        result = classify_intent_sync(query, history=history)
        return _route_from_intent(
            state,
            skip_search=result.skip_search,
            skip_memory=result.skip_memory,
            need_tools=need_tools,
            source=result.source,
            confidence=result.confidence,
            intent=result.intent,
        )

    workflow = StateGraph(AgentState)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("memory_searcher", memory_searcher_node)
    workflow.add_node("tool_caller", tool_caller_node)
    workflow.add_node("responder", responder_agent)

    workflow.add_conditional_edges(
        "__start__",
        start_parallel_search,
        {
            "web_searcher": "web_searcher",
            "memory_searcher": "memory_searcher",
            "tool_caller": "tool_caller",
            "responder": "responder",
        }
    )
    workflow.add_edge("web_searcher", "responder")
    workflow.add_edge("memory_searcher", "responder")
    workflow.add_edge("tool_caller", "responder")
    workflow.add_edge("responder", END)
    return workflow.compile()


