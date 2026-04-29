from langgraph.graph import StateGraph, END
from langgraph.types import Send

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


# 轻量级意图关键词：匹配到则跳过搜索，直接由 Responder 回复
_SKIP_SEARCH_PATTERNS = [
    # 问候
    "你好", "您好", "嗨", "hello", "hi", "hey",
    "早上好", "下午好", "晚上好", "早安", "晚安",
    # 感谢
    "谢谢", "感谢", "thx", "thanks", "thank you",
    # 告别
    "再见", "拜拜", "bye", "goodbye",
    # 确认/否定
    "是的", "没错", "对的", "好", "ok", "okay", "嗯", "哦",
    "不是", "不对", "不行", "不可以",
]


def _should_skip_search(query: str) -> bool:
    """轻量级意图检测：问候/闲聊/感谢等直接跳过搜索。"""
    q = query.strip().lower()
    # 短句（≤6字）且匹配关键词 → 跳过搜索
    if len(query.strip()) <= 6:
        for pat in _SKIP_SEARCH_PATTERNS:
            if pat in q:
                return True
    return False


def create_fast_graph(web_searcher, memory_searcher, responder_agent):
    """快速/计划模式：并行 WebSearcher + MemorySearcher → Responder

    无 Coordinator、无 Researcher LLM 调用，直接并行搜索后生成回答。
    比协调模式少 1-2 次 LLM 调用，速度最快。
    """

    async def web_searcher_node(state: AgentState) -> dict:
        """联网搜索节点"""
        query = state["messages"][-1].content
        result = await web_searcher(query)
        if result:
            from langchain_core.messages import SystemMessage
            return {"messages": [SystemMessage(content=result)]}
        return {"messages": []}

    async def memory_searcher_node(state: AgentState) -> dict:
        """记忆搜索节点"""
        query = state["messages"][-1].content
        user_id = state.get("task_context", {}).get("user_id", "")
        result = await memory_searcher(query, user_id)
        if result:
            from langchain_core.messages import SystemMessage
            return {"messages": [SystemMessage(content=result)]}
        return {"messages": []}

    def start_parallel_search(state: AgentState):
        """并行启动两个搜索子 Agent。

        对问候/闲聊等短句直接跳过搜索，进入 Responder，节省 2~5 秒。
        """
        query = state["messages"][-1].content
        if _should_skip_search(query):
            return Send("responder", state)
        return [
            Send("web_searcher", state),
            Send("memory_searcher", state),
        ]

    workflow = StateGraph(AgentState)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("memory_searcher", memory_searcher_node)
    workflow.add_node("responder", responder_agent)

    # __start__ 是 LangGraph 隐式起点，从这里分发给搜索节点或 responder
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
