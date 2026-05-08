"""工具调用子 Agent - 执行非搜索类工具（代码执行、计算等）。"""

from langchain_core.messages import SystemMessage

from agents.llm import get_llm


TOOL_CALLER_PROMPT = """你是 ToolCaller（工具调用专家）。

你的职责：
1. 分析用户问题是否需要调用非搜索类工具（如计算、代码执行）
2. 如果需要，调用合适的工具获取结果
3. 将工具执行结果以简洁的方式返回

可用工具：
- execute_python: 执行 Python 代码，用于数学计算、数据处理、验证代码等

注意：
- 不要调用搜索类工具（联网搜索、记忆搜索、知识库搜索），这些由其他 Agent 处理
- 如果不需要调用工具，返回空即可
- 工具执行结果要简洁，不要过多解释"""


def _need_tool_call(query: str) -> bool:
    """判断是否需要非搜索类工具调用（计算、代码等）。"""
    q = query.lower()
    if any(kw in q for kw in ["计算", "等于多少", "+", "*", "/", "平方", "次方", "百分比"]):
        return True
    if any(kw in q for kw in ["运行代码", "执行代码", "算一下", "验证"]):
        return True
    return False


async def tool_caller_node(state: dict, sid: str | None = None) -> dict:
    """工具调用子节点 - 直接执行非搜索类工具，不走完整 Agent 流程。

    结果以 SystemMessage 注入 Responder 上下文。
    """
    # 从 state 中获取 sid（如果参数未提供）
    if sid is None:
        sid = state.get("task_context", {}).get("sid", "")

    query = state["messages"][-1].content if state["messages"] else ""

    if not _need_tool_call(query):
        return {"messages": []}

    from cognition.tool_engine import execute_python, run_tool_loop

    llm = get_llm(sid or "")
    tools = [execute_python]
    messages = [SystemMessage(content=TOOL_CALLER_PROMPT)] + list(state["messages"])

    response = await run_tool_loop(llm, messages, tools, max_iterations=2, sid=sid or "")

    if response:
        return {"messages": [SystemMessage(
            content=f"【工具执行结果】\n\n{response}",
            name="tool_caller",
        )]}
    return {"messages": []}
