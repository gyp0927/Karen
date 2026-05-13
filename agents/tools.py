"""工具调用子 Agent - 执行非搜索类工具（代码执行、计算等）。"""

import re

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
    """判断是否需要非搜索类工具调用（计算、代码等）。

    只触发真正的复杂计算或代码执行请求，
    避免简单数学问题（如'1+1'）走工具调用慢路径。
    """
    q = query.lower().strip()
    # 简单算术表达式（只有数字和 +-*/，无复杂运算）→ 不走工具
    if re.match(r"^[\d\s+\-*/().]+$", q) and len(q) <= 20 and any(c.isdigit() for c in q):
        return False
    # 复杂计算关键词（中英文）
    if any(kw in q for kw in [
        "计算", "等于多少", "平方", "次方", "百分比", "积分", "导数", "方程",
        "calculate", "computation", "compute", "square", "power", "percentage",
        "integral", "derivative", "equation", "solve",
    ]):
        return True
    # 代码执行请求（中英文）
    if any(kw in q for kw in [
        "运行代码", "执行代码", "算一下", "验证", "帮我算",
        "run code", "execute code", "execute python", "run python",
        "code execution", "python script",
    ]):
        return True
    return False


async def tool_caller_node(state: dict, sid: str | None = None) -> dict:
    """工具调用子节点 - 直接执行非搜索类工具，不走完整 Agent 流程。

    结果以 SystemMessage 注入 Responder 上下文。
    注意：Responder 节点内部也会检查工具需求并调用。为避免重复执行，
    如果 state 中已包含工具执行结果，则跳过。
    """
    # 从 state 中获取 sid（如果参数未提供）
    if sid is None:
        sid = state.get("task_context", {}).get("sid", "")

    query = state["messages"][-1].content if state["messages"] else ""

    if not _need_tool_call(query):
        return {"messages": []}

    # 检查是否已有工具执行结果，避免与 responder_node 重复执行
    existing_msgs = state.get("messages", [])
    for msg in existing_msgs:
        if getattr(msg, "name", None) == "tool_caller":
            return {"messages": []}
        if "【工具执行结果】" in getattr(msg, "content", ""):
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
