"""工具引擎 —— 让 LLM 自主决定调用外部工具

使用 LangChain 的 bind_tools() 实现标准的 Function Calling 流程：
1. LLM 第一次调用 → 可能输出 tool_calls
2. 执行工具 → 得到结果
3. 将工具结果加入 messages → LLM 第二次调用生成最终回答
"""
import asyncio
import logging
import os
from typing import Optional, Any

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import ToolMessage, AIMessage

logger = logging.getLogger(__name__)

# 代码执行默认关闭,需显式开启 ENABLE_CODE_EXECUTION=true
_CODE_EXEC_ENABLED = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() in ("true", "1", "yes")


# ========== 工具定义 ==========

@tool
async def web_search(query: str) -> str:
    """联网搜索。当你需要查询实时信息、新闻、事实、数据时使用此工具。

    Args:
        query: 搜索关键词或问题

    Returns:
        格式化的搜索结果文本，包含标题、链接和网页摘要
    """
    # 同步导入避免循环依赖;在 to_thread 中执行避免阻塞事件循环
    from tools.search import search_and_summarize
    try:
        return await asyncio.to_thread(search_and_summarize, query, max_results=2)
    except Exception as e:
        logger.warning(f"Web search tool failed: {e}")
        return f"[搜索失败: {e}]"


@tool
async def memory_search(query: str) -> str:
    """搜索长期记忆。当你需要回忆用户的个人信息、历史对话内容、偏好设置时使用此工具。

    Args:
        query: 记忆检索关键词

    Returns:
        相关记忆内容，如果没有找到则返回空字符串
    """
    try:
        from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE
        if not _MEMORY_SYSTEM_AVAILABLE:
            return ""
        store = get_memory_store()
        memories = await store.retrieve(query, top_k=3)
        if memories:
            formatted = store.format_memories_for_prompt(memories)
            return formatted or ""
    except Exception as e:
        logger.warning(f"Memory search tool failed: {e}")
    return ""


@tool
async def knowledge_search(query: str) -> str:
    """搜索知识库。当你需要查询已上传的文档、资料、RAG 内容时使用此工具。

    Args:
        query: 知识库检索关键词

    Returns:
        相关知识库内容，如果没有找到则返回空字符串
    """
    try:
        from core.rag import search_knowledge
        return await search_knowledge(query, top_k=3)
    except Exception as e:
        logger.warning(f"Knowledge search tool failed: {e}")
    return ""


@tool
async def execute_python(code: str) -> str:
    """执行 Python 代码。当你需要计算数学问题、处理数据、验证代码时使用此工具。

    Args:
        code: 要执行的 Python 代码字符串

    Returns:
        代码执行结果（stdout 输出或错误信息）
    """
    if not _CODE_EXEC_ENABLED:
        return "[代码执行已禁用:请在环境变量中设置 ENABLE_CODE_EXECUTION=true 才能启用]"
    # 安全限制：代码长度、黑名单关键词检查
    if len(code) > 5000:
        return "[代码执行被拒绝: 代码长度超过 5000 字符限制]"
    _FORBIDDEN_CODE_PATTERNS = (
        "import os", "import sys", "import subprocess", "import socket",
        "import urllib", "import http", "import ftplib", "import smtplib",
        "__import__", "eval(", "exec(", "open(", "compile(", "globals()", "locals()",
    )
    lowered = code.lower()
    for pat in _FORBIDDEN_CODE_PATTERNS:
        if pat in lowered:
            logger.warning(f"[CodeExec] 拒绝执行含危险模式的代码: {pat}")
            return f"[代码执行被拒绝: 检测到危险模式 '{pat}']"
    # 审计日志：记录完整代码内容
    logger.info(f"[CodeExec] 即将执行代码 ({len(code)} chars):\n{code}")
    try:
        from tools.code_executor import execute_python
        result = await asyncio.to_thread(execute_python, code)
        if result.get("success"):
            output = result.get("stdout", "")
            if output:
                return f"执行成功:\n{output}"
            return "执行成功（无输出）"
        else:
            error = result.get("error", "未知错误")
            return f"执行失败: {error}"
    except Exception as e:
        logger.warning(f"Code execution tool failed: {e}")
        return f"[代码执行失败: {e}]"


# ========== 工具注册表 ==========

DEFAULT_TOOLS: list[BaseTool] = [
    web_search,
    memory_search,
    knowledge_search,
]
if _CODE_EXEC_ENABLED:
    DEFAULT_TOOLS.append(execute_python)


def get_available_tools() -> list[BaseTool]:
    """获取所有可用工具列表"""
    return list(DEFAULT_TOOLS)


async def execute_tool_call(tool_call: dict) -> str:
    """执行单个 tool_call 并返回结果字符串。

    Args:
        tool_call: {"name": str, "args": dict, "id": str, ...}

    Returns:
        工具执行结果文本
    """
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    # 查找匹配的工具
    for t in DEFAULT_TOOLS:
        if t.name == tool_name:
            try:
                logger.info(f"[Tool] 执行 {tool_name}({tool_args})")
                # 所有工具统一走异步路径，避免同步调用阻塞事件循环
                if asyncio.iscoroutinefunction(t.ainvoke):
                    result = await t.ainvoke(tool_args)
                else:
                    result = await asyncio.to_thread(t.invoke, tool_args)
                logger.info(f"[Tool] {tool_name} 完成")
                return str(result) if result else ""
            except Exception as e:
                logger.exception(f"[Tool] {tool_name} 执行失败")
                return f"[工具执行错误: {e}]"

    logger.warning(f"[Tool] 未找到工具: {tool_name}")
    return f"[未找到工具: {tool_name}]"


async def run_tool_loop(
    llm,
    messages: list,
    tools: list[BaseTool],
    max_iterations: int = 3,
    sid: str = "",
    on_token: Optional[callable] = None,
) -> str:
    """执行工具调用循环。

    流程：
    1. 绑定工具到 LLM
    2. 调用 LLM（非流式）检查是否有 tool_calls
    3. 如果有 tool_calls，执行工具，将结果加入 messages
    4. 重复步骤 2-3，直到没有 tool_calls 或达到最大迭代次数
    5. 最终调用用流式输出给用户

    Returns:
        最终响应文本
    """
    llm_with_tools = llm.bind_tools(tools)
    current_messages = list(messages)

    for iteration in range(max_iterations):
        # 非流式调用，检查是否有 tool_calls
        response = await llm_with_tools.ainvoke(current_messages)

        # 如果没有 tool_calls，直接返回内容
        if not getattr(response, "tool_calls", None):
            # 最终输出：流式输出给用户
            return await _stream_final_response(
                llm_with_tools, current_messages, on_token, sid
            )

        # 有 tool_calls，执行工具
        logger.info(f"[ToolLoop] 第 {iteration + 1} 轮，检测到 {len(response.tool_calls)} 个工具调用")

        # 添加 LLM 的 tool_calls 消息到上下文
        current_messages.append(response)

        # 并行执行所有工具调用
        tool_tasks = []
        for tc in response.tool_calls:
            tool_tasks.append(execute_tool_call(tc))

        tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

        # 添加工具结果到上下文
        for tc, result in zip(response.tool_calls, tool_results):
            if isinstance(result, Exception):
                result = f"[工具执行异常: {result}]"
            current_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", ""),
            ))

    # 达到最大迭代次数，做最后一次流式输出
    logger.warning(f"[ToolLoop] 达到最大迭代次数 {max_iterations}，强制结束")
    return await _stream_final_response(
        llm_with_tools, current_messages, on_token, sid
    )


async def _stream_final_response(
    llm_with_tools,
    messages: list,
    on_token: Optional[callable],
    sid: str = "",
) -> str:
    """最终流式输出响应。"""
    from state.stop_flag import is_stopped

    response = ""
    async for chunk in llm_with_tools.astream(messages):
        if is_stopped(sid):
            break
        if chunk.content:
            response += chunk.content
            if on_token:
                on_token(chunk.content)
    return response
