"""搜索子 Agent - 联网搜索、记忆搜索、知识库搜索。"""

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# 子 Agent 的搜索超时（秒）。比单源搜索的内部超时大,留出 fallback 时间。
# 协调模式下 Coordinator 先耗 ~1-2s,叠加网页抓取(每页 8s),需留足余量。
WEB_SEARCH_TIMEOUT_S = 25.0
WEB_SEARCH_TIMEOUT_FAST_S = 1.5    # 快速模式： aggressively 短
# 记忆系统首次初始化需加载 embedding 模型(~10-20s),3s 必然超时。
MEMORY_SEARCH_TIMEOUT_S = 15.0
MEMORY_SEARCH_TIMEOUT_FAST_S = 3.0  # 快速模式：未初始化就快速跳过


async def _safe_search(
    search_fn: Callable,
    label: str,
    *args,
    success_check: Callable | None = None,
    **kwargs,
) -> str:
    """通用搜索包装器——统一异常处理和结果格式化。

    异常一律打 warning 日志，避免静默失败掩盖根因。
    """
    try:
        result = await search_fn(*args, **kwargs)
        if success_check:
            if success_check(result):
                return f"[{label}]\n\n{result}"
            logger.debug(f"{label}: 结果未通过 success_check，丢弃")
        elif result:
            return f"[{label}]\n\n{result}"
        else:
            logger.debug(f"{label}: 空结果")
    except asyncio.TimeoutError:
        logger.warning(f"{label} 超时")
    except TimeoutError:
        logger.warning(f"{label} 超时")
    except ConnectionError as e:
        logger.warning(f"{label} 网络错误: {e}")
    except ImportError as e:
        logger.warning(f"{label} 依赖缺失: {e}")
    except Exception as e:
        # 捕获 httpx.TimeoutException 等第三方库的超时异常
        if type(e).__name__.endswith("TimeoutException") or type(e).__name__.endswith("TimeoutError"):
            logger.warning(f"{label} 超时")
        else:
            logger.warning(f"{label} failed: {e}")
    return ""


async def web_searcher_agent(query: str, user_id: str = "", session_id: str = "", fast_mode: bool = False) -> str:
    """联网搜索子 Agent - 执行 DuckDuckGo 搜索并总结结果。"""
    from tools.search import search_and_summarize

    def _check(result: str) -> bool:
        return result and "未找到" not in result

    async def _search_with_timeout(q: str) -> str:
        timeout = WEB_SEARCH_TIMEOUT_FAST_S if fast_mode else WEB_SEARCH_TIMEOUT_S
        return await asyncio.wait_for(
            asyncio.to_thread(search_and_summarize, q, max_results=2, fast_mode=fast_mode),
            timeout=timeout,
        )

    return await _safe_search(_search_with_timeout, "联网搜索结果", query, success_check=_check)


async def memory_searcher_agent(query: str, user_id: str = "", session_id: str = "", fast_mode: bool = False) -> str:
    """记忆搜索子 Agent - 从自适应记忆系统检索相关记忆。

    session_id 非空时按会话隔离,只检索本会话写入的记忆;空串=不过滤(全局检索)。
    fast_mode: 快速模式下如果 embedding 模型未加载完则快速跳过，不阻塞等待。
    """
    from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE

    async def _do_search(q: str, uid: str, sess: str) -> str:
        if not _MEMORY_SYSTEM_AVAILABLE:
            return ""
        store = get_memory_store()

        # 快速模式：如果 embedding 模型还没加载完，不等待，直接跳过
        if fast_mode and not store.is_initialized():
            logger.debug("Memory store not initialized in fast mode, skipping")
            return ""

        # 首次调用需初始化（加载 embedding 模型 ~10-20s + 存储后端），
        # 与检索分离超时：初始化给 60s，检索本身 <1s。
        if not store.is_initialized():
            init_timeout = 10.0 if fast_mode else 60.0
            try:
                await asyncio.wait_for(store.initialize(), timeout=init_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Memory store initialization timed out (>{init_timeout}s)")
                return ""

        retrieve_timeout = 3.0 if fast_mode else 5.0
        try:
            memories = await asyncio.wait_for(
                store.retrieve(q, top_k=5, user_id=uid, source=sess),
                timeout=retrieve_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Memory retrieve timed out (>{retrieve_timeout}s)")
            return ""
        if memories:
            return store.format_memories_for_prompt(memories) or ""
        return ""

    return await _safe_search(_do_search, "记忆检索结果", query, user_id, session_id)


async def knowledge_searcher_agent(query: str, user_id: str = "") -> str:
    """知识库搜索子 Agent - 从 RAG 向量库检索相关文档。"""
    from core.rag import search_knowledge

    def _check(result: str) -> bool:
        return result and "知识库为空" not in result and "未启用" not in result

    return await _safe_search(
        lambda q: search_knowledge(q, top_k=3),
        "知识库检索结果",
        query,
        success_check=_check,
    )


async def run_parallel_search(state: dict) -> str:
    """根据 state 中的 mode 并行运行搜索子 Agent，返回整合后的搜索上下文。"""
    # 取最后一条 HumanMessage 的内容作为搜索关键词，避免 Coordinator
    # 的 route 回复被当成查询（协调模式下 messages[-1] 是 AIMessage）。
    from langchain_core.messages import HumanMessage
    user_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    if not user_message:
        user_message = state["messages"][-1].content if state.get("messages") else ""

    mode = state.get("task_context", {}).get("mode", "coordination")
    user_id = state.get("task_context", {}).get("user_id", "")
    session_id = state.get("task_context", {}).get("session_id", "")
    fast_mode = mode in ("fast", "planning")

    tasks = []
    if fast_mode:
        tasks.append(web_searcher_agent(user_message, fast_mode=True))
        tasks.append(memory_searcher_agent(user_message, user_id, session_id, fast_mode=True))
    else:
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id, session_id))
        tasks.append(knowledge_searcher_agent(user_message))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    search_parts = [r for r in results if isinstance(r, str) and r]
    return "\n\n".join(search_parts) if search_parts else ""
