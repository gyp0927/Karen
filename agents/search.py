"""搜索子 Agent - 联网搜索、记忆搜索、知识库搜索。"""

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)


async def _safe_search(
    search_fn: Callable,
    label: str,
    *args,
    success_check: Callable = None,
    **kwargs,
) -> str:
    """通用搜索包装器——统一异常处理和结果格式化。"""
    try:
        result = await search_fn(*args, **kwargs)
        if success_check:
            if success_check(result):
                return f"[{label}]\n\n{result}"
        elif result:
            return f"[{label}]\n\n{result}"
    except asyncio.TimeoutError:
        pass
    except (TimeoutError, ConnectionError):
        pass
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"{label} failed: {e}")
    return ""


async def web_searcher_agent(query: str) -> str:
    """联网搜索子 Agent - 执行 DuckDuckGo 搜索并总结结果。"""
    from tools.search import search_and_summarize

    def _check(result: str) -> bool:
        return result and "未找到" not in result

    async def _search_with_timeout(q: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(search_and_summarize, q, max_results=2),
            timeout=6.0,
        )

    return await _safe_search(_search_with_timeout, "联网搜索结果", query, success_check=_check)


async def memory_searcher_agent(query: str, user_id: str = "") -> str:
    """记忆搜索子 Agent - 从自适应记忆系统检索相关记忆。"""
    from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE

    async def _do_search(q: str, uid: str) -> str:
        if not _MEMORY_SYSTEM_AVAILABLE:
            return ""
        store = get_memory_store()
        memories = await asyncio.wait_for(store.retrieve(q, top_k=5, user_id=uid), timeout=1.0)
        if memories:
            return store.format_memories_for_prompt(memories) or ""
        return ""

    return await _safe_search(_do_search, "记忆检索结果", query, user_id)


async def knowledge_searcher_agent(query: str) -> str:
    """知识库搜索子 Agent - 从 RAG 向量库检索相关文档。"""
    from core.rag import search_knowledge

    def _check(result: str) -> bool:
        return result and "知识库为空" not in result and "未启用" not in result

    return await _safe_search(
        lambda q: asyncio.to_thread(search_knowledge, q, top_k=3),
        "知识库检索结果",
        query,
        success_check=_check,
    )


async def run_parallel_search(state: dict) -> str:
    """根据 state 中的 mode 并行运行搜索子 Agent，返回整合后的搜索上下文。"""
    user_message = state["messages"][-1].content
    mode = state.get("task_context", {}).get("mode", "coordination")
    user_id = state.get("task_context", {}).get("user_id", "")

    tasks = []
    if mode in ("fast", "planning"):
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id))
    else:
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id))
        tasks.append(knowledge_searcher_agent(user_message))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    search_parts = [r for r in results if isinstance(r, str) and r]
    return "\n\n".join(search_parts) if search_parts else ""
