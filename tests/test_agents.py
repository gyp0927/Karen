"""Agent 节点、搜索、缓存、配置、记忆、RAG、模型路由、状态管理测试"""

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_responder():
    from agents.nodes import responder_node

    state = {
        "messages": [HumanMessage(content="你好")],
        "task_context": {"detected_language": "zh"},
    }
    result = await responder_node(state)
    assert "messages" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_web_searcher():
    from agents.search import web_searcher_agent

    result = await web_searcher_agent("什么是Python")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_memory_searcher():
    from agents.search import memory_searcher_agent

    result = await memory_searcher_agent("你好", user_id="")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_knowledge_searcher():
    from agents.search import knowledge_searcher_agent

    result = await knowledge_searcher_agent("测试")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_cache_basic():
    from core.cache import get_cache

    cache = get_cache()
    messages = [HumanMessage(content="测试缓存")]
    cache.set(messages, "test", "model", "这是一个测试响应，内容超过十个字符")
    result = cache.get(messages, "test", "model")
    assert result == "这是一个测试响应，内容超过十个字符"


@pytest.mark.asyncio
async def test_cache_stats():
    from core.cache import get_cache

    cache = get_cache()
    stats = cache.get_stats()
    assert "total_entries" in stats
    assert "enabled" in stats


@pytest.mark.asyncio
async def test_providers():
    from core.config import list_providers

    providers = list_providers()
    assert "siliconflow" in providers
    assert "deepseek" in providers


@pytest.mark.asyncio
async def test_model_name():
    from core.config import get_model_name

    model = get_model_name()
    assert model is not None
    assert len(model) > 0


@pytest.mark.asyncio
async def test_memory():
    from core.memory_client import _MEMORY_SYSTEM_AVAILABLE, get_memory_store

    if not _MEMORY_SYSTEM_AVAILABLE:
        pytest.skip("记忆系统依赖未安装")

    store = get_memory_store()
    try:
        await store.initialize()
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            pytest.skip("Qdrant 存储已被其他实例锁定")
        raise

    result = await store.save_memory(
        content="用户喜欢Python编程",
        memory_type="fact",
        source="test",
        importance=0.8,
    )
    assert "memory_id" in result

    memories = await store.retrieve("Python", top_k=5)
    assert isinstance(memories, list)


@pytest.mark.asyncio
async def test_rag():
    from core.rag import add_document, get_knowledge_stats, search_knowledge

    chunks = await add_document("Python是一种高级编程语言。", source="test_doc")
    assert chunks >= 0

    result = await search_knowledge("Python", top_k=3)
    assert isinstance(result, str)

    stats = get_knowledge_stats()
    assert "total_chunks" in stats


@pytest.mark.asyncio
async def test_model_router():
    from core.model_router import get_router

    router = get_router()
    result = router.route("你好，今天天气怎么样？", history_turns=0)
    assert "tier" in result
    assert result["tier"] in ("light", "default", "powerful")


@pytest.mark.asyncio
async def test_session_manager():
    from state.manager import SessionManager

    mgr = SessionManager(user_id="test_user")
    session_id = mgr.new_session("测试会话")
    assert session_id is not None
    mgr.add_human_message("你好")
    messages = mgr.get_messages()
    assert len(messages) == 1
    mgr.delete_session(session_id)
