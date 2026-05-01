"""全功能测试脚本"""
import asyncio
import sys
import traceback

# 设置项目根目录
sys.path.insert(0, "E:/果冻ai-memory/多agent聊天")

results = []

def test(name):
    def decorator(func):
        async def wrapper():
            try:
                await func()
                results.append((name, "PASS", ""))
                print(f"  PASS: {name}")
            except Exception as e:
                results.append((name, "FAIL", str(e)))
                print(f"  FAIL: {name} - {e}")
                traceback.print_exc()
        return wrapper
    return decorator

# ========== 测试 1: 图编译 ==========
@test("图编译 - 协调模式")
async def test_coordination_graph():
    from graph.orchestrator import create_coordination_graph
    from agents.nodes import coordinator_node, researcher_node, responder_node
    from agents.tools import tool_caller_node
    graph = create_coordination_graph(coordinator_node, researcher_node, tool_caller_node, responder_node)
    assert graph is not None

@test("图编译 - 快速模式")
async def test_fast_graph():
    from graph.orchestrator import create_fast_graph
    from agents.search import web_searcher_agent, memory_searcher_agent
    from agents.tools import tool_caller_node
    from agents.nodes import responder_node
    graph = create_fast_graph(web_searcher_agent, memory_searcher_agent, tool_caller_node, responder_node)
    assert graph is not None

@test("图编译 - 审查模式")
async def test_multi_agent_graph():
    from graph.orchestrator import create_multi_agent_graph
    from agents.nodes import coordinator_node, researcher_node, responder_node, reviewer_node
    graph = create_multi_agent_graph(coordinator_node, researcher_node, responder_node, reviewer_node)
    assert graph is not None

# ========== 测试 2: Agent 节点 ==========
@test("Agent 节点 - Coordinator")
async def test_coordinator():
    from agents.nodes import coordinator_node
    from langchain_core.messages import HumanMessage
    state = {"messages": [HumanMessage(content="什么是量子计算？")]}
    result = await coordinator_node(state)
    assert "messages" in result
    assert len(result["messages"]) > 0

@test("Agent 节点 - Responder")
async def test_responder():
    from agents.nodes import responder_node
    from langchain_core.messages import HumanMessage
    state = {
        "messages": [HumanMessage(content="你好")],
        "task_context": {"detected_language": "zh"},
    }
    result = await responder_node(state)
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert "我是果冻ai" in result["messages"][-1].content

# ========== 测试 3: 搜索子 Agent ==========
@test("搜索子 Agent - 联网搜索")
async def test_web_searcher():
    from agents.search import web_searcher_agent
    result = await web_searcher_agent("什么是Python")
    # 可能返回空（网络问题），但至少不报错
    assert isinstance(result, str)

@test("搜索子 Agent - 记忆搜索")
async def test_memory_searcher():
    from agents.search import memory_searcher_agent
    result = await memory_searcher_agent("你好", user_id="")
    assert isinstance(result, str)

@test("搜索子 Agent - 知识库搜索")
async def test_knowledge_searcher():
    from agents.search import knowledge_searcher_agent
    result = await knowledge_searcher_agent("测试")
    assert isinstance(result, str)

# ========== 测试 4: 缓存 ==========
@test("缓存 - 基本读写")
async def test_cache():
    from core.cache import get_cache
    cache = get_cache()
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content="测试缓存")]
    cache.set(messages, "test", "model", "这是一个测试响应，内容超过十个字符")
    result = cache.get(messages, "test", "model")
    assert result == "这是一个测试响应，内容超过十个字符"

@test("缓存 - 统计信息")
async def test_cache_stats():
    from core.cache import get_cache
    cache = get_cache()
    stats = cache.get_stats()
    assert "total_entries" in stats
    assert "enabled" in stats

# ========== 测试 5: 配置系统 ==========
@test("配置 - 提供商列表")
async def test_providers():
    from core.config import list_providers, PROVIDER_CONFIG
    providers = list_providers()
    assert "siliconflow" in providers
    assert "deepseek" in providers

@test("配置 - 获取模型名称")
async def test_model_name():
    from core.config import get_model_name
    model = get_model_name()
    assert model is not None
    assert len(model) > 0

# ========== 测试 6: 记忆系统 ==========
@test("记忆系统 - 存储和检索")
async def test_memory():
    from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE
    if not _MEMORY_SYSTEM_AVAILABLE:
        print("    SKIP: 记忆系统依赖未安装")
        return
    store = get_memory_store()
    await store.initialize()
    # 保存记忆
    result = await store.save_memory(
        content="用户喜欢Python编程",
        memory_type="fact",
        source="test",
        importance=0.8,
    )
    assert "memory_id" in result
    # 检索记忆
    memories = await store.retrieve("Python", top_k=5)
    assert isinstance(memories, list)

# ========== 测试 7: RAG 知识库 ==========
@test("RAG - 基本操作")
async def test_rag():
    from core.rag import add_document, search_knowledge, get_knowledge_stats
    # 添加文档
    chunks = add_document("Python是一种高级编程语言。", source="test_doc")
    assert chunks >= 0
    # 搜索
    result = search_knowledge("Python", top_k=3)
    assert isinstance(result, str)
    # 统计
    stats = get_knowledge_stats()
    assert "total_chunks" in stats

# ========== 测试 8: 模型路由 ==========
@test("模型路由 - 复杂度分析")
async def test_model_router():
    from core.model_router import get_router
    router = get_router()
    result = router.route("你好，今天天气怎么样？", history_turns=0)
    assert "tier" in result
    assert result["tier"] in ("light", "default", "powerful")

# ========== 测试 9: 状态管理 ==========
@test("状态管理 - 会话管理器")
async def test_session_manager():
    from state.manager import SessionManager
    mgr = SessionManager(user_id="test_user")
    session_id = mgr.new_session("测试会话")
    assert session_id is not None
    mgr.add_human_message("你好")
    messages = mgr.get_messages()
    assert len(messages) == 1
    # 清理测试数据，避免污染 web 界面的会话列表
    mgr.delete_session(session_id)

# ========== 测试 10: 端到端聊天（协调模式）==========
@test("端到端 - 协调模式")
async def test_chat_coordination():
    from graph.orchestrator import create_coordination_graph
    from agents.nodes import coordinator_node, researcher_node
    from agents.tools import tool_caller_node
    from agents.nodes import responder_node
    from langchain_core.messages import HumanMessage

    graph = create_coordination_graph(coordinator_node, researcher_node, tool_caller_node, responder_node)
    initial_state = {
        "messages": [HumanMessage(content="你好")],
        "active_agent": None,
        "task_context": {
            "user_input": "你好",
            "detected_language": "zh",
            "user_id": "",
            "mode": "coordination",
        },
        "human_input_required": False,
        "base_model_response": None,
        "review_result": None,
        "awaiting_review": True,
    }

    final_state = None
    async for event in graph.astream(initial_state):
        for node_name, node_output in event.items():
            final_state = node_output

    assert final_state is not None
    assert "messages" in final_state
    assert len(final_state["messages"]) > 0
    response = final_state["messages"][-1].content
    assert "我是果冻ai" in response

# ========== 测试 11: 端到端聊天（快速模式）==========
@test("端到端 - 快速模式")
async def test_chat_fast():
    from graph.orchestrator import create_fast_graph
    from agents.search import web_searcher_agent, memory_searcher_agent
    from agents.tools import tool_caller_node
    from agents.nodes import responder_node
    from langchain_core.messages import HumanMessage

    graph = create_fast_graph(web_searcher_agent, memory_searcher_agent, tool_caller_node, responder_node)
    initial_state = {
        "messages": [HumanMessage(content="你好")],
        "active_agent": None,
        "task_context": {
            "user_input": "你好",
            "detected_language": "zh",
            "user_id": "",
            "mode": "fast",
        },
        "human_input_required": False,
        "base_model_response": None,
        "review_result": None,
        "awaiting_review": True,
    }

    final_state = None
    async for event in graph.astream(initial_state):
        for node_name, node_output in event.items():
            final_state = node_output

    assert final_state is not None
    assert "messages" in final_state
    assert len(final_state["messages"]) > 0
    response = final_state["messages"][-1].content
    assert "我是果冻ai" in response


async def main():
    print("=" * 60)
    print("果冻ai 全功能测试")
    print("=" * 60)

    tests = [
        test_coordination_graph,
        test_fast_graph,
        test_multi_agent_graph,
        test_coordinator,
        test_responder,
        test_web_searcher,
        test_memory_searcher,
        test_knowledge_searcher,
        test_cache,
        test_cache_stats,
        test_providers,
        test_model_name,
        test_memory,
        test_rag,
        test_model_router,
        test_session_manager,
        test_chat_coordination,
        test_chat_fast,
    ]

    for t in tests:
        await t()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    for name, status, msg in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}")
        if msg:
            print(f"       {msg}")
    print(f"\n总计: {passed} 通过, {failed} 失败, {len(results)} 项测试")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
