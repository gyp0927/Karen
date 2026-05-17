"""全功能测试脚本"""

import asyncio
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

# 设置项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

results: list[dict[str, Any]] = []


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
@test("图编译 - 快速模式")
async def test_fast_graph():
    from agents.nodes import responder_node
    from graph.orchestrator import create_fast_graph

    graph = create_fast_graph(responder_node)
    assert graph is not None


@test("图编译 - 协调模式")
async def test_coordination_graph():
    from agents.nodes import coordinator_node, researcher_node, responder_node
    from agents.tools import tool_caller_node
    from graph.orchestrator import create_coordination_graph

    graph = create_coordination_graph(coordinator_node, researcher_node, tool_caller_node, responder_node)
    assert graph is not None


# ========== 测试 2: Agent 节点 ==========
@test("Agent 节点 - Responder")
async def test_responder():
    from langchain_core.messages import HumanMessage

    from agents.nodes import responder_node

    state = {
        "messages": [HumanMessage(content="你好")],
        "task_context": {"detected_language": "zh"},
    }
    result = await responder_node(state)
    assert "messages" in result
    assert len(result["messages"]) > 0
    # 不再断言"我是凯伦"前缀(已从 prompt 中清掉)


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
    from core.config import list_providers

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
    from core.memory_client import _MEMORY_SYSTEM_AVAILABLE, get_memory_store

    if not _MEMORY_SYSTEM_AVAILABLE:
        print("    SKIP: 记忆系统依赖未安装")
        return
    store = get_memory_store()
    try:
        await store.initialize()
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            print("    SKIP: Qdrant 存储已被其他实例锁定")
            return
        raise
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
    from core.rag import add_document, get_knowledge_stats, search_knowledge

    # 添加文档
    chunks = await add_document("Python是一种高级编程语言。", source="test_doc")
    assert chunks >= 0
    # 搜索
    result = await search_knowledge("Python", top_k=3)
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


# ========== 测试 10: 技能系统 ==========
@test("技能系统 - SkillLoader 加载")
async def test_skill_loader():
    from core.skills import get_skill_loader

    loader = get_skill_loader()
    skills = loader.list_skills()
    assert len(skills) >= 3  # 至少加载了 code-review, debug, refactor
    skill_names = {s.name for s in skills}
    assert "code-review" in skill_names
    assert "debug" in skill_names
    assert "refactor" in skill_names


@test("技能系统 - 触发器匹配")
async def test_skill_trigger_match():
    from core.skills import match_skill

    # 完全匹配
    skill = match_skill("代码审查")
    assert skill is not None
    assert skill.name == "code-review"

    # contains 匹配
    skill = match_skill("请帮我 review this code")
    assert skill is not None
    assert skill.name == "code-review"

    # 未匹配
    skill = match_skill("今天天气怎么样")
    assert skill is None


@test("技能系统 - 意图分类集成")
async def test_skill_intent_integration():
    from core.intent import IntentType, classify_intent_sync

    result = classify_intent_sync("帮我看看这段代码")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "code-review"
    assert result.source == "skill"

    # Skill 退出检测
    result = classify_intent_sync("完成了", active_skill="code-review")
    assert result.source == "skill_exit"


# ========== 测试 11: 端到端聊天（快速模式）==========
@test("子 Agent - 调度器并发")
async def test_subagent_scheduler():
    from core.subagent.scheduler import TaskScheduler

    scheduler = TaskScheduler(max_concurrency=2, default_agent_factory=lambda: None)
    assert scheduler.max_concurrency == 2


@test("子 Agent - 聚合器代码审查")
async def test_subagent_aggregator():
    from core.subagent.aggregator import merge_code_reviews
    from core.subagent.base import SubTaskResult

    results = [
        SubTaskResult(task_id="sec", success=True, output="- P0 app.py:1 - SQL注入"),
        SubTaskResult(task_id="perf", success=True, output="- P1 app.py:2 - 低效循环"),
    ]
    output = merge_code_reviews(results)
    assert "P0" in output
    assert "SQL注入" in output


@test("子 Agent - 场景检测")
async def test_subagent_scenario():
    from langchain_core.messages import HumanMessage

    from graph.nodes.subagent import _detect_scenario

    messages = [HumanMessage(content="深入研究一下AI")]
    scenario = _detect_scenario("深入研究一下AI", messages)
    assert scenario == "research"


@test("端到端 - 快速模式")
async def test_chat_fast():
    from langchain_core.messages import HumanMessage

    from agents.nodes import responder_node
    from graph.orchestrator import create_fast_graph

    graph = create_fast_graph(responder_node)
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
    assert response  # 不再要求 "我是凯伦"


async def main():
    print("=" * 60)
    print("凯伦 全功能测试")
    print("=" * 60)

    # ---- 后台预热记忆系统（首次加载 embedding 模型约 10-20s） ----
    _memory_warmup_done = threading.Event()

    def _bg_memory_warmup():
        try:
            from core.memory_client import _MEMORY_SYSTEM_AVAILABLE, get_memory_store

            if _MEMORY_SYSTEM_AVAILABLE:
                asyncio.run(get_memory_store().initialize())
                print("  记忆系统预热完成")
        except Exception as e:
            print(f"  记忆系统预热失败: {e}")
        finally:
            _memory_warmup_done.set()

    threading.Thread(target=_bg_memory_warmup, daemon=True, name="mem-warmup").start()

    tests = [
        test_fast_graph,
        test_coordination_graph,
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
        test_skill_loader,
        test_skill_trigger_match,
        test_skill_intent_integration,
        test_subagent_scheduler,
        test_subagent_aggregator,
        test_subagent_scenario,
        test_chat_fast,
    ]

    # 前 6 个测试（图编译 + 节点 + Web 搜索）本身需要时间，
    # 后台预热线程大概率在 test_memory_searcher 前完成。
    # 若 embedding 模型尚未加载完毕，memory_searcher_agent 会自动 await initialize()（60s 超时）。
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
