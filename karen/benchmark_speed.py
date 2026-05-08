import asyncio
import time
import logging
import sys
import threading
from pathlib import Path

# 抑制第三方库的 verbose 日志
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

async def benchmark_fast():
    """测试快速模式（跳过Coordinator，并行搜索+Responder）"""
    from graph.orchestrator import create_fast_graph
    from agents.search import web_searcher_agent, memory_searcher_agent
    from agents.tools import tool_caller_node
    from agents.nodes import responder_node

    graph = create_fast_graph(
        web_searcher_agent, memory_searcher_agent,
        tool_caller_node, responder_node,
    )

    state = {
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

    t0 = time.time()
    final = None
    async for event in graph.astream(state):
        for _, output in event.items():
            final = output
    elapsed = time.time() - t0
    return elapsed, final

async def benchmark_coordination():
    """测试协调模式（Coordinator + 并行搜索 + Responder）"""
    from graph.orchestrator import create_coordination_graph
    from agents.nodes import coordinator_node, researcher_node, responder_node
    from agents.tools import tool_caller_node

    graph = create_coordination_graph(
        coordinator_node, researcher_node, tool_caller_node, responder_node,
    )

    state = {
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

    t0 = time.time()
    final = None
    async for event in graph.astream(state):
        for _, output in event.items():
            final = output
    elapsed = time.time() - t0
    return elapsed, final

async def main():
    print("=" * 60)
    print("凯伦 响应速度基准测试")
    print("=" * 60)

    # ---- 后台预热记忆系统（首次加载 embedding 模型约 10-20s） ----
    _memory_warmup_done = threading.Event()

    def _bg_memory_warmup():
        try:
            from core.memory_client import _MEMORY_SYSTEM_AVAILABLE, get_memory_store
            if _MEMORY_SYSTEM_AVAILABLE:
                asyncio.run(get_memory_store().initialize())
        except Exception:
            pass
        finally:
            _memory_warmup_done.set()

    threading.Thread(target=_bg_memory_warmup, daemon=True, name="mem-warmup").start()

    # 预热（加载模型、建立连接）
    print("\n[1/4] 预热中...（首次加载需等待）")

    # 确保记忆系统预热完成后再开始 benchmark，避免首次测量被初始化耗时污染
    if not _memory_warmup_done.is_set():
        print("   等待记忆系统预热...")
        await asyncio.to_thread(_memory_warmup_done.wait, 60)

    try:
        await benchmark_fast()
    except Exception as e:
        print(f"   预热失败: {e}")

    # 快速模式测试
    print("\n[2/4] 快速模式测试...")
    fast_times = []
    for i in range(3):
        t, result = await benchmark_fast()
        fast_times.append(t)
        print(f"   第{i+1}次: {t:.2f}秒")
    print(f"   快速模式平均: {sum(fast_times)/len(fast_times):.2f}秒")

    # 协调模式测试
    print("\n[3/4] 协调模式测试...")
    coord_times = []
    for i in range(3):
        t, result = await benchmark_coordination()
        coord_times.append(t)
        print(f"   第{i+1}次: {t:.2f}秒")
    print(f"   协调模式平均: {sum(coord_times)/len(coord_times):.2f}秒")

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    fast_avg = sum(fast_times) / len(fast_times)
    coord_avg = sum(coord_times) / len(coord_times)
    print(f"快速模式:   {fast_avg:.2f}秒 (最快 {min(fast_times):.2f}, 最慢 {max(fast_times):.2f})")
    print(f"协调模式:   {coord_avg:.2f}秒 (最快 {min(coord_times):.2f}, 最慢 {max(coord_times):.2f})")
    print(f"速度提升:   {(coord_avg/fast_avg - 1)*100:.0f}%")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
