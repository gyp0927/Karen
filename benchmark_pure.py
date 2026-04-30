import asyncio
import time
import sys
sys.path.insert(0, "E:/果冻ai-memory/多agent聊天")

from langchain_core.messages import HumanMessage

# 使用 mock 搜索，排除网络波动
async def mock_web_searcher(query):
    return "[联网搜索结果]\n\nPython是一种编程语言。"

async def mock_memory_searcher(query, user_id=""):
    return ""

async def mock_tool_caller(state):
    return {"messages": []}

async def benchmark_fast():
    from graph.orchestrator import create_fast_graph
    from agents.factory import responder_node

    graph = create_fast_graph(mock_web_searcher, mock_memory_searcher, mock_tool_caller, responder_node)
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
    async for _ in graph.astream(state):
        pass
    return time.time() - t0

async def benchmark_coordination():
    from graph.orchestrator import create_coordination_graph
    from agents.factory import coordinator_node, researcher_node, responder_node

    graph = create_coordination_graph(coordinator_node, researcher_node, mock_tool_caller, responder_node)
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
    async for _ in graph.astream(state):
        pass
    return time.time() - t0

async def main():
    print("=" * 60)
    print("纯净基准测试（排除网络波动，使用 mock 搜索）")
    print("=" * 60)

    print("\n[1/3] 快速模式 x3...")
    fast_times = []
    for i in range(3):
        t = await benchmark_fast()
        fast_times.append(t)
        print(f"   第{i+1}次: {t:.2f}秒")

    print("\n[2/3] 协调模式 x3...")
    coord_times = []
    for i in range(3):
        t = await benchmark_coordination()
        coord_times.append(t)
        print(f"   第{i+1}次: {t:.2f}秒")

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    fa = sum(fast_times) / len(fast_times)
    ca = sum(coord_times) / len(coord_times)
    print(f"快速模式:   {fa:.2f}秒 (仅 Responder 1次 LLM)")
    print(f"协调模式:   {ca:.2f}秒 (Coordinator[80t] + Researcher[搜索] + Responder[1次 LLM])")
    print(f"差距:       {abs(ca-fa):.2f}秒 ({((ca/fa-1)*100) if ca>fa else ((fa/ca-1)*100):.0f}%)")
    print("=" * 60)
    print("\n说明：")
    print("- Coordinator 限制 80 tokens，约 0.3~0.5 秒")
    print("- Researcher 不再做 LLM 生成，只做搜索（mock 搜索≈0秒）")
    print("- 优化前协调模式有 3 次 LLM，优化后只剩 1 次")

if __name__ == "__main__":
    asyncio.run(main())
