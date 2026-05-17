"""图编译测试"""

import os

import pytest
from langchain_core.messages import HumanMessage

_CI = os.getenv("CI") == "true"


@pytest.mark.asyncio
async def test_fast_graph():
    from agents.nodes import responder_node
    from graph.orchestrator import create_fast_graph

    graph = create_fast_graph(responder_node)
    assert graph is not None


@pytest.mark.asyncio
async def test_coordination_graph():
    from agents.nodes import coordinator_node, researcher_node, responder_node
    from agents.tools import tool_caller_node
    from graph.orchestrator import create_coordination_graph

    graph = create_coordination_graph(coordinator_node, researcher_node, tool_caller_node, responder_node)
    assert graph is not None


@pytest.mark.skipif(_CI, reason="需要 LLM 服务")
@pytest.mark.asyncio
async def test_chat_fast():
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
        for _node_name, node_output in event.items():
            final_state = node_output

    assert final_state is not None
    assert "messages" in final_state
    assert len(final_state["messages"]) > 0
    assert final_state["messages"][-1].content
