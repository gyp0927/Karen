"""agents/factory.py - 兼容层，所有符号从子模块重新导出。

注意：新代码请直接从子模块导入：
    from agents.llm import get_llm, set_streaming_callback
    from agents.nodes import coordinator_node, responder_node
    from agents.search import web_searcher_agent
    from agents.tools import tool_caller_node
"""

# LLM 基础设施
from agents.llm import (
    set_current_llm_config,
    set_streaming_callback,
    get_streaming_callback,
    clear_streaming_callback,
    get_llm,
    clear_llm_cache,
)

# 搜索子 Agent
from agents.search import (
    web_searcher_agent,
    memory_searcher_agent,
    knowledge_searcher_agent,
)

# Agent 节点
from agents.nodes import (
    coordinator_node,
    researcher_node,
    responder_node,
    reviewer_node,
    planner_node,
    parse_plan_from_response,
    create_agents,
)

# 工具调用
from agents.tools import (
    tool_caller_node,
    _need_tool_call,
)
