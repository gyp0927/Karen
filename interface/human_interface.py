import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph

from graph.orchestrator import create_coordination_graph, create_fast_graph
from state.manager import SessionManager
from core.model_router import get_router

logger = logging.getLogger(__name__)


class HumanInterface:
    """人类用户接口 - 连接用户与多Agent系统"""

    def __init__(self, message_manager: SessionManager, coordinator=None, researcher=None,
                 responder=None, reviewer=None, fast_mode: bool = True, review: bool = False,
                 review_language: str = "zh"):
        """
        参数:
            message_manager: 会话管理器
            coordinator/researcher/responder/reviewer: Agent 节点函数
            fast_mode: 是否跳过 coordinator/researcher
            review: 是否启用审查流程
            review_language: 审查语言（"zh" 或 "en"）
        """
        self.messages = message_manager
        self.fast_mode = fast_mode
        self.review = review
        self.review_language = review_language

        if fast_mode or coordinator is None:
            self.graph = create_fast_graph(responder)
        else:
            self.graph = create_coordination_graph(coordinator, researcher, responder)

        self.responder = responder
        self.reviewer = reviewer

    async def send_message(self, content: str) -> str:
        """发送用户消息并获取响应（异步）"""
        # 模型路由：分析复杂度并切换 LLM 配置
        router = get_router()
        history = self.messages.get_messages()
        history_turns = len(history) // 2
        route_result = router.route(content, history_turns)
        logger.info(f"Model routing: tier={route_result['tier']}, score={route_result['analysis']['score']}")

        self.messages.add_human_message(content)

        initial_state = {
            "messages": self.messages.get_messages_for_model(max_turns=10),
            "active_agent": None,
            "task_context": {"user_input": content},
            "human_input_required": False,
            "base_model_response": None,
            "review_result": None,
            "awaiting_review": False,
        }

        logger.info(f"Sending message, graph_type={'fast' if self.fast_mode else 'coordination'}")

        result = await self.graph.ainvoke(initial_state)

        # 将 agent 响应添加到消息管理器
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                agent_name = getattr(msg, "name", None) or "assistant"
                self.messages.add_agent_message(msg.content, agent_name)

        # 获取最后一条 Agent 消息
        response = result["messages"][-1].content

        # 如果启用了审查，执行审查
        if self.review and self.reviewer:
            review_result = await self._do_review(content, response)
            logger.info("Review completed")
            return f"{response}\n\n--- 审查意见 ---\n{review_result}"

        return response

    async def _do_review(self, user_message: str, base_response: str) -> str:
        """执行审查流程"""
        from prompts.reviewer_prompt import build_review_prompt
        review_prompt = build_review_prompt(user_message, base_response, self.review_language)

        review_state = {
            "messages": [HumanMessage(content=review_prompt)],
            "active_agent": "reviewer",
            "task_context": {"user_input": user_message, "base_response": base_response},
            "human_input_required": False,
            "base_model_response": base_response,
            "review_result": None,
            "awaiting_review": False,
        }

        result = await self.reviewer(review_state)
        msgs = result.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content
        return "No review available"

    def get_history(self) -> list[BaseMessage]:
        """获取完整消息历史"""
        return self.messages.get_messages()
