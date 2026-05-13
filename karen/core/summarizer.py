"""对话自动摘要 - 长对话时自动压缩历史消息，减少 token 消耗。"""

import logging
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# 默认摘要阈值：消息数超过此值时触发摘要
_DEFAULT_SUMMARY_THRESHOLD = 20
# 保留最近多少轮完整对话（不摘要）
_DEFAULT_KEEP_RECENT = 5


class ConversationSummarizer:
    """对话摘要器"""

    def __init__(self, threshold: int = _DEFAULT_SUMMARY_THRESHOLD, keep_recent: int = _DEFAULT_KEEP_RECENT):
        self.threshold = threshold
        self.keep_recent = keep_recent

    def should_summarize(self, message_count: int) -> bool:
        """判断是否需要摘要。"""
        return message_count >= self.threshold

    def summarize(self, messages: list[BaseMessage], llm=None) -> Optional[str]:
        """生成对话摘要。

        Args:
            messages: 需要摘要的消息列表
            llm: 可选的 LLM 实例（如果为 None，使用简单规则摘要）

        Returns:
            摘要文本，或 None（如果不需要摘要）
        """
        if not messages:
            return None

        if llm is not None:
            return self._llm_summarize(messages, llm)
        return self._rule_based_summarize(messages)

    def _llm_summarize(self, messages: list[BaseMessage], llm) -> str:
        """使用 LLM 生成摘要。"""
        # 构建摘要提示词
        conversation_text = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            sender = getattr(msg, "name", role)
            content = msg.content[:500]  # 每条消息最多取 500 字符
            conversation_text.append(f"{sender}: {content}")

        summary_prompt = f"""请对以下对话进行摘要，保留关键信息：

要求：
1. 总结用户的主要需求和问题
2. 记录 AI 提供的关键回答和事实
3. 保留任何待办事项或待确认的事项
4. 保留用户明确表达的偏好
5. 摘要控制在 300 字以内

对话内容：
{"\n".join(conversation_text)}

请输出摘要："""

        try:
            from langchain_core.messages import SystemMessage
            response = llm.invoke([SystemMessage(content=summary_prompt)])
            summary = response.content.strip()
            logger.info(f"Generated LLM summary, length={len(summary)}")
            return summary
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}, falling back to rule-based")
            return self._rule_based_summarize(messages)

    def _rule_based_summarize(self, messages: list[BaseMessage]) -> str:
        """基于规则的简单摘要（无需 LLM）。"""
        topics = []
        key_facts = []

        for msg in messages:
            content = msg.content[:200]
            if isinstance(msg, HumanMessage):
                # 提取用户问题（取前 50 字符作为主题）
                topic = content[:50].strip().replace("\n", " ")
                if topic:
                    topics.append(topic)
            elif isinstance(msg, AIMessage):
                # 尝试提取关键事实（包含数字、日期的句子）
                import re
                facts = re.findall(r'[^。！?.]+(?:\d{4}|\d+%|第[一二三四五]|首先|关键|重要)[^。！?.]*[。！?.]?', content)
                key_facts.extend(facts[:2])  # 每条 AI 消息最多取 2 个事实

        lines = ["[对话摘要]"]
        if topics:
            lines.append(f"讨论主题: {'; '.join(topics[:3])}")
        if key_facts:
            lines.append(f"关键信息: {'; '.join(key_facts[:3])}")

        summary = "\n".join(lines)
        logger.info(f"Generated rule-based summary, length={len(summary)}")
        return summary

    def prepare_messages_for_model(
        self,
        messages: list[BaseMessage],
        max_turns: int = 10,
        llm=None,
    ) -> list[BaseMessage]:
        """准备消息列表供 LLM 使用，必要时进行摘要。

        策略：
        1. 如果消息数未超过阈值，直接返回最近 max_turns 轮
        2. 如果超过阈值：
           - 保留最近 keep_recent 轮完整对话
           - 对之前的消息生成摘要
           - 将摘要作为 SystemMessage 插入

        Returns:
            处理后的消息列表
        """
        # 计算一轮 = 用户 + AI = 2 条消息
        max_msgs = max_turns * 2

        if len(messages) <= max_msgs:
            return list(messages)

        # 消息数超过限制，需要摘要
        if not self.should_summarize(len(messages)):
            # 未达摘要阈值，直接截断
            return list(messages[-max_msgs:])

        # 保留最近 keep_recent 轮
        keep_msgs = self.keep_recent * 2
        recent_messages = messages[-keep_msgs:]
        older_messages = messages[:-keep_msgs]

        # 对旧消息生成摘要
        summary = self.summarize(older_messages, llm)
        if summary:
            summary_msg = SystemMessage(content=f"[历史对话摘要]\n{summary}")
            return [summary_msg] + list(recent_messages)

        # 摘要失败，直接截断
        return list(messages[-max_msgs:])
