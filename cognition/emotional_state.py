"""情感状态管理器——给Agent一颗心"""
import logging
from typing import Optional

from cognition.types import EmotionalState, Mood, CognitiveState

logger = logging.getLogger(__name__)


class EmotionalStateManager:
    """管理所有Agent的情感状态"""

    def __init__(self):
        # 每个agent_name对应一个情感状态
        self._states: dict[str, EmotionalState] = {}

    def get_state(self, agent_name: str) -> EmotionalState:
        """获取或创建agent的情感状态"""
        if agent_name not in self._states:
            self._states[agent_name] = EmotionalState()
        return self._states[agent_name]

    def update_after_interaction(
        self,
        agent_name: str,
        success: bool = True,
        complexity: float = 0.5,
        user_emotion_hint: Optional[str] = None,
    ) -> EmotionalState:
        """交互后更新情感状态

        Args:
            agent_name: agent名称
            success: 交互是否成功/满意
            complexity: 任务复杂度 0-1
            user_emotion_hint: 用户情绪线索（如"愤怒"、"困惑"）
        """
        state = self.get_state(agent_name)

        result = "success" if success else "failure"
        state.update(result, complexity)

        # 响应用户情绪
        if user_emotion_hint:
            self._respond_to_user_emotion(state, user_emotion_hint)

        logger.debug(f"[{agent_name}] 情感状态更新: {state.mood.value}, "
                    f"信心{state.confidence:.1f}, 疲劳{state.fatigue:.1f}")
        return state

    def _respond_to_user_emotion(self, state: EmotionalState, hint: str) -> None:
        """根据用户情绪调整自身状态"""
        hint = hint.lower()
        if any(w in hint for w in ["怒", "生气", "angry", "frustrated", "pissed"]):
            state.urgency = min(1.0, state.urgency + 0.3)
            state.mood = Mood.CAUTIOUS
        elif any(w in hint for w in ["困惑", "不懂", "confused", "lost", "?"]):
            state.confidence = max(0.1, state.confidence - 0.1)
            state.curiosity = min(1.0, state.curiosity + 0.2)
            state.mood = Mood.CURIOUS
        elif any(w in hint for w in ["急", "urgent", "快", " hurry", " asap"]):
            state.urgency = min(1.0, state.urgency + 0.4)
        elif any(w in hint for w in ["谢", "感谢", "thank", "good", "棒", "厉害"]):
            state.confidence = min(1.0, state.confidence + 0.15)
            state.fatigue = max(0.0, state.fatigue - 0.2)
            state.mood = Mood.SATISFIED

# 全局单例
_emotional_manager: Optional[EmotionalStateManager] = None


def get_emotional_manager() -> EmotionalStateManager:
    """获取全局情感管理器"""
    global _emotional_manager
    if _emotional_manager is None:
        _emotional_manager = EmotionalStateManager()
    return _emotional_manager


def inject_emotion_to_prompt(agent_name: str, base_prompt: str) -> str:
    """将情感状态注入到提示词中"""
    manager = get_emotional_manager()
    state = manager.get_state(agent_name)
    emotion_text = state.to_prompt_text()
    return f"{emotion_text}\n\n{base_prompt}"
