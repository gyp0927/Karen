"""认知系统 —— 给AI Agent加入人类思维

核心模块：
- types: 认知状态类型定义
- emotional_state: 情感状态系统
- inner_monologue: 内心独白引擎
- intuition: 直觉引擎（系统1）
- metacognition: 元认知反射
- persona: 人格系统
- human_mind: 统一入口

快速开始：
    from cognition.human_mind import HumanMind
    mind = HumanMind()
    enhanced_prompt = mind.enhance_prompt("responder", base_prompt, query, state)
"""

from cognition.types import (
    CognitiveState,
    EmotionalState,
    Mood,
    ThinkingMode,
    InnerThought,
    IntuitionResult,
    MetacognitionResult,
    PersonaConfig,
)
from cognition.human_mind import HumanMind, enhance_agent_prompt, process_agent_response

__all__ = [
    "CognitiveState",
    "EmotionalState",
    "Mood",
    "ThinkingMode",
    "InnerThought",
    "IntuitionResult",
    "MetacognitionResult",
    "PersonaConfig",
    "HumanMind",
    "enhance_agent_prompt",
    "process_agent_response",
]
