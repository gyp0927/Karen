"""认知系统 —— 给AI Agent加入人类思维

核心模块：
- types: 认知状态类型定义
- engine: 五个认知子系统的统一实现(情感/独白/直觉/元认知/人格)
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
