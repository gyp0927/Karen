"""认知系统工具函数——减少重复的单例模式和序列化逻辑"""
from typing import TypeVar, Optional
from dataclasses import asdict

from cognition.types import (
    CognitiveState,
    EmotionalState,
    Mood,
    ThinkingMode,
    InnerThought,
    MetacognitionResult,
    PersonaConfig,
)

T = TypeVar("T")


def singleton(factory: type[T]) -> tuple[Optional[T], callable]:
    """创建单例模式和获取函数。

    返回 (instance_ref, get_instance)，其中 get_instance 是闭包函数。
    用法：
        _instance, get_instance = singleton(MyClass)
    """
    _inst: Optional[T] = None

    def get_instance(*args, **kwargs) -> T:
        nonlocal _inst
        if _inst is None:
            _inst = factory(*args, **kwargs)
        return _inst

    return _inst, get_instance


def get_cognitive_state_from_dict(state: dict) -> CognitiveState:
    """从 state 字典中提取或创建认知状态。

    注意：asdict() 会递归地将嵌套 dataclass 转为 dict，
    所以反序列化时需要手动恢复嵌套对象。
    """
    cog_dict = state.get("cognitive_state")
    if not cog_dict:
        return CognitiveState()

    # 手动恢复嵌套的 EmotionalState 对象（asdict 递归转 dict 后丢失类型）
    emotional_dict = cog_dict.get("emotional")
    if emotional_dict and isinstance(emotional_dict, dict):
        # 恢复 Mood 枚举
        mood_val = emotional_dict.get("mood")
        if isinstance(mood_val, Mood):
            pass  # 已经是枚举类型
        elif mood_val and isinstance(mood_val, str):
            try:
                emotional_dict["mood"] = Mood(mood_val)
            except ValueError:
                emotional_dict["mood"] = Mood.CALM
        else:
            emotional_dict["mood"] = Mood.CALM
        emotional = EmotionalState(**emotional_dict)
    else:
        emotional = EmotionalState()

    # 恢复 ThinkingMode 枚举
    thinking_mode_val = cog_dict.get("thinking_mode")
    if thinking_mode_val:
        try:
            thinking_mode = ThinkingMode(thinking_mode_val)
        except ValueError:
            thinking_mode = ThinkingMode.INTUITION
    else:
        thinking_mode = ThinkingMode.INTUITION

    # 恢复 InnerThought 列表
    thoughts_raw = cog_dict.get("thoughts", [])
    thoughts = []
    for t in thoughts_raw:
        if isinstance(t, dict):
            tm_val = t.get("thinking_mode")
            if tm_val:
                try:
                    t["thinking_mode"] = ThinkingMode(tm_val)
                except ValueError:
                    t["thinking_mode"] = ThinkingMode.INTUITION
            thoughts.append(InnerThought(**t))
        elif isinstance(t, InnerThought):
            thoughts.append(t)

    # 恢复 last_metacognition
    meta_raw = cog_dict.get("last_metacognition")
    if meta_raw and isinstance(meta_raw, dict):
        last_metacognition = MetacognitionResult(**meta_raw)
    else:
        last_metacognition = None

    return CognitiveState(
        emotional=emotional,
        persona=PersonaConfig(**cog_dict.get("persona", {})),
        thoughts=thoughts,
        thinking_mode=thinking_mode,
        turn_count=cog_dict.get("turn_count", 0),
        last_metacognition=last_metacognition,
    )


def save_cognitive_state_to_dict(state: dict, cognitive_state: CognitiveState) -> None:
    """将认知状态保存回 state 字典。"""
    state["cognitive_state"] = asdict(cognitive_state)


def serialize_cognitive_state(cognitive_state: CognitiveState) -> dict:
    """序列化认知状态为字典。"""
    return asdict(cognitive_state)
