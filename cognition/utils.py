"""认知系统工具函数——减少重复的单例模式和序列化逻辑"""
from typing import TypeVar, Optional
from dataclasses import asdict

from cognition.types import CognitiveState

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

    多处使用（agents/factory.py, interface/human_interface.py 等），
    统一在这里避免重复代码。
    """
    cog_dict = state.get("cognitive_state")
    if cog_dict:
        return CognitiveState(**cog_dict)
    return CognitiveState()


def save_cognitive_state_to_dict(state: dict, cognitive_state: CognitiveState) -> None:
    """将认知状态保存回 state 字典。"""
    state["cognitive_state"] = asdict(cognitive_state)


def serialize_cognitive_state(cognitive_state: CognitiveState) -> dict:
    """序列化认知状态为字典。"""
    return asdict(cognitive_state)
