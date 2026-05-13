"""
事件总线 - 模块间通信的核心基础设施

采用发布-订阅模式，支持事件驱动和状态驱动的混合通信
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import threading


class SignalType(Enum):
    """信号类型"""
    FAST_PATHWAY = "fast_pathway"           # 快速通路（潜意识）
    SLOW_PATHWAY = "slow_pathway"           # 慢速通路（意识）
    META_FEEDBACK = "meta_feedback"         # 元认知反馈
    SPREADING_ACTIVATION = "spreading_activation"  # 激活扩散
    EMOTION_CHANGE = "emotion_change"       # 情感状态变化
    MEMORY_RETRIEVED = "memory_retrieved"   # 记忆被提取
    MEMORY_ENCODED = "memory_encoded"       # 记忆被编码
    ATTENTION_SHIFT = "attention_shift"     # 注意力转移


@dataclass
class Signal:
    """信号 - 模块间通信的基本单元"""
    signal_type: SignalType
    source: str
    target: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5                       # 1-10, 1最高

    def __repr__(self):
        return f"Signal({self.signal_type.value}, src={self.source}, tgt={self.target})"


class EventBus:
    """
    事件总线 - 全系统的中枢神经系统

    所有模块通过事件总线通信，实现松耦合
    """

    def __init__(self):
        self._subscribers: Dict[SignalType, List[Callable]] = {
            st: [] for st in SignalType
        }
        self._global_subscribers: List[Callable] = []  # 订阅所有事件
        self._queue: List[Signal] = []
        self._lock = threading.Lock()
        self._state_callbacks: Dict[str, List[Callable]] = {}  # 状态变化回调

    def subscribe(self, event_type: SignalType, callback: Callable[[Signal], None]):
        """订阅特定类型的事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[Signal], None]):
        """订阅所有事件"""
        self._global_subscribers.append(callback)

    def publish(self, signal: Signal):
        """发布事件到总线"""
        with self._lock:
            self._queue.append(signal)
            # 按优先级排序
            self._queue.sort(key=lambda s: s.priority)

        # 立即分发（同步）
        self._dispatch(signal)

    def publish_quick(self, signal_type: SignalType, source: str,
                      target: List[str] = None, payload: Dict = None,
                      priority: int = 5):
        """快捷发布方法"""
        signal = Signal(
            signal_type=signal_type,
            source=source,
            target=target or [],
            payload=payload or {},
            priority=priority
        )
        self.publish(signal)

    def _dispatch(self, signal: Signal):
        """分发信号到订阅者"""
        # 分发给特定类型的订阅者
        callbacks = self._subscribers.get(signal.signal_type, [])
        for callback in callbacks:
            try:
                callback(signal)
            except Exception as e:
                print(f"[EventBus] Error dispatching to {callback}: {e}")

        # 分发给全局订阅者
        for callback in self._global_subscribers:
            try:
                callback(signal)
            except Exception as e:
                print(f"[EventBus] Error in global dispatch: {e}")

    def get_queue_size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)

    def clear_queue(self):
        """清空队列"""
        with self._lock:
            self._queue.clear()
