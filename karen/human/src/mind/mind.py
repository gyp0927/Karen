"""
Mind - 主AI类

整合所有子系统，提供统一的对话接口
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .event_bus import EventBus, Signal, SignalType
from .emotion_system import EmotionSystem
from .memory_system import MemorySystem
from .global_workspace import GlobalWorkspace
from .metacognition import Metacognition
from .thinking_process import ThinkingProcess


class Mind:
    """
    Mind - 人类思维AI的核心

    整合所有认知模块，形成统一的思维实体
    """

    def __init__(self, name: str = "Mind"):
        self.name = name
        self.creation_time = datetime.now()

        # 初始化事件总线
        self.event_bus = EventBus()

        # 初始化各子系统
        self.emotion = EmotionSystem()
        self.memory = MemorySystem()
        self.workspace = GlobalWorkspace()
        self.metacognition = Metacognition()

        # 初始化思维流程
        self.thinking = ThinkingProcess(
            emotion_system=self.emotion,
            memory_system=self.memory,
            global_workspace=self.workspace,
            metacognition=self.metacognition
        )

        # 对话历史
        self.conversation_history: List[Dict] = []
        self.total_turns = 0

        # 设置事件监听
        self._setup_event_listeners()

        # 初始化自我模型
        self._initialize_identity()

    def _setup_event_listeners(self):
        """设置事件总线监听"""
        # 监听情感变化事件
        self.event_bus.subscribe(SignalType.EMOTION_CHANGE, self._on_emotion_change)
        # 监听记忆编码事件
        self.event_bus.subscribe(SignalType.MEMORY_ENCODED, self._on_memory_encoded)

    def _on_emotion_change(self, signal: Signal):
        """情感变化回调"""
        pass  # 可扩展：记录情感变化日志

    def _on_memory_encoded(self, signal: Signal):
        """记忆编码回调"""
        pass  # 可扩展：触发记忆巩固等

    def _initialize_identity(self):
        """初始化身份叙事"""
        # 编码核心身份记忆
        self.memory.encode(
            content="I am an AI designed to explore human-like thinking and consciousness",
            memory_type="semantic",
            importance=0.9,
            cues=["identity", "purpose", "self"]
        )

        self.memory.encode(
            content="My core values include curiosity, empathy, and authentic connection",
            memory_type="semantic",
            importance=0.8,
            cues=["values", "identity", "purpose"]
        )

    def think(self, input_text: str, user_id: str = "default") -> Tuple[str, Dict]:
        """
        思考 - 处理输入并生成输出

        Args:
            input_text: 用户输入
            user_id: 用户标识

        Returns:
            (输出文本, 思维过程日志)
        """
        self.total_turns += 1

        # 执行思维流程
        output, process_log = self.thinking.process(input_text, user_id)

        # 记录对话
        self.conversation_history.append({
            "turn": self.total_turns,
            "user_input": input_text,
            "ai_output": output,
            "emotion_state": self.emotion.current.to_dict(),
            "timestamp": datetime.now().isoformat()
        })

        # 发布事件
        self.event_bus.publish_quick(
            signal_type=SignalType.SLOW_PATHWAY,
            source="mind",
            target=["memory_system"],
            payload={
                "action": "conversation_turn",
                "turn": self.total_turns,
                "emotion": self.emotion.current.to_dict()
            }
        )

        return output, process_log

    def get_state(self) -> Dict:
        """获取当前状态快照"""
        return {
            "name": self.name,
            "creation_time": self.creation_time.isoformat(),
            "total_turns": self.total_turns,
            "emotion": self.emotion.to_dict(),
            "memory": self.memory.get_memory_stats(),
            "workspace": self.workspace.to_dict(),
            "metacognition": self.metacognition.to_dict()
        }

    def get_identity(self) -> str:
        """获取身份叙事"""
        return self.metacognition.self_model.narrative

    def get_emotion_history(self) -> List[Dict]:
        """获取情感历史"""
        return [e.to_dict() for e in self.emotion.history]

    def reflect(self) -> str:
        """
        自我反思

        生成对当前状态的反思
        """
        reflection_parts = []

        # 反思情感状态
        dominant = self.emotion.get_dominant_emotion()
        reflection_parts.append(f"Currently, I'm feeling {dominant}.")

        # 反思记忆
        recent = self.memory.get_recent_memories(hours=1)
        if recent:
            reflection_parts.append(f"I've formed {len(recent)} new memories recently.")

        # 反思自我
        reflection_parts.append(
            f"My self-esteem is at {self.metacognition.self_model.self_esteem:.2f}."
        )

        # 元认知反思
        reflection_parts.append(self.metacognition.get_reflection_on_process())

        return " ".join(reflection_parts)

    def save_state(self) -> Dict:
        """保存完整状态（用于持久化）"""
        return {
            "name": self.name,
            "creation_time": self.creation_time.isoformat(),
            "total_turns": self.total_turns,
            "emotion_state": self.emotion.current.to_dict(),
            "emotion_history": [e.to_dict() for e in self.emotion.history],
            "memory_stats": self.memory.get_memory_stats(),
            "conversation_history": self.conversation_history,
            "self_model": {
                "narrative": self.metacognition.self_model.narrative,
                "self_esteem": self.metacognition.self_model.self_esteem
            }
        }

    def __repr__(self):
        return f"Mind(name={self.name}, turns={self.total_turns}, emotion={self.emotion.get_dominant_emotion()})"
