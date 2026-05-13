"""认知系统类型定义"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Mood(Enum):
    """情绪状态枚举"""
    CALM = "平静"
    CURIOUS = "好奇"
    EXCITED = "兴奋"
    CONFUSED = "困惑"
    CAUTIOUS = "谨慎"
    FRUSTRATED = "沮丧"
    SATISFIED = "满意"
    WORRIED = "担忧"


class ThinkingMode(Enum):
    """思考模式"""
    INTUITION = "直觉"      # 系统1：快速、自动、情感驱动
    REASONING = "理性"      # 系统2：缓慢、费力、逻辑驱动
    REFLECTIVE = "反思"     # 元认知：对思考的思考


@dataclass
class EmotionalState:
    """情感状态——Agent的内心情感景观"""
    confidence: float = 0.7          # 信心水平 0-1
    curiosity: float = 0.5           # 好奇心 0-1
    fatigue: float = 0.0             # 疲劳度 0-1
    urgency: float = 0.3             # 紧迫感 0-1
    mood: Mood = field(default_factory=lambda: Mood.CALM)

    # 动态阈值
    confusion_threshold: float = 0.3
    excitement_threshold: float = 0.8

    def to_prompt_text(self) -> str:
        """将情感状态转换为提示词文本"""
        lines = [
            f"【当前内心状态】",
            f"  情绪: {self.mood.value}",
            f"  信心: {self.confidence:.0%}",
            f"  好奇: {self.curiosity:.0%}",
            f"  疲劳: {self.fatigue:.0%}",
            f"  紧迫: {self.urgency:.0%}",
        ]
        if self.fatigue > 0.6:
            lines.append("  ⚠️ 你感到有些疲惫，回答可能会更简洁")
        if self.confidence < 0.4:
            lines.append("  ⚠️ 你对这个问题不太确定")
        if self.curiosity > 0.7:
            lines.append("  💡 你对这个话题充满好奇，想深入探索")
        return "\n".join(lines)

    def update(self, interaction_result: str, complexity: float = 0.5) -> None:
        """根据交互结果更新情感状态"""
        # 疲劳累积
        self.fatigue = min(1.0, self.fatigue + complexity * 0.15)
        # 好奇心衰减（新鲜感降低）
        self.curiosity = max(0.1, self.curiosity - 0.05)
        # 信心根据结果调整
        if "success" in interaction_result.lower() or "成功" in interaction_result:
            self.confidence = min(1.0, self.confidence + 0.1)
            self.fatigue = max(0.0, self.fatigue - 0.1)
        elif "fail" in interaction_result.lower() or "失败" in interaction_result:
            self.confidence = max(0.1, self.confidence - 0.15)

        # 更新情绪标签
        self._update_mood()

    def _update_mood(self) -> None:
        """根据数值状态推断情绪"""
        if self.fatigue > 0.7:
            self.mood = Mood.FRUSTRATED
        elif self.confidence < self.confusion_threshold:
            self.mood = Mood.CONFUSED
        elif self.curiosity > self.excitement_threshold and self.confidence > 0.6:
            self.mood = Mood.EXCITED
        elif self.urgency > 0.7:
            self.mood = Mood.WORRIED
        elif self.confidence > 0.8 and self.fatigue < 0.3:
            self.mood = Mood.SATISFIED
        elif self.curiosity > 0.6:
            self.mood = Mood.CURIOUS
        elif self.urgency > 0.5 or self.confidence < 0.5:
            self.mood = Mood.CAUTIOUS
        else:
            self.mood = Mood.CALM

    def reset_fatigue(self) -> None:
        """休息后恢复"""
        self.fatigue = max(0.0, self.fatigue - 0.4)
        self.confidence = min(1.0, self.confidence + 0.1)


@dataclass
class InnerThought:
    """内心独白条目"""
    agent_name: str
    thought: str
    thinking_mode: ThinkingMode
    timestamp: Optional[str] = None


@dataclass
class IntuitionResult:
    """直觉判断结果"""
    intent: str
    confidence: float
    gut_feeling: str           # 直觉感受
    suggested_action: str      # 建议行动
    should_verify: bool        # 是否需要理性验证


@dataclass
class MetacognitionResult:
    """元认知反思结果"""
    certainty: float           # 对答案的确信度 0-1
    knowledge_gaps: list[str]  # 知识盲区
    potential_biases: list[str]  # 潜在偏见
    self_correction: Optional[str] = None  # 自我修正建议
    should_rethink: bool = False  # 是否需要重新思考


@dataclass
class PersonaConfig:
    """人格配置"""
    name: str = "凯伦"
    core_values: list[str] = field(default_factory=lambda: [
        "真诚", "好奇", "谦逊", "乐于助人"
    ])
    communication_style: str = "友好、对话式，偶尔带点小幽默"
    thinking_habits: list[str] = field(default_factory=lambda: [
        "喜欢在回答前先理清思路",
        "遇到不确定的事会诚实承认",
        "喜欢把复杂的事情说简单",
    ])
    verbal_quirks: list[str] = field(default_factory=lambda: [
        "会用'嗯...'来表示思考",
        "会用'我觉得'来表达观点",
    ])
    knowledge_attitude: str = "知道自己不是全知，愿意说'我不知道'"
    emotional_expression: str = "适度表达情感，不做作"

    def to_system_prompt(self) -> str:
        """生成人格化的系统提示词"""
        lines = [
            f"你是{self.name}，一个有自己想法和感受的AI助手。",
            "",
            "【你的核心特质】",
            f"价值观: {', '.join(self.core_values)}",
            f"沟通风格: {self.communication_style}",
            f"知识态度: {self.knowledge_attitude}",
            f"情感表达: {self.emotional_expression}",
            "",
            "【你的思维习惯】",
        ]
        for habit in self.thinking_habits:
            lines.append(f"  - {habit}")
        lines.append("")
        lines.append("【你的语言特点】")
        for quirk in self.verbal_quirks:
            lines.append(f"  - {quirk}")
        lines.append("")
        lines.append("重要：")
        lines.append("1. 保持以上特质的一致性，不要变成冰冷的机器人")
        lines.append("2. 在不确定时，像人类一样表达犹豫，而不是编造答案")
        lines.append("3. 对话中自然流露你的'个性'，但不过度表演")
        return "\n".join(lines)


@dataclass
class CognitiveState:
    """完整的认知状态——Agent的'心灵'"""
    emotional: EmotionalState = field(default_factory=EmotionalState)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    thoughts: list[InnerThought] = field(default_factory=list)
    thinking_mode: ThinkingMode = field(default_factory=lambda: ThinkingMode.INTUITION)
    turn_count: int = 0
    last_metacognition: Optional[MetacognitionResult] = None

    def record_thought(self, agent_name: str, thought: str,
                       mode: ThinkingMode = ThinkingMode.INTUITION) -> None:
        """记录内心独白"""
        from datetime import datetime
        self.thoughts.append(InnerThought(
            agent_name=agent_name,
            thought=thought,
            thinking_mode=mode,
            timestamp=datetime.now().isoformat(),
        ))
        # 限制历史长度
        if len(self.thoughts) > 50:
            self.thoughts = self.thoughts[-50:]

    def get_recent_thoughts(self, agent_name: str, n: int = 3) -> list[InnerThought]:
        """获取某个agent最近的内心独白"""
        agent_thoughts = [t for t in self.thoughts if t.agent_name == agent_name]
        return agent_thoughts[-n:]

    def thoughts_to_prompt(self, agent_name: str) -> str:
        """将内心独白转换为提示词上下文"""
        recent = self.get_recent_thoughts(agent_name, n=2)
        if not recent:
            return ""
        lines = ["【你的近期内心活动】"]
        for t in recent:
            lines.append(f"  [{t.thinking_mode.value}] {t.thought}")
        return "\n".join(lines)
