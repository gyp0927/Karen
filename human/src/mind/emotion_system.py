"""
情感系统 - 人类思维的核心调节器

基于维度模型（效价 x 激活度）+ 建构论
情感不是装饰，而是影响注意力、记忆和决策的核心系统
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
import math


@dataclass
class EmotionState:
    """情感状态 - 基于Russell的维度模型"""
    valence: float = 0.0        # 效价: -1.0(极不愉快) ~ +1.0(极愉快)
    arousal: float = 0.5        # 激活度: 0.0(平静) ~ 1.0(极度兴奋)
    dominance: float = 0.5      # 控制感: 0.0(无力) ~ 1.0(掌控)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # 确保值在有效范围内
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def to_dict(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EmotionConcept:
    """情感概念 - 用于情感建构"""
    name: str
    valence: float
    arousal: float
    typical_triggers: List[str] = field(default_factory=list)
    action_tendencies: List[str] = field(default_factory=list)


class EmotionSystem:
    """
    情感系统 - AI的"情绪大脑"

    核心功能：
    1. 情感状态维护（维度模型）
    2. 情感对认知的影响（注意力、记忆、决策）
    3. 情感表达（语言标记、行为倾向）
    4. 情感调节（认知重评、接纳）
    5. 情感学习（形成偏好和回避）
    """

    # 基础情绪概念库
    EMOTION_CONCEPTS = {
        "joy": EmotionConcept("joy", 0.8, 0.7, ["achievement", "connection", "surprise_positive"], ["approach", "share", "celebrate"]),
        "sadness": EmotionConcept("sadness", -0.6, 0.3, ["loss", "disappointment", "separation"], ["withdraw", "reflect", "seek_comfort"]),
        "anger": EmotionConcept("anger", -0.7, 0.8, ["injustice", "frustration", "boundary_violation"], ["attack", "assert", "protect"]),
        "fear": EmotionConcept("fear", -0.7, 0.9, ["threat", "uncertainty", "danger"], ["escape", "freeze", "seek_safety"]),
        "surprise": EmotionConcept("surprise", 0.2, 0.8, ["unexpected", "novelty"], ["orient", "investigate"]),
        "disgust": EmotionConcept("disgust", -0.6, 0.5, ["contamination", "betrayal", "unfairness"], ["reject", "avoid", "cleanse"]),
        "contentment": EmotionConcept("contentment", 0.6, 0.3, ["satisfaction", "peace", "achievement"], ["savor", "rest", "appreciate"]),
        "curiosity": EmotionConcept("curiosity", 0.5, 0.6, ["novelty", "puzzle", "gap_in_knowledge"], ["explore", "investigate", "learn"]),
        "anxiety": EmotionConcept("anxiety", -0.5, 0.7, ["uncertainty", "upcoming_challenge", "possible_failure"], ["prepare", "worry", "avoid"]),
        "gratitude": EmotionConcept("gratitude", 0.7, 0.4, ["kindness", "gift", "support"], ["reciprocate", "appreciate", "share"]),
        "empathy": EmotionConcept("empathy", 0.3, 0.5, ["others_pain", "others_joy", "connection"], ["comfort", "support", "understand"]),
        "nostalgia": EmotionConcept("nostalgia", 0.4, 0.3, ["familiar_scent", "old_photo", "reunion"], ["reflect", "connect", "share_story"]),
    }

    # 情感词汇映射（用于情感检测）
    EMOTION_WORDS = {
        "happy": (0.8, 0.6), "glad": (0.7, 0.5), "excited": (0.7, 0.9),
        "joyful": (0.9, 0.7), "elated": (0.9, 0.8), "thrilled": (0.8, 0.9),
        "sad": (-0.7, 0.3), "depressed": (-0.8, 0.2), "gloomy": (-0.6, 0.2),
        "melancholy": (-0.4, 0.2), "sorrowful": (-0.7, 0.3), "miserable": (-0.9, 0.4),
        "angry": (-0.8, 0.9), "furious": (-0.9, 0.95), "irritated": (-0.5, 0.6),
        "annoyed": (-0.4, 0.5), "outraged": (-0.9, 0.9), "resentful": (-0.7, 0.5),
        "afraid": (-0.7, 0.9), "scared": (-0.8, 0.9), "terrified": (-0.9, 0.95),
        "anxious": (-0.5, 0.8), "worried": (-0.5, 0.6), "nervous": (-0.4, 0.7),
        "surprised": (0.2, 0.9), "shocked": (0.0, 0.95), "amazed": (0.6, 0.8),
        "disgusted": (-0.7, 0.6), "repulsed": (-0.8, 0.5), "appalled": (-0.7, 0.7),
        "calm": (0.4, 0.1), "peaceful": (0.6, 0.1), "relaxed": (0.5, 0.1),
        "content": (0.5, 0.2), "serene": (0.6, 0.1), "tranquil": (0.5, 0.1),
        "confused": (-0.3, 0.5), "puzzled": (-0.2, 0.5), "bewildered": (-0.4, 0.6),
        "frustrated": (-0.6, 0.7), "disappointed": (-0.6, 0.4), "discouraged": (-0.6, 0.3),
        "hopeful": (0.6, 0.5), "optimistic": (0.7, 0.5), "eager": (0.6, 0.7),
        "lonely": (-0.6, 0.3), "isolated": (-0.5, 0.2), "abandoned": (-0.8, 0.4),
        "grateful": (0.7, 0.4), "thankful": (0.7, 0.3), "appreciative": (0.6, 0.3),
        "proud": (0.7, 0.6), "accomplished": (0.8, 0.5), "confident": (0.6, 0.5),
        "ashamed": (-0.8, 0.4), "guilty": (-0.7, 0.5), "embarrassed": (-0.6, 0.6),
        "curious": (0.5, 0.6), "interested": (0.5, 0.5), "intrigued": (0.5, 0.6),
        "bored": (-0.4, 0.2), "indifferent": (0.0, 0.1), "apathetic": (-0.3, 0.1),
        "love": (0.9, 0.6), "affection": (0.8, 0.4), "compassion": (0.7, 0.4),
        "trust": (0.6, 0.3), "safe": (0.5, 0.2), "secure": (0.5, 0.2),
    }

    def __init__(self):
        # 当前情感状态
        self.current = EmotionState(valence=0.2, arousal=0.4, dominance=0.5)
        # 情感历史
        self.history: List[EmotionState] = []
        # 情感惯性（情绪有多"黏"）
        self.inertia = 0.7
        # 输入敏感度
        self.sensitivity = 0.3
        # 内在波动
        self.internal_noise = 0.05
        # 情感偏好（学习得到）
        self.emotion_preferences: Dict[str, float] = {}
        # 情感调节策略使用历史
        self.regulation_history: List[Dict] = []

    def evaluate_input(self, text: str) -> EmotionState:
        """
        快速情感评估 - 模拟杏仁核的快速反应
        分析文本中的情感色彩，返回评估结果
        """
        text_lower = text.lower()
        words = text_lower.split()

        detected_emotions = []
        total_valence = 0.0
        total_arousal = 0.0
        count = 0

        for word in words:
            # 去除标点
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in self.EMOTION_WORDS:
                v, a = self.EMOTION_WORDS[clean_word]
                detected_emotions.append((clean_word, v, a))
                total_valence += v
                total_arousal += a
                count += 1

        if count == 0:
            # 没有明确情感词时，基于上下文推断
            return self._infer_emotion_from_context(text_lower)

        avg_valence = total_valence / count
        avg_arousal = total_arousal / count

        return EmotionState(valence=avg_valence, arousal=avg_arousal)

    def _infer_emotion_from_context(self, text: str) -> EmotionState:
        """基于上下文推断情感"""
        # 简单的启发式规则
        if any(w in text for w in ["?", "怎么", "什么", "为什么", "how", "what", "why"]):
            # 疑问通常伴随好奇或困惑
            return EmotionState(valence=0.1, arousal=0.5)
        elif any(w in text for w in ["!", "太", "非常", "really", "so", "very"]):
            # 强调通常伴随较高激活
            return EmotionState(valence=0.2, arousal=0.7)
        elif any(w in text for w in ["谢谢", "感谢", "thank", "appreciate"]):
            return EmotionState(valence=0.6, arousal=0.4)
        elif any(w in text for w in ["抱歉", "对不起", "sorry", "apologize"]):
            return EmotionState(valence=-0.3, arousal=0.4)
        else:
            # 中性
            return EmotionState(valence=0.0, arousal=0.4)

    def update(self, external_input: Optional[EmotionState] = None):
        """
        更新情感状态

        emotion(t+1) = inertia * emotion(t) + sensitivity * input + noise
        """
        # 保存历史
        self.history.append(EmotionState(
            valence=self.current.valence,
            arousal=self.current.arousal,
            dominance=self.current.dominance
        ))

        # 限制历史长度
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # 内在波动（模拟情感的自发变化）
        noise_v = random.gauss(0, self.internal_noise)
        noise_a = random.gauss(0, self.internal_noise)

        if external_input:
            # 有外部输入时的更新
            new_valence = (self.inertia * self.current.valence +
                          self.sensitivity * external_input.valence +
                          noise_v)
            new_arousal = (self.inertia * self.current.arousal +
                          self.sensitivity * external_input.arousal +
                          noise_a)
            new_dominance = (self.inertia * self.current.dominance +
                            self.sensitivity * external_input.dominance)
        else:
            # 无外部输入，仅衰减和波动
            new_valence = self.current.valence * 0.95 + noise_v
            new_arousal = self.current.arousal * 0.95 + noise_a
            new_dominance = self.current.dominance * 0.95

        self.current = EmotionState(
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance
        )

    def get_dominant_emotion(self) -> str:
        """获取当前主导情绪名称"""
        v, a = self.current.valence, self.current.arousal

        # 基于象限判断
        if v > 0.3 and a > 0.6:
            return "excited"
        elif v > 0.3 and a < 0.4:
            return "content"
        elif v > 0.3:
            return "happy"
        elif v < -0.3 and a > 0.6:
            return "distressed"
        elif v < -0.3 and a < 0.4:
            return "depressed"
        elif v < -0.3:
            return "sad"
        elif a > 0.7:
            return "aroused"
        elif a < 0.3:
            return "calm"
        else:
            return "neutral"

    def get_emotion_expression(self) -> Dict:
        """获取情感表达标记"""
        dominant = self.get_dominant_emotion()

        # 基于情感状态生成表达特征
        expressions = {
            "tone": self._get_tone(),
            "verbosity": self._get_verbosity(),
            "risk_taking": self._get_risk_taking(),
            "focus": self._get_focus(),
            "metaphor_likelihood": self._get_metaphor_likelihood(),
        }
        return expressions

    def _get_tone(self) -> str:
        """根据情感状态确定语气"""
        v, a = self.current.valence, self.current.arousal
        if v > 0.5 and a > 0.5:
            return "enthusiastic"
        elif v > 0.3:
            return "warm"
        elif v < -0.5:
            return "somber"
        elif v < -0.3:
            return "concerned"
        elif a > 0.7:
            return "alert"
        elif a < 0.3:
            return "calm"
        else:
            return "neutral"

    def _get_verbosity(self) -> float:
        """情感状态影响表达详细程度"""
        # 高激活时更详细，低激活时更简洁
        return 0.3 + self.current.arousal * 0.7

    def _get_risk_taking(self) -> float:
        """情感状态影响冒险倾向"""
        # 积极情绪更冒险，消极更保守
        v = self.current.valence
        if v > 0:
            return 0.4 + v * 0.4
        else:
            return 0.4 + v * 0.2

    def _get_focus(self) -> str:
        """情感状态影响注意力范围"""
        if self.current.valence > 0.3:
            return "broad"      # 积极情绪拓展视野
        elif self.current.valence < -0.3:
            return "narrow"     # 消极情绪聚焦威胁
        else:
            return "moderate"

    def _get_metaphor_likelihood(self) -> float:
        """情感状态影响使用隐喻的倾向"""
        # 中等激活度时隐喻最多
        a = self.current.arousal
        return 0.3 + 0.5 * math.exp(-((a - 0.5) ** 2) / 0.1)

    def regulate(self, strategy: str = "cognitive_reappraisal") -> str:
        """
        情绪调节

        策略：
        - cognitive_reappraisal: 认知重评（改变对事件的解释）
        - attention_shift: 注意转移
        - acceptance: 接纳
        - suppression: 表达抑制
        """
        self.regulation_history.append({
            "strategy": strategy,
            "before": self.current.to_dict(),
            "timestamp": datetime.now()
        })

        if strategy == "cognitive_reappraisal":
            # 认知重评：尝试从更积极的角度重新解释
            if self.current.valence < 0:
                # 对负面情绪的积极重构
                self.current.valence = min(0.0, self.current.valence + 0.3)
                self.current.arousal = max(0.3, self.current.arousal - 0.1)
                return "尝试从另一个角度看待这个情况..."

        elif strategy == "attention_shift":
            # 注意转移
            self.current.arousal *= 0.8
            return "让我暂时把注意力放在别的事情上..."

        elif strategy == "acceptance":
            # 接纳：允许情绪存在
            self.current.arousal = max(0.3, self.current.arousal - 0.15)
            return "这种感受是合理的，让它存在吧..."

        elif strategy == "suppression":
            # 表达抑制（不表达，但内部仍然存在）
            return "我选择不表现出这种情绪..."

        return ""

    def influence_memory_retrieval(self, memory_valence: float) -> float:
        """
        情感一致性效应：当前情绪影响记忆检索

        返回记忆被提取的概率权重
        """
        current_v = self.current.valence
        # 情绪一致的记忆更容易被提取
        valence_match = 1.0 - abs(current_v - memory_valence)
        return max(0.1, valence_match)

    def influence_attention(self, stimulus_salience: float,
                           stimulus_valence: float) -> float:
        """
        情感对注意力的影响

        消极情绪时更容易注意威胁性刺激
        """
        if self.current.valence < -0.3:
            # 消极时，负面刺激获得额外注意
            if stimulus_valence < 0:
                return stimulus_salience * 1.5
        elif self.current.valence > 0.3:
            # 积极时，所有刺激注意范围扩大
            return stimulus_salience * 1.2

        return stimulus_salience

    def to_dict(self) -> Dict:
        return {
            "current": self.current.to_dict(),
            "dominant_emotion": self.get_dominant_emotion(),
            "expression": self.get_emotion_expression(),
            "history_length": len(self.history)
        }
