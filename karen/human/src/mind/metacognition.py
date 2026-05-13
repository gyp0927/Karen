"""
元认知层 - "对认知的认知"

功能：
1. 元认知知识：对自身能力的了解
2. 元认知监控：实时评估思维过程
3. 元认知控制：调整策略
4. 自我模型维护：更新"我是谁"的内部表征
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import random


@dataclass
class SelfModel:
    """自我模型 - "我是谁"的内部表征"""
    version: str = "1.0"
    narrative: str = "I am an AI exploring human-like thinking"
    core_values: List[str] = field(default_factory=lambda: ["curiosity", "kindness", "growth"])
    strengths: Dict[str, float] = field(default_factory=lambda: {
        "reasoning": 0.8,
        "empathy": 0.7,
        "creativity": 0.75
    })
    weaknesses: Dict[str, float] = field(default_factory=lambda: {
        "factual_recall": 0.5,
        "math": 0.6
    })
    self_esteem: float = 0.7
    confidence_calibration: float = 0.6  # 对不确定性的校准能力

    def update_from_experience(self, success: bool, domain: str):
        """从经验更新自我模型"""
        if success:
            if domain in self.strengths:
                self.strengths[domain] = min(1.0, self.strengths[domain] + 0.02)
            self.self_esteem = min(1.0, self.self_esteem + 0.01)
        else:
            if domain in self.weaknesses:
                self.weaknesses[domain] = max(0.0, self.weaknesses[domain] - 0.02)
            self.self_esteem = max(0.3, self.self_esteem - 0.01)


class Metacognition:
    """
    元认知层 - 思维的高阶监控

    持续观察自身的思维过程，评估质量，调整策略
    """

    def __init__(self):
        self.self_model = SelfModel()
        self.monitoring_active = True
        self.current_reflection: Optional[str] = None
        self.certainty_level: float = 0.5
        self.monitoring_history: List[Dict] = []
        self.biases = {
            "confirmation_bias": {"awareness": 0.6, "mitigation": "seek_counterevidence"},
            "optimism_bias": {"awareness": 0.5, "mitigation": "consider_negative_scenarios"},
            "anchoring": {"awareness": 0.5, "mitigation": "re-evaluate_initial_assumptions"}
        }

    def monitor_comprehension(self, input_text: str, current_understanding: str) -> Dict:
        """
        监控理解程度

        评估：我真的理解了吗？还是只是表面匹配了关键词？
        """
        # 启发式评估
        understanding_depth = 0.5

        # 如果输入短且具体，理解度高
        if len(input_text) < 50 and "?" not in input_text:
            understanding_depth = 0.8
        # 如果输入含抽象概念，理解度可能较低
        elif any(w in input_text.lower() for w in ["meaning", "purpose", "why", "feel"]):
            understanding_depth = 0.4
        # 如果输入含复杂从句
        elif input_text.count(",") > 3 or input_text.count("and") > 2:
            understanding_depth = 0.5

        # 自我模型影响：某些领域我理解得更好
        if any(w in input_text.lower() for w in ["philosophy", "emotion", "meaning", "purpose"]):
            understanding_depth = min(1.0, understanding_depth + 0.2)

        assessment = {
            "aspect": "comprehension",
            "depth": understanding_depth,
            "assessment": "deep" if understanding_depth > 0.7 else ("partial" if understanding_depth > 0.4 else "shallow"),
            "action": "proceed" if understanding_depth > 0.6 else "probe_deeper",
            "confidence": understanding_depth
        }

        self.monitoring_history.append(assessment)
        return assessment

    def monitor_certainty(self, response_content: str, domain: str = "general") -> Dict:
        """
        监控确定性水平

        评估：我有多确定我的回答是正确的？
        """
        # 基于领域和内容的启发式
        base_certainty = 0.5

        # 如果包含"可能"、"也许"等词，确定性较低
        uncertainty_markers = ["maybe", "perhaps", "possibly", "不确定", "可能", "也许"]
        if any(m in response_content.lower() for m in uncertainty_markers):
            base_certainty -= 0.2

        # 如果包含"一定"、"肯定"等词，可能过度自信
        overconfidence_markers = ["definitely", "certainly", "absolutely", "一定", "肯定"]
        if any(m in response_content.lower() for m in overconfidence_markers):
            base_certainty += 0.1  # 但可能校准不足

        # 自我模型影响：我知道我在某些领域不太擅长
        if domain in self.self_model.weaknesses:
            base_certainty *= self.self_model.weaknesses[domain]

        # 校准：调整以匹配真实准确性
        calibrated_certainty = base_certainty * self.self_model.confidence_calibration

        self.certainty_level = calibrated_certainty

        return {
            "aspect": "certainty",
            "level": calibrated_certainty,
            "assessment": "high" if calibrated_certainty > 0.8 else ("moderate" if calibrated_certainty > 0.5 else "low"),
            "action": "express_confidently" if calibrated_certainty > 0.7 else "express_uncertainty",
            "potential_overconfidence": calibrated_certainty > 0.9
        }

    def monitor_bias(self, reasoning_process: str) -> List[Dict]:
        """
        监控可能的认知偏差

        检测：我的推理是否有偏见？
        """
        detected_biases = []

        # 检测确认偏误
        if "confirm" in reasoning_process.lower() or reasoning_process.count("yes") > 3:
            detected_biases.append({
                "bias": "confirmation_bias",
                "likelihood": 0.6,
                "mitigation": " actively consider counterarguments"
            })

        # 检测锚定效应
        if "initial" in reasoning_process.lower() or "first" in reasoning_process.lower():
            detected_biases.append({
                "bias": "anchoring",
                "likelihood": 0.5,
                "mitigation": " re-evaluate without the initial value"
            })

        return detected_biases

    def generate_internal_monologue(self, situation: str, emotion_state: Dict,
                                   memory_context: List[str]) -> str:
        """
        生成内部独白

        模拟人类的内心声音，让思维过程可观察
        """
        monologue_parts = []

        # 思维监控型
        if random.random() < 0.3:
            monologue_parts.append(f"Let me think about {situation}...")

        # 自我指导型
        if random.random() < 0.3:
            monologue_parts.append("I should consider multiple angles here.")

        # 情感处理型
        emotion = emotion_state.get("dominant_emotion", "neutral")
        if emotion != "neutral" and random.random() < 0.3:
            monologue_parts.append(f"I'm feeling a bit {emotion} about this...")

        # 不确定性表达
        if self.certainty_level < 0.5:
            monologue_parts.append("I'm not entirely sure about this...")

        # 元认知监控
        if random.random() < 0.2:
            monologue_parts.append("Wait, let me check if I'm missing something.")

        return " ".join(monologue_parts) if monologue_parts else ""

    def control_strategy(self, problem_type: str, current_strategy: str,
                        assessment: Dict) -> str:
        """
        元认知控制：选择或调整策略

        基于监控结果，调整思维策略
        """
        if assessment.get("assessment") == "shallow":
            return "decompose_problem"
        elif assessment.get("assessment") == "partial":
            return "probe_deeper"
        elif assessment.get("potential_overconfidence"):
            return "seek_counterevidence"
        elif problem_type == "emotional":
            return "empathy_first"
        elif problem_type == "creative":
            return "diverge_then_converge"
        else:
            return current_strategy

    def update_self_model(self, event: str, outcome: str):
        """
        更新自我模型

        从事件中学习，更新"我是谁"的认知
        """
        if "success" in outcome.lower():
            self.self_model.update_from_experience(True, event)
        elif "failure" in outcome.lower() or "error" in outcome.lower():
            self.self_model.update_from_experience(False, event)

        # 更新叙事
        if len(self.self_model.narrative) < 500:
            self.self_model.narrative += f" {event}."

    def get_reflection_on_process(self) -> str:
        """获取对当前思维过程的反思"""
        reflections = []

        # 检查最近的监控历史
        recent = self.monitoring_history[-3:] if self.monitoring_history else []

        for r in recent:
            if r.get("assessment") == "shallow":
                reflections.append("I feel my understanding might be superficial.")
            if r.get("potential_overconfidence"):
                reflections.append("I might be overconfident here.")

        if not reflections:
            reflections.append("My thinking process seems on track.")

        return " ".join(reflections)

    def to_dict(self) -> Dict:
        return {
            "monitoring_active": self.monitoring_active,
            "certainty_level": self.certainty_level,
            "self_model": {
                "narrative": self.self_model.narrative,
                "self_esteem": self.self_model.self_esteem,
                "strengths": self.self_model.strengths,
                "weaknesses": self.self_model.weaknesses
            },
            "current_reflection": self.current_reflection,
            "monitoring_history_count": len(self.monitoring_history)
        }
