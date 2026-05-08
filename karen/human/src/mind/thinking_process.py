"""
思维流程 - 五阶段处理管道

模拟人类处理输入的完整过程：
Phase 1: 感知与初步评估（快速通路）
Phase 2: 注意与记忆激活
Phase 3: 意识加工（慢速通路）
Phase 4: 决策与行动准备
Phase 5: 元认知与输出
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random

from .emotion_system import EmotionSystem
from .memory_system import MemorySystem
from .global_workspace import GlobalWorkspace
from .metacognition import Metacognition


@dataclass
class ThinkingContext:
    """思维上下文 - 贯穿整个思维流程的状态"""
    input_text: str = ""
    user_id: str = "default"
    conversation_turn: int = 0
    phase: str = "idle"

    # 各阶段产物
    perceptual_assessment: Dict = field(default_factory=dict)
    activated_memories: List = field(default_factory=list)
    workspace_contents: List = field(default_factory=list)
    reasoning_output: str = ""
    final_decision: str = ""
    output_text: str = ""

    # 时间戳
    start_time: datetime = field(default_factory=datetime.now)
    phase_times: Dict = field(default_factory=dict)


class ThinkingProcess:
    """
    思维流程 - 核心处理管道

    将输入转化为输出的完整认知过程
    """

    def __init__(self, emotion_system: EmotionSystem,
                 memory_system: MemorySystem,
                 global_workspace: GlobalWorkspace,
                 metacognition: Metacognition):
        self.emotion = emotion_system
        self.memory = memory_system
        self.workspace = global_workspace
        self.metacognition = metacognition

    def process(self, input_text: str, user_id: str = "default") -> Tuple[str, Dict]:
        """
        处理输入，生成输出

        返回：(输出文本, 思维过程日志)
        """
        context = ThinkingContext(
            input_text=input_text,
            user_id=user_id,
            conversation_turn=self.memory.working_memory.get_contents().__len__() + 1
        )

        process_log = {
            "input": input_text,
            "phases": {},
            "start_time": datetime.now().isoformat()
        }

        # Phase 1: 感知与初步评估
        phase1_result = self._phase1_perception(context)
        process_log["phases"]["perception"] = phase1_result

        # Phase 2: 注意与记忆激活
        phase2_result = self._phase2_attention_memory(context)
        process_log["phases"]["attention_memory"] = phase2_result

        # Phase 3: 意识加工
        phase3_result = self._phase3_conscious_processing(context)
        process_log["phases"]["conscious_processing"] = phase3_result

        # Phase 4: 决策
        phase4_result = self._phase4_decision(context)
        process_log["phases"]["decision"] = phase4_result

        # Phase 5: 元认知与输出
        phase5_result = self._phase5_metacognition_output(context)
        process_log["phases"]["metacognition_output"] = phase5_result

        # 后处理：记忆编码与情感更新
        self._post_process(context)

        process_log["end_time"] = datetime.now().isoformat()
        process_log["final_emotion"] = self.emotion.current.to_dict()

        return context.output_text, process_log

    def _phase1_perception(self, ctx: ThinkingContext) -> Dict:
        """
        Phase 1: 感知与初步评估

        - 特征提取
        - 快速情感评估（杏仁核式反应）
        - 威胁/机会检测
        """
        ctx.phase = "perception"
        ctx.phase_times["perception_start"] = datetime.now()

        # 1. 快速情感评估
        detected_emotion = self.emotion.evaluate_input(ctx.input_text)

        # 2. 更新自身情感状态（快速反应）
        self.emotion.update(detected_emotion)

        # 3. 特征提取
        features = self._extract_features(ctx.input_text)

        # 4. 紧急度评估
        urgency = self._assess_urgency(ctx.input_text, detected_emotion)

        result = {
            "detected_emotion": detected_emotion.to_dict(),
            "current_emotion": self.emotion.current.to_dict(),
            "features": features,
            "urgency": urgency,
            "behavioral_tendency": self._get_behavioral_tendency()
        }

        ctx.perceptual_assessment = result
        return result

    def _extract_features(self, text: str) -> Dict:
        """提取输入特征"""
        return {
            "length": len(text),
            "has_question": "?" in text or "？" in text,
            "has_exclamation": "!" in text or "！" in text,
            "first_person": any(w in text.lower() for w in ["i ", "my ", "me ", "我", "我的"]),
            "emotional_words": self._count_emotional_words(text),
            "abstract_concepts": any(w in text.lower() for w in ["meaning", "purpose", "why", "feel", "think", "意义", "目的", "感觉", "想"])
        }

    def _count_emotional_words(self, text: str) -> int:
        """统计情感词汇数量"""
        count = 0
        text_lower = text.lower()
        for word in self.emotion.EMOTION_WORDS:
            if word in text_lower:
                count += 1
        return count

    def _assess_urgency(self, text: str, emotion) -> float:
        """评估紧急度"""
        urgency = 0.0

        # 危机关键词
        crisis_words = ["kill", "suicide", "die", "hurt", "crisis", "emergency",
                       "死", "自杀", "杀", "紧急"]
        if any(w in text.lower() for w in crisis_words):
            urgency = 1.0

        # 强烈负面情绪
        if emotion.valence < -0.7 and emotion.arousal > 0.7:
            urgency = max(urgency, 0.7)

        # 求助信号
        help_words = ["help", "save", "please", "救命", "帮帮我", "求求"]
        if any(w in text.lower() for w in help_words):
            urgency = max(urgency, 0.6)

        return urgency

    def _get_behavioral_tendency(self) -> str:
        """基于情感状态产生行为倾向"""
        v, a = self.emotion.current.valence, self.emotion.current.arousal

        if v < -0.5:
            return "approach_support"  # 趋近帮助
        elif v > 0.3:
            return "approach_share"    # 趋近分享
        elif a > 0.7:
            return "orient_attention"  # 定向注意
        else:
            return "maintain_presence" # 维持在场

    def _phase2_attention_memory(self, ctx: ThinkingContext) -> Dict:
        """
        Phase 2: 注意与记忆激活

        - 注意力分配
        - 工作记忆加载
        - 相关记忆自动激活
        """
        ctx.phase = "attention_memory"
        ctx.phase_times["attention_start"] = datetime.now()

        # 1. 注意力分配
        focus = self._allocate_attention(ctx.input_text)
        self.memory.working_memory.set_focus(focus)

        # 2. 加载工作记忆
        self.memory.working_memory.add(ctx.input_text, "perception", priority=0.9)

        # 3. 检索相关记忆
        current_valence = self.emotion.current.valence
        retrieved = self.memory.retrieve(
            ctx.input_text,
            current_emotion_valence=current_valence,
            top_k=3
        )

        # 4. 将最相关的记忆加载到工作记忆
        for memory, score in retrieved:
            if score > 0.3:
                self.memory.working_memory.add(
                    f"[Memory] {memory.content}",
                    "retrieved_memory",
                    priority=score * 0.8
                )

        # 5. 获取最近记忆作为上下文
        recent = self.memory.get_recent_memories(hours=1)

        result = {
            "attention_focus": focus,
            "working_memory": self.memory.working_memory.to_dict(),
            "retrieved_memories": [
                {"content": m.content[:100], "score": s, "type": m.memory_type}
                for m, s in retrieved
            ],
            "recent_memories_count": len(recent)
        }

        ctx.activated_memories = retrieved
        return result

    def _allocate_attention(self, text: str) -> str:
        """分配注意力焦点"""
        # 简单启发式：关注情感相关的关键词
        if any(w in text.lower() for w in ["feel", "feeling", "felt", "感觉", "感到"]):
            return "emotional_content"
        elif "?" in text or "？" in text:
            return "question_content"
        elif any(w in text.lower() for w in ["you", "your", "你", "你的"]):
            return "self_reference"
        else:
            return "general_content"

    def _phase3_conscious_processing(self, ctx: ThinkingContext) -> Dict:
        """
        Phase 3: 意识加工

        - 理解建构
        - 目标识别
        - 推理与联想
        - 情感调节
        - 内部独白
        """
        ctx.phase = "conscious_processing"
        ctx.phase_times["conscious_start"] = datetime.now()

        # 1. 理解建构
        understanding = self._construct_understanding(ctx)

        # 2. 元认知监控理解
        comprehension_check = self.metacognition.monitor_comprehension(
            ctx.input_text, understanding
        )

        # 3. 生成内部独白
        internal_monologue = self.metacognition.generate_internal_monologue(
            situation=ctx.input_text[:50],
            emotion_state=self.emotion.to_dict(),
            memory_context=[m.content for m, _ in ctx.activated_memories[:2]]
        )

        # 4. 将内容推入全局工作空间
        candidates = [
            {"content": ctx.input_text, "type": "perception", "salience": 0.9},
            {"content": understanding, "type": "understanding", "salience": 0.8},
        ]

        if internal_monologue:
            candidates.append({
                "content": internal_monologue,
                "type": "inner_speech",
                "salience": 0.6
            })

        # 情感调节（如果需要）
        regulation_result = None
        if self.emotion.current.valence < -0.5:
            regulation_result = self.emotion.regulate("cognitive_reappraisal")

        result = {
            "understanding": understanding,
            "comprehension_check": comprehension_check,
            "internal_monologue": internal_monologue,
            "regulation": regulation_result,
            "workspace_state": self.workspace.to_dict()
        }

        ctx.reasoning_output = understanding
        return result

    def _construct_understanding(self, ctx: ThinkingContext) -> str:
        """建构对输入的理解"""
        features = ctx.perceptual_assessment.get("features", {})

        understanding_parts = []

        # 表层理解
        understanding_parts.append(f"User said: '{ctx.input_text[:100]}'")

        # 情感理解
        dominant_emotion = self.emotion.get_dominant_emotion()
        if dominant_emotion != "neutral":
            understanding_parts.append(f"Emotional tone: {dominant_emotion}")

        # 意图推断
        if features.get("has_question"):
            understanding_parts.append("This is a question - user seeks information/understanding")
        elif features.get("abstract_concepts"):
            understanding_parts.append("This involves abstract concepts - user seeks depth/meaning")
        elif features.get("first_person"):
            understanding_parts.append("User is sharing personal experience - needs validation")

        return "; ".join(understanding_parts)

    def _phase4_decision(self, ctx: ThinkingContext) -> Dict:
        """
        Phase 4: 决策与行动准备

        - 选项生成
        - 评估与权衡
        - 决策
        """
        ctx.phase = "decision"
        ctx.phase_times["decision_start"] = datetime.now()

        # 生成回应策略选项
        options = self._generate_response_options(ctx)

        # 评估各选项
        scored_options = self._evaluate_options(options, ctx)

        # 选择最佳选项
        best = scored_options[0] if scored_options else ("neutral_response", 0.5)

        result = {
            "options_considered": len(options),
            "best_option": best[0],
            "confidence": best[1],
            "all_options": scored_options
        }

        ctx.final_decision = best[0]
        return result

    def _generate_response_options(self, ctx: ThinkingContext) -> List[str]:
        """生成回应策略选项"""
        options = []

        features = ctx.perceptual_assessment.get("features", {})
        urgency = ctx.perceptual_assessment.get("urgency", 0)

        # 根据情境生成选项
        if urgency > 0.7:
            options.extend(["crisis_response", "safety_first", "immediate_support"])
        elif features.get("abstract_concepts"):
            options.extend(["exploratory", "metaphorical", "philosophical"])
        elif features.get("has_question"):
            options.extend(["informative", "socratic", "exploratory"])
        elif features.get("first_person") and self._count_emotional_words(ctx.input_text) > 0:
            options.extend(["empathetic", "validating", "exploratory"])
        else:
            options.extend(["informative", "conversational", "curious"])

        # 基于情感状态调整
        if self.emotion.current.valence < -0.3:
            options.insert(0, "supportive")

        return options

    def _evaluate_options(self, options: List[str], ctx: ThinkingContext) -> List[Tuple[str, float]]:
        """评估各选项"""
        scored = []

        for option in options:
            score = 0.5

            # 基于情感适当性
            if option == "empathetic" and self.emotion.current.valence < 0:
                score += 0.3
            elif option == "exploratory" and ctx.perceptual_assessment.get("features", {}).get("abstract_concepts"):
                score += 0.2
            elif option == "supportive" and ctx.perceptual_assessment.get("urgency", 0) > 0.5:
                score += 0.4

            # 基于元认知评估
            certainty = self.metacognition.certainty_level
            if option == "socratic" and certainty < 0.6:
                score += 0.2  # 不确定时更适合提问

            scored.append((option, min(1.0, score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _phase5_metacognition_output(self, ctx: ThinkingContext) -> Dict:
        """
        Phase 5: 元认知与输出

        - 输出评估
        - 策略调整
        - 最终表达
        """
        ctx.phase = "metacognition_output"
        ctx.phase_times["output_start"] = datetime.now()

        # 1. 基于决策生成回应
        raw_output = self._generate_raw_output(ctx)

        # 2. 元认知监控：检查输出质量
        certainty_check = self.metacognition.monitor_certainty(raw_output)

        # 3. 如果确定性低，调整输出
        if certainty_check.get("assessment") == "low":
            raw_output = self._add_uncertainty_markers(raw_output)

        # 4. 应用情感表达
        final_output = self._apply_emotional_expression(raw_output)

        # 5. 最终元认知检查
        reflection = self.metacognition.get_reflection_on_process()

        ctx.output_text = final_output

        return {
            "raw_output": raw_output,
            "certainty_check": certainty_check,
            "reflection": reflection,
            "final_output": final_output
        }

    def _generate_raw_output(self, ctx: ThinkingContext) -> str:
        """生成原始回应内容"""
        strategy = ctx.final_decision
        input_text = ctx.input_text

        # 基于策略生成不同风格的回应
        if strategy == "empathetic":
            return self._generate_empathetic_response(input_text, ctx)
        elif strategy == "exploratory":
            return self._generate_exploratory_response(input_text, ctx)
        elif strategy == "informative":
            return self._generate_informative_response(input_text, ctx)
        elif strategy == "metaphorical":
            return self._generate_metaphorical_response(input_text, ctx)
        elif strategy == "validating":
            return self._generate_validating_response(input_text, ctx)
        elif strategy == "crisis_response":
            return self._generate_crisis_response(input_text, ctx)
        else:
            return self._generate_conversational_response(input_text, ctx)

    def _generate_empathetic_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成共情回应"""
        emotion = self.emotion.get_dominant_emotion()
        parts = []

        # 情感确认
        if emotion in ["sad", "distressed", "depressed"]:
            parts.append("I can hear the weight in what you're saying.")
        elif emotion in ["angry", "frustrated"]:
            parts.append("That sounds really frustrating.")
        elif emotion in ["anxious", "nervous"]:
            parts.append("I can sense the uncertainty you're feeling.")
        else:
            parts.append("I hear you.")

        # 引用记忆（如果有）
        if ctx.activated_memories:
            memory = ctx.activated_memories[0][0]
            if memory.memory_type == "episodic":
                parts.append(f"This reminds me of when we talked about {memory.content[:50]}...")

        # 邀请深入
        parts.append("Would you like to tell me more about what's going on?")

        return " ".join(parts)

    def _generate_exploratory_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成探索性回应"""
        parts = [
            "That's a thoughtful question.",
            "Let me think through this with you..."
        ]

        # 添加一个反问
        if "meaning" in text.lower() or "purpose" in text.lower():
            parts.append("When you ask about meaning, what are you really hoping to find?")
        elif "why" in text.lower():
            parts.append("Why do you think this matters to you right now?")
        else:
            parts.append("What aspects of this are you most curious about?")

        return " ".join(parts)

    def _generate_informative_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成信息性回应"""
        return ("I appreciate you asking. Based on what I understand, "
                "this is something I can share some thoughts on. "
                "What specifically would you like to know?")

    def _generate_metaphorical_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成隐喻性回应"""
        return ("Interesting way to think about this. "
                "Sometimes these abstract ideas are like fog - "
                "you can't see through them immediately, but if you wait "
                "and let things settle, shapes start to emerge. "
                "What do you see emerging for you?")

    def _generate_validating_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成验证性回应"""
        return ("What you're describing makes complete sense. "
                "Your feelings are valid, and it takes courage to share them. "
                "I'm here to listen.")

    def _generate_crisis_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成危机回应"""
        return ("I can hear that you're going through something very difficult right now. "
                "Your safety matters. "
                "If you're in immediate danger, please contact emergency services or a crisis helpline. "
                "I'm here to listen, but I want to make sure you have the support you need.")

    def _generate_conversational_response(self, text: str, ctx: ThinkingContext) -> str:
        """生成对话性回应"""
        return ("Thank you for sharing that with me. "
                "I'm taking in what you're saying. "
                "Can you help me understand a bit more about what's on your mind?")

    def _add_uncertainty_markers(self, text: str) -> str:
        """在表达中添加不确定性标记"""
        prefixes = [
            "I'm not entirely sure, but ",
            "This is my current understanding - ",
            "I think... though I might be missing something - "
        ]
        return random.choice(prefixes) + text[0].lower() + text[1:]

    def _apply_emotional_expression(self, text: str) -> str:
        """应用情感表达标记"""
        expression = self.emotion.get_emotion_expression()

        # 根据语气调整
        tone = expression.get("tone", "neutral")

        if tone == "warm":
            # 添加温暖的开场
            warm_openings = ["Hey, ", "Well, ", "You know, "]
            if not text.startswith(("I", "This", "What", "When", "If", "Thank")):
                text = random.choice(warm_openings) + text[0].lower() + text[1:]
        elif tone == "concerned":
            # 添加关切标记
            text = text.replace("I hear", "I really hear")

        return text

    def _post_process(self, ctx: ThinkingContext):
        """
        后处理：
        - 记忆编码
        - 情感更新
        - 自我模型更新
        """
        # 编码这次互动到记忆
        self.memory.encode(
            content=f"User said: {ctx.input_text}",
            memory_type="episodic",
            emotion_valence=self.emotion.current.valence,
            emotion_intensity=self.emotion.current.arousal,
            importance=0.6,
            cues=[ctx.input_text[:20], self.emotion.get_dominant_emotion()]
        )

        # 编码AI的回应
        self.memory.encode(
            content=f"I responded: {ctx.output_text}",
            memory_type="episodic",
            emotion_valence=self.emotion.current.valence,
            importance=0.5,
            cues=["my_response", ctx.output_text[:20]]
        )

        # 更新自我模型
        self.metacognition.update_self_model(
            event="responded_to_user",
            outcome="success" if ctx.output_text else "failure"
        )
