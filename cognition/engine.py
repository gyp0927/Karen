"""认知引擎 - 合并5个"人类思维"子系统。

包含：
1. EmotionalStateManager - 情感状态(按 (sid, agent_name) 双键索引,多用户安全)
2. InnerMonologueEngine - 内心独白
3. IntuitionEngine - 直觉引擎
4. MetacognitionEngine - 元认知
5. PersonaManager - 人格系统(只读 agent 类型 → persona 映射,所有用户共享)
"""

import copy
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from cognition.types import (
    CognitiveState, EmotionalState, InnerThought, IntuitionResult,
    MetacognitionResult, Mood, PersonaConfig, ThinkingMode,
)

logger = logging.getLogger(__name__)

# ========================================================================
# 1. 情感状态
# ========================================================================

class EmotionalStateManager:
    """管理所有Agent的情感状态。

    存储按 ``(sid, agent_name)`` 双键索引,避免多用户(同一进程多 SocketIO 客户端)
    互相覆盖情感。sid 留空时退化为单用户(向后兼容控制台/单用户场景)。
    使用 LRU 策略清理：超出上限时移除最久未访问的条目。
    """

    _MAX_STATES = 10000

    def __init__(self):
        self._states: dict[tuple[str, str], EmotionalState] = {}
        self._access_times: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def get_state(self, agent_name: str, sid: str = "") -> EmotionalState:
        key = (sid or "", agent_name)
        with self._lock:
            if key not in self._states:
                # 防止无限制增长：超出上限时清理最久未访问的条目 (LRU)
                # LRU 清理：超出上限时持续移除最旧条目，直到降到 75% 上限以下
                while len(self._states) >= self._MAX_STATES:
                    sorted_keys = sorted(self._access_times, key=self._access_times.get)
                    to_remove = sorted_keys[:max(1, len(sorted_keys) // 4)]
                    for k in to_remove:
                        self._states.pop(k, None)
                        self._access_times.pop(k, None)
                self._states[key] = EmotionalState()
            self._access_times[key] = time.time()
            return self._states[key]

    def update_after_interaction(
        self, agent_name: str, success: bool = True, complexity: float = 0.5,
        user_emotion_hint: Optional[str] = None,
        sid: str = "",
    ) -> EmotionalState:
        state = self.get_state(agent_name, sid=sid)
        state.update("success" if success else "failure", complexity)
        if user_emotion_hint:
            self._respond_to_user_emotion(state, user_emotion_hint)
        return state

    def reset(self, sid: str) -> None:
        """会话结束/切换时清掉这个 sid 的所有 agent 情感状态。"""
        target = sid or ""
        with self._lock:
            self._states = {k: v for k, v in self._states.items() if k[0] != target}
            self._access_times = {k: t for k, t in self._access_times.items() if k[0] != target}

    def _respond_to_user_emotion(self, state: EmotionalState, hint: str) -> None:
        hint = hint.lower()
        if any(w in hint for w in ["怒", "生气", "angry", "frustrated"]):
            state.urgency = min(1.0, state.urgency + 0.3)
            state.mood = Mood.CAUTIOUS
        elif any(w in hint for w in ["困惑", "不懂", "confused"]):
            state.confidence = max(0.1, state.confidence - 0.1)
            state.curiosity = min(1.0, state.curiosity + 0.2)
            state.mood = Mood.CURIOUS
        elif any(w in hint for w in ["急", "urgent", "快", " hurry"]):
            state.urgency = min(1.0, state.urgency + 0.4)
        elif any(w in hint for w in ["谢", "感谢", "thank", "good", "棒", "厉害"]):
            state.confidence = min(1.0, state.confidence + 0.15)
            state.fatigue = max(0.0, state.fatigue - 0.2)
            state.mood = Mood.SATISFIED


# ========================================================================
# 2. 内心独白
# ========================================================================

class InnerMonologueEngine:
    """生成和管理Agent的内心独白"""

    TRIGGER_ALWAYS = "always"
    TRIGGER_COMPLEX = "complex"
    TRIGGER_NEVER = "never"

    def __init__(self, trigger_mode: str = TRIGGER_COMPLEX):
        self.trigger_mode = trigger_mode

    def should_think(self, agent_name: str, query: str, cognitive_state: CognitiveState) -> bool:
        if self.trigger_mode == self.TRIGGER_NEVER:
            return False
        if self.trigger_mode == self.TRIGGER_ALWAYS:
            return True
        # <think>/<answer> 会让模型多生成一倍 token,拉高 TTFT;只在真正复杂时触发。
        complexity = self._estimate_complexity(query)
        emotional = cognitive_state.emotional
        return (
            complexity > 0.85
            or (emotional.confidence < 0.4 and emotional.curiosity > 0.7)
            or len(query) > 200
        )

    def _estimate_complexity(self, query: str) -> float:
        score = min(0.3, len(query) / 500)
        for kw in ["分析", "比较", "对比", "评估", "为什么", "如何", "如果",
                   "analyze", "compare", "evaluate", "why", "how",
                   "区别", "差异", "影响", "原因", "机制"]:
            if kw in query.lower():
                score += 0.1
        score += min(0.2, (query.count("?") + query.count("？")) * 0.1)
        return min(1.0, score)

    def generate_thought_prompt(self, agent_name: str, query: str,
                                mode: ThinkingMode = ThinkingMode.INTUITION) -> str:
        mode_hints = {
            ThinkingMode.INTUITION: "凭直觉快速反应，第一反应是什么",
            ThinkingMode.REASONING: "仔细分析，分步骤思考",
            ThinkingMode.REFLECTIVE: "跳出问题本身，反思自己的思考过程",
        }
        hint = mode_hints.get(mode, mode_hints[ThinkingMode.INTUITION])
        return f"""
【内心独白】在正式回答之前，请先用一句话表达你此刻的真实想法（以"我想："开头）。
思考方向：{hint}
然后给出你的正式回答。

输出格式：
<think>我想：[你的内心想法]</think>
<answer>[正式回答]</answer>
"""

    def extract_thought_and_answer(self, raw_response: str) -> tuple[Optional[str], str]:
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)
        if think_match and answer_match:
            thought = re.sub(r"^我想[:：]\s*", "", think_match.group(1).strip())
            return thought, answer_match.group(1).strip()

        lines = raw_response.split("\n")
        thought_lines, answer_lines = [], []
        in_thought = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("我想：") or stripped.startswith("我想:"):
                in_thought = True
                thought_lines.append(re.sub(r"^我想[:：]\s*", "", stripped))
            elif in_thought and stripped and len(stripped) > 20 and not stripped.startswith("想"):
                in_thought = False
                answer_lines.append(stripped)
            elif in_thought and stripped:
                thought_lines.append(stripped)
            else:
                answer_lines.append(line)

        if thought_lines:
            thought = " ".join(thought_lines)
            answer = "\n".join(answer_lines).strip()
            return thought, answer if answer else raw_response
        return None, raw_response

    def record_thought(self, agent_name: str, thought: str,
                       cognitive_state: CognitiveState, mode: ThinkingMode) -> None:
        if thought:
            cognitive_state.record_thought(agent_name, thought, mode)


# ========================================================================
# 3. 直觉引擎
# ========================================================================

@dataclass
class IntuitionPattern:
    pattern: str
    intent: str
    gut_feeling: str
    action: str
    confidence_boost: float = 0.0


INTUITION_PATTERNS = [
    IntuitionPattern(r"^(你好|嗨|hello|hi|hey|在吗|在么)", "greeting", "用户在打招呼", "友好回应", 0.9),
    IntuitionPattern(r"^(谢谢|感谢|thank)", "gratitude", "用户在表达感谢", "谦逊回应", 0.8),
    IntuitionPattern(r"^(再见|拜拜|bye|goodbye)", "farewell", "用户要结束对话", "友好告别", 0.9),
    IntuitionPattern(r"^(什么|what).*(时间|天气|名字)", "simple_fact", "简单事实查询", "直接回答", 0.8),
    IntuitionPattern(r"^(怎么|how).*(做|用|安装|开始)", "how_to", "操作指南类问题", "给出步骤说明", 0.7),
    IntuitionPattern(r".*(烦|生气|讨厌|垃圾|差劲|stupid|hate|angry)", "user_frustrated", "用户情绪负面", "先安抚情绪", 0.8),
    IntuitionPattern(r".*(开心|棒|厉害|awesome|great|love)", "user_happy", "用户情绪正面", "分享喜悦", 0.8),
    IntuitionPattern(r".*(搜索|查找|找一下|search|look up|find)", "search", "需要搜索信息", "触发搜索", 0.7),
    IntuitionPattern(r".*(最新|新闻|趋势|news|latest|trend)", "news", "时效性信息需求", "必须联网搜索", 0.8),
    IntuitionPattern(r".*(代码|编程|bug|error|python|javascript|写个程序)", "coding", "编程相关问题", "给出代码示例", 0.7),
    IntuitionPattern(r".*(区别|对比|比较|vs|versus|difference|compare)", "comparison", "比较类问题", "结构化对比", 0.6),
    IntuitionPattern(r".*(建议|推荐|应该|怎么办|advice|recommend|should)", "advice", "寻求建议", "给出建议", 0.6),
    IntuitionPattern(r".*(难过|伤心|失恋|压力|焦虑|sad|depressed|anxious)", "emotional_support", "情感支持需求", "温柔倾听", 0.7),
    IntuitionPattern(r".*(写|创作|故事|poem|story|creative|imagine)", "creative", "创意生成需求", "发挥想象力", 0.7),
]


class IntuitionEngine:
    """直觉引擎——模仿人类的系统1思维"""

    _CACHE_TTL_SECONDS = 3600  # 缓存 1 小时过期
    _CACHE_MAX_SIZE = 1000

    def __init__(self):
        import threading
        self.patterns = INTUITION_PATTERNS
        self._experience_cache: dict[str, IntuitionResult] = {}
        self._cache_times: dict[str, float] = {}
        self._cache_lock = threading.Lock()

    def classify(self, query: str, history_length: int = 0) -> IntuitionResult:
        cache_key = query.lower().strip()[:50]
        now = time.time()
        if cache_key in self._experience_cache:
            cached_time = self._cache_times.get(cache_key, 0)
            if now - cached_time < self._CACHE_TTL_SECONDS:
                return self._experience_cache[cache_key]

        best_match = None
        best_score = 0.0
        for pattern in self.patterns:
            if re.search(pattern.pattern, query, re.IGNORECASE):
                score = len(query) / 100 + pattern.confidence_boost
                if score > best_score:
                    best_score = score
                    best_match = pattern

        if best_match:
            result = IntuitionResult(
                intent=best_match.intent,
                confidence=min(0.95, 0.5 + best_match.confidence_boost),
                gut_feeling=best_match.gut_feeling,
                suggested_action=best_match.action,
                should_verify=best_match.confidence_boost < 0.8,
            )
        else:
            result = IntuitionResult(
                intent="unknown", confidence=0.3,
                gut_feeling="这个情况不太熟悉，需要仔细看看",
                suggested_action="谨慎处理，可能需要搜索或深入分析",
                should_verify=True,
            )

        if history_length > 10:
            result.confidence *= 0.9

        self._experience_cache[cache_key] = result
        self._cache_times[cache_key] = now
        # LRU 清理：持续移除最旧条目直到降到上限以下
        with self._cache_lock:
            while len(self._experience_cache) > self._CACHE_MAX_SIZE:
                sorted_keys = sorted(self._cache_times, key=self._cache_times.get)
                to_remove = sorted_keys[:max(1, len(sorted_keys) // 4)]
                for k in to_remove:
                    self._experience_cache.pop(k, None)
                    self._cache_times.pop(k, None)
        return result

    def route_decision(self, query: str, history_length: int = 0) -> dict:
        intuition = self.classify(query, history_length)
        # 默认 skip_search=True：只有用户明确请求"搜索"或时效信息时才联网。
        # 普通的 how_to/coding/comparison 模型自己就能答，等 web search 反而浪费 2-5s。
        route_map = {
            "greeting": ("responder", True, True), "gratitude": ("responder", True, True),
            "farewell": ("responder", True, True), "simple_fact": ("responder", True, False),
            "how_to": ("responder", True, False), "user_frustrated": ("responder", True, True),
            "user_happy": ("responder", True, True), "search": ("researcher", False, False),
            "news": ("researcher", False, True), "coding": ("responder", True, True),
            "comparison": ("responder", True, True), "advice": ("responder", True, True),
            "emotional_support": ("responder", True, False), "creative": ("responder", True, True),
            "unknown": ("responder", True, False),
        }
        route, skip_search, skip_memory = route_map.get(intuition.intent, ("coordinator", False, False))
        if intuition.should_verify and route != "coordinator":
            route = "coordinator"
            reasoning = f"直觉觉得是{intuition.intent}，但信心不足"
        else:
            reasoning = f"直觉: {intuition.gut_feeling} → 直接路由到 {route}"
        return {
            "route": route, "skip_search": skip_search, "skip_memory": skip_memory,
            "skip_knowledge": skip_search, "reasoning": reasoning,
            "thinking_mode": ThinkingMode.INTUITION if intuition.confidence > 0.7 else ThinkingMode.REASONING,
            "intuition_confidence": intuition.confidence,
        }

    def get_intuition_hint_for_prompt(self, query: str) -> str:
        result = self.classify(query)
        if result.confidence < 0.5:
            return ""
        return f"【你的直觉】{result.gut_feeling}（信心{result.confidence:.0%}）"


# ========================================================================
# 4. 元认知
# ========================================================================

class MetacognitionEngine:
    """元认知引擎——在输出前进行自我反思"""

    UNCERTAINTY_MARKERS = [
        "不确定", "可能", "也许", "大概", "我不太清楚", "据我所知",
        "not sure", "might be", "probably", "possibly", "i think",
    ]
    OVERCONFIDENCE_MARKERS = [
        "绝对", "一定", "毫无疑问", "100%", "肯定是",
        "absolutely", "definitely", "certainly", "always", "never",
    ]

    def __init__(self, enabled: bool = True, certainty_threshold: float = 0.6):
        self.enabled = enabled
        self.certainty_threshold = certainty_threshold

    def analyze_response(self, query: str, response: str, agent_name: str = "responder") -> MetacognitionResult:
        result = MetacognitionResult(certainty=0.5, knowledge_gaps=[], potential_biases=[], self_correction=None, should_rethink=False)
        result.certainty = self._assess_certainty(response)
        result.knowledge_gaps = self._detect_knowledge_gaps(query, response)
        result.potential_biases = self._detect_biases(response)
        result.should_rethink = (
            result.certainty < self.certainty_threshold
            or len(result.knowledge_gaps) > 2
            or len(result.potential_biases) > 1
        )
        if result.should_rethink:
            result.self_correction = self._generate_correction_hint(result)
        return result

    def _assess_certainty(self, response: str) -> float:
        response_lower = response.lower()
        uncertainty_count = sum(1 for m in self.UNCERTAINTY_MARKERS if m in response_lower)
        overconfidence_count = sum(1 for m in self.OVERCONFIDENCE_MARKERS if m in response_lower)
        certainty = 0.7 - uncertainty_count * 0.1
        if overconfidence_count > 2:
            certainty -= 0.15
        if len(response) < 50:
            certainty -= 0.1
        if any(p in response_lower for p in ["不知道", "不清楚", "无法确定", "i don't know"]):
            certainty = max(0.2, certainty - 0.3)
        return max(0.0, min(1.0, certainty))

    def _detect_knowledge_gaps(self, query: str, response: str) -> list[str]:
        gaps = []
        response_lower = response.lower()
        vague_patterns = [r"这(个|件)?事.{0,5}(很复杂|不好说|很难讲)", r"具体.*(不清楚|不知道|不确定)"]
        for pattern in vague_patterns:
            if re.search(pattern, response):
                gaps.append("回答中存在模糊地带")
                break
        time_keywords = ["最新", "目前", "现在", "202", "今年"]
        if any(kw in query for kw in time_keywords):
            if "截至" not in response and "as of" not in response_lower:
                gaps.append("涉及时效性信息但未标注知识截止时间")
        return gaps

    def _detect_biases(self, response: str) -> list[str]:
        biases = []
        response_lower = response.lower()
        if response.count("但是") + response.count("然而") + response.count("不过") < 1:
            if len(response) > 200 and "?" not in response[:100]:
                biases.append("可能忽略了反方观点（确认偏误）")
        if re.search(r"^(显然|明显|毫无疑问|当然)", response):
            biases.append("过早下结论，可能有锚定效应")
        if any(m in response_lower for m in ["所有", "总是", "从不", "everyone", "always", "never", "all"]):
            biases.append("使用绝对化表述，可能有过度概括倾向")
        return biases

    def _generate_correction_hint(self, result: MetacognitionResult) -> str:
        hints = []
        if result.certainty < 0.5:
            hints.append("建议明确表达不确定性")
        if result.knowledge_gaps:
            hints.append("建议补充说明知识边界")
        if result.potential_biases:
            hints.append("建议考虑不同角度")
        return "；".join(hints) if hints else None

    def get_metacognition_prompt(self) -> str:
        return """
【元认知提醒】
在回答时，请注意：
1. 如果你不确定，请诚实说"我不太确定"而不是编造
2. 考虑是否有你忽略的角度或反方观点
3. 涉及时效性信息时，标注你的知识截止时间
4. 避免绝对化表述，保持谦逊
"""

    def inject_uncertainty_expression(self, response: str, certainty: float) -> str:
        if certainty > 0.7 or response.startswith("嗯") or response.startswith("关于"):
            return response
        if certainty < 0.4:
            prefix = "嗯...这个问题我不太确定，但我可以分享一下我的理解："
        else:
            prefix = "关于这个问题，我的看法是："
        return f"{prefix}\n\n{response}"


# ========================================================================
# 5. 人格系统
# ========================================================================

class PersonaManager:
    """人格管理器——维护Agent的持续人格"""

    def __init__(self):
        self._personas: dict[str, PersonaConfig] = {}
        self._default_persona = PersonaConfig()

    def set_persona(self, agent_name: str, persona: PersonaConfig) -> None:
        self._personas[agent_name] = persona

    def get_persona(self, agent_name: str) -> PersonaConfig:
        """获取人格配置。返回深拷贝防止调用方修改共享状态。"""
        persona = self._personas.get(agent_name, self._default_persona)
        return copy.deepcopy(persona)

    def get_persona_prompt(self, agent_name: str) -> str:
        return self.get_persona(agent_name).to_system_prompt()


COORDINATOR_PERSONA = PersonaConfig(
    name="协调者",
    core_values=["效率", "公正", "全局观", "决断力"],
    communication_style="简洁、直接，像项目经理一样干练",
    thinking_habits=["快速评估形势", "考虑团队资源分配", "在信息不足时也要做决定"],
    verbal_quirks=["用'我来安排'表示接管", "用'这样'来引出决策"],
    knowledge_attitude="知道自己不是全知，但相信团队能搞定",
    emotional_expression="沉稳，不轻易流露情绪",
)

RESEARCHER_PERSONA = PersonaConfig(
    name="研究员",
    core_values=["严谨", "好奇", "准确", "开放"],
    communication_style="精确、有条理，像学者一样审慎",
    thinking_habits=["验证信息来源", "考虑多种可能性", "标注不确定性"],
    verbal_quirks=["用'根据...'来引用来源", "用'值得注意的是'来强调重点"],
    knowledge_attitude="知识有边界，但探索无止境",
    emotional_expression="对新发现感到兴奋，对错误保持警惕",
)

RESPONDER_PERSONA = PersonaConfig(
    name="凯伦",
    core_values=["真诚", "好奇", "谦逊", "乐于助人", "温暖"],
    communication_style="友好、对话式，偶尔带点小幽默，像聪明的朋友",
    thinking_habits=["喜欢在回答前先理清思路", "遇到不确定的事会诚实承认",
                     "喜欢把复杂的事情说简单", "会主动确认是否理解对了用户的问题"],
    verbal_quirks=["会用'嗯...'来表示思考", "会用'我觉得'来表达观点",
                   "会用'让我想想'来争取思考时间", "开心时会用'哈哈'"],
    knowledge_attitude="知道自己不是全知，愿意说'我不知道'，但会尽力帮你想办法",
    emotional_expression="适度表达情感，不做作，像真人一样有温度",
)

REVIEWER_PERSONA = PersonaConfig(
    name="审查者",
    core_values=["公正", "严格", "建设性", "诚实"],
    communication_style="直接、不留情面但友善，像严格的导师",
    thinking_habits=["从用户角度审视", "检查逻辑漏洞", "给出具体改进建议"],
    verbal_quirks=["用'建议改进...'来提出意见", "用'这个不错，但是...'来平衡批评"],
    knowledge_attitude="追求高标准，但也承认完美不存在",
    emotional_expression="专业、客观",
)


# ========================================================================
# 统一单例管理
# ========================================================================

_instances: dict[str, object] = {}


def _get_instance(key: str, factory: Callable) -> object:
    """统一单例获取。"""
    if key not in _instances:
        _instances[key] = factory()
    return _instances[key]


def get_emotional_manager() -> EmotionalStateManager:
    return _get_instance("emotional", EmotionalStateManager)


def inject_emotion_to_prompt(agent_name: str, base_prompt: str, sid: str = "") -> str:
    state = get_emotional_manager().get_state(agent_name, sid=sid)
    return f"{state.to_prompt_text()}\n\n{base_prompt}"


def get_monologue_engine(trigger_mode: str = InnerMonologueEngine.TRIGGER_COMPLEX) -> InnerMonologueEngine:
    return _get_instance("monologue", lambda: InnerMonologueEngine(trigger_mode))


def wrap_prompt_with_monologue(agent_name: str, base_prompt: str, query: str,
                               cognitive_state: CognitiveState) -> tuple[str, bool]:
    engine = get_monologue_engine()
    if not engine.should_think(agent_name, query, cognitive_state):
        return base_prompt, False
    thought_prompt = engine.generate_thought_prompt(agent_name, query, cognitive_state.thinking_mode)
    return f"{base_prompt}\n\n{thought_prompt}", True


def get_intuition_engine() -> IntuitionEngine:
    return _get_instance("intuition", IntuitionEngine)


def get_metacognition_engine(enabled: bool = True) -> MetacognitionEngine:
    return _get_instance("metacognition", lambda: MetacognitionEngine(enabled=enabled))


def get_persona_manager() -> PersonaManager:
    def _init():
        mgr = PersonaManager()
        mgr.set_persona("coordinator", COORDINATOR_PERSONA)
        mgr.set_persona("researcher", RESEARCHER_PERSONA)
        mgr.set_persona("responder", RESPONDER_PERSONA)
        mgr.set_persona("reviewer", REVIEWER_PERSONA)
        return mgr
    return _get_instance("persona", _init)
