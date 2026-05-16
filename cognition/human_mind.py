"""Human Mind —— 统一认知系统入口

将五种"人类思维"整合到一个统一的认知处理管道中：
1. 内心独白（Inner Monologue）
2. 情感状态（Emotional State）
3. 直觉引擎（Intuition Engine）
4. 元认知反射（Metacognition）
5. 人格系统（Persona）

使用方式：
    from cognition.human_mind import HumanMind
    mind = HumanMind()
    enhanced_prompt = mind.enhance_prompt("responder", base_prompt, query, state)
    response = mind.process_response("responder", query, raw_response, state)
"""
import logging
import os
from typing import Optional

from langchain_core.messages import AIMessage

from cognition.types import CognitiveState, ThinkingMode

logger = logging.getLogger(__name__)

# cognition.engine 内部依赖（如 anyio、ordered-set 等）可能缺失，
# 导入失败时降级为 pass-through，避免整个系统起不来。
_engine_available = True
try:
    from cognition.engine import (
        get_monologue_engine,
        wrap_prompt_with_monologue,
        get_emotional_manager,
        inject_emotion_to_prompt,
        get_intuition_engine,
        get_metacognition_engine,
        get_persona_manager,
    )
except Exception as _e:
    _engine_available = False
    logger.warning(f"Cognition engine import failed, falling back to pass-through: {_e}")

    def _noop(*args, **kwargs):
        pass

    get_monologue_engine = _noop
    wrap_prompt_with_monologue = lambda *a, **k: (a[1] if len(a) > 1 else "", False)
    get_emotional_manager = _noop
    inject_emotion_to_prompt = lambda *a, **k: a[1] if len(a) > 1 else ""
    get_intuition_engine = _noop
    get_metacognition_engine = _noop
    get_persona_manager = _noop


class HumanMind:
    """人类思维整合器——给Agent一个完整的'心智'"""

    def __init__(
        self,
        enable_monologue: bool = True,
        enable_emotion: bool = True,
        enable_intuition: bool = True,
        enable_metacognition: bool = True,
        enable_persona: bool = True,
    ):
        # 全局开关：通过环境变量禁用整个认知层
        _disabled = os.environ.get("DISABLE_COGNITION", "0").lower() in ("1", "true", "yes")
        self.enable_monologue = enable_monologue and not _disabled
        self.enable_emotion = enable_emotion and not _disabled
        self.enable_intuition = enable_intuition and not _disabled
        self.enable_metacognition = enable_metacognition and not _disabled
        self.enable_persona = enable_persona and not _disabled

        # 初始化各子系统
        self.monologue = get_monologue_engine()
        self.emotion = get_emotional_manager()
        self.intuition = get_intuition_engine()
        self.metacognition = get_metacognition_engine()
        self.persona = get_persona_manager()

    def enhance_prompt(
        self,
        agent_name: str,
        base_prompt: str,
        query: str,
        cognitive_state: Optional[CognitiveState] = None,
        sid: str = "",
    ) -> tuple[str, bool]:
        """增强系统提示词，注入人类思维元素

        Args:
            sid: socket/会话 ID,用于隔离多用户的情感状态。留空退化为单用户。

        Returns:
            (增强后的提示词, 是否启用了内心独白格式)
        """
        if cognitive_state is None:
            cognitive_state = CognitiveState()

        enhanced = base_prompt
        has_monologue = False

        # 1. 注入人格（最底层，定义"我是谁"）
        if self.enable_persona:
            persona_prompt = self.persona.get_persona_prompt(agent_name)
            # 将人格提示词插入到开头（替换掉基础提示词中的身份定义）
            enhanced = self._merge_persona_into_prompt(enhanced, persona_prompt)

        # 2. 注入情感状态(按 sid 隔离)
        if self.enable_emotion:
            enhanced = inject_emotion_to_prompt(agent_name, enhanced, sid=sid)

        # 3. 注入直觉提示
        if self.enable_intuition:
            intuition_hint = self.intuition.get_intuition_hint_for_prompt(query)
            if intuition_hint:
                enhanced = f"{intuition_hint}\n\n{enhanced}"

        # 4. 注入近期内心独白
        if self.enable_monologue:
            thoughts_text = cognitive_state.thoughts_to_prompt(agent_name)
            if thoughts_text:
                enhanced = f"{thoughts_text}\n\n{enhanced}"

        # 5. 注入元认知提醒
        if self.enable_metacognition:
            meta_prompt = self.metacognition.get_metacognition_prompt()
            enhanced = f"{enhanced}\n\n{meta_prompt}"

        # 6. 包装内心独白触发（最后一步，控制输出格式）
        if self.enable_monologue:
            enhanced, has_monologue = wrap_prompt_with_monologue(
                agent_name, enhanced, query, cognitive_state
            )

        return enhanced, has_monologue

    def process_response(
        self,
        agent_name: str,
        query: str,
        raw_response: str,
        cognitive_state: Optional[CognitiveState] = None,
        had_monologue: bool = False,
        sid: str = "",
    ) -> str:
        """处理原始响应，应用人类思维后处理

        Args:
            sid: socket/会话 ID,情感状态按 (sid, agent_name) 隔离。

        - 提取内心独白
        - 元认知分析
        - 情感状态更新
        - 不确定性表达注入
        """
        if cognitive_state is None:
            cognitive_state = CognitiveState()

        response = raw_response
        meta_result = None  # 元认知未启用 / 非 responder 时保持 None,后续 success 计算用 None 兜底

        # 1. 提取内心独白（如果启用了的话）
        if had_monologue and self.enable_monologue:
            thought, response = self.monologue.extract_thought_and_answer(raw_response)
            if thought:
                self.monologue.record_thought(
                    agent_name, thought, cognitive_state,
                    mode=cognitive_state.thinking_mode,
                )
                logger.info(f"[{agent_name}] 💭 {thought[:100]}...")

        # 2. 元认知分析
        if self.enable_metacognition and agent_name == "responder":
            meta_result = self.metacognition.analyze_response(query, response, agent_name)
            cognitive_state.last_metacognition = meta_result

            # 如果元认知认为需要重新思考，且确定性很低，注入不确定性表达
            if meta_result.certainty < 0.5:
                response = self.metacognition.inject_uncertainty_expression(
                    response, meta_result.certainty
                )

            # 记录元认知结果到内心独白
            if self.enable_monologue:
                meta_thought = (
                    f"元认知自检：把握度{meta_result.certainty:.0%}"
                )
                if meta_result.knowledge_gaps:
                    meta_thought += f"，盲区：{', '.join(meta_result.knowledge_gaps[:2])}"
                if meta_result.should_rethink:
                    meta_thought += "，觉得需要再想想"
                self.monologue.record_thought(
                    agent_name, meta_thought, cognitive_state,
                    mode=ThinkingMode.REFLECTIVE,
                )

        # 3. 更新情感状态(按 sid 隔离)
        if self.enable_emotion:
            success = not meta_result.should_rethink if meta_result is not None else True
            complexity = len(response) / 1000  # 简单用长度估算
            self.emotion.update_after_interaction(
                agent_name, success=success, complexity=min(1.0, complexity), sid=sid,
            )

        # 4. 增加回合计数
        cognitive_state.turn_count += 1

        return response

    def get_intuition_route(self, query: str, history_length: int = 0) -> dict:
        """获取直觉路由决策"""
        if not self.enable_intuition:
            return {
                "route": "coordinator",
                "skip_search": False,
                "skip_memory": False,
                "skip_knowledge": False,
                "reasoning": "直觉引擎已关闭",
                "thinking_mode": ThinkingMode.REASONING,
                "intuition_confidence": 0.0,
            }
        return self.intuition.route_decision(query, history_length)

    def _merge_persona_into_prompt(self, base_prompt: str, persona_prompt: str) -> str:
        """将人格提示词合并到基础提示词中

        策略：保留基础提示词的任务指令，但替换身份定义部分
        """
        # 简单策略：在基础提示词前面插入人格定义
        # 更复杂的策略可以解析基础提示词并智能合并
        return f"{persona_prompt}\n\n---\n\n{base_prompt}"


# 便捷函数：快速增强单个agent的prompt
def enhance_agent_prompt(
    agent_name: str,
    base_prompt: str,
    query: str = "",
    cognitive_state: Optional[CognitiveState] = None,
    sid: str = "",
) -> tuple[str, bool]:
    """便捷函数：增强agent提示词"""
    mind = HumanMind()
    return mind.enhance_prompt(agent_name, base_prompt, query, cognitive_state, sid=sid)


def process_agent_response(
    agent_name: str,
    query: str,
    raw_response: str,
    cognitive_state: Optional[CognitiveState] = None,
    had_monologue: bool = False,
    sid: str = "",
) -> str:
    """便捷函数：处理agent响应"""
    mind = HumanMind()
    return mind.process_response(agent_name, query, raw_response, cognitive_state, had_monologue, sid=sid)
