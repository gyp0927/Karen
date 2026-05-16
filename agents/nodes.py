"""Agent 节点函数 - Coordinator、Researcher、Responder、Reviewer、Planner。"""

import json
import logging
import re
import time
import asyncio
import threading
from dataclasses import asdict
from typing import Optional, Callable

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from core.cache import get_cache
from core.plugin_system import get_plugins_prompt
from agents.prompts import COORDINATOR_PROMPT, get_reviewer_prompt, build_responder_prompt, PLANNER_PROMPT
from agents.llm import get_llm, get_streaming_callback, get_llm_provider_model
from agents.search import run_parallel_search
from agents.tools import _need_tool_call
from state.stop_flag import is_stopped
from state.stats import record_call, estimate_cost, CallRecord

from cognition.human_mind import HumanMind
from cognition.types import CognitiveState
from cognition.utils import get_cognitive_state_from_dict, save_cognitive_state_to_dict
from cognition.tool_engine import run_tool_loop

logger = logging.getLogger(__name__)

# LLM 并发控制:同时最多 N 个流式请求,防止突发流量打满连接池或触发 API 限流。
# 值过大 → API 限流/连接池耗尽;值过小 → 高并发时排队。
# 当前连接池 max_connections=100,keepalive=20,设 8-12 比较安全。
# 按 event loop 隔离 Semaphore，避免 threading 模式下跨 loop 使用导致异常。
_LLM_SEM_LIMIT = 8
_llm_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_llm_sem() -> asyncio.Semaphore:
    """获取绑定到当前 event loop 的并发 Semaphore。"""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    sem = _llm_semaphores.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(_LLM_SEM_LIMIT)
        _llm_semaphores[loop_id] = sem
        # 防止 dict 无限增长：loop 被 GC 后 id 可能重用，数量通常很少，
        # 极端场景下（线程池频繁回收）做上限兜底
        if len(_llm_semaphores) > 64:
            _llm_semaphores.clear()
            _llm_semaphores[loop_id] = sem
    return sem


# 单条 LLM 流式响应超时（秒）。超时后返回已收到的 partial response。
# 180s 覆盖 reasoning 模型(如 kimi-for-coding):reasoning 阶段 ~50s + content 阶段 ~30s,
# 还需留余量给慢一点的代码生成。普通模型远用不到这么久。
_LLM_STREAM_TIMEOUT = 180.0


# 常见问候语快速回复模板
_GREETING_PATTERNS: dict[str, str] = {
    # 中文
    "你好": "你好！很高兴见到你，有什么我可以帮你的吗？",
    "在吗": "在的！有什么可以帮你的吗？",
    "在么": "在的！有什么可以帮你的吗？",
    "您好": "您好！很高兴为您服务。",
    "嗨": "嗨！有什么想聊的吗？",
    "哈喽": "哈喽！很高兴见到你。",
    "早上好": "早上好！愿你今天有个好心情。",
    "下午好": "下午好！有什么我可以帮你的吗？",
    "晚上好": "晚上好！今天过得怎么样？",
    "谢谢": "不客气！随时为你效劳。",
    "再见": "再见！有需要随时找我。",
    "拜拜": "拜拜！祝你有美好的一天。",
    # 英文
    "hello": "Hello! Nice to meet you. How can I help?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! What's up?",
    "thanks": "You're welcome! Happy to help anytime.",
    "thank you": "You're welcome! Let me know if you need anything else.",
    "good morning": "Good morning! Hope you have a great day ahead.",
    "good afternoon": "Good afternoon! How can I help you?",
    "good evening": "Good evening! How was your day?",
    "bye": "Bye! Feel free to come back anytime.",
    "goodbye": "Goodbye! Take care.",
}


_GLOWING_GREETINGS: set[str] = set(_GREETING_PATTERNS.keys())
# 按长度降序的 greeting 列表，前缀匹配时优先长串（"good morning" 先于 "good"）
_GREETINGS_BY_LEN: list[str] = sorted(_GREETING_PATTERNS.keys(), key=len, reverse=True)
# 问候后可能跟着的语气词/填充字符（中文 + 标点 + 空格）
_GREETING_FILLERS = "啊呀呢嘛哦哇吧嗯呦唉哈喔嘞欸 ~～!?！？.。,，"


def _quick_greeting(text: str) -> str | None:
    """检测简单问候语，返回模板回复或 None（非问候走 LLM）。"""
    if not text:
        return None
    t = text.strip().lower().rstrip("!?！？.。")
    # 精确匹配
    if t in _GLOWING_GREETINGS:
        return _GREETING_PATTERNS[t]
    # 去掉标点后再试一次（如"你好！""hello~"）
    t2 = t.rstrip("~～ ")
    if t2 in _GLOWING_GREETINGS:
        return _GREETING_PATTERNS[t2]
    # 前缀匹配：剩余部分仅为语气词/填充才视为同一问候
    # 避免把"你好我想问个问题"误判成"你好"
    for key in _GREETINGS_BY_LEN:
        if t.startswith(key):
            tail = t[len(key):].strip(_GREETING_FILLERS).strip()
            if not tail:
                return _GREETING_PATTERNS[key]
    return None


_REPLAY_CHUNK_SIZE = 50


def _replay_stream(stream_cb: Optional[Callable[[str], None]], text: str, sid: str = "") -> None:
    """把缓存/模板回复按 chunk 喂给前端回调,模拟 LLM 流式效果。"""
    if not stream_cb or not text:
        return
    for i in range(0, len(text), _REPLAY_CHUNK_SIZE):
        if sid and is_stopped(sid):
            break
        stream_cb(text[i:i + _REPLAY_CHUNK_SIZE])


def _spawn_bg(fn: Callable, *args, **kwargs) -> None:
    """把一个同步函数扔后台 daemon 线程跑,不阻塞事件循环。

    用于 stats.db / cache.db 等同步 SQLite 写——之前用 asyncio.to_thread + create_task
    会占用 LangGraph 默认线程池并多绕一层 asyncio Task。daemon 线程更轻量,
    主进程退出即结束。
    """
    try:
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
    except Exception as e:
        logger.debug(f"bg thread spawn failed: {e}")


def _estimate_tokens(text: str) -> int:
    """粗估 token 数。中文字符 ≈1 token/字,英文 ≈1 token/4 字符。"""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    non_chinese = len(text) - chinese_chars
    return max(1, chinese_chars + non_chinese // 4)


def _normalize_message_order(messages: list) -> list:
    """把相邻具名 SystemMessage 按 name 排序,保证多个并行 searcher 结果顺序确定。

    LangGraph 的 add reducer 按 task 完成时序拼 messages,fast_graph 三个 searcher
    并行 Send 时位置不固定。这里在传给 LLM 之前 stable sort,不破坏
    Human/AI 历史交错(只动连续的具名 SystemMessage 段)。
    """
    out = []
    buffer = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and getattr(msg, "name", None):
            buffer.append(msg)
        else:
            if buffer:
                buffer.sort(key=lambda m: m.name or "")
                out.extend(buffer)
                buffer = []
            out.append(msg)
    if buffer:
        buffer.sort(key=lambda m: m.name or "")
        out.extend(buffer)
    return out


def _reorder_system_first(messages: list) -> list:
    """把所有 SystemMessage 移到非 SystemMessage 之前。

    V3.2-Pro 等模型在 SystemMessage 出现在 HumanMessage 之后时，
    可能无法正确理解上下文，导致输出过长或为空。将 SystemMessage
    统一放在前面可避免此问题。
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    return system_msgs + other_msgs


def _record_llm_call(
    agent_name: str,
    sid: str | None,
    messages: list,
    response: str,
    duration_ms: int,
    status: str = "success",
) -> None:
    """把一次 LLM 调用打到 stats.db。失败仅警告,不影响主流程。"""
    try:
        provider, model = get_llm_provider_model(sid or "")
        prompt_tokens = sum(_estimate_tokens(getattr(m, "content", "") or "") for m in messages)
        completion_tokens = _estimate_tokens(response)
        total = prompt_tokens + completion_tokens
        cost = estimate_cost(provider, prompt_tokens, completion_tokens)
        record_call(CallRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            agent_name=agent_name,
            session_id=sid or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            duration_ms=duration_ms,
            estimated_cost_usd=cost,
            status=status,
        ))
    except Exception as e:
        logger.debug(f"stats record failed: {e}")


def _build_result_dict(
    response: str,
    agent_name: str,
    cognitive_state: CognitiveState | None = None,
) -> dict:
    """构建标准返回字典，自动包含认知状态（如果提供）。"""
    result = {"messages": [AIMessage(content=response, name=agent_name)]}
    if cognitive_state is not None:
        result["cognitive_state"] = asdict(cognitive_state)
    return result


def _check_greeting_cache(query: str, agent_name: str, tools: Optional[list[BaseTool]], sid: str | None, on_token: Optional[Callable[[str], None]]) -> dict | None:
    """检查问候语快速路径。命中时返回 result_dict，否则返回 None。"""
    if agent_name != "responder" or tools:
        return None
    greeting_reply = _quick_greeting(query)
    if not greeting_reply:
        return None
    logger.info(f"[{agent_name}] Quick greeting shortcut")
    stream_cb = on_token if on_token else get_streaming_callback(sid)
    _replay_stream(stream_cb, greeting_reply, sid or "")
    return _build_result_dict(greeting_reply, agent_name, None)


def _build_cache_messages(messages: list, system_prompt: str, agent_name: str) -> list:
    """构建用于缓存键的消息列表（归一化动态内容）。"""
    if agent_name != "responder":
        return messages
    cache_messages = []
    swapped_system = False
    for m in messages:
        if isinstance(m, AIMessage) and "[route:" in (m.content or ""):
            continue
        if isinstance(m, SystemMessage) and getattr(m, "name", None) in ("search_result", "researcher"):
            continue
        if isinstance(m, SystemMessage) and not getattr(m, "name", None) and not swapped_system:
            cache_messages.append(SystemMessage(content=system_prompt))
            swapped_system = True
        else:
            cache_messages.append(m)
    return cache_messages


def _try_read_cache(cache, cache_messages: list, cache_key: tuple, agent_name: str, sid: str | None, on_token: Optional[Callable[[str], None]]) -> dict | None:
    """尝试从缓存读取。命中时返回 result_dict，否则返回 None。"""
    if not cache:
        return None
    try:
        cached = cache.get(cache_messages, cache_key[0], cache_key[1])
        if cached is not None and cached.strip():
            logger.info(f"[{agent_name}] Cache hit")
            if agent_name == "coordinator":
                return {"messages": [AIMessage(content=cached, name=agent_name)]}
            stream_cb = on_token if on_token else get_streaming_callback(sid)
            _replay_stream(stream_cb, cached, sid or "")
            return {"messages": [AIMessage(content=cached, name=agent_name)]}
    except (OSError, ValueError) as e:
        logger.warning(f"Cache lookup failed: {e}")
    return None


def _bind_llm_params(llm, agent_name: str, provider: str):
    """根据 agent 类型绑定 LLM 参数。"""
    if agent_name == "coordinator":
        return llm.bind(max_tokens=60, temperature=0.1)
    if agent_name == "responder":
        # kimi-code 是 reasoning 模型,reasoning 阶段会消耗 1000+ token,
        # 若 max_tokens 太小,reasoning 没结束就停了,content 一个字都输出不了。
        if provider == "kimi-code":
            return llm.bind(temperature=0.3)
        return llm.bind(max_tokens=180, temperature=0.3)
    return llm


async def _invoke_llm_stream(llm, messages: list, stream_cb: Optional[Callable[[str], None]], sid: str | None, agent_name: str) -> str:
    """执行流式 LLM 调用，返回响应文本。处理超时和连接中断。"""
    response = ""

    async def _stream():
        nonlocal response
        async for chunk in llm.astream(messages):
            if is_stopped(sid):
                break
            if chunk.content:
                response += chunk.content
                if stream_cb:
                    stream_cb(chunk.content)

    try:
        await asyncio.wait_for(_stream(), timeout=_LLM_STREAM_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"[{agent_name}] LLM streaming timed out after {_LLM_STREAM_TIMEOUT}s, returning partial response ({len(response)} chars)")
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)
        if (error_name == "RemoteProtocolError"
                or "peer closed connection" in error_msg
                or "incomplete chunked read" in error_msg):
            if is_stopped(sid):
                logger.info(f"[{agent_name}] Streaming interrupted but stop requested, returning partial response")
            else:
                logger.warning(f"[{agent_name}] Streaming interrupted, retrying once: {e}")
                if response:
                    logger.info(f"[{agent_name}] Returning partial response ({len(response)} chars)")
                else:
                    try:
                        await asyncio.wait_for(_stream(), timeout=_LLM_STREAM_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning(f"[{agent_name}] LLM retry timed out after {_LLM_STREAM_TIMEOUT}s")
        else:
            raise
    return response


def _write_cache_async(cache, cache_messages: list, cache_key: tuple, response: str, sid: str | None) -> None:
    """后台异步写入缓存（fire-and-forget）。"""
    if not cache or not response or is_stopped(sid):
        return
    def _cache_write():
        try:
            cache.set(cache_messages, cache_key[0], cache_key[1], response)
        except (OSError, ValueError) as e:
            logger.warning(f"Cache write failed: {e}")
    _spawn_bg(_cache_write)


def _build_empty_fallback(agent_name: str, cognitive_state: CognitiveState | None, enable_cognition: bool) -> dict:
    """构建空响应的 fallback 结果。"""
    logger.warning(f"[{agent_name}] LLM returned empty response. Returning fallback message.")
    fallback = "抱歉,我现在没法回答这个问题。可能是模型暂时无法处理,请稍后再试或换个问法。"
    return _build_result_dict(fallback, agent_name, cognitive_state if enable_cognition else None)


async def _run_agent(
    state: dict,
    system_prompt: str,
    agent_name: str,
    sid: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
    enable_cognition: bool = True,
    enable_monologue: bool = True,
    tools: Optional[list[BaseTool]] = None,
) -> dict:
    """通用 Agent 执行函数（接入认知系统 + 工具调用）。"""
    if sid is None:
        sid = state.get("task_context", {}).get("sid", "")

    cognitive_state = get_cognitive_state_from_dict(state)
    is_responder = agent_name == "responder"
    mind = HumanMind(
        enable_monologue=enable_monologue and is_responder,
        enable_emotion=is_responder,
        enable_intuition=enable_cognition,
        enable_metacognition=is_responder and enable_cognition,
        enable_persona=is_responder,
    ) if enable_cognition else None
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    # ---- Phase 1: 问候语快速路径 ----
    greeting_result = _check_greeting_cache(query, agent_name, tools, sid, on_token)
    if greeting_result:
        return greeting_result

    # ---- Phase 2: 认知增强 prompt ----
    enhanced_prompt = system_prompt
    had_monologue = False
    if mind and enable_cognition:
        enhanced_prompt, had_monologue = mind.enhance_prompt(
            agent_name, system_prompt, query, cognitive_state, sid=sid or "",
        )

    # ---- Phase 3: 构建消息列表 ----
    messages = [SystemMessage(content=enhanced_prompt)] + _reorder_system_first(
        _normalize_message_order(list(state["messages"]))
    )

    # ---- Phase 4: 缓存配置 ----
    cache_enabled = agent_name in ("responder", "coordinator")
    cache = get_cache() if cache_enabled else None
    provider, model = get_llm_provider_model(sid or "")
    tools_hash = hash(tuple(sorted(t.name for t in tools))) if tools else "none"
    model_key = f"{model}:{agent_name}:{enable_cognition}:{enable_monologue}:{tools_hash}"
    cache_key = (provider, model_key)
    cache_messages = _build_cache_messages(messages, system_prompt, agent_name)

    # ---- Phase 5: 缓存查找 ----
    cached_result = _try_read_cache(cache, cache_messages, cache_key, agent_name, sid, on_token)
    if cached_result:
        return cached_result

    # ---- Phase 6: LLM 调用 ----
    async with _get_llm_sem():
        llm = get_llm(sid or "")
        llm = _bind_llm_params(llm, agent_name, provider)
        _llm_t0 = time.time()

        try:
            if tools:
                stream_cb = on_token if on_token else get_streaming_callback(sid)
                response = await run_tool_loop(llm, messages, tools, max_iterations=3, sid=sid or "", on_token=stream_cb)
            else:
                stream_cb = on_token if on_token else (get_streaming_callback(sid) if agent_name == "responder" else None)
                response = await _invoke_llm_stream(llm, messages, stream_cb, sid, agent_name)
        except Exception:
            _spawn_bg(
                _record_llm_call, agent_name, sid, messages, "",
                int((time.time() - _llm_t0) * 1000), "error",
            )
            raise

    # ---- Phase 7: 统计记录 ----
    _llm_duration_ms = int((time.time() - _llm_t0) * 1000)
    _llm_status = "stopped" if is_stopped(sid) else "success"
    _spawn_bg(
        _record_llm_call, agent_name, sid, messages, response, _llm_duration_ms, _llm_status,
    )

    # ---- Phase 8: 缓存写入 ----
    _write_cache_async(cache, cache_messages, cache_key, response, sid)

    # ---- Phase 9: 认知处理 ----
    if mind and enable_cognition:
        response = mind.process_response(
            agent_name, query, response, cognitive_state, had_monologue, sid=sid or "",
        )
        save_cognitive_state_to_dict(state, cognitive_state)

    # ---- Phase 10: 空响应 fallback ----
    if not response or not response.strip():
        return _build_empty_fallback(agent_name, cognitive_state, enable_cognition)

    return _build_result_dict(response, agent_name, cognitive_state if enable_cognition else None)


async def coordinator_node(state: dict, sid: str | None = None) -> dict:
    """协调者Agent - 分析需求并决定路由。"""
    return await _run_agent(state, COORDINATOR_PROMPT, "coordinator", sid, enable_cognition=True)


async def researcher_node(state: dict, sid: str | None = None) -> dict:
    """搜索聚合节点 - 并行执行 web/memory/knowledge 搜索并把结果注入上下文。

    注意：此节点不调用 LLM，也不走 _run_agent。它只是 run_parallel_search
    的薄包装，把搜索文本以 SystemMessage 的形式塞进 state，供下游 Responder 使用。
    """
    search_context = await run_parallel_search(state)
    if search_context:
        return {"messages": [SystemMessage(
            content=f"【搜索结果】\n\n{search_context}\n\n请基于以上搜索结果生成最终回答。",
            name="researcher",
        )]}
    return {"messages": []}


async def responder_node(state: dict, sid: str | None = None) -> dict:
    """响应者Agent - 生成最终回答。集成异步搜索和工具调用。

    搜索在 responder 内部异步处理：启动后最多等待 1.2 秒，
    超时则基于已有知识直接生成，避免用户长时间等待。
    """
    plugin_prompt = get_plugins_prompt()
    from core.i18n import LANG_INSTRUCTIONS
    detected_lang = state.get("task_context", {}).get("detected_language", "zh")
    lang_instr = LANG_INSTRUCTIONS.get(detected_lang, "")
    responder_prompt = build_responder_prompt(plugin_prompt, lang_instr)

    # ---- 意图判断：每次都重新分类，避免跨轮次缓存导致错误意图复用 ----
    from langchain_core.messages import HumanMessage
    from core.intent import classify_intent_sync
    # 协调模式下 state["messages"][-1] 可能是 AIMessage，需向前找到最后一个 HumanMessage
    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    history = state.get("messages", [])
    result = classify_intent_sync(query, history=history)
    intent = {
        "intent": result.intent,
        "confidence": result.confidence,
        "skip_search": result.skip_search,
        "skip_memory": result.skip_memory,
        "skip_knowledge": result.skip_knowledge,
        "source": result.source,
    }
    state = {k: (list(v) if k == "messages" else v) for k, v in state.items()}
    task_ctx = dict(state.get("task_context", {}))
    task_ctx["intent_result"] = intent
    state["task_context"] = task_ctx

    is_simple = intent.get("skip_search") and intent.get("skip_memory")
    needs_search = not intent.get("skip_search", True)
    needs_memory = not intent.get("skip_memory", True)

    # ---- 保存原始用户 query（搜索注入后 messages[-1] 会变成 SystemMessage）----
    # query 已在上面意图分类时找到最后一个 HumanMessage，直接复用
    original_query = query

    # ---- 异步搜索：启动搜索，最多等待 1.2 秒 ----
    if needs_search or needs_memory:
        try:
            search_context = await asyncio.wait_for(
                run_parallel_search(state),
                timeout=1.2,
            )
            if search_context:
                search_msg = SystemMessage(
                    content=(
                        f"{search_context}\n\n"
                        "请基于以上搜索结果生成最终回答，"
                        "涉及实时信息、最新数据、价格、天气等时效内容时，"
                        "优先以此处的搜索结果为准，而非你的训练知识。"
                    ),
                    name="search_result",
                )
                state = dict(state)
                state["messages"] = list(state.get("messages", [])) + [search_msg]
        except asyncio.TimeoutError:
            logger.info("[responder] Search timeout (1.2s), proceeding with knowledge")
        except Exception as e:
            logger.warning(f"[responder] Search failed: {e}")

    # ---- 工具调用判断（用原始 query，不能用注入搜索后的 messages[-1]）----
    tools = None
    if _need_tool_call(original_query):
        from cognition.tool_engine import execute_python
        tools = [execute_python]

    # Fast mode：简单查询完全禁用 cognition 开销；复杂查询启用 persona+emotion+intuition+metacognition
    use_cognition = not is_simple
    return await _run_agent(
        state, responder_prompt, "responder", sid,
        enable_cognition=use_cognition,
        enable_monologue=False,
        tools=tools,
    )


async def reviewer_node(state: dict, language: str = "zh", sid: str | None = None) -> dict:
    """检查者Agent - 审查回答质量。"""
    reviewer_prompt = get_reviewer_prompt(language)
    return await _run_agent(state, reviewer_prompt, "reviewer", sid, enable_cognition=True)


async def planner_node(state: dict, sid: str | None = None) -> dict:
    """计划者Agent - 分析需求并生成结构化计划。"""
    return await _run_agent(state, PLANNER_PROMPT, "planner", sid)


def parse_plan_from_response(text: str) -> dict | None:
    """从 Agent 输出中提取 JSON 计划。"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\"title\"[\s\S]*\"steps\"[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def create_agents(language: str = "zh", fast_mode: bool = False):
    """创建所有 Agent 节点函数的便捷函数。"""
    async def _reviewer_node(state: dict, sid: str | None = None) -> dict:
        return await reviewer_node(state, language=language, sid=sid)

    return (coordinator_node, researcher_node, responder_node, _reviewer_node)
