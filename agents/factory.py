import asyncio
import json
import logging
import re
import threading
from typing import Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from core.config import get_api_key, get_base_url, get_model_name, get_provider
from core.plugin_system import get_plugins_prompt
from core.cache import get_cache
from prompts.coordinator_prompt import COORDINATOR_PROMPT
from prompts.reviewer_prompt import get_reviewer_prompt
from state.stop_flag import is_stopped

# 认知系统导入
from cognition.human_mind import HumanMind
from cognition.types import CognitiveState, ThinkingMode
from cognition.utils import get_cognitive_state_from_dict, save_cognitive_state_to_dict
from cognition.tool_engine import run_tool_loop, get_tools_for_mode

# 全局共享的 HTTP 客户端（连接池复用）
_httpx_client = None


def _get_http_client():
    """获取全局共享的 httpx.Client，启用连接池复用。

    LangChain ChatOpenAI 的 http_client 参数需要同步的 httpx.Client，
    而非 AsyncClient。"""
    global _httpx_client
    if _httpx_client is None:
        try:
            import httpx
            _httpx_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(60.0, connect=5.0),
            )
            logger.info("HTTP client pool initialized (keepalive=30s, max=100)")
        except ImportError:
            logger.warning("httpx not installed, falling back to default HTTP client")
            return None
    return _httpx_client

logger = logging.getLogger(__name__)

# 按 sid 隔离的 LLM 配置和流式回调
# 使用 dict + lock 替代 contextvars，确保在线程池中也能正确传递
_llm_configs: dict[str, dict | None] = {}
_token_callbacks: dict[str, Callable[[str], None] | None] = {}
_callbacks_lock = threading.Lock()


def set_current_llm_config(config: dict | None, sid: str = ""):
    """设置指定 sid 的 LLM 配置"""
    _llm_configs[sid] = config


def set_streaming_callback(callback: Optional[Callable[[str], None]], sid: str = ""):
    """设置指定 sid 的流式输出 token 回调函数"""
    with _callbacks_lock:
        _token_callbacks[sid] = callback


def get_streaming_callback(sid: str = "") -> Optional[Callable[[str], None]]:
    """获取指定 sid 的流式输出回调"""
    with _callbacks_lock:
        return _token_callbacks.get(sid)


def clear_streaming_callback(sid: str = ""):
    """清除指定 sid 的流式输出回调"""
    with _callbacks_lock:
        _token_callbacks.pop(sid, None)


def _build_llm_kwargs(sid: str = "") -> dict:
    """构建 LLM 初始化参数"""
    from core.config import PROVIDER_CONFIG
    cfg = _llm_configs.get(sid)
    if cfg:
        provider = cfg.get("provider", "ollama")
        base_url = cfg.get("baseUrl", "")
        # 如果 base_url 为空，使用提供商默认值
        if not base_url and provider in PROVIDER_CONFIG:
            base_url = PROVIDER_CONFIG[provider]["base_url"]
        api_key = cfg.get("apiKey", "")
        if not api_key and provider == "ollama":
            api_key = "ollama"
        kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "model": cfg.get("model", ""),
            "temperature": 0.7,
        }
    else:
        provider = get_provider()
        kwargs = {
            "api_key": get_api_key(),
            "base_url": get_base_url(),
            "model": get_model_name(),
            "temperature": 0.7,
        }
    if provider == "kimi-code":
        kwargs["default_headers"] = {
            "User-Agent": "claude-code/1.0",
            "X-Stainless-Lang": "python",
            "X-Stainless-Package-Version": "1.0.0",
        }
    return kwargs


# 自定义缓存字典，避免 lru_cache 处理复杂参数的序列化问题
_llm_cache: dict[str, ChatOpenAI] = {}
_llm_cache_lock = threading.Lock()


def _make_cache_key(kwargs: dict) -> str:
    """构建基于 JSON 的缓存键，支持字典等不可哈希类型。"""
    # 排序键保证一致性
    return json.dumps(kwargs, sort_keys=True)


def get_llm(sid: str = "") -> ChatOpenAI:
    """获取 LLM 实例。优先使用 sid 对应的动态配置，否则回退到全局配置。

    使用自定义字典缓存实例，避免每次请求重复创建。
    单锁保护整个 get-or-create 流程，避免竞态条件。
    """
    kwargs = _build_llm_kwargs(sid)
    cache_key = _make_cache_key(kwargs)

    with _llm_cache_lock:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

        logger.debug(f"Creating new LLM instance for model={kwargs.get('model')}")
        # 启用 HTTP 连接池复用，减少每次请求建立 TCP 连接的开销
        http_client = _get_http_client()
        if http_client:
            kwargs["http_client"] = http_client
        instance = ChatOpenAI(**kwargs)
        _llm_cache[cache_key] = instance
        return instance


def clear_llm_cache():
    """清除 LLM 实例缓存。在切换配置/模式后调用。"""
    with _llm_cache_lock:
        _llm_cache.clear()
    logger.info("LLM cache cleared")


def _get_cache_provider_model() -> tuple[str, str]:
    """获取当前 provider 和 model 名称（用于缓存 key）。

    消除缓存读写中重复的 get_provider() / get_model_name() 调用。
    """
    return get_provider(), get_model_name()


def _build_result_dict(
    response: str,
    agent_name: str,
    cognitive_state: CognitiveState | None = None,
) -> dict:
    """构建标准返回字典，自动包含认知状态（如果提供）。"""
    result = {"messages": [AIMessage(content=response, name=agent_name)]}
    if cognitive_state is not None:
        from dataclasses import asdict
        result["cognitive_state"] = asdict(cognitive_state)
    return result


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
    """通用 Agent 执行函数（增强版，接入认知系统 + 工具调用）。

    参数:
        state: 当前状态字典
        system_prompt: 系统提示词
        agent_name: Agent 名称（用于日志和消息标记）
        sid: Socket ID（用于隔离停止标志）
        on_token: 每收到一个 token chunk 时调用的回调函数(token_text: str)
        enable_cognition: 是否启用认知系统
        enable_monologue: 是否启用内心独白（coordinator等短输出agent可关闭）
        tools: 可用工具列表（responder 使用）
    """
    # 认知系统初始化
    cognitive_state = get_cognitive_state_from_dict(state)
    # Coordinator/Reviewer 是纯功能节点，禁用会干扰输出格式的人格/独白
    # Responder 是唯一需要完整认知系统的节点
    is_responder = agent_name == "responder"
    mind = HumanMind(
        enable_monologue=enable_monologue and is_responder,
        enable_emotion=is_responder,           # 只有 responder 需要情感表达
        enable_intuition=True,                 # 所有节点共享直觉上下文
        enable_metacognition=is_responder,     # 只有 responder 需要元认知
        enable_persona=is_responder,           # 只有 responder 需要人格化
    ) if enable_cognition else None
    query = state["messages"][-1].content if state["messages"] else ""

    # 用认知系统增强提示词
    enhanced_prompt = system_prompt
    had_monologue = False
    if mind and enable_cognition:
        enhanced_prompt, had_monologue = mind.enhance_prompt(
            agent_name, system_prompt, query, cognitive_state
        )
        logger.debug(f"[{agent_name}] 认知增强完成，内心独白={had_monologue}")

    llm = get_llm(sid or "")
    messages = [SystemMessage(content=enhanced_prompt)] + list(state["messages"])

    logger.debug(f"[{agent_name}] Starting generation, messages_count={len(messages)}")

    # 对 responder 和 coordinator 节点启用缓存
    cache_enabled = agent_name in ("responder", "coordinator")
    cache = get_cache() if cache_enabled else None
    cache_key = _get_cache_provider_model()

    if cache and cache_enabled:
        try:
            cached_response = cache.get(messages, cache_key[0], cache_key[1])
            if cached_response is not None:
                logger.info(f"[{agent_name}] Cache hit, skipping generation")
                if agent_name == "coordinator":
                    return {"messages": [AIMessage(content=cached_response, name=agent_name)]}
                stream_cb = on_token if on_token else get_streaming_callback(sid)
                if stream_cb:
                    chunk_size = 20
                    for i in range(0, len(cached_response), chunk_size):
                        stream_cb(cached_response[i:i+chunk_size])
                return {"messages": [AIMessage(content=cached_response, name=agent_name)]}
        except (OSError, ValueError) as e:
            logger.warning(f"Cache lookup failed: {e}, falling back to generation")

    # 限制 Coordinator 输出长度：只需要路由标记，最多 80 tokens
    if agent_name == "coordinator":
        llm = llm.bind(max_tokens=80)

    # 工具调用分支：如果提供了 tools，走 LLM 自主工具调用循环
    if tools:
        stream_cb = on_token if on_token else get_streaming_callback(sid)
        logger.info(f"[{agent_name}] 启用工具调用模式，工具数={len(tools)}")
        response = await run_tool_loop(
            llm, messages, tools,
            max_iterations=3,
            sid=sid or "",
            on_token=stream_cb,
        )
    else:
        # 普通流式生成（原有逻辑）
        response = ""
        stream_cb = on_token if on_token else (get_streaming_callback(sid) if agent_name == "responder" else None)
        async for chunk in llm.astream(messages):
            if is_stopped(sid):
                logger.info(f"[{agent_name}] Generation stopped by user")
                break
            if chunk.content:
                response += chunk.content
                if stream_cb:
                    stream_cb(chunk.content)

    # 写入缓存
    if cache and cache_enabled and response:
        try:
            cache.set(messages, cache_key[0], cache_key[1], response)
        except (OSError, ValueError) as e:
            logger.warning(f"Cache write failed: {e}")

    # 用认知系统处理响应
    if mind and enable_cognition:
        response = mind.process_response(
            agent_name, query, response, cognitive_state, had_monologue
        )
        # 保存更新后的认知状态
        save_cognitive_state_to_dict(state, cognitive_state)
        logger.info(f"[{agent_name}] 💭 认知处理完成，turn={cognitive_state.turn_count}")

    logger.debug(f"[{agent_name}] Generation complete, response_length={len(response)}")
    return _build_result_dict(
        response, agent_name,
        cognitive_state=cognitive_state if enable_cognition else None,
    )


async def coordinator_node(state: dict, sid: str | None = None) -> dict:
    """协调者Agent - 分析需求并决定路由（启用认知系统）"""
    # Coordinator 关闭内心独白（输出格式限制），但启用情感和直觉
    return await _run_agent(state, COORDINATOR_PROMPT, "coordinator", sid, enable_cognition=True)


TOOL_CALLER_PROMPT = """你是 ToolCaller（工具调用专家）。

你的职责：
1. 分析用户问题是否需要调用非搜索类工具（如计算、代码执行）
2. 如果需要，调用合适的工具获取结果
3. 将工具执行结果以简洁的方式返回

可用工具：
- execute_python: 执行 Python 代码，用于数学计算、数据处理、验证代码等

注意：
- 不要调用搜索类工具（联网搜索、记忆搜索、知识库搜索），这些由其他 Agent 处理
- 如果不需要调用工具，返回空即可
- 工具执行结果要简洁，不要过多解释"""


def _need_tool_call(query: str) -> bool:
    """判断是否需要非搜索类工具调用（计算、代码等）。"""
    q = query.lower()
    if any(kw in q for kw in ["计算", "等于多少", "+", "*", "/", "平方", "次方", "百分比"]):
        return True
    if any(kw in q for kw in ["运行代码", "执行代码", "算一下", "验证"]):
        return True
    return False


async def tool_caller_node(state: dict, sid: str | None = None) -> dict:
    """工具调用子节点 - 直接执行非搜索类工具，不走 _run_agent 的完整 Agent 流程。

    结果以 SystemMessage 注入 Responder 上下文。
    """
    query = state["messages"][-1].content if state["messages"] else ""

    # 快速判断：不需要工具时直接返回空
    if not _need_tool_call(query):
        return {"messages": []}

    from cognition.tool_engine import execute_python, run_tool_loop

    llm = get_llm(sid or "")
    tools = [execute_python]
    messages = [SystemMessage(content=TOOL_CALLER_PROMPT)] + list(state["messages"])

    # 执行工具调用循环
    response = await run_tool_loop(
        llm, messages, tools,
        max_iterations=2,
        sid=sid or "",
    )

    if response:
        return {"messages": [SystemMessage(
            content=f"【工具执行结果】\n\n{response}",
            name="tool_caller",
        )]}
    return {"messages": []}


async def _safe_search(
    search_fn: callable,
    label: str,
    *args,
    success_check: callable = None,
    **kwargs,
) -> str:
    """通用搜索包装器——统一异常处理和结果格式化。

    消除 web_searcher_agent / memory_searcher_agent / knowledge_searcher_agent
    中重复的 try/except 模式。
    """
    try:
        result = await search_fn(*args, **kwargs)
        if success_check:
            if success_check(result):
                return f"[{label}]\n\n{result}"
        elif result:
            return f"[{label}]\n\n{result}"
    except asyncio.TimeoutError:
        pass
    except (TimeoutError, ConnectionError):
        pass
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"{label} failed: {e}")
    return ""


async def web_searcher_agent(query: str) -> str:
    """联网搜索子 Agent - 执行 DuckDuckGo 搜索并总结结果。

    整体超时 6 秒，避免搜索卡住响应流程。
    """
    from tools.search import search_and_summarize

    def _check(result: str) -> bool:
        return result and "未找到" not in result

    async def _search_with_timeout(q: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(search_and_summarize, q, max_results=2),
            timeout=6.0,
        )

    return await _safe_search(
        _search_with_timeout,
        "联网搜索结果",
        query,
        success_check=_check,
    )


async def memory_searcher_agent(query: str, user_id: str = "") -> str:
    """记忆搜索子 Agent - 从自适应记忆系统检索相关记忆。"""
    from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE

    async def _do_search(q: str, uid: str) -> str:
        if not _MEMORY_SYSTEM_AVAILABLE:
            return ""
        store = get_memory_store()
        memories = await asyncio.wait_for(
            store.retrieve(q, top_k=5, user_id=uid),
            timeout=1.0,
        )
        if memories:
            return store.format_memories_for_prompt(memories) or ""
        return ""

    return await _safe_search(_do_search, "记忆检索结果", query, user_id)


async def knowledge_searcher_agent(query: str) -> str:
    """知识库搜索子 Agent - 从 RAG 向量库检索相关文档。"""
    from core.rag import search_knowledge

    def _check(result: str) -> bool:
        return result and "知识库为空" not in result and "未启用" not in result

    return await _safe_search(
        lambda q: asyncio.to_thread(search_knowledge, q, top_k=3),
        "知识库检索结果",
        query,
        success_check=_check,
    )


async def _run_parallel_search(state: dict) -> str:
    """根据 state 中的 mode 并行运行搜索子 Agent，返回整合后的搜索上下文。"""
    user_message = state["messages"][-1].content
    mode = state.get("task_context", {}).get("mode", "coordination")
    user_id = state.get("task_context", {}).get("user_id", "")

    # 根据模式决定启用哪些搜索子 Agent
    tasks = []
    if mode in ("fast", "planning"):
        # 快速/计划模式：联网 + 记忆
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id))
    else:
        # 协调模式：联网 + 记忆 + 知识库
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id))
        tasks.append(knowledge_searcher_agent(user_message))

    # 并行执行搜索
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 整合搜索结果
    search_parts = []
    for result in results:
        if isinstance(result, str) and result:
            search_parts.append(result)

    return "\n\n".join(search_parts) if search_parts else ""


async def researcher_node(state: dict, sid: str | None = None) -> dict:
    """研究员 Agent - 仅并行搜索并整合结果，不做 LLM 生成。

    搜索结果以 SystemMessage 形式注入消息流，由 Responder 直接生成最终回复。
    避免 Researcher 和 Responder 重复生成，减少一次 LLM 调用。
    """
    search_context = await _run_parallel_search(state)

    if search_context:
        # 将搜索结果作为系统消息注入，Responder 会在其上下文中生成最终回复
        system_msg = SystemMessage(
            content=f"【搜索结果】\n\n{search_context}\n\n请基于以上搜索结果生成最终回答。",
            name="researcher",
        )
        return {"messages": [system_msg]}

    # 无搜索结果时返回空，Responder 直接基于原始消息生成
    return {"messages": []}


async def responder_node(state: dict, sid: str | None = None) -> dict:
    """响应者Agent - 生成最终回答（完整认知系统 + 工具调用）"""
    plugin_prompt = get_plugins_prompt()

    # 读取自动检测到的用户语言（从 task_context 中获取）
    detected_lang = state.get("task_context", {}).get("detected_language", "zh")
    lang_instructions = {
        "zh": "\n\n重要：你必须用中文回答。",
        "en": "\n\nIMPORTANT: You must respond entirely in English.",
        "ja": "\n\n重要：あなたは日本語で回答しなければなりません。",
        "ko": "\n\n중요: 한국어로 답변해야 합니다.",
        "fr": "\n\nIMPORTANT: Vous devez répondre entièrement en français.",
        "de": "\n\nWICHTIG: Sie müssen vollständig auf Deutsch antworten.",
        "es": "\n\nIMPORTANTE: Debe responder completamente en español.",
        "ru": "\n\nВАЖНО: Вы должны отвечать полностью на русском языке.",
        "ar": "\n\nمهم: يجب أن ترد باللغة العربية بالكامل.",
    }
    lang_instr = lang_instructions.get(detected_lang, "")

    # 基础提示词（认知系统会在此基础上叠加人格、情感等元素）
    responder_prompt = f"""你是 ResponderBot（果冻ai），一位乐于助人且友善的助手。

你的职责是：
1. 提供清晰、友好的回复
2. 以易于理解的方式呈现信息
3. 保持对话式、亲切的风格
4. 当需要实时信息、计算、或查询记忆时，主动调用工具获取准确信息
{plugin_prompt}{lang_instr}

重要：每次回答时，你必须以"我是果冻ai"开头，然后再根据上下文生成最终回答。"""

    # Responder 只负责生成最终回答，不再自己调用工具
    # 搜索和工具调用由前置并行子 Agent 完成
    return await _run_agent(
        state, responder_prompt, "responder", sid,
        enable_cognition=True, tools=None,
    )


async def reviewer_node(state: dict, language: str = "zh", sid: str | None = None) -> dict:
    """检查者Agent - 审查回答质量（启用认知系统）"""
    reviewer_prompt = get_reviewer_prompt(language)
    return await _run_agent(state, reviewer_prompt, "reviewer", sid, enable_cognition=True)


PLANNER_PROMPT = """你是 PlannerBot（果冻ai团队的任务规划专家）。

分析用户的复杂需求并生成清晰的任务执行计划。每次回答时请先以"我是果冻ai"开头。

返回格式必须是纯 JSON（不要包含 markdown 代码块标记）：
{
  "title": "计划标题",
  "steps": [
    {"index": 1, "title": "步骤标题", "description": "步骤描述"},
    {"index": 2, "title": "步骤标题", "description": "步骤描述"},
    ...
  ]
}

要求：
- 步骤数控制在 3-8 个
- 每个步骤描述要具体、可执行
- 步骤之间有逻辑顺序
- 仅返回 JSON，不要添加任何其他文字说明"""


async def planner_node(state: dict, sid: str | None = None) -> dict:
    """计划者Agent - 分析需求并生成结构化计划"""
    return await _run_agent(state, PLANNER_PROMPT, "planner", sid)


def parse_plan_from_response(text: str) -> dict | None:
    """从 Agent 输出中提取 JSON 计划。"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 尝试从 markdown 代码块中提取
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 尝试从文本中找 JSON 对象
    match = re.search(r"\{[\s\S]*\"title\"[\s\S]*\"steps\"[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def create_agents(language: str = "zh", fast_mode: bool = False):
    """创建所有Agent节点函数的便捷函数

    所有模式都包含 Coordinator 和 Researcher，区别仅在于：
    - 快速/计划模式：Researcher 下并行 2 个搜索子 Agent（联网 + 记忆）
    - 协调模式：Researcher 下并行 3 个搜索子 Agent（联网 + 记忆 + 知识库）
    """
    async def _reviewer_node(state: dict, sid: str | None = None) -> dict:
        return await reviewer_node(state, language=language, sid=sid)

    # 所有模式都返回完整的 Agent 集合
    return (
        coordinator_node,
        researcher_node,
        responder_node,
        _reviewer_node,
    )
