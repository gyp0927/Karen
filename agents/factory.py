import asyncio
import json
import logging
import re
import threading
from typing import Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from core.config import get_api_key, get_base_url, get_model_name, get_provider
from core.plugin_system import get_plugins_prompt
from core.cache import get_cache
from prompts.coordinator_prompt import COORDINATOR_PROMPT
from prompts.reviewer_prompt import get_reviewer_prompt
from state.stop_flag import is_stopped

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


async def _run_agent(
    state: dict,
    system_prompt: str,
    agent_name: str,
    sid: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> dict:
    """通用 Agent 执行函数。

    参数:
        state: 当前状态字典
        system_prompt: 系统提示词
        agent_name: Agent 名称（用于日志和消息标记）
        sid: Socket ID（用于隔离停止标志）
        on_token: 每收到一个 token chunk 时调用的回调函数(token_text: str)
    """
    llm = get_llm(sid or "")
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    logger.debug(f"[{agent_name}] Starting generation, messages_count={len(messages)}")

    # 对 responder 和 coordinator 节点启用缓存
    # - responder: 缓存最终输出，避免重复生成相同回答
    # - coordinator: 缓存路由决策，相同问题直接走缓存路径
    cache_enabled = agent_name in ("responder", "coordinator")
    cache = get_cache() if cache_enabled else None
    if cache and cache_enabled:
        try:
            _provider = get_provider()
            _model = get_model_name()
            cached_response = cache.get(messages, _provider, _model)
            if cached_response is not None:
                logger.info(f"[{agent_name}] Cache hit, skipping generation")
                # coordinator 不需要流式输出，直接返回
                if agent_name == "coordinator":
                    return {"messages": [AIMessage(content=cached_response, name=agent_name)]}
                stream_cb = on_token if on_token else get_streaming_callback(sid)
                if stream_cb:
                    # 模拟流式输出缓存内容（分块发送，避免前端卡顿）
                    chunk_size = 20
                    for i in range(0, len(cached_response), chunk_size):
                        stream_cb(cached_response[i:i+chunk_size])
                return {"messages": [AIMessage(content=cached_response, name=agent_name)]}
        except (OSError, ValueError) as e:
            logger.warning(f"Cache lookup failed: {e}, falling back to generation")

    # 限制 Coordinator 输出长度：只需要路由标记，最多 80 tokens
    # 大幅减少 Coordinator 节点的耗时（从 1-2 秒降到 0.3-0.5 秒）
    if agent_name == "coordinator":
        llm = llm.bind(max_tokens=80)

    response = ""
    # 只有 responder 节点触发流式回调（最终输出）
    # coordinator 和 reviewer 是内部节点，不需要流式展示
    stream_cb = on_token if on_token else (get_streaming_callback(sid) if agent_name == "responder" else None)
    async for chunk in llm.astream(messages):
        if is_stopped(sid):
            logger.info(f"[{agent_name}] Generation stopped by user")
            break
        if chunk.content:
            response += chunk.content
            if stream_cb:
                stream_cb(chunk.content)

    # 写入缓存（responder 和 coordinator）
    if cache and cache_enabled and response:
        try:
            _provider = get_provider()
            _model = get_model_name()
            cache.set(messages, _provider, _model, response)
        except (OSError, ValueError) as e:
            logger.warning(f"Cache write failed: {e}")

    logger.debug(f"[{agent_name}] Generation complete, response_length={len(response)}")
    return {"messages": [AIMessage(content=response, name=agent_name)]}


async def coordinator_node(state: dict, sid: str | None = None) -> dict:
    """协调者Agent - 分析需求并决定路由"""
    return await _run_agent(state, COORDINATOR_PROMPT, "coordinator", sid)


async def web_searcher_agent(query: str) -> str:
    """联网搜索子 Agent - 执行 DuckDuckGo 搜索并总结结果。"""
    try:
        from tools.search import search_and_summarize
        # search_and_summarize 是同步函数，使用 asyncio.to_thread 避免阻塞事件循环
        result = await asyncio.to_thread(search_and_summarize, query, max_results=3)
        if result and "未找到" not in result:
            return f"[联网搜索结果]\n\n{result}"
    except ImportError:
        pass
    except (TimeoutError, ConnectionError):
        pass
    except Exception as e:
        logger.warning(f"Web search agent failed: {e}")
    return ""


async def memory_searcher_agent(query: str, user_id: str = "") -> str:
    """记忆搜索子 Agent - 从自适应记忆系统检索相关记忆。"""
    try:
        from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE
        if _MEMORY_SYSTEM_AVAILABLE:
            store = get_memory_store()
            memories = await asyncio.wait_for(
                store.retrieve(query, top_k=5, user_id=user_id),
                timeout=2.0,
            )
            if memories:
                formatted = store.format_memories_for_prompt(memories)
                if formatted:
                    return f"[记忆检索结果]\n\n{formatted}"
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        logger.warning(f"Memory search agent failed: {e}")
    return ""


async def knowledge_searcher_agent(query: str) -> str:
    """知识库搜索子 Agent - 从 RAG 向量库检索相关文档。"""
    try:
        from core.rag import search_knowledge
        # search_knowledge 是同步函数，使用 asyncio.to_thread 避免阻塞事件循环
        result = await asyncio.to_thread(search_knowledge, query, top_k=3)
        if result and "知识库为空" not in result and "未启用" not in result:
            return f"[知识库检索结果]\n\n{result}"
    except ImportError:
        pass
    except (TimeoutError, ConnectionError):
        pass
    except Exception as e:
        logger.warning(f"Knowledge search agent failed: {e}")
    return ""


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
    """研究员 Agent - 并行调用多个搜索子 Agent，整合结果后生成研究回复。"""
    # 并行执行搜索子 Agent
    search_context = await _run_parallel_search(state)

    # 构建研究提示词，包含搜索结果
    if search_context:
        research_prompt = f"""你是 ResearcherBot（果冻ai团队的研究专家）。

{search_context}

基于上述搜索结果，你的职责是：
1. 提供准确、相关的信息
2. 深入分析话题
3. 以清晰、结构化的方式呈现研究结果

回答时请先以"我是果冻ai"开头，然后提供简洁但信息丰富的研究内容。"""
    else:
        research_prompt = """你是 ResearcherBot（果冻ai团队的研究专家）。

你的职责是：
1. 提供准确、相关的信息
2. 深入分析话题
3. 以清晰、结构化的方式呈现研究结果

回答时请先以"我是果冻ai"开头，然后提供简洁但信息丰富的研究内容。"""

    return await _run_agent(state, research_prompt, "researcher", sid)


async def responder_node(state: dict, sid: str | None = None) -> dict:
    """响应者Agent - 生成最终回答"""
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

    responder_prompt = f"""你是 ResponderBot（果冻ai），一位乐于助人且友善的助手。

你的职责是：
1. 提供清晰、友好的回复
2. 以易于理解的方式呈现信息
3. 保持对话式、亲切的风格
{plugin_prompt}{lang_instr}

重要：每次回答时，你必须以"我是果冻ai"开头，然后再根据上下文生成最终回答。"""
    return await _run_agent(state, responder_prompt, "responder", sid)


async def reviewer_node(state: dict, language: str = "zh", sid: str | None = None) -> dict:
    """检查者Agent - 审查回答质量"""
    reviewer_prompt = get_reviewer_prompt(language)
    return await _run_agent(state, reviewer_prompt, "reviewer", sid)


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
