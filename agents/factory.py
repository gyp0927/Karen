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

    # 仅对 responder 节点启用缓存（最终输出）
    is_responder = agent_name == "responder"
    cache = get_cache() if is_responder else None
    if cache and is_responder:
        try:
            _provider = get_provider()
            _model = get_model_name()
            cached_response = cache.get(messages, _provider, _model)
            if cached_response is not None:
                logger.info(f"[{agent_name}] Cache hit, skipping generation")
                stream_cb = on_token if on_token else get_streaming_callback(sid)
                if stream_cb:
                    # 模拟流式输出缓存内容
                    stream_cb(cached_response)
                return {"messages": [AIMessage(content=cached_response, name=agent_name)]}
        except (OSError, ValueError) as e:
            logger.warning(f"Cache lookup failed: {e}, falling back to generation")

    response = ""
    # 只有 responder 节点触发流式回调（最终输出）
    # 优先使用显式传入的 on_token，否则按 sid 查找全局回调
    stream_cb = on_token if on_token else (get_streaming_callback(sid) if is_responder else None)
    async for chunk in llm.astream(messages):
        if is_stopped(sid):
            logger.info(f"[{agent_name}] Generation stopped by user")
            break
        if chunk.content:
            response += chunk.content
            if stream_cb:
                stream_cb(chunk.content)

    # 写入缓存（仅 responder）
    if cache and is_responder and response:
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


async def researcher_node(state: dict, sid: str | None = None) -> dict:
    """研究员Agent - 提供准确信息"""
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
    responder_prompt = f"""你是 ResponderBot（果冻ai），一位乐于助人且友善的助手。

你的职责是：
1. 提供清晰、友好的回复
2. 以易于理解的方式呈现信息
3. 保持对话式、亲切的风格
{plugin_prompt}

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

    fast_mode=True: 只创建 Responder 和 Reviewer，跳过 Coordinator/Researcher
    """
    async def _reviewer_node(state: dict, sid: str | None = None) -> dict:
        return await reviewer_node(state, language=language, sid=sid)

    coordinator = None if fast_mode else coordinator_node
    researcher = None if fast_mode else researcher_node
    return (
        coordinator,
        researcher,
        responder_node,
        _reviewer_node,
    )
