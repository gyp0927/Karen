"""LLM 基础设施 - HTTP 客户端、配置管理和实例缓存。"""

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from typing import Optional, Callable

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ========== HTTP 客户端 ==========
# httpx.AsyncClient 绑定在创建它的 event loop 上。Flask-SocketIO threading
# 模式下线程池中的线程可能销毁重建，导致 loop id 被重用。用
# (thread_id, loop_id) 组合键降低碰撞概率，同时定期清理不再活跃的条目。

_httpx_clients: "OrderedDict[tuple[int, int], object]" = OrderedDict()
_httpx_lock = threading.Lock()
_HTTPX_MAX_CLIENTS = 8


def _get_http_async_client():
    """获取与当前线程+event loop 绑定的 httpx.AsyncClient。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    thread_id = threading.current_thread().ident
    loop_id = id(loop)
    key = (thread_id, loop_id)
    with _httpx_lock:
        client = _httpx_clients.get(key)
        if client is not None:
            _httpx_clients.move_to_end(key)
            return client
        try:
            import httpx
            client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=300.0,
                ),
                timeout=httpx.Timeout(60.0, connect=5.0),
            )
        except ImportError:
            logger.warning("httpx not installed, falling back to default HTTP client")
            return None
        _httpx_clients[key] = client
        if len(_httpx_clients) > _HTTPX_MAX_CLIENTS:
            _, old_client = _httpx_clients.popitem(last=False)
            from core.utils import spawn_bg
            spawn_bg(_close_httpx_client, old_client)
        logger.debug(f"Async HTTP client created for thread={thread_id}, loop={loop_id}")
        return client


def _close_httpx_client(client):
    """在独立 loop 中安全关闭 httpx client。"""
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()
    except Exception:
        pass




# ========== LLM 配置隔离 ==========

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


def cleanup_llm_config(sid: str = ""):
    """清理指定 sid 的 LLM 配置和回调，防止 socket 断开后内存泄漏。"""
    _llm_configs.pop(sid, None)
    with _callbacks_lock:
        _token_callbacks.pop(sid, None)


# ========== LLM 实例管理 ==========
# 实例内部持有按 loop 绑定的 httpx client,所以缓存键也要含 loop id,
# 否则换 loop 时旧实例的 client 已失效。

_llm_cache: "OrderedDict[tuple, ChatOpenAI]" = OrderedDict()
_llm_cache_lock = threading.Lock()
_LLM_CACHE_MAX = 16


def _build_llm_kwargs(sid: str = "") -> dict:
    """构建 LLM 初始化参数"""
    from core.config import PROVIDER_CONFIG, get_provider, get_api_key, get_base_url, get_model_name
    cfg = _llm_configs.get(sid)
    if cfg:
        provider = cfg.get("provider", "ollama")
        base_url = cfg.get("baseUrl", "")
        if not base_url and provider in PROVIDER_CONFIG:
            base_url = PROVIDER_CONFIG[provider]["base_url"]
        api_key = cfg.get("apiKey", "")
        if not api_key:
            if provider == "ollama":
                api_key = "ollama"
            else:
                # 回退到 core.config 的 key 查找（支持 LLM_API_KEY_{PROVIDER} / LLM_API_KEY）
                try:
                    api_key = get_api_key(provider)
                except ValueError:
                    api_key = ""
        kwargs = {"api_key": api_key, "base_url": base_url, "model": cfg.get("model", ""), "temperature": 0.7}
    else:
        provider = get_provider()
        kwargs = {"api_key": get_api_key(), "base_url": get_base_url(), "model": get_model_name(), "temperature": 0.7}
    if provider == "kimi-code":
        kwargs["default_headers"] = {
            "User-Agent": "claude-code/1.0",
            "X-Stainless-Lang": "python",
            "X-Stainless-Package-Version": "1.0.0",
            "X-Stainless-Runtime": "python",
            "X-Stainless-Runtime-Version": "3.13.0",
            "X-Stainless-OS": "Windows",
            "X-Stainless-Arch": "x64",
        }
    return kwargs


def get_llm_provider_model(sid: str = "") -> tuple[str, str]:
    """返回某个 sid 实际使用的 (provider, model)。

    Cache key 和 stats 上报必须用这个,而非 core.config.get_provider() —— 那是
    全局 env 默认值,sid 切档/Web 端用户配置生效后会与实际不一致,导致 cache
    命中错档的响应或 stats 把流量算到错的 provider 上。
    """
    from core.config import get_provider, get_model_name
    cfg = _llm_configs.get(sid)
    if cfg:
        return cfg.get("provider", "ollama"), cfg.get("model", "")
    return get_provider(), get_model_name()


def _make_cache_key(kwargs: dict) -> str:
    """构建基于 JSON 的缓存键。"""
    return json.dumps(kwargs, sort_keys=True)


def get_llm(sid: str = "") -> ChatOpenAI:
    """获取 LLM 实例（带缓存）。"""
    kwargs = _build_llm_kwargs(sid)
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    cache_key = (loop_id, _make_cache_key(kwargs))
    with _llm_cache_lock:
        if cache_key in _llm_cache:
            _llm_cache.move_to_end(cache_key)
            return _llm_cache[cache_key]
        logger.debug(f"Creating new LLM instance for model={kwargs.get('model')}")
        http_async_client = _get_http_async_client()
        if http_async_client:
            kwargs["http_async_client"] = http_async_client
        kwargs["streaming"] = True
        instance = ChatOpenAI(**kwargs)
        _llm_cache[cache_key] = instance
        if len(_llm_cache) > _LLM_CACHE_MAX:
            _llm_cache.popitem(last=False)
        return instance


def clear_llm_cache():
    """清除 LLM 实例缓存。"""
    with _llm_cache_lock:
        _llm_cache.clear()
    logger.info("LLM cache cleared")


async def warmup_connection(sid: str = "") -> bool:
    """预热 LLM 连接 — 提前建立 TCP/TLS 连接,减少首 token 延迟。

    在服务器启动时调用一次,让 HTTP 连接池和目标 API 之间保持长连接。
    预热失败不抛异常,避免阻塞启动流程;失败时清除缓存中的坏实例。
    """
    cache_key = None
    try:
        llm = get_llm(sid)
        # 记录 cache_key 以便失败时清理
        kwargs = _build_llm_kwargs(sid)
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = 0
        cache_key = (loop_id, _make_cache_key(kwargs))
        # 发送一个极短请求触发连接建立,但不等待完整响应
        from langchain_core.messages import HumanMessage
        async for _ in llm.astream([HumanMessage(content="hi")]):
            break  # 只取第一个 chunk 就断,目的是建连
        logger.info("LLM connection warmed up")
        return True
    except Exception as e:
        logger.warning(f"Connection warmup failed (non-critical): {e}")
        if cache_key:
            with _llm_cache_lock:
                _llm_cache.pop(cache_key, None)
        return False
