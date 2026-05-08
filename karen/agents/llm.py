"""LLM 基础设施 - HTTP 客户端、配置管理和实例缓存。"""

import json
import logging
import threading
from typing import Optional, Callable

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ========== HTTP 客户端 ==========

_httpx_client = None


def _get_http_client():
    """获取全局共享的 httpx.Client，启用连接池复用。"""
    global _httpx_client
    if _httpx_client is None:
        try:
            import httpx
            _httpx_client = httpx.Client(
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0),
                timeout=httpx.Timeout(60.0, connect=5.0),
            )
            logger.info("HTTP client pool initialized (keepalive=30s, max=100)")
        except ImportError:
            logger.warning("httpx not installed, falling back to default HTTP client")
            return None
    return _httpx_client


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


# ========== LLM 实例管理 ==========

_llm_cache: dict[str, ChatOpenAI] = {}
_llm_cache_lock = threading.Lock()


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
        if not api_key and provider == "ollama":
            api_key = "ollama"
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
    cache_key = _make_cache_key(kwargs)
    with _llm_cache_lock:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]
        logger.debug(f"Creating new LLM instance for model={kwargs.get('model')}")
        http_client = _get_http_client()
        if http_client:
            kwargs["http_client"] = http_client
        instance = ChatOpenAI(**kwargs)
        _llm_cache[cache_key] = instance
        return instance


def clear_llm_cache():
    """清除 LLM 实例缓存。"""
    with _llm_cache_lock:
        _llm_cache.clear()
    logger.info("LLM cache cleared")
