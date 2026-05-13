"""配置系统 - 统一的配置入口。

所有配置从 state/model_configs.json 读取（由 model_config_manager 管理）。
保留 .env 加载用于向后兼容和其他环境变量。
"""

import os
from typing import Any

from dotenv import load_dotenv

# 加载 .env（用于向后兼容和其他环境变量）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(_ENV_PATH, override=True)


def _get_active_config() -> dict | None:
    """获取当前活跃配置（从 model_config_manager）。"""
    from state.model_config_manager import get_active_config
    return get_active_config()


def _get_config_value(cfg: dict | None, key: str, env_var: str = "", provider_fallback: str = "") -> str:
    """通用配置获取逻辑：优先从配置字典读取，再回退到环境变量。"""
    if cfg:
        value = cfg.get(key, "")
        if value:
            return value
    if env_var:
        value = os.getenv(env_var, "")
        if value:
            return value
    return provider_fallback


def get_provider() -> str:
    cfg = _get_active_config()
    if cfg:
        return cfg.get("provider", "ollama").lower()
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def get_api_key(provider: str | None = None) -> str:
    """获取指定提供商的 API Key，各提供商完全隔离"""
    cfg = _get_active_config()
    p = (provider or (cfg.get("provider", "") if cfg else get_provider())).lower()
    if p == "ollama":
        return "ollama"

    key = _get_config_value(cfg, "apiKey", f"LLM_API_KEY_{p.upper().replace('-', '_')}")
    if key:
        return key
    raise ValueError(f"API Key not set for provider '{p}'")


def get_base_url() -> str:
    cfg = _get_active_config()
    url = _get_config_value(cfg, "baseUrl", "LLM_BASE_URL")
    if url:
        return url

    provider = (cfg.get("provider", "") if cfg else get_provider()).lower()
    if provider in PROVIDER_CONFIG:
        return PROVIDER_CONFIG[provider]["base_url"]
    raise ValueError(f"Unknown provider '{provider}'")


def get_model_name() -> str:
    cfg = _get_active_config()
    model = _get_config_value(cfg, "model", "LLM_MODEL_NAME")
    if model:
        return model

    provider = (cfg.get("provider", "") if cfg else get_provider()).lower()
    if provider in PROVIDER_CONFIG:
        return PROVIDER_CONFIG[provider]["default_model"]
    raise ValueError(f"Unknown provider '{provider}'")


def list_providers() -> list[str]:
    return list(PROVIDER_CONFIG.keys())


# 国内外大模型厂商配置
PROVIDER_CONFIG = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "default_model": "qwen-plus"},
    "minimax": {"base_url": "https://api.minimax.chat/v1", "default_model": "MiniMax-Text-01"},
    "doubao": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "default_model": "doubao-pro-32k"},
    "glm": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "default_model": "glm-4-flash"},
    "ernie": {"base_url": "https://qianfan.baidubce.com/v1", "default_model": "ernie-4.0-8k-latest"},
    "hunyuan": {"base_url": "https://hunyuan.tencentcloudapi.com/v1", "default_model": "hunyuan-pro"},
    "spark": {"base_url": "https://spark-api.xf-yun.com/v3.1", "default_model": "spark-4.0"},
    "kimi": {"base_url": "https://api.moonshot.cn/v1", "default_model": "kimi-k2.6"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "default_model": "deepseek-ai/DeepSeek-V3"},
    "kimi-code": {"base_url": "https://api.kimi.com/coding/v1", "default_model": "kimi-for-coding"},
    "yi": {"base_url": "https://api.lingyiwanwu.com/v1", "default_model": "yi-large"},
    "baichuan": {"base_url": "https://api.baichuan-ai.com/v1", "default_model": "baichuan4"},
    "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "default_model": "claude-sonnet-4-20250514"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "default_model": "gemini-2.0-flash"},
    "grok": {"base_url": "https://api.x.ai/v1", "default_model": "grok-3"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "default_model": "mistral-large-latest"},
    "cohere": {"base_url": "https://api.cohere.com/compatibility/v1", "default_model": "command-r-plus"},
    "perplexity": {"base_url": "https://api.perplexity.ai", "default_model": "sonar-pro"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"},
    "together": {"base_url": "https://api.together.xyz/v1", "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "azure": {"base_url": "", "default_model": "gpt-4o"},
    "ollama": {"base_url": "http://localhost:11434/v1", "default_model": "llama3.2"},
}


# 提供商中文名称映射
PROVIDER_NAMES = {
    "ollama": "Ollama 本地",
    "deepseek": "DeepSeek",
    "qwen": "阿里 Qwen",
    "minimax": "MiniMax",
    "doubao": "字节豆包",
    "glm": "智谱 GLM",
    "ernie": "百度文心",
    "hunyuan": "腾讯混元",
    "spark": "讯飞星火",
    "kimi": "月之暗面 Kimi",
    "siliconflow": "硅基流动",
    "kimi-code": "Kimi Code",
    "yi": "零一万物 Yi",
    "baichuan": "百川 Baichuan",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "grok": "xAI Grok",
    "mistral": "Mistral AI",
    "cohere": "Cohere",
    "perplexity": "Perplexity",
    "groq": "Groq",
    "together": "Together AI",
    "azure": "Azure OpenAI",
}


# OpenAI 兼容 base_url 映射
BASE_URLS = {name: cfg["base_url"] for name, cfg in PROVIDER_CONFIG.items()}


# ========================================================================
# 运行时调优常量 - 之前散落在 web/app.py、core/cache.py
# ========================================================================

# LangGraph 整体执行超时(秒)。超时后中断流式响应,避免网络或 LLM 挂起。
GRAPH_TIMEOUT = 60.0

# Socket 状态不活跃超时(秒)。超过此值的会话会被清理。
SOCKET_INACTIVE_TIMEOUT = 30 * 60

# 服务端 token 批处理阈值:凑齐这么多字符再 emit,减少 socketio 帧数。
# 值越小响应越"实时"但网络帧数越多;值越大批处理效率越高但用户感觉越"卡"。
# 本地/低延迟 API 建议 12-20;海外高延迟 API 建议 6-10。
TOKEN_FLUSH_CHARS = 12

# LLM 响应缓存默认 TTL(秒)。
CACHE_DEFAULT_TTL = 24 * 3600
