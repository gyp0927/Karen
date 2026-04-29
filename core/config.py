import hashlib
import json
import os
import time
from typing import Any

from dotenv import load_dotenv

# 显式加载项目根目录的 .env，避免工作目录不同导致找不到
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(_ENV_PATH, override=True)

# 配置缓存（避免每次调用都读取文件）
_config_cache: dict | None = None
_config_cache_mtime: float = 0
_config_cache_hash: str = ""
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "state", "model_configs.json")


def _compute_file_hash(file_path: str) -> str:
    """计算文件内容的 SHA-256 哈希，用于精确检测内容变化。"""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _try_load_from_model_configs():
    """尝试从 model_configs.json 加载活跃配置（带 mtime + 内容哈希双重缓存）。"""
    global _config_cache, _config_cache_mtime, _config_cache_hash

    try:
        mtime = os.path.getmtime(_CONFIG_FILE)
    except OSError:
        mtime = 0

    # 快速路径：mtime 未变化时直接返回缓存
    if _config_cache is not None and mtime == _config_cache_mtime:
        return _config_cache

    if not os.path.exists(_CONFIG_FILE):
        _config_cache = None
        _config_cache_mtime = mtime if mtime else time.time()
        _config_cache_hash = ""
        return None

    content_hash = _compute_file_hash(_CONFIG_FILE)

    # 如果 mtime 变了但内容哈希没变（某些编辑器会 touch 文件），保留缓存
    if _config_cache is not None and content_hash == _config_cache_hash:
        _config_cache_mtime = mtime if mtime else time.time()
        return _config_cache

    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        _config_cache = None
        _config_cache_mtime = mtime if mtime else time.time()
        _config_cache_hash = content_hash
        return None

    configs = data.get("configs", [])
    active_id = data.get("activeConfigId")
    result = None
    if active_id:
        for c in configs:
            if c.get("id") == active_id:
                result = c
                break
    if result is None and configs:
        result = configs[0]

    _config_cache = result
    _config_cache_mtime = mtime if mtime else time.time()
    _config_cache_hash = content_hash
    return result


# 国内外大模型厂商配置
PROVIDER_CONFIG = {
    # 国内
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-Text-01",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
    },
    "ernie": {
        "base_url": "https://qianfan.baidubce.com/v1",
        "default_model": "ernie-4.0-8k-latest",
    },
    "hunyuan": {
        "base_url": "https://hunyuan.tencentcloudapi.com/v1",
        "default_model": "hunyuan-pro",
    },
    "spark": {
        "base_url": "https://spark-api.xf-yun.com/v3.1",
        "default_model": "spark-4.0",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.6",
    },
    "kimi-code": {
        "base_url": "https://api.kimi.com/coding/v1",
        "default_model": "kimi-for-coding",
    },
    "yi": {
        "base_url": "https://api.lingyiwanwu.com/v1",
        "default_model": "yi-large",
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "baichuan4",
    },
    # 国外
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-20250514",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
    },
    "cohere": {
        "base_url": "https://api.cohere.com/compatibility/v1",
        "default_model": "command-r-plus",
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "default_model": "sonar-pro",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "azure": {
        "base_url": "",
        "default_model": "gpt-4o",
    },
    # 本地
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
    },
}


def get_provider() -> str:
    cfg = _try_load_from_model_configs()
    if cfg:
        return cfg.get("provider", "ollama").lower()
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def _get_config_value(
    cfg: dict | None,
    key: str,
    env_var: str,
    provider_fallback: str = "",
) -> str:
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


def get_api_key(provider: str | None = None) -> str:
    """获取指定提供商的 API Key，各提供商完全隔离"""
    cfg = _try_load_from_model_configs()
    p = (provider or (cfg.get("provider", "") if cfg else get_provider())).lower()
    if p == "ollama":
        return "ollama"

    key = _get_config_value(cfg, "apiKey", f"LLM_API_KEY_{p.upper().replace('-', '_')}")
    if key:
        return key
    raise ValueError(f"API Key not set for provider '{p}'")


def get_base_url() -> str:
    cfg = _try_load_from_model_configs()
    url = _get_config_value(cfg, "baseUrl", "LLM_BASE_URL")
    if url:
        return url

    provider = (cfg.get("provider", "") if cfg else get_provider()).lower()
    if provider in PROVIDER_CONFIG:
        return PROVIDER_CONFIG[provider]["base_url"]
    raise ValueError(f"Unknown provider '{provider}'. Supported: {list(PROVIDER_CONFIG.keys())}")


def get_model_name() -> str:
    cfg = _try_load_from_model_configs()
    model = _get_config_value(cfg, "model", "LLM_MODEL_NAME")
    if model:
        return model

    provider = (cfg.get("provider", "") if cfg else get_provider()).lower()
    if provider in PROVIDER_CONFIG:
        return PROVIDER_CONFIG[provider]["default_model"]
    raise ValueError(f"Unknown provider '{provider}'. Supported: {list(PROVIDER_CONFIG.keys())}")


def list_providers() -> list[str]:
    return list(PROVIDER_CONFIG.keys())


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