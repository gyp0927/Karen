import json
import os
import uuid
from datetime import datetime

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "model_configs.json")


def _ensure_file():
    """确保配置文件存在"""
    if not os.path.exists(_CONFIG_FILE):
        _save_data({"configs": [], "activeConfigId": None})


def _load_data() -> dict:
    """加载配置数据"""
    _ensure_file()
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"configs": [], "activeConfigId": None}


def _save_data(data: dict):
    """保存配置数据"""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_configs() -> list[dict]:
    """获取所有配置列表（API Key 脱敏）"""
    data = _load_data()
    configs = []
    for c in data.get("configs", []):
        cfg = dict(c)
        if cfg.get("apiKey"):
            cfg["apiKey"] = _mask_key(cfg["apiKey"])
        configs.append(cfg)
    return configs


def list_configs_full() -> list[dict]:
    """获取所有配置列表（包含完整 API Key，仅后端使用）"""
    data = _load_data()
    return data.get("configs", [])


def get_config(config_id: str) -> dict | None:
    """获取单个配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            return dict(c)
    return None


def get_active_config() -> dict | None:
    """获取当前活跃配置"""
    data = _load_data()
    active_id = data.get("activeConfigId")
    if not active_id:
        # 如果没有活跃配置，返回第一个
        configs = data.get("configs", [])
        if configs:
            return dict(configs[0])
        return None
    for c in data.get("configs", []):
        if c.get("id") == active_id:
            return dict(c)
    return None


def add_config(name: str, provider: str, model: str, api_key: str, base_url: str = "") -> dict:
    """新增一个配置"""
    data = _load_data()
    config = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "provider": provider,
        "model": model,
        "apiKey": api_key,
        "baseUrl": base_url,
        "createdAt": datetime.now().isoformat(),
    }
    data["configs"].append(config)
    # 如果是第一个配置，自动设为活跃
    if len(data["configs"]) == 1:
        data["activeConfigId"] = config["id"]
    _save_data(data)
    # 返回脱敏版本
    result = dict(config)
    result["apiKey"] = _mask_key(result["apiKey"])
    return result


def update_config(config_id: str, **kwargs) -> dict | None:
    """更新配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            for key in ["name", "provider", "model", "apiKey", "baseUrl"]:
                if key in kwargs:
                    c[key] = kwargs[key]
            _save_data(data)
            result = dict(c)
            result["apiKey"] = _mask_key(result["apiKey"])
            return result
    return None


def delete_config(config_id: str) -> bool:
    """删除配置"""
    data = _load_data()
    configs = data.get("configs", [])
    for i, c in enumerate(configs):
        if c.get("id") == config_id:
            configs.pop(i)
            # 如果删除的是活跃配置，重新设置活跃配置
            if data.get("activeConfigId") == config_id:
                if configs:
                    data["activeConfigId"] = configs[0]["id"]
                else:
                    data["activeConfigId"] = None
            _save_data(data)
            return True
    return False


def set_active_config(config_id: str) -> bool:
    """设置活跃配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            data["activeConfigId"] = config_id
            _save_data(data)
            return True
    return False


def _mask_key(key: str) -> str:
    """API Key 脱敏显示"""
    if not key or len(key) <= 8:
        return key
    return key[:4] + "****" + key[-4:]


def sync_to_env(config: dict):
    """将配置同步到环境变量和 .env 文件（兼容旧系统）"""
    import os
    from dotenv import load_dotenv

    provider = config.get("provider", "ollama")
    model = config.get("model", "")
    api_key = config.get("apiKey", "")
    base_url = config.get("baseUrl", "")

    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL_NAME"] = model
    os.environ["LLM_BASE_URL"] = base_url

    key_env_name = f"LLM_API_KEY_{provider.upper().replace('-', '_')}"
    os.environ[key_env_name] = api_key

    # 同时更新 .env 文件
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(_PROJECT_ROOT, ".env")

    existing = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k] = v

    existing["LLM_PROVIDER"] = provider
    existing["LLM_BASE_URL"] = base_url
    existing["LLM_MODEL_NAME"] = model
    existing[key_env_name] = api_key
    existing.pop("LLM_API_KEY", None)

    lines = ["# LLM 配置\n"]
    for k, v in existing.items():
        lines.append(f"{k}={v}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    try:
        os.chmod(env_path, 0o600)
    except Exception:
        pass
