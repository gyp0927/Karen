import json
import os
import threading
import uuid
from datetime import datetime
from typing import cast

import keyring

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "model_configs.json")
_KEYRING_SERVICE = "karen-ai-model-config"

# 内存缓存：避免每次读配置都打开文件
_cache_lock = threading.Lock()
_cache_data: dict | None = None
_cache_mtime: float = 0.0


def _keyring_username(config_id: str) -> str:
    """生成 keyring 条目用户名。"""
    return f"config-{config_id}"


def _store_api_key(config_id: str, api_key: str) -> None:
    """将 API Key 存入系统密钥环。"""
    if not api_key:
        _remove_api_key(config_id)
        return
    try:
        keyring.set_password(_KEYRING_SERVICE, _keyring_username(config_id), api_key)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to store API key in keyring: {e}")
        raise


def _load_api_key(config_id: str) -> str:
    """从系统密钥环读取 API Key；读取失败返回空字符串。"""
    try:
        value = keyring.get_password(_KEYRING_SERVICE, _keyring_username(config_id))
        return value or ""
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to load API key from keyring: {e}")
        return ""


def _remove_api_key(config_id: str) -> None:
    """从系统密钥环删除 API Key。"""
    try:
        keyring.delete_password(_KEYRING_SERVICE, _keyring_username(config_id))
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to remove API key from keyring: {e}")


def _migrate_legacy_encrypted_key(config_id: str, encrypted_value: str) -> str:
    """兼容旧版 Fernet 加密：尝试解密并迁移到 keyring。

    解密失败时返回原值（可能是旧版明文），调用方自行处理。
    """
    if not encrypted_value:
        return ""
    # 如果值看起来不是 Fernet 密文（长度太短或不含 Fernet 前缀），直接返回
    if len(encrypted_value) < 20 or not encrypted_value.startswith("gAAAA"):
        return encrypted_value
    try:
        from base64 import urlsafe_b64encode as b64encode
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        _LEGACY_KEY_FILE = os.path.join(_CONFIG_DIR, ".model_configs_key")
        salt = None
        if os.path.exists(_LEGACY_KEY_FILE):
            with open(_LEGACY_KEY_FILE, "rb") as f:
                salt = f.read()
        if salt is None:
            return encrypted_value
        try:
            login = os.getlogin()
        except OSError:
            login = "user"
        password = f"{login}@{os.environ.get('COMPUTERNAME', '')}:{_CONFIG_DIR}"
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100_000)
        key = b64encode(kdf.derive(password.encode("utf-8")))
        decrypted = Fernet(key).decrypt(encrypted_value.encode("utf-8")).decode("utf-8")
        # 解密成功后迁移到 keyring，并清空配置文件中的密文字段
        _store_api_key(config_id, decrypted)
        return decrypted
    except Exception:
        return encrypted_value


def _with_api_key(cfg: dict) -> dict:
    """将配置文件中的 apiKey 字段替换为 keyring 里的真实值。"""
    config_id = cfg.get("id")
    if not config_id:
        return cfg
    stored = cfg.get("apiKey", "")
    # 优先从 keyring 读取
    key = _load_api_key(config_id)
    if key:
        cfg["apiKey"] = key
        return cfg
    # keyring 没有，尝试兼容旧版加密
    if stored:
        migrated = _migrate_legacy_encrypted_key(config_id, stored)
        if migrated and migrated != stored:
            # 迁移成功：持久化到 keyring，并清空文件中的密文
            cfg["apiKey"] = migrated
            _store_api_key(config_id, migrated)
            stored_cfg = cfg.copy()
            stored_cfg["apiKey"] = ""
            _update_config_in_place(stored_cfg)
        else:
            # 旧版密文无法解密（可能已损坏或环境变化），返回空字符串，
            # 避免前端把密文当真实 key 使用或显示误导性的脱敏值。
            cfg["apiKey"] = ""
    return cfg


def _update_config_in_place(updated: dict) -> None:
    """用更新后的配置替换文件中的同 id 配置。"""
    data = _load_data()
    for i, c in enumerate(data.get("configs", [])):
        if c.get("id") == updated.get("id"):
            data["configs"][i] = updated
            _save_data(data)
            return


def _ensure_file():
    """确保配置文件存在"""
    if not os.path.exists(_CONFIG_FILE):
        _save_data({"configs": [], "activeConfigId": None})


def _load_data() -> dict:
    """加载配置数据（带内存缓存，文件未变更时直接返回缓存）"""
    global _cache_data, _cache_mtime
    _ensure_file()
    try:
        mtime = os.path.getmtime(_CONFIG_FILE)
    except OSError:
        mtime = 0.0
    with _cache_lock:
        if _cache_data is not None and mtime == _cache_mtime:
            return _cache_data
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        data = {"configs": [], "activeConfigId": None}
    with _cache_lock:
        _cache_data = data
        _cache_mtime = mtime
    return cast(dict, data)


def _save_data(data: dict):
    """保存配置数据（更新文件同时刷新内存缓存）"""
    global _cache_data, _cache_mtime
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with _cache_lock:
        _cache_data = data
        try:
            _cache_mtime = os.path.getmtime(_CONFIG_FILE)
        except OSError:
            _cache_mtime = 0.0


def list_configs() -> list[dict]:
    """获取所有配置列表（API Key 脱敏）"""
    data = _load_data()
    configs = []
    for c in data.get("configs", []):
        cfg = dict(c)
        cfg = _with_api_key(cfg)
        if cfg.get("apiKey"):
            cfg["apiKey"] = _mask_key(cfg["apiKey"])
        configs.append(cfg)
    return configs


def list_configs_full() -> list[dict]:
    """获取所有配置列表（包含完整 API Key，仅后端使用）"""
    data = _load_data()
    configs = []
    for c in data.get("configs", []):
        cfg = dict(c)
        cfg = _with_api_key(cfg)
        configs.append(cfg)
    return configs


def get_config(config_id: str) -> dict | None:
    """获取单个配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            cfg = dict(c)
            cfg = _with_api_key(cfg)
            return cfg
    return None


def get_active_config() -> dict | None:
    """获取当前活跃配置"""
    data = _load_data()
    active_id = data.get("activeConfigId")
    if not active_id:
        # 如果没有活跃配置，返回第一个
        configs = data.get("configs", [])
        if configs:
            cfg = dict(configs[0])
            cfg = _with_api_key(cfg)
            return cfg
        return None
    for c in data.get("configs", []):
        if c.get("id") == active_id:
            cfg = dict(c)
            cfg = _with_api_key(cfg)
            return cfg
    return None


def add_config(name: str, provider: str, model: str, api_key: str, base_url: str = "") -> dict:
    """新增一个配置"""
    data = _load_data()
    # 如果同名同 provider 同 model 已存在，直接更新旧配置而不是新增
    for c in data.get("configs", []):
        if c.get("name") == name and c.get("provider") == provider and c.get("model") == model:
            _store_api_key(c["id"], api_key)
            if base_url:
                c["baseUrl"] = base_url
            _save_data(data)
            result = dict(c)
            result["apiKey"] = _mask_key(api_key)
            return result
    config_id = str(uuid.uuid4())[:8]
    config = {
        "id": config_id,
        "name": name,
        "provider": provider,
        "model": model,
        "apiKey": "",
        "baseUrl": base_url,
        "createdAt": datetime.now().isoformat(),
    }
    data["configs"].append(config)
    # 如果是第一个配置，自动设为活跃
    if len(data["configs"]) == 1:
        data["activeConfigId"] = config_id
    _save_data(data)
    _store_api_key(config_id, api_key)
    # 返回脱敏版本
    result = dict(config)
    result["apiKey"] = _mask_key(api_key)
    return result


def update_config(config_id: str, **kwargs) -> dict | None:
    """更新配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            api_key = None
            for key in ["name", "provider", "model", "apiKey", "baseUrl"]:
                if key in kwargs:
                    value = kwargs[key]
                    if key == "apiKey":
                        api_key = value
                    else:
                        c[key] = value
            _save_data(data)
            if api_key is not None:
                _store_api_key(config_id, api_key)
            result = dict(c)
            result = _with_api_key(result)
            if result.get("apiKey"):
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
            _remove_api_key(config_id)
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
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k] = v
    existing[key_env_name] = api_key
    existing["LLM_PROVIDER"] = provider
    existing["LLM_MODEL_NAME"] = model
    existing["LLM_BASE_URL"] = base_url
    with open(env_path, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")
