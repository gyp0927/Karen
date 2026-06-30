import json
import os
import threading
import uuid
from base64 import urlsafe_b64decode as b64decode, urlsafe_b64encode as b64encode
from datetime import datetime
from typing import cast

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "model_configs.json")
_KEY_FILE = os.path.join(_CONFIG_DIR, ".model_configs_key")

# 内存缓存：避免每次读配置都打开文件
_cache_lock = threading.Lock()
_cache_data: dict | None = None
_cache_mtime: float = 0.0


def _derive_key(password: str | None = None, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """用 PBKDF2 从机器相关数据派生加密密钥。"""
    if salt is None:
        salt = os.urandom(16)
    if password is None:
        # 用机器名 + 用户名 + 项目路径作为默认密码，保证同一台机器能解密。
        try:
            login = os.getlogin()
        except OSError:
            login = "user"
        password = f"{login}@{os.environ.get('COMPUTERNAME', '')}:{_CONFIG_DIR}"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    key = b64encode(kdf.derive(password.encode("utf-8")))
    return key, salt


def _get_fernet() -> Fernet:
    """获取或创建 Fernet 实例。"""
    salt: bytes | None = None
    if os.path.exists(_KEY_FILE):
        try:
            with open(_KEY_FILE, "rb") as f:
                salt = f.read()
        except OSError:
            pass
    key, salt = _derive_key(salt=salt)
    if salt is not None:
        with open(_KEY_FILE, "wb") as f:
            f.write(salt)
    return Fernet(key)


def _encrypt(value: str) -> str:
    """加密字符串，返回 base64 编码的密文。"""
    if not value:
        return value
    try:
        return _get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")
    except Exception as e:
        # 加密失败时不应阻塞业务，但应该记录。
        import logging

        logging.getLogger(__name__).warning(f"Encryption failed: {e}")
        return value


def _decrypt(value: str) -> str:
    """解密 _encrypt 生成的密文。"""
    if not value:
        return value
    try:
        return _get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Decryption failed: {e}")
        # 解密失败可能是旧版明文数据，直接返回原值，兼容迁移。
        return value


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
        if cfg.get("apiKey"):
            cfg["apiKey"] = _mask_key(_decrypt(cfg["apiKey"]))
        configs.append(cfg)
    return configs


def list_configs_full() -> list[dict]:
    """获取所有配置列表（包含完整 API Key，仅后端使用）"""
    data = _load_data()
    configs = []
    for c in data.get("configs", []):
        cfg = dict(c)
        if cfg.get("apiKey"):
            cfg["apiKey"] = _decrypt(cfg["apiKey"])
        configs.append(cfg)
    return configs


def get_config(config_id: str) -> dict | None:
    """获取单个配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            cfg = dict(c)
            if cfg.get("apiKey"):
                cfg["apiKey"] = _decrypt(cfg["apiKey"])
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
            if cfg.get("apiKey"):
                cfg["apiKey"] = _decrypt(cfg["apiKey"])
            return cfg
        return None
    for c in data.get("configs", []):
        if c.get("id") == active_id:
            cfg = dict(c)
            if cfg.get("apiKey"):
                cfg["apiKey"] = _decrypt(cfg["apiKey"])
            return cfg
    return None


def add_config(name: str, provider: str, model: str, api_key: str, base_url: str = "") -> dict:
    """新增一个配置"""
    data = _load_data()
    encrypted_key = _encrypt(api_key)
    # 如果同名同 provider 同 model 已存在，直接更新旧配置而不是新增
    for c in data.get("configs", []):
        if c.get("name") == name and c.get("provider") == provider and c.get("model") == model:
            c["apiKey"] = encrypted_key
            if base_url:
                c["baseUrl"] = base_url
            _save_data(data)
            result = dict(c)
            result["apiKey"] = _mask_key(api_key)
            return result
    config = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "provider": provider,
        "model": model,
        "apiKey": encrypted_key,
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
    result["apiKey"] = _mask_key(api_key)
    return result


def update_config(config_id: str, **kwargs) -> dict | None:
    """更新配置"""
    data = _load_data()
    for c in data.get("configs", []):
        if c.get("id") == config_id:
            for key in ["name", "provider", "model", "apiKey", "baseUrl"]:
                if key in kwargs:
                    value = kwargs[key]
                    if key == "apiKey":
                        value = _encrypt(value)
                    c[key] = value
            _save_data(data)
            result = dict(c)
            if result.get("apiKey"):
                result["apiKey"] = _mask_key(_decrypt(result["apiKey"]))
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
