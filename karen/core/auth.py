"""用户认证系统 - 基于 API Key 的简单多用户认证。

设计原则：
- 无需密码，使用 API Key 作为身份凭证
- 向后兼容：未启用时行为完全一致
- 用户数据隔离：会话、消息按用户隔离

启用方式：环境变量 ENABLE_AUTH=true
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from functools import wraps
from typing import Optional

from flask import request

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "auth.db")
_lock = threading.RLock()

# 是否启用认证
AUTH_ENABLED = os.getenv("ENABLE_AUTH", "false").lower() in ("true", "1", "yes")


def _get_conn() -> sqlite3.Connection:
    from core.db_utils import get_sqlite_conn
    return get_sqlite_conn(_DB_PATH, enable_wal=False)


def init_auth_db():
    """初始化认证数据库"""
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL UNIQUE,
                    config_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_users_key ON users(api_key_hash);
            """)
            conn.commit()
        finally:
            conn.close()


class User:
    """用户对象"""

    def __init__(self, user_id: str, name: str, config: dict = None):
        self.id = user_id
        self.name = name
        self.config = config or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
        }


def _hash_key(api_key: str) -> str:
    """对 API Key 进行哈希（用于存储和查找）。"""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_user(name: str, api_key: str, config: dict = None) -> User:
    """创建新用户。"""
    with _lock:
        conn = _get_conn()
        try:
            user_id = str(uuid.uuid4())[:8]
            key_hash = _hash_key(api_key)
            now = time.time()
            conn.execute(
                "INSERT INTO users (id, name, api_key_hash, config_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, key_hash, json.dumps(config or {}), now)
            )
            conn.commit()
            logger.info(f"Created user: {user_id} ({name})")
            return User(user_id, name, config)
        except sqlite3.IntegrityError:
            raise ValueError("API Key 已被使用")
        finally:
            conn.close()


def authenticate(api_key: str) -> Optional[User]:
    """验证 API Key，返回用户对象。"""
    if not api_key:
        return None

    with _lock:
        conn = _get_conn()
        try:
            key_hash = _hash_key(api_key)
            cursor = conn.execute(
                "SELECT id, name, config_json FROM users WHERE api_key_hash = ?",
                (key_hash,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    row["id"],
                    row["name"],
                    json.loads(row["config_json"] or "{}")
                )
            return None
        finally:
            conn.close()


def get_user_by_id(user_id: str) -> Optional[User]:
    """通过 ID 获取用户。"""
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, name, config_json FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    row["id"],
                    row["name"],
                    json.loads(row["config_json"] or "{}")
                )
            return None
        finally:
            conn.close()


def list_users() -> list[dict]:
    """列出所有用户（不含敏感信息）。"""
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, name, created_at FROM users ORDER BY created_at DESC"
            )
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


def delete_user(user_id: str) -> bool:
    """删除用户。"""
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


def update_user_config(user_id: str, config: dict) -> bool:
    """更新用户配置。"""
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                "UPDATE users SET config_json = ? WHERE id = ?",
                (json.dumps(config), user_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


def auth_required(f):
    """认证装饰器（用于 HTTP 路由）。

    如果认证未启用，直接放行。
    否则检查 X-API-Key Header 或 api_key Cookie。
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)

        api_key = request.headers.get("X-API-Key", "") or request.cookies.get("api_key", "")
        user = authenticate(api_key)
        if not user:
            return {"success": False, "message": "未认证，请提供有效的 API Key"}, 401

        # 将用户对象附加到请求上下文
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


def get_current_user() -> Optional[User]:
    """获取当前请求的用户（在 auth_required 装饰器保护的路由中可用）。"""
    if not AUTH_ENABLED:
        return None
    return getattr(request, "current_user", None)


# 初始化数据库
init_auth_db()
