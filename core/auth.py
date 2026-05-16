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


# ========== Per-IP 限流(防 API Key 暴力枚举) ==========
# 在 _AUTH_RATE_WINDOW 秒内累计失败 _AUTH_RATE_MAX 次,后续 _AUTH_RATE_BLOCK 秒拒绝。
# 不持久化:进程重启重置,够防自动化扫,不影响合法用户偶尔输错。
_AUTH_RATE_WINDOW = 60.0
_AUTH_RATE_MAX = 5
_AUTH_RATE_BLOCK = 300.0
_auth_failures: dict[str, list[float]] = {}
_auth_rate_lock = threading.RLock()

# 字典清理：每记录 _AUTH_PURGE_EVERY 次失败做一次全局扫描，回收已过 BLOCK 窗口的 IP 条目，
# 避免大量"打一枪就跑"的扫描 IP 把字典撑爆。同时设置硬上限作为最后兜底。
_AUTH_PURGE_EVERY = 100
_AUTH_MAX_ENTRIES = 10000
_auth_purge_counter = 0


def _purge_expired_failures(now: float) -> None:
    """清理 _auth_failures 中所有已过 BLOCK 窗口的条目。调用方需持有 _auth_rate_lock。"""
    expired = []
    for ip, attempts in _auth_failures.items():
        fresh = [t for t in attempts if now - t < _AUTH_RATE_BLOCK]
        if fresh:
            _auth_failures[ip] = fresh
        else:
            expired.append(ip)
    for ip in expired:
        _auth_failures.pop(ip, None)


def _check_rate_limit(ip: str) -> bool:
    """True 表示允许尝试,False 表示限流中。空 ip 不限流(本地调用)。"""
    if not ip:
        return True
    now = time.time()
    with _auth_rate_lock:
        attempts = [t for t in _auth_failures.get(ip, []) if now - t < _AUTH_RATE_BLOCK]
        if attempts:
            _auth_failures[ip] = attempts
        else:
            _auth_failures.pop(ip, None)
        recent = sum(1 for t in attempts if now - t < _AUTH_RATE_WINDOW)
        return recent < _AUTH_RATE_MAX


def _record_auth_failure(ip: str) -> None:
    global _auth_purge_counter
    if not ip:
        return
    now = time.time()
    with _auth_rate_lock:
        _auth_failures.setdefault(ip, []).append(now)
        _auth_purge_counter += 1
        if _auth_purge_counter >= _AUTH_PURGE_EVERY:
            _auth_purge_counter = 0
            _purge_expired_failures(now)
        # 硬上限兜底：极端场景下短时间内涌入大量唯一 IP，清扫后仍超限则丢弃最老的
        if len(_auth_failures) > _AUTH_MAX_ENTRIES:
            _purge_expired_failures(now)
            if len(_auth_failures) > _AUTH_MAX_ENTRIES:
                overflow = len(_auth_failures) - _AUTH_MAX_ENTRIES
                stalest = sorted(
                    _auth_failures.items(),
                    key=lambda kv: max(kv[1]) if kv[1] else 0,
                )[:overflow]
                for stale_ip, _ in stalest:
                    _auth_failures.pop(stale_ip, None)


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


def authenticate(api_key: str, ip: str = "") -> Optional[User]:
    """验证 API Key,返回用户对象。

    ip 用于触发 per-IP 失败次数限流,防止 API Key 暴力枚举。
    本地内部调用可省略 ip。
    """
    if not api_key:
        return None

    if not _check_rate_limit(ip):
        logger.warning(f"Auth rate-limited from ip={ip}")
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
                # 认证成功：清除该 IP 的失败记录，避免合法用户被累积失败拖入限流
                with _auth_rate_lock:
                    _auth_failures.pop(ip, None)
                return User(
                    row["id"],
                    row["name"],
                    json.loads(row["config_json"] or "{}")
                )
            _record_auth_failure(ip)
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
        ip = request.remote_addr or ""
        user = authenticate(api_key, ip=ip)
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
