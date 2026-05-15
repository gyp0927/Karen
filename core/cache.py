"""LLM 响应缓存层 - 基于 SQLite 的轻量级缓存。

缓存策略：
- 缓存键由 (provider, model, messages_hash) 计算
- 默认 TTL 24 小时
- 仅缓存成功响应
- 不缓存包含敏感上下文（代码执行、个人身份信息）的消息
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "cache.db")
_lock = threading.RLock()

# 默认 TTL 从 core.config 集中管理
from core.config import CACHE_DEFAULT_TTL as _DEFAULT_TTL


def _get_conn() -> sqlite3.Connection:
    from core.db_utils import get_sqlite_conn
    # 线程本地持久连接:避免每次 get/set 都新建 SQLite 连接(每次连接要重做
    # PRAGMA journal_mode=WAL/synchronous=NORMAL,空响应路径压力下成为瓶颈)。
    return get_sqlite_conn(_DB_PATH, use_thread_local=True)


def init_db():
    """初始化缓存数据库"""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                messages_preview TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache_entries(key_hash);
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
        """)
        conn.commit()


class ResponseCache:
    """LLM 响应缓存管理器"""

    # 每多少次 get 才执行一次过期清理，减少写操作
    _CLEANUP_INTERVAL = 50

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL, enabled: bool = True):
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._get_count = 0
        init_db()

    # 单条消息最大参与哈希长度,避免超长消息拖慢缓存键计算
    _MAX_MSG_HASH_LEN = 500

    def _get_cache_key(self, messages: list, provider: str, model: str) -> str:
        """生成缓存键（SHA256 哈希）。"""
        max_len = self._MAX_MSG_HASH_LEN
        content_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content[:max_len] if msg.content else ""
                content_parts.append(f"{getattr(msg, 'type', 'unknown')}:{content}")
            else:
                content_parts.append(str(msg)[:max_len])
        content_str = "\n".join(content_parts)
        raw_key = f"{provider}:{model}:{content_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _should_skip_cache(self, messages: list) -> bool:
        """判断是否应该跳过缓存。"""
        # 检查消息中是否包含敏感关键词
        skip_keywords = [
            "密码", "password", "token", "secret", "api_key",
            "身份证", "手机号", "信用卡", "cvv",
        ]
        for msg in messages:
            content = getattr(msg, "content", "")
            if any(kw in content.lower() for kw in skip_keywords):
                return True
        return False

    def get(self, messages: list, provider: str, model: str) -> Optional[str]:
        """从缓存获取响应。"""
        if not self.enabled:
            return None
        if self._should_skip_cache(messages):
            return None

        key_hash = self._get_cache_key(messages, provider, model)

        with _lock:
            conn = _get_conn()
            # 周期性清理过期条目（每 CLEANUP_INTERVAL 次 get 执行一次），
            # 避免每次读取都触发写操作，提升并发性能
            self._get_count += 1
            if self._get_count % self._CLEANUP_INTERVAL == 0:
                conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (time.time(),))
                conn.commit()

            cursor = conn.execute(
                """SELECT response, hit_count FROM cache_entries
                   WHERE key_hash = ? AND expires_at > ?""",
                (key_hash, time.time())
            )
            row = cursor.fetchone()
            if row:
                # 更新命中计数
                conn.execute(
                    "UPDATE cache_entries SET hit_count = ? WHERE key_hash = ?",
                    (row["hit_count"] + 1, key_hash)
                )
                conn.commit()
                logger.debug(f"Cache hit: {key_hash[:16]}... (hits={row['hit_count'] + 1})")
                return row["response"]
            return None

    def set(self, messages: list, provider: str, model: str, response: str):
        """将响应写入缓存。"""
        if not self.enabled:
            return
        if self._should_skip_cache(messages):
            return
        if not response or len(response) < 5:
            return  # 太短不缓存（简单数字回答不需要缓存）

        key_hash = self._get_cache_key(messages, provider, model)
        now = time.time()
        expires = now + self.ttl

        # 生成消息预览（前 100 字符）
        preview = ""
        for msg in messages:
            content = getattr(msg, "content", "")[:50]
            preview += content + "; "
            if len(preview) > 100:
                break
        preview = preview[:100]

        with _lock:
            conn = _get_conn()
            conn.execute(
                """INSERT INTO cache_entries
                   (key_hash, provider, model, messages_preview, response, created_at, expires_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(key_hash) DO UPDATE SET
                   response=excluded.response,
                   created_at=excluded.created_at,
                   expires_at=excluded.expires_at,
                   hit_count=0""",
                (key_hash, provider, model, preview, response, now, expires)
            )
            conn.commit()
            logger.debug(f"Cache set: {key_hash[:16]}...")

    def invalidate(self, messages: list, provider: str, model: str) -> bool:
        """使指定缓存失效。"""
        key_hash = self._get_cache_key(messages, provider, model)
        with _lock:
            conn = _get_conn()
            cursor = conn.execute("DELETE FROM cache_entries WHERE key_hash = ?", (key_hash,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self):
        """清空所有缓存。"""
        with _lock:
            conn = _get_conn()
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """获取缓存统计。"""
        with _lock:
            conn = _get_conn()
            total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?", (time.time(),)
            ).fetchone()[0]
            total_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM cache_entries"
            ).fetchone()[0]
            db_size = os.path.getsize(_DB_PATH) if os.path.exists(_DB_PATH) else 0
            return {
                "enabled": self.enabled,
                "ttl_hours": self.ttl / 3600,
                "total_entries": total,
                "expired_entries": expired,
                "total_hits": total_hits,
                "db_size_bytes": db_size,
            }


# 全局缓存实例
_cache_instance: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """获取全局缓存实例。"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance


def configure_cache(enabled: bool = True, ttl_hours: int = 24):
    """配置缓存参数。"""
    global _cache_instance
    _cache_instance = ResponseCache(
        ttl_seconds=ttl_hours * 3600,
        enabled=enabled
    )
    logger.info(f"Cache configured: enabled={enabled}, ttl={ttl_hours}h")
