"""SQLite 持久化层 - 存储会话和消息历史。"""

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "chat_history.db")
_lock = threading.RLock()


# 线程本地连接缓存，减少重复打开/关闭
_thread_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（每个线程缓存一个连接）。"""
    from core.db_utils import get_sqlite_conn
    return get_sqlite_conn(
        _DB_PATH,
        enable_wal=True,
        enable_foreign_keys=True,
        use_thread_local=True,
    )


def init_db():
    """初始化数据库表结构"""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'human' | 'assistant' | 'system'
                content TEXT NOT NULL,
                agent_name TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        """)
        # 添加 user_id 列（向后兼容）
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        # 清理之前外键未启用时遗留的孤立消息
        cleanup_orphaned_messages()
        logger.info(f"Database initialized at {_DB_PATH}")


def save_session(session_id: str, title: str, created_at: float, updated_at: float, user_id: str = ""):
    """保存或更新会话"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO sessions (id, title, created_at, updated_at, user_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
               title=excluded.title, updated_at=excluded.updated_at, user_id=excluded.user_id""",
            (session_id, title, created_at, updated_at, user_id)
        )
        conn.commit()


def save_message(session_id: str, role: str, content: str, agent_name: str | None = None, user_id: str = ""):
    """保存消息"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO messages (session_id, role, content, agent_name, created_at, user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, agent_name, time.time(), user_id)
        )
        conn.commit()


def load_sessions(user_id: str) -> list[dict]:
    """加载所有会话。当 user_id 为空时加载所有会话（兼容未开启认证的场景）。"""
    with _lock:
        conn = _get_conn()
        if user_id:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,)
            )
        else:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


def load_messages(session_id: str, user_id: str) -> list[dict]:
    """加载指定会话的所有消息。当 user_id 为空时不按 user_id 过滤。"""
    with _lock:
        conn = _get_conn()
        if user_id:
            cursor = conn.execute(
                """SELECT role, content, agent_name, created_at FROM messages
                   WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC""",
                (session_id, user_id)
            )
        else:
            cursor = conn.execute(
                """SELECT role, content, agent_name, created_at FROM messages
                   WHERE session_id = ? ORDER BY created_at ASC""",
                (session_id,)
            )
        rows = cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "agent_name": row["agent_name"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def delete_session(session_id: str, user_id: str):
    """删除会话及其消息（外键级联删除）。当 user_id 为空时不按 user_id 过滤。"""
    with _lock:
        conn = _get_conn()
        if user_id:
            conn.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        else:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        logger.info(f"Deleted session from DB: {session_id}")


def update_session_title(session_id: str, title: str, updated_at: float, user_id: str):
    """更新会话标题。当 user_id 为空时不按 user_id 过滤。"""
    with _lock:
        conn = _get_conn()
        if user_id:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, updated_at, session_id, user_id)
            )
        else:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, updated_at, session_id)
            )
        conn.commit()


def cleanup_orphaned_messages() -> int:
    """清理没有对应会话的孤立消息（修复外键未启用时遗留的数据）。"""
    with _lock:
        conn = _get_conn()
        cursor = conn.execute(
            """DELETE FROM messages
               WHERE session_id NOT IN (SELECT id FROM sessions)"""
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} orphaned messages")
        return deleted


def get_db_stats() -> dict:
    """获取数据库统计信息"""
    with _lock:
        conn = _get_conn()
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        orphaned = conn.execute(
            """SELECT COUNT(*) FROM messages
               WHERE session_id NOT IN (SELECT id FROM sessions)"""
        ).fetchone()[0]
        db_size = os.path.getsize(_DB_PATH) if os.path.exists(_DB_PATH) else 0
        return {
            "session_count": session_count,
            "message_count": message_count,
            "orphaned_messages": orphaned,
            "db_size_bytes": db_size,
            "db_path": _DB_PATH,
        }
