"""SQLite 数据库连接工具 - 统一连接逻辑，消除重复"""
import os
import sqlite3
import threading
import time


_THREAD_LOCAL = threading.local()
# 线程本地连接最大存活时间（秒），超时后自动重建，避免连接无限累积
_CONN_MAX_AGE_S = 300


def get_sqlite_conn(
    db_path: str,
    *,
    enable_wal: bool = True,
    enable_foreign_keys: bool = False,
    use_thread_local: bool = False,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """获取 SQLite 数据库连接。

    Args:
        db_path: 数据库文件路径（目录会自动创建）
        enable_wal: 是否启用 WAL 模式（提升并发性能）
        enable_foreign_keys: 是否启用外键约束
        use_thread_local: 是否使用线程本地连接缓存
        check_same_thread: 是否检查同一线程

    Returns:
        sqlite3.Connection 对象
    """
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    if use_thread_local:
        conn_key = f"conn_{db_path}"
        ts_key = f"conn_ts_{db_path}"
        conn = getattr(_THREAD_LOCAL, conn_key, None)
        ts = getattr(_THREAD_LOCAL, ts_key, 0)
        now = time.time()
        if conn is not None and (now - ts) < _CONN_MAX_AGE_S:
            return conn
        # 连接过期或不存在：关闭旧连接（如果存在）
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            setattr(_THREAD_LOCAL, conn_key, None)

    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row

    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")
    if enable_wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    if use_thread_local:
        setattr(_THREAD_LOCAL, conn_key, conn)
        setattr(_THREAD_LOCAL, ts_key, time.time())

    return conn


def close_thread_local_conn(db_path: str) -> None:
    """关闭指定 db_path 的线程本地连接（用于显式清理）。"""
    conn_key = f"conn_{db_path}"
    ts_key = f"conn_ts_{db_path}"
    conn = getattr(_THREAD_LOCAL, conn_key, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        setattr(_THREAD_LOCAL, conn_key, None)
        setattr(_THREAD_LOCAL, ts_key, 0)
