"""SQLite 数据库连接工具 - 统一连接逻辑，消除重复"""
import os
import sqlite3
import threading


_THREAD_LOCAL = threading.local()


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
        conn = getattr(_THREAD_LOCAL, f"conn_{db_path}", None)
        if conn is not None:
            return conn

    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row

    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")
    if enable_wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    if use_thread_local:
        setattr(_THREAD_LOCAL, f"conn_{db_path}", conn)

    return conn
