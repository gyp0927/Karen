"""API 调用统计 - 记录每次 LLM 调用的 token、耗时和费用。"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DB_DIR, "stats.db")
_lock = threading.RLock()

# 各提供商大致价格（每 1K tokens，USD）- 用于估算
_PRICE_PER_1K = {
    "openai": {"input": 0.005, "output": 0.015},      # gpt-4o
    "anthropic": {"input": 0.003, "output": 0.015},   # claude-sonnet
    "deepseek": {"input": 0.0015, "output": 0.006},
    "qwen": {"input": 0.001, "output": 0.002},
    "kimi": {"input": 0.001, "output": 0.002},
    "kimi-code": {"input": 0.001, "output": 0.002},
    "glm": {"input": 0.001, "output": 0.001},
    "ollama": {"input": 0, "output": 0},              # 本地免费
    "default": {"input": 0.005, "output": 0.015},
}


def _get_conn() -> sqlite3.Connection:
    from core.db_utils import get_sqlite_conn
    # 线程本地持久连接:每次 record_call 都新建 SQLite 连接会重复执行
    # PRAGMA journal_mode=WAL/synchronous=NORMAL,密集请求下成为瓶颈。
    return get_sqlite_conn(_DB_PATH, use_thread_local=True)


def init_db():
    """初始化统计数据库"""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                agent_name TEXT,
                session_id TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0,
                status TEXT DEFAULT 'success'  -- 'success' | 'error' | 'stopped'
            );
            CREATE INDEX IF NOT EXISTS idx_calls_time ON api_calls(timestamp);
            CREATE INDEX IF NOT EXISTS idx_calls_session ON api_calls(session_id);
        """)
        conn.commit()


@dataclass
class CallRecord:
    timestamp: float
    provider: str
    model: str
    agent_name: str = ""
    session_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "success"


def record_call(record: CallRecord):
    """记录一次 API 调用"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO api_calls
               (timestamp, provider, model, agent_name, session_id,
                prompt_tokens, completion_tokens, total_tokens,
                duration_ms, estimated_cost_usd, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.timestamp, record.provider, record.model,
             record.agent_name, record.session_id,
             record.prompt_tokens, record.completion_tokens, record.total_tokens,
             record.duration_ms, record.estimated_cost_usd, record.status)
        )
        conn.commit()


def estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算调用费用（USD）"""
    prices = _PRICE_PER_1K.get(provider, _PRICE_PER_1K["default"])
    input_cost = (prompt_tokens / 1000) * prices["input"]
    output_cost = (completion_tokens / 1000) * prices["output"]
    return round(input_cost + output_cost, 6)


def get_stats_summary(days: int = 7) -> dict:
    """获取最近 N 天的统计摘要"""
    with _lock:
        conn = _get_conn()
        since = time.time() - days * 86400
        cursor = conn.execute(
            """SELECT
                COUNT(*) as call_count,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                AVG(duration_ms) as avg_duration,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
               FROM api_calls WHERE timestamp > ?""",
            (since,)
        )
        row = cursor.fetchone()
        return {
            "call_count": row["call_count"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "total_cost_usd": round(row["total_cost"] or 0, 4),
            "avg_duration_ms": round(row["avg_duration"] or 0, 2),
            "success_rate": round((row["success_count"] or 0) / max(row["call_count"], 1) * 100, 1),
            "period_days": days,
        }


def get_daily_stats(days: int = 7) -> list[dict]:
    """获取每日统计（用于图表）"""
    with _lock:
        conn = _get_conn()
        since = time.time() - days * 86400
        cursor = conn.execute(
            """SELECT
                date(datetime(timestamp, 'unixepoch')) as day,
                COUNT(*) as calls,
                SUM(total_tokens) as tokens,
                SUM(estimated_cost_usd) as cost
               FROM api_calls WHERE timestamp > ?
               GROUP BY day ORDER BY day DESC""",
            (since,)
        )
        return [
            {"day": row["day"], "calls": row["calls"], "tokens": row["tokens"], "cost": row["cost"]}
            for row in cursor.fetchall()
        ]


# 初始化数据库
init_db()
