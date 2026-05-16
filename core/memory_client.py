"""Agent memory system client - direct import from hot-and-cold-memory.

This module wraps the self-organizing knowledge base (adaptive memory system)
for use within the multi-agent chat system. All storage happens locally
(Qdrant file-based vector store + SQLite metadata store).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDER_WARMUP_TIMEOUT_S = 60.0

# The memory system expects DOCUMENT_STORE_PATH but the Settings class only
# defines MEMORY_STORE_PATH. We set a fallback env var before importing so
# that Pydantic Settings picks it up.
if not os.environ.get("DOCUMENT_STORE_PATH"):
    os.environ["DOCUMENT_STORE_PATH"] = os.environ.get(
        "MEMORY_STORE_PATH", "./data/memories"
    )

# Memory system imports (may fail if deps are missing – guarded in __init__)
_import_err: Exception | None = None
try:
    from hot_and_cold_memory.core.config import get_settings
    from hot_and_cold_memory.core.logging import setup_logging
    from hot_and_cold_memory.frequency.tracker import FrequencyTracker
    from hot_and_cold_memory.ingestion.embedder import Embedder
    from hot_and_cold_memory.ingestion.pipeline import MemoryPipeline
    from hot_and_cold_memory.migration.engine import MigrationEngine
    from hot_and_cold_memory.retrieval.retriever import UnifiedRetriever
    from hot_and_cold_memory.storage.cache.memory_cache import MemoryCache
    from hot_and_cold_memory.storage.document_store.local_store import LocalDocumentStore
    from hot_and_cold_memory.storage.metadata_store.postgres_store import PostgresMetadataStore
    from hot_and_cold_memory.storage.vector_store.local_qdrant_store import LocalQdrantStore
    from hot_and_cold_memory.tiers.cold_tier import ColdTier
    from hot_and_cold_memory.tiers.compression import CompressionEngine
    from hot_and_cold_memory.tiers.hot_tier import HotTier

    _MEMORY_SYSTEM_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _MEMORY_SYSTEM_AVAILABLE = False
    _import_err = _e
    logger.debug(f"Memory system import failed: {_e}")

# ---------------------------------------------------------------------------
# SQLite fallback memory store (当完整记忆系统依赖缺失时降级使用)
# ---------------------------------------------------------------------------
import sqlite3 as _sqlite3
import uuid as _uuid
import time as _time


class _SQLiteMemoryFallback:
    """纯 SQLite 记忆降级存储。无向量检索，仅做文本子串匹配。

    接口与 AgentMemoryStore 保持一致，便于调用方无感知切换。
    """

    _DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memories_fallback.db")
    _INIT_SQL = """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type TEXT DEFAULT 'observation',
            source TEXT DEFAULT '',
            importance REAL DEFAULT 0.5,
            tags TEXT DEFAULT '[]',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
        CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
    """

    def __init__(self) -> None:
        import threading
        os.makedirs(os.path.dirname(self._DB_PATH), exist_ok=True)
        self._lock = threading.RLock()
        with self._lock:
            conn = _sqlite3.connect(self._DB_PATH, check_same_thread=False)
            conn.executescript(self._INIT_SQL)
            conn.close()
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        pass

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "",
        source: str = "",
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = _sqlite3.connect(self._DB_PATH, check_same_thread=False)
            conn.row_factory = _sqlite3.Row
            try:
                # 简单文本匹配：关键词拆成 LIKE 子句
                words = [w for w in query.split() if len(w) >= 2]
                if not words:
                    words = [query]
                words = words[:20]
                sql = "SELECT * FROM memories WHERE 1=1"
                params: list[Any] = []
                if source:
                    sql += " AND source = ?"
                    params.append(source)
                for w in words:
                    sql += " AND content LIKE ?"
                    params.append(f"%{w}%")
                sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
                params.append(top_k)
                rows = conn.execute(sql, params).fetchall()
                return [
                    {
                        "memory_id": row["memory_id"],
                        "content": row["content"],
                        "score": 0.5,
                        "tier": "cold",
                        "memory_type": row["memory_type"],
                        "frequency_score": 0.0,
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    async def save_memory(
        self,
        content: str,
        memory_type: str = "observation",
        source: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            conn = _sqlite3.connect(self._DB_PATH, check_same_thread=False)
            try:
                mid = str(_uuid.uuid4())
                conn.execute(
                    "INSERT INTO memories (memory_id, content, memory_type, source, importance, tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (mid, content, memory_type, source, importance, json.dumps(tags or []), _time.time()),
                )
                conn.commit()
                return {"memory_id": mid, "status": "created", "tier": "cold"}
            finally:
                conn.close()

    async def save_memories_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.save_memory(**item) for item in items]

    async def delete_session_memories(self, session_id: str) -> int:
        if not session_id:
            return 0
        with self._lock:
            conn = _sqlite3.connect(self._DB_PATH, check_same_thread=False)
            try:
                cursor = conn.execute("DELETE FROM memories WHERE source = ?", (session_id,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    @staticmethod
    def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = ["[相关记忆 / Relevant Memories]"]
        for i, mem in enumerate(memories, 1):
            tier_tag = f"[{mem['tier']}]" if mem.get("tier") else ""
            lines.append(f"{i}. {mem['content']} {tier_tag}".strip())
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            conn = _sqlite3.connect(self._DB_PATH, check_same_thread=False)
            try:
                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                return {"initialized": True, "backend": "sqlite_fallback", "total_memories": total}
            finally:
                conn.close()

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Singleton storage – one global store per process
# ---------------------------------------------------------------------------
_store_instance: AgentMemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store():
    """Return the global memory store singleton.

    当完整记忆系统依赖缺失时，自动降级到 SQLite fallback。
    """
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                if _MEMORY_SYSTEM_AVAILABLE:
                    _store_instance = AgentMemoryStore()
                else:
                    logger.warning(
                        "Memory system dependencies missing, falling back to SQLite. "
                        f"Import error: {_import_err}"
                    )
                    _store_instance = _SQLiteMemoryFallback()
    return _store_instance


class AgentMemoryStore:
    """Simplified wrapper around the adaptive memory system.

    Usage (async):
        store = get_memory_store()
        await store.initialize()
        memories = await store.retrieve("user's question", top_k=5)
        await store.save_memory("User likes Python", memory_type="fact")
    """

    def __init__(self) -> None:
        if not _MEMORY_SYSTEM_AVAILABLE:
            raise RuntimeError(
                "Memory system dependencies are not available. "
                f"Import error: {_import_err}"
            )

        self._services: dict[str, Any] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Lazy-init all memory system backend services."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            settings = get_settings()
            setup_logging(settings.LOG_LEVEL)

            # ---- storage backends ----
            vector_store = LocalQdrantStore()
            await vector_store.initialize()
            await vector_store.ensure_collection(f"{settings.VECTOR_DB_COLLECTION}_cold")

            metadata_store = PostgresMetadataStore()
            await metadata_store.initialize()

            document_store = LocalDocumentStore()
            cache = MemoryCache()
            embedder = Embedder()

            # ---- tiers ----
            hot_tier = HotTier(
                vector_store=vector_store,
                metadata_store=metadata_store,
                document_store=document_store,
                cache=cache,
            )

            compression_engine = CompressionEngine()
            cold_tier = ColdTier(
                vector_store=vector_store,
                metadata_store=metadata_store,
                document_store=document_store,
                compression_engine=compression_engine,
                cache=cache,
                embedder=embedder,
            )

            # ---- frequency tracking ----
            frequency_tracker = FrequencyTracker(
                metadata_store=metadata_store,
                vector_store=vector_store,
                embedder=embedder,
            )

            # ---- retrieval ----
            retriever = UnifiedRetriever(
                hot_tier=hot_tier,
                cold_tier=cold_tier,
                frequency_tracker=frequency_tracker,
                embedder=embedder,
            )

            # ---- migration ----
            migration_engine = MigrationEngine(
                hot_tier=hot_tier,
                cold_tier=cold_tier,
                metadata_store=metadata_store,
                embedder=embedder,
            )

            # ---- ingestion pipeline ----
            pipeline = MemoryPipeline(
                metadata_store=metadata_store,
                hot_tier=hot_tier,
                cold_tier=cold_tier,
                embedder=embedder,
                frequency_tracker=frequency_tracker,
                migration_engine=migration_engine,
            )

            self._services = {
                "vector_store": vector_store,
                "metadata_store": metadata_store,
                "document_store": document_store,
                "cache": cache,
                "embedder": embedder,
                "hot_tier": hot_tier,
                "cold_tier": cold_tier,
                "frequency_tracker": frequency_tracker,
                "retriever": retriever,
                "pipeline": pipeline,
                "migration_engine": migration_engine,
            }

            self._initialized = True
            logger.info("Agent memory store initialized successfully")

            # 校验 embedder 维度与 Qdrant 集合配置一致;不一致时切模型(384↔1536)
            # 后续 upsert 会静默写错维度,先 fail-fast 提醒。
            settings = get_settings()
            try:
                actual = await embedder.embed("dim_check")
                if len(actual) != settings.EMBEDDING_DIMENSION:
                    logger.error(
                        "Embedding dimension mismatch: model returns %d, "
                        "EMBEDDING_DIMENSION=%d. 切模型后请清空 ./data/qdrant_storage 并重启。",
                        len(actual), settings.EMBEDDING_DIMENSION,
                    )
            except Exception as e:
                logger.warning(f"Dimension check skipped: {e}")

            # sentence-transformers 首次 embed 调用会同步加载模型（10-20s 网络 + 解压），
            # 不预热则首条用户消息必然超时。失败仅警告，retrieve 时再走 lazy load 兜底。
            try:
                await asyncio.wait_for(embedder.embed("warmup"), timeout=EMBEDDER_WARMUP_TIMEOUT_S)
                logger.info("Embedder warmed up")
            except asyncio.TimeoutError:
                logger.warning(f"Embedder warmup timed out (>{EMBEDDER_WARMUP_TIMEOUT_S}s)")
            except Exception as e:
                logger.warning(f"Embedder warmup failed: {e}")

    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "",
        source: str = "",
    ) -> list[dict[str, Any]]:
        """Retrieve semantically relevant memories.

        Args:
            query: The user's question / search text.
            top_k: Maximum number of memories to return.
            user_id: Optional user ID (kept for back-compat, not used for filtering).
            source: 会话隔离用。传 session_id 时只返该会话的记忆。空串=不过滤。

        Returns:
            List of memory dicts with keys: memory_id, content, score, tier,
            memory_type, frequency_score.
        """
        if not self._initialized:
            await self.initialize()

        retriever: UnifiedRetriever = self._services["retriever"]
        filters = {"source": source} if source else None
        result = await retriever.query(query_text=query, top_k=top_k, filters=filters)

        memories = []
        for chunk in result.chunks:
            memories.append(
                {
                    "memory_id": str(chunk.memory_id),
                    "content": chunk.content,
                    "score": round(chunk.score, 4),
                    "tier": chunk.tier.value,
                    "memory_type": chunk.memory_type,
                    "frequency_score": round(chunk.frequency_score, 4),
                }
            )
        return memories

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def save_memory(
        self,
        content: str,
        memory_type: str = "observation",
        source: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist a new memory into the adaptive store.

        Args:
            content: Raw memory text.
            memory_type: One of observation / fact / reflection / summary.
            source: Source identifier, e.g. ``user_123`` or session ID.
            importance: 0.0 – 1.0 (higher = more important).
            tags: Optional list of tags.

        Returns:
            Dict with memory_id, status, tier.
        """
        if not self._initialized:
            await self.initialize()

        pipeline: MemoryPipeline = self._services["pipeline"]
        result = await pipeline.write_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            importance=importance,
            tags=tags or [],
        )
        return {
            "memory_id": str(result.memory_id),
            "status": result.status,
            "tier": result.tier,
        }

    async def save_memories_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Batch-save memories.

        Args:
            items: List of dicts with keys: content, memory_type, source,
                   importance, tags.

        Returns:
            List of result dicts (same shape as ``save_memory``).
        """
        if not self._initialized:
            await self.initialize()

        pipeline: MemoryPipeline = self._services["pipeline"]
        results = await pipeline.write_memories_batch(items)
        return [
            {
                "memory_id": str(r.memory_id),
                "status": r.status,
                "tier": r.tier,
            }
            for r in results
        ]

    async def delete_session_memories(self, session_id: str) -> int:
        """删除某个会话的全部记忆(hot+cold tier 一起清,不影响其他会话)。

        save_memory 时把 session_id 写到 source 字段,这里按 source 反查删除。
        会话删除路径调用,不阻塞 UI(调用方建议 fire-and-forget)。

        Returns: 删除的记忆条数,会话 ID 为空或不存在记忆时返回 0。
        """
        if not session_id:
            return 0
        if not self._initialized:
            await self.initialize()
        pipeline: MemoryPipeline = self._services["pipeline"]
        deleted = await pipeline.delete_by_source(session_id)
        # 清 retriever 5s TTL cache,否则刚删的记忆仍可能在缓存里返
        if deleted:
            retriever: UnifiedRetriever = self._services["retriever"]
            retriever._cache.clear()
        return deleted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
        """Convert retrieved memories into a prompt prefix for the LLM.

        Returns an empty string when no memories are found.
        """
        if not memories:
            return ""

        lines = ["[相关记忆 / Relevant Memories]"]
        for i, mem in enumerate(memories, 1):
            tier_tag = f"[{mem['tier']}]" if mem.get("tier") else ""
            lines.append(f"{i}. {mem['content']} {tier_tag}".strip())
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Return basic memory system statistics."""
        if not self._initialized:
            return {"initialized": False}

        # TODO: query metadata_store for counts if needed
        return {
            "initialized": True,
            "vector_collection": get_settings().VECTOR_DB_COLLECTION,
        }

    async def shutdown(self) -> None:
        """关闭底层存储客户端。

        Qdrant local 持有 file lock,SQLAlchemy engine 持有连接池;
        进程退出时不主动关闭会触发 __del__ 里的 ImportError(Python 已开始关闭)。
        """
        if not self._initialized:
            return

        async with self._lock:
            # Qdrant 的 close 是同步的
            vs = self._services.get("vector_store")
            client = getattr(vs, "client", None)
            if client is not None:
                try:
                    close = getattr(client, "close", None)
                    if callable(close):
                        await asyncio.to_thread(close)
                except Exception as e:
                    logger.warning(f"Qdrant close failed: {e}")

            # SQLAlchemy AsyncEngine 用 dispose
            ms = self._services.get("metadata_store")
            engine = getattr(ms, "engine", None)
            if engine is not None:
                try:
                    await engine.dispose()
                except Exception as e:
                    logger.warning(f"Metadata store dispose failed: {e}")

            self._services.clear()
            self._initialized = False
            logger.info("Agent memory store shut down")
