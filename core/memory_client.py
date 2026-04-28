"""Agent memory system client - direct import from hot-and-cold-memory.

This module wraps the self-organizing knowledge base (adaptive memory system)
for use within the multi-agent chat system. All storage happens locally
(Qdrant file-based vector store + SQLite metadata store).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the memory system's src directory is on sys.path so that its
# package ``hot_and_cold_memory`` can be imported directly.
# ---------------------------------------------------------------------------
_MEMORY_SYS_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Self-organizing Knowledge Base",
        "src",
    )
)
if _MEMORY_SYS_SRC not in sys.path:
    sys.path.insert(0, _MEMORY_SYS_SRC)

# The memory system expects DOCUMENT_STORE_PATH but the Settings class only
# defines MEMORY_STORE_PATH. We set a fallback env var before importing so
# that Pydantic Settings picks it up.
if not os.environ.get("DOCUMENT_STORE_PATH"):
    os.environ["DOCUMENT_STORE_PATH"] = os.environ.get(
        "MEMORY_STORE_PATH", "./data/memories"
    )

# Memory system imports (may fail if deps are missing – guarded in __init__)
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
except Exception as _import_err:  # pragma: no cover
    _MEMORY_SYSTEM_AVAILABLE = False
    _import_err = _import_err

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton storage – one global store per process
# ---------------------------------------------------------------------------
_store_instance: AgentMemoryStore | None = None


def get_memory_store() -> AgentMemoryStore:
    """Return the global memory store singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = AgentMemoryStore()
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
    ) -> list[dict[str, Any]]:
        """Retrieve semantically relevant memories.

        Args:
            query: The user's question / search text.
            top_k: Maximum number of memories to return.
            user_id: Optional user ID for source filtering (not yet enforced).

        Returns:
            List of memory dicts with keys: memory_id, content, score, tier,
            memory_type, frequency_score.
        """
        if not self._initialized:
            await self.initialize()

        retriever: UnifiedRetriever = self._services["retriever"]
        result = await retriever.query(query_text=query, top_k=top_k)

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
