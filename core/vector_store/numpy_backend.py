"""numpy 向量存储后端 - 基于 numpy 的轻量级实现（默认）。"""

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入 numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
_STORE_PATH = os.path.join(_STORE_DIR, "rag_store.json")


class NumpyBackend:
    """基于 numpy 的向量存储后端"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_store()
        return cls._instance

    def _init_store(self):
        self.vectors: list[Any] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self._lock = threading.RLock()
        self._dirty = False
        self._load()

    def _load(self):
        """从磁盘加载向量存储"""
        if not HAS_NUMPY or not os.path.exists(_STORE_PATH):
            return
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self.vectors.append(np.array(item["vector"], dtype=np.float32))
                self.texts.append(item["text"])
                self.metadatas.append(item.get("metadata", {}))
            logger.info(f"Loaded {len(self.vectors)} vectors from {_STORE_PATH}")
        except Exception:
            logger.exception("Failed to load RAG store")

    def _save(self):
        """保存向量存储到磁盘"""
        if not HAS_NUMPY:
            return
        try:
            os.makedirs(_STORE_DIR, exist_ok=True)
            data = []
            for i in range(len(self.vectors)):
                data.append({
                    "vector": self.vectors[i].tolist(),
                    "text": self.texts[i],
                    "metadata": self.metadatas[i],
                })
            with open(_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.debug(f"Saved {len(data)} vectors to {_STORE_PATH}")
        except Exception:
            logger.exception("Failed to save RAG store")

    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        with self._lock:
            # 维度一致性校验
            if self.vectors and vector.shape != self.vectors[0].shape:
                logger.warning(
                    f"Vector dimension mismatch: expected {self.vectors[0].shape}, "
                    f"got {vector.shape}. Skipping."
                )
                return
            self.vectors.append(vector)
            self.texts.append(text)
            self.metadatas.append(metadata or {})
            self._dirty = True
            if auto_save:
                self._save()

    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        """余弦相似度检索"""
        with self._lock:
            if not self.vectors:
                return []
            # 维度一致性校验
            if query_vector.shape != self.vectors[0].shape:
                logger.warning(
                    f"Query vector dimension mismatch: stored {self.vectors[0].shape}, "
                    f"query {query_vector.shape}. Returning empty."
                )
                return []
            vectors = np.stack(self.vectors)
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(vectors_norm, query_norm)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(similarities[idx]),
                })
            return results

    def clear(self):
        with self._lock:
            self.vectors.clear()
            self.texts.clear()
            self.metadatas.clear()
            self._dirty = True
            self._save()

    def count(self) -> int:
        with self._lock:
            return len(self.vectors)

    def save(self):
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False

    def list_documents(self) -> list[dict]:
        """按 source 分组列出所有文档。"""
        with self._lock:
            sources = {}
            for meta in self.metadatas:
                source = meta.get("source", "未知来源")
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
            return [{"source": s, "chunks": c} for s, c in sorted(sources.items())]

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的所有文档块。

        Returns:
            删除的块数
        """
        with self._lock:
            indices_to_remove = []
            for i, meta in enumerate(self.metadatas):
                if meta.get("source") == source:
                    indices_to_remove.append(i)

            if not indices_to_remove:
                return 0

            # 从后往前删除，避免索引偏移
            for idx in sorted(indices_to_remove, reverse=True):
                self.vectors.pop(idx)
                self.texts.pop(idx)
                self.metadatas.pop(idx)

            self._dirty = True
            self._save()
            return len(indices_to_remove)
