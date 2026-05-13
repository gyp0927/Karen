"""向量存储 - 统一入口，支持 numpy 和 chroma 两种后端。

使用方式：
    from core.vector_store import get_vector_store
    store = get_vector_store()  # 默认 numpy 后端
"""

import abc
import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ========== 配置 ==========

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STORE_DIR = os.path.join(_PROJECT_ROOT, "data")
_STORE_PATH = os.path.join(_STORE_DIR, "rag_store.json")
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "state", "rag_config.json")
_DEFAULT_CHROMA_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma_db")


# ========== 抽象基类 ==========

class VectorStoreBackend(abc.ABC):
    """向量存储后端抽象基类。"""

    @abc.abstractmethod
    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        """添加向量到存储。"""
        pass

    @abc.abstractmethod
    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        """搜索最相似的向量。"""
        pass

    @abc.abstractmethod
    def clear(self):
        """清空所有数据。"""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """返回存储的向量数量。"""
        pass

    @abc.abstractmethod
    def save(self):
        """保存数据到磁盘。"""
        pass


# ========== Numpy 后端 ==========

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class NumpyBackend(VectorStoreBackend):
    """基于 numpy 的轻量级向量存储后端（默认）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._init_store()
                cls._instance = instance
        return cls._instance

    def _init_store(self):
        self.vectors: list[Any] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self._lock = threading.RLock()
        self._dirty = False
        self._load()

    def _load(self):
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
        except Exception:
            logger.exception("Failed to save RAG store")

    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        with self._lock:
            if self.vectors and vector.shape != self.vectors[0].shape:
                logger.warning(f"Vector dimension mismatch: expected {self.vectors[0].shape}, got {vector.shape}")
                return
            self.vectors.append(vector)
            self.texts.append(text)
            self.metadatas.append(metadata or {})
            self._dirty = True
            if auto_save:
                self._save()

    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        with self._lock:
            if not self.vectors:
                return []
            if query_vector.shape != self.vectors[0].shape:
                logger.warning(f"Query vector dimension mismatch")
                return []
            vectors = np.stack(self.vectors)
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(vectors_norm, query_norm)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [
                {"text": self.texts[idx], "metadata": self.metadatas[idx], "score": float(similarities[idx])}
                for idx in top_indices
            ]

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
        with self._lock:
            sources = {}
            for meta in self.metadatas:
                source = meta.get("source", "未知来源")
                sources[source] = sources.get(source, 0) + 1
            return [{"source": s, "chunks": c} for s, c in sorted(sources.items())]

    def delete_by_source(self, source: str) -> int:
        with self._lock:
            indices = [i for i, meta in enumerate(self.metadatas) if meta.get("source") == source]
            for idx in sorted(indices, reverse=True):
                self.vectors.pop(idx)
                self.texts.pop(idx)
                self.metadatas.pop(idx)
            if indices:
                self._dirty = True
                self._save()
            return len(indices)


# ========== Chroma 后端 ==========

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    chromadb = None  # type: ignore


class ChromaBackend(VectorStoreBackend):
    """基于 ChromaDB 的持久化向量存储后端"""

    def __init__(self, persist_path: str = None):
        if not HAS_CHROMA:
            raise ImportError("ChromaDB 未安装。请运行: pip install chromadb")
        self.persist_path = persist_path or _DEFAULT_CHROMA_PATH
        os.makedirs(self.persist_path, exist_ok=True)
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_path,
            anonymized_telemetry=False,
        ))
        self.collection = self.client.get_or_create_collection("rag_documents")
        logger.info(f"Chroma backend initialized at {self.persist_path}")

    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        import uuid
        meta = metadata or {}
        self.collection.add(
            embeddings=[vector.tolist() if hasattr(vector, "tolist") else list(vector)],
            documents=[text],
            metadatas=[meta],
            ids=[meta.get("chunk_id", str(uuid.uuid4()))],
        )

    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        results = self.collection.query(query_embeddings=[vector_list], n_results=top_k)
        output = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                output.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": float(results["distances"][0][i]) if results["distances"] else 0.0,
                })
        return output

    def clear(self):
        self.client.delete_collection("rag_documents")
        self.collection = self.client.get_or_create_collection("rag_documents")

    def count(self) -> int:
        return self.collection.count()

    def save(self):
        pass

    def list_documents(self) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.get(limit=count, include=["metadatas"])
        sources = {}
        for meta in results.get("metadatas", []):
            if meta:
                source = meta.get("source", "未知来源")
                sources[source] = sources.get(source, 0) + 1
        return [{"source": s, "chunks": c} for s, c in sorted(sources.items())]

    def delete_by_source(self, source: str) -> int:
        count = self.collection.count()
        if count == 0:
            return 0
        results = self.collection.get(limit=count, include=["metadatas"])
        ids_to_remove = []
        for i, meta in enumerate(results.get("metadatas", [])):
            if meta and meta.get("source") == source:
                doc_id = results.get("ids", [])[i] if i < len(results.get("ids", [])) else None
                if doc_id:
                    ids_to_remove.append(doc_id)
        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
        return len(ids_to_remove)


# ========== 工厂函数 ==========

def _get_backend_from_config() -> str:
    try:
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("backend", "numpy")
    except Exception:
        pass
    return "numpy"


def _save_backend_config(backend: str, **kwargs):
    try:
        data = {"backend": backend}
        data.update(kwargs)
        os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save RAG backend config: {e}")


def get_vector_store(backend: str = None, persist_path: str = None):
    """获取向量存储后端实例。"""
    if backend is None:
        backend = _get_backend_from_config()
    backend = backend.lower()
    if backend == "chroma":
        if not HAS_CHROMA:
            logger.warning("ChromaDB 未安装，回退到 numpy 后端")
            return NumpyBackend()
        try:
            return ChromaBackend(persist_path=persist_path)
        except Exception as e:
            logger.warning(f"Chroma backend failed: {e}, falling back to numpy")
            return NumpyBackend()
    if not HAS_NUMPY:
        logger.warning("numpy 未安装，RAG 功能不可用")
    return NumpyBackend()


def set_backend(backend: str, persist_path: str = None) -> bool:
    """切换向量存储后端。"""
    backend = backend.lower()
    if backend == "chroma" and not HAS_CHROMA:
        return False
    _save_backend_config(backend, persist_path=persist_path)
    return True


def list_backends() -> list[dict]:
    """列出可用的后端及其状态。"""
    return [
        {"name": "numpy", "available": HAS_NUMPY, "description": "基于 numpy 的轻量级实现（默认）"},
        {"name": "chroma", "available": HAS_CHROMA, "description": "基于 ChromaDB 的持久化实现"},
    ]


def list_documents() -> list[dict]:
    """列出知识库中的所有文档（按来源分组）。"""
    store = get_vector_store()
    if hasattr(store, "list_documents"):
        return store.list_documents()
    return []


def delete_by_source(source: str) -> int:
    """删除指定来源的所有文档块。"""
    store = get_vector_store()
    if hasattr(store, "delete_by_source"):
        return store.delete_by_source(source)
    return 0
