"""RAG (检索增强生成) - 基于 OpenAI 兼容 Embedding API 的向量检索。

使用可插拔向量存储后端（numpy 或 chroma），无需额外向量数据库依赖。
支持文档分块、嵌入、检索。
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入 numpy
from core.vector_store import HAS_NUMPY, get_vector_store

if HAS_NUMPY:
    import numpy as np


# 全局向量存储实例（懒加载）
_vector_store = None
_vs_lock = threading.Lock()


def _get_store():
    """获取向量存储实例（懒加载 + 线程安全）。"""
    global _vector_store
    if _vector_store is None:
        with _vs_lock:
            if _vector_store is None:
                _vector_store = get_vector_store()
    return _vector_store


def reset_store(backend: str = None, persist_path: str = None):
    """重置向量存储实例（切换后端时使用）。"""
    global _vector_store
    with _vs_lock:
        _vector_store = get_vector_store(backend=backend, persist_path=persist_path)
    logger.info(f"Vector store reset to backend={backend or 'default'}")


_EMBEDDING_MAX_CHARS = 8000


def get_embedding(text: str, model: str = "text-embedding-3-small") -> Any | None:
    """获取文本的 embedding 向量。

    使用 OpenAI 兼容的 embedding API。
    如果提供商不支持 embedding，返回 None。
    """
    if not HAS_NUMPY:
        logger.warning("numpy not available, skipping embedding")
        return None

    try:
        import requests
        from core.config import get_api_key, get_base_url, get_provider

        base_url = get_base_url()
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # 安全截断：避免在多字节字符中间截断
        truncated = text[:_EMBEDDING_MAX_CHARS]
        # 如果截断后的字符串不是合法的 UTF-8 结尾（比如截断了多字节字符），回退到更短的位置
        while truncated:
            try:
                truncated.encode("utf-8")
                break
            except UnicodeEncodeError:
                truncated = truncated[:-1]

        body = {
            "model": model,
            "input": truncated,
        }
        resp = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json=body,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            vector = data["data"][0]["embedding"]
            return np.array(vector, dtype=np.float32)
        else:
            logger.warning(f"Embedding API error: {resp.status_code} - {resp.text[:200]}")
            return None
    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        logger.warning(f"Embedding request failed: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning(f"Embedding response parsing failed: {e}")
        return None


def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """将文本分割成重叠的块"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 尽量在句子边界截断
        if end < len(text):
            for sep in ["\n\n", "\n", "。", ". ", "! ", "? "]:
                pos = text.rfind(sep, start, end)
                if pos > start + chunk_size // 2:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end
    return [c for c in chunks if c]


def add_document(text: str, source: str = "", chunk_size: int = 500) -> int:
    """添加文档到知识库。

    参数:
        text: 文档全文
        source: 文档来源标识（如文件名）
        chunk_size: 分块大小

    返回:
        添加的块数
    """
    if not HAS_NUMPY:
        logger.warning("RAG disabled: numpy not installed")
        return 0

    store = _get_store()
    chunks = _split_text(text, chunk_size)
    count = 0
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        if vector is not None:
            store.add(
                vector=vector,
                text=chunk,
                metadata={"source": source, "chunk_index": i, "total_chunks": len(chunks), "chunk_id": f"{source}_{i}"},
                auto_save=False,
            )
            count += 1
    store.save()
    logger.info(f"Added {count}/{len(chunks)} chunks from {source} to RAG")
    return count


def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索知识库并返回格式化结果。

    参数:
        query: 查询文本
        top_k: 返回的最大结果数

    返回:
        格式化的检索结果文本（供 LLM 使用）
    """
    if not HAS_NUMPY:
        return "[RAG 功能未启用：请安装 numpy]"

    store = _get_store()
    if store.count() == 0:
        return "[知识库为空，请先上传文档]"

    query_vector = get_embedding(query)
    if query_vector is None:
        return "[无法获取查询向量，Embedding API 可能不可用]"

    results = store.search(query_vector, top_k=top_k)
    if not results:
        return "[未找到相关知识]"

    lines = ["## 知识库检索结果\n"]
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "未知来源")
        score = r.get("score", 0)
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
        lines.append(f"**[{i}]** （来源: {source}，相关度: {score_str}）")
        lines.append(r["text"])
        lines.append("")

    return "\n".join(lines)


def get_knowledge_stats() -> dict[str, Any]:
    """获取知识库统计信息"""
    store = _get_store()
    from core.vector_store import list_backends, _get_backend_from_config
    return {
        "total_chunks": store.count(),
        "enabled": HAS_NUMPY,
        "current_backend": _get_backend_from_config(),
        "available_backends": list_backends(),
    }


def clear_knowledge():
    """清空知识库"""
    store = _get_store()
    store.clear()
    logger.info("Knowledge base cleared")


def list_documents() -> list[dict]:
    """列出知识库中的所有文档（按来源分组）。"""
    store = _get_store()
    if hasattr(store, "list_documents"):
        return store.list_documents()
    return []


def delete_document_by_source(source: str) -> int:
    """删除指定来源的所有文档块。

    Returns:
        删除的块数
    """
    store = _get_store()
    if hasattr(store, "delete_by_source"):
        count = store.delete_by_source(source)
        logger.info(f"Deleted {count} chunks from source '{source}'")
        return count
    return 0
