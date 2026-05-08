"""In-memory cache implementation."""

import time
from collections import OrderedDict
from typing import Any

from hot_and_cold_memory.core.config import get_settings

from .base import BaseCache

# 默认上限,可被 settings.MEMORY_CACHE_MAX_ITEMS 覆盖
_DEFAULT_MAX_ITEMS = 5000


class MemoryCache(BaseCache):
    """Simple in-memory cache with TTL + LRU eviction.

    没有 LRU 上限的字典在长会话/高 QPS 下会无界增长。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._max_items = getattr(self.settings, "MEMORY_CACHE_MAX_ITEMS", _DEFAULT_MAX_ITEMS)
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._expires: dict[str, float] = {}

    async def get(self, key: str) -> Any | None:
        """Get value from cache. 访问触发 LRU move-to-end。"""
        if key in self._expires and time.time() > self._expires[key]:
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return None
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache。超过上限时按 LRU 淘汰最早访问项。"""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if ttl is not None:
            self._expires[key] = time.time() + ttl
        else:
            self._expires[key] = time.time() + self.settings.CACHE_TTL_SECONDS
        # 容量上限
        while len(self._data) > self._max_items:
            old_key, _ = self._data.popitem(last=False)
            self._expires.pop(old_key, None)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        existed = key in self._data
        self._data.pop(key, None)
        self._expires.pop(key, None)
        return existed

    async def exists(self, key: str) -> bool:
        """Check if key exists and not expired."""
        if key in self._expires and time.time() > self._expires[key]:
            return False
        return key in self._data

    async def flush(self) -> None:
        """Clear all cached data."""
        self._data.clear()
        self._expires.clear()
