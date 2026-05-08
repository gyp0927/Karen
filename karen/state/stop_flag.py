"""按 Socket 隔离的停止标志，用于异步架构中中断生成。"""

import logging
import threading

logger = logging.getLogger(__name__)

# 按 socket sid 存储的停止事件字典
_stop_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def set_stop(sid: str):
    """设置指定 socket 的停止标志。"""
    with _lock:
        _stop_events.setdefault(sid, threading.Event()).set()
    logger.debug(f"Stop flag set for sid={sid}")


def clear_stop(sid: str):
    """清除指定 socket 的停止标志。"""
    with _lock:
        event = _stop_events.get(sid)
        if event:
            event.clear()
        else:
            _stop_events[sid] = threading.Event()
    logger.debug(f"Stop flag cleared for sid={sid}")


def is_stopped(sid: str | None = None) -> bool:
    """检查指定 socket 是否已请求停止。

    如果 sid 为 None，检查是否有任何 socket 请求了停止（兼容旧用法）。
    """
    if sid is None:
        with _lock:
            return any(e.is_set() for e in _stop_events.values())
    with _lock:
        return _stop_events.get(sid, threading.Event()).is_set()


def cleanup_sid(sid: str):
    """清理指定 socket 的停止标志，避免内存泄漏。"""
    with _lock:
        _stop_events.pop(sid, None)
    logger.debug(f"Stop flag cleaned up for sid={sid}")
