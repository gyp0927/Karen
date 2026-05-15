"""Web 层通用工具函数

无 Flask/SocketIO 依赖的纯工具函数，可在任何上下文安全使用。
"""
import asyncio
import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)

_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
os.makedirs(_GENERATED_DIR, exist_ok=True)


def run_async_in_thread(coro):
    """在线程中安全运行异步协程。优先使用 asyncio.run()，
    若检测到已有事件循环在运行，则在新线程中创建独立循环执行。
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):

            def _run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_new_loop)
                return future.result()
        raise
