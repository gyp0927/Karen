"""Socket 状态管理和 Agent 图初始化。

将 socket 级状态（会话、配置、图实例）集中管理，
与 HTTP/SocketIO 事件处理器解耦，便于测试和维护。
"""
import asyncio
import logging
import threading
import time

from agents.llm import clear_streaming_callback, cleanup_llm_config
from agents.nodes import create_agents
from graph.orchestrator import create_fast_graph
from state.manager import SessionManager
from state.stop_flag import set_stop, is_stopped, cleanup_sid

from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE
from core.config import SOCKET_INACTIVE_TIMEOUT
from state.model_config_manager import get_active_config

logger = logging.getLogger(__name__)


class SocketState:
    """每个 Socket 连接的隔离状态"""

    def __init__(self, sid: str):
        self.sid = sid
        self.user_id: str = ""
        self.msg_manager = SessionManager(user_id=self.user_id)
        self.current_base_response: str | None = None
        self.fast_mode = True
        self.review_language = "zh"
        self.detected_language: str | None = None
        self.last_active = time.time()
        # 预启动的搜索任务（和 LLM 回复并行）
        self.pending_search: "asyncio.Task | None" = None

    def touch(self):
        """更新最后活跃时间。"""
        self.last_active = time.time()

    def set_user_id(self, user_id: str):
        """设置用户 ID，如果变化则重新创建 SessionManager。"""
        if self.user_id != user_id:
            self.user_id = user_id
            self.msg_manager = SessionManager(user_id=user_id)

    def reset_session(self):
        """切/新/删会话时重置 per-socket 瞬态字段。

        否则 detected_language 会从前一会话沿用,
        流式回调也可能继续把 token_chunk 打到新会话。
        """
        set_stop(self.sid)
        clear_streaming_callback(self.sid)
        self.current_base_response = None
        self.detected_language = None


# 按 socket sid 存储的隔离状态
socket_states: dict[str, SocketState] = {}
_socket_states_lock = threading.Lock()

# Socket 级配置隔离：key = socket sid, value = {provider, model, apiKey, baseUrl, name}
socket_configs = {}
_socket_configs_lock = threading.Lock()

# 全局预编译的图（图本身不区分 socket，Agent 函数通过 sid 获取配置）
fast_graph = None


def get_socket_state(sid: str) -> SocketState:
    """获取或创建指定 socket 的状态（线程安全），并更新活跃时间。"""
    with _socket_states_lock:
        if sid not in socket_states:
            socket_states[sid] = SocketState(sid)
            logger.info(f"Created socket state for sid={sid}")
        state = socket_states[sid]
        state.touch()
        return state


def cleanup_socket(sid: str):
    """清理 socket 相关资源，避免内存泄漏（线程安全）"""
    with _socket_states_lock:
        socket_states.pop(sid, None)
    with _socket_configs_lock:
        socket_configs.pop(sid, None)
    cleanup_sid(sid)
    cleanup_llm_config(sid)
    logger.info(f"Cleaned up socket resources for sid={sid}")


_cleanup_timer: threading.Timer | None = None


def _cleanup_inactive_sockets():
    """清理长时间不活跃的 socket 状态，防止内存泄漏。
    每 10 分钟执行一次检查。
    """
    global _cleanup_timer
    try:
        now = time.time()
        inactive_sids = []
        with _socket_states_lock:
            for sid, state in socket_states.items():
                if now - state.last_active > SOCKET_INACTIVE_TIMEOUT:
                    inactive_sids.append(sid)
        for sid in inactive_sids:
            cleanup_socket(sid)
            logger.info(f"Cleaned up inactive socket: sid={sid}")
    except Exception as e:
        logger.warning(f"Socket cleanup failed: {e}")
    finally:
        _cleanup_timer = threading.Timer(600, _cleanup_inactive_sockets)
        _cleanup_timer.daemon = True
        _cleanup_timer.start()


def start_socket_cleanup():
    """启动 socket 状态定时清理任务。"""
    global _cleanup_timer
    if _cleanup_timer is None:
        _cleanup_timer = threading.Timer(600, _cleanup_inactive_sockets)
        _cleanup_timer.daemon = True
        _cleanup_timer.start()
        logger.info("Socket cleanup timer started")


# 常见的 API Key 占位符/默认值，视为无效配置
_INVALID_API_KEY_PATTERNS = {
    "", "your_api_key_here", "your-api-key", "your_api_key",
    "sk-xxxx", "sk-xxxxxxxx", "placeholder", "none", "null",
}


def _is_valid_api_key(key: str | None) -> bool:
    """检查 API Key 是否有效（非空、非占位符）"""
    if not key or not isinstance(key, str):
        return False
    stripped = key.strip().lower()
    return stripped not in _INVALID_API_KEY_PATTERNS and len(stripped) > 4


def has_socket_config(sid: str) -> bool:
    """检查指定 socket 是否有有效配置"""
    with _socket_configs_lock:
        cfg = socket_configs.get(sid)
    if not cfg:
        return False
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return True
    return _is_valid_api_key(cfg.get("apiKey"))


def has_valid_config(sid: str = None) -> bool:
    """检查是否有有效配置。优先检查 socket 级配置，再回退到全局配置"""
    if sid and has_socket_config(sid):
        return True
    cfg = get_active_config()
    if cfg:
        provider = cfg.get("provider", "ollama")
        if provider == "ollama":
            return True
        return _is_valid_api_key(cfg.get("apiKey"))
    # 兼容旧版 .env
    try:
        from core.config import get_api_key, get_provider
        key = get_api_key()
        if _is_valid_api_key(key):
            return True
    except (ValueError, KeyError) as e:
        logger.debug(f"No valid config found: {e}")
    return False


_init_lock = threading.Lock()
_agents_initialized = False


def init_agents():
    """预编译所有图结构,并初始化记忆系统。

    幂等:首次调用做完整初始化,后续调用直接返回。
    Graph 是无状态的(运行时通过 sid 取 LLM 配置),memory store 自带
    `_initialized` 守卫;重复 init 等于浪费 30s embedder 预热。
    """
    global fast_graph, _agents_initialized

    # double-checked: 已初始化时跳过 lock,避免每次连接都进 contended path
    if _agents_initialized:
        return

    with _init_lock:
        if _agents_initialized:
            return

        # 只编译快速模式图：单 Responder 节点，搜索在 responder 内部异步处理
        _, _, responder, _ = create_agents(language="zh", fast_mode=True)
        fast_graph = create_fast_graph(responder)

        logger.info("Agent graphs initialized")

        # 初始化记忆系统(后台 fire-and-forget,sentence-transformers 首次加载~10-15s,
        # 不阻塞 init_agents 返回。第一次 search/save 调用时若未就绪会自动 await initialize)
        if _MEMORY_SYSTEM_AVAILABLE:
            def _bg_memory_init():
                try:
                    asyncio.run(get_memory_store().initialize())
                    logger.info("Memory system initialized")
                except Exception as e:
                    logger.warning(f"Memory system initialization failed: {e}")
            threading.Thread(target=_bg_memory_init, name="memory-warmup", daemon=True).start()
        else:
            logger.info("Memory system not available (dependencies missing)")

        _agents_initialized = True
