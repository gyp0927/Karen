import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import io
import logging
import secrets
import asyncio
import tempfile
import threading
import time
import traceback

from core.utils import detect_language
from core.i18n import LANG_NAMES, get_lang_instruction

from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agents.llm import (
    set_current_llm_config, set_streaming_callback,
    clear_streaming_callback, clear_llm_cache, get_llm,
)
from agents.nodes import create_agents
from agents.search import web_searcher_agent, memory_searcher_agent
from agents.tools import tool_caller_node
from graph.orchestrator import create_fast_graph
from state.manager import SessionManager
from state.stop_flag import set_stop, clear_stop, is_stopped, cleanup_sid
from state.model_config_manager import (
    list_configs, list_configs_full, get_config, get_active_config,
    add_config, update_config, delete_config, set_active_config, sync_to_env
)
from core.config import PROVIDER_NAMES, BASE_URLS
from core.rag import add_document, search_knowledge, get_knowledge_stats, clear_knowledge, list_documents, delete_document_by_source
from core.export import export_markdown, export_json, export_html, export_pdf, get_export_filename
from core.plugin_system import get_registry, list_plugins, execute_plugin
from core.model_router import get_router
from core.auth import (
    AUTH_ENABLED, create_user, authenticate, get_user_by_id,
    list_users, delete_user, update_user_config, auth_required
)
from state.stats import record_call, estimate_cost, get_stats_summary, get_daily_stats, CallRecord
from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
# 限制上传体积,防止磁盘被恶意大文件耗尽
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# CORS 与 SocketIO origin 白名单 - 默认 localhost,生产环境需通过 CORS_ORIGINS 显式开放
_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
_cors_origins = [o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip()] or ["http://localhost:5000"]
CORS(app, origins=_cors_origins)
socketio = SocketIO(app, cors_allowed_origins=_cors_origins, async_mode="threading")

# 是否信任反向代理传来的 X-Forwarded-For / X-Real-Ip
# 默认不信任(否则任何外部用户可伪造 127.0.0.1 绕过 LOCAL_ONLY 校验)
TRUST_PROXY = os.getenv("TRUST_PROXY", "false").lower() in ("true", "1", "yes")

# 生成的文件保存目录
_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
os.makedirs(_GENERATED_DIR, exist_ok=True)

# 配置相关路由仅允许本机访问（防止局域网用户窃取 API Key）
LOCAL_ONLY_PREFIXES = ["/config", "/api/config", "/api/configs", "/knowledge", "/plugins", "/api/plugins/upload", "/mcp", "/api/mcp"]


@app.route("/api/export", methods=["POST"])
@auth_required
def export_chat():
    """导出聊天记录为指定格式"""
    try:
        data = request.get_json() or {}
        sid = data.get("sid", "")
        fmt = data.get("format", "md")

        if not sid or sid not in socket_states:
            return {"success": False, "message": "无法获取会话状态"}, 400

        state = socket_states[sid]

        # 认证启用时，验证会话所有权
        if AUTH_ENABLED:
            from core.auth import get_current_user
            current_user = get_current_user()
            if current_user and state.user_id != current_user.id:
                return {"success": False, "message": "无权访问该会话"}, 403

        messages = state.msg_manager.get_messages()
        if not messages:
            return {"success": False, "message": "当前会话没有消息"}, 400

        title = state.msg_manager._current().get("title", "聊天记录")

        if fmt == "md":
            content = export_markdown(messages, title)
            mime = "text/markdown"
        elif fmt == "json":
            content = export_json(messages, title)
            mime = "application/json"
        elif fmt == "html":
            content = export_html(messages, title)
            mime = "text/html"
        elif fmt == "pdf":
            pdf_bytes, error = export_pdf(messages, title)
            if error:
                # 回退到 HTML
                content = export_html(messages, title)
                mime = "text/html"
                fmt = "html"
                filename = get_export_filename(title, "html")
            else:
                filename = get_export_filename(title, "pdf")
                return send_file(
                    io.BytesIO(pdf_bytes),
                    mimetype="application/pdf",
                    as_attachment=True,
                    download_name=filename
                )
        else:
            return {"success": False, "message": f"不支持的格式: {fmt}"}, 400

        filename = get_export_filename(title, fmt)

        return send_file(
            io.BytesIO(content.encode("utf-8")),
            mimetype=mime,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.exception("Export failed")
        return {"success": False, "message": f"导出失败: {str(e)}"}, 500


@app.route("/api/stats", methods=["GET"])
@auth_required
def get_stats():
    """获取 API 调用统计"""
    try:
        days = request.args.get("days", 7, type=int)
        summary = get_stats_summary(days=days)
        daily = get_daily_stats(days=days)
        return {
            "success": True,
            "summary": summary,
            "daily": daily,
        }
    except Exception as e:
        logger.exception("Failed to get stats")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/rag/upload", methods=["POST"])
@auth_required
def upload_to_rag():
    """上传文件到知识库"""
    try:
        if "file" not in request.files:
            return {"success": False, "message": "没有文件"}, 400

        file = request.files["file"]
        if file.filename == "":
            return {"success": False, "message": "文件名为空"}, 400

        from core.document_parser import parse_document

        # 必须用 secure_filename 防 ../ 与绝对路径(Windows 上 ..\..\Windows\Temp 同样危险)
        safe_name = secure_filename(file.filename) or "upload.bin"
        temp_dir = tempfile.mkdtemp(prefix="rag_")
        file_path = os.path.join(temp_dir, safe_name)
        file.save(file_path)

        try:
            content = parse_document(file_path)
            chunks = run_async_in_thread(add_document(content, source=file.filename))
            return {
                "success": True,
                "message": f"已添加 {chunks} 个文档块到知识库",
                "filename": file.filename,
                "chunks": chunks,
            }
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                os.rmdir(temp_dir)
            except OSError:
                pass
    except Exception as e:
        logger.exception("RAG upload failed")
        return {"success": False, "message": f"上传失败: {str(e)}"}, 500


@app.route("/api/rag/clear", methods=["POST"])
@auth_required
def clear_rag_api():
    """清空知识库"""
    try:
        clear_knowledge()
        return {"success": True, "message": "知识库已清空"}
    except Exception as e:
        logger.exception("Failed to clear RAG")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/rag/stats", methods=["GET"])
def get_rag_stats():
    """获取知识库统计"""
    return {"success": True, **get_knowledge_stats()}


@app.route("/api/rag/documents", methods=["GET"])
def get_rag_documents():
    """获取知识库文档列表（按来源分组）"""
    try:
        docs = list_documents()
        return {"success": True, "documents": docs}
    except Exception as e:
        logger.exception("Failed to list RAG documents")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/rag/documents/<path:source>", methods=["DELETE"])
@auth_required
def delete_rag_document(source):
    """删除指定来源的文档"""
    try:
        count = delete_document_by_source(source)
        if count > 0:
            return {"success": True, "message": f"已删除 {count} 个文档块", "deleted": count}
        return {"success": False, "message": "未找到该文档"}, 404
    except Exception as e:
        logger.exception(f"Failed to delete RAG document: {source}")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/execute", methods=["POST"])
@auth_required
def execute_code_api():
    """执行 Python 代码（HTTP API） — 仅认证用户可调用"""
    try:
        data = request.get_json() or {}
        code = data.get("code", "")
        if not code:
            return {"success": False, "message": "代码为空"}, 400

        from tools.code_executor import execute_python, format_result
        result = execute_python(code, timeout=30)
        return {
            "success": result["success"],
            "result": result,
            "formatted": format_result(result),
        }
    except Exception as e:
        logger.exception("Code execution API failed")
        return {"success": False, "message": str(e)}, 500


def _get_real_remote_addr():
    """获取真实的客户端 IP。

    仅在 TRUST_PROXY=true 时才接受 X-Forwarded-For / X-Real-Ip,
    否则返回直连地址 — 不然任何外部客户端可伪造 127.0.0.1 绕过 LOCAL_ONLY。
    """
    if TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-Ip", "")
        if real_ip:
            return real_ip
    return request.remote_addr


@app.before_request
def restrict_local_only():
    path = request.path
    for prefix in LOCAL_ONLY_PREFIXES:
        if path.startswith(prefix):
            remote = _get_real_remote_addr()
            if remote not in ("127.0.0.1", "::1", "localhost"):
                return "Access denied: configuration is local only", 403


# ===== 语言检测 =====
# detect_language, LANG_NAMES, get_lang_instruction 已从 core.utils 和 core.i18n 导入


# ===== Socket 状态隔离 =====

class SocketState:
    """每个 Socket 连接的隔离状态"""

    # 不活跃超时时间（秒）：30 分钟
    INACTIVE_TIMEOUT = 30 * 60

    def __init__(self, sid: str):
        self.sid = sid
        self.user_id: str = ""  # 认证用户 ID
        self.msg_manager = SessionManager(user_id=self.user_id)
        self.current_base_response: str | None = None
        self.fast_mode = True
        self.review_language = "zh"
        self.detected_language: str | None = None  # 自动检测的用户语言（仅第一条消息）
        self.last_active = time.time()  # 最后活跃时间戳

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
fast_graph = None           # 快速模式：并行搜索 → Responder


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
                if now - state.last_active > SocketState.INACTIVE_TIMEOUT:
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

        # 只编译快速模式图：并行 WebSearcher + MemorySearcher + ToolCaller → Responder
        _, _, responder, _ = create_agents(language="zh", fast_mode=True)
        fast_graph = create_fast_graph(web_searcher_agent, memory_searcher_agent, tool_caller_node, responder)

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


def run_async_in_thread(coro):
    """在线程中安全运行异步协程。优先使用 asyncio.run()，
    若检测到已有事件循环在运行，则在新线程中创建独立循环执行。"""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            import concurrent.futures

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


def _send_history(sid: str):
    """发送聊天历史给前端"""
    state = get_socket_state(sid)
    messages = state.msg_manager.get_messages()
    if not messages:
        return

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            emit("user_message", {"message": msg.content})
        elif msg_type == "AIMessage":
            sender = msg.name if hasattr(msg, "name") else "assistant"
            if sender == "base_model":
                emit("bot_history", {"message": msg.content, "sender": sender, "awaiting_review": False})
            elif sender == "reviewer":
                emit("review_history", {"review_result": msg.content})
            else:
                emit("bot_history", {"message": msg.content, "sender": sender, "awaiting_review": False})

    emit("history_restored", {"awaiting_review": state.current_base_response is not None})


# ===== SocketIO Events =====

def socketio_auth_required(handler):
    """SocketIO 事件处理器认证装饰器。

    当 AUTH_ENABLED 为 True 时，检查 socket 状态中的 user_id 是否有效。
    未认证时断开连接并返回错误。
    """
    @functools.wraps(handler)
    def wrapper(*args, **kwargs):
        sid = request.sid
        if AUTH_ENABLED:
            state = get_socket_state(sid)
            if not state.user_id:
                emit("error", {"message": "认证失败，请先提供有效的 API Key"})
                socketio.disconnect(sid)
                return None
        return handler(*args, **kwargs)
    return wrapper


@socketio.on("connect")
def on_connect():
    sid = request.sid
    logger.info(f"Client connected: sid={sid}")

    if AUTH_ENABLED:
        api_key = request.args.get("api_key", "").strip()
        if not api_key:
            logger.warning(f"Connection rejected: no api_key from sid={sid}")
            return False
        user = authenticate(api_key)
        if not user:
            logger.warning(f"Connection rejected: invalid api_key from sid={sid}")
            return False
        state = get_socket_state(sid)
        state.set_user_id(user.id)
        logger.info(f"User authenticated: {user.name} (id={user.id})")

    emit("status", {"message": "Connected to 凯伦"})
    emit("auth_status", {"enabled": AUTH_ENABLED})
    _send_history(sid)


@socketio.on("send_message")
@socketio_auth_required
def handle_message(data):
    """处理用户消息"""
    sid = request.sid
    state = get_socket_state(sid)

    user_message = data.get("message", "")
    document_context = data.get("document_context", "")
    if not user_message:
        emit("error", {"message": "Empty message"})
        return

    # 检查是否已配置模型
    if not has_valid_config(sid):
        emit("config_required", {
            "message": "请先配置 AI 模型",
            "config_url": "/config"
        })
        return

    clear_stop(sid)

    expected_session_id = state.msg_manager.get_current_session_id()

    # 注入当前 socket 的 LLM 配置
    with _socket_configs_lock:
        user_cfg = socket_configs.get(sid)

    # 模型路由：根据问题复杂度选择模型档位
    try:
        router = get_router()
        if router.enabled:
            history = state.msg_manager.get_messages()
            history_turns = len(history) // 2
            route_result = router.route(user_message, history_turns)
            if route_result["tier"] != "default":
                tier_config = route_result["config"]
                # 合并路由配置到用户配置
                routed_cfg = dict(user_cfg) if user_cfg else {}
                routed_cfg.update({
                    "provider": tier_config.get("provider", routed_cfg.get("provider", "ollama")),
                    "model": tier_config.get("model", routed_cfg.get("model", "")),
                })
                if tier_config.get("apiKey"):
                    routed_cfg["apiKey"] = tier_config["apiKey"]
                if tier_config.get("baseUrl"):
                    routed_cfg["baseUrl"] = tier_config["baseUrl"]
                user_cfg = routed_cfg
                logger.info(f"Model routed to tier={route_result['tier']} for sid={sid}")
                emit("model_routed", {
                    "tier": route_result["tier"],
                    "score": route_result["analysis"]["score"],
                })
    except (OSError, ValueError) as e:
        logger.warning(f"Model routing failed, using default config: {e}")

    set_current_llm_config(user_cfg, sid)

    # 立即在前端显示用户消息并保存到数据库（不等待异步处理）
    emit("user_message", {"message": user_message})
    state.msg_manager.add_human_message(user_message)

    try:
        run_async_in_thread(_async_handle_message(
            sid, user_message, document_context, expected_session_id
        ))
    except Exception as e:
        logger.exception(f"Error handling message from sid={sid}")
        emit("error", {"message": str(e)})
    finally:
        set_current_llm_config(None, sid)
        clear_streaming_callback(sid)


def _emit_agent_reset(sid: str = ""):
    """发送 Agent 空闲状态到前端"""
    try:
        socketio.emit("agent_finish", {"agent": "web_searcher", "message": "空闲"}, room=sid)
        socketio.emit("agent_finish", {"agent": "memory_searcher", "message": "空闲"}, room=sid)
        socketio.emit("agent_finish", {"agent": "responder", "message": "空闲"}, room=sid)
    except Exception:
        pass


def _record_api_stats(sid: str, messages_for_llm: list, final_state: dict | None,
                       expected_session_id: str, call_start: float):
    """记录 API 调用统计"""
    try:
        from core.config import get_provider, get_model_name
        duration_ms = int((time.time() - call_start) * 1000)
        provider = get_provider()
        model = get_model_name()
        # 估算 token 数：优先使用 tiktoken，回退到字符数估算
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            all_content = "\n".join(m.content for m in messages_for_llm if hasattr(m, "content"))
            prompt_tokens = len(enc.encode(all_content))
            resp_content = final_state["messages"][-1].content if final_state else ""
            completion_tokens = len(enc.encode(resp_content))
        except Exception:
            # 回退：粗略估算（1 token ≈ 3 字符）
            all_content = "\n".join(m.content for m in messages_for_llm if hasattr(m, "content"))
            prompt_tokens = len(all_content) // 3
            resp_content = final_state["messages"][-1].content if final_state else ""
            completion_tokens = len(resp_content) // 3
        status = "stopped" if is_stopped(sid) else "success"
        record = CallRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            agent_name="multi_agent",
            session_id=expected_session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration_ms=duration_ms,
            estimated_cost_usd=estimate_cost(provider, prompt_tokens, completion_tokens),
            status=status,
        )
        record_call(record)
        logger.debug(f"API call recorded: {provider}/{model}, {duration_ms}ms, {prompt_tokens + completion_tokens} tokens")

        # 向前端发送 token 使用统计（使用 socketio.emit 避免后台线程上下文丢失）
        try:
            socketio.emit("token_usage", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": round(record.estimated_cost_usd, 6),
                "duration_ms": duration_ms,
                "provider": provider,
                "model": model,
            }, room=sid)
        except Exception as e:
            logger.debug(f"Failed to emit token_usage: {e}")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to record API stats: {e}")


async def _async_handle_message(sid: str, user_message: str, document_context: str, expected_session_id: str):
    """异步处理用户消息的核心逻辑（集成 RAG + 统计）

    消息持久化和前端显示仅在图执行成功后进行。如果执行失败，
    消息不会存入数据库，也不会出现在聊天界面中。

    搜索流程（由 Researcher 节点内部并行执行）：
    - 快速/计划模式：并行 2 个搜索子 Agent（联网 + 记忆）
    - 协调模式：并行 3 个搜索子 Agent（联网 + 记忆 + 知识库）
    """
    from langchain_core.messages import HumanMessage
    state = get_socket_state(sid)

    # 获取历史消息（快速模式保留 5 轮，协调模式 10 轮）
    history_turns = 5 if state.fast_mode else 10
    raw_messages = list(state.msg_manager.get_messages_for_model(max_turns=history_turns))

    # 过滤掉 content 为空的 assistant 消息（避免 API 400 错误）
    messages = [m for m in raw_messages if getattr(m, "content", "") or getattr(m, "type", None) != "ai"]

    # 构建当前用户消息（用于传给 LLM，但先不保存到数据库）
    current_msg = HumanMessage(content=user_message, name="Human")

    # === 上传文件内容 ===
    # 用户主动上传的文档内容直接注入上下文（不走搜索子 Agent）
    if document_context:
        current_msg = HumanMessage(
            content=f"{document_context}\n\n用户问题：{user_message}",
            name="Human",
        )

    # === 自动语言检测（仅第一条用户消息） ===
    # 如果当前会话还没有检测过语言，且这是第一条用户消息，进行检测
    if state.detected_language is None:
        state.detected_language = detect_language(user_message)
        lang_name = LANG_NAMES.get(state.detected_language, state.detected_language)
        logger.info(f"Auto-detected language for sid={sid}: {state.detected_language} ({lang_name})")

    # 传给 LLM 的消息列表（历史 + 当前）
    messages_for_llm = messages + [current_msg]

    # 当前模式固定为快速模式
    current_mode = "fast"

    initial_state = {
        "messages": messages_for_llm,
        "active_agent": None,
        "task_context": {
            "user_input": user_message,
            "detected_language": state.detected_language,
            "user_id": state.user_id,
            "session_id": expected_session_id,
            "mode": current_mode,
            "sid": sid,
        },
        "human_input_required": False,
        "base_model_response": None,
        "review_result": None,
        "awaiting_review": True
    }

    # 始终使用快速模式图
    graph = fast_graph

    # 安全 emit：后台线程中请求上下文可能丢失，使用 socketio.emit 代替
    def _safe_emit(event, data):
        try:
            socketio.emit(event, data, room=sid)
        except Exception as e:
            logger.debug(f"Failed to emit {event} to {sid}: {e}")

    # 设置流式输出回调：直接发送每个 token，零延迟
    _stream_started = False

    def on_token_chunk(token: str):
        nonlocal _stream_started
        if not _stream_started:
            _stream_started = True
            socketio.emit("stream_start", {"agent": "responder"}, room=sid)
        if token:
            socketio.emit("token_chunk", {"token": token}, room=sid)

    def flush_tokens():
        pass  # 实时发送无需 flush

    set_streaming_callback(on_token_chunk, sid)

    # ===== 搜索阶段显示 thinking 提示，LLM 输出时再创建流式气泡 =====
    # 避免占位文本和实际回复拼接在一起
    _safe_emit("thinking", {"message": "🔍 正在搜索相关信息，请稍候..."})
    # _stream_started 保持 False，让 LLM 回调的第一次 on_token_chunk 触发 stream_start

    final_state = None
    call_start = time.time()

    # === 执行图（核心逻辑，用 try/except 包裹）===
    # 所有模式统一走 LangGraph：Coordinator → Researcher(并行搜索) → Responder
    try:
        async for event in graph.astream(initial_state):
            if is_stopped(sid):
                break
            for node_name, node_output in event.items():
                if node_name == "coordinator":
                    _safe_emit("agent_start", {"agent": "coordinator", "message": "分析需求中..."})
                elif node_name == "researcher":
                    # 协调模式才有 coordinator
                    if "coordinator" in event:
                        _safe_emit("agent_finish", {"agent": "coordinator", "message": "分析完成"})
                    _safe_emit("thinking", {"message": "🔍 正在搜索网络信息..."})
                    _safe_emit("agent_start", {"agent": "researcher", "message": "调研中..."})
                elif node_name == "web_searcher":
                    # 快速/计划模式的并行搜索子 Agent
                    _safe_emit("thinking", {"message": "🔍 正在搜索网络信息..."})
                    _safe_emit("agent_start", {"agent": node_name, "message": "联网搜索中..."})
                elif node_name == "memory_searcher":
                    _safe_emit("thinking", {"message": "🧠 正在检索记忆..."})
                    _safe_emit("agent_start", {"agent": node_name, "message": "记忆检索中..."})
                elif node_name == "responder":
                    # 完成前置节点
                    if "researcher" in event:
                        _safe_emit("agent_finish", {"agent": "researcher", "message": "调研完成"})
                    if "web_searcher" in event or "memory_searcher" in event:
                        _safe_emit("agent_finish", {"agent": "search_hub", "message": "搜索完成"})
                    _safe_emit("agent_start", {"agent": "responder", "message": "生成回答中..."})
                final_state = node_output
    except Exception as e:
        logger.exception(f"Error processing message for sid={sid}")
        clear_streaming_callback(sid)
        _emit_agent_reset(sid)
        clear_stop(sid)
        # 如果流式气泡已创建，发送 stream_end 清理占位符
        if _stream_started:
            _safe_emit("stream_end", {"message": f"❌ 处理出错: {str(e)}", "awaiting_review": False})
        _safe_emit("message_failed", {"message": user_message, "error": str(e)})
        return

    # === 图执行成功后的处理 ===
    # 用户消息已在发送时保存，这里只保存 AI 回复

    # 1. 记录 API 调用统计
    _record_api_stats(sid, messages_for_llm, final_state, expected_session_id, call_start)

    # 3. 重置 Agent 状态
    _emit_agent_reset(sid)

    if is_stopped(sid):
        # 清理流式占位符，避免用户看到未完成的"正在搜索..."
        if _stream_started:
            _safe_emit("stream_end", {"message": "（生成已停止）", "awaiting_review": False})
        _safe_emit("generation_stopped", {"message": "生成已停止"})
        return

    if final_state is None:
        clear_streaming_callback(sid)
        if _stream_started:
            _safe_emit("stream_end", {"message": "未生成回复", "awaiting_review": False})
        _safe_emit("error", {"message": "No response generated"})
        return

    # 清理流式回调
    clear_streaming_callback(sid)

    # 会话隔离检查
    if state.msg_manager.get_current_session_id() != expected_session_id:
        logger.info(f"Session changed during generation, discarding result for sid={sid}")
        return

    base_response = final_state["messages"][-1].content
    state.current_base_response = base_response

    # 避免保存空响应到历史记录
    if base_response:
        state.msg_manager.add_agent_message(base_response, "base_model")

    # === 保存对话记忆到自适应记忆系统 ===
    # source=session_id 是为了让 delete_session 能 cascade 清掉本会话的记忆,
    # 不影响其他会话。user_id 退到 tags 里保留追踪能力。
    if _MEMORY_SYSTEM_AVAILABLE:
        try:
            store = get_memory_store()
            user_tag = f"user_{state.user_id}" if state.user_id else f"sid_{sid}"
            # 保存用户输入作为 observation
            await store.save_memory(
                content=f"用户说: {user_message}",
                memory_type="observation",
                source=expected_session_id,
                importance=0.4,
                tags=["user_input", user_tag],
            )
            # 保存 AI 回复作为 observation
            await store.save_memory(
                content=f"AI回复: {base_response[:500]}",  # 限制长度避免过大
                memory_type="observation",
                source=expected_session_id,
                importance=0.3,
                tags=["ai_response", user_tag],
            )
            logger.debug(f"Conversation memories saved for sid={sid} session={expected_session_id}")
        except Exception as e:
            logger.warning(f"Failed to save conversation memory: {e}")

    # 发送流式结束标记（如果使用了流式输出）
    if _stream_started:
        _safe_emit("stream_end", {"message": base_response, "awaiting_review": True})
    else:
        # 未使用流式输出（如 fast_mode），一次性发送完整消息
        _safe_emit("base_response", {
            "message": base_response,
            "awaiting_review": True
        })


@socketio.on("trigger_review")
@socketio_auth_required
def handle_review():
    """第二阶段：用户点击检查，让审查者只提供审查意见"""
    sid = request.sid
    state = get_socket_state(sid)

    if state.current_base_response is None:
        emit("error", {"message": "No base response to review"})
        return

    clear_stop(sid)
    expected_session_id = state.msg_manager.get_current_session_id()

    # 注入当前 socket 的 LLM 配置
    with _socket_configs_lock:
        user_cfg = socket_configs.get(sid)
    set_current_llm_config(user_cfg, sid)

    try:
        run_async_in_thread(_async_handle_review(sid, expected_session_id))
    except Exception as e:
        logger.exception(f"Error handling review from sid={sid}")
        emit("error", {"message": str(e)})
    finally:
        set_current_llm_config(None, sid)


async def _async_handle_review(sid: str, expected_session_id: str):
    """异步处理审查的核心逻辑（直接调用 reviewer，无需 LangGraph）"""
    state = get_socket_state(sid)

    from langchain_core.messages import HumanMessage
    # 获取用户原始问题
    all_messages = state.msg_manager.get_messages()
    user_message = ""
    for msg in reversed(all_messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    lang_name = "中文" if state.review_language == "zh" else "English"

    def _safe_emit_review(event, data):
        try:
            socketio.emit(event, data, room=sid)
        except Exception:
            pass

    _safe_emit_review("agent_start", {
        "agent": "reviewer",
        "message": "正在审查..." if state.review_language == "zh" else "Reviewing response..."
    })

    # 构建审查提示
    from agents.prompts import build_review_prompt
    review_prompt = build_review_prompt(user_message, state.current_base_response, state.review_language)

    # 直接调用 reviewer 节点函数（无需 LangGraph）
    from agents.llm import get_llm
    from agents.prompts import get_reviewer_prompt
    from langchain_core.messages import SystemMessage

    llm = get_llm(sid)
    reviewer_system = get_reviewer_prompt(state.review_language)
    messages = [SystemMessage(content=reviewer_system)]
    messages.append(HumanMessage(content=review_prompt))

    review_result = ""
    async for chunk in llm.astream(messages):
        if is_stopped(sid):
            break
        if chunk.content:
            review_result += chunk.content

    _safe_emit_review("agent_finish", {"agent": "reviewer", "message": "空闲"})

    if is_stopped(sid):
        _safe_emit_review("generation_stopped", {"message": "生成已停止"})
        return

    # 会话隔离检查
    if state.msg_manager.get_current_session_id() != expected_session_id:
        logger.info(f"Session changed during review, discarding result for sid={sid}")
        return

    _safe_emit_review("review_complete", {
        "review_result": review_result or "No review available",
        "original_response": state.current_base_response
    })

    state.current_base_response = None


def _create_new_session(sid: str, state):
    """新建会话的通用逻辑"""
    state.reset_session()
    session_id = state.msg_manager.new_session("新对话")
    emit("session_created", {
        "session_id": session_id,
        "sessions": state.msg_manager.list_sessions()
    })


@socketio.on("clear_history")
@socketio_auth_required
def handle_clear():
    """新建对话"""
    sid = request.sid
    _create_new_session(sid, get_socket_state(sid))


@socketio.on("new_session")
@socketio_auth_required
def handle_new_session():
    """新建会话"""
    sid = request.sid
    _create_new_session(sid, get_socket_state(sid))


@socketio.on("switch_session")
@socketio_auth_required
def handle_switch_session(data):
    """切换会话"""
    sid = request.sid
    state = get_socket_state(sid)
    session_id = data.get("session_id", "")
    if state.msg_manager.switch_session(session_id):
        state.reset_session()
        emit("session_switched", {
            "session_id": session_id,
            "sessions": state.msg_manager.list_sessions()
        })
        _send_history(sid)
    else:
        emit("error", {"message": "会话不存在"})


@socketio.on("delete_session")
@socketio_auth_required
def handle_delete_session(data):
    """删除会话"""
    sid = request.sid
    state = get_socket_state(sid)
    session_id = data.get("session_id", "")
    is_current = session_id == state.msg_manager.get_current_session_id()
    if state.msg_manager.delete_session(session_id):
        if is_current:
            state.reset_session()
        emit("session_deleted", {
            "session_id": session_id,
            "sessions": state.msg_manager.list_sessions()
        })
        _send_history(sid)
        # cascade 清记忆: source=session_id 的记忆从 hot+cold 一起清,
        # 不影响其他会话。fire-and-forget 不阻塞 UI。
        if _MEMORY_SYSTEM_AVAILABLE and session_id:
            def _bg_clear_memories():
                try:
                    n = asyncio.run(get_memory_store().delete_session_memories(session_id))
                    if n:
                        logger.info(f"Cleared {n} memories for deleted session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to clear memories for session {session_id}: {e}")
            threading.Thread(target=_bg_clear_memories, name=f"clear-mem-{session_id[:8]}", daemon=True).start()
    else:
        emit("error", {"message": "删除失败"})


@socketio.on("get_sessions")
def handle_get_sessions():
    """获取所有会话列表"""
    sid = request.sid
    state = get_socket_state(sid)
    emit("sessions_list", {
        "sessions": state.msg_manager.list_sessions()
    })


@socketio.on("get_model_info")
def handle_get_model_info():
    sid = request.sid
    cfg = None
    with _socket_configs_lock:
        cfg = socket_configs.get(sid)
    if cfg:
        provider = cfg.get("provider", "ollama")
        model = cfg.get("model", "")
        name = cfg.get("name", "")
        server_has_config = True
    else:
        active = get_active_config()
        if active:
            provider = active.get("provider", "ollama")
            model = active.get("model", "")
            name = active.get("name", "")
            server_has_config = True
        else:
            from core.config import get_provider, get_model_name
            provider = get_provider()
            model = get_model_name()
            name = ""
            server_has_config = False
    emit("model_info", {
        "provider": provider,
        "provider_name": PROVIDER_NAMES.get(provider, provider),
        "model": model,
        "name": name,
        "is_local": provider == "ollama",
        "has_config": cfg is not None,
        "server_has_config": server_has_config
    })


@socketio.on("get_available_models")
def handle_get_available_models():
    """获取可用的模型列表"""
    import requests
    models = []
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for m in data.get("models", []):
                model_name = m.get("name", "")
                size_bytes = m.get("size", 0)
                size_str = format_size(size_bytes) if size_bytes else ""
                full_name = model_name
                if size_str:
                    full_name = f"{model_name}:{size_str}"
                models.append({
                    "name": model_name,
                    "full": full_name,
                    "size": size_str,
                    "size_bytes": size_bytes,
                    "modified": m.get("modified_at", "")
                })
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to fetch Ollama models: {e}")

    from core.config import get_model_name
    current = get_model_name()

    emit("available_models", {
        "models": [m["full"] for m in models],
        "model_details": models,
        "current": current
    })


def format_size(bytes_size):
    """格式化文件大小"""
    if not bytes_size:
        return ""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"


@socketio.on("set_model")
@socketio_auth_required
def handle_set_model(data):
    """切换模型"""
    model_name = data.get("model", "")
    if not model_name:
        emit("error", {"message": "Invalid model name"})
        return

    try:
        import os
        os.environ["LLM_MODEL_NAME"] = model_name
        clear_llm_cache()
        init_agents()

        emit("model_changed", {
            "model": model_name,
            "message": f"Model changed to {model_name}"
        })
    except Exception as e:
        logger.exception("Failed to change model")
        emit("error", {"message": f"Failed to change model: {str(e)}"})


@socketio.on("activate_config")
@socketio_auth_required
def handle_activate_config(data):
    """通过 Socket 切换活跃配置"""
    config_id = data.get("configId", "")
    if not config_id:
        emit("error", {"message": "Invalid config ID"})
        return

    try:
        if set_active_config(config_id):
            active = get_active_config()
            if active:
                sync_to_env(active)
                clear_llm_cache()
                init_agents()
                emit("config_activated", {
                    "configId": config_id,
                    "name": active.get("name", ""),
                    "provider": active.get("provider", ""),
                    "provider_name": PROVIDER_NAMES.get(active.get("provider", ""), active.get("provider", "")),
                    "model": active.get("model", ""),
                    "message": f"已切换到: {active.get('name', '')}"
                })
        else:
            emit("error", {"message": "配置不存在"})
    except Exception as e:
        logger.exception("Failed to activate config")
        emit("error", {"message": f"切换失败: {str(e)}"})


@socketio.on("get_configs")
def handle_get_configs():
    """获取所有配置列表"""
    configs = list_configs()
    active = get_active_config()
    emit("configs_list", {
        "configs": configs,
        "activeConfigId": active.get("id") if active else None
    })


@socketio.on("stop_generation")
def handle_stop_generation():
    """用户请求停止生成"""
    sid = request.sid
    set_stop(sid)
    emit("generation_stopping", {"message": "正在停止..."})


def _get_mode(state: SocketState) -> str:
    """返回模式名称（固定为快速模式）"""
    return "fast"


@socketio.on("set_mode")
@socketio_auth_required
def handle_set_mode(data):
    """模式切换（固定为快速模式，忽略其他模式请求）"""
    sid = request.sid
    state = get_socket_state(sid)
    mode = data.get("mode", "fast")

    state.fast_mode = True
    mode_name = "快速模式"

    logger.info(f"Mode set to fast for sid={sid} (requested: {mode})")
    emit("mode_changed", {
        "mode": "fast",
        "message": f"已切换到{mode_name}"
    })


@socketio.on("get_mode")
def handle_get_mode():
    """获取当前工作模式"""
    sid = request.sid
    state = get_socket_state(sid)
    emit("mode_status", {"mode": _get_mode(state)})


@socketio.on("set_review_language")
@socketio_auth_required
def handle_set_review_language(data):
    """设置审查语言（按 socket 隔离）"""
    sid = request.sid
    state = get_socket_state(sid)
    lang = data.get("language", "zh")
    if lang not in ("zh", "en"):
        emit("error", {"message": f"不支持的语言: {lang}"})
        return

    try:
        state.review_language = lang
        lang_names = {"zh": "中文", "en": "English"}
        logger.info(f"Review language changed to {lang} for sid={sid}")
        emit("review_language_changed", {
            "language": lang,
            "message": f"审查语言已切换为: {lang_names.get(lang, lang)}"
        })
    except Exception as e:
        logger.exception("Failed to set review language")
        emit("error", {"message": f"切换审查语言失败: {str(e)}"})


@socketio.on("set_user_config")
@socketio_auth_required
def handle_set_user_config(data):
    """用户设置自己的 LLM 配置（LAN 共享场景，每个用户独立）"""
    sid = request.sid
    provider = data.get("provider", "ollama")
    model = data.get("model", "")
    api_key = data.get("apiKey", "")
    name = data.get("name", "")

    base_url = BASE_URLS.get(provider, "")
    if not base_url and provider == "ollama":
        base_url = "http://localhost:11434/v1"

    if provider != "ollama" and not api_key:
        emit("config_error", {"message": "请输入 API Key"})
        return
    if not model:
        emit("config_error", {"message": "请选择模型"})
        return

    with _socket_configs_lock:
        socket_configs[sid] = {
            "provider": provider,
            "model": model,
            "apiKey": api_key,
            "baseUrl": base_url,
            "name": name or f"{PROVIDER_NAMES.get(provider, provider)} · {model}"
        }

    logger.info(f"User config set for sid={sid}, provider={provider}, model={model}")
    emit("user_config_set", {
        "success": True,
        "provider": provider,
        "provider_name": PROVIDER_NAMES.get(provider, provider),
        "model": model,
        "name": name or f"{PROVIDER_NAMES.get(provider, provider)} · {model}"
    })



@socketio.on("disconnect")
def handle_disconnect():
    """断开连接时清理该 socket 的所有资源"""
    sid = request.sid
    logger.info(f"Client disconnected: sid={sid}")
    cleanup_socket(sid)


if __name__ == "__main__":
    print("Initializing agents...")
    init_agents()
    start_socket_cleanup()
    print("Agents initialized!")
    print("Starting Flask server at http://0.0.0.0:5000")
    print("Local access: http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


from web.api import api_bp
app.register_blueprint(api_bp)

# 模块加载时即触发 Agent 图编译与记忆系统后台预热，
# 确保 WSGI 入口(gunicorn 等)和 __main__ 入口行为一致。
init_agents()
