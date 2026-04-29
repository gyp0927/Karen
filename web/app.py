import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import logging
import secrets
import asyncio
import tempfile
import threading
import time
import traceback

from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from agents.factory import (
    create_agents, planner_node, parse_plan_from_response,
    set_current_llm_config, set_streaming_callback,
    clear_streaming_callback, clear_llm_cache, get_llm,
)
from graph.orchestrator import create_coordination_graph, create_fast_graph
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
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 生成的文件保存目录
_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
os.makedirs(_GENERATED_DIR, exist_ok=True)

# 配置相关路由仅允许本机访问（防止局域网用户窃取 API Key）
LOCAL_ONLY_PREFIXES = ["/config", "/api/config", "/api/configs", "/knowledge", "/plugins", "/api/plugins/upload", "/mcp", "/api/mcp"]


@app.route("/api/export", methods=["POST"])
def export_chat():
    """导出聊天记录为指定格式"""
    try:
        data = request.get_json() or {}
        sid = data.get("sid", "")
        fmt = data.get("format", "md")

        if sid and sid in socket_states:
            state = socket_states[sid]
        else:
            # 尝试从请求参数获取
            return {"success": False, "message": "无法获取会话状态"}, 400

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
def upload_to_rag():
    """上传文件到知识库"""
    try:
        if "file" not in request.files:
            return {"success": False, "message": "没有文件"}, 400

        file = request.files["file"]
        if file.filename == "":
            return {"success": False, "message": "文件名为空"}, 400

        from core.document_parser import parse_document

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        try:
            content = parse_document(file_path)
            chunks = add_document(content, source=file.filename)
            return {
                "success": True,
                "message": f"已添加 {chunks} 个文档块到知识库",
                "filename": file.filename,
                "chunks": chunks,
            }
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        logger.exception("RAG upload failed")
        return {"success": False, "message": f"上传失败: {str(e)}"}, 500


@app.route("/api/rag/clear", methods=["POST"])
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
def execute_code_api():
    """执行 Python 代码（HTTP API）"""
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
    """获取真实的客户端 IP，考虑反向代理。"""
    # X-Forwarded-For 格式: client, proxy1, proxy2
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
        # 计划模式状态
        self.planning_mode = False
        self.current_plan: dict | None = None  # { title, steps[] }
        self.current_step_index: int = 0
        self.plan_results: dict[int, str] = {}  # step_index -> result
        self.last_active = time.time()  # 最后活跃时间戳

    def touch(self):
        """更新最后活跃时间。"""
        self.last_active = time.time()

    def set_user_id(self, user_id: str):
        """设置用户 ID，如果变化则重新创建 SessionManager。"""
        if self.user_id != user_id:
            self.user_id = user_id
            self.msg_manager = SessionManager(user_id=user_id)


# 按 socket sid 存储的隔离状态
socket_states: dict[str, SocketState] = {}
_socket_states_lock = threading.Lock()

# Socket 级配置隔离：key = socket sid, value = {provider, model, apiKey, baseUrl, name}
socket_configs = {}
_socket_configs_lock = threading.Lock()

# 全局预编译的图（图本身不区分 socket，Agent 函数通过 sid 获取配置）
coordination_graph = None   # coordinator -> researcher(optional) -> responder
fast_graph = None           # 快速模式：直接 responder


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


def init_agents():
    """预编译所有图结构，并初始化记忆系统。"""
    global coordination_graph, fast_graph

    # 始终编译两种图，运行时根据 socket 的 fast_mode 选择
    coordinator, researcher, responder, reviewer = create_agents(language="zh", fast_mode=False)
    coordination_graph = create_coordination_graph(coordinator, researcher, responder)
    fast_graph = create_fast_graph(responder)

    logger.info("Agent graphs initialized")

    # 初始化记忆系统（如果可用）
    if _MEMORY_SYSTEM_AVAILABLE:
        try:
            run_async_in_thread(get_memory_store().initialize())
            logger.info("Memory system initialized")
        except Exception as e:
            logger.warning(f"Memory system initialization failed: {e}")
    else:
        logger.info("Memory system not available (dependencies missing)")


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


# ===== HTTP Routes =====

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/config")
def config_page():
    return render_template("config.html", first_run=not has_valid_config())


@app.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")


@app.route("/plugins")
def plugins_page():
    return render_template("plugins.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """上传并解析文件，返回文件内容"""
    if "file" not in request.files:
        return {"success": False, "message": "没有文件"}, 400

    file = request.files["file"]
    if file.filename == "":
        return {"success": False, "message": "文件名为空"}, 400

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    try:
        from core.document_parser import parse_document, truncate_text
        content = parse_document(file_path)
        truncated = truncate_text(content)
        logger.info(f"Uploaded file parsed: {file.filename}, length={len(content)}")
        return {
            "success": True,
            "filename": file.filename,
            "content": truncated
        }
    except Exception as e:
        logger.exception(f"Failed to parse file: {file.filename}")
        return {"success": False, "message": f"解析失败: {str(e)}"}, 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/api/generate-file", methods=["POST"])
def generate_file():
    """根据代码内容生成指定格式的文件"""
    try:
        data = request.get_json()
        if not data:
            return {"success": False, "message": "请求数据为空"}, 400

        content = data.get("content", "")
        filename = data.get("filename", "generated")
        file_format = data.get("format", "html")

        if not content:
            return {"success": False, "message": "内容为空"}, 400

        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-").strip()
        if not safe_name:
            safe_name = "generated"

        ext_map = {
            "html": "html", "doc": "doc", "txt": "txt", "md": "md",
            "css": "css", "js": "js", "json": "json", "py": "py",
        }
        ext = ext_map.get(file_format, file_format)
        full_filename = f"{safe_name}.{ext}"
        file_path = os.path.join(_GENERATED_DIR, full_filename)

        if file_format == "doc":
            html_wrapper = f"""<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head><meta charset="utf-8"><title>{safe_name}</title></head>
<body>
{content}
</body>
</html>"""
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_wrapper)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        _cleanup_generated_files(max_files=50)
        logger.info(f"Generated file: {full_filename}")

        return {
            "success": True,
            "filename": full_filename,
            "download_url": f"/api/download/{full_filename}",
            "message": f"文件已生成: {full_filename}"
        }
    except Exception as e:
        logger.exception("Failed to generate file")
        return {"success": False, "message": f"生成失败: {str(e)}"}, 500


def _cleanup_generated_files(max_files=50):
    """清理旧文件，只保留最新的 N 个"""
    try:
        files = []
        for f in os.listdir(_GENERATED_DIR):
            path = os.path.join(_GENERATED_DIR, f)
            if os.path.isfile(path):
                files.append((path, os.path.getmtime(path)))
        if len(files) > max_files:
            files.sort(key=lambda x: x[1], reverse=True)
            for path, _ in files[max_files:]:
                os.remove(path)
    except (OSError, PermissionError) as e:
        logger.debug(f"Cleanup generated files failed: {e}")


@app.route("/api/download/<filename>")
def download_file(filename):
    """下载生成的文件"""
    try:
        # 使用 pathlib 进行安全的路径校验
        base_dir = Path(_GENERATED_DIR).resolve()
        target = (base_dir / filename).resolve()

        # 确保目标在 base_dir 内部（resolve() 会处理 .. 等路径穿越）
        if base_dir not in target.parents and target != base_dir:
            return {"success": False, "message": "非法路径"}, 403

        if not target.is_file():
            return {"success": False, "message": "文件不存在"}, 404

        ext = target.suffix.lower()
        mime_map = {
            ".html": "text/html", ".doc": "application/msword",
            ".txt": "text/plain", ".md": "text/markdown",
            ".css": "text/css", ".js": "application/javascript",
            ".json": "application/json", ".py": "text/x-python",
        }
        mimetype = mime_map.get(ext, "application/octet-stream")

        return send_file(str(target), mimetype=mimetype, as_attachment=True, download_name=filename)
    except (OSError, PermissionError) as e:
        logger.warning(f"Download failed for {filename}: {e}")
        return {"success": False, "message": f"下载失败: {str(e)}"}, 500


# ===== Config API Routes =====

@app.route("/api/configs", methods=["GET"])
def get_configs():
    """获取所有保存的模型配置列表"""
    configs = list_configs()
    active = get_active_config()
    return {
        "success": True,
        "configs": configs,
        "activeConfigId": active.get("id") if active else None
    }


@app.route("/api/configs", methods=["POST"])
def save_config_api():
    """保存/新增模型配置"""
    try:
        data = request.get_json()
        if not data:
            return {"success": False, "message": "请求数据为空"}, 400

        name = data.get("name", "").strip()
        provider = data.get("provider", "ollama")
        model = data.get("model", "")
        api_key = data.get("apiKey", "")
        config_id = data.get("id", "")

        if not name:
            return {"success": False, "message": "请填写配置名称"}, 400

        base_url = BASE_URLS.get(provider, BASE_URLS.get("ollama", ""))

        if config_id:
            result = update_config(config_id, name=name, provider=provider, model=model,
                                   apiKey=api_key, baseUrl=base_url)
            if result:
                active = get_active_config()
                if active and active.get("id") == config_id:
                    sync_to_env(active)
                    clear_llm_cache()
                    init_agents()
                return {"success": True, "message": "配置已更新", "config": result}
            return {"success": False, "message": "配置不存在"}, 404
        else:
            result = add_config(name=name, provider=provider, model=model,
                                api_key=api_key, base_url=base_url)
            all_configs = list_configs_full()
            if len(all_configs) == 1:
                active = get_active_config()
                if active:
                    sync_to_env(active)
                    clear_llm_cache()
                    init_agents()
            return {"success": True, "message": "配置已保存", "config": result}
    except Exception as e:
        logger.exception("Failed to save config")
        return {"success": False, "message": f"保存失败: {str(e)}"}, 500


@app.route("/api/configs/<config_id>", methods=["DELETE"])
def delete_config_api(config_id):
    """删除模型配置"""
    try:
        if delete_config(config_id):
            active = get_active_config()
            if active:
                sync_to_env(active)
                clear_llm_cache()
                init_agents()
            return {"success": True, "message": "配置已删除"}
        return {"success": False, "message": "配置不存在"}, 404
    except Exception as e:
        logger.exception("Failed to delete config")
        return {"success": False, "message": f"删除失败: {str(e)}"}, 500


@app.route("/api/configs/<config_id>/activate", methods=["POST"])
def activate_config_api(config_id):
    """激活指定配置"""
    try:
        if set_active_config(config_id):
            active = get_active_config()
            if active:
                sync_to_env(active)
                clear_llm_cache()
                init_agents()
            return {"success": True, "message": "配置已激活"}
        return {"success": False, "message": "配置不存在"}, 404
    except Exception as e:
        logger.exception("Failed to activate config")
        return {"success": False, "message": f"激活失败: {str(e)}"}, 500


@app.route("/api/configs/test", methods=["POST"])
def test_config_api():
    """测试 API 连通性"""
    try:
        data = request.get_json()
        if not data:
            return {"success": False, "message": "请求数据为空"}, 400

        provider = data.get("provider", "ollama")
        model = data.get("model", "")
        api_key = data.get("apiKey", "")

        if provider == "ollama":
            import requests
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    if model and model in model_names:
                        return {"success": True, "message": f"Ollama 连接正常，已找到模型 {model}"}
                    return {"success": True, "message": f"Ollama 连接正常，共 {len(models)} 个模型"}
                return {"success": False, "message": f"Ollama 返回状态码 {resp.status_code}"}
            except requests.exceptions.ConnectionError:
                return {"success": False, "message": "无法连接到 Ollama，请确认是否已启动"}
            except Exception as e:
                return {"success": False, "message": f"连接失败: {str(e)}"}

        if not api_key:
            return {"success": False, "message": "请先输入 API Key"}

        base_url = BASE_URLS.get(provider, "")
        if not base_url:
            return {"success": False, "message": f"未知提供商: {provider}"}

        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.get(f"{base_url}/models", headers=headers, timeout=15)
            if resp.status_code == 200:
                return {"success": True, "message": "API 连接正常，授权验证通过"}
            elif resp.status_code == 401:
                return {"success": False, "message": "API Key 无效或已过期"}
            elif resp.status_code == 403:
                return {"success": False, "message": "API Key 没有访问权限"}
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.debug(f"GET /models failed for {provider}, falling back to chat test: {e}")

        try:
            body = {
                "model": model or "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1
            }
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=20)
            if resp.status_code == 200:
                return {"success": True, "message": "API 连接正常，模型可正常调用"}
            elif resp.status_code == 401:
                return {"success": False, "message": "API Key 无效或已过期"}
            elif resp.status_code == 404:
                return {"success": False, "message": f"模型 {model} 不存在或无权访问"}
            else:
                err = resp.json().get("error", {}).get("message", resp.text[:200])
                return {"success": False, "message": f"请求失败 ({resp.status_code}): {err}"}
        except requests.exceptions.Timeout:
            return {"success": False, "message": "请求超时，请检查网络连接"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": "无法连接到 API 服务器，请检查网络"}
        except Exception as e:
            return {"success": False, "message": f"测试失败: {str(e)}"}

    except Exception as e:
        logger.exception("Config test failed")
        return {"success": False, "message": f"测试失败: {str(e)}"}, 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """获取当前活跃配置（兼容旧版）"""
    active = get_active_config()
    if active:
        return {
            "provider": active.get("provider", "ollama"),
            "model": active.get("model", ""),
            "apiKey": active.get("apiKey", "")
        }
    from core.config import get_provider, get_model_name
    from dotenv import load_dotenv
    import os
    load_dotenv()
    provider = request.args.get("provider", get_provider()).lower()
    key_env_name = f"LLM_API_KEY_{provider.upper().replace('-', '_')}"
    api_key = os.getenv(key_env_name, "")
    if not api_key:
        api_key = os.getenv("LLM_API_KEY", "")
    return {
        "provider": provider,
        "model": get_model_name() if provider == get_provider() else "",
        "apiKey": api_key
    }


@app.route("/api/config", methods=["POST"])
def legacy_save_config():
    """兼容旧版保存配置接口"""
    return save_config_api()


# ===== Plugin API Routes =====

@app.route("/api/plugins", methods=["GET"])
def get_plugins_api():
    """获取所有插件列表"""
    try:
        return {"success": True, "plugins": list_plugins()}
    except Exception as e:
        logger.exception("Failed to list plugins")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/plugins/<name>/execute", methods=["POST"])
def execute_plugin_api(name):
    """执行指定插件"""
    try:
        data = request.get_json() or {}
        args = data.get("args", {})
        result = execute_plugin(name, args)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "message": str(e)}, 400
    except Exception as e:
        logger.exception(f"Failed to execute plugin {name}")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/plugins/<name>/enable", methods=["POST"])
def enable_plugin_api(name):
    """启用插件"""
    try:
        registry = get_registry()
        if registry.enable(name):
            return {"success": True, "message": f"插件 '{name}' 已启用"}
        return {"success": False, "message": f"插件 '{name}' 不存在"}, 404
    except Exception as e:
        logger.exception(f"Failed to enable plugin {name}")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/plugins/<name>/disable", methods=["POST"])
def disable_plugin_api(name):
    """禁用插件"""
    try:
        registry = get_registry()
        registry.disable(name)
        return {"success": True, "message": f"插件 '{name}' 已禁用"}
    except Exception as e:
        logger.exception(f"Failed to disable plugin {name}")
        return {"success": False, "message": str(e)}, 500


# 插件上传安全限制
_MAX_PLUGIN_SIZE = 256 * 1024  # 256KB
_PLUGIN_FORBIDDEN_IMPORTS = {"os.system", "subprocess", "eval", "exec", "compile", "__import__"}


def _is_safe_plugin_filename(filename: str) -> bool:
    """检查插件文件名是否安全（只允许简单的 .py 文件名）。"""
    if not filename or not filename.endswith(".py"):
        return False
    name = filename[:-3]
    # 只允许字母、数字、下划线、连字符
    return name.isidentifier() or all(c.isalnum() or c in "_-" for c in name)


def _scan_plugin_content(content: str) -> list[str]:
    """扫描插件内容中的可疑代码。"""
    issues = []
    content_lower = content.lower()
    for forbidden in _PLUGIN_FORBIDDEN_IMPORTS:
        if forbidden in content_lower:
            issues.append(f"发现危险操作: {forbidden}")
    return issues


@app.route("/api/plugins/upload", methods=["POST"])
def upload_plugin_api():
    """上传安装新插件（带安全校验）"""
    try:
        if "file" not in request.files:
            return {"success": False, "message": "没有文件"}, 400

        file = request.files["file"]
        if file.filename == "":
            return {"success": False, "message": "文件名为空"}, 400

        if not _is_safe_plugin_filename(file.filename):
            return {"success": False, "message": "文件名不合法（只允许 .py 文件，文件名只能包含字母、数字、下划线、连字符）"}, 400

        content = file.read().decode("utf-8", errors="replace")
        if len(content) > _MAX_PLUGIN_SIZE:
            return {"success": False, "message": f"插件文件过大（最大 {_MAX_PLUGIN_SIZE // 1024}KB）"}, 400

        issues = _scan_plugin_content(content)
        if issues:
            return {"success": False, "message": f"安全扫描未通过: {'; '.join(issues)}"}, 400

        import core.plugin_system as ps
        plugins_dir = Path(ps._PLUGINS_DIR)
        plugins_dir.mkdir(parents=True, exist_ok=True)

        file_path = plugins_dir / file.filename
        file_path.write_text(content, encoding="utf-8")

        # 重新扫描插件
        registry = get_registry()
        registry.discover()

        return {"success": True, "message": f"插件 '{file.filename}' 安装成功", "filename": file.filename}
    except Exception as e:
        logger.exception("Failed to upload plugin")
        return {"success": False, "message": f"安装失败: {str(e)}"}, 500


# ===== Cache API Routes =====

@app.route("/api/cache/stats", methods=["GET"])
def get_cache_stats_api():
    """获取缓存统计"""
    try:
        from core.cache import get_cache
        stats = get_cache().get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.exception("Failed to get cache stats")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache_api():
    """清空缓存"""
    try:
        from core.cache import get_cache
        get_cache().clear()
        return {"success": True, "message": "缓存已清空"}
    except Exception as e:
        logger.exception("Failed to clear cache")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/cache/config", methods=["POST"])
def config_cache_api():
    """配置缓存参数"""
    try:
        from core.cache import configure_cache
        data = request.get_json() or {}
        enabled = data.get("enabled", True)
        ttl_hours = data.get("ttl_hours", 24)
        configure_cache(enabled=enabled, ttl_hours=ttl_hours)
        return {"success": True, "message": f"缓存配置已更新: enabled={enabled}, ttl={ttl_hours}h"}
    except Exception as e:
        logger.exception("Failed to config cache")
        return {"success": False, "message": str(e)}, 500


# ===== RAG Backend API Routes =====

@app.route("/api/rag/backends", methods=["GET"])
def get_rag_backends_api():
    """获取可用的向量存储后端列表"""
    try:
        from core.vector_store.factory import list_backends, _get_backend_from_config
        return {
            "success": True,
            "backends": list_backends(),
            "current": _get_backend_from_config(),
        }
    except Exception as e:
        logger.exception("Failed to get RAG backends")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/rag/backend", methods=["POST"])
def set_rag_backend_api():
    """切换向量存储后端"""
    try:
        from core.vector_store.factory import set_backend
        from core.rag import reset_store
        data = request.get_json() or {}
        backend = data.get("backend", "numpy")
        persist_path = data.get("persist_path")

        if not set_backend(backend, persist_path):
            return {"success": False, "message": f"后端 '{backend}' 不可用，请检查依赖是否安装"}, 400

        reset_store(backend=backend, persist_path=persist_path)
        return {"success": True, "message": f"向量存储后端已切换为: {backend}"}
    except Exception as e:
        logger.exception("Failed to set RAG backend")
        return {"success": False, "message": str(e)}, 500


# ===== MCP API Routes =====

@app.route("/mcp")
def mcp_page():
    """MCP 服务器管理页面"""
    return render_template("mcp.html")


@app.route("/api/mcp/servers", methods=["GET"])
def list_mcp_servers_api():
    """列出所有 MCP 服务器"""
    try:
        from core.mcp_manager import get_mcp_manager
        servers = get_mcp_manager().list_servers()
        return {"success": True, "servers": servers}
    except Exception as e:
        logger.exception("Failed to list MCP servers")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/mcp/servers", methods=["POST"])
def add_mcp_server_api():
    """添加 MCP 服务器"""
    try:
        from core.mcp_manager import get_mcp_manager
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        if not name:
            return {"success": False, "message": "服务器名称不能为空"}, 400

        get_mcp_manager().add_server(
            name=name,
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url", ""),
            transport=data.get("transport", "stdio"),
        )
        # 重新加载插件以包含新 MCP 工具
        from core.plugin_system import get_registry
        get_registry().discover()
        return {"success": True, "message": f"服务器 '{name}' 已添加"}
    except Exception as e:
        logger.exception("Failed to add MCP server")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/mcp/servers/<name>", methods=["DELETE"])
def delete_mcp_server_api(name):
    """删除 MCP 服务器"""
    try:
        from core.mcp_manager import get_mcp_manager
        if get_mcp_manager().remove_server(name):
            from core.plugin_system import get_registry
            get_registry().discover()
            return {"success": True, "message": f"服务器 '{name}' 已删除"}
        return {"success": False, "message": "服务器不存在"}, 404
    except Exception as e:
        logger.exception(f"Failed to delete MCP server {name}")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/mcp/servers/<name>/toggle", methods=["POST"])
def toggle_mcp_server_api(name):
    """启用/禁用 MCP 服务器"""
    try:
        from core.mcp_manager import get_mcp_manager
        data = request.get_json() or {}
        enabled = data.get("enabled", True)
        if get_mcp_manager().toggle_server(name, enabled):
            from core.plugin_system import get_registry
            get_registry().discover()
            return {"success": True, "message": f"服务器 '{name}' 已{'启用' if enabled else '禁用'}"}
        return {"success": False, "message": "服务器不存在"}, 404
    except Exception as e:
        logger.exception(f"Failed to toggle MCP server {name}")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/mcp/tools", methods=["GET"])
def list_mcp_tools_api():
    """列出所有 MCP 工具"""
    try:
        from core.mcp_manager import list_mcp_tools
        tools = list_mcp_tools()
        return {"success": True, "tools": tools}
    except Exception as e:
        logger.exception("Failed to list MCP tools")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/mcp/tools/<server>/<tool>", methods=["POST"])
def call_mcp_tool_api(server, tool):
    """测试调用 MCP 工具"""
    try:
        from core.mcp_manager import get_mcp_manager
        data = request.get_json() or {}
        args = data.get("args", {})
        result = get_mcp_manager().call_tool(server, tool, args)
        return {"success": True, "result": result}
    except Exception as e:
        logger.exception(f"Failed to call MCP tool {server}/{tool}")
        return {"success": False, "message": str(e)}, 500


# ===== Model Router API Routes =====

@app.route("/api/router/status", methods=["GET"])
def get_router_status_api():
    """获取模型路由状态"""
    try:
        router = get_router()
        return {
            "success": True,
            "enabled": router.enabled,
            "tiers": router.get_all_tiers(),
        }
    except Exception as e:
        logger.exception("Failed to get router status")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/router/config", methods=["POST"])
def config_router_api():
    """配置模型路由"""
    try:
        from core.model_router import configure_router
        data = request.get_json() or {}
        enabled = data.get("enabled", True)
        configure_router(enabled=enabled)
        return {"success": True, "message": f"模型路由已{'启用' if enabled else '禁用'}"}
    except Exception as e:
        logger.exception("Failed to config router")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/router/tiers/<tier>", methods=["POST"])
def set_router_tier_api(tier):
    """设置模型档位"""
    try:
        router = get_router()
        data = request.get_json() or {}
        provider = data.get("provider", "")
        model = data.get("model", "")
        api_key = data.get("apiKey", "")
        base_url = data.get("baseUrl", "")
        router.set_tier(tier, provider, model, api_key, base_url)
        return {"success": True, "message": f"档位 '{tier}' 已更新"}
    except Exception as e:
        logger.exception(f"Failed to set router tier {tier}")
        return {"success": False, "message": str(e)}, 500


# ===== Auth API Routes =====

@app.route("/api/auth/status", methods=["GET"])
def get_auth_status_api():
    """获取认证系统状态"""
    return {
        "success": True,
        "enabled": AUTH_ENABLED,
    }


@app.route("/api/auth/register", methods=["POST"])
def register_api():
    """注册新用户"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        api_key = data.get("apiKey", "").strip()
        if not name:
            return {"success": False, "message": "请填写用户名"}, 400
        if not api_key:
            return {"success": False, "message": "请提供 API Key"}, 400
        user = create_user(name, api_key)
        return {"success": True, "user": user.to_dict()}
    except ValueError as e:
        return {"success": False, "message": str(e)}, 400
    except Exception as e:
        logger.exception("Failed to register user")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/auth/login", methods=["POST"])
def login_api():
    """登录（验证 API Key）"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        data = request.get_json() or {}
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            return {"success": False, "message": "请提供 API Key"}, 400
        user = authenticate(api_key)
        if not user:
            return {"success": False, "message": "API Key 无效"}, 401
        return {"success": True, "user": user.to_dict()}
    except Exception as e:
        logger.exception("Login failed")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/auth/users", methods=["GET"])
def list_users_api():
    """列出所有用户"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        return {"success": True, "users": list_users()}
    except Exception as e:
        logger.exception("Failed to list users")
        return {"success": False, "message": str(e)}, 500


@app.route("/api/auth/users/<user_id>", methods=["DELETE"])
def delete_user_api(user_id):
    """删除用户"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        if delete_user(user_id):
            return {"success": True, "message": "用户已删除"}
        return {"success": False, "message": "用户不存在"}, 404
    except Exception as e:
        logger.exception(f"Failed to delete user {user_id}")
        return {"success": False, "message": str(e)}, 500


# ===== SocketIO Events =====

@socketio.on("connect")
def on_connect():
    sid = request.sid
    logger.info(f"Client connected: sid={sid}")

    # 如果启用认证，尝试从 query 参数获取 api_key
    if AUTH_ENABLED:
        api_key = request.args.get("api_key", "").strip()
        if api_key:
            user = authenticate(api_key)
            if user:
                state = get_socket_state(sid)
                state.set_user_id(user.id)
                logger.info(f"User authenticated: {user.name} (id={user.id})")
            else:
                logger.warning(f"Invalid api_key from sid={sid}")

    emit("status", {"message": "Connected to 果冻ai"})
    emit("auth_status", {"enabled": AUTH_ENABLED})
    _send_history(sid)


@socketio.on("send_message")
def handle_message(data):
    """处理用户消息"""
    sid = request.sid
    state = get_socket_state(sid)

    user_message = data.get("message", "")
    document_context = data.get("document_context", "")
    web_search = data.get("web_search", False)
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
        if state.planning_mode:
            run_async_in_thread(_async_handle_planning(
                sid, user_message, expected_session_id
            ))
        else:
            run_async_in_thread(_async_handle_message(
                sid, user_message, document_context, expected_session_id, web_search
            ))
    except Exception as e:
        logger.exception(f"Error handling message from sid={sid}")
        emit("error", {"message": str(e)})
    finally:
        set_current_llm_config(None, sid)
        clear_streaming_callback(sid)


def _emit_agent_reset(fast_mode: bool):
    """发送 Agent 空闲状态到前端"""
    if fast_mode:
        emit("agent_finish", {"agent": "responder", "message": "空闲"})
    else:
        emit("agent_finish", {"agent": "coordinator", "message": "空闲"})
        emit("agent_finish", {"agent": "researcher", "message": "空闲"})
        emit("agent_finish", {"agent": "responder", "message": "空闲"})


def _record_api_stats(sid: str, messages_for_llm: list, final_state: dict | None,
                       expected_session_id: str, call_start: float):
    """记录 API 调用统计"""
    try:
        from core.config import get_provider, get_model_name
        duration_ms = int((time.time() - call_start) * 1000)
        provider = get_provider()
        model = get_model_name()
        # 估算 token 数（粗略：1 token ≈ 3 字符）
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

        # 向前端发送 token 使用统计
        try:
            emit("token_usage", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": round(record.estimated_cost_usd, 6),
                "duration_ms": duration_ms,
                "provider": provider,
                "model": model,
            })
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"Failed to emit token_usage: {e}")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to record API stats: {e}")


async def _async_handle_message(sid: str, user_message: str, document_context: str, expected_session_id: str, web_search: bool = False):
    """异步处理用户消息的核心逻辑（集成 RAG + 统计）

    消息持久化和前端显示仅在图执行成功后进行。如果执行失败，
    消息不会存入数据库，也不会出现在聊天界面中。
    """
    from langchain_core.messages import HumanMessage
    state = get_socket_state(sid)

    # 获取历史消息（快速模式保留 5 轮，协调模式 10 轮）
    history_turns = 5 if state.fast_mode else 10
    messages = list(state.msg_manager.get_messages_for_model(max_turns=history_turns))

    # 构建当前用户消息（用于传给 LLM，但先不保存到数据库）
    current_msg = HumanMessage(content=user_message, name="Human")

    # === 联网搜索 ===
    search_context = ""
    if web_search:
        try:
            from tools.search import search_and_summarize
            search_result = search_and_summarize(user_message, max_results=3)
            if search_result and "未找到" not in search_result:
                search_context = search_result
        except ImportError:
            logger.warning("Search module not available")
        except (TimeoutError, ConnectionError) as e:
            logger.warning(f"Search timed out or connection failed: {e}")

    # === RAG 知识库检索 ===
    rag_context = ""
    try:
        rag_result = search_knowledge(user_message, top_k=3)
        if rag_result and "知识库为空" not in rag_result and "未启用" not in rag_result:
            rag_context = rag_result
    except ImportError:
        logger.warning("RAG module not available")
    except (TimeoutError, ConnectionError) as e:
        logger.warning(f"RAG query timed out or connection failed: {e}")

    if document_context or search_context or rag_context:
        extra_parts = []
        if search_context:
            extra_parts.append(search_context)
        if document_context:
            extra_parts.append(document_context)
        if rag_context:
            extra_parts.append(rag_context)
        combined = "\n\n".join(extra_parts)
        combined_content = f"{combined}\n\n用户问题：{user_message}"
        current_msg = HumanMessage(content=combined_content, name="Human")

    # === 自适应记忆系统检索 ===
    memory_context = ""
    if _MEMORY_SYSTEM_AVAILABLE:
        try:
            store = get_memory_store()
            memories = await store.retrieve(user_message, top_k=5, user_id=state.user_id)
            if memories:
                memory_context = store.format_memories_for_prompt(memories)
                logger.info(f"Retrieved {len(memories)} memories for sid={sid}")
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

    # 传给 LLM 的消息列表（历史 + 当前）
    # 如果有相关记忆，以 SystemMessage 形式注入到消息列表开头
    messages_for_llm = messages + [current_msg]
    if memory_context:
        from langchain_core.messages import SystemMessage
        memory_msg = SystemMessage(content=memory_context)
        messages_for_llm = [memory_msg] + messages_for_llm

    initial_state = {
        "messages": messages_for_llm,
        "active_agent": None,
        "task_context": {"user_input": user_message},
        "human_input_required": False,
        "base_model_response": None,
        "review_result": None,
        "awaiting_review": True
    }

    # 根据 socket 的 fast_mode 选择图
    graph = fast_graph if state.fast_mode else coordination_graph

    if state.fast_mode:
        emit("thinking", {"message": "正在生成回答..."})
        emit("agent_start", {"agent": "responder", "message": "生成回答中..."})
    else:
        emit("thinking", {"message": "Coordinator 正在分析需求..."})

    # 设置流式输出回调 - 低延迟 batch 策略：每 2 个 token flush，
    # 或遇到换行/句末标点立即 flush，同时设置 60ms 超时避免单个 token 延迟
    streaming_buffer = {"started": False, "batch": [], "timer": None}

    def _do_flush():
        if streaming_buffer["batch"]:
            batch_text = "".join(streaming_buffer["batch"])
            streaming_buffer["batch"].clear()
            # Use socketio.emit with room=sid to avoid request context issues
            # when called from threading.Timer callback
            socketio.emit("token_chunk", {"token": batch_text}, room=sid)
        streaming_buffer["timer"] = None

    def on_token_chunk(token: str):
        if not streaming_buffer["started"]:
            streaming_buffer["started"] = True
            emit("stream_start", {"agent": "responder"})
        streaming_buffer["batch"].append(token)
        # 累积 2 个 token 或包含换行/句末标点时立即 flush
        should_flush = (
            len(streaming_buffer["batch"]) >= 2
            or "\n" in token
            or (token and token[-1] in ".。!！?？")
        )
        if should_flush:
            if streaming_buffer["timer"]:
                streaming_buffer["timer"].cancel()
                streaming_buffer["timer"] = None
            _do_flush()
        elif streaming_buffer["timer"] is None:
            # 启动超时定时器，避免单个 token 延迟
            streaming_buffer["timer"] = threading.Timer(0.06, _do_flush)
            streaming_buffer["timer"].start()

    def flush_tokens():
        if streaming_buffer["timer"]:
            streaming_buffer["timer"].cancel()
            streaming_buffer["timer"] = None
        if streaming_buffer["batch"]:
            batch_text = "".join(streaming_buffer["batch"])
            streaming_buffer["batch"].clear()
            emit("token_chunk", {"token": batch_text})

    set_streaming_callback(on_token_chunk, sid)

    final_state = None
    call_start = time.time()

    # === 执行图（核心逻辑，用 try/except 包裹）===
    try:
        if state.fast_mode:
            # 快速模式：直接调用 LLM，绕过 LangGraph 状态管理开销
            llm = get_llm(sid)
            from langchain_core.messages import AIMessage, SystemMessage
            responder_prompt = """你是 ResponderBot（果冻ai），一位乐于助人且友善的助手。\n\n你的职责是：\n1. 提供清晰、友好的回复\n2. 以易于理解的方式呈现信息\n3. 保持对话式、亲切的风格\n\n重要：每次回答时，你必须以"我是果冻ai"开头，然后再根据上下文生成最终回答。"""
            all_messages = [SystemMessage(content=responder_prompt)] + messages_for_llm
            response = ""
            async for chunk in llm.astream(all_messages):
                if is_stopped(sid):
                    break
                if chunk.content:
                    response += chunk.content
                    on_token_chunk(chunk.content)
            flush_tokens()
            final_state = {"messages": [AIMessage(content=response, name="responder")]}
        else:
            async for event in graph.astream(initial_state):
                if is_stopped(sid):
                    break
                for node_name, node_output in event.items():
                    if node_name == "coordinator":
                        emit("agent_start", {"agent": "coordinator", "message": "分析需求中..."})
                    elif node_name == "researcher":
                        emit("agent_finish", {"agent": "coordinator", "message": "分析完成"})
                        emit("agent_start", {"agent": "researcher", "message": "调研中..."})
                    elif node_name == "responder":
                        if "researcher" in event:
                            emit("agent_finish", {"agent": "researcher", "message": "调研完成"})
                        emit("agent_start", {"agent": "responder", "message": "生成回答中..."})
                    final_state = node_output
    except Exception as e:
        logger.exception(f"Error processing message for sid={sid}")
        clear_streaming_callback(sid)
        _emit_agent_reset(state.fast_mode)
        clear_stop(sid)
        emit("message_failed", {"message": user_message, "error": str(e)})
        return

    # === 图执行成功后的处理 ===
    # 用户消息已在发送时保存，这里只保存 AI 回复

    # 1. 记录 API 调用统计
    _record_api_stats(sid, messages_for_llm, final_state, expected_session_id, call_start)

    # 3. 重置 Agent 状态
    _emit_agent_reset(state.fast_mode)

    if is_stopped(sid):
        emit("generation_stopped", {"message": "生成已停止"})
        return

    if final_state is None:
        clear_streaming_callback(sid)
        emit("error", {"message": "No response generated"})
        return

    # 清理流式回调
    clear_streaming_callback(sid)

    # 会话隔离检查
    if state.msg_manager.get_current_session_id() != expected_session_id:
        logger.info(f"Session changed during generation, discarding result for sid={sid}")
        return

    base_response = final_state["messages"][-1].content
    state.current_base_response = base_response

    state.msg_manager.add_agent_message(base_response, "base_model")

    # === 保存对话记忆到自适应记忆系统 ===
    if _MEMORY_SYSTEM_AVAILABLE:
        try:
            store = get_memory_store()
            source = state.user_id or sid
            # 保存用户输入作为 observation
            await store.save_memory(
                content=f"用户说: {user_message}",
                memory_type="observation",
                source=source,
                importance=0.4,
                tags=["user_input", f"session_{expected_session_id}"],
            )
            # 保存 AI 回复作为 observation
            await store.save_memory(
                content=f"AI回复: {base_response[:500]}",  # 限制长度避免过大
                memory_type="observation",
                source=source,
                importance=0.3,
                tags=["ai_response", f"session_{expected_session_id}"],
            )
            logger.debug(f"Conversation memories saved for sid={sid}")
        except Exception as e:
            logger.warning(f"Failed to save conversation memory: {e}")

    # 发送流式结束标记（如果使用了流式输出）
    if streaming_buffer["started"]:
        emit("stream_end", {"message": base_response, "awaiting_review": True})
    else:
        # 未使用流式输出（如 fast_mode），一次性发送完整消息
        emit("base_response", {
            "message": base_response,
            "awaiting_review": True
        })


@socketio.on("trigger_review")
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
    emit("agent_start", {
        "agent": "reviewer",
        "message": "正在审查..." if state.review_language == "zh" else "Reviewing response..."
    })

    # 构建审查提示
    from prompts.reviewer_prompt import build_review_prompt
    review_prompt = build_review_prompt(user_message, state.current_base_response, state.review_language)

    # 直接调用 reviewer 节点函数（无需 LangGraph）
    from agents.factory import get_llm
    from prompts.reviewer_prompt import get_reviewer_prompt
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

    emit("agent_finish", {"agent": "reviewer", "message": "空闲"})

    if is_stopped(sid):
        emit("generation_stopped", {"message": "生成已停止"})
        return

    # 会话隔离检查
    if state.msg_manager.get_current_session_id() != expected_session_id:
        logger.info(f"Session changed during review, discarding result for sid={sid}")
        return

    emit("review_complete", {
        "review_result": review_result or "No review available",
        "original_response": state.current_base_response
    })

    state.current_base_response = None


def _create_new_session(sid: str, state):
    """新建会话的通用逻辑"""
    set_stop(sid)
    state.current_base_response = None
    session_id = state.msg_manager.new_session("新对话")
    emit("session_created", {
        "session_id": session_id,
        "sessions": state.msg_manager.list_sessions()
    })


@socketio.on("clear_history")
def handle_clear():
    """新建对话"""
    sid = request.sid
    _create_new_session(sid, get_socket_state(sid))


@socketio.on("new_session")
def handle_new_session():
    """新建会话"""
    sid = request.sid
    _create_new_session(sid, get_socket_state(sid))


@socketio.on("switch_session")
def handle_switch_session(data):
    """切换会话"""
    sid = request.sid
    state = get_socket_state(sid)
    set_stop(sid)
    session_id = data.get("session_id", "")
    if state.msg_manager.switch_session(session_id):
        state.current_base_response = None
        emit("session_switched", {
            "session_id": session_id,
            "sessions": state.msg_manager.list_sessions()
        })
        _send_history(sid)
    else:
        emit("error", {"message": "会话不存在"})


@socketio.on("delete_session")
def handle_delete_session(data):
    """删除会话"""
    sid = request.sid
    state = get_socket_state(sid)
    session_id = data.get("session_id", "")
    if session_id == state.msg_manager.get_current_session_id():
        set_stop(sid)
    if state.msg_manager.delete_session(session_id):
        state.current_base_response = None
        emit("session_deleted", {
            "session_id": session_id,
            "sessions": state.msg_manager.list_sessions()
        })
        _send_history(sid)
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
    """根据 fast_mode 和 planning_mode 返回统一的模式名称"""
    if state.planning_mode:
        return "planning"
    if state.fast_mode:
        return "fast"
    return "coordination"


@socketio.on("set_mode")
def handle_set_mode(data):
    """统一切换工作模式：协调 / 快速 / 计划"""
    sid = request.sid
    state = get_socket_state(sid)
    mode = data.get("mode", "coordination")

    if mode == "fast":
        state.fast_mode = True
        state.planning_mode = False
        mode_name = "快速模式"
    elif mode == "planning":
        state.fast_mode = False
        state.planning_mode = True
        mode_name = "计划模式"
    else:
        state.fast_mode = False
        state.planning_mode = False
        mode_name = "协调模式"

    logger.info(f"Mode changed to {mode} for sid={sid}")
    emit("mode_changed", {
        "mode": mode,
        "message": f"已切换到{mode_name}"
    })


@socketio.on("get_mode")
def handle_get_mode():
    """获取当前工作模式"""
    sid = request.sid
    state = get_socket_state(sid)
    emit("mode_status", {"mode": _get_mode(state)})


@socketio.on("set_review_language")
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


@socketio.on("execute_code")
def handle_execute_code(data):
    """执行 Python 代码"""
    code = data.get("code", "")
    if not code:
        emit("code_result", {"success": False, "error": "代码为空"})
        return

    try:
        from tools.code_executor import execute_python, format_result
        result = execute_python(code, timeout=data.get("timeout", 30))
        emit("code_result", {
            "success": result["success"],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "error": result.get("error", ""),
            "duration_ms": result.get("duration_ms", 0),
            "formatted": format_result(result),
        })
    except Exception as e:
        logger.exception("Code execution failed")
        emit("code_result", {"success": False, "error": str(e)})


@socketio.on("web_search")
def handle_web_search(data):
    """联网搜索"""
    query = data.get("query", "")
    if not query:
        emit("search_result", {"success": False, "error": "查询为空"})
        return

    try:
        from tools.search import search_and_summarize
        result = search_and_summarize(query, max_results=data.get("max_results", 3))
        emit("search_result", {
            "success": True,
            "query": query,
            "result": result,
        })
    except Exception as e:
        logger.exception("Web search failed")
        emit("search_result", {"success": False, "error": str(e)})


@socketio.on("confirm_plan")
def handle_confirm_plan(data):
    """用户确认计划后开始执行"""
    sid = request.sid
    state = get_socket_state(sid)

    if not state.current_plan:
        emit("error", {"message": "当前没有可执行的计划"})
        return

    # 用户可能修改了计划，更新
    user_plan = data.get("plan")
    if user_plan and isinstance(user_plan, dict) and "steps" in user_plan:
        state.current_plan = user_plan
        # 重新编号
        for i, step in enumerate(state.current_plan.get("steps", []), 1):
            step["index"] = i

    state.current_step_index = 0
    state.plan_results = {}

    emit("plan_confirmed", {
        "plan": state.current_plan,
        "message": "计划已确认，开始执行"
    })

    # 自动开始执行第一步
    clear_stop(sid)
    expected_session_id = state.msg_manager.get_current_session_id()

    with _socket_configs_lock:
        user_cfg = socket_configs.get(sid)
    set_current_llm_config(user_cfg, sid)

    try:
        run_async_in_thread(_async_execute_plan_step(
            sid, 0, expected_session_id
        ))
    except Exception as e:
        logger.exception(f"Error starting plan execution for sid={sid}")
        emit("error", {"message": str(e)})
    finally:
        set_current_llm_config(None, sid)


@socketio.on("skip_step")
def handle_skip_step(data):
    """用户跳过当前步骤"""
    sid = request.sid
    state = get_socket_state(sid)

    if not state.current_plan:
        emit("error", {"message": "当前没有正在执行的计划"})
        return

    step_index = data.get("step_index", state.current_step_index)
    steps = state.current_plan.get("steps", [])

    if step_index >= len(steps):
        emit("error", {"message": "步骤索引超出范围"})
        return

    emit("plan_step_skipped", {
        "step_index": step_index,
        "step_title": steps[step_index].get("title", ""),
        "message": f"已跳过步骤 {step_index + 1}"
    })

    # 继续执行下一步
    state.current_step_index = step_index + 1
    if state.current_step_index >= len(steps):
        _finish_plan(sid)
        return

    clear_stop(sid)
    expected_session_id = state.msg_manager.get_current_session_id()

    with _socket_configs_lock:
        user_cfg = socket_configs.get(sid)
    set_current_llm_config(user_cfg, sid)

    try:
        run_async_in_thread(_async_execute_plan_step(
            sid, state.current_step_index, expected_session_id
        ))
    except Exception as e:
        logger.exception(f"Error skipping step for sid={sid}")
        emit("error", {"message": str(e)})
    finally:
        set_current_llm_config(None, sid)


@socketio.on("cancel_plan")
def handle_cancel_plan():
    """用户取消当前计划"""
    sid = request.sid
    state = get_socket_state(sid)

    state.current_plan = None
    state.current_step_index = 0
    state.plan_results = {}
    set_stop(sid)

    emit("plan_cancelled", {"message": "计划已取消"})


async def _async_handle_planning(sid: str, user_message: str, expected_session_id: str):
    """计划模式：生成任务计划"""
    state = get_socket_state(sid)
    from langchain_core.messages import HumanMessage

    emit("thinking", {"message": "Planner 正在分析需求并制定计划..."})
    emit("agent_start", {"agent": "planner", "message": "制定计划中..."})

    # 构建 Planner 输入
    plan_prompt = (
        f"请为以下需求制定一个详细的执行计划：\n\n"
        f"{user_message}\n\n"
        f"请分析需求并输出 JSON 格式的任务计划。"
    )

    plan_state = {
        "messages": [HumanMessage(content=plan_prompt, name="Human")],
    }

    try:
        result = await planner_node(plan_state, sid=sid)
        planner_msg = result["messages"][-1]
        plan_text = planner_msg.content

        plan = parse_plan_from_response(plan_text)
        if not plan or "steps" not in plan:
            logger.warning(f"Failed to parse plan from response for sid={sid}")
            emit("error", {"message": "计划解析失败，请重试或用更明确的描述"})
            emit("agent_finish", {"agent": "planner", "message": "空闲"})
            return

        # 确保步骤有正确的 index
        for i, step in enumerate(plan.get("steps", []), 1):
            step["index"] = i

        state.current_plan = plan
        state.current_step_index = 0
        state.plan_results = {}

        # 保存用户消息到数据库
        state.msg_manager.add_human_message(user_message)

        emit("plan_generated", {
            "title": plan.get("title", "任务计划"),
            "steps": plan.get("steps", []),
            "message": f"已生成计划：{plan.get('title', '任务计划')}，共 {len(plan.get('steps', []))} 个步骤"
        })
        emit("agent_finish", {"agent": "planner", "message": "空闲"})

    except Exception as e:
        logger.exception(f"Error generating plan for sid={sid}")
        emit("error", {"message": f"计划生成失败: {str(e)}"})
        emit("agent_finish", {"agent": "planner", "message": "空闲"})


async def _async_execute_plan_step(sid: str, step_index: int, expected_session_id: str):
    """执行计划中的某一步"""
    state = get_socket_state(sid)

    if not state.current_plan or step_index >= len(state.current_plan.get("steps", [])):
        return

    steps = state.current_plan["steps"]
    step = steps[step_index]

    emit("plan_step_started", {
        "step_index": step_index,
        "step_title": step.get("title", ""),
        "message": f"正在执行步骤 {step_index + 1}/{len(steps)}：{step.get('title', '')}"
    })
    emit("agent_start", {"agent": "coordinator", "message": f"执行步骤 {step_index + 1}..."})

    from langchain_core.messages import HumanMessage

    # 构建步骤执行任务
    step_prompt = (
        f"当前执行计划第 {step_index + 1} 步：{step.get('title', '')}\n"
        f"步骤描述：{step.get('description', '')}\n\n"
        f"请完成这个步骤的任务。如果需要研究或搜索信息，请先进行研究。"
    )

    # 获取历史消息（用于上下文）
    messages = list(state.msg_manager.get_messages_for_model(max_turns=5))
    current_msg = HumanMessage(content=step_prompt, name="Human")
    messages_for_llm = messages + [current_msg]

    initial_state = {
        "messages": messages_for_llm,
        "active_agent": None,
        "task_context": {"user_input": step_prompt, "step_index": step_index},
        "human_input_required": False,
        "base_model_response": None,
        "review_result": None,
        "awaiting_review": False,
    }

    graph = coordination_graph  # 计划模式始终使用协调图
    final_state = None

    try:
        async for event in graph.astream(initial_state):
            if is_stopped(sid):
                break
            for node_name, node_output in event.items():
                if node_name == "coordinator":
                    emit("agent_start", {"agent": "coordinator", "message": "分析步骤需求..."})
                elif node_name == "researcher":
                    emit("agent_finish", {"agent": "coordinator", "message": "分析完成"})
                    emit("agent_start", {"agent": "researcher", "message": "调研中..."})
                elif node_name == "responder":
                    if "researcher" in event:
                        emit("agent_finish", {"agent": "researcher", "message": "调研完成"})
                    emit("agent_start", {"agent": "responder", "message": "生成结果..."})
                final_state = node_output
    except Exception as e:
        logger.exception(f"Error executing plan step {step_index} for sid={sid}")
        _emit_agent_reset(False)
        clear_stop(sid)
        emit("plan_step_error", {
            "step_index": step_index,
            "error": str(e),
            "message": f"步骤 {step_index + 1} 执行失败"
        })
        return

    _emit_agent_reset(False)

    if is_stopped(sid):
        emit("generation_stopped", {"message": "生成已停止"})
        return

    if final_state is None:
        emit("error", {"message": "步骤执行未产生结果"})
        return

    # 会话隔离检查
    if state.msg_manager.get_current_session_id() != expected_session_id:
        logger.info(f"Session changed during plan step, discarding result for sid={sid}")
        return

    step_result = final_state["messages"][-1].content
    state.plan_results[step_index] = step_result

    # 将步骤结果保存到消息历史
    state.msg_manager.add_agent_message(
        f"【步骤 {step_index + 1}】{step.get('title', '')}\n\n{step_result}",
        "base_model"
    )

    # 自动标记步骤完成（AI 打勾）
    emit("plan_step_completed", {
        "step_index": step_index,
        "step_title": step.get("title", ""),
        "result": step_result,
        "message": f"步骤 {step_index + 1}/{len(steps)} 已完成 ✅"
    })

    # 继续执行下一步
    state.current_step_index = step_index + 1
    if state.current_step_index >= len(steps):
        _finish_plan(sid)
        return

    # 执行下一步（无延迟，连续执行提高效率）
    await _async_execute_plan_step(sid, state.current_step_index, expected_session_id)


def _finish_plan(sid: str):
    """计划全部完成后生成总结"""
    state = get_socket_state(sid)

    if not state.current_plan:
        return

    steps = state.current_plan.get("steps", [])
    emit("plan_completed", {
        "title": state.current_plan.get("title", "任务计划"),
        "total_steps": len(steps),
        "completed_steps": len(state.plan_results),
        "message": f"🎉 计划全部完成！共 {len(steps)} 个步骤"
    })

    # 清空当前计划，但保留 planning_mode 为 True 以便继续
    state.current_plan = None
    state.current_step_index = 0
    state.plan_results = {}


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
