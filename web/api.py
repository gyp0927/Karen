"""HTTP API Blueprint - 从 web.app 拆分出来的路由。"""
import ast
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

from flask import Blueprint, request, send_file, render_template
from werkzeug.utils import secure_filename

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

# 延迟加载符号（避免循环导入）
_GENERATED_DIR = None

def _get_generated_dir():
    global _GENERATED_DIR
    if _GENERATED_DIR is None:
        from web.utils import _GENERATED_DIR as gd
        _GENERATED_DIR = gd
    return _GENERATED_DIR

from core.auth import (
    auth_required, AUTH_ENABLED, create_user, authenticate,
    get_user_by_id, list_users, delete_user, update_user_config,
)
from core.config import PROVIDER_NAMES, BASE_URLS
from core.export import export_markdown, export_json, export_html, export_pdf, get_export_filename
from core.plugin_system import list_plugins, get_registry, execute_plugin
from core.model_router import get_router
from agents.llm import clear_llm_cache
from state.model_config_manager import (
    list_configs, list_configs_full, get_active_config,
    add_config, update_config, delete_config, set_active_config, sync_to_env,
)
from state.stats import record_call, estimate_cost, get_stats_summary, get_daily_stats, CallRecord
from web.state import has_valid_config

# ===== HTTP Routes =====

@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/config")
def config_page():
    return render_template("config.html", first_run=not has_valid_config())


@api_bp.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")


@api_bp.route("/plugins")
def plugins_page():
    return render_template("plugins.html")


@api_bp.route("/api/upload", methods=["POST"])
@auth_required
def upload_file():
    """上传并解析文件，返回文件内容"""
    if "file" not in request.files:
        return {"success": False, "message": "没有文件"}, 400

    file = request.files["file"]
    if file.filename == "":
        return {"success": False, "message": "文件名为空"}, 400

    # secure_filename 防 ../ 与绝对路径
    safe_name = secure_filename(file.filename) or "upload.bin"
    temp_dir = tempfile.mkdtemp(prefix="upload_")
    file_path = os.path.join(temp_dir, safe_name)
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
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass


@api_bp.route("/api/generate-file", methods=["POST"])
@auth_required
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
        file_path = os.path.join(_get_generated_dir(), full_filename)

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
        for f in os.listdir(_get_generated_dir()):
            path = os.path.join(_get_generated_dir(), f)
            if os.path.isfile(path):
                files.append((path, os.path.getmtime(path)))
        if len(files) > max_files:
            files.sort(key=lambda x: x[1], reverse=True)
            for path, _ in files[max_files:]:
                os.remove(path)
    except (OSError, PermissionError) as e:
        logger.debug(f"Cleanup generated files failed: {e}")


@api_bp.route("/api/download/<filename>")
@auth_required
def download_file(filename):
    """下载生成的文件"""
    try:
        # 使用 pathlib 进行安全的路径校验
        base_dir = Path(_get_generated_dir()).resolve()
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

@api_bp.route("/api/configs", methods=["GET"])
def get_configs():
    """获取所有保存的模型配置列表"""
    configs = list_configs()
    active = get_active_config()
    return {
        "success": True,
        "configs": configs,
        "activeConfigId": active.get("id") if active else None
    }


@api_bp.route("/api/configs", methods=["POST"])
@auth_required
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
                    from web.state import init_agents
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
                    from web.state import init_agents
                    init_agents()
            return {"success": True, "message": "配置已保存", "config": result}
    except Exception as e:
        logger.exception("Failed to save config")
        return {"success": False, "message": f"保存失败: {str(e)}"}, 500


@api_bp.route("/api/configs/<config_id>", methods=["DELETE"])
@auth_required
def delete_config_api(config_id):
    """删除模型配置"""
    try:
        if delete_config(config_id):
            active = get_active_config()
            if active:
                sync_to_env(active)
                clear_llm_cache()
                from web.state import init_agents
                init_agents()
            return {"success": True, "message": "配置已删除"}
        return {"success": False, "message": "配置不存在"}, 404
    except Exception as e:
        logger.exception("Failed to delete config")
        return {"success": False, "message": f"删除失败: {str(e)}"}, 500


@api_bp.route("/api/configs/<config_id>/activate", methods=["POST"])
@auth_required
def activate_config_api(config_id):
    """激活指定配置"""
    try:
        if set_active_config(config_id):
            active = get_active_config()
            if active:
                sync_to_env(active)
                clear_llm_cache()
                from web.state import init_agents
                init_agents()
            return {"success": True, "message": "配置已激活"}
        return {"success": False, "message": "配置不存在"}, 404
    except Exception as e:
        logger.exception("Failed to activate config")
        return {"success": False, "message": f"激活失败: {str(e)}"}, 500


@api_bp.route("/api/configs/test", methods=["POST"])
@auth_required
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


def _mask_api_key(key: str | None) -> str:
    """对 API Key 进行脱敏处理：只显示前4位和后4位。"""
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


@api_bp.route("/api/config", methods=["GET"])
def get_config():
    """获取当前活跃配置（兼容旧版）——返回脱敏后的 API Key。"""
    active = get_active_config()
    if active:
        return {
            "provider": active.get("provider", "ollama"),
            "model": active.get("model", ""),
            "apiKey": _mask_api_key(active.get("apiKey", ""))
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
        "apiKey": _mask_api_key(api_key)
    }


@api_bp.route("/api/config", methods=["POST"])
def legacy_save_config():
    """兼容旧版保存配置接口"""
    return save_config_api()


# ===== Plugin API Routes =====

@api_bp.route("/api/plugins", methods=["GET"])
def get_plugins_api():
    """获取所有插件列表"""
    try:
        return {"success": True, "plugins": list_plugins()}
    except Exception as e:
        logger.exception("Failed to list plugins")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/plugins/<name>/execute", methods=["POST"])
@auth_required
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


@api_bp.route("/api/plugins/<name>/enable", methods=["POST"])
@auth_required
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


@api_bp.route("/api/plugins/<name>/disable", methods=["POST"])
@auth_required
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
_PLUGIN_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "ctypes", "socket", "importlib",
    "urllib", "http", "ftplib", "smtplib", "pickle", "marshal",
    "shutil", "pathlib", "tempfile", "multiprocessing",
}
_PLUGIN_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "system", "popen", "call", "run", "spawn", "fork",
    "getattr", "setattr", "delattr",
    "import_module", "find_loader", "spec_from_file_location",
}
_PLUGIN_FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__base__", "__subclasses__",
    "__mro__", "__globals__", "__builtins__", "__import__",
    "__getattribute__",
}


def _is_safe_plugin_filename(filename: str) -> bool:
    """检查插件文件名是否安全（只允许简单的 .py 文件名）。"""
    if not filename or not filename.endswith(".py"):
        return False
    name = filename[:-3]
    # 只允许字母、数字、下划线、连字符
    return name.isidentifier() or all(c.isalnum() or c in "_-" for c in name)


def _scan_plugin_content(content: str) -> list[str]:
    """AST 级扫描插件代码,挡 substring 黑名单挡不住的绕过手法。

    挡的攻击向量:
    - 危险模块 import / from import
    - 危险调用名(eval/exec/getattr/system/...)
    - 危险 dunder 属性访问(__class__/__subclasses__/__globals__ 链式)
    - 字符串拼接 + getattr 间接调用(getattr 已直接禁)
    - __builtins__['eval'] 等下标访问绕过
    - Lambda / comprehension 中隐藏的危险调用
    """
    issues: list[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        issues.append(f"语法错误: {e}")
        return issues

    def _check_name(node, name: str) -> bool:
        """递归检查 Name 节点，处理字符串拼接和 f-string 构造的调用名。"""
        if isinstance(node, ast.Name):
            return node.id == name
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value == name
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _check_name(node.left, name) or _check_name(node.right, name)
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str) and name in v.value:
                    return True
        return False

    for node in ast.walk(tree):
        # import xxx
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _PLUGIN_FORBIDDEN_MODULES:
                    issues.append(f"禁止 import: {alias.name}")
        # from xxx import yyy
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _PLUGIN_FORBIDDEN_MODULES:
                issues.append(f"禁止 from-import: {node.module}")
            for alias in node.names:
                if alias.name in _PLUGIN_FORBIDDEN_CALLS:
                    issues.append(f"禁止导入名: {alias.name}")
        # 调用名
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _PLUGIN_FORBIDDEN_CALLS:
                issues.append(f"禁止调用: {node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _PLUGIN_FORBIDDEN_CALLS:
                issues.append(f"禁止调用: .{node.func.attr}()")
            # 下标调用: __builtins__['eval']('code')
            elif isinstance(node.func, ast.Subscript):
                sub = node.func
                if isinstance(sub.value, ast.Name) and sub.value.id == "__builtins__":
                    if isinstance(sub.slice, ast.Constant) and sub.slice.value in _PLUGIN_FORBIDDEN_CALLS:
                        issues.append(f"禁止调用: __builtins__['{sub.slice.value}']()")
                    elif isinstance(sub.slice, ast.Name) and sub.slice.id in _PLUGIN_FORBIDDEN_CALLS:
                        issues.append(f"禁止调用: __builtins__['{sub.slice.id}']()")
            # 间接调用: getattr(obj, 'eval')()
            elif isinstance(node.func, ast.Call):
                inner = node.func
                if isinstance(inner.func, ast.Name) and inner.func.id == "getattr":
                    if len(inner.args) >= 2:
                        if _check_name(inner.args[1], "eval") or _check_name(inner.args[1], "exec"):
                            issues.append("禁止间接调用: getattr(..., 'eval/exec')()")
        # 危险属性访问 (.__class__ / .__subclasses__ 等)
        elif isinstance(node, ast.Attribute):
            if node.attr in _PLUGIN_FORBIDDEN_ATTRS:
                issues.append(f"禁止访问 dunder 属性: .{node.attr}")
        # 下标访问: __builtins__['__import__']
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "__builtins__":
                if isinstance(node.slice, ast.Constant) and node.slice.value in _PLUGIN_FORBIDDEN_ATTRS:
                    issues.append(f"禁止下标访问: __builtins__['{node.slice.value}']")
    return list(set(issues))


@api_bp.route("/api/plugins/upload", methods=["POST"])
@auth_required
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

@api_bp.route("/api/cache/stats", methods=["GET"])
def get_cache_stats_api():
    """获取缓存统计"""
    try:
        from core.cache import get_cache
        stats = get_cache().get_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.exception("Failed to get cache stats")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/cache/clear", methods=["POST"])
@auth_required
def clear_cache_api():
    """清空缓存"""
    try:
        from core.cache import get_cache
        get_cache().clear()
        return {"success": True, "message": "缓存已清空"}
    except Exception as e:
        logger.exception("Failed to clear cache")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/cache/config", methods=["POST"])
@auth_required
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

@api_bp.route("/api/rag/backends", methods=["GET"])
def get_rag_backends_api():
    """获取可用的向量存储后端列表"""
    try:
        from core.vector_store import list_backends, _get_backend_from_config
        return {
            "success": True,
            "backends": list_backends(),
            "current": _get_backend_from_config(),
        }
    except Exception as e:
        logger.exception("Failed to get RAG backends")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/rag/backend", methods=["POST"])
@auth_required
def set_rag_backend_api():
    """切换向量存储后端"""
    try:
        from core.vector_store import set_backend
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

@api_bp.route("/mcp")
def mcp_page():
    """MCP 服务器管理页面"""
    return render_template("mcp.html")


@api_bp.route("/api/mcp/servers", methods=["GET"])
def list_mcp_servers_api():
    """列出所有 MCP 服务器"""
    try:
        from core.mcp_manager import get_mcp_manager
        servers = get_mcp_manager().list_servers()
        return {"success": True, "servers": servers}
    except Exception as e:
        logger.exception("Failed to list MCP servers")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/mcp/servers", methods=["POST"])
@auth_required
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


@api_bp.route("/api/mcp/servers/<name>", methods=["DELETE"])
@auth_required
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


@api_bp.route("/api/mcp/servers/<name>/toggle", methods=["POST"])
@auth_required
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


@api_bp.route("/api/mcp/tools", methods=["GET"])
def list_mcp_tools_api():
    """列出所有 MCP 工具"""
    try:
        from core.mcp_manager import list_mcp_tools
        tools = list_mcp_tools()
        return {"success": True, "tools": tools}
    except Exception as e:
        logger.exception("Failed to list MCP tools")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/mcp/tools/<server>/<tool>", methods=["POST"])
@auth_required
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

@api_bp.route("/api/router/status", methods=["GET"])
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


@api_bp.route("/api/router/config", methods=["POST"])
@auth_required
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


@api_bp.route("/api/router/tiers/<tier>", methods=["POST"])
@auth_required
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

@api_bp.route("/api/auth/status", methods=["GET"])
def get_auth_status_api():
    """获取认证系统状态"""
    return {
        "success": True,
        "enabled": AUTH_ENABLED,
    }


@api_bp.route("/api/auth/register", methods=["POST"])
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


@api_bp.route("/api/auth/login", methods=["POST"])
def login_api():
    """登录（验证 API Key）"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        data = request.get_json() or {}
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            return {"success": False, "message": "请提供 API Key"}, 400
        user = authenticate(api_key, ip=request.remote_addr or "")
        if not user:
            return {"success": False, "message": "API Key 无效"}, 401
        return {"success": True, "user": user.to_dict()}
    except Exception as e:
        logger.exception("Login failed")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/auth/users", methods=["GET"])
@auth_required
def list_users_api():
    """列出所有用户(需认证 + LOCAL_ONLY 双重保护)"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        return {"success": True, "users": list_users()}
    except Exception as e:
        logger.exception("Failed to list users")
        return {"success": False, "message": str(e)}, 500


@api_bp.route("/api/auth/users/<user_id>", methods=["DELETE"])
@auth_required
def delete_user_api(user_id):
    """删除用户(需认证 + LOCAL_ONLY 双重保护)。用户只能删除自己。"""
    if not AUTH_ENABLED:
        return {"success": False, "message": "认证系统未启用"}, 400
    try:
        from core.auth import get_current_user
        current_user = get_current_user()
        if not current_user or current_user.id != user_id:
            return {"success": False, "message": "只能删除自己的账号"}, 403
        if delete_user(user_id):
            return {"success": True, "message": "用户已删除"}
        return {"success": False, "message": "用户不存在"}, 404
    except Exception as e:
        logger.exception(f"Failed to delete user {user_id}")
        return {"success": False, "message": str(e)}, 500




