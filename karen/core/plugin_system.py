"""插件系统 - 动态加载和管理自定义工具。

插件是放在 plugins/ 目录下的 Python 模块，继承 Plugin 基类并实现 execute 方法。
"""

import abc
import importlib.util
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PLUGINS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
_ENABLED_FILE = os.path.join(_PLUGINS_DIR, "enabled.json")


class Plugin(abc.ABC):
    """插件抽象基类。所有插件必须继承此类。"""

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True

    @abc.abstractmethod
    def execute(self, args: dict) -> str:
        """执行插件功能。

        Args:
            args: 前端或 Agent 传入的参数字典

        Returns:
            插件执行结果文本
        """
        pass

    def get_schema(self) -> dict:
        """返回插件的输入参数 schema（供 Agent 使用）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }


class PluginRegistry:
    """插件注册表，负责扫描、加载和管理插件。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: dict[str, Plugin] = {}
            cls._instance._enabled: set[str] = set()
            cls._instance._load_enabled()
            cls._instance.discover()
        return cls._instance

    def _load_enabled(self):
        """从 enabled.json 加载启用状态。"""
        if os.path.exists(_ENABLED_FILE):
            try:
                with open(_ENABLED_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._enabled = set(data.get("enabled", []))
            except Exception:
                self._enabled = set()
        else:
            self._enabled = set()

    def _save_enabled(self):
        """保存启用状态到 enabled.json。"""
        os.makedirs(_PLUGINS_DIR, exist_ok=True)
        with open(_ENABLED_FILE, "w", encoding="utf-8") as f:
            json.dump({"enabled": sorted(self._enabled)}, f, ensure_ascii=False, indent=2)

    def discover(self):
        """扫描 plugins/ 目录，发现并加载所有插件。"""
        self._plugins.clear()
        if not os.path.exists(_PLUGINS_DIR):
            os.makedirs(_PLUGINS_DIR, exist_ok=True)
            return

        for filename in os.listdir(_PLUGINS_DIR):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            filepath = os.path.join(_PLUGINS_DIR, filename)
            self._load_plugin_file(filepath)

        # 注册 MCP 工具为插件
        try:
            from core.mcp_plugin_adapter import register_mcp_plugins
            register_mcp_plugins()
        except Exception:
            pass

        logger.info(f"Discovered {len(self._plugins)} plugins: {list(self._plugins.keys())}")

    def _load_plugin_file(self, filepath: str):
        """从文件加载单个插件模块。"""
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            # 安全扫描：先读取文件内容，进行 AST 安全检查
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            if not self._scan_plugin_ast(source, filepath):
                return

            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找模块中继承 Plugin 的类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, Plugin)
                        and attr is not Plugin and not getattr(attr, "__abstractmethods__", None)):
                    try:
                        instance = attr()
                        self._plugins[instance.name] = instance
                        # 默认启用新发现的插件
                        if not self._enabled and module_name != "enabled":
                            self._enabled.add(instance.name)
                    except Exception as e:
                        logger.warning(f"Failed to instantiate plugin {attr_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load plugin file {filepath}: {e}")

    def list_plugins(self) -> list[dict]:
        """列出所有插件信息。"""
        result = []
        for name, plugin in self._plugins.items():
            result.append({
                "name": plugin.name,
                "description": plugin.description,
                "version": plugin.version,
                "enabled": name in self._enabled,
                "schema": plugin.get_schema(),
            })
        return sorted(result, key=lambda x: x["name"])

    def get_plugin(self, name: str) -> Plugin | None:
        """获取指定名称的插件实例。"""
        return self._plugins.get(name)

    def is_enabled(self, name: str) -> bool:
        """检查插件是否已启用。"""
        return name in self._enabled and name in self._plugins

    def enable(self, name: str) -> bool:
        """启用插件。"""
        if name in self._plugins:
            self._enabled.add(name)
            self._save_enabled()
            return True
        return False

    def disable(self, name: str) -> bool:
        """禁用插件。"""
        self._enabled.discard(name)
        self._save_enabled()
        return True

    def execute(self, name: str, args: dict) -> str:
        """执行指定插件。"""
        if not self.is_enabled(name):
            raise ValueError(f"插件 '{name}' 未启用或不存在")
        plugin = self._plugins.get(name)
        if plugin is None:
            raise ValueError(f"插件 '{name}' 不存在")
        return plugin.execute(args)

    def _scan_plugin_ast(self, source: str, filepath: str) -> bool:
        """对插件源码进行 AST 安全检查。

        TODO: 添加可选的插件签名/哈希验证机制，防止插件被篡改。
        当前仅做 AST 静态扫描，不验证文件完整性。

        禁止导入危险模块和调用危险函数。
        这是最后一道防线——插件仍然运行在宿主进程中。
        """
        import ast

        _FORBIDDEN_MODULES = {
            "os", "sys", "subprocess", "ctypes", "socket",
            "urllib", "http", "pickle", "marshal", "shutil",
            "pathlib", "tempfile", "multiprocessing", "builtins",
            "io", "operator", "inspect", "types", "code", "codeop",
        }
        _FORBIDDEN_CALLS = {
            "eval", "exec", "compile", "open", "input", "__import__",
            "getattr", "setattr", "delattr", "globals", "locals", "vars",
            "attrgetter", "itemgetter", "methodcaller",
        }
        _FORBIDDEN_ATTRS = {
            "__class__", "__bases__", "__base__", "__subclasses__",
            "__mro__", "__globals__", "__builtins__", "__dict__",
            "__code__", "__closure__", "f_globals", "f_locals",
        }

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Plugin syntax error in {filepath}: {e}")
            return False

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                else:
                    mod = (node.module or "").split(".")[0]
                    names = [mod] if mod else []
                for name in names:
                    if name in _FORBIDDEN_MODULES:
                        logger.warning(f"Plugin {filepath}: forbidden import '{name}'")
                        return False

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in _FORBIDDEN_CALLS:
                        logger.warning(f"Plugin {filepath}: forbidden call '{node.func.id}'")
                        return False
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in _FORBIDDEN_CALLS:
                        logger.warning(f"Plugin {filepath}: forbidden call '{node.func.attr}'")
                        return False

            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRS:
                    logger.warning(f"Plugin {filepath}: forbidden attribute '{node.attr}'")
                    return False

        return True

    def get_enabled_plugins_prompt(self) -> str:
        """生成供 Agent 使用的插件列表提示词。"""
        enabled = [p for p in self._plugins.values() if p.name in self._enabled]
        if not enabled:
            return ""
        lines = ["\n\n你有以下可用工具，当用户需要时可以使用它们："]
        for p in enabled:
            lines.append(f"- {p.name}: {p.description}")
        lines.append("\n使用工具时，请在回复中按以下格式调用：")
        lines.append("[tool: 工具名]\n参数: JSON 格式的参数")
        return "\n".join(lines)


# 全局注册表单例
_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """获取插件注册表单例。"""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def list_plugins() -> list[dict]:
    return get_registry().list_plugins()


def execute_plugin(name: str, args: dict) -> str:
    return get_registry().execute(name, args)


def get_plugins_prompt() -> str:
    return get_registry().get_enabled_plugins_prompt()
