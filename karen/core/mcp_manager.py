"""MCP (Model Context Protocol) 管理器。

支持连接多个 MCP 服务器（stdio 和 sse 模式），列出工具、调用工具。
配置保存在 state/mcp_servers.json。
"""

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "state", "mcp_servers.json"
)

# 通过环境变量限制 stdio MCP 可启动的命令(防 add_server 任意命令注入)
# 默认拒绝所有命令，必须显式配置白名单才允许启动 MCP 服务器
# 建议设为如 "npx,uvx,python"
_DEFAULT_ALLOWED = "npx,uvx,python"
_mcp_env = os.getenv("MCP_ALLOWED_COMMANDS", _DEFAULT_ALLOWED)
# 如果环境变量显式设为空字符串，也表示拒绝所有（而非允许所有）
if _mcp_env.strip() == "":
    _MCP_ALLOWED = set()
else:
    _MCP_ALLOWED = {c.strip() for c in _mcp_env.split(",") if c.strip()}

# 透传给 MCP 子进程的最小环境变量集合 — 不含项目 API Key
# 防止恶意/失误的 MCP 拿到所有凭据
_MCP_ENV_PASSTHROUGH = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE",
    "USERPROFILE", "USERNAME", "HOMEPATH", "HOMEDRIVE",  # Windows
    "HOME", "USER", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",  # Windows 必需
}


def _build_mcp_env(extra: dict | None) -> dict:
    """为 MCP 子进程构造受限的环境变量。

    只透传上面白名单里的 + 用户在 mcp_servers.json 显式配置的 env。
    不再无条件继承宿主全部 env(里面可能有 OPENAI_API_KEY 等)。
    """
    base = {k: v for k, v in os.environ.items() if k in _MCP_ENV_PASSTHROUGH}
    if extra:
        base.update(extra)
    return base


def _load_config() -> dict:
    """加载 MCP 服务器配置"""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(data: dict):
    """保存 MCP 服务器配置"""
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class MCPManager:
    """MCP 服务器管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._servers = {}
            cls._instance._load()
        return cls._instance

    def _load(self):
        """从配置文件加载服务器列表"""
        data = _load_config()
        self._servers = data.get("servers", {})

    def _save(self):
        """保存到配置文件"""
        _save_config({"servers": self._servers})

    # ===== 服务器管理 =====

    def list_servers(self) -> list[dict]:
        """列出所有服务器配置"""
        result = []
        for name, cfg in self._servers.items():
            result.append({
                "name": name,
                "transport": cfg.get("transport", "stdio"),
                "command": cfg.get("command", ""),
                "args": cfg.get("args", []),
                "url": cfg.get("url", ""),
                "enabled": cfg.get("enabled", True),
            })
        return result

    def add_server(self, name: str, command: str = "", args: list = None,
                   env: dict = None, url: str = "", transport: str = "stdio") -> bool:
        """添加/更新服务器配置"""
        # 命令白名单校验:防止远程攻击者(在 LOCAL_ONLY 误绕过等场景下)注入任意 shell 命令
        if transport == "stdio" and command and _MCP_ALLOWED:
            cmd_basename = os.path.basename(command).lower()
            if cmd_basename not in _MCP_ALLOWED and command not in _MCP_ALLOWED:
                raise ValueError(
                    f"命令 '{command}' 不在 MCP_ALLOWED_COMMANDS 白名单中"
                )
        self._servers[name] = {
            "transport": transport,
            "command": command,
            "args": args or [],
            "env": env or {},
            "url": url,
            "enabled": True,
        }
        self._save()
        return True

    def remove_server(self, name: str) -> bool:
        """删除服务器配置"""
        if name in self._servers:
            del self._servers[name]
            self._save()
            return True
        return False

    def toggle_server(self, name: str, enabled: bool) -> bool:
        """启用/禁用服务器"""
        if name in self._servers:
            self._servers[name]["enabled"] = enabled
            self._save()
            return True
        return False

    # ===== 工具操作 =====

    def list_tools(self, server_name: str) -> list[dict]:
        """列出指定服务器的所有工具（同步包装）"""
        return self._run_async(self._alist_tools(server_name))

    def list_all_tools(self) -> list[dict]:
        """列出所有服务器的所有工具"""
        all_tools = []
        for name in self._servers:
            if not self._servers[name].get("enabled", True):
                continue
            try:
                tools = self.list_tools(name)
                for t in tools:
                    all_tools.append({
                        "name": f"{name}__{t['name']}",
                        "original_name": t["name"],
                        "server": name,
                        "description": t.get("description", ""),
                        "schema": t.get("inputSchema", {}),
                    })
            except Exception as e:
                logger.warning(f"Failed to list tools from MCP server '{name}': {e}")
        return all_tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """调用指定服务器的工具（同步包装）"""
        return self._run_async(self._acall_tool(server_name, tool_name, arguments))

    # ===== 异步内部方法 =====

    def _run_async(self, coro):
        """在线程中安全运行异步协程"""
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # 如果当前线程已有事件循环
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    async def _alist_tools(self, server_name: str) -> list[dict]:
        """异步：列出工具"""
        cfg = self._servers.get(server_name)
        if not cfg:
            raise ValueError(f"MCP server '{server_name}' not found")

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client

        transport = cfg.get("transport", "stdio")

        try:
            if transport == "sse":
                url = cfg.get("url", "")
                if not url:
                    raise ValueError("SSE URL not set")
                async with sse_client(url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        return [t.model_dump() for t in result.tools]
            else:
                # stdio 模式
                params = StdioServerParameters(
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=_build_mcp_env(cfg.get("env", {})),
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        return [t.model_dump() for t in result.tools]
        except Exception:
            logger.exception(f"Failed to connect MCP server '{server_name}'")
            return []

    async def _acall_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """异步：调用工具"""
        cfg = self._servers.get(server_name)
        if not cfg:
            raise ValueError(f"MCP server '{server_name}' not found")

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client

        transport = cfg.get("transport", "stdio")

        try:
            if transport == "sse":
                url = cfg.get("url", "")
                async with sse_client(url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        return self._format_tool_result(result)
            else:
                params = StdioServerParameters(
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    env=_build_mcp_env(cfg.get("env", {})),
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        return self._format_tool_result(result)
        except Exception as e:
            logger.exception(f"MCP tool call failed: {server_name}/{tool_name}")
            return f"[MCP 工具调用失败: {str(e)}]"

    @staticmethod
    def _format_tool_result(result) -> str:
        """格式化工具返回结果"""
        lines = []
        for content in result.content:
            if getattr(content, "type", None) == "text":
                lines.append(content.text)
            else:
                lines.append(str(content))
        return "\n".join(lines) if lines else "[工具无输出]"


# 全局实例
_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager:
    """获取 MCP 管理器实例"""
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager


def list_mcp_tools() -> list[dict]:
    """列出所有 MCP 工具"""
    return get_mcp_manager().list_all_tools()


def call_mcp_tool(full_name: str, arguments: dict) -> str:
    """调用 MCP 工具（full_name 格式: server__tool_name）"""
    parts = full_name.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid MCP tool name: {full_name}, expected 'server__tool_name'")
    server_name, tool_name = parts
    return get_mcp_manager().call_tool(server_name, tool_name, arguments)
