"""MCP 工具 → Plugin 适配器。

把 MCP 服务器的工具包装成本地 Plugin 接口，让现有 Agent 可以直接调用。
"""

import logging

from core.plugin_system import Plugin, get_registry
from core.mcp_manager import list_mcp_tools, call_mcp_tool

logger = logging.getLogger(__name__)


class MCPToolPlugin(Plugin):
    """包装单个 MCP 工具的 Plugin"""

    def __init__(self, server_name: str, tool_name: str, description: str, schema: dict):
        self._server_name = server_name
        self._tool_name = tool_name
        self._schema = schema or {}
        super().__init__()

    @property
    def name(self) -> str:
        return f"mcp__{self._server_name}__{self._tool_name}"

    @property
    def description(self) -> str:
        return self._schema.get("description", f"MCP tool {self._tool_name} from {self._server_name}")

    @property
    def version(self) -> str:
        return "1.0.0"

    def execute(self, args: dict) -> str:
        return call_mcp_tool(f"{self._server_name}__{self._tool_name}", args)

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._schema,
        }


def register_mcp_plugins():
    """扫描所有 MCP 服务器，将其工具注册为 Plugin"""
    try:
        tools = list_mcp_tools()
        registry = get_registry()
        for t in tools:
            plugin = MCPToolPlugin(
                server_name=t["server"],
                tool_name=t["original_name"],
                description=t.get("description", ""),
                schema=t.get("schema", {}),
            )
            registry._plugins[plugin.name] = plugin
            logger.info(f"Registered MCP tool as plugin: {plugin.name}")
    except Exception as e:
        logger.warning(f"Failed to register MCP plugins: {e}")
