"""
MCP (Model Context Protocol) Base Infrastructure

Demonstrates:
- MCP protocol implementation
- Tool definition pattern
- Server abstraction
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MCP TYPES
# ============================================================================

class ToolParameterType(str, Enum):
    """Tool parameter types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class ToolParameter:
    """Tool parameter definition.

    Demonstrates:
    - Value object pattern
    - Type safety
    """
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        param = {
            "type": self.type.value,
            "description": self.description
        }
        if self.enum:
            param["enum"] = self.enum
        if self.default is not None:
            param["default"] = self.default
        return param


@dataclass
class Tool:
    """MCP Tool definition.

    Demonstrates:
    - Tool definition pattern
    - Schema-based validation
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_dict() for p in self.parameters}

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool result
        """
        if self.handler is None:
            raise NotImplementedError(f"No handler for tool: {self.name}")

        # Validate required parameters
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(kwargs.keys())
        missing = required_params - provided_params

        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Execute handler
        return await self.handler(**kwargs)


# ============================================================================
# MCP SERVER INTERFACE
# ============================================================================

class MCPServer(ABC):
    """Abstract MCP server.

    Demonstrates:
    - Abstract base class
    - Template method pattern
    """

    def __init__(self, name: str, description: str):
        """Initialize MCP server.

        Args:
            name: Server name
            description: Server description
        """
        self.name = name
        self.description = description
        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools (implemented by subclass)."""
        pass

    def register_tool(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} in server {self.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None
        """
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools.

        Returns:
            List of tool schemas
        """
        return [tool.to_schema() for tool in self.tools.values()]

    async def execute_tool(self, name: str, **parameters) -> Dict[str, Any]:
        """Execute a tool by name.

        Args:
            name: Tool name
            **parameters: Tool parameters

        Returns:
            Tool result
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        logger.info(f"Executing tool: {name} with params: {parameters}")
        result = await tool.execute(**parameters)
        logger.info(f"Tool {name} completed")

        return result

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information.

        Returns:
            Server info
        """
        return {
            "name": self.name,
            "description": self.description,
            "tools": self.list_tools(),
            "version": "1.0.0",
            "protocol": "mcp"
        }


# ============================================================================
# MCP CLIENT
# ============================================================================

class MCPClient:
    """MCP client for calling tools.

    Demonstrates:
    - Client pattern
    - Server registry
    """

    def __init__(self):
        """Initialize MCP client."""
        self.servers: Dict[str, MCPServer] = {}

    def register_server(self, server: MCPServer) -> None:
        """Register an MCP server.

        Args:
            server: MCP server instance
        """
        self.servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def list_servers(self) -> List[str]:
        """List registered servers.

        Returns:
            List of server names
        """
        return list(self.servers.keys())

    def get_server(self, name: str) -> Optional[MCPServer]:
        """Get server by name.

        Args:
            name: Server name

        Returns:
            Server or None
        """
        return self.servers.get(name)

    def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all tools from all servers.

        Returns:
            Dictionary of server -> tools
        """
        return {
            server_name: server.list_tools()
            for server_name, server in self.servers.items()
        }

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        **parameters
    ) -> Dict[str, Any]:
        """Call a tool on a specific server.

        Args:
            server_name: Server name
            tool_name: Tool name
            **parameters: Tool parameters

        Returns:
            Tool result
        """
        server = self.get_server(server_name)
        if server is None:
            raise ValueError(f"Unknown server: {server_name}")

        return await server.execute_tool(tool_name, **parameters)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    # Example tool handler
    async def example_handler(query: str, limit: int = 10) -> Dict[str, Any]:
        """Example tool handler."""
        return {
            "results": [f"Result {i}" for i in range(limit)],
            "query": query,
            "count": limit
        }

    # Create tool
    tool = Tool(
        name="search",
        description="Search for items",
        parameters=[
            ToolParameter("query", ToolParameterType.STRING, "Search query"),
            ToolParameter("limit", ToolParameterType.NUMBER, "Result limit", required=False, default=10)
        ],
        handler=example_handler
    )

    print("Tool Schema:")
    print(tool.to_schema())

    # Execute tool
    async def test():
        result = await tool.execute(query="test", limit=5)
        print("\nTool Result:")
        print(result)

    asyncio.run(test())
