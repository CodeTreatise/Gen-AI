---
title: "MCP Fundamentals"
---

# MCP Fundamentals

## Introduction

The Model Context Protocol (MCP) enables AI models to interact with external tools and services through a standardized interface. MCP servers expose tools that models can discover and invoke, extending capabilities beyond built-in functions.

### What We'll Cover

- What MCP is and why it matters
- Remote MCP server integration
- Server URL configuration
- Transport mechanisms (HTTP, SSE)
- Tool discovery and invocation

### Prerequisites

- Understanding of API communication
- OpenAI Responses API familiarity
- Basic server concepts

---

## What is MCP?

### The Protocol Overview

```python
"""
Model Context Protocol (MCP) enables:
1. Dynamic tool discovery
2. Standardized tool invocation
3. Remote server integration
4. Cross-platform compatibility
"""

# MCP architecture diagram
"""
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  AI Model   │────▶│ MCP Server  │
│   (Your     │     │  (GPT-4o)   │     │  (Remote)   │
│    App)     │◀────│             │◀────│             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   Tools   │
                    │ - weather │
                    │ - search  │
                    │ - database│
                    └───────────┘
"""
```

### MCP vs Function Calling

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class ToolExecutionModel(Enum):
    FUNCTION_CALLING = "function_calling"
    MCP = "mcp"
    BUILT_IN = "built_in"


@dataclass
class ToolComparison:
    """Comparison of tool execution models."""
    
    model: ToolExecutionModel
    execution_location: str
    discovery: str
    setup_complexity: str
    use_case: str


TOOL_MODELS = [
    ToolComparison(
        model=ToolExecutionModel.FUNCTION_CALLING,
        execution_location="Your server",
        discovery="Static (you define)",
        setup_complexity="Medium",
        use_case="Custom business logic"
    ),
    ToolComparison(
        model=ToolExecutionModel.MCP,
        execution_location="Remote MCP server",
        discovery="Dynamic (server provides)",
        setup_complexity="Low",
        use_case="Third-party integrations"
    ),
    ToolComparison(
        model=ToolExecutionModel.BUILT_IN,
        execution_location="OpenAI servers",
        discovery="Static (predefined)",
        setup_complexity="Very low",
        use_case="Common capabilities"
    )
]


def print_comparison():
    """Print tool model comparison."""
    print("Tool Execution Models Comparison:")
    print("-" * 60)
    for tool in TOOL_MODELS:
        print(f"\n{tool.model.value}:")
        print(f"  Execution: {tool.execution_location}")
        print(f"  Discovery: {tool.discovery}")
        print(f"  Complexity: {tool.setup_complexity}")
        print(f"  Use case: {tool.use_case}")
```

---

## Remote MCP Server Integration

### Basic Server Connection

```python
from openai import OpenAI

client = OpenAI()

# Connect to an MCP server
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "weather-server",
        "server_url": "https://mcp.example.com/weather",
        "require_approval": "never"
    }],
    input="What's the weather in San Francisco?"
)

print(response.output_text)
```

### MCP Server Configuration

```python
@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    label: str
    url: str
    require_approval: str = "never"  # never, always, auto
    allowed_tools: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    
    def to_tool(self) -> dict:
        """Convert to tool definition."""
        config = {
            "type": "mcp",
            "server_label": self.label,
            "server_url": self.url,
            "require_approval": self.require_approval
        }
        
        if self.allowed_tools:
            config["allowed_tools"] = self.allowed_tools
        
        return config


class MCPClient:
    """Client for MCP server interactions."""
    
    def __init__(self, servers: List[MCPServerConfig] = None):
        self.client = OpenAI()
        self.servers = servers or []
    
    def add_server(self, server: MCPServerConfig):
        """Add an MCP server."""
        self.servers.append(server)
    
    def query(
        self,
        prompt: str,
        model: str = "gpt-4o"
    ) -> dict:
        """Query with all configured MCP servers."""
        
        tools = [server.to_tool() for server in self.servers]
        
        response = self.client.responses.create(
            model=model,
            tools=tools,
            input=prompt
        )
        
        return {
            "output": response.output_text,
            "model": response.model,
            "servers_available": len(self.servers)
        }


# Usage
mcp_client = MCPClient()

mcp_client.add_server(MCPServerConfig(
    label="weather",
    url="https://mcp.example.com/weather",
    require_approval="never"
))

mcp_client.add_server(MCPServerConfig(
    label="calendar",
    url="https://mcp.example.com/calendar",
    require_approval="always",
    allowed_tools=["get_events", "create_event"]
))

# result = mcp_client.query("What's my schedule for today?")
```

---

## Server URL Configuration

### URL Patterns

```python
from urllib.parse import urlparse, urljoin
from typing import Optional

@dataclass
class MCPEndpoint:
    """MCP server endpoint configuration."""
    
    base_url: str
    path: str = ""
    api_version: Optional[str] = None
    
    def get_url(self) -> str:
        """Get full server URL."""
        url = self.base_url.rstrip("/")
        
        if self.api_version:
            url = f"{url}/{self.api_version}"
        
        if self.path:
            url = f"{url}/{self.path.lstrip('/')}"
        
        return url
    
    def validate(self) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(self.get_url())
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False


class MCPServerRegistry:
    """Registry for MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, MCPEndpoint] = {}
    
    def register(
        self,
        label: str,
        base_url: str,
        path: str = "",
        api_version: str = None
    ):
        """Register an MCP server."""
        
        endpoint = MCPEndpoint(
            base_url=base_url,
            path=path,
            api_version=api_version
        )
        
        if not endpoint.validate():
            raise ValueError(f"Invalid URL for server {label}")
        
        self.servers[label] = endpoint
    
    def get_url(self, label: str) -> str:
        """Get URL for registered server."""
        
        if label not in self.servers:
            raise KeyError(f"Server not found: {label}")
        
        return self.servers[label].get_url()
    
    def list_servers(self) -> List[dict]:
        """List all registered servers."""
        return [
            {"label": label, "url": endpoint.get_url()}
            for label, endpoint in self.servers.items()
        ]


# Usage
registry = MCPServerRegistry()

registry.register(
    label="weather",
    base_url="https://api.example.com",
    path="mcp/weather",
    api_version="v1"
)

registry.register(
    label="database",
    base_url="https://mcp.internal.corp",
    path="tools"
)

print(registry.get_url("weather"))
# https://api.example.com/v1/mcp/weather
```

### Environment-Based Configuration

```python
import os
from typing import Dict

class MCPEnvironmentConfig:
    """Load MCP configuration from environment."""
    
    def __init__(self, prefix: str = "MCP_SERVER_"):
        self.prefix = prefix
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_from_env()
    
    def _load_from_env(self):
        """Load server configs from environment variables."""
        
        # Expected format: MCP_SERVER_<LABEL>_URL, MCP_SERVER_<LABEL>_APPROVAL
        server_labels = set()
        
        for key in os.environ:
            if key.startswith(self.prefix):
                parts = key[len(self.prefix):].split("_")
                if len(parts) >= 2:
                    label = parts[0].lower()
                    server_labels.add(label)
        
        for label in server_labels:
            url_key = f"{self.prefix}{label.upper()}_URL"
            approval_key = f"{self.prefix}{label.upper()}_APPROVAL"
            tools_key = f"{self.prefix}{label.upper()}_TOOLS"
            
            url = os.environ.get(url_key)
            if url:
                approval = os.environ.get(approval_key, "never")
                tools = os.environ.get(tools_key, "").split(",")
                tools = [t.strip() for t in tools if t.strip()]
                
                self.servers[label] = MCPServerConfig(
                    label=label,
                    url=url,
                    require_approval=approval,
                    allowed_tools=tools if tools else None
                )
    
    def get_server(self, label: str) -> Optional[MCPServerConfig]:
        """Get server configuration."""
        return self.servers.get(label)
    
    def get_all_tools(self) -> List[dict]:
        """Get tool definitions for all servers."""
        return [server.to_tool() for server in self.servers.values()]


# Usage with environment variables:
# export MCP_SERVER_WEATHER_URL=https://mcp.example.com/weather
# export MCP_SERVER_WEATHER_APPROVAL=never
# export MCP_SERVER_CALENDAR_URL=https://mcp.example.com/calendar
# export MCP_SERVER_CALENDAR_TOOLS=get_events,create_event

# config = MCPEnvironmentConfig()
# tools = config.get_all_tools()
```

---

## Transport Mechanisms

### HTTP Transport

```python
import httpx
from typing import AsyncIterator

@dataclass
class HTTPTransportConfig:
    """Configuration for HTTP transport."""
    
    timeout: float = 30.0
    max_retries: int = 3
    headers: Dict[str, str] = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.headers:
            headers.update(self.headers)
        return headers


class HTTPMCPTransport:
    """HTTP transport for MCP communication."""
    
    def __init__(self, config: HTTPTransportConfig = None):
        self.config = config or HTTPTransportConfig()
        self.client = httpx.Client(
            timeout=self.config.timeout,
            headers=self.config.get_headers()
        )
    
    def list_tools(self, server_url: str) -> List[dict]:
        """List available tools from server."""
        
        response = self.client.get(f"{server_url}/tools")
        response.raise_for_status()
        
        return response.json().get("tools", [])
    
    def call_tool(
        self,
        server_url: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> dict:
        """Call a tool on the server."""
        
        response = self.client.post(
            f"{server_url}/tools/{tool_name}",
            json={"arguments": arguments}
        )
        response.raise_for_status()
        
        return response.json()
    
    def close(self):
        """Close the client."""
        self.client.close()


# Usage
transport = HTTPMCPTransport(HTTPTransportConfig(
    timeout=30.0,
    headers={"Authorization": "Bearer token"}
))

# tools = transport.list_tools("https://mcp.example.com")
# result = transport.call_tool(
#     "https://mcp.example.com",
#     "get_weather",
#     {"location": "San Francisco"}
# )
```

### Server-Sent Events (SSE) Transport

```python
import json
from typing import Generator

class SSEMCPTransport:
    """SSE transport for streaming MCP responses."""
    
    def __init__(self, config: HTTPTransportConfig = None):
        self.config = config or HTTPTransportConfig()
    
    def stream_response(
        self,
        server_url: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Generator[dict, None, None]:
        """Stream tool response via SSE."""
        
        with httpx.stream(
            "POST",
            f"{server_url}/tools/{tool_name}/stream",
            json={"arguments": arguments},
            headers={
                **self.config.get_headers(),
                "Accept": "text/event-stream"
            },
            timeout=self.config.timeout
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield json.loads(data)
    
    def stream_to_completion(
        self,
        server_url: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> dict:
        """Stream and collect full response."""
        
        chunks = []
        for chunk in self.stream_response(server_url, tool_name, arguments):
            chunks.append(chunk)
        
        # Combine chunks
        if not chunks:
            return {}
        
        # Assume last chunk has final result
        return chunks[-1] if chunks else {}


# Usage
sse_transport = SSEMCPTransport()

# for chunk in sse_transport.stream_response(
#     "https://mcp.example.com",
#     "long_running_task",
#     {"query": "analyze data"}
# ):
#     print(chunk)
```

### Streamable HTTP (Modern Transport)

```python
class StreamableHTTPTransport:
    """Streamable HTTP transport for bidirectional communication."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id: Optional[str] = None
    
    async def connect(self) -> str:
        """Establish connection and get session ID."""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/connect",
                json={}
            )
            response.raise_for_status()
            
            data = response.json()
            self.session_id = data.get("session_id")
            return self.session_id
    
    async def send_message(self, message: dict) -> dict:
        """Send message to server."""
        
        if not self.session_id:
            await self.connect()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/message",
                json={
                    "session_id": self.session_id,
                    "message": message
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def list_tools(self) -> List[dict]:
        """List available tools."""
        
        result = await self.send_message({
            "type": "list_tools"
        })
        return result.get("tools", [])
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> dict:
        """Call a tool."""
        
        return await self.send_message({
            "type": "call_tool",
            "tool": tool_name,
            "arguments": arguments
        })


# Usage
# transport = StreamableHTTPTransport("https://mcp.example.com")
# await transport.connect()
# tools = await transport.list_tools()
# result = await transport.call_tool("get_weather", {"location": "NYC"})
```

---

## Tool Discovery

### Discovering Available Tools

```python
from openai import OpenAI

client = OpenAI()

# Request tool discovery
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "tools-server",
        "server_url": "https://mcp.example.com",
        "require_approval": "never"
    }],
    input="What tools are available?"
)

# Check for mcp_list_tools call
for item in response.output:
    if hasattr(item, 'type') and item.type == 'mcp_list_tools':
        print(f"Server: {item.server_label}")
        for tool in item.tools:
            print(f"  - {tool.name}: {tool.description}")
```

### Tool Discovery Manager

```python
@dataclass
class MCPTool:
    """Discovered MCP tool."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    server_label: str


class ToolDiscoveryManager:
    """Manage tool discovery across MCP servers."""
    
    def __init__(self):
        self.client = OpenAI()
        self.discovered_tools: Dict[str, List[MCPTool]] = {}
    
    def discover(
        self,
        servers: List[MCPServerConfig]
    ) -> Dict[str, List[MCPTool]]:
        """Discover tools from all servers."""
        
        tools_config = [server.to_tool() for server in servers]
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=tools_config,
            input="List all available tools from all servers."
        )
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_list_tools':
                server_label = item.server_label
                self.discovered_tools[server_label] = []
                
                for tool in item.tools:
                    self.discovered_tools[server_label].append(MCPTool(
                        name=tool.name,
                        description=getattr(tool, 'description', ''),
                        parameters=getattr(tool, 'parameters', {}),
                        server_label=server_label
                    ))
        
        return self.discovered_tools
    
    def find_tool(self, name: str) -> Optional[MCPTool]:
        """Find a tool by name across all servers."""
        
        for server_tools in self.discovered_tools.values():
            for tool in server_tools:
                if tool.name == name:
                    return tool
        return None
    
    def search_tools(self, query: str) -> List[MCPTool]:
        """Search tools by description."""
        
        query_lower = query.lower()
        results = []
        
        for server_tools in self.discovered_tools.values():
            for tool in server_tools:
                if query_lower in tool.name.lower() or \
                   query_lower in tool.description.lower():
                    results.append(tool)
        
        return results
    
    def get_tools_by_server(self, server_label: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        return self.discovered_tools.get(server_label, [])


# Usage
discovery = ToolDiscoveryManager()

# servers = [
#     MCPServerConfig(label="weather", url="https://mcp.example.com/weather"),
#     MCPServerConfig(label="database", url="https://mcp.example.com/database")
# ]
# 
# all_tools = discovery.discover(servers)
# 
# # Find specific tool
# weather_tool = discovery.find_tool("get_forecast")
# 
# # Search for tools
# data_tools = discovery.search_tools("data")
```

---

## Hands-on Exercise

### Your Task

Build an MCP client with dynamic tool discovery.

### Requirements

1. Configure multiple MCP servers
2. Implement tool discovery
3. Cache discovered tools
4. Create tool invocation helper

<details>
<summary>💡 Hints</summary>

- Use server labels to identify sources
- Cache tools to avoid repeated discovery
- Handle connection errors gracefully
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

@dataclass
class CachedTool:
    """Tool with cache metadata."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    server_label: str
    cached_at: datetime
    
    def is_expired(self, ttl_minutes: int = 60) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.cached_at > timedelta(minutes=ttl_minutes)


@dataclass
class MCPServerStatus:
    """Status of an MCP server."""
    
    label: str
    url: str
    is_connected: bool
    last_check: datetime
    tool_count: int
    error: Optional[str] = None


class MCPToolManager:
    """Complete MCP tool management system."""
    
    def __init__(self, cache_ttl_minutes: int = 60):
        self.client = OpenAI()
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tool_cache: Dict[str, List[CachedTool]] = {}
        self.server_status: Dict[str, MCPServerStatus] = {}
        self.cache_ttl = cache_ttl_minutes
    
    def register_server(self, config: MCPServerConfig):
        """Register an MCP server."""
        self.servers[config.label] = config
        self.server_status[config.label] = MCPServerStatus(
            label=config.label,
            url=config.url,
            is_connected=False,
            last_check=datetime.now(),
            tool_count=0
        )
    
    def discover_tools(
        self,
        server_label: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, List[CachedTool]]:
        """Discover tools from servers."""
        
        # Check cache
        if not force_refresh:
            cached = self._get_from_cache(server_label)
            if cached:
                return cached
        
        # Select servers to query
        if server_label:
            servers_to_query = [self.servers[server_label]]
        else:
            servers_to_query = list(self.servers.values())
        
        # Build tools config
        tools_config = [s.to_tool() for s in servers_to_query]
        
        try:
            response = self.client.responses.create(
                model="gpt-4o",
                tools=tools_config,
                input="List all available tools from all connected servers."
            )
            
            # Parse response
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'mcp_list_tools':
                    label = item.server_label
                    self.tool_cache[label] = []
                    
                    for tool in item.tools:
                        self.tool_cache[label].append(CachedTool(
                            name=tool.name,
                            description=getattr(tool, 'description', ''),
                            parameters=getattr(tool, 'parameters', {}),
                            server_label=label,
                            cached_at=datetime.now()
                        ))
                    
                    # Update status
                    self.server_status[label].is_connected = True
                    self.server_status[label].tool_count = len(self.tool_cache[label])
                    self.server_status[label].last_check = datetime.now()
                    self.server_status[label].error = None
        
        except Exception as e:
            for server in servers_to_query:
                self.server_status[server.label].is_connected = False
                self.server_status[server.label].error = str(e)
                self.server_status[server.label].last_check = datetime.now()
        
        return self.tool_cache
    
    def _get_from_cache(
        self,
        server_label: Optional[str]
    ) -> Optional[Dict[str, List[CachedTool]]]:
        """Get tools from cache if valid."""
        
        if server_label:
            tools = self.tool_cache.get(server_label, [])
            if tools and not tools[0].is_expired(self.cache_ttl):
                return {server_label: tools}
            return None
        
        # Check all cached
        result = {}
        all_valid = True
        
        for label, tools in self.tool_cache.items():
            if tools and not tools[0].is_expired(self.cache_ttl):
                result[label] = tools
            else:
                all_valid = False
        
        return result if all_valid and result else None
    
    def get_tool(self, tool_name: str) -> Optional[CachedTool]:
        """Get a specific tool."""
        for tools in self.tool_cache.values():
            for tool in tools:
                if tool.name == tool_name:
                    return tool
        return None
    
    def invoke_tool(
        self,
        tool_name: str,
        prompt: str
    ) -> dict:
        """Invoke a tool through the model."""
        
        # Find the tool's server
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool not found: {tool_name}"}
        
        server = self.servers.get(tool.server_label)
        if not server:
            return {"error": f"Server not found: {tool.server_label}"}
        
        # Invoke
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[server.to_tool()],
            input=prompt
        )
        
        # Parse result
        result = {
            "tool": tool_name,
            "server": tool.server_label,
            "output": response.output_text
        }
        
        # Check for tool calls
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_call':
                result["tool_call"] = {
                    "name": getattr(item, 'name', ''),
                    "arguments": getattr(item, 'arguments', {})
                }
        
        return result
    
    def get_all_tools(self) -> List[CachedTool]:
        """Get all cached tools."""
        all_tools = []
        for tools in self.tool_cache.values():
            all_tools.extend(tools)
        return all_tools
    
    def get_status_report(self) -> dict:
        """Get status report for all servers."""
        return {
            "servers": [
                {
                    "label": status.label,
                    "url": status.url,
                    "connected": status.is_connected,
                    "tools": status.tool_count,
                    "last_check": status.last_check.isoformat(),
                    "error": status.error
                }
                for status in self.server_status.values()
            ],
            "total_tools": sum(s.tool_count for s in self.server_status.values()),
            "cache_ttl_minutes": self.cache_ttl
        }
    
    def search_tools(self, query: str) -> List[CachedTool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        return [
            tool for tool in self.get_all_tools()
            if query_lower in tool.name.lower() or 
               query_lower in tool.description.lower()
        ]


# Usage example
manager = MCPToolManager(cache_ttl_minutes=30)

# Register servers
manager.register_server(MCPServerConfig(
    label="weather",
    url="https://mcp.example.com/weather",
    require_approval="never"
))

manager.register_server(MCPServerConfig(
    label="calendar",
    url="https://mcp.example.com/calendar",
    require_approval="always"
))

manager.register_server(MCPServerConfig(
    label="database",
    url="https://mcp.internal.corp/db",
    require_approval="auto",
    allowed_tools=["query", "get_schema"]
))

# Discover all tools
# all_tools = manager.discover_tools()
# print(f"Discovered {len(manager.get_all_tools())} tools")

# Find specific tool
# weather_tool = manager.get_tool("get_forecast")
# if weather_tool:
#     print(f"Found: {weather_tool.name} on {weather_tool.server_label}")

# Search tools
# data_tools = manager.search_tools("data")
# print(f"Found {len(data_tools)} data-related tools")

# Invoke tool
# result = manager.invoke_tool(
#     "get_forecast",
#     "What's the weather forecast for New York this week?"
# )
# print(result)

# Status report
status = manager.get_status_report()
print(f"Connected servers: {sum(1 for s in status['servers'] if s['connected'])}")
print(f"Total tools available: {status['total_tools']}")
```

</details>

---

## Summary

✅ MCP standardizes AI-tool communication  
✅ Remote servers expose tools dynamically  
✅ Server URLs use HTTP/HTTPS protocols  
✅ Transport options include HTTP, SSE, and Streamable HTTP  
✅ Tool discovery reveals available capabilities  
✅ Caching improves discovery performance

**Next:** [MCP Workflow](./05-mcp-workflow.md)

---

## Further Reading

- [OpenAI MCP Integration](https://platform.openai.com/docs/guides/mcp) — Official documentation
- [MCP Specification](https://modelcontextprotocol.io/) — Protocol specification
- [Building MCP Servers](https://modelcontextprotocol.io/quickstart/server) — Server development guide
