---
title: "MCP Workflow"
---

# MCP Workflow

## Introduction

Understanding the MCP workflow is essential for building robust integrations. This lesson covers the complete lifecycle from tool discovery to execution, including response handling and error management.

### What We'll Cover

- Tool listing with mcp_list_tools
- Tool definitions from servers
- Executing tools with mcp_call
- Handling responses and errors
- Building complete workflows

### Prerequisites

- MCP fundamentals
- OpenAI Responses API
- Async programming concepts

---

## Tool Listing with mcp_list_tools

### Basic Tool Discovery

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

client = OpenAI()

# Request triggers tool listing
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "productivity",
        "server_url": "https://mcp.example.com/productivity",
        "require_approval": "never"
    }],
    input="What tools are available on the productivity server?"
)

# Parse tool listing response
for item in response.output:
    if hasattr(item, 'type') and item.type == 'mcp_list_tools':
        print(f"Server: {item.server_label}")
        print(f"Tools: {len(item.tools)}")
        for tool in item.tools:
            print(f"  - {tool.name}")
```

### Tool Listing Handler

```python
@dataclass
class ToolParameter:
    """Parameter for an MCP tool."""
    
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class DiscoveredTool:
    """A tool discovered from an MCP server."""
    
    name: str
    description: str
    parameters: List[ToolParameter]
    server_label: str
    
    @classmethod
    def from_api_response(cls, tool_data, server_label: str) -> 'DiscoveredTool':
        """Create from API response."""
        
        params = []
        schema = getattr(tool_data, 'parameters', {})
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get('type', 'string'),
                description=prop.get('description', ''),
                required=name in required,
                default=prop.get('default')
            ))
        
        return cls(
            name=tool_data.name,
            description=getattr(tool_data, 'description', ''),
            parameters=params,
            server_label=server_label
        )


class ToolListingHandler:
    """Handle tool listing responses."""
    
    def __init__(self):
        self.tools: Dict[str, List[DiscoveredTool]] = {}
    
    def process_response(self, response) -> Dict[str, List[DiscoveredTool]]:
        """Process response and extract tools."""
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_list_tools':
                server_label = item.server_label
                self.tools[server_label] = []
                
                for tool_data in item.tools:
                    tool = DiscoveredTool.from_api_response(
                        tool_data, 
                        server_label
                    )
                    self.tools[server_label].append(tool)
        
        return self.tools
    
    def get_tool(self, name: str) -> Optional[DiscoveredTool]:
        """Find a tool by name."""
        for server_tools in self.tools.values():
            for tool in server_tools:
                if tool.name == name:
                    return tool
        return None
    
    def format_for_display(self) -> str:
        """Format tools for display."""
        lines = []
        
        for server, tools in self.tools.items():
            lines.append(f"\n[{server}]")
            for tool in tools:
                lines.append(f"  {tool.name}: {tool.description}")
                for param in tool.parameters:
                    req = "*" if param.required else ""
                    lines.append(f"    - {param.name}{req}: {param.type}")
        
        return "\n".join(lines)


# Usage
handler = ToolListingHandler()
# tools = handler.process_response(response)
# print(handler.format_for_display())
```

---

## Tool Definitions from Servers

### Understanding Tool Schemas

```python
from typing import Union

@dataclass
class ToolSchema:
    """JSON Schema for a tool."""
    
    type: str = "object"
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False
    
    def validate(self, arguments: Dict[str, Any]) -> List[str]:
        """Validate arguments against schema."""
        errors = []
        
        # Check required
        for req in self.required:
            if req not in arguments:
                errors.append(f"Missing required parameter: {req}")
        
        # Check types
        for name, value in arguments.items():
            if name in self.properties:
                expected_type = self.properties[name].get('type')
                if not self._check_type(value, expected_type):
                    errors.append(f"Invalid type for {name}: expected {expected_type}")
        
        # Check for unknown properties
        if not self.additional_properties:
            for name in arguments:
                if name not in self.properties:
                    errors.append(f"Unknown parameter: {name}")
        
        return errors
    
    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Unknown type, allow
        
        return isinstance(value, expected_types)


class ToolDefinitionManager:
    """Manage tool definitions."""
    
    def __init__(self):
        self.definitions: Dict[str, ToolSchema] = {}
    
    def register(self, tool_name: str, schema: ToolSchema):
        """Register a tool definition."""
        self.definitions[tool_name] = schema
    
    def from_discovered_tool(self, tool: DiscoveredTool):
        """Create definition from discovered tool."""
        
        properties = {}
        required = []
        
        for param in tool.parameters:
            properties[param.name] = {
                'type': param.type,
                'description': param.description
            }
            if param.default is not None:
                properties[param.name]['default'] = param.default
            
            if param.required:
                required.append(param.name)
        
        schema = ToolSchema(
            properties=properties,
            required=required
        )
        
        self.register(tool.name, schema)
        return schema
    
    def validate_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> List[str]:
        """Validate a tool call."""
        
        if tool_name not in self.definitions:
            return [f"Unknown tool: {tool_name}"]
        
        return self.definitions[tool_name].validate(arguments)


# Usage
definition_manager = ToolDefinitionManager()

# Register tool schema
definition_manager.register("get_weather", ToolSchema(
    properties={
        "location": {"type": "string", "description": "City name"},
        "units": {"type": "string", "description": "celsius or fahrenheit"}
    },
    required=["location"]
))

# Validate call
errors = definition_manager.validate_call("get_weather", {
    "location": "San Francisco"
})
print(f"Validation errors: {errors}")  # []
```

---

## Executing Tools with mcp_call

### Basic Tool Execution

```python
# Model makes tool call
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "weather",
        "server_url": "https://mcp.example.com/weather",
        "require_approval": "never"
    }],
    input="What's the weather in Paris?"
)

# Check for tool calls
for item in response.output:
    if hasattr(item, 'type') and item.type == 'mcp_call':
        print(f"Tool: {item.name}")
        print(f"Arguments: {item.arguments}")
        print(f"Result: {item.result}")
```

### MCP Call Handler

```python
from enum import Enum
from datetime import datetime

class CallStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MCPCall:
    """Record of an MCP tool call."""
    
    id: str
    tool_name: str
    server_label: str
    arguments: Dict[str, Any]
    status: CallStatus = CallStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get call duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


class MCPCallExecutor:
    """Execute and track MCP calls."""
    
    def __init__(self):
        self.client = OpenAI()
        self.calls: List[MCPCall] = []
        self.call_counter = 0
    
    def execute(
        self,
        server_config: dict,
        prompt: str
    ) -> List[MCPCall]:
        """Execute a request that may trigger MCP calls."""
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[server_config],
            input=prompt
        )
        
        calls = []
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'mcp_call':
                self.call_counter += 1
                
                call = MCPCall(
                    id=f"call_{self.call_counter}",
                    tool_name=item.name,
                    server_label=server_config.get('server_label', 'unknown'),
                    arguments=getattr(item, 'arguments', {}),
                    status=CallStatus.COMPLETED,
                    result=getattr(item, 'result', None),
                    started_at=datetime.now(),
                    completed_at=datetime.now()
                )
                
                calls.append(call)
                self.calls.append(call)
        
        return calls
    
    def get_call_history(
        self,
        tool_name: Optional[str] = None,
        status: Optional[CallStatus] = None
    ) -> List[MCPCall]:
        """Get call history with optional filters."""
        
        result = self.calls
        
        if tool_name:
            result = [c for c in result if c.tool_name == tool_name]
        
        if status:
            result = [c for c in result if c.status == status]
        
        return result
    
    def get_statistics(self) -> dict:
        """Get call statistics."""
        
        if not self.calls:
            return {"message": "No calls recorded"}
        
        by_status = {}
        by_tool = {}
        durations = []
        
        for call in self.calls:
            # By status
            status = call.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # By tool
            by_tool[call.tool_name] = by_tool.get(call.tool_name, 0) + 1
            
            # Duration
            if call.duration_ms:
                durations.append(call.duration_ms)
        
        return {
            "total_calls": len(self.calls),
            "by_status": by_status,
            "by_tool": by_tool,
            "avg_duration_ms": sum(durations) / len(durations) if durations else None
        }


# Usage
executor = MCPCallExecutor()

# Execute request
# calls = executor.execute(
#     {
#         "type": "mcp",
#         "server_label": "weather",
#         "server_url": "https://mcp.example.com/weather",
#         "require_approval": "never"
#     },
#     "What's the weather in Tokyo?"
# )
# 
# for call in calls:
#     print(f"{call.tool_name}: {call.result}")
```

---

## Handling Responses and Errors

### Response Types

```python
class MCPResponseType(Enum):
    TEXT = "text"
    TOOL_LIST = "mcp_list_tools"
    TOOL_CALL = "mcp_call"
    APPROVAL_REQUEST = "mcp_approval_request"
    ERROR = "error"


@dataclass
class ParsedResponse:
    """Parsed MCP response."""
    
    response_type: MCPResponseType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPResponseParser:
    """Parse various MCP response types."""
    
    def parse(self, response) -> List[ParsedResponse]:
        """Parse response into structured format."""
        
        results = []
        
        # Check for text output
        if response.output_text:
            results.append(ParsedResponse(
                response_type=MCPResponseType.TEXT,
                content=response.output_text
            ))
        
        # Process output items
        for item in response.output:
            parsed = self._parse_item(item)
            if parsed:
                results.append(parsed)
        
        return results
    
    def _parse_item(self, item) -> Optional[ParsedResponse]:
        """Parse a single output item."""
        
        if not hasattr(item, 'type'):
            return None
        
        if item.type == 'mcp_list_tools':
            return ParsedResponse(
                response_type=MCPResponseType.TOOL_LIST,
                content=[{
                    "name": t.name,
                    "description": getattr(t, 'description', '')
                } for t in item.tools],
                metadata={"server": item.server_label}
            )
        
        elif item.type == 'mcp_call':
            return ParsedResponse(
                response_type=MCPResponseType.TOOL_CALL,
                content={
                    "tool": item.name,
                    "arguments": getattr(item, 'arguments', {}),
                    "result": getattr(item, 'result', None)
                },
                metadata={"server": getattr(item, 'server_label', '')}
            )
        
        elif item.type == 'mcp_approval_request':
            return ParsedResponse(
                response_type=MCPResponseType.APPROVAL_REQUEST,
                content={
                    "tool": item.name,
                    "arguments": getattr(item, 'arguments', {})
                },
                metadata={"approval_id": getattr(item, 'id', '')}
            )
        
        return None


# Usage
parser = MCPResponseParser()
# parsed = parser.parse(response)
# 
# for item in parsed:
#     print(f"Type: {item.response_type.value}")
#     print(f"Content: {item.content}")
```

### Error Handling

```python
class MCPErrorType(Enum):
    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


@dataclass
class MCPError:
    """Structured MCP error."""
    
    error_type: MCPErrorType
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_after: Optional[int] = None


class MCPErrorHandler:
    """Handle and recover from MCP errors."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_history: List[MCPError] = []
    
    def classify_error(self, exception: Exception) -> MCPError:
        """Classify an exception into MCPError."""
        
        error_str = str(exception).lower()
        
        if "connection" in error_str or "network" in error_str:
            return MCPError(
                error_type=MCPErrorType.CONNECTION_ERROR,
                message="Failed to connect to MCP server",
                recoverable=True
            )
        
        elif "401" in error_str or "403" in error_str or "auth" in error_str:
            return MCPError(
                error_type=MCPErrorType.AUTHENTICATION_ERROR,
                message="Authentication failed",
                recoverable=False
            )
        
        elif "429" in error_str or "rate" in error_str:
            return MCPError(
                error_type=MCPErrorType.RATE_LIMIT,
                message="Rate limit exceeded",
                recoverable=True,
                retry_after=60
            )
        
        elif "timeout" in error_str:
            return MCPError(
                error_type=MCPErrorType.TIMEOUT,
                message="Request timed out",
                recoverable=True
            )
        
        elif "not found" in error_str or "404" in error_str:
            return MCPError(
                error_type=MCPErrorType.TOOL_NOT_FOUND,
                message="Tool not found",
                recoverable=False
            )
        
        else:
            return MCPError(
                error_type=MCPErrorType.UNKNOWN,
                message=str(exception),
                recoverable=True
            )
    
    def handle_error(
        self,
        error: MCPError,
        retry_count: int = 0
    ) -> dict:
        """Handle an error and determine action."""
        
        self.error_history.append(error)
        
        action = {
            "error": error,
            "should_retry": False,
            "wait_seconds": 0,
            "fallback_action": None
        }
        
        if not error.recoverable:
            action["fallback_action"] = "notify_user"
            return action
        
        if retry_count >= self.max_retries:
            action["fallback_action"] = "use_cached_result"
            return action
        
        # Determine retry strategy
        if error.error_type == MCPErrorType.RATE_LIMIT:
            action["should_retry"] = True
            action["wait_seconds"] = error.retry_after or 60
        
        elif error.error_type == MCPErrorType.CONNECTION_ERROR:
            action["should_retry"] = True
            action["wait_seconds"] = 2 ** retry_count  # Exponential backoff
        
        elif error.error_type == MCPErrorType.TIMEOUT:
            action["should_retry"] = True
            action["wait_seconds"] = 5
        
        return action
    
    def get_error_summary(self) -> dict:
        """Get summary of errors."""
        
        by_type = {}
        for error in self.error_history:
            t = error.error_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_type": by_type,
            "recoverable_count": sum(1 for e in self.error_history if e.recoverable)
        }


# Usage
error_handler = MCPErrorHandler(max_retries=3)

# try:
#     result = executor.execute(...)
# except Exception as e:
#     error = error_handler.classify_error(e)
#     action = error_handler.handle_error(error)
#     
#     if action["should_retry"]:
#         time.sleep(action["wait_seconds"])
#         # Retry...
```

---

## Complete Workflow

### End-to-End MCP Workflow

```python
import time
from typing import Callable

class MCPWorkflow:
    """Complete MCP workflow manager."""
    
    def __init__(
        self,
        servers: List[dict],
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        self.client = OpenAI()
        self.servers = servers
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        
        self.parser = MCPResponseParser()
        self.error_handler = MCPErrorHandler(max_retries)
        
        self.discovered_tools: Dict[str, List[DiscoveredTool]] = {}
        self.call_history: List[MCPCall] = []
    
    def initialize(self) -> Dict[str, List[DiscoveredTool]]:
        """Initialize by discovering all tools."""
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=self.servers,
            input="List all available tools."
        )
        
        handler = ToolListingHandler()
        self.discovered_tools = handler.process_response(response)
        
        return self.discovered_tools
    
    def execute_query(
        self,
        query: str,
        on_tool_call: Callable[[MCPCall], None] = None
    ) -> dict:
        """Execute a query that may involve MCP tools."""
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                response = self.client.responses.create(
                    model="gpt-4o",
                    tools=self.servers,
                    input=query
                )
                
                # Parse response
                parsed = self.parser.parse(response)
                
                # Process tool calls
                calls = []
                for item in parsed:
                    if item.response_type == MCPResponseType.TOOL_CALL:
                        call = MCPCall(
                            id=f"call_{len(self.call_history) + 1}",
                            tool_name=item.content["tool"],
                            server_label=item.metadata.get("server", ""),
                            arguments=item.content["arguments"],
                            status=CallStatus.COMPLETED,
                            result=item.content["result"],
                            started_at=datetime.now(),
                            completed_at=datetime.now()
                        )
                        calls.append(call)
                        self.call_history.append(call)
                        
                        if on_tool_call:
                            on_tool_call(call)
                
                # Get text response
                text = ""
                for item in parsed:
                    if item.response_type == MCPResponseType.TEXT:
                        text = item.content
                        break
                
                return {
                    "success": True,
                    "text": text,
                    "tool_calls": calls,
                    "parsed_items": len(parsed)
                }
            
            except Exception as e:
                error = self.error_handler.classify_error(e)
                action = self.error_handler.handle_error(error, retry_count)
                
                if action["should_retry"]:
                    time.sleep(action["wait_seconds"])
                    retry_count += 1
                    continue
                
                return {
                    "success": False,
                    "error": error.message,
                    "error_type": error.error_type.value
                }
        
        return {
            "success": False,
            "error": "Max retries exceeded"
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        tools = []
        for server_tools in self.discovered_tools.values():
            tools.extend([t.name for t in server_tools])
        return tools
    
    def get_workflow_stats(self) -> dict:
        """Get workflow statistics."""
        
        return {
            "servers_configured": len(self.servers),
            "tools_discovered": sum(
                len(tools) for tools in self.discovered_tools.values()
            ),
            "total_calls": len(self.call_history),
            "successful_calls": sum(
                1 for c in self.call_history 
                if c.status == CallStatus.COMPLETED
            ),
            "error_summary": self.error_handler.get_error_summary()
        }


# Usage
workflow = MCPWorkflow(
    servers=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "https://mcp.example.com/weather",
            "require_approval": "never"
        },
        {
            "type": "mcp",
            "server_label": "calendar",
            "server_url": "https://mcp.example.com/calendar",
            "require_approval": "always"
        }
    ],
    max_retries=3
)

# Initialize
# tools = workflow.initialize()
# print(f"Discovered tools: {workflow.get_available_tools()}")

# Execute query
# result = workflow.execute_query(
#     "What's the weather in London and do I have any meetings today?",
#     on_tool_call=lambda c: print(f"Called: {c.tool_name}")
# )

# Get stats
# stats = workflow.get_workflow_stats()
# print(f"Total calls: {stats['total_calls']}")
```

---

## Hands-on Exercise

### Your Task

Build a complete MCP workflow system with error recovery.

### Requirements

1. Discover tools from multiple servers
2. Execute queries with automatic tool selection
3. Handle errors with retry logic
4. Track call history and statistics

<details>
<summary>💡 Hints</summary>

- Use exponential backoff for retries
- Cache discovered tools
- Track timing for calls
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import json

class WorkflowState(Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    
    query: str
    success: bool
    text_response: str
    tool_calls: List[dict]
    duration_ms: float
    retries: int
    error: Optional[str] = None


class RobustMCPWorkflow:
    """Robust MCP workflow with error recovery."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        tool_cache_ttl_minutes: int = 30
    ):
        self.client = OpenAI()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.cache_ttl = timedelta(minutes=tool_cache_ttl_minutes)
        
        self.servers: Dict[str, dict] = {}
        self.tool_cache: Dict[str, dict] = {}  # {server: {tools, cached_at}}
        self.call_history: List[dict] = []
        self.results: List[WorkflowResult] = []
        
        self.state = WorkflowState.IDLE
    
    def add_server(
        self,
        label: str,
        url: str,
        require_approval: str = "never",
        allowed_tools: List[str] = None
    ):
        """Add an MCP server."""
        
        config = {
            "type": "mcp",
            "server_label": label,
            "server_url": url,
            "require_approval": require_approval
        }
        
        if allowed_tools:
            config["allowed_tools"] = allowed_tools
        
        self.servers[label] = config
    
    def discover_tools(
        self,
        force_refresh: bool = False
    ) -> Dict[str, List[dict]]:
        """Discover tools from all servers."""
        
        self.state = WorkflowState.DISCOVERING
        
        # Check cache
        if not force_refresh:
            cached = self._get_cached_tools()
            if cached:
                self.state = WorkflowState.IDLE
                return cached
        
        # Discover from servers
        try:
            response = self.client.responses.create(
                model="gpt-4o",
                tools=list(self.servers.values()),
                input="List all available tools from all servers."
            )
            
            # Parse tools
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'mcp_list_tools':
                    server_label = item.server_label
                    tools = []
                    
                    for tool in item.tools:
                        tools.append({
                            "name": tool.name,
                            "description": getattr(tool, 'description', ''),
                            "parameters": getattr(tool, 'parameters', {})
                        })
                    
                    self.tool_cache[server_label] = {
                        "tools": tools,
                        "cached_at": datetime.now()
                    }
            
            self.state = WorkflowState.IDLE
            return self._get_cached_tools()
        
        except Exception as e:
            self.state = WorkflowState.ERROR
            raise
    
    def _get_cached_tools(self) -> Optional[Dict[str, List[dict]]]:
        """Get tools from cache if valid."""
        
        now = datetime.now()
        result = {}
        all_valid = True
        
        for server, cache_entry in self.tool_cache.items():
            if now - cache_entry["cached_at"] < self.cache_ttl:
                result[server] = cache_entry["tools"]
            else:
                all_valid = False
        
        return result if all_valid and result else None
    
    def execute(
        self,
        query: str,
        callback: Callable[[str, Any], None] = None
    ) -> WorkflowResult:
        """Execute a query with automatic retry."""
        
        start_time = time.time()
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            self.state = WorkflowState.EXECUTING
            
            try:
                response = self.client.responses.create(
                    model="gpt-4o",
                    tools=list(self.servers.values()),
                    input=query
                )
                
                # Parse response
                tool_calls = []
                text_response = response.output_text
                
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'mcp_call':
                        call = {
                            "tool": item.name,
                            "server": getattr(item, 'server_label', ''),
                            "arguments": getattr(item, 'arguments', {}),
                            "result": getattr(item, 'result', None),
                            "timestamp": datetime.now().isoformat()
                        }
                        tool_calls.append(call)
                        self.call_history.append(call)
                        
                        if callback:
                            callback("tool_call", call)
                
                duration_ms = (time.time() - start_time) * 1000
                
                result = WorkflowResult(
                    query=query,
                    success=True,
                    text_response=text_response,
                    tool_calls=tool_calls,
                    duration_ms=duration_ms,
                    retries=retries
                )
                
                self.results.append(result)
                self.state = WorkflowState.IDLE
                return result
            
            except Exception as e:
                last_error = str(e)
                retries += 1
                
                if retries <= self.max_retries:
                    delay = self._calculate_delay(retries, e)
                    self.state = WorkflowState.WAITING
                    
                    if callback:
                        callback("retry", {
                            "attempt": retries,
                            "delay": delay,
                            "error": last_error
                        })
                    
                    time.sleep(delay)
        
        # All retries failed
        duration_ms = (time.time() - start_time) * 1000
        
        result = WorkflowResult(
            query=query,
            success=False,
            text_response="",
            tool_calls=[],
            duration_ms=duration_ms,
            retries=retries,
            error=last_error
        )
        
        self.results.append(result)
        self.state = WorkflowState.ERROR
        return result
    
    def _calculate_delay(self, retry: int, error: Exception) -> float:
        """Calculate delay with exponential backoff."""
        
        error_str = str(error).lower()
        
        # Rate limit: longer delay
        if "429" in error_str or "rate" in error_str:
            base = 60.0
        else:
            base = self.base_delay
        
        delay = base * (2 ** (retry - 1))
        return min(delay, self.max_delay)
    
    def get_tool_by_name(self, name: str) -> Optional[dict]:
        """Find a tool by name."""
        
        for server_cache in self.tool_cache.values():
            for tool in server_cache["tools"]:
                if tool["name"] == name:
                    return tool
        return None
    
    def get_statistics(self) -> dict:
        """Get workflow statistics."""
        
        if not self.results:
            return {"message": "No queries executed"}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        durations = [r.duration_ms for r in successful]
        retries = [r.retries for r in self.results]
        
        tool_usage = {}
        for call in self.call_history:
            tool = call["tool"]
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return {
            "total_queries": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) * 100,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "total_retries": sum(retries),
            "tool_calls": len(self.call_history),
            "tool_usage": tool_usage,
            "servers_configured": len(self.servers),
            "tools_cached": sum(
                len(c["tools"]) for c in self.tool_cache.values()
            )
        }
    
    def export_history(self) -> str:
        """Export call history as JSON."""
        
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "servers": list(self.servers.keys()),
            "call_history": self.call_history,
            "results": [
                {
                    "query": r.query,
                    "success": r.success,
                    "tool_calls": len(r.tool_calls),
                    "duration_ms": r.duration_ms,
                    "retries": r.retries,
                    "error": r.error
                }
                for r in self.results
            ],
            "statistics": self.get_statistics()
        }, indent=2)


# Usage example
workflow = RobustMCPWorkflow(
    max_retries=3,
    base_delay=1.0,
    tool_cache_ttl_minutes=30
)

# Add servers
workflow.add_server(
    label="weather",
    url="https://mcp.example.com/weather",
    require_approval="never"
)

workflow.add_server(
    label="calendar",
    url="https://mcp.example.com/calendar",
    require_approval="always"
)

workflow.add_server(
    label="database",
    url="https://mcp.internal.corp/db",
    require_approval="never",
    allowed_tools=["query", "get_schema"]
)

# Discover tools
# tools = workflow.discover_tools()
# print(f"Discovered tools: {sum(len(t) for t in tools.values())}")

# Execute queries with callback
def on_event(event_type, data):
    if event_type == "tool_call":
        print(f"  Called: {data['tool']}")
    elif event_type == "retry":
        print(f"  Retrying ({data['attempt']}) after {data['delay']}s")

# result = workflow.execute(
#     "What's the weather in Paris and do I have any meetings tomorrow?",
#     callback=on_event
# )
# 
# print(f"Success: {result.success}")
# print(f"Response: {result.text_response}")
# print(f"Tools used: {len(result.tool_calls)}")

# Get statistics
stats = workflow.get_statistics()
print(f"\nWorkflow Statistics:")
print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
print(f"  Tool calls: {stats.get('tool_calls', 0)}")
print(f"  Servers: {stats.get('servers_configured', 0)}")
```

</details>

---

## Summary

✅ mcp_list_tools discovers available tools dynamically  
✅ Tool definitions include name, description, and parameters  
✅ mcp_call executes tools and returns results  
✅ Response parsing handles multiple output types  
✅ Error handling enables recovery from failures  
✅ Complete workflows combine all components

**Next:** [MCP Approval System](./06-mcp-approval-system.md)

---

## Further Reading

- [MCP Tool Discovery](https://platform.openai.com/docs/guides/mcp#tool-discovery) — Official documentation
- [MCP Error Handling](https://platform.openai.com/docs/guides/mcp#error-handling) — Error patterns
- [Building MCP Clients](https://modelcontextprotocol.io/quickstart/client) — Client development
