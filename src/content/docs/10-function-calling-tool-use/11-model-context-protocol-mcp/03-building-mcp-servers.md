---
title: "Building MCP Servers"
---

# Building MCP Servers

## Introduction

Now that we understand MCP's architecture and primitives, it's time to build a real MCP server. The Python MCP SDK provides `FastMCP` — a high-level class that uses Python type hints and docstrings to automatically generate tool definitions. This means you can go from a Python function to a fully discoverable MCP tool in just a few lines of code.

In this sub-lesson, we build a complete MCP server from scratch, register tools with proper schemas, run the server, and test it using the MCP Inspector.

### What we'll cover

- Setting up a Python MCP project with `uv`
- The `FastMCP` class and server initialization
- Registering tools with the `@mcp.tool()` decorator
- Handling async operations in tool implementations
- Running and testing MCP servers
- Logging best practices for stdio servers

### Prerequisites

- Completed [Server Primitives](./02-server-primitives.md)
- Python 3.10+ installed
- Familiarity with Python async/await and type hints
- `uv` package manager (recommended) or `pip`

---

## Setting up the project

The recommended way to create an MCP server project is using `uv`, the fast Python package manager:

```bash
# Create a new project directory
uv init my-mcp-server
cd my-mcp-server

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install the MCP SDK with CLI tools
uv add "mcp[cli]"

# Create the server file
touch server.py
```

> **Note:** You need Python 3.10 or higher and MCP SDK version 1.2.0+. The `mcp[cli]` extra installs command-line tools for testing and debugging.

---

## The FastMCP class

`FastMCP` is the primary entry point for building MCP servers. It handles protocol negotiation, message routing, and automatic schema generation:

```python
from mcp.server.fastmcp import FastMCP

# Initialize the server with a name
mcp = FastMCP("my-server")
```

The server name is used during the initialization handshake — clients see it in the `serverInfo` response. Choose a descriptive name that identifies what your server does.

---

## Registering tools with `@mcp.tool()`

The `@mcp.tool()` decorator converts a Python function into an MCP tool. The SDK automatically generates the JSON Schema from your function's **type hints** and **docstring**:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: The first number
        b: The second number
    """
    result = a + b
    return f"{a} + {b} = {result}"
```

**What happens behind the scenes:**

1. The decorator reads the function signature: `a: float, b: float`
2. It parses the docstring for parameter descriptions
3. It generates a JSON Schema tool definition:

```json
{
  "name": "add",
  "description": "Add two numbers together.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": { "type": "number", "description": "The first number" },
      "b": { "type": "number", "description": "The second number" }
    },
    "required": ["a", "b"]
  }
}
```

> **Tip:** Always include type hints and docstrings. Without them, the generated schema is vague and the LLM won't know how to use your tool effectively.

### Multiple tools on one server

A single server can expose as many tools as needed:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math-tools")

@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: The first number
        b: The second number
    """
    return f"{a} + {b} = {a + b}"

@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: The first number
        b: The second number
    """
    return f"{a} × {b} = {a * b}"

@mcp.tool()
def factorial(n: int) -> str:
    """Calculate the factorial of a non-negative integer.

    Args:
        n: A non-negative integer (0-20 for practical limits)
    """
    if n < 0:
        return "Error: factorial is not defined for negative numbers"
    if n > 20:
        return "Error: input too large (max 20)"
    result = 1
    for i in range(2, n + 1):
        result *= i
    return f"{n}! = {result}"
```

---

## Async tools with external API calls

Most practical MCP tools interact with external services. Use `async` functions with `httpx` for non-blocking HTTP requests:

```python
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-mcp-server/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get active weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g., CA, NY, TX)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return f"No active alerts for {state}."

    alerts = []
    for feature in data["features"]:
        props = feature["properties"]
        alert = (
            f"Event: {props.get('event', 'Unknown')}\n"
            f"Area: {props.get('areaDesc', 'Unknown')}\n"
            f"Severity: {props.get('severity', 'Unknown')}\n"
            f"Description: {props.get('description', 'N/A')}"
        )
        alerts.append(alert)

    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location using latitude and longitude.

    Args:
        latitude: Latitude of the location (e.g., 40.7128 for New York)
        longitude: Longitude of the location (e.g., -74.0060 for New York)
    """
    # First, get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the next 5 periods
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecast = (
            f"{period['name']}:\n"
            f"  Temperature: {period['temperature']}°{period['temperatureUnit']}\n"
            f"  Wind: {period['windSpeed']} {period['windDirection']}\n"
            f"  Forecast: {period['detailedForecast']}"
        )
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)
```

**Key points about async tools:**
- Use `async def` for any tool that makes I/O calls (HTTP requests, database queries, file reads)
- Use `httpx.AsyncClient` instead of `requests` (which blocks the event loop)
- Always set timeouts on external requests
- Return user-friendly strings, not raw JSON — the LLM passes these to the user

---

## Running the server

Add the entry point to run the server with stdio transport:

```python
def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

Start the server:

```bash
# Using uv (recommended)
uv run server.py

# Using Python directly
python server.py
```

The server now listens for JSON-RPC messages on stdin and responds on stdout. It won't produce any visible output — it's waiting for a client to connect.

> **Warning:** For stdio servers, **never use `print()`**. It writes to stdout and corrupts the JSON-RPC message stream. Use `logging` instead.

### Correct logging in stdio servers

```python
import logging

# Configure logging to write to stderr
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-server")

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get active weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g., CA, NY, TX)
    """
    logger.info(f"Fetching alerts for state: {state}")  # ✅ Goes to stderr

    # print(f"Fetching alerts for {state}")  ❌ NEVER do this in stdio servers

    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)
    # ... rest of implementation
```

---

## Testing with MCP Inspector

The **MCP Inspector** is a development tool that lets you interact with your MCP server through a web interface. It connects to your server, lists tools, and lets you call them:

```bash
# Install and run the inspector
npx @modelcontextprotocol/inspector uv run server.py
```

The Inspector opens a browser window where you can:
1. See all registered tools with their schemas
2. Fill in parameters and call tools
3. View the raw JSON-RPC messages
4. Debug response formatting

> **Tip:** The MCP Inspector is the fastest way to verify your tool definitions and test execution before connecting to an actual AI model.

---

## Configuring with Claude Desktop

To use your MCP server with Claude Desktop, add it to the configuration file:

**macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/my-mcp-server",
        "run",
        "server.py"
      ]
    }
  }
}
```

After saving and restarting Claude Desktop, the weather tools appear in the interface. When you ask "What's the weather in Sacramento?", Claude discovers the `get_forecast` tool via MCP and calls it automatically.

### What happens under the hood

1. Claude Desktop launches your server as a subprocess using the configured command
2. It sends an `initialize` request and negotiates capabilities
3. It calls `tools/list` to discover your tools
4. When the user asks a weather question, Claude's model decides to call `get_forecast`
5. Claude Desktop sends `tools/call` via the MCP client
6. Your server executes the function and returns the result
7. Claude uses the result to formulate a natural language response

---

## Complete server example

Here's a full, working MCP server that you can use as a template:

```python
"""A task management MCP server."""

import logging
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Configure logging (stderr only for stdio servers)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task-server")

# Initialize the server
mcp = FastMCP("task-manager")

# In-memory task storage (use a database in production)
tasks: dict[str, dict] = {}
task_counter = 0


@mcp.tool()
def create_task(title: str, priority: str = "medium") -> str:
    """Create a new task with a title and priority level.

    Args:
        title: Short description of the task
        priority: Priority level - low, medium, high, or critical
    """
    global task_counter
    task_counter += 1
    task_id = f"TASK-{task_counter:04d}"

    if priority not in ("low", "medium", "high", "critical"):
        return f"Error: Invalid priority '{priority}'. Use low, medium, high, or critical."

    tasks[task_id] = {
        "id": task_id,
        "title": title,
        "priority": priority,
        "status": "todo",
        "created": datetime.now().isoformat(),
    }

    logger.info(f"Created task {task_id}: {title}")
    return f"Created task {task_id}: '{title}' (priority: {priority})"


@mcp.tool()
def list_tasks(status: str = "all") -> str:
    """List all tasks, optionally filtered by status.

    Args:
        status: Filter by status - todo, in-progress, done, or all
    """
    if not tasks:
        return "No tasks found."

    filtered = tasks.values()
    if status != "all":
        filtered = [t for t in filtered if t["status"] == status]

    if not filtered:
        return f"No tasks with status '{status}'."

    lines = []
    for task in filtered:
        lines.append(
            f"[{task['id']}] {task['title']} "
            f"| Priority: {task['priority']} "
            f"| Status: {task['status']}"
        )
    return "\n".join(lines)


@mcp.tool()
def update_status(task_id: str, status: str) -> str:
    """Update the status of an existing task.

    Args:
        task_id: The task identifier (e.g., TASK-0001)
        status: New status - todo, in-progress, or done
    """
    if task_id not in tasks:
        return f"Error: Task '{task_id}' not found."

    if status not in ("todo", "in-progress", "done"):
        return f"Error: Invalid status '{status}'. Use todo, in-progress, or done."

    old_status = tasks[task_id]["status"]
    tasks[task_id]["status"] = status
    logger.info(f"Updated {task_id}: {old_status} → {status}")
    return f"Updated {task_id}: {old_status} → {status}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**Output (when interacting via Claude Desktop):**

```
User: Create a task to fix the login bug, high priority
Claude: Created task TASK-0001: 'Fix the login bug' (priority: high)

User: Create another task to update the docs
Claude: Created task TASK-0002: 'Update the docs' (priority: medium)

User: What tasks do I have?
Claude: Here are your current tasks:
  [TASK-0001] Fix the login bug | Priority: high | Status: todo
  [TASK-0002] Update the docs | Priority: medium | Status: todo

User: Mark the login bug as in progress
Claude: Updated TASK-0001: todo → in-progress
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Always use type hints on tool parameters | SDK generates accurate JSON Schema from types |
| Write detailed docstrings with `Args:` section | Parameter descriptions help the LLM provide correct values |
| Return human-readable strings, not raw data | The LLM passes tool output directly to the user |
| Use `async def` for I/O-bound tools | Prevents blocking the event loop during API calls |
| Set timeouts on all external requests | Avoids hanging on unresponsive services |
| Use `logging` instead of `print()` | `print()` corrupts stdio transport |
| Validate inputs before processing | Return clear error messages for invalid parameters |
| Keep tools focused (single responsibility) | One tool = one action makes LLM selection more reliable |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `print()` for debugging | Use `logging.info()` — it writes to stderr |
| Missing type hints on parameters | Always annotate: `def tool(name: str, count: int)` |
| No docstring or vague descriptions | Write: "Search flights between cities" not "Search stuff" |
| Returning raw JSON/dict from tools | Convert to formatted string: `f"{key}: {value}"` |
| Using `requests` instead of `httpx` | `requests` blocks the event loop; use `async httpx` |
| Not handling errors gracefully | Return error strings: `"Error: City not found"` instead of raising |
| Hardcoding the transport type | Default to stdio for development; make it configurable |
| Testing only via AI model | Use MCP Inspector first for rapid iteration |

---

## Hands-on exercise

### Your task

Build a complete MCP server called `bookmark-manager` that manages a collection of bookmarks (URLs with titles and tags).

### Requirements

1. Create a new MCP server using `FastMCP("bookmark-manager")`
2. Implement three tools:
   - `add_bookmark(url: str, title: str, tags: str = "")` — Saves a bookmark. Tags are comma-separated
   - `search_bookmarks(query: str)` — Searches bookmarks by title or tag
   - `list_bookmarks(tag: str = "all")` — Lists all bookmarks, optionally filtered by tag
3. Use in-memory storage (a dictionary)
4. Include proper type hints, docstrings, and error handling
5. Add the `main()` entry point with stdio transport

### Expected result

A working `bookmark_server.py` that passes MCP Inspector testing — you can add, search, and list bookmarks through the tool interface.

<details>
<summary>💡 Hints (click to expand)</summary>

- Store bookmarks in a `dict[str, dict]` keyed by URL
- For `search_bookmarks`, check if the query appears in the title or tags (case-insensitive)
- Split tags on commas and strip whitespace: `[t.strip() for t in tags.split(",")]`
- Return "No bookmarks found" instead of empty strings

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
"""Bookmark Manager MCP Server."""

import logging
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bookmark-server")

mcp = FastMCP("bookmark-manager")

# In-memory storage
bookmarks: dict[str, dict] = {}


@mcp.tool()
def add_bookmark(url: str, title: str, tags: str = "") -> str:
    """Save a new bookmark with a URL, title, and optional tags.

    Args:
        url: The full URL to bookmark (e.g., https://example.com)
        title: A descriptive title for the bookmark
        tags: Comma-separated tags (e.g., "python, tutorial, beginner")
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL '{url}'. Must start with http:// or https://"

    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()] if tags else []

    bookmarks[url] = {
        "url": url,
        "title": title,
        "tags": tag_list,
    }

    tag_display = ", ".join(tag_list) if tag_list else "none"
    logger.info(f"Added bookmark: {title} ({url})")
    return f"Saved bookmark: '{title}'\n  URL: {url}\n  Tags: {tag_display}"


@mcp.tool()
def search_bookmarks(query: str) -> str:
    """Search bookmarks by title or tag.

    Args:
        query: Search term to match against bookmark titles and tags
    """
    if not bookmarks:
        return "No bookmarks saved yet."

    query_lower = query.lower()
    results = []

    for bm in bookmarks.values():
        title_match = query_lower in bm["title"].lower()
        tag_match = any(query_lower in tag for tag in bm["tags"])

        if title_match or tag_match:
            tags = ", ".join(bm["tags"]) if bm["tags"] else "none"
            results.append(f"• {bm['title']}\n  {bm['url']}\n  Tags: {tags}")

    if not results:
        return f"No bookmarks matching '{query}'."

    return f"Found {len(results)} bookmark(s):\n\n" + "\n\n".join(results)


@mcp.tool()
def list_bookmarks(tag: str = "all") -> str:
    """List all saved bookmarks, optionally filtered by tag.

    Args:
        tag: Filter by tag name, or 'all' to show everything
    """
    if not bookmarks:
        return "No bookmarks saved yet."

    filtered = bookmarks.values()
    if tag != "all":
        filtered = [bm for bm in filtered if tag.lower() in bm["tags"]]

    if not filtered:
        return f"No bookmarks with tag '{tag}'."

    lines = []
    for bm in filtered:
        tags = ", ".join(bm["tags"]) if bm["tags"] else "none"
        lines.append(f"• {bm['title']}\n  {bm['url']}\n  Tags: {tags}")

    return f"Showing {len(lines)} bookmark(s):\n\n" + "\n\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

</details>

### Bonus challenges

- [ ] Add a `delete_bookmark(url: str)` tool that removes a bookmark
- [ ] Add a `export_bookmarks()` tool that returns all bookmarks in Markdown link format
- [ ] Persist bookmarks to a JSON file so they survive server restarts

---

## Summary

✅ `FastMCP` auto-generates tool schemas from Python **type hints** and **docstrings** — no manual JSON Schema needed

✅ Use `@mcp.tool()` to register functions as MCP tools — both sync and async functions work

✅ For async tools, use `httpx.AsyncClient` instead of `requests` to avoid blocking the event loop

✅ **Never use `print()`** in stdio servers — use `logging` (which writes to stderr) instead

✅ Test with **MCP Inspector** (`npx @modelcontextprotocol/inspector`) before connecting to AI models

✅ Configure Claude Desktop by adding your server to `claude_desktop_config.json` with the launch command

**Next:** [Client Implementation →](./04-client-implementation.md)

---

*Previous:* [Server Primitives](./02-server-primitives.md) | *Next:* [Client Implementation →](./04-client-implementation.md)

<!--
Sources Consulted:
- MCP Build Server Guide: https://modelcontextprotocol.io/docs/develop/build-server
- MCP Server Concepts: https://modelcontextprotocol.io/docs/learn/server-concepts
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Example Servers: https://modelcontextprotocol.io/examples
-->
