---
title: "Building MCP Servers"
---

# Building MCP Servers

## Introduction

An MCP server is a program that exposes tools, resources, and prompts to AI applications through the standardized protocol. In this lesson, we build MCP servers using the **Python `mcp` SDK** and its high-level **FastMCP** interface.

FastMCP transforms Python functions into MCP-compliant tools using type hints and docstrings — no manual JSON Schema definitions required. We will create a working server from scratch, register tools with input validation, expose resources with URI patterns, define prompt templates, and run the server for testing.

### What We'll Cover

- Setting up the Python MCP SDK environment
- Creating a server with `FastMCP`
- Registering tools with automatic schema generation
- Exposing resources (static and templated)
- Defining prompt templates
- Declaring capabilities and running the server
- Logging best practices for MCP servers

### Prerequisites

- Python 3.10+ installed
- Familiarity with Python type hints and async/await
- Understanding of MCP primitives (previous lesson)
- `uv` package manager (recommended) or `pip`

---

## Setting Up the Environment

We use `uv` for fast, reliable Python project management:

```bash
# Create a new project directory
uv init my-mcp-server
cd my-mcp-server

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the MCP SDK with CLI tools
uv add "mcp[cli]"
```

> **Note:** The `mcp[cli]` extra includes the `mcp` command-line inspector for testing servers interactively.

If you prefer `pip`:

```bash
pip install "mcp[cli]"
```

---

## Creating a Server with FastMCP

The `FastMCP` class is the entry point for building MCP servers. It handles protocol negotiation, capability declaration, and message routing automatically.

```python
from mcp.server.fastmcp import FastMCP

# Initialize the server with a name
mcp = FastMCP("my-server")
```

That single line creates a fully functional MCP server skeleton. The name `"my-server"` appears in the `serverInfo` during initialization, helping clients identify which server they are connected to.

---

## Registering Tools

Tools are the most common primitive. With FastMCP, we register tools using the `@mcp.tool()` decorator. The SDK inspects type hints and docstrings to generate JSON Schema automatically.

### Basic Tool Registration

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    result = a + b
    return f"{a} + {b} = {result}"
```

**What FastMCP generates from this:**

```json
{
  "name": "add",
  "description": "Add two numbers together.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": { "type": "number", "description": "First number" },
      "b": { "type": "number", "description": "Second number" }
    },
    "required": ["a", "b"]
  }
}
```

> **🔑 Key Concept:** FastMCP reads the function's docstring for the tool description and the `Args:` section for parameter descriptions. Always write clear docstrings — the LLM uses them to decide when to call your tool.

### Tools with Complex Inputs

For more complex inputs, use Python's `typing` module and Pydantic-style patterns:

```python
from typing import Optional

@mcp.tool()
async def search_files(
    query: str,
    directory: str = ".",
    max_results: int = 10,
    file_type: Optional[str] = None,
) -> str:
    """Search for files matching a query pattern.

    Args:
        query: Search pattern (supports wildcards)
        directory: Directory to search in (default: current)
        max_results: Maximum number of results to return
        file_type: Filter by file extension (e.g., '.py', '.md')
    """
    # Implementation here
    results = []  # Actual file search logic
    return f"Found {len(results)} files matching '{query}'"
```

Parameters with default values become **optional** in the JSON Schema. Parameters without defaults are **required**.

### Tools That Return Structured Data

Tools always return strings (or lists of content objects). For structured data, format it as readable text:

```python
import json

@mcp.tool()
async def get_project_stats(project_path: str) -> str:
    """Get statistics about a project directory.

    Args:
        project_path: Path to the project directory
    """
    stats = {
        "total_files": 42,
        "python_files": 15,
        "total_lines": 3200,
        "test_coverage": "78%",
    }
    return json.dumps(stats, indent=2)
```

**Output:**

```
{
  "total_files": 42,
  "python_files": 15,
  "total_lines": 3200,
  "test_coverage": "78%"
}
```

### Error Handling in Tools

Return error information as text with the `isError` pattern, or raise exceptions that FastMCP converts to error responses:

```python
@mcp.tool()
async def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file
    """
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except PermissionError:
        raise ValueError(f"Permission denied: {path}")
```

> **Warning:** Never silently swallow errors. The LLM needs to know when a tool fails so it can try a different approach.

---

## Exposing Resources

Resources provide read-only data that applications can fetch for context. Use the `@mcp.resource()` decorator with a URI pattern.

### Static Resources

Static resources have fixed URIs:

```python
@mcp.resource("config://app/settings")
async def get_settings() -> str:
    """Application configuration settings."""
    settings = {
        "debug": False,
        "log_level": "INFO",
        "max_connections": 100,
    }
    return json.dumps(settings, indent=2)
```

The client can read this resource with `resources/read` using the URI `config://app/settings`.

### Dynamic Resource Templates

For parameterized resources, use URI templates with curly braces:

```python
@mcp.resource("users://{user_id}/profile")
async def get_user_profile(user_id: str) -> str:
    """Get a user's profile information.

    Args:
        user_id: The user's unique identifier
    """
    # In practice, fetch from a database
    profile = {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "role": "developer",
    }
    return json.dumps(profile, indent=2)
```

The application can request `users://alice/profile` or `users://bob/profile`, and the `user_id` parameter is extracted from the URI automatically.

### Resources with MIME Types

Specify the content type for proper handling:

```python
@mcp.resource("docs://api/openapi", mime_type="application/json")
async def get_api_spec() -> str:
    """OpenAPI specification for the project's REST API."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "My API", "version": "1.0.0"},
        "paths": {},
    }
    return json.dumps(spec)
```

---

## Defining Prompt Templates

Prompts let server authors provide pre-built workflows. Use the `@mcp.prompt()` decorator:

### Basic Prompt

```python
@mcp.prompt()
async def code_review(language: str, focus: str = "general") -> str:
    """Review code for quality, bugs, and improvements.

    Args:
        language: Programming language of the code
        focus: Review focus area (general, security, performance)
    """
    return f"""Please review the following {language} code with a focus on {focus}.

Look for:
- Bugs and logic errors
- Code style and readability issues
- {"Security vulnerabilities (SQL injection, XSS, etc.)" if focus == "security" else ""}
- {"Performance bottlenecks and optimization opportunities" if focus == "performance" else ""}
- Best practices for {language}

Provide specific, actionable feedback with line references."""
```

### Prompt with Multiple Messages

For more complex prompts that set up a conversation:

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

@mcp.prompt()
async def debug_assistant(error_message: str, language: str = "python") -> list[dict]:
    """Help debug an error message.

    Args:
        error_message: The error message to debug
        language: Programming language context
    """
    return [
        {
            "role": "user",
            "content": f"I'm getting this error in my {language} code:\n\n```\n{error_message}\n```\n\nPlease help me understand what's causing it and how to fix it.",
        }
    ]
```

---

## Complete Server Example

Here is a complete, working MCP server that combines tools, resources, and prompts:

```python
"""A note-taking MCP server with tools, resources, and prompts."""

import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Initialize server
mcp = FastMCP("notes")

# In-memory storage (use a database in production)
notes: dict[str, dict] = {}


# --- Tools ---

@mcp.tool()
async def create_note(title: str, content: str, tags: str = "") -> str:
    """Create a new note.

    Args:
        title: Note title
        content: Note body content
        tags: Comma-separated tags (optional)
    """
    note_id = f"note_{len(notes) + 1}"
    notes[note_id] = {
        "id": note_id,
        "title": title,
        "content": content,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created_at": datetime.now().isoformat(),
    }
    return f"Created note '{title}' with ID: {note_id}"


@mcp.tool()
async def search_notes(query: str) -> str:
    """Search notes by title or content.

    Args:
        query: Search term to look for in titles and content
    """
    results = []
    for note in notes.values():
        if query.lower() in note["title"].lower() or query.lower() in note["content"].lower():
            results.append(f"[{note['id']}] {note['title']}")

    if not results:
        return f"No notes found matching '{query}'"
    return f"Found {len(results)} notes:\n" + "\n".join(results)


@mcp.tool()
async def delete_note(note_id: str) -> str:
    """Delete a note by its ID.

    Args:
        note_id: The unique identifier of the note to delete
    """
    if note_id not in notes:
        raise ValueError(f"Note not found: {note_id}")
    title = notes[note_id]["title"]
    del notes[note_id]
    return f"Deleted note: '{title}'"


# --- Resources ---

@mcp.resource("notes://all")
async def list_all_notes() -> str:
    """List of all stored notes."""
    if not notes:
        return json.dumps({"notes": [], "count": 0})
    return json.dumps({"notes": list(notes.values()), "count": len(notes)}, indent=2)


@mcp.resource("notes://{note_id}")
async def get_note(note_id: str) -> str:
    """Get a specific note by ID.

    Args:
        note_id: The note identifier
    """
    if note_id not in notes:
        return json.dumps({"error": f"Note not found: {note_id}"})
    return json.dumps(notes[note_id], indent=2)


# --- Prompts ---

@mcp.prompt()
async def summarize_notes(topic: str = "all") -> str:
    """Summarize notes, optionally filtered by topic.

    Args:
        topic: Topic to filter by, or 'all' for everything
    """
    return f"""Please summarize the following notes{f' related to {topic}' if topic != 'all' else ''}.
Organize the summary by theme and highlight key points.
If there are action items, list them separately."""


# --- Run ---

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

---

## Running and Testing the Server

### Running with stdio

The simplest way to run an MCP server:

```bash
uv run server.py
```

The server listens on stdin and responds on stdout. In practice, a host application launches the server as a subprocess.

### Testing with the MCP Inspector

The MCP SDK includes an interactive inspector for testing:

```bash
mcp dev server.py
```

This opens a web UI where you can:
- See all registered tools, resources, and prompts
- Call tools with custom arguments
- Read resources by URI
- View the raw JSON-RPC messages

### Configuring with Claude Desktop

To use your server with Claude Desktop, add it to the configuration file:

```json
{
  "mcpServers": {
    "notes": {
      "command": "uv",
      "args": ["--directory", "/path/to/my-mcp-server", "run", "server.py"]
    }
  }
}
```

> **Warning:** Use the **absolute path** to your server directory. Relative paths may not resolve correctly depending on how the host launches the process.

---

## Logging Best Practices

For **stdio-based servers**, logging requires special care. The stdout channel is reserved for JSON-RPC messages — writing anything else to stdout corrupts the protocol.

```python
# ❌ Bad — prints to stdout, breaks the protocol
print("Processing request")

# ✅ Good — logs to stderr, safe for stdio servers
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Processing request")
```

For **HTTP-based servers**, standard output logging is fine since it does not interfere with HTTP responses.

> **Important:** Python's `print()` writes to stdout by default. In stdio MCP servers, **never use `print()`** — always use the `logging` module configured to write to stderr.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use clear, specific function docstrings | FastMCP generates tool descriptions from them |
| Make tool functions async | MCP operations are inherently asynchronous |
| Validate inputs early in tool functions | Return helpful error messages before doing work |
| Use descriptive parameter names | They appear in the JSON Schema the LLM reads |
| Return structured text from tools | Makes it easier for the LLM to parse results |
| Keep servers focused on one domain | Smaller servers are easier to maintain and test |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `print()` in stdio servers | Use `logging` module with stderr handler |
| Missing docstrings on tool functions | Always write docstrings — they become tool descriptions |
| Returning `None` from tools | Always return a string describing what happened |
| Not handling exceptions in tools | Wrap operations in try/except, raise `ValueError` for user-visible errors |
| Hardcoding file paths | Accept paths as parameters, validate against roots |
| Putting all logic in one giant server | Split into focused servers: one for files, one for database, etc. |

---

## Hands-on Exercise

### Your Task

Build a "Bookmark Manager" MCP server that lets an AI assistant manage web bookmarks.

### Requirements

1. Create a FastMCP server named `"bookmarks"`
2. Implement **3 tools**:
   - `add_bookmark(url, title, category)` — Save a new bookmark
   - `search_bookmarks(query)` — Search bookmarks by title or URL
   - `delete_bookmark(url)` — Remove a bookmark
3. Implement **2 resources**:
   - `bookmarks://all` — List all bookmarks as JSON
   - `bookmarks://category/{category}` — List bookmarks filtered by category
4. Implement **1 prompt**:
   - `organize_bookmarks` — Generate instructions to categorize uncategorized bookmarks
5. Add the `main()` function to run with stdio transport

### Expected Result

A working Python file that can be tested with `mcp dev your_server.py`.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a dictionary as in-memory storage: `bookmarks: dict[str, dict] = {}`
- The URL makes a good dictionary key
- For `search_bookmarks`, check both `url` and `title` fields
- The resource template `bookmarks://category/{category}` should extract the `category` parameter

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
"""Bookmark Manager MCP Server."""

import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bookmarks")

# In-memory storage
bookmarks: dict[str, dict] = {}


@mcp.tool()
async def add_bookmark(url: str, title: str, category: str = "uncategorized") -> str:
    """Save a new web bookmark.

    Args:
        url: The URL to bookmark
        title: A descriptive title for the bookmark
        category: Category for organization (default: uncategorized)
    """
    bookmarks[url] = {
        "url": url,
        "title": title,
        "category": category,
        "added_at": datetime.now().isoformat(),
    }
    return f"Bookmarked: '{title}' in category '{category}'"


@mcp.tool()
async def search_bookmarks(query: str) -> str:
    """Search bookmarks by title or URL.

    Args:
        query: Search term to match against titles and URLs
    """
    results = [
        bm for bm in bookmarks.values()
        if query.lower() in bm["title"].lower() or query.lower() in bm["url"].lower()
    ]
    if not results:
        return f"No bookmarks found matching '{query}'"
    lines = [f"- [{bm['title']}]({bm['url']}) [{bm['category']}]" for bm in results]
    return f"Found {len(results)} bookmarks:\n" + "\n".join(lines)


@mcp.tool()
async def delete_bookmark(url: str) -> str:
    """Remove a bookmark by its URL.

    Args:
        url: The URL of the bookmark to delete
    """
    if url not in bookmarks:
        raise ValueError(f"Bookmark not found: {url}")
    title = bookmarks[url]["title"]
    del bookmarks[url]
    return f"Deleted bookmark: '{title}'"


@mcp.resource("bookmarks://all")
async def list_all_bookmarks() -> str:
    """All saved bookmarks."""
    return json.dumps(list(bookmarks.values()), indent=2)


@mcp.resource("bookmarks://category/{category}")
async def list_by_category(category: str) -> str:
    """Bookmarks filtered by category.

    Args:
        category: Category name to filter by
    """
    filtered = [bm for bm in bookmarks.values() if bm["category"] == category]
    return json.dumps(filtered, indent=2)


@mcp.prompt()
async def organize_bookmarks() -> str:
    """Generate instructions to categorize uncategorized bookmarks."""
    return """Review the list of bookmarks and suggest appropriate categories
for any that are currently 'uncategorized'. Group related bookmarks together
and suggest new category names where appropriate."""


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

</details>

### Bonus Challenges

- [ ] Add a `tag` field to bookmarks and implement a `bookmarks://tag/{tag}` resource template
- [ ] Implement `import_bookmarks(json_data)` tool that accepts JSON-formatted bookmark data
- [ ] Add a `reading_list` prompt that creates a curated reading plan from saved bookmarks

---

## Summary

✅ **FastMCP** generates JSON Schema from Python type hints and docstrings — no manual schema writing needed

✅ Use `@mcp.tool()` for functions the LLM can call, `@mcp.resource()` for data, and `@mcp.prompt()` for templates

✅ **Never use `print()`** in stdio servers — it corrupts the JSON-RPC channel; use `logging` instead

✅ Test servers interactively with `mcp dev server.py` before connecting to host applications

✅ Keep servers **focused and composable** — one domain per server, multiple servers per host

**Next:** [Building MCP Clients](./04-building-mcp-clients.md)

---

## Further Reading

- [Build an MCP Server — Official Tutorial](https://modelcontextprotocol.io/docs/develop/build-server) — Step-by-step guide
- [Python MCP SDK on GitHub](https://github.com/modelcontextprotocol/python-sdk) — Source code and examples
- [FastMCP API Reference](https://github.com/modelcontextprotocol/python-sdk#fastmcp) — Decorator reference
- [MCP Inspector](https://modelcontextprotocol.io/legacy/tools/debugging) — Debugging MCP servers

---

[Back to MCP Overview](./00-model-context-protocol-mcp.md)

<!-- Sources Consulted:
- Build MCP Server Tutorial: https://modelcontextprotocol.io/docs/develop/build-server
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Server Concepts: https://modelcontextprotocol.io/docs/learn/server-concepts
- MCP Specification 2025-06-18: https://modelcontextprotocol.io/specification/2025-06-18
-->
