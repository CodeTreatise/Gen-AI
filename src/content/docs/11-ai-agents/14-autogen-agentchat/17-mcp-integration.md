---
title: "MCP integration"
---

# MCP integration

The Model Context Protocol (MCP) is an open standard that lets AI applications connect to external tools and data sources through a unified interface. AutoGen AgentChat integrates with MCP via the `McpWorkbench` class from the `autogen-ext` package, enabling agents to discover and call tools provided by any MCP-compatible server — file systems, databases, APIs, and more — without writing custom tool wrappers. In this lesson we learn how to connect AutoGen agents to MCP servers, manage server lifecycles, work with multiple servers, and use the newer Streamable HTTP transport for remote connections.

## Prerequisites

Before starting this lesson, you should be familiar with:

- AutoGen AgentChat core concepts (AssistantAgent, tools)
- The `autogen-ext` extensions package (`pip install "autogen-ext[mcp]"`)
- MCP fundamentals from Unit 10 (Function Calling / Tool Use)
- Python `async with` context managers

---

## What is MCP in AutoGen?

MCP provides a standard protocol for LLM applications to access tools, resources, and prompts from external servers. Instead of writing bespoke tool functions for every service, we connect to an MCP server that already exposes those capabilities.

In AutoGen's architecture, the integration works like this:

1. An **MCP server** exposes tools (functions) over a transport protocol (stdio or HTTP).
2. A **`McpWorkbench`** instance connects to the server and discovers available tools.
3. An **`AssistantAgent`** receives the workbench and can call any discovered tool during conversation.

This is powerful because the MCP ecosystem already includes servers for file systems, GitHub, Slack, databases, web search, and dozens of other services. By connecting to these servers, our agents gain capabilities without us writing a single tool function.

> **Note:** MCP in AutoGen builds directly on the concepts covered in Unit 10 (Function Calling / Tool Use). If you have not studied MCP fundamentals there, we recommend reviewing that material first.

---

## McpWorkbench setup

The `McpWorkbench` class is the bridge between AutoGen agents and MCP servers. It handles connection management, tool discovery, and tool invocation.

### Installation

```bash
pip install "autogen-ext[openai,mcp]"
```

### Basic usage with a filesystem server

The most common MCP server is the filesystem server, which gives agents the ability to read, write, list, and search files:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    )

    async with McpWorkbench(server_params=server_params) as workbench:
        agent = AssistantAgent(
            name="file_agent",
            model_client=model_client,
            workbench=workbench,
        )
        result = await agent.run(task="List all Python files in the directory")
        print(result.messages[-1].content)

asyncio.run(main())
```

**Output:**
```
Found the following Python files:
- /path/to/dir/main.py
- /path/to/dir/utils.py
- /path/to/dir/tests/test_main.py
```

Let us break down what happens here:

1. We create `StdioServerParams` that tell `McpWorkbench` how to launch the MCP server as a subprocess.
2. The `async with` block starts the server, connects to it, and discovers all available tools.
3. We pass the `workbench` to the `AssistantAgent`, which makes every discovered tool available for the agent to call.
4. When the agent runs, it can invoke tools like `read_file`, `write_file`, `list_directory`, and `search_files` — all provided by the MCP server.
5. On exit, the context manager stops the server process.

---

## StdioServerParams

`StdioServerParams` configures subprocess-based MCP servers — the most common transport for local development. The server runs as a child process and communicates with AutoGen over stdin/stdout.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | The executable to run (e.g., `"npx"`, `"python"`, `"node"`) |
| `args` | `list[str]` | Arguments passed to the command |
| `env` | `dict[str, str]` | Optional environment variables for the subprocess |

### Examples

**Filesystem server:**

```python
server_params = StdioServerParams(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"],
)
```

**Output:**
```
(Params configured for the MCP filesystem server)
```

**GitHub server:**

```python
server_params = StdioServerParams(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"},
)
```

**Output:**
```
(Params configured for the MCP GitHub server with auth token)
```

**Custom Python MCP server:**

```python
server_params = StdioServerParams(
    command="python",
    args=["-m", "my_mcp_server"],
    env={"DATABASE_URL": "postgresql://localhost/mydb"},
)
```

**Output:**
```
(Params configured for a custom Python-based MCP server)
```

> **Warning:** The `env` dictionary replaces the entire environment for the subprocess. If your MCP server needs access to system PATH or other variables, merge them explicitly: `env={**os.environ, "MY_KEY": "value"}`.

---

## Connecting to multiple MCP servers

Real-world agents often need tools from multiple sources — files from a filesystem server, issues from GitHub, messages from Slack. We can create multiple `McpWorkbench` instances and combine them:

```python
import asyncio
from contextlib import AsyncExitStack
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    fs_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    )

    github_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"},
    )

    async with AsyncExitStack() as stack:
        fs_workbench = await stack.enter_async_context(
            McpWorkbench(server_params=fs_params)
        )
        github_workbench = await stack.enter_async_context(
            McpWorkbench(server_params=github_params)
        )

        # File agent uses filesystem tools
        file_agent = AssistantAgent(
            name="file_agent",
            model_client=model_client,
            workbench=fs_workbench,
        )

        # GitHub agent uses GitHub tools
        github_agent = AssistantAgent(
            name="github_agent",
            model_client=model_client,
            workbench=github_workbench,
        )

        # Use agents in a team...

asyncio.run(main())
```

**Output:**
```
(Two agents, each connected to a different MCP server via separate workbenches)
```

> **Note:** We use `AsyncExitStack` to manage multiple async context managers cleanly. This ensures all servers are properly shut down, even if an error occurs.

### Alternative: single agent with multiple tool sources

If a single agent needs tools from multiple servers, we can gather tools from each workbench and pass them together:

```python
async with AsyncExitStack() as stack:
    fs_workbench = await stack.enter_async_context(
        McpWorkbench(server_params=fs_params)
    )
    github_workbench = await stack.enter_async_context(
        McpWorkbench(server_params=github_params)
    )

    # Combine tools from both workbenches
    all_tools = fs_workbench.list_tools() + github_workbench.list_tools()

    super_agent = AssistantAgent(
        name="super_agent",
        model_client=model_client,
        tools=all_tools,
    )
```

**Output:**
```
(Single agent with tools from both filesystem and GitHub MCP servers)
```

---

## Session lifecycle

`McpWorkbench` manages the full lifecycle of the MCP server connection:

| Phase | What happens | Triggered by |
|-------|-------------|--------------|
| **Start** | Launches the server subprocess (stdio) or opens HTTP connection | Entering `async with` |
| **Discovery** | Queries the server for available tools | Automatic after start |
| **Invocation** | Calls tools as the agent requests them | Agent tool calls during `run()` |
| **Shutdown** | Stops the subprocess or closes the connection | Exiting `async with` |

This lifecycle model is critical for resource management. Each MCP server is a running process that consumes memory and potentially holds connections (to databases, APIs, etc.). The context manager pattern ensures we never leak these resources.

```python
# The lifecycle in code
async with McpWorkbench(server_params=server_params) as workbench:
    # Server is RUNNING — tools are available
    agent = AssistantAgent(name="agent", model_client=client, workbench=workbench)
    result = await agent.run(task="Do something with the tools")
# Server is STOPPED — process terminated, resources freed
```

**Output:**
```
(Server starts on enter, stops on exit — clean resource management)
```

> **Warning:** Never store a `McpWorkbench` reference outside its `async with` block. Once the context exits, the server is gone and tool calls will fail.

---

## Streamable HTTP transport

While stdio is perfect for local development, production deployments often need remote MCP servers. The **Streamable HTTP** transport protocol enables this by connecting to MCP servers over HTTP, supporting both streaming and non-streaming responses.

```python
from autogen_ext.tools.mcp import McpWorkbench, StreamableHttpServerParams

http_params = StreamableHttpServerParams(
    url="https://mcp.example.com/sse",
)

async with McpWorkbench(server_params=http_params) as workbench:
    agent = AssistantAgent(
        name="remote_agent",
        model_client=model_client,
        workbench=workbench,
    )
    result = await agent.run(task="Query the remote service")
```

**Output:**
```
(Agent connected to a remote MCP server over Streamable HTTP)
```

### When to use each transport

| Transport | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Stdio** (`StdioServerParams`) | Local development, single-machine deployments | Simple setup, no network config | Server must run on same machine |
| **Streamable HTTP** (`StreamableHttpServerParams`) | Production, remote servers, shared infrastructure | Network-accessible, scalable | Requires HTTP endpoint, auth setup |

---

## Available MCP servers

The MCP ecosystem is growing rapidly. Here are some popular servers that work well with AutoGen:

| Server | Package | Capabilities |
|--------|---------|-------------|
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | Read, write, list, search files |
| **GitHub** | `@modelcontextprotocol/server-github` | Issues, PRs, repos, code search |
| **Slack** | `@modelcontextprotocol/server-slack` | Read/send messages, list channels |
| **PostgreSQL** | `@modelcontextprotocol/server-postgres` | Query databases, inspect schemas |
| **Brave Search** | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| **Memory** | `@modelcontextprotocol/server-memory` | Persistent key-value memory |
| **Puppeteer** | `@modelcontextprotocol/server-puppeteer` | Browser automation, screenshots |

You can find a comprehensive directory of MCP servers at [modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers).

> **Note:** Any MCP-compatible server works with AutoGen's `McpWorkbench`. As the ecosystem grows, your agents automatically gain access to new capabilities without code changes — just point them at a new server.

---

## Best practices

1. **Always use async context managers.** The `async with` pattern ensures MCP servers are properly started and stopped. Never manage the lifecycle manually.
2. **Scope workbenches narrowly.** Create `McpWorkbench` instances close to where they are used and let them close as soon as possible. Long-lived server processes waste resources.
3. **Use `AsyncExitStack` for multiple servers.** It handles cleanup in reverse order and is exception-safe.
4. **Prefer stdio for development, HTTP for production.** Stdio is simpler to set up; Streamable HTTP scales across machines.
5. **Pass secrets via `env`, not `args`.** Command-line arguments are visible in process listings. Use the `env` parameter to pass API keys and tokens securely.
6. **Limit tools per agent.** An agent with too many tools may struggle to choose the right one. Assign specific workbenches to specific agents based on their roles.
7. **Cross-reference with Unit 10.** The MCP concepts here build directly on the Function Calling / Tool Use unit. Review that material for protocol-level details.

---

## Common pitfalls

| Pitfall | Consequence | Fix |
|---------|------------|-----|
| Using `McpWorkbench` without `async with` | Server never starts or never stops | Always use `async with McpWorkbench(...) as wb:` |
| Storing workbench reference outside context | `ConnectionError` on tool calls | Keep all usage inside the `async with` block |
| Passing secrets in `args` | Credentials visible in `ps` output | Use `env` parameter instead |
| Overloading one agent with many servers | Poor tool selection by the LLM | Assign one workbench per agent by role |
| Forgetting to install MCP server packages | `npx` downloads on every run (slow) | Install servers globally: `npm install -g @modelcontextprotocol/server-filesystem` |
| Not merging `os.environ` in `env` | Server subprocess cannot find executables | Use `env={**os.environ, "KEY": "val"}` |

---

## Exercise

Build an agent that uses MCP to interact with the local filesystem:

1. Install the required packages:
   ```bash
   pip install "autogen-ext[openai,mcp]"
   ```
2. Create `StdioServerParams` pointing to the MCP filesystem server and a local directory of your choice.
3. Create an `McpWorkbench` with those parameters inside an `async with` block.
4. Create an `AssistantAgent` with the workbench.
5. Run the task: *"List all files in the directory, then read the contents of the first Python file you find and summarise what it does."*
6. Print the agent's final response.

**Bonus:** Add a second `McpWorkbench` connected to the MCP memory server (`@modelcontextprotocol/server-memory`). Have the agent store its summary in memory, then retrieve it in a subsequent run.

---

## Summary

AutoGen's MCP integration lets agents connect to any MCP-compatible server and use its tools without writing custom wrappers. The `McpWorkbench` class handles server lifecycle, tool discovery, and invocation through a clean async context manager pattern. `StdioServerParams` powers local subprocess-based servers, while `StreamableHttpServerParams` enables remote HTTP connections for production deployments. By assigning specific workbenches to specific agents, we can build teams where each member has access to exactly the tools it needs — filesystem operations for one agent, GitHub for another, databases for a third. Combined with the growing MCP ecosystem of pre-built servers, this approach makes it straightforward to give our agents real-world capabilities at scale.

---

**Previous:** [Reasoning Models and Providers](./16-reasoning-models-and-providers.md) | **Next:** [Google Agent Development Kit](../15-google-agent-development-kit/00-google-agent-development-kit.md)

---

## Further reading

- [AutoGen MCP tools documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/mcp-tools.html)
- [Model Context Protocol specification](https://modelcontextprotocol.io/)
- [MCP server directory](https://modelcontextprotocol.io/servers)
- [Unit 10 — Function Calling / Tool Use](../../10-function-calling-tool-use/)
- [McpWorkbench API reference](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.tools.mcp.html)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources Consulted:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/mcp-tools.html
https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.tools.mcp.html
https://modelcontextprotocol.io/
https://modelcontextprotocol.io/servers
https://spec.modelcontextprotocol.io/
-->
