---
title: "Extensions and ecosystem"
---

# Extensions and ecosystem

AutoGen AgentChat's core library provides the foundational building blocks — agents, teams, and messaging — but the real power multiplies when we tap into the extension ecosystem. The `autogen-ext` package bundles production-ready integrations for model providers, specialized agents, memory backends, code executors, and more. On top of that, AutoGen Studio offers a visual, no-code interface for designing and testing agent workflows. In this lesson we explore the key extensions, learn how to install and configure them, and see how they fit together in real projects.

## Prerequisites

Before starting this lesson, you should be familiar with:

- AutoGen AgentChat core concepts (AssistantAgent, teams, tools)
- Python virtual environments and `pip` extras syntax
- Basic Docker concepts (for sandboxed code execution)

---

## The autogen-ext package

The `autogen-ext` package ships separately from `autogen-agentchat` so that you only install the dependencies you actually need. Each integration area is exposed as a pip *extra*:

```bash
# Install specific extras
pip install "autogen-ext[openai]"

# Install multiple extras at once
pip install "autogen-ext[openai,azure,magentic-one,chromadb,redis,mcp,docker]"
```

> **Note:** Each extra pulls in its own dependencies. Installing `docker` adds the Docker SDK, `chromadb` adds ChromaDB, and so on. Only install what your project requires to keep your environment lean.

After installation, every extension follows the same import convention:

```python
from autogen_ext.<category>.<name> import ClassName
```

This predictable structure makes discovery straightforward — if you know the category (models, agents, tools, memory, code_executors, teams) you can guess the import path.

---

## Key extensions reference

The table below summarises the most important extensions available today. We will dive deeper into several of them throughout this lesson and the remaining lessons in this unit.

| Extra | Import Path | Key Classes | Purpose |
|-------|-------------|-------------|---------|
| `openai` | `autogen_ext.models.openai` | `OpenAIChatCompletionClient` | Chat completions via OpenAI API |
| `azure` | `autogen_ext.models.openai` | `AzureOpenAIChatCompletionClient` | Chat completions via Azure OpenAI |
| `magentic-one` | `autogen_ext.agents.magentic_one` | `MagenticOneCoderAgent` | Specialised coding agent |
| `magentic-one` | `autogen_ext.agents.web_surfer` | `MultimodalWebSurfer` | Web browsing with vision |
| `magentic-one` | `autogen_ext.agents.file_surfer` | `FileSurfer` | Local file reading and navigation |
| `magentic-one` | `autogen_ext.teams.magentic_one` | `MagenticOne` | Pre-built multi-agent team helper |
| `mcp` | `autogen_ext.tools.mcp` | `McpWorkbench`, `StdioServerParams` | Model Context Protocol tool bridge |
| `chromadb` | `autogen_ext.memory.chromadb` | `ChromaDBVectorMemory` | Vector memory with ChromaDB |
| `redis` | `autogen_ext.memory.redis` | `RedisMemory` | Distributed memory via Redis |
| — | `autogen_ext.memory.mem0` | `Mem0Memory` | Personalised memory layer |
| — | `autogen_ext.code_executors.local` | `LocalCommandLineCodeExecutor` | Run code on local machine |
| `docker` | `autogen_ext.code_executors.docker` | `DockerCommandLineCodeExecutor` | Run code in Docker containers |

### Model clients

The model client extensions are the most commonly used. `OpenAIChatCompletionClient` wraps the OpenAI chat completions API, while `AzureOpenAIChatCompletionClient` targets Azure-hosted deployments with parameters like `azure_deployment`, `api_version`, and `azure_endpoint`.

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")
```

**Output:**
```
(Returns a configured model client ready for use with agents)
```

### Specialised agents

The Magentic-One family provides battle-tested agents for common tasks:

```python
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent

web_agent = MultimodalWebSurfer(
    name="web_surfer",
    model_client=model_client,
)

file_agent = FileSurfer(
    name="file_surfer",
    model_client=model_client,
)
```

**Output:**
```
(Agents initialised and ready to join a team)
```

`MultimodalWebSurfer` can browse the web and interpret screenshots, `FileSurfer` reads and navigates local files, and `MagenticOneCoderAgent` generates and iterates on code.

### Memory backends

Memory extensions let agents remember information across conversations:

```python
from autogen_ext.memory.chromadb import ChromaDBVectorMemory

memory = ChromaDBVectorMemory(
    collection_name="project_knowledge",
    persist_directory="./chroma_data",
)
```

**Output:**
```
(ChromaDB-backed vector memory ready for agent use)
```

`ChromaDBVectorMemory` stores embeddings locally, `RedisMemory` distributes state across a Redis cluster, and `Mem0Memory` provides a personalised memory layer that learns user preferences over time.

---

## AutoGen Studio

AutoGen Studio is a no-code/low-code companion application that wraps AutoGen AgentChat in a web-based visual interface. It is ideal for rapid prototyping, team demonstrations, and users who prefer a graphical workflow.

### Installation and launch

```bash
pip install autogenstudio
autogenstudio ui
```

**Output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8081
```

Open the URL in your browser to access the Studio interface.

### Core capabilities

AutoGen Studio provides several features that accelerate agent development:

| Feature | Description |
|---------|-------------|
| **Visual workflow design** | Drag-and-drop agent and team composition |
| **Team configuration UI** | Set team type, termination conditions, model clients |
| **Testing interface** | Send messages and observe multi-agent conversations in real time |
| **Component gallery** | Browse and reuse pre-built agents, tools, and teams |
| **Import / export** | Save and load component configurations as JSON |

### Working with JSON configurations

Every component you create in AutoGen Studio serialises to JSON. This means you can version-control your agent architectures, share them with teammates, or load them programmatically:

```json
{
  "type": "RoundRobinGroupChat",
  "participants": [
    {
      "type": "AssistantAgent",
      "name": "planner",
      "model_client": {
        "type": "OpenAIChatCompletionClient",
        "model": "gpt-4o"
      }
    },
    {
      "type": "AssistantAgent",
      "name": "executor",
      "model_client": {
        "type": "OpenAIChatCompletionClient",
        "model": "gpt-4o-mini"
      }
    }
  ]
}
```

> **Note:** AutoGen Studio is an excellent on-ramp, but production deployments typically graduate to the programmatic API for full control over error handling, logging, and deployment pipelines.

---

## Docker code execution

When agents generate and run code, security is paramount. The `DockerCommandLineCodeExecutor` runs code inside an isolated Docker container, preventing untrusted code from accessing the host filesystem or network.

### Setup

First, ensure Docker is installed and running on your machine:

```bash
pip install "autogen-ext[docker]"
```

Then configure the executor:

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

code_executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",
    container_name="autogen_executor",
    timeout=60,
    work_dir="/workspace",
)
```

**Output:**
```
(Docker code executor configured — container created on first use)
```

### Using with a CodeExecutorAgent

The executor plugs directly into a `CodeExecutorAgent`:

```python
from autogen_agentchat.agents import CodeExecutorAgent

executor_agent = CodeExecutorAgent(
    name="code_runner",
    code_executor=code_executor,
)
```

**Output:**
```
(CodeExecutorAgent ready to execute code blocks in Docker)
```

### Comparing execution options

| Feature | `LocalCommandLineCodeExecutor` | `DockerCommandLineCodeExecutor` |
|---------|-------------------------------|--------------------------------|
| **Isolation** | None — runs on host | Full container isolation |
| **Setup** | Zero — works immediately | Requires Docker daemon |
| **Performance** | Fastest | Slight container overhead |
| **Security** | Low — host access | High — sandboxed |
| **Use case** | Local development, trusted code | Production, untrusted code |

> **Warning:** Never use `LocalCommandLineCodeExecutor` with untrusted input in production. LLM-generated code can contain arbitrary system commands. Always prefer Docker execution for any environment where safety matters.

### Lifecycle management

The Docker executor supports async context management for clean startup and teardown:

```python
async with DockerCommandLineCodeExecutor(
    image="python:3.12-slim",
    timeout=60,
) as executor:
    agent = CodeExecutorAgent(name="runner", code_executor=executor)
    # Use agent...
    # Container is automatically stopped and removed on exit
```

**Output:**
```
(Container created on enter, destroyed on exit)
```

---

## Community extensions

AutoGen's architecture is deliberately extensible. Every major abstraction — model clients, agents, tools, memory, and code executors — follows the `Component` protocol, which means anyone can create a new extension by implementing the right interface.

### The Component protocol

At its core, a Component must support two operations:

1. **Serialisation** — convert the component's configuration to a JSON-compatible dictionary via `dump_component()`.
2. **Deserialisation** — reconstruct the component from that dictionary via `Component.load_component()`.

This protocol is what enables AutoGen Studio's import/export feature and makes it possible to share agent configurations as plain JSON files.

### Creating a custom extension

Here is a minimal example of a custom model client that wraps a hypothetical API:

```python
from autogen_core.models import ChatCompletionClient
from autogen_core import Component

class CustomModelClient(ChatCompletionClient, Component):
    """Custom model client following the Component protocol."""

    def __init__(self, api_key: str, model: str = "custom-v1"):
        self.api_key = api_key
        self.model = model

    def dump_component(self) -> dict:
        return {
            "type": "CustomModelClient",
            "model": self.model,
            # Never serialise secrets in production
        }

    # Implement remaining abstract methods...
```

**Output:**
```
(Custom component that serialises/deserialises via the Component protocol)
```

> **Note:** When building community extensions, follow the existing naming conventions (`autogen_ext.<category>.<name>`) and include comprehensive type hints so that tools like AutoGen Studio can discover your extension automatically.

---

## Best practices

1. **Install only what you need.** Each extra adds dependencies. A minimal install keeps builds fast and reduces version conflicts.
2. **Pin versions in production.** Use `pip freeze` to lock `autogen-agentchat` and `autogen-ext` versions together — they must stay compatible.
3. **Prefer Docker execution for untrusted code.** The local executor is convenient during development but dangerous in production.
4. **Use AutoGen Studio for prototyping, not production.** Graduate to the programmatic API once your team design stabilises.
5. **Follow the Component protocol.** If you build custom extensions, implement `dump_component()` and `load_component()` so they work seamlessly with serialisation and Studio.
6. **Keep secrets out of serialised configs.** Use environment variables or a secrets manager — never embed API keys in JSON exports.

---

## Common pitfalls

| Pitfall | Consequence | Fix |
|---------|------------|-----|
| Installing `autogen-ext` without extras | `ImportError` on first use | Use `pip install "autogen-ext[openai]"` |
| Mixing `autogen-agentchat` and `autogen-ext` version mismatches | Subtle runtime errors | Pin both to the same release series |
| Using `LocalCommandLineCodeExecutor` in production | Security vulnerability | Switch to `DockerCommandLineCodeExecutor` |
| Forgetting async context manager for Docker executor | Orphaned containers | Always use `async with` |
| Serialising API keys in component JSON | Credential leakage | Load keys from environment variables |

---

## Exercise

Build a research assistant team that uses multiple extensions:

1. Install the required extras: `pip install "autogen-ext[openai,docker]"`
2. Create an `AssistantAgent` with `OpenAIChatCompletionClient` as the model client.
3. Create a `CodeExecutorAgent` backed by `DockerCommandLineCodeExecutor`.
4. Combine both agents in a `RoundRobinGroupChat`.
5. Run a task: *"Write a Python script that fetches the current Bitcoin price from the CoinGecko API and prints it."*
6. Observe the assistant generating code and the executor running it safely in Docker.

**Bonus:** Export your team configuration as JSON using `team.dump_component()` and reload it with `RoundRobinGroupChat.load_component()`.

---

## Summary

AutoGen's extension ecosystem transforms a capable agent framework into a full-stack AI platform. The `autogen-ext` package provides plug-and-play integrations for model providers (OpenAI, Azure), specialised agents (web surfer, file surfer, coder), memory backends (ChromaDB, Redis, Mem0), code executors (local, Docker), and MCP tool bridges. AutoGen Studio adds a visual layer for rapid prototyping, while the Component protocol ensures that anyone can contribute new extensions. By choosing the right extensions for each project and following security best practices — especially around code execution — we can build production-grade agent systems efficiently.

---

**Next:** [Reasoning Models and Providers](./16-reasoning-models-and-providers.md)

---

## Further reading

- [AutoGen Extension Packages documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/extensions.html)
- [AutoGen Studio documentation](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html)
- [Docker Code Executor guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/code-executors.html)
- [Component protocol reference](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components.html)

[Back to AutoGen AgentChat Overview](./00-autogen-agentchat.md)

<!-- Sources Consulted:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/extensions.html
https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/code-executors.html
https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components.html
https://pypi.org/project/autogen-ext/
https://pypi.org/project/autogenstudio/
-->
