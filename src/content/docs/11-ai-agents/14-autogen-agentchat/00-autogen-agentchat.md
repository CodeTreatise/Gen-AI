---
title: "AutoGen AgentChat"
---

# AutoGen AgentChat

## Overview

Microsoft's **AutoGen** framework (v0.7.5, stable) provides a comprehensive platform for building single-agent and multi-agent AI applications. Built on an async-first, event-driven architecture, AutoGen's **AgentChat** high-level API makes it straightforward to create intelligent agents that use tools, collaborate in teams, and solve complex tasks through structured workflows.

AutoGen distinguishes itself through its modular package design (`autogen-agentchat`, `autogen-core`, `autogen-ext`), its rich set of team orchestration patterns (round-robin, model-selected, swarm, graph-based), and its built-in support for state persistence, memory, and component serialization. Whether you need a single assistant agent with tools or a sophisticated multi-agent system like Magentic-One that browses the web and writes code, AutoGen provides the building blocks.

### What we'll cover in this lesson

This lesson is organized into 17 sub-lessons covering every major aspect of AutoGen AgentChat:

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Architecture and core concepts](./01-architecture-and-core-concepts.md) | Async-first design, event-driven messaging, package structure |
| 02 | [AgentChat high-level API](./02-agentchat-high-level-api.md) | AssistantAgent, UserProxyAgent, run/run_stream, messaging |
| 03 | [Tools and function calling](./03-tools-and-function-calling.md) | FunctionTool, BaseTool, McpWorkbench, AgentTool |
| 04 | [Team orchestration patterns](./04-team-orchestration-patterns.md) | RoundRobinGroupChat, team operations, single-agent teams |
| 05 | [Termination conditions](./05-termination-conditions.md) | 11 built-in conditions, combining with `\|`/`&`, custom conditions |
| 06 | [SelectorGroupChat](./06-selector-group-chat.md) | Model-based speaker selection, custom selector/candidate functions |
| 07 | [Swarm pattern](./07-swarm-pattern.md) | HandoffMessage, agent handoffs, human-in-the-loop |
| 08 | [GraphFlow DAG workflows](./08-graphflow-dag-workflows.md) | DiGraphBuilder, sequential/parallel/conditional flows |
| 09 | [Magentic-One system](./09-magentic-one-system.md) | Orchestrator, WebSurfer, FileSurfer, Coder agents |
| 10 | [Custom agents](./10-custom-agents.md) | BaseChatAgent, on_messages, on_messages_stream |
| 11 | [State and memory](./11-state-and-memory.md) | save_state/load_state, Memory protocol, ListMemory |
| 12 | [Memory and RAG](./12-memory-and-rag.md) | ChromaDB, Redis, RAG pattern, Mem0 integration |
| 13 | [Structured output and streaming](./13-structured-output-and-streaming.md) | output_content_type, streaming tokens, model context |
| 14 | [Component serialization](./14-component-serialization.md) | dump_component/load_component, JSON config, portability |
| 15 | [Extensions and ecosystem](./15-extensions-and-ecosystem.md) | autogen-ext, AutoGen Studio, Docker executor |
| 16 | [Reasoning models and providers](./16-reasoning-models-and-providers.md) | GPT-5, o3/o4-mini, Anthropic thinking mode |
| 17 | [MCP integration](./17-mcp-integration.md) | McpWorkbench, MCP server connectivity, session management |

### Prerequisites

- Completion of [CrewAI with Flows](../13-crewai-with-flows/00-crewai-with-flows.md) or equivalent agent framework experience
- Python 3.10+ installed
- Understanding of async/await in Python (see [Unit 02: Async Programming](../../02-python-for-ai-development/09-async-programming/00-async-programming.md))
- An OpenAI API key (or Azure OpenAI credentials)

### Installation

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

For additional capabilities:

```bash
# Azure OpenAI support
pip install -U "autogen-ext[azure]"

# Magentic-One agents (web surfer, file surfer)
pip install -U "autogen-ext[magentic-one,openai]"

# Memory extensions
pip install -U "autogen-ext[chromadb]" "autogen-ext[redis]"

# MCP server integration
pip install -U "autogen-ext[mcp]"
```

### Key resources

- 📖 [AutoGen Documentation](https://microsoft.github.io/autogen/stable/) — Official docs (v0.7.5)
- 🐙 [GitHub Repository](https://github.com/microsoft/autogen) — Source code and examples
- 📦 [PyPI: autogen-agentchat](https://pypi.org/project/autogen-agentchat/) — Package listing
- 📄 [Magentic-One Paper](https://arxiv.org/abs/2411.04468) — Technical report on the multi-agent system

---

**Next:** [Architecture and Core Concepts](./01-architecture-and-core-concepts.md)

---

[Back to AI Agents Overview](../00-overview.md)

<!--
Sources Consulted:
- AutoGen AgentChat Tutorial: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
- AutoGen Teams Tutorial: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html
- AutoGen Quickstart: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html
-->
