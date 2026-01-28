---
title: "Built-in Tools & MCP Integration"
---

# Built-in Tools & MCP Integration

## Introduction

Modern AI APIs provide powerful built-in tools that extend model capabilities beyond text generation. This lesson covers OpenAI's built-in tools, Anthropic's computer use capabilities, and the Model Context Protocol (MCP) for connecting to external services.

### What We'll Cover

This lesson explores native tool capabilities and external integrations:

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01-openai-built-in-tools.md](./01-openai-built-in-tools.md) | OpenAI Built-in Tools | Web search, code interpreter, file search, image generation |
| [02-file-search-configuration.md](./02-file-search-configuration.md) | File Search Advanced | Ranking options, chunking, metadata filtering |
| [03-computer-use.md](./03-computer-use.md) | Computer Use | Screen interaction, actions, safety, sandboxing |
| [04-mcp-fundamentals.md](./04-mcp-fundamentals.md) | MCP Basics | Protocol overview, remote servers, transports |
| [05-mcp-workflow.md](./05-mcp-workflow.md) | MCP Workflow | Tool discovery, execution, error handling |
| [06-mcp-approval-system.md](./06-mcp-approval-system.md) | MCP Approvals | Approval requests, trusted servers, security |
| [07-openai-connectors.md](./07-openai-connectors.md) | OpenAI Connectors | Service integrations, OAuth, configuration |
| [08-mcp-security.md](./08-mcp-security.md) | MCP Security | Prompt injection, logging, data residency |

### Prerequisites

- Experience with function calling
- Understanding of API authentication
- Python development environment

---

## Tool Categories Overview

### Built-in vs External Tools

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI API REQUEST                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   BUILT-IN TOOLS    │    │   EXTERNAL TOOLS    │            │
│  │   (Native)          │    │   (MCP/Connectors)  │            │
│  │                     │    │                     │            │
│  │  • Web Search       │    │  • MCP Servers      │            │
│  │  • Code Interpreter │    │  • OAuth Connectors │            │
│  │  • File Search      │    │  • Custom APIs      │            │
│  │  • Image Generation │    │  • Database Access  │            │
│  │  • Computer Use     │    │  • Cloud Services   │            │
│  └─────────────────────┘    └─────────────────────┘            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   FUNCTION CALLING                         │ │
│  │                   (Custom Tools)                           │ │
│  │   • Your own function definitions                          │ │
│  │   • Execute in your application                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Comparison

| Feature | Built-in Tools | MCP Tools | Function Calling |
|---------|---------------|-----------|------------------|
| Execution | Server-side | Remote server | Your code |
| Setup | Minimal | Server URL | Schema definition |
| Control | Limited | Approval system | Full control |
| Latency | Optimized | Variable | Your infrastructure |
| Security | Provider managed | Trust required | Your responsibility |

---

## Quick Start Examples

### OpenAI Web Search

```python
from openai import OpenAI

client = OpenAI()

# Enable web search
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search"}],
    input="What are the latest developments in AI regulation?"
)

# Results include citations
print(response.output_text)
```

### Anthropic Computer Use

```python
from anthropic import Anthropic

client = Anthropic()

# Enable computer use
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080
    }],
    messages=[{
        "role": "user",
        "content": "Take a screenshot of the current screen"
    }]
)
```

### MCP Server Integration

```python
# Connect to remote MCP server
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_url": "https://mcp.example.com/api",
        "require_approval": "never"  # For trusted servers
    }],
    input="Search the database for recent orders"
)
```

---

## Learning Path

1. **Start with Built-in Tools** → Understand native capabilities
2. **Master File Search** → Configure advanced search options
3. **Explore Computer Use** → Understand automation potential
4. **Learn MCP Fundamentals** → Grasp the protocol
5. **Implement MCP Workflow** → Build integrations
6. **Configure Approvals** → Secure your implementation
7. **Use Connectors** → Integrate with services
8. **Apply Security** → Protect your application

---

## Further Reading

- [OpenAI Tools Documentation](https://platform.openai.com/docs/guides/tools) — Official guide
- [Anthropic Computer Use](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) — Computer control
- [Model Context Protocol](https://modelcontextprotocol.io/) — MCP specification
- [MCP Servers Registry](https://github.com/modelcontextprotocol/servers) — Available servers
