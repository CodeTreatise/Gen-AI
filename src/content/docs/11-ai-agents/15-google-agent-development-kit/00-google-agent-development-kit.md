---
title: "Google Agent Development Kit (ADK)"
---

# Google Agent Development Kit (ADK)

## Overview

Google's **Agent Development Kit (ADK)** is a production-ready, open-source framework for building, deploying, and orchestrating AI agents. Released in 2025, ADK provides a code-first, modular approach to agent development with native support for multi-agent architectures, rich tool ecosystems, and flexible deployment options. ADK is model-agnostic — while optimized for Gemini, it works with any LLM through its extensible model interface.

ADK represents Google's answer to the growing demand for enterprise-grade agent frameworks, offering built-in session management, state persistence, memory systems, and the Agent-to-Agent (A2A) protocol for cross-framework agent communication.

### What makes ADK unique

| Feature | Description |
|---------|-------------|
| **Multi-language** | SDKs for Python, TypeScript, Go, and Java |
| **Model-agnostic** | Works with Gemini, Claude, Ollama, vLLM, LiteLLM, and more |
| **Deployment-flexible** | Run locally, on Cloud Run, GKE, or Vertex AI Agent Engine |
| **Rich tool ecosystem** | FunctionTool, MCP tools, OpenAPI tools, built-in Google tools |
| **A2A Protocol** | Standardized agent-to-agent communication (Linux Foundation) |
| **Visual Builder** | No-code drag-and-drop agent design interface |
| **Bidi-streaming** | Real-time bidirectional audio/video/text streaming |

---

## Lesson contents

This lesson covers 17 sub-topics that progressively build your mastery of Google ADK:

| # | Topic | Description |
|---|-------|-------------|
| 01 | [ADK Architecture Overview](./01-adk-architecture-overview.md) | Core architecture, Agent class, project structure |
| 02 | [Agent Definition Patterns](./02-agent-definition-patterns.md) | LlmAgent, custom agents, configuration options |
| 03 | [Tool Creation in ADK](./03-tool-creation-in-adk.md) | FunctionTool, async tools, built-in integrations |
| 04 | [Multi-Agent Hierarchies](./04-multi-agent-hierarchies.md) | Parent-child relationships, delegation, AgentTool |
| 05 | [Session and State Management](./05-session-and-state-management.md) | Sessions, state prefixes, templating |
| 06 | [Memory Systems](./06-memory-systems.md) | Cross-session memory, PreloadMemoryTool |
| 07 | [Workflow Patterns](./07-workflow-patterns.md) | Coordinator, pipeline, fan-out, generator-critic |
| 08 | [Callbacks and Observability](./08-callbacks-and-observability.md) | Before/after hooks for agents, models, tools |
| 09 | [ADK Web Integration](./09-adk-web-integration.md) | FastAPI deployment, API server, REST endpoints |
| 10 | [Multi-Language SDK Support](./10-multi-language-sdk-support.md) | Python, TypeScript, Go, Java SDKs |
| 11 | [A2A Protocol](./11-a2a-protocol.md) | Agent-to-Agent communication standard |
| 12 | [Workflow Agents](./12-workflow-agents.md) | SequentialAgent, ParallelAgent, LoopAgent |
| 13 | [Visual Builder](./13-visual-builder.md) | No-code agent design interface |
| 14 | [Plugins System](./14-plugins-system.md) | Reflect/retry, extensible plugin architecture |
| 15 | [Resume Agents](./15-resume-agents.md) | Checkpoint recovery, workflow resumption |
| 16 | [Third-Party Tool Integrations](./16-third-party-tool-integrations.md) | MCP tools, Atlassian, GitHub, Slack |
| 17 | [Bidi-Streaming Live](./17-bidi-streaming-live.md) | Real-time audio/video/text streaming |

---

## Prerequisites

Before starting this lesson, you should be familiar with:

- ✅ Python async/await programming (Unit 2, Lesson 9)
- ✅ AI agent fundamentals (Unit 11, Lessons 1-3)
- ✅ Function calling and tool use (Unit 10)
- ✅ API integration basics (Unit 4)
- ✅ Multi-agent system concepts (Unit 12)

### Environment setup

```bash
# Install ADK
pip install google-adk

# Verify installation
adk --version

# Set up API key (for Gemini)
export GOOGLE_API_KEY="your-api-key-here"
```

---

## Learning objectives

By the end of this lesson, you will be able to:

1. ✅ Design and implement ADK agents with proper architecture patterns
2. ✅ Create custom tools and integrate third-party tools via MCP
3. ✅ Build multi-agent systems with hierarchical delegation
4. ✅ Manage sessions, state, and memory across conversations
5. ✅ Implement workflow patterns using SequentialAgent, ParallelAgent, and LoopAgent
6. ✅ Use callbacks for observability, guardrails, and behavior customization
7. ✅ Deploy agents via FastAPI, Cloud Run, or Vertex AI Agent Engine
8. ✅ Enable real-time bidirectional streaming for voice/video applications
9. ✅ Connect agents across frameworks using the A2A protocol

---

## Key resources

| Resource | URL |
|----------|-----|
| ADK Documentation | [google.github.io/adk-docs](https://google.github.io/adk-docs/) |
| ADK Python SDK | [github.com/google/adk-python](https://github.com/google/adk-python) |
| ADK TypeScript SDK | [github.com/google/adk-js](https://github.com/google/adk-js) |
| ADK Go SDK | [github.com/google/adk-go](https://github.com/google/adk-go) |
| ADK Java SDK | [github.com/google/adk-java](https://github.com/google/adk-java) |
| ADK Samples | [github.com/google/adk-samples](https://github.com/google/adk-samples) |
| A2A Protocol | [a2a-protocol.org](https://a2a-protocol.org/) |
| API Reference | [ADK API Reference](https://google.github.io/adk-docs/api-reference/) |

---

**Next:** [ADK Architecture Overview](./01-adk-architecture-overview.md)

---

[Back to AI Agents Unit Overview](../00-overview.md)
