---
title: "OpenAI Agents SDK"
---

# OpenAI Agents SDK

## Overview

The **OpenAI Agents SDK** is a production-grade, Python-first framework for building agentic AI applications. Originally inspired by the experimental Swarm project, it has evolved into a fully supported, open-source SDK (v0.8.1, 18.8k+ GitHub stars) that provides the primitives you need to build single-agent tools, multi-agent orchestrations, and voice-enabled AI systems.

In this lesson, we explore every major feature of the SDK — from the `Agent` class and `Runner` execution model to tools, handoffs, guardrails, tracing, sessions, voice, and more.

### Why the OpenAI Agents SDK?

| Feature | Benefit |
|---------|---------|
| Python-first design | No DSLs or YAML — pure Python classes and decorators |
| Minimal abstraction | Just `Agent`, `Runner`, and `Tool` — learn in minutes |
| Built-in tracing | Free dashboard at platform.openai.com/traces |
| Multi-agent support | Handoffs and agents-as-tools out of the box |
| Model flexibility | Use OpenAI, Anthropic, Google, or local models via LiteLLM |
| Production features | Sessions, guardrails, streaming, voice — all included |

### Prerequisites

Before diving in, you should be comfortable with:

- Python fundamentals (async/await, decorators, type hints)
- AI agent concepts from earlier lessons in this unit
- Basic understanding of LLM APIs and tool calling

### What we'll cover

This lesson is organized into the following sub-lessons:

| File | Topic |
|------|-------|
| [01 — Agent class fundamentals](./01-agent-class-fundamentals.md) | Defining agents, instructions, output types, hooks |
| [02 — Runner execution model](./02-runner-execution-model.md) | Running agents, the agent loop, RunConfig |
| [03 — Tool implementation patterns](./03-tool-implementation-patterns.md) | Function tools, hosted tools, agents as tools |
| [04 — Handoffs and multi-agent orchestration](./04-handoffs-multi-agent.md) | Agent delegation, input filters, handoff prompts |
| [05 — Guardrails and safety](./05-guardrails-safety.md) | Input/output/tool guardrails, tripwires |
| [06 — Context management](./06-context-management.md) | RunContextWrapper, ToolContext, dependency injection |
| [07 — Tracing and observability](./07-tracing-observability.md) | Traces, spans, custom processors, external integrations |
| [08 — Voice and realtime agents](./08-voice-realtime-agents.md) | VoicePipeline, RealtimeAgent, audio streaming |
| [09 — Sessions and persistence](./09-sessions-persistence.md) | SQLiteSession, SQLAlchemy, encryption, compaction |
| [10 — Development utilities](./10-development-utilities.md) | REPL, agent visualization, debugging |
| [11 — LiteLLM model support](./11-litellm-model-support.md) | Non-OpenAI models, setup, usage tracking |
| [12 — Computer use capabilities](./12-computer-use-capabilities.md) | ComputerTool, ShellTool, ApplyPatchTool |

### Installation

```bash
pip install openai-agents
```

For optional features:

```bash
# Voice support
pip install 'openai-agents[voice]'

# LiteLLM integration
pip install 'openai-agents[litellm]'
```

### Quick taste

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
)

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)
```

**Output:**
```
The capital of France is Paris.
```

---

**Next:** [Agent Class Fundamentals](./01-agent-class-fundamentals.md)

---

*[Back to AI Agents Unit Overview](../00-overview.md)*

<!-- 
Sources Consulted:
- OpenAI Agents SDK homepage: https://openai.github.io/openai-agents-python/
- OpenAI Agents SDK quickstart: https://openai.github.io/openai-agents-python/quickstart/
- GitHub repository: https://github.com/openai/openai-agents-python
-->
