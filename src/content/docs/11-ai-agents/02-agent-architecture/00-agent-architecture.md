---
title: "Agent Architecture"
---

# Agent Architecture

## Overview

This lesson explores the internal architecture that makes AI agents work. While Lesson 01 defined *what* agents are and *why* they matter, this lesson examines *how* they're built — the components, loops, data flows, and state management patterns that power modern agent systems.

Understanding architecture is essential for designing agents that are reliable, debuggable, and production-ready. Whether you use OpenAI Agents SDK, LangGraph, Google ADK, or build from scratch, the same fundamental patterns appear across every framework.

## Lessons in this section

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Core components](./01-core-components.md) | The five building blocks — planner, memory, tools, executor, and orchestrator — and how they interact |
| 02 | [Agent loop structure](./02-agent-loop-structure.md) | The perception-reasoning-action cycle that drives agent behavior, with framework implementations |
| 03 | [Input processing](./03-input-processing.md) | How agents parse user goals, gather context, extract constraints, and determine priorities |
| 04 | [Output generation](./04-output-generation.md) | How agents synthesize results, format responses, report actions, and generate summaries |
| 05 | [State management](./05-state-management.md) | State representation, persistence, transitions, and concurrency across agent frameworks |

## Prerequisites

- Completed [Agent Fundamentals](../01-agent-fundamentals/00-agent-fundamentals.md)
- Understanding of LLM APIs and function calling
- Familiarity with async/await patterns
- Basic understanding of design patterns

## Key takeaways

After completing this section, you will be able to:

✅ Identify and explain the five core components of any agent system
✅ Trace the flow of data through the agent loop from input to output
✅ Implement input processing pipelines that parse goals and extract constraints
✅ Design output generation systems that communicate results effectively
✅ Build state management systems that persist and transition cleanly

---

[Back to Unit 11 Overview](../00-overview.md) | [Previous: Agent Fundamentals](../01-agent-fundamentals/00-agent-fundamentals.md) | [Next: Core Components](./01-core-components.md)
