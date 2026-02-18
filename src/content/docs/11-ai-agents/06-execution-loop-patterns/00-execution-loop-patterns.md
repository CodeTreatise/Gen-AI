---
title: "Execution Loop Patterns"
---

# Execution Loop Patterns

## Overview

Execution loop patterns define how agents cycle through reasoning, acting, and observing results. While the previous lessons covered individual components — tools, memory, reasoning — this lesson focuses on how those components are orchestrated into repeatable execution cycles that drive agent behavior from start to finish.

Understanding these patterns is the difference between building a brittle agent that works for one scenario and building a robust system that handles diverse, unpredictable tasks. Each pattern represents a different trade-off between simplicity, flexibility, performance, and cost.

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Simple Loop (Reason → Act → Observe)](./01-simple-loop.md) | The foundational ReAct cycle — one tool call at a time with observation |
| 02 | [Iterative Refinement Loops](./02-iterative-refinement-loops.md) | Evaluator-optimizer patterns for improving output quality through multiple passes |
| 03 | [Parallel Execution](./03-parallel-execution.md) | Concurrent tool calls, independent subtasks, and result aggregation |
| 04 | [Conditional Branching](./04-conditional-branching.md) | Routing, decision points, and path selection based on context |
| 05 | [Loop Termination Conditions](./05-loop-termination-conditions.md) | Goal detection, quality thresholds, and knowing when to stop |
| 06 | [Maximum Iteration Limits](./06-maximum-iteration-limits.md) | Safety limits, cost protection, stuck detection, and escalation |

## Prerequisites

- [Tool Integration](../05-tool-integration/00-tool-integration.md) — tool invocation and coordination patterns
- [Reasoning and Planning](../03-reasoning-and-planning/00-reasoning-and-planning.md) — ReAct and other reasoning strategies
- [Agent Architecture](../02-agent-architecture/00-agent-architecture.md) — agent components and structure
- Async Python proficiency (`async`/`await`, `asyncio`)

## Learning objectives

By the end of this lesson, you will be able to:

- Implement the simple ReAct loop (reason → act → observe) from scratch and with frameworks
- Build evaluator-optimizer loops for iterative quality improvement
- Design parallel execution patterns with fan-out/fan-in aggregation
- Create conditional branching workflows with routing and merge strategies
- Define effective termination conditions that balance thoroughness and efficiency
- Set appropriate iteration limits with stuck detection and escalation triggers

---

**Next:** [Simple Loop (Reason → Act → Observe)](./01-simple-loop.md)

*[Back to Unit Overview](../00-overview.md)*
