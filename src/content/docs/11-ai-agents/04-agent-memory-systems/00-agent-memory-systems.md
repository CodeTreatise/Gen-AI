---
title: "Agent Memory Systems"
---

# Agent Memory Systems

## Introduction

Memory is what transforms a stateless LLM into a capable agent. Without memory, every interaction starts from zero — the agent has no context about what it has already done, what the user prefers, or what failed previously. With memory, agents can maintain conversations across turns, learn from past interactions, and build personalized, adaptive experiences.

This lesson explores the complete landscape of agent memory: from short-term conversation context that lasts a single session, to long-term persistent storage that spans months of interactions. We'll examine how modern frameworks implement these concepts and how to design memory systems that scale.

### What we'll cover

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Short-term memory](./01-short-term-memory.md) | Conversation context, session state, recent actions, and temporary data |
| 02 | [Long-term memory](./02-long-term-memory.md) | Cross-session persistence, knowledge accumulation, and user preferences |
| 03 | [Working memory](./03-working-memory.md) | Active information management, attention focus, and capacity limits |
| 04 | [Memory retrieval strategies](./04-memory-retrieval-strategies.md) | Relevance-based, recency-weighted, and hybrid retrieval approaches |
| 05 | [Memory summarization](./05-memory-summarization.md) | Periodic compression, key point extraction, and hierarchical summaries |
| 06 | [Episodic vs. semantic memory](./06-episodic-vs-semantic-memory.md) | Event-based vs. fact-based memories, procedural memory, and integration patterns |

### Prerequisites

- Completion of [Reasoning & Planning](../03-reasoning-planning/00-reasoning-planning.md)
- Understanding of agent architecture and the [agent loop](../02-agent-architecture/02-agent-loop-structure.md)
- Familiarity with [state management](../02-agent-architecture/05-state-management.md) concepts
- Basic understanding of databases and key-value stores
- Python async/await knowledge

### Learning objectives

By the end of this lesson, you will be able to:

- ✅ Implement short-term memory using OpenAI Sessions, LangGraph checkpoints, and ADK session state
- ✅ Design long-term memory systems with cross-session persistence and namespaced storage
- ✅ Manage context window limits through trimming, compaction, and summarization
- ✅ Implement retrieval strategies that combine relevance, recency, and importance
- ✅ Choose the right memory type (episodic, semantic, procedural) for different use cases
- ✅ Build agents that learn and adapt from past interactions

---

**Next:** [Short-term Memory](./01-short-term-memory.md)

---

*[Back to Unit Overview](../00-overview.md)*
