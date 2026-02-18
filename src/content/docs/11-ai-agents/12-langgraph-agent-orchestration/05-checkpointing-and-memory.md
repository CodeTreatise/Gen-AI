---
title: "Checkpointing and Memory"
---

# Checkpointing and Memory

## Introduction

Agents without memory are stateless — every invocation starts from scratch. In production, you need agents that remember conversations, recover from failures, and let users pick up where they left off. LangGraph's persistence layer provides all of this through **checkpointers** and a **memory store**.

This lesson covers the complete persistence system: from in-memory checkpointers for development to PostgreSQL-backed storage for production, plus the cross-thread memory store for long-term user knowledge.

### What We'll Cover

- What checkpointers are and why they matter
- Thread-based conversation management
- Checkpointer libraries: `MemorySaver`, `SqliteSaver`, `PostgresSaver`
- State snapshots: inspecting, replaying, and updating state
- The memory store for cross-thread, long-term memory
- Semantic search in the memory store

### Prerequisites

- Completed [Core Concepts](./01-core-concepts.md) and [State Management](./03-state-management.md)
- Understanding of `StateGraph` compilation and invocation

---

## What Checkpointers Do

A checkpointer saves the graph state after every node execution. This unlocks several critical capabilities:

```mermaid
flowchart LR
    CP[Checkpointer] --> C1[Conversation Memory]
    CP --> C2[Fault Recovery]
    CP --> C3[Human-in-the-Loop]
    CP --> C4[Time Travel]
    CP --> C5[Debugging]
```

| Capability | Description |
|-----------|-------------|
| **Conversation memory** | Resume conversations across multiple invocations |
| **Fault recovery** | Restart from the last successful node after a crash |
| **Human-in-the-loop** | Pause, review, and resume execution |
| **Time travel** | Replay from any previous checkpoint |
| **Debugging** | Inspect state at every step of execution |

### Enabling Persistence

Pass a checkpointer when compiling your graph:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

def respond(state: MessagesState):
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Echo: {last.content}")]}

builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# Compile WITH a checkpointer
graph = builder.compile(checkpointer=MemorySaver())
```

---

## Threads: Conversation Management

Threads are the primary mechanism for separating conversations. Each thread has a unique `thread_id` that tells the checkpointer which state to load and save.

```python
# Thread 1: Alice's conversation
config_alice = {"configurable": {"thread_id": "alice-001"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello!")]},
    config=config_alice,
)
print(result["messages"][-1].content)

# Continue Alice's conversation — state is preserved
result = graph.invoke(
    {"messages": [HumanMessage(content="How are you?")]},
    config=config_alice,
)
print(f"Alice has {len(result['messages'])} messages")

# Thread 2: Bob's conversation — completely separate state
config_bob = {"configurable": {"thread_id": "bob-001"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="Hi there!")]},
    config=config_bob,
)
print(f"Bob has {len(result['messages'])} messages")
```

**Output:**
```
Echo: Hello!
Alice has 4 messages
Bob has 2 messages
```

> **🔑 Key concept:** The `thread_id` is your persistent cursor. Reusing it resumes the same conversation. Using a new value starts a brand-new thread with empty state.

---

## Checkpointer Libraries

LangGraph provides several checkpointer implementations for different environments:

### MemorySaver (Development)

Stores checkpoints in-memory. Data is lost when the process exits.

```python
from langgraph.checkpoint.memory import MemorySaver

# Also aliased as InMemorySaver
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

> **Warning:** `MemorySaver` is for development and testing only. It does not persist data across restarts.

### SqliteSaver (Local Persistence)

Stores checkpoints in a SQLite database. Good for local development and single-server deployments.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based SQLite
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)

# Async version
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async_checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
```

### PostgresSaver (Production)

Stores checkpoints in PostgreSQL. Recommended for production multi-server deployments.

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)

# Required: create the checkpoint tables
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)

# Async version
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async_checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)
await async_checkpointer.setup()
```

### Choosing a Checkpointer

| Checkpointer | Persistence | Multi-server | Use Case |
|--------------|-------------|-------------|----------|
| `MemorySaver` | ❌ In-memory only | ❌ | Development, testing |
| `SqliteSaver` | ✅ Disk | ❌ Single server | Local apps, prototyping |
| `PostgresSaver` | ✅ Database | ✅ | Production deployments |
| `CosmosDBSaver` | ✅ Cloud | ✅ | Azure deployments |

---

## State Snapshots

With checkpointing enabled, you can inspect the graph's state at any point.

### Getting Current State

```python
config = {"configurable": {"thread_id": "thread-1"}}

# Run the graph
graph.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

# Get the current state snapshot
snapshot = graph.get_state(config)
print(f"Values: {snapshot.values}")
print(f"Next nodes: {snapshot.next}")
print(f"Config: {snapshot.config}")
```

**Output:**
```
Values: {'messages': [HumanMessage(content='Hello'), AIMessage(content='Echo: Hello')]}
Next nodes: ()
Config: {'configurable': {'thread_id': 'thread-1', 'checkpoint_id': '...'}}
```

### State History

View all checkpoints for a thread:

```python
# Get all state snapshots in reverse chronological order
for snapshot in graph.get_state_history(config):
    print(f"Step: {snapshot.metadata.get('step', '?')}")
    print(f"  Next: {snapshot.next}")
    print(f"  Messages: {len(snapshot.values.get('messages', []))}")
    print()
```

### Replaying From a Checkpoint

Resume execution from a specific checkpoint by providing its `checkpoint_id`:

```python
# Get a previous checkpoint
history = list(graph.get_state_history(config))
old_checkpoint = history[-1]  # The earliest state

# Replay from that point
replay_config = old_checkpoint.config
result = graph.invoke(None, config=replay_config)
```

### Updating State Manually

You can modify the state of a thread directly, without running the graph:

```python
# Update the state as if a specific node produced the update
graph.update_state(
    config,
    values={"messages": [AIMessage(content="Injected message")]},
    as_node="respond",
)

# The next invocation will see the injected message
result = graph.invoke(
    {"messages": [HumanMessage(content="What was your last message?")]},
    config=config,
)
```

> **💡 Tip:** `update_state` with `as_node` is powerful for testing and debugging. It lets you simulate what a specific node would have returned without actually running it.

---

## Memory Store: Cross-Thread Memory

Checkpointers manage per-thread state. The **memory store** provides a separate key-value store for information that spans across threads — like user preferences, past interactions, or learned facts.

### Basic Usage

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage

# Create a store
store = InMemoryStore()

def respond(state: MessagesState, *, store):
    """Node that accesses the memory store."""
    user_id = "user-123"
    namespace = ("users", user_id)
    
    # Read from the store
    memories = store.search(namespace)
    memory_text = "; ".join(m.value.get("fact", "") for m in memories)
    
    # Save a new fact
    store.put(namespace, key="greeting", value={"fact": "User said hello"})
    
    return {"messages": [AIMessage(content=f"Known facts: {memory_text}")]}

builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# Pass the store at compile time
graph = builder.compile(
    checkpointer=MemorySaver(),
    store=store,
)
```

### Namespaced Storage

The store organizes data into namespaces — tuple-based paths that act like directories:

```python
# User preferences
store.put(("users", "alice"), "preferences", {"theme": "dark", "language": "en"})

# Project data
store.put(("projects", "project-1"), "config", {"model": "claude-sonnet"})

# Search within a namespace
results = store.search(("users", "alice"))
for item in results:
    print(f"  {item.key}: {item.value}")
```

**Output:**
```
  preferences: {'theme': 'dark', 'language': 'en'}
```

### Semantic Search

The memory store supports embedding-based semantic search for finding relevant memories:

```python
from langchain_openai import OpenAIEmbeddings

store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
        "fields": ["fact"],  # Which fields to embed
    }
)

# Store some facts
store.put(("users", "alice"), "fact-1", {"fact": "Alice loves hiking in the mountains"})
store.put(("users", "alice"), "fact-2", {"fact": "Alice is a software engineer"})
store.put(("users", "alice"), "fact-3", {"fact": "Alice prefers dark mode"})

# Semantic search
results = store.search(
    ("users", "alice"),
    query="outdoor activities",
    limit=1,
)
print(results[0].value["fact"])
```

**Output:**
```
Alice loves hiking in the mountains
```

> **🤖 AI Context:** The memory store is how you build agents that learn about users over time. Unlike checkpointers (which store conversation history per thread), the memory store persists knowledge across all conversations — enabling personalization, preference learning, and contextual awareness.

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use `MemorySaver` for development, `PostgresSaver` for production | Match persistence to your environment |
| Always set `thread_id` in config | Without it, the checkpointer can't load or save state |
| Use the memory store for cross-thread knowledge | Checkpointers are per-thread; the store spans all threads |
| Call `checkpointer.setup()` for database-backed stores | Creates the required tables on first use |
| Keep stored values JSON-serializable | Non-serializable values will fail with some checkpointers |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `MemorySaver` in production | Switch to `PostgresSaver` or `SqliteSaver` for persistence |
| Forgetting `thread_id` in config | Always pass `{"configurable": {"thread_id": "..."}}` |
| Storing large objects in checkpoints | Keep state lean — move large data to external storage |
| Not calling `setup()` for database checkpointers | Call `checkpointer.setup()` before first use |
| Confusing checkpointer and store | Checkpointer = per-thread state history; Store = cross-thread memory |

---

## Hands-on Exercise

### Your Task

Build an agent with both conversation memory (checkpointer) and user knowledge (store).

### Requirements

1. Create a graph with `MemorySaver` checkpointer and `InMemoryStore`
2. The agent should remember facts the user tells it (store them in the memory store)
3. When asked "What do you know about me?", retrieve and display stored facts
4. Conversation history should persist across invocations within the same thread
5. User facts should persist across different threads

### Expected Result

```python
# Thread 1
graph.invoke({"messages": [HumanMessage(content="I love Python")]}, config=thread_1_config)

# Thread 2 (different conversation, same user)
result = graph.invoke({"messages": [HumanMessage(content="What do you know about me?")]}, config=thread_2_config)
# Should include "I love Python" from the memory store
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `store.put(("users", user_id), key, value)` to save facts
- Use `store.search(("users", user_id))` to retrieve facts
- The `user_id` can come from the `config["configurable"]` dict
- Pass both `checkpointer` and `store` to `builder.compile()`

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage
import uuid


store = InMemoryStore()


def respond(state: MessagesState, *, store, config):
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = ("users", user_id)
    last_msg = state["messages"][-1].content.lower()

    if "what do you know" in last_msg:
        memories = store.search(namespace)
        facts = [m.value.get("fact", "") for m in memories]
        response = f"I know: {', '.join(facts)}" if facts else "I don't know anything yet!"
    else:
        # Store the fact
        store.put(namespace, str(uuid.uuid4()), {"fact": last_msg})
        response = f"Got it! I'll remember that."

    return {"messages": [AIMessage(content=response)]}


builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile(checkpointer=MemorySaver(), store=store)

# Thread 1
config_1 = {"configurable": {"thread_id": "t1", "user_id": "alice"}}
graph.invoke({"messages": [HumanMessage(content="I love Python")]}, config=config_1)

# Thread 2 — different thread, same user
config_2 = {"configurable": {"thread_id": "t2", "user_id": "alice"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="What do you know about me?")]},
    config=config_2,
)
print(result["messages"][-1].content)
```

**Output:**
```
I know: i love python
```

</details>

### Bonus Challenges

- [ ] Add semantic search to the memory store with embeddings
- [ ] Implement a "forget" command that removes a specific memory
- [ ] View the state history using `graph.get_state_history(config)`

---

## Summary

✅ Checkpointers save graph state after every node, enabling memory and fault recovery

✅ Threads (`thread_id`) separate conversations — each thread has its own state history

✅ Use `MemorySaver` for development, `PostgresSaver` for production

✅ State snapshots let you inspect, replay, and manually update graph state

✅ The memory store provides cross-thread, namespaced storage with semantic search

**Next:** [Human-in-the-Loop Patterns](./06-human-in-the-loop-patterns.md)

---

## Further Reading

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — Full persistence documentation
- [LangGraph Memory Store](https://docs.langchain.com/oss/python/langgraph/persistence) — Cross-thread memory
- [PostgresSaver Setup](https://docs.langchain.com/oss/python/langgraph/persistence) — Production checkpointer guide

*Back to [LangGraph Agent Orchestration](./00-langgraph-agent-orchestration.md)*

<!-- 
Sources Consulted:
- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
-->
