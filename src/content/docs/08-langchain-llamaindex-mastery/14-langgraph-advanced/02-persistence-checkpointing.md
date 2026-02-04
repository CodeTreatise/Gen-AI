---
title: "Persistence & Checkpointing"
---

# Persistence & Checkpointing

## Introduction

Persistence is what makes LangGraph production-ready. By saving graph state at each step, checkpointers enable conversation memory, fault recovery, time-travel debugging, and human-in-the-loop workflows.

In this lesson, we'll explore checkpointer types, thread management, state access patterns, and durability modes.

### What We'll Cover

- How checkpointing works
- Threads and checkpoints
- Checkpointer implementations (Memory, SQLite, PostgreSQL)
- Accessing and updating state
- Time travel debugging
- Durability modes
- Memory store for cross-thread data

### Prerequisites

- StateGraph fundamentals
- Basic graph execution
- Understanding of state management

---

## How Checkpointing Works

```mermaid
flowchart TD
    subgraph Execution["Graph Execution"]
        N1[Node 1] --> CP1[(Checkpoint 1)]
        CP1 --> N2[Node 2]
        N2 --> CP2[(Checkpoint 2)]
        CP2 --> N3[Node 3]
        N3 --> CP3[(Checkpoint 3)]
    end
    
    subgraph Storage["Checkpointer"]
        CP1 --> DB[(Database)]
        CP2 --> DB
        CP3 --> DB
    end
    
    DB --> Resume[Resume from Any Point]
```

A checkpoint is a **snapshot of graph state** saved at each super-step. Checkpoints enable:

| Capability | Description |
|------------|-------------|
| **Memory** | Conversations persist across interactions |
| **Interrupts** | Pause and resume execution |
| **Fault Recovery** | Resume from last successful step |
| **Time Travel** | Replay and fork from any checkpoint |
| **Auditing** | Track state changes for debugging |

---

## Threads

A **thread** is a unique identifier for a sequence of checkpoints:

```python
# Thread ID identifies the conversation
config = {"configurable": {"thread_id": "user-123-chat-1"}}

# First interaction
result1 = app.invoke({"messages": [...]}, config)

# Second interaction - same thread continues conversation
result2 = app.invoke({"messages": [...]}, config)
```

### Thread Properties

| Property | Description |
|----------|-------------|
| `thread_id` | Primary key for storing checkpoints |
| Isolated | Each thread has independent state |
| Persistent | Survives across process restarts (with durable checkpointer) |
| Required | Must be provided for any graph with a checkpointer |

---

## Checkpointer Implementations

### MemorySaver (Development)

In-memory storage for development and testing:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "dev-session"}}
result = app.invoke(input, config)
```

> **Warning:** Data is lost when the process terminates. Use only for development.

### SqliteSaver (Local Persistence)

SQLite database for local development with persistence:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Create connection
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)

# Create saver
sqlite_saver = SqliteSaver(conn)

# Compile with saver
app = graph.compile(checkpointer=sqlite_saver)
```

Installation:

```bash
pip install langgraph-checkpoint-sqlite
```

### PostgresSaver (Production)

PostgreSQL for production deployments:

```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# Create connection pool
pool = ConnectionPool(
    conninfo="postgresql://user:password@localhost:5432/langgraph"
)

# Create saver
postgres_saver = PostgresSaver(pool)

# Initialize tables (run once)
postgres_saver.setup()

# Compile
app = graph.compile(checkpointer=postgres_saver)
```

Installation:

```bash
pip install langgraph-checkpoint-postgres
```

### Async Checkpointers

For async applications:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Use with async methods
result = await app.ainvoke(input, config)
```

---

## Checkpointer Comparison

| Checkpointer | Persistence | Use Case | Performance |
|--------------|-------------|----------|-------------|
| `MemorySaver` | None (RAM only) | Development, testing | Fastest |
| `SqliteSaver` | File-based | Local dev, single-server | Fast |
| `PostgresSaver` | Database | Production, multi-server | Scalable |

---

## Accessing State

### Get Current State

```python
config = {"configurable": {"thread_id": "my-thread"}}

# Get the latest state snapshot
state = app.get_state(config)

print(state.values)      # Current state values
print(state.next)        # Next nodes to execute
print(state.config)      # Configuration including checkpoint_id
print(state.metadata)    # Step number, writes, etc.
```

### StateSnapshot Properties

| Property | Description |
|----------|-------------|
| `values` | The state values at this checkpoint |
| `next` | Tuple of node names to execute next |
| `config` | Config with thread_id and checkpoint_id |
| `metadata` | Source, step number, writes |
| `tasks` | Pending tasks, interrupts, errors |
| `created_at` | Timestamp of checkpoint creation |
| `parent_config` | Config of the previous checkpoint |

---

## State History

Retrieve all checkpoints for a thread:

```python
config = {"configurable": {"thread_id": "my-thread"}}

# Get all checkpoints (most recent first)
for checkpoint in app.get_state_history(config):
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values}")
    print(f"  Next: {checkpoint.next}")
    print(f"  ID: {checkpoint.config['configurable']['checkpoint_id']}")
    print()
```

### Example Output

```
Step 2: {'foo': 'b', 'bar': ['a', 'b']}
  Next: ()
  ID: 1ef663ba-28fe-6528-8002-5a559208592c

Step 1: {'foo': 'a', 'bar': ['a']}
  Next: ('node_b',)
  ID: 1ef663ba-28f9-6ec4-8001-31981c2c39f8

Step 0: {'foo': '', 'bar': []}
  Next: ('node_a',)
  ID: 1ef663ba-28f4-6b4a-8000-ca575a13d36a
```

---

## Updating State Externally

Modify state directly (for human-in-the-loop):

```python
# Update state values
app.update_state(
    config,
    {"messages": [HumanMessage(content="Manual addition")]}
)

# Update as if from a specific node
app.update_state(
    config,
    {"result": "Modified value"},
    as_node="processing_node"
)
```

### How Updates Work

Updates go through reducers:

```python
# If state has: bar: Annotated[list, operator.add]

# Current state: {"foo": 1, "bar": ["a"]}
app.update_state(config, {"foo": 2, "bar": ["b"]})
# New state: {"foo": 2, "bar": ["a", "b"]}
#             ↑ overwrites    ↑ appends (has reducer)
```

---

## Time Travel

### Replaying from a Checkpoint

Resume from any previous checkpoint:

```python
# Get checkpoint history
checkpoints = list(app.get_state_history(config))

# Get a specific checkpoint ID
checkpoint_id = checkpoints[2].config["configurable"]["checkpoint_id"]

# Create replay config
replay_config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": checkpoint_id
    }
}

# Replay from that point (with new input)
result = app.invoke(new_input, replay_config)
```

### Forking Execution

Create an alternative timeline:

```python
# Fork from a previous checkpoint
fork_config = {
    "configurable": {
        "thread_id": "my-thread-fork",  # New thread!
        "checkpoint_id": checkpoint_id   # Starting point
    }
}

# This creates a new branch of execution
result = app.invoke(different_input, fork_config)
```

```mermaid
flowchart TD
    C0[Checkpoint 0] --> C1[Checkpoint 1]
    C1 --> C2[Checkpoint 2]
    C2 --> C3[Checkpoint 3]
    
    C1 -->|Fork| F1[Fork Checkpoint 1]
    F1 --> F2[Fork Checkpoint 2]
```

---

## Durability Modes

Control how frequently state is persisted:

```python
# In stream or invoke
result = app.invoke(
    input,
    config,
    durability="sync"  # Options: "exit", "async", "sync"
)
```

| Mode | When State Saves | Use Case |
|------|------------------|----------|
| `"exit"` | Only on graph exit | Best performance, less durable |
| `"async"` | Async during execution | Good balance |
| `"sync"` | Before each step | Highest durability, slower |

> **Tip:** Use `"sync"` when you need to recover from crashes mid-execution.

---

## Memory Store

For data that needs to persist **across threads** (like user preferences):

```python
from langgraph.store.memory import InMemoryStore
import uuid

# Create store
store = InMemoryStore()

# Namespace for this user's memories
namespace = ("user-123", "preferences")

# Store a memory
memory_id = str(uuid.uuid4())
store.put(namespace, memory_id, {"theme": "dark", "language": "en"})

# Retrieve memories
memories = store.search(namespace)
print(memories[-1].value)  # {"theme": "dark", "language": "en"}
```

### Using Store in Graphs

```python
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig


def node_with_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    """Access cross-thread memory in a node."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    
    # Retrieve user's memories
    memories = store.search(namespace)
    
    # Use memories in processing...
    
    # Store new memory
    store.put(namespace, str(uuid.uuid4()), {"last_topic": "persistence"})
    
    return {"result": "processed with memory"}


# Compile with both checkpointer and store
app = graph.compile(
    checkpointer=MemorySaver(),
    store=store
)

# Invoke with user_id
config = {
    "configurable": {
        "thread_id": "chat-1",
        "user_id": "user-123"
    }
}
```

### Semantic Search in Store

```python
from langchain.embeddings import init_embeddings

# Create store with embeddings
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["content"]
    }
)

# Store with embeddings
store.put(namespace, memory_id, {"content": "User loves Italian food"})

# Semantic search
results = store.search(
    namespace,
    query="What does the user like to eat?",
    limit=3
)
```

---

## Encryption

Encrypt persisted state for security:

```python
import os
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

# Set encryption key
os.environ["LANGGRAPH_AES_KEY"] = "your-32-byte-key-here"

# Create encrypted serializer
serde = EncryptedSerializer.from_pycryptodome_aes()

# Use with checkpointer
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://...",
    serde=serde
)
```

---

## Complete Example: Persistent Chat

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini")


def chat(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(MessagesState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile with persistence
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# Conversation with persistence
config = {"configurable": {"thread_id": "alice-chat"}}

# First message
result1 = app.invoke(
    {"messages": [("user", "Hi! My name is Alice.")]},
    config
)
print(result1["messages"][-1].content)

# Second message - remembers context!
result2 = app.invoke(
    {"messages": [("user", "What's my name?")]},
    config
)
print(result2["messages"][-1].content)  # "Your name is Alice!"


# View conversation history
for checkpoint in app.get_state_history(config):
    step = checkpoint.metadata.get("step", "?")
    msg_count = len(checkpoint.values.get("messages", []))
    print(f"Step {step}: {msg_count} messages")
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| Use meaningful thread IDs | Include user/session context |
| Choose appropriate checkpointer | Memory for dev, Postgres for prod |
| Set durability mode based on needs | Higher durability for critical workflows |
| Use Memory Store for cross-thread data | User preferences, long-term memory |
| Enable encryption in production | Protect sensitive conversation data |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Missing thread_id | Always provide in config |
| Using MemorySaver in production | Use SqliteSaver or PostgresSaver |
| Not calling setup() for Postgres | Run `saver.setup()` once |
| Forgetting checkpointer with interrupts | Interrupts require checkpointer |
| Same thread_id for different users | Include user context in thread_id |

---

## Hands-on Exercise

### Your Task

Build a note-taking app with:
1. Persistent notes across sessions
2. Ability to view note history
3. Time travel to restore previous versions

### Requirements

1. State: `note_content`, `version`, `last_modified`
2. Nodes: `add_note`, `view_history`
3. Use SqliteSaver for persistence
4. Implement version restoration

### Expected Flow

```python
# Add first note
app.invoke({"note_content": "First note"}, config)

# Add second note
app.invoke({"note_content": "Updated note"}, config)

# View history
for cp in app.get_state_history(config):
    print(f"v{cp.values['version']}: {cp.values['note_content']}")

# Restore to version 1
# ... implementation
```

<details>
<summary>💡 Hints (click to expand)</summary>

1. Increment version in each note update
2. Use `get_state_history()` to list versions
3. Use `checkpoint_id` to restore specific version
4. Consider using `update_state()` for restoration

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import sqlite3
from datetime import datetime
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


class NoteState(TypedDict):
    note_content: str
    version: int
    last_modified: str


def update_note(state: NoteState) -> dict:
    """Update note with new version."""
    return {
        "version": state.get("version", 0) + 1,
        "last_modified": datetime.now().isoformat()
    }


# Build graph
graph = StateGraph(NoteState)
graph.add_node("update", update_note)
graph.add_edge(START, "update")
graph.add_edge("update", END)

# Compile with SQLite persistence
conn = sqlite3.connect("notes.db", check_same_thread=False)
saver = SqliteSaver(conn)
app = graph.compile(checkpointer=saver)


# Usage
config = {"configurable": {"thread_id": "my-notes"}}

# Add notes
app.invoke({"note_content": "First note", "version": 0, "last_modified": ""}, config)
app.invoke({"note_content": "Updated note", "version": 0, "last_modified": ""}, config)
app.invoke({"note_content": "Third version", "version": 0, "last_modified": ""}, config)

# View history
print("Note History:")
checkpoints = list(app.get_state_history(config))
for cp in checkpoints:
    v = cp.values.get("version", 0)
    content = cp.values.get("note_content", "")
    print(f"  v{v}: {content}")

# Restore to v1
if len(checkpoints) >= 3:
    restore_id = checkpoints[2].config["configurable"]["checkpoint_id"]
    restore_config = {
        "configurable": {
            "thread_id": "my-notes",
            "checkpoint_id": restore_id
        }
    }
    
    # Fork to a restored version
    restored = app.invoke(
        {"note_content": "Restored to v1", "version": 0, "last_modified": ""},
        restore_config
    )
    print(f"\nRestored: v{restored['version']}: {restored['note_content']}")
```

</details>

---

## Summary

✅ Checkpointers save state at each step for persistence and recovery

✅ Threads isolate conversations using unique `thread_id`

✅ Use `MemorySaver` for development, `PostgresSaver` for production

✅ `get_state()` and `get_state_history()` access checkpoint data

✅ Time travel enables replaying and forking from any checkpoint

✅ Memory Store persists data across threads (user preferences, long-term memory)

**Next:** [Subgraphs & Composition →](./03-subgraphs-composition.md)

---

## Further Reading

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [Time Travel Guide](https://docs.langchain.com/oss/python/langgraph/use-time-travel)

---

<!-- 
Sources Consulted:
- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Durable Execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
-->
