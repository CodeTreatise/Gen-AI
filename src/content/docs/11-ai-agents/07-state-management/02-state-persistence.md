---
title: "State Persistence"
---

# State Persistence

## Introduction

An agent that loses its state when the process ends is useless in production. Users expect conversations to survive server restarts. Long-running research agents need their progress saved if a network call fails. Multi-agent systems need shared state that persists across independent processes. State persistence transforms agents from fragile prototypes into reliable systems.

This lesson covers the full spectrum of persistence strategies: from in-memory stores for development, through SQLite for single-server deployments, to PostgreSQL for production. We'll explore how LangGraph's checkpointers and OpenAI's session backends handle persistence automatically, how cross-session memory enables agents to remember across conversations, and how serialization and encryption protect stored state.

### What we'll cover

- LangGraph checkpointer backends (InMemorySaver, SQLite, PostgreSQL)
- OpenAI Agents SDK session types (SQLiteSession, SQLAlchemy, Encrypted)
- Cross-session memory with LangGraph's Memory Store
- Serialization formats and encrypted state storage
- Choosing the right persistence backend for your use case

### Prerequisites

- Understanding of [Agent State Representation](./01-agent-state-representation.md)
- Familiarity with [Agent Memory Systems](../04-agent-memory-systems/00-agent-memory-systems.md)
- Basic knowledge of databases (SQL concepts)

---

## LangGraph checkpointers

LangGraph persists state through **checkpointers** — pluggable backends that save a snapshot of the graph state after every super-step. When you compile a graph with a checkpointer, persistence is automatic:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

class AgentState(MessagesState):
    task: str
    results: list[str]

def process_node(state: AgentState) -> dict:
    return {"results": [f"Processed: {state['task']}"]}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile WITH checkpointer — state is now persisted
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Every invocation saves state automatically
config = {"configurable": {"thread_id": "session_001"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Analyze this data"}], "task": "analysis"},
    config=config
)
```

**Output:**
```
# State is automatically saved after every super-step
# The thread_id "session_001" acts as the primary key
# Next invocation with the same thread_id resumes from saved state
```

> **🔑 Key concept:** The `thread_id` is the primary key for state persistence. Each unique thread_id maintains its own independent state history. Think of it as a conversation ID.

### Available checkpointer backends

LangGraph provides several checkpointer implementations, each suited to different environments:

```python
# 1. InMemorySaver — development and testing only
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 2. SqliteSaver — single-server, persistent storage
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("agent_state.db")

# 3. AsyncSqliteSaver — async-compatible SQLite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_conn_string("agent_state.db")

# 4. PostgresSaver — production multi-server
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agents"
)

# 5. AsyncPostgresSaver — async-compatible PostgreSQL
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql+asyncpg://user:pass@localhost:5432/agents"
)
```

| Backend | Package | Use Case | Persistence | Multi-Process |
|---------|---------|----------|:-----------:|:------------:|
| **InMemorySaver** | `langgraph-checkpoint` | Development, tests | ❌ | ❌ |
| **SqliteSaver** | `langgraph-checkpoint-sqlite` | Local apps, prototypes | ✅ | ❌ |
| **PostgresSaver** | `langgraph-checkpoint-postgres` | Production systems | ✅ | ✅ |
| **CosmosDBSaver** | `langgraph-checkpoint-cosmosdb` | Azure deployments | ✅ | ✅ |

> **Warning:** `InMemorySaver` loses all state when the process ends. Never use it in production — it exists only for rapid prototyping and unit tests.

### Installing checkpointer packages

Each backend requires its own package:

```bash
# Base (includes InMemorySaver)
pip install langgraph-checkpoint

# SQLite support
pip install langgraph-checkpoint-sqlite

# PostgreSQL support  
pip install langgraph-checkpoint-postgres

# Azure CosmosDB support
pip install langgraph-checkpoint-cosmosdb
```

### Thread-based state isolation

Each `thread_id` creates an independent state timeline. Multiple threads share the same database but never interfere with each other:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("conversations.db")
graph = builder.compile(checkpointer=checkpointer)

# User A's conversation
config_a = {"configurable": {"thread_id": "user_alice_session_1"}}
result_a = graph.invoke({"messages": [{"role": "user", "content": "Help with Python"}]}, config_a)

# User B's conversation — completely independent
config_b = {"configurable": {"thread_id": "user_bob_session_1"}}
result_b = graph.invoke({"messages": [{"role": "user", "content": "Help with Rust"}]}, config_b)

# User A continues — sees only their history
result_a2 = graph.invoke({"messages": [{"role": "user", "content": "Show me decorators"}]}, config_a)
```

**Output:**
```
# User A sees: Python context → decorators follow-up
# User B sees: Rust context only
# No cross-contamination between threads
```

---

## OpenAI Agents SDK sessions

The OpenAI Agents SDK manages persistence through **Sessions** — objects that automatically store and retrieve conversation history:

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# SQLiteSession — lightweight, file-based
session = SQLiteSession("user_123", "conversations.db")

# First turn
result = await Runner.run(agent, "What is state management?", session=session)
print(result.final_output)

# Second turn — session automatically loads previous history
result = await Runner.run(agent, "Give me an example", session=session)
print(result.final_output)
```

**Output:**
```
# Turn 1: "State management is the process of..."
# Turn 2: Agent sees Turn 1 and gives a contextual example
# No manual history management needed
```

### Session type comparison

The SDK provides several session implementations for different scales:

```python
# 1. SQLiteSession — development, small apps
from agents import SQLiteSession

# In-memory (lost on restart)
session = SQLiteSession("user_123")

# File-based (persistent)
session = SQLiteSession("user_123", "conversations.db")

# 2. SQLAlchemySession — production databases
from agents.extensions.memory import SQLAlchemySession

session = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True
)

# 3. AdvancedSQLiteSession — branching, analytics
from agents.extensions.memory import AdvancedSQLiteSession

session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="conversations.db",
    create_tables=True
)

# Track token usage automatically
result = await Runner.run(agent, "Hello", session=session)
await session.store_run_usage(result)

# Branch from a specific turn
await session.create_branch_from_turn(2)

# 4. EncryptedSession — encryption wrapper
from agents.extensions.memory import EncryptedSession

underlying = SQLAlchemySession.from_url(
    "user_123",
    url="sqlite+aiosqlite:///conversations.db",
    create_tables=True
)
session = EncryptedSession(
    session_id="user_123",
    underlying_session=underlying,
    encryption_key="your-secret-key",
    ttl=600  # Auto-expire after 10 minutes
)
```

| Session Type | Storage | Features | Use Case |
|-------------|---------|----------|----------|
| **SQLiteSession** | SQLite file / memory | Basic CRUD | Development, small apps |
| **SQLAlchemySession** | Any SQL database | Production-grade, async | Production systems |
| **AdvancedSQLiteSession** | SQLite file | Branching, usage analytics | Research, debugging |
| **EncryptedSession** | Wraps any session | AES encryption, TTL | Sensitive data, compliance |
| **OpenAIConversationsSession** | OpenAI servers | Cloud-hosted | OpenAI-managed storage |

### Session operations

Sessions support CRUD operations for managing conversation history:

```python
from agents import SQLiteSession

session = SQLiteSession("user_123", "conversations.db")

# Get all conversation items
items = await session.get_items()

# Add items manually
new_items = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
await session.add_items(new_items)

# Remove the last item (useful for corrections)
last_item = await session.pop_item()

# Clear entire session
await session.clear_session()
```

**Output:**
```
# get_items() returns the full conversation history
# pop_item() removes and returns the most recent item
# Useful for "undo" functionality in chat interfaces
```

### Multiple agents sharing a session

Different agents can share the same session, enabling handoff patterns:

```python
from agents import Agent, Runner, SQLiteSession

support_agent = Agent(name="Support", instructions="Handle general questions")
billing_agent = Agent(name="Billing", instructions="Handle payment questions")

# Both agents share the same session
session = SQLiteSession("user_123", "conversations.db")

# Support agent handles first message
result = await Runner.run(support_agent, "I have a question about my account", session=session)

# Billing agent sees the full conversation history
result = await Runner.run(billing_agent, "What are my charges?", session=session)
```

**Output:**
```
# Billing agent sees: "I have a question about my account" + Support's response
# Enables seamless handoffs between specialized agents
```

---

## Cross-session memory with Memory Store

LangGraph checkpointers save state *per thread* — each conversation is independent. But what if an agent needs to remember things *across* conversations? That's where **Memory Store** comes in:

```python
from langgraph.store.memory import InMemoryStore

# Create a memory store
store = InMemoryStore()

# Store a memory — namespaced by user ID
store.put(
    namespace=("user_alice", "preferences"),
    key="communication_style",
    value={"style": "concise", "format": "bullet_points"}
)

store.put(
    namespace=("user_alice", "preferences"),
    key="expertise_level",
    value={"level": "advanced", "domain": "machine_learning"}
)

# Retrieve memories
items = store.search(namespace=("user_alice", "preferences"))
for item in items:
    print(f"{item.key}: {item.value}")
```

**Output:**
```
communication_style: {"style": "concise", "format": "bullet_points"}
expertise_level: {"level": "advanced", "domain": "machine_learning"}
```

### Using Memory Store in graph nodes

Nodes can access the store through the `store` parameter:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

class PersonalizedState(MessagesState):
    user_id: str

def personalized_node(state: PersonalizedState, store: BaseStore) -> dict:
    """Use cross-session memories to personalize responses."""
    user_id = state["user_id"]
    
    # Retrieve user preferences from memory store
    preferences = store.search(namespace=(user_id, "preferences"))
    pref_dict = {item.key: item.value for item in preferences}
    
    # Use preferences for personalization
    style = pref_dict.get("communication_style", {}).get("style", "standard")
    
    # Save new observations
    store.put(
        namespace=(user_id, "interactions"),
        key=f"last_topic",
        value={"topic": state["messages"][-1].content, "timestamp": "2025-01-15"}
    )
    
    return {"messages": [{"role": "assistant", "content": f"[{style} mode] Response here..."}]}

# Build graph with BOTH checkpointer and store
builder = StateGraph(PersonalizedState)
builder.add_node("respond", personalized_node)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore()
)
```

**Output:**
```
# Checkpointer: saves per-thread conversation state
# Memory Store: saves cross-thread user preferences and observations
# Together: each conversation is independent, but the agent remembers the user
```

> **🔑 Key concept:** Checkpointers save *conversation state* (per thread). Memory Store saves *long-term memories* (across threads). Use both together for agents that remember users across separate conversations.

### Semantic search in Memory Store

Memory Store supports embedding-based semantic search to find relevant memories:

```python
from langgraph.store.memory import InMemoryStore

# Initialize with embedding support
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)

# Store memories with natural language
store.put(
    namespace=("user_alice", "memories"),
    key="coding_preference",
    value={"text": "Alice prefers Python with type hints and pytest for testing"}
)

store.put(
    namespace=("user_alice", "memories"),
    key="project_context",
    value={"text": "Alice is building a RAG pipeline with LangChain and Pinecone"}
)

# Semantic search — finds relevant memories by meaning
results = store.search(
    namespace=("user_alice", "memories"),
    query="What testing framework does Alice use?"
)
# Returns the "coding_preference" memory (mentions pytest)
```

**Output:**
```
# Semantic search matches by meaning, not exact keywords
# Query about "testing framework" finds the memory mentioning "pytest"
# Enables natural retrieval of relevant user context
```

---

## Serialization and encryption

### Default serialization

LangGraph uses `JsonPlusSerializer` by default, which extends standard JSON to handle Python types like `datetime`, `set`, `bytes`, and dataclasses:

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serializer = JsonPlusSerializer()

# Handles types that standard JSON can't
data = {
    "timestamp": datetime.now(),
    "tags": {"python", "langgraph"},
    "raw_data": b"binary content"
}

# Serializes transparently — you don't need to do anything special
# The checkpointer handles this automatically
```

For objects that can't be JSON-serialized, the serializer falls back to `pickle` (configurable):

```python
# Pickle fallback for complex objects
serializer = JsonPlusSerializer(pickle_fallback=True)
```

### Encrypted state storage

For sensitive data, LangGraph provides an `EncryptedSerializer` that wraps the default serializer with AES encryption:

```python
import os
from langgraph.checkpoint.serde.enc import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# Set encryption key via environment variable
os.environ["LANGGRAPH_AES_KEY"] = "your-32-byte-encryption-key-here!!"

# Create encrypted serializer
encrypted_serde = EncryptedSerializer.from_pycryptodome_aes()

# Use with any checkpointer
checkpointer = SqliteSaver.from_conn_string(
    "agent_state.db",
    serde=encrypted_serde
)

# All state data is now encrypted at rest
graph = builder.compile(checkpointer=checkpointer)
```

**Output:**
```
# State data is encrypted before writing to the database
# Decrypted transparently when loading
# Protects sensitive conversation data at rest
```

For OpenAI Agents SDK, the `EncryptedSession` wrapper provides similar protection:

```python
from agents.extensions.memory import EncryptedSession, SQLAlchemySession

underlying = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True
)

session = EncryptedSession(
    session_id="user_123",
    underlying_session=underlying,
    encryption_key="your-secret-key",
    ttl=600  # Sessions expire after 10 minutes
)
```

> **🔒 Security:** Always encrypt state at rest when it contains user data, API keys, or PII. Both LangGraph and OpenAI SDK provide built-in encryption — use it.

---

## Choosing the right backend

```mermaid
flowchart TD
    A{Development or Production?} -->|Development| B{Need persistence across restarts?}
    B -->|No| C[InMemorySaver]
    B -->|Yes| D[SqliteSaver]
    A -->|Production| E{Single server or distributed?}
    E -->|Single server| D
    E -->|Distributed| F{Cloud provider?}
    F -->|Azure| G[CosmosDBSaver]
    F -->|AWS/GCP/Self-hosted| H[PostgresSaver]
```

| Scenario | LangGraph | OpenAI SDK |
|----------|-----------|------------|
| **Unit tests** | `InMemorySaver()` | `SQLiteSession("test")` (in-memory) |
| **Local development** | `SqliteSaver.from_conn_string("dev.db")` | `SQLiteSession("id", "dev.db")` |
| **Production (single)** | `SqliteSaver` | `SQLAlchemySession` |
| **Production (multi)** | `PostgresSaver` | `SQLAlchemySession` with PostgreSQL |
| **Sensitive data** | `EncryptedSerializer` + any saver | `EncryptedSession` wrapper |
| **Azure** | `CosmosDBSaver` | `SQLAlchemySession` |

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Always use a checkpointer in production | Without it, server restarts lose all conversation state |
| Use meaningful `thread_id` values | `"user_123_support_ticket_456"` > `"abc123"` for debugging |
| Separate checkpointer from Memory Store | Conversation state ≠ long-term memory — they have different lifecycles |
| Encrypt state containing user data | Compliance (GDPR, HIPAA) and basic security hygiene |
| Use PostgreSQL for multi-server deployments | SQLite doesn't handle concurrent writes from multiple processes |
| Set TTL on encrypted sessions | Prevents stale sensitive data from persisting indefinitely |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using `InMemorySaver` in production | Use `SqliteSaver` or `PostgresSaver` for persistence |
| Forgetting `thread_id` in config | Every invocation needs `{"configurable": {"thread_id": "..."}}` |
| Storing large files in state | Store file paths or URLs, not file contents |
| No encryption for sensitive conversations | Use `EncryptedSerializer` or `EncryptedSession` |
| Same `thread_id` for all users | Each user/conversation needs a unique `thread_id` |
| Ignoring Memory Store for cross-session data | Checkpointers are per-thread — use Store for shared data |

---

## Hands-on exercise

### Your task

Build a persistent agent that remembers user preferences across separate conversations. The agent should:
1. Use a SQLite checkpointer for conversation state
2. Use a Memory Store for cross-session user preferences
3. Detect when a user states a preference and store it
4. Load preferences at the start of each conversation

### Requirements

1. Define a state schema with `messages`, `user_id`, and `detected_preferences`
2. Create a `detect_preferences_node` that scans the last message for preference keywords
3. Create a `respond_node` that loads existing preferences from the store
4. Use `SqliteSaver` for the checkpointer (or `InMemorySaver` if SQLite package unavailable)
5. Demonstrate two separate conversations (different `thread_id`) where the second uses preferences from the first

### Expected result

Conversation 1: User says "I prefer detailed explanations" → agent detects and stores the preference. Conversation 2: New thread, same user → agent loads the stored preference and responds accordingly.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `store.put(namespace=(user_id, "preferences"), key="style", value={...})` to save preferences
- Use `store.search(namespace=(user_id, "preferences"))` to load preferences
- Pass a different `thread_id` for conversation 2 but keep the same `user_id` in state
- Access the store in nodes via `store: BaseStore` parameter

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

class PreferenceState(MessagesState):
    user_id: str
    detected_preferences: Annotated[list[str], add]

def detect_preferences_node(state: PreferenceState, store: BaseStore) -> dict:
    """Detect and store user preferences."""
    last_msg = state["messages"][-1].content.lower()
    detected = []
    
    preference_keywords = {
        "detailed": "detailed_explanations",
        "concise": "concise_responses",
        "examples": "include_examples",
        "formal": "formal_tone",
        "casual": "casual_tone"
    }
    
    for keyword, pref_key in preference_keywords.items():
        if keyword in last_msg:
            detected.append(pref_key)
            store.put(
                namespace=(state["user_id"], "preferences"),
                key=pref_key,
                value={"detected_from": last_msg, "active": True}
            )
    
    return {"detected_preferences": detected}

def respond_node(state: PreferenceState, store: BaseStore) -> dict:
    """Respond using stored preferences."""
    # Load all preferences
    prefs = store.search(namespace=(state["user_id"], "preferences"))
    pref_list = [item.key for item in prefs]
    
    if pref_list:
        style_note = f"[Using preferences: {', '.join(pref_list)}] "
    else:
        style_note = "[No preferences stored yet] "
    
    return {
        "messages": [{"role": "assistant", "content": f"{style_note}Here's my response..."}]
    }

# Build graph
builder = StateGraph(PreferenceState)
builder.add_node("detect", detect_preferences_node)
builder.add_node("respond", respond_node)
builder.add_edge(START, "detect")
builder.add_edge("detect", "respond")
builder.add_edge("respond", END)

store = InMemoryStore()
graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# Conversation 1: User states a preference
config_1 = {"configurable": {"thread_id": "conv_001"}}
result_1 = graph.invoke(
    {"messages": [{"role": "user", "content": "I prefer detailed explanations with examples"}],
     "user_id": "alice", "detected_preferences": []},
    config_1
)
print("Conv 1:", result_1["messages"][-1].content)

# Conversation 2: New thread, same user — preferences loaded
config_2 = {"configurable": {"thread_id": "conv_002"}}
result_2 = graph.invoke(
    {"messages": [{"role": "user", "content": "Tell me about state management"}],
     "user_id": "alice", "detected_preferences": []},
    config_2
)
print("Conv 2:", result_2["messages"][-1].content)
```

**Output:**
```
Conv 1: [Using preferences: detailed_explanations, include_examples] Here's my response...
Conv 2: [Using preferences: detailed_explanations, include_examples] Here's my response...
```

</details>

### Bonus challenges

- [ ] Add semantic search to the Memory Store to find relevant preferences by query
- [ ] Implement preference expiration using timestamps
- [ ] Add an `EncryptedSession` wrapper for sensitive preference data

---

## Summary

✅ LangGraph checkpointers (InMemorySaver, SqliteSaver, PostgresSaver) automatically save state after every super-step, keyed by `thread_id`

✅ OpenAI Sessions (SQLiteSession, SQLAlchemySession, EncryptedSession) manage conversation history automatically with CRUD operations

✅ Memory Store provides cross-session memory — shared data across threads, namespaced by user or context, with optional semantic search

✅ Serialization (JsonPlusSerializer) handles Python types transparently; EncryptedSerializer adds AES encryption at rest

✅ Choose your backend based on environment: InMemorySaver for tests, SQLite for local, PostgreSQL for production, encryption for sensitive data

**Next:** [Checkpointing for Recovery](./03-checkpointing-for-recovery.md)

---

## Further reading

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — Threads, checkpoints, Memory Store
- [LangGraph Checkpointer Libraries](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries) — Package installation and usage
- [OpenAI Agents SDK Sessions](https://openai.github.io/openai-agents-python/sessions/) — Session types and management
- [OpenAI EncryptedSession](https://openai.github.io/openai-agents-python/sessions/encrypted_session/) — Encryption wrapper docs

---

*[Back to State Management Overview](./00-state-management.md)*

<!-- 
Sources Consulted:
- LangGraph Persistence (Threads, Checkpoints, Memory Store): https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Graph API (State, Schemas): https://docs.langchain.com/oss/python/langgraph/graph-api
- OpenAI Agents SDK Sessions: https://openai.github.io/openai-agents-python/sessions/
- OpenAI Running Agents: https://openai.github.io/openai-agents-python/running_agents/
-->
