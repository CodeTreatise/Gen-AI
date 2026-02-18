---
title: "Short-term memory"
---

# Short-term memory

## Introduction

Short-term memory is the agent's ability to remember what has happened within the current conversation. When a user asks "What state is it in?" after discussing the Golden Gate Bridge, the agent needs to recall the previous turn to understand that "it" refers to the bridge. This is the most fundamental form of agent memory — without it, every message would be an isolated, contextless interaction.

In practice, short-term memory is implemented as conversation history: the list of messages exchanged between the user and agent within a single session. Modern frameworks provide built-in mechanisms to manage this automatically, so you rarely need to handle message lists manually.

### What we'll cover

- Current session state and conversation history
- Recent actions and tool results tracking
- Active goal tracking across turns
- Temporary data within an invocation
- Framework implementations: OpenAI Sessions, LangGraph checkpoints, ADK session state

### Prerequisites

- Understanding of the [agent loop](../02-agent-architecture/02-agent-loop-structure.md) — how agents process messages
- Basic familiarity with async Python
- Understanding of key-value data structures

---

## Current session state

Session state is the most common form of short-term memory. It tracks the conversation history — every user message, assistant response, and tool call — within a single thread of interaction.

### How conversation context works

When an agent processes a message, it doesn't just see the latest input. It sees the entire conversation history for that session:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Memory as Session Memory

    User->>Agent: "What's the weather in Paris?"
    Agent->>Memory: Store user message
    Agent->>Agent: Call weather tool → 72°F
    Agent->>Memory: Store tool result
    Agent->>User: "It's 72°F in Paris"
    Agent->>Memory: Store assistant response

    User->>Agent: "How about London?"
    Memory->>Agent: Load full history (3 items)
    Note over Agent: Sees: Paris question + response + "How about London?"
    Agent->>Agent: Understands "How about" refers to weather
    Agent->>User: "It's 59°F in London"
```

The agent understands "How about London?" because it can see the full conversation context. Without session memory, it would have no idea what "How about" means.

### OpenAI Agents SDK: Sessions

The OpenAI Agents SDK provides the most straightforward approach with its `Session` system. Sessions automatically store and retrieve conversation history:

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a session — this handles all memory automatically
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session,
)
print(result.final_output)

# Second turn — agent automatically remembers previous context
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session,
)
print(result.final_output)
```

**Output:**
```
San Francisco
California
```

Behind the scenes, the `SQLiteSession`:
1. **Before each run:** Retrieves the conversation history and prepends it to the input
2. **After each run:** Stores all new items (user input, assistant responses, tool calls) in the session
3. **Context preservation:** Each subsequent run includes the full history

> **🔑 Key concept:** You don't need to manually call `.to_input_list()` or manage message arrays. The session handles everything — creating, storing, and retrieving conversation history automatically.

### Manual conversation management

Before Sessions existed, developers managed conversation history manually. This approach gives you more control but requires more code:

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Reply very concisely.")

# First turn
result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
print(result.final_output)

# Manually carry forward the conversation
new_input = result.to_input_list() + [
    {"role": "user", "content": "What state is it in?"}
]
result = await Runner.run(agent, new_input)
print(result.final_output)
```

**Output:**
```
San Francisco
California
```

### LangGraph: Thread-scoped checkpoints

LangGraph manages short-term memory through checkpointing. Every time a graph step completes, the entire state is saved to a checkpoint associated with a thread ID:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]

def chatbot(state: State):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile with checkpointer — this enables memory
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Use thread_id to maintain conversation context
config = {"configurable": {"thread_id": "thread_1"}}

result = app.invoke(
    {"messages": [{"role": "user", "content": "Hi, I'm Alice"}]},
    config=config,
)
print(result["messages"][-1].content)

# Same thread_id = same conversation
result = app.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config,
)
print(result["messages"][-1].content)
```

**Output:**
```
Hello Alice! Nice to meet you. How can I help you today?
Your name is Alice! You introduced yourself just a moment ago.
```

> **🤖 AI Context:** LangGraph's checkpointing saves the *entire* graph state — not just messages. This means tool results, intermediate computations, and any custom state fields are all preserved as part of short-term memory.

### Google ADK: Session state

Google ADK uses a `session.state` dictionary as the agent's scratchpad. State is scoped by key prefix:

```python
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

agent = LlmAgent(
    name="Assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. The user's name is {user_name}.",
    output_key="last_response",  # Auto-save response to state
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

# Create session with initial state
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_1",
    session_id="session_1",
    state={"user_name": "Alice"},
)
```

---

## Recent actions and results

Beyond conversation history, agents need to track what tools they've called and what results they received. This "action memory" is critical for multi-step tasks where later steps depend on earlier results.

### Tracking tool execution history

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ActionRecord:
    """Records a single tool invocation and its result."""
    tool_name: str
    arguments: dict
    result: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    duration_ms: float = 0.0

@dataclass
class ActionHistory:
    """Maintains a history of agent actions within a session."""
    records: list[ActionRecord] = field(default_factory=list)
    max_records: int = 100

    def add(self, record: ActionRecord):
        self.records.append(record)
        # Trim oldest if at capacity
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

    def get_recent(self, n: int = 5) -> list[ActionRecord]:
        """Get the N most recent actions."""
        return self.records[-n:]

    def get_by_tool(self, tool_name: str) -> list[ActionRecord]:
        """Get all actions for a specific tool."""
        return [r for r in self.records if r.tool_name == tool_name]

    def get_failures(self) -> list[ActionRecord]:
        """Get all failed actions."""
        return [r for r in self.records if not r.success]

# Usage
history = ActionHistory()
history.add(ActionRecord(
    tool_name="web_search",
    arguments={"query": "Python 3.13 release date"},
    result="October 7, 2024",
    success=True,
    duration_ms=450.0,
))
print(f"Recent actions: {len(history.get_recent())}")
```

**Output:**
```
Recent actions: 1
```

---

## Active goal tracking

In multi-turn interactions, the agent may be working toward a goal that spans multiple messages. Tracking the active goal helps the agent stay focused:

```python
from dataclasses import dataclass

@dataclass
class ActiveGoal:
    """Tracks the agent's current objective across turns."""
    description: str
    subtasks: list[str]
    completed_subtasks: list[str] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, completed, abandoned

    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 0.0
        return len(self.completed_subtasks) / len(self.subtasks) * 100

    def complete_subtask(self, subtask: str):
        if subtask in self.subtasks and subtask not in self.completed_subtasks:
            self.completed_subtasks.append(subtask)
        if len(self.completed_subtasks) == len(self.subtasks):
            self.status = "completed"

# Track a multi-step goal
goal = ActiveGoal(
    description="Compare Python web frameworks",
    subtasks=["Research Django", "Research FastAPI", "Research Flask", "Write comparison"],
)
goal.complete_subtask("Research Django")
goal.complete_subtask("Research FastAPI")
print(f"Goal progress: {goal.progress:.0f}%")
print(f"Status: {goal.status}")
```

**Output:**
```
Goal progress: 50%
Status: in_progress
```

---

## Temporary data

Some data only needs to exist for a single invocation — intermediate calculations, raw API responses before processing, or validation flags. Google ADK formalizes this with the `temp:` prefix:

```python
# Google ADK temp state — discarded after invocation completes
def process_order(context):
    # Temp data: only needed during this invocation
    context.state["temp:raw_api_response"] = {"status": 200, "data": {...}}
    context.state["temp:validation_passed"] = True

    # Session data: persists across invocations
    context.state["order_status"] = "confirmed"

    # User data: persists across all sessions for this user
    context.state["user:total_orders"] = context.state.get("user:total_orders", 0) + 1
```

| Prefix | Scope | Persistence | Use case |
|--------|-------|-------------|----------|
| *(none)* | Current session | Session lifetime | Task progress, conversation flags |
| `temp:` | Current invocation | Discarded after invocation | Raw API data, intermediate calculations |
| `user:` | All sessions for user | Persistent | Preferences, profile, history |
| `app:` | All users, all sessions | Persistent | Global settings, shared templates |

> **💡 Tip:** Even without ADK's prefix system, you can implement the same pattern by using namespaced dictionaries or separate storage backends for different scopes. The concept of scoped memory applies to all frameworks.

---

## Session storage backends

Different session backends suit different deployment scenarios:

| Backend | Framework | Persistence | Best for |
|---------|-----------|-------------|----------|
| In-memory | All | Lost on restart | Development, testing |
| SQLite | OpenAI SDK, LangGraph | File-based | Local apps, prototypes |
| PostgreSQL | OpenAI (SQLAlchemy), LangGraph | Full persistence | Production |
| Encrypted sessions | OpenAI SDK | With TTL expiry | Sensitive data |
| Server-managed | OpenAI (Conversations API) | OpenAI-hosted | Simplest production path |
| `InMemorySessionService` | ADK | Lost on restart | Development |
| `DatabaseSessionService` | ADK | Full persistence | Production |
| `VertexAiSessionService` | ADK | Google Cloud | Cloud-native production |

### Choosing a backend

```python
# Development — simple, in-memory
from agents import SQLiteSession
session = SQLiteSession("dev_session")  # In-memory by default

# Local persistence — file-based SQLite
session = SQLiteSession("user_123", "conversations.db")

# Production — PostgreSQL via SQLAlchemy
from agents.extensions.memory import SQLAlchemySession
session = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True,
)

# Sensitive data — encrypted with TTL
from agents.extensions.memory import EncryptedSession
session = EncryptedSession(
    session_id="user_123",
    underlying_session=underlying,
    encryption_key="your-secret-key",
    ttl=600,  # 10 minutes
)
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use session IDs that encode context | `"user_123_support_ticket_456"` makes debugging easier than `"abc123"` |
| Choose the right persistence backend | In-memory for dev, database for production — never mix them up |
| Scope data appropriately | Use temp for intermediates, session for conversation, user for preferences |
| Don't store non-serializable objects | Keep state as basic types (strings, numbers, lists, dicts) |
| Set maximum history lengths | Unbounded conversation history will eventually exceed context windows |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Manually managing `to_input_list()` when Sessions are available | Use `SQLiteSession` or `SQLAlchemySession` for automatic management |
| Storing everything in conversation history | Use structured state (dictionaries, databases) for non-message data |
| Using in-memory storage in production | Switch to file-based SQLite or PostgreSQL before deploying |
| Not separating session-scoped from user-scoped data | Use prefixes or namespaces to scope data correctly |
| Assuming state persists after restart with `InMemorySaver` | Use persistent checkpointers (SQLite, Postgres) for durability |

---

## Hands-on exercise

### Your task

Build a multi-turn customer support agent that maintains session state across turns, tracks the user's issue, and remembers context from earlier in the conversation.

### Requirements

1. Create an agent with session-based memory (use `SQLiteSession` or manual `to_input_list()`)
2. The agent should handle at least 3 turns of conversation about a support issue
3. Track the current support status (e.g., "gathering info", "diagnosing", "resolved") in state
4. Show that the agent remembers details from turn 1 when responding in turn 3

### Expected result

A conversation where the agent references earlier information without the user repeating it.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `SQLiteSession("support_session")` for automatic conversation management
- Add instructions that tell the agent to track the issue status
- Use `dynamic_instructions` to inject state into the agent's instructions
- The key test: in turn 3, ask something that requires knowledge from turn 1

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import asyncio
from agents import Agent, Runner, SQLiteSession

async def main():
    agent = Agent(
        name="SupportBot",
        instructions="""You are a customer support agent. 
        Track the customer's issue across the conversation.
        Reference details from earlier messages when relevant.
        Be concise but thorough.""",
    )

    session = SQLiteSession("support_demo")

    # Turn 1: Customer describes issue
    print("Turn 1:")
    result = await Runner.run(
        agent,
        "Hi, I'm having trouble with my order #12345. "
        "It was supposed to arrive yesterday but tracking shows it's stuck in Chicago.",
        session=session,
    )
    print(f"Agent: {result.final_output}\n")

    # Turn 2: Customer provides more detail
    print("Turn 2:")
    result = await Runner.run(
        agent,
        "I ordered the blue wireless headphones, the premium model.",
        session=session,
    )
    print(f"Agent: {result.final_output}\n")

    # Turn 3: Ask something requiring context from turn 1
    print("Turn 3:")
    result = await Runner.run(
        agent,
        "Can you check if there are weather delays in that city?",
        session=session,
    )
    print(f"Agent: {result.final_output}\n")
    # Agent should know "that city" = Chicago from turn 1

asyncio.run(main())
```

**Expected output:**
```
Turn 1:
Agent: I'm sorry to hear about the delay with order #12345. Let me look into why it's stuck in Chicago...

Turn 2:
Agent: Thank you for confirming — the blue premium wireless headphones on order #12345...

Turn 3:
Agent: I'll check for any weather-related delays in Chicago that might be affecting your shipment...
```

The agent remembers "Chicago" from turn 1 without the user repeating it in turn 3.

</details>

### Bonus challenges

- [ ] Add a `dynamic_instructions` function that injects the current support status into the agent's prompt
- [ ] Implement `pop_item()` to allow the user to "undo" their last message
- [ ] Use `AdvancedSQLiteSession` to track token usage across turns

---

## Summary

✅ **Short-term memory** maintains conversation context within a single session, enabling multi-turn interactions

✅ **OpenAI Sessions** (`SQLiteSession`, `SQLAlchemySession`, `EncryptedSession`) provide automatic conversation history management

✅ **LangGraph checkpoints** save the entire graph state — messages plus custom state — at each step, scoped by thread ID

✅ **Google ADK** uses prefixed state keys (`temp:`, `user:`, `app:`) to scope data to different lifetimes

✅ **Choose storage backends** based on your deployment: in-memory for dev, SQLite for local, PostgreSQL for production

**Next:** [Long-term Memory](./02-long-term-memory.md)

---

## Further reading

- [OpenAI Agents SDK Sessions](https://openai.github.io/openai-agents-python/sessions/) — Complete Sessions documentation
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — Checkpointing and threads
- [Google ADK Session State](https://google.github.io/adk-docs/sessions/state/) — State prefixes and scoping
- [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory) — Short vs long-term memory concepts

*[Back to Agent Memory Systems Overview](./00-agent-memory-systems.md)*

<!--
Sources Consulted:
- OpenAI Agents SDK Sessions docs: https://openai.github.io/openai-agents-python/sessions/
- OpenAI Agents SDK Running Agents (conversation management): https://openai.github.io/openai-agents-python/running_agents/
- LangGraph Memory overview (short-term memory, thread-scoped checkpoints): https://docs.langchain.com/oss/python/langgraph/memory
- LangGraph Persistence (checkpointers, threads, state snapshots): https://docs.langchain.com/oss/python/langgraph/persistence
- Google ADK Session State (prefixes, scoping, state updates): https://google.github.io/adk-docs/sessions/state/
- Anthropic Building Effective Agents (augmented LLM, memory): https://www.anthropic.com/engineering/building-effective-agents
-->
