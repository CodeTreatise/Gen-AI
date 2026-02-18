---
title: "State management"
---

# State management

## Introduction

State is the agent's memory of everything that has happened during execution. It tracks the conversation so far, tools that have been called, results received, decisions made, and progress toward the goal. Without state management, agents forget what they've done, repeat actions, lose context, and fail on multi-step tasks.

This lesson explores how production agent frameworks handle state — from simple conversation history to complex graph state with reducers, checkpointers, and persistent sessions.

### What we'll cover

- State representation: how different frameworks model agent state
- State persistence: saving and restoring state across sessions
- State transitions: how state changes as the agent loop runs
- Concurrent state handling: managing state when multiple processes access it

### Prerequisites

- Completed [Output Generation](./04-output-generation.md)
- Understanding of the agent loop from [Agent Loop Structure](./02-agent-loop-structure.md)
- Familiarity with Python data classes and type hints

---

## State representation

Every agent framework models state differently. Understanding these representations is essential for choosing the right framework and designing effective agents.

### Conversation-based state

The simplest state model is a list of messages. This is the default in the OpenAI Agents SDK:

```python
# OpenAI Agents SDK: state is the conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "22°C, sunny", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "It's 22°C and sunny in Paris!"},
]
```

Each message captures one step of the agent loop. The conversation history *is* the state — the LLM reads it to understand what has happened and decide what to do next.

> **Note:** Conversation-based state is simple and works well for single-agent, single-session tasks. It breaks down when you need structured data, multi-agent coordination, or state that persists across sessions.

### Graph-based state (LangGraph)

LangGraph uses a typed state schema that goes far beyond conversation history. State is a structured object that nodes read from and write to:

```python
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState

# Simple: just use MessagesState (includes messages with add_messages reducer)
class BasicState(MessagesState):
    pass

# Complex: custom state with multiple fields
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query: str                                # The research question
    sources: Annotated[list[str], lambda a, b: a + b]  # Accumulates sources
    findings: Annotated[list[dict], lambda a, b: a + b]  # Accumulates findings
    draft_report: str                         # Current draft
    review_feedback: str                      # Reviewer's feedback
    iteration_count: int                      # How many review cycles
    status: str                               # Current phase
```

### Reducers: controlling how state updates

A critical concept in LangGraph is the **reducer** — a function that determines how new values combine with existing state. Without reducers, new values simply overwrite old ones:

```python
from typing import Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # OVERWRITE: new value replaces old (default behavior)
    status: str
    
    # APPEND: new values are added to the list
    messages: Annotated[list, add_messages]
    
    # CUSTOM REDUCER: merge logic you define
    scores: Annotated[list[float], lambda old, new: old + new]
    
    # ACCUMULATOR: count operations
    tool_calls_count: Annotated[int, lambda old, new: old + new]
```

Here's how reducers affect state updates:

```python
# Node returns a partial state update
def analyze_node(state: AgentState) -> dict:
    return {
        "status": "analyzing",              # Overwrites previous status
        "messages": [new_message],           # Appended via add_messages reducer
        "scores": [0.85],                    # Appended via custom reducer
        "tool_calls_count": 1,              # Added to existing count
    }

# Before: {"status": "idle", "messages": [...], "scores": [0.7], "tool_calls_count": 3}
# After:  {"status": "analyzing", "messages": [..., new_message], "scores": [0.7, 0.85], "tool_calls_count": 4}
```

> **Important:** Forgetting to add a reducer to a list field is one of the most common LangGraph bugs. Without the reducer, each node overwrites the list instead of appending to it, causing data loss.

### Session-based state (Google ADK)

Google ADK uses a `Session` object managed by a `SessionService`. State is split into two parts:

```python
# Google ADK session state
session = session_service.create_session(
    app_name="my_agent",
    user_id="user_123",
)

# Session state: persistent key-value pairs
session.state["user_name"] = "Alice"
session.state["preferences"] = {"language": "en", "theme": "dark"}

# Session events: the conversation and action history
# Events are appended automatically as the agent runs
for event in session.events:
    print(f"[{event.author}] {event.content}")
```

The ADK's state model separates *what happened* (events) from *what we know* (state dictionary). This makes it easy to track metadata without cluttering the conversation history:

```python
# In a tool or callback, update session state
def process_order(tool_context, order_id: str) -> str:
    """Process an order and track it in state."""
    result = order_api.process(order_id)
    
    # Update session state (persists across turns)
    tool_context.state["last_order_id"] = order_id
    tool_context.state["orders_processed"] = (
        tool_context.state.get("orders_processed", 0) + 1
    )
    
    return f"Order {order_id} processed successfully"
```

### Comparison of state models

| Feature | OpenAI Agents SDK | LangGraph | Google ADK |
|---------|------------------|-----------|------------|
| **State model** | Message list | Typed schema | Session + events |
| **State updates** | Append messages | Reducers (overwrite/append/custom) | Key-value + events |
| **Type safety** | Minimal | Strong (TypedDict/Pydantic) | Flexible dict |
| **Custom fields** | Via `context` object | Direct state fields | `session.state` dict |
| **Multi-agent state** | Per-agent messages | Shared graph state | Shared session |
| **Built-in history** | Conversation messages | `MessagesState` with `add_messages` | Session events |

---

## State persistence

In-memory state disappears when the process ends. For production agents, state must be persisted to survive restarts, handle long-running tasks, and support multi-session workflows.

### Checkpointing in LangGraph

LangGraph's checkpointing system saves the full graph state after every node execution. This enables pause/resume, time travel, and fault recovery:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Create a persistent checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Build the graph
graph = StateGraph(ResearchState)
graph.add_node("search", search_sources)
graph.add_node("analyze", analyze_findings)
graph.add_node("report", generate_report)
graph.add_edge(START, "search")
graph.add_edge("search", "analyze")
graph.add_edge("analyze", "report")
graph.add_edge("report", END)

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Run with a thread ID (acts as session identifier)
config = {"configurable": {"thread_id": "research-001"}}
result = await app.ainvoke(
    {"query": "AI trends 2025", "sources": [], "findings": []},
    config=config,
)
```

### Resuming from checkpoints

If a graph execution fails midway or needs human input, you can resume from the last checkpoint:

```python
# Get the state at any point
state = await app.aget_state(config)
print(f"Current status: {state.values['status']}")
print(f"Sources found: {len(state.values['sources'])}")

# Resume execution (picks up from where it stopped)
result = await app.ainvoke(None, config=config)
```

### Time travel

Checkpointing also enables "time travel" — going back to a previous state:

```python
# List all checkpoints for this thread
history = [s async for s in app.aget_state_history(config)]

# Go back to an earlier state
previous_state = history[2]  # Third checkpoint
await app.aupdate_state(config, previous_state.values)

# Re-run from that point (branching the execution)
result = await app.ainvoke(None, config=config)
```

### Sessions in the OpenAI Agents SDK

The OpenAI Agents SDK uses `Session` objects for conversation persistence:

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="Assistant",
    instructions="Help users across multiple conversations.",
)

# SQLiteSession persists conversation history
session = SQLiteSession("user_alice_session_1")

# Turn 1
result = await Runner.run(agent, "My name is Alice", session=session)

# ... hours later, new process ...

# Turn 2 (session restored from SQLite)
session = SQLiteSession("user_alice_session_1")
result = await Runner.run(agent, "What's my name?", session=session)
# Agent responds: "Your name is Alice."
```

### Google ADK session persistence

ADK supports multiple session backends through service interfaces:

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# Development: in-memory (lost on restart)
dev_service = InMemorySessionService()

# Production: database-backed (persists across restarts)
prod_service = DatabaseSessionService(
    connection_string="postgresql://user:pass@host/db"
)

# Both use the same API
session = prod_service.create_session(
    app_name="support_agent",
    user_id="user_456",
)
```

> **🤖 AI Context:** In the Google ADK, state changes from tools are not committed immediately. They're collected in a `state_delta` and committed only when the event is finalized. This prevents partial state corruption if the agent errors mid-execution.

---

## State transitions

State transitions define how state changes as the agent moves through its loop. Understanding transitions helps you design agents that make progress and don't get stuck.

### The state machine pattern

Many agent architectures can be modeled as state machines:

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: New input received
    Processing --> Planning: Goal parsed
    Planning --> Executing: Plan ready
    Executing --> Evaluating: Action complete
    Evaluating --> Planning: More work needed
    Evaluating --> Responding: Goal achieved
    Responding --> Idle: Response sent
    Executing --> Error: Action failed
    Error --> Planning: Retry
    Error --> Responding: Max retries reached
```

### Implementing state transitions in LangGraph

LangGraph's conditional edges are the primary mechanism for state transitions:

```python
from langgraph.graph import StateGraph, START, END

class TaskState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    status: str  # "planning", "executing", "evaluating", "complete", "error"
    attempts: int
    max_attempts: int
    result: str

def route_by_status(state: TaskState) -> str:
    """Determine next node based on current status."""
    if state["status"] == "complete":
        return "respond"
    if state["status"] == "error":
        if state["attempts"] < state["max_attempts"]:
            return "plan"  # Retry
        return "respond"  # Give up
    if state["status"] == "evaluating":
        return "evaluate"
    return "execute"

graph = StateGraph(TaskState)
graph.add_node("plan", plan_task)
graph.add_node("execute", execute_task)
graph.add_node("evaluate", evaluate_result)
graph.add_node("respond", generate_response)

graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", route_by_status)
graph.add_conditional_edges("execute", lambda s: "evaluate")
graph.add_conditional_edges("evaluate", route_by_status)
graph.add_edge("respond", END)
```

### Tracking transition history

For debugging and auditing, you should track state transitions:

```python
from datetime import datetime

class TransitionLog(BaseModel):
    """Record of a state transition."""
    timestamp: str
    from_state: str
    to_state: str
    trigger: str
    details: str

def log_transition(state: TaskState, new_status: str, trigger: str) -> dict:
    """Update status and log the transition."""
    log_entry = TransitionLog(
        timestamp=datetime.now().isoformat(),
        from_state=state.get("status", "initial"),
        to_state=new_status,
        trigger=trigger,
        details=f"Attempt {state.get('attempts', 0)}",
    )
    
    return {
        "status": new_status,
        "transition_log": [log_entry.model_dump()],
    }
```

---

## Concurrent state handling

When multiple agents or processes access shared state, concurrency issues arise. Race conditions, dirty reads, and lost updates can cause subtle bugs.

### The problem: shared mutable state

```mermaid
sequenceDiagram
    participant Agent A
    participant State
    participant Agent B
    
    Agent A->>State: Read score = 10
    Agent B->>State: Read score = 10
    Agent A->>State: Write score = 15 (added 5)
    Agent B->>State: Write score = 13 (added 3)
    Note over State: Final: 13 ❌<br/>Expected: 18
```

Agent B's write overwrites Agent A's update because both read the same initial value. This is a classic lost update problem.

### LangGraph's reducer solution

LangGraph's reducers solve the concurrent update problem for graph state. Because reducers define *how* to combine values rather than what the new value is, parallel nodes can safely update the same field:

```python
class SharedState(TypedDict):
    # The reducer ensures both updates are applied
    scores: Annotated[list[float], lambda old, new: old + new]
    errors: Annotated[list[str], lambda old, new: old + new]

# These can run in parallel via Send
def agent_a(state: SharedState) -> dict:
    return {"scores": [5.0]}  # Appends, doesn't overwrite

def agent_b(state: SharedState) -> dict:
    return {"scores": [3.0]}  # Also appends

# Result: scores = [5.0, 3.0] ✅ Both contributions preserved
```

### LangGraph's `Send` for parallel state updates

The `Send` API enables map-reduce patterns where multiple nodes run in parallel and their state updates are merged:

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

class FanOutState(TypedDict):
    topics: list[str]
    results: Annotated[list[dict], lambda old, new: old + new]

def distribute(state: FanOutState) -> list[Send]:
    """Fan out: send each topic to a research node."""
    return [
        Send("research", {"topic": topic})
        for topic in state["topics"]
    ]

def research(state: dict) -> dict:
    """Run in parallel for each topic."""
    topic = state["topic"]
    return {"results": [{"topic": topic, "summary": f"Research on {topic}"}]}

graph = StateGraph(FanOutState)
graph.add_node("research", research)
graph.add_conditional_edges(START, distribute)
graph.add_edge("research", END)
```

### Google ADK's state_delta approach

Google ADK handles concurrency through `state_delta` — a deferred state update mechanism:

```python
# ADK state updates are NOT immediate
# They're collected in state_delta and committed atomically

# During agent execution:
#   1. Tool writes to state → stored in state_delta (not committed)
#   2. Another tool reads state → sees OLD value (dirty read possible)
#   3. Event is finalized → all state_delta changes committed atomically

# This means within a single turn:
# - Tools see stale state from other tools in the same turn
# - But cross-turn state is always consistent
```

> **Warning:** Be aware of ADK's "dirty read" behavior. If Tool A updates `state["count"]` and Tool B reads `state["count"]` in the same turn, Tool B sees the old value. Design your tools to be independent within a single turn.

### Best practices for concurrent state

| Pattern | When to use | Framework support |
|---------|-------------|------------------|
| **Reducers** | Parallel nodes updating same field | LangGraph ✅ |
| **Atomic updates** | All-or-nothing state changes | ADK state_delta ✅ |
| **Pessimistic locking** | Critical sections (e.g., payments) | Custom implementation |
| **Event sourcing** | Full audit trail of all changes | ADK events, LangGraph checkpoints |
| **Idempotent operations** | Safe retries after failures | All frameworks (design pattern) |

---

## Best practices

| Practice | Why it matters |
|----------|---------------|
| Define a typed state schema upfront | Prevents schema drift and makes state predictable |
| Use reducers for list fields in LangGraph | Prevents accidental overwrites from parallel nodes |
| Enable checkpointing in production | Enables pause/resume, debugging, and fault recovery |
| Keep state minimal | Large state increases memory, slows serialization, and inflates LLM context |
| Log state transitions | Essential for debugging agent behavior |
| Design tools to be idempotent | Safe retries when state operations fail partway |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Forgetting reducers on list fields (LangGraph) | Always annotate accumulation fields with reducers |
| Putting all data in conversation messages | Use structured state fields for non-conversation data |
| No checkpointing in production | Enable persistent checkpointing for fault tolerance |
| Unbounded state growth | Set limits on list sizes and prune old entries |
| Reading stale state in ADK same-turn tools | Design tools to be independent within a single turn |
| Not handling state serialization errors | Ensure all state values are serializable (no lambdas, file handles, etc.) |

---

## Hands-on exercise

### Your task

Build a stateful research agent using LangGraph that tracks its progress through multiple phases and can be resumed from any checkpoint.

### Requirements

1. Define a `ResearchState` with typed fields and appropriate reducers
2. Implement at least 3 nodes: `gather`, `analyze`, `report`
3. Add conditional edges for state transitions (retry on failure)
4. Enable checkpointing with `SqliteSaver`
5. Demonstrate pausing and resuming execution

### Expected result

```
[Checkpoint 1] Status: gathering, Sources: 0
[Checkpoint 2] Status: gathering, Sources: 3
[Checkpoint 3] Status: analyzing, Findings: 0
[Checkpoint 4] Status: analyzing, Findings: 5
[Checkpoint 5] Status: complete, Report generated

--- Simulating crash and resume ---

Resumed from checkpoint 4
[Checkpoint 5] Status: complete, Report generated (re-run)
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `SqliteSaver.from_conn_string(":memory:")` for testing
- Pass `config={"configurable": {"thread_id": "my-thread"}}` to `ainvoke`
- Use `app.aget_state(config)` to inspect current state
- Use `app.aget_state_history(config)` to list all checkpoints
- Use `app.aupdate_state(config, values)` to set state for resume

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio

class ResearchState(TypedDict):
    query: str
    status: str
    sources: Annotated[list[str], lambda a, b: a + b]
    findings: Annotated[list[dict], lambda a, b: a + b]
    report: str
    attempts: int

def gather_sources(state: ResearchState) -> dict:
    """Gather research sources."""
    sources = [
        "arxiv.org/paper/123",
        "docs.python.org/3/tutorial",
        "blog.openai.com/agents",
    ]
    return {"sources": sources, "status": "analyzing", "attempts": 0}

def analyze_findings(state: ResearchState) -> dict:
    """Analyze gathered sources."""
    findings = [
        {"source": s, "insight": f"Key insight from {s}"}
        for s in state["sources"]
    ]
    return {"findings": findings, "status": "reporting"}

def generate_report(state: ResearchState) -> dict:
    """Generate final report."""
    report = f"Research Report: {state['query']}\n"
    report += f"Sources: {len(state['sources'])}\n"
    report += f"Findings: {len(state['findings'])}\n"
    for f in state["findings"]:
        report += f"  - {f['insight']}\n"
    return {"report": report, "status": "complete"}

def route(state: ResearchState) -> str:
    status = state.get("status", "gathering")
    if status == "analyzing":
        return "analyze"
    if status == "reporting":
        return "report"
    return "end"

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("gather", gather_sources)
graph.add_node("analyze", analyze_findings)
graph.add_node("report", generate_report)

graph.add_edge(START, "gather")
graph.add_conditional_edges("gather", route)
graph.add_conditional_edges("analyze", route)
graph.add_edge("report", END)

# Compile with checkpointer
checkpointer = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=checkpointer)

async def main():
    config = {"configurable": {"thread_id": "research-001"}}
    
    result = await app.ainvoke(
        {"query": "AI Agent Architectures", "sources": [], "findings": [], "status": "gathering", "attempts": 0, "report": ""},
        config=config,
    )
    
    # Print checkpoint history
    async for state in app.aget_state_history(config):
        v = state.values
        print(f"Status: {v.get('status')}, Sources: {len(v.get('sources', []))}, Findings: {len(v.get('findings', []))}")
    
    print(f"\nFinal Report:\n{result['report']}")

asyncio.run(main())
```

</details>

### Bonus challenges

- [ ] Add a `review` node that can send the report back to `analyze` for revision
- [ ] Implement a `max_iterations` guard that prevents infinite loops
- [ ] Add parallel source gathering using `Send` to process multiple queries simultaneously

---

## Summary

✅ **State representation** varies by framework: message lists (OpenAI SDK), typed schemas with reducers (LangGraph), or session dictionaries (Google ADK)

✅ **Reducers** in LangGraph control how state updates combine — critical for preventing data loss in parallel execution

✅ **Checkpointing** enables pause/resume, time travel, and fault recovery in production

✅ **Session persistence** (SQLite, PostgreSQL) ensures state survives process restarts

✅ **Concurrent state** requires careful design — use reducers, atomic updates, or idempotent operations

✅ Google ADK's `state_delta` provides atomic commits but allows dirty reads within a turn

**Next:** [Reasoning & Planning](../03-reasoning-planning/00-reasoning-planning.md)

---

## Further reading

- [LangGraph: State Management](https://docs.langchain.com/oss/python/langgraph/graph-api) — State schemas, reducers, and checkpointers
- [LangGraph: Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — Checkpointing and time travel
- [OpenAI Agents SDK: Sessions](https://openai.github.io/openai-agents-python/running_agents/#sessions) — Session-based persistence
- [Google ADK: Sessions & State](https://google.github.io/adk-docs/sessions/) — ADK session management

[Back to Agent Architecture Overview](./00-agent-architecture.md)

<!--
Sources Consulted:
- LangGraph Graph API (StateGraph, reducers, Send, Command): https://docs.langchain.com/oss/python/langgraph/graph-api
- OpenAI Agents SDK - Running Agents (Sessions, SQLiteSession): https://openai.github.io/openai-agents-python/running_agents/
- Google ADK - Event Loop (state_delta, dirty reads, atomic commits): https://google.github.io/adk-docs/runtime/event-loop/
- Google ADK - Core Concepts (Session, SessionService, State): https://google.github.io/adk-docs/get-started/about/
- LangGraph - Workflows and Agents: https://docs.langchain.com/oss/python/langgraph/workflows-agents
-->
