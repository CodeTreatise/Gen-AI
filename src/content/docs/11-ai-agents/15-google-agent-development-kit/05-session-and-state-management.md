---
title: "Session and State Management in Google ADK"
---

# Session and State Management in Google ADK

Conversational AI is only as good as its memory. Without sessions and state, every interaction starts from zero — the agent forgets the user's name, loses track of preferences, and cannot maintain a coherent multi-turn conversation. Google's Agent Development Kit (ADK) solves this with a layered session management system that gives us scoped state, automatic persistence, and template-based access to state values right inside agent instructions. In this lesson, we'll explore how ADK sessions work, how state is scoped across different lifetimes, and how to wire it all together for production-ready agents.

**Prerequisites:**

- Familiarity with Google ADK agent definitions ([Agent Definition Patterns](./02-agent-definition-patterns.md))
- Understanding of multi-agent hierarchies ([Multi-Agent Hierarchies](./04-multi-agent-hierarchies.md))
- Python async/await basics

---

## What Is a Session?

A **Session** in ADK represents a single conversation between a user and an agent system. Think of it like a browser session — it starts when a user begins a conversation and persists until it is explicitly closed or expires.

Every session contains:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier, auto-generated on creation |
| `app_name` | `str` | The application this session belongs to |
| `user_id` | `str` | The user who owns this session |
| `events` | `list[Event]` | Ordered message history (user turns, agent responses, tool calls) |
| `state` | `dict` | Key-value store for arbitrary data |

Sessions are **isolated by design**. A single user can have many concurrent sessions (e.g., one about travel planning, another about cooking), and state in one session does not leak into another — unless we explicitly use cross-session state prefixes.

```python
from google.adk.sessions import InMemorySessionService

service = InMemorySessionService()

# One user, two completely independent sessions
planning_session = await service.create_session(
    app_name="assistant", user_id="user_42"
)
cooking_session = await service.create_session(
    app_name="assistant", user_id="user_42"
)

print(planning_session.id)  # e.g., "sess_abc123"
print(cooking_session.id)   # e.g., "sess_def456"
```

Expected output:

```
sess_abc123
sess_def456
```

---

## SessionService Implementations

ADK abstracts storage behind the `SessionService` interface. We pick the implementation that matches our deployment stage.

| Service | Storage | Use Case |
|---------|---------|----------|
| `InMemorySessionService` | RAM | Development, testing |
| `DatabaseSessionService` | SQLite / PostgreSQL | Production, persistent |
| `VertexAiSessionService` | Google Cloud | Enterprise, managed |

Let's look at how we instantiate each one:

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# --- Development ---
dev_service = InMemorySessionService()

# --- Production with SQLite ---
prod_service_sqlite = DatabaseSessionService(
    db_url="sqlite+aiosqlite:///sessions.db"
)

# --- Production with PostgreSQL ---
prod_service_pg = DatabaseSessionService(
    db_url="postgresql+asyncpg://user:pass@localhost/mydb"
)
```

Expected output (no runtime output — these are instantiation calls):

```
# No output; services are ready to accept create/get/list/delete calls.
```

The `InMemorySessionService` is perfect for rapid prototyping because it requires zero configuration. Once we move to staging or production, we swap in `DatabaseSessionService` with a connection string — our code stays identical because both implement the same interface.

> **Tip:** For Google Cloud deployments, `VertexAiSessionService` handles scaling, replication, and backups automatically. We only need a project ID and location.

---

## Session CRUD Operations

Every `SessionService` exposes four async operations: **create**, **read**, **list**, and **delete**.

```python
from google.adk.sessions import InMemorySessionService

service = InMemorySessionService()

# CREATE — start a new session
session = await service.create_session(
    app_name="my_app", user_id="user_1"
)
print(f"Created: {session.id}")

# READ — retrieve an existing session by ID
fetched = await service.get_session(
    app_name="my_app", user_id="user_1", session_id=session.id
)
print(f"Fetched: {fetched.id}")

# LIST — get all sessions for a user
all_sessions = await service.list_sessions(
    app_name="my_app", user_id="user_1"
)
print(f"Total sessions: {len(all_sessions)}")

# DELETE — remove a session permanently
await service.delete_session(
    app_name="my_app", user_id="user_1", session_id=session.id
)
remaining = await service.list_sessions(
    app_name="my_app", user_id="user_1"
)
print(f"After delete: {len(remaining)}")
```

Expected output:

```
Created: sess_abc123
Fetched: sess_abc123
Total sessions: 1
After delete: 0
```

We can also **seed a session with initial state** at creation time. This is useful when we already know something about the user — their name, subscription tier, or locale:

```python
session = await service.create_session(
    app_name="my_app",
    user_id="user_1",
    state={"user:name": "Alice", "user:tier": "premium"},
)
print(session.state["user:name"])   # "Alice"
print(session.state["user:tier"])   # "premium"
```

Expected output:

```
Alice
premium
```

---

## State Prefixes — ADK's Scoping System

Not all state is created equal. A user's preferred language should persist across every conversation, but a temporary step counter should vanish after the current turn. ADK uses **key prefixes** to control scope and lifetime:

| Prefix | Scope | Lifetime | Example Key |
|--------|-------|----------|-------------|
| *(none)* | Session | Current session only | `state["counter"]` |
| `user:` | User | All sessions for this user | `state["user:language"]` |
| `app:` | Application | All users, all sessions | `state["app:version"]` |
| `temp:` | Invocation | Current turn only | `state["temp:step"]` |

```mermaid
graph TD
    A["app: state<br/>(global)"] --> B["user: state<br/>(per user)"]
    B --> C["session state<br/>(per conversation)"]
    C --> D["temp: state<br/>(per turn)"]

    style A fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style B fill:#7b68ee,stroke:#5a4cbf,color:#fff
    style C fill:#50c878,stroke:#3a9a5c,color:#fff
    style D fill:#f4a460,stroke:#c8834d,color:#fff
```

The scoping model follows a hierarchy: `app` → `user` → `session` → `temp`. Broader scopes survive longer and are visible across more contexts.

### Setting State from Tools

The most common way to write state is inside a tool function via `ToolContext`:

```python
from google.adk.tools import ToolContext

def set_language(language: str, tool_context: ToolContext) -> str:
    """Set the user's preferred language."""
    # Persists across ALL sessions for this user
    tool_context.state["user:language"] = language

    # Visible only during THIS turn
    tool_context.state["temp:action"] = "language_set"

    # Session-scoped — survives across turns but not across sessions
    tool_context.state["interaction_count"] = (
        tool_context.state.get("interaction_count", 0) + 1
    )

    return f"Language set to {language}"
```

Expected output (when the tool is invoked by the agent):

```
Language set to Spanish
```

After this tool runs:

- `state["user:language"]` → `"Spanish"` — available in every future session for this user.
- `state["temp:action"]` → `"language_set"` — gone after the current turn ends.
- `state["interaction_count"]` → `1` — available for the rest of this session.

---

## State Templating in Instructions

One of ADK's most powerful features is **automatic state interpolation** in agent instructions. We write `{key}` placeholders, and ADK substitutes them with the corresponding `session.state` values at runtime.

```python
from google.adk.agents import Agent

agent = Agent(
    name="personalized_agent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful assistant.
    The user's name is {user:name}.
    Their preferred language is {user:language}.
    Current session topic: {topic}.""",
)
```

Expected output (the instruction the LLM actually sees at runtime):

```
You are a helpful assistant.
The user's name is Alice.
Their preferred language is Spanish.
Current session topic: travel planning.
```

This eliminates boilerplate code for injecting context. We don't need to manually format strings or rebuild prompts — ADK handles it each turn, always using the latest state values.

> **Important:** If a referenced key doesn't exist in state, ADK leaves the placeholder as-is (e.g., `{topic}` renders literally as `{topic}`). Always ensure required state keys are initialized.

---

## EventActions and state_delta

Every event in a session can carry **actions** that modify state. When the runner processes an event, it inspects `event.actions.state_delta` — a dictionary of state changes that occurred during that event.

```python
from google.adk.events import EventActions

# Internally, ADK constructs something like:
actions = EventActions(
    state_delta={
        "user:language": "Spanish",
        "temp:action": "language_set",
        "interaction_count": 1,
    }
)

# The runner applies this delta to session.state automatically.
print(actions.state_delta)
```

Expected output:

```python
{'user:language': 'Spanish', 'temp:action': 'language_set', 'interaction_count': 1}
```

We can also modify state in **callbacks** via `CallbackContext`, which wraps the same state interface:

```python
def before_agent_callback(callback_context) -> None:
    """Runs before the agent processes each turn."""
    turn = callback_context.state.get("turn_count", 0)
    callback_context.state["turn_count"] = turn + 1
```

Expected output (no direct output — state is mutated silently):

```
# turn_count incremented from 0 → 1, 1 → 2, etc., on each turn.
```

---

## output_key for Agent-to-Agent Data Flow

In multi-agent systems, we often need one agent's output to flow into another agent's context. The `output_key` parameter saves an agent's final text response directly into session state.

```python
from google.adk.agents import Agent

classifier = Agent(
    name="classifier",
    model="gemini-2.0-flash",
    instruction=(
        "Classify the user's intent. "
        "Respond with exactly one word: billing, technical, or general."
    ),
    output_key="intent",  # Agent's response → state["intent"]
)

router = Agent(
    name="router",
    model="gemini-2.0-flash",
    instruction="""You are a routing agent.
    The classified intent is: {intent}.
    Route the user to the appropriate department.""",
)
```

Expected output (at runtime, after the classifier runs):

```
# state["intent"] = "billing"
# The router's instruction becomes:
# "...The classified intent is: billing. Route the user..."
```

This pattern is the glue of multi-agent orchestration. The classifier writes to `state["intent"]`, and the router reads it via `{intent}` — no manual wiring required.

---

## Putting It All Together

Let's build a complete example — a customer support agent that remembers user preferences across sessions and routes by intent:

```python
import asyncio
from google.adk.agents import Agent, SequentialAgent
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.tools import ToolContext
from google.genai.types import Content, Part


def set_preference(key: str, value: str, tool_context: ToolContext) -> str:
    """Save a user-level preference."""
    tool_context.state[f"user:{key}"] = value
    return f"Preference '{key}' saved as '{value}'."


classifier = Agent(
    name="classifier",
    model="gemini-2.0-flash",
    instruction="Classify intent as: billing, technical, or general. One word only.",
    output_key="intent",
)

support_agent = Agent(
    name="support",
    model="gemini-2.0-flash",
    instruction="""You are a support agent.
    User: {user:name} (Tier: {user:tier})
    Intent: {intent}
    Respond helpfully.""",
    tools=[set_preference],
)

pipeline = SequentialAgent(
    name="support_pipeline",
    sub_agents=[classifier, support_agent],
)

session_service = DatabaseSessionService(
    db_url="sqlite+aiosqlite:///support.db"
)

runner = Runner(
    agent=pipeline,
    app_name="support_app",
    session_service=session_service,
)


async def main():
    session = await session_service.create_session(
        app_name="support_app",
        user_id="user_42",
        state={"user:name": "Alice", "user:tier": "premium"},
    )

    user_message = Content(
        parts=[Part(text="I was overcharged on my last invoice.")]
    )

    async for event in runner.run_async(
        session_id=session.id, user_id="user_42", new_message=user_message
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)


asyncio.run(main())
```

Expected output:

```
I understand your concern about the overcharge, Alice. As a premium member,
you're entitled to priority billing support. Let me look into your last
invoice right away...
```

---

## Best Practices

| Practice | Why It Matters |
|----------|---------------|
| Use `user:` prefix for preferences and profile data | Survives across sessions — users don't repeat themselves |
| Use `temp:` for intermediate computation | Keeps state clean; no stale data on the next turn |
| Seed sessions with `state={}` at creation | Ensures templates never render with missing placeholders |
| Use `output_key` instead of manual state writes for agent outputs | Cleaner, declarative, and less error-prone |
| Switch to `DatabaseSessionService` before staging | Avoid data loss from server restarts |
| Keep state keys descriptive and namespaced | Prevents collisions in multi-agent systems (e.g., `user:lang` vs `session:lang`) |

---

## Common Pitfalls

| ❌ Don't | ✅ Do |
|----------|-------|
| Store large blobs (images, PDFs) in state | Store a URL or reference; fetch on demand |
| Forget to initialize state keys referenced in templates | Pass `state={...}` at session creation or use defaults |
| Use session-scoped state for data that should persist across conversations | Use the `user:` prefix for cross-session data |
| Assume `temp:` state survives across turns | Treat `temp:` as single-turn scratch space only |
| Hard-code context into instructions | Use `{key}` templates so instructions adapt dynamically |
| Use `InMemorySessionService` in production | Use `DatabaseSessionService` or `VertexAiSessionService` for persistence |

---

## Hands-on Exercise

**Build a language-aware greeter agent** that:

1. Creates a `DatabaseSessionService` backed by SQLite.
2. Seeds a session with `user:name` and `user:language`.
3. Defines a tool `update_language` that changes `user:language` via `ToolContext`.
4. Uses state templating in the agent's instruction to greet the user in their preferred language.
5. Runs two turns: the first asks for a greeting, the second changes the language and asks again.

**Acceptance criteria:**

- The first greeting uses the seeded language.
- After calling `update_language`, the second greeting uses the new language.
- Session state persists in a `.db` file.

<details>
<summary>💡 Hints</summary>

- Use `sqlite+aiosqlite:///greeter.db` as the database URL.
- Remember that `ToolContext` is injected automatically — just add it as a parameter.
- Use `{user:name}` and `{user:language}` in your instruction string.
- Run two separate `runner.run_async()` calls with different messages in the same session.

</details>

<details>
<summary>✅ Solution</summary>

```python
import asyncio
from google.adk.agents import Agent
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.tools import ToolContext
from google.genai.types import Content, Part


def update_language(new_language: str, tool_context: ToolContext) -> str:
    """Update the user's preferred language."""
    tool_context.state["user:language"] = new_language
    return f"Language updated to {new_language}."


agent = Agent(
    name="greeter",
    model="gemini-2.0-flash",
    instruction="""Greet the user by name in their preferred language.
    User's name: {user:name}
    Preferred language: {user:language}""",
    tools=[update_language],
)

session_service = DatabaseSessionService(
    db_url="sqlite+aiosqlite:///greeter.db"
)

runner = Runner(
    agent=agent,
    app_name="greeter_app",
    session_service=session_service,
)


async def main():
    session = await session_service.create_session(
        app_name="greeter_app",
        user_id="user_1",
        state={"user:name": "Carlos", "user:language": "Spanish"},
    )

    # Turn 1 — greet in Spanish
    msg1 = Content(parts=[Part(text="Hello! Please greet me.")])
    async for event in runner.run_async(
        session_id=session.id, user_id="user_1", new_message=msg1
    ):
        if event.content and event.content.parts:
            print("Turn 1:", event.content.parts[0].text)

    # Turn 2 — switch to French and greet again
    msg2 = Content(
        parts=[Part(text="Change my language to French, then greet me again.")]
    )
    async for event in runner.run_async(
        session_id=session.id, user_id="user_1", new_message=msg2
    ):
        if event.content and event.content.parts:
            print("Turn 2:", event.content.parts[0].text)


asyncio.run(main())
```

Expected output:

```
Turn 1: ¡Hola, Carlos! ¿Cómo puedo ayudarte hoy?
Turn 2: Bonjour, Carlos! Comment puis-je vous aider aujourd'hui?
```

</details>

---

## Summary

✅ A **Session** encapsulates one conversation — it holds an ID, user ID, event history, and a state dictionary.

✅ **SessionService** abstracts storage: use `InMemorySessionService` for dev, `DatabaseSessionService` for production, and `VertexAiSessionService` for managed cloud deployments.

✅ **State prefixes** (`user:`, `app:`, `temp:`, or none) control scope and lifetime — from a single turn to the entire application.

✅ **State templating** with `{key}` in instructions lets ADK inject dynamic context without manual string formatting.

✅ **`output_key`** saves an agent's response to state, enabling seamless data flow between agents in a pipeline.

✅ **EventActions** and `state_delta` give us fine-grained visibility into what changed and when.

✅ Always **seed initial state** at session creation and use **descriptive, namespaced keys** to avoid collisions.

---

**Next:** [Memory Systems](./06-memory-systems.md)

**Previous:** [Multi-Agent Hierarchies](./04-multi-agent-hierarchies.md)

[Back to Google ADK Overview](./00-google-agent-development-kit.md)

---

## Further Reading

- [Google ADK Sessions Documentation](https://google.github.io/adk-docs/sessions/)
- [Google ADK State Management](https://google.github.io/adk-docs/sessions/state/)
- [ADK Session Service API Reference](https://google.github.io/adk-docs/api-reference/sessions/)
- [Google ADK GitHub — Session Examples](https://github.com/google/adk-python/tree/main/examples)
- [Building Stateful AI Agents (Google Cloud Blog)](https://cloud.google.com/blog/products/ai-machine-learning)

<!-- Sources:
- Google ADK Documentation: https://google.github.io/adk-docs/
- Google ADK Sessions & State: https://google.github.io/adk-docs/sessions/
- Google ADK GitHub Repository: https://github.com/google/adk-python
- ADK API Reference — SessionService: https://google.github.io/adk-docs/api-reference/
- Google Cloud Vertex AI Agent Builder: https://cloud.google.com/vertex-ai/docs/agents
-->
