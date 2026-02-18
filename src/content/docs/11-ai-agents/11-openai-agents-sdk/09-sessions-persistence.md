---
title: "Sessions and Persistence"
---

# Sessions and Persistence

## Introduction

Agents are stateless by default — each `Runner.run()` call starts fresh. For multi-turn conversations, we need **sessions** to persist conversation history between calls. The OpenAI Agents SDK provides built-in session types and an interface for custom implementations.

Sessions handle the mechanics of saving and loading conversation history, so our agent code stays focused on behavior rather than state management.

### What we'll cover

- What sessions solve and how they work
- Built-in session types (SQLite, SQLAlchemy, OpenAI-managed)
- Multi-turn conversation patterns
- Session lifecycle (get, add, pop, clear)
- Custom session implementations
- Advanced patterns (branching, encryption)

### Prerequisites

- [Agent Class Fundamentals](./01-agent-class-fundamentals.md)
- [Runner Execution Model](./02-runner-execution-model.md)

---

## How sessions work

Without sessions, every `Runner.run()` call is independent:

```mermaid
flowchart LR
    A["Run 1: 'My name is Alice'"] --> B[Agent]
    C["Run 2: 'What's my name?'"] --> D[Agent]
    B --> E["'Nice to meet you, Alice!'"]
    D --> F["'I don't know your name.'"]
    
    style F fill:#ffcccc
```

With sessions, history persists between runs:

```mermaid
flowchart LR
    A["Run 1: 'My name is Alice'"] --> B[Agent + Session]
    C["Run 2: 'What's my name?'"] --> B
    B --> D["Session Storage"]
    B --> E["'Nice to meet you, Alice!'"]
    B --> F["'Your name is Alice!'"]
    
    style F fill:#ccffcc
```

| Without sessions | With sessions |
|-----------------|---------------|
| Each run is independent | History carries across runs |
| Agent forgets everything | Agent remembers context |
| Manual history management | Automatic save/load |

---

## SQLiteSession (recommended start)

The simplest way to add persistence. Stores conversations in a local SQLite database:

```python
from agents import Agent, Runner
from agents.extensions.sessions import SQLiteSession

agent = Agent(
    name="Memory Bot",
    instructions="You are a helpful assistant that remembers previous conversations.",
)

# Create a session — history is stored in memory (or specify a file)
session = SQLiteSession("conversation_001")

# First interaction
result = await Runner.run(
    agent,
    "My favorite color is blue.",
    session=session,
)
print(result.final_output)

# Second interaction — the agent remembers!
result = await Runner.run(
    agent,
    "What's my favorite color?",
    session=session,
)
print(result.final_output)
```

**Output:**
```
Got it! I'll remember that your favorite color is blue.
Your favorite color is blue!
```

### File-based persistence

```python
# Persist to a file (survives process restarts)
session = SQLiteSession("user_123", db_path="conversations.db")

# Different users get different sessions
alice_session = SQLiteSession("alice", db_path="conversations.db")
bob_session = SQLiteSession("bob", db_path="conversations.db")
```

### Session lifecycle methods

```python
session = SQLiteSession("my_session")

# get_items() — Retrieve stored history
items = await session.get_items()
print(f"History has {len(items)} items")

# add_items() — Manually add items
await session.add_items([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
])

# pop_item() — Remove the last item
removed = await session.pop_item()

# clear_session() — Remove all history
await session.clear_session()
```

---

## Other built-in session types

### SQLAlchemySession

For production databases (PostgreSQL, MySQL, etc.):

```python
from agents.extensions.sessions import SQLAlchemySession

# Connect to PostgreSQL
session = await SQLAlchemySession.from_url(
    session_id="user_456",
    url="postgresql://user:pass@localhost/mydb",
)

result = await Runner.run(agent, "Hello!", session=session)
```

### OpenAI-managed sessions

Use OpenAI's server-side session storage:

```python
from agents.extensions.sessions import OpenAIConversationsSession

# Conversations API manages the history
session = OpenAIConversationsSession(session_id="conv_789")
result = await Runner.run(agent, "Hello!", session=session)
```

### OpenAI compaction session

Automatically compacts long conversation histories to stay within context limits:

```python
from agents.extensions.sessions import OpenAIResponsesCompactionSession

session = OpenAIResponsesCompactionSession(session_id="long_conv")

# After many turns, the session automatically compacts older messages
# to fit within the model's context window
for i in range(50):
    result = await Runner.run(agent, f"Message {i}", session=session)
```

> **🤖 AI Context:** Compaction is critical for long-running agents. Without it, conversations eventually exceed the context window, causing errors or truncated history.

---

## Session type comparison

| Session type | Storage | Best for | Persistence |
|-------------|---------|----------|-------------|
| `SQLiteSession` | Local SQLite file | Development, simple apps | File-based |
| `SQLAlchemySession` | Any SQL database | Production apps | Database |
| `OpenAIConversationsSession` | OpenAI servers | Using Conversations API | Cloud |
| `OpenAIResponsesCompactionSession` | OpenAI servers | Long conversations | Cloud + auto-compaction |

---

## Multi-turn conversation patterns

### Pattern 1: Simple loop

```python
from agents import Agent, Runner
from agents.extensions.sessions import SQLiteSession

agent = Agent(name="Chat Bot", instructions="Be helpful and conversational.")
session = SQLiteSession("chat_session", db_path="chat.db")

async def chat_loop():
    print("Chat started. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("quit", "exit"):
            break
        
        result = await Runner.run(agent, user_input, session=session)
        print(f"Bot: {result.final_output}")

import asyncio
asyncio.run(chat_loop())
```

### Pattern 2: Session per user

```python
from agents import Agent, Runner
from agents.extensions.sessions import SQLiteSession

agent = Agent(name="Support Agent", instructions="Help customers.")

def get_session(user_id: str) -> SQLiteSession:
    """One session per user, all in the same database."""
    return SQLiteSession(user_id, db_path="support.db")

async def handle_message(user_id: str, message: str) -> str:
    session = get_session(user_id)
    result = await Runner.run(agent, message, session=session)
    return result.final_output

# Each user has independent conversation history
await handle_message("user_alice", "I need help with billing")
await handle_message("user_bob", "How do I reset my password?")
await handle_message("user_alice", "Can you check my last invoice?")  # Remembers billing context
```

### Pattern 3: Manual history (no sessions)

For maximum control, manage history manually using `to_input_list()`:

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Be helpful.")
history = []

async def chat(message: str) -> str:
    # Include previous history as input
    input_items = history + [{"role": "user", "content": message}]
    
    result = await Runner.run(agent, input_items)
    
    # Save the full turn to history
    history.extend(result.to_input_list())
    
    return result.final_output
```

---

## Custom session implementation

Implement `SessionABC` to create custom storage backends:

```python
from agents.extensions.sessions import SessionABC

class RedisSession(SessionABC):
    """Store conversation history in Redis."""
    
    def __init__(self, session_id: str, redis_client):
        self.session_id = session_id
        self.redis = redis_client
        self._key = f"session:{session_id}"
    
    async def get_items(self) -> list:
        """Load history from Redis."""
        import json
        data = await self.redis.get(self._key)
        if data is None:
            return []
        return json.loads(data)
    
    async def add_items(self, items: list) -> None:
        """Append items to history in Redis."""
        import json
        existing = await self.get_items()
        existing.extend(items)
        await self.redis.set(self._key, json.dumps(existing))
    
    async def pop_item(self):
        """Remove and return the last item."""
        import json
        items = await self.get_items()
        if not items:
            return None
        removed = items.pop()
        await self.redis.set(self._key, json.dumps(items))
        return removed
    
    async def clear_session(self) -> None:
        """Delete all history."""
        await self.redis.delete(self._key)
```

### SessionABC interface

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_items()` | Load stored conversation history | `list` of input items |
| `add_items(items)` | Append new items to history | `None` |
| `pop_item()` | Remove and return the last item | The removed item or `None` |
| `clear_session()` | Delete all stored history | `None` |

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Use meaningful session IDs | `f"user_{user_id}_conv_{conv_id}"` makes debugging easier |
| Use file-based SQLite for development | Survives restarts; easy to inspect |
| Use SQLAlchemy for production | Proper database with backups and scaling |
| Use compaction for long conversations | Prevents context window overflow |
| Clear sessions explicitly | Don't rely on garbage collection for cleanup |
| Separate sessions per conversation | One user can have multiple independent conversations |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Reusing the same session ID for all users | Use unique IDs: `f"user_{user_id}"` |
| Forgetting to await session methods | All session methods are async — use `await` |
| Not handling context window limits | Use `OpenAIResponsesCompactionSession` for long conversations |
| Storing sessions in memory only | Pass `db_path` to `SQLiteSession` for persistence across restarts |
| Mixing session types in one app | Pick one session backend and use it consistently |

---

## Hands-on exercise

### Your task

Build a **note-taking agent** with persistent sessions that remembers notes across restarts.

### Requirements

1. Create an agent with tools: `save_note(title, content)`, `list_notes()`, `search_notes(query)`
2. Use `SQLiteSession` with a file-based database (`notes.db`)
3. The agent should remember conversation context between runs
4. Implement a simple chat loop that persists between program restarts
5. Use the session ID to separate different users' conversations

### Expected result

After restarting the program, the agent remembers previously saved notes and conversation context.

<details>
<summary>💡 Hints (click to expand)</summary>

- Store notes in context (a list in a dataclass), and use session for conversation history
- `SQLiteSession("user_alice", db_path="notes.db")` persists to disk
- The chat loop runs with `while True` and breaks on "quit"
- Use `result.to_input_list()` if managing history manually

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import asyncio
from dataclasses import dataclass, field
from agents import Agent, Runner, RunContextWrapper, function_tool
from agents.extensions.sessions import SQLiteSession

@dataclass
class NotesContext:
    user_id: str
    notes: list[dict] = field(default_factory=list)

@function_tool
async def save_note(ctx: RunContextWrapper[NotesContext], title: str, content: str) -> str:
    """Save a new note.
    
    Args:
        title: Title of the note.
        content: Content of the note.
    """
    note = {"title": title, "content": content}
    ctx.context.notes.append(note)
    return f"Saved note: '{title}'"

@function_tool
async def list_notes(ctx: RunContextWrapper[NotesContext]) -> str:
    """List all saved notes."""
    if not ctx.context.notes:
        return "No notes saved yet."
    lines = [f"- {n['title']}: {n['content']}" for n in ctx.context.notes]
    return "\n".join(lines)

@function_tool
async def search_notes(ctx: RunContextWrapper[NotesContext], query: str) -> str:
    """Search notes by keyword.
    
    Args:
        query: The keyword to search for.
    """
    matches = [n for n in ctx.context.notes 
               if query.lower() in n["title"].lower() or query.lower() in n["content"].lower()]
    if not matches:
        return f"No notes matching '{query}'."
    lines = [f"- {n['title']}: {n['content']}" for n in matches]
    return "\n".join(lines)

agent = Agent[NotesContext](
    name="Notes Agent",
    instructions="Help users manage their notes. Use the tools to save, list, and search notes.",
    tools=[save_note, list_notes, search_notes],
)

async def main():
    user_id = "alice"
    ctx = NotesContext(user_id=user_id)
    session = SQLiteSession(f"notes_{user_id}", db_path="notes.db")
    
    print("📝 Notes Agent (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("quit", "exit"):
            break
        
        result = await Runner.run(
            agent, user_input, context=ctx, session=session
        )
        print(f"Agent: {result.final_output}")

asyncio.run(main())
```

</details>

### Bonus challenges

- [ ] Add `delete_note(title)` and `update_note(title, new_content)` tools
- [ ] Implement a custom `SessionABC` that stores history in a JSON file
- [ ] Use `OpenAIResponsesCompactionSession` and test with 50+ turns

---

## Summary

✅ Sessions persist conversation history between `Runner.run()` calls automatically

✅ `SQLiteSession` is the simplest option — pass `db_path` for file-based persistence

✅ `SQLAlchemySession` supports production databases (PostgreSQL, MySQL)

✅ `OpenAIResponsesCompactionSession` auto-compacts long conversations to fit context windows

✅ Custom sessions implement `SessionABC` with four methods: `get_items`, `add_items`, `pop_item`, `clear_session`

**Next:** [Development Utilities](./10-development-utilities.md)

---

## Further reading

- [Sessions docs](https://openai.github.io/openai-agents-python/sessions/) — Session types and configuration
- [Multi-turn docs](https://openai.github.io/openai-agents-python/running_agents/#multi-turn-conversations) — Conversation patterns
- [SessionABC reference](https://openai.github.io/openai-agents-python/ref/sessions/) — Custom session interface

---

*[Back to OpenAI Agents SDK Overview](./00-openai-agents-sdk.md)*

<!-- 
Sources Consulted:
- Sessions documentation: https://openai.github.io/openai-agents-python/sessions/
- Running agents multi-turn: https://openai.github.io/openai-agents-python/running_agents/
- Session API reference: https://openai.github.io/openai-agents-python/ref/sessions/
-->
