---
title: "Memory Integration: Modern Patterns and Production Systems"
---

# Memory Integration: Modern Patterns and Production Systems

## Introduction

The previous lessons covered individual memory types. Now we bring everything together with **modern integration patterns**—specifically `RunnableWithMessageHistory`, which is LangChain's recommended approach for managing conversation memory in production applications.

This lesson focuses on practical implementation: how to wire memory into LCEL chains, manage multiple sessions, persist across restarts, and build robust production-ready memory systems.

### What We'll Cover

- RunnableWithMessageHistory deep dive
- LCEL integration patterns
- Multi-session and multi-tenant architectures
- Persistence backends (Redis, PostgreSQL, MongoDB)
- Message history implementations
- Production patterns and error handling
- Migration from legacy memory classes

### Prerequisites

- Understanding of all memory types from previous lessons
- Familiarity with LCEL (LangChain Expression Language)
- Basic async Python knowledge

---

## RunnableWithMessageHistory

`RunnableWithMessageHistory` is the modern, recommended way to add memory to any LCEL chain. It wraps a runnable and automatically:

1. **Loads** message history before each call
2. **Injects** history into the chain
3. **Saves** new messages after each call

```mermaid
sequenceDiagram
    participant User
    participant RunnableWithHistory
    participant HistoryStore
    participant Chain
    participant LLM

    User->>RunnableWithHistory: invoke(input, config)
    RunnableWithHistory->>HistoryStore: get_session_history(session_id)
    HistoryStore-->>RunnableWithHistory: ChatMessageHistory
    RunnableWithHistory->>Chain: Add history + invoke
    Chain->>LLM: Prompt with history
    LLM-->>Chain: Response
    Chain-->>RunnableWithHistory: Result
    RunnableWithHistory->>HistoryStore: Save new messages
    RunnableWithHistory-->>User: Response
```

### Basic Usage

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 1. Create the chain
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# 2. Create session storage
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Factory function to get/create session history."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. Wrap with message history
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 4. Use with session configuration
config = {"configurable": {"session_id": "user-123"}}

response = chain_with_memory.invoke(
    {"input": "Hi, I'm Alex!"},
    config=config
)
print(response.content)

response = chain_with_memory.invoke(
    {"input": "What's my name?"},
    config=config
)
print(response.content)
```

**Output:**
```
Hello Alex! Nice to meet you. How can I help you today?
Your name is Alex! You just introduced yourself. 😊
```

### Configuration Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `runnable` | Runnable | Yes | The chain to wrap |
| `get_session_history` | Callable | Yes | Factory function returning history |
| `input_messages_key` | str | No | Key for user input in chain input |
| `output_messages_key` | str | No | Key for AI output in chain output |
| `history_messages_key` | str | No | Key for history in prompt |
| `history_factory_config` | list | No | Custom configurable fields |

---

## Input/Output Key Configuration

Understanding the key configuration is crucial for correct wiring:

### Scenario 1: String Input Chain

```python
# Chain takes string input directly
chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    # No input_messages_key needed - detects single string input
    history_messages_key="history"
)

# Invoke with string
response = chain_with_memory.invoke(
    "Hello!",  # String, not dict
    config={"configurable": {"session_id": "user-1"}}
)
```

### Scenario 2: Dict Input Chain

```python
# Chain takes dict input
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for {topic}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",      # Which key has user input
    history_messages_key="chat_history"  # Which placeholder gets history
)

response = chain_with_memory.invoke(
    {"question": "What is Python?", "topic": "programming"},
    config={"configurable": {"session_id": "user-1"}}
)
```

### Scenario 3: Messages Input/Output

```python
from langchain_core.messages import HumanMessage

# Chain that works with message lists
chain_with_memory = RunnableWithMessageHistory(
    llm,  # Just the LLM, no prompt
    get_session_history,
    # For chains that take/return Message lists, no keys needed
)

response = chain_with_memory.invoke(
    [HumanMessage(content="Hello!")],
    config={"configurable": {"session_id": "user-1"}}
)
```

---

## Multi-Session Architecture

Production apps need to handle multiple users with isolated sessions:

```python
"""Multi-session chat application."""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class Session:
    """Session metadata."""
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    history: InMemoryChatMessageHistory


class MultiSessionManager:
    """Manages multiple user sessions."""
    
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> session_ids
    
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            history=InMemoryChatMessageHistory()
        )
        
        self.sessions[session_id] = session
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        return session_id
    
    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get history for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.last_active = datetime.now()
        return session.history
    
    def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all sessions for a user."""
        session_ids = self.user_sessions.get(user_id, [])
        return [self.sessions[sid] for sid in session_ids]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            age = (now - session.last_active).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            session = self.sessions.pop(session_id)
            self.user_sessions[session.user_id].remove(session_id)


# Usage
session_manager = MultiSessionManager()

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return session_manager.get_history(session_id)

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain_with_memory = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


# API-like usage
def chat(user_id: str, message: str, session_id: str | None = None) -> tuple[str, str]:
    """Chat with optional session continuation."""
    if session_id is None:
        session_id = session_manager.create_session(user_id)
    
    config = {"configurable": {"session_id": session_id}}
    response = chain_with_memory.invoke({"input": message}, config=config)
    
    return session_id, response.content


# Test multi-user
session_1, resp = chat("alice", "I'm building a Python app")
print(f"Alice session {session_1[:8]}: {resp[:50]}...")

session_2, resp = chat("bob", "I'm working on JavaScript")
print(f"Bob session {session_2[:8]}: {resp[:50]}...")

# Alice continues
_, resp = chat("alice", "What am I building?", session_1)
print(f"Alice followup: {resp[:50]}...")  # Remembers Python app
```

---

## Multi-Key Session Configuration

Sometimes you need to identify sessions by multiple fields (user_id + conversation_id):

```python
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store keyed by (user_id, conversation_id)
store: dict[tuple[str, str], InMemoryChatMessageHistory] = {}


def get_history(user_id: str, conversation_id: str) -> InMemoryChatMessageHistory:
    """Get history for user + conversation combination."""
    key = (user_id, conversation_id)
    if key not in store:
        store[key] = InMemoryChatMessageHistory()
    return store[key]


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation",
            default="",
            is_shared=True,
        ),
    ],
)


# Usage with multi-key config
config = {
    "configurable": {
        "user_id": "alice",
        "conversation_id": "project-1"
    }
}

response = chain_with_memory.invoke({"input": "Hello!"}, config=config)
```

---

## Persistence Backends

### Redis Chat History

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        key_prefix="chat:",
        ttl=3600  # 1 hour expiration
    )


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="input",
    history_messages_key="history"
)

# History persists across restarts and scales horizontally
```

### PostgreSQL Chat History

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory


def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string="postgresql://user:pass@localhost/chatdb",
        table_name="chat_history"
    )


# Auto-creates table if not exists
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_postgres_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### MongoDB Chat History

```python
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


def get_mongo_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost:27017",
        database_name="chat_app",
        collection_name="histories"
    )


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_mongo_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### Backend Comparison

| Backend | Persistence | Scale | TTL Support | Best For |
|---------|-------------|-------|-------------|----------|
| **InMemory** | ❌ | Single process | ❌ | Development |
| **Redis** | ✅ | Horizontal | ✅ | Production (stateless) |
| **PostgreSQL** | ✅ | Vertical | Via triggers | ACID requirements |
| **MongoDB** | ✅ | Horizontal | ✅ | Document-oriented apps |
| **SQLite** | ✅ | Single file | ❌ | Edge/embedded |

---

## Custom Message History Implementation

Implement your own history backend:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from typing import Sequence
import json
from pathlib import Path


class FileChatMessageHistory(BaseChatMessageHistory):
    """File-based chat message history."""
    
    def __init__(self, session_id: str, storage_dir: str = "./chat_history"):
        self.session_id = session_id
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / f"{session_id}.json"
        self._messages: list[BaseMessage] = self._load()
    
    def _load(self) -> list[BaseMessage]:
        """Load messages from file."""
        if self.file_path.exists():
            with open(self.file_path) as f:
                data = json.load(f)
                return messages_from_dict(data)
        return []
    
    def _save(self) -> None:
        """Save messages to file."""
        with open(self.file_path, 'w') as f:
            json.dump(messages_to_dict(self._messages), f, indent=2)
    
    @property
    def messages(self) -> list[BaseMessage]:
        """Get all messages."""
        return self._messages
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages and persist."""
        self._messages.extend(messages)
        self._save()
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()


# Usage
def get_file_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(session_id)

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_file_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

---

## Async Support

`RunnableWithMessageHistory` supports async operations:

```python
import asyncio
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

llm = ChatOpenAI(model="gpt-4o-mini")

chain_with_memory = RunnableWithMessageHistory(
    llm,
    get_history,
)


async def chat_async(session_id: str, message: str) -> str:
    config = {"configurable": {"session_id": session_id}}
    response = await chain_with_memory.ainvoke(message, config=config)
    return response.content


async def main():
    # Parallel conversations
    results = await asyncio.gather(
        chat_async("user-1", "I like Python"),
        chat_async("user-2", "I prefer JavaScript"),
        chat_async("user-3", "Rust is the best"),
    )
    
    for i, result in enumerate(results, 1):
        print(f"User {i}: {result[:50]}...")


asyncio.run(main())
```

---

## Error Handling and Resilience

### Graceful Degradation

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
import logging

logger = logging.getLogger(__name__)


class ResilientHistoryManager:
    """History manager with fallback to in-memory."""
    
    def __init__(self, primary_backend: str = "redis"):
        self.primary_backend = primary_backend
        self.fallback_store: dict[str, InMemoryChatMessageHistory] = {}
    
    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        try:
            if self.primary_backend == "redis":
                return self._get_redis_history(session_id)
        except Exception as e:
            logger.warning(f"Primary backend failed: {e}. Using fallback.")
        
        # Fallback to in-memory
        if session_id not in self.fallback_store:
            self.fallback_store[session_id] = InMemoryChatMessageHistory()
        return self.fallback_store[session_id]
    
    def _get_redis_history(self, session_id: str):
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        return RedisChatMessageHistory(
            session_id=session_id,
            url="redis://localhost:6379"
        )
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential


class RetryingHistory(InMemoryChatMessageHistory):
    """History with retry logic for saves."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def add_messages(self, messages):
        # Attempt to save with retries
        super().add_messages(messages)
        self._persist_to_backend()
    
    def _persist_to_backend(self):
        # Your persistence logic here
        pass
```

---

## Migration from Legacy Memory

### Before (Legacy)

```python
# ⚠️ Deprecated - will be removed in LangChain v1.0
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
chain.invoke({"input": "Hello"})
```

### After (Modern)

```python
# ✅ Recommended
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "user-123"}}
chain_with_memory.invoke({"input": "Hello"}, config=config)
```

### Migration Mapping

| Legacy | Modern Equivalent |
|--------|------------------|
| `ConversationBufferMemory` | `InMemoryChatMessageHistory` |
| `ConversationBufferWindowMemory` | Custom history with trimming |
| `ConversationSummaryMemory` | Custom history with summarization |
| `memory.save_context()` | Automatic via `RunnableWithMessageHistory` |
| `memory.load_memory_variables()` | Automatic via `RunnableWithMessageHistory` |

---

## Production Patterns

### Pattern 1: FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

app = FastAPI()

# Session store (use Redis in production)
sessions: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id]

llm = ChatOpenAI(model="gpt-4o-mini")

chain_with_memory = RunnableWithMessageHistory(
    llm,
    get_history,
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}
    
    try:
        response = await chain_with_memory.ainvoke(
            request.message,
            config=config
        )
        return ChatResponse(
            session_id=request.session_id,
            response=response.content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id].clear()
        return {"status": "cleared"}
    raise HTTPException(status_code=404, detail="Session not found")
```

### Pattern 2: Streaming with Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Chain with streaming
chain = prompt | llm | StrOutputParser()

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


async def stream_chat(session_id: str, message: str):
    config = {"configurable": {"session_id": session_id}}
    
    async for chunk in chain_with_memory.astream(
        {"input": message},
        config=config
    ):
        yield chunk  # Stream to client
```

### Pattern 3: Message Trimming Middleware

```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Trim messages to fit token budget
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=llm,
    include_system=True,
)

# Apply trimming before LLM call
chain = trimmer | prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| **Use configurable session IDs** | UUID prevents guessing |
| **Set TTL on persistent stores** | Prevent unlimited growth |
| **Implement graceful degradation** | App works even if Redis fails |
| **Use async in web apps** | Better concurrency |
| **Monitor session counts** | Detect memory leaks |
| **Trim messages before LLM** | Prevent context overflow |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No session ID validation | Validate format, check existence |
| Session ID in URL path | Use headers/cookies for security |
| No TTL on Redis history | Set reasonable expiration |
| Blocking sync in async app | Use `ainvoke`, `astream` |
| History grows forever | Implement trimming strategy |
| Shared session across users | Validate user owns session |

---

## Hands-on Exercise

### Your Task

Build a complete chat application with:

1. Session management (create, list, delete)
2. Redis persistence
3. Message count limiting (max 50 messages)
4. Session expiration (1 hour TTL)
5. Basic analytics (message count, session age)

### Requirements

```python
# Expected interface
chat_app = ChatApplication(redis_url="redis://localhost:6379")

# Create session
session_id = chat_app.create_session(user_id="alice")

# Chat
response = chat_app.chat(session_id, "Hello!")

# Get session info
info = chat_app.get_session_info(session_id)
# {'message_count': 2, 'created_at': '...', 'last_active': '...'}

# List user sessions
sessions = chat_app.list_sessions(user_id="alice")

# Delete session
chat_app.delete_session(session_id)
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use RedisChatMessageHistory for persistence
- Store session metadata in separate Redis keys
- Implement message trimming in custom history class
- Use Redis EXPIRE for TTL

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
"""Complete chat application with session management."""
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import redis


class LimitedRedisChatHistory(BaseChatMessageHistory):
    """Redis chat history with message limit."""
    
    def __init__(
        self,
        session_id: str,
        redis_client: redis.Redis,
        max_messages: int = 50,
        ttl_seconds: int = 3600
    ):
        self.session_id = session_id
        self.redis = redis_client
        self.max_messages = max_messages
        self.ttl = ttl_seconds
        self.key = f"chat:{session_id}:messages"
    
    @property
    def messages(self) -> list[BaseMessage]:
        data = self.redis.get(self.key)
        if data:
            return messages_from_dict(json.loads(data))
        return []
    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        current = self.messages
        current.extend(messages)
        
        # Trim to limit
        if len(current) > self.max_messages:
            current = current[-self.max_messages:]
        
        # Save and set TTL
        self.redis.setex(
            self.key,
            self.ttl,
            json.dumps(messages_to_dict(current))
        )
    
    def clear(self) -> None:
        self.redis.delete(self.key)


@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    message_count: int


class ChatApplication:
    """Complete chat application with session management."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_messages: int = 50,
        session_ttl: int = 3600
    ):
        self.redis = redis.from_url(redis_url)
        self.max_messages = max_messages
        self.session_ttl = session_ttl
        
        # Build chain
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        
        self.chain_with_memory = RunnableWithMessageHistory(
            chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    def _get_history(self, session_id: str) -> LimitedRedisChatHistory:
        return LimitedRedisChatHistory(
            session_id=session_id,
            redis_client=self.redis,
            max_messages=self.max_messages,
            ttl_seconds=self.session_ttl
        )
    
    def _session_meta_key(self, session_id: str) -> str:
        return f"chat:{session_id}:meta"
    
    def _user_sessions_key(self, user_id: str) -> str:
        return f"user:{user_id}:sessions"
    
    def create_session(self, user_id: str) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Store session metadata
        meta = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "last_active": now
        }
        self.redis.setex(
            self._session_meta_key(session_id),
            self.session_ttl,
            json.dumps(meta)
        )
        
        # Add to user's session list
        self.redis.sadd(self._user_sessions_key(user_id), session_id)
        
        return session_id
    
    def chat(self, session_id: str, message: str) -> str:
        """Send a message and get response."""
        # Validate session exists
        meta_key = self._session_meta_key(session_id)
        if not self.redis.exists(meta_key):
            raise ValueError(f"Session {session_id} not found or expired")
        
        # Update last active
        meta = json.loads(self.redis.get(meta_key))
        meta["last_active"] = datetime.now().isoformat()
        self.redis.setex(meta_key, self.session_ttl, json.dumps(meta))
        
        # Get response
        config = {"configurable": {"session_id": session_id}}
        response = self.chain_with_memory.invoke(
            {"input": message},
            config=config
        )
        
        return response.content
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        meta_key = self._session_meta_key(session_id)
        meta_data = self.redis.get(meta_key)
        
        if not meta_data:
            return None
        
        meta = json.loads(meta_data)
        history = self._get_history(session_id)
        
        return SessionInfo(
            session_id=meta["session_id"],
            user_id=meta["user_id"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            last_active=datetime.fromisoformat(meta["last_active"]),
            message_count=len(history.messages)
        )
    
    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        """List all sessions for a user."""
        session_ids = self.redis.smembers(self._user_sessions_key(user_id))
        
        sessions = []
        for session_id in session_ids:
            info = self.get_session_info(session_id.decode())
            if info:
                sessions.append(info)
        
        return sorted(sessions, key=lambda s: s.last_active, reverse=True)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        meta_key = self._session_meta_key(session_id)
        meta_data = self.redis.get(meta_key)
        
        if not meta_data:
            return False
        
        meta = json.loads(meta_data)
        
        # Remove from user's sessions
        self.redis.srem(self._user_sessions_key(meta["user_id"]), session_id)
        
        # Delete session data
        self.redis.delete(meta_key)
        self.redis.delete(f"chat:{session_id}:messages")
        
        return True


# Test (requires Redis running)
if __name__ == "__main__":
    app = ChatApplication()
    
    # Create session
    session = app.create_session("alice")
    print(f"Created session: {session}")
    
    # Chat
    response = app.chat(session, "Hello, I'm Alice!")
    print(f"Response: {response[:50]}...")
    
    response = app.chat(session, "What's my name?")
    print(f"Response: {response[:50]}...")
    
    # Get info
    info = app.get_session_info(session)
    print(f"Session info: {info}")
    
    # List sessions
    sessions = app.list_sessions("alice")
    print(f"Alice's sessions: {len(sessions)}")
    
    # Cleanup
    app.delete_session(session)
    print("Session deleted")
```

</details>

### Bonus Challenges

- [ ] Add conversation export (to JSON/Markdown)
- [ ] Implement session sharing between users
- [ ] Add conversation branching (fork from a point)

---

## Summary

Modern memory integration in LangChain centers around `RunnableWithMessageHistory`:

✅ **RunnableWithMessageHistory** — Wraps any LCEL chain with automatic history management  
✅ **Session-based architecture** — Each user/conversation gets isolated history  
✅ **Multiple persistence options** — InMemory, Redis, PostgreSQL, MongoDB  
✅ **Custom implementations** — Implement `BaseChatMessageHistory` for any backend  
✅ **Async support** — Full `ainvoke`, `astream` compatibility  
✅ **Production patterns** — FastAPI integration, streaming, error handling

This completes the Memory Systems lesson. You now have the tools to build sophisticated, production-ready conversational AI applications with proper context management.

---

## Further Reading

- [RunnableWithMessageHistory Documentation](https://python.langchain.com/docs/how_to/message_history/)
- [Message History Implementations](https://python.langchain.com/docs/integrations/memory/)
- [LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [LangChain Memory Migration Guide](https://python.langchain.com/docs/versions/migrating_memory/)

---

<!-- 
Sources Consulted:
- LangChain GitHub: langchain_core/runnables/history.py
- LangChain GitHub: langchain_core/chat_history.py
- LangChain Memory Migration documentation
- Redis, PostgreSQL, MongoDB chat history implementations
-->
