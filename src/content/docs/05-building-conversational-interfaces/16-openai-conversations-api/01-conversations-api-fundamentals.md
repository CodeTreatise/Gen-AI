---
title: "Conversations API Fundamentals"
---

# Conversations API Fundamentals

## Introduction

The Conversations API provides durable, server-side conversation objects that persist across sessions and devices. Unlike manual message array management, conversations are first-class API resources that you can create, retrieve, update, and delete.

This lesson covers the core CRUD operations for conversation objects and how to use metadata for organization.

### What We'll Cover

- Creating conversation objects
- Retrieving conversation details
- Updating conversation metadata
- Deleting conversations
- Using metadata for organization and querying

### Prerequisites

- OpenAI API key configured
- Python `openai` package installed (v1.50.0+)

---

## The Conversation Object

Every conversation has this structure:

```json
{
  "id": "conv_689667905b048191b4740501625afd940c7533ace33a2dab",
  "object": "conversation",
  "created_at": 1741900000,
  "metadata": {
    "topic": "customer_support",
    "user_id": "user_123"
  }
}
```

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique conversation identifier (starts with `conv_`) |
| `object` | string | Always `"conversation"` |
| `created_at` | integer | Unix timestamp of creation |
| `metadata` | object | Up to 16 key-value pairs for custom data |

---

## Creating Conversations

### Basic Creation

```python
from openai import OpenAI

client = OpenAI()

conversation = client.conversations.create()

print(conversation.id)
# conv_689667905b048191b4740501625afd940c7533ace33a2dab
```

### With Metadata

Metadata helps you organize and query conversations:

```python
conversation = client.conversations.create(
    metadata={
        "user_id": "user_123",
        "topic": "billing_inquiry",
        "session_id": "sess_abc",
        "priority": "high"
    }
)
```

> **Note:** Metadata is limited to 16 key-value pairs. Keys can be up to 64 characters, values up to 512 characters.

### With Initial Items

You can seed a conversation with up to 20 initial items:

```python
conversation = client.conversations.create(
    metadata={"topic": "demo"},
    items=[
        {
            "type": "message",
            "role": "user",
            "content": "Hello!"
        },
        {
            "type": "message",
            "role": "assistant", 
            "content": "Hi there! How can I help you today?"
        }
    ]
)
```

This is useful for:
- Migrating existing chat histories
- Setting up conversation context
- Pre-populating with system context

---

## Retrieving Conversations

### Get a Single Conversation

```python
conversation = client.conversations.retrieve("conv_123")

print(f"Created: {conversation.created_at}")
print(f"Metadata: {conversation.metadata}")
```

### Listing Conversations

> **Note:** As of the current API, listing all conversations requires using the dashboard or tracking IDs in your own database. The API focuses on individual conversation operations.

### Practical Pattern: Track Conversation IDs

```python
import json

class ConversationStore:
    def __init__(self, filepath="conversations.json"):
        self.filepath = filepath
        self._load()
    
    def _load(self):
        try:
            with open(self.filepath) as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
    
    def save(self, user_id: str, conversation_id: str, metadata: dict = None):
        self.data[user_id] = {
            "conversation_id": conversation_id,
            "metadata": metadata or {}
        }
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)
    
    def get(self, user_id: str) -> str | None:
        return self.data.get(user_id, {}).get("conversation_id")

# Usage
store = ConversationStore()

def get_or_create_conversation(client, user_id: str):
    existing_id = store.get(user_id)
    
    if existing_id:
        try:
            return client.conversations.retrieve(existing_id)
        except Exception:
            pass  # Conversation may have been deleted
    
    # Create new conversation
    conversation = client.conversations.create(
        metadata={"user_id": user_id}
    )
    store.save(user_id, conversation.id)
    return conversation
```

---

## Updating Conversations

You can update a conversation's metadata at any time:

```python
updated = client.conversations.update(
    "conv_123",
    metadata={
        "topic": "billing_resolved",
        "resolution": "refund_issued",
        "resolved_at": "2025-01-30"
    }
)

print(updated.metadata)
```

### Use Cases for Updates

| Scenario | Metadata Update |
|----------|-----------------|
| Mark resolved | `{"status": "resolved"}` |
| Add tags | `{"tags": "urgent,escalated"}` |
| Track agent | `{"assigned_agent": "agent_456"}` |
| Store summary | `{"summary": "Customer asked about..."}` |

> **Important:** Updating metadata replaces the entire metadata object. Include all keys you want to keep.

---

## Deleting Conversations

Delete a conversation when it's no longer needed:

```python
result = client.conversations.delete("conv_123")

print(result)
# {"id": "conv_123", "object": "conversation.deleted", "deleted": true}
```

> **Warning:** Deleting a conversation does **not** delete its items. Items remain stored separately.

### When to Delete

| Scenario | Action |
|----------|--------|
| User requests data deletion | Delete conversation + items |
| Conversation resolved | Keep or delete based on retention policy |
| Testing/development | Delete test conversations |
| Compliance requirement | Delete per policy timeline |

### Complete Deletion (Conversation + Items)

```python
def delete_conversation_completely(client, conversation_id: str):
    # First, delete all items
    items = client.conversations.items.list(conversation_id)
    
    for item in items.data:
        try:
            client.conversations.items.delete(
                conversation_id, 
                item.id
            )
        except Exception as e:
            print(f"Failed to delete item {item.id}: {e}")
    
    # Then delete the conversation
    client.conversations.delete(conversation_id)
    print(f"Deleted conversation {conversation_id} and all items")
```

---

## Metadata Best Practices

### Schema Design

```python
# Recommended metadata schema
metadata = {
    # Identifiers
    "user_id": "user_123",
    "session_id": "sess_abc",
    "org_id": "org_456",
    
    # Classification
    "topic": "billing",
    "category": "refund",
    "priority": "high",
    
    # Status tracking
    "status": "active",  # active, resolved, archived
    "created_by": "web_app",
    
    # Timestamps (as strings)
    "started_at": "2025-01-30T10:00:00Z",
}
```

### Querying by Metadata

Since the API doesn't provide server-side filtering, maintain an index:

```python
from dataclasses import dataclass
from typing import Optional
import sqlite3

@dataclass
class ConversationRecord:
    id: str
    user_id: str
    topic: Optional[str]
    status: str
    created_at: int

class ConversationIndex:
    def __init__(self, db_path="conversations.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                topic TEXT,
                status TEXT DEFAULT 'active',
                created_at INTEGER
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user ON conversations(user_id)"
        )
        self.conn.commit()
    
    def add(self, record: ConversationRecord):
        self.conn.execute(
            """INSERT OR REPLACE INTO conversations 
               (id, user_id, topic, status, created_at) 
               VALUES (?, ?, ?, ?, ?)""",
            (record.id, record.user_id, record.topic, 
             record.status, record.created_at)
        )
        self.conn.commit()
    
    def find_by_user(self, user_id: str) -> list[ConversationRecord]:
        cursor = self.conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        return [ConversationRecord(*row) for row in cursor.fetchall()]
    
    def find_active(self) -> list[ConversationRecord]:
        cursor = self.conn.execute(
            "SELECT * FROM conversations WHERE status = 'active'"
        )
        return [ConversationRecord(*row) for row in cursor.fetchall()]
```

---

## Error Handling

### Common Errors

```python
from openai import NotFoundError, BadRequestError, RateLimitError

def safe_get_conversation(client, conversation_id: str):
    try:
        return client.conversations.retrieve(conversation_id)
    
    except NotFoundError:
        print(f"Conversation {conversation_id} not found")
        return None
    
    except BadRequestError as e:
        print(f"Invalid request: {e}")
        return None
    
    except RateLimitError:
        print("Rate limited - retry with backoff")
        raise
```

### Validation

```python
def validate_metadata(metadata: dict) -> bool:
    if len(metadata) > 16:
        raise ValueError("Metadata limited to 16 key-value pairs")
    
    for key, value in metadata.items():
        if len(key) > 64:
            raise ValueError(f"Key '{key}' exceeds 64 character limit")
        if len(str(value)) > 512:
            raise ValueError(f"Value for '{key}' exceeds 512 character limit")
    
    return True
```

---

## Complete Example

```python
from openai import OpenAI
from datetime import datetime

client = OpenAI()

class ChatSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation = None
    
    def start(self, topic: str = "general"):
        """Start a new conversation session."""
        self.conversation = client.conversations.create(
            metadata={
                "user_id": self.user_id,
                "topic": topic,
                "status": "active",
                "started_at": datetime.now().isoformat()
            }
        )
        print(f"Started conversation: {self.conversation.id}")
        return self.conversation.id
    
    def resume(self, conversation_id: str):
        """Resume an existing conversation."""
        self.conversation = client.conversations.retrieve(conversation_id)
        print(f"Resumed conversation: {self.conversation.id}")
    
    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        if not self.conversation:
            raise ValueError("No active conversation. Call start() or resume() first.")
        
        response = client.responses.create(
            model="gpt-4o",
            input=message,
            conversation=self.conversation.id
        )
        
        return response.output_text
    
    def update_topic(self, topic: str):
        """Update the conversation topic."""
        if not self.conversation:
            raise ValueError("No active conversation")
        
        current_metadata = dict(self.conversation.metadata)
        current_metadata["topic"] = topic
        
        self.conversation = client.conversations.update(
            self.conversation.id,
            metadata=current_metadata
        )
    
    def resolve(self, resolution: str = None):
        """Mark conversation as resolved."""
        if not self.conversation:
            raise ValueError("No active conversation")
        
        metadata = dict(self.conversation.metadata)
        metadata["status"] = "resolved"
        metadata["resolved_at"] = datetime.now().isoformat()
        if resolution:
            metadata["resolution"] = resolution
        
        self.conversation = client.conversations.update(
            self.conversation.id,
            metadata=metadata
        )
        print(f"Resolved conversation: {self.conversation.id}")
    
    def end(self, delete: bool = False):
        """End the session, optionally deleting the conversation."""
        if delete and self.conversation:
            client.conversations.delete(self.conversation.id)
            print(f"Deleted conversation: {self.conversation.id}")
        
        self.conversation = None


# Usage
session = ChatSession(user_id="user_123")

# Start a new conversation
conv_id = session.start(topic="product_inquiry")

# Have a conversation
print(session.chat("What products do you offer?"))
print(session.chat("Tell me more about the premium plan"))

# Update metadata
session.update_topic("premium_plan_inquiry")

# Mark as resolved
session.resolve(resolution="customer_subscribed")

# Or resume later
# session.resume(conv_id)
# print(session.chat("I have a follow-up question"))
```

---

## Summary

✅ Conversations are persistent API resources with unique IDs

✅ Create conversations with optional metadata and initial items

✅ Retrieve conversations to access their metadata and details

✅ Update metadata to track status, topics, and custom data

✅ Delete conversations when no longer needed (items persist separately)

✅ Use local indexes for efficient querying by metadata

**Next:** [Conversation Items](./02-conversation-items.md)

---

## Further Reading

- [Conversations API Reference](https://platform.openai.com/docs/api-reference/conversations) — Full API documentation
- [Create Conversation](https://platform.openai.com/docs/api-reference/conversations/create) — Creation endpoint
- [Conversation Object](https://platform.openai.com/docs/api-reference/conversations/object) — Object schema

---

<!-- 
Sources Consulted:
- Conversations API Reference: https://platform.openai.com/docs/api-reference/conversations
- Conversation State Guide: https://platform.openai.com/docs/guides/conversation-state
-->
