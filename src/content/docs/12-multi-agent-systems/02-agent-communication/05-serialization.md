---
title: "Serialization"
---

# Serialization

## Introduction

When agents exchange messages, those messages exist as Python objects in memory. But agents might run on different machines, communicate through message queues, or need their conversations persisted to disk. Serialization — converting objects to a portable format and back — is the bridge between in-memory agent communication and durable, transportable messages.

Getting serialization wrong leads to subtle bugs: lost fields, type confusion, and version mismatches when one agent is updated but others aren't. A well-designed serialization strategy ensures agents can communicate reliably regardless of where or when they're running.

### What We'll Cover
- Designing message formats for agent communication
- JSON serialization with type preservation
- Using Pydantic models for validated messages
- Schema evolution and backward compatibility
- Breaking changes and version negotiation

### Prerequisites
- Python dataclasses and type hints
- JSON structure and syntax
- Pydantic basics (helpful but not required)

---

## Message Format Design

A good message format balances human readability, machine parseability, and extensibility. We need structured formats that agents can produce and consume reliably.

### Choosing a Format

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| **JSON** | Human-readable, universal support | No native datetime/enum, verbose | HTTP APIs, debugging, LLM agents |
| **MessagePack** | Compact binary, fast parsing | Not human-readable | High-throughput agent pipelines |
| **Protocol Buffers** | Schema-enforced, very efficient | Requires `.proto` files, setup overhead | Large-scale production systems |
| **YAML** | Very human-readable | Slow parsing, security concerns | Configuration, not runtime messages |

For most multi-agent systems, **JSON** is the right default — LLM-based agents already think in text, debugging is straightforward, and every language supports it natively.

### Designing an Agent Message Schema

```python
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Message types use string values for JSON compatibility."""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    HANDOFF = "handoff"


@dataclass
class AgentMessage:
    """Standard message format for agent communication."""
    # Required fields
    sender: str
    receiver: str
    msg_type: MessageType
    content: str
    
    # Auto-generated fields
    message_id: str = field(
        default_factory=lambda: __import__('uuid').uuid4().hex[:12]
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    # Optional fields
    correlation_id: str | None = None  # Links request to response
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = asdict(self)
        # Enum values serialize as their string values automatically
        # because MessageType inherits from str
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        # Convert string back to enum
        data["msg_type"] = MessageType(data["msg_type"])
        return cls(**data)


# Demo: round-trip serialization
msg = AgentMessage(
    sender="researcher",
    receiver="coordinator",
    msg_type=MessageType.RESULT,
    content="Found 5 key AI trends for 2025",
    correlation_id="req-001",
    metadata={"sources": 3, "confidence": 0.92}
)

# Serialize
json_str = msg.to_json()
print("Serialized message:")
print(json_str)

# Deserialize
restored = AgentMessage.from_json(json_str)
print(f"\nRestored: {restored.sender} → {restored.receiver}")
print(f"Type: {restored.msg_type}")
print(f"Content: {restored.content}")
print(f"Match: {msg.content == restored.content}")
```

**Output:**
```
Serialized message:
{
  "sender": "researcher",
  "receiver": "coordinator",
  "msg_type": "result",
  "content": "Found 5 key AI trends for 2025",
  "message_id": "a3f7c2e1b8d4",
  "timestamp": "2025-01-15T10:30:00.000000",
  "correlation_id": "req-001",
  "version": "1.0",
  "metadata": {
    "sources": 3,
    "confidence": 0.92
  }
}

Restored: researcher → coordinator
Type: MessageType.RESULT
Content: Found 5 key AI trends for 2025
Match: True
```

> **Note:** By making `MessageType` inherit from both `str` and `Enum`, the enum values serialize to clean strings (`"result"`) instead of `"MessageType.RESULT"`. This makes JSON output clean and compatible with non-Python consumers.

---

## JSON Serialization with Type Preservation

Standard JSON doesn't support Python-specific types like `datetime`, `set`, `bytes`, or custom classes. We need a strategy for handling these types without losing information.

### Custom JSON Encoder/Decoder

```python
import json
from datetime import datetime, date
from enum import Enum
from dataclasses import is_dataclass, asdict
from typing import Any


class AgentMessageEncoder(json.JSONEncoder):
    """Handles Python types that JSON doesn't support natively."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        if isinstance(obj, date):
            return {"__type__": "date", "value": obj.isoformat()}
        if isinstance(obj, set):
            return {"__type__": "set", "value": list(obj)}
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "value": obj.decode("utf-8", errors="replace")
            }
        if isinstance(obj, Enum):
            return {"__type__": "enum", "class": type(obj).__name__,
                    "value": obj.value}
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                "__type__": "dataclass",
                "class": type(obj).__name__,
                "data": asdict(obj)
            }
        return super().default(obj)


def agent_message_decoder(obj: dict) -> Any:
    """Object hook for json.loads to restore Python types."""
    if "__type__" not in obj:
        return obj
    
    type_tag = obj["__type__"]
    if type_tag == "datetime":
        return datetime.fromisoformat(obj["value"])
    if type_tag == "date":
        return date.fromisoformat(obj["value"])
    if type_tag == "set":
        return set(obj["value"])
    if type_tag == "bytes":
        return obj["value"].encode("utf-8")
    
    # For enums and dataclasses, return as dict
    # (full reconstruction requires a type registry)
    return obj


# Demo
data = {
    "created_at": datetime(2025, 1, 15, 10, 30),
    "tags": {"ai", "agents", "python"},
    "raw_data": b"binary content here",
    "results": {
        "scores": [0.95, 0.87, 0.91],
        "completed_at": datetime(2025, 1, 15, 11, 0),
    }
}

# Encode
encoded = json.dumps(data, cls=AgentMessageEncoder, indent=2)
print("Encoded with type tags:")
print(encoded)

# Decode
decoded = json.loads(encoded, object_hook=agent_message_decoder)
print(f"\nDecoded created_at type: {type(decoded['created_at']).__name__}")
print(f"Decoded tags type: {type(decoded['tags']).__name__}")
print(f"Tags match: {decoded['tags'] == data['tags']}")
```

**Output:**
```
Encoded with type tags:
{
  "created_at": {
    "__type__": "datetime",
    "value": "2025-01-15T10:30:00"
  },
  "tags": {
    "__type__": "set",
    "value": ["agents", "ai", "python"]
  },
  "raw_data": {
    "__type__": "bytes",
    "value": "binary content here"
  },
  "results": {
    "scores": [0.95, 0.87, 0.91],
    "completed_at": {
      "__type__": "datetime",
      "value": "2025-01-15T11:00:00"
    }
  }
}

Decoded created_at type: datetime
Decoded tags type: set
Tags match: True
```

> **Warning:** The `__type__` tagging approach works within your own system. If agents communicate with external systems, use ISO 8601 strings for dates and arrays for sets — don't assume other systems understand your type tags.

---

## Pydantic Models for Validated Messages

Pydantic models provide automatic validation, serialization, and documentation. They're the standard approach in AutoGen 0.4+ and a natural fit for structured agent messages.

### Validated Message Models

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from typing import Any


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskMessage(BaseModel):
    """A validated task message between agents."""
    sender: str = Field(min_length=1, description="Agent sending the message")
    receiver: str = Field(min_length=1, description="Agent receiving the message")
    task: str = Field(min_length=1, max_length=5000)
    priority: Priority = Priority.NORMAL
    deadline_minutes: int | None = Field(default=None, ge=1, le=1440)
    context: dict[str, Any] = Field(default_factory=dict)
    
    # Auto-generated
    message_id: str = Field(
        default_factory=lambda: __import__('uuid').uuid4().hex[:12]
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"
    
    @field_validator("sender", "receiver")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Agent names must be alphanumeric with underscores."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Agent name must be alphanumeric: '{v}'"
            )
        return v


class ResultMessage(BaseModel):
    """A validated result message."""
    sender: str
    receiver: str
    result: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    sources: list[str] = Field(default_factory=list)
    correlation_id: str  # Must link back to original task
    
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"


# Demo: validation in action
print("1. Valid message:")
task = TaskMessage(
    sender="coordinator",
    receiver="researcher",
    task="Analyze AI market trends",
    priority=Priority.HIGH,
    deadline_minutes=30,
    context={"region": "North America"}
)
print(f"   {task.sender} → {task.receiver}: {task.task}")
print(f"   Priority: {task.priority.value}, Deadline: {task.deadline_minutes}m")

# Serialize to JSON
json_str = task.model_dump_json(indent=2)
print(f"\n2. Serialized ({len(json_str)} chars)")

# Deserialize
restored = TaskMessage.model_validate_json(json_str)
print(f"   Restored: {restored.task}")
print(f"   Match: {task.task == restored.task}")

# Validation errors
print("\n3. Validation errors:")
try:
    bad_task = TaskMessage(
        sender="",  # Empty — violates min_length=1
        receiver="researcher",
        task="Do something"
    )
except Exception as e:
    error_msg = str(e)
    print(f"   Empty sender: {error_msg[:80]}...")

try:
    bad_deadline = TaskMessage(
        sender="coordinator",
        receiver="researcher",
        task="Do something",
        deadline_minutes=0  # Below minimum of 1
    )
except Exception as e:
    error_msg = str(e)
    print(f"   Bad deadline: {error_msg[:80]}...")
```

**Output:**
```
1. Valid message:
   coordinator → researcher: Analyze AI market trends
   Priority: high, Deadline: 30m

2. Serialized (312 chars)
   Restored: Analyze AI market trends
   Match: True

3. Validation errors:
   Empty sender: 1 validation error for TaskMessage
sender
  String should have at le...
   Bad deadline: 1 validation error for TaskMessage
deadline_minutes
  Input should be gre...
```

### AutoGen 0.4+ Message Design

AutoGen 0.4+ uses Pydantic or dataclass messages as first-class routing keys:

```python
from dataclasses import dataclass
from pydantic import BaseModel


# Approach 1: Dataclass messages (lightweight)
@dataclass
class ResearchRequest:
    topic: str
    depth: str = "standard"  # "quick", "standard", "deep"


@dataclass
class ResearchResult:
    topic: str
    findings: str
    sources: int
    confidence: float


# Approach 2: Pydantic messages (validated)
class AnalysisRequest(BaseModel):
    data: dict
    analysis_type: str = "summary"


class AnalysisResult(BaseModel):
    insights: list[str]
    confidence: float

# In AutoGen, message types determine routing:
# @message_handler decorator routes ResearchRequest 
# to the research handler and AnalysisRequest to 
# the analysis handler automatically.
```

> **🤖 AI Context:** AutoGen 0.4+ routes messages by type — a `ResearchRequest` message automatically goes to the handler registered for that type. This means your message classes double as routing keys. The serialization format (dataclass vs Pydantic) affects validation but not routing.

---

## Schema Evolution

Agents don't all update at the same time. When you change a message schema — adding a field, removing one, or changing a type — some agents will still send the old format. Schema evolution strategies handle this gracefully.

### Backward-Compatible Changes

```python
from pydantic import BaseModel, Field
from typing import Any


# Version 1.0 — Original message
class TaskMessageV1(BaseModel):
    sender: str
    receiver: str
    task: str
    version: str = "1.0"


# Version 1.1 — Added optional fields (BACKWARD COMPATIBLE)
class TaskMessageV1_1(BaseModel):
    sender: str
    receiver: str
    task: str
    priority: str = "normal"  # New field with default
    tags: list[str] = Field(default_factory=list)  # New field with default
    version: str = "1.1"


# Version 1.2 — Added more optional fields (BACKWARD COMPATIBLE)
class TaskMessageV1_2(BaseModel):
    sender: str
    receiver: str
    task: str
    priority: str = "normal"
    tags: list[str] = Field(default_factory=list)
    deadline_minutes: int | None = None  # New optional field
    context: dict[str, Any] = Field(default_factory=dict)  # New optional
    version: str = "1.2"


# Demo: V1 message parsed by V1.2 model
v1_json = '{"sender": "old_agent", "receiver": "new_agent", "task": "Research AI"}'

print("Parsing V1 message with V1.2 model:")
parsed = TaskMessageV1_2.model_validate_json(v1_json)
print(f"  sender: {parsed.sender}")
print(f"  task: {parsed.task}")
print(f"  priority: {parsed.priority} (default applied)")
print(f"  tags: {parsed.tags} (default applied)")
print(f"  deadline: {parsed.deadline_minutes} (default applied)")
print(f"  version: {parsed.version} (default applied)")

# V1.2 message parsed by V1.2 model
v12_json = '''
{
    "sender": "new_agent",
    "receiver": "coordinator",
    "task": "Deep analysis",
    "priority": "high",
    "tags": ["urgent", "ai"],
    "deadline_minutes": 30,
    "context": {"region": "NA"},
    "version": "1.2"
}
'''
print("\nParsing V1.2 message with V1.2 model:")
parsed = TaskMessageV1_2.model_validate_json(v12_json)
print(f"  All fields present: priority={parsed.priority}, "
      f"tags={parsed.tags}, deadline={parsed.deadline_minutes}m")
```

**Output:**
```
Parsing V1 message with V1.2 model:
  sender: old_agent
  task: Research AI
  priority: normal (default applied)
  tags: [] (default applied)
  deadline: None (default applied)
  version: 1.2 (default applied)

Parsing V1.2 message with V1.2 model:
  All fields present: priority=high, tags=['urgent', 'ai'], deadline=30m
```

### Rules for Safe Schema Evolution

| Change Type | Safe? | Example |
|-------------|-------|---------|
| Add optional field with default | ✅ Yes | `tags: list = []` |
| Add required field | ❌ No | `priority: str` (no default) |
| Remove optional field | ⚠️ Risky | Old senders still send it |
| Change field type | ❌ No | `priority: str` → `priority: int` |
| Rename field | ❌ No | `task` → `task_description` |
| Add new enum value | ✅ Yes | `Priority.CRITICAL` added |
| Remove enum value | ❌ No | Old messages may contain it |

---

## Version Negotiation

When breaking changes are unavoidable, agents need to negotiate which schema version to use. Version negotiation lets agents discover each other's capabilities and agree on a compatible format.

### Version-Aware Message Router

```python
from pydantic import BaseModel, Field
from typing import Any
import json


class VersionedMessage(BaseModel):
    """Base message with version information."""
    version: str
    sender: str
    payload: dict[str, Any]


class SchemaRegistry:
    """Registry of message schemas by version."""
    
    def __init__(self):
        self._schemas: dict[str, type[BaseModel]] = {}
        self._transformers: dict[tuple[str, str], callable] = {}
    
    def register(self, version: str, schema: type[BaseModel]):
        """Register a schema for a version."""
        self._schemas[version] = schema
        print(f"  [Registry] Registered schema v{version}: "
              f"{schema.__name__}")
    
    def register_transformer(
        self, from_version: str, to_version: str, 
        transformer: callable
    ):
        """Register a function to convert between versions."""
        self._transformers[(from_version, to_version)] = transformer
        print(f"  [Registry] Transformer: v{from_version} → v{to_version}")
    
    def parse(
        self, data: dict, target_version: str
    ) -> BaseModel | None:
        """Parse message data, transforming if needed."""
        source_version = data.get("version", "1.0")
        
        if source_version == target_version:
            schema = self._schemas.get(target_version)
            if schema:
                return schema.model_validate(data)
        
        # Try transformation
        transformer = self._transformers.get(
            (source_version, target_version)
        )
        if transformer:
            transformed = transformer(data)
            schema = self._schemas.get(target_version)
            if schema:
                return schema.model_validate(transformed)
        
        print(f"  [Registry] No path from v{source_version} "
              f"to v{target_version}")
        return None


# Define versioned schemas
class TaskV1(BaseModel):
    version: str = "1.0"
    sender: str
    receiver: str
    task: str


class TaskV2(BaseModel):
    version: str = "2.0"
    sender: str
    receiver: str
    task: str
    priority: str = "normal"
    max_tokens: int = 1000


# Transformer: V1 → V2
def v1_to_v2(data: dict) -> dict:
    """Transform V1 message to V2 format."""
    return {
        **data,
        "version": "2.0",
        "priority": "normal",  # Default for upgraded messages
        "max_tokens": 1000,    # Default for upgraded messages
    }


# Demo
print("Schema Registry Demo:\n")
registry = SchemaRegistry()
registry.register("1.0", TaskV1)
registry.register("2.0", TaskV2)
registry.register_transformer("1.0", "2.0", v1_to_v2)

# V1 agent sends a message
v1_data = {
    "version": "1.0",
    "sender": "old_agent",
    "receiver": "new_agent",
    "task": "Research agentic AI"
}

print(f"\n  Incoming message: v{v1_data['version']}")

# V2 agent receives and needs V2 format
result = registry.parse(v1_data, target_version="2.0")
if result:
    print(f"  Transformed to v2.0:")
    print(f"    task: {result.task}")
    print(f"    priority: {result.priority} (default)")
    print(f"    max_tokens: {result.max_tokens} (default)")

# V2 message — no transformation needed
v2_data = {
    "version": "2.0",
    "sender": "new_agent",
    "receiver": "coordinator",
    "task": "Analyze results",
    "priority": "high",
    "max_tokens": 2000
}

print(f"\n  Incoming message: v{v2_data['version']}")
result = registry.parse(v2_data, target_version="2.0")
if result:
    print(f"  Parsed directly as v2.0:")
    print(f"    task: {result.task}")
    print(f"    priority: {result.priority}")
    print(f"    max_tokens: {result.max_tokens}")
```

**Output:**
```
Schema Registry Demo:

  [Registry] Registered schema v1.0: TaskV1
  [Registry] Registered schema v2.0: TaskV2
  [Registry] Transformer: v1.0 → v2.0

  Incoming message: v1.0
  Transformed to v2.0:
    task: Research agentic AI
    priority: normal (default)
    max_tokens: 1000 (default)

  Incoming message: v2.0
  Parsed directly as v2.0:
    task: Analyze results
    priority: high
    max_tokens: 2000
```

### Version Negotiation Handshake

```python
from dataclasses import dataclass


@dataclass
class VersionCapability:
    agent_id: str
    supported_versions: list[str]
    preferred_version: str


def negotiate_version(
    sender: VersionCapability,
    receiver: VersionCapability
) -> str | None:
    """Find the highest mutually supported version."""
    common = set(sender.supported_versions) & set(
        receiver.supported_versions
    )
    if not common:
        return None
    
    # Pick the highest common version
    return max(common, key=lambda v: [int(x) for x in v.split(".")])


# Demo
agent_old = VersionCapability(
    agent_id="legacy_researcher",
    supported_versions=["1.0", "1.1"],
    preferred_version="1.1"
)

agent_new = VersionCapability(
    agent_id="modern_coordinator",
    supported_versions=["1.1", "2.0", "2.1"],
    preferred_version="2.1"
)

agent_cutting = VersionCapability(
    agent_id="cutting_edge",
    supported_versions=["3.0"],
    preferred_version="3.0"
)

v = negotiate_version(agent_old, agent_new)
print(f"old ↔ new: v{v}")

v = negotiate_version(agent_new, agent_old)
print(f"new ↔ old: v{v}")

v = negotiate_version(agent_old, agent_cutting)
print(f"old ↔ cutting: {v} (incompatible!)")
```

**Output:**
```
old ↔ new: v1.1
new ↔ old: v1.1
old ↔ cutting: None (incompatible!)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use Pydantic for all external message schemas | Validation catches errors at the boundary, not deep in processing |
| Include a `version` field in every message | Enables schema evolution and version negotiation |
| Use `str` enums for JSON compatibility | `class Priority(str, Enum)` serializes cleanly |
| Default new fields for backward compatibility | Old agents sending V1 messages should work with V2 parsers |
| Keep messages as data, not behavior | Messages are nouns (data), not verbs (methods) |
| Use ISO 8601 for timestamps | Universal format that every language can parse |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Adding required fields without defaults | Always give new fields defaults — old agents can't provide them |
| Using Python-specific types in JSON | Stick to JSON-native types or use explicit type tags |
| No version field in messages | Add `version: str = "1.0"` from day one |
| Renaming fields in updated schemas | Use field aliases instead: `Field(alias="old_name")` |
| Returning Pydantic models as raw dicts | Use `model_dump_json()` to ensure proper serialization |
| Ignoring unknown fields | Use `model_config = ConfigDict(extra="ignore")` to be tolerant |

---

## Hands-on Exercise

### Your Task

Build a version-aware message system that handles backward-compatible communication between agents running different schema versions.

### Requirements

1. Define `TaskV1` (sender, receiver, task) and `TaskV2` (adds priority, deadline_minutes, tags)
2. Create a `MessageSerializer` that detects version and parses accordingly
3. Implement a V1→V2 transformer that fills defaults for missing fields
4. Serialize and deserialize messages, verifying round-trip integrity
5. Handle an invalid message gracefully (return an error, don't crash)

### Expected Result

```
V1 message serialized → deserialized: ✅
V1 message upgraded to V2: priority=normal, deadline=None
V2 message serialized → deserialized: ✅ (priority=high, deadline=30)
Invalid message: Error caught — "validation error..."
```

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `model_validate_json()` / `model_dump_json()` for round-trips
- Check the `version` field first to determine which model to use
- Wrap parsing in try/except for `ValidationError`
- V1→V2 transformer: copy all V1 fields, add defaults for V2 fields

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Any
import json


class TaskV1(BaseModel):
    version: str = "1.0"
    sender: str
    receiver: str
    task: str


class TaskV2(BaseModel):
    version: str = "2.0"
    sender: str
    receiver: str
    task: str
    priority: str = "normal"
    deadline_minutes: int | None = None
    tags: list[str] = Field(default_factory=list)


class MessageSerializer:
    def __init__(self):
        self._schemas = {"1.0": TaskV1, "2.0": TaskV2}
    
    def serialize(self, msg: BaseModel) -> str:
        return msg.model_dump_json(indent=2)
    
    def deserialize(self, json_str: str) -> BaseModel | None:
        try:
            data = json.loads(json_str)
            version = data.get("version", "1.0")
            schema = self._schemas.get(version)
            if not schema:
                print(f"Unknown version: {version}")
                return None
            return schema.model_validate(data)
        except ValidationError as e:
            print(f"Error caught — \"{str(e)[:40]}...\"")
            return None
    
    def upgrade_v1_to_v2(self, v1_msg: TaskV1) -> TaskV2:
        return TaskV2(
            sender=v1_msg.sender,
            receiver=v1_msg.receiver,
            task=v1_msg.task,
            priority="normal",
            deadline_minutes=None,
            tags=[]
        )


# Test
s = MessageSerializer()

# V1 round-trip
v1 = TaskV1(sender="old", receiver="new", task="Research AI")
v1_json = s.serialize(v1)
v1_back = s.deserialize(v1_json)
print(f"V1 message serialized → deserialized: "
      f"{'✅' if v1_back and v1_back.task == v1.task else '❌'}")

# V1 → V2 upgrade
v2_upgraded = s.upgrade_v1_to_v2(v1)
print(f"V1 message upgraded to V2: priority={v2_upgraded.priority}, "
      f"deadline={v2_upgraded.deadline_minutes}")

# V2 round-trip
v2 = TaskV2(
    sender="new", receiver="coord", task="Analyze",
    priority="high", deadline_minutes=30, tags=["urgent"]
)
v2_json = s.serialize(v2)
v2_back = s.deserialize(v2_json)
print(f"V2 message serialized → deserialized: "
      f"{'✅' if v2_back else '❌'} "
      f"(priority={v2_back.priority}, deadline={v2_back.deadline_minutes})")

# Invalid message
bad_json = '{"version": "2.0", "sender": "", "task": 123}'
print(f"Invalid message: ", end="")
s.deserialize(bad_json)
```

</details>

### Bonus Challenges
- [ ] Add a V3 schema and chain transformers (V1→V2→V3)
- [ ] Implement a schema diff tool that shows changes between versions
- [ ] Add MessagePack serialization as an alternative to JSON

---

## Summary

✅ **JSON is the right default** for agent messages — universally supported, human-readable, and compatible with LLM-based agents

✅ **Pydantic models** provide validated serialization with automatic type coercion, clear field documentation, and built-in JSON support through `model_dump_json()` / `model_validate_json()`

✅ **Schema evolution** must be backward compatible — add optional fields with defaults, never remove or rename required fields, and always include a version field

✅ **Type preservation** requires explicit strategies — use ISO 8601 for dates, string enums for JSON compatibility, and `__type__` tags only within your own ecosystem

✅ **Version negotiation** enables mixed-version environments — agents discover common versions and transformers bridge incompatible schemas

**Next:** [Communication Security](./06-communication-security.md)

**Previous:** [Conversation Management](./04-conversation-management.md)

---

## Further Reading

- [Pydantic V2 Serialization](https://docs.pydantic.dev/latest/concepts/serialization/) - Complete serialization guide
- [AutoGen Message Design](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html) - Framework message conventions
- [JSON Schema](https://json-schema.org/) - Standard for describing JSON structures
- [Protobuf Python Tutorial](https://protobuf.dev/getting-started/pythontutorial/) - Alternative for high-performance messaging

<!-- 
Sources Consulted:
- AutoGen message-and-communication: https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html
- Pydantic V2 serialization: https://docs.pydantic.dev/latest/concepts/serialization/
- JSON Schema specification: https://json-schema.org/
- Python json module: https://docs.python.org/3/library/json.html
-->
