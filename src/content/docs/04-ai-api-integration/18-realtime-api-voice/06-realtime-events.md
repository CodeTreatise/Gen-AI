---
title: "Realtime Events"
---

# Realtime Events

## Introduction

The Realtime API uses a bidirectional event system. Client events control the session and send input, while server events deliver responses and status updates.

### What We'll Cover

- Client event types and usage
- Server event handling
- Conversation item events
- Error events and recovery
- Event sequencing patterns

### Prerequisites

- Understanding of WebSocket communication
- Session management concepts
- Async event handling

---

## Client Events Overview

### Client Event Types

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import asyncio


class ClientEventType(Enum):
    """Client event types for Realtime API."""
    
    # Session events
    SESSION_UPDATE = "session.update"
    
    # Audio buffer events
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # Conversation events
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    
    # Response events
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"


@dataclass
class ClientEvent:
    """Base class for client events."""
    
    type: ClientEventType
    event_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for sending."""
        data = {"type": self.type.value}
        if self.event_id:
            data["event_id"] = self.event_id
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class ClientEventBuilder:
    """Build client events with validation."""
    
    def __init__(self):
        self._event_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter}_{datetime.now().strftime('%H%M%S%f')}"
    
    def session_update(
        self,
        modalities: List[str] = None,
        voice: str = None,
        instructions: str = None,
        input_audio_format: str = None,
        output_audio_format: str = None,
        turn_detection: dict = None,
        tools: List[dict] = None,
        temperature: float = None,
        max_response_output_tokens: int = None
    ) -> dict:
        """Build session.update event."""
        
        session = {}
        
        if modalities:
            session["modalities"] = modalities
        if voice:
            session["voice"] = voice
        if instructions:
            session["instructions"] = instructions
        if input_audio_format:
            session["input_audio_format"] = input_audio_format
        if output_audio_format:
            session["output_audio_format"] = output_audio_format
        if turn_detection:
            session["turn_detection"] = turn_detection
        if tools:
            session["tools"] = tools
        if temperature is not None:
            session["temperature"] = temperature
        if max_response_output_tokens:
            session["max_response_output_tokens"] = max_response_output_tokens
        
        return {
            "event_id": self._generate_id(),
            "type": "session.update",
            "session": session
        }
    
    def input_audio_buffer_append(self, audio_base64: str) -> dict:
        """Build input_audio_buffer.append event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
    
    def input_audio_buffer_commit(self) -> dict:
        """Build input_audio_buffer.commit event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "input_audio_buffer.commit"
        }
    
    def input_audio_buffer_clear(self) -> dict:
        """Build input_audio_buffer.clear event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "input_audio_buffer.clear"
        }
    
    def conversation_item_create(
        self,
        item_type: str,
        role: str = None,
        content: List[dict] = None,
        call_id: str = None,
        name: str = None,
        arguments: str = None,
        output: str = None
    ) -> dict:
        """Build conversation.item.create event."""
        
        item = {"type": item_type}
        
        if item_type == "message":
            item["role"] = role
            item["content"] = content
        elif item_type == "function_call":
            item["call_id"] = call_id
            item["name"] = name
            item["arguments"] = arguments
        elif item_type == "function_call_output":
            item["call_id"] = call_id
            item["output"] = output
        
        return {
            "event_id": self._generate_id(),
            "type": "conversation.item.create",
            "item": item
        }
    
    def conversation_item_truncate(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int
    ) -> dict:
        """Build conversation.item.truncate event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms
        }
    
    def conversation_item_delete(self, item_id: str) -> dict:
        """Build conversation.item.delete event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "conversation.item.delete",
            "item_id": item_id
        }
    
    def response_create(
        self,
        modalities: List[str] = None,
        instructions: str = None,
        voice: str = None,
        tools: List[dict] = None,
        max_output_tokens: int = None
    ) -> dict:
        """Build response.create event."""
        
        event = {
            "event_id": self._generate_id(),
            "type": "response.create"
        }
        
        response = {}
        if modalities:
            response["modalities"] = modalities
        if instructions:
            response["instructions"] = instructions
        if voice:
            response["voice"] = voice
        if tools:
            response["tools"] = tools
        if max_output_tokens:
            response["max_output_tokens"] = max_output_tokens
        
        if response:
            event["response"] = response
        
        return event
    
    def response_cancel(self) -> dict:
        """Build response.cancel event."""
        
        return {
            "event_id": self._generate_id(),
            "type": "response.cancel"
        }


# Usage
builder = ClientEventBuilder()

# Session update
session_event = builder.session_update(
    modalities=["text", "audio"],
    voice="coral",
    instructions="You are a helpful assistant.",
    temperature=0.8
)
print(f"Session update: {json.dumps(session_event, indent=2)}")

# Audio append
audio_event = builder.input_audio_buffer_append("SGVsbG8gV29ybGQ=")
print(f"\nAudio append: {audio_event['type']}")

# Conversation item
message_event = builder.conversation_item_create(
    item_type="message",
    role="user",
    content=[{"type": "input_text", "text": "Hello!"}]
)
print(f"\nMessage create: {json.dumps(message_event, indent=2)}")
```

---

## Server Events Overview

### Server Event Types

```python
class ServerEventType(Enum):
    """Server event types from Realtime API."""
    
    # Error events
    ERROR = "error"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    
    # Conversation events
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = \
        "conversation.item.input_audio_transcription.completed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = \
        "conversation.item.input_audio_transcription.failed"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    
    # Audio buffer events
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    
    # Response events
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    
    # Streaming delta events
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    
    # Function call events
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    
    # Rate limit events
    RATE_LIMITS_UPDATED = "rate_limits.updated"


@dataclass
class ServerEvent:
    """Parsed server event."""
    
    type: str
    event_id: str
    data: dict
    received_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ServerEvent':
        """Parse server event from JSON."""
        data = json.loads(json_str)
        return cls(
            type=data.get("type", "unknown"),
            event_id=data.get("event_id", ""),
            data=data
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from event data."""
        return self.data.get(key, default)


# Event category helpers
SESSION_EVENTS = {
    ServerEventType.SESSION_CREATED,
    ServerEventType.SESSION_UPDATED
}

AUDIO_BUFFER_EVENTS = {
    ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED,
    ServerEventType.INPUT_AUDIO_BUFFER_CLEARED,
    ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED,
    ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED
}

RESPONSE_EVENTS = {
    ServerEventType.RESPONSE_CREATED,
    ServerEventType.RESPONSE_DONE,
    ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED,
    ServerEventType.RESPONSE_OUTPUT_ITEM_DONE
}

STREAMING_EVENTS = {
    ServerEventType.RESPONSE_TEXT_DELTA,
    ServerEventType.RESPONSE_AUDIO_DELTA,
    ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA
}


print("Server Event Categories:")
print(f"  Session events: {len(SESSION_EVENTS)}")
print(f"  Audio buffer events: {len(AUDIO_BUFFER_EVENTS)}")
print(f"  Response events: {len(RESPONSE_EVENTS)}")
print(f"  Streaming events: {len(STREAMING_EVENTS)}")
```

---

## Event Handler Pattern

### Event Router

```python
class EventRouter:
    """Route server events to handlers."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._wildcard_handlers: List[Callable] = []
        self._category_handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event_type: str, handler: Callable):
        """Register handler for specific event type."""
        
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def on_all(self, handler: Callable):
        """Register handler for all events."""
        self._wildcard_handlers.append(handler)
    
    def on_category(self, category: str, handler: Callable):
        """Register handler for event category."""
        
        if category not in self._category_handlers:
            self._category_handlers[category] = []
        self._category_handlers[category].append(handler)
    
    def _get_category(self, event_type: str) -> Optional[str]:
        """Get category from event type."""
        
        if event_type.startswith("session."):
            return "session"
        elif event_type.startswith("conversation."):
            return "conversation"
        elif event_type.startswith("input_audio_buffer."):
            return "audio_buffer"
        elif event_type.startswith("response."):
            return "response"
        elif event_type.startswith("rate_limits."):
            return "rate_limits"
        elif event_type == "error":
            return "error"
        
        return None
    
    async def route(self, event: ServerEvent):
        """Route event to appropriate handlers."""
        
        # Call wildcard handlers first
        for handler in self._wildcard_handlers:
            await self._call_handler(handler, event)
        
        # Call category handlers
        category = self._get_category(event.type)
        if category and category in self._category_handlers:
            for handler in self._category_handlers[category]:
                await self._call_handler(handler, event)
        
        # Call specific handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                await self._call_handler(handler, event)
    
    async def _call_handler(self, handler: Callable, event: ServerEvent):
        """Call handler (sync or async)."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Handler error: {e}")


# Usage
router = EventRouter()

# Register specific handler
router.on("session.created", lambda e: print(f"Session created: {e.event_id}"))

# Register category handler
router.on_category("response", lambda e: print(f"Response event: {e.type}"))

# Register wildcard handler
router.on_all(lambda e: print(f"Event: {e.type}"))

# Route an event
# await router.route(event)

print("Event Router configured")
print(f"  Specific handlers: {len(router._handlers)}")
print(f"  Category handlers: {len(router._category_handlers)}")
print(f"  Wildcard handlers: {len(router._wildcard_handlers)}")
```

### Streaming Event Handler

```python
class StreamingEventHandler:
    """Handle streaming delta events."""
    
    def __init__(self):
        self.current_text = ""
        self.current_audio_transcript = ""
        self.audio_chunks: List[str] = []
        
        self._text_callbacks: List[Callable] = []
        self._audio_callbacks: List[Callable] = []
        self._transcript_callbacks: List[Callable] = []
    
    def on_text_delta(self, callback: Callable):
        """Register text delta callback."""
        self._text_callbacks.append(callback)
    
    def on_audio_delta(self, callback: Callable):
        """Register audio delta callback."""
        self._audio_callbacks.append(callback)
    
    def on_transcript_delta(self, callback: Callable):
        """Register transcript delta callback."""
        self._transcript_callbacks.append(callback)
    
    async def handle_event(self, event: ServerEvent):
        """Handle streaming event."""
        
        if event.type == "response.text.delta":
            await self._handle_text_delta(event)
        elif event.type == "response.audio.delta":
            await self._handle_audio_delta(event)
        elif event.type == "response.audio_transcript.delta":
            await self._handle_transcript_delta(event)
        elif event.type in ["response.text.done", "response.audio.done"]:
            await self._handle_stream_done(event)
    
    async def _handle_text_delta(self, event: ServerEvent):
        """Handle text delta."""
        
        delta = event.get("delta", "")
        self.current_text += delta
        
        for callback in self._text_callbacks:
            await self._call_callback(callback, delta, self.current_text)
    
    async def _handle_audio_delta(self, event: ServerEvent):
        """Handle audio delta."""
        
        delta = event.get("delta", "")
        self.audio_chunks.append(delta)
        
        for callback in self._audio_callbacks:
            await self._call_callback(callback, delta, len(self.audio_chunks))
    
    async def _handle_transcript_delta(self, event: ServerEvent):
        """Handle audio transcript delta."""
        
        delta = event.get("delta", "")
        self.current_audio_transcript += delta
        
        for callback in self._transcript_callbacks:
            await self._call_callback(
                callback, delta, self.current_audio_transcript
            )
    
    async def _handle_stream_done(self, event: ServerEvent):
        """Handle stream completion."""
        
        if event.type == "response.text.done":
            print(f"Text complete: {len(self.current_text)} chars")
        elif event.type == "response.audio.done":
            print(f"Audio complete: {len(self.audio_chunks)} chunks")
    
    async def _call_callback(self, callback: Callable, *args):
        """Call callback safely."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Callback error: {e}")
    
    def reset(self):
        """Reset state for new response."""
        self.current_text = ""
        self.current_audio_transcript = ""
        self.audio_chunks = []


# Usage
streaming = StreamingEventHandler()

# Register callbacks
streaming.on_text_delta(lambda d, t: print(f"Text: +{len(d)} chars"))
streaming.on_audio_delta(lambda d, n: print(f"Audio chunk {n}"))
streaming.on_transcript_delta(lambda d, t: print(f"Transcript: {t}"))

print("Streaming handler configured")
```

---

## Conversation Item Events

### Managing Conversation Items

```python
@dataclass
class ConversationItem:
    """Represents a conversation item."""
    
    id: str
    object: str
    type: str
    status: str
    role: Optional[str] = None
    content: List[dict] = field(default_factory=list)
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_event(cls, event_data: dict) -> 'ConversationItem':
        """Create from event data."""
        item = event_data.get("item", {})
        return cls(
            id=item.get("id", ""),
            object=item.get("object", "realtime.item"),
            type=item.get("type", ""),
            status=item.get("status", "completed"),
            role=item.get("role"),
            content=item.get("content", []),
            call_id=item.get("call_id"),
            name=item.get("name"),
            arguments=item.get("arguments"),
            output=item.get("output")
        )


class ConversationManager:
    """Manage conversation items."""
    
    def __init__(self):
        self.items: Dict[str, ConversationItem] = {}
        self.item_order: List[str] = []
        
        self._on_item_added: Optional[Callable] = None
        self._on_item_updated: Optional[Callable] = None
        self._on_item_deleted: Optional[Callable] = None
    
    def on_item_added(self, callback: Callable):
        """Register item added callback."""
        self._on_item_added = callback
    
    def on_item_updated(self, callback: Callable):
        """Register item updated callback."""
        self._on_item_updated = callback
    
    def on_item_deleted(self, callback: Callable):
        """Register item deleted callback."""
        self._on_item_deleted = callback
    
    async def handle_event(self, event: ServerEvent):
        """Handle conversation item event."""
        
        if event.type == "conversation.item.created":
            await self._handle_item_created(event)
        elif event.type == "conversation.item.truncated":
            await self._handle_item_truncated(event)
        elif event.type == "conversation.item.deleted":
            await self._handle_item_deleted(event)
        elif event.type == "conversation.item.input_audio_transcription.completed":
            await self._handle_transcription_completed(event)
    
    async def _handle_item_created(self, event: ServerEvent):
        """Handle item creation."""
        
        item = ConversationItem.from_event(event.data)
        self.items[item.id] = item
        self.item_order.append(item.id)
        
        if self._on_item_added:
            await self._call_callback(self._on_item_added, item)
    
    async def _handle_item_truncated(self, event: ServerEvent):
        """Handle item truncation."""
        
        item_id = event.get("item_id")
        if item_id in self.items:
            # Update content based on truncation
            audio_end_ms = event.get("audio_end_ms", 0)
            self.items[item_id].status = "truncated"
            
            if self._on_item_updated:
                await self._call_callback(
                    self._on_item_updated, 
                    self.items[item_id]
                )
    
    async def _handle_item_deleted(self, event: ServerEvent):
        """Handle item deletion."""
        
        item_id = event.get("item_id")
        if item_id in self.items:
            item = self.items.pop(item_id)
            self.item_order.remove(item_id)
            
            if self._on_item_deleted:
                await self._call_callback(self._on_item_deleted, item)
    
    async def _handle_transcription_completed(self, event: ServerEvent):
        """Handle audio transcription completion."""
        
        item_id = event.get("item_id")
        transcript = event.get("transcript", "")
        
        if item_id in self.items:
            # Update item with transcription
            item = self.items[item_id]
            for content in item.content:
                if content.get("type") == "input_audio":
                    content["transcript"] = transcript
            
            if self._on_item_updated:
                await self._call_callback(self._on_item_updated, item)
    
    async def _call_callback(self, callback: Callable, *args):
        """Call callback safely."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Callback error: {e}")
    
    def get_messages(self) -> List[ConversationItem]:
        """Get all message items."""
        return [
            self.items[id] for id in self.item_order
            if self.items[id].type == "message"
        ]
    
    def get_transcript(self) -> str:
        """Get conversation transcript."""
        
        lines = []
        for id in self.item_order:
            item = self.items[id]
            if item.type == "message":
                role = item.role or "unknown"
                for content in item.content:
                    if content.get("type") in ["text", "input_text"]:
                        lines.append(f"{role}: {content.get('text', '')}")
                    elif content.get("type") == "input_audio":
                        transcript = content.get("transcript", "[audio]")
                        lines.append(f"{role}: {transcript}")
        
        return "\n".join(lines)


# Usage
conversation = ConversationManager()

# Register callbacks
conversation.on_item_added(
    lambda item: print(f"Item added: {item.id} ({item.type})")
)

print("Conversation Manager configured")
print(f"Items: {len(conversation.items)}")
```

---

## Error Events

### Error Handling

```python
class ErrorType(Enum):
    """Error types from Realtime API."""
    
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_error"
    SERVER = "server_error"
    TIMEOUT = "timeout_error"
    CONNECTION = "connection_error"
    UNKNOWN = "unknown_error"


@dataclass
class RealtimeError:
    """Parsed error from server."""
    
    type: ErrorType
    code: str
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_event(cls, event: ServerEvent) -> 'RealtimeError':
        """Create from error event."""
        
        error_data = event.get("error", {})
        error_type = error_data.get("type", "unknown_error")
        
        try:
            type_enum = ErrorType(error_type)
        except ValueError:
            type_enum = ErrorType.UNKNOWN
        
        return cls(
            type=type_enum,
            code=error_data.get("code", "unknown"),
            message=error_data.get("message", "Unknown error"),
            param=error_data.get("param"),
            event_id=event.event_id
        )
    
    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable."""
        return self.type in [
            ErrorType.RATE_LIMIT,
            ErrorType.SERVER,
            ErrorType.TIMEOUT,
            ErrorType.CONNECTION
        ]


class ErrorHandler:
    """Handle Realtime API errors."""
    
    def __init__(self):
        self.errors: List[RealtimeError] = []
        self.max_retries = 3
        self.retry_delay_seconds = 1.0
        
        self._error_callbacks: Dict[ErrorType, List[Callable]] = {}
        self._global_callback: Optional[Callable] = None
    
    def on_error(self, error_type: ErrorType, callback: Callable):
        """Register callback for specific error type."""
        
        if error_type not in self._error_callbacks:
            self._error_callbacks[error_type] = []
        self._error_callbacks[error_type].append(callback)
    
    def on_any_error(self, callback: Callable):
        """Register callback for any error."""
        self._global_callback = callback
    
    async def handle_error(self, event: ServerEvent) -> Optional[dict]:
        """Handle error event, return retry action if applicable."""
        
        error = RealtimeError.from_event(event)
        self.errors.append(error)
        
        # Call global callback
        if self._global_callback:
            await self._call_callback(self._global_callback, error)
        
        # Call type-specific callbacks
        if error.type in self._error_callbacks:
            for callback in self._error_callbacks[error.type]:
                await self._call_callback(callback, error)
        
        # Determine retry action
        if error.is_retryable:
            return self._get_retry_action(error)
        
        return None
    
    def _get_retry_action(self, error: RealtimeError) -> Optional[dict]:
        """Get retry action for retryable error."""
        
        recent_same_errors = [
            e for e in self.errors[-10:]
            if e.type == error.type
        ]
        
        if len(recent_same_errors) >= self.max_retries:
            return None  # Max retries exceeded
        
        delay = self.retry_delay_seconds * (2 ** (len(recent_same_errors) - 1))
        
        return {
            "action": "retry",
            "delay_seconds": delay,
            "attempt": len(recent_same_errors),
            "max_attempts": self.max_retries
        }
    
    async def _call_callback(self, callback: Callable, error: RealtimeError):
        """Call callback safely."""
        try:
            result = callback(error)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Error callback failed: {e}")
    
    def get_error_summary(self) -> dict:
        """Get error summary."""
        
        by_type = {}
        for error in self.errors:
            type_name = error.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "by_type": by_type,
            "retryable": sum(1 for e in self.errors if e.is_retryable)
        }


# Usage
error_handler = ErrorHandler()

# Register handlers
error_handler.on_error(
    ErrorType.RATE_LIMIT,
    lambda e: print(f"Rate limit: {e.message}")
)
error_handler.on_any_error(
    lambda e: print(f"Error: {e.type.value} - {e.message}")
)

print("Error Handler configured")
print(f"Max retries: {error_handler.max_retries}")
```

---

## Complete Event System

### Integrated Event Manager

```python
class RealtimeEventManager:
    """Complete event management system."""
    
    def __init__(self):
        # Component managers
        self.router = EventRouter()
        self.streaming = StreamingEventHandler()
        self.conversation = ConversationManager()
        self.error_handler = ErrorHandler()
        
        # Event log
        self.event_log: List[ServerEvent] = []
        self.max_log_size = 1000
        
        # Statistics
        self.event_counts: Dict[str, int] = {}
        
        # Setup routing
        self._setup_routing()
    
    def _setup_routing(self):
        """Setup internal event routing."""
        
        # Log all events
        self.router.on_all(self._log_event)
        
        # Route to appropriate handlers
        self.router.on_category("response", self._route_response_event)
        self.router.on_category("conversation", self._route_conversation_event)
        self.router.on("error", self._route_error_event)
    
    async def _log_event(self, event: ServerEvent):
        """Log event."""
        
        self.event_log.append(event)
        
        # Trim log if needed
        if len(self.event_log) > self.max_log_size:
            self.event_log = self.event_log[-self.max_log_size:]
        
        # Update counts
        self.event_counts[event.type] = (
            self.event_counts.get(event.type, 0) + 1
        )
    
    async def _route_response_event(self, event: ServerEvent):
        """Route response events."""
        
        # Handle streaming events
        if event.type in [
            "response.text.delta",
            "response.audio.delta",
            "response.audio_transcript.delta",
            "response.text.done",
            "response.audio.done"
        ]:
            await self.streaming.handle_event(event)
        
        # Reset streaming on new response
        if event.type == "response.created":
            self.streaming.reset()
    
    async def _route_conversation_event(self, event: ServerEvent):
        """Route conversation events."""
        await self.conversation.handle_event(event)
    
    async def _route_error_event(self, event: ServerEvent):
        """Route error events."""
        await self.error_handler.handle_error(event)
    
    async def process_event(self, json_data: str):
        """Process incoming event JSON."""
        
        event = ServerEvent.from_json(json_data)
        await self.router.route(event)
        
        return event
    
    def get_statistics(self) -> dict:
        """Get event statistics."""
        
        return {
            "total_events": len(self.event_log),
            "event_counts": dict(sorted(
                self.event_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "conversation_items": len(self.conversation.items),
            "errors": self.error_handler.get_error_summary()
        }
    
    def get_recent_events(self, count: int = 10) -> List[dict]:
        """Get recent events."""
        
        return [
            {"type": e.type, "event_id": e.event_id}
            for e in self.event_log[-count:]
        ]


# Usage
event_manager = RealtimeEventManager()

# Register custom handlers
event_manager.router.on(
    "session.created",
    lambda e: print(f"Session: {e.get('session', {}).get('id')}")
)

print("Event Manager configured")

# Simulate processing events
sample_events = [
    '{"type": "session.created", "event_id": "evt_1", "session": {"id": "sess_123"}}',
    '{"type": "response.text.delta", "event_id": "evt_2", "delta": "Hello"}',
    '{"type": "response.text.delta", "event_id": "evt_3", "delta": " World"}'
]

# Would process:
# for event_json in sample_events:
#     await event_manager.process_event(event_json)

# Get stats
stats = event_manager.get_statistics()
print(f"\nStatistics: {json.dumps(stats, indent=2)}")
```

---

## Hands-on Exercise

### Your Task

Build a complete event processing system with logging and analysis.

### Requirements

1. Process all event types
2. Track event sequence
3. Handle errors with retry logic
4. Generate event analytics
5. Support event replay

<details>
<summary>💡 Hints</summary>

- Use event IDs for ordering
- Store raw JSON for replay
- Calculate event timing metrics
</details>

<details>
<summary>✅ Solution</summary>

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import asyncio


class ComprehensiveEventSystem:
    """Complete event processing with analytics."""
    
    def __init__(self):
        # Core components
        self.router = EventRouter()
        self.streaming = StreamingEventHandler()
        self.conversation = ConversationManager()
        self.error_handler = ErrorHandler()
        self.event_builder = ClientEventBuilder()
        
        # Storage
        self.raw_events: List[str] = []  # For replay
        self.parsed_events: List[ServerEvent] = []
        self.max_storage = 5000
        
        # Analytics
        self.event_timing: Dict[str, List[float]] = {}
        self.response_times: List[float] = []
        self.last_event_time: Optional[datetime] = None
        
        # Sequence tracking
        self.event_sequence: List[str] = []
        self.expected_sequences = {
            "response_flow": [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.audio.delta",
                "response.audio_transcript.delta",
                "response.content_part.done",
                "response.output_item.done",
                "response.done"
            ]
        }
        
        self._setup()
    
    def _setup(self):
        """Setup event handlers."""
        
        # Track all events
        self.router.on_all(self._track_event)
        
        # Route to handlers
        self.router.on_category("response", self._handle_response)
        self.router.on_category("conversation", self._handle_conversation)
        self.router.on("error", self._handle_error)
    
    async def _track_event(self, event: ServerEvent):
        """Track event for analytics."""
        
        now = datetime.now()
        
        # Calculate timing
        if self.last_event_time:
            delta_ms = (now - self.last_event_time).total_seconds() * 1000
            
            if event.type not in self.event_timing:
                self.event_timing[event.type] = []
            self.event_timing[event.type].append(delta_ms)
        
        self.last_event_time = now
        
        # Track sequence
        self.event_sequence.append(event.type)
        
        # Store parsed
        self.parsed_events.append(event)
        if len(self.parsed_events) > self.max_storage:
            self.parsed_events = self.parsed_events[-self.max_storage:]
    
    async def _handle_response(self, event: ServerEvent):
        """Handle response events."""
        
        await self.streaming.handle_event(event)
        
        if event.type == "response.created":
            self._start_response_timer()
            self.streaming.reset()
        elif event.type == "response.done":
            self._end_response_timer()
    
    async def _handle_conversation(self, event: ServerEvent):
        """Handle conversation events."""
        await self.conversation.handle_event(event)
    
    async def _handle_error(self, event: ServerEvent):
        """Handle error events."""
        
        retry_action = await self.error_handler.handle_error(event)
        
        if retry_action:
            print(f"Retry scheduled: {retry_action}")
    
    def _start_response_timer(self):
        """Start response timing."""
        self._response_start = datetime.now()
    
    def _end_response_timer(self):
        """End response timing."""
        if hasattr(self, '_response_start'):
            duration = (datetime.now() - self._response_start).total_seconds()
            self.response_times.append(duration)
    
    async def process_raw(self, json_str: str) -> ServerEvent:
        """Process raw JSON event."""
        
        # Store raw for replay
        self.raw_events.append(json_str)
        if len(self.raw_events) > self.max_storage:
            self.raw_events = self.raw_events[-self.max_storage:]
        
        # Parse and route
        event = ServerEvent.from_json(json_str)
        await self.router.route(event)
        
        return event
    
    async def replay_events(
        self,
        start_idx: int = 0,
        end_idx: int = None,
        speed: float = 1.0
    ):
        """Replay stored events."""
        
        events = self.raw_events[start_idx:end_idx]
        
        for i, json_str in enumerate(events):
            # Process event
            event = await self.process_raw(json_str)
            print(f"Replayed [{i+1}/{len(events)}]: {event.type}")
            
            # Delay based on speed
            if speed > 0 and i < len(events) - 1:
                await asyncio.sleep(0.1 / speed)
    
    def check_sequence(self, sequence_name: str) -> dict:
        """Check if event sequence matches expected."""
        
        expected = self.expected_sequences.get(sequence_name, [])
        actual = self.event_sequence[-len(expected):]
        
        matches = []
        mismatches = []
        
        for i, (exp, act) in enumerate(zip(expected, actual)):
            if exp == act:
                matches.append((i, exp))
            else:
                mismatches.append((i, exp, act))
        
        return {
            "sequence": sequence_name,
            "match_rate": len(matches) / len(expected) if expected else 0,
            "matches": len(matches),
            "mismatches": mismatches
        }
    
    def get_analytics(self) -> dict:
        """Get comprehensive analytics."""
        
        # Event counts
        event_counts = {}
        for event in self.parsed_events:
            event_counts[event.type] = event_counts.get(event.type, 0) + 1
        
        # Timing stats
        timing_stats = {}
        for event_type, times in self.event_timing.items():
            if times:
                timing_stats[event_type] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
        
        # Response stats
        response_stats = {}
        if self.response_times:
            response_stats = {
                "count": len(self.response_times),
                "avg_seconds": sum(self.response_times) / len(self.response_times),
                "min_seconds": min(self.response_times),
                "max_seconds": max(self.response_times)
            }
        
        return {
            "total_events": len(self.parsed_events),
            "event_counts": dict(sorted(
                event_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "timing_stats": timing_stats,
            "response_stats": response_stats,
            "conversation": {
                "items": len(self.conversation.items),
                "transcript_length": len(self.conversation.get_transcript())
            },
            "errors": self.error_handler.get_error_summary()
        }
    
    def get_event_flow(self, last_n: int = 20) -> List[str]:
        """Get recent event flow."""
        return self.event_sequence[-last_n:]
    
    def export_session(self) -> str:
        """Export session data for analysis."""
        
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "analytics": self.get_analytics(),
            "event_flow": self.get_event_flow(50),
            "conversation_transcript": self.conversation.get_transcript(),
            "raw_event_count": len(self.raw_events)
        }, indent=2)


# Usage
system = ComprehensiveEventSystem()

print("Comprehensive Event System")
print("=" * 50)

# Simulate events
sample_events = [
    {"type": "session.created", "event_id": "e1", "session": {"id": "s1"}},
    {"type": "response.created", "event_id": "e2", "response": {"id": "r1"}},
    {"type": "response.output_item.added", "event_id": "e3"},
    {"type": "response.audio.delta", "event_id": "e4", "delta": "audio_data"},
    {"type": "response.audio_transcript.delta", "event_id": "e5", "delta": "Hello"},
    {"type": "response.done", "event_id": "e6"}
]

# Would process:
# for event in sample_events:
#     await system.process_raw(json.dumps(event))

# Get analytics
analytics = system.get_analytics()
print(f"\nAnalytics:")
print(f"  Total events: {analytics['total_events']}")
print(f"  Conversation items: {analytics['conversation']['items']}")

# Check sequence
# sequence_check = system.check_sequence("response_flow")
# print(f"\nSequence check: {sequence_check['match_rate']:.0%} match")

# Export
export = system.export_session()
print(f"\nExport size: {len(export)} bytes")
```

</details>

---

## Summary

✅ Client events control session and input  
✅ Server events deliver responses and status  
✅ Event routing enables modular handling  
✅ Streaming handlers accumulate deltas  
✅ Error events support retry logic

**Next:** [Voice Best Practices](./07-voice-best-practices.md)

---

## Further Reading

- [Client Events Reference](https://platform.openai.com/docs/api-reference/realtime-client-events) — All client event types
- [Server Events Reference](https://platform.openai.com/docs/api-reference/realtime-server-events) — All server event types
- [Event Examples](https://platform.openai.com/docs/guides/realtime#events) — Event flow examples
