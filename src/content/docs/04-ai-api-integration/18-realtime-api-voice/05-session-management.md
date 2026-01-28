---
title: "Session Management"
---

# Session Management

## Introduction

Realtime API sessions require careful lifecycle management. You'll handle initialization, turn-taking, interruptions, and graceful termination.

### What We'll Cover

- Session lifecycle events
- Turn-taking patterns
- Handling user interruptions
- Session state persistence
- Timeout and cleanup

### Prerequisites

- Understanding of Realtime API connections
- WebSocket or WebRTC experience
- Async programming patterns

---

## Session Lifecycle

### Session States

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import asyncio


class SessionState(Enum):
    """Session lifecycle states."""
    
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    CONFIGURING = "configuring"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class SessionMetrics:
    """Track session metrics."""
    
    created_at: datetime = field(default_factory=datetime.now)
    connected_at: Optional[datetime] = None
    ready_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    interruptions: int = 0
    errors: int = 0
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_audio_ms: int = 0
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration."""
        end = self.closed_at or datetime.now()
        return (end - self.created_at).total_seconds()
    
    @property
    def time_to_ready_ms(self) -> Optional[float]:
        """Get time from creation to ready state."""
        if self.ready_at:
            return (self.ready_at - self.created_at).total_seconds() * 1000
        return None


@dataclass
class SessionConfig:
    """Session configuration."""
    
    model: str = "gpt-4o-realtime-preview"
    voice: str = "alloy"
    instructions: str = ""
    tools: List[Dict] = field(default_factory=list)
    
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    
    turn_detection: Optional[Dict] = None
    temperature: float = 0.8
    max_response_output_tokens: Optional[int] = None
    
    timeout_seconds: int = 600  # 10 minutes
    idle_timeout_seconds: int = 60


class SessionLifecycleManager:
    """Manage session lifecycle."""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.state = SessionState.INITIALIZING
        self.metrics = SessionMetrics()
        
        # Event handlers
        self._state_handlers: Dict[SessionState, List[Callable]] = {}
        self._transition_handlers: List[Callable] = []
        
        # State tracking
        self._previous_state: Optional[SessionState] = None
        self._state_history: List[tuple] = []
    
    def on_state(self, state: SessionState, handler: Callable):
        """Register handler for specific state."""
        
        if state not in self._state_handlers:
            self._state_handlers[state] = []
        self._state_handlers[state].append(handler)
    
    def on_transition(self, handler: Callable):
        """Register handler for any state transition."""
        self._transition_handlers.append(handler)
    
    async def transition_to(self, new_state: SessionState):
        """Transition to new state."""
        
        old_state = self.state
        self._previous_state = old_state
        self.state = new_state
        
        # Record transition
        self._state_history.append((
            datetime.now(),
            old_state,
            new_state
        ))
        
        # Update metrics
        self.metrics.last_activity = datetime.now()
        
        if new_state == SessionState.CONNECTED:
            self.metrics.connected_at = datetime.now()
        elif new_state == SessionState.READY:
            self.metrics.ready_at = datetime.now()
        elif new_state == SessionState.CLOSED:
            self.metrics.closed_at = datetime.now()
        elif new_state == SessionState.ERROR:
            self.metrics.errors += 1
        elif new_state == SessionState.INTERRUPTED:
            self.metrics.interruptions += 1
        
        # Notify handlers
        for handler in self._transition_handlers:
            await self._call_handler(handler, old_state, new_state)
        
        if new_state in self._state_handlers:
            for handler in self._state_handlers[new_state]:
                await self._call_handler(handler, old_state, new_state)
    
    async def _call_handler(self, handler: Callable, *args):
        """Call handler (sync or async)."""
        result = handler(*args)
        if asyncio.iscoroutine(result):
            await result
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state not in [
            SessionState.CLOSED,
            SessionState.ERROR
        ]
    
    def can_accept_input(self) -> bool:
        """Check if session can accept user input."""
        return self.state in [
            SessionState.READY,
            SessionState.LISTENING,
            SessionState.RESPONDING  # For interrupts
        ]
    
    def get_state_duration(self) -> float:
        """Get duration of current state."""
        if self._state_history:
            last_transition = self._state_history[-1][0]
            return (datetime.now() - last_transition).total_seconds()
        return 0.0


# Example usage
config = SessionConfig(
    model="gpt-4o-realtime-preview",
    voice="coral",
    instructions="You are a helpful assistant.",
    timeout_seconds=300
)

lifecycle = SessionLifecycleManager(config)

print(f"Initial state: {lifecycle.state.value}")
print(f"Can accept input: {lifecycle.can_accept_input()}")
```

### Session Initialization

```python
class RealtimeSession:
    """Complete session with lifecycle management."""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.lifecycle = SessionLifecycleManager(config)
        self.session_id: Optional[str] = None
        
        # Setup lifecycle handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup lifecycle event handlers."""
        
        self.lifecycle.on_state(
            SessionState.CONNECTED,
            self._on_connected
        )
        self.lifecycle.on_state(
            SessionState.ERROR,
            self._on_error
        )
    
    async def _on_connected(self, old_state, new_state):
        """Handle connection established."""
        print(f"Session connected, configuring...")
        await self._configure_session()
    
    async def _on_error(self, old_state, new_state):
        """Handle error state."""
        print(f"Session error from {old_state.value}")
    
    async def _configure_session(self):
        """Send session configuration."""
        
        await self.lifecycle.transition_to(SessionState.CONFIGURING)
        
        config_event = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.config.voice,
                "instructions": self.config.instructions,
                "input_audio_format": self.config.input_audio_format,
                "output_audio_format": self.config.output_audio_format,
                "tools": self.config.tools,
                "temperature": self.config.temperature
            }
        }
        
        if self.config.turn_detection:
            config_event["session"]["turn_detection"] = self.config.turn_detection
        
        if self.config.max_response_output_tokens:
            config_event["session"]["max_response_output_tokens"] = (
                self.config.max_response_output_tokens
            )
        
        # Send configuration (implement in subclass)
        await self._send_event(config_event)
    
    async def _send_event(self, event: dict):
        """Send event to server (override in subclass)."""
        print(f"Would send: {event['type']}")
    
    async def start(self):
        """Start the session."""
        
        # Connect to server
        await self.lifecycle.transition_to(SessionState.CONNECTED)
        
        # Wait for ready
        await self.lifecycle.transition_to(SessionState.READY)
    
    async def close(self):
        """Close the session gracefully."""
        
        await self.lifecycle.transition_to(SessionState.CLOSING)
        
        # Cleanup
        await self._cleanup()
        
        await self.lifecycle.transition_to(SessionState.CLOSED)
    
    async def _cleanup(self):
        """Perform cleanup tasks."""
        print("Cleaning up session resources")


# Usage
session = RealtimeSession(config)

# Check lifecycle
print(f"State: {session.lifecycle.state.value}")
print(f"Active: {session.lifecycle.is_active()}")
```

---

## Turn-Taking Patterns

### Understanding Turns

```python
class TurnType(Enum):
    """Types of conversation turns."""
    
    USER_AUDIO = "user_audio"
    USER_TEXT = "user_text"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class ConversationTurn:
    """Represents a conversation turn."""
    
    id: str
    type: TurnType
    started_at: datetime
    completed_at: Optional[datetime] = None
    content: Optional[str] = None
    audio_duration_ms: Optional[int] = None
    interrupted: bool = False
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get turn duration in milliseconds."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None


class TurnManager:
    """Manage conversation turns."""
    
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.current_turn: Optional[ConversationTurn] = None
        self._turn_counter = 0
    
    def start_turn(self, turn_type: TurnType) -> ConversationTurn:
        """Start a new turn."""
        
        self._turn_counter += 1
        
        turn = ConversationTurn(
            id=f"turn_{self._turn_counter}",
            type=turn_type,
            started_at=datetime.now()
        )
        
        self.current_turn = turn
        return turn
    
    def complete_turn(
        self,
        content: str = None,
        audio_duration_ms: int = None
    ) -> ConversationTurn:
        """Complete current turn."""
        
        if not self.current_turn:
            raise ValueError("No active turn")
        
        self.current_turn.completed_at = datetime.now()
        self.current_turn.content = content
        self.current_turn.audio_duration_ms = audio_duration_ms
        
        self.turns.append(self.current_turn)
        completed = self.current_turn
        self.current_turn = None
        
        return completed
    
    def interrupt_turn(self) -> Optional[ConversationTurn]:
        """Interrupt current turn."""
        
        if not self.current_turn:
            return None
        
        self.current_turn.completed_at = datetime.now()
        self.current_turn.interrupted = True
        
        self.turns.append(self.current_turn)
        interrupted = self.current_turn
        self.current_turn = None
        
        return interrupted
    
    def get_last_turn(self, turn_type: TurnType = None) -> Optional[ConversationTurn]:
        """Get last turn, optionally filtered by type."""
        
        for turn in reversed(self.turns):
            if turn_type is None or turn.type == turn_type:
                return turn
        return None
    
    def get_turn_stats(self) -> dict:
        """Get turn statistics."""
        
        user_turns = [t for t in self.turns if t.type in [
            TurnType.USER_AUDIO, TurnType.USER_TEXT
        ]]
        assistant_turns = [t for t in self.turns if t.type == TurnType.ASSISTANT]
        interrupted = [t for t in self.turns if t.interrupted]
        
        return {
            "total_turns": len(self.turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "interrupted_turns": len(interrupted),
            "interrupt_rate": len(interrupted) / len(self.turns) if self.turns else 0
        }


# Example
turn_manager = TurnManager()

# Start user turn
user_turn = turn_manager.start_turn(TurnType.USER_AUDIO)
print(f"Started turn: {user_turn.id}")

# Complete user turn
turn_manager.complete_turn(
    content="Hello, how are you?",
    audio_duration_ms=2500
)

# Start assistant turn
assistant_turn = turn_manager.start_turn(TurnType.ASSISTANT)

# Complete assistant turn
turn_manager.complete_turn(
    content="I'm doing well, thank you!",
    audio_duration_ms=3000
)

# Get stats
stats = turn_manager.get_turn_stats()
print(f"Turn stats: {stats}")
```

### Server VAD Turn Detection

```python
@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    
    type: str = "server_vad"
    threshold: float = 0.5  # 0.0 to 1.0
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True


class ServerVADManager:
    """Manage server-side VAD."""
    
    def __init__(self, config: VADConfig = None):
        self.config = config or VADConfig()
        
        # State tracking
        self.is_speaking = False
        self.speech_start: Optional[datetime] = None
        self.last_speech_end: Optional[datetime] = None
        
        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
    
    def get_config_event(self) -> dict:
        """Get turn detection configuration for session.update."""
        
        return {
            "type": self.config.type,
            "threshold": self.config.threshold,
            "prefix_padding_ms": self.config.prefix_padding_ms,
            "silence_duration_ms": self.config.silence_duration_ms,
            "create_response": self.config.create_response
        }
    
    def on_speech_start(self, handler: Callable):
        """Register speech start handler."""
        self._on_speech_start = handler
    
    def on_speech_end(self, handler: Callable):
        """Register speech end handler."""
        self._on_speech_end = handler
    
    async def handle_event(self, event: dict):
        """Handle VAD-related events."""
        
        event_type = event.get("type", "")
        
        if event_type == "input_audio_buffer.speech_started":
            await self._handle_speech_started(event)
        elif event_type == "input_audio_buffer.speech_stopped":
            await self._handle_speech_stopped(event)
    
    async def _handle_speech_started(self, event: dict):
        """Handle speech start event."""
        
        self.is_speaking = True
        self.speech_start = datetime.now()
        
        if self._on_speech_start:
            result = self._on_speech_start(event)
            if asyncio.iscoroutine(result):
                await result
    
    async def _handle_speech_stopped(self, event: dict):
        """Handle speech stop event."""
        
        self.is_speaking = False
        self.last_speech_end = datetime.now()
        
        if self._on_speech_end:
            result = self._on_speech_end(event)
            if asyncio.iscoroutine(result):
                await result
    
    @property
    def speech_duration_ms(self) -> Optional[float]:
        """Get current/last speech duration."""
        
        if not self.speech_start:
            return None
        
        end = self.last_speech_end or datetime.now()
        return (end - self.speech_start).total_seconds() * 1000


# Example
vad = ServerVADManager(VADConfig(
    threshold=0.5,
    silence_duration_ms=700
))

# Register handlers
vad.on_speech_start(lambda e: print("User started speaking"))
vad.on_speech_end(lambda e: print("User stopped speaking"))

# Get config for session update
vad_config = vad.get_config_event()
print(f"VAD config: {vad_config}")
```

---

## Handling Interruptions

### Interruption Detection and Response

```python
class InterruptionType(Enum):
    """Types of user interruptions."""
    
    SPEECH_OVERLAP = "speech_overlap"  # User speaks during assistant
    CANCEL_REQUEST = "cancel_request"  # User says "stop" or similar
    MANUAL_CANCEL = "manual_cancel"  # UI button press


@dataclass
class InterruptionEvent:
    """Record of an interruption."""
    
    type: InterruptionType
    timestamp: datetime
    audio_position_ms: int  # Where in response interrupted
    transcript_position: int  # Characters of response spoken
    user_input: Optional[str] = None  # What user said


class InterruptionHandler:
    """Handle conversation interruptions."""
    
    def __init__(self):
        self.interruptions: List[InterruptionEvent] = []
        self.allow_interruptions = True
        
        # Callbacks
        self._on_interrupt: Optional[Callable] = None
    
    def on_interrupt(self, handler: Callable):
        """Register interruption handler."""
        self._on_interrupt = handler
    
    async def handle_speech_started_during_response(
        self,
        audio_position_ms: int,
        transcript_position: int
    ) -> bool:
        """Handle user speaking during assistant response."""
        
        if not self.allow_interruptions:
            return False
        
        event = InterruptionEvent(
            type=InterruptionType.SPEECH_OVERLAP,
            timestamp=datetime.now(),
            audio_position_ms=audio_position_ms,
            transcript_position=transcript_position
        )
        
        self.interruptions.append(event)
        
        # Notify handler
        if self._on_interrupt:
            result = self._on_interrupt(event)
            if asyncio.iscoroutine(result):
                await result
        
        return True
    
    def create_cancel_event(self) -> dict:
        """Create response.cancel event."""
        return {"type": "response.cancel"}
    
    def create_truncate_event(
        self,
        item_id: str,
        audio_end_ms: int
    ) -> dict:
        """Create conversation.item.truncate event."""
        
        return {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }
    
    def get_interruption_stats(self) -> dict:
        """Get interruption statistics."""
        
        if not self.interruptions:
            return {
                "count": 0,
                "avg_position_ms": 0,
                "types": {}
            }
        
        positions = [i.audio_position_ms for i in self.interruptions]
        type_counts = {}
        for i in self.interruptions:
            type_counts[i.type.value] = type_counts.get(i.type.value, 0) + 1
        
        return {
            "count": len(self.interruptions),
            "avg_position_ms": sum(positions) / len(positions),
            "types": type_counts
        }


class InterruptibleResponse:
    """Manage an interruptible assistant response."""
    
    def __init__(
        self,
        item_id: str,
        handler: InterruptionHandler
    ):
        self.item_id = item_id
        self.handler = handler
        
        self.audio_position_ms = 0
        self.transcript = ""
        self.is_complete = False
        self.was_interrupted = False
    
    def update_audio_position(self, position_ms: int):
        """Update current audio playback position."""
        self.audio_position_ms = position_ms
    
    def append_transcript(self, text: str):
        """Append to transcript."""
        self.transcript += text
    
    async def check_for_interruption(
        self,
        speech_started: bool
    ) -> Optional[dict]:
        """Check if response should be interrupted."""
        
        if not speech_started or self.is_complete:
            return None
        
        # Record interruption
        interrupted = await self.handler.handle_speech_started_during_response(
            self.audio_position_ms,
            len(self.transcript)
        )
        
        if interrupted:
            self.was_interrupted = True
            
            # Return truncate event
            return self.handler.create_truncate_event(
                self.item_id,
                self.audio_position_ms
            )
        
        return None
    
    def complete(self):
        """Mark response as complete."""
        self.is_complete = True


# Example
handler = InterruptionHandler()

# Register handler
handler.on_interrupt(lambda e: print(f"Interrupted at {e.audio_position_ms}ms"))

# Create interruptible response
response = InterruptibleResponse("resp_123", handler)

# Simulate playback
response.update_audio_position(1500)
response.append_transcript("Hello, I'm here to help you with")

# Check interruption would be:
# event = await response.check_for_interruption(speech_started=True)

# Get stats
stats = handler.get_interruption_stats()
print(f"Interruption stats: {stats}")
```

---

## Session State Persistence

### Persisting Conversation State

```python
import json
from pathlib import Path


@dataclass
class ConversationItem:
    """A conversation item for persistence."""
    
    id: str
    type: str  # "message", "function_call", "function_call_output"
    role: Optional[str] = None  # "user", "assistant", "system"
    content: Optional[List[Dict]] = None
    status: str = "completed"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "content": self.content,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationItem':
        return cls(**data)


@dataclass
class SessionSnapshot:
    """Snapshot of session state."""
    
    session_id: str
    created_at: str
    config: dict
    conversation_items: List[dict]
    metrics: dict
    
    def to_json(self) -> str:
        return json.dumps({
            "session_id": self.session_id,
            "created_at": self.created_at,
            "config": self.config,
            "conversation_items": self.conversation_items,
            "metrics": self.metrics
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionSnapshot':
        data = json.loads(json_str)
        return cls(**data)


class SessionPersistence:
    """Persist and restore session state."""
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("./session_data")
        self.storage_path.mkdir(exist_ok=True)
    
    def save_snapshot(
        self,
        session_id: str,
        config: SessionConfig,
        items: List[ConversationItem],
        metrics: SessionMetrics
    ) -> Path:
        """Save session snapshot."""
        
        snapshot = SessionSnapshot(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            config={
                "model": config.model,
                "voice": config.voice,
                "instructions": config.instructions,
                "tools": config.tools
            },
            conversation_items=[item.to_dict() for item in items],
            metrics={
                "total_turns": metrics.total_turns,
                "interruptions": metrics.interruptions,
                "duration_seconds": metrics.duration_seconds
            }
        )
        
        file_path = self.storage_path / f"{session_id}.json"
        file_path.write_text(snapshot.to_json())
        
        return file_path
    
    def load_snapshot(self, session_id: str) -> Optional[SessionSnapshot]:
        """Load session snapshot."""
        
        file_path = self.storage_path / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        return SessionSnapshot.from_json(file_path.read_text())
    
    def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        
        return [
            f.stem for f in self.storage_path.glob("*.json")
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete saved session."""
        
        file_path = self.storage_path / f"{session_id}.json"
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class ResumableSession:
    """Session that can be paused and resumed."""
    
    def __init__(
        self,
        config: SessionConfig,
        persistence: SessionPersistence
    ):
        self.config = config
        self.persistence = persistence
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.lifecycle = SessionLifecycleManager(config)
        self.turn_manager = TurnManager()
        self.conversation_items: List[ConversationItem] = []
    
    def add_conversation_item(self, item: ConversationItem):
        """Add item to conversation."""
        self.conversation_items.append(item)
    
    def save_state(self) -> Path:
        """Save current session state."""
        
        return self.persistence.save_snapshot(
            self.session_id,
            self.config,
            self.conversation_items,
            self.lifecycle.metrics
        )
    
    @classmethod
    def resume_from(
        cls,
        session_id: str,
        persistence: SessionPersistence
    ) -> Optional['ResumableSession']:
        """Resume session from saved state."""
        
        snapshot = persistence.load_snapshot(session_id)
        if not snapshot:
            return None
        
        # Reconstruct config
        config = SessionConfig(
            model=snapshot.config["model"],
            voice=snapshot.config["voice"],
            instructions=snapshot.config["instructions"],
            tools=snapshot.config["tools"]
        )
        
        session = cls(config, persistence)
        session.session_id = session_id
        
        # Restore conversation items
        session.conversation_items = [
            ConversationItem.from_dict(item)
            for item in snapshot.conversation_items
        ]
        
        return session
    
    def get_conversation_for_resume(self) -> List[dict]:
        """Get conversation items for resuming session."""
        
        return [
            {
                "type": "conversation.item.create",
                "item": item.to_dict()
            }
            for item in self.conversation_items
        ]


# Example
persistence = SessionPersistence()
session = ResumableSession(config, persistence)

# Add conversation item
item = ConversationItem(
    id="msg_001",
    type="message",
    role="user",
    content=[{"type": "text", "text": "Hello!"}]
)
session.add_conversation_item(item)

# Save state
# path = session.save_state()
# print(f"Saved to: {path}")

# List sessions
sessions = persistence.list_sessions()
print(f"Saved sessions: {sessions}")
```

---

## Timeout and Cleanup

### Session Timeout Management

```python
class TimeoutType(Enum):
    """Types of session timeouts."""
    
    IDLE = "idle"  # No activity
    TOTAL = "total"  # Session duration
    RESPONSE = "response"  # Waiting for response
    CONNECTION = "connection"  # Connection timeout


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    
    idle_timeout_seconds: int = 60
    total_timeout_seconds: int = 600
    response_timeout_seconds: int = 30
    connection_timeout_seconds: int = 10


class SessionTimeoutManager:
    """Manage session timeouts."""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        
        self.last_activity = datetime.now()
        self.session_start = datetime.now()
        
        self._timeout_tasks: Dict[TimeoutType, asyncio.Task] = {}
        self._timeout_callbacks: Dict[TimeoutType, Callable] = {}
    
    def on_timeout(self, timeout_type: TimeoutType, callback: Callable):
        """Register timeout callback."""
        self._timeout_callbacks[timeout_type] = callback
    
    def record_activity(self):
        """Record activity to reset idle timeout."""
        self.last_activity = datetime.now()
    
    def get_idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return (datetime.now() - self.last_activity).total_seconds()
    
    def get_session_seconds(self) -> float:
        """Get total session duration."""
        return (datetime.now() - self.session_start).total_seconds()
    
    async def start_monitoring(self):
        """Start timeout monitoring."""
        
        # Start idle timeout monitor
        self._timeout_tasks[TimeoutType.IDLE] = asyncio.create_task(
            self._monitor_idle_timeout()
        )
        
        # Start total timeout monitor
        self._timeout_tasks[TimeoutType.TOTAL] = asyncio.create_task(
            self._monitor_total_timeout()
        )
    
    async def stop_monitoring(self):
        """Stop all timeout monitoring."""
        
        for task in self._timeout_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._timeout_tasks.clear()
    
    async def _monitor_idle_timeout(self):
        """Monitor for idle timeout."""
        
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            if self.get_idle_seconds() >= self.config.idle_timeout_seconds:
                await self._trigger_timeout(TimeoutType.IDLE)
                break
    
    async def _monitor_total_timeout(self):
        """Monitor for total session timeout."""
        
        await asyncio.sleep(self.config.total_timeout_seconds)
        await self._trigger_timeout(TimeoutType.TOTAL)
    
    async def start_response_timeout(self) -> asyncio.Task:
        """Start response timeout timer."""
        
        async def timeout_check():
            await asyncio.sleep(self.config.response_timeout_seconds)
            await self._trigger_timeout(TimeoutType.RESPONSE)
        
        task = asyncio.create_task(timeout_check())
        self._timeout_tasks[TimeoutType.RESPONSE] = task
        return task
    
    def cancel_response_timeout(self):
        """Cancel response timeout."""
        
        if TimeoutType.RESPONSE in self._timeout_tasks:
            self._timeout_tasks[TimeoutType.RESPONSE].cancel()
            del self._timeout_tasks[TimeoutType.RESPONSE]
    
    async def _trigger_timeout(self, timeout_type: TimeoutType):
        """Trigger timeout callback."""
        
        if timeout_type in self._timeout_callbacks:
            callback = self._timeout_callbacks[timeout_type]
            result = callback(timeout_type)
            if asyncio.iscoroutine(result):
                await result


class SessionCleanup:
    """Handle session cleanup."""
    
    def __init__(self):
        self._cleanup_tasks: List[Callable] = []
    
    def register_cleanup(self, task: Callable):
        """Register cleanup task."""
        self._cleanup_tasks.append(task)
    
    async def perform_cleanup(self):
        """Perform all cleanup tasks."""
        
        for task in self._cleanup_tasks:
            try:
                result = task()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        self._cleanup_tasks.clear()


# Example
timeout_config = TimeoutConfig(
    idle_timeout_seconds=30,
    total_timeout_seconds=300
)

timeout_manager = SessionTimeoutManager(timeout_config)

# Register handlers
timeout_manager.on_timeout(
    TimeoutType.IDLE,
    lambda t: print(f"Session idle timeout: {t.value}")
)

# Check timeouts
print(f"Idle seconds: {timeout_manager.get_idle_seconds():.1f}")
print(f"Session seconds: {timeout_manager.get_session_seconds():.1f}")
```

---

## Hands-on Exercise

### Your Task

Build a complete session manager with lifecycle, turn-taking, and persistence.

### Requirements

1. Implement full lifecycle management
2. Track conversation turns
3. Handle interruptions
4. Support session persistence
5. Manage timeouts

<details>
<summary>💡 Hints</summary>

- Use state machine pattern for lifecycle
- Store minimal data for persistence
- Cancel timeouts on activity
</details>

<details>
<summary>✅ Solution</summary>

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
import asyncio
import json


class ComprehensiveSessionManager:
    """Complete session manager with all features."""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Components
        self.lifecycle = SessionLifecycleManager(config)
        self.turn_manager = TurnManager()
        self.vad_manager = ServerVADManager()
        self.interrupt_handler = InterruptionHandler()
        self.timeout_manager = SessionTimeoutManager(TimeoutConfig(
            idle_timeout_seconds=config.idle_timeout_seconds,
            total_timeout_seconds=config.timeout_seconds
        ))
        self.persistence = SessionPersistence()
        self.cleanup = SessionCleanup()
        
        # State
        self.conversation_items: List[ConversationItem] = []
        self.current_response: Optional[InterruptibleResponse] = None
        
        # Setup
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup all event handlers."""
        
        # Lifecycle handlers
        self.lifecycle.on_state(
            SessionState.CONNECTED,
            self._on_connected
        )
        self.lifecycle.on_state(
            SessionState.CLOSED,
            self._on_closed
        )
        
        # VAD handlers
        self.vad_manager.on_speech_start(self._on_speech_start)
        self.vad_manager.on_speech_end(self._on_speech_end)
        
        # Timeout handlers
        self.timeout_manager.on_timeout(
            TimeoutType.IDLE,
            self._on_idle_timeout
        )
        self.timeout_manager.on_timeout(
            TimeoutType.TOTAL,
            self._on_total_timeout
        )
        
        # Cleanup tasks
        self.cleanup.register_cleanup(self._cleanup_resources)
    
    async def _on_connected(self, old_state, new_state):
        """Handle connection."""
        print(f"[{self.session_id}] Connected")
        await self.timeout_manager.start_monitoring()
    
    async def _on_closed(self, old_state, new_state):
        """Handle session close."""
        print(f"[{self.session_id}] Closed")
        await self.timeout_manager.stop_monitoring()
        await self.cleanup.perform_cleanup()
    
    async def _on_speech_start(self, event):
        """Handle user speech start."""
        
        self.timeout_manager.record_activity()
        
        # Start user turn
        self.turn_manager.start_turn(TurnType.USER_AUDIO)
        
        # Check for interruption
        if self.current_response and not self.current_response.is_complete:
            event = await self.current_response.check_for_interruption(True)
            if event:
                # Send truncate event
                await self._send_event(event)
                self.lifecycle.metrics.interruptions += 1
    
    async def _on_speech_end(self, event):
        """Handle user speech end."""
        
        self.timeout_manager.record_activity()
        
        # Complete user turn
        if self.turn_manager.current_turn:
            self.turn_manager.complete_turn()
    
    async def _on_idle_timeout(self, timeout_type):
        """Handle idle timeout."""
        print(f"[{self.session_id}] Idle timeout")
        await self.close("idle_timeout")
    
    async def _on_total_timeout(self, timeout_type):
        """Handle total session timeout."""
        print(f"[{self.session_id}] Session timeout")
        await self.close("session_timeout")
    
    async def _cleanup_resources(self):
        """Cleanup session resources."""
        print(f"[{self.session_id}] Cleaning up resources")
        # Save final state
        self.save_state()
    
    async def _send_event(self, event: dict):
        """Send event to server."""
        print(f"[{self.session_id}] Send: {event['type']}")
    
    async def start(self):
        """Start the session."""
        
        await self.lifecycle.transition_to(SessionState.CONNECTED)
        await self.lifecycle.transition_to(SessionState.READY)
        
        print(f"[{self.session_id}] Ready")
    
    async def close(self, reason: str = "user_request"):
        """Close the session."""
        
        print(f"[{self.session_id}] Closing: {reason}")
        
        await self.lifecycle.transition_to(SessionState.CLOSING)
        await self.lifecycle.transition_to(SessionState.CLOSED)
    
    async def handle_event(self, event: dict):
        """Handle incoming server event."""
        
        event_type = event.get("type", "")
        self.timeout_manager.record_activity()
        
        # Route to appropriate handler
        if event_type.startswith("input_audio_buffer"):
            await self.vad_manager.handle_event(event)
        
        elif event_type == "response.created":
            await self._handle_response_created(event)
        
        elif event_type == "response.audio_transcript.delta":
            await self._handle_transcript_delta(event)
        
        elif event_type == "response.done":
            await self._handle_response_done(event)
        
        elif event_type == "conversation.item.created":
            await self._handle_item_created(event)
    
    async def _handle_response_created(self, event: dict):
        """Handle response creation."""
        
        await self.lifecycle.transition_to(SessionState.RESPONDING)
        
        # Start assistant turn
        self.turn_manager.start_turn(TurnType.ASSISTANT)
        
        # Create interruptible response
        response_id = event.get("response", {}).get("id", "unknown")
        self.current_response = InterruptibleResponse(
            response_id,
            self.interrupt_handler
        )
    
    async def _handle_transcript_delta(self, event: dict):
        """Handle transcript update."""
        
        if self.current_response:
            delta = event.get("delta", "")
            self.current_response.append_transcript(delta)
    
    async def _handle_response_done(self, event: dict):
        """Handle response completion."""
        
        if self.current_response:
            self.current_response.complete()
            self.current_response = None
        
        # Complete assistant turn
        if self.turn_manager.current_turn:
            self.turn_manager.complete_turn()
        
        await self.lifecycle.transition_to(SessionState.READY)
    
    async def _handle_item_created(self, event: dict):
        """Handle conversation item creation."""
        
        item_data = event.get("item", {})
        
        item = ConversationItem(
            id=item_data.get("id", ""),
            type=item_data.get("type", ""),
            role=item_data.get("role"),
            content=item_data.get("content"),
            status=item_data.get("status", "completed")
        )
        
        self.conversation_items.append(item)
    
    def save_state(self) -> Path:
        """Save session state."""
        
        return self.persistence.save_snapshot(
            self.session_id,
            self.config,
            self.conversation_items,
            self.lifecycle.metrics
        )
    
    def get_session_summary(self) -> dict:
        """Get session summary."""
        
        turn_stats = self.turn_manager.get_turn_stats()
        interrupt_stats = self.interrupt_handler.get_interruption_stats()
        
        return {
            "session_id": self.session_id,
            "state": self.lifecycle.state.value,
            "duration_seconds": self.lifecycle.metrics.duration_seconds,
            "turns": turn_stats,
            "interruptions": interrupt_stats,
            "conversation_items": len(self.conversation_items),
            "idle_seconds": self.timeout_manager.get_idle_seconds()
        }


# Usage
config = SessionConfig(
    model="gpt-4o-realtime-preview",
    voice="coral",
    instructions="You are a helpful voice assistant.",
    timeout_seconds=600,
    idle_timeout_seconds=60
)

manager = ComprehensiveSessionManager(config)

print("Session Manager Created")
print("=" * 50)
print(f"Session ID: {manager.session_id}")
print(f"State: {manager.lifecycle.state.value}")

# Simulate session
# await manager.start()
# await manager.handle_event({"type": "input_audio_buffer.speech_started"})
# await manager.handle_event({"type": "input_audio_buffer.speech_stopped"})
# await manager.handle_event({"type": "response.created", "response": {"id": "resp_123"}})
# await manager.handle_event({"type": "response.done"})

# Get summary
summary = manager.get_session_summary()
print(f"\nSession Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

</details>

---

## Summary

✅ Session lifecycle follows defined state transitions  
✅ Turn-taking manages user and assistant exchanges  
✅ Interruptions truncate responses gracefully  
✅ Persistence enables session resume  
✅ Timeouts prevent resource leaks

**Next:** [Realtime Events](./06-realtime-events.md)

---

## Further Reading

- [OpenAI Realtime Guide](https://platform.openai.com/docs/guides/realtime) — Session management details
- [Session Events Reference](https://platform.openai.com/docs/api-reference/realtime-client-events) — Event documentation
- [Turn Detection](https://platform.openai.com/docs/guides/realtime#turn-detection) — VAD configuration
