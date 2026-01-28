---
title: "Voice Agent Building"
---

# Voice Agent Building

## Introduction

The OpenAI Agents SDK provides high-level abstractions for building voice agents. Using `RealtimeAgent` and `RealtimeSession`, you can create sophisticated voice assistants with automatic audio handling and connection management.

### What We'll Cover

- Agents SDK for TypeScript
- RealtimeAgent and RealtimeSession
- Microphone and audio output handling
- Automatic connection management

### Prerequisites

- Understanding of Realtime API basics
- TypeScript/JavaScript experience
- Familiarity with async patterns

---

## Agents SDK Overview

### Understanding the Agents SDK

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime


class AgentCapability(Enum):
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    TEXT_INPUT = "text_input"
    TEXT_OUTPUT = "text_output"
    FUNCTION_CALLING = "function_calling"
    TRANSCRIPTION = "transcription"


@dataclass
class AgentConfig:
    """Configuration for a realtime agent."""
    
    name: str
    instructions: str
    voice: str = "alloy"
    model: str = "gpt-4o-realtime-preview"
    capabilities: List[AgentCapability] = field(default_factory=lambda: [
        AgentCapability.VOICE_INPUT,
        AgentCapability.VOICE_OUTPUT,
        AgentCapability.TRANSCRIPTION
    ])
    temperature: float = 0.8
    tools: List[dict] = field(default_factory=list)


class RealtimeAgentBase:
    """Base class for realtime voice agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session = None
        self.is_active = False
        self.conversation_history: List[dict] = []
        
        # Event handlers
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None
    
    def on_speech_start(self, handler: Callable):
        """Register speech start handler."""
        self._on_speech_start = handler
    
    def on_speech_end(self, handler: Callable):
        """Register speech end handler."""
        self._on_speech_end = handler
    
    def on_response(self, handler: Callable):
        """Register response handler."""
        self._on_response = handler
    
    def on_transcript(self, handler: Callable):
        """Register transcript handler."""
        self._on_transcript = handler


# TypeScript Agents SDK equivalent
TYPESCRIPT_AGENT_CODE = '''
import { RealtimeAgent, RealtimeSession } from '@openai/agents';

// Define agent
const voiceAgent = new RealtimeAgent({
  name: "VoiceAssistant",
  instructions: "You are a helpful voice assistant.",
  model: "gpt-4o-realtime-preview",
  voice: "alloy"
});

// Create session
const session = new RealtimeSession(voiceAgent);

// Handle events
session.on('speech_start', () => {
  console.log('User started speaking');
});

session.on('speech_end', () => {
  console.log('User stopped speaking');
});

session.on('response', (response) => {
  console.log('Agent response:', response.text);
});

// Connect
await session.connect();

// Start listening
await session.startListening();
'''

print("TypeScript Agents SDK example:")
print(TYPESCRIPT_AGENT_CODE)
```

### Python Agent Implementation

```python
from openai import OpenAI
import asyncio


class RealtimeAgent:
    """Python implementation of a realtime voice agent."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI()
        self.session_id: Optional[str] = None
        self.is_connected = False
        self.is_listening = False
        
        # Audio buffers
        self.input_audio_buffer: bytes = b""
        self.output_audio_buffer: bytes = b""
        
        # Conversation state
        self.turns: List[dict] = []
        self.current_turn: Optional[dict] = None
    
    def get_session_config(self) -> dict:
        """Get session configuration for API."""
        
        return {
            "model": self.config.model,
            "voice": self.config.voice,
            "instructions": self.config.instructions,
            "temperature": self.config.temperature,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 500
            },
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "tools": self.config.tools
        }
    
    async def connect(self) -> bool:
        """Connect to Realtime API."""
        
        try:
            # Create session
            response = self.client.realtime.sessions.create(
                **self.get_session_config()
            )
            
            self.session_id = response.id
            self.is_connected = True
            
            print(f"Connected: {self.session_id}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Realtime API."""
        
        if self.is_connected:
            self.is_connected = False
            self.is_listening = False
            self.session_id = None
            print("Disconnected")
    
    def start_turn(self):
        """Start a new conversation turn."""
        
        self.current_turn = {
            "id": f"turn_{len(self.turns) + 1}",
            "started_at": datetime.now().isoformat(),
            "user_audio": b"",
            "user_transcript": "",
            "agent_audio": b"",
            "agent_transcript": ""
        }
    
    def end_turn(self):
        """End current turn."""
        
        if self.current_turn:
            self.current_turn["ended_at"] = datetime.now().isoformat()
            self.turns.append(self.current_turn)
            self.current_turn = None
    
    def get_conversation_transcript(self) -> str:
        """Get full conversation transcript."""
        
        lines = []
        for turn in self.turns:
            if turn.get("user_transcript"):
                lines.append(f"User: {turn['user_transcript']}")
            if turn.get("agent_transcript"):
                lines.append(f"Agent: {turn['agent_transcript']}")
        
        return "\n".join(lines)


# Usage
agent_config = AgentConfig(
    name="CustomerService",
    instructions="You are a helpful customer service agent for TechCorp.",
    voice="coral",
    tools=[]
)

agent = RealtimeAgent(agent_config)
print(f"Agent created: {agent.config.name}")
print(f"Session config: {agent.get_session_config()}")
```

---

## RealtimeSession Management

### Session Lifecycle

```python
class SessionState(Enum):
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"


@dataclass
class SessionMetrics:
    """Metrics for a realtime session."""
    
    start_time: datetime
    turn_count: int = 0
    total_audio_duration_ms: float = 0
    average_latency_ms: float = 0
    interruption_count: int = 0


class RealtimeSession:
    """Manage a realtime voice session."""
    
    def __init__(self, agent: RealtimeAgent):
        self.agent = agent
        self.state = SessionState.DISCONNECTED
        self.metrics = None
        
        # Audio handling
        self.audio_input_enabled = True
        self.audio_output_enabled = True
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit(self, event_type: str, data: Any = None):
        """Emit event to handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                print(f"Handler error: {e}")
    
    async def connect(self) -> bool:
        """Connect session."""
        
        self.state = SessionState.INITIALIZING
        
        if await self.agent.connect():
            self.state = SessionState.CONNECTED
            self.metrics = SessionMetrics(start_time=datetime.now())
            self.emit("connected")
            return True
        
        self.state = SessionState.DISCONNECTED
        return False
    
    async def start_listening(self):
        """Start listening for audio input."""
        
        if self.state != SessionState.CONNECTED:
            raise RuntimeError("Session not connected")
        
        self.state = SessionState.LISTENING
        self.agent.is_listening = True
        self.emit("listening_started")
    
    async def stop_listening(self):
        """Stop listening for audio input."""
        
        self.agent.is_listening = False
        if self.state == SessionState.LISTENING:
            self.state = SessionState.CONNECTED
        self.emit("listening_stopped")
    
    async def pause(self):
        """Pause the session."""
        
        previous_state = self.state
        self.state = SessionState.PAUSED
        self.emit("paused", {"previous_state": previous_state.value})
    
    async def resume(self):
        """Resume the session."""
        
        if self.state == SessionState.PAUSED:
            self.state = SessionState.LISTENING
            self.emit("resumed")
    
    async def disconnect(self):
        """Disconnect session."""
        
        await self.agent.disconnect()
        self.state = SessionState.DISCONNECTED
        self.emit("disconnected", self.get_summary())
    
    def handle_speech_start(self):
        """Handle user speech start."""
        
        self.agent.start_turn()
        self.emit("speech_start")
    
    def handle_speech_end(self):
        """Handle user speech end."""
        
        self.state = SessionState.PROCESSING
        self.emit("speech_end")
    
    def handle_response_start(self):
        """Handle agent response start."""
        
        self.state = SessionState.RESPONDING
        self.emit("response_start")
    
    def handle_response_end(self):
        """Handle agent response end."""
        
        self.agent.end_turn()
        self.metrics.turn_count += 1
        self.state = SessionState.LISTENING
        self.emit("response_end")
    
    def handle_interruption(self):
        """Handle user interruption."""
        
        self.metrics.interruption_count += 1
        self.emit("interruption")
    
    def get_summary(self) -> dict:
        """Get session summary."""
        
        duration = 0
        if self.metrics:
            duration = (datetime.now() - self.metrics.start_time).total_seconds()
        
        return {
            "state": self.state.value,
            "duration_seconds": duration,
            "turn_count": self.metrics.turn_count if self.metrics else 0,
            "interruption_count": self.metrics.interruption_count if self.metrics else 0,
            "transcript": self.agent.get_conversation_transcript()
        }


# Usage
session = RealtimeSession(agent)

# Register handlers
session.on("connected", lambda: print("✓ Connected"))
session.on("speech_start", lambda: print("🎤 User speaking..."))
session.on("speech_end", lambda: print("🎤 User finished"))
session.on("response_start", lambda: print("🤖 Agent responding..."))
session.on("response_end", lambda: print("🤖 Agent finished"))

print("Session created with event handlers")
```

---

## Microphone and Audio Handling

### Audio Input Handler

```python
import base64


@dataclass
class AudioInputConfig:
    """Configuration for audio input."""
    
    sample_rate: int = 24000
    channels: int = 1
    bits_per_sample: int = 16
    chunk_duration_ms: int = 20
    
    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes."""
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        bytes_per_sample = self.bits_per_sample // 8
        return samples_per_chunk * self.channels * bytes_per_sample


class MicrophoneHandler:
    """Handle microphone input for voice agents."""
    
    def __init__(self, config: AudioInputConfig = None):
        self.config = config or AudioInputConfig()
        self.is_capturing = False
        self.audio_callback: Optional[Callable[[bytes], None]] = None
        
        # Audio buffer
        self.buffer: bytes = b""
        self.total_captured_bytes = 0
    
    def start_capture(self, callback: Callable[[bytes], None]):
        """Start capturing audio."""
        
        self.is_capturing = True
        self.audio_callback = callback
        
        # In production, use pyaudio or sounddevice
        print("Microphone capture started")
        print(f"  Sample rate: {self.config.sample_rate}")
        print(f"  Chunk size: {self.config.chunk_size_bytes} bytes")
    
    def stop_capture(self):
        """Stop capturing audio."""
        
        self.is_capturing = False
        self.audio_callback = None
        print(f"Microphone capture stopped. Total: {self.total_captured_bytes} bytes")
    
    def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk."""
        
        if not self.is_capturing:
            return
        
        self.buffer += audio_data
        self.total_captured_bytes += len(audio_data)
        
        # Send chunks when buffer is full
        while len(self.buffer) >= self.config.chunk_size_bytes:
            chunk = self.buffer[:self.config.chunk_size_bytes]
            self.buffer = self.buffer[self.config.chunk_size_bytes:]
            
            if self.audio_callback:
                self.audio_callback(chunk)
    
    def get_captured_duration_ms(self) -> float:
        """Get total captured duration in milliseconds."""
        
        bytes_per_second = (
            self.config.sample_rate * 
            self.config.channels * 
            (self.config.bits_per_sample // 8)
        )
        
        return (self.total_captured_bytes / bytes_per_second) * 1000


# JavaScript equivalent for browser microphone access
BROWSER_MICROPHONE_CODE = '''
class BrowserMicrophone {
  private mediaStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  
  async start(onAudioChunk: (chunk: Float32Array) => void): Promise<void> {
    // Get microphone access
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
    
    // Create audio context
    this.audioContext = new AudioContext({ sampleRate: 24000 });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    // Create processor for audio chunks
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => {
      const audioData = event.inputBuffer.getChannelData(0);
      onAudioChunk(audioData);
    };
    
    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }
  
  stop(): void {
    if (this.processor) {
      this.processor.disconnect();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
  }
}
'''

print("Browser microphone handler:")
print(BROWSER_MICROPHONE_CODE)
```

### Audio Output Handler

```python
@dataclass
class AudioOutputConfig:
    """Configuration for audio output."""
    
    sample_rate: int = 24000
    channels: int = 1
    bits_per_sample: int = 16
    buffer_size_ms: int = 100  # Pre-buffer before playback


class AudioOutputHandler:
    """Handle audio output for voice agents."""
    
    def __init__(self, config: AudioOutputConfig = None):
        self.config = config or AudioOutputConfig()
        self.is_playing = False
        
        # Audio buffers
        self.playback_buffer: bytes = b""
        self.total_played_bytes = 0
        
        # Callbacks
        self.on_playback_start: Optional[Callable] = None
        self.on_playback_end: Optional[Callable] = None
    
    def add_audio(self, audio_data: bytes):
        """Add audio to playback buffer."""
        
        self.playback_buffer += audio_data
        
        # Start playback if buffer is full enough
        if not self.is_playing:
            buffer_duration = self._get_buffer_duration_ms()
            if buffer_duration >= self.config.buffer_size_ms:
                self._start_playback()
    
    def _get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        
        bytes_per_second = (
            self.config.sample_rate * 
            self.config.channels * 
            (self.config.bits_per_sample // 8)
        )
        
        return (len(self.playback_buffer) / bytes_per_second) * 1000
    
    def _start_playback(self):
        """Start audio playback."""
        
        self.is_playing = True
        if self.on_playback_start:
            self.on_playback_start()
        print("Audio playback started")
    
    def _stop_playback(self):
        """Stop audio playback."""
        
        self.is_playing = False
        if self.on_playback_end:
            self.on_playback_end()
        print(f"Audio playback ended. Total: {self.total_played_bytes} bytes")
    
    def clear_buffer(self):
        """Clear playback buffer (for interruption)."""
        
        self.playback_buffer = b""
        if self.is_playing:
            self._stop_playback()
    
    def get_playback_position_ms(self) -> float:
        """Get current playback position."""
        
        bytes_per_second = (
            self.config.sample_rate * 
            self.config.channels * 
            (self.config.bits_per_sample // 8)
        )
        
        return (self.total_played_bytes / bytes_per_second) * 1000


# Usage
audio_output = AudioOutputHandler()

# Set callbacks
audio_output.on_playback_start = lambda: print("🔊 Playing audio...")
audio_output.on_playback_end = lambda: print("🔊 Audio finished")

# Simulate receiving audio
# audio_output.add_audio(b"\x00\x01" * 2400)  # 100ms of audio
```

---

## Automatic Connection Management

### Connection Manager

```python
class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class ConnectionOptions:
    """Options for connection management."""
    
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay_ms: int = 1000
    reconnect_backoff_multiplier: float = 1.5
    heartbeat_interval_ms: int = 30000


class ConnectionManager:
    """Manage realtime connection with auto-reconnect."""
    
    def __init__(
        self,
        agent: RealtimeAgent,
        options: ConnectionOptions = None
    ):
        self.agent = agent
        self.options = options or ConnectionOptions()
        self.state = ConnectionState.DISCONNECTED
        
        self.reconnect_attempts = 0
        self.last_heartbeat: Optional[datetime] = None
        
        # Event handlers
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_reconnecting: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def connect(self) -> bool:
        """Establish connection."""
        
        self.state = ConnectionState.CONNECTING
        
        try:
            if await self.agent.connect():
                self.state = ConnectionState.CONNECTED
                self.reconnect_attempts = 0
                self.last_heartbeat = datetime.now()
                
                if self.on_connected:
                    self.on_connected()
                
                # Start heartbeat
                asyncio.create_task(self._heartbeat_loop())
                
                return True
        except Exception as e:
            if self.on_error:
                self.on_error(e)
        
        # Try reconnect
        if self.options.auto_reconnect:
            return await self._reconnect()
        
        self.state = ConnectionState.DISCONNECTED
        return False
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect."""
        
        self.state = ConnectionState.RECONNECTING
        
        if self.on_reconnecting:
            self.on_reconnecting(self.reconnect_attempts)
        
        while self.reconnect_attempts < self.options.max_reconnect_attempts:
            self.reconnect_attempts += 1
            
            # Calculate delay with backoff
            delay = self.options.reconnect_delay_ms * (
                self.options.reconnect_backoff_multiplier ** (self.reconnect_attempts - 1)
            )
            
            print(f"Reconnecting in {delay}ms (attempt {self.reconnect_attempts})")
            await asyncio.sleep(delay / 1000)
            
            try:
                if await self.agent.connect():
                    self.state = ConnectionState.CONNECTED
                    self.reconnect_attempts = 0
                    
                    if self.on_connected:
                        self.on_connected()
                    
                    return True
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
        
        self.state = ConnectionState.DISCONNECTED
        if self.on_disconnected:
            self.on_disconnected("Max reconnect attempts reached")
        
        return False
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        
        while self.state == ConnectionState.CONNECTED:
            await asyncio.sleep(self.options.heartbeat_interval_ms / 1000)
            
            if self.state != ConnectionState.CONNECTED:
                break
            
            try:
                # Send heartbeat (implementation depends on protocol)
                self.last_heartbeat = datetime.now()
            except Exception:
                # Connection lost, try reconnect
                if self.options.auto_reconnect:
                    await self._reconnect()
                else:
                    self.state = ConnectionState.DISCONNECTED
                    if self.on_disconnected:
                        self.on_disconnected("Heartbeat failed")
    
    async def disconnect(self):
        """Disconnect gracefully."""
        
        self.state = ConnectionState.DISCONNECTED
        await self.agent.disconnect()
        
        if self.on_disconnected:
            self.on_disconnected("Manual disconnect")


# Usage
connection_options = ConnectionOptions(
    auto_reconnect=True,
    max_reconnect_attempts=5,
    reconnect_delay_ms=1000,
    reconnect_backoff_multiplier=1.5
)

connection = ConnectionManager(agent, connection_options)
connection.on_connected = lambda: print("✓ Connected")
connection.on_reconnecting = lambda n: print(f"⟳ Reconnecting (attempt {n})")
connection.on_disconnected = lambda r: print(f"✗ Disconnected: {r}")

print("Connection manager configured with auto-reconnect")
```

---

## Complete Voice Agent

### Full Implementation

```python
class VoiceAgent:
    """Complete voice agent with all components."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        voice: str = "alloy",
        tools: List[dict] = None
    ):
        # Configuration
        self.config = AgentConfig(
            name=name,
            instructions=instructions,
            voice=voice,
            tools=tools or []
        )
        
        # Core components
        self.agent = RealtimeAgent(self.config)
        self.session = RealtimeSession(self.agent)
        self.connection = ConnectionManager(
            self.agent,
            ConnectionOptions(auto_reconnect=True)
        )
        
        # Audio handlers
        self.microphone = MicrophoneHandler()
        self.audio_output = AudioOutputHandler()
        
        # State
        self.is_running = False
        
        # Set up internal event routing
        self._setup_event_routing()
    
    def _setup_event_routing(self):
        """Set up internal event handlers."""
        
        # Connection events
        self.connection.on_connected = self._on_connected
        self.connection.on_disconnected = self._on_disconnected
        
        # Audio output events
        self.audio_output.on_playback_start = self._on_audio_start
        self.audio_output.on_playback_end = self._on_audio_end
    
    def _on_connected(self):
        """Handle connection established."""
        print(f"[{self.config.name}] Connected")
    
    def _on_disconnected(self, reason: str):
        """Handle disconnection."""
        print(f"[{self.config.name}] Disconnected: {reason}")
    
    def _on_audio_start(self):
        """Handle audio playback start."""
        print(f"[{self.config.name}] Speaking...")
    
    def _on_audio_end(self):
        """Handle audio playback end."""
        print(f"[{self.config.name}] Finished speaking")
    
    def _on_audio_chunk(self, chunk: bytes):
        """Handle incoming audio chunk from microphone."""
        
        # In production, send to Realtime API
        pass
    
    async def start(self):
        """Start the voice agent."""
        
        print(f"Starting voice agent: {self.config.name}")
        
        # Connect
        if not await self.connection.connect():
            raise RuntimeError("Failed to connect")
        
        # Start session
        await self.session.start_listening()
        
        # Start microphone capture
        self.microphone.start_capture(self._on_audio_chunk)
        
        self.is_running = True
        print(f"[{self.config.name}] Ready and listening")
    
    async def stop(self):
        """Stop the voice agent."""
        
        print(f"Stopping voice agent: {self.config.name}")
        
        # Stop microphone
        self.microphone.stop_capture()
        
        # Stop session
        await self.session.disconnect()
        
        # Disconnect
        await self.connection.disconnect()
        
        self.is_running = False
        print(f"[{self.config.name}] Stopped")
    
    def interrupt(self):
        """Interrupt current response."""
        
        self.audio_output.clear_buffer()
        self.session.handle_interruption()
        print(f"[{self.config.name}] Interrupted")
    
    def get_transcript(self) -> str:
        """Get conversation transcript."""
        return self.agent.get_conversation_transcript()
    
    def get_summary(self) -> dict:
        """Get agent summary."""
        return {
            "name": self.config.name,
            "is_running": self.is_running,
            "session": self.session.get_summary(),
            "connection_state": self.connection.state.value
        }


# Usage
voice_agent = VoiceAgent(
    name="TechSupport",
    instructions="""You are a technical support agent for a software company.
    Be helpful, patient, and thorough in your explanations.
    Always verify the customer's issue before suggesting solutions.""",
    voice="coral"
)

print(f"Voice agent created: {voice_agent.config.name}")
print(f"Voice: {voice_agent.config.voice}")
```

---

## Hands-on Exercise

### Your Task

Build a complete voice agent with custom tools.

### Requirements

1. Create agent with function calling
2. Handle audio input/output
3. Implement connection management
4. Add conversation state tracking

<details>
<summary>💡 Hints</summary>

- Define tools as function schemas
- Track conversation turns with metadata
- Handle connection errors gracefully
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import asyncio
import json


# Tool definitions
class CustomerServiceTools:
    """Tools for customer service agent."""
    
    @staticmethod
    def get_tools() -> List[dict]:
        return [
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up order status by order ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "type": "function",
                "name": "schedule_callback",
                "description": "Schedule a callback from a human agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {
                            "type": "string",
                            "description": "Customer phone number"
                        },
                        "preferred_time": {
                            "type": "string",
                            "description": "Preferred callback time"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for callback"
                        }
                    },
                    "required": ["phone_number", "reason"]
                }
            },
            {
                "type": "function",
                "name": "transfer_to_department",
                "description": "Transfer call to a specific department",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {
                            "type": "string",
                            "enum": ["billing", "technical", "sales", "returns"]
                        }
                    },
                    "required": ["department"]
                }
            }
        ]
    
    @staticmethod
    def lookup_order(order_id: str) -> dict:
        """Simulate order lookup."""
        # Mock database lookup
        orders = {
            "ORD-12345": {"status": "shipped", "tracking": "1Z999AA1"},
            "ORD-67890": {"status": "processing", "eta": "2-3 days"}
        }
        
        if order_id in orders:
            return {"found": True, **orders[order_id]}
        return {"found": False, "message": "Order not found"}
    
    @staticmethod
    def schedule_callback(
        phone_number: str,
        reason: str,
        preferred_time: str = None
    ) -> dict:
        """Simulate scheduling callback."""
        return {
            "scheduled": True,
            "confirmation_id": f"CB-{datetime.now().strftime('%Y%m%d%H%M')}",
            "estimated_wait": "24-48 hours"
        }
    
    @staticmethod
    def transfer_to_department(department: str) -> dict:
        """Simulate department transfer."""
        return {
            "transferred": True,
            "department": department,
            "message": f"Transferring to {department} department..."
        }


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    
    turn_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    user_transcript: str = ""
    agent_transcript: str = ""
    user_audio_duration_ms: float = 0
    agent_audio_duration_ms: float = 0
    tool_calls: List[dict] = field(default_factory=list)
    latency_ms: float = 0


class ConversationTracker:
    """Track conversation state and history."""
    
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.current_turn: Optional[ConversationTurn] = None
        self.metadata: Dict[str, Any] = {}
    
    def start_turn(self) -> ConversationTurn:
        """Start a new turn."""
        self.current_turn = ConversationTurn(
            turn_id=f"turn_{len(self.turns) + 1}",
            started_at=datetime.now()
        )
        return self.current_turn
    
    def end_turn(self):
        """End current turn."""
        if self.current_turn:
            self.current_turn.ended_at = datetime.now()
            self.turns.append(self.current_turn)
            self.current_turn = None
    
    def add_tool_call(self, tool_name: str, args: dict, result: Any):
        """Add tool call to current turn."""
        if self.current_turn:
            self.current_turn.tool_calls.append({
                "tool": tool_name,
                "arguments": args,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
    
    def set_metadata(self, key: str, value: Any):
        """Set conversation metadata."""
        self.metadata[key] = value
    
    def get_transcript(self) -> str:
        """Get full transcript."""
        lines = []
        for turn in self.turns:
            if turn.user_transcript:
                lines.append(f"Customer: {turn.user_transcript}")
            if turn.agent_transcript:
                lines.append(f"Agent: {turn.agent_transcript}")
        return "\n".join(lines)
    
    def get_summary(self) -> dict:
        """Get conversation summary."""
        total_duration = 0
        if self.turns:
            first = self.turns[0].started_at
            last = self.turns[-1].ended_at or datetime.now()
            total_duration = (last - first).total_seconds()
        
        tool_call_count = sum(len(t.tool_calls) for t in self.turns)
        
        return {
            "turn_count": len(self.turns),
            "duration_seconds": total_duration,
            "tool_calls": tool_call_count,
            "metadata": self.metadata
        }


class CustomerServiceVoiceAgent:
    """Complete customer service voice agent."""
    
    def __init__(self):
        # Tools
        self.tools = CustomerServiceTools()
        
        # Agent configuration
        self.config = AgentConfig(
            name="CustomerService",
            instructions="""You are a helpful customer service agent.
            
Your capabilities:
- Look up order status using the lookup_order tool
- Schedule callbacks for customers using schedule_callback
- Transfer to specialized departments when needed

Guidelines:
- Be patient and friendly
- Verify customer information before sharing order details
- Offer to schedule callbacks for complex issues
- Transfer to appropriate department when you can't resolve the issue
- Summarize the conversation before ending""",
            voice="coral",
            tools=self.tools.get_tools()
        )
        
        # Components
        self.agent = RealtimeAgent(self.config)
        self.conversation = ConversationTracker()
        self.connection = ConnectionManager(
            self.agent,
            ConnectionOptions(
                auto_reconnect=True,
                max_reconnect_attempts=3
            )
        )
        
        # Audio
        self.microphone = MicrophoneHandler()
        self.audio_output = AudioOutputHandler()
        
        # State
        self.is_active = False
        self.customer_id: Optional[str] = None
    
    def _handle_tool_call(
        self,
        tool_name: str,
        arguments: dict
    ) -> Any:
        """Handle tool call from agent."""
        
        result = None
        
        if tool_name == "lookup_order":
            result = self.tools.lookup_order(arguments["order_id"])
            
        elif tool_name == "schedule_callback":
            result = self.tools.schedule_callback(
                arguments["phone_number"],
                arguments["reason"],
                arguments.get("preferred_time")
            )
            
        elif tool_name == "transfer_to_department":
            result = self.tools.transfer_to_department(
                arguments["department"]
            )
        
        # Track tool call
        self.conversation.add_tool_call(tool_name, arguments, result)
        
        return result
    
    async def start(self, customer_id: str = None):
        """Start the agent."""
        
        self.customer_id = customer_id
        self.conversation.set_metadata("customer_id", customer_id)
        self.conversation.set_metadata("session_start", datetime.now().isoformat())
        
        # Connect
        print(f"Starting customer service agent...")
        
        if not await self.connection.connect():
            raise RuntimeError("Failed to connect")
        
        # Start audio
        self.microphone.start_capture(self._on_audio_input)
        
        self.is_active = True
        
        # Greeting
        print(f"Agent ready. Greeting customer...")
    
    async def stop(self):
        """Stop the agent."""
        
        self.microphone.stop_capture()
        await self.connection.disconnect()
        
        self.conversation.set_metadata("session_end", datetime.now().isoformat())
        self.is_active = False
        
        print("Agent stopped")
        print(f"\n=== Session Summary ===")
        print(json.dumps(self.conversation.get_summary(), indent=2))
    
    def _on_audio_input(self, chunk: bytes):
        """Handle audio input."""
        # Send to Realtime API
        pass
    
    def handle_user_speech(self, transcript: str):
        """Handle user speech event."""
        
        turn = self.conversation.start_turn()
        turn.user_transcript = transcript
        
        print(f"Customer: {transcript}")
    
    def handle_agent_response(self, transcript: str, audio: bytes = None):
        """Handle agent response."""
        
        if self.conversation.current_turn:
            self.conversation.current_turn.agent_transcript = transcript
            self.conversation.end_turn()
        
        print(f"Agent: {transcript}")
        
        if audio:
            self.audio_output.add_audio(audio)
    
    def get_call_summary(self) -> dict:
        """Get complete call summary."""
        
        return {
            "customer_id": self.customer_id,
            "conversation": self.conversation.get_summary(),
            "transcript": self.conversation.get_transcript(),
            "tool_usage": [
                call
                for turn in self.conversation.turns
                for call in turn.tool_calls
            ]
        }


# Usage
customer_service = CustomerServiceVoiceAgent()

print("Customer Service Voice Agent")
print("=" * 40)
print(f"Name: {customer_service.config.name}")
print(f"Voice: {customer_service.config.voice}")
print(f"Tools: {[t['name'] for t in customer_service.config.tools]}")
print()

# Simulate a conversation
customer_service.conversation.set_metadata("customer_id", "CUST-12345")

# Turn 1
turn1 = customer_service.conversation.start_turn()
turn1.user_transcript = "Hi, I want to check on my order ORD-12345"
customer_service._handle_tool_call("lookup_order", {"order_id": "ORD-12345"})
turn1.agent_transcript = "I found your order. It has been shipped with tracking number 1Z999AA1."
customer_service.conversation.end_turn()

# Turn 2
turn2 = customer_service.conversation.start_turn()
turn2.user_transcript = "Great, thanks! Can I also get a callback about a billing question?"
customer_service._handle_tool_call("schedule_callback", {
    "phone_number": "555-1234",
    "reason": "Billing question"
})
turn2.agent_transcript = "I've scheduled a callback for you. You should receive a call within 24-48 hours."
customer_service.conversation.end_turn()

# Print summary
print("\n=== Call Summary ===")
summary = customer_service.get_call_summary()
print(f"Customer: {summary['customer_id']}")
print(f"Turns: {summary['conversation']['turn_count']}")
print(f"Tool calls: {len(summary['tool_usage'])}")
print()
print("Transcript:")
print(summary['transcript'])
```

</details>

---

## Summary

✅ Agents SDK provides high-level abstractions for voice agents  
✅ RealtimeAgent configures model, voice, and capabilities  
✅ RealtimeSession manages lifecycle and events  
✅ MicrophoneHandler captures audio input  
✅ AudioOutputHandler plays agent responses  
✅ ConnectionManager handles auto-reconnect

**Next:** [Audio in Responses API](./04-audio-responses-api.md)

---

## Further Reading

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-js) — TypeScript SDK
- [Realtime API Guide](https://platform.openai.com/docs/guides/realtime) — Official documentation
- [Voice Agent Patterns](https://platform.openai.com/docs/guides/realtime-conversations) — Best practices
