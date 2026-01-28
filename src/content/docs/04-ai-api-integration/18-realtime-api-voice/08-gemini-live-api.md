---
title: "Gemini Live API"
---

# Gemini Live API

## Introduction

Google's Gemini 2.0 Live API enables real-time, bidirectional streaming for voice and video interactions. It supports audio input/output, video understanding, and tool use within live sessions.

### What We'll Cover

- Gemini 2.0 Live API overview
- WebSocket connection setup
- Bidirectional audio streaming
- Video input handling
- Tool integration in live sessions

### Prerequisites

- Google Cloud project with Gemini API access
- Understanding of WebSocket protocols
- Experience with async Python

---

## Gemini Live API Overview

### API Capabilities

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import asyncio
import json
import base64


class GeminiLiveCapability(Enum):
    """Gemini Live API capabilities."""
    
    AUDIO_INPUT = "audio_input"       # Voice input
    AUDIO_OUTPUT = "audio_output"     # Voice output
    VIDEO_INPUT = "video_input"       # Camera/video input
    SCREEN_SHARE = "screen_share"     # Screen sharing input
    TEXT_INPUT = "text_input"         # Text messages
    TOOL_USE = "tool_use"             # Function calling


@dataclass
class GeminiLiveConfig:
    """Configuration for Gemini Live session."""
    
    model: str = "gemini-2.0-flash-exp"
    
    # Voice settings
    voice_name: str = "Puck"  # Puck, Charon, Kore, Fenrir, Aoede
    language: str = "en-US"
    
    # Audio settings
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    audio_encoding: str = "LINEAR16"  # LINEAR16 or MULAW
    
    # Generation settings
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 8192
    
    # Features
    enable_audio: bool = True
    enable_video: bool = False
    enable_tools: bool = False
    
    # System instruction
    system_instruction: str = ""


GEMINI_VOICES = {
    "Puck": {
        "description": "Energetic and bright",
        "style": "Casual",
        "best_for": ["Entertainment", "Gaming", "Youth content"]
    },
    "Charon": {
        "description": "Deep and authoritative",
        "style": "Formal",
        "best_for": ["Narration", "News", "Business"]
    },
    "Kore": {
        "description": "Warm and friendly",
        "style": "Conversational",
        "best_for": ["Customer service", "Education", "Support"]
    },
    "Fenrir": {
        "description": "Confident and dynamic",
        "style": "Professional",
        "best_for": ["Presentations", "Marketing", "Announcements"]
    },
    "Aoede": {
        "description": "Calm and soothing",
        "style": "Gentle",
        "best_for": ["Meditation", "Healthcare", "Accessibility"]
    }
}


# Feature comparison
GEMINI_VS_OPENAI = {
    "audio_input": {"gemini": True, "openai": True},
    "audio_output": {"gemini": True, "openai": True},
    "video_input": {"gemini": True, "openai": False},
    "screen_share": {"gemini": True, "openai": False},
    "native_multimodal": {"gemini": True, "openai": True},
    "function_calling": {"gemini": True, "openai": True},
    "code_execution": {"gemini": True, "openai": False}
}


print("Gemini Live Voices:")
for voice, info in GEMINI_VOICES.items():
    print(f"  {voice}: {info['description']} ({info['style']})")

print("\nFeature Comparison (Gemini vs OpenAI):")
for feature, support in GEMINI_VS_OPENAI.items():
    gemini = "✓" if support["gemini"] else "✗"
    openai = "✓" if support["openai"] else "✗"
    print(f"  {feature}: Gemini {gemini} | OpenAI {openai}")
```

---

## WebSocket Connection

### Setting Up the Connection

```python
import websockets
from google.auth import default
from google.auth.transport.requests import Request


class GeminiLiveConnection:
    """Manage WebSocket connection to Gemini Live API."""
    
    API_VERSION = "v1alpha"
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        config: GeminiLiveConfig = None
    ):
        self.project_id = project_id
        self.location = location
        self.config = config or GeminiLiveConfig()
        
        self.websocket = None
        self.session_id: Optional[str] = None
        self._connected = False
        
        # Event handlers
        self._on_audio: Optional[Callable] = None
        self._on_text: Optional[Callable] = None
        self._on_tool_call: Optional[Callable] = None
    
    def _get_endpoint(self) -> str:
        """Get WebSocket endpoint URL."""
        
        return (
            f"wss://{self.location}-aiplatform.googleapis.com/"
            f"{self.API_VERSION}/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/google/"
            f"models/{self.config.model}:streamGenerateContent"
        )
    
    async def _get_auth_token(self) -> str:
        """Get authentication token."""
        
        credentials, _ = default()
        credentials.refresh(Request())
        return credentials.token
    
    def _build_setup_message(self) -> dict:
        """Build initial setup message."""
        
        setup = {
            "setup": {
                "model": f"models/{self.config.model}",
                "generation_config": {
                    "response_modalities": ["AUDIO"] if self.config.enable_audio else ["TEXT"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self.config.voice_name
                            }
                        }
                    },
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "top_p": self.config.top_p,
                    "max_output_tokens": self.config.max_output_tokens
                }
            }
        }
        
        if self.config.system_instruction:
            setup["setup"]["system_instruction"] = {
                "parts": [{"text": self.config.system_instruction}]
            }
        
        if self.config.enable_tools:
            setup["setup"]["tools"] = []
        
        return setup
    
    async def connect(self):
        """Connect to Gemini Live API."""
        
        token = await self._get_auth_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        self.websocket = await websockets.connect(
            self._get_endpoint(),
            extra_headers=headers
        )
        
        # Send setup message
        setup_msg = self._build_setup_message()
        await self.websocket.send(json.dumps(setup_msg))
        
        # Wait for setup confirmation
        response = await self.websocket.recv()
        setup_response = json.loads(response)
        
        if "setupComplete" in setup_response:
            self._connected = True
            self.session_id = setup_response.get("setupComplete", {}).get("sessionId")
            print(f"Connected to Gemini Live: {self.session_id}")
        else:
            raise ConnectionError(f"Setup failed: {setup_response}")
    
    async def disconnect(self):
        """Disconnect from API."""
        
        if self.websocket:
            await self.websocket.close()
            self._connected = False
            self.session_id = None
    
    def on_audio(self, handler: Callable):
        """Register audio response handler."""
        self._on_audio = handler
    
    def on_text(self, handler: Callable):
        """Register text response handler."""
        self._on_text = handler
    
    def on_tool_call(self, handler: Callable):
        """Register tool call handler."""
        self._on_tool_call = handler
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# Usage
config = GeminiLiveConfig(
    model="gemini-2.0-flash-exp",
    voice_name="Kore",
    enable_audio=True,
    system_instruction="You are a helpful voice assistant."
)

connection = GeminiLiveConnection(
    project_id="your-project-id",
    config=config
)

print(f"Endpoint: {connection._get_endpoint()[:80]}...")
print(f"Voice: {config.voice_name}")
```

---

## Bidirectional Audio Streaming

### Sending and Receiving Audio

```python
class GeminiAudioStreamer:
    """Handle bidirectional audio streaming."""
    
    def __init__(self, connection: GeminiLiveConnection):
        self.connection = connection
        
        # Audio buffers
        self.input_buffer: List[bytes] = []
        self.output_buffer: List[bytes] = []
        
        # State
        self.is_streaming = False
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
    
    def _encode_audio(self, audio_data: bytes) -> str:
        """Encode audio to base64."""
        return base64.b64encode(audio_data).decode("utf-8")
    
    def _decode_audio(self, base64_data: str) -> bytes:
        """Decode audio from base64."""
        return base64.b64decode(base64_data)
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to Gemini."""
        
        if not self.connection.is_connected:
            raise ConnectionError("Not connected")
        
        message = {
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": f"audio/pcm;rate={self.connection.config.input_sample_rate}",
                    "data": self._encode_audio(audio_data)
                }]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def send_text(self, text: str):
        """Send text input."""
        
        if not self.connection.is_connected:
            raise ConnectionError("Not connected")
        
        message = {
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": text}]
                }],
                "turnComplete": True
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def _receive_loop(self):
        """Receive and process responses."""
        
        while self.is_streaming:
            try:
                message = await asyncio.wait_for(
                    self.connection.websocket.recv(),
                    timeout=0.1
                )
                
                await self._process_response(json.loads(message))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                break
    
    async def _process_response(self, response: dict):
        """Process server response."""
        
        # Handle different response types
        if "serverContent" in response:
            await self._handle_content(response["serverContent"])
        elif "toolCall" in response:
            await self._handle_tool_call(response["toolCall"])
        elif "turnComplete" in response:
            print("Turn complete")
    
    async def _handle_content(self, content: dict):
        """Handle content response."""
        
        model_turn = content.get("modelTurn", {})
        parts = model_turn.get("parts", [])
        
        for part in parts:
            if "text" in part:
                # Text response
                if self.connection._on_text:
                    await self._call_handler(
                        self.connection._on_text,
                        part["text"]
                    )
            
            elif "inlineData" in part:
                # Audio response
                inline = part["inlineData"]
                if inline.get("mimeType", "").startswith("audio/"):
                    audio_data = self._decode_audio(inline["data"])
                    self.output_buffer.append(audio_data)
                    
                    if self.connection._on_audio:
                        await self._call_handler(
                            self.connection._on_audio,
                            audio_data
                        )
    
    async def _handle_tool_call(self, tool_call: dict):
        """Handle tool call request."""
        
        if self.connection._on_tool_call:
            await self._call_handler(
                self.connection._on_tool_call,
                tool_call
            )
    
    async def _call_handler(self, handler: Callable, *args):
        """Call handler safely."""
        result = handler(*args)
        if asyncio.iscoroutine(result):
            await result
    
    async def start_streaming(self):
        """Start bidirectional streaming."""
        
        if not self.connection.is_connected:
            raise ConnectionError("Not connected")
        
        self.is_streaming = True
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def stop_streaming(self):
        """Stop streaming."""
        
        self.is_streaming = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass


class AudioChunker:
    """Chunk audio for streaming."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.bytes_per_sample = 2  # 16-bit
        
        self.chunk_size = (
            sample_rate *
            self.bytes_per_sample *
            chunk_duration_ms // 1000
        )
        
        self.buffer = b""
    
    def add_audio(self, audio_data: bytes) -> List[bytes]:
        """Add audio and return complete chunks."""
        
        self.buffer += audio_data
        chunks = []
        
        while len(self.buffer) >= self.chunk_size:
            chunks.append(self.buffer[:self.chunk_size])
            self.buffer = self.buffer[self.chunk_size:]
        
        return chunks
    
    def flush(self) -> Optional[bytes]:
        """Flush remaining audio."""
        
        if self.buffer:
            remaining = self.buffer
            self.buffer = b""
            return remaining
        return None


# Usage
streamer = GeminiAudioStreamer(connection)

# Register handlers
streamer.connection.on_audio(lambda data: print(f"Audio: {len(data)} bytes"))
streamer.connection.on_text(lambda text: print(f"Text: {text}"))

# Chunker for input
chunker = AudioChunker(
    sample_rate=16000,
    chunk_duration_ms=100
)

print(f"Chunk size: {chunker.chunk_size} bytes")
print(f"Duration: {chunker.chunk_duration_ms}ms")
```

---

## Video Input Handling

### Processing Video Input

```python
class VideoInputType(Enum):
    """Types of video input."""
    
    CAMERA = "camera"
    SCREEN = "screen"
    FILE = "file"


@dataclass
class VideoConfig:
    """Video input configuration."""
    
    input_type: VideoInputType = VideoInputType.CAMERA
    frame_rate: int = 1  # Frames per second to send
    max_width: int = 640
    max_height: int = 480
    format: str = "jpeg"  # jpeg or png


class GeminiVideoHandler:
    """Handle video input for Gemini Live."""
    
    def __init__(
        self,
        connection: GeminiLiveConnection,
        config: VideoConfig = None
    ):
        self.connection = connection
        self.config = config or VideoConfig()
        
        self.is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
    
    def _encode_frame(self, frame_data: bytes) -> str:
        """Encode frame to base64."""
        return base64.b64encode(frame_data).decode("utf-8")
    
    def _get_mime_type(self) -> str:
        """Get MIME type for video format."""
        return f"image/{self.config.format}"
    
    async def send_frame(self, frame_data: bytes):
        """Send a single video frame."""
        
        if not self.connection.is_connected:
            raise ConnectionError("Not connected")
        
        message = {
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": self._get_mime_type(),
                    "data": self._encode_frame(frame_data)
                }]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def send_audio_and_video(
        self,
        audio_data: bytes,
        frame_data: bytes
    ):
        """Send synchronized audio and video."""
        
        if not self.connection.is_connected:
            raise ConnectionError("Not connected")
        
        # Combined message with both audio and video
        message = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": f"audio/pcm;rate={self.connection.config.input_sample_rate}",
                        "data": base64.b64encode(audio_data).decode("utf-8")
                    },
                    {
                        "mimeType": self._get_mime_type(),
                        "data": self._encode_frame(frame_data)
                    }
                ]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def start_camera_stream(self, frame_generator):
        """Start streaming from camera."""
        
        self.is_streaming = True
        
        async def stream_loop():
            frame_interval = 1.0 / self.config.frame_rate
            
            while self.is_streaming:
                try:
                    # Get frame from generator
                    frame = next(frame_generator)
                    await self.send_frame(frame)
                    await asyncio.sleep(frame_interval)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Frame error: {e}")
                    break
        
        self._stream_task = asyncio.create_task(stream_loop())
    
    async def stop_stream(self):
        """Stop video streaming."""
        
        self.is_streaming = False
        
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass


class ScreenShareHandler:
    """Handle screen sharing input."""
    
    def __init__(
        self,
        connection: GeminiLiveConnection,
        config: VideoConfig = None
    ):
        self.connection = connection
        self.config = config or VideoConfig(input_type=VideoInputType.SCREEN)
    
    async def send_screenshot(self, screenshot_data: bytes):
        """Send a screenshot."""
        
        message = {
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": "image/png",
                    "data": base64.b64encode(screenshot_data).decode("utf-8")
                }]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def describe_screen(self, screenshot_data: bytes, question: str):
        """Send screenshot with question."""
        
        # First send the screenshot
        await self.send_screenshot(screenshot_data)
        
        # Then send the question
        message = {
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": question}]
                }],
                "turnComplete": True
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))


# Usage
video_config = VideoConfig(
    input_type=VideoInputType.CAMERA,
    frame_rate=1,
    max_width=640,
    max_height=480
)

video_handler = GeminiVideoHandler(connection, video_config)
screen_handler = ScreenShareHandler(connection)

print(f"Video config: {video_config.frame_rate} fps, {video_config.max_width}x{video_config.max_height}")
```

---

## Tool Integration

### Tools in Live Sessions

```python
@dataclass
class GeminiTool:
    """Tool definition for Gemini Live."""
    
    name: str
    description: str
    parameters: dict
    handler: Callable
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        return {
            "functionDeclarations": [{
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }]
        }


class GeminiToolManager:
    """Manage tools for Gemini Live sessions."""
    
    def __init__(self, connection: GeminiLiveConnection):
        self.connection = connection
        self.tools: Dict[str, GeminiTool] = {}
    
    def register_tool(self, tool: GeminiTool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tools_config(self) -> List[dict]:
        """Get tools configuration for setup."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    async def handle_tool_call(self, tool_call: dict):
        """Handle a tool call from Gemini."""
        
        function_calls = tool_call.get("functionCalls", [])
        
        for call in function_calls:
            name = call.get("name")
            args = call.get("args", {})
            call_id = call.get("id")
            
            if name in self.tools:
                tool = self.tools[name]
                
                try:
                    # Execute tool
                    result = tool.handler(**args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    
                    # Send result back
                    await self._send_tool_result(call_id, name, result)
                    
                except Exception as e:
                    await self._send_tool_error(call_id, name, str(e))
    
    async def _send_tool_result(
        self,
        call_id: str,
        name: str,
        result: Any
    ):
        """Send tool result back to Gemini."""
        
        message = {
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "name": name,
                    "response": {"result": result}
                }]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))
    
    async def _send_tool_error(
        self,
        call_id: str,
        name: str,
        error: str
    ):
        """Send tool error back to Gemini."""
        
        message = {
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "name": name,
                    "response": {"error": error}
                }]
            }
        }
        
        await self.connection.websocket.send(json.dumps(message))


# Example tools
def get_weather(location: str) -> dict:
    """Get weather for a location."""
    # Simulated response
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny"
    }


def search_knowledge_base(query: str) -> dict:
    """Search internal knowledge base."""
    return {
        "query": query,
        "results": ["Result 1", "Result 2"]
    }


WEATHER_TOOL = GeminiTool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or zip code"
            }
        },
        "required": ["location"]
    },
    handler=get_weather
)

SEARCH_TOOL = GeminiTool(
    name="search_knowledge_base",
    description="Search the internal knowledge base",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    },
    handler=search_knowledge_base
)


# Usage
tool_manager = GeminiToolManager(connection)
tool_manager.register_tool(WEATHER_TOOL)
tool_manager.register_tool(SEARCH_TOOL)

tools_config = tool_manager.get_tools_config()
print(f"Registered tools: {list(tool_manager.tools.keys())}")
```

---

## Complete Gemini Live Client

### Full Implementation

```python
class GeminiLiveClient:
    """Complete Gemini Live API client."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        config: GeminiLiveConfig = None
    ):
        self.config = config or GeminiLiveConfig()
        
        # Connection
        self.connection = GeminiLiveConnection(
            project_id=project_id,
            location=location,
            config=self.config
        )
        
        # Handlers
        self.audio_streamer = GeminiAudioStreamer(self.connection)
        self.tool_manager = GeminiToolManager(self.connection)
        
        if self.config.enable_video:
            self.video_handler = GeminiVideoHandler(
                self.connection,
                VideoConfig()
            )
        
        # State
        self._is_active = False
        
        # Callbacks
        self._on_response: Optional[Callable] = None
        self._on_audio_response: Optional[Callable] = None
        self._on_turn_complete: Optional[Callable] = None
    
    def on_response(self, callback: Callable):
        """Register response callback."""
        self._on_response = callback
        self.connection.on_text(callback)
    
    def on_audio_response(self, callback: Callable):
        """Register audio response callback."""
        self._on_audio_response = callback
        self.connection.on_audio(callback)
    
    def on_turn_complete(self, callback: Callable):
        """Register turn complete callback."""
        self._on_turn_complete = callback
    
    def register_tool(self, tool: GeminiTool):
        """Register a tool."""
        self.tool_manager.register_tool(tool)
    
    async def connect(self):
        """Connect to Gemini Live."""
        
        # Update config with tools if registered
        if self.tool_manager.tools:
            self.config.enable_tools = True
        
        await self.connection.connect()
        
        # Setup tool handling
        self.connection.on_tool_call(self.tool_manager.handle_tool_call)
        
        # Start streaming
        await self.audio_streamer.start_streaming()
        
        self._is_active = True
    
    async def disconnect(self):
        """Disconnect from Gemini Live."""
        
        self._is_active = False
        await self.audio_streamer.stop_streaming()
        await self.connection.disconnect()
    
    async def send_audio(self, audio_data: bytes):
        """Send audio input."""
        await self.audio_streamer.send_audio_chunk(audio_data)
    
    async def send_text(self, text: str):
        """Send text input."""
        await self.audio_streamer.send_text(text)
    
    async def send_video_frame(self, frame_data: bytes):
        """Send video frame."""
        if hasattr(self, 'video_handler'):
            await self.video_handler.send_frame(frame_data)
    
    @property
    def is_connected(self) -> bool:
        return self._is_active and self.connection.is_connected


# Usage example
async def main():
    # Configure client
    config = GeminiLiveConfig(
        model="gemini-2.0-flash-exp",
        voice_name="Kore",
        enable_audio=True,
        enable_video=False,
        temperature=0.8,
        system_instruction="You are a helpful voice assistant named Aria. "
                          "Be friendly and concise in your responses."
    )
    
    client = GeminiLiveClient(
        project_id="your-project-id",
        config=config
    )
    
    # Register callbacks
    client.on_response(lambda text: print(f"Response: {text}"))
    client.on_audio_response(lambda audio: print(f"Audio: {len(audio)} bytes"))
    
    # Register tools
    client.register_tool(WEATHER_TOOL)
    
    # Connect
    # await client.connect()
    
    # Send audio
    # await client.send_audio(audio_chunk)
    
    # Or send text
    # await client.send_text("What's the weather like?")
    
    # Disconnect
    # await client.disconnect()
    
    return client


# Create client (without running)
# client = asyncio.run(main())
print("Gemini Live Client ready for use")
```

---

## Hands-on Exercise

### Your Task

Build a multimodal Gemini Live assistant that handles audio, video, and tools.

### Requirements

1. Set up bidirectional audio streaming
2. Handle video input
3. Integrate tools
4. Manage session lifecycle

<details>
<summary>💡 Hints</summary>

- Use separate handlers for each modality
- Coordinate audio/video timing
- Handle tool calls asynchronously
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import asyncio
import json
import base64


class MultimodalGeminiAssistant:
    """Complete multimodal Gemini Live assistant."""
    
    def __init__(self, project_id: str, assistant_name: str = "Aria"):
        self.project_id = project_id
        self.assistant_name = assistant_name
        
        # Configuration
        self.config = GeminiLiveConfig(
            model="gemini-2.0-flash-exp",
            voice_name="Kore",
            enable_audio=True,
            enable_video=True,
            enable_tools=True,
            system_instruction=f"""You are {assistant_name}, a helpful multimodal assistant.

## Capabilities
- Voice conversation with natural speech
- Visual understanding from camera or screen
- Tool use for real-world tasks

## Guidelines
- Be friendly and conversational
- Describe what you see when asked about visuals
- Use tools proactively when helpful
- Keep responses concise for voice delivery
"""
        )
        
        # Components
        self.connection = GeminiLiveConnection(
            project_id=project_id,
            config=self.config
        )
        self.audio_streamer = GeminiAudioStreamer(self.connection)
        self.video_handler = GeminiVideoHandler(self.connection)
        self.tool_manager = GeminiToolManager(self.connection)
        
        # State
        self.session_start: Optional[datetime] = None
        self.turn_count = 0
        self.is_listening = False
        self.is_speaking = False
        
        # Metrics
        self.metrics = {
            "audio_chunks_sent": 0,
            "video_frames_sent": 0,
            "tool_calls": 0,
            "responses": 0
        }
        
        # Setup
        self._setup_tools()
        self._setup_handlers()
    
    def _setup_tools(self):
        """Register available tools."""
        
        # Weather tool
        self.tool_manager.register_tool(GeminiTool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            },
            handler=self._weather_handler
        ))
        
        # Calendar tool
        self.tool_manager.register_tool(GeminiTool(
            name="check_calendar",
            description="Check calendar for events",
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check (YYYY-MM-DD)"
                    }
                },
                "required": ["date"]
            },
            handler=self._calendar_handler
        ))
        
        # Web search tool
        self.tool_manager.register_tool(GeminiTool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            handler=self._search_handler
        ))
    
    def _setup_handlers(self):
        """Setup event handlers."""
        
        self.connection.on_text(self._on_text_response)
        self.connection.on_audio(self._on_audio_response)
        self.connection.on_tool_call(self._on_tool_call)
    
    async def _weather_handler(self, location: str) -> dict:
        """Handle weather requests."""
        self.metrics["tool_calls"] += 1
        return {
            "location": location,
            "temperature": 72,
            "conditions": "partly cloudy",
            "forecast": "Mild temperatures expected"
        }
    
    async def _calendar_handler(self, date: str) -> dict:
        """Handle calendar requests."""
        self.metrics["tool_calls"] += 1
        return {
            "date": date,
            "events": [
                {"time": "10:00 AM", "title": "Team standup"},
                {"time": "2:00 PM", "title": "Project review"}
            ]
        }
    
    async def _search_handler(self, query: str) -> dict:
        """Handle search requests."""
        self.metrics["tool_calls"] += 1
        return {
            "query": query,
            "results": [
                {"title": "Result 1", "snippet": "Relevant information..."},
                {"title": "Result 2", "snippet": "More details..."}
            ]
        }
    
    async def _on_text_response(self, text: str):
        """Handle text response."""
        self.metrics["responses"] += 1
        print(f"\n{self.assistant_name}: {text}")
    
    async def _on_audio_response(self, audio_data: bytes):
        """Handle audio response."""
        self.is_speaking = True
        # Play audio or buffer for playback
        print(f"[Playing {len(audio_data)} bytes of audio]")
    
    async def _on_tool_call(self, tool_call: dict):
        """Handle tool call."""
        await self.tool_manager.handle_tool_call(tool_call)
    
    async def start(self):
        """Start the assistant."""
        
        print(f"\n{self.assistant_name} is starting up...")
        
        await self.connection.connect()
        await self.audio_streamer.start_streaming()
        
        self.session_start = datetime.now()
        self.is_listening = True
        
        print(f"{self.assistant_name} is ready! Say something...")
    
    async def stop(self):
        """Stop the assistant."""
        
        self.is_listening = False
        await self.audio_streamer.stop_streaming()
        await self.connection.disconnect()
        
        duration = (datetime.now() - self.session_start).total_seconds()
        print(f"\n{self.assistant_name} session ended")
        print(f"Duration: {duration:.1f}s")
        print(f"Metrics: {self.metrics}")
    
    async def send_audio(self, audio_chunk: bytes):
        """Send audio input."""
        if not self.is_listening:
            return
        
        await self.audio_streamer.send_audio_chunk(audio_chunk)
        self.metrics["audio_chunks_sent"] += 1
    
    async def send_text(self, text: str):
        """Send text input."""
        if not self.connection.is_connected:
            return
        
        self.turn_count += 1
        print(f"\nYou: {text}")
        await self.audio_streamer.send_text(text)
    
    async def send_video_frame(self, frame_data: bytes):
        """Send video frame."""
        if not self.connection.is_connected:
            return
        
        await self.video_handler.send_frame(frame_data)
        self.metrics["video_frames_sent"] += 1
    
    async def describe_visual(
        self,
        frame_data: bytes,
        question: str = "What do you see?"
    ):
        """Send visual with description request."""
        
        # Send frame
        await self.send_video_frame(frame_data)
        
        # Ask about it
        await self.send_text(question)
    
    def get_session_info(self) -> dict:
        """Get session information."""
        
        if not self.session_start:
            return {"status": "not_started"}
        
        duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "assistant": self.assistant_name,
            "duration_seconds": duration,
            "turn_count": self.turn_count,
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "metrics": self.metrics,
            "tools_available": list(self.tool_manager.tools.keys())
        }


# Usage example
async def demo_assistant():
    """Demonstrate multimodal assistant."""
    
    assistant = MultimodalGeminiAssistant(
        project_id="your-project-id",
        assistant_name="Aria"
    )
    
    # Get session info before starting
    print("Session info:", assistant.get_session_info())
    
    # In production:
    # await assistant.start()
    # await assistant.send_text("What's the weather in San Francisco?")
    # await asyncio.sleep(5)  # Wait for response
    # await assistant.stop()
    
    return assistant


# Create assistant
assistant = MultimodalGeminiAssistant(
    project_id="demo-project",
    assistant_name="Aria"
)

print("Multimodal Gemini Assistant")
print("=" * 50)
print(f"Name: {assistant.assistant_name}")
print(f"Model: {assistant.config.model}")
print(f"Voice: {assistant.config.voice_name}")
print(f"Tools: {list(assistant.tool_manager.tools.keys())}")
print(f"\nSession: {assistant.get_session_info()}")
```

</details>

---

## Summary

✅ Gemini 2.0 Live API supports audio, video, and text  
✅ WebSocket connection enables bidirectional streaming  
✅ Video input allows visual understanding in real-time  
✅ Tools integrate seamlessly in live sessions  
✅ Multiple voice options available for different use cases

**Next:** [Anthropic Voice](./09-anthropic-voice.md)

---

## Further Reading

- [Gemini Live API Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/realtime) — Official docs
- [Gemini 2.0 Capabilities](https://deepmind.google/technologies/gemini/) — Model overview
- [Multimodal AI Guide](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/overview) — Google Cloud multimodal
