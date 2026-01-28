---
title: "Connection Methods"
---

# Connection Methods

## Introduction

The Realtime API supports multiple connection methods optimized for different deployment scenarios. Choose WebRTC for browser applications, WebSocket for server-side control, or SIP for telephony integration.

### What We'll Cover

- WebRTC for browser applications
- WebSocket for server applications
- SIP for VoIP telephony
- Ephemeral API keys for client-side

### Prerequisites

- Understanding of Realtime API basics
- Familiarity with network protocols
- JavaScript/TypeScript knowledge

---

## WebRTC for Browser Applications

### WebRTC Overview

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime


class ConnectionType(Enum):
    WEBRTC = "webrtc"
    WEBSOCKET = "websocket"
    SIP = "sip"


@dataclass
class WebRTCConfig:
    """WebRTC connection configuration."""
    
    model: str = "gpt-4o-realtime-preview"
    ephemeral_key: Optional[str] = None
    ice_servers: List[Dict[str, str]] = field(default_factory=list)
    audio_codec: str = "opus"
    dtls_fingerprint: Optional[str] = None
    
    def get_connection_url(self) -> str:
        """Get WebRTC connection endpoint."""
        return "https://api.openai.com/v1/realtime"


class WebRTCConnectionManager:
    """Manage WebRTC connections for browser clients."""
    
    def __init__(self, config: WebRTCConfig):
        self.config = config
        self.connection_state = "new"
        self.data_channel_open = False
        self.audio_track_active = False
    
    def get_offer_config(self) -> dict:
        """Get configuration for creating WebRTC offer."""
        
        return {
            "iceServers": self.config.ice_servers or [
                {"urls": "stun:stun.l.google.com:19302"}
            ],
            "sdpSemantics": "unified-plan",
            "bundlePolicy": "max-bundle"
        }
    
    def get_sdp_constraints(self) -> dict:
        """Get SDP offer/answer constraints."""
        
        return {
            "mandatory": {
                "OfferToReceiveAudio": True,
                "OfferToReceiveVideo": False
            }
        }


# JavaScript/TypeScript equivalent for browser:
WEBRTC_BROWSER_CODE = '''
// Browser-side WebRTC connection
async function connectWebRTC(ephemeralKey: string) {
  // Create peer connection
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
  });
  
  // Add audio track from microphone
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  stream.getTracks().forEach(track => pc.addTrack(track, stream));
  
  // Create data channel for events
  const dc = pc.createDataChannel("oai-events");
  
  // Handle incoming audio
  pc.ontrack = (event) => {
    const audioEl = new Audio();
    audioEl.srcObject = event.streams[0];
    audioEl.play();
  };
  
  // Create and set offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  
  // Send offer to OpenAI
  const response = await fetch("https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${ephemeralKey}`,
      "Content-Type": "application/sdp"
    },
    body: offer.sdp
  });
  
  // Set remote description
  const answer = await response.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answer });
  
  return { pc, dc };
}
'''

print("WebRTC connection code for browser:")
print(WEBRTC_BROWSER_CODE)
```

### Ephemeral Key Generation

```python
from openai import OpenAI


class EphemeralKeyManager:
    """Manage ephemeral API keys for client-side use."""
    
    def __init__(self):
        self.client = OpenAI()
        self.active_keys: Dict[str, dict] = {}
    
    def create_ephemeral_key(
        self,
        user_id: str,
        session_duration_minutes: int = 60
    ) -> dict:
        """Create ephemeral key for client."""
        
        # Create ephemeral key via API
        response = self.client.realtime.sessions.create(
            model="gpt-4o-realtime-preview",
            modalities=["audio", "text"],
            voice="alloy"
        )
        
        key_info = {
            "key": response.client_secret.value,
            "expires_at": response.client_secret.expires_at,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
        
        self.active_keys[user_id] = key_info
        
        return {
            "ephemeral_key": key_info["key"],
            "expires_at": key_info["expires_at"]
        }
    
    def revoke_key(self, user_id: str) -> bool:
        """Revoke an ephemeral key."""
        
        if user_id in self.active_keys:
            del self.active_keys[user_id]
            return True
        return False
    
    def is_key_valid(self, user_id: str) -> bool:
        """Check if user's key is still valid."""
        
        if user_id not in self.active_keys:
            return False
        
        key_info = self.active_keys[user_id]
        expires = datetime.fromisoformat(key_info["expires_at"])
        
        return datetime.now() < expires


# Server-side endpoint for generating keys
class EphemeralKeyEndpoint:
    """Server endpoint for ephemeral key generation."""
    
    def __init__(self):
        self.key_manager = EphemeralKeyManager()
    
    def handle_request(
        self,
        user_id: str,
        session_config: dict = None
    ) -> dict:
        """Handle ephemeral key request."""
        
        # Validate user (authentication)
        if not self._validate_user(user_id):
            return {"error": "Unauthorized"}
        
        # Generate key
        key_data = self.key_manager.create_ephemeral_key(user_id)
        
        return {
            "status": "success",
            "ephemeral_key": key_data["ephemeral_key"],
            "expires_at": key_data["expires_at"],
            "connection_url": "https://api.openai.com/v1/realtime"
        }
    
    def _validate_user(self, user_id: str) -> bool:
        """Validate user authentication."""
        # Implement your auth logic
        return True


# Usage
# key_manager = EphemeralKeyManager()
# key = key_manager.create_ephemeral_key("user_123")
# print(f"Ephemeral key: {key['ephemeral_key'][:20]}...")
```

---

## WebSocket for Server Applications

### WebSocket Connection

```python
import asyncio
import websockets
import json


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""
    
    api_key: str
    model: str = "gpt-4o-realtime-preview"
    base_url: str = "wss://api.openai.com/v1/realtime"
    
    def get_connection_url(self) -> str:
        """Get WebSocket connection URL."""
        return f"{self.base_url}?model={self.model}"
    
    def get_headers(self) -> dict:
        """Get connection headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }


class RealtimeWebSocketClient:
    """WebSocket client for Realtime API."""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.websocket = None
        self.connected = False
        self.session_id: Optional[str] = None
        
        # Event handlers
        self.on_message: Optional[Callable] = None
        self.on_audio: Optional[Callable] = None
        self.on_transcript: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def connect(self):
        """Establish WebSocket connection."""
        
        headers = self.config.get_headers()
        url = self.config.get_connection_url()
        
        self.websocket = await websockets.connect(
            url,
            extra_headers=headers
        )
        self.connected = True
        print(f"Connected to {url}")
        
        # Start message receiver
        asyncio.create_task(self._receive_messages())
    
    async def _receive_messages(self):
        """Receive and dispatch messages."""
        
        try:
            async for message in self.websocket:
                event = json.loads(message)
                await self._handle_event(event)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            print("Connection closed")
    
    async def _handle_event(self, event: dict):
        """Handle incoming event."""
        
        event_type = event.get("type", "")
        
        if event_type == "session.created":
            self.session_id = event.get("session", {}).get("id")
            print(f"Session created: {self.session_id}")
            
        elif event_type == "response.audio.delta":
            if self.on_audio:
                audio_data = event.get("delta", "")
                self.on_audio(audio_data)
                
        elif event_type == "conversation.item.input_audio_transcription.completed":
            if self.on_transcript:
                transcript = event.get("transcript", "")
                self.on_transcript(transcript)
                
        elif event_type == "error":
            if self.on_error:
                self.on_error(event)
            print(f"Error: {event}")
        
        # General handler
        if self.on_message:
            self.on_message(event)
    
    async def send_event(self, event: dict):
        """Send event to server."""
        
        if not self.connected:
            raise RuntimeError("Not connected")
        
        await self.websocket.send(json.dumps(event))
    
    async def configure_session(self, config: dict):
        """Configure the session."""
        
        await self.send_event({
            "type": "session.update",
            "session": config
        })
    
    async def send_audio(self, audio_base64: str):
        """Send audio data."""
        
        await self.send_event({
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        })
    
    async def commit_audio(self):
        """Commit audio buffer to trigger response."""
        
        await self.send_event({
            "type": "input_audio_buffer.commit"
        })
    
    async def send_text(self, text: str):
        """Send text message."""
        
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })
        
        # Request response
        await self.send_event({"type": "response.create"})
    
    async def cancel_response(self):
        """Cancel current response (for interruption)."""
        
        await self.send_event({
            "type": "response.cancel"
        })
    
    async def close(self):
        """Close connection."""
        
        if self.websocket:
            await self.websocket.close()
            self.connected = False


# Usage example (async context)
async def websocket_example():
    config = WebSocketConfig(
        api_key="your-api-key",
        model="gpt-4o-realtime-preview"
    )
    
    client = RealtimeWebSocketClient(config)
    
    # Set up handlers
    client.on_transcript = lambda t: print(f"Transcript: {t}")
    client.on_audio = lambda a: print(f"Audio chunk: {len(a)} bytes")
    
    # Connect
    await client.connect()
    
    # Configure session
    await client.configure_session({
        "voice": "alloy",
        "instructions": "You are a helpful assistant.",
        "turn_detection": {"type": "server_vad"}
    })
    
    # Send a text message
    await client.send_text("Hello, how are you?")
    
    # Wait for response
    await asyncio.sleep(5)
    
    # Close
    await client.close()


# Run example
# asyncio.run(websocket_example())
```

### WebSocket Event Loop

```python
class WebSocketEventLoop:
    """Event loop for WebSocket realtime sessions."""
    
    def __init__(self, client: RealtimeWebSocketClient):
        self.client = client
        self.running = False
        self.audio_buffer: List[bytes] = []
        self.pending_responses: List[str] = []
    
    async def start(self, audio_source: Optional[Callable] = None):
        """Start the event loop."""
        
        self.running = True
        
        while self.running:
            # Get audio from source if available
            if audio_source:
                audio = audio_source()
                if audio:
                    await self.client.send_audio(audio)
            
            await asyncio.sleep(0.02)  # 20ms intervals
    
    async def stop(self):
        """Stop the event loop."""
        self.running = False
    
    def on_audio_received(self, audio_data: str):
        """Handle received audio."""
        import base64
        decoded = base64.b64decode(audio_data)
        self.audio_buffer.append(decoded)
    
    def get_audio_output(self) -> bytes:
        """Get and clear audio buffer."""
        audio = b"".join(self.audio_buffer)
        self.audio_buffer.clear()
        return audio


# Usage
# loop = WebSocketEventLoop(client)
# await loop.start()
```

---

## SIP for VoIP Telephony

### SIP Connection Configuration

```python
@dataclass
class SIPConfig:
    """SIP connection configuration."""
    
    sip_uri: str  # e.g., "sip:+15551234567@pstn.twilio.com"
    model: str = "gpt-4o-realtime-preview"
    voice: str = "alloy"
    audio_format: str = "g711_ulaw"  # Standard for telephony
    
    # SIP-specific settings
    from_number: Optional[str] = None
    trunk_provider: str = "twilio"  # twilio, telnyx, etc.
    
    def get_audio_config(self) -> dict:
        """Get audio configuration for telephony."""
        return {
            "input_audio_format": self.audio_format,
            "output_audio_format": self.audio_format,
            "sample_rate": 8000  # Standard telephony rate
        }


class SIPTrunkManager:
    """Manage SIP trunk connections."""
    
    def __init__(self, provider: str, credentials: dict):
        self.provider = provider
        self.credentials = credentials
        self.active_calls: Dict[str, dict] = {}
    
    def initiate_call(
        self,
        to_number: str,
        from_number: str,
        session_config: dict
    ) -> dict:
        """Initiate outbound SIP call."""
        
        call_id = f"call_{datetime.now().timestamp()}"
        
        # Provider-specific call initiation
        if self.provider == "twilio":
            call_config = self._twilio_call(to_number, from_number)
        elif self.provider == "telnyx":
            call_config = self._telnyx_call(to_number, from_number)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        self.active_calls[call_id] = {
            "id": call_id,
            "to": to_number,
            "from": from_number,
            "status": "initiating",
            "started_at": datetime.now().isoformat(),
            "session_config": session_config
        }
        
        return {"call_id": call_id, **call_config}
    
    def _twilio_call(self, to: str, from_: str) -> dict:
        """Initiate Twilio SIP call."""
        
        # In production, use Twilio SDK
        return {
            "provider": "twilio",
            "sip_endpoint": f"sip:{to}@your-trunk.pstn.twilio.com",
            "status": "pending"
        }
    
    def _telnyx_call(self, to: str, from_: str) -> dict:
        """Initiate Telnyx SIP call."""
        
        return {
            "provider": "telnyx",
            "sip_endpoint": f"sip:{to}@sip.telnyx.com",
            "status": "pending"
        }
    
    def handle_inbound(
        self,
        call_sid: str,
        from_number: str,
        to_number: str
    ) -> dict:
        """Handle inbound SIP call."""
        
        call_id = f"inbound_{call_sid}"
        
        self.active_calls[call_id] = {
            "id": call_id,
            "to": to_number,
            "from": from_number,
            "status": "ringing",
            "direction": "inbound",
            "started_at": datetime.now().isoformat()
        }
        
        return {
            "call_id": call_id,
            "action": "connect_to_realtime",
            "stream_url": "wss://api.openai.com/v1/realtime"
        }
    
    def end_call(self, call_id: str) -> bool:
        """End an active call."""
        
        if call_id in self.active_calls:
            self.active_calls[call_id]["status"] = "ended"
            self.active_calls[call_id]["ended_at"] = datetime.now().isoformat()
            return True
        return False
    
    def get_active_calls(self) -> List[dict]:
        """Get list of active calls."""
        return [
            c for c in self.active_calls.values()
            if c["status"] not in ["ended", "failed"]
        ]


# Twilio integration example
TWILIO_TWIML_EXAMPLE = '''
<!-- TwiML for connecting inbound call to Realtime API -->
<Response>
  <Connect>
    <Stream url="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview">
      <Parameter name="Authorization" value="Bearer YOUR_API_KEY" />
    </Stream>
  </Connect>
</Response>
'''

print("TwiML for Twilio integration:")
print(TWILIO_TWIML_EXAMPLE)
```

### Telephony Audio Handling

```python
class TelephonyAudioHandler:
    """Handle audio for telephony connections."""
    
    def __init__(self, audio_format: str = "g711_ulaw"):
        self.audio_format = audio_format
        self.sample_rate = 8000
        self.input_buffer: bytes = b""
        self.output_buffer: bytes = b""
    
    def convert_to_realtime_format(self, telephony_audio: bytes) -> bytes:
        """Convert telephony audio to Realtime API format."""
        
        if self.audio_format == "g711_ulaw":
            # μ-law to PCM16 conversion
            return self._ulaw_to_pcm16(telephony_audio)
        elif self.audio_format == "g711_alaw":
            # A-law to PCM16 conversion
            return self._alaw_to_pcm16(telephony_audio)
        else:
            return telephony_audio
    
    def convert_from_realtime_format(self, realtime_audio: bytes) -> bytes:
        """Convert Realtime API audio to telephony format."""
        
        if self.audio_format == "g711_ulaw":
            return self._pcm16_to_ulaw(realtime_audio)
        elif self.audio_format == "g711_alaw":
            return self._pcm16_to_alaw(realtime_audio)
        else:
            return realtime_audio
    
    def _ulaw_to_pcm16(self, ulaw_data: bytes) -> bytes:
        """Convert μ-law to 16-bit PCM."""
        # Simplified - use audioop in production
        # import audioop
        # return audioop.ulaw2lin(ulaw_data, 2)
        return ulaw_data  # Placeholder
    
    def _pcm16_to_ulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to μ-law."""
        # Simplified - use audioop in production
        # import audioop
        # return audioop.lin2ulaw(pcm_data, 2)
        return pcm_data  # Placeholder
    
    def _alaw_to_pcm16(self, alaw_data: bytes) -> bytes:
        """Convert A-law to 16-bit PCM."""
        return alaw_data  # Placeholder
    
    def _pcm16_to_alaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to A-law."""
        return pcm_data  # Placeholder


# Usage
handler = TelephonyAudioHandler("g711_ulaw")
# telephony_audio = get_audio_from_call()
# realtime_audio = handler.convert_to_realtime_format(telephony_audio)
```

---

## Connection Selection

### Choosing the Right Connection

```python
class ConnectionSelector:
    """Select appropriate connection method."""
    
    @staticmethod
    def recommend_connection(
        use_case: str,
        environment: str,
        latency_requirement_ms: int = 500
    ) -> ConnectionType:
        """Recommend connection type based on requirements."""
        
        # Browser applications
        if environment == "browser":
            if latency_requirement_ms < 300:
                return ConnectionType.WEBRTC  # Lowest latency
            else:
                return ConnectionType.WEBRTC  # Still best for browser
        
        # Server applications
        if environment == "server":
            return ConnectionType.WEBSOCKET  # Full control
        
        # Telephony/VoIP
        if use_case in ["phone", "ivr", "call_center"]:
            return ConnectionType.SIP
        
        # Default
        return ConnectionType.WEBSOCKET
    
    @staticmethod
    def get_connection_requirements(
        connection_type: ConnectionType
    ) -> dict:
        """Get requirements for connection type."""
        
        requirements = {
            ConnectionType.WEBRTC: {
                "environment": "browser",
                "key_type": "ephemeral",
                "latency": "lowest (P2P)",
                "complexity": "medium",
                "bidirectional": True,
                "requirements": [
                    "Ephemeral key from backend",
                    "Microphone access",
                    "WebRTC-capable browser"
                ]
            },
            ConnectionType.WEBSOCKET: {
                "environment": "server",
                "key_type": "standard",
                "latency": "low",
                "complexity": "low",
                "bidirectional": True,
                "requirements": [
                    "Standard API key",
                    "WebSocket library",
                    "Audio capture (server-side)"
                ]
            },
            ConnectionType.SIP: {
                "environment": "telephony",
                "key_type": "standard",
                "latency": "varies",
                "complexity": "high",
                "bidirectional": True,
                "requirements": [
                    "SIP trunk provider",
                    "Phone number",
                    "G.711 audio support"
                ]
            }
        }
        
        return requirements.get(connection_type, {})


# Usage
selector = ConnectionSelector()

# Web app recommendation
web_conn = selector.recommend_connection(
    use_case="voice_assistant",
    environment="browser",
    latency_requirement_ms=300
)
print(f"Web app: {web_conn.value}")
print(f"Requirements: {selector.get_connection_requirements(web_conn)}")

# Server recommendation
server_conn = selector.recommend_connection(
    use_case="voice_bot",
    environment="server",
    latency_requirement_ms=500
)
print(f"\nServer: {server_conn.value}")
```

---

## Hands-on Exercise

### Your Task

Build a multi-connection manager for the Realtime API.

### Requirements

1. Support WebRTC, WebSocket, and SIP connections
2. Handle ephemeral key generation
3. Abstract connection specifics
4. Provide unified event handling

<details>
<summary>💡 Hints</summary>

- Use abstract base class for connections
- Factory pattern for connection creation
- Unified event interface across connection types
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Protocol
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ConnectionEvent:
    """Unified connection event."""
    
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: ConnectionType


class RealtimeConnection(ABC):
    """Abstract base for realtime connections."""
    
    def __init__(self):
        self.state = ConnectionState.DISCONNECTED
        self.session_id: Optional[str] = None
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection."""
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data."""
        pass
    
    @abstractmethod
    async def send_text(self, text: str) -> bool:
        """Send text message."""
        pass
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit(self, event: ConnectionEvent):
        """Emit event to handlers."""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error: {e}")


class WebRTCConnection(RealtimeConnection):
    """WebRTC connection implementation."""
    
    def __init__(self, ephemeral_key: str, model: str = "gpt-4o-realtime-preview"):
        super().__init__()
        self.ephemeral_key = ephemeral_key
        self.model = model
        self.connection_type = ConnectionType.WEBRTC
    
    async def connect(self) -> bool:
        """Connect via WebRTC."""
        self.state = ConnectionState.CONNECTING
        
        # In browser, this would use RTCPeerConnection
        # Server-side simulation
        try:
            # Simulate connection
            await asyncio.sleep(0.1)
            self.state = ConnectionState.CONNECTED
            self.session_id = f"webrtc_{datetime.now().timestamp()}"
            
            self.emit(ConnectionEvent(
                event_type="connected",
                timestamp=datetime.now(),
                data={"session_id": self.session_id},
                source=self.connection_type
            ))
            
            return True
        except Exception as e:
            self.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect WebRTC."""
        self.state = ConnectionState.DISCONNECTED
        return True
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio via WebRTC data channel."""
        if self.state != ConnectionState.CONNECTED:
            return False
        # WebRTC audio sent via media track
        return True
    
    async def send_text(self, text: str) -> bool:
        """Send text via data channel."""
        if self.state != ConnectionState.CONNECTED:
            return False
        return True


class WebSocketConnection(RealtimeConnection):
    """WebSocket connection implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.connection_type = ConnectionType.WEBSOCKET
        self.websocket = None
    
    async def connect(self) -> bool:
        """Connect via WebSocket."""
        self.state = ConnectionState.CONNECTING
        
        try:
            url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Simulate connection
            await asyncio.sleep(0.1)
            self.state = ConnectionState.CONNECTED
            self.session_id = f"ws_{datetime.now().timestamp()}"
            
            self.emit(ConnectionEvent(
                event_type="connected",
                timestamp=datetime.now(),
                data={"session_id": self.session_id},
                source=self.connection_type
            ))
            
            return True
        except Exception as e:
            self.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect WebSocket."""
        if self.websocket:
            # await self.websocket.close()
            pass
        self.state = ConnectionState.DISCONNECTED
        return True
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio via WebSocket."""
        if self.state != ConnectionState.CONNECTED:
            return False
        
        import base64
        audio_b64 = base64.b64encode(audio_data).decode()
        # Send via WebSocket
        return True
    
    async def send_text(self, text: str) -> bool:
        """Send text via WebSocket."""
        if self.state != ConnectionState.CONNECTED:
            return False
        return True


class SIPConnection(RealtimeConnection):
    """SIP connection implementation."""
    
    def __init__(
        self,
        api_key: str,
        sip_uri: str,
        trunk_provider: str = "twilio"
    ):
        super().__init__()
        self.api_key = api_key
        self.sip_uri = sip_uri
        self.trunk_provider = trunk_provider
        self.connection_type = ConnectionType.SIP
    
    async def connect(self) -> bool:
        """Connect via SIP."""
        self.state = ConnectionState.CONNECTING
        
        try:
            # SIP connection via trunk provider
            await asyncio.sleep(0.1)
            self.state = ConnectionState.CONNECTED
            self.session_id = f"sip_{datetime.now().timestamp()}"
            
            self.emit(ConnectionEvent(
                event_type="connected",
                timestamp=datetime.now(),
                data={"session_id": self.session_id, "sip_uri": self.sip_uri},
                source=self.connection_type
            ))
            
            return True
        except Exception as e:
            self.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect SIP."""
        self.state = ConnectionState.DISCONNECTED
        return True
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio via SIP RTP stream."""
        if self.state != ConnectionState.CONNECTED:
            return False
        return True
    
    async def send_text(self, text: str) -> bool:
        """Send text (converted to audio for SIP)."""
        if self.state != ConnectionState.CONNECTED:
            return False
        # For SIP, text would be converted to speech
        return True


class MultiConnectionManager:
    """Manage multiple realtime connections."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.connections: Dict[str, RealtimeConnection] = {}
        self.ephemeral_key_manager = EphemeralKeyManager()
        
        # Global event handlers
        self.global_handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event_type: str, handler: Callable):
        """Register global event handler."""
        if event_type not in self.global_handlers:
            self.global_handlers[event_type] = []
        self.global_handlers[event_type].append(handler)
    
    def _dispatch_event(self, event: ConnectionEvent):
        """Dispatch event to global handlers."""
        handlers = self.global_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Global handler error: {e}")
    
    async def create_webrtc_connection(
        self,
        connection_id: str,
        user_id: str
    ) -> RealtimeConnection:
        """Create WebRTC connection with ephemeral key."""
        
        # Generate ephemeral key
        key_data = self.ephemeral_key_manager.create_ephemeral_key(user_id)
        
        connection = WebRTCConnection(
            ephemeral_key=key_data["ephemeral_key"]
        )
        
        # Register for global events
        connection.on("connected", self._dispatch_event)
        connection.on("audio", self._dispatch_event)
        connection.on("transcript", self._dispatch_event)
        
        self.connections[connection_id] = connection
        return connection
    
    async def create_websocket_connection(
        self,
        connection_id: str
    ) -> RealtimeConnection:
        """Create WebSocket connection."""
        
        connection = WebSocketConnection(api_key=self.api_key)
        
        connection.on("connected", self._dispatch_event)
        connection.on("audio", self._dispatch_event)
        connection.on("transcript", self._dispatch_event)
        
        self.connections[connection_id] = connection
        return connection
    
    async def create_sip_connection(
        self,
        connection_id: str,
        sip_uri: str,
        trunk_provider: str = "twilio"
    ) -> RealtimeConnection:
        """Create SIP connection."""
        
        connection = SIPConnection(
            api_key=self.api_key,
            sip_uri=sip_uri,
            trunk_provider=trunk_provider
        )
        
        connection.on("connected", self._dispatch_event)
        connection.on("audio", self._dispatch_event)
        connection.on("transcript", self._dispatch_event)
        
        self.connections[connection_id] = connection
        return connection
    
    def get_connection(self, connection_id: str) -> Optional[RealtimeConnection]:
        """Get connection by ID."""
        return self.connections.get(connection_id)
    
    async def connect(self, connection_id: str) -> bool:
        """Connect a specific connection."""
        connection = self.connections.get(connection_id)
        if connection:
            return await connection.connect()
        return False
    
    async def disconnect(self, connection_id: str) -> bool:
        """Disconnect a specific connection."""
        connection = self.connections.get(connection_id)
        if connection:
            return await connection.disconnect()
        return False
    
    async def disconnect_all(self):
        """Disconnect all connections."""
        for conn_id in list(self.connections.keys()):
            await self.disconnect(conn_id)
    
    def get_status(self) -> Dict[str, dict]:
        """Get status of all connections."""
        return {
            conn_id: {
                "type": conn.connection_type.value,
                "state": conn.state.value,
                "session_id": conn.session_id
            }
            for conn_id, conn in self.connections.items()
        }


# Usage example
async def demo():
    manager = MultiConnectionManager(api_key="your-api-key")
    
    # Global event handler
    def on_connected(event: ConnectionEvent):
        print(f"Connection established: {event.source.value} - {event.data}")
    
    manager.on("connected", on_connected)
    
    # Create WebSocket connection
    ws_conn = await manager.create_websocket_connection("ws_1")
    await ws_conn.connect()
    
    # Create WebRTC connection (for browser client)
    # webrtc_conn = await manager.create_webrtc_connection("rtc_1", "user_123")
    
    # Create SIP connection (for telephony)
    # sip_conn = await manager.create_sip_connection(
    #     "sip_1",
    #     "sip:+15551234567@pstn.twilio.com"
    # )
    
    # Check status
    status = manager.get_status()
    print(f"\nConnection status: {status}")
    
    # Cleanup
    await manager.disconnect_all()


# Run demo
# asyncio.run(demo())

# Show status of multi-connection setup
manager = MultiConnectionManager(api_key="demo-key")
print("Multi-Connection Manager initialized")
print("Supported connection types:")
print("  - WebRTC (browser)")
print("  - WebSocket (server)")
print("  - SIP (telephony)")
```

</details>

---

## Summary

✅ WebRTC provides lowest latency for browser apps  
✅ WebSocket offers full server-side control  
✅ SIP enables VoIP and telephony integration  
✅ Ephemeral keys secure client-side connections  
✅ Choose connection type based on environment and use case

**Next:** [Voice Agent Building](./03-voice-agent-building.md)

---

## Further Reading

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Connection documentation
- [WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) — Browser WebRTC
- [Twilio SIP Trunking](https://www.twilio.com/docs/sip-trunking) — SIP integration example
