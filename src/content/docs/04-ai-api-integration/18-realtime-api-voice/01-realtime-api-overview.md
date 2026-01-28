---
title: "Realtime API Overview"
---

# Realtime API Overview

## Introduction

The OpenAI Realtime API enables low-latency, multimodal communication for building voice assistants and interactive applications. Unlike traditional request-response APIs, it maintains persistent connections for natural, conversational experiences.

### What We'll Cover

- Low-latency multimodal communication
- Speech-to-speech interactions
- Audio input/output support
- Real-time transcription
- Comparison with traditional APIs

### Prerequisites

- Understanding of streaming concepts
- Basic knowledge of audio formats
- Familiarity with async programming

---

## Low-Latency Multimodal Communication

### Understanding Latency Requirements

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class CommunicationMode(Enum):
    TEXT_ONLY = "text"
    AUDIO_INPUT = "audio_in"
    AUDIO_OUTPUT = "audio_out"
    AUDIO_BIDIRECTIONAL = "audio_bidirectional"
    MULTIMODAL = "multimodal"


@dataclass
class LatencyMetrics:
    """Track latency for realtime communication."""
    
    time_to_first_byte_ms: float
    total_response_time_ms: float
    audio_processing_ms: float = 0.0
    model_inference_ms: float = 0.0
    
    @property
    def perceived_latency(self) -> float:
        """Perceived latency is time to first meaningful output."""
        return self.time_to_first_byte_ms + self.audio_processing_ms
    
    def is_conversational(self) -> bool:
        """Check if latency is acceptable for conversation."""
        # Under 500ms feels natural
        return self.perceived_latency < 500


@dataclass
class RealtimeCapabilities:
    """Capabilities of realtime communication."""
    
    mode: CommunicationMode
    input_modalities: List[str]
    output_modalities: List[str]
    supports_interruption: bool
    supports_transcription: bool
    max_audio_duration_seconds: int


# Define realtime vs traditional capabilities
REALTIME_CAPABILITIES = RealtimeCapabilities(
    mode=CommunicationMode.AUDIO_BIDIRECTIONAL,
    input_modalities=["audio", "text"],
    output_modalities=["audio", "text"],
    supports_interruption=True,
    supports_transcription=True,
    max_audio_duration_seconds=300
)

TRADITIONAL_CAPABILITIES = RealtimeCapabilities(
    mode=CommunicationMode.TEXT_ONLY,
    input_modalities=["text"],
    output_modalities=["text"],
    supports_interruption=False,
    supports_transcription=False,
    max_audio_duration_seconds=0
)


def compare_approaches() -> dict:
    """Compare realtime vs traditional API approaches."""
    
    return {
        "realtime": {
            "latency": "200-500ms to first audio",
            "connection": "Persistent WebSocket/WebRTC",
            "flow": "Bidirectional streaming",
            "interruption": "Natural handling",
            "use_case": "Voice assistants, live chat"
        },
        "traditional": {
            "latency": "1-3 seconds per exchange",
            "connection": "Per-request HTTP",
            "flow": "Request-response",
            "interruption": "Not supported",
            "use_case": "Batch processing, async tasks"
        }
    }


# Usage
comparison = compare_approaches()
print("Realtime API:")
for key, value in comparison["realtime"].items():
    print(f"  {key}: {value}")
```

### Realtime Session Configuration

```python
from openai import OpenAI

client = OpenAI()


@dataclass
class RealtimeConfig:
    """Configuration for realtime session."""
    
    model: str = "gpt-4o-realtime-preview"
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: bool = True
    turn_detection: str = "server_vad"  # Server-side Voice Activity Detection
    instructions: str = ""
    temperature: float = 0.8
    max_response_output_tokens: int = 4096
    
    def to_session_config(self) -> dict:
        """Convert to API session configuration."""
        
        config = {
            "model": self.model,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": {
                "type": self.turn_detection
            },
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens
        }
        
        if self.input_audio_transcription:
            config["input_audio_transcription"] = {
                "model": "whisper-1"
            }
        
        if self.instructions:
            config["instructions"] = self.instructions
        
        return config


class RealtimeSessionManager:
    """Manage realtime API sessions."""
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.session_id: Optional[str] = None
        self.connection_status = "disconnected"
        self.metrics: List[LatencyMetrics] = []
    
    def get_session_config(self) -> dict:
        """Get configuration for session creation."""
        return self.config.to_session_config()
    
    def record_metrics(
        self,
        ttfb_ms: float,
        total_ms: float,
        audio_ms: float = 0
    ):
        """Record latency metrics."""
        
        metrics = LatencyMetrics(
            time_to_first_byte_ms=ttfb_ms,
            total_response_time_ms=total_ms,
            audio_processing_ms=audio_ms
        )
        
        self.metrics.append(metrics)
        
        if not metrics.is_conversational():
            print(f"Warning: High latency detected: {metrics.perceived_latency}ms")
    
    def get_average_latency(self) -> float:
        """Get average perceived latency."""
        
        if not self.metrics:
            return 0
        
        total = sum(m.perceived_latency for m in self.metrics)
        return total / len(self.metrics)


# Usage
config = RealtimeConfig(
    model="gpt-4o-realtime-preview",
    voice="alloy",
    instructions="You are a helpful voice assistant.",
    turn_detection="server_vad"
)

manager = RealtimeSessionManager(config)
print(f"Session config: {manager.get_session_config()}")
```

---

## Speech-to-Speech Interactions

### Direct Speech Pipeline

```python
class SpeechInteractionMode(Enum):
    SPEECH_TO_TEXT = "stt"  # Audio in, text out
    TEXT_TO_SPEECH = "tts"  # Text in, audio out
    SPEECH_TO_SPEECH = "sts"  # Audio in, audio out


@dataclass
class SpeechInteraction:
    """A speech-to-speech interaction."""
    
    mode: SpeechInteractionMode
    input_audio: Optional[bytes] = None
    input_text: Optional[str] = None
    output_audio: Optional[bytes] = None
    output_text: Optional[str] = None
    transcript: Optional[str] = None
    latency_ms: float = 0
    
    @property
    def has_audio_output(self) -> bool:
        return self.output_audio is not None


class SpeechToSpeechHandler:
    """Handle speech-to-speech interactions."""
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.client = OpenAI()
        self.interactions: List[SpeechInteraction] = []
    
    def process_audio_input(
        self,
        audio_data: bytes,
        audio_format: str = "pcm16"
    ) -> SpeechInteraction:
        """Process audio input and get speech output."""
        
        start_time = datetime.now()
        
        # In realtime API, this happens over WebSocket
        # This is a conceptual representation
        interaction = SpeechInteraction(
            mode=SpeechInteractionMode.SPEECH_TO_SPEECH,
            input_audio=audio_data
        )
        
        # The realtime API handles:
        # 1. Voice Activity Detection (VAD)
        # 2. Automatic transcription
        # 3. Model inference
        # 4. Audio generation
        
        # Track interaction
        self.interactions.append(interaction)
        
        return interaction
    
    def get_conversation_transcript(self) -> str:
        """Get full conversation transcript."""
        
        lines = []
        for i, interaction in enumerate(self.interactions):
            if interaction.transcript:
                lines.append(f"User: {interaction.transcript}")
            if interaction.output_text:
                lines.append(f"Assistant: {interaction.output_text}")
        
        return "\n".join(lines)


# The key advantage of speech-to-speech:
# No intermediate text step = lower latency
print("Speech-to-Speech advantages:")
print("1. Direct audio processing")
print("2. Preserved prosody and emotion")
print("3. Lower latency (no STT→LLM→TTS chain)")
print("4. More natural conversation flow")
```

### Voice Activity Detection

```python
class VADMode(Enum):
    SERVER = "server_vad"  # Server detects speech
    CLIENT = "client_vad"  # Client sends speech boundaries
    MANUAL = "manual"  # Manual turn control


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    
    mode: VADMode
    threshold: float = 0.5  # Sensitivity
    prefix_padding_ms: int = 300  # Audio before speech
    silence_duration_ms: int = 500  # Silence to end turn
    
    def to_dict(self) -> dict:
        """Convert to API format."""
        
        if self.mode == VADMode.SERVER:
            return {
                "type": "server_vad",
                "threshold": self.threshold,
                "prefix_padding_ms": self.prefix_padding_ms,
                "silence_duration_ms": self.silence_duration_ms
            }
        elif self.mode == VADMode.CLIENT:
            return {"type": "client_vad"}
        else:
            return {"type": "none"}


class VADHandler:
    """Handle Voice Activity Detection."""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.is_speaking = False
        self.speech_start_time: Optional[datetime] = None
        self.last_speech_time: Optional[datetime] = None
    
    def on_speech_start(self):
        """Called when speech is detected."""
        self.is_speaking = True
        self.speech_start_time = datetime.now()
        print("Speech started")
    
    def on_speech_end(self):
        """Called when speech ends."""
        self.is_speaking = False
        self.last_speech_time = datetime.now()
        
        if self.speech_start_time:
            duration = (self.last_speech_time - self.speech_start_time).total_seconds()
            print(f"Speech ended after {duration:.2f}s")
    
    def get_speech_duration(self) -> float:
        """Get current speech duration in seconds."""
        
        if not self.is_speaking or not self.speech_start_time:
            return 0
        
        return (datetime.now() - self.speech_start_time).total_seconds()


# Usage
vad_config = VADConfig(
    mode=VADMode.SERVER,
    threshold=0.5,
    silence_duration_ms=500
)

print(f"VAD config: {vad_config.to_dict()}")
```

---

## Audio Input/Output Support

### Audio Format Configuration

```python
class AudioFormat(Enum):
    PCM16 = "pcm16"  # 16-bit PCM, recommended for realtime
    G711_ULAW = "g711_ulaw"  # μ-law for telephony
    G711_ALAW = "g711_alaw"  # A-law for telephony


@dataclass
class AudioConfig:
    """Audio configuration for realtime sessions."""
    
    format: AudioFormat
    sample_rate: int = 24000  # 24kHz for realtime
    channels: int = 1  # Mono
    bits_per_sample: int = 16
    
    @property
    def bytes_per_second(self) -> int:
        """Calculate bytes per second of audio."""
        return self.sample_rate * self.channels * (self.bits_per_sample // 8)
    
    @property
    def bytes_per_ms(self) -> float:
        """Calculate bytes per millisecond."""
        return self.bytes_per_second / 1000


class AudioStreamHandler:
    """Handle audio streaming for realtime API."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.input_buffer: bytes = b""
        self.output_buffer: bytes = b""
        self.chunk_duration_ms: int = 20  # 20ms chunks typical
    
    def get_chunk_size(self) -> int:
        """Get size of audio chunk in bytes."""
        return int(self.config.bytes_per_ms * self.chunk_duration_ms)
    
    def add_input_audio(self, audio_data: bytes):
        """Add audio to input buffer."""
        self.input_buffer += audio_data
    
    def get_next_input_chunk(self) -> Optional[bytes]:
        """Get next chunk for sending."""
        
        chunk_size = self.get_chunk_size()
        
        if len(self.input_buffer) < chunk_size:
            return None
        
        chunk = self.input_buffer[:chunk_size]
        self.input_buffer = self.input_buffer[chunk_size:]
        
        return chunk
    
    def add_output_audio(self, audio_data: bytes):
        """Add received audio to output buffer."""
        self.output_buffer += audio_data
    
    def get_output_duration_ms(self) -> float:
        """Get duration of buffered output audio."""
        return len(self.output_buffer) / self.config.bytes_per_ms
    
    def clear_output_buffer(self):
        """Clear output buffer (e.g., on interruption)."""
        self.output_buffer = b""


# Usage
audio_config = AudioConfig(
    format=AudioFormat.PCM16,
    sample_rate=24000,
    channels=1
)

stream_handler = AudioStreamHandler(audio_config)
print(f"Chunk size: {stream_handler.get_chunk_size()} bytes")
print(f"Bytes per second: {audio_config.bytes_per_second}")
```

### Audio Format Conversion

```python
import base64


class AudioConverter:
    """Convert between audio formats for realtime API."""
    
    @staticmethod
    def pcm16_to_base64(pcm_data: bytes) -> str:
        """Convert PCM16 audio to base64 for API."""
        return base64.b64encode(pcm_data).decode('utf-8')
    
    @staticmethod
    def base64_to_pcm16(base64_data: str) -> bytes:
        """Convert base64 audio from API to PCM16."""
        return base64.b64decode(base64_data)
    
    @staticmethod
    def calculate_duration_ms(
        audio_bytes: bytes,
        sample_rate: int = 24000,
        channels: int = 1,
        bits_per_sample: int = 16
    ) -> float:
        """Calculate audio duration in milliseconds."""
        
        bytes_per_sample = bits_per_sample // 8
        total_samples = len(audio_bytes) // (bytes_per_sample * channels)
        
        return (total_samples / sample_rate) * 1000


# Usage
converter = AudioConverter()

# Example: Prepare audio for API
raw_audio = b"\x00\x01" * 24000  # 1 second of audio at 24kHz
base64_audio = converter.pcm16_to_base64(raw_audio)
print(f"Base64 audio length: {len(base64_audio)}")

# Calculate duration
duration = converter.calculate_duration_ms(raw_audio)
print(f"Audio duration: {duration}ms")
```

---

## Real-Time Transcription

### Transcription Configuration

```python
@dataclass
class TranscriptionConfig:
    """Configuration for real-time transcription."""
    
    enabled: bool = True
    model: str = "whisper-1"
    language: Optional[str] = None  # Auto-detect if None
    include_timestamps: bool = False


@dataclass
class TranscriptionResult:
    """Result of real-time transcription."""
    
    text: str
    confidence: float
    language: str
    is_final: bool
    start_time_ms: float = 0
    end_time_ms: float = 0


class RealtimeTranscriber:
    """Handle real-time transcription."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.partial_results: List[TranscriptionResult] = []
        self.final_results: List[TranscriptionResult] = []
    
    def on_partial_transcript(
        self,
        text: str,
        confidence: float = 0.0
    ):
        """Handle partial (interim) transcript."""
        
        result = TranscriptionResult(
            text=text,
            confidence=confidence,
            language="auto",
            is_final=False
        )
        
        self.partial_results.append(result)
        print(f"Partial: {text}")
    
    def on_final_transcript(
        self,
        text: str,
        confidence: float,
        language: str
    ):
        """Handle final transcript."""
        
        result = TranscriptionResult(
            text=text,
            confidence=confidence,
            language=language,
            is_final=True
        )
        
        self.final_results.append(result)
        self.partial_results.clear()
        
        print(f"Final ({language}, {confidence:.2f}): {text}")
    
    def get_full_transcript(self) -> str:
        """Get complete transcript from final results."""
        return " ".join(r.text for r in self.final_results)
    
    def get_current_state(self) -> dict:
        """Get current transcription state."""
        
        return {
            "final_text": self.get_full_transcript(),
            "partial_text": self.partial_results[-1].text if self.partial_results else "",
            "turn_count": len(self.final_results)
        }


# Usage
transcription_config = TranscriptionConfig(
    enabled=True,
    model="whisper-1",
    language=None  # Auto-detect
)

transcriber = RealtimeTranscriber(transcription_config)

# Simulate receiving transcripts
transcriber.on_partial_transcript("Hello")
transcriber.on_partial_transcript("Hello, how")
transcriber.on_final_transcript("Hello, how are you?", 0.95, "en")

print(f"\nFull transcript: {transcriber.get_full_transcript()}")
```

---

## Hands-on Exercise

### Your Task

Build a realtime session configuration manager.

### Requirements

1. Configure audio formats and VAD
2. Handle latency tracking
3. Manage transcription settings
4. Support multiple voice options

<details>
<summary>💡 Hints</summary>

- Use dataclasses for configuration
- Track metrics for optimization
- Consider telephony vs web use cases
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta


class Voice(Enum):
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"


class UseCase(Enum):
    VOICE_ASSISTANT = "voice_assistant"
    CUSTOMER_SERVICE = "customer_service"
    TELEPHONY = "telephony"
    INTERACTIVE_DEMO = "demo"


@dataclass
class RealtimeSessionConfig:
    """Complete realtime session configuration."""
    
    # Core settings
    model: str = "gpt-4o-realtime-preview"
    voice: Voice = Voice.ALLOY
    instructions: str = ""
    
    # Audio settings
    input_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_format: AudioFormat = AudioFormat.PCM16
    sample_rate: int = 24000
    
    # VAD settings
    vad_mode: VADMode = VADMode.SERVER
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500
    
    # Transcription
    transcription_enabled: bool = True
    transcription_model: str = "whisper-1"
    transcription_language: Optional[str] = None
    
    # Generation
    temperature: float = 0.8
    max_response_tokens: int = 4096


class RealtimeConfigBuilder:
    """Builder for realtime configurations."""
    
    def __init__(self):
        self.config = RealtimeSessionConfig()
    
    def for_use_case(self, use_case: UseCase) -> 'RealtimeConfigBuilder':
        """Configure for specific use case."""
        
        if use_case == UseCase.VOICE_ASSISTANT:
            self.config.voice = Voice.ALLOY
            self.config.vad_mode = VADMode.SERVER
            self.config.silence_duration_ms = 500
            self.config.temperature = 0.8
            
        elif use_case == UseCase.CUSTOMER_SERVICE:
            self.config.voice = Voice.CORAL
            self.config.vad_mode = VADMode.SERVER
            self.config.silence_duration_ms = 700  # More patience
            self.config.temperature = 0.6  # More consistent
            
        elif use_case == UseCase.TELEPHONY:
            self.config.input_audio_format = AudioFormat.G711_ULAW
            self.config.output_audio_format = AudioFormat.G711_ULAW
            self.config.sample_rate = 8000
            self.config.voice = Voice.SAGE
            
        elif use_case == UseCase.INTERACTIVE_DEMO:
            self.config.voice = Voice.SHIMMER
            self.config.vad_mode = VADMode.SERVER
            self.config.temperature = 1.0  # More creative
        
        return self
    
    def with_voice(self, voice: Voice) -> 'RealtimeConfigBuilder':
        """Set voice."""
        self.config.voice = voice
        return self
    
    def with_instructions(self, instructions: str) -> 'RealtimeConfigBuilder':
        """Set system instructions."""
        self.config.instructions = instructions
        return self
    
    def with_transcription(
        self,
        enabled: bool = True,
        language: str = None
    ) -> 'RealtimeConfigBuilder':
        """Configure transcription."""
        self.config.transcription_enabled = enabled
        self.config.transcription_language = language
        return self
    
    def with_vad(
        self,
        mode: VADMode,
        threshold: float = 0.5,
        silence_ms: int = 500
    ) -> 'RealtimeConfigBuilder':
        """Configure VAD."""
        self.config.vad_mode = mode
        self.config.vad_threshold = threshold
        self.config.silence_duration_ms = silence_ms
        return self
    
    def build(self) -> RealtimeSessionConfig:
        """Build the configuration."""
        return self.config


class RealtimeMetricsTracker:
    """Track realtime session metrics."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.turn_counts: int = 0
        self.audio_duration_ms: float = 0
        self.interruptions: int = 0
        self.transcription_errors: int = 0
        self.session_start: Optional[datetime] = None
    
    def start_session(self):
        """Start tracking session."""
        self.session_start = datetime.now()
    
    def record_turn(
        self,
        latency_ms: float,
        audio_duration_ms: float
    ):
        """Record a conversation turn."""
        self.latencies.append(latency_ms)
        self.turn_counts += 1
        self.audio_duration_ms += audio_duration_ms
    
    def record_interruption(self):
        """Record an interruption."""
        self.interruptions += 1
    
    def get_metrics(self) -> dict:
        """Get session metrics."""
        
        session_duration = None
        if self.session_start:
            session_duration = (datetime.now() - self.session_start).total_seconds()
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        p95_latency = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) >= 20 else avg_latency
        
        return {
            "session_duration_s": session_duration,
            "turn_count": self.turn_counts,
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "total_audio_ms": self.audio_duration_ms,
            "interruptions": self.interruptions,
            "interruption_rate": self.interruptions / max(self.turn_counts, 1)
        }


class RealtimeSessionManager:
    """Complete realtime session manager."""
    
    def __init__(self, config: RealtimeSessionConfig):
        self.config = config
        self.metrics = RealtimeMetricsTracker()
        self.audio_handler = AudioStreamHandler(
            AudioConfig(
                format=config.input_audio_format,
                sample_rate=config.sample_rate
            )
        )
        self.transcriber = RealtimeTranscriber(
            TranscriptionConfig(
                enabled=config.transcription_enabled,
                model=config.transcription_model,
                language=config.transcription_language
            )
        )
        
        self.session_id: Optional[str] = None
        self.connection_state = "disconnected"
    
    def get_api_config(self) -> dict:
        """Get configuration for API session creation."""
        
        config = {
            "model": self.config.model,
            "voice": self.config.voice.value,
            "input_audio_format": self.config.input_audio_format.value,
            "output_audio_format": self.config.output_audio_format.value,
            "turn_detection": {
                "type": self.config.vad_mode.value,
                "threshold": self.config.vad_threshold,
                "silence_duration_ms": self.config.silence_duration_ms
            },
            "temperature": self.config.temperature,
            "max_response_output_tokens": self.config.max_response_tokens
        }
        
        if self.config.transcription_enabled:
            config["input_audio_transcription"] = {
                "model": self.config.transcription_model
            }
            if self.config.transcription_language:
                config["input_audio_transcription"]["language"] = self.config.transcription_language
        
        if self.config.instructions:
            config["instructions"] = self.config.instructions
        
        return config
    
    def on_connect(self, session_id: str):
        """Handle successful connection."""
        self.session_id = session_id
        self.connection_state = "connected"
        self.metrics.start_session()
        print(f"Connected: {session_id}")
    
    def on_audio_received(self, audio_data: bytes):
        """Handle received audio."""
        self.audio_handler.add_output_audio(audio_data)
    
    def on_transcript_received(
        self,
        text: str,
        is_final: bool,
        confidence: float = 0.0
    ):
        """Handle received transcript."""
        if is_final:
            self.transcriber.on_final_transcript(text, confidence, "auto")
        else:
            self.transcriber.on_partial_transcript(text, confidence)
    
    def on_response_complete(self, latency_ms: float):
        """Handle response completion."""
        audio_duration = self.audio_handler.get_output_duration_ms()
        self.metrics.record_turn(latency_ms, audio_duration)
    
    def on_interruption(self):
        """Handle user interruption."""
        self.metrics.record_interruption()
        self.audio_handler.clear_output_buffer()
    
    def get_session_summary(self) -> dict:
        """Get session summary."""
        
        return {
            "session_id": self.session_id,
            "state": self.connection_state,
            "transcript": self.transcriber.get_full_transcript(),
            "metrics": self.metrics.get_metrics()
        }


# Usage
# Build configuration for customer service use case
config = (
    RealtimeConfigBuilder()
    .for_use_case(UseCase.CUSTOMER_SERVICE)
    .with_voice(Voice.CORAL)
    .with_instructions("You are a helpful customer service agent.")
    .with_transcription(enabled=True, language="en")
    .with_vad(VADMode.SERVER, threshold=0.5, silence_ms=700)
    .build()
)

# Create session manager
session = RealtimeSessionManager(config)

# Get API config
api_config = session.get_api_config()
print("API Configuration:")
print(f"  Model: {api_config['model']}")
print(f"  Voice: {api_config['voice']}")
print(f"  VAD: {api_config['turn_detection']['type']}")

# Simulate session events
session.on_connect("session_123")
session.on_transcript_received("Hello, I need help", True, 0.95)
session.on_response_complete(latency_ms=350)

# Get summary
summary = session.get_session_summary()
print(f"\nSession summary:")
print(f"  Transcript: {summary['transcript']}")
print(f"  Metrics: {summary['metrics']}")
```

</details>

---

## Summary

✅ Realtime API enables low-latency voice communication  
✅ Speech-to-speech reduces latency vs traditional pipelines  
✅ VAD automatically detects speech boundaries  
✅ Audio formats optimized for web (PCM16) and telephony (G.711)  
✅ Real-time transcription provides text alongside audio  
✅ Latency under 500ms enables natural conversation

**Next:** [Connection Methods](./02-connection-methods.md)

---

## Further Reading

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Official documentation
- [WebRTC Basics](https://webrtc.org/getting-started/overview) — Browser-based communication
- [Voice Activity Detection](https://en.wikipedia.org/wiki/Voice_activity_detection) — VAD concepts
