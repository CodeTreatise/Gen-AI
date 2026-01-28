---
title: "Audio in Responses API"
---

# Audio in Responses API

## Introduction

The OpenAI Responses API supports audio input and output through modality configuration. You can generate spoken responses, transcribe audio input, and select from multiple voice options.

### What We'll Cover

- Audio modalities configuration
- Audio input with `input_audio` content type
- Audio format support
- Voice selection options
- Audio token usage and pricing

### Prerequisites

- Understanding of Responses API basics
- Familiarity with audio formats
- Python SDK knowledge

---

## Audio Modalities Configuration

### Enabling Audio Output

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

client = OpenAI()


class Modality(Enum):
    TEXT = "text"
    AUDIO = "audio"


@dataclass
class AudioOutputConfig:
    """Configuration for audio output."""
    
    voice: str = "alloy"
    format: str = "wav"  # wav, mp3, flac, opus, pcm16
    speed: float = 1.0  # 0.25 to 4.0
    
    def to_dict(self) -> dict:
        return {
            "voice": self.voice,
            "format": self.format,
            "speed": self.speed
        }


def generate_audio_response(
    prompt: str,
    voice: str = "alloy"
) -> dict:
    """Generate a response with audio output."""
    
    response = client.responses.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={
            "voice": voice,
            "format": "wav"
        },
        input=prompt
    )
    
    return {
        "text": response.output_text,
        "audio_data": response.audio.data if response.audio else None,
        "audio_transcript": response.audio.transcript if response.audio else None
    }


# Usage
# result = generate_audio_response(
#     "Explain how neural networks work in simple terms.",
#     voice="sage"
# )
# print(f"Text: {result['text']}")
# if result['audio_data']:
#     print(f"Audio: {len(result['audio_data'])} bytes")


# Modality combinations
MODALITY_OPTIONS = {
    "text_only": {
        "modalities": ["text"],
        "description": "Standard text response"
    },
    "text_and_audio": {
        "modalities": ["text", "audio"],
        "description": "Text response with spoken audio"
    },
    "audio_only": {
        "modalities": ["audio"],
        "description": "Audio response only (for transcription input)"
    }
}

print("Available modality configurations:")
for name, config in MODALITY_OPTIONS.items():
    print(f"  {name}: {config['description']}")
```

### Audio Input Configuration

```python
@dataclass
class AudioInputConfig:
    """Configuration for audio input."""
    
    format: str = "wav"  # wav, mp3, flac, webm, ogg
    source: str = "file"  # file, base64, url
    transcription_model: str = "whisper-1"


class AudioInputHandler:
    """Handle audio input for Responses API."""
    
    def __init__(self, config: AudioInputConfig = None):
        self.config = config or AudioInputConfig()
        self.client = OpenAI()
    
    def from_file(self, file_path: str) -> bytes:
        """Read audio from file."""
        with open(file_path, "rb") as f:
            return f.read()
    
    def to_base64(self, audio_data: bytes) -> str:
        """Convert audio to base64."""
        import base64
        return base64.b64encode(audio_data).decode("utf-8")
    
    def create_audio_content(
        self,
        audio_data: bytes,
        format: str = None
    ) -> dict:
        """Create audio content for API request."""
        
        return {
            "type": "input_audio",
            "input_audio": {
                "data": self.to_base64(audio_data),
                "format": format or self.config.format
            }
        }
    
    def send_audio_message(
        self,
        audio_data: bytes,
        text_context: str = None
    ) -> dict:
        """Send audio with optional text context."""
        
        content = []
        
        # Add text context if provided
        if text_context:
            content.append({
                "type": "input_text",
                "text": text_context
            })
        
        # Add audio
        content.append(self.create_audio_content(audio_data))
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            input=content
        )
        
        return {
            "text": response.output_text,
            "audio": response.audio
        }


# Usage
handler = AudioInputHandler()

# Create audio content
# audio_bytes = handler.from_file("question.wav")
# content = handler.create_audio_content(audio_bytes, "wav")
# print(f"Audio content type: {content['type']}")
```

---

## Audio Format Support

### Supported Formats

```python
class AudioFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OPUS = "opus"
    OGG = "ogg"
    WEBM = "webm"
    PCM16 = "pcm16"


@dataclass
class FormatSpec:
    """Specification for an audio format."""
    
    format: AudioFormat
    mime_type: str
    sample_rates: List[int]
    channels: List[int]
    use_case: str
    file_extension: str


AUDIO_FORMATS = {
    AudioFormat.WAV: FormatSpec(
        format=AudioFormat.WAV,
        mime_type="audio/wav",
        sample_rates=[8000, 16000, 24000, 44100, 48000],
        channels=[1, 2],
        use_case="High quality, uncompressed audio",
        file_extension=".wav"
    ),
    AudioFormat.MP3: FormatSpec(
        format=AudioFormat.MP3,
        mime_type="audio/mpeg",
        sample_rates=[8000, 16000, 22050, 44100, 48000],
        channels=[1, 2],
        use_case="Compressed audio, smaller file size",
        file_extension=".mp3"
    ),
    AudioFormat.FLAC: FormatSpec(
        format=AudioFormat.FLAC,
        mime_type="audio/flac",
        sample_rates=[8000, 16000, 24000, 44100, 48000, 96000],
        channels=[1, 2, 6, 8],
        use_case="Lossless compression, archival quality",
        file_extension=".flac"
    ),
    AudioFormat.OPUS: FormatSpec(
        format=AudioFormat.OPUS,
        mime_type="audio/opus",
        sample_rates=[8000, 12000, 16000, 24000, 48000],
        channels=[1, 2],
        use_case="Real-time streaming, low latency",
        file_extension=".opus"
    ),
    AudioFormat.PCM16: FormatSpec(
        format=AudioFormat.PCM16,
        mime_type="audio/pcm",
        sample_rates=[8000, 16000, 24000],
        channels=[1],
        use_case="Raw audio for Realtime API",
        file_extension=".raw"
    )
}


def get_format_recommendation(use_case: str) -> AudioFormat:
    """Recommend format based on use case."""
    
    recommendations = {
        "web_streaming": AudioFormat.OPUS,
        "file_storage": AudioFormat.MP3,
        "high_quality": AudioFormat.FLAC,
        "realtime_api": AudioFormat.PCM16,
        "general": AudioFormat.WAV
    }
    
    return recommendations.get(use_case, AudioFormat.WAV)


# Display format info
print("Supported Audio Formats:")
print("-" * 50)
for fmt, spec in AUDIO_FORMATS.items():
    print(f"{spec.format.value}:")
    print(f"  MIME: {spec.mime_type}")
    print(f"  Use: {spec.use_case}")
    print()
```

### Format Conversion

```python
class AudioFormatConverter:
    """Convert between audio formats."""
    
    def __init__(self):
        self.supported_input = [
            AudioFormat.WAV, AudioFormat.MP3, 
            AudioFormat.FLAC, AudioFormat.OGG
        ]
        self.supported_output = [
            AudioFormat.WAV, AudioFormat.MP3,
            AudioFormat.OPUS, AudioFormat.FLAC
        ]
    
    def get_format_from_header(self, audio_data: bytes) -> Optional[AudioFormat]:
        """Detect format from file header."""
        
        # Check magic bytes
        if audio_data[:4] == b'RIFF':
            return AudioFormat.WAV
        elif audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
            return AudioFormat.MP3
        elif audio_data[:4] == b'fLaC':
            return AudioFormat.FLAC
        elif audio_data[:4] == b'OggS':
            return AudioFormat.OGG
        
        return None
    
    def validate_format(
        self,
        audio_data: bytes,
        expected_format: AudioFormat
    ) -> bool:
        """Validate audio format."""
        
        detected = self.get_format_from_header(audio_data)
        return detected == expected_format
    
    def to_api_format(
        self,
        audio_data: bytes,
        source_format: AudioFormat
    ) -> tuple[bytes, str]:
        """Convert to API-compatible format."""
        
        # API accepts most formats directly
        # Return as-is with format string
        return audio_data, source_format.value


# Usage
converter = AudioFormatConverter()

# Example: Detect format
sample_wav = b'RIFF\x00\x00\x00\x00WAVEfmt '
detected = converter.get_format_from_header(sample_wav)
print(f"Detected format: {detected.value if detected else 'unknown'}")
```

---

## Voice Selection

### Available Voices

```python
class Voice(Enum):
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"


@dataclass
class VoiceProfile:
    """Profile for a voice option."""
    
    voice: Voice
    description: str
    style: str
    best_for: List[str]
    gender_presentation: str


VOICE_PROFILES = {
    Voice.ALLOY: VoiceProfile(
        voice=Voice.ALLOY,
        description="Balanced and versatile",
        style="Neutral",
        best_for=["General purpose", "Documentation", "Tutorials"],
        gender_presentation="Neutral"
    ),
    Voice.ASH: VoiceProfile(
        voice=Voice.ASH,
        description="Warm and approachable",
        style="Conversational",
        best_for=["Customer service", "Friendly chat", "Support"],
        gender_presentation="Masculine"
    ),
    Voice.BALLAD: VoiceProfile(
        voice=Voice.BALLAD,
        description="Expressive and dramatic",
        style="Storytelling",
        best_for=["Narration", "Stories", "Audiobooks"],
        gender_presentation="Neutral"
    ),
    Voice.CORAL: VoiceProfile(
        voice=Voice.CORAL,
        description="Clear and articulate",
        style="Professional",
        best_for=["Instructions", "Educational", "Business"],
        gender_presentation="Feminine"
    ),
    Voice.ECHO: VoiceProfile(
        voice=Voice.ECHO,
        description="Authoritative and confident",
        style="Commanding",
        best_for=["Announcements", "News", "Formal content"],
        gender_presentation="Masculine"
    ),
    Voice.SAGE: VoiceProfile(
        voice=Voice.SAGE,
        description="Calm and reassuring",
        style="Soothing",
        best_for=["Meditation", "Support", "Healthcare"],
        gender_presentation="Neutral"
    ),
    Voice.SHIMMER: VoiceProfile(
        voice=Voice.SHIMMER,
        description="Bright and friendly",
        style="Enthusiastic",
        best_for=["Marketing", "Entertainment", "Youth content"],
        gender_presentation="Feminine"
    ),
    Voice.VERSE: VoiceProfile(
        voice=Voice.VERSE,
        description="Dynamic and engaging",
        style="Energetic",
        best_for=["Entertainment", "Gaming", "Podcasts"],
        gender_presentation="Neutral"
    )
}


def recommend_voice(use_case: str) -> Voice:
    """Recommend voice based on use case."""
    
    use_case_mapping = {
        "customer_service": Voice.ASH,
        "narration": Voice.BALLAD,
        "education": Voice.CORAL,
        "announcements": Voice.ECHO,
        "meditation": Voice.SAGE,
        "marketing": Voice.SHIMMER,
        "gaming": Voice.VERSE,
        "general": Voice.ALLOY
    }
    
    return use_case_mapping.get(use_case.lower(), Voice.ALLOY)


# Display voice profiles
print("Available Voices:")
print("=" * 60)
for voice, profile in VOICE_PROFILES.items():
    print(f"\n{voice.value.upper()}")
    print(f"  Style: {profile.style}")
    print(f"  Best for: {', '.join(profile.best_for)}")
```

### Voice Configuration in Requests

```python
class VoiceConfigBuilder:
    """Build voice configuration for API requests."""
    
    def __init__(self, voice: Voice = Voice.ALLOY):
        self.voice = voice
        self.format = "wav"
        self.speed = 1.0
    
    def with_voice(self, voice: Voice) -> 'VoiceConfigBuilder':
        """Set voice."""
        self.voice = voice
        return self
    
    def with_format(self, format: str) -> 'VoiceConfigBuilder':
        """Set output format."""
        if format not in ["wav", "mp3", "flac", "opus", "pcm16"]:
            raise ValueError(f"Unsupported format: {format}")
        self.format = format
        return self
    
    def with_speed(self, speed: float) -> 'VoiceConfigBuilder':
        """Set speech speed (0.25 to 4.0)."""
        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        self.speed = speed
        return self
    
    def build(self) -> dict:
        """Build configuration dict."""
        return {
            "voice": self.voice.value,
            "format": self.format,
            "speed": self.speed
        }


# Usage
voice_config = (
    VoiceConfigBuilder()
    .with_voice(Voice.CORAL)
    .with_format("mp3")
    .with_speed(1.1)
    .build()
)

print(f"Voice config: {voice_config}")
```

---

## Audio Token Usage and Pricing

### Understanding Audio Tokens

```python
@dataclass
class AudioTokenUsage:
    """Track audio token usage."""
    
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0
    
    @property
    def total_input_tokens(self) -> int:
        return self.input_text_tokens + self.input_audio_tokens
    
    @property
    def total_output_tokens(self) -> int:
        return self.output_text_tokens + self.output_audio_tokens
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class AudioPricing:
    """Pricing for audio tokens."""
    
    input_text_per_1m: float = 2.50  # per 1M tokens
    input_audio_per_1m: float = 100.0  # per 1M tokens
    output_text_per_1m: float = 10.0  # per 1M tokens
    output_audio_per_1m: float = 200.0  # per 1M tokens
    
    def calculate_cost(self, usage: AudioTokenUsage) -> dict:
        """Calculate cost for token usage."""
        
        input_text_cost = (usage.input_text_tokens / 1_000_000) * self.input_text_per_1m
        input_audio_cost = (usage.input_audio_tokens / 1_000_000) * self.input_audio_per_1m
        output_text_cost = (usage.output_text_tokens / 1_000_000) * self.output_text_per_1m
        output_audio_cost = (usage.output_audio_tokens / 1_000_000) * self.output_audio_per_1m
        
        return {
            "input_text": input_text_cost,
            "input_audio": input_audio_cost,
            "output_text": output_text_cost,
            "output_audio": output_audio_cost,
            "total": input_text_cost + input_audio_cost + output_text_cost + output_audio_cost
        }


class AudioCostTracker:
    """Track audio API costs."""
    
    def __init__(self, pricing: AudioPricing = None):
        self.pricing = pricing or AudioPricing()
        self.usage_history: List[AudioTokenUsage] = []
    
    def record_usage(self, response) -> AudioTokenUsage:
        """Record usage from API response."""
        
        usage = AudioTokenUsage(
            input_text_tokens=getattr(response.usage, 'input_tokens', 0),
            input_audio_tokens=getattr(response.usage, 'input_audio_tokens', 0),
            output_text_tokens=getattr(response.usage, 'output_tokens', 0),
            output_audio_tokens=getattr(response.usage, 'output_audio_tokens', 0)
        )
        
        self.usage_history.append(usage)
        return usage
    
    def get_total_usage(self) -> AudioTokenUsage:
        """Get total usage across all calls."""
        
        total = AudioTokenUsage()
        
        for usage in self.usage_history:
            total.input_text_tokens += usage.input_text_tokens
            total.input_audio_tokens += usage.input_audio_tokens
            total.output_text_tokens += usage.output_text_tokens
            total.output_audio_tokens += usage.output_audio_tokens
        
        return total
    
    def get_total_cost(self) -> dict:
        """Get total cost."""
        return self.pricing.calculate_cost(self.get_total_usage())
    
    def get_summary(self) -> dict:
        """Get usage summary."""
        
        total_usage = self.get_total_usage()
        costs = self.get_total_cost()
        
        return {
            "call_count": len(self.usage_history),
            "total_tokens": total_usage.total_tokens,
            "breakdown": {
                "input_text_tokens": total_usage.input_text_tokens,
                "input_audio_tokens": total_usage.input_audio_tokens,
                "output_text_tokens": total_usage.output_text_tokens,
                "output_audio_tokens": total_usage.output_audio_tokens
            },
            "estimated_cost_usd": costs["total"]
        }


# Example usage calculation
example_usage = AudioTokenUsage(
    input_text_tokens=1000,
    input_audio_tokens=5000,  # ~30 seconds of audio
    output_text_tokens=500,
    output_audio_tokens=10000  # ~1 minute of audio output
)

pricing = AudioPricing()
cost = pricing.calculate_cost(example_usage)

print("Example Audio Cost Calculation:")
print(f"  Input text: ${cost['input_text']:.4f}")
print(f"  Input audio: ${cost['input_audio']:.4f}")
print(f"  Output text: ${cost['output_text']:.4f}")
print(f"  Output audio: ${cost['output_audio']:.4f}")
print(f"  Total: ${cost['total']:.4f}")
```

---

## Complete Audio Client

### Audio-Enabled Responses Client

```python
class AudioResponsesClient:
    """Client for audio-enabled Responses API."""
    
    def __init__(self, default_voice: Voice = Voice.ALLOY):
        self.client = OpenAI()
        self.default_voice = default_voice
        self.cost_tracker = AudioCostTracker()
        self.audio_handler = AudioInputHandler()
    
    def text_to_audio(
        self,
        text: str,
        voice: Voice = None,
        format: str = "wav"
    ) -> dict:
        """Convert text to audio response."""
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={
                "voice": (voice or self.default_voice).value,
                "format": format
            },
            input=text
        )
        
        self.cost_tracker.record_usage(response)
        
        return {
            "text": response.output_text,
            "audio_data": response.audio.data if response.audio else None,
            "transcript": response.audio.transcript if response.audio else None
        }
    
    def audio_to_text(
        self,
        audio_data: bytes,
        audio_format: str = "wav"
    ) -> dict:
        """Process audio input, get text response."""
        
        content = [{
            "type": "input_audio",
            "input_audio": {
                "data": self.audio_handler.to_base64(audio_data),
                "format": audio_format
            }
        }]
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            input=content
        )
        
        self.cost_tracker.record_usage(response)
        
        return {
            "text": response.output_text,
            "input_transcript": getattr(response, 'input_audio_transcript', None)
        }
    
    def audio_to_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        voice: Voice = None,
        output_format: str = "wav"
    ) -> dict:
        """Process audio input, get audio response."""
        
        content = [{
            "type": "input_audio",
            "input_audio": {
                "data": self.audio_handler.to_base64(audio_data),
                "format": audio_format
            }
        }]
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={
                "voice": (voice or self.default_voice).value,
                "format": output_format
            },
            input=content
        )
        
        self.cost_tracker.record_usage(response)
        
        return {
            "text": response.output_text,
            "audio_data": response.audio.data if response.audio else None,
            "input_transcript": getattr(response, 'input_audio_transcript', None),
            "output_transcript": response.audio.transcript if response.audio else None
        }
    
    def get_cost_summary(self) -> dict:
        """Get cost summary for all calls."""
        return self.cost_tracker.get_summary()


# Usage
audio_client = AudioResponsesClient(default_voice=Voice.CORAL)

# Text to audio
# result = audio_client.text_to_audio(
#     "Welcome to our customer service line. How may I help you today?",
#     voice=Voice.CORAL
# )
# print(f"Generated audio: {len(result['audio_data'])} bytes")

# Cost summary
# summary = audio_client.get_cost_summary()
# print(f"Total cost: ${summary['estimated_cost_usd']:.4f}")

print("Audio Responses Client initialized")
print(f"Default voice: {audio_client.default_voice.value}")
```

---

## Hands-on Exercise

### Your Task

Build an audio transcription and response system.

### Requirements

1. Accept audio input in multiple formats
2. Generate spoken responses
3. Track token usage and costs
4. Support voice selection

<details>
<summary>💡 Hints</summary>

- Validate audio format before sending
- Cache voice configurations
- Aggregate costs across calls
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import base64
import json


class AudioTranscriptionSystem:
    """Complete audio transcription and response system."""
    
    def __init__(self):
        self.client = OpenAI()
        
        # Configuration
        self.supported_input_formats = ["wav", "mp3", "flac", "ogg", "webm"]
        self.supported_output_formats = ["wav", "mp3", "opus", "flac"]
        
        # Voice configuration
        self.voice_cache: Dict[str, dict] = {}
        self.default_voice = Voice.ALLOY
        
        # Tracking
        self.cost_tracker = AudioCostTracker()
        self.conversation_history: List[dict] = []
    
    def configure_voice(
        self,
        name: str,
        voice: Voice,
        format: str = "wav",
        speed: float = 1.0
    ):
        """Configure and cache a voice setting."""
        
        self.voice_cache[name] = {
            "voice": voice.value,
            "format": format,
            "speed": speed
        }
    
    def get_voice_config(self, name: str = None) -> dict:
        """Get voice configuration."""
        
        if name and name in self.voice_cache:
            return self.voice_cache[name]
        
        return {
            "voice": self.default_voice.value,
            "format": "wav"
        }
    
    def validate_audio(
        self,
        audio_data: bytes,
        claimed_format: str
    ) -> tuple[bool, str]:
        """Validate audio data and format."""
        
        if not audio_data:
            return False, "Empty audio data"
        
        if claimed_format not in self.supported_input_formats:
            return False, f"Unsupported format: {claimed_format}"
        
        # Basic validation - check minimum size
        if len(audio_data) < 100:
            return False, "Audio data too small"
        
        return True, "Valid"
    
    def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav"
    ) -> dict:
        """Transcribe audio to text only."""
        
        # Validate
        valid, message = self.validate_audio(audio_data, audio_format)
        if not valid:
            return {"error": message}
        
        content = [{
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_data).decode("utf-8"),
                "format": audio_format
            }
        }]
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            input=content
        )
        
        self.cost_tracker.record_usage(response)
        
        return {
            "transcript": response.output_text,
            "model_response": response.output_text
        }
    
    def process_audio_conversation(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        voice_config_name: str = None,
        context: str = None
    ) -> dict:
        """Process audio input and generate audio response."""
        
        # Validate input
        valid, message = self.validate_audio(audio_data, audio_format)
        if not valid:
            return {"error": message}
        
        # Build input content
        content = []
        
        if context:
            content.append({
                "type": "input_text",
                "text": context
            })
        
        content.append({
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_data).decode("utf-8"),
                "format": audio_format
            }
        })
        
        # Get voice config
        voice_config = self.get_voice_config(voice_config_name)
        
        # Make request
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio=voice_config,
            input=content
        )
        
        # Track usage
        self.cost_tracker.record_usage(response)
        
        # Store in history
        result = {
            "timestamp": datetime.now().isoformat(),
            "input_audio_size": len(audio_data),
            "input_format": audio_format,
            "text_response": response.output_text,
            "audio_response": response.audio.data if response.audio else None,
            "audio_transcript": response.audio.transcript if response.audio else None,
            "voice_used": voice_config["voice"]
        }
        
        self.conversation_history.append(result)
        
        return result
    
    def generate_speech(
        self,
        text: str,
        voice_config_name: str = None
    ) -> dict:
        """Generate speech from text."""
        
        voice_config = self.get_voice_config(voice_config_name)
        
        response = self.client.responses.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio=voice_config,
            input=text
        )
        
        self.cost_tracker.record_usage(response)
        
        return {
            "text": response.output_text,
            "audio": response.audio.data if response.audio else None,
            "transcript": response.audio.transcript if response.audio else None
        }
    
    def get_conversation_transcript(self) -> str:
        """Get transcript of all conversations."""
        
        lines = []
        for entry in self.conversation_history:
            if entry.get("audio_transcript"):
                lines.append(f"User: [audio]")
            lines.append(f"Assistant: {entry['text_response']}")
        
        return "\n".join(lines)
    
    def get_usage_report(self) -> dict:
        """Get usage and cost report."""
        
        summary = self.cost_tracker.get_summary()
        
        return {
            "conversation_count": len(self.conversation_history),
            "total_calls": summary["call_count"],
            "token_usage": summary["breakdown"],
            "estimated_cost_usd": summary["estimated_cost_usd"],
            "voices_used": list(set(
                entry["voice_used"] 
                for entry in self.conversation_history
            ))
        }
    
    def export_session(self) -> str:
        """Export session data as JSON."""
        
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "conversation_history": self.conversation_history,
            "usage_report": self.get_usage_report()
        }, indent=2)


# Usage example
system = AudioTranscriptionSystem()

# Configure voices
system.configure_voice("customer_service", Voice.CORAL, "mp3", 1.0)
system.configure_voice("narrator", Voice.BALLAD, "wav", 0.9)
system.configure_voice("assistant", Voice.ALLOY, "opus", 1.1)

print("Audio Transcription System")
print("=" * 50)
print(f"Input formats: {system.supported_input_formats}")
print(f"Output formats: {system.supported_output_formats}")
print(f"Configured voices: {list(system.voice_cache.keys())}")

# Simulate conversation
# Normally you'd have actual audio data
mock_audio = b"\x00\x01" * 1000  # Placeholder

# Validate audio
valid, msg = system.validate_audio(mock_audio, "wav")
print(f"\nAudio validation: {msg}")

# Generate speech from text
# result = system.generate_speech(
#     "Hello! How can I help you today?",
#     voice_config_name="customer_service"
# )

# Get usage report
report = system.get_usage_report()
print(f"\nUsage Report:")
print(f"  Conversations: {report['conversation_count']}")
print(f"  Total calls: {report['total_calls']}")
print(f"  Estimated cost: ${report['estimated_cost_usd']:.4f}")
```

</details>

---

## Summary

✅ Modalities configure text and audio output  
✅ `input_audio` content type accepts multiple formats  
✅ Eight voices available for different use cases  
✅ Audio tokens priced separately from text  
✅ Track usage for cost management

**Next:** [Session Management](./05-session-management.md)

---

## Further Reading

- [OpenAI Audio Guide](https://platform.openai.com/docs/guides/audio) — Official audio documentation
- [Voice Options](https://platform.openai.com/docs/guides/text-to-speech) — Voice samples and selection
- [Audio Pricing](https://openai.com/pricing) — Current audio token pricing
