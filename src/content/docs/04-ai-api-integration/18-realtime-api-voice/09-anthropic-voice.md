---
title: "Anthropic Voice Integration"
---

# Anthropic Voice Integration

## Introduction

Anthropic Claude supports audio input for multimodal understanding. While Claude doesn't have native speech output, you can build voice applications by combining Claude's audio comprehension with third-party text-to-speech services.

### What We'll Cover

- Claude audio input capabilities
- Audio processing and transcription
- Voice-to-text-to-voice patterns
- Third-party TTS integration
- Building complete voice assistants

### Prerequisites

- Anthropic API access
- Understanding of audio formats
- Experience with async Python

---

## Claude Audio Input Capabilities

### Audio Input Support

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import base64
import httpx


class AudioInputFormat(Enum):
    """Supported audio input formats for Claude."""
    
    MP3 = "audio/mp3"
    WAV = "audio/wav"
    OGG = "audio/ogg"
    WEBM = "audio/webm"
    FLAC = "audio/flac"
    M4A = "audio/m4a"


@dataclass
class ClaudeAudioConfig:
    """Configuration for Claude audio input."""
    
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 1.0
    
    # Audio limits
    max_audio_duration_seconds: int = 600  # 10 minutes
    max_audio_size_mb: int = 25
    
    # Processing options
    include_transcription: bool = True
    analyze_audio_content: bool = True


# Models supporting audio input
AUDIO_CAPABLE_MODELS = {
    "claude-sonnet-4-20250514": {
        "description": "Claude Sonnet 4",
        "audio_support": True,
        "max_audio_files": 5
    },
    "claude-3-5-sonnet-20241022": {
        "description": "Claude 3.5 Sonnet v2",
        "audio_support": True,
        "max_audio_files": 5
    },
    "claude-3-5-sonnet-20240620": {
        "description": "Claude 3.5 Sonnet v1",
        "audio_support": False,
        "max_audio_files": 0
    },
    "claude-3-opus-20240229": {
        "description": "Claude 3 Opus",
        "audio_support": False,
        "max_audio_files": 0
    }
}


def validate_audio_file(
    audio_data: bytes,
    format: AudioInputFormat
) -> tuple[bool, str]:
    """Validate audio file for Claude."""
    
    # Check size
    size_mb = len(audio_data) / (1024 * 1024)
    if size_mb > 25:
        return False, f"Audio too large: {size_mb:.1f}MB (max 25MB)"
    
    # Check format
    valid_formats = [f.value for f in AudioInputFormat]
    if format.value not in valid_formats:
        return False, f"Unsupported format: {format.value}"
    
    return True, "Valid"


print("Audio-Capable Claude Models:")
for model, info in AUDIO_CAPABLE_MODELS.items():
    if info["audio_support"]:
        print(f"  ✓ {model}")
    else:
        print(f"  ✗ {model} (no audio support)")
```

### Sending Audio to Claude

```python
class ClaudeAudioClient:
    """Client for Claude audio input."""
    
    def __init__(
        self,
        api_key: str,
        config: ClaudeAudioConfig = None
    ):
        self.api_key = api_key
        self.config = config or ClaudeAudioConfig()
        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        )
    
    def _encode_audio(self, audio_data: bytes) -> str:
        """Encode audio to base64."""
        return base64.standard_b64encode(audio_data).decode("utf-8")
    
    def _build_audio_content(
        self,
        audio_data: bytes,
        media_type: str
    ) -> dict:
        """Build audio content block."""
        
        return {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": self._encode_audio(audio_data)
            }
        }
    
    async def transcribe(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav"
    ) -> str:
        """Transcribe audio to text."""
        
        content = [
            self._build_audio_content(audio_data, media_type),
            {
                "type": "text",
                "text": "Please transcribe this audio exactly as spoken. "
                       "Include any notable features like tone or emotion."
            }
        ]
        
        response = await self._send_message(content)
        return response
    
    async def analyze_audio(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav",
        question: str = None
    ) -> str:
        """Analyze audio content."""
        
        prompt = question or (
            "Analyze this audio. Describe:\n"
            "1. What is being said (transcription)\n"
            "2. The speaker's tone and emotion\n"
            "3. Any background sounds or context\n"
            "4. The likely purpose or context of this audio"
        )
        
        content = [
            self._build_audio_content(audio_data, media_type),
            {"type": "text", "text": prompt}
        ]
        
        response = await self._send_message(content)
        return response
    
    async def respond_to_audio(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav",
        system_prompt: str = None
    ) -> str:
        """Respond to audio input conversationally."""
        
        content = [
            self._build_audio_content(audio_data, media_type),
            {
                "type": "text",
                "text": "Please respond to what was said in this audio."
            }
        ]
        
        response = await self._send_message(
            content,
            system=system_prompt or "You are a helpful assistant responding to voice messages."
        )
        
        return response
    
    async def _send_message(
        self,
        content: List[dict],
        system: str = None
    ) -> str:
        """Send message to Claude."""
        
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{
                "role": "user",
                "content": content
            }]
        }
        
        if system:
            payload["system"] = system
        
        response = await self.client.post(
            "/v1/messages",
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        return data["content"][0]["text"]
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


# Usage example (configuration only)
config = ClaudeAudioConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    include_transcription=True
)

print(f"Claude Audio Client configured")
print(f"  Model: {config.model}")
print(f"  Max audio: {config.max_audio_size_mb}MB, {config.max_audio_duration_seconds}s")
```

---

## Voice-to-Text-to-Voice Pattern

### Architecture Overview

```python
class VoiceAssistantComponent(Enum):
    """Components in voice-to-text-to-voice pipeline."""
    
    AUDIO_INPUT = "audio_input"        # Microphone/audio source
    SPEECH_TO_TEXT = "speech_to_text"  # Transcription (Claude or Whisper)
    LLM_PROCESSING = "llm_processing"  # Claude response generation
    TEXT_TO_SPEECH = "text_to_speech"  # TTS (ElevenLabs, OpenAI, etc.)
    AUDIO_OUTPUT = "audio_output"      # Speaker/audio playback


@dataclass
class VoiceAssistantConfig:
    """Configuration for voice assistant."""
    
    # Claude config
    claude_model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    
    # STT options
    use_claude_for_transcription: bool = True
    fallback_stt: str = "whisper"  # openai whisper
    
    # TTS options
    tts_provider: str = "elevenlabs"  # elevenlabs, openai, google
    voice_id: str = ""
    speaking_rate: float = 1.0
    
    # Audio settings
    input_format: str = "wav"
    output_format: str = "mp3"
    sample_rate: int = 24000


class VoicePipeline:
    """Voice-to-text-to-voice pipeline."""
    
    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        
        # Component timing
        self.timings: Dict[str, float] = {}
    
    async def process_audio_input(
        self,
        audio_data: bytes,
        claude_client: 'ClaudeAudioClient',
        tts_provider: 'TTSProvider'
    ) -> bytes:
        """Process audio through full pipeline."""
        
        import time
        
        # Step 1: Speech-to-Text (using Claude)
        start = time.time()
        
        if self.config.use_claude_for_transcription:
            transcript = await claude_client.transcribe(
                audio_data,
                f"audio/{self.config.input_format}"
            )
        else:
            # Use fallback STT
            transcript = await self._fallback_transcribe(audio_data)
        
        self.timings["stt"] = time.time() - start
        
        # Step 2: Generate response with Claude
        start = time.time()
        
        response_text = await claude_client.respond_to_audio(
            audio_data,
            f"audio/{self.config.input_format}",
            self.config.system_prompt
        )
        
        self.timings["llm"] = time.time() - start
        
        # Step 3: Convert response to speech
        start = time.time()
        
        audio_response = await tts_provider.synthesize(
            response_text,
            voice_id=self.config.voice_id
        )
        
        self.timings["tts"] = time.time() - start
        
        return audio_response
    
    async def _fallback_transcribe(self, audio_data: bytes) -> str:
        """Fallback transcription using Whisper."""
        # Would integrate OpenAI Whisper API
        return "[Fallback transcription not implemented]"
    
    def get_latency_report(self) -> dict:
        """Get latency report."""
        
        total = sum(self.timings.values())
        
        return {
            "components": self.timings,
            "total_seconds": total,
            "bottleneck": max(self.timings, key=self.timings.get) if self.timings else None
        }


# Pipeline configuration
pipeline_config = VoiceAssistantConfig(
    claude_model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful voice assistant. Keep responses concise.",
    tts_provider="elevenlabs",
    voice_id="pNInz6obpgDQGcFmaJgB",  # Example voice ID
    speaking_rate=1.0
)

pipeline = VoicePipeline(pipeline_config)

print("Voice Pipeline configured:")
print(f"  Claude: {pipeline_config.claude_model}")
print(f"  TTS: {pipeline_config.tts_provider}")
```

---

## Third-Party TTS Integration

### TTS Provider Interface

```python
from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Abstract base for TTS providers."""
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str = None
    ) -> bytes:
        """Synthesize text to speech."""
        pass
    
    @abstractmethod
    def list_voices(self) -> List[dict]:
        """List available voices."""
        pass


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS integration."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.client = httpx.AsyncClient(
            headers={"xi-api-key": api_key}
        )
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = "eleven_monolingual_v1",
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ) -> bytes:
        """Synthesize text to speech."""
        
        voice = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default voice
        
        response = await self.client.post(
            f"{self.base_url}/text-to-speech/{voice}",
            json={
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost
                }
            }
        )
        
        response.raise_for_status()
        return response.content
    
    def list_voices(self) -> List[dict]:
        """List available voices."""
        # Synchronous for simplicity
        import httpx as sync_httpx
        
        response = sync_httpx.get(
            f"{self.base_url}/voices",
            headers={"xi-api-key": self.api_key}
        )
        
        data = response.json()
        return [
            {
                "voice_id": v["voice_id"],
                "name": v["name"],
                "category": v.get("category", "unknown")
            }
            for v in data.get("voices", [])
        ]
    
    async def close(self):
        await self.client.aclose()


class OpenAITTS(TTSProvider):
    """OpenAI TTS integration."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        model: str = "tts-1",
        speed: float = 1.0
    ) -> bytes:
        """Synthesize text to speech."""
        
        voice = voice_id or "alloy"
        
        response = await self.client.post(
            "/audio/speech",
            json={
                "model": model,
                "input": text,
                "voice": voice,
                "speed": speed
            }
        )
        
        response.raise_for_status()
        return response.content
    
    def list_voices(self) -> List[dict]:
        """List available voices."""
        return [
            {"voice_id": "alloy", "name": "Alloy", "category": "standard"},
            {"voice_id": "echo", "name": "Echo", "category": "standard"},
            {"voice_id": "fable", "name": "Fable", "category": "standard"},
            {"voice_id": "onyx", "name": "Onyx", "category": "standard"},
            {"voice_id": "nova", "name": "Nova", "category": "standard"},
            {"voice_id": "shimmer", "name": "Shimmer", "category": "standard"}
        ]
    
    async def close(self):
        await self.client.aclose()


class GoogleCloudTTS(TTSProvider):
    """Google Cloud TTS integration."""
    
    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path
        # Would use google-cloud-texttospeech library
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = None,
        language_code: str = "en-US",
        speaking_rate: float = 1.0
    ) -> bytes:
        """Synthesize text to speech."""
        
        # Placeholder - would use Google Cloud TTS API
        # from google.cloud import texttospeech
        
        raise NotImplementedError("Google Cloud TTS requires google-cloud-texttospeech")
    
    def list_voices(self) -> List[dict]:
        """List available voices."""
        return [
            {"voice_id": "en-US-Standard-A", "name": "Standard A", "category": "standard"},
            {"voice_id": "en-US-Wavenet-A", "name": "WaveNet A", "category": "premium"},
            {"voice_id": "en-US-Neural2-A", "name": "Neural2 A", "category": "premium"}
        ]


# Provider factory
class TTSProviderFactory:
    """Factory for TTS providers."""
    
    @staticmethod
    def create(
        provider: str,
        api_key: str = None,
        **kwargs
    ) -> TTSProvider:
        """Create TTS provider."""
        
        if provider == "elevenlabs":
            return ElevenLabsTTS(api_key)
        elif provider == "openai":
            return OpenAITTS(api_key)
        elif provider == "google":
            return GoogleCloudTTS(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Usage
print("TTS Providers:")
print("  - ElevenLabs: High-quality voice cloning")
print("  - OpenAI: Simple, reliable TTS")
print("  - Google Cloud: Multiple languages, Neural voices")
```

---

## Building a Complete Voice Assistant

### Full Integration

```python
class ClaudeVoiceAssistant:
    """Complete voice assistant using Claude + TTS."""
    
    def __init__(
        self,
        claude_api_key: str,
        tts_api_key: str,
        config: VoiceAssistantConfig = None
    ):
        self.config = config or VoiceAssistantConfig()
        
        # Initialize clients
        self.claude = ClaudeAudioClient(
            claude_api_key,
            ClaudeAudioConfig(model=self.config.claude_model)
        )
        
        self.tts = TTSProviderFactory.create(
            self.config.tts_provider,
            tts_api_key
        )
        
        # Conversation state
        self.conversation_history: List[dict] = []
        self.is_active = False
        
        # Metrics
        self.metrics = {
            "turns": 0,
            "total_audio_seconds": 0,
            "avg_response_time": 0
        }
    
    async def start(self):
        """Start the assistant."""
        self.is_active = True
        self.conversation_history = []
        print("Claude Voice Assistant started")
    
    async def stop(self):
        """Stop the assistant."""
        self.is_active = False
        await self.claude.close()
        await self.tts.close()
        print("Claude Voice Assistant stopped")
    
    async def process_voice_input(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav"
    ) -> bytes:
        """Process voice input and return voice response."""
        
        if not self.is_active:
            raise RuntimeError("Assistant not started")
        
        import time
        start_time = time.time()
        
        # Get Claude's response to the audio
        text_response = await self.claude.respond_to_audio(
            audio_data,
            media_type,
            self.config.system_prompt
        )
        
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": "[Audio input]"
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": text_response
        })
        
        # Convert to speech
        audio_response = await self.tts.synthesize(
            text_response,
            voice_id=self.config.voice_id
        )
        
        # Update metrics
        self.metrics["turns"] += 1
        response_time = time.time() - start_time
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["turns"] - 1) +
             response_time) / self.metrics["turns"]
        )
        
        return audio_response
    
    async def process_text_input(self, text: str) -> bytes:
        """Process text input and return voice response."""
        
        if not self.is_active:
            raise RuntimeError("Assistant not started")
        
        # Build message with history context
        messages = self.conversation_history + [
            {"role": "user", "content": text}
        ]
        
        # Get response from Claude
        response = await self.claude._send_message(
            [{"type": "text", "text": text}],
            system=self.config.system_prompt
        )
        
        # Update history
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Convert to speech
        audio_response = await self.tts.synthesize(
            response,
            voice_id=self.config.voice_id
        )
        
        self.metrics["turns"] += 1
        
        return audio_response
    
    def get_conversation_transcript(self) -> str:
        """Get text transcript of conversation."""
        
        lines = []
        for msg in self.conversation_history:
            lines.append(f"{msg['role'].title()}: {msg['content']}")
        
        return "\n".join(lines)
    
    def get_metrics(self) -> dict:
        """Get assistant metrics."""
        return {
            **self.metrics,
            "history_length": len(self.conversation_history)
        }


# Streaming variant for lower latency
class StreamingClaudeVoiceAssistant(ClaudeVoiceAssistant):
    """Voice assistant with streaming TTS for lower latency."""
    
    async def process_voice_input_streaming(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav",
        on_audio_chunk: Callable = None
    ):
        """Process input with streaming audio output."""
        
        if not self.is_active:
            raise RuntimeError("Assistant not started")
        
        # Get Claude's response
        text_response = await self.claude.respond_to_audio(
            audio_data,
            media_type,
            self.config.system_prompt
        )
        
        # Stream TTS in chunks
        if hasattr(self.tts, 'synthesize_stream'):
            async for chunk in self.tts.synthesize_stream(
                text_response,
                voice_id=self.config.voice_id
            ):
                if on_audio_chunk:
                    await on_audio_chunk(chunk)
        else:
            # Fall back to non-streaming
            audio = await self.tts.synthesize(
                text_response,
                voice_id=self.config.voice_id
            )
            if on_audio_chunk:
                await on_audio_chunk(audio)
        
        # Update history
        self.conversation_history.append({
            "role": "user",
            "content": "[Audio input]"
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": text_response
        })
        
        self.metrics["turns"] += 1


# Configuration example
assistant_config = VoiceAssistantConfig(
    claude_model="claude-sonnet-4-20250514",
    system_prompt="""You are Aria, a helpful voice assistant.

Guidelines:
- Be conversational and friendly
- Keep responses concise for voice delivery
- Ask clarifying questions when needed
- Remember context from the conversation
""",
    tts_provider="openai",
    voice_id="nova",
    speaking_rate=1.0
)

print("Claude Voice Assistant Configuration")
print("=" * 50)
print(f"Model: {assistant_config.claude_model}")
print(f"TTS: {assistant_config.tts_provider}")
print(f"Voice: {assistant_config.voice_id}")
```

---

## Real-Time Transcription Integration

### Live Transcription System

```python
class TranscriptionMode(Enum):
    """Transcription modes."""
    
    CLAUDE = "claude"  # Use Claude for transcription
    WHISPER = "whisper"  # Use OpenAI Whisper
    HYBRID = "hybrid"  # Use both for verification


class LiveTranscriptionSystem:
    """Real-time transcription with Claude."""
    
    def __init__(
        self,
        claude_client: ClaudeAudioClient,
        mode: TranscriptionMode = TranscriptionMode.CLAUDE
    ):
        self.claude = claude_client
        self.mode = mode
        
        # State
        self.is_transcribing = False
        self.transcript_buffer: List[str] = []
        
        # Callbacks
        self._on_transcript: Optional[Callable] = None
        self._on_final_transcript: Optional[Callable] = None
    
    def on_transcript(self, callback: Callable):
        """Register interim transcript callback."""
        self._on_transcript = callback
    
    def on_final_transcript(self, callback: Callable):
        """Register final transcript callback."""
        self._on_final_transcript = callback
    
    async def transcribe_chunk(
        self,
        audio_chunk: bytes,
        media_type: str = "audio/wav"
    ) -> Optional[str]:
        """Transcribe an audio chunk."""
        
        if not self.is_transcribing:
            return None
        
        try:
            transcript = await self.claude.transcribe(
                audio_chunk,
                media_type
            )
            
            self.transcript_buffer.append(transcript)
            
            if self._on_transcript:
                await self._call_callback(
                    self._on_transcript,
                    transcript
                )
            
            return transcript
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    async def finalize(self) -> str:
        """Finalize transcription."""
        
        full_transcript = " ".join(self.transcript_buffer)
        self.transcript_buffer = []
        
        if self._on_final_transcript:
            await self._call_callback(
                self._on_final_transcript,
                full_transcript
            )
        
        return full_transcript
    
    async def start(self):
        """Start transcription."""
        self.is_transcribing = True
        self.transcript_buffer = []
    
    async def stop(self) -> str:
        """Stop transcription and return final transcript."""
        
        self.is_transcribing = False
        return await self.finalize()
    
    async def _call_callback(self, callback: Callable, *args):
        """Call callback safely."""
        import asyncio
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result


class WhisperFallback:
    """Fallback to Whisper for transcription."""
    
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en"
    ) -> str:
        """Transcribe audio using Whisper."""
        
        # Prepare multipart form data
        files = {
            "file": ("audio.wav", audio_data, "audio/wav"),
            "model": (None, "whisper-1"),
            "language": (None, language)
        }
        
        response = await self.client.post(
            "/audio/transcriptions",
            files=files
        )
        
        response.raise_for_status()
        return response.json()["text"]
    
    async def close(self):
        await self.client.aclose()


# Usage
print("Transcription Modes:")
for mode in TranscriptionMode:
    print(f"  - {mode.value}")
```

---

## Hands-on Exercise

### Your Task

Build a complete voice assistant using Claude for understanding and a TTS provider for speech output.

### Requirements

1. Process audio input with Claude
2. Generate natural responses
3. Convert to speech with TTS
4. Track conversation history
5. Handle errors gracefully

<details>
<summary>💡 Hints</summary>

- Use Claude's audio understanding for context
- Keep responses concise for voice
- Add error recovery with retries
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import asyncio
import base64


class ProductionVoiceAssistant:
    """Production-ready Claude voice assistant."""
    
    def __init__(
        self,
        claude_api_key: str,
        tts_api_key: str,
        assistant_name: str = "Aria"
    ):
        self.assistant_name = assistant_name
        
        # Configuration
        self.config = VoiceAssistantConfig(
            claude_model="claude-sonnet-4-20250514",
            system_prompt=self._build_system_prompt(),
            tts_provider="openai",
            voice_id="nova",
            speaking_rate=1.0
        )
        
        # Clients (would be initialized with real API keys)
        self.claude_api_key = claude_api_key
        self.tts_api_key = tts_api_key
        self.claude: Optional[ClaudeAudioClient] = None
        self.tts: Optional[TTSProvider] = None
        
        # State
        self.is_active = False
        self.session_id: Optional[str] = None
        self.conversation_history: List[dict] = []
        
        # Metrics
        self.metrics = {
            "session_start": None,
            "turns": 0,
            "audio_input_seconds": 0,
            "audio_output_seconds": 0,
            "errors": 0,
            "latencies": []
        }
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the assistant."""
        
        return f"""You are {self.assistant_name}, a helpful voice assistant.

## Voice Delivery Guidelines
- Keep responses under 100 words for natural voice delivery
- Use conversational, natural language
- Avoid technical jargon unless the user uses it
- Ask clarifying questions when the request is ambiguous

## Conversation Style
- Be warm and friendly
- Remember context from earlier in the conversation
- Acknowledge emotions when users express them
- Be honest when you don't know something

## Capabilities
- Answer questions on a wide range of topics
- Help with tasks, planning, and problem-solving
- Provide recommendations and suggestions
- Explain complex topics simply

## Limitations
- You cannot browse the internet in real-time
- You cannot execute code or access external systems
- For medical, legal, or financial advice, recommend consulting professionals
"""
    
    async def start(self):
        """Start the assistant session."""
        
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize clients
        self.claude = ClaudeAudioClient(
            self.claude_api_key,
            ClaudeAudioConfig(model=self.config.claude_model)
        )
        
        self.tts = TTSProviderFactory.create(
            self.config.tts_provider,
            self.tts_api_key
        )
        
        self.is_active = True
        self.conversation_history = []
        self.metrics["session_start"] = datetime.now().isoformat()
        
        # Generate greeting
        greeting = f"Hello! I'm {self.assistant_name}. How can I help you today?"
        greeting_audio = await self._text_to_speech(greeting)
        
        print(f"\n{self.assistant_name} is ready!")
        
        return greeting_audio
    
    async def stop(self):
        """Stop the assistant session."""
        
        self.is_active = False
        
        if self.claude:
            await self.claude.close()
        if self.tts:
            await self.tts.close()
        
        # Log session summary
        print(f"\n{self.assistant_name} Session Summary:")
        print(f"  Duration: {self._get_session_duration():.1f}s")
        print(f"  Turns: {self.metrics['turns']}")
        print(f"  Avg latency: {self._get_avg_latency():.2f}s")
        print(f"  Errors: {self.metrics['errors']}")
    
    async def process_audio(
        self,
        audio_data: bytes,
        media_type: str = "audio/wav"
    ) -> bytes:
        """Process audio input and return audio response."""
        
        if not self.is_active:
            raise RuntimeError("Assistant not started")
        
        start_time = datetime.now()
        
        try:
            # Get response from Claude
            response_text = await self._with_retry(
                self._get_claude_response,
                audio_data,
                media_type
            )
            
            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": "[Audio message]",
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Convert to speech
            audio_response = await self._with_retry(
                self._text_to_speech,
                response_text
            )
            
            # Record metrics
            latency = (datetime.now() - start_time).total_seconds()
            self.metrics["latencies"].append(latency)
            self.metrics["turns"] += 1
            
            return audio_response
            
        except Exception as e:
            self.metrics["errors"] += 1
            error_response = "I'm sorry, I encountered an error. Could you please try again?"
            return await self._text_to_speech(error_response)
    
    async def process_text(self, text: str) -> bytes:
        """Process text input and return audio response."""
        
        if not self.is_active:
            raise RuntimeError("Assistant not started")
        
        # Build context with history
        context = self._build_conversation_context()
        
        try:
            # Get response
            response = await self.claude._send_message(
                [{"type": "text", "text": text}],
                system=self.config.system_prompt
            )
            
            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": text,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Convert to speech
            audio_response = await self._text_to_speech(response)
            self.metrics["turns"] += 1
            
            return audio_response
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise
    
    async def _get_claude_response(
        self,
        audio_data: bytes,
        media_type: str
    ) -> str:
        """Get response from Claude for audio input."""
        
        return await self.claude.respond_to_audio(
            audio_data,
            media_type,
            self.config.system_prompt
        )
    
    async def _text_to_speech(self, text: str) -> bytes:
        """Convert text to speech."""
        
        return await self.tts.synthesize(
            text,
            voice_id=self.config.voice_id
        )
    
    async def _with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retries."""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(
                        self.retry_delay * (2 ** attempt)
                    )
        
        raise last_error
    
    def _build_conversation_context(self) -> str:
        """Build context from conversation history."""
        
        # Include last 10 turns for context
        recent = self.conversation_history[-20:]
        
        lines = []
        for msg in recent:
            role = msg["role"].title()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _get_session_duration(self) -> float:
        """Get session duration in seconds."""
        
        if not self.metrics["session_start"]:
            return 0
        
        start = datetime.fromisoformat(self.metrics["session_start"])
        return (datetime.now() - start).total_seconds()
    
    def _get_avg_latency(self) -> float:
        """Get average latency."""
        
        if not self.metrics["latencies"]:
            return 0
        
        return sum(self.metrics["latencies"]) / len(self.metrics["latencies"])
    
    def get_transcript(self) -> str:
        """Get conversation transcript."""
        
        lines = []
        for msg in self.conversation_history:
            role = msg["role"].title()
            content = msg["content"]
            lines.append(f"[{msg['timestamp']}] {role}: {content}")
        
        return "\n".join(lines)
    
    def get_status(self) -> dict:
        """Get assistant status."""
        
        return {
            "name": self.assistant_name,
            "session_id": self.session_id,
            "is_active": self.is_active,
            "turns": self.metrics["turns"],
            "duration_seconds": self._get_session_duration(),
            "avg_latency": self._get_avg_latency(),
            "errors": self.metrics["errors"],
            "history_length": len(self.conversation_history)
        }


# Usage example
async def demo():
    """Demonstrate the voice assistant."""
    
    assistant = ProductionVoiceAssistant(
        claude_api_key="your-claude-key",
        tts_api_key="your-tts-key",
        assistant_name="Aria"
    )
    
    print(f"Assistant: {assistant.assistant_name}")
    print(f"Status: {assistant.get_status()}")
    
    # In production:
    # greeting = await assistant.start()
    # play_audio(greeting)
    
    # audio_response = await assistant.process_audio(user_audio)
    # play_audio(audio_response)
    
    # await assistant.stop()
    
    return assistant


# Create assistant
assistant = ProductionVoiceAssistant(
    claude_api_key="demo-key",
    tts_api_key="demo-key",
    assistant_name="Aria"
)

print("Production Voice Assistant")
print("=" * 50)
print(f"Name: {assistant.assistant_name}")
print(f"Model: {assistant.config.claude_model}")
print(f"TTS: {assistant.config.tts_provider}")
print(f"Status: {assistant.get_status()}")
```

</details>

---

## Summary

✅ Claude supports audio input for understanding  
✅ Voice-to-text-to-voice pattern enables full voice assistants  
✅ Multiple TTS providers available (ElevenLabs, OpenAI, Google)  
✅ Error handling and retries improve reliability  
✅ Conversation history maintains context

**Next:** [Back to Realtime API Overview](./00-realtime-api-voice.md)

---

## Further Reading

- [Anthropic Claude Documentation](https://docs.anthropic.com/en/docs/build-with-claude/audio) — Audio input guide
- [ElevenLabs API](https://elevenlabs.io/docs) — High-quality TTS
- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) — OpenAI voice synthesis
- [Google Cloud TTS](https://cloud.google.com/text-to-speech) — Google voice options
