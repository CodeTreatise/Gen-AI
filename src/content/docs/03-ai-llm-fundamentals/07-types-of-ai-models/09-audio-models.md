---
title: "Audio Models"
---

# Audio Models

## Introduction

Audio models handle speech-to-text (transcription), text-to-speech (synthesis), and audio generation. These models power voice assistants, transcription services, and audio content creation.

### What We'll Cover

- Speech-to-text (Whisper, AssemblyAI)
- Text-to-speech (OpenAI TTS, ElevenLabs)
- Voice cloning
- Real-time audio processing

---

## Speech-to-Text

### OpenAI Whisper

```python
from openai import OpenAI

client = OpenAI()

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper"""
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    return response.text

# With options
def transcribe_with_options(
    audio_path: str,
    language: str = None,
    prompt: str = None
) -> dict:
    """Transcribe with timestamps and options"""
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,  # ISO code: "en", "es", "fr"
            prompt=prompt,      # Context for better accuracy
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    
    return {
        "text": response.text,
        "segments": response.segments,
        "words": response.words
    }

result = transcribe_with_options("meeting.mp3", language="en")
print(result["text"])
```

### AssemblyAI

```python
import assemblyai as aai

aai.settings.api_key = "YOUR_KEY"

def transcribe_with_assemblyai(audio_url: str) -> dict:
    """Transcribe with AssemblyAI features"""
    
    config = aai.TranscriptionConfig(
        speaker_labels=True,       # Diarization
        auto_chapters=True,        # Chapter summaries
        entity_detection=True,     # Named entities
        sentiment_analysis=True,   # Sentiment per sentence
    )
    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_url, config)
    
    return {
        "text": transcript.text,
        "speakers": [
            {"speaker": u.speaker, "text": u.text}
            for u in transcript.utterances
        ],
        "chapters": transcript.chapters,
        "entities": transcript.entities,
        "sentiment": transcript.sentiment_analysis_results
    }
```

### Translation (Audio to English)

```python
def translate_audio(audio_path: str) -> str:
    """Translate audio to English"""
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
    
    return response.text

# Spanish audio → English text
english_text = translate_audio("spanish_audio.mp3")
```

---

## Text-to-Speech

### OpenAI TTS

```python
def text_to_speech(
    text: str,
    voice: str = "alloy",  # alloy, echo, fable, onyx, nova, shimmer
    output_path: str = "output.mp3"
) -> str:
    """Convert text to speech"""
    
    response = client.audio.speech.create(
        model="tts-1",      # or tts-1-hd for higher quality
        voice=voice,
        input=text
    )
    
    response.stream_to_file(output_path)
    return output_path

# Generate speech
text_to_speech(
    "Hello! Welcome to our AI course.",
    voice="nova",
    output_path="welcome.mp3"
)
```

### ElevenLabs

```python
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key="YOUR_KEY")

def generate_elevenlabs_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
) -> bytes:
    """Generate speech with ElevenLabs"""
    
    audio = client.generate(
        text=text,
        voice=voice_id,
        model="eleven_multilingual_v2"
    )
    
    return audio

# Stream audio
def stream_speech(text: str):
    audio_stream = client.generate(
        text=text,
        voice="Rachel",
        model="eleven_multilingual_v2",
        stream=True
    )
    
    for chunk in audio_stream:
        yield chunk
```

### Voice Comparison

| Feature | OpenAI TTS | ElevenLabs |
|---------|-----------|------------|
| Quality | Good | Excellent |
| Voices | 6 built-in | 1000s + cloning |
| Languages | ~50 | 29 |
| Streaming | Yes | Yes |
| Cost | $0.015/1K chars | $0.30/1K chars |
| Cloning | No | Yes |

---

## Voice Cloning

### ElevenLabs Voice Cloning

```python
def clone_voice(
    name: str,
    audio_files: list,
    description: str = ""
) -> str:
    """Clone a voice from audio samples"""
    
    voice = client.clone(
        name=name,
        description=description,
        files=audio_files  # List of audio file paths
    )
    
    return voice.voice_id

# Clone from samples
voice_id = clone_voice(
    name="My Custom Voice",
    audio_files=["sample1.mp3", "sample2.mp3", "sample3.mp3"],
    description="Professional narrator voice"
)

# Use cloned voice
audio = client.generate(
    text="This is my cloned voice speaking.",
    voice=voice_id
)
```

### Instant Voice Cloning

```python
# Instant cloning from single sample (lower quality)
def instant_clone(audio_sample: str, text: str) -> bytes:
    """Clone and generate in one step"""
    
    voice = client.clone(
        name="Instant Clone",
        files=[audio_sample]
    )
    
    return client.generate(
        text=text,
        voice=voice.voice_id
    )
```

---

## Real-Time Audio Processing

### OpenAI Realtime API

```python
import asyncio
import websockets
import json

async def realtime_conversation():
    """Real-time voice conversation with GPT-4"""
    
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
        additional_headers={"Authorization": f"Bearer {API_KEY}"}
    ) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy"
            }
        }))
        
        # Send audio input
        audio_chunk = get_microphone_audio()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_chunk).decode()
        }))
        
        # Receive response
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "response.audio.delta":
                play_audio(base64.b64decode(data["delta"]))
```

### Streaming Transcription

```python
import pyaudio
import wave
from queue import Queue
import threading

class RealtimeTranscriber:
    """Stream audio for real-time transcription"""
    
    def __init__(self):
        self.audio_queue = Queue()
        self.is_recording = False
    
    def start_recording(self):
        """Start recording from microphone"""
        self.is_recording = True
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        while self.is_recording:
            data = stream.read(1024)
            self.audio_queue.put(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def transcribe_stream(self):
        """Transcribe audio as it arrives"""
        buffer = b""
        
        while self.is_recording or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                buffer += self.audio_queue.get()
                
                # Transcribe every 5 seconds of audio
                if len(buffer) >= 16000 * 2 * 5:  # 5 seconds
                    text = transcribe_chunk(buffer)
                    print(text, end=" ", flush=True)
                    buffer = b""
```

---

## Music Generation

### Suno AI (via API)

```python
# Suno generates music from text descriptions
def generate_music(
    prompt: str,
    duration: int = 30,  # seconds
    style: str = "pop"
) -> str:
    """Generate music with Suno"""
    
    # Note: Suno API access may require specific integration
    response = requests.post(
        "https://api.suno.ai/v1/generate",
        headers={"Authorization": f"Bearer {SUNO_KEY}"},
        json={
            "prompt": prompt,
            "style": style,
            "duration": duration
        }
    )
    
    return response.json()["audio_url"]

# Generate a song
url = generate_music(
    prompt="An upbeat electronic track about coding",
    style="electronic",
    duration=60
)
```

---

## Hands-on Exercise

### Your Task

Build an audio processing pipeline:

```python
from openai import OpenAI
import os

client = OpenAI()

class AudioProcessor:
    """Complete audio processing pipeline"""
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio with timestamps"""
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
        return {
            "text": response.text,
            "duration": response.duration,
            "segments": response.segments
        }
    
    def translate(self, audio_path: str) -> str:
        """Translate audio to English"""
        with open(audio_path, "rb") as f:
            response = client.audio.translations.create(
                model="whisper-1",
                file=f
            )
        return response.text
    
    def synthesize(
        self, 
        text: str, 
        voice: str = "nova",
        output_path: str = "output.mp3"
    ) -> str:
        """Generate speech from text"""
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )
        response.stream_to_file(output_path)
        return output_path
    
    def transcribe_and_translate(self, audio_path: str) -> dict:
        """Full pipeline: transcribe, translate, synthesize"""
        
        # Transcribe
        transcript = self.transcribe(audio_path)
        
        # Summarize with LLM
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this transcript:\n\n{transcript['text']}"
            }]
        ).choices[0].message.content
        
        # Synthesize summary
        summary_audio = self.synthesize(summary, output_path="summary.mp3")
        
        return {
            "transcript": transcript,
            "summary_text": summary,
            "summary_audio": summary_audio
        }

# Test
processor = AudioProcessor()
# result = processor.transcribe("recording.mp3")
# print(result["text"])
```

---

## Summary

✅ **Whisper**: Best open STT, supports 100+ languages

✅ **OpenAI TTS**: Good quality, affordable

✅ **ElevenLabs**: Best quality, voice cloning

✅ **Real-time**: WebSocket APIs for live audio

✅ **Music generation**: Emerging capability (Suno, etc.)

**Next:** [Video Models](./10-video-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Image Understanding](./08-image-understanding-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Video Models](./10-video-models.md) |

