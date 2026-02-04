---
title: "Video and Audio Prompting"
---

# Video and Audio Prompting

## Introduction

Video and audio add temporal dimensions to multimodal prompting. From analyzing meeting recordings to processing surveillance footage, understanding how to work with time-based media unlocks powerful applications. This lesson covers video input, timestamp referencing, audio analysis, and cost optimization for long-form content.

> **🔑 Key Insight:** Gemini is the primary provider for video understanding, with native support for video files, YouTube URLs, and audio extraction. OpenAI and Anthropic focus on images, requiring frame extraction for video analysis.

### What We'll Cover

- Video upload methods (Files API, inline, YouTube)
- Temporal referencing with timestamps
- Frame rate control and clipping
- Audio input and analysis
- Token cost optimization for long content

### Prerequisites

- [Image Prompting Fundamentals](./01-image-prompting.md)
- [Vision Capabilities](./03-vision-capabilities.md)

---

## Video Input Methods

### Gemini Files API (Recommended for Large Videos)

For videos over 20MB, use the Files API:

```python
from google import genai
from google.genai import types
import time

client = genai.Client()

# Upload video (stored for 48 hours, no charge)
print("Uploading video...")
video_file = client.files.upload(file="product_demo.mp4")

# Wait for processing to complete
while video_file.state.name == "PROCESSING":
    print("Processing...")
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(f"Video processing failed: {video_file.state.name}")

print(f"Upload complete: {video_file.name}")
print(f"MIME type: {video_file.mime_type}")

# Use in request
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        video_file,
        "Summarize this product demonstration. What features are shown?"
    ]
)

print(response.text)
```

### Inline Video (Small Files < 20MB)

```python
from google import genai
from google.genai import types

client = genai.Client()

with open("short_clip.mp4", "rb") as f:
    video_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        "Describe what happens in this video clip."
    ]
)
```

### YouTube URL Support

Gemini can analyze YouTube videos directly:

```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        "https://www.youtube.com/watch?v=VIDEO_ID",
        "Summarize the main points of this video."
    ]
)
```

**YouTube Limitations:**

| Tier | Daily Limit | Notes |
|------|-------------|-------|
| Free | 8 hours | Resets daily |
| Pay-as-you-go | Higher | Based on usage |
| Enterprise | Custom | Contact sales |

> **Note:** YouTube support requires the video to be publicly accessible or unlisted with the URL. Private videos cannot be analyzed.

### Multiple Videos in One Request

```python
# Gemini 2.5+ supports up to 10 videos per request
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        video_file_1,
        "Video 1: Product A demo",
        video_file_2,
        "Video 2: Product B demo",
        """
        Compare these two product demonstrations:
        1. Which explains features more clearly?
        2. Which has better production quality?
        3. Which would be more effective for marketing?
        """
    ]
)
```

---

## Temporal References with Timestamps

### Asking About Specific Moments

Use `MM:SS` format for timestamp references:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        video_file,
        """
        Watch this video and answer:
        1. What happens at 01:30?
        2. At what timestamp does the speaker mention pricing?
        3. Describe the scene between 02:15 and 02:45.
        """
    ]
)
```

### Getting Timestamp-Referenced Summaries

```python
from pydantic import BaseModel

class VideoMoment(BaseModel):
    timestamp: str  # MM:SS format
    description: str
    importance: str  # high, medium, low

class VideoSummary(BaseModel):
    title: str
    duration: str
    key_moments: list[VideoMoment]
    topics_covered: list[str]
    call_to_action: str | None

prompt = """
Analyze this video and provide a timestamped summary.
For each key moment:
1. Note the timestamp in MM:SS format
2. Describe what happens
3. Rate importance (high/medium/low)
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, prompt],
    config={
        "response_mime_type": "application/json",
        "response_schema": VideoSummary.model_json_schema()
    }
)

import json
summary = VideoSummary(**json.loads(response.text))

print(f"Video: {summary.title} ({summary.duration})")
print("\nKey Moments:")
for moment in summary.key_moments:
    print(f"  [{moment.timestamp}] {moment.description} ({moment.importance})")
```

### Scene Detection

```python
prompt = """
Identify distinct scenes in this video.
For each scene:
1. Start and end timestamps
2. Description of the scene
3. Main subjects/objects
4. Any text or graphics shown

Example format:
Scene 1: 00:00 - 00:45
Description: Opening logo animation with company branding
Text shown: "Company Name - Innovating Since 2020"
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, prompt]
)
```

---

## Frame Rate Control and Clipping

### Custom Frame Rate Sampling

By default, Gemini samples at 1 frame per second. Adjust for different needs:

```python
from google.genai import types

# Higher frame rate for fast action
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
            video_metadata=types.VideoMetadata(fps=5)  # 5 frames per second
        ),
        "Analyze the athlete's form during this high-speed maneuver."
    ]
)

# Lower frame rate for long videos (cost savings)
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
            video_metadata=types.VideoMetadata(fps=0.25)  # 1 frame every 4 seconds
        ),
        "Give a general overview of this hour-long lecture."
    ]
)
```

### Frame Rate Guidelines

| Content Type | Recommended FPS | Use Case |
|--------------|-----------------|----------|
| Static presentations | 0.25 | Slides, lectures |
| Normal video | 1 (default) | General analysis |
| Fast motion | 3-5 | Sports, action |
| Frame-by-frame | 10+ | Quality inspection |

### Video Clipping

Analyze only a portion of a video:

```python
from google.genai import types

# Analyze 30-second segment starting at 2 minutes
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
            video_metadata=types.VideoMetadata(
                start_offset=types.Duration(seconds=120),  # Start at 2:00
                end_offset=types.Duration(seconds=150)     # End at 2:30
            )
        ),
        "Describe what happens in this segment."
    ]
)
```

### Multiple Clips from Same Video

```python
# Analyze specific segments
clips = [
    {"start": 0, "end": 30, "label": "Introduction"},
    {"start": 120, "end": 180, "label": "Main demo"},
    {"start": 300, "end": 330, "label": "Conclusion"}
]

contents = []
for clip in clips:
    contents.append(
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
            video_metadata=types.VideoMetadata(
                start_offset=types.Duration(seconds=clip["start"]),
                end_offset=types.Duration(seconds=clip["end"])
            )
        )
    )
    contents.append(f"[{clip['label']}]")

contents.append("Compare these three segments. Which is most engaging?")

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=contents
)
```

---

## Audio Input and Analysis

### Audio Files with Gemini

```python
from google import genai
from google.genai import types

client = genai.Client()

# Upload audio file
audio_file = client.files.upload(file="podcast_episode.mp3")

# Wait for processing
import time
while audio_file.state.name == "PROCESSING":
    time.sleep(2)
    audio_file = client.files.get(name=audio_file.name)

# Transcribe and analyze
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        audio_file,
        """
        Analyze this audio recording:
        1. Transcribe the full conversation
        2. Identify speakers (Speaker 1, Speaker 2, etc.)
        3. Summarize key points discussed
        4. Note any action items mentioned
        """
    ]
)
```

### Inline Audio for Short Clips

```python
with open("voice_note.wav", "rb") as f:
    audio_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        "Transcribe this voice note and identify the speaker's intent."
    ]
)
```

### Supported Audio Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| MP3 | audio/mp3, audio/mpeg | Common, compressed |
| WAV | audio/wav | Uncompressed |
| FLAC | audio/flac | Lossless |
| AAC | audio/aac | Common in video |
| OGG | audio/ogg | Open format |
| WEBM | audio/webm | Web format |

### Audio Analysis Tasks

```python
# Speaker identification
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        audio_file,
        """
        Identify distinct speakers in this recording.
        For each speaker:
        1. Assign a label (Speaker A, Speaker B, etc.)
        2. Estimate their speaking time percentage
        3. Characterize their speaking style
        """
    ]
)

# Sentiment analysis
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        audio_file,
        """
        Analyze the emotional tone throughout this recording.
        Track sentiment changes with timestamps.
        Note any moments of tension, enthusiasm, or concern.
        """
    ]
)

# Meeting minutes
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        audio_file,
        """
        Generate meeting minutes from this recording:
        
        ## Meeting Summary
        [Brief overview]
        
        ## Attendees
        [List identified speakers]
        
        ## Discussion Points
        [Key topics with timestamps]
        
        ## Decisions Made
        [Any conclusions reached]
        
        ## Action Items
        [Tasks assigned, with owners if mentioned]
        
        ## Next Steps
        [Follow-up plans]
        """
    ]
)
```

---

## Token Cost Optimization

### Video Token Calculation

```python
def estimate_video_tokens(
    duration_seconds: int,
    fps: float = 1.0,
    has_audio: bool = True,
    media_resolution: str = "medium"
) -> dict:
    """
    Estimate token cost for Gemini video processing.
    
    Based on:
    - 258 tokens per frame (at default resolution)
    - 32 tokens per second of audio
    """
    
    # Frame tokens
    resolution_multipliers = {
        "low": 0.25,    # 66 tokens/frame
        "medium": 1.0,  # 258 tokens/frame
        "high": 1.0     # 258+ depending on resolution
    }
    
    base_frame_tokens = 258
    frame_tokens = base_frame_tokens * resolution_multipliers.get(media_resolution, 1.0)
    
    total_frames = int(duration_seconds * fps)
    video_tokens = int(total_frames * frame_tokens)
    
    # Audio tokens
    audio_tokens = duration_seconds * 32 if has_audio else 0
    
    total_tokens = video_tokens + audio_tokens
    
    return {
        "total_tokens": total_tokens,
        "video_tokens": video_tokens,
        "audio_tokens": audio_tokens,
        "frames": total_frames,
        "fps": fps,
        "duration_seconds": duration_seconds
    }

# Examples
print("1-minute video at 1 FPS:")
print(estimate_video_tokens(60, fps=1.0, has_audio=True))
# ~15,480 + 1,920 = ~17,400 tokens

print("\n10-minute video at 0.5 FPS:")
print(estimate_video_tokens(600, fps=0.5, has_audio=True))
# ~77,400 + 19,200 = ~96,600 tokens

print("\n1-hour video at 0.25 FPS:")
print(estimate_video_tokens(3600, fps=0.25, has_audio=True))
# ~232,200 + 115,200 = ~347,400 tokens
```

### Cost Optimization Strategies

```python
# Strategy 1: Use lower FPS for talking-head videos
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(fps=0.25)  # Much cheaper
        ),
        "Summarize this lecture."
    ]
)

# Strategy 2: Use clipping to analyze only relevant portions
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=video_file.uri,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(
                start_offset=types.Duration(seconds=120),
                end_offset=types.Duration(seconds=180)
            )
        ),
        "Analyze this specific segment."
    ]
)

# Strategy 3: Lower resolution for general understanding
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, "What is this video about?"],
    config={"media_resolution": "low"}
)

# Strategy 4: Audio-only for podcasts/interviews
# Extract audio first, then analyze (cheaper than video)
import subprocess

def extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video."""
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "mp3",
        audio_path
    ], check=True)
    return audio_path
```

### Cost Comparison Table

For a 10-minute video:

| Strategy | FPS | Tokens | Cost Reduction |
|----------|-----|--------|----------------|
| Default | 1.0 | ~174,000 | Baseline |
| Low FPS | 0.25 | ~58,000 | ~67% |
| Audio only | N/A | ~19,200 | ~89% |
| 2-min clip | 1.0 | ~35,000 | ~80% |

---

## Video Analysis Patterns

### Content Moderation

```python
from pydantic import BaseModel
from enum import Enum

class ContentRating(str, Enum):
    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    FLAGGED = "flagged"

class ModerationResult(BaseModel):
    rating: ContentRating
    violence_detected: bool
    explicit_content: bool
    hate_speech: bool
    dangerous_activities: bool
    timestamps_of_concern: list[str]
    notes: str

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        video_file,
        """
        Review this video for content policy violations.
        Check for:
        - Violence or graphic content
        - Explicit or adult content
        - Hate speech or discrimination
        - Dangerous activities without safety warnings
        
        Note specific timestamps for any concerns.
        """
    ],
    config={
        "response_mime_type": "application/json",
        "response_schema": ModerationResult.model_json_schema()
    }
)
```

### Tutorial Quality Analysis

```python
prompt = """
Evaluate this tutorial video for educational effectiveness:

1. Clarity (1-10)
   - Is the topic clearly introduced?
   - Are concepts explained well?
   - Is the pacing appropriate?

2. Production Quality (1-10)
   - Audio clarity
   - Visual quality
   - Screen visibility

3. Engagement (1-10)
   - Does it maintain interest?
   - Are there interactive elements?
   - Appropriate length?

4. Accessibility
   - Would a beginner understand?
   - Are prerequisites stated?
   - Are resources linked?

5. Specific Improvement Suggestions
   - List 3-5 actionable improvements

Provide timestamps for examples of good and bad practices.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, prompt]
)
```

---

## Common Mistakes

### ❌ Processing Long Videos at High FPS

```python
# Bad: Expensive for a 1-hour video
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=hour_long_video.uri,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(fps=5)  # 18,000 frames!
        ),
        "Summarize this lecture."
    ]
)

# Good: Lower FPS for lectures
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[
        types.Part.from_uri(
            file_uri=hour_long_video.uri,
            mime_type="video/mp4",
            video_metadata=types.VideoMetadata(fps=0.25)  # 900 frames
        ),
        "Summarize this lecture."
    ]
)
```

### ❌ Not Waiting for Upload Processing

```python
# Bad: Using file before processing complete
video_file = client.files.upload(file="video.mp4")
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, "Analyze this."]  # May fail!
)

# Good: Wait for processing
video_file = client.files.upload(file="video.mp4")
while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "ACTIVE":
    response = client.models.generate_content(...)
```

### ❌ Ignoring Audio for Transcription Tasks

```python
# Bad: Sending video when you only need audio
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[video_file, "Transcribe the speech."]  # Wastes video tokens
)

# Good: Extract and send audio only
extract_audio("video.mp4", "audio.mp3")
audio_file = client.files.upload(file="audio.mp3")
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[audio_file, "Transcribe the speech."]
)
```

---

## Hands-on Exercise

### Your Task

Build a video chapter generator that:
1. Accepts a tutorial/educational video
2. Identifies natural chapter breaks
3. Generates timestamped chapter titles
4. Creates a brief description for each chapter
5. Outputs YouTube-compatible chapter format

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from google import genai
from google.genai import types
from pydantic import BaseModel
import time
import json

class Chapter(BaseModel):
    timestamp: str  # MM:SS or HH:MM:SS format
    title: str
    description: str
    key_topics: list[str]

class VideoChapters(BaseModel):
    video_title: str
    total_duration: str
    chapters: list[Chapter]
    target_audience: str

def generate_chapters(video_path: str) -> VideoChapters:
    """Generate YouTube-style chapters for a video."""
    
    client = genai.Client()
    
    # Upload video
    print("Uploading video...")
    video_file = client.files.upload(file=video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("Processing...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed")
    
    print("Generating chapters...")
    
    # Use moderate FPS for chapter detection
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type,
                video_metadata=types.VideoMetadata(fps=0.5)  # 1 frame per 2 seconds
            ),
            """
            Generate YouTube-style chapters for this video.
            
            Guidelines:
            1. First chapter MUST start at 00:00
            2. Each chapter should be 2-10 minutes long
            3. Identify natural topic transitions
            4. Create concise, descriptive titles (under 50 chars)
            5. Write brief descriptions (1-2 sentences)
            6. List 2-3 key topics per chapter
            
            Consider:
            - Visual transitions (slides, scene changes)
            - Verbal cues ("Next, we'll discuss...")
            - Topic shifts in content
            """
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": VideoChapters.model_json_schema()
        }
    )
    
    return VideoChapters(**json.loads(response.text))

def format_youtube_chapters(chapters: VideoChapters) -> str:
    """Format chapters for YouTube description."""
    
    lines = [f"📚 Chapters for: {chapters.video_title}", ""]
    
    for chapter in chapters.chapters:
        lines.append(f"{chapter.timestamp} - {chapter.title}")
    
    lines.extend(["", "---", ""])
    
    for chapter in chapters.chapters:
        lines.append(f"**{chapter.timestamp} - {chapter.title}**")
        lines.append(chapter.description)
        lines.append("")
    
    return "\n".join(lines)

# Usage
chapters = generate_chapters("tutorial.mp4")

print(f"Video: {chapters.video_title}")
print(f"Duration: {chapters.total_duration}")
print(f"Target Audience: {chapters.target_audience}")
print(f"\nGenerated {len(chapters.chapters)} chapters:")

for ch in chapters.chapters:
    print(f"\n{ch.timestamp} - {ch.title}")
    print(f"  {ch.description}")
    print(f"  Topics: {', '.join(ch.key_topics)}")

print("\n" + "="*50)
print("YouTube Description Format:")
print("="*50)
print(format_youtube_chapters(chapters))
```

</details>

---

## Summary

✅ **Gemini dominates video analysis:** Files API, inline, and YouTube URL support
✅ **Use timestamps:** `MM:SS` format for referencing specific moments
✅ **Control frame rate:** Lower FPS for talks, higher for action
✅ **Clipping saves tokens:** Analyze only relevant portions
✅ **Audio-only is cheapest:** Extract audio for transcription tasks

**Previous:** [Vision Capabilities](./03-vision-capabilities.md)
**Back to:** [Multimodal Prompting Overview](./00-multimodal-prompting-overview.md)

---

## Further Reading

- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Files API](https://ai.google.dev/gemini-api/docs/files)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - For audio extraction

---

<!-- 
Sources Consulted:
- Gemini Video Understanding: Files API, timestamps, videoMetadata, FPS control
- Gemini Audio: Supported formats, token calculation
- Google AI Studio: YouTube URL support, daily limits
-->
