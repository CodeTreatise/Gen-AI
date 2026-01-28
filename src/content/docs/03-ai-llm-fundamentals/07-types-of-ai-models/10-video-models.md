---
title: "Video Models"
---

# Video Models

## Introduction

Video models generate, edit, and understand video content. From text-to-video generation to video analysis, these models are rapidly evolving.

### What We'll Cover

- Text-to-video generation
- Video editing with AI
- Current capabilities and limitations
- Commercial availability

---

## Text-to-Video Generation

### OpenAI Sora

```python
# Sora API (when available)
def generate_video_sora(
    prompt: str,
    duration: int = 10,  # seconds
    resolution: str = "1080p"
) -> str:
    """Generate video with Sora"""
    
    response = client.videos.generate(
        model="sora-1",
        prompt=prompt,
        duration=duration,
        resolution=resolution
    )
    
    return response.video_url

# Example prompt
video_url = generate_video_sora(
    "A serene mountain lake at sunrise, camera slowly panning across "
    "the water with mist rising, photorealistic, 4K quality",
    duration=15
)
```

**Capabilities:**
- Up to 60 seconds
- 1080p resolution
- Realistic physics
- Complex camera movements
- Consistent subjects

### Runway Gen-3

```python
import requests

def generate_runway_video(
    prompt: str,
    duration: int = 4,  # 4-16 seconds
    aspect_ratio: str = "16:9"
) -> str:
    """Generate video with Runway"""
    
    response = requests.post(
        "https://api.runwayml.com/v1/generate",
        headers={"Authorization": f"Bearer {RUNWAY_KEY}"},
        json={
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "model": "gen-3-alpha"
        }
    )
    
    task_id = response.json()["task_id"]
    
    # Poll for completion
    while True:
        status = requests.get(
            f"https://api.runwayml.com/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {RUNWAY_KEY}"}
        ).json()
        
        if status["status"] == "completed":
            return status["video_url"]
        
        time.sleep(5)
```

### Pika Labs

```python
def generate_pika_video(
    prompt: str,
    style: str = "cinematic"
) -> str:
    """Generate video with Pika"""
    
    # Pika offers Discord bot and API access
    response = requests.post(
        "https://api.pika.art/v1/generate",
        headers={"Authorization": f"Bearer {PIKA_KEY}"},
        json={
            "prompt": prompt,
            "style": style,  # cinematic, anime, 3d, etc.
            "fps": 24,
            "duration": 3
        }
    )
    
    return response.json()["video_url"]
```

---

## Image-to-Video

### Animate a Still Image

```python
def image_to_video(
    image_url: str,
    motion_prompt: str = "subtle movement"
) -> str:
    """Convert image to video with motion"""
    
    response = requests.post(
        "https://api.runwayml.com/v1/image-to-video",
        headers={"Authorization": f"Bearer {RUNWAY_KEY}"},
        json={
            "image_url": image_url,
            "motion_prompt": motion_prompt,
            "duration": 4
        }
    )
    
    return poll_for_result(response.json()["task_id"])

# Animate a photo
video = image_to_video(
    "https://example.com/landscape.jpg",
    "camera slowly zooms in, clouds moving gently"
)
```

### Extend Video

```python
def extend_video(
    video_url: str,
    prompt: str,
    extend_seconds: int = 4
) -> str:
    """Extend video with additional generated content"""
    
    response = requests.post(
        "https://api.runwayml.com/v1/extend",
        headers={"Authorization": f"Bearer {RUNWAY_KEY}"},
        json={
            "video_url": video_url,
            "prompt": prompt,
            "extend_by": extend_seconds
        }
    )
    
    return poll_for_result(response.json()["task_id"])
```

---

## Video Understanding

### Video Analysis with Gemini

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")

def analyze_video(video_path: str, prompt: str) -> str:
    """Analyze video content"""
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Upload video
    video_file = genai.upload_file(video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    
    # Analyze
    response = model.generate_content([video_file, prompt])
    
    return response.text

# Example
analysis = analyze_video(
    "product_demo.mp4",
    "Summarize this product demo and list key features shown"
)
```

### Frame-by-Frame Analysis

```python
import cv2
from openai import OpenAI
import base64

client = OpenAI()

def analyze_video_frames(
    video_path: str,
    prompt: str,
    frame_interval: int = 30  # Every 30 frames
) -> list:
    """Analyze video by sampling frames"""
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode()
            frames.append(base64_frame)
        
        frame_count += 1
    
    cap.release()
    
    # Analyze frames with vision model
    content = [{"type": "text", "text": prompt}]
    for frame in frames[:20]:  # Limit to 20 frames
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )
    
    return response.choices[0].message.content
```

---

## Current Capabilities

### What Works Well

| Capability | Quality | Notes |
|------------|---------|-------|
| Short clips (3-5s) | ⭐⭐⭐⭐ | Most reliable |
| Nature scenes | ⭐⭐⭐⭐⭐ | Excellent |
| Abstract art | ⭐⭐⭐⭐⭐ | Very good |
| Camera movements | ⭐⭐⭐⭐ | Pan, zoom work well |
| Slow motion | ⭐⭐⭐⭐ | Natural looking |

### Current Limitations

| Challenge | Status |
|-----------|--------|
| Human hands/fingers | Often malformed |
| Text in video | Usually garbled |
| Consistent characters | Difficult |
| Physics accuracy | Sometimes wrong |
| Long duration (>30s) | Quality degrades |
| Fine control | Limited |

---

## Model Comparison

| Feature | Sora | Runway Gen-3 | Pika | Kling |
|---------|------|--------------|------|-------|
| Max duration | 60s | 16s | 3s | 10s |
| Resolution | 1080p | 1080p | 1080p | 1080p |
| API Access | Limited | Yes | Yes | Limited |
| Cost | TBD | ~$0.05/s | ~$0.05/s | ~$0.03/s |
| Quality | Highest | High | Good | Good |
| Speed | Minutes | ~1 min | ~30s | ~1 min |

---

## Video Editing with AI

### Remove/Replace Objects

```python
def remove_object_from_video(
    video_path: str,
    object_mask: str,  # Mask indicating object to remove
) -> str:
    """Remove object from video using inpainting"""
    
    response = requests.post(
        "https://api.runwayml.com/v1/inpaint-video",
        headers={"Authorization": f"Bearer {RUNWAY_KEY}"},
        json={
            "video_url": video_path,
            "mask_url": object_mask,
            "prompt": "clean background"
        }
    )
    
    return poll_for_result(response.json()["task_id"])
```

### Style Transfer

```python
def stylize_video(
    video_path: str,
    style: str  # "anime", "oil_painting", "sketch"
) -> str:
    """Apply style to video"""
    
    response = requests.post(
        "https://api.runwayml.com/v1/stylize",
        json={
            "video_url": video_path,
            "style": style
        },
        headers={"Authorization": f"Bearer {RUNWAY_KEY}"}
    )
    
    return poll_for_result(response.json()["task_id"])
```

---

## Hands-on Exercise

### Your Task

Build a video processing wrapper:

```python
import google.generativeai as genai
import time

class VideoProcessor:
    """Video analysis and generation toolkit"""
    
    def __init__(self, gemini_key: str):
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
    
    def upload_video(self, path: str):
        """Upload and process video"""
        video_file = genai.upload_file(path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        return video_file
    
    def summarize(self, video_file) -> str:
        """Generate video summary"""
        response = self.model.generate_content([
            video_file,
            "Provide a detailed summary of this video."
        ])
        return response.text
    
    def extract_key_moments(self, video_file) -> list:
        """Extract key moments with timestamps"""
        response = self.model.generate_content([
            video_file,
            """Identify key moments in this video.
            Return as JSON: [{"timestamp": "0:30", "description": "..."}]"""
        ])
        import json
        try:
            return json.loads(response.text)
        except:
            return [{"raw": response.text}]
    
    def answer_question(self, video_file, question: str) -> str:
        """Answer question about video content"""
        response = self.model.generate_content([
            video_file,
            question
        ])
        return response.text

# Usage
processor = VideoProcessor("YOUR_GEMINI_KEY")
# video = processor.upload_video("demo.mp4")
# summary = processor.summarize(video)
# print(summary)
```

---

## Summary

✅ **Sora**: Highest quality, limited access

✅ **Runway Gen-3**: Best commercial API

✅ **Pika**: Fast, good for short clips

✅ **Gemini**: Best for video understanding

✅ **Limitations**: Hands, text, long duration still challenging

**Next:** [Multimodal Models](./11-multimodal-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Audio Models](./09-audio-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Multimodal Models](./11-multimodal-models.md) |

