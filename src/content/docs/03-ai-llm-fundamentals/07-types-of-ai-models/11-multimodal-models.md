---
title: "Multimodal Models"
---

# Multimodal Models

## Introduction

Multimodal models process multiple types of input and output—text, images, audio, and video. These unified models simplify architectures and enable rich cross-modal interactions.

### What We'll Cover

- What makes models multimodal
- GPT-4o capabilities
- Gemini native multimodal
- Claude vision and documents
- Unified vs separate models

---

## What Are Multimodal Models?

### Definition

```
Traditional Models:
──────────────────
Text Model:  Text → Text
Vision Model: Image → Text
Audio Model: Audio → Text
(Separate models, separate APIs)

Multimodal Models:
──────────────────
Single Model: Text + Image + Audio + Video → Text + Image + Audio
(One model, one API, all modalities)
```

### Benefits

```python
multimodal_advantages = {
    "unified_context": "All inputs understood together",
    "cross_modal_reasoning": "Can reason across text and images",
    "simpler_architecture": "One model instead of many",
    "emergent_capabilities": "Novel cross-modal abilities",
    "natural_interaction": "More human-like understanding",
}
```

---

## GPT-4o Multimodal

### Text + Image

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_with_text(image_path: str, text: str) -> str:
    """Combine text and image analysis"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]
    )
    
    return response.choices[0].message.content

# Example: Analyze product with text context
result = analyze_with_text(
    "product.jpg",
    "This is our new product. Write marketing copy based on what you see."
)
```

### Text + Multiple Images

```python
def compare_images_with_context(
    images: list,
    context: str,
    question: str
) -> str:
    """Analyze multiple images with text context"""
    
    content = [
        {"type": "text", "text": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    for img_path in images:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )
    
    return response.choices[0].message.content

# Compare design iterations
result = compare_images_with_context(
    ["v1.png", "v2.png", "v3.png"],
    "These are iterations of our logo design",
    "Which version is most professional and why?"
)
```

### Audio Capabilities

```python
# GPT-4o Realtime for native audio
import websockets
import json

async def multimodal_conversation():
    """Real-time multimodal conversation"""
    
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
        additional_headers={"Authorization": f"Bearer {API_KEY}"}
    ) as ws:
        # Send text + image + audio together
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "audio", "audio": {"data": audio_base64}}
                ]
            }
        }))
```

---

## Gemini Native Multimodal

Gemini is built from the ground up as multimodal.

### All Modalities at Once

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

def process_all_modalities(
    text: str = None,
    image_path: str = None,
    video_path: str = None,
    audio_path: str = None
) -> str:
    """Process multiple modalities together"""
    
    content = []
    
    if text:
        content.append(text)
    
    if image_path:
        content.append(Image.open(image_path))
    
    if video_path:
        video_file = genai.upload_file(video_path)
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        content.append(video_file)
    
    if audio_path:
        audio_file = genai.upload_file(audio_path)
        content.append(audio_file)
    
    response = model.generate_content(content)
    return response.text

# Example: Analyze video with context
result = process_all_modalities(
    text="Summarize this video and identify any issues",
    video_path="demo.mp4"
)
```

### Long Context Advantage

```python
# Gemini 1.5 Pro: 1 million token context
# Can process:
# - ~1 hour of video
# - ~11 hours of audio
# - ~700,000 words of text
# - ~3,000 images

def analyze_long_video(video_path: str) -> str:
    """Analyze hour-long video"""
    
    video = genai.upload_file(video_path)
    
    response = model.generate_content([
        video,
        """Provide comprehensive analysis:
        1. Full summary
        2. Key moments with timestamps
        3. Important quotes
        4. Action items mentioned
        5. Overall sentiment"""
    ])
    
    return response.text
```

---

## Claude Vision and Documents

### Image Understanding

```python
from anthropic import Anthropic
import base64

client = Anthropic()

def analyze_with_claude(image_path: str, prompt: str) -> str:
    """Analyze image with Claude"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    media_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    return response.content[0].text
```

### Document Processing

```python
def analyze_pdf_claude(pdf_path: str, prompt: str) -> str:
    """Analyze PDF natively with Claude"""
    
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    
    return response.content[0].text

# Analyze a contract
result = analyze_pdf_claude(
    "contract.pdf",
    "Extract key terms, obligations, and any concerning clauses"
)
```

---

## Unified vs Separate Models

### Comparison

| Aspect | Unified (GPT-4o) | Separate Models |
|--------|------------------|-----------------|
| Latency | Lower | Higher (multiple calls) |
| Cost | Often lower | Can be optimized |
| Context | Shared across modalities | Separate contexts |
| Capabilities | Jack of all trades | Specialists |
| Flexibility | Less | More |

### When to Use Each

```python
# Use UNIFIED models when:
unified_use_cases = [
    "Cross-modal reasoning required",
    "Real-time interaction",
    "Simplicity is priority",
    "Context must be shared",
]

# Use SEPARATE models when:
separate_use_cases = [
    "Maximum quality per modality",
    "Cost optimization",
    "Specialized tasks",
    "Fine-grained control needed",
]
```

### Hybrid Architecture

```python
class HybridProcessor:
    """Combine unified and specialized models"""
    
    def __init__(self):
        self.unified = OpenAI()  # GPT-4o
        self.specialized_ocr = OpenAI()  # Specialized
        self.specialized_audio = AssemblyAI()  # Specialized
    
    def process(self, inputs: dict) -> dict:
        """Route to best model for each task"""
        
        results = {}
        
        # Use specialized for specific tasks
        if "audio" in inputs and inputs.get("need_diarization"):
            results["transcript"] = self.specialized_audio.transcribe(
                inputs["audio"],
                speaker_labels=True
            )
        
        # Use unified for cross-modal
        if "image" in inputs and "text" in inputs:
            results["analysis"] = self.unified.analyze(
                inputs["image"],
                inputs["text"]
            )
        
        return results
```

---

## Hands-on Exercise

### Your Task

Build a multimodal assistant:

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

class MultimodalAssistant:
    """Assistant that handles multiple input types"""
    
    def __init__(self):
        self.conversation = []
    
    def chat(
        self,
        text: str = None,
        images: list = None,
        audio: str = None
    ) -> str:
        """Process multimodal input"""
        
        content = []
        
        # Add text
        if text:
            content.append({"type": "text", "text": text})
        
        # Add images
        if images:
            for img_path in images:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                })
        
        # Add audio (transcribe first)
        if audio:
            with open(audio, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            content.append({
                "type": "text",
                "text": f"[Audio transcript]: {transcript.text}"
            })
        
        # Add to conversation
        self.conversation.append({
            "role": "user",
            "content": content
        })
        
        # Get response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation
        )
        
        assistant_message = response.choices[0].message.content
        
        self.conversation.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset(self):
        """Reset conversation"""
        self.conversation = []

# Usage
assistant = MultimodalAssistant()

# Text only
print(assistant.chat(text="Hello! What can you help me with?"))

# Text + image
# print(assistant.chat(
#     text="What's in this image?",
#     images=["photo.jpg"]
# ))

# Multiple images
# print(assistant.chat(
#     text="Compare these designs",
#     images=["design1.png", "design2.png"]
# ))
```

---

## Summary

✅ **GPT-4o**: Excellent multimodal, real-time capable

✅ **Gemini**: Native multimodal, massive context

✅ **Claude**: Strong vision and document processing

✅ **Unified models**: Simpler, shared context

✅ **Hybrid approach**: Best of both worlds

**Next:** [Document Understanding Models](./12-document-understanding-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Video Models](./10-video-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Document Understanding](./12-document-understanding-models.md) |

