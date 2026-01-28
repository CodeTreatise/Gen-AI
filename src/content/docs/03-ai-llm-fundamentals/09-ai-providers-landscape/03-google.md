---
title: "Google (Gemini)"
---

# Google (Gemini)

## Introduction

Google's Gemini models are built natively multimodal and offer industry-leading context windows up to 2 million tokens. Deep integration with Google Cloud makes them excellent for enterprise applications.

### What We'll Cover

- Gemini model variants
- Native multimodal capabilities
- Massive context windows
- Vertex AI platform

---

## Model Lineup

### Current Models (2025-2026)

| Model | Context | Best For |
|-------|---------|----------|
| Gemini 2.5 Pro | 1M-2M | Flagship, all tasks |
| Gemini 2.5 Flash | 1M | Fast, cost-effective |
| Gemini 2.0 Flash | 1M | Previous gen fast |
| Gemini 1.5 Pro | 1M | Proven, stable |

### Model Selection

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")

def select_gemini_model(task: dict) -> str:
    """Select appropriate Gemini model"""
    
    if task.get("context_tokens", 0) > 200000:
        # Gemini's strength: massive context
        if task.get("priority") == "speed":
            return "gemini-2.5-flash"
        return "gemini-2.5-pro"
    
    if task.get("priority") == "cost":
        return "gemini-1.5-flash"
    
    return "gemini-2.5-pro"
```

---

## Native Multimodal

### All Modalities Built-In

```python
from PIL import Image

model = genai.GenerativeModel("gemini-2.5-pro")

# Text + Image
def analyze_image(image_path: str, question: str) -> str:
    image = Image.open(image_path)
    response = model.generate_content([image, question])
    return response.text

# Text + Video
def analyze_video(video_path: str, question: str) -> str:
    video_file = genai.upload_file(video_path)
    
    # Wait for processing
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    
    response = model.generate_content([video_file, question])
    return response.text

# Text + Audio
def analyze_audio(audio_path: str, question: str) -> str:
    audio_file = genai.upload_file(audio_path)
    response = model.generate_content([audio_file, question])
    return response.text
```

### Combined Inputs

```python
def multimodal_analysis(
    text: str = None,
    image_path: str = None,
    video_path: str = None,
    audio_path: str = None
) -> str:
    """Analyze multiple modalities together"""
    
    content = []
    
    if image_path:
        content.append(Image.open(image_path))
    
    if video_path:
        video = genai.upload_file(video_path)
        content.append(video)
    
    if audio_path:
        audio = genai.upload_file(audio_path)
        content.append(audio)
    
    if text:
        content.append(text)
    
    response = model.generate_content(content)
    return response.text
```

---

## Massive Context Window

### 1-2 Million Tokens

```python
# Gemini 2.5 Pro: Up to 2M tokens
# That's approximately:
# - 4,000 pages of text
# - 2 hours of video
# - 22 hours of audio

def analyze_entire_codebase(files: list) -> str:
    """Analyze large codebase in single request"""
    
    content = []
    
    for file_path in files:
        with open(file_path, 'r') as f:
            content.append(f"# {file_path}\n{f.read()}\n\n")
    
    full_content = "".join(content)
    
    response = model.generate_content([
        full_content,
        "Analyze this codebase. Identify patterns, issues, and improvements."
    ])
    
    return response.text
```

### Long Video Analysis

```python
def analyze_long_video(video_path: str) -> dict:
    """Analyze hour-long video"""
    
    video = genai.upload_file(video_path)
    
    # Wait for processing
    import time
    while video.state.name == "PROCESSING":
        time.sleep(5)
        video = genai.get_file(video.name)
    
    # Full analysis
    response = model.generate_content([
        video,
        """Provide comprehensive analysis:
        1. Summary
        2. Key moments with timestamps
        3. Main topics discussed
        4. Action items mentioned
        5. Notable quotes"""
    ])
    
    return {"analysis": response.text}
```

---

## API Usage

### Basic Chat

```python
def gemini_chat(prompt: str) -> str:
    """Basic Gemini chat"""
    
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text
```

### Conversation

```python
def gemini_conversation():
    """Multi-turn conversation"""
    
    model = genai.GenerativeModel("gemini-2.5-pro")
    chat = model.start_chat(history=[])
    
    response1 = chat.send_message("Hello! I'm building an AI app.")
    print(response1.text)
    
    response2 = chat.send_message("What framework do you recommend?")
    print(response2.text)
```

### Streaming

```python
def gemini_stream(prompt: str):
    """Stream Gemini response"""
    
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt, stream=True)
    
    for chunk in response:
        print(chunk.text, end="", flush=True)
```

---

## Function Calling

```python
def get_weather(location: str) -> dict:
    return {"temperature": 22, "condition": "sunny"}

weather_tool = genai.protos.Tool(
    function_declarations=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }]
)

model = genai.GenerativeModel(
    "gemini-2.5-pro",
    tools=[weather_tool]
)

response = model.generate_content("What's the weather in Tokyo?")
```

---

## Vertex AI

### Enterprise Platform

```python
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
aiplatform.init(project="your-project", location="us-central1")

# Use Gemini via Vertex
model = GenerativeModel("gemini-2.5-pro")
response = model.generate_content("Hello!")
```

### Vertex Benefits

```python
vertex_advantages = {
    "enterprise_security": "VPC, IAM, encryption",
    "compliance": "SOC 2, HIPAA, FedRAMP",
    "global_regions": "Deploy in specific regions",
    "unified_billing": "Single Google Cloud bill",
    "grounding": "Ground responses in Google Search",
    "model_garden": "Access to many models",
}
```

---

## Pricing

### Current Pricing

| Model | Input/1M | Output/1M |
|-------|----------|-----------|
| Gemini 2.5 Pro | ~$1.25 | ~$5.00 |
| Gemini 2.5 Flash | ~$0.075 | ~$0.30 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |

### Context Pricing Note

```python
# Gemini charges for context length
# Prompts >128K tokens may have different rates
# Check current pricing for long context
```

---

## Best Use Cases

```python
gemini_best_for = [
    "Very long documents (books, transcripts)",
    "Video analysis and understanding",
    "Multi-modal applications",
    "Google Cloud integrated apps",
    "Cost-effective high-quality inference",
    "Massive context requirements",
]
```

---

## Summary

✅ **Native multimodal**: Image, video, audio built-in

✅ **Massive context**: Up to 2M tokens

✅ **Gemini Flash**: Very fast and affordable

✅ **Vertex AI**: Enterprise-ready platform

✅ **Google ecosystem**: Search grounding, Cloud integration

**Next:** [Meta (Llama)](./04-meta.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Anthropic](./02-anthropic.md) | [AI Providers](./00-ai-providers-landscape.md) | [Meta](./04-meta.md) |

