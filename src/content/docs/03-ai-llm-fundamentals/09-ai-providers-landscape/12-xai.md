---
title: "xAI (Grok)"
---

# xAI (Grok)

## Introduction

xAI, founded by Elon Musk in 2023, develops the Grok family of AI models. Grok is designed to be witty, provide real-time information through X (Twitter) integration, and tackle questions that other AI systems might avoid. The API offers OpenAI-compatible endpoints with competitive pricing.

### What We'll Cover

- Grok model family and capabilities
- API setup and usage patterns
- Vision and multimodal features
- Real-time X integration
- Function calling and structured outputs
- Pricing and use cases

### Prerequisites

- Python 3.8+
- xAI API key from [console.x.ai](https://console.x.ai/)
- Familiarity with OpenAI-style APIs

---

## Model Lineup

### Current Models (2025-2026)

```
┌─────────────────────────────────────────────────────────────────┐
│                     GROK MODEL FAMILY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Grok 4.1 Fast (NEW)                                            │
│  ├── Context: 2,000,000 tokens (2M!)                            │
│  ├── Modalities: Text, Vision, Audio, Documents                 │
│  ├── Features: Function calling, Structured outputs             │
│  └── Optimized for: Agentic tool calling                        │
│                                                                  │
│  Grok 2                                                          │
│  ├── Context: 128K tokens                                       │
│  ├── Flagship general-purpose model                             │
│  └── Vision capable                                              │
│                                                                  │
│  Grok 2 Mini                                                     │
│  ├── Context: 128K tokens                                       │
│  ├── Faster, more cost-effective                                │
│  └── Good for simpler tasks                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Model | Context | Vision | Speed | Cost |
|-------|---------|--------|-------|------|
| Grok 4.1 Fast | 2M | ✅ | Very Fast | Medium |
| Grok 2 | 128K | ✅ | Fast | Higher |
| Grok 2 Mini | 128K | ✅ | Very Fast | Low |

---

## API Usage

### Setup

```python
from openai import OpenAI

# xAI uses OpenAI-compatible API
client = OpenAI(
    api_key="YOUR_XAI_KEY",
    base_url="https://api.x.ai/v1"
)
```

### Basic Chat

```python
def grok_chat(prompt: str, model: str = "grok-2-latest") -> str:
    """Basic chat with Grok"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Grok has a witty, conversational style
print(grok_chat("Tell me something interesting about black holes"))
```

### Streaming

```python
def grok_stream(prompt: str):
    """Stream responses for real-time output"""
    stream = client.chat.completions.create(
        model="grok-2-latest",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # Newline at end
```

### System Prompts

```python
def grok_with_persona(prompt: str, persona: str = None) -> str:
    """Customize Grok's personality"""
    
    messages = []
    if persona:
        messages.append({
            "role": "system",
            "content": persona
        })
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=messages
    )
    return response.choices[0].message.content

# Example: More formal Grok
result = grok_with_persona(
    "Explain quantum computing",
    persona="You are a formal academic lecturer. Be precise and educational."
)
```

---

## Vision Capabilities

### Image Understanding

```python
import base64

def grok_vision(image_path: str, question: str) -> str:
    """Analyze images with Grok Vision"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="grok-2-vision-latest",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]
    )
    return response.choices[0].message.content
```

### URL-Based Images

```python
def grok_vision_url(image_url: str, question: str) -> str:
    """Analyze images from URLs"""
    
    response = client.chat.completions.create(
        model="grok-2-vision-latest",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
    )
    return response.choices[0].message.content

# Example: OCR and document analysis
result = grok_vision_url(
    "https://example.com/receipt.jpg",
    "Extract all text from this receipt and itemize the purchases"
)
```

### Multiple Images

```python
def grok_compare_images(images: list[str], question: str) -> str:
    """Compare or analyze multiple images"""
    
    content = [{"type": "text", "text": question}]
    
    for img_path in images:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
        })
    
    response = client.chat.completions.create(
        model="grok-2-vision-latest",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content
```

---

## Function Calling

### Tool Definition

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_x",
            "description": "Search recent posts on X (Twitter)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["hour", "day", "week"],
                        "description": "How recent"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    }
]

def grok_with_tools(prompt: str):
    """Use Grok with function calling"""
    
    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Args: {tool_call.function.arguments}")
    
    return message
```

---

## Unique Features

### X/Twitter Integration

```python
grok_features = {
    "real_time_info": "Access to recent X/Twitter posts and trends",
    "current_events": "Up-to-date knowledge from social media",
    "personality": "Witty, conversational, willing to engage edgy topics",
    "vision": "Strong image understanding and OCR",
    "function_calling": "Full tool use support",
    "massive_context": "2M tokens on Grok 4.1 Fast"
}
```

### Best Use Cases

```python
grok_excels_at = [
    "Current events and trending topics",
    "Social media analysis",
    "Real-time information queries",
    "Conversational AI with personality",
    "Image analysis and OCR",
    "Long document processing (2M context)"
]
```

---

## Pricing

### Current Pricing (Per Million Tokens)

| Model | Input | Output |
|-------|-------|--------|
| Grok 4.1 Fast | $1.00 | $4.00 |
| Grok 2 | $2.00 | $10.00 |
| Grok 2 Mini | $0.30 | $0.50 |
| Grok 2 Vision | $2.00 | $10.00 |

### Cost Comparison

```python
cost_analysis = {
    "grok_2_mini": {
        "position": "Very competitive for fast tasks",
        "comparable_to": "GPT-4o-mini pricing tier"
    },
    "grok_2": {
        "position": "Premium tier",
        "comparable_to": "GPT-4o, Claude 3.5 Sonnet"
    },
    "grok_4_1_fast": {
        "position": "Best for agentic workflows",
        "unique": "2M context window"
    }
}
```

---

## When to Choose Grok

```python
choose_grok_when = [
    "Need real-time X/Twitter integration",
    "Want current events awareness",
    "Prefer witty, engaging AI personality",
    "Need massive context (2M tokens)",
    "Building social media applications",
    "Want OpenAI-compatible API"
]

consider_alternatives_when = [
    "Need maximum reasoning (consider Claude, o1)",
    "Require specialized code (consider DeepSeek)",
    "Want lowest cost (consider DeepSeek, Grok Mini)"
]
```

---

## Summary

✅ **Real-time info**: Integrated with X/Twitter data

✅ **Massive context**: 2M tokens on Grok 4.1 Fast

✅ **Vision capable**: Strong image understanding and OCR

✅ **OpenAI compatible**: Drop-in API replacement

✅ **Grok 2 Mini**: Cost-effective for simpler tasks

✅ **Unique personality**: Witty, engaging, fewer refusals

**Next:** [Alibaba Qwen](./13-alibaba-qwen.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [DeepSeek](./11-deepseek.md) | [AI Providers](./00-ai-providers-landscape.md) | [Alibaba Qwen](./13-alibaba-qwen.md) |

