---
title: "Groq"
---

# Groq

## Introduction

Groq provides the fastest LLM inference available, powered by their custom LPU (Language Processing Unit) hardware. Ideal for real-time applications where latency is critical.

### What We'll Cover

- LPU technology
- Available models
- API usage
- Real-time use cases

---

## LPU Technology

### What Makes Groq Fast

```
┌─────────────────────────────────────────────────────────────────┐
│                    GROQ LPU ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional GPU:                                                │
│  ┌──────────────────────────────────────┐                       │
│  │ Memory ←→ Compute (bottleneck)       │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
│  Groq LPU:                                                       │
│  ┌──────────────────────────────────────┐                       │
│  │ Deterministic execution flow         │                       │
│  │ No memory bandwidth bottleneck       │                       │
│  │ Predictable, ultra-low latency       │                       │
│  └──────────────────────────────────────┘                       │
│                                                                  │
│  Speed Comparison:                                               │
│  • GPT-4o: ~50 tokens/sec                                       │
│  • Llama 70B (GPU): ~30-50 tokens/sec                           │
│  • Llama 70B (Groq): ~300+ tokens/sec                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Metrics

```python
groq_performance = {
    "llama_3_70b": {
        "tokens_per_second": 300,
        "time_to_first_token": "<100ms",
        "comparison": "10x faster than typical GPU inference"
    },
    "mixtral_8x7b": {
        "tokens_per_second": 500,
        "time_to_first_token": "<50ms",
        "comparison": "Among fastest available"
    }
}
```

---

## Available Models

### Current Models (2025-2026)

| Model | Context | Speed |
|-------|---------|-------|
| Llama 3.3 70B | 128K | ~300 tok/s |
| Llama 3.1 70B | 128K | ~300 tok/s |
| Llama 3.1 8B | 128K | ~1000 tok/s |
| Mixtral 8x7B | 32K | ~500 tok/s |
| Gemma 2 9B | 8K | ~800 tok/s |

---

## API Usage

### Basic Chat

```python
from groq import Groq

client = Groq()

def groq_chat(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Ultra-fast response
print(groq_chat("Hello!"))
```

### Streaming

```python
def groq_stream(prompt: str):
    """Stream for real-time output"""
    
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools,
    tool_choice="auto"
)
```

---

## Real-Time Use Cases

### Voice Applications

```python
import time

class RealtimeAssistant:
    """Ultra-low latency for voice"""
    
    def __init__(self):
        self.client = Groq()
        self.model = "llama-3.1-8b-instant"  # Fastest
    
    def respond(self, transcript: str) -> str:
        """Get response fast enough for voice"""
        
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": transcript}],
            max_tokens=100  # Limit for speed
        )
        
        latency = (time.time() - start) * 1000
        print(f"Response in {latency:.0f}ms")
        
        return response.choices[0].message.content

# Usage
assistant = RealtimeAssistant()
# Response typically in <200ms total
```

### Gaming NPCs

```python
class GameNPC:
    """Fast NPC responses for games"""
    
    def __init__(self, character: str):
        self.client = Groq()
        self.character = character
        self.model = "llama-3.1-8b-instant"
    
    def speak(self, player_input: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.character}. Respond in character, briefly."
                },
                {"role": "user", "content": player_input}
            ],
            max_tokens=50,  # Keep responses short
            temperature=0.8
        )
        return response.choices[0].message.content

# NPC responds in real-time
npc = GameNPC("a grumpy blacksmith named Gorn")
print(npc.speak("Can you fix my sword?"))
```

### Autocomplete

```python
def fast_autocomplete(partial_text: str) -> str:
    """Real-time autocomplete suggestions"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Complete this text naturally: {partial_text}"
        }],
        max_tokens=30,
        temperature=0.3
    )
    
    return response.choices[0].message.content
```

---

## Pricing

### Current Pricing

| Model | Input/1M | Output/1M |
|-------|----------|-----------|
| Llama 3.3 70B | $0.59 | $0.79 |
| Llama 3.1 8B | $0.05 | $0.08 |
| Mixtral 8x7B | $0.24 | $0.24 |

### Value Proposition

```python
groq_value = {
    "speed": "10x faster than GPU alternatives",
    "price": "Competitive with other providers",
    "tradeoff": "Limited to open models (no GPT-4, Claude)",
    "best_for": "When latency matters more than model choice"
}
```

---

## Limitations

```python
groq_limitations = {
    "models": "Only open-source models (Llama, Mixtral)",
    "context": "Some models have shorter context",
    "features": "Fewer advanced features than OpenAI",
    "rate_limits": "Lower limits on free tier",
    "vision": "Limited multimodal support"
}
```

---

## Summary

✅ **Fastest inference**: 300+ tokens/sec on Llama 70B

✅ **LPU technology**: Purpose-built for LLMs

✅ **Open models**: Llama, Mixtral, Gemma

✅ **Real-time ready**: <100ms TTFT possible

✅ **Competitive pricing**: Speed without premium

**Next:** [Together AI](./08-together-ai.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Cohere](./06-cohere.md) | [AI Providers](./00-ai-providers-landscape.md) | [Together AI](./08-together-ai.md) |

