---
title: "Fireworks AI"
---

# Fireworks AI

## Introduction

Fireworks AI is a high-performance inference platform optimized for open-source models. Known for production-ready infrastructure with enterprise SLAs, sub-100ms latencies, and seamless OpenAI API compatibility. Ideal for teams migrating from OpenAI or building production applications.

### What We'll Cover

- Platform architecture and speed optimizations
- Comprehensive model offerings (text, vision, audio, embeddings)
- OpenAI-compatible API usage
- Function calling and structured outputs
- Deployment options and pricing

### Prerequisites

- Python 3.8+
- Fireworks API key from [fireworks.ai](https://fireworks.ai/)
- Basic understanding of LLM APIs

---

## Platform Architecture

### Why Fireworks Is Fast

```
┌─────────────────────────────────────────────────────────────────┐
│                 FIREWORKS AI ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Key Optimizations:                                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ • Custom inference engine (not standard vLLM)           │     │
│  │ • Speculative decoding for faster generation           │     │
│  │ • Continuous batching for high throughput              │     │
│  │ • Optimized CUDA kernels                               │     │
│  │ • Smart model caching and preloading                   │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Deployment Options:                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────┐   │
│  │   Serverless    │   │   On-Demand     │   │  Fine-tuned  │   │
│  │   Pay-per-token │   │   Dedicated GPU │   │  Custom      │   │
│  │   Auto-scaling  │   │   Fast scaling  │   │  Models      │   │
│  └─────────────────┘   └─────────────────┘   └──────────────┘   │
│                                                                  │
│  Performance Metrics:                                            │
│  • TTFT: <100ms typical                                         │
│  • Throughput: Optimized for high concurrency                   │
│  • Uptime: 99.9% SLA available                                  │
│  • Global: Multiple regions for low latency                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Available Models

### Text Models

| Model | Context | Speed | Best For |
|-------|---------|-------|----------|
| Llama 3.3 70B | 128K | Fast | General chat, reasoning |
| Llama 3.1 405B | 128K | Moderate | Maximum quality |
| DeepSeek V3 | 128K | Fast | Code, reasoning |
| Mixtral 8x22B | 64K | Fast | MoE efficiency |
| Qwen 2.5 72B | 128K | Fast | Multilingual |
| FireFunction V2 | 32K | Very Fast | Tool calling |

### Vision Models

| Model | Context | Features |
|-------|---------|----------|
| Llama 3.2 Vision 90B | 128K | Image understanding |
| Llama 3.2 Vision 11B | 128K | Fast image analysis |
| Phi-3.5 Vision | 128K | Lightweight vision |

### Audio Models

| Model | Use Case | Features |
|-------|----------|----------|
| Whisper Large V3 | Transcription | 100+ languages |
| Whisper Large V3 Turbo | Fast transcription | 4x faster |

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| Nomic Embed | 768 | General embeddings |
| E5 Mistral | 4096 | High-quality retrieval |

---

## API Usage

### Basic Chat

```python
import fireworks.client as fireworks

fireworks.client.api_key = "YOUR_KEY"

def fireworks_chat(prompt: str) -> str:
    response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Streaming

```python
def fireworks_stream(prompt: str):
    response = fireworks.client.ChatCompletion.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### Function Calling (FireFunction)

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

# FireFunction optimized for tool use
response = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/firefunction-v2",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=tools
)
```

### OpenAI-Compatible API

```python
from openai import OpenAI

# Drop-in replacement - just change the base URL
client = OpenAI(
    api_key="YOUR_FIREWORKS_KEY",
    base_url="https://api.fireworks.ai/inference/v1"
)

def fireworks_chat_openai(prompt: str) -> str:
    """Use familiar OpenAI SDK with Fireworks"""
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Structured Outputs (JSON Mode)

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{
        "role": "user",
        "content": "Get weather info for San Francisco"
    }],
    response_format={
        "type": "json_object",
        "schema": WeatherInfo.model_json_schema()
    }
)
```

### Vision Models

```python
import base64

def fireworks_vision(image_path: str, question: str) -> str:
    """Analyze images with Llama Vision"""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }]
    )
    return response.choices[0].message.content
```

### Audio Transcription

```python
def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper"""
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-v3-turbo",
            file=audio_file,
            response_format="text"
        )
    return response

# Fast batch transcription available
```

### Embeddings

```python
def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for text"""
    
    response = client.embeddings.create(
        model="nomic-ai/nomic-embed-text-v1.5",
        input=texts
    )
    return [item.embedding for item in response.data]
```

---

## Deployment Options

### Serverless vs On-Demand

```python
deployment_options = {
    "serverless": {
        "pricing": "Pay per token",
        "scaling": "Automatic",
        "cold_start": "Minimal (models pre-loaded)",
        "best_for": "Variable workloads, prototyping"
    },
    "on_demand": {
        "pricing": "Per GPU hour",
        "scaling": "Fast auto-scaling",
        "cold_start": "Near-zero",
        "best_for": "Production, predictable traffic"
    },
    "fine_tuned": {
        "pricing": "Per GPU hour + training",
        "scaling": "Dedicated resources",
        "features": "Custom models up to 1T+ parameters",
        "best_for": "Specialized use cases"
    }
}
```

---

## Speed Features

### Low Latency

```python
fireworks_performance = {
    "time_to_first_token": "<100ms typical",
    "throughput": "Optimized for high concurrency",
    "reliability": "99.9% uptime SLA",
    "global": "Multiple regions available"
}
```

### Batch Inference

```python
# For high-volume, non-real-time workloads
batch_benefits = {
    "cost": "Up to 50% cheaper than real-time",
    "throughput": "Higher total throughput",
    "use_cases": [
        "Bulk document processing",
        "Dataset annotation",
        "Offline analysis"
    ]
}
```

---

## Pricing

### Serverless Pricing (Per Million Tokens)

| Model | Input | Output |
|-------|-------|--------|
| Llama 3.3 70B | $0.90 | $0.90 |
| Llama 3.1 405B | $3.00 | $3.00 |
| DeepSeek V3 | $0.90 | $0.90 |
| Mixtral 8x22B | $0.90 | $0.90 |
| FireFunction V2 | $0.90 | $0.90 |
| Whisper V3 | — | $0.20/audio min |
| Nomic Embed | $0.008 | — |

### On-Demand GPU Pricing

| GPU | Price/Hour | Best For |
|-----|------------|----------|
| A10G | ~$0.80 | Smaller models |
| A100 40GB | ~$2.50 | Medium models |
| A100 80GB | ~$3.50 | Large models |
| H100 | ~$5.00 | Maximum performance |

> **Tip:** On-demand deployments are more cost-effective for sustained traffic above ~100K tokens/hour.

---

## Migration from OpenAI

### Simple Migration

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY

# After (Fireworks)
from openai import OpenAI
client = OpenAI(
    api_key="YOUR_FIREWORKS_KEY",
    base_url="https://api.fireworks.ai/inference/v1"
)

# Same code works - just update model names
response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",  # Was "gpt-4"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Model Mapping

```python
model_migration = {
    "gpt-4o": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "gpt-4o-mini": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "gpt-4-turbo": "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "text-embedding-3-small": "nomic-ai/nomic-embed-text-v1.5"
}
```

---

## Best Practices

### When to Choose Fireworks

```python
choose_fireworks_when = [
    "Need OpenAI compatibility with open models",
    "Require enterprise SLAs (99.9% uptime)",
    "Want structured outputs with JSON schema",
    "Building production agentic applications",
    "Need multi-modal (text + vision + audio)",
    "Cost-sensitive but quality-focused"
]

consider_alternatives_when = [
    "Need proprietary models (GPT-4, Claude)",
    "Require absolute lowest latency (consider Groq)",
    "Want maximum cost savings (consider DeepSeek direct)"
]
```

---

## Summary

✅ **OpenAI compatible**: Drop-in replacement, same SDK

✅ **Production ready**: 99.9% uptime SLA, enterprise support

✅ **Multi-modal**: Text, vision, audio, embeddings

✅ **FireFunction**: Specialized function calling model

✅ **Flexible deployment**: Serverless, on-demand, fine-tuned

✅ **Structured outputs**: Native JSON schema support

**Next:** [Replicate](./10-replicate.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Together AI](./08-together-ai.md) | [AI Providers](./00-ai-providers-landscape.md) | [Replicate](./10-replicate.md) |

