---
title: "Other Providers"
---

# Other Providers

## Introduction

Beyond the major players, several other companies offer compelling AI models with unique strengths. This lesson covers AI21 Labs, MiniMax, NVIDIA, Zhipu AI, and Moonshot AI.

---

## AI21 Labs

### Jamba Models

AI21 Labs developed Jamba, a hybrid architecture combining Transformers with Mamba (state-space models).

```python
from ai21 import AI21Client

client = AI21Client(api_key="YOUR_KEY")

def ai21_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="jamba-1.5-large",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Key Features

```python
ai21_features = {
    "jamba_architecture": "Transformer + Mamba hybrid",
    "context": "256K tokens",
    "efficiency": "Fast inference, lower memory",
    "specialties": ["Enterprise", "RAG", "Summarization"]
}
```

---

## MiniMax

### Chinese AI Company

MiniMax offers models with strong multilingual capabilities and long context.

```python
minimax_models = {
    "abab6.5": "Latest flagship",
    "context": "Up to 1M tokens",
    "specialties": ["Chinese", "Creative writing", "Long documents"]
}
```

---

## NVIDIA

### Nemotron Models

NVIDIA's Nemotron models are optimized for enterprise deployment on NVIDIA hardware.

```python
from openai import OpenAI

# Via NVIDIA API Catalog
client = OpenAI(
    api_key="YOUR_NVIDIA_KEY",
    base_url="https://integrate.api.nvidia.com/v1"
)

def nemotron_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### NVIDIA AI Enterprise

```python
nvidia_offerings = {
    "nemotron": "High-performance models",
    "nim": "NVIDIA Inference Microservices",
    "api_catalog": "Access to many models",
    "optimized": "Best performance on NVIDIA GPUs"
}
```

---

## Zhipu AI (GLM)

### Chinese AI Research

Zhipu AI develops the GLM (General Language Model) series, strong in Chinese.

```python
zhipu_models = {
    "glm-4": "Flagship model",
    "glm-4v": "Vision capable",
    "context": "128K tokens",
    "strengths": ["Chinese NLP", "Vision", "Code"]
}

# API access via zhipuai.cn
```

---

## Moonshot AI (Kimi)

### Long Context Specialist

Moonshot AI's Kimi models are known for extremely long context windows.

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_MOONSHOT_KEY",
    base_url="https://api.moonshot.cn/v1"
)

def kimi_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="moonshot-v1-128k",  # or 32k, 8k
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Context Options

```python
kimi_contexts = {
    "moonshot-v1-8k": "8K tokens",
    "moonshot-v1-32k": "32K tokens",
    "moonshot-v1-128k": "128K tokens"
}
```

---

## Comparison Table

| Provider | Model | Strength |
|----------|-------|----------|
| AI21 Labs | Jamba | Hybrid architecture |
| MiniMax | abab | Long context, Chinese |
| NVIDIA | Nemotron | Enterprise, hardware-optimized |
| Zhipu | GLM-4 | Chinese, multimodal |
| Moonshot | Kimi | Very long context |

---

## Summary

✅ **AI21 Jamba**: Innovative hybrid architecture

✅ **NVIDIA Nemotron**: Enterprise-ready, hardware-optimized

✅ **Zhipu GLM**: Strong Chinese, vision

✅ **Moonshot Kimi**: Long context specialist

✅ **Each has niche**: Choose based on specific needs

**Next:** [Open Source Tools](./15-open-source-tools.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Alibaba Qwen](./13-alibaba-qwen.md) | [AI Providers](./00-ai-providers-landscape.md) | [Open Source Tools](./15-open-source-tools.md) |

