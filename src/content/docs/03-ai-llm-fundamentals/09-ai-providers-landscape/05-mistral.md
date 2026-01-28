---
title: "Mistral AI"
---

# Mistral AI

## Introduction

Mistral AI is a French AI company known for efficient, high-quality models and the innovative Mixture of Experts (MoE) architecture. They offer both open-weight and commercial models.

### What We'll Cover

- Model lineup
- Mixture of Experts architecture
- API and pricing
- Open-weight options

---

## Model Lineup

### Current Models (2025-2026)

| Model | Type | Context | Best For |
|-------|------|---------|----------|
| Mistral Large | Proprietary | 128K | Top quality |
| Mistral Medium | Proprietary | 32K | Balanced |
| Mistral Small | Proprietary | 32K | Cost-effective |
| Codestral | Specialized | 32K | Code generation |
| Mixtral 8x7B | Open MoE | 32K | Self-hosting |
| Mixtral 8x22B | Open MoE | 64K | High-quality open |
| Mistral 7B | Open | 32K | Small, fast |

---

## Mixture of Experts (MoE)

### How MoE Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIXTURE OF EXPERTS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Token                                                     │
│       ↓                                                          │
│  ┌─────────┐                                                    │
│  │  Router │ → Selects best experts for this token              │
│  └─────────┘                                                    │
│       ↓                                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
│  │ E1  │ │ E2  │ │ E3  │ │ E4  │ │ E5  │ │ E6  │ │ E7  │ │ E8  ││
│  └──✓──┘ └─────┘ └──✓──┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
│     ↓               ↓    (Only 2 of 8 experts activated)        │
│  ┌───────────────────────┐                                      │
│  │    Combine Outputs    │                                      │
│  └───────────────────────┘                                      │
│                                                                  │
│  Benefits:                                                       │
│  - 8x7B parameters, but only uses 2x7B per token                │
│  - More knowledge, same inference cost                          │
│  - Different experts specialize in different tasks              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MoE Benefits

```python
moe_advantages = {
    "efficiency": "Only subset of parameters used per token",
    "capacity": "More total knowledge with same compute",
    "specialization": "Experts can specialize in domains",
    "speed": "Fast inference despite large total size"
}

# Mixtral 8x7B:
# - 47B total parameters
# - 13B active parameters per token
# - Quality closer to 70B dense model
```

---

## API Usage

### Basic Chat

```python
from mistralai import Mistral

client = Mistral(api_key="YOUR_KEY")

def mistral_chat(prompt: str, model: str = "mistral-large-latest") -> str:
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Usage
print(mistral_chat("Hello!"))
```

### Streaming

```python
def mistral_stream(prompt: str):
    stream = client.chat.stream(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    for chunk in stream:
        if chunk.data.choices[0].delta.content:
            print(chunk.data.choices[0].delta.content, end="", flush=True)
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

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools
)
```

---

## Codestral

### Code-Specialized Model

```python
def codestral_complete(code: str, instruction: str) -> str:
    """Code completion and generation"""
    
    response = client.chat.complete(
        model="codestral-latest",
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\n```python\n{code}\n```"
        }]
    )
    return response.choices[0].message.content

# Fill-in-the-middle (FIM)
def codestral_fim(prefix: str, suffix: str) -> str:
    response = client.fim.complete(
        model="codestral-latest",
        prompt=prefix,
        suffix=suffix
    )
    return response.choices[0].message.content
```

---

## Pricing

### Current Pricing

| Model | Input/1M | Output/1M |
|-------|----------|-----------|
| Mistral Large | $2.00 | $6.00 |
| Mistral Small | $0.20 | $0.60 |
| Codestral | $0.20 | $0.60 |
| Mixtral 8x7B (self-host) | Free | Free |

---

## Open-Weight Models

### Self-Hosting Mixtral

```python
from vllm import LLM, SamplingParams

# Mixtral 8x22B
llm = LLM(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    tensor_parallel_size=4
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2048
)

def generate(prompt: str) -> str:
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
```

### With Ollama

```bash
# Pull and run
ollama pull mixtral
ollama run mixtral "Hello!"
```

---

## European AI Focus

### Compliance Benefits

```python
mistral_compliance = {
    "headquarters": "Paris, France",
    "data_processing": "EU-based options available",
    "gdpr": "Strong GDPR compliance focus",
    "sovereignty": "European AI alternative"
}
```

---

## Summary

✅ **MoE architecture**: Efficient, high capacity

✅ **Codestral**: Excellent for code tasks

✅ **Open weights**: Mixtral available for self-hosting

✅ **European provider**: EU compliance benefits

✅ **Competitive pricing**: Good value proposition

**Next:** [Cohere](./06-cohere.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Meta](./04-meta.md) | [AI Providers](./00-ai-providers-landscape.md) | [Cohere](./06-cohere.md) |

