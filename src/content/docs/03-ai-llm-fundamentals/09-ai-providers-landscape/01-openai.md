---
title: "OpenAI"
---

# OpenAI

## Introduction

OpenAI is the pioneer of modern LLMs and remains a leader in AI capabilities. Their GPT series, o-series reasoning models, and multimodal offerings make them a go-to choice for many applications.

### What We'll Cover

- Model lineup (GPT-4o, o-series, GPT-5)
- Pricing and tiers
- API features
- Enterprise offerings

---

## Model Lineup

### Current Models (2025-2026)

| Model | Context | Best For | Released |
|-------|---------|----------|----------|
| GPT-4o | 128K | General purpose, multimodal | 2024 |
| GPT-4o-mini | 128K | Cost-effective, fast | 2024 |
| GPT-4.1 | 1M | Long context tasks | 2025 |
| GPT-4.1-mini | 1M | Fast, long context | 2025 |
| GPT-5 | TBD | Flagship, latest capabilities | 2025 |
| o1 | 128K | Complex reasoning, math | 2024 |
| o3 | 200K | Advanced reasoning | 2025 |
| o4-mini | 128K | Fast reasoning | 2025 |

### Model Selection

```python
from openai import OpenAI

client = OpenAI()

def select_openai_model(task: dict) -> str:
    """Select appropriate OpenAI model"""
    
    if task.get("needs_reasoning"):
        if task.get("priority") == "speed":
            return "o4-mini"
        return "o3"  # or o1 for complex math
    
    if task.get("context_tokens", 0) > 128000:
        return "gpt-4.1"  # 1M context
    
    if task.get("priority") == "cost":
        return "gpt-4o-mini"
    
    if task.get("needs_latest"):
        return "gpt-5"  # Flagship
    
    return "gpt-4o"  # Default: great all-around

# Usage
model = select_openai_model({
    "needs_reasoning": False,
    "priority": "quality"
})
```

---

## GPT-4o Family

### Capabilities

```python
gpt4o_capabilities = {
    "text": True,
    "vision": True,
    "audio_input": True,   # Realtime API
    "audio_output": True,  # Realtime API
    "function_calling": True,
    "json_mode": True,
    "streaming": True,
    "structured_outputs": True,
}

# Vision example
def analyze_image(image_url: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
    )
    return response.choices[0].message.content
```

### GPT-4o-mini

```python
# Perfect for high-volume, cost-sensitive applications
def quick_classification(text: str, categories: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ~20x cheaper than GPT-4o
        messages=[{
            "role": "user",
            "content": f"Classify this text into one of {categories}:\n\n{text}"
        }],
        max_tokens=50
    )
    return response.choices[0].message.content
```

---

## O-Series Reasoning Models

### Extended Thinking

```python
# o1 and o3 use "chain of thought" internally
def solve_complex_problem(problem: str) -> str:
    response = client.chat.completions.create(
        model="o3",  # or "o1" for math-heavy
        messages=[{
            "role": "user",
            "content": problem
        }]
        # Note: o-series may take longer but produce better reasoning
    )
    return response.choices[0].message.content

# Good for:
# - Mathematical proofs
# - Complex coding challenges
# - Multi-step reasoning
# - Scientific analysis
```

### When to Use O-Series

```python
use_o_series_for = [
    "PhD-level questions",
    "Mathematical proofs",
    "Complex code debugging",
    "Scientific reasoning",
    "Multi-step planning",
    "Competition-level problems"
]

use_gpt4o_for = [
    "General conversation",
    "Content generation",
    "Simple code tasks",
    "Image analysis",
    "Real-time applications",
    "High-volume processing"
]
```

---

## Pricing Structure

### Current Pricing

| Model | Input/1M | Output/1M | Cached Input |
|-------|----------|-----------|--------------|
| GPT-4o | $2.50 | $10.00 | $1.25 |
| GPT-4o-mini | $0.15 | $0.60 | $0.075 |
| o1 | $15.00 | $60.00 | $7.50 |
| o3 | ~$20.00 | ~$80.00 | ~$10.00 |
| GPT-4.1 | ~$2.00 | ~$8.00 | ~$1.00 |

### Prompt Caching

```python
# Automatically caches repeated prompt prefixes
# Saves 50% on input tokens for cached content

# Good for:
# - System prompts
# - Few-shot examples
# - Repeated context
```

---

## API Features

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
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
```

### Structured Outputs

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "John Doe, 30, john@example.com"}],
    response_format=UserInfo
)

user = response.choices[0].message.parsed
print(user.name)  # "John Doe"
```

### Batch API

```python
# 50% discount for non-time-sensitive workloads
batch = client.batches.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Check status
status = client.batches.retrieve(batch.id)
```

---

## Rate Limits and Tiers

### Tier System

| Tier | Qualification | GPT-4o RPM | GPT-4o TPM |
|------|--------------|------------|------------|
| Free | New accounts | 3 | 40,000 |
| Tier 1 | $5+ paid | 500 | 30,000 |
| Tier 2 | $50+ paid | 5,000 | 450,000 |
| Tier 3 | $100+ paid | 5,000 | 800,000 |
| Tier 4 | $250+ paid | 10,000 | 2,000,000 |
| Tier 5 | $1,000+ paid | 10,000 | 10,000,000 |

---

## Enterprise Offerings

### OpenAI Enterprise

```python
enterprise_features = {
    "soc2_compliance": True,
    "sso": True,
    "admin_console": True,
    "usage_analytics": True,
    "dedicated_support": True,
    "custom_rate_limits": True,
    "data_not_used_for_training": True,  # API default now
    "sla": "99.9%"
}
```

---

## Summary

✅ **GPT-4o**: Best general-purpose model

✅ **GPT-4o-mini**: Cost-effective workhorse

✅ **O-series**: Complex reasoning tasks

✅ **GPT-5**: Latest flagship capabilities

✅ **Rich API features**: Functions, vision, structured outputs

**Next:** [Anthropic](./02-anthropic.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Overview](./00-ai-providers-landscape.md) | [AI Providers](./00-ai-providers-landscape.md) | [Anthropic](./02-anthropic.md) |

