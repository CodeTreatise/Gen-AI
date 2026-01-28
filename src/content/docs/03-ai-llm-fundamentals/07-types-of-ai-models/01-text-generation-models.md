---
title: "Text Generation Models"
---

# Text Generation Models

## Introduction

Text generation models are the foundation of modern AI applications. These large language models (LLMs) generate human-like text for chat, content creation, analysis, and more.

### What We'll Cover

- Leading text generation models
- Model comparison and strengths
- Pricing and context windows
- Choosing the right model

---

## The Major Players

### GPT-4 / GPT-4o (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",  # Latest multimodal model
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

**Strengths:**
- Strong reasoning and instruction following
- Excellent at complex tasks
- Broad knowledge base
- Great tool use / function calling

**Best for:** General-purpose applications, complex reasoning, code generation

### Claude 3.5 (Anthropic)

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.content[0].text)
```

**Strengths:**
- Excellent at long documents (200K context)
- Strong writing quality
- Good at nuanced tasks
- Honest about limitations

**Best for:** Document analysis, writing, research, nuanced conversations

### Gemini (Google)

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content("Explain quantum computing")
print(response.text)
```

**Strengths:**
- Massive context window (1M+ tokens)
- Native multimodal (text, image, video, audio)
- Strong reasoning
- Google ecosystem integration

**Best for:** Long documents, multimodal applications, Google Cloud users

### LLaMA / Open Models (Meta)

```python
# Via Ollama (local)
import ollama

response = ollama.chat(
    model="llama3.1:70b",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response["message"]["content"])
```

**Strengths:**
- Open weights (can run locally)
- No API costs for self-hosted
- Customizable / fine-tunable
- Privacy (data stays local)

**Best for:** Self-hosted, privacy-sensitive, custom fine-tuning

---

## Model Comparison

### Capabilities Matrix

| Capability | GPT-4o | Claude 3.5 | Gemini 1.5 | LLaMA 3.1 |
|------------|--------|------------|------------|-----------|
| Reasoning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Writing | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Code | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Math | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Vision | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Tool Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Context Windows (2025)

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128K tokens |
| GPT-4 Turbo | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Claude 3 Opus | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Gemini 1.5 Flash | 1M tokens |
| LLaMA 3.1 405B | 128K tokens |

### Pricing Comparison (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |

> **Note:** Prices as of late 2024. Check provider websites for current pricing.

---

## Model Tiers

### Tier 1: Flagship Models

```python
# Maximum capability, highest cost
flagship_models = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-pro",
}

# Use for:
# - Complex reasoning
# - Production applications
# - Quality-critical tasks
```

### Tier 2: Balanced Models

```python
# Good performance, lower cost
balanced_models = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "google": "gemini-1.5-flash",
}

# Use for:
# - High-volume applications
# - Simpler tasks
# - Cost-sensitive production
```

### Tier 3: Economy Models

```python
# Fast and cheap
economy_models = {
    "openai": "gpt-3.5-turbo",
    "google": "gemini-1.5-flash-8b",
}

# Use for:
# - Very high volume
# - Simple classification
# - Prototyping
```

---

## Strengths and Weaknesses

### GPT-4o

| Strengths | Weaknesses |
|-----------|------------|
| Best tool use / function calling | Can be verbose |
| Strong reasoning | Occasional refusals |
| Reliable outputs | Higher latency |
| Excellent code | Price for high volume |

### Claude 3.5 Sonnet

| Strengths | Weaknesses |
|-----------|------------|
| Excellent writing quality | Slightly slower |
| Great at long documents | Less tool use polish |
| Nuanced understanding | Occasional verbosity |
| Honest responses | Smaller ecosystem |

### Gemini 1.5

| Strengths | Weaknesses |
|-----------|------------|
| Massive context window | Variable quality |
| Native multimodal | Less predictable |
| Very fast (Flash) | Google lock-in risk |
| Good pricing | Newer ecosystem |

---

## Choosing the Right Model

### Decision Framework

```python
def choose_text_model(requirements: dict) -> str:
    """Select best model based on requirements"""
    
    # Priority: capability > cost > latency
    
    if requirements.get("max_context") and requirements["max_context"] > 200_000:
        return "gemini-1.5-pro"  # Only option for >200K
    
    if requirements.get("local_deployment"):
        return "llama-3.1-70b"  # Open weights
    
    if requirements.get("tool_use_heavy"):
        return "gpt-4o"  # Best function calling
    
    if requirements.get("long_documents"):
        return "claude-3-5-sonnet"  # Great at analysis
    
    if requirements.get("high_volume") and requirements.get("simple_task"):
        return "gpt-4o-mini"  # Balance of speed/cost
    
    # Default: balanced flagship
    return "gpt-4o"
```

### Use Case Mapping

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Customer support chat | GPT-4o-mini | Fast, reliable |
| Legal document review | Claude 3.5 Sonnet | Long context, nuance |
| Code generation | GPT-4o | Best code quality |
| Content writing | Claude 3.5 Sonnet | Writing quality |
| Data extraction | GPT-4o | Tool use / JSON |
| Video analysis | Gemini 1.5 Pro | Native multimodal |
| Privacy-sensitive | LLaMA 3.1 | Local deployment |

---

## Hands-on Exercise

### Your Task

Compare models on the same prompt:

```python
from openai import OpenAI
from anthropic import Anthropic
import time

openai_client = OpenAI()
anthropic_client = Anthropic()

prompt = """
Analyze this code and suggest improvements:

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)
"""

def test_model(name: str, call_fn):
    start = time.time()
    result = call_fn()
    elapsed = time.time() - start
    return {
        "model": name,
        "time": elapsed,
        "length": len(result),
        "response": result[:500] + "..." if len(result) > 500 else result
    }

# Test GPT-4o
gpt_result = test_model("gpt-4o", lambda: openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
).choices[0].message.content)

# Test Claude
claude_result = test_model("claude-3-5-sonnet", lambda: anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
).content[0].text)

# Compare
print(f"GPT-4o: {gpt_result['time']:.2f}s, {gpt_result['length']} chars")
print(f"Claude: {claude_result['time']:.2f}s, {claude_result['length']} chars")
```

---

## Summary

✅ **GPT-4o**: Best for tool use, code, general tasks

✅ **Claude 3.5 Sonnet**: Best for writing, documents, nuance

✅ **Gemini 1.5**: Best for massive context, multimodal

✅ **LLaMA**: Best for local/private deployment

✅ **Choose based on**: context needs, task type, cost, privacy

**Next:** [Code Generation Models](./02-code-generation-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Lesson Overview](./00-types-of-ai-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Code Generation Models](./02-code-generation-models.md) |

