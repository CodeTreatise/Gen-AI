---
title: "What Are Reasoning Models?"
---

# What Are Reasoning Models?

## Introduction

Reasoning models are a new class of AI models designed to "think" before answering. Unlike traditional models that respond immediately, reasoning models use extended internal computation to work through complex problems step by step.

### What We'll Cover

- How reasoning models differ
- OpenAI o1 and o3 series
- Claude extended thinking
- Gemini thinking capabilities
- Open-source options

---

## How Reasoning Models Work

### Internal Chain-of-Thought

```
┌─────────────────────────────────────────────────────────────────┐
│                 REASONING MODEL PROCESS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Receive Problem                                         │
│  "Prove that √2 is irrational"                                  │
│                                                                  │
│  Step 2: Internal Reasoning (hidden or visible)                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Let me approach this with proof by contradiction...         ││
│  │ Assume √2 = a/b where a,b are coprime integers...          ││
│  │ Then 2 = a²/b², so a² = 2b²...                             ││
│  │ This means a² is even, so a is even...                     ││
│  │ Let a = 2k, then 4k² = 2b², so 2k² = b²...                ││
│  │ This means b² is even, so b is even...                     ││
│  │ But both a and b even contradicts coprime assumption...    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Step 3: Return Answer                                           │
│  "√2 is irrational because..." (polished explanation)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Test-Time Compute

```python
# Traditional: Fixed computation per token
# Reasoning: Variable computation based on problem complexity

reasoning_model_concept = {
    "key_innovation": "Test-time compute scaling",
    "meaning": "Model can 'think longer' on harder problems",
    "result": "Better accuracy on complex tasks",
    "trade_off": "Higher latency and cost"
}
```

---

## OpenAI o-Series

### Model Lineup

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| o1 | Math, science, complex reasoning | Slow | $$$ |
| o3 | General reasoning, coding | Medium | $$ |
| o4-mini | Fast reasoning tasks | Fast | $ |

### Using o-Series

```python
from openai import OpenAI

client = OpenAI()

def use_o3(problem: str) -> str:
    """Use o3 for complex reasoning"""
    
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": problem}]
    )
    return response.choices[0].message.content

# o-series excels at:
complex_problems = [
    "Prove the Pythagorean theorem",
    "Debug this algorithm and explain the fix",
    "Design a system architecture for X requirements",
    "Solve this competition math problem"
]
```

### o3 Reasoning Effort

```python
# o3 allows configuring reasoning effort
response = client.chat.completions.create(
    model="o3",
    reasoning_effort="high",  # low, medium, high
    messages=[{"role": "user", "content": "Complex problem..."}]
)

reasoning_effort_levels = {
    "low": "Quick reasoning, lower cost",
    "medium": "Balanced (default)",
    "high": "Maximum reasoning, best quality, highest cost"
}
```

---

## Claude Extended Thinking

### How It Works

```python
from anthropic import Anthropic

client = Anthropic()

def claude_with_thinking(problem: str) -> dict:
    """Claude with extended thinking enabled"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000  # Thinking token budget
        },
        messages=[{"role": "user", "content": problem}]
    )
    
    thinking = None
    answer = None
    
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text
    
    return {
        "thinking": thinking,  # Visible chain-of-thought
        "answer": answer
    }

result = claude_with_thinking("Solve x² + 5x + 6 = 0")
print("Thinking:", result["thinking"])
print("Answer:", result["answer"])
```

### Thinking Visibility

```python
# Claude's key advantage: VISIBLE thinking
# You can see the model's reasoning process

claude_thinking_benefits = {
    "debugging": "See where reasoning goes wrong",
    "trust": "Verify the model's logic",
    "learning": "Understand how it solves problems",
    "control": "Set thinking budget"
}
```

---

## Gemini Thinking

### Thinking Mode

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-pro")

def gemini_with_thinking(problem: str) -> str:
    """Use Gemini's thinking capabilities"""
    
    response = model.generate_content(
        problem,
        generation_config={
            "thinking": True,  # Enable thinking mode
        }
    )
    
    return response.text
```

---

## Open-Source Reasoning

### DeepSeek R1

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DEEPSEEK_KEY",
    base_url="https://api.deepseek.com"
)

def deepseek_reasoning(problem: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-reasoner",  # R1
        messages=[{"role": "user", "content": problem}]
    )
    return response.choices[0].message.content
```

### QwQ (Qwen Reasoning)

```python
# QwQ is Alibaba's open-weight reasoning model
# Available on HuggingFace and via providers

qwq_features = {
    "type": "Open-weight reasoning model",
    "parameters": "32B",
    "context": "128K",
    "availability": "HuggingFace, Together AI, local"
}
```

---

## Comparison

| Model | Provider | Thinking Visible | Cost | Open Source |
|-------|----------|------------------|------|-------------|
| o1 | OpenAI | Summary only | $$$ | ❌ |
| o3 | OpenAI | Summary only | $$ | ❌ |
| Claude thinking | Anthropic | Full visibility | $$ | ❌ |
| Gemini thinking | Google | Accessible | $$ | ❌ |
| DeepSeek R1 | DeepSeek | Visible | $ | ✅ |
| QwQ | Alibaba | Visible | Free | ✅ |

---

## Summary

✅ **Reasoning models think before answering** - extended internal computation

✅ **o-series from OpenAI** - hidden reasoning, summary output

✅ **Claude extended thinking** - visible thinking blocks

✅ **Open-source options** - DeepSeek R1, QwQ

✅ **Trade-off** - better accuracy for higher latency and cost

**Next:** [Extended Thinking vs. Fast Responses](./02-extended-vs-fast.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Overview](./00-reasoning-thinking-models.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [Extended vs Fast](./02-extended-vs-fast.md) |

