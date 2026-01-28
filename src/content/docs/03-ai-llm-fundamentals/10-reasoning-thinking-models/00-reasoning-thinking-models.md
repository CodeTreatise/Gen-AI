---
title: "Reasoning & Thinking Models"
---

# Reasoning & Thinking Models

## Overview

Reasoning models represent a new paradigm in AI - models that "think" before responding, using extended computation to solve complex problems more accurately.

### What You'll Learn

| Topic | Description |
|-------|-------------|
| [What Are Reasoning Models?](./01-what-are-reasoning-models.md) | o1, o3, Claude thinking explained |
| [Extended Thinking vs. Fast Responses](./02-extended-vs-fast.md) | When to use each mode |
| [When to Use Reasoning Models](./03-when-to-use.md) | Best use cases |
| [Cost Implications](./04-cost-implications.md) | Thinking token costs |
| [Reasoning Budget & Test-Time Compute](./05-reasoning-budget.md) | Configuring compute allocation |
| [Reasoning Token Visibility](./06-token-visibility.md) | Hidden vs visible reasoning |
| [Streaming Thinking Blocks](./07-streaming-thinking.md) | UX for thinking display |
| [Verifiable Reasoning Outputs](./08-verifiable-reasoning.md) | Debugging and auditing |

### Prerequisites

Before this lesson, understand:
- Types of AI models (Lesson 07)
- Model selection criteria (Lesson 08)
- AI providers landscape (Lesson 09)

---

## The Reasoning Revolution

### Traditional vs. Reasoning Models

```
┌─────────────────────────────────────────────────────────────────┐
│                 TRADITIONAL vs REASONING MODELS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRADITIONAL (GPT-4o, Claude 3.5):                              │
│  ┌────────┐     ┌────────────────┐                             │
│  │ Prompt │ ──→ │ Direct Answer  │     (~1 second)             │
│  └────────┘     └────────────────┘                             │
│                                                                  │
│  REASONING (o1, o3, Claude thinking):                           │
│  ┌────────┐     ┌─────────────────────────┐     ┌────────────┐ │
│  │ Prompt │ ──→ │ Extended Thinking       │ ──→ │ Answer     │ │
│  └────────┘     │ (chain of thought)      │     └────────────┘ │
│                 │ 5-60+ seconds           │                     │
│                 └─────────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

```python
reasoning_models_key_insight = {
    "traditional": "Fast response, single pass",
    "reasoning": "Extended thinking, multiple iterations",
    "when_reasoning_wins": [
        "Complex math problems",
        "Multi-step planning", 
        "Scientific reasoning",
        "Debugging complex code",
        "PhD-level questions"
    ],
    "when_traditional_wins": [
        "Simple questions",
        "Creative writing",
        "Real-time chat",
        "High-volume processing"
    ]
}
```

---

## Current Reasoning Models

| Provider | Model | Notes |
|----------|-------|-------|
| OpenAI | o1 | Mathematical, scientific reasoning |
| OpenAI | o3 | Advanced reasoning, configurable |
| OpenAI | o4-mini | Fast reasoning |
| Anthropic | Claude (extended thinking) | Visible thinking blocks |
| Google | Gemini (thinking mode) | Thinking process access |
| DeepSeek | R1 | Open-weight reasoning |
| Alibaba | QwQ | Open-weight reasoning |

---

## Quick Example

```python
from openai import OpenAI

client = OpenAI()

# Traditional model - fast, single pass
def quick_answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Reasoning model - thinks before answering
def reasoned_answer(question: str) -> str:
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Use reasoning for complex problems
question = """
A bat and ball cost $1.10 together. The bat costs $1.00 more 
than the ball. How much does the ball cost?
"""

# GPT-4o might say $0.10 (wrong, intuitive answer)
# o3 will reason through and get $0.05 (correct)
```

---

## Navigation

| Previous Lesson | Unit Home | First Topic |
|-----------------|-----------|-------------|
| [AI Providers Landscape](../09-ai-providers-landscape/00-ai-providers-landscape.md) | [AI/LLM Fundamentals](../00-overview.md) | [What Are Reasoning Models?](./01-what-are-reasoning-models.md) |

