---
title: "Chain-of-Thought Prompting"
---

# Chain-of-Thought Prompting

## Overview

Chain-of-thought (CoT) prompting is a technique that improves LLM reasoning by encouraging the model to show its work. Instead of jumping straight to an answer, the model breaks down problems into steps, leading to more accurate results on complex tasks.

> **🔑 Critical Update:** With modern reasoning models (GPT-5, o3, o4-mini), chain-of-thought happens automatically inside the model. CoT prompting techniques are primarily for non-reasoning models like GPT-4.1.

---

## What You'll Learn

| Lesson | Topic | Description |
|--------|-------|-------------|
| [01](./01-what-is-chain-of-thought.md) | What is Chain-of-Thought? | Explicit reasoning, step-by-step solving |
| [02](./02-eliciting-reasoning-steps.md) | Eliciting Reasoning Steps | Trigger phrases and structured requests |
| [03](./03-step-by-step-instructions.md) | Step-by-Step Instructions | Breaking down complex tasks |
| [04](./04-zero-shot-chain-of-thought.md) | Zero-Shot CoT | "Let's think step by step" technique |
| [05](./05-when-not-to-use-cot.md) | When NOT to Use CoT | Reasoning models and anti-patterns |
| [06](./06-when-to-use-chain-of-thought.md) | When to Use CoT | Complex reasoning, math, accuracy |
| [07](./07-verification-self-correction.md) | Verification & Self-Correction | Self-checking and error detection |

---

## Understanding Model Types

```
┌─────────────────────────────────────────────────────────────┐
│                    REASONING MODELS                         │
│         GPT-5, GPT-5-mini, o3, o4-mini                     │
│                                                             │
│  • Internal chain-of-thought (reasoning tokens)             │
│  • Control via reasoning.effort parameter                   │
│  • DON'T add "think step by step"                          │
│  • Give high-level goals, not detailed instructions         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     GPT MODELS                              │
│              GPT-4.1, GPT-4o, GPT-4                         │
│                                                             │
│  • No internal reasoning                                    │
│  • Benefit from CoT prompting                               │
│  • "Let's think step by step" helps                         │
│  • Need explicit instructions                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### CoT Trigger Phrases (for GPT models)

```python
# Zero-shot CoT
"Let's think through this step by step."
"Let's work through this problem carefully."
"Before answering, reason through the problem."

# Few-shot CoT
"Here's how to solve similar problems: [examples with reasoning]"

# Self-verification
"After reaching your answer, verify each step."
```

### Using Reasoning Models (GPT-5, o3, o4-mini)

```python
from openai import OpenAI
client = OpenAI()

# Reasoning happens automatically - just set effort level
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "medium"},  # low, medium, or high
    input="What is 127 × 83?"
)

# DON'T do this with reasoning models:
# "Let's think step by step..."  # Reduces performance!
```

---

## When to Use Each Approach

| Scenario | Model Type | Approach |
|----------|------------|----------|
| Complex math problem | Reasoning | Just ask, set effort |
| Complex math problem | GPT | Add CoT prompting |
| Multi-step analysis | Reasoning | Set high effort |
| Multi-step analysis | GPT | Break into steps |
| Simple factual query | Either | No CoT needed |
| Speed-critical task | Either | Skip CoT |

---

## Navigation

| Previous | Up | Next |
|----------|-----|------|
| [JSON Mode](../06-json-mode-structured-outputs/00-json-mode-overview.md) | [Unit Overview](../00-overview.md) | [What is Chain-of-Thought?](./01-what-is-chain-of-thought.md) |

---

<!-- 
Sources Consulted:
- OpenAI Reasoning: https://platform.openai.com/docs/guides/reasoning
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
-->
