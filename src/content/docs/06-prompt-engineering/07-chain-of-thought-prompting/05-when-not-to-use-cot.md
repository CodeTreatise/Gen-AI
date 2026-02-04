---
title: "When NOT to Use Chain-of-Thought"
---

# When NOT to Use Chain-of-Thought

## Introduction

Chain-of-thought prompting revolutionized how we use language models—but the landscape has changed. Modern reasoning models like GPT-5, o3, and o4-mini have internal chain-of-thought built in. Using explicit CoT prompts with these models can actually *reduce* performance.

This lesson covers when to skip CoT entirely and when it actively hurts results.

> **⚠️ Warning:** Adding "think step by step" to reasoning models is an anti-pattern. These models reason internally and perform worse with explicit CoT instructions.

### What We'll Cover

- Reasoning models vs GPT models
- Why CoT hurts reasoning models
- Other scenarios to skip CoT
- Identifying which model type you're using

### Prerequisites

- [What is Chain-of-Thought?](./01-what-is-chain-of-thought.md)
- [Zero-Shot Chain-of-Thought](./04-zero-shot-chain-of-thought.md)

---

## Reasoning Models Don't Need CoT

### The Two Model Types

| Model Type | Examples | CoT Behavior |
|------------|----------|--------------|
| **Reasoning models** | GPT-5, GPT-5-mini, o3, o4-mini, o1 | Internal automatic CoT |
| **GPT models** | GPT-4.1, GPT-4o, GPT-4o-mini | Need explicit CoT prompts |

### How Reasoning Models Work

```
GPT Models (GPT-4.1, GPT-4o):
  Input → [Generate tokens] → Output
  (No internal reasoning unless prompted)

Reasoning Models (GPT-5, o3, o4-mini):
  Input → [Reasoning tokens - internal CoT] → Output
  (Automatic multi-step reasoning)
```

Reasoning models use **reasoning tokens**—internal chain-of-thought that happens automatically before generating the visible response. These tokens:

- Are processed internally (not visible via API by default)
- Count toward output token limits
- Provide the reasoning benefit you'd manually prompt for

### Why Explicit CoT Hurts Reasoning Models

When you add "think step by step" to a reasoning model:

| What You Expect | What Actually Happens |
|-----------------|----------------------|
| More reasoning | Redundant instruction confuses the model |
| Better accuracy | Performance often decreases |
| Clearer steps | Model may explain its internal reasoning poorly |

The model is already reasoning. Asking it to reason about its reasoning adds noise.

---

## The Mental Model

### Reasoning Models = Senior Developer

Think of reasoning models like a senior developer:

```python
# ❌ Wrong approach (micromanaging)
prompt = """You are given this task. First, think about what 
steps you need. Then consider edge cases. Then implement 
step by step. Show your work at each stage..."""

# ✅ Right approach (high-level goal)
prompt = "Implement a function that validates email addresses. 
Handle edge cases appropriately."
```

Give goals, not procedures. The model knows how to reason.

### GPT Models = Junior Developer

Think of GPT models like a talented but inexperienced developer:

```python
# ✅ Right approach (explicit guidance)
prompt = """Implement an email validator. 

Think through these steps:
1. What makes an email valid?
2. What edge cases exist?
3. Implement the validation logic
4. Test against edge cases

Show your reasoning at each step."""
```

They benefit from structure and explicit reasoning prompts.

---

## Quick Reference: When to Skip CoT

| Skip CoT When... | Reason |
|------------------|--------|
| Using reasoning models (o3, o4-mini, GPT-5) | Internal CoT already active |
| Simple factual questions | No reasoning needed |
| Speed is critical | CoT adds latency |
| Token budget is tight | CoT uses more tokens |
| Creative writing | Overthinking hurts creativity |
| Following fixed formats | Structure alone is enough |

---

## Simple Tasks Don't Need CoT

### Examples Where CoT Adds Nothing

```python
# ❌ Wasteful - no reasoning needed
prompt = "What is the capital of France? Let's think step by step."

# ✅ Direct - faster, same result
prompt = "What is the capital of France?"
```

```python
# ❌ Wasteful - simple extraction
prompt = "Extract the email from this text: 'Contact us at support@example.com'. 
Let's think through this carefully."

# ✅ Direct
prompt = "Extract the email from: 'Contact us at support@example.com'"
```

### Rule of Thumb

| Task Type | CoT Value |
|-----------|-----------|
| Multi-step math | High |
| Logic puzzles | High |
| Single-step lookup | None |
| Direct translation | None |
| Simple formatting | None |

---

## Speed-Critical Scenarios

### When Latency Matters

```python
# Real-time chat assistance
# ❌ Adds 1-3 seconds of latency
prompt = f"Quick question: {user_query}. Think step by step."

# ✅ Faster response
prompt = f"Quick question: {user_query}"
```

### Token Economics

| Prompt Type | Input Tokens | Output Tokens | Cost Factor |
|-------------|--------------|---------------|-------------|
| Direct | 50 | 30 | 1x |
| Zero-shot CoT | 55 | 150 | ~2x |
| Few-shot CoT | 200 | 180 | ~3x |

For high-volume applications, this multiplies quickly.

---

## Using `reasoning.effort` Instead

### For Reasoning Models

Instead of adding CoT prompts, control reasoning depth with the `reasoning.effort` parameter:

```python
from openai import OpenAI

client = OpenAI()

# Quick reasoning - faster, cheaper
response = client.chat.completions.create(
    model="o4-mini",
    reasoning={"effort": "low"},
    messages=[{"role": "user", "content": "What's 25 x 37?"}]
)

# Deep reasoning - thorough, more tokens
response = client.chat.completions.create(
    model="o4-mini", 
    reasoning={"effort": "high"},
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
)
```

### Effort Levels

| Level | Use When | Token Usage |
|-------|----------|-------------|
| `low` | Quick decisions, simple problems | Minimal |
| `medium` | Balanced tasks (default) | Moderate |
| `high` | Complex reasoning, high stakes | Maximum |

This is the proper way to control reasoning depth in reasoning models—not CoT prompts.

---

## Getting Reasoning Summaries

If you need to see the model's reasoning:

```python
response = client.chat.completions.create(
    model="o4-mini",
    reasoning={"summary": "auto"},  # Get reasoning summary
    messages=[{"role": "user", "content": "Solve this equation: 3x + 7 = 22"}]
)

# Access reasoning summary if provided
# (Available in response metadata)
```

This gives visibility into the reasoning process without degrading performance.

---

## Identifying Your Model Type

### Quick Check

```python
def needs_cot_prompting(model: str) -> bool:
    """Check if a model benefits from explicit CoT prompts."""
    
    # Reasoning models - DON'T use CoT prompts
    reasoning_models = {
        "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini",
        "o4-mini",
        "gpt-5", "gpt-5-mini"
    }
    
    # Check if model is a reasoning model
    for rm in reasoning_models:
        if rm in model.lower():
            return False  # Skip CoT
    
    return True  # Use CoT for GPT models
```

### Adaptive Prompting

```python
def create_prompt(question: str, model: str) -> str:
    """Create appropriate prompt based on model type."""
    
    if needs_cot_prompting(model):
        # GPT models benefit from CoT
        return f"{question}\n\nLet's think step by step."
    else:
        # Reasoning models - just give the goal
        return question
```

---

## Anti-Patterns to Avoid

### ❌ Double Reasoning

```python
# DON'T do this with reasoning models
prompt = """Think carefully about this problem.
Break it down step by step.
Consider all possibilities.
Show your work.
Then give me your final answer.

Question: What is 127 x 83?"""
```

This adds noise and can confuse the model's internal reasoning.

### ❌ Reasoning Instructions in System Prompts

```python
# DON'T do this with reasoning models
messages = [
    {"role": "system", "content": "Always think step by step and show your reasoning."},
    {"role": "user", "content": "Calculate the area of a triangle with base 10 and height 5."}
]
```

### ✅ Clean Prompts for Reasoning Models

```python
# DO this instead
messages = [
    {"role": "user", "content": "Calculate the area of a triangle with base 10 and height 5."}
]

# If you need specific output format:
messages = [
    {"role": "user", "content": "Calculate the area of a triangle with base 10 and height 5. Provide the answer in square units."}
]
```

---

## Decision Framework

```
Is this a reasoning model (o3, o4-mini, GPT-5)?
├── YES → Skip CoT prompts
│         Use reasoning.effort for depth control
│         Give high-level goals, not procedures
│
└── NO → Is the task complex (multi-step reasoning)?
         ├── YES → Use CoT (zero-shot or few-shot)
         └── NO → Skip CoT (direct prompt is fine)
```

---

## Summary

- Reasoning models (GPT-5, o3, o4-mini) have built-in CoT
- Adding "think step by step" to reasoning models hurts performance
- Use `reasoning.effort` to control depth instead
- Skip CoT for simple factual questions regardless of model
- Match your prompting strategy to your model type

**Next:** [When to Use Chain-of-Thought](./06-when-to-use-chain-of-thought.md)

---

<!-- Sources: OpenAI Reasoning Guide, OpenAI Prompt Engineering Best Practices -->
