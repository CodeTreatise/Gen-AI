---
title: "Reasoning Token Visibility"
---

# Reasoning Token Visibility

## Introduction

Different providers handle reasoning visibility differently. Some show the full thinking process, others only provide summaries. Understanding these differences helps you choose the right approach.

### What We'll Cover

- OpenAI: Hidden reasoning with summaries
- Claude: Optional full visibility
- Gemini: Thinking process access
- Debugging with visible reasoning

---

## Visibility Comparison

### Provider Approaches

```
┌─────────────────────────────────────────────────────────────────┐
│                 REASONING VISIBILITY BY PROVIDER                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OpenAI (o1, o3)                                                │
│  ├── Thinking: HIDDEN                                           │
│  ├── Available: Summary in response                             │
│  └── Reason: Safety, prevent prompt injection via reasoning     │
│                                                                  │
│  Anthropic (Claude extended thinking)                           │
│  ├── Thinking: VISIBLE (optional)                               │
│  ├── Available: Full thinking block                             │
│  └── Reason: Transparency, debugging                            │
│                                                                  │
│  Google (Gemini thinking)                                       │
│  ├── Thinking: ACCESSIBLE                                       │
│  ├── Available: Thinking metadata                               │
│  └── Reason: Developer insights                                 │
│                                                                  │
│  Open Source (DeepSeek R1, QwQ)                                 │
│  ├── Thinking: FULLY VISIBLE                                    │
│  ├── Available: Complete chain-of-thought                       │
│  └── Reason: Open development                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## OpenAI: Hidden Reasoning

### What You Get

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}]
)

# You get the final answer
print(response.choices[0].message.content)
# "To prove that √2 is irrational, we use proof by contradiction..."

# But NOT the internal reasoning process
# OpenAI considers this a safety feature
```

### Usage Information

```python
# You can see token counts
print(response.usage)
# CompletionUsage(
#     prompt_tokens=15,
#     completion_tokens=500,
#     total_tokens=515,
#     completion_tokens_details=CompletionTokensDetails(
#         reasoning_tokens=8500  # Hidden thinking tokens
#     )
# )

# You pay for reasoning tokens but don't see them
```

---

## Claude: Full Visibility

### Accessing Thinking Blocks

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}]
)

# Parse response blocks
for block in response.content:
    if block.type == "thinking":
        print("=== THINKING ===")
        print(block.thinking)
        print()
    elif block.type == "text":
        print("=== ANSWER ===")
        print(block.text)
```

### Example Output

```
=== THINKING ===
Let me approach this with a proof by contradiction.

Assume √2 is rational. Then √2 = a/b where a and b are integers 
with no common factors (reduced form).

Squaring both sides: 2 = a²/b²
Therefore: 2b² = a²

This means a² is even. If a² is even, then a must be even 
(since odd² = odd).

Let a = 2k for some integer k.
Then: 2b² = (2k)² = 4k²
So: b² = 2k²

This means b² is even, so b is also even.

But wait - if both a and b are even, they share a common factor of 2.
This contradicts our assumption that a/b is in reduced form.

Therefore, √2 cannot be rational. It is irrational. ∎

=== ANSWER ===
To prove that √2 is irrational, we use proof by contradiction...
```

---

## Gemini: Thinking Access

### Accessing Thinking Metadata

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-pro")

response = model.generate_content(
    "Prove sqrt(2) is irrational",
    generation_config={"thinking": True}
)

# Access thinking information
if hasattr(response, 'thinking'):
    print("Thinking process:", response.thinking)

print("Answer:", response.text)
```

---

## Open Source: Full Transparency

### DeepSeek R1

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DEEPSEEK_KEY",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}]
)

# R1 shows full chain-of-thought in the response
print(response.choices[0].message.content)
# Includes <think>...</think> blocks with reasoning
```

### QwQ

```python
# QwQ (Qwen reasoning) via local or API
# Shows complete reasoning process

# Self-hosted example
import ollama

response = ollama.chat(
    model="qwq:32b",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}]
)

# Full reasoning visible
print(response['message']['content'])
```

---

## Debugging with Visible Reasoning

### Why Visibility Matters

```python
visibility_benefits = {
    "debugging": "See where model went wrong",
    "trust": "Verify the reasoning is sound",
    "learning": "Understand model's approach",
    "improvement": "Identify prompting issues",
    "safety": "Catch problematic reasoning"
}
```

### Debugging Example

```python
def debug_reasoning(problem: str) -> dict:
    """Use visible reasoning to debug issues"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": problem}]
    )
    
    thinking = ""
    answer = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text
    
    # Analyze reasoning
    issues = []
    
    if "I'm not sure" in thinking:
        issues.append("Model expressed uncertainty")
    
    if "Let me reconsider" in thinking:
        issues.append("Model backtracked - may indicate difficulty")
    
    if "This contradicts" in thinking and "wait" in thinking.lower():
        issues.append("Model found contradiction - good sign for proofs")
    
    return {
        "thinking": thinking,
        "answer": answer,
        "potential_issues": issues,
        "thinking_length": len(thinking.split())
    }

result = debug_reasoning("Is 0.999... equal to 1?")
print("Issues found:", result["potential_issues"])
```

---

## Choosing Based on Visibility Needs

### Decision Matrix

| Need | Recommended Provider |
|------|---------------------|
| Just need answer | OpenAI o3 |
| Need to verify reasoning | Claude |
| Need to debug failures | Claude, DeepSeek R1 |
| Regulatory audit requirements | Claude, Open source |
| Cost-sensitive + visibility | DeepSeek R1, QwQ |

---

## Summary

✅ **OpenAI**: Hidden reasoning (safety focus)

✅ **Claude**: Full visibility (transparency focus)

✅ **Gemini**: Accessible thinking metadata

✅ **Open source**: Complete transparency

✅ **Use visible reasoning for debugging and trust**

**Next:** [Streaming Thinking Blocks](./07-streaming-thinking.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Reasoning Budget](./05-reasoning-budget.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [Streaming Thinking](./07-streaming-thinking.md) |

