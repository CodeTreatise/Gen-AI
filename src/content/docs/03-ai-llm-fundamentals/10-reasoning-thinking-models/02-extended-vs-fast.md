---
title: "Extended Thinking vs. Fast Responses"
---

# Extended Thinking vs. Fast Responses

## Introduction

Choosing between reasoning models and traditional models is a key architectural decision. This lesson helps you understand when extended thinking is worth the trade-offs.

### What We'll Cover

- When to use each mode
- Latency implications
- Quality improvements from thinking
- User experience considerations

---

## The Trade-Off

```
┌─────────────────────────────────────────────────────────────────┐
│              FAST vs REASONING TRADE-OFFS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FAST MODELS (GPT-4o, Claude Sonnet)                            │
│  ├── Latency: 1-5 seconds                                       │
│  ├── Cost: $                                                    │
│  ├── Quality: Good for most tasks                               │
│  └── UX: Feels responsive                                       │
│                                                                  │
│  REASONING MODELS (o3, Claude thinking)                         │
│  ├── Latency: 5-120 seconds                                     │
│  ├── Cost: $$$                                                  │
│  ├── Quality: Excellent for complex tasks                       │
│  └── UX: Requires loading state                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each

### Use Fast Models For

```python
fast_model_use_cases = [
    # Real-time interaction
    "Chat conversations",
    "Customer support",
    "Quick Q&A",
    
    # Creative tasks
    "Writing assistance",
    "Brainstorming",
    "Content generation",
    
    # High-volume processing
    "Document classification",
    "Sentiment analysis",
    "Data extraction",
    
    # Time-sensitive
    "Autocomplete",
    "Real-time suggestions",
    "Voice assistants"
]
```

### Use Reasoning Models For

```python
reasoning_model_use_cases = [
    # Complex analysis
    "Mathematical proofs",
    "Scientific reasoning",
    "Legal analysis",
    
    # Code tasks
    "Complex bug debugging",
    "Architecture design",
    "Code review with explanations",
    
    # Planning
    "Multi-step problem solving",
    "Strategic planning",
    "Project breakdown",
    
    # High-stakes
    "Medical diagnosis support",
    "Financial analysis",
    "Critical decision support"
]
```

---

## Latency Comparison

### Real-World Timings

```python
import time
from openai import OpenAI

client = OpenAI()

def compare_latency(prompt: str):
    """Compare response times"""
    
    # Fast model
    start = time.time()
    fast_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    fast_time = time.time() - start
    
    # Reasoning model
    start = time.time()
    reasoning_response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}]
    )
    reasoning_time = time.time() - start
    
    return {
        "fast_model": {
            "time": f"{fast_time:.1f}s",
            "answer": fast_response.choices[0].message.content[:100]
        },
        "reasoning_model": {
            "time": f"{reasoning_time:.1f}s",
            "answer": reasoning_response.choices[0].message.content[:100]
        }
    }

# Typical results:
# Simple question: Fast 1.2s vs Reasoning 8s
# Complex math: Fast 1.5s vs Reasoning 45s
```

### Latency Expectations

| Task Complexity | Fast Model | Reasoning Model |
|-----------------|------------|-----------------|
| Simple question | 1-2s | 5-10s |
| Moderate problem | 2-3s | 15-30s |
| Complex reasoning | 2-4s | 30-60s |
| Very hard problem | 3-5s | 60-120s+ |

---

## Quality Improvements

### Where Reasoning Shines

```python
# Classic example: The bat and ball problem
prompt = """
A bat and ball cost $1.10 together. 
The bat costs $1.00 more than the ball. 
How much does the ball cost?
"""

# Fast model often gives intuitive (wrong) answer: $0.10
# Reasoning model works through and gets correct: $0.05

# Why? The reasoning model:
# 1. Sets up equations
# 2. Checks the intuitive answer
# 3. Realizes $0.10 + $1.10 = $1.20 ≠ $1.10
# 4. Solves correctly: x + (x + 1) = 1.10, so x = 0.05
```

### Benchmark Differences

```python
benchmark_comparison = {
    "MATH (competition math)": {
        "gpt-4o": "~60%",
        "o3": "~90%"
    },
    "GPQA (PhD-level science)": {
        "gpt-4o": "~50%",
        "o3": "~85%"
    },
    "Code (SWE-bench)": {
        "gpt-4o": "~30%",
        "o3": "~70%"
    }
}
```

---

## User Experience Considerations

### Progressive Loading

```python
async def reasoning_with_ux(problem: str):
    """Show progress while reasoning"""
    
    # Start with immediate feedback
    yield {"status": "thinking", "message": "Analyzing problem..."}
    
    # For Claude, stream thinking blocks
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{"role": "user", "content": problem}]
    ) as stream:
        async for event in stream:
            if event.type == "thinking_delta":
                yield {"status": "thinking", "content": event.thinking}
            elif event.type == "text_delta":
                yield {"status": "answering", "content": event.text}
```

### UX Patterns

```python
ux_patterns = {
    "show_progress": "Indicate thinking is happening",
    "estimate_time": "'This may take 30-60 seconds'",
    "interruptible": "Allow user to cancel",
    "show_thinking": "Display reasoning (if visible)",
    "fallback": "Offer faster model as alternative"
}
```

---

## Hybrid Approach

### Router Pattern

```python
def select_model(task: dict) -> str:
    """Route to appropriate model"""
    
    if task.get("complexity") == "high":
        return "o3"
    
    if task.get("type") in ["math", "science", "code_debug"]:
        return "o3"
    
    if task.get("time_sensitive"):
        return "gpt-4o"
    
    if task.get("interactive"):
        return "gpt-4o"
    
    return "gpt-4o"  # Default to fast

# Use reasoning for initial analysis, fast for follow-ups
def smart_conversation(messages: list) -> str:
    if len(messages) == 1 and is_complex(messages[0]):
        model = "o3"
    else:
        model = "gpt-4o"
    
    return call_model(model, messages)
```

---

## Summary

✅ **Fast models**: Real-time, high-volume, interactive

✅ **Reasoning models**: Complex, high-stakes, accuracy-critical

✅ **Latency**: 1-5s vs 5-120s

✅ **Quality**: 60% vs 90% on hard benchmarks

✅ **UX matters**: Show progress for long operations

**Next:** [When to Use Reasoning Models](./03-when-to-use.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [What Are Reasoning Models?](./01-what-are-reasoning-models.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [When to Use](./03-when-to-use.md) |

