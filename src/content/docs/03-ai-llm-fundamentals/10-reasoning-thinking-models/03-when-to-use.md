---
title: "When to Use Reasoning Models"
---

# When to Use Reasoning Models

## Introduction

Knowing when reasoning models provide value versus when they're overkill is essential. This lesson covers the specific scenarios where extended thinking pays off.

### What We'll Cover

- Complex multi-step problems
- Mathematical and scientific reasoning
- Code debugging and analysis
- Strategic planning
- When NOT to use reasoning models

---

## Complex Multi-Step Problems

### Breaking Down Complexity

```python
# Reasoning models excel when problems require multiple steps

multi_step_examples = [
    # Business analysis
    """Analyze our SaaS metrics:
    - MRR: $50K, growing 10% monthly
    - Churn: 5% monthly
    - CAC: $200, LTV: $1,500
    Calculate runway, break-even, and recommend actions.""",
    
    # System design
    """Design a real-time notification system:
    - 10M users, 1M daily active
    - Max 100ms delivery latency
    - Support push, email, SMS
    Consider scaling, reliability, and cost.""",
    
    # Legal analysis
    """Review this contract clause for risks:
    [clause text]
    Identify issues, suggest alternatives, 
    and explain implications for each party."""
]
```

### Why Reasoning Helps

```python
multi_step_value = {
    "problem_decomposition": "Breaks into manageable parts",
    "dependency_tracking": "Remembers what depends on what",
    "verification": "Checks each step before proceeding",
    "coherence": "Maintains logical consistency throughout"
}
```

---

## Mathematical Reasoning

### Where Reasoning Models Excel

```python
math_use_cases = [
    # Proofs
    "Prove that the sum of angles in a triangle is 180°",
    
    # Competition math
    "Find all positive integers n where n! + 1 is a perfect square",
    
    # Applied math
    "Derive the Black-Scholes equation from first principles",
    
    # Word problems
    """A train leaves station A at 60 mph. Another train leaves 
    station B (100 miles away) at 40 mph toward A. When do they meet?"""
]
```

### Comparison Example

```python
from openai import OpenAI

client = OpenAI()

problem = """
Prove that for any positive integers a and b, 
if a² + b² is divisible by 3, then both a and b are divisible by 3.
"""

# GPT-4o might attempt but miss cases
# o3 will:
# 1. Consider a mod 3 and b mod 3
# 2. Enumerate all combinations
# 3. Show only (0,0) works
# 4. Conclude proof
```

---

## Code Debugging and Analysis

### Complex Bug Finding

```python
code_debugging_use_cases = [
    # Race conditions
    "Find the race condition in this concurrent code",
    
    # Logic errors
    "This algorithm sometimes returns wrong results. Why?",
    
    # Performance issues
    "Why does this function become slow with large inputs?",
    
    # Security vulnerabilities
    "Identify security issues in this authentication flow"
]

# Example prompt
debugging_prompt = """
This code sometimes fails. Find the bug:

def merge_sorted(lists):
    if not lists:
        return []
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged.append(merge_two(lists[i], lists[i+1]))
            else:
                merged.append(lists[i])
        lists = merged
    return lists[0]
"""

# Reasoning model will:
# 1. Trace through execution
# 2. Identify edge cases
# 3. Find the bug
# 4. Explain why it occurs
```

---

## Strategic Planning

### Long-Range Reasoning

```python
planning_use_cases = [
    # Project planning
    """Create a 6-month roadmap for launching an AI product:
    - Current: MVP complete
    - Goal: 10K paying users
    - Team: 5 engineers, 1 designer
    - Budget: $500K""",
    
    # Architecture decisions
    """We need to migrate from monolith to microservices.
    Current: 50K LOC Python app, 10K requests/sec
    Constraints: Zero downtime, 3 months timeline
    Propose a migration strategy.""",
    
    # Optimization
    """Our CI/CD pipeline takes 45 minutes.
    Analyze and propose optimizations to get under 15 minutes.
    Current steps: [list of pipeline steps]"""
]
```

---

## Scientific Reasoning

### Research-Level Problems

```python
scientific_use_cases = [
    # Literature analysis
    "Compare methodologies in these three papers: [papers]",
    
    # Experimental design
    "Design an experiment to test hypothesis X with constraints Y",
    
    # Data interpretation
    "Explain these unexpected results: [data]",
    
    # Mechanism reasoning
    "Propose mechanisms that could explain observation X"
]
```

---

## When NOT to Use Reasoning Models

### Overkill Scenarios

```python
dont_use_reasoning_for = [
    # Simple questions
    "What's the capital of France?",
    
    # Creative writing
    "Write a poem about autumn",
    
    # Conversational
    "How are you today?",
    
    # Classification
    "Is this email spam?",
    
    # Simple extraction
    "Extract the date from this text",
    
    # High-volume processing
    "Classify these 10,000 documents",
    
    # Real-time applications
    "Autocomplete this sentence"
]
```

### Decision Framework

```python
def should_use_reasoning(task: dict) -> bool:
    """Decide if reasoning model is appropriate"""
    
    # Definitely use reasoning
    if task.get("requires_proof"):
        return True
    if task.get("multi_step") and task.get("steps") > 5:
        return True
    if task.get("domain") in ["math", "science", "law"]:
        return True
    if task.get("stakes") == "high":
        return True
    
    # Definitely don't use reasoning
    if task.get("time_sensitive"):
        return False
    if task.get("volume") > 100:
        return False
    if task.get("complexity") == "low":
        return False
    
    # Default to fast model
    return False
```

---

## Summary

✅ **Use for**: Multi-step, math, debugging, planning, science

✅ **Don't use for**: Simple, creative, high-volume, real-time

✅ **Key signal**: "Does this require careful step-by-step reasoning?"

✅ **Decision**: Worth extra time/cost if accuracy is critical

**Next:** [Cost Implications of Thinking Tokens](./04-cost-implications.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Extended vs Fast](./02-extended-vs-fast.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [Cost Implications](./04-cost-implications.md) |

