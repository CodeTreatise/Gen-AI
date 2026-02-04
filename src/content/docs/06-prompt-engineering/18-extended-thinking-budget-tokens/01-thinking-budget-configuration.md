---
title: "Thinking Budget Configuration"
---

# Thinking Budget Configuration

## Introduction

The thinking budget controls how many tokens the model can use for internal reasoning. A larger budget allows deeper exploration of complex problems, while a smaller budget optimizes for speed and cost. Getting this balance right is key to effective use of extended thinking.

This lesson covers how to configure thinking budgets across different providers and optimize for your specific use cases.

### What We'll Cover

- Budget tokens configuration (Anthropic)
- Thinking levels (Gemini 3)
- Thinking budgets (Gemini 2.5)
- Dynamic vs fixed budgets
- Budget optimization strategies
- Pricing implications

### Prerequisites

- [Extended Thinking Overview](./00-extended-thinking-overview.md)

---

## Anthropic: Budget Tokens

### Basic Configuration

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Maximum thinking tokens
    },
    messages=[
        {"role": "user", "content": "Solve this complex optimization problem..."}
    ]
)
```

### Budget Token Rules

| Rule | Details |
|------|---------|
| **Minimum** | 1,024 tokens |
| **Maximum** | Limited by `max_tokens` |
| **Relationship** | `budget_tokens` must be less than `max_tokens` |
| **Exception** | With interleaved thinking, budget can exceed `max_tokens` |

### Budget Guidelines by Task

```python
THINKING_BUDGETS = {
    # Simple tasks - minimum budget
    "basic_analysis": 1024,
    "simple_math": 2000,
    
    # Medium complexity
    "code_review": 5000,
    "data_analysis": 8000,
    
    # Complex tasks
    "algorithm_design": 16000,
    "complex_debugging": 20000,
    
    # Very complex
    "research_problems": 32000,
    "novel_solutions": 50000,
}

def get_thinking_budget(task_type: str) -> int:
    """Get recommended thinking budget for task type."""
    return THINKING_BUDGETS.get(task_type, 10000)
```

### How Claude Uses the Budget

```
Budget: 10,000 tokens

Scenario 1: Simple problem
├── Claude thinks for 2,000 tokens
├── Reaches solution early
└── Returns response (8,000 tokens unused)

Scenario 2: Complex problem
├── Claude uses full 10,000 tokens
├── Still working through problem
└── Returns best answer so far

Scenario 3: Very complex problem
├── Claude uses 10,000 tokens
├── Notes it could use more thinking
└── Consider increasing budget
```

> **Note:** The model may not use the entire budget. It stops thinking when it has a satisfactory answer.

---

## Gemini 3: Thinking Levels

Gemini 3 models use qualitative thinking levels instead of token counts:

### Available Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `minimal` | Minimal thinking (may still think for complex coding) | Chat, high-throughput |
| `low` | Reduced latency and cost | Simple instructions |
| `medium` | Balanced thinking | Most general tasks |
| `high` | Maximum reasoning depth (default) | Complex problems |

### Configuration

```python
from google import genai
from google.genai import types

client = genai.Client()

# High thinking for complex problems
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Solve this complex mathematical proof...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="high"
        )
    )
)

# Low thinking for simple tasks
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="List 3 famous physicists",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        )
    )
)
```

### Level Comparison

```
Task: "What is 2 + 2?"

minimal: ────► Response (fastest)

low:    ──────► Response

medium: ─────────► Response

high:   ──────────────────► Response (most thorough)
```

### Important Notes

- Gemini 3 Pro **cannot** disable thinking
- Gemini 3 Flash `minimal` means thinking is *unlikely* but not guaranteed off
- Default is `high` with dynamic adjustment based on complexity

---

## Gemini 2.5: Thinking Budgets

Gemini 2.5 models use explicit token budgets (similar to Anthropic):

### Budget Ranges by Model

| Model | Default | Range | Can Disable |
|-------|---------|-------|-------------|
| Gemini 2.5 Pro | Dynamic (-1) | 128 - 32,768 | No |
| Gemini 2.5 Flash | Dynamic (-1) | 0 - 24,576 | Yes (budget=0) |
| Gemini 2.5 Flash Lite | Off by default | 512 - 24,576 | Yes (budget=0) |

### Configuration

```python
from google import genai
from google.genai import types

client = genai.Client()

# Explicit budget
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Complex problem...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=10000  # Fixed budget
        )
    )
)

# Dynamic thinking (model decides)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Problem...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1  # Dynamic
        )
    )
)

# Disable thinking
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Simple task...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0  # Off
        )
    )
)
```

---

## Dynamic vs Fixed Budgets

### Dynamic Budgets

Both Anthropic and Gemini support dynamic thinking where the model adjusts effort based on problem complexity:

**Gemini:**
```python
thinking_config=types.ThinkingConfig(thinking_budget=-1)
```

**Claude:** Uses the budget as a *maximum*, thinking less for simpler problems.

### When to Use Fixed Budgets

| Scenario | Recommendation |
|----------|----------------|
| Cost-sensitive applications | Use fixed, conservative budgets |
| Predictable latency requirements | Use fixed budgets |
| Variable task complexity | Use dynamic budgets |
| Research/exploration | Use high fixed budgets |
| Production chat | Use low fixed or dynamic |

### Adaptive Budget Strategy

```python
def adaptive_thinking_budget(
    task_description: str,
    max_budget: int = 32000,
    min_budget: int = 1024
) -> int:
    """Estimate appropriate thinking budget based on task signals."""
    
    budget = min_budget
    
    # Complexity indicators
    complexity_signals = {
        "prove": 8000,
        "derive": 6000,
        "optimize": 6000,
        "debug": 5000,
        "analyze": 4000,
        "explain": 2000,
        "compare": 3000,
        "step by step": 4000,
        "multiple": 3000,
        "complex": 5000,
    }
    
    task_lower = task_description.lower()
    
    for signal, additional_budget in complexity_signals.items():
        if signal in task_lower:
            budget += additional_budget
    
    # Cap at max
    return min(budget, max_budget)

# Usage
budget = adaptive_thinking_budget(
    "Prove that the square root of 2 is irrational using a step by step approach"
)
print(f"Recommended budget: {budget}")  # ~13,024 tokens
```

---

## Max Tokens and Context Window

### The Relationship

With extended thinking, `max_tokens` has special behavior:

```
Context Window = Input Tokens + max_tokens
                                  │
                                  ├─ Thinking tokens (budget)
                                  └─ Output tokens (response)
```

### Validation Rules

```python
# Claude 4 models enforce strict limits
# This will ERROR if: input_tokens + max_tokens > context_window

# Example for 200K context window:
# Input: 50K tokens
# max_tokens: 160K → ERROR (50K + 160K > 200K)
# max_tokens: 150K → OK (50K + 150K = 200K)
```

### Setting Appropriate Limits

```python
def calculate_safe_max_tokens(
    input_tokens: int,
    context_window: int,
    thinking_budget: int,
    expected_output: int
) -> int:
    """Calculate safe max_tokens value."""
    
    # Leave buffer for safety
    buffer = 500
    
    available = context_window - input_tokens - buffer
    needed = thinking_budget + expected_output
    
    if needed > available:
        raise ValueError(
            f"Request too large. Need {needed} tokens but only {available} available."
        )
    
    return min(needed, available)

# Usage
max_tokens = calculate_safe_max_tokens(
    input_tokens=5000,
    context_window=200000,
    thinking_budget=10000,
    expected_output=4000
)
```

---

## Pricing Implications

### How Thinking Tokens Are Billed

**Anthropic:**
- Thinking tokens count as **output tokens**
- Billed at output token rates
- Previous turn thinking is stripped (not billed as input)

**Gemini:**
- Full thinking tokens billed (not summaries)
- `thoughtsTokenCount` in usage metadata shows actual count

```python
# Get thinking token count (Gemini)
print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
```

### Cost Comparison Example

```python
# Task: Complex mathematical proof

# Without extended thinking (standard model)
# Input: 1000 tokens, Output: 2000 tokens
# Cost: (1000 * $0.003/1K) + (2000 * $0.015/1K) = $0.033

# With extended thinking
# Input: 1000 tokens, Thinking: 8000 tokens, Output: 2000 tokens  
# Cost: (1000 * $0.003/1K) + (8000 + 2000) * $0.015/1K = $0.153

# 4.6x cost increase, but potentially much higher quality
```

### Streaming Requirements

> **Warning:** For Claude, streaming is **required** when `max_tokens` exceeds 21,333 tokens.

```python
# Required for large max_tokens
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=30000,
    thinking={"type": "enabled", "budget_tokens": 20000},
    messages=[...]
) as stream:
    for event in stream:
        # Handle events
        pass
```

---

## Budget Optimization Strategies

### Start Small, Scale Up

```python
def iterative_budget_approach(task: str, client) -> dict:
    """Try increasing budgets until quality is satisfactory."""
    
    budgets = [1024, 4000, 10000, 20000, 32000]
    
    for budget in budgets:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=budget + 5000,
            thinking={"type": "enabled", "budget_tokens": budget},
            messages=[{"role": "user", "content": task}]
        )
        
        # Check if model used most of the budget
        thinking_block = next(
            (b for b in response.content if b.type == "thinking"), None
        )
        
        if thinking_block:
            thinking_length = len(thinking_block.thinking)
            # If model used less than 80% of budget, we're probably good
            if thinking_length < budget * 0.8:
                return {
                    "response": response,
                    "optimal_budget": budget,
                    "thinking_used": thinking_length
                }
    
    # Used maximum budget
    return {
        "response": response,
        "optimal_budget": budgets[-1],
        "note": "Consider even higher budget for this task type"
    }
```

### Task-Based Budget Selection

```python
class ThinkingBudgetSelector:
    """Select thinking budget based on task characteristics."""
    
    TASK_PATTERNS = {
        # Pattern: (keywords, base_budget, multiplier_per_constraint)
        "math": (["calculate", "prove", "derive", "equation"], 8000, 2000),
        "code": (["code", "implement", "debug", "algorithm"], 10000, 2500),
        "analysis": (["analyze", "evaluate", "compare", "assess"], 6000, 1500),
        "creative": (["write", "create", "design", "brainstorm"], 4000, 1000),
    }
    
    def select_budget(self, task: str, num_constraints: int = 0) -> int:
        task_lower = task.lower()
        
        for category, (keywords, base, multiplier) in self.TASK_PATTERNS.items():
            if any(kw in task_lower for kw in keywords):
                return base + (num_constraints * multiplier)
        
        # Default for unknown tasks
        return 5000 + (num_constraints * 1000)
```

---

## Common Mistakes

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Always using maximum budget | Start with minimum, increase as needed |
| Ignoring latency impact | Use streaming, set appropriate timeouts |
| Not checking thinking output | Review thinking to debug and optimize |
| Fixed budget for all tasks | Match budget to task complexity |
| Forgetting `max_tokens` relationship | Ensure `budget_tokens` < `max_tokens` |

---

## Hands-on Exercise

### Your Task

Write a function that:
1. Classifies a task by complexity (low, medium, high)
2. Returns appropriate thinking configuration for Claude and Gemini
3. Includes cost estimation

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ThinkingConfig:
    claude_budget: int
    gemini_level: Literal["minimal", "low", "medium", "high"]
    gemini_budget: int
    estimated_cost_per_request: float
    latency_estimate: str

def classify_and_configure(task: str) -> ThinkingConfig:
    """Classify task and return thinking configuration."""
    
    task_lower = task.lower()
    
    # Complexity signals
    high_complexity_signals = [
        "prove", "derive", "optimize", "algorithm", "mathematical",
        "complex", "multi-step", "research"
    ]
    medium_complexity_signals = [
        "analyze", "compare", "explain", "debug", "review",
        "evaluate", "summarize"
    ]
    
    # Count signals
    high_count = sum(1 for s in high_complexity_signals if s in task_lower)
    medium_count = sum(1 for s in medium_complexity_signals if s in task_lower)
    
    # Classify
    if high_count >= 2 or "prove" in task_lower or "algorithm" in task_lower:
        # High complexity
        return ThinkingConfig(
            claude_budget=20000,
            gemini_level="high",
            gemini_budget=16000,
            estimated_cost_per_request=0.35,
            latency_estimate="10-30 seconds"
        )
    elif high_count >= 1 or medium_count >= 2:
        # Medium complexity
        return ThinkingConfig(
            claude_budget=8000,
            gemini_level="medium",
            gemini_budget=8000,
            estimated_cost_per_request=0.15,
            latency_estimate="5-15 seconds"
        )
    else:
        # Low complexity
        return ThinkingConfig(
            claude_budget=2000,
            gemini_level="low",
            gemini_budget=2000,
            estimated_cost_per_request=0.05,
            latency_estimate="2-5 seconds"
        )

# Test
config = classify_and_configure(
    "Prove that the algorithm has O(n log n) time complexity"
)
print(f"Claude budget: {config.claude_budget}")
print(f"Gemini level: {config.gemini_level}")
print(f"Estimated cost: ${config.estimated_cost_per_request}")
print(f"Latency: {config.latency_estimate}")
```

</details>

---

## Summary

✅ **Anthropic uses `budget_tokens`**: Minimum 1,024, must be less than `max_tokens`
✅ **Gemini 3 uses `thinkingLevel`**: minimal, low, medium, high
✅ **Gemini 2.5 uses `thinkingBudget`**: Token count or -1 for dynamic
✅ **Start small**: Begin with minimum budget and scale up
✅ **Thinking tokens cost money**: Billed as output tokens

**Next:** [Prompting for Extended Thinking](./02-prompting-for-extended-thinking.md)

---

## Further Reading

- [Anthropic Extended Thinking Guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Gemini Thinking Configuration](https://ai.google.dev/gemini-api/docs/thinking)
- [Anthropic Token Counting](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)

---

<!-- 
Sources Consulted:
- Anthropic Extended Thinking: budget_tokens configuration, limits, max_tokens relationship
- Gemini Thinking: thinkingLevel, thinkingBudget, model-specific ranges
- Anthropic: Streaming requirements for large max_tokens
-->
