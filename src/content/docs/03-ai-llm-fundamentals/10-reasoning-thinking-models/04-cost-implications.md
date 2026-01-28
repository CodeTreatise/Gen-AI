---
title: "Cost Implications of Thinking Tokens"
---

# Cost Implications of Thinking Tokens

## Introduction

Reasoning models generate "thinking tokens" during extended computation, which significantly impacts costs. Understanding this is crucial for budgeting and optimization.

### What We'll Cover

- How thinking tokens add to costs
- Budgeting for reasoning
- Cost vs quality trade-off
- Optimization strategies

---

## How Thinking Tokens Work

### Token Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REASONING TOKEN FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 500 tokens                                               │
│  ↓                                                               │
│  Thinking: 5,000-50,000 tokens (internal reasoning)              │
│  ↓                                                               │
│  Output: 1,000 tokens                                            │
│                                                                  │
│  BILLING (OpenAI o-series):                                      │
│  ├── Input tokens: Charged at input rate                        │
│  ├── Thinking tokens: Charged at output rate (!)                │
│  └── Output tokens: Charged at output rate                       │
│                                                                  │
│  Total: 500 input + 5,000 thinking + 1,000 output                │
│       = 500 input + 6,000 output tokens                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Comparison

```python
# Example: Solving a complex math problem

# GPT-4o approach
gpt4o_cost = {
    "input_tokens": 500,
    "output_tokens": 500,
    "input_rate": 2.50 / 1_000_000,
    "output_rate": 10.00 / 1_000_000,
    "total": (500 * 2.50 + 500 * 10.00) / 1_000_000
    # = $0.00625
}

# o3 approach
o3_cost = {
    "input_tokens": 500,
    "thinking_tokens": 10000,  # Internal reasoning
    "output_tokens": 500,
    "input_rate": 10.00 / 1_000_000,  # Higher base rate
    "output_rate": 40.00 / 1_000_000,  # Much higher
    "total": (500 * 10.00 + (10000 + 500) * 40.00) / 1_000_000
    # = $0.425
}

# o3 is ~68x more expensive for this task!
```

---

## Pricing Comparison

### Current Rates (Per 1M Tokens)

| Model | Input | Output | Thinking |
|-------|-------|--------|----------|
| GPT-4o | $2.50 | $10.00 | N/A |
| o1 | $15.00 | $60.00 | @ output rate |
| o3 | ~$10.00 | ~$40.00 | @ output rate |
| o4-mini | ~$1.00 | ~$4.00 | @ output rate |
| Claude thinking | $3.00 | $15.00 | Included in output |

---

## Budgeting Strategies

### Token Budget Limits

```python
from anthropic import Anthropic

client = Anthropic()

def claude_with_budget(problem: str, thinking_budget: int = 5000) -> str:
    """Control thinking token spend"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": thinking_budget  # Limit thinking
        },
        messages=[{"role": "user", "content": problem}]
    )
    
    return response.content

# Adjust budget based on problem complexity
budgets = {
    "simple": 2000,
    "moderate": 5000,
    "complex": 10000,
    "very_complex": 20000
}
```

### Cost Estimation

```python
def estimate_reasoning_cost(
    input_tokens: int,
    expected_thinking: int,
    expected_output: int,
    model: str = "o3"
) -> dict:
    """Estimate cost before running"""
    
    rates = {
        "o1": {"input": 15.00, "output": 60.00},
        "o3": {"input": 10.00, "output": 40.00},
        "o4-mini": {"input": 1.00, "output": 4.00},
    }
    
    rate = rates.get(model, rates["o3"])
    
    input_cost = (input_tokens * rate["input"]) / 1_000_000
    thinking_cost = (expected_thinking * rate["output"]) / 1_000_000
    output_cost = (expected_output * rate["output"]) / 1_000_000
    
    return {
        "input_cost": f"${input_cost:.4f}",
        "thinking_cost": f"${thinking_cost:.4f}",
        "output_cost": f"${output_cost:.4f}",
        "total": f"${input_cost + thinking_cost + output_cost:.4f}"
    }

# Example
estimate = estimate_reasoning_cost(
    input_tokens=1000,
    expected_thinking=15000,
    expected_output=2000,
    model="o3"
)
print(estimate)
# {'input_cost': '$0.0100', 'thinking_cost': '$0.6000', 
#  'output_cost': '$0.0800', 'total': '$0.6900'}
```

---

## Cost vs Quality Trade-Off

### When Higher Cost Is Worth It

```python
cost_justified_scenarios = {
    "high_stakes_decision": {
        "example": "Medical diagnosis support",
        "cost_per_query": "$1.00",
        "value_of_accuracy": "Critical",
        "justified": True
    },
    "code_review": {
        "example": "Security audit of authentication",
        "cost_per_query": "$0.50",
        "value_of_accuracy": "High (prevents breaches)",
        "justified": True
    },
    "content_classification": {
        "example": "Spam detection",
        "cost_per_query": "$0.001",
        "value_of_accuracy": "Low per item",
        "justified": False  # Use fast model
    }
}
```

### ROI Calculation

```python
def reasoning_roi(
    task_value: float,
    fast_accuracy: float,
    reasoning_accuracy: float,
    fast_cost: float,
    reasoning_cost: float,
    volume: int
) -> dict:
    """Calculate ROI of using reasoning model"""
    
    # Expected value with each approach
    fast_ev = task_value * fast_accuracy - fast_cost
    reasoning_ev = task_value * reasoning_accuracy - reasoning_cost
    
    # Total difference at volume
    total_difference = (reasoning_ev - fast_ev) * volume
    
    return {
        "fast_model_ev": f"${fast_ev:.2f}/task",
        "reasoning_model_ev": f"${reasoning_ev:.2f}/task",
        "benefit_per_task": f"${reasoning_ev - fast_ev:.2f}",
        "total_benefit": f"${total_difference:.2f}",
        "recommendation": "reasoning" if reasoning_ev > fast_ev else "fast"
    }

# Example: Legal document review
roi = reasoning_roi(
    task_value=100,       # Value of correct review
    fast_accuracy=0.80,   # 80% accuracy
    reasoning_accuracy=0.95,  # 95% accuracy
    fast_cost=0.01,       # Cheap
    reasoning_cost=0.50,  # More expensive
    volume=1000           # Monthly reviews
)
# Reasoning justified: 95% × $100 - $0.50 > 80% × $100 - $0.01
```

---

## Optimization Strategies

### 1. Tiered Approach

```python
def tiered_processing(task: dict) -> str:
    """Use reasoning only when needed"""
    
    # First try with fast model
    fast_result = call_model("gpt-4o", task)
    
    # Check confidence or complexity
    if needs_reasoning(fast_result, task):
        return call_model("o3", task)
    
    return fast_result
```

### 2. Reasoning Effort Selection

```python
def adaptive_reasoning(problem: str) -> str:
    """Adjust reasoning effort to problem complexity"""
    
    complexity = estimate_complexity(problem)
    
    effort = {
        "low": "low",
        "medium": "medium", 
        "high": "high"
    }.get(complexity, "medium")
    
    response = client.chat.completions.create(
        model="o3",
        reasoning_effort=effort,
        messages=[{"role": "user", "content": problem}]
    )
    
    return response.choices[0].message.content
```

### 3. Caching Reasoning

```python
import hashlib

reasoning_cache = {}

def cached_reasoning(problem: str) -> str:
    """Cache reasoning results for repeated queries"""
    
    cache_key = hashlib.md5(problem.encode()).hexdigest()
    
    if cache_key in reasoning_cache:
        return reasoning_cache[cache_key]
    
    result = call_reasoning_model(problem)
    reasoning_cache[cache_key] = result
    
    return result
```

---

## Summary

✅ **Thinking tokens are expensive** - billed at output rate

✅ **Can be 10-100x more expensive** than fast models

✅ **Set budgets** to control costs

✅ **Calculate ROI** - sometimes reasoning pays for itself

✅ **Use tiered approaches** - reasoning only when needed

**Next:** [Reasoning Budget & Test-Time Compute](./05-reasoning-budget.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [When to Use](./03-when-to-use.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [Reasoning Budget](./05-reasoning-budget.md) |

