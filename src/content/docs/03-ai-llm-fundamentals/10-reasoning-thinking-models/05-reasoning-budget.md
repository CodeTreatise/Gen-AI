---
title: "Reasoning Budget & Test-Time Compute"
---

# Reasoning Budget & Test-Time Compute

## Introduction

Test-time compute scaling is the key innovation behind reasoning models. You can configure how much "thinking" the model does, trading off cost/latency for quality.

### What We'll Cover

- Configurable compute allocation
- Test-time compute scaling
- Reasoning token limits
- Dynamic budget strategies

---

## Test-Time Compute Scaling

### The Core Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                TEST-TIME COMPUTE SCALING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional Training: Invest compute during training            │
│  Test-Time Scaling: Invest compute during inference              │
│                                                                  │
│  Key Insight: More thinking at inference = better answers        │
│                                                                  │
│  Compute Level:     LOW        MEDIUM        HIGH                │
│                      │            │            │                 │
│  Thinking Tokens:   2K          10K          50K+                │
│  Quality:           ★★          ★★★          ★★★★                │
│  Latency:           5s          15s          60s+                │
│  Cost:              $           $$           $$$$                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why It Works

```python
test_time_compute_insight = {
    "analogy": "Like giving a human more time to think",
    "mechanism": "Model can explore more reasoning paths",
    "verification": "Model can check its own work",
    "backtracking": "Model can abandon wrong approaches",
    "result": "Higher accuracy on complex problems"
}
```

---

## OpenAI o3 Reasoning Effort

### Configuring Compute

```python
from openai import OpenAI

client = OpenAI()

def o3_with_effort(problem: str, effort: str = "medium") -> str:
    """Control o3 reasoning effort level"""
    
    response = client.chat.completions.create(
        model="o3",
        reasoning_effort=effort,  # "low", "medium", "high"
        messages=[{"role": "user", "content": problem}]
    )
    
    return response.choices[0].message.content

# Low: Quick problems, cost-sensitive
result_low = o3_with_effort("Calculate 2 + 2", effort="low")

# Medium: Balanced (default)
result_med = o3_with_effort("Solve this algebra problem", effort="medium")

# High: Maximum accuracy, complex problems
result_high = o3_with_effort("Prove this theorem", effort="high")
```

### Effort Level Comparison

| Effort | Thinking Tokens | Latency | Cost | Accuracy |
|--------|-----------------|---------|------|----------|
| Low | ~1K-5K | 5-15s | $ | Good |
| Medium | ~5K-20K | 15-45s | $$ | Better |
| High | ~20K-100K | 45s-2min+ | $$$ | Best |

---

## Claude Thinking Budget

### Setting Token Limits

```python
from anthropic import Anthropic

client = Anthropic()

def claude_thinking(problem: str, budget: int) -> dict:
    """Set explicit thinking token budget"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": budget  # Max thinking tokens
        },
        messages=[{"role": "user", "content": problem}]
    )
    
    # Extract thinking and answer
    thinking = ""
    answer = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text
    
    return {
        "thinking_tokens_used": len(thinking.split()),  # Approximate
        "thinking": thinking,
        "answer": answer
    }

# Low budget for simple problems
simple_result = claude_thinking("What is 2+2?", budget=1000)

# High budget for complex problems
complex_result = claude_thinking("Prove the Riemann hypothesis... (kidding)", budget=50000)
```

---

## Dynamic Budget Strategies

### Complexity-Based Allocation

```python
def estimate_problem_complexity(problem: str) -> str:
    """Estimate problem complexity for budget allocation"""
    
    # Simple heuristics
    complexity_signals = {
        "prove": 3,
        "explain why": 2,
        "calculate": 1,
        "multi-step": 2,
        "analyze": 2,
        "compare": 1,
    }
    
    score = 0
    problem_lower = problem.lower()
    
    for signal, points in complexity_signals.items():
        if signal in problem_lower:
            score += points
    
    # Word count as complexity signal
    word_count = len(problem.split())
    if word_count > 200:
        score += 2
    elif word_count > 100:
        score += 1
    
    if score >= 5:
        return "high"
    elif score >= 2:
        return "medium"
    return "low"

def adaptive_budget(problem: str) -> int:
    """Set budget based on estimated complexity"""
    
    complexity = estimate_problem_complexity(problem)
    
    budgets = {
        "low": 2000,
        "medium": 8000,
        "high": 25000
    }
    
    return budgets[complexity]

# Usage
budget = adaptive_budget("Prove that there are infinitely many primes")
result = claude_thinking(problem, budget)
```

### Progressive Reasoning

```python
async def progressive_reasoning(problem: str, max_attempts: int = 3):
    """Start with low budget, increase if needed"""
    
    budgets = [2000, 10000, 30000]
    
    for attempt, budget in enumerate(budgets[:max_attempts]):
        result = await claude_thinking_async(problem, budget)
        
        # Check if answer is satisfactory
        if is_confident_answer(result):
            return {
                "answer": result["answer"],
                "attempts": attempt + 1,
                "final_budget": budget
            }
    
    return {
        "answer": result["answer"],
        "attempts": max_attempts,
        "final_budget": budgets[max_attempts - 1],
        "note": "Max attempts reached"
    }

def is_confident_answer(result: dict) -> bool:
    """Check if model seems confident"""
    
    # Look for uncertainty signals
    uncertainty_phrases = [
        "I'm not sure",
        "This might be wrong",
        "I need more information",
        "Let me reconsider"
    ]
    
    answer = result["answer"].lower()
    return not any(phrase in answer for phrase in uncertainty_phrases)
```

---

## Budget Monitoring

### Tracking Usage

```python
class ReasoningBudgetTracker:
    """Track reasoning token usage and costs"""
    
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.today_spend = 0.0
        self.queries = []
    
    def log_query(self, thinking_tokens: int, model: str = "o3"):
        cost = self.estimate_cost(thinking_tokens, model)
        self.today_spend += cost
        self.queries.append({
            "tokens": thinking_tokens,
            "cost": cost,
            "model": model
        })
    
    def estimate_cost(self, thinking_tokens: int, model: str) -> float:
        rates = {"o3": 0.04, "o1": 0.06, "claude": 0.015}  # per 1K tokens
        rate = rates.get(model, 0.04)
        return (thinking_tokens / 1000) * rate
    
    def can_afford(self, estimated_tokens: int, model: str = "o3") -> bool:
        estimated_cost = self.estimate_cost(estimated_tokens, model)
        return (self.today_spend + estimated_cost) <= self.daily_budget
    
    def get_remaining_budget(self) -> float:
        return self.daily_budget - self.today_spend

# Usage
tracker = ReasoningBudgetTracker(daily_budget=50.0)

if tracker.can_afford(10000, "o3"):
    result = call_o3(problem)
    tracker.log_query(10000, "o3")
else:
    result = call_fast_model(problem)  # Fallback
```

---

## Summary

✅ **Test-time compute**: More thinking = better answers

✅ **o3 reasoning_effort**: low/medium/high settings

✅ **Claude budget_tokens**: Explicit token limits

✅ **Dynamic allocation**: Match budget to complexity

✅ **Monitor spending**: Track and limit reasoning costs

**Next:** [Reasoning Token Visibility](./06-token-visibility.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Cost Implications](./04-cost-implications.md) | [Reasoning Models](./00-reasoning-thinking-models.md) | [Token Visibility](./06-token-visibility.md) |

