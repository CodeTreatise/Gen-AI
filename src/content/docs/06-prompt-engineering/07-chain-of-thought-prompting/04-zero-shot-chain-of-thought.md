---
title: "Zero-Shot Chain-of-Thought"
---

# Zero-Shot Chain-of-Thought

## Introduction

Zero-shot chain-of-thought is remarkably simple: add "Let's think step by step" to your prompt. No examples needed. This single phrase triggers reasoning behavior across a wide range of tasks, making it one of the most cost-effective prompting techniques available.

> **🔑 Key Insight:** Zero-shot CoT works because the phrase activates reasoning patterns the model learned during training. It's essentially a shortcut to few-shot CoT without the example overhead.

### What We'll Cover

- The "Let's think step by step" technique
- How zero-shot CoT works
- Effectiveness across task types
- Combining with other techniques

### Prerequisites

- [Eliciting Reasoning Steps](./02-eliciting-reasoning-steps.md)
- [Step-by-Step Instructions](./03-step-by-step-instructions.md)

---

## The Magic Phrase

### Basic Usage

```python
# Without zero-shot CoT
prompt = "What is 127 x 83?"
# Response: "10541" (may be wrong, no work shown)

# With zero-shot CoT
prompt = "What is 127 x 83? Let's think step by step."
# Response shows work:
# "Let me break this down:
#  127 x 83 = 127 x (80 + 3)
#           = 127 x 80 + 127 x 3
#           = 10160 + 381
#           = 10541"
```

### Variations That Work

| Phrase | Strength | Use Case |
|--------|----------|----------|
| "Let's think step by step" | Strong | General reasoning |
| "Let's work through this" | Strong | Problem-solving |
| "Let's break this down" | Strong | Analysis tasks |
| "Think carefully before answering" | Medium | Accuracy-critical |
| "Reason through this problem" | Medium | Logic tasks |
| "Show your work" | Medium | Math problems |

### Phrase Placement

```python
# End placement (most common)
prompt = "Calculate the area of a circle with radius 5. Let's think step by step."

# Beginning placement
prompt = "Let's think step by step about calculating the area of a circle with radius 5."

# System message (for consistent behavior)
system = "You always think step by step before answering."
user = "Calculate the area of a circle with radius 5."
```

---

## How Zero-Shot CoT Works

### The Mechanism

```
Standard prompt:
  Input → [Model weights] → Output

Zero-shot CoT:
  Input + "think step by step" → [Reasoning trigger] → 
  Step 1 output → [Context update] →
  Step 2 output → [Context update] →
  ... →
  Final output
```

The key insight: each generated token becomes context for the next. "Let's think step by step" creates a reasoning scaffold that the model then fills.

### Why It Works

| Factor | Explanation |
|--------|-------------|
| Training data | Models saw many examples of step-by-step solutions |
| Pattern activation | The phrase triggers learned reasoning patterns |
| Token generation | Each step becomes input for the next |
| Error visibility | Mistakes in reasoning become visible and correctable |

---

## Effectiveness Across Tasks

### Strong Performance

| Task Type | Improvement | Example |
|-----------|-------------|---------|
| Arithmetic | +20-40% | Multi-step calculations |
| Word problems | +20-35% | Story-based math |
| Logical reasoning | +15-30% | Syllogisms, puzzles |
| Common sense | +10-20% | Physical reasoning |
| Symbolic reasoning | +25-40% | Pattern matching |

### Example: Arithmetic

```python
# Without CoT - prone to errors
prompt = "If a shirt costs $24.99 and is 30% off, and there's an additional 8% sales tax on the discounted price, what's the final cost?"
# Common wrong answer: Just applies 30% off

# With zero-shot CoT
prompt = """If a shirt costs $24.99 and is 30% off, and there's an additional 8% sales tax on the discounted price, what's the final cost?

Let's work through this step by step."""

# Response:
# 1. Original price: $24.99
# 2. 30% discount: $24.99 x 0.30 = $7.50
# 3. Discounted price: $24.99 - $7.50 = $17.49
# 4. 8% tax on discounted: $17.49 x 0.08 = $1.40
# 5. Final price: $17.49 + $1.40 = $18.89
```

### Example: Logic Puzzle

```python
prompt = """Alice is older than Bob. Bob is older than Charlie. 
Charlie is older than Diana. Is Alice older than Diana?

Let's reason through this carefully."""

# Response:
# Let me trace the age relationships:
# 1. Alice > Bob (Alice is older than Bob)
# 2. Bob > Charlie (Bob is older than Charlie)
# 3. Charlie > Diana (Charlie is older than Diana)
# 
# Following the chain:
# Alice > Bob > Charlie > Diana
# 
# Since age is transitive, and Alice is at the top of the chain,
# Alice is older than Diana.
# 
# Answer: Yes, Alice is older than Diana.
```

---

## When Zero-Shot CoT Helps Most

### High-Value Scenarios

| Scenario | Why CoT Helps |
|----------|---------------|
| Multi-step math | Prevents compounding errors |
| Word problems | Forces extraction of numbers |
| Logic chains | Makes deductions explicit |
| Comparative analysis | Shows evaluation criteria |
| Debugging | Traces execution path |

### Low-Value Scenarios

| Scenario | Why CoT Adds Little |
|----------|---------------------|
| Simple facts | "What's the capital of France?" |
| Direct translation | "Translate 'hello' to Spanish" |
| Simple lookup | "What year was X born?" |
| Speed-critical | Latency matters more than accuracy |
| Creative writing | Overthinking hurts flow |

---

## Combining with Other Techniques

### Zero-Shot CoT + Role

```python
prompt = """You are an expert mathematician.

Calculate the probability of getting exactly 3 heads in 5 coin flips.

Let's think through this step by step."""
```

### Zero-Shot CoT + Format Specification

```python
prompt = """Determine whether this number is prime: 97

Let's think step by step.

Format your response as:
REASONING: [your steps]
ANSWER: [Prime/Not Prime]"""
```

### Zero-Shot CoT + Verification

```python
prompt = """Calculate: 234 x 56

Let's work through this carefully.

After calculating, verify your answer by:
1. Checking with estimation (200 x 50 = 10,000, so answer should be near 13,000)
2. Confirming the units digit is correct"""
```

### Zero-Shot CoT + JSON Output

```python
prompt = """Analyze whether 2024 is a leap year.

Let's think through the rules step by step.

Then provide your answer as JSON:
{
  "reasoning": "brief summary",
  "is_leap_year": true/false
}"""
```

---

## Zero-Shot vs Few-Shot CoT

### Comparison

| Aspect | Zero-Shot CoT | Few-Shot CoT |
|--------|---------------|--------------|
| Setup cost | None | Need good examples |
| Token usage | Lower | Higher |
| Consistency | Variable | More consistent |
| Task coverage | Broad | Tailored |
| Accuracy | Good | Often better |

### When to Use Each

```python
# Use Zero-Shot CoT when:
# - Quick prototyping
# - One-off questions
# - Token budget is tight
# - Task is straightforward

zero_shot = "Solve for x: 2x + 5 = 17. Let's think step by step."

# Use Few-Shot CoT when:
# - High accuracy needed
# - Specific format required
# - Complex domain-specific reasoning
# - Production systems

few_shot = """Solve linear equations step by step.

Example: 3x + 6 = 15
Step 1: Subtract 6: 3x = 9
Step 2: Divide by 3: x = 3

Now solve: 2x + 5 = 17
"""
```

---

## Implementation Example

```python
from openai import OpenAI

client = OpenAI()

def solve_with_cot(problem: str, model: str = "gpt-4o") -> str:
    """Solve a problem using zero-shot chain-of-thought."""
    
    prompt = f"{problem}\n\nLet's think through this step by step."
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Usage
result = solve_with_cot(
    "A store buys items for $15 and sells them for $24. "
    "If they sell 150 items, what's their total profit?"
)
print(result)
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Use natural phrasing | "Let's" works better than "You must" |
| Place at prompt end | Strong trigger position |
| Match to task complexity | Don't use for trivial tasks |
| Combine with verification | Catch reasoning errors |
| Extract final answer | Prevent buried conclusions |

---

## Common Pitfalls

| Mistake | Solution |
|---------|----------|
| Using with simple questions | Skip CoT for trivial tasks |
| Forgetting answer extraction | Add "FINAL ANSWER:" prompt |
| Using with reasoning models | Not needed - they do it internally |
| Expecting perfection | CoT improves odds, doesn't guarantee |

---

## Hands-on Exercise

### Your Task

Test zero-shot CoT on a multi-step problem.

### Problem

A farmer has a rectangular field that is 120 meters long and 80 meters wide. They want to:
1. Fence the entire perimeter
2. Divide it into 4 equal plots with internal fencing
3. Calculate total fencing needed

Test with and without "Let's think step by step."

<details>
<summary>Expected Improvement</summary>

**Without CoT:** Model might miss internal fencing or miscalculate.

**With CoT:** Model should show:
1. Perimeter = 2(120 + 80) = 400m
2. Internal fencing for 4 plots: divide 120m and 80m
   - 1 fence at 60m across width = 80m
   - 1 fence at 40m across length = 120m
   - Or: 3 internal fences of 80m or 120m depending on division
3. Total = perimeter + internal fencing

The step-by-step approach catches the ambiguity in "4 equal plots" (2x2 grid vs 1x4 strip).

</details>

---

## Summary

- "Let's think step by step" is a simple, powerful trigger
- Zero-shot CoT requires no examples
- Most effective on multi-step reasoning tasks
- Combine with other techniques for best results
- Not needed for reasoning models (GPT-5, o3, o4-mini)

**Next:** [When NOT to Use Chain-of-Thought](./05-when-not-to-use-cot.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide, Chain-of-Thought research -->
