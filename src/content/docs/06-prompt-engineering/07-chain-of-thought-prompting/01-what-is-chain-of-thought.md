---
title: "What is Chain-of-Thought?"
---

# What is Chain-of-Thought?

## Introduction

Chain-of-thought (CoT) prompting is a technique where you encourage an LLM to explicitly show its reasoning process before arriving at an answer. Instead of producing just a final result, the model generates intermediate steps—"showing its work" like a student solving a math problem.

> **🤖 AI Context:** CoT emerged from research showing that asking models to reason step-by-step dramatically improved accuracy on complex tasks. It's now built into reasoning models automatically.

### What We'll Cover

- What chain-of-thought prompting is
- How explicit reasoning improves accuracy
- The mechanics of step-by-step problem solving
- Why "showing work" helps

### Prerequisites

- [Few-Shot Prompting](../04-few-shot-prompting/00-few-shot-prompting-overview.md)
- Basic understanding of LLM responses

---

## The Core Idea

### Without Chain-of-Thought

```
User: If a store has 15 apples and sells 8, then receives a shipment 
of 12 more, how many apples does the store have?

Model: 19 apples.
```

The model jumps straight to the answer. Sometimes correct, sometimes not—especially with more complex problems.

### With Chain-of-Thought

```
User: If a store has 15 apples and sells 8, then receives a shipment 
of 12 more, how many apples does the store have? Think through this 
step by step.

Model: Let me work through this step by step:
1. Start: 15 apples
2. After selling 8: 15 - 8 = 7 apples
3. After shipment of 12: 7 + 12 = 19 apples

The store has 19 apples.
```

The model shows its reasoning, making errors visible and often preventing them.

---

## Why CoT Works

### Cognitive Offloading

When the model generates intermediate steps, it:

| Benefit | Description |
|---------|-------------|
| Reduces working memory load | Each step becomes input for the next |
| Creates checkpoints | Errors become visible mid-reasoning |
| Enables self-correction | Model can catch mistakes |
| Improves coherence | Each step builds logically |

### Token-by-Token Generation

LLMs generate one token at a time. Without CoT:

```
Input: Complex problem
↓
[Internal processing limited to model weights]
↓
Output: Final answer (may be wrong)
```

With CoT:

```
Input: Complex problem
↓
Output token: "Let"
↓
Output token: "me"
↓
Output token: "think"
↓
... [reasoning tokens] ...
↓
Output: Final answer (more likely correct)
```

Each reasoning token becomes part of the context for generating the next token, effectively extending the model's "thinking space."

---

## Types of Reasoning

### Deductive Reasoning

Moving from general principles to specific conclusions:

```
Prompt: All mammals are warm-blooded. Dolphins are mammals. 
Are dolphins warm-blooded? Explain your reasoning.

Response: Let me work through this:
1. Premise 1: All mammals are warm-blooded
2. Premise 2: Dolphins are mammals
3. Since dolphins belong to the category "mammals"
4. And all members of "mammals" are warm-blooded
5. Therefore, dolphins are warm-blooded.

Yes, dolphins are warm-blooded.
```

### Mathematical Reasoning

Breaking down calculations:

```
Prompt: A train travels 120 miles in 2 hours, then 90 miles in 
1.5 hours. What's the average speed? Show your work.

Response: Let me calculate this step by step:

1. Total distance:
   - First leg: 120 miles
   - Second leg: 90 miles
   - Total: 120 + 90 = 210 miles

2. Total time:
   - First leg: 2 hours
   - Second leg: 1.5 hours
   - Total: 2 + 1.5 = 3.5 hours

3. Average speed = Total distance / Total time
   - 210 miles / 3.5 hours = 60 mph

The average speed is 60 mph.
```

### Analytical Reasoning

Examining multiple factors:

```
Prompt: Should a startup focus on user growth or profitability first? 
Analyze the tradeoffs.

Response: Let me analyze this systematically:

1. User Growth Focus:
   - Pros: Network effects, market share, investor appeal
   - Cons: Cash burn, may never find profitability path
   - Best for: Platforms, marketplaces, social networks

2. Profitability Focus:
   - Pros: Sustainability, capital efficiency, optionality
   - Cons: Slower growth, competitors may outpace
   - Best for: Services, niche products, bootstrapped

3. Key Factors to Consider:
   - Available runway
   - Market dynamics (winner-take-all vs. fragmented)
   - Business model (unit economics)
   - Competitive landscape

4. Recommendation Framework:
   If unit economics are proven → Growth
   If unit economics are unproven → Profitability first
```

---

## Accuracy Improvements

Research and practice show significant improvements with CoT:

| Task Type | Without CoT | With CoT | Improvement |
|-----------|-------------|----------|-------------|
| Multi-step math | ~60-70% | ~85-95% | +20-30% |
| Logic puzzles | ~50-65% | ~80-90% | +25-30% |
| Word problems | ~65-75% | ~85-92% | +15-20% |
| Common sense reasoning | ~70-80% | ~85-92% | +10-15% |

> **Note:** Actual improvements vary by model, task complexity, and prompt quality.

---

## Showing the "Work"

### The Teaching Analogy

Think of prompting for CoT like asking a student to show their work:

| Without "show work" | With "show work" |
|---------------------|------------------|
| Teacher can't diagnose errors | Errors become visible |
| Student may guess | Student must reason |
| No learning reinforcement | Reasoning reinforces learning |
| Hard to give partial credit | Can identify where confusion starts |

### Practical Example: Debugging

```python
# Prompt for code review with reasoning
prompt = """Review this function and identify any bugs. 
Think through each line's behavior before concluding.

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""

# Response with CoT:
"""
Let me trace through this function step by step:

1. `total = 0` - Initializes sum accumulator ✓
2. `for num in numbers` - Iterates through input list ✓
3. `total += num` - Accumulates sum ✓
4. `return total / len(numbers)` - Calculates average

Potential issues identified:

Issue 1: Division by zero
- If `numbers` is empty, `len(numbers)` is 0
- Division by zero raises ZeroDivisionError

Issue 2: Type handling
- No check that all elements are numeric
- Will fail if list contains strings

Issue 3: Integer division (Python 2 only)
- In Python 2, int/int = int (truncates)
- Python 3 handles this correctly

Recommended fix:
def calculate_average(numbers):
    if not numbers:
        return 0  # or raise ValueError
    return sum(numbers) / len(numbers)
"""
```

---

## When CoT Matters Most

### High-Value Scenarios

| Scenario | Why CoT Helps |
|----------|---------------|
| Multi-step calculations | Reduces compounding errors |
| Complex decisions | Makes tradeoffs explicit |
| Ambiguous inputs | Exposes interpretation choices |
| High-stakes outputs | Creates audit trail |

### Low-Value Scenarios

| Scenario | Why CoT Adds Little |
|----------|---------------------|
| Simple factual queries | "What's the capital of France?" |
| Direct transformations | "Translate to Spanish" |
| Speed-critical tasks | Latency matters more than accuracy |
| Creative generation | Overthinking hurts creativity |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Match technique to model | GPT models need prompting; reasoning models don't |
| Request explicit steps | "Show your work" beats "be careful" |
| Keep reasoning visible | Don't hide intermediate steps |
| Verify the reasoning | Model can still make logical errors |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using CoT for simple tasks | Match technique to complexity |
| Trusting CoT blindly | Reasoning can still be flawed |
| Using CoT with reasoning models | Let them handle it internally |
| Vague CoT requests | Be specific: "List each step" |

---

## Hands-on Exercise

### Your Task

Compare outputs with and without chain-of-thought prompting.

### Problem

A farmer has 3 fields. Field A produces 450 bushels of wheat. Field B produces 30% more than Field A. Field C produces half as much as Fields A and B combined. What is the total harvest?

### Test Both Approaches

1. **Without CoT**: Ask directly for the answer
2. **With CoT**: Ask to show step-by-step reasoning

<details>
<summary>💡 Expected Difference</summary>

**Without CoT**: Model might give just "1462.5 bushels" (correct or wrong)

**With CoT**: Model shows:
1. Field A: 450 bushels
2. Field B: 450 × 1.30 = 585 bushels
3. A + B = 450 + 585 = 1035 bushels
4. Field C: 1035 / 2 = 517.5 bushels
5. Total: 450 + 585 + 517.5 = 1552.5 bushels

You can verify each step!

</details>

<details>
<summary>✅ CoT Prompt</summary>

```python
from openai import OpenAI
client = OpenAI()

prompt = """A farmer has 3 fields:
- Field A produces 450 bushels of wheat
- Field B produces 30% more than Field A
- Field C produces half as much as Fields A and B combined

What is the total harvest?

Please solve this step by step, showing your calculations 
for each field before giving the final total.
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

</details>

---

## Summary

✅ **Chain-of-thought** makes models show their reasoning

✅ **Explicit steps** reduce errors on complex tasks

✅ **Works best** on math, logic, and multi-step problems

✅ **Creates visibility** into model's thought process

✅ **Not needed** for reasoning models (GPT-5, o3, o4-mini)

**Next:** [Eliciting Reasoning Steps](./02-eliciting-reasoning-steps.md)

---

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)

---

<!-- 
Sources Consulted:
- OpenAI Reasoning Guide: https://platform.openai.com/docs/guides/reasoning
- OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
-->
