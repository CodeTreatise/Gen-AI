---
title: "Eliciting Reasoning Steps"
---

# Eliciting Reasoning Steps

## Introduction

Getting a model to reason step-by-step requires specific techniques. We'll explore trigger phrases, example-based elicitation, and format specifications that reliably produce chain-of-thought responses.

### What We'll Cover

- Trigger phrases that initiate reasoning
- Example-based elicitation (few-shot CoT)
- Structured reasoning requests
- Reasoning format specification

### Prerequisites

- [What is Chain-of-Thought?](./01-what-is-chain-of-thought.md)

---

## Trigger Phrases

### Common CoT Triggers

| Category | Phrase | Best For |
|----------|--------|----------|
| General | "Let's think step by step" | Any reasoning task |
| Mathematical | "Show your calculations" | Math problems |
| Analytical | "Break this down" | Complex analysis |
| Procedural | "Walk through the process" | How-to tasks |
| Verification | "Reason through this carefully" | High-stakes decisions |

### Trigger Phrase Examples

```python
# General reasoning trigger
prompt = """What is the probability of rolling at least one 6 
in three dice rolls?

Let's think through this step by step."""

# Mathematical trigger
prompt = """Calculate the compound interest on $10,000 at 5% 
annual rate, compounded quarterly, over 3 years.

Show all your calculations clearly."""
```

### Placement Matters

| Position | Effect | Example |
|----------|--------|---------|
| End of prompt | Strong trigger | "...Let's think step by step." |
| Start of prompt | Sets expectation | "Step by step, let's figure out..." |
| In system message | Consistent behavior | "Always show your reasoning..." |

---

## Example-Based Elicitation (Few-Shot CoT)

### The Power of Examples

Showing the model examples of reasoning is more powerful than just asking for it:

```python
system_prompt = """You solve word problems by thinking step by step.

# Example 1

Problem: A bookshelf has 5 shelves. Each shelf holds 12 books. 
If 17 books are removed, how many remain?

Reasoning:
1. Total capacity: 5 shelves x 12 books = 60 books
2. After removal: 60 - 17 = 43 books

Answer: 43 books

# Example 2

Problem: A train leaves at 9:15 AM and arrives at 2:45 PM. 
If it makes three 15-minute stops, what's the moving time?

Reasoning:
1. Total travel time: 9:15 AM to 2:45 PM = 5 hours 30 minutes
2. Convert to minutes: 5 x 60 + 30 = 330 minutes
3. Stop time: 3 x 15 = 45 minutes
4. Moving time: 330 - 45 = 285 minutes = 4 hours 45 minutes

Answer: 4 hours 45 minutes

---

Now solve the following problem using the same step-by-step approach:
"""
```

### Crafting Good Examples

| Principle | Implementation |
|-----------|----------------|
| Diverse problems | Cover different sub-types |
| Clear structure | Numbered steps, explicit labels |
| Appropriate difficulty | Match expected query complexity |
| Complete reasoning | Don't skip "obvious" steps |

---

## Structured Reasoning Requests

### Template-Based Reasoning

```python
template = """Given the problem below, analyze it using this framework:

PROBLEM: {problem}

Step 1: UNDERSTAND - What is being asked?
Step 2: PLAN - What approach will you use?
Step 3: EXECUTE - Show each calculation
Step 4: VERIFY - Does the answer make sense?
Step 5: ANSWER - State the final answer
"""
```

---

## Reasoning Format Specification

### Different Formats

**Numbered List:**
```
1. First, identify...
2. Then, calculate...
3. Finally, conclude...
```

**Labeled Sections:**
```
GIVEN: [inputs]
FIND: [goal]
SOLUTION: [work]
ANSWER: [result]
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Be explicit about format | Models follow clear structure |
| Show complete examples | Don't skip steps |
| Match complexity | Simple triggers for simple tasks |
| Use system messages | Consistent behavior |

---

## Common Pitfalls

| Mistake | Solution |
|---------|----------|
| Vague triggers | Use specific phrases |
| Incomplete examples | Show full reasoning chains |
| Over-constraining | Allow model flexibility |

---

## Hands-on Exercise

### Your Task

Create a few-shot prompt for solving percentage problems.

### Requirements

1. Include 2 examples with step-by-step reasoning
2. Use a consistent format
3. Test with a new problem

<details>
<summary>Solution</summary>

```python
prompt = """You solve percentage problems step by step.

Example 1:
Q: What is 15% of 80?
A: Step 1: Convert 15% to decimal: 15/100 = 0.15
   Step 2: Multiply: 0.15 x 80 = 12
   Answer: 12

Example 2:
Q: 24 is what percent of 60?
A: Step 1: Divide: 24/60 = 0.4
   Step 2: Convert to percent: 0.4 x 100 = 40%
   Answer: 40%

Now solve: What is 35% of 240?
"""
```

</details>

---

## Summary

- Trigger phrases activate step-by-step reasoning
- Examples are more powerful than instructions alone
- Structured formats improve consistency
- Match technique complexity to task complexity

**Next:** [Step-by-Step Instructions](./03-step-by-step-instructions.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide -->
