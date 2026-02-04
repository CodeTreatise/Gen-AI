---
title: "When to Use Chain-of-Thought"
---

# When to Use Chain-of-Thought

## Introduction

Chain-of-thought prompting is most valuable for GPT models (GPT-4o, GPT-4.1) tackling complex reasoning tasks. This lesson provides a practical decision framework for when CoT will give you the best return on investment.

> **🔑 Key Insight:** CoT shines on tasks where a human would need to work through steps—multi-step math, logical deductions, and complex analysis.

### What We'll Cover

- High-value use cases for CoT
- Task complexity assessment
- Model selection matrix
- Implementation patterns

### Prerequisites

- [When NOT to Use CoT](./05-when-not-to-use-cot.md)
- [Zero-Shot Chain-of-Thought](./04-zero-shot-chain-of-thought.md)

---

## High-Value CoT Use Cases

### Multi-Step Mathematics

```python
# Perfect for CoT - multiple operations, error-prone without steps
prompt = """A store offers a 20% discount on all items. Then, they apply 
an additional 15% member discount on the discounted price. Finally, 
8.25% sales tax is added. If an item originally costs $249.99, 
what is the final price?

Let's work through this step by step."""

# Without CoT, models often:
# - Apply discounts incorrectly (to original vs discounted price)
# - Forget the order of operations
# - Make arithmetic errors
```

**Why CoT helps:** Each step builds on the previous. Errors compound without explicit tracking.

### Word Problems

```python
prompt = """Sarah has 3 times as many apples as Tom. If Sarah gives 
Tom 10 apples, they will have the same number. How many apples 
does each person have originally?

Let's solve this step by step."""

# CoT Response:
# Let Tom's apples = T
# Sarah's apples = 3T
# After giving 10: Sarah has 3T - 10, Tom has T + 10
# They're equal: 3T - 10 = T + 10
# 2T = 20
# T = 10
# Tom has 10, Sarah has 30
```

**Why CoT helps:** Forces explicit variable definition and algebraic setup.

### Logical Reasoning

```python
prompt = """Consider these statements:
1. All programmers know logic
2. Some mathematicians are programmers  
3. No artists know logic

Can we conclude that no mathematicians are artists?

Let's reason through this carefully."""
```

**Why CoT helps:** Syllogistic reasoning requires explicit chain of deductions.

### Comparative Analysis

```python
prompt = """Compare these two database architectures for a social media 
application with 10M users:

Option A: Single PostgreSQL instance with read replicas
Option B: Sharded MongoDB cluster

Evaluate on: scalability, consistency, query flexibility, and cost.

Let's analyze each factor systematically."""
```

**Why CoT helps:** Multiple evaluation criteria need organized assessment.

---

## Task Complexity Assessment

### Complexity Indicators

| Indicator | Low Complexity | High Complexity |
|-----------|----------------|-----------------|
| Steps required | 1-2 | 3+ |
| Information sources | 1 | Multiple to synthesize |
| Intermediate results | None | Several |
| Error risk | Low | Compounding |
| Human approach | Immediate answer | Work on paper |

### Quick Assessment Questions

Ask yourself:

1. **Would I need paper?** → If yes, use CoT
2. **Are there multiple steps?** → If yes, use CoT
3. **Could intermediate errors compound?** → If yes, use CoT
4. **Does order matter?** → If yes, use CoT

---

## Model × Task Matrix

| Task Type | GPT Models | Reasoning Models |
|-----------|------------|------------------|
| Simple math | Direct prompt | Direct prompt |
| Multi-step math | Zero-shot CoT | Direct prompt |
| Logic puzzles | Few-shot CoT | Direct prompt |
| Code debugging | Step-by-step | Direct prompt |
| Word problems | Zero-shot CoT | Direct prompt |
| Factual Q&A | Direct prompt | Direct prompt |
| Creative writing | Direct prompt | Direct prompt |

### Decision Tree

```
What model am I using?
│
├── Reasoning model (o3, GPT-5) → Skip CoT, use direct prompts
│
└── GPT model (GPT-4o, GPT-4.1) 
    │
    └── Is the task complex (multi-step reasoning)?
        │
        ├── YES → Use CoT
        │   ├── Need consistent format? → Few-shot CoT
        │   └── One-off question? → Zero-shot CoT
        │
        └── NO → Direct prompt (faster, cheaper)
```

---

## Specific Domain Applications

### Financial Calculations

```python
# Compound interest with multiple contributions
prompt = """Calculate the future value of an investment with:
- Initial investment: $10,000
- Monthly contribution: $500
- Annual interest rate: 7% (compounded monthly)
- Time period: 10 years

Let's work through this step by step."""
```

### Code Debugging

```python
prompt = """Debug this function that should return the second largest number:

def second_largest(numbers):
    if len(numbers) < 2:
        return None
    largest = max(numbers)
    return max(n for n in numbers if n < largest)

What's wrong with it?

Let's trace through with an example step by step."""
```

### Data Analysis

```python
prompt = """Given this sales data pattern:
- Q1: $1.2M (up 5% from prev Q1)
- Q2: $1.5M (up 8% from prev Q2)
- Q3: $1.1M (down 3% from prev Q3)
- Q4: $2.1M (up 12% from prev Q4)

Analyze the trend and project Q1 next year.

Let's break down the analysis step by step."""
```

---

## Implementation Patterns

### Basic Zero-Shot CoT

```python
def solve_with_cot(problem: str) -> str:
    """Add CoT trigger to any problem."""
    return f"{problem}\n\nLet's think step by step."
```

### Conditional CoT

```python
def adaptive_prompt(problem: str, complexity: str) -> str:
    """Apply CoT based on task complexity."""
    
    if complexity == "high":
        return f"{problem}\n\nLet's work through this step by step."
    elif complexity == "medium":
        return f"{problem}\n\nThink carefully before answering."
    else:
        return problem
```

### Domain-Specific CoT

```python
def math_cot(problem: str) -> str:
    """CoT template for math problems."""
    return f"""Solve the following math problem:

{problem}

Show your work:
1. Identify what we're solving for
2. List known values
3. Apply relevant formulas
4. Calculate step by step
5. Verify the answer makes sense"""
```

### Structured Output CoT

```python
def structured_cot(problem: str) -> str:
    """CoT with explicit output structure."""
    return f"""{problem}

Work through this step by step, then provide your answer in this format:

STEPS:
- Step 1: [description]
- Step 2: [description]
...

FINAL ANSWER: [answer]"""
```

---

## Measuring CoT Effectiveness

### A/B Testing Approach

```python
import random

def test_cot_effectiveness(problems: list, model: str):
    """Compare direct prompts vs CoT on accuracy."""
    
    results = {"direct": 0, "cot": 0}
    
    for problem, expected in problems:
        # Randomly assign to test group
        use_cot = random.choice([True, False])
        
        if use_cot:
            prompt = f"{problem}\n\nLet's think step by step."
            answer = call_model(prompt, model)
            if verify_answer(answer, expected):
                results["cot"] += 1
        else:
            answer = call_model(problem, model)
            if verify_answer(answer, expected):
                results["direct"] += 1
    
    return results
```

### Metrics to Track

| Metric | How to Measure |
|--------|----------------|
| Accuracy | % correct answers |
| Latency | Response time (ms) |
| Token usage | Input + output tokens |
| Cost per correct answer | $/correct response |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Match technique to model | GPT models → CoT, Reasoning models → direct |
| Assess complexity first | Use mental checklist before prompting |
| Start with zero-shot | Add few-shot only if needed |
| Always verify results | CoT improves but doesn't guarantee accuracy |
| Monitor costs | Track token usage vs accuracy gains |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Using CoT on simple tasks | Assess complexity first |
| Using CoT with reasoning models | Check model type |
| Expecting perfection | CoT improves odds, doesn't guarantee |
| Ignoring latency costs | Consider response time requirements |

---

## Hands-on Exercise

### Your Task

Classify each problem and decide: CoT or Direct?

1. "What year did World War II end?"
2. "If a train travels 120 km in 2 hours, then 180 km in 2.5 hours, what's the average speed for the entire journey?"
3. "Translate 'Good morning' to Spanish."
4. "A rectangle's length is twice its width. If the perimeter is 36 cm, find the area."
5. "What's the capital of Japan?"

<details>
<summary>Answers</summary>

1. **Direct** - Simple factual lookup
2. **CoT** - Multi-step: total distance ÷ total time, not average of speeds
3. **Direct** - Simple translation
4. **CoT** - Need to set up equation, solve for dimensions, then calculate area
5. **Direct** - Simple factual lookup

Key insight: Problems 2 and 4 have a common trap (averaging speeds incorrectly, or confusing perimeter/area formulas) that CoT helps avoid.

</details>

---

## Summary

- Use CoT for multi-step reasoning with GPT models
- High-value tasks: math, logic, analysis, debugging
- Assess complexity: "Would I need paper?"
- Match technique to model type
- Zero-shot for quick tests, few-shot for production

**Next:** [Verification and Self-Correction](./07-verification-self-correction.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide, CoT research -->
