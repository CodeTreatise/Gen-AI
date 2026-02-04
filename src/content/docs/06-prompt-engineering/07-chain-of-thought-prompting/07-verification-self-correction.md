---
title: "Verification and Self-Correction"
---

# Verification and Self-Correction

## Introduction

Chain-of-thought prompting surfaces the model's reasoning—but that reasoning can still be wrong. Verification techniques add a crucial second pass: the model checks its own work. This catches errors that would otherwise slip through.

> **🔑 Key Insight:** Self-verification works because the model treats its own output as input, allowing a fresh evaluation with the same capabilities used to generate the answer.

### What We'll Cover

- Self-checking prompts and techniques
- Error detection patterns
- Correction strategies
- Multi-pass verification

### Prerequisites

- [Step-by-Step Instructions](./03-step-by-step-instructions.md)
- [When to Use Chain-of-Thought](./06-when-to-use-chain-of-thought.md)

---

## Basic Self-Verification

### The Simple Approach

```python
prompt = """Calculate: 847 + 368

Let's work through this step by step.
After calculating, verify your answer is correct."""
```

### Explicit Verification Request

```python
prompt = """Solve: What is 15% of 240?

Show your calculation step by step.

Then verify by:
1. Checking if 10% would be 24, so 15% should be slightly higher
2. Confirming 15% is between 10% (24) and 20% (48)"""
```

### Response Pattern

```
Step-by-step solution:
15% of 240 = 0.15 × 240 = 36

Verification:
- 10% of 240 = 24 ✓
- 20% of 240 = 48 ✓
- 36 is between 24 and 48 ✓
- Answer 36 passes verification
```

---

## Verification Techniques

### Reverse Calculation

```python
prompt = """Solve: If 3x + 7 = 22, what is x?

After finding x, verify by substituting back into the original equation."""
```

**Output:**
```
Solving:
3x + 7 = 22
3x = 15
x = 5

Verification:
3(5) + 7 = 15 + 7 = 22 ✓
The answer x = 5 is correct.
```

### Estimation Check

```python
prompt = """Calculate: 487 × 23

First estimate: 500 × 20 = 10,000 (rough range)

Now calculate precisely, then confirm the answer is close to your estimate."""
```

### Alternative Method

```python
prompt = """Find the area of a triangle with base 12 and height 8.

Method 1: Use the formula A = ½bh
Method 2: Think of it as half of a 12×8 rectangle

Both methods should give the same answer."""
```

### Unit/Dimension Check

```python
prompt = """A car travels at 60 mph for 2.5 hours. How far does it go?

Calculate the distance, then verify:
- Units: miles/hour × hours = miles ✓
- Sanity: At 60 mph, 1 hour = 60 miles, so 2.5 hours should be more than 60 miles"""
```

---

## Error Detection Patterns

### Common Error Types

| Error Type | Detection Method |
|------------|------------------|
| Arithmetic mistakes | Reverse calculation |
| Order of operations | Step-by-step trace |
| Unit errors | Dimensional analysis |
| Logic gaps | Premise checking |
| Off-by-one | Boundary testing |

### Arithmetic Error Detection

```python
prompt = """Calculate the total cost:
- 3 items at $12.99 each
- 2 items at $7.50 each  
- 8% sales tax on total

After calculating, verify:
1. Subtotal should be roughly 3×13 + 2×8 = 55
2. Tax on ~$55 should be around $4-5
3. Total should be approximately $60"""
```

### Logic Gap Detection

```python
prompt = """Determine if this argument is valid:

Premise 1: All cats are mammals
Premise 2: All mammals are animals
Premise 3: Whiskers is a cat
Conclusion: Whiskers is an animal

Trace the logical chain step by step.
Then verify no steps are skipped or assumed."""
```

---

## Self-Correction Strategies

### Iterative Refinement

```python
prompt = """Write a function to check if a string is a palindrome.

After writing, review your code for:
1. Edge cases (empty string, single character)
2. Case sensitivity (should "Racecar" match?)
3. Non-alphanumeric characters (should "A man, a plan" match?)

If any issues found, provide a corrected version."""
```

### Explicit Error Fixing

```python
prompt = """Solve this problem: Find two numbers that add to 10 and multiply to 24.

After finding your answer, verify by:
1. Checking they add to 10
2. Checking they multiply to 24

If verification fails, identify the error and try again."""
```

### Confidence Assessment

```python
prompt = """Answer this question: What is the population of Tokyo?

Then assess your confidence:
- HIGH: Recent, well-known fact
- MEDIUM: Approximate or may be outdated
- LOW: Uncertain, should verify

Explain your confidence rating."""
```

---

## Multi-Pass Verification

### Two-Pass Pattern

```python
# Pass 1: Solve
prompt_1 = """Solve: A store has a 25% off sale. An item originally costs $80.
What's the sale price? Show your work."""

response_1 = call_model(prompt_1)

# Pass 2: Verify
prompt_2 = f"""Review this solution for errors:

{response_1}

Check:
1. Is the discount calculated correctly?
2. Is the subtraction correct?
3. Does the answer make sense (should be less than $80)?

If errors found, provide the correct answer."""
```

### Adversarial Verification

```python
prompt = """Solve: If 5 workers can complete a job in 8 hours, 
how long would it take 4 workers?

After solving, try to find a flaw in your reasoning.
Consider: Did you account for work rate properly?

If you find an error, correct it."""
```

### Consensus Verification

```python
prompt = """Solve this problem three different ways, then compare answers:

Problem: What is the sum of integers from 1 to 100?

Method 1: Use the formula n(n+1)/2
Method 2: Pair numbers (1+100, 2+99, etc.)
Method 3: Calculate directly for smaller case and extrapolate

All three methods should agree."""
```

---

## Implementation Patterns

### Basic Verification Wrapper

```python
def solve_with_verification(problem: str, model: str = "gpt-4o") -> dict:
    """Solve a problem with built-in verification."""
    
    prompt = f"""{problem}

Solve step by step.

After solving, verify your answer using an appropriate check:
- Substitute back into original equation
- Estimate to confirm reasonableness  
- Use an alternative method

Format:
SOLUTION: [your step-by-step work]
VERIFICATION: [your verification check]
FINAL ANSWER: [answer]
VERIFIED: [Yes/No]"""
    
    response = call_model(prompt, model)
    
    return {
        "response": response,
        "verified": "VERIFIED: Yes" in response
    }
```

### Two-Stage Verification

```python
def solve_and_verify(problem: str, model: str = "gpt-4o") -> dict:
    """Two-stage solving with separate verification."""
    
    # Stage 1: Solve
    solution = call_model(f"{problem}\n\nSolve step by step.", model)
    
    # Stage 2: Verify
    verification_prompt = f"""Review this solution for errors:

Problem: {problem}
Solution: {solution}

Check for:
1. Calculation errors
2. Logic errors
3. Missing steps
4. Incorrect assumptions

VERDICT: [Correct/Incorrect]
If incorrect, provide CORRECTED ANSWER: [answer]"""
    
    verification = call_model(verification_prompt, model)
    
    return {
        "solution": solution,
        "verification": verification,
        "is_correct": "VERDICT: Correct" in verification
    }
```

### Retry on Failure

```python
def solve_with_retry(problem: str, max_attempts: int = 3) -> str:
    """Retry solving if verification fails."""
    
    for attempt in range(max_attempts):
        result = solve_and_verify(problem)
        
        if result["is_correct"]:
            return result["solution"]
        
        # Extract corrected answer if available
        if "CORRECTED ANSWER:" in result["verification"]:
            return result["verification"]
    
    return f"Could not verify solution after {max_attempts} attempts"
```

---

## Domain-Specific Verification

### Math Problems

```python
prompt = """Solve: Find x if 2^x = 32

After solving, verify by computing 2^(your answer) = 32"""
```

### Code Generation

```python
prompt = """Write a function to find the maximum value in a list.

After writing, trace through with these test cases:
- [3, 1, 4, 1, 5, 9] → should return 9
- [-1, -5, -2] → should return -1
- [42] → should return 42
- [] → should handle gracefully

Fix any issues discovered."""
```

### Text Analysis

```python
prompt = """Identify the main argument in this paragraph:
"[paragraph text]"

After identifying, verify by:
1. Checking if the argument is explicitly stated or implied
2. Confirming no other statement better represents the main point
3. Ensuring you haven't confused supporting evidence with the main claim"""
```

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Verification finds error but doesn't fix | Explicitly request correction |
| Same error repeated in verification | Use different method to verify |
| Verification adds token overhead | Reserve for high-stakes tasks |
| Model confirms wrong answer | Use two-pass with fresh context |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Match verification to problem type | Reverse calculation for equations, estimation for arithmetic |
| Request explicit confirmation | "State whether verification passed" |
| Use different method for verification | Catches systematic errors |
| Build verification into templates | Consistency across prompts |
| Reserve for complex tasks | Overhead isn't worth it for simple questions |

---

## Hands-on Exercise

### Your Task

Create a verification-enhanced prompt for this problem:

**Problem:** A store has 150 items. On Monday, 20% are sold. On Tuesday, 25% of the remaining items are sold. How many items are left?

**Requirements:**
1. Request step-by-step solution
2. Add at least two verification checks
3. Request explicit pass/fail for each check
4. Format for clear final answer

<details>
<summary>Solution</summary>

```python
prompt = """A store has 150 items. On Monday, 20% are sold. 
On Tuesday, 25% of the remaining items are sold. 
How many items are left?

Solve step by step:
1. Calculate items sold Monday
2. Calculate remaining after Monday
3. Calculate items sold Tuesday
4. Calculate final remaining

Then verify with these checks:

CHECK 1 (Estimation):
- 20% + 25% is less than 50%, so over half should remain
- Does your answer show more than 75 items? [PASS/FAIL]

CHECK 2 (Reverse):
- Start with your final answer
- Add back Tuesday sales (1/3 of remaining, since 25% sold = 75% left)
- Add back Monday sales (25% of that, since 20% sold = 80% left)
- Do you get 150? [PASS/FAIL]

FINAL ANSWER: [X items]
ALL CHECKS PASSED: [Yes/No]
```

</details>

---

## Summary

- Self-verification catches errors that slip through CoT
- Match verification method to problem type
- Two-pass verification provides fresh evaluation
- Build verification into prompt templates
- Reserve for complex or high-stakes tasks

**Next:** Return to [Chain-of-Thought Overview](./00-chain-of-thought-overview.md) or continue to [Lesson 8](../08-role-and-persona-prompts.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide, Self-consistency research -->
