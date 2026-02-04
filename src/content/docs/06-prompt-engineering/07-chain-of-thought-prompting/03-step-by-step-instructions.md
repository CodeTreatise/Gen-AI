---
title: "Step-by-Step Instructions"
---

# Step-by-Step Instructions

## Introduction

Breaking down complex tasks into explicit steps guides the model through your reasoning process. Instead of asking for an end result, you specify intermediate checkpoints that ensure accuracy and provide visibility into the model's work.

### What We'll Cover

- Breaking down complex tasks
- Intermediate checkpoints
- Self-verification steps
- Final answer extraction

### Prerequisites

- [Eliciting Reasoning Steps](./02-eliciting-reasoning-steps.md)

---

## Breaking Down Complex Tasks

### The Decomposition Principle

Complex tasks fail when models try to solve everything at once. Decomposition forces sequential reasoning:

```python
# Bad: One big ask
prompt = "Analyze this sales data and create a marketing strategy."

# Good: Step-by-step decomposition
prompt = """Analyze this sales data and create a marketing strategy.

Follow these steps:

Step 1: DATA SUMMARY
- What are the total sales figures?
- What's the time period covered?
- Which products/categories are included?

Step 2: TREND ANALYSIS
- Are sales increasing or decreasing?
- What's the growth rate?
- Are there seasonal patterns?

Step 3: TOP PERFORMERS
- Which products sell best?
- Which customer segments are most valuable?
- What channels drive the most revenue?

Step 4: PROBLEM AREAS
- Where are sales declining?
- Which products underperform?
- What gaps exist in the data?

Step 5: MARKETING STRATEGY
Based on steps 1-4, recommend:
- Target audience priorities
- Product focus areas
- Channel allocation
- Timing considerations
"""
```

### Task Decomposition Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| Sequential | Order matters | "First read, then summarize, then critique" |
| Parallel | Independent sub-tasks | "Analyze pros AND cons separately" |
| Hierarchical | Nested complexity | "For each category, list items, then evaluate" |
| Conditional | Branching logic | "If X, do A. Otherwise, do B." |

---

## Intermediate Checkpoints

### Why Checkpoints Matter

Checkpoints:
- Make progress visible
- Catch errors early
- Provide restart points
- Enable verification

### Checkpoint Examples

```python
prompt = """Debug this function by following these checkpoints:

def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

CHECKPOINT 1: Trace normal execution
- Input: [1, 2, 3, 4, 5]
- Expected output: 3.0
- Does it work? [Yes/No and why]

CHECKPOINT 2: Test edge cases
- Empty list []
- Single element [42]
- Report behavior for each

CHECKPOINT 3: Identify issues
- List any bugs found
- Explain why each is a problem

CHECKPOINT 4: Propose fixes
- For each bug, show the fix
- Explain why the fix works

CHECKPOINT 5: Final solution
- Show the corrected function
- Verify it passes all checkpoints
"""
```

### Checkpoint Verification Pattern

```python
prompt = """Solve this problem with verification at each step:

Problem: A store offers 20% off, then an additional 10% off 
the reduced price. What's the total discount on a $100 item?

Step 1: Calculate first discount
- Show calculation
- VERIFY: Is the result reasonable?

Step 2: Calculate second discount
- Apply to result from Step 1
- VERIFY: Is this applied to the REDUCED price?

Step 3: Calculate total discount
- Compare final price to original
- VERIFY: Is total discount less than 30%? (Why?)

Step 4: State final answer
- Total discount percentage
- Final price
"""
```

---

## Self-Verification Steps

### Built-in Verification

Ask the model to check its own work:

```python
prompt = """Calculate the monthly payment for a $300,000 mortgage 
at 6.5% annual interest over 30 years.

Steps:
1. Identify the formula needed
2. Convert annual rate to monthly rate
3. Calculate total number of payments
4. Apply the formula
5. State the monthly payment

SELF-VERIFICATION:
After calculating, verify your answer by:
- Checking that total payments exceed the principal
- Confirming the interest portion makes sense
- Comparing to typical mortgage payments for this amount
"""
```

### Verification Strategies

| Strategy | Implementation | Best For |
|----------|----------------|----------|
| Sanity check | "Does this answer make sense?" | Magnitude errors |
| Unit check | "Are the units correct?" | Conversion errors |
| Reverse verify | "Work backwards from answer" | Calculation errors |
| Boundary check | "Test with extreme values" | Logic errors |
| Comparison | "Compare to known values" | Reasonableness |

### Example: Multi-Check Verification

```python
prompt = """A car travels 240 miles using 8 gallons of gas. 
How far can it travel on 15 gallons?

Solve step by step, then verify:

SOLUTION:
[Your step-by-step work here]

VERIFICATION CHECKLIST:
[ ] Units check: Is the answer in miles?
[ ] Proportionality: More gas = more miles?
[ ] Sanity check: Is 400-500 miles reasonable for 15 gallons?
[ ] Cross-check: Does (answer/15) equal (240/8)?

If any check fails, revise your answer.
"""
```

---

## Final Answer Extraction

### The Extraction Problem

After reasoning, models sometimes bury the answer in text. Force explicit extraction:

```python
# Problem: Answer lost in reasoning
response = """Let me think through this... First I'll calculate...
Then considering... So that gives us... Which means...
approximately 42.7 when we factor in..."""

# Solution: Explicit extraction
prompt = """[Your reasoning task here]

After completing your analysis:
1. Show all your reasoning steps
2. Then clearly state:

FINAL ANSWER: [Your answer here]

The final answer must be a single, clear statement.
"""
```

### Extraction Patterns

```python
# Pattern 1: Boxed answer
prompt = """Solve and put your final answer in a box.

SOLUTION:
[work here]

FINAL ANSWER:
┌─────────────────┐
│                 │
└─────────────────┘
"""

# Pattern 2: Structured output
prompt = """Analyze and respond with:

REASONING:
[Your step-by-step analysis]

CONFIDENCE: [High/Medium/Low]

ANSWER: [Single clear answer]

ALTERNATIVES: [If applicable]
"""

# Pattern 3: JSON extraction
prompt = """Solve the problem, then format your answer as:

{
  "reasoning": "brief summary of steps",
  "answer": "final answer",
  "units": "applicable units"
}
"""
```

---

## Complete Step-by-Step Template

```python
step_by_step_template = """You are a careful problem solver who shows all work.

PROBLEM:
{problem}

INSTRUCTIONS:
Complete each step before moving to the next.

STEP 1: UNDERSTAND THE PROBLEM
- What is being asked?
- What information is given?
- What assumptions are needed?

STEP 2: PLAN YOUR APPROACH
- What method will you use?
- What formulas or rules apply?
- What's the order of operations?

STEP 3: EXECUTE THE PLAN
- Show each calculation or reasoning step
- Label intermediate results
- Keep track of units

STEP 4: VERIFY YOUR WORK
- Does the answer make sense?
- Check with estimation
- Verify units are correct

STEP 5: STATE YOUR ANSWER
- Clear, final statement
- Include appropriate units
- Note any assumptions made

---

FINAL ANSWER: [State clearly here]
"""
```

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Number your steps | Creates clear sequence |
| Use action verbs | "Calculate", "Identify", "Compare" |
| Include verification | Catches errors before output |
| Extract answer explicitly | Prevents buried answers |
| Match step granularity to complexity | Too many steps = confusion |

---

## Common Pitfalls

| Mistake | Solution |
|---------|----------|
| Steps too vague | Be specific: "Calculate X" not "Figure out" |
| No verification | Always include sanity checks |
| Answer buried in text | Use explicit extraction patterns |
| Skipping steps | Make each step required |
| Over-decomposition | 5-7 steps usually sufficient |

---

## Hands-on Exercise

### Your Task

Create a step-by-step prompt for a multi-part word problem.

### Problem

A company has 3 departments:
- Sales: 45 employees earning $55,000 average
- Engineering: 30 employees earning $85,000 average
- Marketing: 25 employees earning $62,000 average

Find: Total payroll, average salary across all employees, and which department has the highest total payroll.

<details>
<summary>Solution</summary>

```python
prompt = """Solve this problem step by step.

GIVEN:
- Sales: 45 employees at $55,000 average
- Engineering: 30 employees at $85,000 average
- Marketing: 25 employees at $62,000 average

STEP 1: CALCULATE EACH DEPARTMENT'S PAYROLL
- Sales payroll = 45 x $55,000 = ?
- Engineering payroll = 30 x $85,000 = ?
- Marketing payroll = 25 x $62,000 = ?

STEP 2: CALCULATE TOTAL PAYROLL
- Sum of all department payrolls

STEP 3: CALCULATE TOTAL EMPLOYEES
- Sum of all department employees

STEP 4: CALCULATE AVERAGE SALARY
- Total payroll / Total employees

STEP 5: IDENTIFY HIGHEST PAYROLL DEPARTMENT
- Compare the three department payrolls

VERIFICATION:
- Is total payroll in millions? (reasonable for ~100 employees)
- Is average salary between $55K and $85K? (must be in range)
- Does highest make sense given employee count and salary?

FINAL ANSWERS:
- Total Payroll: $______
- Average Salary: $______
- Highest Payroll Department: ______
"""
```

</details>

---

## Summary

- Decompose complex tasks into numbered steps
- Include intermediate checkpoints for verification
- Add self-verification before final answer
- Use explicit extraction patterns
- Match step granularity to task complexity

**Next:** [Zero-Shot Chain-of-Thought](./04-zero-shot-chain-of-thought.md)

---

<!-- Sources: OpenAI Prompt Engineering Guide -->
