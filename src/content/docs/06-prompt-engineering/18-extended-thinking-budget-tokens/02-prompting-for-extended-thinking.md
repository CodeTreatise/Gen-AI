---
title: "Prompting for Extended Thinking"
---

# Prompting for Extended Thinking

## Introduction

Prompting for extended thinking differs from standard prompting. The model now has a dedicated space to reason internally, which changes how you should structure your requests. Overly prescriptive prompts can actually *hinder* performance by constraining the model's natural reasoning process.

This lesson covers prompting techniques specifically optimized for extended thinking mode.

> **🔑 Key Insight:** With extended thinking, the model's creativity in approaching problems may exceed a human's ability to prescribe the optimal thinking process. Let the model think freely.

### What We'll Cover

- High-level vs. prescriptive prompting
- When to let the model explore vs. constrain
- Multishot prompting with thinking examples
- Verification and self-checking patterns
- Long-form output generation

### Prerequisites

- [Extended Thinking Overview](./00-extended-thinking-overview.md)
- [Thinking Budget Configuration](./01-thinking-budget-configuration.md)

---

## The Extended Thinking Difference

### Standard Prompting vs. Extended Thinking

**Standard mode (chain-of-thought):**
```
"Think step by step: First identify the variables, then set up the equation, 
then solve for x, then verify your answer."
```

**Extended thinking mode:**
```
"Think deeply about this math problem. Consider multiple approaches."
```

The model now has a dedicated "thinking space"—you don't need to ask it to think step by step in the output.

---

## General Instructions First

### Start High-Level

Claude often performs better with high-level instructions rather than step-by-step prescriptive guidance:

```python
# ❌ Overly prescriptive (can hinder performance)
prompt = """
Think through this math problem step by step:
1. First, identify the variables
2. Then, set up the equation  
3. Next, solve for x
4. Then check your work
5. Finally, provide the answer
"""

# ✅ High-level (lets model explore)
prompt = """
Please think about this math problem thoroughly and in great detail.
Consider multiple approaches and show your complete reasoning.
Try different methods if your first approach doesn't work.
"""
```

### Why This Works

With extended thinking, the model:
1. Has dedicated space for internal reasoning
2. Can explore multiple approaches naturally
3. May find better solution paths than you'd prescribe
4. Will backtrack and try alternatives on its own

### When to Add More Structure

Start high-level, then add structure if needed:

```python
# Level 1: Start here
prompt_v1 = "Analyze this complex system and identify potential issues."

# Level 2: Add specifics if results are unfocused
prompt_v2 = """
Analyze this complex system and identify potential issues.
Focus particularly on:
- Performance bottlenecks
- Security vulnerabilities  
- Scalability concerns
"""

# Level 3: Add methodology if still needed
prompt_v3 = """
Analyze this complex system and identify potential issues.
Focus particularly on:
- Performance bottlenecks
- Security vulnerabilities
- Scalability concerns

Work through each component systematically, considering both 
individual issues and interactions between components.
"""
```

---

## Task-Specific Prompting

### Complex STEM Problems

```python
prompt = """
Solve this problem. Take your time to think deeply.

Problem: [complex math/science problem]

Consider multiple solution approaches before committing to one.
Verify your answer makes sense in the context of the problem.
"""
```

### Coding Tasks

```python
prompt = """
Write a function to solve this problem.

Requirements: [problem description]

Before you finish, verify your solution by:
- Walking through edge cases in your thinking
- Checking that the logic handles all requirements
- Considering potential failure modes
"""
```

### Analysis Tasks

```python
prompt = """
Analyze this situation comprehensively.

Context: [situation description]

Consider this from multiple angles. Weigh different factors.
Your analysis should account for trade-offs and uncertainty.
"""
```

### Strategic Planning

```python
prompt = """
Develop a strategy for [goal].

Constraints:
- [constraint 1]
- [constraint 2]
- [constraint 3]

Think through multiple approaches. Consider how different 
stakeholders might react. Identify risks and mitigations.
"""
```

---

## Multishot Prompting with Thinking

You can provide examples that include thinking patterns. Claude will generalize these to its extended thinking:

```python
prompt = """
I'm going to show you how to analyze a problem, then ask you to analyze a similar one.

Example Problem: Should we expand into the European market?

<thinking>
Let me consider this from multiple angles:

Market opportunity:
- EU has 450M consumers
- Growing demand for our product category
- Strong regulatory environment for quality products

Challenges:
- Different regulations per country
- Language/localization costs
- Established local competitors

Financial considerations:
- Estimated setup cost: $2-5M
- Break-even timeline: 18-24 months
- Currency risk with EUR/USD fluctuation

Conclusion after weighing factors: Moderate opportunity with manageable risk.
</thinking>

Recommendation: Proceed with a phased approach, starting with UK and Germany as test markets before broader expansion.

---

Now analyze this:
Problem: Should we acquire our main competitor?

[Additional context about the competitor and acquisition terms]
"""
```

### Using XML Tags for Thinking Examples

The `<thinking>` or `<scratchpad>` tags in examples help Claude understand the pattern:

```python
prompt = """
Solve these optimization problems using constraint analysis.

Example:
Problem: Minimize cost with quality >= 8

<thinking>
Objective: minimize cost
Constraint: quality >= 8

Let me map the feasible region...
[detailed working]
</thinking>

Answer: Optimal solution is x=5, y=3 with cost=$45

Now solve:
Problem: [new optimization problem]
"""
```

---

## Verification and Self-Checking

### Ask for Verification

Claude can check its own work during extended thinking:

```python
prompt = """
Write a function to calculate factorial.

Before you finish, please verify your solution with test cases for:
- n=0 (edge case)
- n=1 (base case)
- n=5 (normal case)
- n=10 (larger number)

Fix any issues you find before presenting your final answer.
"""
```

### Error Recovery Patterns

```python
prompt = """
Solve this problem.

If your first approach doesn't seem to be working:
1. Pause and reconsider the problem statement
2. Try a different method
3. Check for simplifying assumptions you might have missed

Don't give up on the first difficulty - explore alternatives.
"""
```

### Consistency Checking

```python
prompt = """
Calculate the expected revenue for Q4.

After you arrive at a number:
1. Sanity check: Does this number make sense given Q3 revenue?
2. Verify: Work backward from your answer to confirm the inputs
3. Consider: What would need to be true for this to be wrong?
"""
```

---

## Long-Form Output Generation

### Detailed Content Generation

For long outputs, be explicit about detail level:

```python
prompt = """
Create an extremely detailed analysis of [topic].

Structure your response with:
- Executive summary (200 words)
- Detailed findings (organized by theme)
- Supporting evidence for each finding
- Implications and recommendations
- Appendix with raw data interpretation

Target length: 3,000-5,000 words.
"""
```

### Outline-First Approach

For very long outputs (20,000+ words), request an outline first:

```python
prompt = """
Write a comprehensive guide to [topic].

First, create a detailed outline with:
- Section titles
- Subsection titles
- Estimated word count for each section
- Key points to cover

Then write the full guide, tracking your progress against the outline.
"""
```

### Incremental Generation

```python
# For extremely long content, generate in sections
section_prompt = """
This is section {n} of a {total}-part document about [topic].

Previous sections covered: {summary_of_previous}

Write section {n}: {section_title}

Requirements:
- Length: approximately {word_count} words
- Connect naturally to previous content
- Set up the next section on {next_section_topic}
"""
```

---

## What NOT to Do

### ❌ Don't Over-Constrain

```python
# Bad: Too prescriptive
prompt = """
Follow EXACTLY these steps:
1. Read the first paragraph
2. Identify the main noun
3. Count the adjectives
4. Calculate the ratio
5. Report the result

Do not deviate from this procedure.
"""
```

### ❌ Don't Ask for Thinking in Output

```python
# Unnecessary with extended thinking
prompt = """
Think out loud as you solve this problem.
Show all your work in your response.
Explain every step of your reasoning.
"""

# With extended thinking, the model already thinks internally.
# Instead:
prompt = """
Solve this problem thoroughly. 
Provide your final answer with a brief explanation of your approach.
"""
```

### ❌ Don't Repeat Chain-of-Thought Instructions

```python
# Redundant with extended thinking
prompt = """
Think step by step.
Take a deep breath.
Reason carefully.
Consider the problem from multiple angles.
Work through each part systematically.
"""

# Extended thinking does this automatically.
# Simpler is better:
prompt = """
Solve this problem carefully. 
Consider multiple approaches before settling on your answer.
"""
```

---

## Language Considerations

> **Important:** Extended thinking performs best in English, even if the final output is in another language.

```python
# For non-English outputs
prompt = """
[Instructions in English]

Think through this problem (you may think in English internally).

Provide your final answer in French.
"""
```

---

## Debugging Thinking

### Review Thinking Output

When results aren't as expected, review the thinking:

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": prompt}]
)

# Inspect thinking
for block in response.content:
    if block.type == "thinking":
        print("THINKING:")
        print(block.thinking)
        print("\n---\n")
    elif block.type == "text":
        print("RESPONSE:")
        print(block.text)
```

### Common Issues and Fixes

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Model goes in circles | Budget too low | Increase `budget_tokens` |
| Missing edge cases | Not asked to check | Add verification instructions |
| Unfocused thinking | Prompt too open | Add specific focus areas |
| Repetitive output | Over-constrained | Remove prescriptive steps |
| Wrong approach | Locked in early | Ask to consider alternatives |

---

## Best Practices Summary

| ✅ Do | ❌ Don't |
|-------|---------|
| Start with high-level instructions | Over-specify every step |
| Ask for verification | Assume first attempt is correct |
| Let model explore approaches | Force a single methodology |
| Provide context and constraints | Give step-by-step procedures |
| Review thinking for debugging | Ignore thinking output |
| Use examples with `<thinking>` tags | Pass thinking back as input |

---

## Hands-on Exercise

### Your Task

Transform this overly-prescriptive prompt into an effective extended thinking prompt:

**Original (too prescriptive):**
```
Think through this coding problem step by step:
1. First, parse the requirements
2. Then identify the data structures needed
3. Next, write pseudocode
4. Then implement the function
5. Then add error handling
6. Then write test cases
7. Finally, optimize if needed

Write code for a function that finds the longest common subsequence of two strings.
```

<details>
<summary>✅ Solution (click to expand)</summary>

**Improved prompt:**
```
Write a function that finds the longest common subsequence of two strings.

Think deeply about this problem. Consider different algorithmic approaches
and choose the most appropriate one for this task.

Before finalizing your solution:
- Verify correctness with a few test cases
- Consider edge cases (empty strings, identical strings, no common subsequence)
- Note the time and space complexity of your approach

Provide clean, well-documented code with a brief explanation of your approach.
```

**Why this works better:**
1. States the goal clearly without prescribing steps
2. Encourages exploration of approaches
3. Includes verification without micro-managing
4. Requests complexity analysis without dictating format
5. Lets the model's extended thinking handle the detailed reasoning

</details>

---

## Summary

✅ **Start high-level** and add structure only as needed
✅ **Let the model explore** different approaches
✅ **Add verification requests** for better accuracy
✅ **Use thinking examples** with XML tags in multishot prompts
✅ **Review thinking output** to debug and improve prompts
✅ **Avoid chain-of-thought instructions**—extended thinking does this internally

**Next:** [Thinking with Tools and Streaming](./03-thinking-with-tools-streaming.md)

---

## Further Reading

- [Anthropic Extended Thinking Tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips)
- [OpenAI Reasoning Prompting Guide](https://platform.openai.com/docs/guides/reasoning-best-practices#how-to-prompt-reasoning-models-effectively)
- [Gemini Prompting Best Practices](https://ai.google.dev/gemini-api/docs/prompting-strategies)

---

<!-- 
Sources Consulted:
- Anthropic Extended Thinking Tips: High-level vs prescriptive, multishot patterns
- OpenAI Reasoning Best Practices: Simple prompts, avoid chain-of-thought
- Anthropic: Verification patterns, language considerations
-->
