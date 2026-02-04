---
title: "Prompt Optimizer"
---

# Prompt Optimizer

## Introduction

The prompt optimizer is an automated tool that improves your prompts based on evaluation data. You provide examples of good and bad outputs, annotate what went wrong, and the optimizer refines your prompt to address those issues.

This lesson covers how to prepare data and use the prompt optimizer effectively.

### What We'll Cover

- Preparing datasets for optimization
- Good/Bad annotations
- Text critiques (output_feedback)
- Using grader results
- The optimization workflow
- Best practices and limitations

### Prerequisites

- [Graders for Automated Testing](./02-graders-automated-testing.md)
- OpenAI dashboard access

---

## What the Optimizer Does

### The Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│  1. Your Prompt                                              │
│     "Summarize the following article..."                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Test with Dataset                                        │
│     Run prompt → Generate outputs                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Annotate Results                                         │
│     Good/Bad ratings + Text critiques                        │
│     "Too verbose" "Missing key point" "Good tone"            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Run Graders                                              │
│     Automated scoring captures additional signals            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Optimize                                                 │
│     System analyzes annotations + grader results             │
│     Generates improved prompt                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Optimized Prompt                                         │
│     "Summarize the article in 2-3 sentences, focusing on     │
│      the main argument and key evidence. Avoid jargon."      │
└─────────────────────────────────────────────────────────────┘
```

---

## Preparing Your Dataset

### Requirements

| Requirement | Details |
|-------------|---------|
| **Minimum rows** | At least 3 with responses |
| **Annotations** | At least 1 per row (grader or human) |
| **Quality** | Diverse examples, including failures |

### Dataset Structure

In the OpenAI Datasets interface:

| Column | Purpose |
|--------|---------|
| **Input** | Your prompt/user message |
| **Output** | Model's response |
| **Good/Bad** | Binary quality rating |
| **output_feedback** | Text critique (optional but valuable) |
| **Custom columns** | Additional annotation fields |
| **Grader results** | Automated scores |

---

## Annotations

### Good/Bad Ratings

The simplest annotation—binary quality signal:

| Rating | When to Use |
|--------|-------------|
| **Good** ✅ | Output meets requirements |
| **Bad** ❌ | Output has significant issues |

> **Tip:** Be consistent. Define what "Good" means before annotating.

### Text Critiques (output_feedback)

Written feedback explaining what went wrong (or right):

```
Bad Examples:
- "Response is too long, should be under 100 words"
- "Missing the product price"
- "Tone is too casual for business context"
- "Hallucinated information about the company"

Good Examples:
- "Perfect length and includes all key points"
- "Correctly handled the edge case with missing data"
```

**Why critiques matter:**
- The optimizer uses these to understand *why* outputs failed
- More specific critiques → more targeted improvements
- "Bad" alone tells the optimizer what to avoid, not what to do

### Custom Annotation Columns

You can add domain-specific annotations:

| Example Column | Values |
|----------------|--------|
| **Accuracy** | 1-5 scale |
| **Tone** | Formal / Casual / Wrong |
| **Completeness** | Full / Partial / Missing |
| **Factual** | Correct / Has Errors |

---

## Using Grader Results

### Why Graders Help

Graders provide:
- **Scalable annotations** without manual review
- **Consistent scoring** across all examples
- **Specific failure signals** (which criteria failed)

### Workflow with Graders

1. Create graders that capture desired output properties
2. Run graders on all dataset rows
3. Grader results are automatically used by optimizer

```
Dataset Row:
├── Input: "Summarize this article..."
├── Output: "The article discusses..."
├── Grader: word_count → 0.8 (slightly over limit)
├── Grader: contains_thesis → 1.0 (pass)
├── Grader: tone_check → 0.3 (too casual)
└── Human: output_feedback = "Good content but wrong tone"
```

### Effective Grader Design

| Principle | Example |
|-----------|---------|
| **Narrow scope** | One grader per criteria |
| **Clear failure** | word_count < 100 → fail |
| **Match failures** | Build graders for observed issues |

---

## Running the Optimizer

### In the Dashboard

1. **Open Datasets** (platform.openai.com/datasets)
2. **Select your dataset** with prompts and outputs
3. **Add annotations** (Good/Bad, critiques)
4. **Run graders** if configured
5. **Click "Optimize"** in the prompt pane
6. **Review optimized prompt** in new tab

### The Optimization Process

When you click "Optimize":

1. System analyzes all annotations and grader results
2. Identifies patterns in failures
3. Generates a new prompt that addresses issues
4. Returns the optimized prompt for testing

### Iterative Optimization

One round may not be enough:

```
Round 1: Original prompt → Accuracy: 72%
         ↓ Optimize
Round 2: Optimized v1 → Accuracy: 84%
         ↓ Annotate new failures, Optimize
Round 3: Optimized v2 → Accuracy: 91%
         ↓ Fine-tune remaining edge cases
Round 4: Final prompt → Accuracy: 95%
```

---

## Example Walkthrough

### Scenario: Customer Support Response Generator

**Original Prompt:**
```
You are a customer support agent. Help the customer with their issue.
```

### Step 1: Generate Test Outputs

Run the prompt on 10+ customer questions, collect outputs.

### Step 2: Annotate

| Input | Output | Rating | Feedback |
|-------|--------|--------|----------|
| "Where's my order?" | "I don't have that info..." | Bad | Should ask for order number |
| "How do I return?" | "Here are the steps..." | Good | Clear and complete |
| "This is garbage!" | "I understand your frustration..." | Good | Good empathy |
| "Cancel my account" | "Done!" | Bad | Should confirm and explain process |
| "Price match?" | "We don't do that" | Bad | Missing policy details, too abrupt |

### Step 3: Add Graders

```python
# Grader: Asks for details when needed
{
    "type": "python",
    "source": """
def grade(sample, item):
    needs_details = item.get("needs_customer_details", False)
    output = sample["output_text"].lower()
    
    if needs_details:
        asks_for_info = any(q in output for q in [
            "order number", "could you provide", "can you share"
        ])
        return 1.0 if asks_for_info else 0.0
    return 1.0
"""
}

# Grader: Professional tone
{
    "type": "score_model",
    "input": [
        {"role": "system", "content": "Rate if this response is professional (1) or not (0)"},
        {"role": "user", "content": "{{ sample.output_text }}"}
    ],
    "model": "gpt-4.1-mini",
    "range": [0, 1]
}
```

### Step 4: Run Optimizer

After annotations and graders, click Optimize.

**Optimized Prompt:**
```
You are a customer support agent for [Company]. 

When responding to customers:
1. Acknowledge their concern with empathy
2. If you need information (order number, account details), ask politely
3. Provide clear, step-by-step instructions when applicable
4. Reference our policies when relevant (returns: 30 days, price match: no)
5. Confirm any actions taken and explain next steps
6. Maintain a professional, helpful tone throughout

If you cannot resolve the issue, explain what the customer should do next.
```

### Step 5: Test and Iterate

Run the optimized prompt, check improvement, annotate new failures, optimize again.

---

## Best Practices

### Annotation Quality

| Do | Don't |
|-----|-------|
| ✅ Specific critiques | ❌ Just "Bad" |
| ✅ Explain *why* it failed | ❌ Vague feedback |
| ✅ Include good examples | ❌ Only failures |
| ✅ Cover edge cases | ❌ Only happy path |

### Dataset Composition

| Guideline | Reason |
|-----------|--------|
| **30%+ failure cases** | Optimizer needs failure signals |
| **Diverse inputs** | Generalize improvements |
| **Include edge cases** | Stress test the prompt |
| **Balanced classes** | Avoid overfitting to one type |

### Grader Design

| Principle | Example |
|-----------|---------|
| **Narrow graders** | One check per grader |
| **Target failures** | Build graders for observed issues |
| **Combine with humans** | Graders miss nuance |

---

## Limitations

> **Important:** Always review and test optimized prompts before production.

### What Can Go Wrong

| Issue | Mitigation |
|-------|------------|
| **Overfitting** | Test on held-out examples |
| **Regression** | Run full eval after each optimization |
| **Worse on some inputs** | Check edge cases manually |
| **Too specific** | Ensure diverse training data |

### When to Stop

- Accuracy plateaus after 2-3 rounds
- Improvements are marginal
- Trade-offs appear (fixing X breaks Y)

At this point, consider:
- Fine-tuning instead of prompt optimization
- Breaking into multiple specialized prompts
- Accepting the current performance level

---

## Prompt Optimizer vs. Fine-Tuning

| Aspect | Prompt Optimizer | Fine-Tuning |
|--------|------------------|-------------|
| **Changes** | Prompt text | Model weights |
| **Speed** | Minutes | Hours |
| **Data needed** | 10-50 examples | 100+ examples |
| **Expertise** | Low | Medium |
| **Cost** | Free (besides API) | Training cost |
| **Reversible** | Instant | New training run |

**Use prompt optimizer when:**
- Quick iteration needed
- Small dataset
- Exploring what works

**Use fine-tuning when:**
- Prompt optimization plateaus
- Consistent style/format needed
- Cost optimization (smaller fine-tuned model)

---

## Hands-on Exercise

### Your Task

Design an annotation strategy for a product description generator:

1. What Good/Bad criteria would you use?
2. What text critiques would be most valuable?
3. What graders would you create?

<details>
<summary>✅ Solution (click to expand)</summary>

**Good/Bad Criteria:**
- **Good**: Includes product name, price, key features, and compelling language
- **Bad**: Missing required info, factually wrong, wrong tone, too long/short

**Valuable Text Critiques:**
- "Missing the price"
- "Didn't mention the warranty"
- "Too technical for general audience"
- "Boring—needs more compelling language"
- "Factually incorrect: said 2-year warranty but it's 1 year"
- "Perfect length and includes all key selling points"

**Graders to Create:**

```python
# 1. Contains product name
{
    "type": "string_check",
    "name": "Has product name",
    "input": "{{ sample.output_text }}",
    "operation": "ilike",
    "reference": "{{ item.product_name }}"
}

# 2. Mentions price
{
    "type": "python",
    "source": """
import re
def grade(sample, item):
    output = sample["output_text"]
    expected_price = item.get("price", "")
    
    # Check for price patterns
    has_dollar = "$" in output
    has_price = str(expected_price) in output
    
    return 1.0 if (has_dollar and has_price) else 0.0
"""
}

# 3. Word count (50-150)
{
    "type": "python",
    "source": """
def grade(sample, item):
    words = len(sample["output_text"].split())
    if 50 <= words <= 150:
        return 1.0
    elif 30 <= words < 50 or 150 < words <= 200:
        return 0.5
    return 0.0
"""
}

# 4. Compelling language (LLM judge)
{
    "type": "score_model",
    "name": "Compelling",
    "input": [
        {"role": "system", "content": "Rate if this product description is compelling and would make someone want to buy (1.0) or is boring/generic (0.0)"},
        {"role": "user", "content": "{{ sample.output_text }}"}
    ],
    "model": "gpt-4.1-mini",
    "range": [0, 1],
    "pass_threshold": 0.7
}
```

</details>

---

## Summary

✅ **Prepare datasets** with diverse examples including failures
✅ **Annotate thoroughly**: Good/Bad + specific text critiques
✅ **Use graders** for scalable, consistent signals
✅ **Iterate**: Optimize → Test → Annotate → Optimize again
✅ **Always review** optimized prompts before production
✅ **Know when to stop** and consider fine-tuning

**Next:** [Eval-Driven Development](./04-eval-driven-development.md)

---

## Further Reading

- [OpenAI Prompt Optimizer Guide](https://platform.openai.com/docs/guides/prompt-optimizer)
- [OpenAI Datasets Documentation](https://platform.openai.com/docs/guides/evaluation-getting-started)
- [OpenAI Cookbook: Building Resilient Prompts](https://cookbook.openai.com/)

---

<!-- 
Sources Consulted:
- OpenAI Prompt Optimizer: Requirements, workflow, best practices
- OpenAI Evaluation Getting Started: Dataset setup
- OpenAI Graders: Integration with optimizer
-->
