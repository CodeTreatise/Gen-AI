---
title: "Graders for Automated Testing"
---

# Graders for Automated Testing

## Introduction

Graders are the scoring engines that evaluate model outputs against your criteria. They range from simple string comparisons to sophisticated LLM-as-judge evaluations. Choosing the right grader for each evaluation criteria is essential for reliable, automated testing.

This lesson covers all grader types and when to use each.

### What We'll Cover

- String check graders (exact match, contains)
- Text similarity graders (ROUGE, BLEU, cosine)
- Score model graders (LLM-as-judge)
- Python graders (custom logic)
- Multigraders (combining multiple graders)

### Prerequisites

- [OpenAI Evals System](./01-openai-evals-system.md)
- Understanding of templating syntax

---

## Graders Overview

### Score Range

All graders return a score from **0 to 1**:
- `1.0` = Pass (fully correct)
- `0.0` = Fail (completely wrong)
- `0.0 - 1.0` = Partial credit (some graders)

### Grader Type Comparison

| Grader | Output | Best For | Limitations |
|--------|--------|----------|-------------|
| `string_check` | 0 or 1 | Exact categories, keywords | Binary only, no nuance |
| `text_similarity` | 0.0-1.0 | Summarization, paraphrase | Doesn't understand semantics |
| `score_model` | 0.0-1.0 | Open-ended, subjective | Costs tokens, slower |
| `python` | 0.0-1.0 | Custom logic, complex rules | Requires coding |
| `multi` | 0.0-1.0 | Combining multiple checks | RFT only currently |

---

## String Check Grader

### When to Use

- Classification tasks with fixed labels
- Keyword presence/absence
- Format validation
- Binary pass/fail checks

### Syntax

```python
{
    "type": "string_check",
    "name": "Descriptive name",
    "input": "{{ sample.output_text }}",
    "operation": "eq",            # eq, neq, like, ilike
    "reference": "{{ item.expected }}"
}
```

### Operations

| Operation | Behavior | Case Sensitive |
|-----------|----------|----------------|
| `eq` | Exact match | Yes |
| `neq` | Not equal | Yes |
| `like` | Contains substring | Yes |
| `ilike` | Contains substring | No |

### Examples

**Exact match (classification):**
```python
{
    "type": "string_check",
    "name": "Correct category",
    "input": "{{ sample.output_text }}",
    "operation": "eq",
    "reference": "{{ item.correct_label }}"
}
```

**Contains keyword (case-insensitive):**
```python
{
    "type": "string_check",
    "name": "Mentions product",
    "input": "{{ sample.output_text }}",
    "operation": "ilike",
    "reference": "{{ item.product_name }}"
}
```

**Must NOT contain:**
```python
{
    "type": "string_check",
    "name": "No profanity",
    "input": "{{ sample.output_text }}",
    "operation": "neq",
    "reference": "inappropriate_word"
}
```

> **Warning:** `neq` only returns 1 if the entire input doesn't equal the reference. For "doesn't contain", you'd need a Python grader.

---

## Text Similarity Grader

### When to Use

- Summarization (compare to reference summary)
- Translation quality
- Paraphrase detection
- Open-ended text with reference answer

### Syntax

```python
{
    "type": "text_similarity",
    "name": "Summary quality",
    "input": "{{ sample.output_text }}",
    "reference": "{{ item.reference_summary }}",
    "evaluation_metric": "rouge_l",    # Choose metric
    "pass_threshold": 0.4              # Minimum to pass
}
```

### Available Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `fuzzy_match` | Fuzzy string matching (rapidfuzz) | Typo-tolerant matching |
| `bleu` | BLEU score | Translation |
| `gleu` | Google BLEU | Translation (variant) |
| `meteor` | METEOR score | Translation with synonyms |
| `cosine` | Cosine similarity (embeddings) | Semantic similarity |
| `rouge_1` | ROUGE-1 (unigram overlap) | Summarization |
| `rouge_2` | ROUGE-2 (bigram overlap) | Summarization |
| `rouge_l` | ROUGE-L (longest common subsequence) | Summarization |

### Examples

**Summarization quality:**
```python
{
    "type": "text_similarity",
    "name": "ROUGE-L score",
    "input": "{{ sample.output_text }}",
    "reference": "{{ item.reference_summary }}",
    "evaluation_metric": "rouge_l",
    "pass_threshold": 0.40
}
```

**Semantic similarity:**
```python
{
    "type": "text_similarity",
    "name": "Semantic match",
    "input": "{{ sample.output_text }}",
    "reference": "{{ item.expected_answer }}",
    "evaluation_metric": "cosine",
    "pass_threshold": 0.85
}
```

> **Note:** `cosine` uses `text-embedding-3-large` embeddings and is only available for evals, not fine-tuning.

---

## Score Model Grader (LLM-as-Judge)

### When to Use

- Open-ended, subjective responses
- Quality assessment (tone, helpfulness, accuracy)
- Complex criteria that can't be codified
- Nuanced evaluation requiring reasoning

### Syntax

```python
{
    "type": "score_model",
    "name": "Quality assessment",
    "input": [
        {"role": "system", "content": "You are an expert grader..."},
        {"role": "user", "content": "Grade this response..."}
    ],
    "model": "o4-mini-2025-04-16",
    "range": [0, 1],
    "pass_threshold": 0.7,
    "sampling_params": {
        "temperature": 0.0,
        "max_completions_tokens": 1024
    }
}
```

### Supported Models

| Model | Notes |
|-------|-------|
| `gpt-4o-2024-08-06` | Fast, capable |
| `gpt-4o-mini-2024-07-18` | Cheaper, good for simple grading |
| `gpt-4.1-2025-04-14` | Latest, high quality |
| `gpt-4.1-mini-2025-04-14` | Good balance |
| `gpt-4.1-nano-2025-04-14` | Fastest, cheapest |
| `o1-2024-12-17` | Strong reasoning |
| `o3-mini-2025-01-31` | Reasoning model |
| `o3-2025-04-16` | Best reasoning |
| `o4-mini-2025-04-16` | Latest reasoning |

### Full Example

```python
{
    "type": "score_model",
    "name": "Response quality",
    "input": [
        {
            "role": "system",
            "content": """You are an expert grader evaluating customer service responses.
            
Score the response based on:
1. Accuracy - Does it correctly answer the question?
2. Helpfulness - Is it actionable and clear?
3. Tone - Is it professional and empathetic?

If all criteria are met, score 1.0.
If mostly met with minor issues, score 0.7.
If significant issues, score 0.3.
If completely wrong or unhelpful, score 0.0.

Return your score in the 'result' field."""
        },
        {
            "role": "user",
            "content": """Customer question: {{ item.question }}
            
Reference answer: {{ item.reference_answer }}

Agent response: {{ sample.output_text }}

Grade this response."""
        }
    ],
    "model": "gpt-4.1",
    "range": [0, 1],
    "pass_threshold": 0.7,
    "sampling_params": {
        "temperature": 0.0,
        "max_completions_tokens": 2048
    }
}
```

### Output Format

The score model internally uses this response format:

```json
{
  "result": 0.85,
  "steps": [
    {
      "description": "Evaluating accuracy",
      "conclusion": "Response correctly identifies the issue"
    },
    {
      "description": "Evaluating helpfulness", 
      "conclusion": "Clear next steps provided"
    }
  ]
}
```

### Best Practices

| Practice | Reason |
|----------|--------|
| Use clear rubrics | Model needs to understand scoring criteria |
| Provide examples | Show what 0, 0.5, 1 look like |
| Use temperature 0 | Reduce scoring variance |
| Use reasoning models | o-series excels at nuanced grading |
| Reference the output fields | Mention `result` and reasoning `steps` in prompt |

---

## Python Grader

### When to Use

- Complex logic that can't be expressed in other graders
- Numeric calculations
- Format validation (JSON, regex)
- Length checks
- Multi-field validation

### Syntax

```python
{
    "type": "python",
    "source": "def grade(sample, item) -> float:\n    return 1.0",
    "image_tag": "2025-05-08"
}
```

### Function Signature

```python
from typing import Any

def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    """
    Args:
        sample: Model output containing:
            - output_text: str
            - output_json: dict (if response_format used)
            - output_tools: list (if tools used)
            - choices: list
            
        item: Test data row fields
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    return 1.0
```

### Examples

**Word count check:**
```python
grading_function = """
def grade(sample, item) -> float:
    output = sample.get("output_text", "")
    word_count = len(output.split())
    max_words = item.get("max_words", 100)
    
    if word_count <= max_words:
        return 1.0
    else:
        # Partial credit for slightly over
        overage = word_count - max_words
        return max(0.0, 1.0 - (overage / max_words))
"""

grader = {
    "type": "python",
    "source": grading_function,
    "image_tag": "2025-05-08"
}
```

**JSON format validation:**
```python
grading_function = """
import json

def grade(sample, item) -> float:
    output = sample.get("output_text", "")
    
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return 0.0  # Not valid JSON
    
    # Check required fields
    required = item.get("required_fields", [])
    missing = [f for f in required if f not in data]
    
    if not missing:
        return 1.0
    else:
        return 1.0 - (len(missing) / len(required))
"""
```

**Fuzzy matching with rapidfuzz:**
```python
grading_function = """
from rapidfuzz import fuzz, utils

def grade(sample, item) -> float:
    output_text = sample["output_text"]
    reference_answer = item["reference_answer"]
    
    # Returns similarity ratio 0-100, convert to 0-1
    return fuzz.WRatio(
        output_text, 
        reference_answer, 
        processor=utils.default_process
    ) / 100.0
"""
```

### Available Packages

The Python runtime includes:

```
numpy, scipy, sympy, pandas, rapidfuzz, scikit-learn,
rouge-score, deepdiff, jsonschema, pydantic, pyyaml,
nltk, sqlparse, rdkit, scikit-bio, ast-grep-py
```

### Technical Constraints

| Limit | Value |
|-------|-------|
| Code size | < 256 KB |
| Execution time | 2 minutes |
| Memory | 2 GB |
| Disk | 1 GB |
| CPU cores | 2 (throttled above) |
| Network | None |

---

## Multigraders

> **Note:** Currently only available for Reinforcement Fine-Tuning (RFT), not standard evals.

### When to Use

- Combine multiple checks into one score
- Weight different criteria
- Complex scoring formulas

### Syntax

```python
{
    "type": "multi",
    "graders": {
        "accuracy": {
            "type": "string_check",
            "name": "Correct answer",
            "input": "{{ sample.output_json.answer }}",
            "reference": "{{ item.correct_answer }}",
            "operation": "eq"
        },
        "format": {
            "type": "text_similarity",
            "name": "Format match",
            "input": "{{ sample.output_text }}",
            "reference": "{{ item.expected_format }}",
            "evaluation_metric": "fuzzy_match",
            "pass_threshold": 0.8
        }
    },
    "calculate_output": "(accuracy * 0.7) + (format * 0.3)"
}
```

### Calculate Output Operators

| Operator/Function | Example |
|-------------------|---------|
| `+`, `-`, `*`, `/` | `(a + b) / 2` |
| `^` (power) | `a ^ 2` |
| `min`, `max` | `min(a, b)` |
| `abs`, `floor`, `ceil` | `floor(a)` |
| `sqrt`, `log`, `exp` | `sqrt(a)` |

### Example: Weighted Scoring

```python
{
    "type": "multi",
    "graders": {
        "name_match": {...},
        "email_match": {...},
        "phone_match": {...}
    },
    "calculate_output": "(name_match * 0.4) + (email_match * 0.4) + (phone_match * 0.2)"
}
```

---

## Grader Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                     What are you evaluating?                     │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    Fixed categories?    Open-ended text?     Complex logic?
            │                   │                   │
            ▼                   ▼                   ▼
      string_check        Have reference?        python
            │                   │                   
            │           ┌───────┼───────┐
            │           ▼               ▼
            │    Factual/literal?   Subjective?
            │           │               │
            │           ▼               ▼
            │    text_similarity    score_model
            │
            └──────────────────────────────────────────
```

### Quick Reference

| Task | Recommended Grader |
|------|-------------------|
| Classification | `string_check` (eq) |
| Keyword detection | `string_check` (ilike) |
| Summarization | `text_similarity` (rouge_l) |
| Translation | `text_similarity` (bleu) |
| Quality assessment | `score_model` |
| Tone evaluation | `score_model` |
| Word count | `python` |
| JSON validation | `python` |
| Regex matching | `python` |
| Multi-criteria | `multi` (RFT only) |

---

## Testing and Validating Graders

### Validate Before Use

```python
import requests

grader = {
    "type": "python",
    "source": "def grade(sample, item):\n    return 1.0"
}

response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"grader": grader}
)

print(response.json())
# {"valid": true} or {"valid": false, "error": "..."}
```

### Test with Sample Data

```python
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "grader": grader,
        "item": {"reference_answer": "Paris"},
        "model_sample": "The capital of France is Paris."
    }
)

print(response.json())
# {"score": 1.0, ...}
```

---

## Hands-on Exercise

### Your Task

Create graders for an email response generator that must:
1. Include a greeting ("Hello" or "Hi")
2. Be between 50-200 words
3. End with a signature

<details>
<summary>✅ Solution (click to expand)</summary>

```python
# Grader 1: Check for greeting
greeting_grader = {
    "type": "python",
    "source": """
def grade(sample, item) -> float:
    output = sample.get("output_text", "").lower()
    if output.startswith("hello") or output.startswith("hi"):
        return 1.0
    return 0.0
"""
}

# Grader 2: Word count check
word_count_grader = {
    "type": "python",
    "source": """
def grade(sample, item) -> float:
    output = sample.get("output_text", "")
    words = len(output.split())
    
    if 50 <= words <= 200:
        return 1.0
    elif 30 <= words < 50 or 200 < words <= 250:
        return 0.5  # Partial credit
    else:
        return 0.0
"""
}

# Grader 3: Signature check
signature_grader = {
    "type": "python",
    "source": """
def grade(sample, item) -> float:
    output = sample.get("output_text", "")
    expected_name = item.get("sender_name", "")
    
    # Check last 50 chars for signature patterns
    ending = output[-100:].lower()
    
    patterns = ["regards,", "sincerely,", "best,", "thanks,"]
    has_closing = any(p in ending for p in patterns)
    has_name = expected_name.lower() in ending
    
    if has_closing and has_name:
        return 1.0
    elif has_closing or has_name:
        return 0.5
    return 0.0
"""
}

# Use all three in testing_criteria
testing_criteria = [
    {**greeting_grader, "name": "Has greeting"},
    {**word_count_grader, "name": "Word count"},
    {**signature_grader, "name": "Has signature"}
]
```

</details>

---

## Summary

✅ **`string_check`**: Fast binary checks for exact match/contains
✅ **`text_similarity`**: ROUGE/BLEU/cosine for comparing to references
✅ **`score_model`**: LLM judges for subjective, nuanced evaluation
✅ **`python`**: Custom logic for anything else
✅ **Test graders** before running full evals
✅ **Choose based on task**: classification → string_check, quality → score_model

**Next:** [Prompt Optimizer](./03-prompt-optimizer.md)

---

## Further Reading

- [OpenAI Graders Documentation](https://platform.openai.com/docs/guides/graders)
- [OpenAI Graders API Reference](https://platform.openai.com/docs/api-reference/graders)
- [ROUGE Scoring Explained](https://aclanthology.org/W04-1013/)

---

<!-- 
Sources Consulted:
- OpenAI Graders: All grader types, syntax, examples
- OpenAI Graders: Python constraints, available packages
- OpenAI Graders: Multigrader calculate_output syntax
-->
