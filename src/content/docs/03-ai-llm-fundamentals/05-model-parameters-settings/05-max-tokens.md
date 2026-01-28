---
title: "Max Tokens: Output Length Control"
---

# Max Tokens: Output Length Control

## Introduction

The `max_tokens` parameter controls the maximum length of the model's response. It's crucial for cost management, format control, and ensuring predictable output sizes.

### What We'll Cover

- What max_tokens controls
- Impact on cost
- Truncated responses
- Task-specific recommendations

---

## What Max Tokens Controls

`max_tokens` sets the upper limit on response length, measured in tokens.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

# Short response
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    max_tokens=50  # Limit to ~50 tokens
)
# Response: ~50 tokens, stops mid-thought if necessary

# Long response
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    max_tokens=2000  # Allow detailed response
)
# Response: As long as needed, up to 2000 tokens
```

### Default Behavior

```python
# If max_tokens is not specified:
# - Model generates until natural completion OR
# - Model reaches its maximum output limit

# Model maximum outputs vary:
model_limits = {
    "gpt-4": 8192,          # tokens for output
    "gpt-4-turbo": 4096,    # default, can request more
    "gpt-4o": 16384,
    "claude-3-sonnet": 8192,
    "claude-3-opus": 4096,
}

# Natural completion example:
# Q: "What is 2+2?"
# A: "4." (naturally short, doesn't need limit)
```

---

## Impact on Cost

Output tokens typically cost 2-4x more than input tokens. Limiting output saves money.

### Cost Calculation

```python
def calculate_response_cost(
    output_tokens: int,
    model: str,
    pricing: dict
) -> float:
    """Calculate cost for output tokens"""
    rate = pricing[model]["output"]  # per million tokens
    return (output_tokens / 1_000_000) * rate

# Example pricing (per million tokens)
pricing = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5": {"input": 10.00, "output": 30.00},
}

# Cost difference
short_response = calculate_response_cost(100, "gpt-4o", pricing)   # $0.001
long_response = calculate_response_cost(4000, "gpt-4o", pricing)   # $0.04

# 40x cost difference!
```

### Cost Control Strategy

```python
def cost_aware_request(
    messages: list,
    task_type: str,
    model: str = "gpt-4o"
) -> dict:
    """
    Request with appropriate max_tokens for the task.
    """
    
    # Task-specific limits
    token_limits = {
        "yes_no": 10,
        "short_answer": 50,
        "summary": 150,
        "explanation": 500,
        "code_snippet": 300,
        "full_code": 1500,
        "article": 2000,
        "unlimited": None,
    }
    
    max_tokens = token_limits.get(task_type, 500)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    
    return {
        "content": response.choices[0].message.content,
        "tokens_used": response.usage.completion_tokens,
        "tokens_limit": max_tokens,
        "cost": calculate_response_cost(
            response.usage.completion_tokens, 
            model, 
            pricing
        )
    }
```

---

## Truncated Responses

If the model hits `max_tokens` before completing its thought:

### Detecting Truncation

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a detailed essay about AI"}],
    max_tokens=100
)

# Check finish reason
finish_reason = response.choices[0].finish_reason

if finish_reason == "length":
    print("Response was truncated due to max_tokens limit")
elif finish_reason == "stop":
    print("Response completed naturally")
elif finish_reason == "content_filter":
    print("Response was filtered for safety")
```

### Handling Truncation

```python
async def complete_response(
    messages: list,
    initial_max: int = 1000,
    max_continuations: int = 3
) -> str:
    """
    Continue generation if truncated.
    """
    
    full_response = ""
    continuation_count = 0
    current_messages = messages.copy()
    
    while continuation_count < max_continuations:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=current_messages,
            max_tokens=initial_max
        )
        
        content = response.choices[0].message.content
        full_response += content
        
        if response.choices[0].finish_reason == "stop":
            break  # Natural completion
        
        # Add partial response and ask to continue
        current_messages.append({
            "role": "assistant",
            "content": content
        })
        current_messages.append({
            "role": "user",
            "content": "Please continue from where you left off."
        })
        
        continuation_count += 1
    
    return full_response
```

---

## Estimating Appropriate Limits

### Rules of Thumb

```python
# Approximate token counts for common outputs
token_estimates = {
    "one_word": 1-2,
    "one_sentence": 15-25,
    "one_paragraph": 50-100,
    "one_page": 300-500,
    "short_article": 500-1000,
    "long_article": 1500-3000,
    "code_function": 50-200,
    "code_class": 200-500,
    "full_code_file": 500-2000,
}
```

### Calculating Limits

```python
def estimate_max_tokens(
    task_description: str,
    expected_format: str,
    safety_margin: float = 1.3
) -> int:
    """
    Estimate appropriate max_tokens for a task.
    """
    
    base_estimates = {
        "single_word": 5,
        "single_line": 20,
        "paragraph": 80,
        "multiple_paragraphs": 300,
        "short_document": 800,
        "long_document": 2000,
        "code": 500,
    }
    
    base = base_estimates.get(expected_format, 500)
    
    # Add safety margin for variability
    return int(base * safety_margin)

# Usage
max_tokens = estimate_max_tokens(
    task_description="Summarize this article",
    expected_format="paragraph"
)
print(f"Recommended max_tokens: {max_tokens}")  # ~104
```

---

## Task-Specific Recommendations

### Comprehensive Guide

| Task | Recommended max_tokens | Reasoning |
|------|----------------------|-----------|
| Yes/No questions | 5-10 | Single word needed |
| Classifications | 10-20 | Label + brief reason |
| Named entity extraction | 50-100 | List of entities |
| Short Q&A | 100-200 | Concise answer |
| Summarization | 150-300 | Key points only |
| Translation | Input × 1.5 | May expand/contract |
| Code snippets | 200-400 | Function or method |
| Full code files | 1000-2000 | Complete implementation |
| Creative writing | 500-1500 | Story/poem/essay |
| Detailed explanations | 500-1000 | Thorough but focused |
| Long-form content | 2000-4000 | Articles, reports |

### Implementation

```python
class TokenLimitManager:
    """
    Manage max_tokens based on task type.
    """
    
    LIMITS = {
        "classification": 20,
        "extraction": 100,
        "short_qa": 150,
        "summary": 250,
        "explanation": 500,
        "code": 400,
        "creative": 800,
        "detailed": 1500,
        "unlimited": 4096,
    }
    
    @classmethod
    def get_limit(cls, task_type: str, custom_limit: int = None) -> int:
        if custom_limit:
            return custom_limit
        return cls.LIMITS.get(task_type, 500)
    
    @classmethod
    def validate_response(cls, response, task_type: str) -> dict:
        """
        Check if response was appropriate length.
        """
        tokens_used = response.usage.completion_tokens
        expected_max = cls.LIMITS.get(task_type, 500)
        
        return {
            "tokens_used": tokens_used,
            "expected_max": expected_max,
            "truncated": response.choices[0].finish_reason == "length",
            "efficiency": tokens_used / expected_max if expected_max else 0,
            "recommendation": cls._get_recommendation(tokens_used, expected_max)
        }
    
    @staticmethod
    def _get_recommendation(used: int, expected: int) -> str:
        ratio = used / expected if expected else 0
        if ratio > 0.95:
            return "Consider increasing max_tokens"
        elif ratio < 0.3:
            return "Could reduce max_tokens to save cost"
        else:
            return "Limit is appropriate"

# Usage
limit = TokenLimitManager.get_limit("summary")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarize..."}],
    max_tokens=limit
)

validation = TokenLimitManager.validate_response(response, "summary")
print(validation)
```

---

## Context Window Interaction

Remember: max_tokens comes from the shared context window.

```python
def calculate_available_output(
    context_window: int,
    input_tokens: int,
    safety_buffer: int = 100
) -> int:
    """
    Calculate maximum possible output tokens.
    """
    available = context_window - input_tokens - safety_buffer
    return max(0, available)

# Example
context_window = 128_000  # GPT-4 Turbo
input_tokens = 100_000    # Large document

available = calculate_available_output(context_window, input_tokens)
print(f"Max possible output: {available:,}")  # 27,900 tokens

# If you set max_tokens > available, you'll hit context limit
```

---

## Hands-on Exercise

### Your Task

Experiment with max_tokens for different tasks:

```python
from openai import OpenAI

client = OpenAI()

# Same question, different limits
prompt = "Explain how machine learning works."

limits = [20, 50, 100, 200, 500, 1000]

for limit in limits:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=limit
    )
    
    content = response.choices[0].message.content
    tokens = response.usage.completion_tokens
    truncated = response.choices[0].finish_reason == "length"
    
    print(f"\n{'='*50}")
    print(f"max_tokens={limit}, used={tokens}, truncated={truncated}")
    print(f"Response: {content[:200]}...")

# Questions:
# 1. At what limit does the response become complete?
# 2. How does truncation affect response quality?
# 3. What's the sweet spot for this task?
```

---

## Summary

✅ **max_tokens** sets the upper limit on output length

✅ **Output tokens cost more** than input—limit for cost control

✅ **Check finish_reason** to detect truncation ("length" vs "stop")

✅ **Estimate limits** based on task type and expected format

✅ **Available output** = context_window - input_tokens

✅ **Task-specific limits** save money without sacrificing quality

**Next:** [Stop Sequences](./06-stop-sequences.md)

---

## Further Reading

- [OpenAI Token Usage](https://platform.openai.com/docs/guides/text-generation/managing-tokens) — Official guide
- [Token Counting](https://platform.openai.com/tokenizer) — Online tokenizer

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Penalties](./04-penalties.md) | [Model Parameters](./00-model-parameters-settings.md) | [Stop Sequences](./06-stop-sequences.md) |

