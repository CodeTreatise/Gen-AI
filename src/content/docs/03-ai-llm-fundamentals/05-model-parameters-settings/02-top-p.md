---
title: "Top-P (Nucleus) Sampling"
---

# Top-P (Nucleus) Sampling

## Introduction

Top-P sampling, also called nucleus sampling, is an alternative way to control randomness. Instead of scaling all probabilities like temperature, top-p dynamically selects from the smallest set of tokens that add up to a probability threshold.

### What We'll Cover

- What nucleus sampling is
- How top-p works (0-1 scale)
- Top-p vs temperature
- When to use each

---

## What Is Nucleus Sampling?

Nucleus sampling selects from the "nucleus" of most probable tokens until their cumulative probability reaches the threshold P.

### How It Works

```python
def nucleus_sampling(probabilities: dict, top_p: float) -> dict:
    """
    Select tokens whose cumulative probability reaches top_p.
    """
    # Sort by probability (highest first)
    sorted_tokens = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    cumulative = 0
    nucleus = {}
    
    for token, prob in sorted_tokens:
        cumulative += prob
        nucleus[token] = prob
        if cumulative >= top_p:
            break
    
    # Renormalize probabilities
    total = sum(nucleus.values())
    return {t: p/total for t, p in nucleus.items()}

# Example
probabilities = {
    "mat": 0.50,
    "floor": 0.25,
    "table": 0.15,
    "chair": 0.05,
    "moon": 0.03,
    "elephant": 0.02,
}

# Top-P = 0.9
# Include tokens until cumulative >= 0.9
# mat (0.50) + floor (0.25) + table (0.15) = 0.90 ✓
# Only these 3 tokens are in the nucleus
```

### Visualization

```
Token Probabilities (Original):
───────────────────────────────
mat       ████████████████████████████████  50%
floor     ████████████████                  25%
table     █████████                         15%
chair     ███                                5%
moon      ██                                 3%
elephant  █                                  2%

Top-P = 0.9 (cumulative threshold):
───────────────────────────────────
Included in nucleus:
mat       ████████████████████████████████  50% ──┐
floor     ████████████████                  25% ──┼── Cumulative: 90%
table     █████████                         15% ──┘

Excluded from nucleus:
chair     ✗ (would push to 95%)
moon      ✗
elephant  ✗
```

---

## The Top-P Scale (0-1)

### Top-P Values

| Top-P Value | Effect | Behavior |
|-------------|--------|----------|
| **0.1** | Very restricted | Only 1-2 most likely tokens |
| **0.5** | Somewhat restricted | Top few tokens |
| **0.9** | Slightly restricted | Most likely tokens |
| **1.0** | No restriction | All tokens possible |

### Code Examples

```python
# Top-P = 0.1 (very deterministic)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Complete: The cat sat on the..."}],
    top_p=0.1  # Only most likely tokens
)
# Usually picks "mat" or similar

# Top-P = 0.5 (balanced)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Complete: The cat sat on the..."}],
    top_p=0.5  # Top few options
)
# Might pick mat, floor, or table

# Top-P = 0.95 (nearly full vocabulary)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Complete: The cat sat on the..."}],
    top_p=0.95  # Almost everything allowed
)
# Could pick unusual options occasionally
```

---

## Dynamic Vocabulary Selection

The key advantage of top-p is **dynamic** token selection:

### Temperature vs Top-P Behavior

```python
# With temperature, ALL tokens are always possible
# (just with adjusted probabilities)

# Scenario: Model is very confident
probabilities = {"yes": 0.99, "no": 0.01, "maybe": 0.001}

# Temperature = 1.0: All three tokens possible
# Temperature = 0.5: "yes" even more likely, but others still possible

# With top-p = 0.9, only "yes" is in nucleus
# The other tokens are ELIMINATED, not just less likely

# Scenario: Model is uncertain
probabilities = {"red": 0.30, "blue": 0.28, "green": 0.22, "yellow": 0.20}

# Top-P = 0.9: All four in nucleus (cumulative = 1.0)
# Top-P = 0.5: Only red + blue in nucleus

# Dynamic! When confident → small nucleus
#            When uncertain → large nucleus
```

### Why This Matters

```
High Confidence Scenario:
─────────────────────────
Temperature 1.0:  Still might pick wrong token (0.01% chance)
Top-P 0.9:        Wrong token eliminated entirely

Low Confidence Scenario:
────────────────────────
Temperature 1.0:  All options available
Top-P 0.9:        All good options available, rare outliers removed

Top-P adapts to the model's confidence automatically!
```

---

## Top-P vs Temperature

### Comparison Table

| Aspect | Temperature | Top-P |
|--------|-------------|-------|
| **Mechanism** | Scales all probabilities | Truncates vocabulary |
| **Low values** | More confident choices | Fewer tokens available |
| **High values** | Flatter distribution | More tokens available |
| **Vocabulary** | Always complete | Dynamically sized |
| **Interaction** | Affects sharpness | Affects options |

### When to Use Each

```python
# Use Temperature when:
# - You want to control overall creativity
# - Fine-tuning response style
# - Task has consistent confidence levels

# Use Top-P when:
# - You want to eliminate low-probability tokens
# - Model confidence varies by input
# - Avoiding occasional nonsense tokens

# Best practice: Usually adjust ONE, keep other at default
temperature_focused = {
    "temperature": 0.7,
    "top_p": 1.0  # Keep at default
}

top_p_focused = {
    "temperature": 1.0,  # Keep at default
    "top_p": 0.9
}
```

### Using Both (Not Recommended)

```python
# ⚠️ Using both simultaneously is confusing
# The effects interact in complex ways

# ❌ Hard to reason about
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.5,
    top_p=0.8  # Both modified - which matters more?
)

# ✅ Easier to understand
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
    top_p=1.0  # Only temperature matters
)
```

---

## Practical Examples

### Example 1: Code Generation

```python
# For code, we want deterministic output
# Both approaches work:

# Temperature approach
code_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a sorting function"}],
    temperature=0,
    top_p=1.0
)

# Top-P approach (less common for code)
code_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a sorting function"}],
    temperature=1.0,
    top_p=0.1  # Only most likely tokens
)
```

### Example 2: Diverse Suggestions

```python
# For brainstorming, we want variety but coherence

# Temperature approach: increases randomness
brainstorm = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Give me 5 product name ideas"}],
    temperature=1.2,  # More creative
    top_p=1.0
)

# Top-P approach: keeps coherence while allowing variety
brainstorm = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Give me 5 product name ideas"}],
    temperature=1.0,
    top_p=0.95  # Removes only outliers
)
```

### Example 3: Factual Q&A

```python
# For facts, we want high confidence only

factual = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the speed of light?"}],
    temperature=0,  # Deterministic
    top_p=1.0
)

# Or using top-p
factual = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the speed of light?"}],
    temperature=1.0,
    top_p=0.1  # Only highest confidence tokens
)
```

---

## OpenAI's Recommendation

From OpenAI's documentation:

> "We generally recommend altering either temperature or top_p but not both."

```python
# OpenAI's suggested defaults
recommended_settings = {
    # Option A: Temperature-focused
    "temperature": 0.7,  # Adjust as needed
    "top_p": 1.0,        # Leave at default
    
    # Option B: Top-P-focused
    "temperature": 1.0,  # Leave at default
    "top_p": 0.9,        # Adjust as needed
}

# Most developers use temperature because it's more intuitive
```

---

## Hands-on Exercise

### Your Task

Compare temperature and top-p effects:

```python
from openai import OpenAI

client = OpenAI()

prompt = "Complete this story in one sentence: Once upon a time, there was a dragon who..."

# Test different settings
settings = [
    {"name": "Deterministic (temp=0)", "temperature": 0, "top_p": 1.0},
    {"name": "Low Top-P (p=0.1)", "temperature": 1.0, "top_p": 0.1},
    {"name": "Balanced Temp (temp=0.7)", "temperature": 0.7, "top_p": 1.0},
    {"name": "Balanced Top-P (p=0.9)", "temperature": 1.0, "top_p": 0.9},
    {"name": "High Temp (temp=1.5)", "temperature": 1.5, "top_p": 1.0},
    {"name": "Both Modified", "temperature": 0.7, "top_p": 0.8},
]

for setting in settings:
    print(f"\n=== {setting['name']} ===")
    print(f"temp={setting['temperature']}, top_p={setting['top_p']}")
    
    # Generate 3 responses to see variation
    for i in range(3):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=setting["temperature"],
            top_p=setting["top_p"],
            max_tokens=30
        )
        print(f"  {i+1}. {response.choices[0].message.content}")
```

### Questions to Consider

- How does top_p=0.1 compare to temperature=0?
- Which setting produces more coherent variety?
- When would you prefer top-p over temperature?

---

## Summary

✅ **Top-P (nucleus sampling)** selects from smallest token set reaching probability P

✅ **Dynamic vocabulary** — adapts to model confidence

✅ **Top-P = 0.1** is very restricted; **Top-P = 1.0** allows all tokens

✅ **Use top-p OR temperature**, not both simultaneously

✅ **Temperature is more common** because it's more intuitive

✅ **Top-P is better** for eliminating low-probability outliers

**Next:** [Top-K Sampling](./03-top-k.md)

---

## Further Reading

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Original nucleus sampling paper
- [Hugging Face Sampling Guide](https://huggingface.co/blog/how-to-generate) — Comprehensive overview

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Temperature](./01-temperature.md) | [Model Parameters](./00-model-parameters-settings.md) | [Top-K Sampling](./03-top-k.md) |

