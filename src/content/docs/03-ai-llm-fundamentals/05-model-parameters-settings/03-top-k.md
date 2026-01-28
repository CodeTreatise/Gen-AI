---
title: "Top-K Sampling"
---

# Top-K Sampling

## Introduction

Top-K sampling is another method to control randomness by limiting token selection to the K most probable tokens. While simpler than top-p, it's less commonly used in modern commercial APIs.

### What We'll Cover

- How top-k works
- Static vs dynamic selection
- Comparison with top-p
- When and where top-k is available

---

## What Is Top-K Sampling?

Top-K simply keeps the K tokens with highest probability and discards the rest.

### How It Works

```python
def top_k_sampling(probabilities: dict, k: int) -> dict:
    """
    Keep only the top K tokens by probability.
    """
    # Sort by probability (highest first)
    sorted_tokens = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Keep only top K
    top_k_tokens = dict(sorted_tokens[:k])
    
    # Renormalize
    total = sum(top_k_tokens.values())
    return {t: p/total for t, p in top_k_tokens.items()}

# Example
probabilities = {
    "mat": 0.50,
    "floor": 0.25,
    "table": 0.15,
    "chair": 0.05,
    "moon": 0.03,
    "elephant": 0.02,
}

# K = 3: Keep only mat, floor, table
top_3 = top_k_sampling(probabilities, k=3)
# {"mat": 0.556, "floor": 0.278, "table": 0.167}  (renormalized)
```

### Visualization

```
Original Probabilities:
───────────────────────────────
mat       ████████████████████████████████  50%
floor     ████████████████                  25%
table     █████████                         15%
chair     ███                                5%
moon      ██                                 3%
elephant  █                                  2%

Top-K = 3:
───────────────────────────────
Included:
mat       ████████████████████████████████  56% (renormalized)
floor     ████████████████                  28%
table     ██████████                        17%

Excluded:
chair     ✗
moon      ✗
elephant  ✗
```

---

## Static vs Dynamic Selection

The key difference between top-k and top-p is that **top-k is static**:

### Top-K: Fixed Number

```python
# Top-K always takes exactly K tokens

# High confidence scenario
high_conf = {"yes": 0.95, "maybe": 0.04, "no": 0.01}
# K=3: All three included (even though "yes" is 95%!)

# Low confidence scenario  
low_conf = {"red": 0.26, "blue": 0.25, "green": 0.25, "yellow": 0.24}
# K=3: Only red, blue, green (yellow excluded despite being close)
```

### Top-P: Dynamic Number

```python
# Top-P adapts to probability distribution

# High confidence scenario
# P=0.9: Only "yes" included (0.95 > 0.9)

# Low confidence scenario
# P=0.9: All four included (need all to reach 0.9)
```

### Comparison

```
High Confidence (95% on one token):
──────────────────────────────────
Top-K=3:  Includes 3 tokens anyway (wasteful)
Top-P=0.9: Includes only 1 token (efficient)

Low Confidence (even split across many):
────────────────────────────────────────
Top-K=3:  Excludes valid options (too restrictive)
Top-P=0.9: Includes all relevant options (appropriate)
```

---

## Top-K vs Top-P

| Aspect | Top-K | Top-P |
|--------|-------|-------|
| **Selection** | Fixed number of tokens | Dynamic based on probability |
| **High confidence** | May include unnecessary tokens | Focuses on confident choices |
| **Low confidence** | May exclude valid options | Includes more options |
| **Simplicity** | Very simple to understand | Slightly more complex |
| **API availability** | Less common | More common |

### When Top-K Works Well

```python
# Top-K works well when:
# 1. Probability distribution is fairly uniform
# 2. You want predictable vocabulary size
# 3. Working with open-source models

# Example: Uniform-ish distribution
uniform_probs = {
    "option1": 0.20,
    "option2": 0.19,
    "option3": 0.18,
    "option4": 0.17,
    "option5": 0.15,
    "option6": 0.11,
}
# K=4 is reasonable here
```

### When Top-K Fails

```python
# Top-K fails when:
# 1. Probability is concentrated on few tokens
# 2. Distribution varies significantly by context

# Example: Concentrated distribution
concentrated = {"correct_answer": 0.99, "wrong": 0.01}
# K=10 would include 9 near-zero probability tokens!
```

---

## API Availability

### Where Top-K Is Available

```python
# OpenAI API: Does NOT expose top-k parameter
# Uses top-p (nucleus sampling) instead

# Anthropic Claude: Does NOT expose top-k
# Uses top_p parameter

# Google Gemini: top_k IS available
from google.generativeai import GenerativeModel

model = GenerativeModel("gemini-pro")
response = model.generate_content(
    "Hello world",
    generation_config={
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }
)

# Hugging Face / Local Models: top_k available
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
output = generator(
    "The future of AI",
    max_length=50,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

### OpenAI and Anthropic

```python
# OpenAI: top_k not available
# Must use top_p instead

response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    temperature=0.7,
    top_p=0.95,  # Use this instead of top_k
    # top_k=40,  # NOT AVAILABLE
)

# Anthropic Claude: Same situation
response = anthropic.messages.create(
    model="claude-3-sonnet",
    messages=[...],
    temperature=0.7,
    top_p=0.95,
    # top_k=40,  # NOT AVAILABLE
)
```

---

## Using Top-K with Open Source Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with different top_k values
for k in [1, 10, 50, None]:
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=20,
        top_k=k,  # Available in Hugging Face
        do_sample=True,
        temperature=0.7,
    )
    print(f"K={k}: {tokenizer.decode(output[0])}")
```

### Typical Values

```python
top_k_recommendations = {
    "k=1": "Greedy decoding (deterministic)",
    "k=10": "Very restricted, similar to low temperature",
    "k=40-50": "Common default for balanced generation",
    "k=100": "Relatively permissive",
    "k=None": "No restriction (uses all vocabulary)",
}
```

---

## Combining Parameters

When using models that support all three (temperature, top-k, top-p):

```python
# Typical combination
generation_config = {
    "temperature": 0.7,   # Scale probabilities
    "top_k": 40,          # First filter: keep top 40
    "top_p": 0.95,        # Second filter: keep nucleus to 0.95
}

# Order of operations:
# 1. Apply temperature to scale logits
# 2. Apply top-k to keep only K tokens
# 3. Apply top-p to nucleus from remaining tokens
# 4. Sample from final distribution

# This is more restrictive than any single parameter
```

### Google Gemini Example

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-pro")

response = model.generate_content(
    "Write a short poem about technology",
    generation_config={
        "temperature": 0.8,
        "top_k": 40,      # Available in Gemini
        "top_p": 0.95,
        "max_output_tokens": 100,
    }
)
print(response.text)
```

---

## Hands-on Exercise

### Your Task

If you have access to Hugging Face or Google's API, experiment with top-k:

```python
# Using Hugging Face
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "In the year 2050, robots will"

top_k_values = [1, 5, 20, 50, 100]

for k in top_k_values:
    print(f"\n=== Top-K = {k} ===")
    for i in range(3):
        output = generator(
            prompt,
            max_length=40,
            num_return_sequences=1,
            top_k=k,
            temperature=0.8,
            do_sample=True,
        )
        print(f"  {i+1}. {output[0]['generated_text']}")
```

### If Using OpenAI Only

Since OpenAI doesn't support top-k, compare how top-p achieves similar effects:

```python
# Map approximate top-k behavior to top-p
equivalents = {
    "top_k=1": "top_p=0.01",   # Nearly deterministic
    "top_k=10": "top_p=0.5",   # Restricted
    "top_k=50": "top_p=0.9",   # Balanced
    "top_k=100": "top_p=0.99", # Permissive
}

for k_desc, p_setting in equivalents.items():
    top_p_value = float(p_setting.split("=")[1])
    print(f"\nApproximate {k_desc} using {p_setting}")
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Complete: The robot said..."}],
        top_p=top_p_value,
        temperature=0.8,
        max_tokens=20
    )
    print(f"  {response.choices[0].message.content}")
```

---

## Summary

✅ **Top-K** keeps exactly K most probable tokens

✅ **Static selection** — always K tokens regardless of distribution

✅ **Simpler** than top-p but less adaptive

✅ **Not available** in OpenAI or Anthropic APIs

✅ **Available** in Google Gemini, Hugging Face, and local models

✅ **Top-P is preferred** for most use cases due to dynamic adaptation

**Next:** [Penalties](./04-penalties.md)

---

## Further Reading

- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies) — Comprehensive guide
- [Google AI Studio](https://ai.google.dev/) — Gemini with top-k support

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Top-P Sampling](./02-top-p.md) | [Model Parameters](./00-model-parameters-settings.md) | [Penalties](./04-penalties.md) |

