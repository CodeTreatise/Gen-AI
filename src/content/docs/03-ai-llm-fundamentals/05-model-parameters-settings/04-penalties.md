---
title: "Frequency and Presence Penalties"
---

# Frequency and Presence Penalties

## Introduction

Penalties help prevent repetitive text. The two main penalties—frequency and presence—discourage the model from repeating tokens, but in different ways.

### What We'll Cover

- Frequency penalty: penalize repeated tokens
- Presence penalty: encourage topic diversity  
- Scale and typical values (-2 to 2)
- Use cases and balancing

---

## Frequency Penalty

Frequency penalty reduces the probability of tokens based on how often they've appeared in the text so far.

### How It Works

```python
# Frequency penalty applies per-occurrence
# More occurrences = stronger penalty

# Without penalty:
# "The cat sat on the mat. The cat looked at the cat."
# "the" appears 3x, "cat" appears 3x

# With frequency_penalty = 0.5:
# Each occurrence of "the" reduces its probability
# Third "the" is much less likely than first "the"

# Effect formula (simplified):
# adjusted_prob = original_prob - (frequency_penalty * count)
```

### Visualization

```
Token: "cat"
───────────────────

Without penalty (frequency_penalty = 0):
  1st occurrence:  probability = 0.3
  2nd occurrence:  probability = 0.3
  3rd occurrence:  probability = 0.3

With frequency_penalty = 0.5:
  1st occurrence:  probability = 0.3
  2nd occurrence:  probability = 0.3 - 0.5 = -0.2 (less likely)
  3rd occurrence:  probability = 0.3 - 1.0 = -0.7 (much less likely)

The more a token appears, the stronger the penalty
```

### Code Example

```python
# Demonstrate frequency penalty effect
prompts = [
    "Write a paragraph about cats. Make it detailed.",
]

for penalty in [0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompts[0]}],
        frequency_penalty=penalty,
        max_tokens=100
    )
    
    text = response.choices[0].message.content
    cat_count = text.lower().count("cat")
    
    print(f"\nFrequency penalty = {penalty}")
    print(f"'cat' appears {cat_count} times")
    print(f"Text: {text[:200]}...")
```

---

## Presence Penalty

Presence penalty reduces the probability of tokens that have appeared at all, regardless of how many times.

### How It Works

```python
# Presence penalty is binary: appeared or not
# Doesn't matter if token appeared 1x or 10x

# Without penalty:
# Model might return to same topics repeatedly

# With presence_penalty = 0.5:
# Any token that has appeared is less likely
# Encourages discussing NEW topics

# Effect formula (simplified):
# adjusted_prob = original_prob - (presence_penalty * (1 if appeared else 0))
```

### Comparison with Frequency

```
Token: "cat"
───────────────────

Frequency Penalty (count-based):
  1st occurrence:  penalty = 0
  2nd occurrence:  penalty = 0.5
  3rd occurrence:  penalty = 1.0

Presence Penalty (binary):
  1st occurrence:  penalty = 0
  2nd occurrence:  penalty = 0.5  (same as 1st after)
  3rd occurrence:  penalty = 0.5  (no increase)

Frequency = increasingly strong penalty
Presence = constant penalty after first use
```

### Code Example

```python
# Demonstrate presence penalty effect
prompt = "List 10 different topics you could write about. Be diverse."

for penalty in [0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        presence_penalty=penalty,
        max_tokens=200
    )
    
    print(f"\nPresence penalty = {penalty}")
    print(response.choices[0].message.content)
```

---

## Scale and Typical Values

Both penalties use the same scale: **-2 to 2**.

### Value Guide

| Value | Effect |
|-------|--------|
| **-2.0** | Strongly encourage repetition |
| **-1.0** | Slightly encourage repetition |
| **0.0** | No effect (default) |
| **0.5** | Slight discouragement of repetition |
| **1.0** | Moderate discouragement |
| **1.5** | Strong discouragement |
| **2.0** | Very strong discouragement (may affect quality) |

### Negative Values

```python
# Negative penalties ENCOURAGE repetition
# Useful in specific scenarios

# Example: Need consistent terminology
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing consistently"}],
    frequency_penalty=-0.5,  # Encourage using same terms
)

# Use case: Technical documentation where consistency matters
```

### Typical Recommendations

```python
recommended_penalties = {
    "default": {"frequency_penalty": 0, "presence_penalty": 0},
    
    # Reduce repetition
    "general_chat": {"frequency_penalty": 0.3, "presence_penalty": 0.3},
    "creative_writing": {"frequency_penalty": 0.5, "presence_penalty": 0.5},
    
    # Encourage diversity
    "brainstorming": {"frequency_penalty": 0.3, "presence_penalty": 0.8},
    "topic_exploration": {"frequency_penalty": 0.2, "presence_penalty": 1.0},
    
    # Allow repetition
    "technical_docs": {"frequency_penalty": -0.2, "presence_penalty": 0},
    "code_generation": {"frequency_penalty": 0, "presence_penalty": 0},
}
```

---

## Use Cases

### Avoiding Repetitive Loops

```python
# Problem: Model gets stuck in repetition
# "The cat sat. The cat looked. The cat meowed. The cat sat. The cat..."

# Solution: Apply frequency penalty
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a story about a cat"}],
    frequency_penalty=0.8,  # Discourage repeating words
    max_tokens=200
)
```

### Encouraging Topic Diversity

```python
# Problem: Model keeps returning to same topics
# "Technology is important. Technology helps us. Technology..."

# Solution: Apply presence penalty
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Discuss future trends"}],
    presence_penalty=1.0,  # Encourage new topics
    max_tokens=300
)
```

### Balancing Both

```python
# Use both for different effects
def get_penalties_for_task(task: str) -> dict:
    """
    Return appropriate penalties for different tasks.
    """
    
    if task == "story":
        # Stories need word variety but theme consistency
        return {"frequency_penalty": 0.6, "presence_penalty": 0.3}
    
    elif task == "brainstorm":
        # Brainstorming needs topic diversity
        return {"frequency_penalty": 0.2, "presence_penalty": 1.2}
    
    elif task == "summary":
        # Summaries should use consistent key terms
        return {"frequency_penalty": 0.3, "presence_penalty": 0}
    
    elif task == "code":
        # Code often repeats variable names, function calls
        return {"frequency_penalty": 0, "presence_penalty": 0}
    
    else:
        # Default: slight variety
        return {"frequency_penalty": 0.3, "presence_penalty": 0.3}

# Usage
penalties = get_penalties_for_task("story")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a short story"}],
    **penalties,
    max_tokens=500
)
```

---

## Interaction with Other Parameters

### Penalties + Temperature

```python
# Low temperature + penalties can be too restrictive
# High temperature + high penalties = very diverse but possibly incoherent

# Good combinations:
combinations = {
    "focused_diverse": {
        "temperature": 0.5,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
    },
    "creative_varied": {
        "temperature": 1.0,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    },
    "deterministic": {
        "temperature": 0,
        "frequency_penalty": 0,  # No effect at temp=0
        "presence_penalty": 0,
    },
}
```

### Note on Deterministic Mode

```python
# At temperature = 0, penalties have limited effect
# The model already picks highest probability token

# Penalties only affect generation when sampling is involved
# temp=0 → no sampling → penalties don't matter much
```

---

## Debugging Repetition Issues

```python
def diagnose_repetition(text: str) -> dict:
    """
    Analyze text for repetition patterns.
    """
    words = text.lower().split()
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(words)
    
    # Find repeated words
    repeated = {word: count for word, count in word_counts.items() if count > 2}
    
    # Find repeated phrases (2-grams, 3-grams)
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    
    repeated_bigrams = {bg: count for bg, count in Counter(bigrams).items() if count > 1}
    repeated_trigrams = {tg: count for tg, count in Counter(trigrams).items() if count > 1}
    
    return {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "uniqueness_ratio": len(set(words)) / len(words) if words else 0,
        "repeated_words": repeated,
        "repeated_bigrams": repeated_bigrams,
        "repeated_trigrams": repeated_trigrams,
        "recommendation": get_recommendation(len(set(words)) / len(words) if words else 0)
    }

def get_recommendation(uniqueness: float) -> str:
    if uniqueness < 0.5:
        return "High repetition. Try frequency_penalty=1.0"
    elif uniqueness < 0.7:
        return "Moderate repetition. Try frequency_penalty=0.5"
    else:
        return "Good variety. No changes needed."

# Usage
text = "The cat sat on the mat. The cat looked at the mat. The mat was red."
diagnosis = diagnose_repetition(text)
print(diagnosis)
```

---

## Hands-on Exercise

### Your Task

Experiment with penalties to control repetition:

```python
from openai import OpenAI

client = OpenAI()

prompt = """
Write a 150-word description of a forest. 
Try to use varied vocabulary and cover different aspects.
"""

penalty_combinations = [
    {"freq": 0, "pres": 0, "desc": "No penalties"},
    {"freq": 0.5, "pres": 0, "desc": "Frequency only"},
    {"freq": 0, "pres": 0.5, "desc": "Presence only"},
    {"freq": 0.5, "pres": 0.5, "desc": "Both moderate"},
    {"freq": 1.0, "pres": 1.0, "desc": "Both high"},
]

for combo in penalty_combinations:
    print(f"\n{'='*50}")
    print(f"{combo['desc']} (freq={combo['freq']}, pres={combo['pres']})")
    print('='*50)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        frequency_penalty=combo["freq"],
        presence_penalty=combo["pres"],
        max_tokens=200
    )
    
    text = response.choices[0].message.content
    print(text)
    
    # Analyze
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words)
    print(f"\nUnique word ratio: {unique_ratio:.2%}")
```

### Questions to Consider

- How does frequency penalty affect word repetition?
- How does presence penalty affect topic variety?
- What combination works best for creative writing?

---

## Summary

✅ **Frequency penalty** reduces probability based on token count (more = stronger)

✅ **Presence penalty** reduces probability if token appeared at all (binary)

✅ **Scale is -2 to 2**, with 0 as default (no effect)

✅ **Negative values** encourage repetition (for consistency)

✅ **Use frequency** to avoid word repetition loops

✅ **Use presence** to encourage topic diversity

✅ **Both interact** with temperature and sampling

**Next:** [Max Tokens](./05-max-tokens.md)

---

## Further Reading

- [OpenAI Parameters](https://platform.openai.com/docs/api-reference/chat/create) — Official documentation
- [Controlling Repetition](https://huggingface.co/blog/how-to-generate) — Hugging Face guide

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Top-K Sampling](./03-top-k.md) | [Model Parameters](./00-model-parameters-settings.md) | [Max Tokens](./05-max-tokens.md) |

