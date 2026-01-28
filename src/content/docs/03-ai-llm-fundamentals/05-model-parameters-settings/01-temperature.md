---
title: "Temperature: Controlling Randomness"
---

# Temperature: Controlling Randomness

## Introduction

Temperature is the most important parameter for controlling LLM output. It determines how "creative" or "random" the model's responses are—from completely deterministic to highly varied.

### What We'll Cover

- What temperature means technically
- Temperature scale (0-2)
- Choosing temperature for different tasks
- Practical examples and trade-offs

---

## What Temperature Means

Temperature controls the randomness of token selection during text generation.

### The Technical Explanation

```python
# Simplified view of how temperature works

def apply_temperature(logits: list, temperature: float) -> list:
    """
    Temperature scales the logits before softmax.
    Lower temperature = sharper distribution (more confident)
    Higher temperature = flatter distribution (more random)
    """
    if temperature == 0:
        # Deterministic: always pick highest probability
        return [1 if i == logits.index(max(logits)) else 0 for i in range(len(logits))]
    
    # Scale logits by temperature
    scaled = [l / temperature for l in logits]
    
    # Apply softmax
    exp_scaled = [math.exp(l) for l in scaled]
    total = sum(exp_scaled)
    probabilities = [e / total for e in exp_scaled]
    
    return probabilities

# Example: Next token probabilities for "The cat sat on the..."
logits = {"mat": 3.5, "floor": 2.8, "table": 2.1, "moon": 0.5}

# Temperature = 0.5 (sharper)
# mat: 85%, floor: 12%, table: 3%, moon: 0%

# Temperature = 1.0 (balanced)
# mat: 60%, floor: 25%, table: 12%, moon: 3%

# Temperature = 2.0 (flatter)
# mat: 35%, floor: 28%, table: 22%, moon: 15%
```

### Visualization

```
Token Probability Distribution:
────────────────────────────────

Temperature = 0 (Deterministic)
┌─────────────────────────────────────┐
│ mat   ████████████████████████ 100% │
│ floor                            0% │
│ table                            0% │
│ moon                             0% │
└─────────────────────────────────────┘

Temperature = 0.5 (Conservative)
┌─────────────────────────────────────┐
│ mat   ████████████████████     85%  │
│ floor ███                      12%  │
│ table █                         3%  │
│ moon                            0%  │
└─────────────────────────────────────┘

Temperature = 1.0 (Balanced)
┌─────────────────────────────────────┐
│ mat   ██████████████           60%  │
│ floor ██████                   25%  │
│ table ███                      12%  │
│ moon  █                         3%  │
└─────────────────────────────────────┘

Temperature = 2.0 (Creative)
┌─────────────────────────────────────┐
│ mat   ████████                 35%  │
│ floor ███████                  28%  │
│ table █████                    22%  │
│ moon  ████                     15%  │
└─────────────────────────────────────┘
```

---

## Temperature Scale (0-2)

### Temperature = 0: Deterministic

```python
# Temperature 0 always picks the highest probability token
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0
)

# Same prompt, same model = same response every time
# "2+2 equals 4."
# "2+2 equals 4."
# "2+2 equals 4."
```

**Best for:**
- Factual questions
- Code generation
- Data extraction
- Consistent outputs needed

### Temperature = 0.3-0.7: Conservative

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Summarize this article..."}],
    temperature=0.5
)

# Slightly varied but mostly consistent
# Good balance of reliability and naturalness
```

**Best for:**
- Summarization
- Translation
- Customer service
- Most production applications

### Temperature = 1.0: Balanced (Default)

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story about a robot."}],
    temperature=1.0
)

# Good variety, natural-sounding
# Default for general use
```

**Best for:**
- Conversational AI
- Content generation
- General purpose
- When variety is acceptable

### Temperature = 1.5-2.0: Creative

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem about technology."}],
    temperature=1.8
)

# Highly varied, unexpected choices
# May produce unusual or surprising outputs
```

**Best for:**
- Brainstorming
- Creative writing
- Generating options
- When surprises are welcome

> **Warning:** Temperature > 2.0 often produces incoherent text. Most APIs cap at 2.0.

---

## Choosing Temperature by Task

### Task-Based Recommendations

| Task | Recommended Temperature | Reasoning |
|------|------------------------|-----------|
| **Code generation** | 0 | Needs exact, reproducible output |
| **Data extraction** | 0 | Must be consistent and accurate |
| **Math problems** | 0 | Single correct answer |
| **Factual Q&A** | 0-0.3 | Accuracy paramount |
| **Summarization** | 0.3-0.5 | Consistent but natural |
| **Translation** | 0.3-0.5 | Accurate but fluent |
| **Customer support** | 0.5-0.7 | Professional but varied |
| **General chat** | 0.7-1.0 | Natural conversation |
| **Creative writing** | 1.0-1.5 | Variety and creativity |
| **Brainstorming** | 1.5-2.0 | Maximum divergence |

### Code Example: Dynamic Temperature

```python
def get_temperature_for_task(task_type: str) -> float:
    """
    Return appropriate temperature for task type.
    """
    temperatures = {
        # Deterministic tasks
        "code": 0,
        "math": 0,
        "extraction": 0,
        "classification": 0,
        
        # Conservative tasks
        "summary": 0.3,
        "translation": 0.4,
        "factual_qa": 0.2,
        
        # Balanced tasks
        "customer_service": 0.6,
        "explanation": 0.7,
        "general_chat": 0.8,
        
        # Creative tasks
        "creative_writing": 1.2,
        "brainstorming": 1.5,
        "poetry": 1.3,
    }
    
    return temperatures.get(task_type, 0.7)  # Default: balanced

# Usage
temp = get_temperature_for_task("code")
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=temp
)
```

---

## Temperature Demonstration

### Same Prompt, Different Temperatures

```python
prompt = "Complete this sentence: The future of AI is..."

temperatures = [0, 0.5, 1.0, 1.5]

for temp in temperatures:
    print(f"\n=== Temperature {temp} ===")
    for run in range(3):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=20
        )
        print(f"  Run {run+1}: {response.choices[0].message.content}")
```

**Example Output:**
```
=== Temperature 0 ===
  Run 1: ...bright, with tremendous potential to transform industries.
  Run 2: ...bright, with tremendous potential to transform industries.
  Run 3: ...bright, with tremendous potential to transform industries.

=== Temperature 0.5 ===
  Run 1: ...bright, with innovations that will reshape society.
  Run 2: ...promising, though careful regulation will be essential.
  Run 3: ...bright, with tremendous potential to help humanity.

=== Temperature 1.0 ===
  Run 1: ...uncertain but exciting, full of both promise and peril.
  Run 2: ...being written by researchers and dreamers alike.
  Run 3: ...a tapestry of human ingenuity and machine capability.

=== Temperature 1.5 ===
  Run 1: ...dancing on the edge of consciousness and silicon dreams.
  Run 2: ...a wild garden where algorithms bloom unexpectedly.
  Run 3: ...as unpredictable as a thunderstorm in August!
```

---

## Common Mistakes

### 1. Using High Temperature for Precision Tasks

```python
# ❌ BAD: High temperature for code
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    temperature=1.5  # Will produce inconsistent, possibly incorrect code
)

# ✅ GOOD: Low temperature for code
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    temperature=0
)
```

### 2. Using Temperature 0 When Variety is Needed

```python
# ❌ BAD: Temperature 0 for creative tasks
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Give me 5 unique product name ideas"}],
    temperature=0  # Will give same ideas every time
)

# ✅ GOOD: Higher temperature for variety
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Give me 5 unique product name ideas"}],
    temperature=1.0
)
```

### 3. Thinking Higher is Always "Better"

```python
# Temperature is not a quality dial
# Higher temperature ≠ better responses
# Higher temperature = more random responses

# For most production use cases, 0.3-0.7 is optimal
```

---

## Hands-on Exercise

### Your Task

Experiment with temperature for different tasks:

```python
from openai import OpenAI

client = OpenAI()

tasks = [
    {
        "name": "Code",
        "prompt": "Write a function to calculate fibonacci numbers",
        "temperatures": [0, 0.5, 1.0]
    },
    {
        "name": "Creative",
        "prompt": "Write a haiku about programming",
        "temperatures": [0.5, 1.0, 1.5]
    },
    {
        "name": "Factual",
        "prompt": "What is the capital of France?",
        "temperatures": [0, 0.5, 1.0]
    }
]

for task in tasks:
    print(f"\n{'='*50}")
    print(f"Task: {task['name']}")
    print(f"Prompt: {task['prompt']}")
    print('='*50)
    
    for temp in task['temperatures']:
        print(f"\n--- Temperature: {temp} ---")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": task['prompt']}],
            temperature=temp,
            max_tokens=100
        )
        print(response.choices[0].message.content)
```

### Questions to Consider

- At what temperature does code generation become unreliable?
- How does creative output differ at temperature 0.5 vs 1.5?
- Why might you want some randomness even for factual tasks?

---

## Summary

✅ **Temperature controls randomness** in token selection (0-2 scale)

✅ **Temperature 0** = deterministic, same output every time

✅ **Temperature 1** = balanced, natural variation

✅ **Temperature 2** = maximum creativity (often too random)

✅ **Match temperature to task**: low for code/facts, high for creativity

✅ **0.3-0.7 is the sweet spot** for most production applications

**Next:** [Top-P Sampling](./02-top-p.md)

---

## Further Reading

- [OpenAI Temperature Guide](https://platform.openai.com/docs/guides/text-generation) — Official documentation
- [Understanding Sampling](https://huggingface.co/blog/how-to-generate) — Hugging Face guide

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Lesson Overview](./00-model-parameters-settings.md) | [Model Parameters](./00-model-parameters-settings.md) | [Top-P Sampling](./02-top-p.md) |

