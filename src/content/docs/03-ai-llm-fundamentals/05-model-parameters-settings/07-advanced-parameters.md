---
title: "Advanced Parameters"
---

# Advanced Parameters

## Introduction

Beyond the basic parameters, LLM APIs offer advanced controls for reproducibility, structured output, token probability analysis, and more. These parameters unlock powerful capabilities for specialized applications.

### What We'll Cover

- Seed parameter for reproducibility
- response_format for JSON mode
- logprobs for token analysis
- strict mode for function calling
- logit_bias for token manipulation

---

## Seed Parameter: Reproducible Outputs

The `seed` parameter enables reproducible generation—same inputs with same seed produce same outputs.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

# With seed: reproducible
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    seed=42,  # Fixed seed
    temperature=0.7
)

response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    seed=42,  # Same seed
    temperature=0.7
)

# response1 and response2 should be identical
print(response1.choices[0].message.content)
print(response2.choices[0].message.content)
print(f"Match: {response1.choices[0].message.content == response2.choices[0].message.content}")
```

### Use Cases

```python
# 1. Testing and debugging
def test_prompt_consistency():
    """Ensure prompt produces consistent results"""
    SEED = 12345
    
    results = []
    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Classify: 'I love this product'"}],
            seed=SEED,
            temperature=0
        )
        results.append(response.choices[0].message.content)
    
    assert all(r == results[0] for r in results), "Responses should match"

# 2. A/B testing with controlled variance
def ab_test_prompts(prompt_a: str, prompt_b: str, seeds: list):
    """Compare prompts with same random conditions"""
    results_a, results_b = [], []
    
    for seed in seeds:
        resp_a = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_a}],
            seed=seed
        )
        resp_b = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_b}],
            seed=seed
        )
        results_a.append(resp_a)
        results_b.append(resp_b)
    
    return results_a, results_b
```

### Important Notes

```python
# ⚠️ Reproducibility is "best effort"
# Same seed + same parameters + same model version → usually same output
# But not 100% guaranteed (infrastructure changes can affect)

# ✅ Check system_fingerprint for debugging
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    seed=42
)
print(f"System fingerprint: {response.system_fingerprint}")
# Different fingerprints may indicate different infrastructure
```

---

## Response Format: JSON Mode

The `response_format` parameter can enforce JSON output.

### Basic JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "List 3 programming languages with their creation year as JSON"
    }],
    response_format={"type": "json_object"}  # Force JSON output
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"languages": [{"name": "Python", "year": 1991}, ...]}
```

### With Schema (Structured Outputs)

```python
# Define expected schema
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Extract information from: 'John Smith, 35 years old, software engineer'"
    }],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"}
                },
                "required": ["name", "age", "occupation"]
            }
        }
    }
)

data = json.loads(response.choices[0].message.content)
# Guaranteed to match schema
print(data)  # {"name": "John Smith", "age": 35, "occupation": "software engineer"}
```

### Important Requirement

```python
# ⚠️ Must mention JSON in your prompt!
# The model needs to know you want JSON

# ❌ Won't work reliably
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List 3 fruits"}],
    response_format={"type": "json_object"}
)

# ✅ Works correctly
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List 3 fruits as JSON"}],
    response_format={"type": "json_object"}
)
```

---

## Logprobs: Token Probability Analysis

The `logprobs` parameter returns probability information for each generated token.

### Basic Usage

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=5,
    logprobs=True,
    top_logprobs=3  # Return top 3 alternatives for each position
)

# Access logprobs
for token_info in response.choices[0].logprobs.content:
    print(f"Token: '{token_info.token}'")
    print(f"  Logprob: {token_info.logprob:.4f}")
    print(f"  Probability: {math.exp(token_info.logprob):.2%}")
    print("  Alternatives:")
    for alt in token_info.top_logprobs:
        print(f"    '{alt.token}': {math.exp(alt.logprob):.2%}")
```

### Use Cases

```python
import math

# 1. Confidence scoring
def get_response_confidence(response) -> float:
    """Calculate confidence score from logprobs"""
    if not response.choices[0].logprobs:
        return None
    
    logprobs = [t.logprob for t in response.choices[0].logprobs.content]
    avg_logprob = sum(logprobs) / len(logprobs)
    confidence = math.exp(avg_logprob)  # Convert to probability
    
    return confidence

# 2. Detecting uncertainty
def detect_uncertainty(response, threshold: float = -1.5) -> bool:
    """Check if model was uncertain about any token"""
    if not response.choices[0].logprobs:
        return None
    
    for token_info in response.choices[0].logprobs.content:
        if token_info.logprob < threshold:
            return True  # Low probability = uncertainty
    
    return False

# 3. Finding alternative completions
def get_alternatives(response) -> dict:
    """Get alternative tokens the model considered"""
    alternatives = {}
    
    for i, token_info in enumerate(response.choices[0].logprobs.content):
        alternatives[i] = {
            "chosen": token_info.token,
            "alternatives": [
                {"token": alt.token, "prob": math.exp(alt.logprob)}
                for alt in token_info.top_logprobs
            ]
        }
    
    return alternatives
```

---

## Strict Mode: Schema Enforcement

For function calling / tools, `strict` mode guarantees the model follows your schema exactly.

### Function Calling with Strict Mode

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            },
            "strict": True  # Enforce exact schema
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

# With strict=True:
# - All required fields are guaranteed
# - Enum values are guaranteed to be valid
# - Types are guaranteed to match
```

### Without vs With Strict

```python
# Without strict=True, model might:
# - Omit required fields
# - Use wrong types
# - Make up enum values

# With strict=True:
# - Guaranteed valid output
# - May refuse if it can't comply
# - Slightly higher latency
```

---

## Logit Bias: Token Manipulation

`logit_bias` directly adjusts the probability of specific tokens.

### Basic Usage

```python
import tiktoken

# Get token IDs
enc = tiktoken.encoding_for_model("gpt-4")

# Increase probability of "Python"
python_token = enc.encode("Python")[0]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Name a programming language"}],
    logit_bias={
        str(python_token): 5  # Increase probability
    }
)
# Much more likely to say "Python"
```

### Use Cases

```python
# 1. Ban specific tokens
def ban_tokens(token_list: list) -> dict:
    """Create logit_bias to ban tokens"""
    enc = tiktoken.encoding_for_model("gpt-4")
    bias = {}
    for word in token_list:
        tokens = enc.encode(word)
        for token_id in tokens:
            bias[str(token_id)] = -100  # Effectively ban
    return bias

banned = ban_tokens(["competitor", "rival"])
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    logit_bias=banned
)

# 2. Encourage specific vocabulary
def encourage_tokens(token_list: list, boost: float = 3) -> dict:
    """Increase probability of preferred tokens"""
    enc = tiktoken.encoding_for_model("gpt-4")
    bias = {}
    for word in token_list:
        tokens = enc.encode(word)
        for token_id in tokens:
            bias[str(token_id)] = boost
    return bias

preferred = encourage_tokens(["sustainable", "eco-friendly", "green"])
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Describe modern architecture"}],
    logit_bias=preferred
)
```

### Value Scale

```python
logit_bias_scale = {
    -100: "Effectively banned (never appears)",
    -5: "Much less likely",
    -1: "Slightly less likely",
    0: "No change (default)",
    1: "Slightly more likely",
    5: "Much more likely",
    100: "Always choose (if valid)",
}
```

---

## Combining Advanced Parameters

```python
def advanced_generation(
    prompt: str,
    require_reproducibility: bool = False,
    require_json: bool = False,
    analyze_confidence: bool = False,
    banned_words: list = None
) -> dict:
    """
    Generate with advanced parameters based on requirements.
    """
    
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    if require_reproducibility:
        kwargs["seed"] = 42
        kwargs["temperature"] = 0
    
    if require_json:
        kwargs["response_format"] = {"type": "json_object"}
    
    if analyze_confidence:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 3
    
    if banned_words:
        kwargs["logit_bias"] = ban_tokens(banned_words)
    
    response = client.chat.completions.create(**kwargs)
    
    result = {
        "content": response.choices[0].message.content,
        "model": response.model,
    }
    
    if require_reproducibility:
        result["system_fingerprint"] = response.system_fingerprint
    
    if analyze_confidence:
        result["confidence"] = get_response_confidence(response)
    
    return result
```

---

## Hands-on Exercise

### Your Task

Experiment with advanced parameters:

```python
import math
from openai import OpenAI
import tiktoken

client = OpenAI()

# 1. Test reproducibility with seeds
def test_seed_reproducibility():
    results = []
    for seed in [42, 42, 123, 42]:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Generate a random number between 1 and 100"}],
            seed=seed,
            temperature=1.0
        )
        results.append(response.choices[0].message.content)
        print(f"Seed {seed}: {results[-1]}")
    
    print(f"Seeds 42 match: {results[0] == results[1] == results[3]}")

# 2. Analyze confidence with logprobs
def analyze_model_confidence(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        logprobs=True,
        top_logprobs=3
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print("\nToken-level confidence:")
    
    for token_info in response.choices[0].logprobs.content:
        prob = math.exp(token_info.logprob)
        print(f"  '{token_info.token}': {prob:.1%}")

# 3. Use logit_bias to control output
def biased_response(prompt: str, boost_word: str, ban_word: str):
    enc = tiktoken.encoding_for_model("gpt-4")
    
    bias = {}
    for token in enc.encode(boost_word):
        bias[str(token)] = 5
    for token in enc.encode(ban_word):
        bias[str(token)] = -100
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        logit_bias=bias
    )
    
    print(f"Boosted '{boost_word}', banned '{ban_word}'")
    print(f"Response: {response.choices[0].message.content}")

# Run experiments
test_seed_reproducibility()
print("\n" + "="*50 + "\n")
analyze_model_confidence("What color is the sky?")
print("\n" + "="*50 + "\n")
biased_response("Name a popular fruit", "mango", "apple")
```

---

## Summary

✅ **seed** enables reproducible outputs for testing and debugging

✅ **response_format** enforces JSON output and schema compliance

✅ **logprobs** reveals token probabilities for confidence analysis

✅ **strict mode** guarantees function call schema compliance

✅ **logit_bias** directly manipulates token probabilities

✅ **Combine parameters** for sophisticated generation control

**Next:** [Parameter Combinations](./08-parameter-combinations.md)

---

## Further Reading

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — JSON mode guide
- [Reproducible Outputs](https://platform.openai.com/docs/guides/text-generation/reproducible-outputs) — Seed parameter
- [Function Calling](https://platform.openai.com/docs/guides/function-calling) — Strict mode

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Stop Sequences](./06-stop-sequences.md) | [Model Parameters](./00-model-parameters-settings.md) | [Parameter Combinations](./08-parameter-combinations.md) |

