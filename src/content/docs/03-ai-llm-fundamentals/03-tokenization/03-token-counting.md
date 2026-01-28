---
title: "Token Counting and Estimation"
---

# Token Counting and Estimation

## Introduction

Accurate token counting is essential for cost estimation, staying within context limits, and optimizing your applications. This section covers the tools and techniques for counting tokens precisely and estimating them quickly.

### What We'll Cover

- Tiktoken library for accurate counting
- Approximate counting rules
- Estimation formulas for different content
- Code-based counting for production

---

## Tiktoken Library

**Tiktoken** is OpenAI's fast tokenization library. It's the most accurate way to count tokens for OpenAI models.

### Installation

```bash
pip install tiktoken
```

### Basic Usage

```python
import tiktoken

# Get the encoding for a specific model
enc = tiktoken.encoding_for_model("gpt-4")

# Count tokens in text
text = "Hello, how are you today?"
tokens = enc.encode(text)
token_count = len(tokens)

print(f"Text: '{text}'")
print(f"Tokens: {tokens}")
print(f"Token count: {token_count}")
```

**Output:**
```
Text: 'Hello, how are you today?'
Tokens: [9906, 11, 1268, 527, 499, 3432, 30]
Token count: 7
```

### Counting for Chat Messages

```python
import tiktoken

def count_chat_tokens(messages, model="gpt-4"):
    """Count tokens in chat messages including overhead"""
    enc = tiktoken.encoding_for_model(model)
    
    # Different models have different overhead
    if model.startswith("gpt-4"):
        tokens_per_message = 3  # <|im_start|>{role}\n...
        tokens_per_name = 1
    else:
        tokens_per_message = 4
        tokens_per_name = -1
    
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(enc.encode(value))
            if key == "name":
                total += tokens_per_name
    
    total += 3  # Reply priming (<|im_start|>assistant<|im_sep|>)
    return total

# Example
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

token_count = count_chat_tokens(messages)
print(f"Total tokens: {token_count}")
```

### Full Message Counter (Production-Ready)

```python
import tiktoken
from typing import List, Dict

def num_tokens_from_messages(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4"
) -> int:
    """
    Returns the number of tokens used by a list of messages.
    Adapted from OpenAI's cookbook.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Model-specific token overhead
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]:
        tokens_per_message = 4
        tokens_per_name = -1
    elif model in ["gpt-4", "gpt-4-0613", "gpt-4-turbo"]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model.startswith("gpt-4o") or model.startswith("gpt-5"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        # Default fallback
        tokens_per_message = 3
        tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # Assistant reply priming
    return num_tokens
```

---

## Approximate Counting Rules

When you need quick estimates without running code:

### Rules of Thumb

| Content Type | Estimation Rule |
|--------------|-----------------|
| English text | 1 token ≈ 4 characters |
| English text | 1 token ≈ 0.75 words |
| Common words | 1 word ≈ 1-1.5 tokens |
| Code | 1 token ≈ 2-3 characters |
| JSON | 1 token ≈ 2-3 characters |
| Numbers | ~1 token per 2-3 digits |

### Quick Formulas

```python
def estimate_tokens_english(text: str) -> int:
    """Quick estimate for English text"""
    return len(text) // 4

def estimate_tokens_words(word_count: int) -> int:
    """Estimate from word count"""
    return int(word_count * 1.33)  # ~1.33 tokens per word

def estimate_tokens_code(code: str) -> int:
    """Estimate for code (more tokens per char)"""
    return len(code) // 3

# Examples
text = "Hello, how are you doing today?"
print(f"Character estimate: {estimate_tokens_english(text)}")
print(f"Word estimate: {estimate_tokens_words(len(text.split()))}")
```

### When to Use Estimates vs. Exact Counts

| Situation | Use |
|-----------|-----|
| Cost forecasting | Estimates OK |
| Context window management | Exact count |
| Billing verification | Exact count |
| Prompt optimization | Exact count |
| Quick planning | Estimates OK |

---

## Counting for Different Content

### Code Analysis

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

code_samples = {
    "python": '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
    "json": '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}',
    "sql": "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.total > 100"
}

for lang, code in code_samples.items():
    tokens = len(enc.encode(code))
    ratio = len(code) / tokens
    print(f"{lang}: {len(code)} chars, {tokens} tokens, ratio: {ratio:.2f}")
```

### Multilingual Content

```python
multilingual_samples = {
    "english": "The quick brown fox jumps over the lazy dog.",
    "spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
    "german": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "chinese": "敏捷的棕色狐狸跳过了懒狗。",
    "japanese": "素早い茶色の狐が怠惰な犬を飛び越えた。",
    "arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول."
}

for lang, text in multilingual_samples.items():
    tokens = len(enc.encode(text))
    ratio = len(text) / tokens
    print(f"{lang:10}: {tokens:3} tokens, {len(text):3} chars, ratio: {ratio:.2f}")
```

**Typical Output:**
```
english   :  10 tokens,  44 chars, ratio: 4.40
spanish   :  16 tokens,  54 chars, ratio: 3.38
german    :  15 tokens,  55 chars, ratio: 3.67
chinese   :  17 tokens,  14 chars, ratio: 0.82
japanese  :  25 tokens,  20 chars, ratio: 0.80
arabic    :  24 tokens,  47 chars, ratio: 1.96
```

> **Note:** Chinese and Japanese have very low char/token ratios, meaning they're much less efficient.

---

## Token Counting Utilities

### Production-Ready Utilities

```python
import tiktoken
from functools import lru_cache

@lru_cache(maxsize=4)
def get_encoding(model: str):
    """Cache encodings for performance"""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text"""
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    encoding = get_encoding(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def split_into_chunks(text: str, chunk_size: int, model: str = "gpt-4") -> list:
    """Split text into chunks of approximately chunk_size tokens"""
    encoding = get_encoding(model)
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
    
    return chunks

# Usage examples
text = "Your long document here..." * 100

# Count
total = count_tokens(text)
print(f"Total tokens: {total}")

# Truncate to fit in context
truncated = truncate_to_tokens(text, max_tokens=1000)
print(f"Truncated to: {count_tokens(truncated)} tokens")

# Split for processing
chunks = split_into_chunks(text, chunk_size=500)
print(f"Split into {len(chunks)} chunks")
```

### Token Budget Calculator

```python
def calculate_token_budget(
    context_window: int,
    system_prompt_tokens: int,
    desired_output_tokens: int,
    safety_margin: float = 0.1
) -> int:
    """
    Calculate how many tokens are available for user content.
    
    Args:
        context_window: Model's maximum context (e.g., 128000)
        system_prompt_tokens: Tokens in system prompt
        desired_output_tokens: Expected response length
        safety_margin: Buffer for overhead (default 10%)
    
    Returns:
        Available tokens for user content
    """
    total_reserved = system_prompt_tokens + desired_output_tokens
    margin = int(context_window * safety_margin)
    available = context_window - total_reserved - margin
    
    return max(0, available)

# Example for GPT-4
budget = calculate_token_budget(
    context_window=128000,
    system_prompt_tokens=500,
    desired_output_tokens=4000,
    safety_margin=0.05
)
print(f"Available for user content: {budget:,} tokens")
# Available for user content: 117,100 tokens
```

---

## JavaScript Token Counting

For web applications, use similar libraries:

```javascript
// Using tiktoken in Node.js
import { encoding_for_model } from "tiktoken";

function countTokens(text, model = "gpt-4") {
  const enc = encoding_for_model(model);
  const tokens = enc.encode(text);
  enc.free(); // Important: free the encoder
  return tokens.length;
}

// Using gpt-tokenizer (browser-compatible)
import { encode, decode } from 'gpt-tokenizer';

const tokens = encode("Hello, world!");
console.log(`Token count: ${tokens.length}`);
```

---

## Hands-on Exercise

### Your Task

Build a token analysis tool:

```python
import tiktoken

def analyze_text(text: str, model: str = "gpt-4"):
    """Complete analysis of text tokenization"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    # Basic stats
    char_count = len(text)
    word_count = len(text.split())
    token_count = len(tokens)
    
    # Ratios
    chars_per_token = char_count / token_count if token_count else 0
    words_per_token = word_count / token_count if token_count else 0
    
    # Cost estimate (GPT-4 Turbo pricing as example)
    input_cost_per_1k = 0.01  # $0.01 per 1K input tokens
    estimated_cost = (token_count / 1000) * input_cost_per_1k
    
    print(f"=== Token Analysis ===")
    print(f"Characters:      {char_count:,}")
    print(f"Words:           {word_count:,}")
    print(f"Tokens:          {token_count:,}")
    print(f"")
    print(f"Chars/Token:     {chars_per_token:.2f}")
    print(f"Words/Token:     {words_per_token:.2f}")
    print(f"")
    print(f"Estimated cost:  ${estimated_cost:.6f}")
    
    return {
        "chars": char_count,
        "words": word_count,
        "tokens": token_count,
        "cost": estimated_cost
    }

# Test with your own text
sample = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals.
"""

analyze_text(sample)
```

### Questions to Consider

- How accurate are the quick estimation formulas?
- What's the relationship between words and tokens?
- How does the cost scale with document size?

---

## Summary

✅ **Tiktoken** provides accurate token counting for OpenAI models

✅ English text averages **~4 characters per token** or **~1.33 tokens per word**

✅ **Code and JSON** use more tokens per character than prose

✅ **Non-English** languages can use 2-3x more tokens

✅ Use **exact counts** for context management, **estimates** for planning

✅ Build **utility functions** for consistent token handling across your app

**Next:** [Impact on Cost](./04-impact-on-cost.md)

---

## Further Reading

- [Tiktoken GitHub](https://github.com/openai/tiktoken) — Official library
- [OpenAI Cookbook: Token Counting](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) — Official guide
- [gpt-tokenizer (npm)](https://www.npmjs.com/package/gpt-tokenizer) — JavaScript implementation

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Text to Tokens](./02-text-to-tokens.md) | [Tokenization](./00-tokenization.md) | [Impact on Cost](./04-impact-on-cost.md) |

