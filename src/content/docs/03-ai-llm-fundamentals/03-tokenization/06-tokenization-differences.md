---
title: "Tokenization Differences Across Models"
---

# Tokenization Differences Across Models

## Introduction

Different models use different tokenizers. The same text can produce different token counts depending on which model you're using. Understanding these differences helps you accurately estimate costs and context usage across providers.

### What We'll Cover

- Each model has its own tokenizer
- Different vocabularies
- SentencePiece vs. Tiktoken differences
- Multilingual tokenization efficiency
- Impact on non-English languages

---

## Each Model Has Its Own Tokenizer

Different AI providers use different tokenization systems:

```python
# Same text, different token counts across models
text = "Hello, how are you doing today?"

token_counts = {
    "gpt-4": 9,       # OpenAI cl100k_base
    "gpt-4o": 8,      # OpenAI o200k_base (larger vocab)
    "claude-4": 9,    # Anthropic tokenizer
    "gemini": 8,      # Google tokenizer
    "llama-3": 9,     # Meta SentencePiece
}

# Same meaning, but billing differs!
```

### Why Different Tokenizers?

| Provider | Reason |
|----------|--------|
| OpenAI | Optimized for their training data distribution |
| Anthropic | Different training corpus, different optimization |
| Google | Multilingual optimization for global use |
| Meta (LLaMA) | Open-source, community-focused |

---

## Vocabulary Size Differences

Larger vocabularies generally mean more efficient tokenization:

```python
vocabulary_sizes = {
    "gpt-2": 50257,           # Original
    "gpt-3/3.5 (p50k)": 50281,
    "gpt-3.5/4 (cl100k)": 100277,
    "gpt-4o (o200k)": 200019, # Largest OpenAI
    "claude": ~100000,        # Approximate
    "llama-2": 32000,         # Smaller
    "llama-3": 128000,        # Much larger
    "gemini": ~256000,        # Very large
}
```

### Impact of Vocabulary Size

```
Larger Vocabulary:
✅ Fewer tokens per text (more efficient)
✅ Better compression of common patterns
✅ Lower API costs per character
❌ Larger model size
❌ More memory required

Smaller Vocabulary:
✅ Smaller model footprint
✅ Simpler implementation
❌ More tokens per text
❌ Higher API costs per character
```

### Practical Example

```python
# Same text with different tokenizers
text = "artificial intelligence and machine learning"

# Hypothetical breakdown:
gpt2_tokens = ["art", "ificial", " intelligence", " and", " machine", " learning"]  # 6 tokens
gpt4o_tokens = ["artificial", " intelligence", " and", " machine", " learning"]  # 5 tokens
# Larger vocabulary = "artificial" is one token instead of two
```

---

## Encoding Comparisons

### OpenAI Encoding Evolution

```python
import tiktoken

text = "The quick brown fox jumps over the lazy dog."

# Compare encodings
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "p50k_base": tiktoken.get_encoding("p50k_base"),
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
}

print("Token counts by encoding:")
for name, enc in encodings.items():
    tokens = enc.encode(text)
    print(f"  {name}: {len(tokens)} tokens")

# Typical output:
# gpt2: 10 tokens
# p50k_base: 10 tokens
# cl100k_base: 10 tokens
# o200k_base: 9 tokens (more efficient)
```

### cl100k_base vs o200k_base

```python
# cl100k_base: Used by GPT-3.5, GPT-4, embeddings
# o200k_base: Used by GPT-4o, GPT-5 series

comparison_text = """
This is a longer text to show the efficiency difference 
between tokenizers. The newer o200k encoding has a larger 
vocabulary which means it can represent the same text 
using fewer tokens.
"""

cl100k = tiktoken.get_encoding("cl100k_base")
o200k = tiktoken.get_encoding("o200k_base")

print(f"cl100k_base: {len(cl100k.encode(comparison_text))} tokens")
print(f"o200k_base: {len(o200k.encode(comparison_text))} tokens")

# o200k is typically 5-15% more efficient
```

---

## Multilingual Tokenization Efficiency

Non-English languages often require more tokens:

### Character Efficiency by Language

```python
# Tokens per character (lower is better)
language_efficiency = {
    "english": 0.25,      # ~4 chars per token
    "spanish": 0.29,      # ~3.5 chars per token
    "french": 0.28,
    "german": 0.33,       # ~3 chars per token
    "russian": 0.40,
    "arabic": 0.50,       # ~2 chars per token
    "chinese": 0.67,      # ~1.5 chars per token
    "japanese": 0.75,     # ~1.3 chars per token
    "korean": 0.60,
}

# Same content costs more in Chinese than English!
```

### Why Non-English Uses More Tokens

```python
# Training data distribution
training_data_distribution = {
    "english": "60%+",
    "other_european": "20%",
    "chinese": "5%",
    "japanese": "3%",
    "other": "12%"
}

# Models optimize tokenization for common patterns
# English patterns are most common → most efficient tokenization
```

### Practical Cost Implications

```python
def calculate_multilingual_cost(
    english_tokens: int,
    language: str,
    language_multipliers: dict
) -> int:
    """Estimate tokens for same content in different language"""
    multiplier = language_multipliers.get(language, 1.5)
    return int(english_tokens * multiplier)

# Example
english_doc = 1000  # tokens for English version

# Same content in other languages
estimates = {
    "spanish": calculate_multilingual_cost(english_doc, "spanish", {"spanish": 1.2}),
    "chinese": calculate_multilingual_cost(english_doc, "chinese", {"chinese": 2.5}),
    "japanese": calculate_multilingual_cost(english_doc, "japanese", {"japanese": 2.8}),
}

print("Token estimates for equivalent content:")
for lang, tokens in estimates.items():
    print(f"  {lang}: {tokens} tokens ({tokens/english_doc:.1f}x)")
```

---

## Handling Cross-Model Token Estimation

### Token Conversion Utilities

```python
import tiktoken

class CrossModelTokenizer:
    """Estimate tokens across different models"""
    
    # Approximate conversion factors (to cl100k_base baseline)
    CONVERSION_FACTORS = {
        "gpt-3.5-turbo": 1.0,
        "gpt-4": 1.0,
        "gpt-4o": 0.9,      # o200k is ~10% more efficient
        "gpt-5": 0.9,
        "claude-3": 1.05,    # Slightly less efficient
        "claude-4": 1.0,
        "gemini-1.5": 0.95,
        "llama-3": 1.1,
    }
    
    def __init__(self):
        self.cl100k = tiktoken.get_encoding("cl100k_base")
    
    def count_baseline(self, text: str) -> int:
        """Count tokens using baseline encoder"""
        return len(self.cl100k.encode(text))
    
    def estimate_for_model(self, text: str, model: str) -> int:
        """Estimate tokens for a specific model"""
        baseline = self.count_baseline(text)
        factor = self.CONVERSION_FACTORS.get(model, 1.0)
        return int(baseline * factor)
    
    def compare_models(self, text: str, models: list) -> dict:
        """Compare token counts across models"""
        baseline = self.count_baseline(text)
        return {
            model: int(baseline * self.CONVERSION_FACTORS.get(model, 1.0))
            for model in models
        }

# Usage
tokenizer = CrossModelTokenizer()
text = "Your document content here..." * 100

comparison = tokenizer.compare_models(text, [
    "gpt-4", "gpt-4o", "claude-4", "llama-3"
])

print("Estimated token counts:")
for model, count in comparison.items():
    print(f"  {model}: {count:,}")
```

### Provider-Specific Token Counting

```python
# For accurate counts, use each provider's tokenizer

# OpenAI
import tiktoken
def count_openai(text: str, model: str) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Anthropic (use their SDK or API)
# Note: Anthropic's tokenizer is not publicly available as a library
# Use the API response's usage field for accurate counts
def count_anthropic(response) -> dict:
    return {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens
    }

# For estimation without API calls, use approximation
def estimate_anthropic(text: str) -> int:
    # Claude uses similar encoding to GPT, rough estimate
    return count_openai(text, "gpt-4") * 1.05
```

---

## Special Cases

### Code Tokenization

```python
# Code often tokenizes differently across models
code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

# Different tokenizers handle:
# - Underscores in names differently
# - Indentation differently
# - Keywords differently

# Generally, code uses 2-3 chars/token vs 4 for prose
```

### JSON and Structured Data

```python
json_data = '{"name": "John", "age": 30, "city": "NYC"}'

# JSON is tokenization-heavy:
# - Every { } [ ] " : , is often a separate token
# - Keys become tokens
# - Values become tokens

# Estimate: ~2 chars per token for JSON
```

### Emojis and Special Characters

```python
emojis = "🎉 🚀 🔥 💡"

# Emoji handling varies by tokenizer
# Some encode each emoji as 1 token
# Complex emojis (skin tones, ZWJ sequences) may be multiple tokens

# Example: "👨‍💻" (technologist) might be 4+ tokens
```

---

## Best Practices

### When Switching Between Models

```python
def switch_model_consideration(
    current_model: str,
    new_model: str,
    current_tokens: int
) -> dict:
    """Estimate impact of switching models"""
    
    efficiency_map = {
        "gpt-4": 1.0,
        "gpt-4o": 0.9,
        "claude-4-haiku": 1.0,
        "claude-4-sonnet": 1.0,
        "gemini-1.5-flash": 0.95,
    }
    
    current_efficiency = efficiency_map.get(current_model, 1.0)
    new_efficiency = efficiency_map.get(new_model, 1.0)
    
    estimated_new_tokens = int(current_tokens * new_efficiency / current_efficiency)
    
    return {
        "current_tokens": current_tokens,
        "estimated_new_tokens": estimated_new_tokens,
        "token_change_pct": (estimated_new_tokens - current_tokens) / current_tokens * 100
    }
```

### Multi-Provider Applications

```python
class MultiProviderTokenManager:
    """Manage tokens across multiple AI providers"""
    
    def __init__(self):
        self.encoders = {
            "openai": tiktoken.get_encoding("cl100k_base"),
            "openai-4o": tiktoken.get_encoding("o200k_base"),
        }
        
        self.conversion_to_openai = {
            "anthropic": 1.05,
            "google": 0.95,
            "cohere": 1.1,
        }
    
    def count_for_provider(self, text: str, provider: str) -> int:
        """Count tokens for a specific provider"""
        if provider in self.encoders:
            return len(self.encoders[provider].encode(text))
        
        # Use OpenAI as baseline with conversion factor
        baseline = len(self.encoders["openai"].encode(text))
        factor = self.conversion_to_openai.get(provider, 1.0)
        return int(baseline * factor)
```

---

## Hands-on Exercise

### Your Task

Compare tokenization across contexts:

```python
import tiktoken

def tokenization_study():
    """Study tokenization patterns"""
    
    test_cases = {
        "simple_english": "Hello world, how are you?",
        "technical": "The API returns a JSON response with OAuth2 tokens.",
        "code": "def greet(name): return f'Hello, {name}!'",
        "numbers": "The year 2024 had 365 days and 8760 hours.",
        "special": "Email: test@example.com, Price: $19.99",
    }
    
    encodings = {
        "cl100k": tiktoken.get_encoding("cl100k_base"),
        "o200k": tiktoken.get_encoding("o200k_base"),
    }
    
    print("Tokenization Comparison:")
    print("-" * 60)
    
    for case_name, text in test_cases.items():
        print(f"\n{case_name}: '{text}'")
        for enc_name, enc in encodings.items():
            tokens = enc.encode(text)
            print(f"  {enc_name}: {len(tokens)} tokens")
            print(f"    Tokens: {[enc.decode([t]) for t in tokens]}")

tokenization_study()
```

### Questions to Consider

- Which encoding is more efficient for code vs. prose?
- How do special characters affect token counts?
- What patterns use surprisingly many tokens?

---

## Summary

✅ **Each model has its own tokenizer** with different vocabulary

✅ **Larger vocabularies** (o200k vs cl100k) are more efficient

✅ **Non-English languages** typically use 1.5-3x more tokens

✅ **Code and JSON** use more tokens per character than prose

✅ Use **approximate conversion factors** when switching models

✅ For **accurate billing**, use each provider's official token counts

**Next Lesson:** [Context Windows](../04-context-windows/00-context-windows.md)

---

## Further Reading

- [OpenAI Tiktoken](https://github.com/openai/tiktoken) — Official tokenizer
- [Tokenizer Arena](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) — Compare tokenizers
- [Multilingual Tokenization Research](https://arxiv.org/abs/2306.08582) — Academic analysis

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Impact on Context](./05-impact-on-context.md) | [Tokenization](./00-tokenization.md) | [Context Windows](../04-context-windows/00-context-windows.md) |

<!-- 
Sources Consulted:
- OpenAI Tiktoken: https://github.com/openai/tiktoken
- OpenAI Platform Docs: https://platform.openai.com/docs
-->

