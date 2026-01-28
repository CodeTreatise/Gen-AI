---
title: "How Text Is Converted to Tokens"
---

# How Text Is Converted to Tokens

## Introduction

Tokenization isn't arbitrary—it uses specific algorithms that determine how text is split. Understanding these algorithms helps you predict token counts and optimize your prompts.

### What We'll Cover

- Byte Pair Encoding (BPE)
- SentencePiece tokenization
- Tokenizer vocabulary
- Special tokens
- Visualizing the tokenization process

---

## Byte Pair Encoding (BPE)

**Byte Pair Encoding** is the most common tokenization algorithm used by GPT models, Claude, and others.

### How BPE Works

BPE starts with individual characters and repeatedly merges the most frequent pairs:

```
Training Process (simplified):

Step 0: Start with characters
        Vocabulary: {a, b, c, d, e, f, g, h, i, ...}

Step 1: Find most common pair in training data
        Example: "t" + "h" appears most often
        Merge into "th"
        Vocabulary: {..., th}

Step 2: Find next most common pair
        Example: "th" + "e" appears often
        Merge into "the"
        Vocabulary: {..., th, the}

Step 3: Continue until vocabulary size reached
        Example: "e" + "r" → "er"
        Example: "er" + "s" → "ers"
        ...

Final: Vocabulary of ~50,000-100,000 tokens
```

### BPE in Action

```python
# Conceptual BPE encoding
def bpe_encode(text, vocabulary, merges):
    """
    Encode text using trained BPE vocabulary
    """
    # Start with characters (or bytes)
    tokens = list(text)
    
    # Apply merges in priority order
    for merge_pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == merge_pair:
                # Merge the pair
                tokens[i:i+2] = [tokens[i] + tokens[i+1]]
            else:
                i += 1
    
    # Convert to token IDs
    token_ids = [vocabulary[t] for t in tokens]
    return token_ids

# Example
text = "lower"
# Step by step:
# ['l', 'o', 'w', 'e', 'r']
# ['lo', 'w', 'e', 'r']      (merge 'l' + 'o')
# ['low', 'e', 'r']          (merge 'lo' + 'w')
# ['low', 'er']              (merge 'e' + 'r')
# ['lower']                  (merge 'low' + 'er')
```

### Why BPE Works Well

| Property | Benefit |
|----------|---------|
| Common words → 1 token | Efficient for frequent text |
| Rare words → subwords | Can handle any text |
| Learned from data | Optimized for training corpus |
| Deterministic | Same text → same tokens |

---

## SentencePiece Tokenization

**SentencePiece** is another popular approach, used by LLaMA, Gemini, and others.

### Key Differences from BPE

```python
# BPE typically works on pre-tokenized text
text = "Hello world"
# First split: ["Hello", "world"]
# Then BPE each word

# SentencePiece works on raw text
text = "Hello world"
# Treats as: "▁Hello▁world" (▁ = space marker)
# Tokenizes without pre-splitting
```

### The Underscore Convention

```
SentencePiece representation:
"Hello world" → "▁Hello▁world"

Where ▁ (U+2581) marks word boundaries

Tokens: ["▁Hello", "▁world"]
        or with more splits:
        ["▁He", "llo", "▁wo", "rld"]
```

### SentencePiece Models

| Model | Description |
|-------|-------------|
| **Unigram** | Probabilistic subword selection |
| **BPE** | SentencePiece can also use BPE |

---

## Tokenizer Vocabulary

Each model has a fixed vocabulary—the set of all possible tokens.

### Vocabulary Structure

```python
# Conceptual vocabulary
vocabulary = {
    # Special tokens
    "<|endoftext|>": 0,
    "<|im_start|>": 1,
    # ... more special tokens
    
    # Regular tokens
    "the": 1169,
    "The": 464,
    " the": 262,
    "hello": 31373,
    "Hello": 15496,
    # ... ~100,000 entries
    
    # Subword tokens
    "ing": 278,
    "tion": 428,
    "er": 263,
    # ...
}
```

### Looking Up Tokens

```python
import tiktoken

# Load tokenizer
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

# Text to tokens
text = "Hello, world!"
tokens = enc.encode(text)
print(tokens)  # [9906, 11, 1917, 0]

# Tokens to text
decoded = enc.decode(tokens)
print(decoded)  # "Hello, world!"

# See individual tokens
for token_id in tokens:
    print(f"{token_id}: '{enc.decode([token_id])}'")
# 9906: 'Hello'
# 11: ','
# 1917: ' world'
# 0: '!'
```

---

## Special Tokens

Tokenizers reserve certain tokens for special purposes:

### Common Special Tokens

```python
special_tokens = {
    # OpenAI GPT-4
    "<|endoftext|>": "End of document",
    "<|fim_prefix|>": "Fill-in-middle prefix",
    "<|fim_middle|>": "Fill-in-middle target",
    "<|fim_suffix|>": "Fill-in-middle suffix",
    
    # Chat format (varies by model)
    "<|im_start|>": "Start of message",
    "<|im_end|>": "End of message",
    
    # BERT-style (embedding models)
    "[CLS]": "Classification token",
    "[SEP]": "Separator token",
    "[PAD]": "Padding token",
    "[MASK]": "Masked token for training",
}
```

### How Chat Messages Use Special Tokens

```python
# What you send:
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi!"}
]

# What's actually tokenized (ChatML format):
"""
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
"""

# Each message boundary adds tokens!
```

> **Note:** Message formatting overhead typically adds 4-8 tokens per message. This matters in long conversations.

---

## Tokenization Process Step by Step

### Complete Example

```python
import tiktoken

def show_tokenization(text, encoding_name="cl100k_base"):
    """Visualize tokenization process"""
    enc = tiktoken.get_encoding(encoding_name)
    
    # Encode
    tokens = enc.encode(text)
    
    print(f"Input: '{text}'")
    print(f"Token count: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    print("\nBreakdown:")
    
    for i, token_id in enumerate(tokens):
        token_text = enc.decode([token_id])
        # Show control characters clearly
        display = repr(token_text) if token_text.strip() != token_text else token_text
        print(f"  {i+1}. ID {token_id:>6} → {display}")
    
    print(f"\nCharacter/token ratio: {len(text)/len(tokens):.2f}")

# Example
show_tokenization("The quick brown fox jumps over the lazy dog.")
```

**Output:**
```
Input: 'The quick brown fox jumps over the lazy dog.'
Token count: 10
Token IDs: [791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5765, 13]

Breakdown:
  1. ID    791 → The
  2. ID   4062 →  quick
  3. ID  14198 →  brown
  4. ID  39935 →  fox
  5. ID  35308 →  jumps
  6. ID    927 →  over
  7. ID    279 →  the
  8. ID  16053 →  lazy
  9. ID   5765 →  dog
  10. ID     13 → .

Character/token ratio: 4.40
```

---

## Encoding Names

Different models use different encodings:

### OpenAI Encodings

```python
import tiktoken

# List available encodings
print(tiktoken.list_encoding_names())
# ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base']

# Model to encoding mapping
model_encodings = {
    "gpt-2": "gpt2",
    "gpt-3": "p50k_base", 
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-5": "o200k_base",  # Expected
    "text-embedding-3-*": "cl100k_base",
}

# Get encoding for a model
enc = tiktoken.encoding_for_model("gpt-4")
```

### cl100k_base vs o200k_base

| Encoding | Vocab Size | Used By |
|----------|------------|---------|
| cl100k_base | ~100,000 | GPT-3.5, GPT-4, embeddings |
| o200k_base | ~200,000 | GPT-4o, GPT-5 series |

Larger vocabulary = more efficient tokenization (fewer tokens for same text).

---

## Handling Edge Cases

### Whitespace and Formatting

```python
# Whitespace affects tokenization
enc = tiktoken.get_encoding("cl100k_base")

print(len(enc.encode("Hello World")))     # 2
print(len(enc.encode("Hello  World")))    # 3 (extra space = extra token)
print(len(enc.encode("Hello\nWorld")))    # 3 (newline = token)
print(len(enc.encode("Hello\t\tWorld")))  # 4 (tabs = tokens)
```

### Unicode and Emojis

```python
# Unicode handling
print(len(enc.encode("Hello")))        # 1
print(len(enc.encode("你好")))          # 2 (Chinese characters)
print(len(enc.encode("🎉")))            # 1 (common emoji)
print(len(enc.encode("🧑‍💻")))          # 4 (combined emoji)
```

### Byte Fallback

```python
# For unknown characters, BPE falls back to byte-level encoding
unknown = "🏴󠁧󠁢󠁷󠁬󠁳󠁿"  # Wales flag (complex emoji)
tokens = enc.encode(unknown)
print(f"Tokens: {len(tokens)}")  # Many tokens for complex Unicode
```

---

## Hands-on Exercise

### Your Task

Use tiktoken to analyze different text types:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "Hello, world!",
    "Supercalifragilisticexpialidocious",
    '{"name": "John", "age": 30}',
    "def hello():\n    return 'world'",
    "The quick brown fox jumps over the lazy dog.",
    "机器学习是人工智能的一个分支",  # Machine learning is a branch of AI
]

for text in texts:
    tokens = enc.encode(text)
    ratio = len(text) / len(tokens)
    print(f"Chars: {len(text):3} | Tokens: {len(tokens):3} | Ratio: {ratio:.2f} | Text: {text[:40]}...")
```

### Questions to Consider

- Which text type has the highest character/token ratio?
- How does code compare to natural language?
- What's the ratio for Chinese text?

---

## Summary

✅ **BPE (Byte Pair Encoding)** merges frequent character pairs into tokens

✅ **SentencePiece** works on raw text with explicit space markers

✅ **Vocabulary** is fixed at training time (~100K tokens typical)

✅ **Special tokens** handle message boundaries and control sequences

✅ **Encoding names** like `cl100k_base` determine the vocabulary

✅ Different content types have different efficiency ratios

**Next:** [Token Counting](./03-token-counting.md)

---

## Further Reading

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Original BPE paper
- [SentencePiece: Unsupervised Text Tokenizer](https://github.com/google/sentencepiece) — Google's implementation
- [Tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenization library

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [What Are Tokens?](./01-what-are-tokens.md) | [Tokenization](./00-tokenization.md) | [Token Counting](./03-token-counting.md) |

