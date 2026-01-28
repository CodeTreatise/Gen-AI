---
title: "What Are Tokens?"
---

# What Are Tokens?

## Introduction

Before an LLM can process text, it must convert that text into numbers—tokens. Understanding what tokens are and how they work is essential for cost estimation, prompt optimization, and debugging model behavior.

### What We'll Cover

- Tokens as the smallest units of text processing
- Not characters, not words—subword units
- Character-to-token ratios by language
- Why tokenization works this way

---

## Tokens: The Building Blocks of LLM Input

**Tokens** are the fundamental units that LLMs process. Think of them as pieces of text that the model treats as single entities.

### Not Characters, Not Words

```
Characters:  H | e | l | l | o |   | w | o | r | l | d | !
             └─────────────────────────────────────────────┘
                              12 characters

Words:       Hello | world | !
             └─────────────────┘
                    3 words

Tokens:      Hello | , |  world | !
             └──────────────────────┘
                   4 tokens (GPT-style)
```

Tokens are **subword units**—somewhere between characters and words. This design balances vocabulary size with the ability to handle new words.

### Why Subword Tokenization?

```python
# If we used character-level:
#   - Every word = many tokens (expensive, slow)
#   - "hello" = 5 tokens
#   - But: can handle any text

# If we used word-level:
#   - Efficient for common words
#   - "hello" = 1 token
#   - But: can't handle new/rare words ("qwertyuiop" = ?)

# Subword tokenization (actual approach):
#   - Common words = 1 token
#   - "hello" = 1 token
#   - Rare words = multiple tokens
#   - "qwertyuiop" = ["qwer", "ty", "uiop"] = 3 tokens
#   - Best of both worlds!
```

---

## How Tokens Look in Practice

### Common English Words

```
Word        Tokens                Count
─────────────────────────────────────
the         [the]                 1
hello       [hello]               1
computer    [computer]            1
understand  [understand]          1
```

### Less Common Words

```
Word            Tokens                   Count
──────────────────────────────────────────────
cryptocurrency  [cry, pt, ocurrency]     3
antidisestabl.  [anti, dis, establish]   3+
onomatopoeia    [on, om, atop, oeia]    4
```

### Special Cases

```
Input                   Tokens                      Count
────────────────────────────────────────────────────────
"Hello, world!"         [Hello, ,, world, !]       4
"Hello,world!"          [Hello, ,world, !]          3
"    spaces"            [   , spaces]               2
"NEW\nLINE"            [NEW, \n, LINE]             3
"123456"               [123, 456]                   2
```

> **Note:** Whitespace, punctuation, and newlines are often separate tokens. This affects prompt formatting.

---

## Average Characters Per Token

The relationship between characters and tokens varies by content type:

### English Text

```
Average: ~4 characters per token
         ~1.5 tokens per word

Example:
"The quick brown fox jumps over the lazy dog"
Characters: 43
Tokens:     ~9
Ratio:      4.8 characters/token
```

### By Content Type

| Content | Chars/Token | Notes |
|---------|-------------|-------|
| Common English | 4-5 | Efficient |
| Technical English | 3-4 | Jargon uses more tokens |
| Code | 2-3 | Syntax characters = many tokens |
| JSON | 2-3 | Brackets, quotes = tokens |
| Numbers | 2-3 | Digits grouped, but not always |

### Non-English Languages

```
Language        Chars/Token    Ratio to English
────────────────────────────────────────────────
English         ~4             1.0x
Spanish         ~3.5           ~1.15x
German          ~3             ~1.3x
Chinese         ~1.5           ~2-3x
Japanese        ~1.5           ~2-3x
Arabic          ~2             ~2x
```

> **Warning:** Non-English text typically uses 1.5-3x more tokens than equivalent English text. This significantly impacts cost for multilingual applications.

---

## Why This Design?

### Handling Unknown Words

```python
# Traditional word tokenizers can't handle:
unknown = "covfefe"  # Not in vocabulary

# Subword tokenizers break it down:
tokens = ["cov", "fe", "fe"]  # Can represent anything

# This is why LLMs can handle:
# - Typos: "teh" → ["te", "h"]
# - New words: "ChatGPT" → ["Chat", "G", "PT"]
# - Code: "myFunctionName" → ["my", "Function", "Name"]
```

### Vocabulary Size Trade-off

```
Tokenizer Vocabulary Sizes:
─────────────────────────────
GPT-2:          50,257 tokens
GPT-3/4:        ~100,000 tokens
Claude:         ~100,000 tokens
LLaMA:          32,000 tokens

Larger vocabulary = fewer tokens per text (more efficient)
                  = larger model (more memory)
```

---

## Special Tokens

Tokenizers include special tokens that aren't regular text:

```python
special_tokens = {
    "<|endoftext|>": "End of document/response",
    "<|im_start|>": "Start of message (some models)",
    "<|im_end|>": "End of message (some models)",
    "[PAD]": "Padding for batch processing",
    "[CLS]": "Classification token (BERT-style)",
    "[SEP]": "Separator between segments",
}

# These consume token slots but serve special purposes
```

### In API Messages

```python
# When you send:
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

# The API adds special tokens (invisible to you):
# <|im_start|>system\nYou are helpful.<|im_end|>
# <|im_start|>user\nHello!<|im_end|>
# <|im_start|>assistant\n

# These special tokens count toward your token usage!
```

---

## Visualizing Tokenization

### Online Tools

Several tools let you see how text is tokenized:

- **OpenAI Tokenizer**: https://platform.openai.com/tokenizer
- **Tiktokenizer**: https://tiktokenizer.vercel.app

### Example Visualizations

```
Input: "I love programming in Python!"

Token breakdown:
┌─────┬──────────────┬───────┐
│ #   │ Token        │ ID    │
├─────┼──────────────┼───────┤
│ 1   │ "I"          │ 40    │
│ 2   │ " love"      │ 3021  │
│ 3   │ " programming"│ 15473 │
│ 4   │ " in"        │ 287   │
│ 5   │ " Python"    │ 11361 │
│ 6   │ "!"          │ 0     │
└─────┴──────────────┴───────┘

Total: 6 tokens
```

Notice how spaces are often attached to the following word (" love" not "love").

---

## Token Boundaries and Model Behavior

Token boundaries can affect how models process text:

### Spaces Matter

```python
# These tokenize differently:
"HelloWorld"    # ["Hello", "World"] - 2 tokens
"Hello World"   # ["Hello", " World"] - 2 tokens
"Hello  World"  # ["Hello", "  ", "World"] - 3 tokens

# Leading spaces:
"Hello"         # ["Hello"] - 1 token
" Hello"        # [" Hello"] - 1 token (different token!)
```

### Case Sensitivity

```python
# Case affects tokenization:
"Hello"   # ["Hello"] - Token ID: 15496
"hello"   # ["hello"] - Token ID: 31373
"HELLO"   # ["HELLO"] - Token ID: 37973

# All different tokens!
```

### Why This Matters

```python
# Prompt engineering implication:
# Consistent casing = more predictable behavior
# The model has seen "Hello" and "hello" in different contexts
```

---

## Hands-on Exercise

### Your Task

Explore tokenization with the OpenAI tokenizer:

1. Go to https://platform.openai.com/tokenizer
2. Try these inputs and count the tokens:

```
Input 1: "Hello, world!"
Input 2: "Supercalifragilisticexpialidocious"
Input 3: "The quick brown fox jumps over the lazy dog"
Input 4: "{ \"name\": \"John\", \"age\": 30 }"
Input 5: "def hello_world():\n    print('Hi')"
Input 6: "你好世界" (Hello world in Chinese)
```

### Questions to Consider

- Which input has the most tokens relative to character count?
- How does JSON/code compare to natural text?
- How does Chinese compare to English?

<details>
<summary>💡 Expected Observations</summary>

1. "Hello, world!" → ~4 tokens
2. Long rare word → many tokens (broken into pieces)
3. Common sentence → efficient (~9 tokens)
4. JSON → many tokens (punctuation, quotes)
5. Code → moderate (function names may be multiple tokens)
6. Chinese → about 3 tokens for 4 characters (high ratio)

</details>

---

## Summary

✅ **Tokens** are subword units—smaller than words, larger than characters

✅ English averages **~4 characters per token**

✅ **Rare words** are split into multiple tokens

✅ **Non-English** text often requires more tokens

✅ **Special tokens** serve control purposes (end of text, message boundaries)

✅ Token boundaries affect model behavior and prompt design

**Next:** [How Text Becomes Tokens](./02-text-to-tokens.md)

---

## Further Reading

- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer) — Visualize tokenization
- [BPE Paper](https://arxiv.org/abs/1508.07909) — Original algorithm
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers) — Deep dive

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Lesson Overview](./00-tokenization.md) | [Tokenization](./00-tokenization.md) | [Text to Tokens](./02-text-to-tokens.md) |

