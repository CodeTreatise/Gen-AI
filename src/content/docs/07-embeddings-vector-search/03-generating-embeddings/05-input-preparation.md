---
title: "Input Preparation"
---

# Input Preparation

## Introduction

The quality of your embeddings starts with the quality of your input. Raw text often contains noise—extra whitespace, encoding issues, irrelevant metadata—that can degrade embedding quality. Proper input preparation ensures your embedding model receives clean, well-structured text that maximizes semantic capture.

In this lesson, we'll explore text cleaning techniques, encoding best practices, and prefix strategies that can improve embedding quality by 5-10% or more.

### What We'll Cover

- Text cleaning and normalization
- Character encoding and UTF-8 handling
- Whitespace normalization
- Special character handling
- Prefix and instruction strategies
- Input length validation

### Prerequisites

- Completed [Embedding API Calls](./01-embedding-api-calls.md)
- Basic understanding of text processing

---

## Why Input Preparation Matters

Consider these two versions of the same content:

**Raw input:**
```
  \n\n  The quick    brown fox   jumps over the\tlazy dog.  \n\n\n
```

**Cleaned input:**
```
The quick brown fox jumps over the lazy dog.
```

While embedding models are somewhat robust to noise, consistent preprocessing:
- Reduces token usage (saving costs)
- Improves embedding consistency for similar content
- Removes artifacts that can shift semantic meaning
- Enables reliable caching (same input → same cache key)

> **🤖 AI Context:** In production RAG systems, input preparation is often the difference between 80% and 95% retrieval accuracy. Garbage in, garbage out applies doubly to embeddings.

---

## Text Cleaning Fundamentals

### Basic Cleaning Function

```python
import re
import unicodedata

def clean_text(text: str) -> str:
    """Clean text for embedding generation."""
    if not text:
        return ""
    
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Example
raw = "  \n\n  The quick    brown fox   jumps over the\tlazy dog.  \n\n\n"
cleaned = clean_text(raw)
print(f"Raw length: {len(raw)}")
print(f"Cleaned length: {len(cleaned)}")
print(f"Cleaned: '{cleaned}'")
```

**Output:**
```
Raw length: 67
Cleaned length: 44
Cleaned: 'The quick brown fox jumps over the lazy dog.'
```

### Unicode Normalization

Unicode has multiple ways to represent the same character. Normalization ensures consistency:

```python
import unicodedata

# These look identical but are different bytes
char1 = "é"  # Single character (U+00E9)
char2 = "é"  # Two characters: e + combining accent (U+0065 + U+0301)

print(f"char1 bytes: {char1.encode('utf-8')}")
print(f"char2 bytes: {char2.encode('utf-8')}")
print(f"Equal? {char1 == char2}")

# After normalization
norm1 = unicodedata.normalize("NFC", char1)
norm2 = unicodedata.normalize("NFC", char2)
print(f"After NFC normalization equal? {norm1 == norm2}")
```

**Output:**
```
char1 bytes: b'\xc3\xa9'
char2 bytes: b'e\xcc\x81'
Equal? False
After NFC normalization equal? True
```

| Form | Description | Use Case |
|------|-------------|----------|
| NFC | Composed (default) | Most text processing |
| NFD | Decomposed | Linguistic analysis |
| NFKC | Compatibility composed | Search, matching |
| NFKD | Compatibility decomposed | Full normalization |

> **Tip:** Use NFC for most embedding tasks. Use NFKC when you want "ﬁ" to match "fi".

---

## UTF-8 Encoding

All major embedding APIs expect UTF-8 encoded text. Handle encoding issues before sending:

```python
def ensure_utf8(text: str | bytes) -> str:
    """Ensure text is properly UTF-8 encoded."""
    if isinstance(text, bytes):
        # Try to decode as UTF-8, fallback to latin-1
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            text = text.decode('latin-1')
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    
    # Encode and decode to ensure valid UTF-8
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    return text

# Example with problematic input
problematic = b"Hello \x80 World"  # Invalid UTF-8 byte
cleaned = ensure_utf8(problematic)
print(f"Cleaned: {cleaned}")
```

**Output:**
```
Cleaned: Hello � World
```

### Common Encoding Issues

| Issue | Example | Solution |
|-------|---------|----------|
| Latin-1 in UTF-8 | `b'\xe9'` for é | Detect and re-encode |
| Null bytes | `\x00` | Remove |
| Surrogate pairs | Invalid surrogates | Use `errors='replace'` |
| BOM markers | `\ufeff` | Strip |

---

## Whitespace Normalization

Different whitespace characters can affect tokenization and embedding consistency:

```python
def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces."""
    # Replace all Unicode whitespace with regular space
    # This includes: tabs, newlines, non-breaking spaces, etc.
    whitespace_chars = [
        '\t',      # Tab
        '\n',      # Newline
        '\r',      # Carriage return
        '\f',      # Form feed
        '\v',      # Vertical tab
        '\xa0',    # Non-breaking space
        '\u2003',  # Em space
        '\u2002',  # En space
        '\u200b',  # Zero-width space
        '\u00a0',  # NBSP
    ]
    
    for char in whitespace_chars:
        text = text.replace(char, ' ')
    
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# Example
text_with_weird_spaces = "Hello\u00a0world\u200bhow\tare\nyou"
normalized = normalize_whitespace(text_with_weird_spaces)
print(f"Original: {repr(text_with_weird_spaces)}")
print(f"Normalized: {repr(normalized)}")
```

**Output:**
```
Original: 'Hello\xa0world\u200bhow\tare\nyou'
Normalized: 'Hello world how are you'
```

---

## Special Character Handling

Decide which special characters to keep, remove, or replace:

```python
import re

def clean_special_chars(
    text: str,
    keep_punctuation: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
) -> str:
    """Clean special characters based on configuration."""
    
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    if remove_emails:
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    if not keep_punctuation:
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

# Example
text = "Check out https://example.com or email us at info@test.com! 🎉"
cleaned = clean_special_chars(text)
print(cleaned)
```

**Output:**
```
Check out [URL] or email us at [EMAIL]! 🎉
```

### Emoji Handling

Emojis are valid Unicode and some models handle them well. Decide based on your use case:

```python
import re

def handle_emojis(text: str, action: str = "keep") -> str:
    """Handle emojis: keep, remove, or replace with description."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )
    
    if action == "remove":
        return emoji_pattern.sub('', text)
    elif action == "replace":
        return emoji_pattern.sub('[EMOJI]', text)
    else:
        return text  # keep as-is

# Examples
text = "I love this! 🎉😊"
print(f"Keep: {handle_emojis(text, 'keep')}")
print(f"Remove: {handle_emojis(text, 'remove')}")
print(f"Replace: {handle_emojis(text, 'replace')}")
```

**Output:**
```
Keep: I love this! 🎉😊
Remove: I love this! 
Replace: I love this! [EMOJI]
```

---

## Prefix and Instruction Strategies

Many embedding models benefit from task-specific prefixes that provide context:

### Query vs Document Prefixes

```python
def add_prefix(text: str, prefix_type: str) -> str:
    """Add a task-specific prefix to the text."""
    prefixes = {
        "query": "query: ",
        "document": "passage: ",
        "search": "search_query: ",
        "code": "Represent the code for retrieval: ",
        "question": "Represent this question for searching answers: ",
    }
    
    prefix = prefixes.get(prefix_type, "")
    return prefix + text

# Example
question = "How do I sort a list in Python?"
document = "To sort a list in Python, use the sorted() function or list.sort() method."

print(add_prefix(question, "query"))
print(add_prefix(document, "document"))
```

**Output:**
```
query: How do I sort a list in Python?
passage: To sort a list in Python, use the sorted() function or list.sort() method.
```

### When to Use Prefixes

| Model/Provider | Prefix Required? | Notes |
|----------------|------------------|-------|
| OpenAI | No | General purpose |
| Gemini | No | Uses `task_type` instead |
| Cohere | No | Uses `input_type` instead |
| Voyage | Optional | Adds internally when `input_type` set |
| E5 models | Yes | `query:` and `passage:` |
| Instructor models | Yes | Detailed instructions |
| BGE models | Optional | `Represent this...` |

### Instruction-Based Prefixes

Some open-source models support detailed instructions:

```python
def add_instruction(text: str, instruction: str) -> str:
    """Add an instruction-style prefix (for Instructor/E5 models)."""
    return f"Instruct: {instruction}\nQuery: {text}"

# Example for Instructor-XL
instruction = "Represent the query for retrieving technical documentation about Python programming"
query = "async await syntax"

prefixed = add_instruction(query, instruction)
print(prefixed)
```

**Output:**
```
Instruct: Represent the query for retrieving technical documentation about Python programming
Query: async await syntax
```

---

## Input Length Validation

Validate and handle text length before embedding:

```python
import tiktoken

def validate_and_truncate(
    text: str,
    max_tokens: int = 8192,
    model: str = "text-embedding-3-small"
) -> tuple[str, bool]:
    """Validate text length and truncate if needed."""
    
    # Get the tokenizer for the model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    was_truncated = False
    
    if len(tokens) > max_tokens:
        # Truncate and decode back to text
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
        was_truncated = True
    
    return text, was_truncated

# Example
long_text = "This is a test. " * 5000  # Very long text
truncated, was_cut = validate_and_truncate(long_text, max_tokens=100)
print(f"Was truncated: {was_cut}")
print(f"Result length: {len(truncated)} chars")
print(f"Result: {truncated[:100]}...")
```

**Output:**
```
Was truncated: True
Result length: 428 chars
Result: This is a test. This is a test. This is a test. This is a test. This is a test. This is a test. Th...
```

### Token Limits by Provider

| Provider | Model | Max Tokens |
|----------|-------|------------|
| OpenAI | text-embedding-3-* | 8,192 |
| Gemini | gemini-embedding-001 | 2,048 |
| Cohere | embed-v4.0 | Varies |
| Voyage | voyage-4-* | 32,000 |

---

## Complete Preprocessing Pipeline

Here's a production-ready preprocessing function:

```python
import re
import unicodedata
import tiktoken

def preprocess_for_embedding(
    text: str,
    prefix_type: str | None = None,
    max_tokens: int = 8192,
    remove_urls: bool = True,
    normalize_unicode: bool = True,
) -> dict:
    """
    Complete preprocessing pipeline for embedding generation.
    
    Returns dict with:
        - text: processed text
        - original_length: original character count
        - processed_length: processed character count
        - was_truncated: bool
        - token_count: approximate token count
    """
    original_length = len(text)
    
    # Step 1: Ensure valid UTF-8
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = text.replace('\x00', '')
    
    # Step 2: Unicode normalization
    if normalize_unicode:
        text = unicodedata.normalize("NFC", text)
    
    # Step 3: URL/email handling
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Step 4: Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 5: Add prefix if specified
    if prefix_type:
        prefixes = {
            "query": "query: ",
            "document": "passage: ",
        }
        text = prefixes.get(prefix_type, "") + text
    
    # Step 6: Validate length and truncate if needed
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    was_truncated = False
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
        was_truncated = True
    
    return {
        "text": text,
        "original_length": original_length,
        "processed_length": len(text),
        "was_truncated": was_truncated,
        "token_count": len(tokens),
    }

# Example usage
raw_text = """
    Check out our website at https://example.com for more info!
    
    Contact us at    support@example.com    with questions.
    
    We're here to help! 🙂
"""

result = preprocess_for_embedding(raw_text, prefix_type="document")
print(f"Processed text: '{result['text']}'")
print(f"Original: {result['original_length']} chars → {result['processed_length']} chars")
print(f"Tokens: {result['token_count']}")
```

**Output:**
```
Processed text: 'passage: Check out our website at for more info! Contact us at with questions. We're here to help! 🙂'
Original: 179 chars → 103 chars
Tokens: 26
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Normalize Unicode (NFC) | Mix Unicode forms |
| Collapse whitespace | Keep raw HTML/markdown formatting |
| Validate length before sending | Let API truncate silently |
| Use consistent preprocessing | Different cleaning per document |
| Document your pipeline | Change preprocessing without re-embedding |
| Remove metadata if not relevant | Embed file paths, timestamps, etc. |

---

## Hands-on Exercise

### Your Task

Build a preprocessing pipeline that:

1. Handles a batch of messy documents
2. Cleans and normalizes each document
3. Reports statistics on what was cleaned
4. Ensures all outputs are within token limits

### Requirements

1. Process at least 5 sample documents with various issues
2. Track: original length, cleaned length, truncation status
3. Return cleaned texts ready for embedding
4. Print a summary report

<details>
<summary>💡 Hints</summary>

- Create documents with different issues: URLs, extra whitespace, emojis, long text
- Use the `preprocess_for_embedding` function as a starting point
- Calculate statistics across all documents

</details>

<details>
<summary>✅ Solution</summary>

```python
from typing import List, Dict

def preprocess_batch(documents: List[str], max_tokens: int = 500) -> Dict:
    """Preprocess a batch of documents with reporting."""
    
    results = []
    stats = {
        "total_documents": len(documents),
        "total_original_chars": 0,
        "total_cleaned_chars": 0,
        "truncated_count": 0,
        "avg_token_reduction": 0,
    }
    
    for doc in documents:
        result = preprocess_for_embedding(doc, max_tokens=max_tokens)
        results.append(result)
        
        stats["total_original_chars"] += result["original_length"]
        stats["total_cleaned_chars"] += result["processed_length"]
        if result["was_truncated"]:
            stats["truncated_count"] += 1
    
    # Calculate averages
    stats["char_reduction_pct"] = (
        1 - stats["total_cleaned_chars"] / stats["total_original_chars"]
    ) * 100 if stats["total_original_chars"] > 0 else 0
    
    return {
        "cleaned_texts": [r["text"] for r in results],
        "details": results,
        "stats": stats,
    }

# Sample messy documents
messy_docs = [
    "  \n\nCheck out https://example.com for more!   \n\n",
    "Contact: info@test.com    or    sales@test.com",
    "Lots    of   \t\t  weird   spacing   here",
    "Unicode test: café résumé naïve " + "word " * 1000,  # Long document
    "Emoji party! 🎉🎊🎁✨ Let's celebrate! 🥳",
]

# Process
output = preprocess_batch(messy_docs, max_tokens=100)

# Print report
print("=" * 50)
print("PREPROCESSING REPORT")
print("=" * 50)
print(f"Documents processed: {output['stats']['total_documents']}")
print(f"Character reduction: {output['stats']['char_reduction_pct']:.1f}%")
print(f"Documents truncated: {output['stats']['truncated_count']}")
print("-" * 50)

for i, (original, detail) in enumerate(zip(messy_docs, output['details'])):
    print(f"\nDoc {i + 1}:")
    print(f"  Original ({detail['original_length']} chars): {original[:50]}...")
    print(f"  Cleaned ({detail['processed_length']} chars): {detail['text'][:50]}...")
    print(f"  Truncated: {detail['was_truncated']}, Tokens: {detail['token_count']}")
```

</details>

---

## Summary

✅ Clean text before embedding—normalize Unicode, collapse whitespace, handle encoding

✅ Use UTF-8 encoding and handle problematic characters gracefully

✅ Remove irrelevant content (URLs, emails) unless semantically important

✅ Add task-specific prefixes when the model supports them

✅ Validate token length to avoid silent truncation

✅ Document your preprocessing pipeline—changing it requires re-embedding

**Next:** [Handling Long Texts](./06-handling-long-texts.md)

---

## Further Reading

- [Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [tiktoken - OpenAI Tokenizer](https://github.com/openai/tiktoken)
- [Python unicodedata Module](https://docs.python.org/3/library/unicodedata.html)

---

<!-- 
Sources Consulted:
- OpenAI Embeddings API: https://platform.openai.com/docs/api-reference/embeddings
- tiktoken documentation: https://github.com/openai/tiktoken
- Unicode Technical Reports: https://unicode.org/reports/tr15/
-->
