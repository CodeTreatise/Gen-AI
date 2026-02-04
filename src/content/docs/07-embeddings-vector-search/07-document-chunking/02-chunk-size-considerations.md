---
title: "Chunk Size Considerations"
---

# Chunk Size Considerations

## Introduction

Chunk size is the most impactful decision in your chunking strategy. Too small and you lose context; too large and you dilute relevance with noise. The optimal size depends on your content type, embedding model, and use case.

---

## The Goldilocks Zone

| Size | Tokens | Effect |
|------|--------|--------|
| **Too small** | < 200 | Lost context, incomplete thoughts |
| **Optimal** | 500-1500 | Balanced context and precision |
| **Too large** | > 2000 | Noise, diluted relevance |

```python
# Too small - loses context
too_small = "The function returns None."
# Question: Which function? What conditions? What should it return?

# Optimal - complete thought with context
optimal = """
The calculate_tax() function returns None when the input amount is 
negative or zero. For valid positive amounts, it returns the tax 
calculated using the current rate from the config file.
"""

# Too large - includes unrelated content
too_large = """
[500 tokens about authentication]
[500 tokens about rate limiting]
The calculate_tax() function returns None...  # buried
[500 tokens about logging]
[500 tokens about deployment]
"""
```

---

## Embedding Model Limits

Every embedding model has a maximum context length. Content beyond this is truncated:

| Model | Max Tokens | Recommended Chunk Size |
|-------|------------|------------------------|
| text-embedding-3-small | 8,191 | 500-1500 |
| text-embedding-3-large | 8,191 | 500-2000 |
| Voyage-3 | 32,000 | 500-2000 |
| BGE-large | 512 | 400-500 |
| all-MiniLM-L6-v2 | 256 | 200-250 |

> **Warning:** If your chunk exceeds the model's limit, content is silently truncated. The embedding won't represent the full text.

```python
def validate_chunk_size(chunk: str, model: str) -> dict:
    """Check if chunk fits model's context limit."""
    import tiktoken
    
    model_limits = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "voyage-3": 32000,
        "bge-large": 512,
    }
    
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(chunk))
    limit = model_limits.get(model, 8191)
    
    return {
        "tokens": token_count,
        "limit": limit,
        "fits": token_count <= limit,
        "overflow": max(0, token_count - limit)
    }

# Example
result = validate_chunk_size(long_text, "bge-large")
if not result["fits"]:
    print(f"Chunk exceeds limit by {result['overflow']} tokens!")
```

---

## Task-Specific Sizing

Different use cases need different chunk sizes:

| Use Case | Recommended Size | Reasoning |
|----------|------------------|-----------|
| **Q&A / Factual** | 300-500 tokens | Precise answers, less noise |
| **Summarization** | 1000-2000 tokens | Need broader context |
| **Code search** | Function-sized | Natural code boundaries |
| **Legal/Medical** | 500-800 tokens | Preserve clause/section context |
| **Chat/Support** | 400-600 tokens | Quick, focused responses |

### Dynamic Sizing Based on Content

```python
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    target_size: int
    min_size: int
    max_size: int
    overlap: int

def get_config_for_content(content_type: str) -> ChunkConfig:
    """Get optimal chunk config for content type."""
    configs = {
        "documentation": ChunkConfig(
            target_size=800, min_size=400, max_size=1200, overlap=100
        ),
        "code": ChunkConfig(
            target_size=1000, min_size=200, max_size=2000, overlap=50
        ),
        "legal": ChunkConfig(
            target_size=600, min_size=300, max_size=900, overlap=150
        ),
        "chat_logs": ChunkConfig(
            target_size=400, min_size=200, max_size=600, overlap=50
        ),
        "research": ChunkConfig(
            target_size=1000, min_size=500, max_size=1500, overlap=200
        ),
    }
    return configs.get(content_type, ChunkConfig(
        target_size=500, min_size=200, max_size=1000, overlap=100
    ))
```

---

## Measuring Tokens vs Characters

Tokens ≠ characters. Always measure in tokens for accuracy:

```python
import tiktoken

def chars_to_tokens_ratio(text: str) -> float:
    """Approximate character to token ratio."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(text))
    chars = len(text)
    return chars / tokens

# English prose: ~4 characters per token
# Code: ~3 characters per token  
# Technical docs: ~3.5 characters per token

def estimate_tokens(text: str, content_type: str = "prose") -> int:
    """Quick token estimate without tokenizing."""
    ratios = {
        "prose": 4.0,
        "code": 3.0,
        "technical": 3.5,
        "mixed": 3.5
    }
    ratio = ratios.get(content_type, 3.5)
    return int(len(text) / ratio)

# For production, always use actual tokenizer
def count_tokens_accurate(text: str) -> int:
    """Accurate token count using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
```

---

## Chunk Size Distribution Analysis

Analyze your chunks to ensure consistent sizing:

```python
import numpy as np
from collections import Counter

def analyze_chunk_distribution(chunks: list[str]) -> dict:
    """Analyze token distribution across chunks."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    sizes = [len(enc.encode(chunk)) for chunk in chunks]
    
    return {
        "count": len(chunks),
        "mean": np.mean(sizes),
        "std": np.std(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "median": np.median(sizes),
        "p25": np.percentile(sizes, 25),
        "p75": np.percentile(sizes, 75),
        "too_small": sum(1 for s in sizes if s < 100),
        "too_large": sum(1 for s in sizes if s > 2000),
    }

# Example output:
# {
#   "count": 1500,
#   "mean": 487.3,
#   "std": 152.1,
#   "min": 45,     # <- investigate these
#   "max": 3421,   # <- investigate these
#   "too_small": 23,
#   "too_large": 8
# }
```

---

## Adaptive Chunking

Adjust chunk size based on content density:

```python
def adaptive_chunk_size(
    text: str,
    base_size: int = 500,
    min_size: int = 200,
    max_size: int = 1000
) -> int:
    """Adjust chunk size based on content characteristics."""
    
    # Information density indicators
    has_code = "```" in text or "def " in text or "class " in text
    has_lists = text.count("\n- ") > 3 or text.count("\n* ") > 3
    has_tables = "|" in text and "-|-" in text
    avg_sentence_length = len(text) / max(1, text.count(". "))
    
    size = base_size
    
    # Code is denser, use larger chunks
    if has_code:
        size = int(size * 1.5)
    
    # Lists are scannable, can be smaller
    if has_lists:
        size = int(size * 0.8)
    
    # Tables need context, keep together
    if has_tables:
        size = int(size * 1.3)
    
    # Long sentences = complex content, larger chunks
    if avg_sentence_length > 150:
        size = int(size * 1.2)
    
    return max(min_size, min(max_size, size))
```

---

## Testing Chunk Size Impact

Always evaluate chunk size choices:

```python
def evaluate_chunk_size(
    test_queries: list[str],
    ground_truth: dict[str, list[str]],  # query -> relevant doc IDs
    chunk_sizes: list[int],
    documents: list[str],
    embed_fn
) -> dict:
    """Compare retrieval quality across chunk sizes."""
    
    results = {}
    
    for size in chunk_sizes:
        # Chunk documents at this size
        chunks = chunk_documents(documents, size)
        
        # Build index
        index = build_index(chunks, embed_fn)
        
        # Evaluate
        recall_at_k = []
        for query in test_queries:
            retrieved = search(index, query, k=5)
            relevant = ground_truth[query]
            hits = len(set(retrieved) & set(relevant))
            recall_at_k.append(hits / len(relevant))
        
        results[size] = {
            "avg_recall@5": np.mean(recall_at_k),
            "chunk_count": len(chunks)
        }
    
    return results

# Example results:
# {
#   256: {"avg_recall@5": 0.72, "chunk_count": 4521},
#   512: {"avg_recall@5": 0.81, "chunk_count": 2312},  # <-- best
#   1024: {"avg_recall@5": 0.78, "chunk_count": 1189},
#   2048: {"avg_recall@5": 0.69, "chunk_count": 612},
# }
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Measure in tokens, not characters | Assume 1 char = 1 token |
| Check embedding model limits | Exceed context length silently |
| Adjust for content type | Use same size for everything |
| Analyze chunk distribution | Ignore outliers |
| Evaluate with real queries | Pick arbitrary sizes |

---

## Summary

✅ **Optimal range is 500-1500 tokens** for most use cases

✅ **Match chunk size to embedding model limits** to avoid truncation

✅ **Adjust for content type** - code, legal, chat all have different needs

✅ **Measure in tokens** - characters are a poor proxy

✅ **Evaluate empirically** - test with your actual queries

**Next:** [Overlap Strategies](./03-overlap-strategies.md)
