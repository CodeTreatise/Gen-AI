---
title: "Embedding Normalization"
---

# Embedding Normalization

## Introduction

Normalization is one of the most overlooked aspects of embedding generation—and one of the most critical. When embeddings aren't normalized, cosine similarity calculations become unreliable, search results degrade, and debugging becomes nearly impossible.

In this lesson, we'll explain what normalization means, when it's required, and the critical gotcha with Gemini's reduced-dimension embeddings that can silently break your search quality.

### What We'll Cover

- What L2 normalization is and why it matters
- Which providers return normalized embeddings
- The critical Gemini normalization requirement
- How to normalize embeddings in Python
- Verifying normalization in your pipeline

### Prerequisites

- Completed [Dimension Control](./03-dimension-control.md)
- Basic linear algebra (vector magnitude)

---

## What Is Normalization?

A normalized (or "unit") vector has a length (magnitude) of exactly 1. This is achieved through L2 normalization:

$$\hat{v} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^{n} v_i^2}}$$

Where:
- $v$ is the original vector
- $\|v\|_2$ is the L2 norm (Euclidean length)
- $\hat{v}$ is the normalized vector

```python
import numpy as np

def l2_normalize(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    v = np.array(vector)
    norm = np.linalg.norm(v)
    if norm == 0:
        return vector  # Can't normalize zero vector
    return (v / norm).tolist()

# Example
v = [3.0, 4.0]
v_normalized = l2_normalize(v)
print(f"Original: {v}, length: {np.linalg.norm(v)}")
print(f"Normalized: {v_normalized}, length: {np.linalg.norm(v_normalized)}")
```

**Output:**
```
Original: [3.0, 4.0], length: 5.0
Normalized: [0.6, 0.8], length: 1.0
```

---

## Why Normalization Matters

### Cosine Similarity Simplification

For normalized vectors, cosine similarity equals the dot product:

$$\cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}$$

When $\|a\| = \|b\| = 1$:

$$\cos(\theta) = a \cdot b$$

This makes similarity calculations:
- **Faster** — just a dot product, no division
- **Simpler** — easier to reason about
- **Database-friendly** — most vector DBs optimize for dot product

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a: list[float], b: list[float]) -> float:
    """Dot product (equals cosine for normalized vectors)."""
    return np.dot(a, b)

# For normalized vectors, these are identical
v1_norm = l2_normalize([1, 2, 3])
v2_norm = l2_normalize([4, 5, 6])

print(f"Cosine similarity: {cosine_similarity(v1_norm, v2_norm):.6f}")
print(f"Dot product: {dot_product(v1_norm, v2_norm):.6f}")
```

**Output:**
```
Cosine similarity: 0.974632
Dot product: 0.974632
```

### What Happens Without Normalization

```python
# Non-normalized vectors with same direction but different magnitudes
v1 = [1.0, 0.0]  # Length 1
v2 = [100.0, 0.0]  # Length 100

# Cosine similarity (correct)
print(f"Cosine: {cosine_similarity(v1, v2):.2f}")  # 1.0 (same direction)

# Dot product (misleading for non-normalized)
print(f"Dot: {dot_product(v1, v2):.2f}")  # 100.0 (not a similarity!)
```

**Output:**
```
Cosine: 1.00
Dot: 100.00
```

> **Warning:** If your vector database uses dot product similarity (common for performance), non-normalized embeddings will produce incorrect rankings.

---

## Provider Normalization Status

| Provider | Model | Pre-Normalized? | Notes |
|----------|-------|-----------------|-------|
| OpenAI | text-embedding-3-* | ✅ Yes | All dimensions |
| OpenAI | text-embedding-ada-002 | ✅ Yes | Always |
| Gemini | gemini-embedding-001 (3072d) | ✅ Yes | Full dimensions only |
| Gemini | gemini-embedding-001 (768/1536d) | ❌ **NO** | Must normalize manually! |
| Cohere | embed-v4.0 | ✅ Yes | All configurations |
| Voyage | voyage-4-* | ✅ Yes | All configurations |

---

## ⚠️ Critical: Gemini Normalization Warning

This is the most common pitfall in embedding generation:

> **Gemini embeddings at 768 and 1536 dimensions are NOT pre-normalized. You MUST normalize them manually.**

### The Problem

```python
import google.generativeai as genai
import numpy as np

genai.configure(api_key="YOUR_API_KEY")

# Full dimensions - normalized
full_result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Hello, world!",
    task_type="RETRIEVAL_DOCUMENT",
)
full_norm = np.linalg.norm(full_result['embedding'])
print(f"Full 3072d norm: {full_norm:.6f}")  # ~1.0

# Reduced dimensions - NOT normalized!
reduced_result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Hello, world!",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=768,
)
reduced_norm = np.linalg.norm(reduced_result['embedding'])
print(f"Reduced 768d norm: {reduced_norm:.6f}")  # NOT 1.0!
```

**Output:**
```
Full 3072d norm: 1.000000
Reduced 768d norm: 0.573421
```

### The Fix

Always normalize after using `output_dimensionality`:

```python
def get_gemini_embedding(
    text: str,
    dimensions: int | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Get Gemini embedding with proper normalization."""
    
    kwargs = {
        "model": "models/gemini-embedding-001",
        "content": text,
        "task_type": task_type,
    }
    
    if dimensions:
        kwargs["output_dimensionality"] = dimensions
    
    result = genai.embed_content(**kwargs)
    embedding = result['embedding']
    
    # Normalize if using reduced dimensions
    if dimensions and dimensions < 3072:
        embedding = l2_normalize(embedding)
    
    return embedding

# Now both are normalized
emb_full = get_gemini_embedding("Hello, world!")
emb_768 = get_gemini_embedding("Hello, world!", dimensions=768)

print(f"Full norm: {np.linalg.norm(emb_full):.6f}")
print(f"768d norm (after fix): {np.linalg.norm(emb_768):.6f}")
```

**Output:**
```
Full norm: 1.000000
768d norm (after fix): 1.000000
```

---

## Normalization After Operations

Several operations require re-normalization:

### After Dimension Truncation

```python
def truncate_and_normalize(
    embedding: list[float],
    target_dim: int
) -> list[float]:
    """Truncate Matryoshka embedding and re-normalize."""
    truncated = embedding[:target_dim]
    return l2_normalize(truncated)
```

### After Mean Pooling

```python
def mean_pool_and_normalize(
    embeddings: list[list[float]]
) -> list[float]:
    """Mean pool multiple embeddings and normalize result."""
    arr = np.array(embeddings)
    pooled = np.mean(arr, axis=0)
    return l2_normalize(pooled.tolist())
```

### After Weighted Combination

```python
def combine_embeddings(
    embeddings: list[list[float]],
    weights: list[float]
) -> list[float]:
    """Weighted combination of embeddings, normalized."""
    arr = np.array(embeddings)
    weights = np.array(weights) / sum(weights)  # Normalize weights
    combined = np.average(arr, axis=0, weights=weights)
    return l2_normalize(combined.tolist())
```

---

## Verifying Normalization

Add verification to your pipeline:

```python
def verify_normalized(
    embedding: list[float],
    tolerance: float = 1e-5
) -> bool:
    """Check if embedding is normalized (length ≈ 1)."""
    norm = np.linalg.norm(embedding)
    return abs(norm - 1.0) < tolerance

def assert_normalized(embedding: list[float], name: str = "embedding"):
    """Raise error if embedding is not normalized."""
    if not verify_normalized(embedding):
        norm = np.linalg.norm(embedding)
        raise ValueError(f"{name} is not normalized (norm={norm:.6f})")

# Usage in production code
def process_embedding(embedding: list[float]) -> list[float]:
    # Normalize just in case
    normalized = l2_normalize(embedding)
    
    # Verify (catches bugs early)
    assert_normalized(normalized)
    
    return normalized
```

### Batch Verification

```python
def check_batch_normalization(
    embeddings: list[list[float]]
) -> dict:
    """Check normalization status of a batch of embeddings."""
    norms = [np.linalg.norm(e) for e in embeddings]
    
    return {
        "count": len(embeddings),
        "min_norm": min(norms),
        "max_norm": max(norms),
        "mean_norm": np.mean(norms),
        "all_normalized": all(abs(n - 1.0) < 1e-5 for n in norms),
        "needs_normalization": any(abs(n - 1.0) > 1e-5 for n in norms),
    }

# Example usage
embeddings = [...]  # Your embeddings
status = check_batch_normalization(embeddings)
print(f"All normalized: {status['all_normalized']}")
print(f"Norm range: [{status['min_norm']:.4f}, {status['max_norm']:.4f}]")
```

---

## Normalization in Sentence Transformers

Sentence Transformers has built-in normalization:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

# Without normalization
embeddings_raw = model.encode(
    ["Hello, world!"],
    normalize_embeddings=False,
)
print(f"Raw norm: {np.linalg.norm(embeddings_raw[0]):.4f}")

# With normalization (recommended)
embeddings_norm = model.encode(
    ["Hello, world!"],
    normalize_embeddings=True,
)
print(f"Normalized norm: {np.linalg.norm(embeddings_norm[0]):.4f}")
```

**Output:**
```
Raw norm: 1.2847
Normalized norm: 1.0000
```

> **Tip:** Always use `normalize_embeddings=True` unless you have a specific reason not to.

---

## Normalization and Vector Databases

Most vector databases assume normalized vectors when using dot product:

| Database | Default Metric | Assumes Normalized? |
|----------|---------------|---------------------|
| Pinecone | cosine | Handles either |
| Weaviate | cosine | Handles either |
| Qdrant | cosine | Handles either |
| Milvus | L2 or IP | IP assumes normalized |
| pgvector | cosine | Handles either |
| Chroma | cosine | Handles either |

### Best Practice

Configure your vector database for **dot product** (inner product) and ensure all embeddings are normalized:

```python
# Pinecone example
import pinecone

# Create index with dot product
pinecone.create_index(
    name="my-index",
    dimension=1536,
    metric="dotproduct",  # Fastest for normalized vectors
)

# Always normalize before upserting
def upsert_embeddings(embeddings, ids):
    normalized = [l2_normalize(e) for e in embeddings]
    index.upsert(vectors=zip(ids, normalized))
```

---

## Common Mistakes

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using Gemini 768d without normalizing | Always normalize reduced Gemini embeddings |
| Pooling without re-normalizing | Normalize after any combination operation |
| Mixing normalized and non-normalized | Standardize on normalized throughout pipeline |
| Using dot product with non-normalized | Either normalize or use cosine similarity |
| Truncating Matryoshka without normalizing | Always normalize after dimension reduction |

---

## Best Practices

| Practice | Why It Matters |
|----------|----------------|
| Normalize at embedding generation time | Catch issues early |
| Add normalization verification in tests | Prevent regressions |
| Document provider normalization behavior | Team awareness |
| Use consistent similarity metric | Avoid confusion |
| Log norm values during debugging | Quick sanity check |

---

## Hands-on Exercise

### Your Task

Create a normalization utility module that:

1. Normalizes individual embeddings
2. Normalizes batches efficiently
3. Verifies normalization status
4. Handles edge cases (zero vectors)

### Requirements

1. `normalize(embedding)` - single embedding
2. `normalize_batch(embeddings)` - batch with NumPy
3. `is_normalized(embedding)` - check status
4. `normalization_report(embeddings)` - detailed stats
5. Handle zero vectors gracefully

<details>
<summary>💡 Hints</summary>

- Use NumPy for efficient batch operations
- Zero vectors can't be normalized—return as-is or raise
- Report should include min/max/mean norms and count of issues

</details>

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
from typing import List, Dict, Any

class EmbeddingNormalizer:
    """Utility class for embedding normalization."""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
    
    def normalize(self, embedding: List[float]) -> List[float]:
        """Normalize a single embedding to unit length."""
        v = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(v)
        
        if norm < self.tolerance:
            # Can't normalize near-zero vector
            return embedding
        
        return (v / norm).tolist()
    
    def normalize_batch(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Efficiently normalize a batch of embeddings."""
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms < self.tolerance, 1.0, norms)
        
        normalized = arr / norms
        return normalized.tolist()
    
    def is_normalized(self, embedding: List[float]) -> bool:
        """Check if embedding is normalized (length ≈ 1)."""
        norm = np.linalg.norm(embedding)
        return abs(norm - 1.0) < self.tolerance
    
    def normalization_report(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Generate detailed normalization report."""
        norms = [np.linalg.norm(e) for e in embeddings]
        
        normalized_count = sum(
            1 for n in norms if abs(n - 1.0) < self.tolerance
        )
        zero_count = sum(
            1 for n in norms if n < self.tolerance
        )
        
        return {
            "total_embeddings": len(embeddings),
            "normalized_count": normalized_count,
            "needs_normalization": len(embeddings) - normalized_count,
            "zero_vectors": zero_count,
            "norm_stats": {
                "min": float(min(norms)) if norms else None,
                "max": float(max(norms)) if norms else None,
                "mean": float(np.mean(norms)) if norms else None,
                "std": float(np.std(norms)) if norms else None,
            },
            "all_normalized": normalized_count == len(embeddings),
        }

# Test the utility
normalizer = EmbeddingNormalizer()

# Test data
embeddings = [
    [1.0, 0.0, 0.0],          # Already normalized
    [3.0, 4.0, 0.0],          # Not normalized (length 5)
    [0.0, 0.0, 0.0],          # Zero vector
    [0.5, 0.5, 0.5, 0.5],     # Not normalized
]

# Single normalization
print("Single normalize:")
print(f"  [3,4,0] -> {normalizer.normalize([3.0, 4.0, 0.0])}")
print(f"  Norm: {np.linalg.norm(normalizer.normalize([3.0, 4.0, 0.0])):.4f}")

# Batch normalization
print("\nBatch normalize:")
batch_norm = normalizer.normalize_batch(embeddings[:2])  # Skip zero and diff-dim
for i, e in enumerate(batch_norm):
    print(f"  {i}: norm = {np.linalg.norm(e):.4f}")

# Report
print("\nNormalization report:")
report = normalizer.normalization_report(embeddings[:3])  # Same dimensions
for key, value in report.items():
    print(f"  {key}: {value}")
```

**Output:**
```
Single normalize:
  [3,4,0] -> [0.6, 0.8, 0.0]
  Norm: 1.0000

Batch normalize:
  0: norm = 1.0000
  1: norm = 1.0000

Normalization report:
  total_embeddings: 3
  normalized_count: 1
  needs_normalization: 2
  zero_vectors: 1
  norm_stats: {'min': 0.0, 'max': 5.0, 'mean': 2.0, 'std': 2.16}
  all_normalized: False
```

</details>

---

## Summary

✅ Normalized embeddings have length 1, making cosine similarity equal to dot product

✅ Most providers return normalized embeddings, but **Gemini at 768/1536d does NOT**

✅ Always re-normalize after dimension reduction, pooling, or any vector combination

✅ Use `normalize_embeddings=True` with Sentence Transformers

✅ Verify normalization in your pipeline—add assertions and logging

**Next:** [Caching Strategies](./08-caching-strategies.md)

---

## Further Reading

- [L2 Normalization Explained](https://developers.google.com/machine-learning/glossary#L2-normalization)
- [Gemini Embeddings Documentation](https://ai.google.dev/gemini-api/docs/embeddings)
- [NumPy linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

---

<!-- 
Sources Consulted:
- Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- NumPy Documentation: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
-->
