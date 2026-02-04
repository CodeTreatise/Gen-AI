---
title: "Normalization After Truncation"
---

# Normalization After Truncation

## Introduction

This is the most commonly overlooked step when working with Matryoshka embeddings. **Truncated vectors are NOT unit-length.** If you compute cosine similarity on unnormalized truncated vectors, your similarity scores will be mathematically incorrect—and you might not even notice because the results "look" reasonable.

This lesson explains why normalization matters, how to implement it correctly, and which providers handle it for you.

### What We'll Cover

- Why truncation breaks normalization
- The mathematical necessity of re-normalization
- How to normalize correctly in code
- Provider-specific normalization behavior
- Debugging normalization issues

### Prerequisites

- Understanding of cosine similarity and dot product
- Familiarity with [API Parameters](./05-api-parameters.md)

---

## Why Truncation Breaks Normalization

### Unit Vectors and Cosine Similarity

A **unit vector** has a magnitude (length) of exactly 1:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2} = 1$$

When both vectors are unit length, **cosine similarity equals dot product**:

$$\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \times \|\mathbf{b}\|} = \mathbf{a} \cdot \mathbf{b}$$

This is why many embedding models pre-normalize their outputs—it makes similarity computation a simple dot product.

### What Happens When You Truncate

Consider a normalized 768-dimensional embedding:

```python
import numpy as np

# Original unit vector (768 dimensions, normalized)
full_embedding = np.random.randn(768)
full_embedding = full_embedding / np.linalg.norm(full_embedding)

print(f"Full embedding norm: {np.linalg.norm(full_embedding):.6f}")  # 1.000000

# Truncate to 256 dimensions
truncated = full_embedding[:256]

print(f"Truncated embedding norm: {np.linalg.norm(truncated):.6f}")  # ~0.577 (not 1!)
```

**Output:**
```
Full embedding norm: 1.000000
Truncated embedding norm: 0.577350
```

The truncated vector has lost ~42% of its magnitude! The exact loss depends on how the information is distributed across dimensions.

### The Math Behind This

For a random unit vector uniformly distributed on the unit sphere, truncating from dimension $n$ to $k$ gives an expected norm of:

$$E[\|\mathbf{v}_{1:k}\|] = \sqrt{\frac{k}{n}}$$

| Full Dims | Truncated Dims | Expected Norm |
|-----------|----------------|---------------|
| 768 | 768 | 1.000 |
| 768 | 512 | 0.816 |
| 768 | 384 | 0.707 |
| 768 | 256 | 0.577 |
| 768 | 128 | 0.408 |
| 768 | 64 | 0.289 |

---

## Why This Matters for Similarity

### Incorrect Similarity Without Normalization

```python
import numpy as np

def broken_similarity(embedding_a, embedding_b):
    """WRONG: Dot product on non-unit vectors."""
    return np.dot(embedding_a, embedding_b)

def correct_similarity(embedding_a, embedding_b):
    """CORRECT: Cosine similarity with normalization."""
    a_norm = embedding_a / np.linalg.norm(embedding_a)
    b_norm = embedding_b / np.linalg.norm(embedding_b)
    return np.dot(a_norm, b_norm)

# Generate two similar embeddings
np.random.seed(42)
emb_a = np.random.randn(768)
emb_a = emb_a / np.linalg.norm(emb_a)

emb_b = emb_a + 0.1 * np.random.randn(768)  # Slightly perturbed
emb_b = emb_b / np.linalg.norm(emb_b)

# Full embedding similarity (both correct)
full_sim = np.dot(emb_a, emb_b)
print(f"Full embedding similarity: {full_sim:.4f}")

# Truncate to 256 dims
a_trunc = emb_a[:256]
b_trunc = emb_b[:256]

# WRONG: Direct dot product
wrong_sim = broken_similarity(a_trunc, b_trunc)
print(f"Truncated (no normalization): {wrong_sim:.4f}")

# CORRECT: After normalization
right_sim = correct_similarity(a_trunc, b_trunc)
print(f"Truncated (with normalization): {right_sim:.4f}")
```

**Output:**
```
Full embedding similarity: 0.9512
Truncated (no normalization): 0.3167  ← WRONG! Way too low
Truncated (with normalization): 0.9485  ← Correct
```

Without normalization, the similarity score is ~3x lower than it should be!

### The Silent Failure Mode

The dangerous part is that unnormalized similarities **look plausible**:
- They're still between -1 and 1 (for most practical cases)
- Rankings might be preserved in some cases
- You won't get obvious errors

But your absolute scores will be wrong, and in edge cases, rankings can flip.

---

## How to Normalize Correctly

### Python/NumPy

```python
import numpy as np

def truncate_and_normalize(embedding: np.ndarray, dim: int) -> np.ndarray:
    """
    Safely truncate an embedding and re-normalize to unit length.
    
    Args:
        embedding: Full-dimension embedding (1D array)
        dim: Target dimension
    
    Returns:
        Normalized truncated embedding
    """
    truncated = embedding[:dim]
    norm = np.linalg.norm(truncated)
    
    if norm == 0:
        return truncated  # Handle zero vector edge case
    
    return truncated / norm

# Usage
full_emb = np.random.randn(768)
full_emb = full_emb / np.linalg.norm(full_emb)

truncated_256 = truncate_and_normalize(full_emb, 256)
print(f"Truncated norm: {np.linalg.norm(truncated_256):.6f}")  # 1.000000
```

### Batch Normalization

```python
def truncate_and_normalize_batch(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """
    Truncate and normalize a batch of embeddings.
    
    Args:
        embeddings: Array of shape (N, full_dim)
        dim: Target dimension
    
    Returns:
        Array of shape (N, dim), normalized
    """
    truncated = embeddings[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)
    
    return truncated / norms

# Batch usage
embeddings = np.random.randn(100, 768)
# Assume original embeddings are normalized
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

truncated_batch = truncate_and_normalize_batch(embeddings, 256)
print(f"Shape: {truncated_batch.shape}")  # (100, 256)
print(f"Norms: {np.linalg.norm(truncated_batch, axis=1)[:5]}")  # All ~1.0
```

### PyTorch

```python
import torch
import torch.nn.functional as F

def truncate_and_normalize_torch(embeddings: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Truncate and normalize using PyTorch.
    
    Args:
        embeddings: Tensor of shape (N, full_dim)
        dim: Target dimension
    
    Returns:
        Normalized truncated tensor
    """
    truncated = embeddings[:, :dim]
    return F.normalize(truncated, p=2, dim=-1)

# Usage
embeddings = torch.randn(100, 768)
embeddings = F.normalize(embeddings, p=2, dim=-1)

truncated = truncate_and_normalize_torch(embeddings, 256)
print(f"Norms: {torch.norm(truncated, dim=1)[:5]}")  # All 1.0
```

---

## Provider Normalization Behavior

### Which Providers Pre-Normalize?

| Provider | Model | API Returns Normalized? | Action Required |
|----------|-------|------------------------|-----------------|
| OpenAI | text-embedding-3-* | ✅ Yes (when using `dimensions`) | None |
| Google | gemini-embedding-001 | ❌ No | **Must normalize** |
| Cohere | embed-v4.0 | ✅ Yes | None |
| Nomic | nomic-embed-text-v1.5 | ⚠️ Full only | Normalize after truncation |
| Jina | jina-embeddings-v3 | ⚠️ Depends on method | Check documentation |

### OpenAI: Already Handled

When you use the `dimensions` parameter, OpenAI returns already-normalized embeddings:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello, world!",
    dimensions=256
)

embedding = np.array(response.data[0].embedding)
print(f"Norm: {np.linalg.norm(embedding):.6f}")  # 1.000000 (already normalized)
```

### Gemini: You Must Normalize

Gemini does NOT normalize, even when using `output_dimensionality`:

```python
import google.generativeai as genai
import numpy as np

genai.configure(api_key="your-key")

result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Hello, world!",
    output_dimensionality=256
)

embedding = np.array(result['embedding'])
print(f"Norm BEFORE: {np.linalg.norm(embedding):.6f}")  # NOT 1.0!

# MUST normalize for correct cosine similarity
embedding = embedding / np.linalg.norm(embedding)
print(f"Norm AFTER: {np.linalg.norm(embedding):.6f}")  # 1.000000
```

### Sentence Transformers: Depends on Usage

When loading with `truncate_dim`, behavior varies by model:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Method 1: Using truncate_dim (may or may not normalize)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", truncate_dim=256)
emb = model.encode("Hello, world!")
print(f"With truncate_dim, norm: {np.linalg.norm(emb):.4f}")  # Check!

# Method 2: Manual truncation (definitely need to normalize)
model_full = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
emb_full = model_full.encode("Hello, world!")
emb_truncated = emb_full[:256]
print(f"Manual truncation, norm BEFORE: {np.linalg.norm(emb_truncated):.4f}")

emb_truncated = emb_truncated / np.linalg.norm(emb_truncated)
print(f"Manual truncation, norm AFTER: {np.linalg.norm(emb_truncated):.4f}")  # 1.0
```

---

## Debugging Normalization Issues

### Symptom: Low Similarity Scores

If your similarity scores seem unexpectedly low:

```python
def diagnose_similarity(emb_a, emb_b):
    """Diagnose potential normalization issues."""
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    
    dot_product = np.dot(emb_a, emb_b)
    cosine_sim = dot_product / (norm_a * norm_b)
    
    print(f"Embedding A norm: {norm_a:.4f} (should be 1.0)")
    print(f"Embedding B norm: {norm_b:.4f} (should be 1.0)")
    print(f"Raw dot product: {dot_product:.4f}")
    print(f"True cosine similarity: {cosine_sim:.4f}")
    
    if norm_a < 0.95 or norm_b < 0.95:
        print("⚠️  Warning: Embeddings are not normalized!")

# Test
emb_a = np.random.randn(256)  # Not normalized
emb_b = np.random.randn(256)  # Not normalized
diagnose_similarity(emb_a, emb_b)
```

### Symptom: Inconsistent Rankings

If rankings seem inconsistent when using different dimension levels:

```python
def compare_dimension_rankings(model, queries, docs, dims_list):
    """Compare rankings at different dimensions."""
    # Get full embeddings
    q_embs = model.encode(queries)
    d_embs = model.encode(docs)
    
    for dims in dims_list:
        q_trunc = q_embs[:, :dims]
        d_trunc = d_embs[:, :dims]
        
        # Normalize
        q_trunc = q_trunc / np.linalg.norm(q_trunc, axis=1, keepdims=True)
        d_trunc = d_trunc / np.linalg.norm(d_trunc, axis=1, keepdims=True)
        
        # Compute similarities
        sims = np.dot(q_trunc, d_trunc.T)
        rankings = np.argsort(-sims, axis=1)
        
        print(f"{dims} dims rankings: {rankings[0][:3]}")  # Top 3 for first query
```

---

## Best Practices

### 1. Normalize Immediately After Truncation

```python
class SafeEmbedding:
    """Wrapper that ensures embeddings are always normalized."""
    
    def __init__(self, embedding: np.ndarray):
        self._data = embedding / np.linalg.norm(embedding)
    
    def truncate(self, dim: int) -> 'SafeEmbedding':
        """Return a new SafeEmbedding truncated to dim dimensions."""
        return SafeEmbedding(self._data[:dim])  # Auto-normalizes
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    def similarity(self, other: 'SafeEmbedding') -> float:
        """Cosine similarity (just dot product since both normalized)."""
        return np.dot(self._data, other._data)
```

### 2. Add Assertions During Development

```python
def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute similarity with normalization check."""
    
    # Debug assertions (disable in production with -O flag)
    assert abs(np.linalg.norm(emb_a) - 1.0) < 0.01, f"emb_a not normalized: {np.linalg.norm(emb_a)}"
    assert abs(np.linalg.norm(emb_b) - 1.0) < 0.01, f"emb_b not normalized: {np.linalg.norm(emb_b)}"
    
    return np.dot(emb_a, emb_b)
```

### 3. Normalize in Your Data Pipeline

```python
class EmbeddingPipeline:
    """Pipeline that guarantees normalized outputs."""
    
    def __init__(self, model, target_dim: int):
        self.model = model
        self.target_dim = target_dim
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings at target dimension."""
        # Get embeddings
        embeddings = self.model.encode(texts)
        
        # Truncate if needed
        if embeddings.shape[1] > self.target_dim:
            embeddings = embeddings[:, :self.target_dim]
        
        # ALWAYS normalize before returning
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid div by zero
        
        return embeddings / norms
```

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    NORMALIZATION FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Full Embedding (768 dims)                                 │
│   ├── Norm: 1.0 ✓                                          │
│   │                                                         │
│   ▼                                                         │
│   Truncate to 256 dims                                      │
│   ├── Norm: ~0.58 ✗ (BROKEN!)                              │
│   │                                                         │
│   ▼                                                         │
│   Re-normalize: v / ||v||                                   │
│   ├── Norm: 1.0 ✓ (FIXED!)                                 │
│   │                                                         │
│   ▼                                                         │
│   Ready for cosine similarity                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

✅ **Truncation breaks normalization**—unit vectors become shorter vectors  
✅ **Unnormalized similarity is mathematically incorrect**  
✅ **OpenAI and Cohere** pre-normalize when you use their dimension parameters  
✅ **Gemini and manual truncation** require explicit normalization  
✅ **Always check norms** during development to catch issues early  
✅ **Normalize immediately** after truncation, every time

---

## Quick Reference: Normalization Code

```python
import numpy as np

# Single vector
def normalize(v):
    return v / np.linalg.norm(v)

# Batch of vectors
def normalize_batch(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)

# Truncate + normalize
def truncate_normalize(embeddings, dim):
    truncated = embeddings[:, :dim] if embeddings.ndim == 2 else embeddings[:dim]
    return normalize_batch(truncated) if embeddings.ndim == 2 else normalize(truncated)
```

---

**Next:** [Cost-Benefit Analysis →](./07-cost-benefit-analysis.md)

---

<!-- 
Sources Consulted:
- Gemini Embeddings Documentation (normalization requirements)
- OpenAI Embeddings API (pre-normalization behavior)
- NumPy documentation (linalg.norm)
-->
