---
title: "Quantization Trade-offs"
---

# Quantization Trade-offs

## Introduction

Quantization reduces storage and speeds up search, but it comes with precision trade-offs. Understanding these trade-offs helps you choose the right compression level for your use case—and implement strategies to mitigate quality loss.

This lesson examines the precision/compression curve, explains two-stage retrieval, and provides frameworks for making quantization decisions.

### What We'll Cover

- Precision loss per compression level
- The compression ratio vs quality curve
- Two-stage retrieval pattern
- When to accept trade-offs

### Prerequisites

- Understanding of [compression types](./02-compression-types.md)
- Basic knowledge of similarity search

---

## The Fundamental Trade-off

```
┌─────────────────────────────────────────────────────────────────┐
│              Compression vs Quality Trade-off                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Quality (Recall@10)                                            │
│     │                                                           │
│ 1.0 ┤ ●──────────●                                              │
│     │            ╲                                              │
│ 0.9 ┤             ●──────●                                      │
│     │                    ╲                                      │
│ 0.8 ┤                     ●                                     │
│     │                      ╲                                    │
│ 0.7 ┤                       ●                                   │
│     │                        ╲                                  │
│ 0.6 ┤                         ●                                 │
│     │                                                           │
│     └────┴────┴────┴────┴────┴────┴────►                       │
│          1x   2x   4x   8x  16x  32x   64x                     │
│                    Compression Ratio                            │
│                                                                 │
│  Sweet spot: 4x compression (int8) with ~99% quality           │
│  High compression: 32x (binary) needs rescoring                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Precision Loss by Method

### Scalar Quantization (int8)

**How it works:** Maps continuous float values to discrete integers.

| Metric | Float32 | Int8 | Loss |
|--------|---------|------|------|
| Recall@10 | 0.98 | 0.97 | ~1% |
| Recall@100 | 0.99 | 0.98 | ~1% |
| MRR | 0.95 | 0.94 | ~1% |

**Why loss is minimal:**
- 256 discrete levels usually capture similarity well
- Most embedding dimensions don't need full float precision
- Distance calculations are monotonic—ranking preserved

### Binary Quantization

**How it works:** Each dimension becomes a single bit (positive → 1, negative → 0).

| Metric | Float32 | Binary (raw) | Binary + rescore |
|--------|---------|--------------|------------------|
| Recall@10 | 0.98 | 0.82 | 0.96 |
| Recall@100 | 0.99 | 0.91 | 0.98 |
| MRR | 0.95 | 0.75 | 0.93 |

**Why loss is significant without rescoring:**
- Magnitude information is completely lost
- All positive values become identical (1)
- Fine-grained similarity distinctions disappear

### Product Quantization (PQ)

**How it works:** Divides vector into subvectors, quantizes each to a codebook ID.

| Configuration | Compression | Recall@10 | Notes |
|---------------|-------------|-----------|-------|
| m=8, bits=8 | 32x | ~0.85 | Moderate quality |
| m=16, bits=8 | 16x | ~0.90 | Better quality |
| m=32, bits=8 | 8x | ~0.95 | Near-optimal |
| m=64, bits=8 | 4x | ~0.97 | Quality close to int8 |

More subvectors (m) = better quality but less compression.

---

## Understanding the Quality Curve

### Why Quality Degrades Non-Linearly

```
┌─────────────────────────────────────────────────────────────────┐
│              Information Loss Visualization                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FLOAT32: Full precision                                        │
│  [0.234, -0.156, 0.512, -0.089, 0.421, -0.312, 0.178, -0.067]  │
│     ↓                                                           │
│  INT8: Discretized (256 levels)                                │
│  [23, -15, 51, -8, 42, -31, 17, -6]                            │
│     ↓                                                           │
│  BINARY: Sign only (2 levels)                                   │
│  [1, 0, 1, 0, 1, 0, 1, 0]                                       │
│                                                                 │
│  Notice: Binary loses that 0.234 ≠ 0.512 ≠ 0.421               │
│          They all become '1' — magnitude is gone               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Insight: High Dimensions Help Binary

Binary quantization works better with more dimensions:

| Dimensions | Binary Quality | Reason |
|------------|----------------|--------|
| 128 | Poor (~70%) | Too few bits to capture similarity |
| 384 | Moderate (~80%) | More bits, better discrimination |
| 768 | Good (~88%) | Statistical patterns emerge |
| 1536 | Very good (~92%) | High-dimensional sign patterns reliable |
| 3072 | Excellent (~95%) | Abundant bits compensate for magnitude loss |

> **Key insight:** Modern high-dimensional embeddings (1536+) are specifically designed to work well with binary quantization.

---

## The Two-Stage Retrieval Pattern

### The Core Idea

Use compressed embeddings for fast initial retrieval, then rescore with full precision:

```
┌─────────────────────────────────────────────────────────────────┐
│              Two-Stage Retrieval Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         1 Million Documents                     │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           STAGE 1: Fast Coarse Search                    │   │
│  │  • Binary/Int8 compressed embeddings                     │   │
│  │  • Hamming distance (binary) or L2 (int8)               │   │
│  │  • Return top 100-1000 candidates                        │   │
│  │  • Time: ~10ms                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                        100 candidates                           │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           STAGE 2: Precise Reranking                     │   │
│  │  • Full float32 embeddings                               │   │
│  │  • Cosine similarity                                     │   │
│  │  • Return top 10                                         │   │
│  │  • Time: ~5ms                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                          10 results                             │
│                                                                 │
│  Total time: ~15ms (vs ~100ms for full float32 scan)           │
│  Quality: 95%+ of pure float32 search                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import numpy as np
from typing import List, Tuple

def two_stage_search(
    query_binary: np.ndarray,
    query_float: np.ndarray,
    doc_binary: np.ndarray,
    doc_float: np.ndarray,
    doc_ids: List[str],
    top_k: int = 10,
    candidates: int = 100
) -> List[Tuple[str, float]]:
    """
    Two-stage retrieval: fast binary + precise float reranking.
    """
    # Stage 1: Fast Hamming distance with binary embeddings
    # XOR and popcount for Hamming distance
    hamming_distances = np.unpackbits(
        np.bitwise_xor(query_binary, doc_binary), axis=1
    ).sum(axis=1)
    
    # Get top candidates (lowest Hamming distance)
    candidate_indices = np.argsort(hamming_distances)[:candidates]
    
    # Stage 2: Precise cosine similarity with float embeddings
    candidate_floats = doc_float[candidate_indices]
    
    # Normalize for cosine similarity
    query_norm = query_float / np.linalg.norm(query_float)
    candidate_norms = candidate_floats / np.linalg.norm(
        candidate_floats, axis=1, keepdims=True
    )
    
    # Compute similarities
    similarities = np.dot(candidate_norms, query_norm)
    
    # Get top-k from candidates
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        (doc_ids[candidate_indices[i]], similarities[i])
        for i in top_indices
    ]
    
    return results
```

### Choosing the Candidate Count

| Scenario | Candidates | Why |
|----------|------------|-----|
| High precision needed | 500-1000 | More candidates = higher recall |
| Speed critical | 50-100 | Fewer candidates = faster reranking |
| Binary embeddings | 200-500 | Binary needs more candidates |
| Int8 embeddings | 50-100 | Int8 ranking is already good |

Rule of thumb: **candidates = 10-20x top_k**

---

## Oversampling Strategy

### What It Is

Oversampling retrieves more candidates than needed from compressed search, then filters with full precision:

```python
# Without oversampling
results = search(query, top_k=10)  # May miss relevant docs

# With oversampling (2x)
candidates = search_compressed(query, top_k=20)  # Get 2x candidates
results = rescore_float(candidates, query, top_k=10)  # Filter to 10
```

### Oversampling Factor Guidelines

| Compression | Oversampling | Effective Recall |
|-------------|--------------|------------------|
| Int8 | 1.2-1.5x | ~99% |
| Binary (1-bit) | 2-3x | ~96% |
| Binary (1.5-bit) | 1.5-2x | ~97% |
| Product Quantization | 2-4x | ~90-95% |

### Qdrant Example

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Search with oversampling for compressed index
results = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10,
    search_params={
        "quantization": {
            "rescore": True,           # Enable rescoring
            "oversampling": 2.0        # 2x oversampling
        }
    }
)
```

---

## When to Accept Trade-offs

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Quantization Decision Tree                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  How many vectors?                                              │
│       │                                                         │
│       ├── < 100K ──────► No compression needed                  │
│       │                                                         │
│       ├── 100K - 10M ──► Int8 (4x, minimal quality loss)       │
│       │                                                         │
│       ├── 10M - 100M ──► Int8 + dimension reduction            │
│       │                  OR Binary + rescoring                  │
│       │                                                         │
│       └── > 100M ──────► Binary first stage + float rescore    │
│                          OR Product Quantization (PQ)           │
│                                                                 │
│  Can you afford quality loss?                                   │
│       │                                                         │
│       ├── No  → Int8 only (4x, ~99% quality)                   │
│       │                                                         │
│       ├── <5% → Binary + rescore (32x, ~96% quality)           │
│       │                                                         │
│       └── <10% → Product Quantization (64x, ~90% quality)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Use Case Recommendations

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| E-commerce search | Int8 | Quality critical for conversions |
| Document search | Binary + rescore | High volume, can tolerate reranking |
| Recommendation | Int8 or Binary | Depends on precision requirements |
| Semantic cache | Binary | Speed matters, fuzzy matching OK |
| Duplicate detection | Binary | Exact matching not needed |
| Legal/Medical | Float32 or Int8 | Accuracy is paramount |

---

## Measuring Quality Impact

### Before You Deploy

Always benchmark on **your** data with **your** queries:

```python
import numpy as np
from typing import List

def measure_recall(
    ground_truth: List[List[str]],
    predictions: List[List[str]],
    k: int = 10
) -> float:
    """
    Measure Recall@K: what fraction of true top-K 
    appears in predicted top-K?
    """
    recalls = []
    for true_ids, pred_ids in zip(ground_truth, predictions):
        true_set = set(true_ids[:k])
        pred_set = set(pred_ids[:k])
        recall = len(true_set & pred_set) / len(true_set)
        recalls.append(recall)
    return np.mean(recalls)

# Compare float32 baseline vs compressed
float_results = search_float32(queries)    # Ground truth
int8_results = search_int8(queries)        # 4x compressed
binary_results = search_binary(queries)    # 32x compressed

print(f"Float32 (baseline): Recall@10 = 1.000")
print(f"Int8: Recall@10 = {measure_recall(float_results, int8_results):.3f}")
print(f"Binary: Recall@10 = {measure_recall(float_results, binary_results):.3f}")
```

**Expected output:**
```
Float32 (baseline): Recall@10 = 1.000
Int8: Recall@10 = 0.987
Binary: Recall@10 = 0.823
```

### Key Metrics to Track

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Recall@K | Fraction of true top-K found | > 0.95 |
| MRR | Mean Reciprocal Rank | > 0.90 |
| NDCG@K | Ranking quality | > 0.90 |
| Latency P50/P99 | Query speed | Application-dependent |

---

## Summary

✅ **Int8 (4x)** provides ~99% quality with minimal tuning required  
✅ **Binary (32x)** needs rescoring but enables massive scale  
✅ **Two-stage retrieval** combines speed of compressed + quality of float  
✅ **Oversampling** compensates for quality loss (2-3x for binary)  
✅ **Always benchmark** on your data before deploying compression

---

**Next:** [Binary Embeddings →](./04-binary-embeddings.md)

---

<!-- 
Sources Consulted:
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Cohere Binary Embeddings: https://docs.cohere.com/docs/embeddings
-->
