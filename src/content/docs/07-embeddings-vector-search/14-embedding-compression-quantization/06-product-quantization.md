---
title: "Product Quantization (PQ)"
---

# Product Quantization (PQ)

## Introduction

Product Quantization is the most aggressive compression technique for embeddings, achieving up to 64x compression. Unlike scalar or binary quantization which work on individual values, PQ divides vectors into segments and quantizes each segment to a small codebook of representative patterns.

This lesson explains how PQ works, when to use it, and how to configure it effectively.

### What We'll Cover

- How Product Quantization works
- Codebooks and subvector training
- Asymmetric distance computation
- Tuning PQ parameters for quality vs compression

### Prerequisites

- Understanding of [quantization trade-offs](./03-quantization-tradeoffs.md)
- Familiarity with clustering concepts (k-means)

---

## How Product Quantization Works

### The Core Idea

Instead of quantizing each dimension independently, PQ:
1. Splits the vector into equal-sized segments (subvectors)
2. Learns a codebook of representative patterns for each segment
3. Replaces each segment with the ID of its nearest codebook entry

```
┌─────────────────────────────────────────────────────────────────┐
│              Product Quantization Process                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original vector (1536 dimensions):                             │
│  [0.12, -0.34, 0.56, ... 1536 values ...]                      │
│                                                                 │
│  Step 1: Split into m=8 subvectors (1536/8 = 192 dims each):   │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬───┐   │
│  │ Sub 1  │ Sub 2  │ Sub 3  │ Sub 4  │ Sub 5  │ Sub 6  │...│   │
│  │192 dims│192 dims│192 dims│192 dims│192 dims│192 dims│   │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┴───┘   │
│                                                                 │
│  Step 2: Each subvector → nearest codebook entry ID:           │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬───┐   │
│  │  42    │  156   │   7    │  201   │  89    │  134   │...│   │
│  │(1 byte)│(1 byte)│(1 byte)│(1 byte)│(1 byte)│(1 byte)│   │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┴───┘   │
│                                                                 │
│  Result: 1536 × 4 = 6144 bytes → 8 bytes                       │
│  Compression: 768x (with m=8, 256 centroids per codebook)      │
│                                                                 │
│  Practical compression accounting for codebook storage:         │
│  ~64x for large collections                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why "Product"?

The name comes from representing the quantization space as a Cartesian product of smaller spaces:

```
Full space: 2^32 possible float32 values × 1536 dimensions
           = Astronomical number of possible vectors

PQ space: 256 codes × 256 codes × ... × 256 codes (m times)
        = 256^m possible quantized vectors

With m=8: 256^8 ≈ 1.8 × 10^19 possible combinations
         Much smaller than original, but still rich
```

---

## Training Codebooks

### K-Means Clustering

Each subvector space has its own codebook, learned via k-means:

```python
import numpy as np
from sklearn.cluster import KMeans

def train_pq_codebooks(
    vectors: np.ndarray,      # Training vectors (n_samples, dim)
    m: int = 8,               # Number of subvectors
    k: int = 256,             # Codebook size (centroids per subvector)
    sample_size: int = 100000 # Training samples
) -> list:
    """
    Train Product Quantization codebooks.
    
    Returns:
        List of m codebooks, each shape (k, dim/m)
    """
    n_samples, dim = vectors.shape
    assert dim % m == 0, f"Dimension {dim} not divisible by m={m}"
    
    subvector_dim = dim // m
    
    # Sample training data if needed
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        vectors = vectors[indices]
    
    codebooks = []
    
    for i in range(m):
        # Extract subvector i from all training vectors
        start = i * subvector_dim
        end = (i + 1) * subvector_dim
        subvectors = vectors[:, start:end]
        
        # Train k-means to find k centroids
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(subvectors)
        
        codebooks.append(kmeans.cluster_centers_)
        print(f"Trained codebook {i+1}/{m}")
    
    return codebooks
```

### Encoding Vectors

Once codebooks are trained, encode vectors by finding nearest centroids:

```python
def encode_pq(
    vectors: np.ndarray,       # Vectors to encode (n, dim)
    codebooks: list            # Trained codebooks
) -> np.ndarray:
    """
    Encode vectors using Product Quantization.
    
    Returns:
        Encoded vectors as uint8 array (n, m)
    """
    n_samples, dim = vectors.shape
    m = len(codebooks)
    subvector_dim = dim // m
    
    codes = np.zeros((n_samples, m), dtype=np.uint8)
    
    for i, codebook in enumerate(codebooks):
        start = i * subvector_dim
        end = (i + 1) * subvector_dim
        subvectors = vectors[:, start:end]
        
        # Find nearest centroid for each subvector
        # Using L2 distance
        distances = np.linalg.norm(
            subvectors[:, np.newaxis, :] - codebook[np.newaxis, :, :],
            axis=2
        )
        codes[:, i] = np.argmin(distances, axis=1)
    
    return codes
```

---

## Distance Computation

### Symmetric vs Asymmetric Distance

Two approaches for computing distances with PQ:

```
┌─────────────────────────────────────────────────────────────────┐
│              PQ Distance Methods                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SYMMETRIC DISTANCE (SDC):                                      │
│  - Both query and database vectors are quantized               │
│  - Compare code to code                                         │
│  - Faster but less accurate                                     │
│                                                                 │
│  Query:    [42, 156, 7, ...]  (quantized)                      │
│  Database: [38, 156, 12, ...] (quantized)                      │
│  Distance: Lookup table(42,38) + Lookup(156,156) + ...         │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  ASYMMETRIC DISTANCE (ADC):                                     │
│  - Query is NOT quantized (original float)                     │
│  - Database vectors are quantized                              │
│  - More accurate                                                │
│                                                                 │
│  Query:    [0.12, -0.34, 0.56, ...] (original)                 │
│  Database: [38, 156, 12, ...]        (quantized)               │
│  Distance: dist(query_sub1, centroid[38]) + ...                │
│                                                                 │
│  ADC is standard in production systems                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Asymmetric Distance Implementation

```python
def compute_distance_table(
    query: np.ndarray,         # Single query vector (dim,)
    codebooks: list            # Trained codebooks
) -> np.ndarray:
    """
    Precompute distances from query subvectors to all centroids.
    
    Returns:
        Distance table of shape (m, k) where k is codebook size
    """
    m = len(codebooks)
    k = codebooks[0].shape[0]
    subvector_dim = len(query) // m
    
    distance_table = np.zeros((m, k))
    
    for i, codebook in enumerate(codebooks):
        start = i * subvector_dim
        end = (i + 1) * subvector_dim
        query_sub = query[start:end]
        
        # Squared L2 distance from query subvector to all centroids
        distance_table[i] = np.sum(
            (codebook - query_sub) ** 2, axis=1
        )
    
    return distance_table


def search_pq(
    query: np.ndarray,         # Query vector (dim,)
    database_codes: np.ndarray,# Encoded database (n, m)
    codebooks: list,
    top_k: int = 10
) -> tuple:
    """
    Asymmetric distance search with PQ.
    
    Returns:
        (indices, distances) of top-k nearest neighbors
    """
    # Precompute distance table
    dist_table = compute_distance_table(query, codebooks)
    
    # Compute total distance for each database vector
    n_samples, m = database_codes.shape
    distances = np.zeros(n_samples)
    
    for i in range(m):
        # Look up precomputed distances for subvector i
        distances += dist_table[i, database_codes[:, i]]
    
    # Find top-k (smallest distances)
    top_indices = np.argpartition(distances, top_k)[:top_k]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    
    return top_indices, distances[top_indices]
```

### Why ADC is Fast

The key insight: precompute query-to-centroid distances once, then lookup:

```
Without precomputation:
- For each database vector: m × k distance calculations
- Total: n × m × k operations

With precomputation (ADC):
- Build table: m × k distance calculations
- For each database vector: m lookups + m additions
- Total: m × k + n × 2m operations

For 1M vectors, m=8, k=256:
- Without: 1M × 8 × 256 = 2 billion operations
- With ADC: 2048 + 1M × 16 = 16 million operations

Speedup: ~125x
```

---

## Tuning PQ Parameters

### Key Parameters

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `m` | Number of subvectors | More = better quality, less compression |
| `k` (nbits) | Codebook size (2^nbits) | More = better quality, larger tables |
| Training data | Vectors used for codebook training | More = better codebook, longer training |

### Compression Calculations

```
┌─────────────────────────────────────────────────────────────────┐
│              PQ Compression Formula                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original size = dim × 4 bytes (float32)                       │
│  Compressed size = m × ceil(log2(k) / 8) bytes                 │
│                                                                 │
│  With k=256 (8 bits = 1 byte per code):                        │
│  Compressed size = m bytes                                      │
│                                                                 │
│  Examples (dim=1536):                                          │
│                                                                 │
│  m    Compressed    Compression Ratio                          │
│  ──────────────────────────────────────                        │
│  8    8 bytes       768x                                       │
│  16   16 bytes      384x                                       │
│  32   32 bytes      192x                                       │
│  48   48 bytes      128x                                       │
│  64   64 bytes      96x                                        │
│  96   96 bytes      64x                                        │
│  192  192 bytes     32x                                        │
│                                                                 │
│  Note: dim must be divisible by m                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quality vs Compression Guidelines

| Use Case | m | Expected Recall@10 | Compression |
|----------|---|-------------------|-------------|
| Aggressive | 8-16 | 70-80% | 96-384x |
| Balanced | 32-48 | 85-90% | 32-64x |
| Quality-focused | 64-96 | 92-96% | 16-32x |
| Conservative | 128+ | 97%+ | 8-16x |

### Milvus PQ Configuration

```python
# Milvus IVF_PQ index
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "IP",
    "params": {
        "nlist": 1024,      # IVF clusters
        "m": 48,            # 1536 / 48 = 32 dims per subvector
        "nbits": 8          # 256 centroids per codebook
    }
}

# For HNSW_PQ
index_params = {
    "index_type": "HNSW_PQ",
    "metric_type": "IP",
    "params": {
        "M": 16,
        "efConstruction": 200,
        "m": 48,
        "nbits": 8
    }
}
```

### Qdrant PQ Configuration

```python
from qdrant_client.models import (
    ProductQuantization, ProductQuantizationConfig
)

# Qdrant uses compression ratio instead of m
client.create_collection(
    collection_name="documents_pq",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=ProductQuantizationConfig.CompressionRatio.X32,
            always_ram=True
        )
    )
)

# Available compression ratios
# X4, X8, X16, X32, X64
```

---

## OPQ: Optimized Product Quantization

### The Problem with PQ

Standard PQ assumes subvectors are independent, but they're not. Adjacent dimensions are often correlated.

### OPQ Solution

Apply a rotation matrix before PQ to minimize quantization error:

```
┌─────────────────────────────────────────────────────────────────┐
│              OPQ (Optimized Product Quantization)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard PQ:                                                   │
│  vector → split into m subvectors → quantize each              │
│                                                                 │
│  OPQ:                                                           │
│  vector → rotate with matrix R → split → quantize              │
│                                                                 │
│  The rotation R is learned to minimize reconstruction error    │
│  Alternating optimization:                                      │
│  1. Fix R, train codebooks                                     │
│  2. Fix codebooks, optimize R                                  │
│  3. Repeat until convergence                                   │
│                                                                 │
│  Result: 10-20% better recall at same compression              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Using OPQ in Milvus

```python
# Milvus automatically uses OPQ with IVF_PQ
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "IP",
    "params": {
        "nlist": 1024,
        "m": 48,
        "nbits": 8
        # OPQ is applied automatically during training
    }
}
```

---

## Residual Product Quantization (RPQ)

### Progressive Refinement

Instead of one-shot quantization, encode residuals (errors) with additional codebooks:

```
┌─────────────────────────────────────────────────────────────────┐
│              Residual Product Quantization                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1 (Coarse PQ):                                          │
│  vector → quantize → reconstruction₁                           │
│  residual₁ = vector - reconstruction₁                          │
│                                                                 │
│  Level 2 (Fine PQ):                                            │
│  residual₁ → quantize → reconstruction₂                        │
│  residual₂ = residual₁ - reconstruction₂                       │
│                                                                 │
│  Final reconstruction:                                          │
│  reconstruction₁ + reconstruction₂                             │
│                                                                 │
│  Result: Better accuracy at same memory cost                   │
│          (or same accuracy at lower memory)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HNSW_PRQ in Milvus

```python
# Progressive Residual Quantization
index_params = {
    "index_type": "HNSW_PRQ",
    "metric_type": "IP",
    "params": {
        "M": 16,
        "efConstruction": 200,
        "m": 48,
        "nbits": 8
    }
}
```

---

## When to Use PQ

### Good Candidates

| Scenario | Why PQ Works |
|----------|--------------|
| Billions of vectors | Storage savings are massive |
| Cost-sensitive | Minimize cloud storage costs |
| Memory-constrained | Fit more in RAM |
| Recall trade-off OK | ~90% recall acceptable |
| Cold/archive data | Access patterns tolerate latency |

### Poor Candidates

| Scenario | Why Avoid PQ |
|----------|--------------|
| < 1M vectors | Simpler methods sufficient |
| Maximum accuracy | ~95%+ recall required |
| Frequent updates | Codebook training is expensive |
| Low latency critical | Distance table computation adds overhead |

### PQ vs Other Methods

| Method | Compression | Recall | Complexity |
|--------|-------------|--------|------------|
| Scalar (int8) | 4x | ~99% | Simple |
| Binary | 32x | ~82% (raw), ~95% (rescore) | Simple |
| PQ (m=32) | ~64x | ~90% | Complex |
| PQ (m=96) | ~32x | ~95% | Complex |

---

## Summary

✅ **PQ divides vectors into subvectors** and quantizes each to a codebook ID  
✅ **Codebooks are trained via k-means** on representative data  
✅ **ADC (Asymmetric Distance)** precomputes query-centroid distances for fast search  
✅ **Tune `m` parameter** to balance compression vs quality  
✅ **OPQ and PRQ** improve upon basic PQ with rotation and residual encoding

---

**Next:** [Best Practices →](./07-best-practices.md)

---

<!-- 
Sources Consulted:
- Milvus Index Documentation: https://milvus.io/docs/index.md
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
-->
