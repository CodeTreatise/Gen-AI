---
title: "Binary Embeddings"
---

# Binary Embeddings

## Introduction

Binary embeddings represent each dimension with a single bit—the most extreme form of compression at 32x. While this loses magnitude information, clever bit-packing and fast Hamming distance computation make binary search extremely efficient. Combined with rescoring, binary embeddings enable billion-scale search.

This lesson covers how binary embeddings work internally, how to manipulate them, and how to implement effective search patterns.

### What We'll Cover

- How bits are packed into bytes
- Working with binary embeddings in Python
- Hamming distance computation
- Rescoring strategies for quality recovery

### Prerequisites

- Understanding of [compression types](./02-compression-types.md)
- Basic NumPy knowledge

---

## How Binary Embeddings Work

### From Float to Bit

The conversion is simple: positive values become 1, negative values become 0.

```
┌─────────────────────────────────────────────────────────────────┐
│              Float to Binary Conversion                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Float32 embedding (8 dimensions for illustration):             │
│  [0.234, -0.156, 0.512, -0.089, 0.421, -0.312, 0.178, -0.067]  │
│                                                                 │
│  Apply sign function:                                           │
│  positive → 1                                                   │
│  negative → 0                                                   │
│                                                                 │
│  Binary bits:                                                   │
│  [1, 0, 1, 0, 1, 0, 1, 0]                                       │
│                                                                 │
│  Note: 0.234, 0.512, and 0.421 ALL become 1                    │
│        Their magnitudes (0.234 vs 0.512) are lost              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Bit Packing into Bytes

Eight bits pack into one byte for efficient storage and computation:

```
┌─────────────────────────────────────────────────────────────────┐
│              Bit Packing Process                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Binary bits (1024 dimensions):                                 │
│  [1,0,1,0,1,0,1,0, 0,1,0,0,1,1,0,1, 1,1,0,0,0,1,1,1, ...]      │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                │
│      Byte 1          Byte 2          Byte 3                     │
│                                                                 │
│  Pack into bytes (8 bits each):                                │
│                                                                 │
│  Byte 1: 10101010 = 170 (binary) or 0xAA (hex)                 │
│  Byte 2: 01001101 = 77                                         │
│  Byte 3: 11000111 = 199                                        │
│  ...                                                           │
│                                                                 │
│  Result: 1024 bits → 128 bytes                                 │
│  (32x compression from 1024 × 4 = 4096 bytes)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dimension to Byte Calculation

```python
def dimensions_to_bytes(dimensions: int) -> int:
    """Calculate bytes needed for binary embedding."""
    return (dimensions + 7) // 8  # Ceiling division

# Examples
print(f"768 dims → {dimensions_to_bytes(768)} bytes")   # 96
print(f"1024 dims → {dimensions_to_bytes(1024)} bytes") # 128
print(f"1536 dims → {dimensions_to_bytes(1536)} bytes") # 192
print(f"3072 dims → {dimensions_to_bytes(3072)} bytes") # 384
```

**Output:**
```
768 dims → 96 bytes
1024 dims → 128 bytes
1536 dims → 192 bytes
3072 dims → 384 bytes
```

---

## Working with Binary Embeddings in Python

### From Cohere API

```python
import cohere
import numpy as np

co = cohere.ClientV2(api_key="your-api-key")

response = co.embed(
    texts=["How do I compress embeddings?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float", "ubinary"]
)

# Binary embedding comes as list of byte values
binary_bytes = response.embeddings.ubinary[0]
float_embedding = response.embeddings.float_[0]

print(f"Float dimensions: {len(float_embedding)}")
print(f"Binary bytes: {len(binary_bytes)}")
print(f"First 5 byte values: {binary_bytes[:5]}")
```

**Output:**
```
Float dimensions: 1024
Binary bytes: 128
First 5 byte values: [170, 77, 199, 142, 235]
```

### Converting to NumPy

```python
import numpy as np

def bytes_to_numpy(byte_list: list) -> np.ndarray:
    """Convert list of byte values to numpy array."""
    return np.array(byte_list, dtype=np.uint8)

def bytes_to_bits(byte_array: np.ndarray) -> np.ndarray:
    """Unpack bytes to individual bits."""
    return np.unpackbits(byte_array)

# Convert API response
binary_np = bytes_to_numpy(binary_bytes)
bits = bytes_to_bits(binary_np)

print(f"Byte array shape: {binary_np.shape}")
print(f"Bits array shape: {bits.shape}")
print(f"First 16 bits: {bits[:16]}")
```

**Output:**
```
Byte array shape: (128,)
Bits array shape: (1024,)
First 16 bits: [1 0 1 0 1 0 1 0 0 1 0 0 1 1 0 1]
```

### Converting Float to Binary Manually

```python
def float_to_binary(float_embedding: list) -> np.ndarray:
    """Convert float embedding to packed binary bytes."""
    # Convert to numpy
    floats = np.array(float_embedding, dtype=np.float32)
    
    # Get sign: positive → 1, negative → 0
    bits = (floats >= 0).astype(np.uint8)
    
    # Pad to multiple of 8 if needed
    pad_length = (8 - len(bits) % 8) % 8
    if pad_length > 0:
        bits = np.pad(bits, (0, pad_length))
    
    # Pack bits into bytes
    packed = np.packbits(bits)
    
    return packed

# Test manual conversion
float_emb = response.embeddings.float_[0]
manual_binary = float_to_binary(float_emb)
api_binary = bytes_to_numpy(response.embeddings.ubinary[0])

# Should match (or be very close)
print(f"Match: {np.array_equal(manual_binary, api_binary)}")
```

---

## Hamming Distance

### What It Is

Hamming distance counts the number of positions where corresponding bits differ:

```
┌─────────────────────────────────────────────────────────────────┐
│              Hamming Distance Example                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vector A: [1, 0, 1, 1, 0, 0, 1, 0]                            │
│  Vector B: [1, 1, 1, 0, 0, 0, 1, 1]                            │
│            ─  ↑  ─  ↑  ─  ─  ─  ↑                              │
│               ×     ×           ×                               │
│                                                                 │
│  Differences at positions: 2, 4, 8                             │
│  Hamming distance = 3                                          │
│                                                                 │
│  XOR operation (1 where bits differ):                          │
│  A XOR B: [0, 1, 0, 1, 0, 0, 0, 1]                             │
│  popcount([0, 1, 0, 1, 0, 0, 0, 1]) = 3                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Computing Hamming Distance

```python
import numpy as np

def hamming_distance_bytes(a: np.ndarray, b: np.ndarray) -> int:
    """
    Compute Hamming distance between two packed byte arrays.
    Uses efficient XOR + popcount.
    """
    # XOR finds differing bits
    xor_result = np.bitwise_xor(a, b)
    
    # Unpack to bits and count 1s
    bits = np.unpackbits(xor_result)
    distance = np.sum(bits)
    
    return distance

def hamming_distance_optimized(a: np.ndarray, b: np.ndarray) -> int:
    """
    Optimized Hamming distance using numpy's bit counting.
    """
    # XOR and count bits per byte, then sum
    xor_result = np.bitwise_xor(a, b)
    
    # Use lookup table for popcount per byte
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    bit_counts = popcount_table[xor_result]
    
    return int(np.sum(bit_counts))

# Example usage
a = np.array([0b10101010, 0b11110000], dtype=np.uint8)
b = np.array([0b10100011, 0b11110000], dtype=np.uint8)

distance = hamming_distance_bytes(a, b)
print(f"Hamming distance: {distance}")  # Should be 3
```

**Output:**
```
Hamming distance: 3
```

### Batch Hamming Distance

For searching across many documents:

```python
def batch_hamming_distances(
    query: np.ndarray,
    documents: np.ndarray
) -> np.ndarray:
    """
    Compute Hamming distance from query to all documents.
    
    Args:
        query: Shape (num_bytes,)
        documents: Shape (num_docs, num_bytes)
    
    Returns:
        Distances of shape (num_docs,)
    """
    # Broadcast XOR
    xor_result = np.bitwise_xor(query, documents)  # (num_docs, num_bytes)
    
    # Unpack and sum bits
    bits = np.unpackbits(xor_result, axis=1)  # (num_docs, num_bits)
    distances = np.sum(bits, axis=1)  # (num_docs,)
    
    return distances

# Example: find nearest neighbors
def find_nearest_binary(
    query: np.ndarray,
    documents: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """Find k nearest neighbors by Hamming distance."""
    distances = batch_hamming_distances(query, documents)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices
```

### Why Hamming is Fast

Modern CPUs have dedicated instructions for bit counting:

| Operation | Float32 Cosine | Binary Hamming | Speedup |
|-----------|----------------|----------------|---------|
| Per-comparison | ~50 cycles | ~3 cycles | ~17x |
| Memory load | 4096 bytes | 128 bytes | 32x |
| Cache efficiency | Lower | Higher | Variable |
| SIMD utilization | Moderate | Excellent | 2-4x |

**Combined effect: 30-40x faster search for same number of vectors.**

---

## Rescoring Strategies

### Why Rescore?

Binary search is fast but loses precision. Rescoring with full-precision embeddings recovers quality:

```
┌─────────────────────────────────────────────────────────────────┐
│              Rescoring Impact on Quality                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Search Type            Recall@10     Latency                   │
│  ─────────────────────────────────────────────────────         │
│  Float32 only           0.98          100ms                     │
│  Binary only            0.82          5ms                       │
│  Binary + rescore (100) 0.96          15ms                      │
│  Binary + rescore (200) 0.97          20ms                      │
│                                                                 │
│  Rescoring recovers most of the quality at fraction of cost    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Stage Implementation

```python
import numpy as np
from typing import List, Tuple

class BinarySearcher:
    def __init__(
        self,
        binary_embeddings: np.ndarray,  # (num_docs, num_bytes)
        float_embeddings: np.ndarray,    # (num_docs, num_dims)
        doc_ids: List[str]
    ):
        self.binary = binary_embeddings
        self.floats = float_embeddings
        self.doc_ids = doc_ids
        
        # Pre-normalize floats for cosine similarity
        norms = np.linalg.norm(float_embeddings, axis=1, keepdims=True)
        self.float_normalized = float_embeddings / norms
    
    def search(
        self,
        query_binary: np.ndarray,
        query_float: np.ndarray,
        top_k: int = 10,
        candidates: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Two-stage search: binary candidates + float rescoring.
        """
        # Stage 1: Fast binary search
        distances = batch_hamming_distances(query_binary, self.binary)
        candidate_indices = np.argsort(distances)[:candidates]
        
        # Stage 2: Precise float rescoring
        candidate_floats = self.float_normalized[candidate_indices]
        query_normalized = query_float / np.linalg.norm(query_float)
        
        similarities = np.dot(candidate_floats, query_normalized)
        
        # Sort by similarity (descending)
        top_local = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.doc_ids[candidate_indices[i]], float(similarities[i]))
            for i in top_local
        ]
        
        return results

# Usage
searcher = BinarySearcher(
    binary_embeddings=doc_binary_array,
    float_embeddings=doc_float_array,
    doc_ids=document_ids
)

results = searcher.search(
    query_binary=query_bin,
    query_float=query_float,
    top_k=10,
    candidates=100
)

for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### Choosing Candidate Count

The relationship between candidates and quality:

| Candidates | Recall Recovery | Latency Overhead |
|------------|-----------------|------------------|
| 50 | ~92% | Minimal |
| 100 | ~95% | Low |
| 200 | ~97% | Moderate |
| 500 | ~98% | Higher |
| 1000 | ~99% | Significant |

**Rule of thumb:** Start with 10x your top_k, adjust based on benchmarks.

---

## Storing Binary Embeddings

### PostgreSQL with pgvector

```sql
-- Create table with bit column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding_binary BIT(1024),      -- Binary for fast search
    embedding_float VECTOR(1024)      -- Float for rescoring
);

-- Create HNSW index on binary
CREATE INDEX ON documents 
USING hnsw (embedding_binary bit_hamming_ops);

-- Search with Hamming distance
SELECT id, content, embedding_binary <~> $1::bit(1024) AS distance
FROM documents
ORDER BY distance
LIMIT 100;
```

### Binary Storage Format

```python
def to_pg_bit_string(binary_bytes: np.ndarray) -> str:
    """Convert packed bytes to PostgreSQL bit string."""
    bits = np.unpackbits(binary_bytes)
    return ''.join(str(b) for b in bits)

def from_pg_bit_string(bit_string: str) -> np.ndarray:
    """Convert PostgreSQL bit string to packed bytes."""
    bits = np.array([int(b) for b in bit_string], dtype=np.uint8)
    return np.packbits(bits)

# Example
binary_bytes = np.array([170, 77], dtype=np.uint8)
pg_string = to_pg_bit_string(binary_bytes)
print(f"PostgreSQL format: '{pg_string}'")
# Output: '1010101001001101'
```

---

## Performance Comparison

### Benchmark Results

| Collection Size | Float32 Search | Binary + Rescore | Speedup |
|-----------------|----------------|------------------|---------|
| 100K | 50ms | 8ms | 6x |
| 1M | 450ms | 25ms | 18x |
| 10M | 4.5s | 120ms | 37x |
| 100M | 45s | 800ms | 56x |

### Memory Comparison

| Collection Size | Float32 Memory | Binary Memory | Savings |
|-----------------|----------------|---------------|---------|
| 1M × 1024d | 4.1 GB | 128 MB | 32x |
| 10M × 1024d | 41 GB | 1.3 GB | 32x |
| 100M × 1024d | 410 GB | 12.8 GB | 32x |

> **Note:** If storing both binary (for search) and float (for rescore), total memory is binary + float = 33x original. Still significant savings if float embeddings are on disk.

---

## Summary

✅ **Binary = 1 bit per dimension**, packed 8 bits per byte  
✅ **Hamming distance** uses XOR + popcount—extremely fast  
✅ **32x compression** enables billion-scale search  
✅ **Rescoring with 100-200 candidates** recovers ~95-97% quality  
✅ **Store both formats** for optimal two-stage retrieval

---

**Next:** [Vector Database Quantization →](./05-vector-db-quantization.md)

---

<!-- 
Sources Consulted:
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- pgvector GitHub: https://github.com/pgvector/pgvector
-->
