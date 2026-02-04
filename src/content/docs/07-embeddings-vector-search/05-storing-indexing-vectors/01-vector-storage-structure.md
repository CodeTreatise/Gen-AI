---
title: "Vector Storage Structure"
---

# Vector Storage Structure

## Introduction

Every vector record in a database consists of three core components: an ID, the vector itself, and associated metadata. Understanding how to design each component impacts deduplication, storage efficiency, and retrieval performance.

```
┌─────────────────────────────────────────────────────────┐
│                    Vector Record                        │
├─────────────┬──────────────────┬───────────────────────┤
│     ID      │     Vector       │      Metadata         │
│  "doc_123"  │  [0.1, 0.2, ...] │  {"category": "AI"}   │
└─────────────┴──────────────────┴───────────────────────┘
```

---

## ID Strategies

The ID uniquely identifies each vector record. Your choice of ID strategy impacts deduplication, debugging, and system coordination.

| Strategy | Format | Pros | Cons |
|----------|--------|------|------|
| **UUID** | `550e8400-e29b-41d4-a716-446655440000` | Universally unique, no coordination | Longer, no ordering |
| **Sequential** | `1`, `2`, `3` | Compact, ordered | Requires coordination |
| **Content hash** | `sha256(content)[:16]` | Deduplication built-in | Collisions possible |
| **Composite** | `source:doc:chunk` | Self-documenting | Longer strings |

```python
import hashlib
import uuid

# UUID approach - universally unique
def generate_uuid_id() -> str:
    return str(uuid.uuid4())

# Content-based ID - enables automatic deduplication
def generate_content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

# Composite ID - self-documenting structure
def generate_composite_id(source: str, doc_id: str, chunk_idx: int) -> str:
    return f"{source}:{doc_id}:chunk_{chunk_idx}"

# Examples
print(generate_uuid_id())        # 550e8400-e29b-41d4-a716-446655440000
print(generate_content_id("Hello world"))  # b94d27b9934d3e08
print(generate_composite_id("confluence", "doc123", 5))  # confluence:doc123:chunk_5
```

**Output:**
```
550e8400-e29b-41d4-a716-446655440000
b94d27b9934d3e08
confluence:doc123:chunk_5
```

> **Tip:** Content-based IDs automatically prevent duplicate content from being indexed twice—if you try to insert the same text again, it generates the same ID.

---

## Vector Data Types

Vectors are arrays of floating-point numbers. The data type you choose affects storage size, memory usage, and search quality.

| Type | Bytes per Dim | Precision | Use Case |
|------|---------------|-----------|----------|
| `float32` | 4 | Full | Default, highest quality |
| `float16` | 2 | Reduced | Memory-constrained |
| `int8` | 1 | Quantized | Large scale, some recall loss |
| `binary` | 1/8 | Binary | Massive scale, specialized |

```python
import numpy as np

# Standard float32 (default)
vector_f32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
print(f"float32: {vector_f32.nbytes} bytes")  # 12 bytes

# Half precision (2x compression)
vector_f16 = vector_f32.astype(np.float16)
print(f"float16: {vector_f16.nbytes} bytes")  # 6 bytes

# Quantized int8 (4x compression)
# Requires scaling to [-128, 127] range
def quantize_to_int8(vector: np.ndarray) -> np.ndarray:
    min_val, max_val = vector.min(), vector.max()
    scale = 255 / (max_val - min_val)
    return ((vector - min_val) * scale - 128).astype(np.int8)

vector_i8 = quantize_to_int8(vector_f32)
print(f"int8: {vector_i8.nbytes} bytes")  # 3 bytes
```

**Output:**
```
float32: 12 bytes
float16: 6 bytes
int8: 3 bytes
```

### Memory Calculation

For a collection of 1 million 1536-dimensional vectors:

| Data Type | Memory per Vector | Total Memory |
|-----------|-------------------|--------------|
| float32 | 6.14 KB | 6.14 GB |
| float16 | 3.07 KB | 3.07 GB |
| int8 | 1.54 KB | 1.54 GB |

```python
def calculate_memory(num_vectors: int, dimensions: int, dtype: str) -> str:
    """Calculate memory requirements."""
    bytes_per_dim = {"float32": 4, "float16": 2, "int8": 1, "binary": 0.125}
    
    total_bytes = num_vectors * dimensions * bytes_per_dim[dtype]
    
    if total_bytes >= 1e9:
        return f"{total_bytes / 1e9:.2f} GB"
    return f"{total_bytes / 1e6:.2f} MB"

# Example: 1M vectors, 1536 dimensions
for dtype in ["float32", "float16", "int8"]:
    memory = calculate_memory(1_000_000, 1536, dtype)
    print(f"{dtype}: {memory}")
```

---

## Storage Format Considerations

How vectors are physically stored affects different access patterns:

| Format | Best For | Trade-offs |
|--------|----------|------------|
| **Row-oriented** | Random access, updates | Slower scans |
| **Column-oriented** | Batch operations, analytics | Slower point lookups |
| **Memory-mapped** | Large datasets, fast loads | Requires SSD |
| **Compressed** | Storage efficiency | CPU overhead |

### Row vs Column Storage

```python
# Row-oriented: Each vector stored contiguously
# Good for: Fetching individual vectors by ID
row_storage = [
    [0.1, 0.2, 0.3, 0.4],  # Vector 1
    [0.5, 0.6, 0.7, 0.8],  # Vector 2
    [0.9, 1.0, 1.1, 1.2],  # Vector 3
]

# Column-oriented: Each dimension stored contiguously
# Good for: Batch operations across all vectors
column_storage = {
    "dim_0": [0.1, 0.5, 0.9],
    "dim_1": [0.2, 0.6, 1.0],
    "dim_2": [0.3, 0.7, 1.1],
    "dim_3": [0.4, 0.8, 1.2],
}
```

### Memory-Mapped Files

For datasets too large for RAM, use memory-mapped files:

```python
import numpy as np

def create_mmap_index(vectors: np.ndarray, filepath: str):
    """Create memory-mapped vector storage."""
    # Save to disk
    fp = np.memmap(filepath, dtype='float32', mode='w+', shape=vectors.shape)
    fp[:] = vectors[:]
    fp.flush()
    return fp

def load_mmap_index(filepath: str, shape: tuple) -> np.memmap:
    """Load memory-mapped vectors (near-instant load time)."""
    return np.memmap(filepath, dtype='float32', mode='r', shape=shape)

# Usage
vectors = np.random.random((1000000, 1536)).astype('float32')
mmap = create_mmap_index(vectors, "vectors.mmap")

# Loading is instant - OS handles paging from disk
loaded = load_mmap_index("vectors.mmap", (1000000, 1536))
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use content-based IDs for deduplication | Use sequential IDs across distributed systems |
| Start with float32, optimize later | Prematurely optimize to int8 |
| Include source tracking in IDs | Use opaque UUIDs without tracking |
| Calculate memory requirements upfront | Assume unlimited memory |

---

## Summary

✅ **ID strategies** determine deduplication and debugging capabilities

✅ **Data types** trade precision for storage efficiency

✅ **Storage formats** optimize for different access patterns

✅ **Memory-mapped files** enable larger-than-RAM datasets

**Next:** [Metadata Storage Patterns](./02-metadata-storage-patterns.md)
