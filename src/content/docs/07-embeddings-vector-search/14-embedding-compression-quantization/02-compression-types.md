---
title: "Compression Types"
---

# Compression Types

## Introduction

Modern embedding APIs offer multiple compression formats in a single call. Rather than choosing one format at embedding time, you can generate multiple representations simultaneously—storing full-precision for quality and compressed for speed.

This lesson covers the five compression types available in Cohere's embed-v4.0 and similar approaches in other providers.

### What We'll Cover

- Float32: full precision baseline
- Int8: 4x compression with minimal quality loss
- Uint8: unsigned 8-bit alternative
- Binary and Ubinary: extreme 32x compression
- Requesting multiple types in a single API call

### Prerequisites

- Understanding of [why compression matters](./01-why-compress-embeddings.md)
- Basic familiarity with embedding APIs

---

## The Compression Spectrum

```
┌─────────────────────────────────────────────────────────────────┐
│              Compression Type Comparison                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Precision                                          Compression │
│     High ◄─────────────────────────────────────────────► High  │
│                                                                 │
│     float32      float16       int8        binary              │
│       │            │            │            │                  │
│       ▼            ▼            ▼            ▼                  │
│     4 bytes     2 bytes      1 byte      1 bit                 │
│     per dim     per dim      per dim     per dim               │
│                                                                 │
│     1x          2x           4x          32x                   │
│   baseline    compression  compression  compression            │
│                                                                 │
│  Best         Good          Slightly    Reduced                │
│  quality      quality       reduced     (needs rescoring)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Float32: Full Precision

### What It Is

Float32 (32-bit floating point) is the standard representation:

- **Size:** 4 bytes per dimension
- **Range:** ±3.4 × 10³⁸ with ~7 decimal digits precision
- **Use case:** Maximum quality, reference for comparisons

### When to Use

- Storing canonical embeddings for later reranking
- When quality is more important than storage
- Benchmarking compressed formats
- Small datasets (< 100K vectors)

### Example

```python
import cohere

co = cohere.ClientV2(api_key="your-api-key")

response = co.embed(
    texts=["How do I compress embeddings?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]  # Full precision
)

# Access float embeddings
float_embedding = response.embeddings.float_[0]

print(f"Dimensions: {len(float_embedding)}")
print(f"Sample values: {float_embedding[:5]}")
print(f"Size: {len(float_embedding) * 4} bytes")
```

**Output:**
```
Dimensions: 1024
Sample values: [0.0234, -0.0156, 0.0512, -0.0089, 0.0421]
Size: 4096 bytes
```

---

## Int8: The Sweet Spot

### What It Is

Int8 (8-bit signed integer) provides 4x compression with minimal quality loss:

- **Size:** 1 byte per dimension
- **Range:** -128 to 127
- **Compression:** 4x vs float32
- **Quality:** ~99%+ of full precision

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│              Float32 to Int8 Quantization                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Float32 values:   [-0.82, 0.15, -0.03, 0.91, -0.45]           │
│                                                                 │
│  Step 1: Find min/max of embedding                             │
│          min = -0.82, max = 0.91                               │
│                                                                 │
│  Step 2: Scale to -128..127 range                              │
│          scale = 255 / (max - min) = 147.4                     │
│          offset = min = -0.82                                  │
│                                                                 │
│  Step 3: Quantize each value                                   │
│          int8 = round((float - offset) * scale) - 128          │
│                                                                 │
│  Result: [-128, -27, 72, 127, 8]                               │
│                                                                 │
│  Store: values + scale + offset for reconstruction             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use

- **Primary recommendation** for most production use cases
- Large collections (millions of vectors)
- When quality cannot be noticeably degraded
- In-memory indices with RAM constraints

### Example

```python
import cohere

co = cohere.ClientV2(api_key="your-api-key")

response = co.embed(
    texts=["How do I compress embeddings?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["int8"]  # 4x compression
)

# Access int8 embeddings
int8_embedding = response.embeddings.int8[0]

print(f"Dimensions: {len(int8_embedding)}")
print(f"Sample values: {int8_embedding[:5]}")
print(f"Size: {len(int8_embedding)} bytes")  # 1 byte per value
```

**Output:**
```
Dimensions: 1024
Sample values: [23, -15, 51, -8, 42]
Size: 1024 bytes
```

---

## Uint8: Unsigned Alternative

### What It Is

Uint8 (8-bit unsigned integer) is similar to int8 but uses positive values only:

- **Size:** 1 byte per dimension
- **Range:** 0 to 255
- **Compression:** 4x vs float32

### Int8 vs Uint8

| Aspect | Int8 | Uint8 |
|--------|------|-------|
| Range | -128 to 127 | 0 to 255 |
| Signed | Yes | No |
| Storage | Same (1 byte) | Same (1 byte) |
| Use case | Centered data | Non-negative data |

### When to Use

- Database systems that prefer unsigned integers
- When your vector database doesn't support signed int8
- Compatibility with certain hardware accelerators

### Example

```python
response = co.embed(
    texts=["How do I compress embeddings?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["uint8"]
)

uint8_embedding = response.embeddings.uint8[0]
print(f"Sample values: {uint8_embedding[:5]}")  # All 0-255
```

**Output:**
```
Sample values: [151, 113, 179, 120, 170]
```

---

## Binary: Maximum Compression

### What It Is

Binary embeddings use a single bit per dimension:

- **Size:** 1 bit per dimension (packed into bytes)
- **Compression:** 32x vs float32
- **Quality:** Reduced, but effective with rescoring

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│              Float32 to Binary Conversion                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Float32 values:   [0.23, -0.15, 0.08, -0.91, 0.45, -0.03, ...] │
│                                                                 │
│  Rule: positive → 1, negative → 0                               │
│                                                                 │
│  Binary bits:      [1, 0, 1, 0, 1, 0, ...]                     │
│                                                                 │
│  Pack into bytes (8 bits each):                                │
│                                                                 │
│  Bits:    1 0 1 0 1 0 1 1 │ 0 1 0 0 1 1 0 1 │ ...             │
│           └─────┬─────────┘ └──────┬────────┘                   │
│               Byte 1             Byte 2                         │
│               (171)              (77)                           │
│                                                                 │
│  1024 dimensions → 1024 bits → 128 bytes                       │
│  3072 dimensions → 3072 bits → 384 bytes                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Binary vs Ubinary

| Type | Description | Bit Interpretation |
|------|-------------|-------------------|
| `binary` | Signed binary | Uses sign bit conventions |
| `ubinary` | Unsigned binary | Simple positive = 1, negative = 0 |

For most use cases, they produce identical results.

### When to Use

- First-stage retrieval in two-stage pipelines
- Billion-scale collections where storage is critical
- When combined with full-precision rescoring
- Hamming distance search (very fast)

### Example

```python
response = co.embed(
    texts=["How do I compress embeddings?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["ubinary"]
)

# Binary embeddings are packed as bytes
ubinary_embedding = response.embeddings.ubinary[0]

print(f"Number of bytes: {len(ubinary_embedding)}")
print(f"Sample bytes: {ubinary_embedding[:5]}")
print(f"Bits represented: {len(ubinary_embedding) * 8}")
```

**Output:**
```
Number of bytes: 128
Sample bytes: [171, 77, 234, 89, 142]
Bits represented: 1024
```

---

## Requesting Multiple Types

### The Power of Multiple Formats

Cohere and similar APIs let you request multiple compression types in a single call:

```python
import cohere

co = cohere.ClientV2(api_key="your-api-key")

response = co.embed(
    texts=[
        "Machine learning fundamentals",
        "Deep neural network architectures",
        "Natural language processing basics"
    ],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float", "int8", "ubinary"]  # All three!
)

# Access each format
for i, text in enumerate(["ML", "DNN", "NLP"]):
    float_emb = response.embeddings.float_[i]
    int8_emb = response.embeddings.int8[i]
    ubinary_emb = response.embeddings.ubinary[i]
    
    print(f"{text}:")
    print(f"  float32: {len(float_emb) * 4} bytes")
    print(f"  int8:    {len(int8_emb)} bytes")
    print(f"  ubinary: {len(ubinary_emb)} bytes")
```

**Output:**
```
ML:
  float32: 4096 bytes
  int8:    1024 bytes
  ubinary: 128 bytes
DNN:
  float32: 4096 bytes
  int8:    1024 bytes
  ubinary: 128 bytes
NLP:
  float32: 4096 bytes
  int8:    1024 bytes
  ubinary: 128 bytes
```

### Two-Stage Storage Pattern

```python
# Store both formats for optimal retrieval
def embed_and_store(texts, collection):
    response = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float", "ubinary"]
    )
    
    for i, text in enumerate(texts):
        collection.insert({
            "text": text,
            "embedding_full": response.embeddings.float_[i],    # For rescoring
            "embedding_binary": response.embeddings.ubinary[i]  # For fast search
        })

# Query: fast binary search, then rescore with float
def search(query, collection, top_k=10, candidates=100):
    query_response = co.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float", "ubinary"]
    )
    
    # Stage 1: Fast binary search for candidates
    candidates = collection.search_binary(
        query_response.embeddings.ubinary[0],
        limit=candidates
    )
    
    # Stage 2: Rescore candidates with full precision
    results = rescore_with_float(
        candidates,
        query_response.embeddings.float_[0],
        top_k=top_k
    )
    
    return results
```

---

## Accessing Embedding Attributes

### Cohere Response Structure

```python
response = co.embed(
    texts=["test"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float", "int8", "uint8", "binary", "ubinary"]
)

# Each type has its own attribute
# Note: float uses float_ (with underscore) to avoid Python keyword
print(type(response.embeddings.float_))     # List[List[float]]
print(type(response.embeddings.int8))       # List[List[int]]
print(type(response.embeddings.uint8))      # List[List[int]]
print(type(response.embeddings.binary))     # List[List[int]] (packed bytes)
print(type(response.embeddings.ubinary))    # List[List[int]] (packed bytes)
```

### Dimension Control

Combine compression with dimension reduction for even more savings:

```python
# Matryoshka embeddings: use fewer dimensions
response = co.embed(
    texts=["test"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["int8", "ubinary"],
    output_dimension=256  # Reduce from 1024 to 256
)

int8_emb = response.embeddings.int8[0]
ubinary_emb = response.embeddings.ubinary[0]

print(f"int8: {len(int8_emb)} bytes")      # 256 bytes
print(f"ubinary: {len(ubinary_emb)} bytes") # 32 bytes (256/8)
```

**Output:**
```
int8: 256 bytes
ubinary: 32 bytes
```

---

## Comparison Table

| Type | Bytes/Dim | Size (1024d) | Compression | Quality | Speed |
|------|-----------|--------------|-------------|---------|-------|
| float32 | 4.0 | 4,096 B | 1x | 100% | Baseline |
| float16 | 2.0 | 2,048 B | 2x | ~99.9% | ~1.2x |
| int8 | 1.0 | 1,024 B | 4x | ~99% | ~2x |
| uint8 | 1.0 | 1,024 B | 4x | ~99% | ~2x |
| binary | 0.125 | 128 B | 32x | ~95%* | ~30x |
| ubinary | 0.125 | 128 B | 32x | ~95%* | ~30x |

*With rescoring; lower without

---

## Summary

✅ **float32** provides full precision—use for reference and rescoring  
✅ **int8** offers the best quality/compression trade-off (4x, ~99% quality)  
✅ **uint8** is equivalent to int8 but unsigned (0-255 range)  
✅ **binary/ubinary** enable 32x compression for first-stage retrieval  
✅ **Request multiple types** in one API call for two-stage search patterns

---

**Next:** [Quantization Trade-offs →](./03-quantization-tradeoffs.md)

---

<!-- 
Sources Consulted:
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- Cohere Embed API Reference: https://docs.cohere.com/reference/embed
-->
