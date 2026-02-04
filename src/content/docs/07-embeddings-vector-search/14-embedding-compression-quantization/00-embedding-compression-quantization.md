---
title: "Embedding Compression & Quantization"
---

# Embedding Compression & Quantization

## Overview

As embedding collections grow, so do storage costs, memory requirements, and search latency. **Compression and quantization** techniques reduce embedding size while preserving semantic similarity—enabling you to store billions of vectors affordably and search them quickly.

This lesson explores compression methods from API-level options (Cohere's int8/binary) to database-level quantization (Qdrant, Milvus, pgvector), helping you choose the right approach for your scale and accuracy requirements.

---

## Why This Matters

```
┌─────────────────────────────────────────────────────────────────┐
│              The Scale Problem                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1 Million Vectors @ 1536 dimensions @ float32:                │
│                                                                 │
│  Storage:    1M × 1536 × 4 bytes = 6.1 GB                      │
│  Memory:     6.1 GB (for in-memory search)                     │
│  Bandwidth:  6.1 GB per full index load                        │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  With int8 Quantization (4x compression):                      │
│                                                                 │
│  Storage:    1.5 GB                                            │
│  Memory:     1.5 GB                                            │
│  Savings:    ~75%                                              │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  With Binary Quantization (32x compression):                   │
│                                                                 │
│  Storage:    190 MB                                            │
│  Memory:     190 MB                                            │
│  Savings:    ~97%                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Why Compress Embeddings](./01-why-compress-embeddings.md) | Storage costs, memory efficiency, search speed, enabling scale |
| 02 | [Compression Types](./02-compression-types.md) | Cohere embed-v4 types: float, int8, uint8, binary, ubinary |
| 03 | [Quantization Trade-offs](./03-quantization-tradeoffs.md) | Precision loss vs compression, two-stage retrieval pattern |
| 04 | [Binary Embeddings](./04-binary-embeddings.md) | Bit packing, Hamming distance, rescoring strategies |
| 05 | [Vector DB Quantization](./05-vector-db-quantization.md) | Qdrant, Milvus, pgvector quantization features |
| 06 | [Product Quantization](./06-product-quantization.md) | PQ algorithm, codebooks, asymmetric distance computation |
| 07 | [Best Practices](./07-best-practices.md) | Benchmarking, quality monitoring, production strategies |

---

## Compression Methods Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Compression Hierarchy                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HIGHEST QUALITY                          MOST COMPRESSED       │
│  ◄─────────────────────────────────────────────────────────────►│
│                                                                 │
│  float32    float16    int8      int4     binary               │
│  (4 bytes)  (2 bytes)  (1 byte)  (0.5b)   (1 bit)              │
│                                                                 │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐          │
│  │ 1.0x   │ │ 2.0x   │ │ 4.0x   │ │ 8.0x │ │ 32.0x  │          │
│  │ compr. │ │ compr. │ │ compr. │ │compr.│ │ compr. │          │
│  └────────┘ └────────┘ └────────┘ └──────┘ └────────┘          │
│                                                                 │
│  Recall:                                                        │
│  ~100%      ~99.9%     ~99%       ~95%     ~90-98%*            │
│                                                                 │
│  *Binary recall varies by model and dimensionality              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Compression Options

### API-Level (Cohere embed-v4)

| Type | Bits/Dim | Compression | Use Case |
|------|----------|-------------|----------|
| `float` | 32 | 1x | Maximum quality |
| `int8` | 8 | 4x | Balanced quality/size |
| `uint8` | 8 | 4x | Same as int8, unsigned |
| `binary` | 1 | 32x | Fast candidate generation |
| `ubinary` | 1 | 32x | Same as binary, unsigned |

### Database-Level

| Database | Scalar | Binary | Product |
|----------|--------|--------|---------|
| Qdrant | ✅ int8 | ✅ 1/1.5/2-bit | ✅ PQ |
| Milvus | ✅ IVF_SQ8 | — | ✅ IVF_PQ, HNSW_PQ |
| pgvector | ✅ halfvec | ✅ bit type | — |
| Pinecone | ✅ auto | — | — |

---

## The Two-Stage Retrieval Pattern

For maximum efficiency with compressed embeddings:

```
┌─────────────────────────────────────────────────────────────────┐
│              Two-Stage Retrieval                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: CANDIDATE GENERATION (Fast, Approximate)             │
│  ─────────────────────────────────────────────────              │
│                                                                 │
│  Query ───▶ Binary/Quantized Search ───▶ Top 100 Candidates    │
│                                                                 │
│  • Uses compressed embeddings (binary, int8)                   │
│  • Very fast (Hamming distance, SIMD operations)               │
│  • Oversample: retrieve more than needed                       │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  STAGE 2: RESCORING (Precise, Accurate)                        │
│  ─────────────────────────────────────────────────              │
│                                                                 │
│  Top 100 ───▶ Full Precision Scoring ───▶ Top 10 Final         │
│                                                                 │
│  • Uses original float32 embeddings                            │
│  • Accurate cosine/dot product similarity                      │
│  • Only on small candidate set                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Example

```python
import cohere
import numpy as np

co = cohere.ClientV2()

# Generate embeddings with multiple compression types in one call
response = co.embed(
    texts=["Machine learning is transforming industries"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float", "int8", "ubinary"],  # Multiple types
    output_dimension=1024
)

# Access different formats
float_embedding = response.embeddings.float_[0]
int8_embedding = response.embeddings.int8[0]
binary_embedding = response.embeddings.ubinary[0]

print(f"Float: {len(float_embedding)} values, {len(float_embedding) * 4} bytes")
print(f"Int8:  {len(int8_embedding)} values, {len(int8_embedding)} bytes")
print(f"Binary: {len(binary_embedding)} bytes (packed bits)")
```

**Output:**
```
Float: 1024 values, 4096 bytes
Int8:  1024 values, 1024 bytes
Binary: 128 bytes (packed bits)
```

---

## Prerequisites

- Understanding of embeddings and vector search
- Familiarity with similarity metrics (cosine, dot product)
- Basic knowledge of vector databases

---

## Learning Objectives

By the end of this lesson, you will be able to:

- ✅ Explain why embedding compression matters at scale
- ✅ Use Cohere's compression types (float, int8, binary)
- ✅ Implement two-stage retrieval with rescoring
- ✅ Configure database-level quantization (Qdrant, Milvus, pgvector)
- ✅ Understand Product Quantization for extreme compression
- ✅ Benchmark and monitor quality after quantization

---

**Start Learning:** [Why Compress Embeddings →](./01-why-compress-embeddings.md)

---

<!-- 
Sources Consulted:
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Milvus Index Types: https://milvus.io/docs/index.md
- pgvector: https://github.com/pgvector/pgvector
-->
