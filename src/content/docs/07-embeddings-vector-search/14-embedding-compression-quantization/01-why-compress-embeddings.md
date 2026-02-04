---
title: "Why Compress Embeddings?"
---

# Why Compress Embeddings?

## Introduction

At small scale, storing embeddings as 32-bit floats is perfectly reasonable. But as your collection grows to millions or billions of vectors, the costs of full-precision storage become prohibitive. Compression isn't just about saving money—it's about making large-scale semantic search practical.

This lesson explores the four key drivers for embedding compression: storage costs, memory efficiency, search speed, and enabling scale.

### What We'll Cover

- Storage cost reduction at scale
- Memory efficiency for in-memory search
- Faster similarity computation
- Enabling larger indices

### Prerequisites

- Basic understanding of embeddings
- Familiarity with vector databases

---

## The Math of Scale

### Storage Requirements

Each embedding dimension stored as float32 takes 4 bytes:

```
┌─────────────────────────────────────────────────────────────────┐
│              Storage Formula                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Storage = Vectors × Dimensions × Bytes per Value               │
│                                                                 │
│  Example (OpenAI text-embedding-3-large, 3072 dimensions):     │
│                                                                 │
│  1K vectors:    1,000 × 3,072 × 4 = 12.3 MB                    │
│  1M vectors:    1,000,000 × 3,072 × 4 = 12.3 GB                │
│  100M vectors:  100,000,000 × 3,072 × 4 = 1.23 TB              │
│  1B vectors:    1,000,000,000 × 3,072 × 4 = 12.3 TB            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Implications

Using typical cloud storage costs:

| Scale | Raw Size | Cloud Storage/Month | With 4x Compression |
|-------|----------|---------------------|---------------------|
| 1M vectors | 12.3 GB | ~$3 | ~$0.75 |
| 10M vectors | 123 GB | ~$30 | ~$7.50 |
| 100M vectors | 1.23 TB | ~$300 | ~$75 |
| 1B vectors | 12.3 TB | ~$3,000 | ~$750 |

---

## Driver 1: Storage Cost Reduction

### The Problem

Every dimension of every vector consumes storage. For production systems with:
- Multiple embedding models (different purposes)
- Multiple versions (A/B testing, rollback capability)
- Backup and replication requirements

Costs multiply quickly.

### Compression Impact

```
┌─────────────────────────────────────────────────────────────────┐
│              Storage Savings by Compression Level                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  100M Vectors × 1536 Dimensions                                 │
│                                                                 │
│  Format      Bytes/Dim   Total Size   Monthly Cost*            │
│  ────────────────────────────────────────────────────          │
│  float32     4.0         614 GB       ~$150                    │
│  float16     2.0         307 GB       ~$75                     │
│  int8        1.0         154 GB       ~$37                     │
│  binary      0.125       19 GB        ~$5                      │
│                                                                 │
│  *Estimated cloud storage at $0.25/GB/month                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Example

A recommendation system with:
- 50M products
- 1536-dim embeddings
- 3 replicas for availability
- Daily backups (7-day retention)

**Without compression:**
```
Base: 50M × 1536 × 4 = 307 GB
Replicas: 307 GB × 3 = 921 GB
Backups: 307 GB × 7 = 2.15 TB
Total: ~3 TB → ~$750/month in storage alone
```

**With int8 compression (4x):**
```
Total: ~750 GB → ~$187/month
Annual savings: ~$6,756
```

---

## Driver 2: Memory Efficiency

### Why Memory Matters

Vector search performance depends heavily on keeping vectors in memory:

| Storage Location | Latency | Throughput |
|------------------|---------|------------|
| L1 Cache | ~1 ns | Highest |
| RAM | ~100 ns | High |
| SSD | ~100 μs | Medium |
| HDD | ~10 ms | Low |

**In-memory search is 1000x+ faster than disk-based search.**

### The RAM Constraint

```
┌─────────────────────────────────────────────────────────────────┐
│              Memory Utilization                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Server: 64 GB RAM (typical cloud instance)                    │
│  OS + Application overhead: ~10 GB                             │
│  Available for vectors: ~54 GB                                 │
│                                                                 │
│  Vectors that fit (1536 dims):                                 │
│                                                                 │
│  float32:  54 GB ÷ 6.1 KB = ~8.8 million vectors               │
│  int8:     54 GB ÷ 1.5 KB = ~35 million vectors                │
│  binary:   54 GB ÷ 192 B  = ~280 million vectors               │
│                                                                 │
│  Compression enables 4-32x more vectors in the same RAM        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory vs Disk Trade-off

Without compression, large collections require:
- Disk-based indices (slower queries)
- More expensive high-memory instances
- Complex sharding across machines

With compression:
- More vectors fit in RAM (faster queries)
- Smaller instances work for larger datasets
- Simpler architecture (fewer shards)

---

## Driver 3: Faster Similarity Computation

### How Compression Speeds Up Search

Compressed vectors enable faster comparison operations:

| Format | Comparison Operation | SIMD Support | Relative Speed |
|--------|---------------------|--------------|----------------|
| float32 | Floating-point multiply-add | ✅ | 1x (baseline) |
| float16 | Half-precision multiply-add | ✅ | ~1.5x |
| int8 | Integer multiply-add | ✅ (optimized) | ~2-4x |
| binary | XOR + popcount (Hamming) | ✅ (single instruction) | ~10-40x |

### Binary Search Advantage

Binary embeddings enable **Hamming distance** computation:

```python
# Float32 cosine similarity (many operations)
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(y**2 for y in b))
    return dot / (norm_a * norm_b)

# Binary Hamming distance (fast bitwise operations)
def hamming_distance(a, b):
    # XOR to find differing bits, then count them
    return bin(a ^ b).count('1')
```

Modern CPUs have dedicated instructions (`POPCNT`) for counting bits, making binary comparison extremely fast.

### Benchmark Comparison

| Operation | float32 (1536d) | binary (1536 bits) | Speedup |
|-----------|-----------------|---------------------|---------|
| Single comparison | ~3 μs | ~0.1 μs | ~30x |
| 1M comparisons | ~3 s | ~100 ms | ~30x |
| Queries per second | ~300 | ~10,000 | ~33x |

---

## Driver 4: Enabling Larger Indices

### Index Size Impact

Vector indices (HNSW, IVF) have memory overhead beyond raw vectors:

```
┌─────────────────────────────────────────────────────────────────┐
│              HNSW Index Memory Breakdown                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Component           float32 (100M vectors)   int8             │
│  ──────────────────────────────────────────────────────────────│
│  Raw vectors         614 GB                   154 GB           │
│  Graph structure     ~50 GB                   ~50 GB           │
│  Metadata            ~10 GB                   ~10 GB           │
│  ──────────────────────────────────────────────────────────────│
│  Total               ~674 GB                  ~214 GB          │
│                                                                 │
│  Graph structure doesn't shrink with quantization,             │
│  but vector data does — and that's the largest component.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Practical Limits

Without compression, you hit practical limits sooner:

| Constraint | float32 Limit | With 4x Compression |
|------------|---------------|---------------------|
| Single machine (256 GB RAM) | ~40M vectors | ~160M vectors |
| Managed service (typical tier) | ~10M vectors | ~40M vectors |
| Serverless (cold start) | ~1M vectors | ~4M vectors |

### Sharding Implications

```
┌─────────────────────────────────────────────────────────────────┐
│              Sharding: Compressed vs Uncompressed                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Target: 1 Billion vectors, 1536 dimensions                    │
│                                                                 │
│  WITHOUT COMPRESSION (float32):                                │
│  Total size: 6.1 TB                                            │
│  Shards needed (256 GB each): 24 shards                        │
│  Coordination overhead: High                                   │
│  Query latency: Higher (fan-out to 24 shards)                  │
│                                                                 │
│  WITH int8 COMPRESSION:                                        │
│  Total size: 1.5 TB                                            │
│  Shards needed: 6 shards                                       │
│  Coordination overhead: Lower                                  │
│  Query latency: Lower (fan-out to 6 shards)                    │
│                                                                 │
│  Fewer shards = simpler ops + faster queries + lower cost      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## When Compression Makes Sense

### Definitely Compress When

| Scenario | Reason |
|----------|--------|
| > 1M vectors | Storage costs become significant |
| Memory-constrained | Need more vectors in RAM |
| High QPS requirements | Need faster comparisons |
| Cost-sensitive deployment | Every GB counts |
| Multi-region replication | Replication multiplies storage |

### Maybe Delay Compression When

| Scenario | Reason |
|----------|--------|
| < 100K vectors | Overhead of compression may not pay off |
| Maximum accuracy required | Some compression reduces quality |
| Prototyping phase | Simplicity matters more than efficiency |
| Unknown query patterns | Hard to benchmark quality impact |

### Always Test First

> **Rule of thumb:** Don't compress blindly. Benchmark your specific queries to measure quality impact before deploying compressed embeddings in production.

---

## Summary

✅ **Storage costs** grow linearly with vector count—compression provides 4-32x savings  
✅ **Memory efficiency** determines how many vectors can be searched in-memory  
✅ **Faster computation** from int8 and binary enables higher throughput  
✅ **Larger indices** become practical with compression—fewer shards, simpler ops  
✅ **Test before deploying** to ensure quality meets your requirements

---

**Next:** [Compression Types →](./02-compression-types.md)

---

<!-- 
Sources Consulted:
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
-->
