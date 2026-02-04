---
title: "Cost-Benefit Analysis"
---

# Cost-Benefit Analysis

## Introduction

Matryoshka embeddings offer a rare opportunity in engineering: **reduce costs without sacrificing much quality**. By using smaller dimensions, you save on storage, speed up searches, and reduce memory usage—all while maintaining nearly identical retrieval accuracy.

This lesson quantifies those tradeoffs with real numbers, helping you make informed decisions for your specific use case.

### What We'll Cover

- Storage cost calculations
- Search latency improvements
- Memory usage analysis
- Quality vs. efficiency tradeoffs
- When the tradeoffs are worth it (and when they're not)

### Prerequisites

- Understanding of [dimension selection](./04-dimension-selection.md)
- Basic knowledge of vector database operations

---

## Storage Savings

### How Storage Scales with Dimensions

Vector embeddings are typically stored as 32-bit floating-point numbers. Each dimension requires 4 bytes:

| Dimensions | Bytes per Vector | Vectors per GB | Reduction |
|------------|------------------|----------------|-----------|
| 3072 | 12,288 | 87,381 | Baseline |
| 1536 | 6,144 | 174,763 | 50% |
| 768 | 3,072 | 349,525 | 75% |
| 512 | 2,048 | 524,288 | 83% |
| 256 | 1,024 | 1,048,576 | 92% |

### Real-World Storage Costs

```python
def calculate_storage_costs(
    num_documents: int,
    full_dims: int = 3072,
    target_dims: int = 768,
    cost_per_gb_month: float = 0.023  # AWS S3 standard
) -> dict:
    """
    Calculate storage savings from dimension reduction.
    
    Args:
        num_documents: Number of documents in your corpus
        full_dims: Full embedding dimensions
        target_dims: Reduced embedding dimensions
        cost_per_gb_month: Storage cost per GB per month
    
    Returns:
        Dictionary with storage analysis
    """
    bytes_per_float = 4
    
    # Full embeddings
    full_bytes = num_documents * full_dims * bytes_per_float
    full_gb = full_bytes / (1024 ** 3)
    full_cost = full_gb * cost_per_gb_month
    
    # Reduced embeddings
    reduced_bytes = num_documents * target_dims * bytes_per_float
    reduced_gb = reduced_bytes / (1024 ** 3)
    reduced_cost = reduced_gb * cost_per_gb_month
    
    return {
        "full_storage_gb": round(full_gb, 2),
        "reduced_storage_gb": round(reduced_gb, 2),
        "storage_savings_gb": round(full_gb - reduced_gb, 2),
        "storage_reduction_pct": round((1 - target_dims/full_dims) * 100, 1),
        "monthly_cost_full": round(full_cost, 2),
        "monthly_cost_reduced": round(reduced_cost, 2),
        "monthly_savings": round(full_cost - reduced_cost, 2)
    }

# Example: 10 million documents
result = calculate_storage_costs(
    num_documents=10_000_000,
    full_dims=3072,
    target_dims=768
)

for key, value in result.items():
    print(f"{key}: {value}")
```

**Output:**
```
full_storage_gb: 114.44
reduced_storage_gb: 28.61
storage_savings_gb: 85.83
storage_reduction_pct: 75.0
monthly_cost_full: 2.63
monthly_cost_reduced: 0.66
monthly_savings: 1.97
```

### Storage at Scale

| Documents | 3072 dims | 768 dims | Savings |
|-----------|-----------|----------|---------|
| 100K | 1.14 GB | 0.29 GB | 0.85 GB |
| 1M | 11.4 GB | 2.9 GB | 8.5 GB |
| 10M | 114 GB | 29 GB | 85 GB |
| 100M | 1.14 TB | 286 GB | 858 GB |
| 1B | 11.4 TB | 2.86 TB | 8.58 TB |

At billion-scale, dimension reduction saves **terabytes** of storage.

---

## Search Latency Improvements

### Why Smaller Dimensions Mean Faster Search

Vector similarity search involves computing dot products between the query vector and database vectors. Computational cost scales **linearly** with dimension count:

$$\text{FLOPS} = 2 \times \text{dimensions} \times \text{num\_vectors}$$

### Brute-Force Search Performance

```python
import numpy as np
import time

def benchmark_search(query_dim: int, num_vectors: int, num_queries: int = 100) -> float:
    """Benchmark brute-force similarity search."""
    # Generate random data
    np.random.seed(42)
    database = np.random.randn(num_vectors, query_dim).astype(np.float32)
    queries = np.random.randn(num_queries, query_dim).astype(np.float32)
    
    # Normalize (for cosine similarity via dot product)
    database = database / np.linalg.norm(database, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Time the search
    start = time.perf_counter()
    for query in queries:
        similarities = np.dot(database, query)
        top_k = np.argsort(-similarities)[:10]
    elapsed = time.perf_counter() - start
    
    return elapsed / num_queries * 1000  # ms per query

# Benchmark different dimensions
dims_to_test = [3072, 1536, 768, 512, 256]
num_vectors = 1_000_000

print("Brute-force search latency (1M vectors):\n")
for dim in dims_to_test:
    latency = benchmark_search(dim, num_vectors)
    print(f"{dim:4d} dims: {latency:.2f} ms/query")
```

**Expected Output:**
```
Brute-force search latency (1M vectors):

3072 dims: 12.4 ms/query
1536 dims:  6.2 ms/query
 768 dims:  3.1 ms/query
 512 dims:  2.1 ms/query
 256 dims:  1.0 ms/query
```

> **Note:** These are CPU benchmarks. GPU performance scales even more dramatically with dimension count.

### HNSW Index Performance

For approximate nearest neighbor indexes like HNSW, smaller dimensions also improve performance:

| Metric | 3072 dims | 768 dims | Improvement |
|--------|-----------|----------|-------------|
| Index build time | 4x | 1x | 4x faster |
| Memory footprint | 4x | 1x | 75% less |
| Query latency | ~1.5x | 1x | 33% faster |

The query latency improvement for HNSW is less dramatic because distance computation is only part of the algorithm, but memory savings remain substantial.

---

## Memory Usage

### RAM Requirements for Vector Databases

Vector databases keep embeddings in memory for fast access. RAM costs significantly more than disk storage:

```python
def calculate_memory_costs(
    num_documents: int,
    dims: int,
    bytes_per_float: int = 4,
    ram_cost_per_gb: float = 5.00  # Cloud RAM: ~$5/GB/month
) -> dict:
    """Calculate RAM costs for vector database."""
    # Vector memory
    vector_bytes = num_documents * dims * bytes_per_float
    vector_gb = vector_bytes / (1024 ** 3)
    
    # Add ~20% overhead for indexes, metadata
    total_gb = vector_gb * 1.2
    
    monthly_cost = total_gb * ram_cost_per_gb
    
    return {
        "pure_vectors_gb": round(vector_gb, 2),
        "with_overhead_gb": round(total_gb, 2),
        "monthly_ram_cost": round(monthly_cost, 2)
    }

# Compare 3072 vs 768 dimensions for 10M vectors
scenarios = [
    {"name": "Full (3072)", "dims": 3072},
    {"name": "Reduced (768)", "dims": 768},
]

print("Memory costs for 10M documents:\n")
for scenario in scenarios:
    result = calculate_memory_costs(10_000_000, scenario["dims"])
    print(f"{scenario['name']:15s}: {result['with_overhead_gb']:6.1f} GB RAM = ${result['monthly_ram_cost']:,.0f}/month")
```

**Output:**
```
Memory costs for 10M documents:

Full (3072)    : 137.3 GB RAM = $687/month
Reduced (768)  :  34.3 GB RAM = $172/month
```

### Instance Sizing Impact

| Documents | 3072 dims RAM | 768 dims RAM | Instance Difference |
|-----------|---------------|--------------|---------------------|
| 1M | 14 GB | 3.5 GB | r6g.large → t3.large |
| 10M | 137 GB | 34 GB | r6g.4xlarge → r6g.xlarge |
| 100M | 1.4 TB | 343 GB | Cluster required → Single instance |

Dimension reduction can mean the difference between needing a cluster vs. a single server.

---

## Quality vs. Efficiency Tradeoffs

### The Efficiency Frontier

Here's the key insight: **quality degrades slowly, but costs drop linearly**.

| Dimension Reduction | Storage/Cost Savings | Quality Loss (MTEB avg) |
|--------------------|---------------------|------------------------|
| 3072 → 1536 (50%) | 50% savings | ~0.1% loss |
| 3072 → 768 (75%) | 75% savings | ~0.2% loss |
| 3072 → 512 (83%) | 83% savings | ~0.5% loss |
| 3072 → 256 (92%) | 92% savings | ~1-2% loss |

### Visualizing the Tradeoff

```python
import matplotlib.pyplot as plt
import numpy as np

# Data from MTEB benchmarks (approximated)
dimensions = [3072, 1536, 768, 512, 256, 128, 64]
quality = [68.2, 68.1, 68.0, 67.7, 67.0, 65.5, 62.0]  # MTEB scores
cost_relative = [100, 50, 25, 17, 8, 4, 2]  # % of full cost

fig, ax1 = plt.subplots(figsize=(10, 6))

# Quality line
ax1.set_xlabel('Dimensions')
ax1.set_ylabel('Quality (MTEB Score)', color='blue')
ax1.plot(dimensions, quality, 'b-o', linewidth=2, label='Quality')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlim(3200, 0)  # Reverse x-axis

# Cost line
ax2 = ax1.twinx()
ax2.set_ylabel('Relative Cost (%)', color='red')
ax2.plot(dimensions, cost_relative, 'r--s', linewidth=2, label='Cost')
ax2.tick_params(axis='y', labelcolor='red')

# Sweet spot annotation
ax1.axvline(x=768, color='green', linestyle=':', alpha=0.7)
ax1.annotate('Sweet spot', xy=(768, 68.0), xytext=(1200, 67.0),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=12, color='green')

plt.title('Matryoshka Embeddings: Quality vs. Cost Tradeoff')
plt.tight_layout()
plt.savefig('quality_vs_cost.png', dpi=150)
plt.show()
```

### The "Sweet Spot"

For most applications, **768 dimensions** represents an excellent tradeoff:
- **75% cost reduction** (storage, memory, compute)
- **<0.5% quality loss** (often imperceptible in practice)
- **Compatible with legacy systems** (many older models used 768)

---

## When Dimension Reduction Is Worth It

### ✅ Strong Use Cases

| Scenario | Why It Works | Recommended Dims |
|----------|--------------|------------------|
| **Large corpus (10M+ docs)** | Storage/RAM savings compound | 256-768 |
| **Real-time search (<50ms)** | Lower latency critical | 256-512 |
| **Mobile/edge deployment** | Memory constrained | 128-256 |
| **High query volume** | Compute costs add up | 512-768 |
| **First-pass retrieval** | Just need rough ranking | 256-512 |

### ❌ When to Keep Full Dimensions

| Scenario | Why Full Dims | Rationale |
|----------|---------------|-----------|
| **Small corpus (<100K)** | Savings negligible | Few dollars difference |
| **Maximum precision required** | Every % matters | Legal discovery, medical |
| **Final re-ranking stage** | Quality is paramount | Use full after reduced retrieval |
| **Rare/specialized queries** | Long tail performance | Reduced dims hurt edge cases |

---

## Hybrid Strategies: Best of Both Worlds

### Two-Stage Retrieval

Store reduced dimensions for fast retrieval, full dimensions for re-ranking:

```python
class HybridRetriever:
    """Two-stage retrieval with dimension reduction."""
    
    def __init__(self, reduced_db, full_db, reduced_dims=256):
        self.reduced_db = reduced_db  # Fast search index (256 dims)
        self.full_db = full_db        # Precise re-rank index (3072 dims)
        self.reduced_dims = reduced_dims
    
    def search(self, query_full: np.ndarray, k: int = 10, candidates: int = 100):
        """
        Two-stage search:
        1. Fast retrieval with reduced dimensions (get candidates)
        2. Precise re-ranking with full dimensions (get final k)
        """
        # Stage 1: Fast search with reduced dimensions
        query_reduced = query_full[:self.reduced_dims]
        query_reduced = query_reduced / np.linalg.norm(query_reduced)
        
        candidate_ids = self.reduced_db.search(query_reduced, k=candidates)
        
        # Stage 2: Re-rank with full dimensions
        candidate_full_embeddings = self.full_db.get_embeddings(candidate_ids)
        
        query_full_norm = query_full / np.linalg.norm(query_full)
        similarities = np.dot(candidate_full_embeddings, query_full_norm)
        
        top_k_indices = np.argsort(-similarities)[:k]
        return [candidate_ids[i] for i in top_k_indices]
```

**Benefits:**
- **Candidate retrieval**: 256 dims, sub-millisecond
- **Re-ranking**: 3072 dims, only on 100 candidates (vs. millions)
- **Result**: Near-full-dimension quality at reduced-dimension cost

### Progressive Dimension Serving

Serve different dimensions based on user tier or query complexity:

```python
def get_dimensions_for_request(user_tier: str, query_complexity: str) -> int:
    """Determine dimensions based on context."""
    
    dimension_matrix = {
        # (user_tier, complexity): dimensions
        ("free", "simple"): 256,
        ("free", "complex"): 512,
        ("pro", "simple"): 512,
        ("pro", "complex"): 768,
        ("enterprise", "simple"): 768,
        ("enterprise", "complex"): 3072,
    }
    
    return dimension_matrix.get((user_tier, query_complexity), 512)
```

---

## Cost Calculator

Here's a complete cost comparison tool:

```python
def full_cost_analysis(
    num_documents: int,
    queries_per_month: int,
    full_dims: int = 3072,
    reduced_dims: int = 768,
    storage_cost_gb: float = 0.023,  # S3
    ram_cost_gb: float = 5.00,       # Cloud RAM
    compute_cost_per_million_ops: float = 0.10
) -> dict:
    """Comprehensive cost comparison between full and reduced dimensions."""
    
    results = {}
    
    for name, dims in [("full", full_dims), ("reduced", reduced_dims)]:
        # Storage
        storage_gb = (num_documents * dims * 4) / (1024 ** 3)
        storage_cost = storage_gb * storage_cost_gb
        
        # RAM
        ram_gb = storage_gb * 1.2  # 20% overhead
        ram_cost = ram_gb * ram_cost_gb
        
        # Compute (approximate: operations scale with dims)
        ops_per_query = dims * num_documents / 1_000_000  # Million ops
        compute_cost = queries_per_month * ops_per_query * compute_cost_per_million_ops / 1_000_000
        
        total_monthly = storage_cost + ram_cost + compute_cost
        
        results[name] = {
            "storage_gb": round(storage_gb, 2),
            "storage_cost": round(storage_cost, 2),
            "ram_gb": round(ram_gb, 2),
            "ram_cost": round(ram_cost, 2),
            "compute_cost": round(compute_cost, 2),
            "total_monthly": round(total_monthly, 2)
        }
    
    results["savings"] = {
        "storage": results["full"]["storage_cost"] - results["reduced"]["storage_cost"],
        "ram": results["full"]["ram_cost"] - results["reduced"]["ram_cost"],
        "compute": results["full"]["compute_cost"] - results["reduced"]["compute_cost"],
        "total": results["full"]["total_monthly"] - results["reduced"]["total_monthly"],
        "percentage": round((1 - results["reduced"]["total_monthly"] / results["full"]["total_monthly"]) * 100, 1)
    }
    
    return results

# Example: 10M documents, 1M queries/month
analysis = full_cost_analysis(
    num_documents=10_000_000,
    queries_per_month=1_000_000
)

print("Full dimensions (3072):")
for k, v in analysis["full"].items():
    print(f"  {k}: ${v}" if "cost" in k or "monthly" in k else f"  {k}: {v}")

print("\nReduced dimensions (768):")
for k, v in analysis["reduced"].items():
    print(f"  {k}: ${v}" if "cost" in k or "monthly" in k else f"  {k}: {v}")

print("\nMonthly Savings:")
print(f"  Total: ${analysis['savings']['total']:.2f} ({analysis['savings']['percentage']}% reduction)")
```

---

## Summary

✅ **75% dimension reduction** = 75% storage savings with <0.5% quality loss  
✅ **Search latency** scales linearly with dimensions  
✅ **RAM costs** often dominate—dimension reduction can change instance requirements  
✅ **768 dimensions** is the common "sweet spot" for most applications  
✅ **Two-stage retrieval** gives near-full quality at reduced-dimension costs  
✅ **Not worth it** for small corpora or maximum-precision requirements

---

**Next:** [Migration Strategies →](./08-migration-strategies.md)

---

<!-- 
Sources Consulted:
- AWS S3 pricing documentation
- MTEB benchmark results for dimension comparison
- Vector database documentation (Pinecone, Weaviate)
-->
