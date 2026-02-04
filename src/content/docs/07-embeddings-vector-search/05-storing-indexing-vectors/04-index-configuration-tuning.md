---
title: "Index Configuration & Tuning"
---

# Index Configuration & Tuning

## Introduction

Index parameters control the trade-off between search quality (recall) and speed (latency). Understanding which parameters to tune—and when—is crucial for production systems.

---

## Build-Time vs Query-Time Parameters

Some parameters are set during index creation and are expensive to change. Others can be adjusted per-query.

```python
class IndexConfig:
    """Separating build and query configuration."""
    
    # Build-time parameters (set once, expensive to change)
    build_params = {
        "M": 32,                  # HNSW: edges per node
        "ef_construction": 200,   # HNSW: build quality
        "nlist": 1000,           # IVF: cluster count
    }
    
    # Query-time parameters (adjustable per request)
    query_params = {
        "ef_search": 100,        # HNSW: search quality
        "nprobe": 10,            # IVF: clusters to search
    }
```

### Parameter Categories

| Parameter | Type | Index | Effect |
|-----------|------|-------|--------|
| `M` | Build | HNSW | More edges = better recall, more memory |
| `ef_construction` | Build | HNSW | Higher = better graph quality, slower build |
| `nlist` | Build | IVF | More clusters = faster search, lower recall |
| `ef_search` | Query | HNSW | Higher = better recall, slower query |
| `nprobe` | Query | IVF | More probes = better recall, slower query |

---

## Recall vs Latency Tuning

### Benchmarking Framework

```python
import time
import numpy as np
import faiss

def benchmark_recall_latency(
    index, 
    queries: np.ndarray, 
    ground_truth: np.ndarray,
    param_name: str, 
    param_values: list,
    k: int = 10
) -> list[dict]:
    """Benchmark different parameter settings."""
    results = []
    
    for value in param_values:
        # Set parameter
        if param_name == "ef_search":
            index.hnsw.efSearch = value
        elif param_name == "nprobe":
            index.nprobe = value
        
        # Measure latency
        start = time.time()
        _, indices = index.search(queries, k)
        total_time = time.time() - start
        latency_ms = (total_time / len(queries)) * 1000
        
        # Calculate recall
        recall = calculate_recall(indices, ground_truth, k)
        
        results.append({
            param_name: value,
            f"recall@{k}": recall,
            "latency_ms": latency_ms,
            "qps": len(queries) / total_time
        })
    
    return results

def calculate_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Calculate recall@k."""
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        correct += len(set(pred[:k]) & set(truth[:k]))
    return correct / (len(predictions) * k)
```

### Example Benchmark Results

```python
# HNSW ef_search tuning
ef_values = [32, 64, 128, 256, 512]
results = benchmark_recall_latency(hnsw_index, queries, ground_truth, "ef_search", ef_values)

for r in results:
    print(f"ef_search={r['ef_search']:3d} | recall@10={r['recall@10']:.3f} | latency={r['latency_ms']:.2f}ms")
```

**Output:**
```
ef_search= 32 | recall@10=0.920 | latency=0.45ms
ef_search= 64 | recall@10=0.960 | latency=0.72ms
ef_search=128 | recall@10=0.985 | latency=1.35ms
ef_search=256 | recall@10=0.995 | latency=2.80ms
ef_search=512 | recall@10=0.998 | latency=5.50ms
```

### Choosing the Right Trade-off

| Use Case | Target Recall | Acceptable Latency |
|----------|---------------|-------------------|
| Real-time search | 95%+ | < 10ms |
| Recommendation | 90%+ | < 50ms |
| Batch processing | 99%+ | < 1s |
| Research/analysis | 99.9%+ | Any |

---

## Comprehensive Benchmarking

```python
def full_benchmark(
    index, 
    test_vectors: np.ndarray, 
    test_queries: np.ndarray, 
    k: int = 10
) -> dict:
    """Comprehensive index benchmark."""
    
    # Build benchmark
    start = time.time()
    index.add(test_vectors)
    build_time = time.time() - start
    
    # Memory benchmark (approximate)
    memory_mb = (index.ntotal * test_vectors.shape[1] * 4) / (1024 * 1024)
    
    # Warm-up queries
    _ = index.search(test_queries[:10], k)
    
    # Query latency benchmark
    latencies = []
    for query in test_queries:
        start = time.time()
        index.search(query.reshape(1, -1), k)
        latencies.append((time.time() - start) * 1000)
    
    # Recall benchmark (compare to exact search)
    exact_index = faiss.IndexFlatL2(test_vectors.shape[1])
    exact_index.add(test_vectors)
    _, ground_truth = exact_index.search(test_queries, k)
    _, approx_results = index.search(test_queries, k)
    recall = calculate_recall(approx_results, ground_truth, k)
    
    return {
        "build_time_sec": round(build_time, 2),
        "memory_mb": round(memory_mb, 2),
        "avg_latency_ms": round(np.mean(latencies), 3),
        "p50_latency_ms": round(np.percentile(latencies, 50), 3),
        "p99_latency_ms": round(np.percentile(latencies, 99), 3),
        f"recall@{k}": round(recall, 4),
        "qps": round(len(test_queries) / sum(latencies) * 1000, 1)
    }

# Usage
results = full_benchmark(index, vectors, queries)
print(f"""
Benchmark Results:
  Build time:    {results['build_time_sec']}s
  Memory:        {results['memory_mb']} MB
  Avg latency:   {results['avg_latency_ms']}ms
  P99 latency:   {results['p99_latency_ms']}ms
  Recall@10:     {results['recall@10']}
  QPS:           {results['qps']}
""")
```

---

## HNSW Tuning Guide

### Build Parameters

| Parameter | Low | Default | High | Trade-off |
|-----------|-----|---------|------|-----------|
| `M` | 8 | 16-32 | 64 | Memory vs recall |
| `ef_construction` | 40 | 100-200 | 500 | Build time vs quality |

```python
# Conservative (low memory)
index_low = faiss.IndexHNSWFlat(dim, M=8)
index_low.hnsw.efConstruction = 40

# Balanced (default)
index_balanced = faiss.IndexHNSWFlat(dim, M=32)
index_balanced.hnsw.efConstruction = 200

# High quality (best recall)
index_high = faiss.IndexHNSWFlat(dim, M=64)
index_high.hnsw.efConstruction = 500
```

### Query Parameters

```python
def adaptive_ef_search(query_priority: str, base_ef: int = 100) -> int:
    """Adjust ef_search based on query priority."""
    multipliers = {
        "fast": 0.5,      # Latency-critical
        "balanced": 1.0,  # Default
        "quality": 2.0,   # Recall-critical
        "exact": 5.0      # Near-exact results
    }
    return int(base_ef * multipliers.get(query_priority, 1.0))

# Usage
index.hnsw.efSearch = adaptive_ef_search("quality")
```

---

## IVF Tuning Guide

### Build Parameters

| Parameter | Formula | Example (1M vectors) |
|-----------|---------|---------------------|
| `nlist` | √N to 4√N | 1000 - 4000 |

```python
import math

def calculate_nlist(num_vectors: int) -> int:
    """Calculate optimal nlist for IVF."""
    sqrt_n = int(math.sqrt(num_vectors))
    # Start with √N, adjust based on memory/speed needs
    return max(100, min(sqrt_n * 2, 65536))

# For 1M vectors
nlist = calculate_nlist(1_000_000)  # Returns ~2000
```

### Query Parameters

```python
def calculate_nprobe(nlist: int, target_recall: float) -> int:
    """Estimate nprobe for target recall."""
    # Rough heuristic: probe 1-10% of clusters
    if target_recall >= 0.99:
        return min(nlist, int(nlist * 0.1))
    elif target_recall >= 0.95:
        return min(nlist, int(nlist * 0.05))
    else:
        return min(nlist, int(nlist * 0.01))

# For nlist=1000, target 95% recall
nprobe = calculate_nprobe(1000, 0.95)  # Returns 50
```

---

## Production Configuration Patterns

### Configuration Class

```python
from dataclasses import dataclass
from enum import Enum

class QualityLevel(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"

@dataclass
class IndexConfiguration:
    """Production index configuration."""
    
    # HNSW parameters
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 100
    
    # IVF parameters (if using IVF)
    nlist: int = 1000
    nprobe: int = 10
    
    @classmethod
    def from_quality_level(cls, level: QualityLevel, num_vectors: int) -> "IndexConfiguration":
        """Create configuration from quality level."""
        configs = {
            QualityLevel.FAST: {
                "M": 16, "ef_construction": 100, "ef_search": 50,
                "nlist": int(math.sqrt(num_vectors)), "nprobe": 5
            },
            QualityLevel.BALANCED: {
                "M": 32, "ef_construction": 200, "ef_search": 100,
                "nlist": int(math.sqrt(num_vectors) * 2), "nprobe": 10
            },
            QualityLevel.HIGH_QUALITY: {
                "M": 64, "ef_construction": 400, "ef_search": 200,
                "nlist": int(math.sqrt(num_vectors) * 4), "nprobe": 20
            }
        }
        return cls(**configs[level])

# Usage
config = IndexConfiguration.from_quality_level(QualityLevel.BALANCED, 1_000_000)
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Benchmark on representative data | Use synthetic data only |
| Measure both recall AND latency | Optimize only one metric |
| Start with defaults, then tune | Over-engineer from the start |
| Document your configuration choices | Leave parameters unexplained |
| Re-benchmark after data changes | Assume static performance |

---

## Summary

✅ **Build parameters** are expensive to change—choose carefully upfront

✅ **Query parameters** can be tuned per-request for different use cases

✅ **Benchmark systematically** with recall AND latency measurements

✅ **Use quality levels** to standardize configurations across your system

**Next:** [Update Strategies](./05-update-strategies.md)
