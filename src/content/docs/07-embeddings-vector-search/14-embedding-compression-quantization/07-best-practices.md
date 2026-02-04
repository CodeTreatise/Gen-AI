---
title: "Best Practices"
---

# Best Practices

## Introduction

Embedding compression is powerful but requires careful implementation. This lesson consolidates best practices for benchmarking, monitoring, and deploying compressed embeddings in production.

Following these guidelines helps you achieve maximum compression benefits while avoiding quality degradation surprises.

### What We'll Cover

- Benchmarking methodology
- Quality monitoring in production
- Deployment strategies
- Common pitfalls and how to avoid them

### Prerequisites

- Familiarity with [quantization trade-offs](./03-quantization-tradeoffs.md)
- Understanding of your vector database's quantization options

---

## Benchmarking Methodology

### The Golden Rule

> **Always benchmark on YOUR data with YOUR queries before deploying compression.**

Generic benchmarks don't reflect your specific:
- Embedding distribution
- Query patterns
- Precision requirements
- Latency budget

### Creating a Benchmark Dataset

```python
import numpy as np
from typing import List, Tuple
import json

def create_benchmark_dataset(
    queries: List[str],
    documents: List[str],
    embed_fn,
    output_path: str,
    k: int = 100
) -> None:
    """
    Create ground truth dataset for benchmarking compression.
    
    Args:
        queries: List of benchmark queries
        documents: List of documents to search
        embed_fn: Function to embed text
        output_path: Path to save benchmark data
        k: Number of ground truth neighbors per query
    """
    # Embed everything at full precision
    print("Embedding documents...")
    doc_embeddings = np.array([embed_fn(d) for d in documents])
    
    print("Embedding queries...")
    query_embeddings = np.array([embed_fn(q) for q in queries])
    
    # Normalize for cosine similarity
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_normalized = doc_embeddings / doc_norms
    query_normalized = query_embeddings / query_norms
    
    # Compute ground truth (brute force, full precision)
    print("Computing ground truth...")
    ground_truth = []
    for i, query in enumerate(query_normalized):
        similarities = np.dot(doc_normalized, query)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        ground_truth.append({
            "query_idx": i,
            "query_text": queries[i],
            "top_k_indices": top_k_indices.tolist(),
            "top_k_scores": similarities[top_k_indices].tolist()
        })
    
    # Save benchmark data
    benchmark = {
        "queries": queries,
        "documents": documents,
        "ground_truth": ground_truth,
        "k": k
    }
    
    with open(output_path, 'w') as f:
        json.dump(benchmark, f)
    
    # Save embeddings separately (for faster loading)
    np.save(output_path.replace('.json', '_docs.npy'), doc_embeddings)
    np.save(output_path.replace('.json', '_queries.npy'), query_embeddings)
    
    print(f"Benchmark saved to {output_path}")
```

### Measuring Compression Quality

```python
def evaluate_compression(
    query_embeddings: np.ndarray,
    compressed_search_fn,
    ground_truth: List[dict],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> dict:
    """
    Evaluate compression against ground truth.
    
    Returns:
        Dictionary of metrics at each k
    """
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"] = []
    
    for i, query_emb in enumerate(query_embeddings):
        # Get compressed search results
        compressed_results = compressed_search_fn(query_emb, max(k_values))
        
        # Get ground truth
        true_top = set(ground_truth[i]["top_k_indices"][:max(k_values)])
        
        # Calculate Recall@k for each k
        for k in k_values:
            true_top_k = set(ground_truth[i]["top_k_indices"][:k])
            pred_top_k = set(compressed_results[:k])
            recall = len(true_top_k & pred_top_k) / k
            results[f"recall@{k}"].append(recall)
        
        # Calculate MRR (Mean Reciprocal Rank)
        true_first = ground_truth[i]["top_k_indices"][0]
        try:
            rank = compressed_results.index(true_first) + 1
            mrr = 1.0 / rank
        except ValueError:
            mrr = 0.0
        results["mrr"].append(mrr)
    
    # Aggregate
    summary = {}
    for metric, values in results.items():
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95)
        }
    
    return summary
```

### What Metrics to Track

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Recall@10 | Top-10 overlap with ground truth | > 0.95 |
| Recall@100 | Broader relevance | > 0.98 |
| MRR | First result quality | > 0.90 |
| P99 Latency | Worst-case speed | Application-specific |
| Memory Usage | RAM consumption | Fit in budget |

---

## Quality Monitoring in Production

### The Monitoring Challenge

Compression quality can degrade over time due to:
- Data drift (embedding distribution changes)
- New query patterns
- Model updates
- Codebook staleness (for PQ)

### Shadow Testing

Run compressed and full-precision search in parallel:

```python
import logging
from typing import List, Tuple
import time

class ShadowSearchMonitor:
    """Monitor compression quality by comparing to full-precision."""
    
    def __init__(
        self,
        compressed_search_fn,
        full_search_fn,
        sample_rate: float = 0.01,  # 1% of queries
        alert_threshold: float = 0.90  # Alert if recall drops below
    ):
        self.compressed_fn = compressed_search_fn
        self.full_fn = full_search_fn
        self.sample_rate = sample_rate
        self.alert_threshold = alert_threshold
        self.logger = logging.getLogger("shadow_search")
        
        # Metrics storage
        self.recent_recalls = []
        self.window_size = 1000
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[int], float]:
        """
        Perform search with optional shadow comparison.
        
        Returns:
            (results, latency)
        """
        # Always run compressed search
        start = time.time()
        compressed_results = self.compressed_fn(query, top_k)
        latency = time.time() - start
        
        # Probabilistically run shadow comparison
        if np.random.random() < self.sample_rate:
            self._shadow_compare(query, compressed_results, top_k)
        
        return compressed_results, latency
    
    def _shadow_compare(
        self,
        query: np.ndarray,
        compressed_results: List[int],
        top_k: int
    ) -> None:
        """Compare compressed results to full-precision."""
        full_results = self.full_fn(query, top_k)
        
        # Calculate recall
        compressed_set = set(compressed_results)
        full_set = set(full_results)
        recall = len(compressed_set & full_set) / top_k
        
        # Track recall
        self.recent_recalls.append(recall)
        if len(self.recent_recalls) > self.window_size:
            self.recent_recalls.pop(0)
        
        # Check for degradation
        avg_recall = np.mean(self.recent_recalls)
        if avg_recall < self.alert_threshold:
            self.logger.warning(
                f"Compression quality degraded! "
                f"Average Recall@{top_k}: {avg_recall:.3f} "
                f"(threshold: {self.alert_threshold})"
            )
    
    def get_metrics(self) -> dict:
        """Get current quality metrics."""
        if not self.recent_recalls:
            return {"status": "no_data"}
        
        return {
            "avg_recall": np.mean(self.recent_recalls),
            "min_recall": np.min(self.recent_recalls),
            "p5_recall": np.percentile(self.recent_recalls, 5),
            "samples": len(self.recent_recalls)
        }
```

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Avg Recall@10 | < 0.93 | < 0.90 |
| Min Recall@10 | < 0.70 | < 0.50 |
| P5 Recall | < 0.85 | < 0.75 |
| Latency P99 | 2x baseline | 5x baseline |

---

## Deployment Strategies

### Gradual Rollout

```
┌─────────────────────────────────────────────────────────────────┐
│              Compression Rollout Strategy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Shadow Mode (1 week)                                  │
│  ┌──────────────────────────────────────────────┐              │
│  │ 100% traffic: Full precision (serving)       │              │
│  │ 10% traffic: Compressed (shadow comparison)  │              │
│  └──────────────────────────────────────────────┘              │
│  Goal: Validate quality metrics match benchmarks               │
│                                                                 │
│  Phase 2: Canary (1-2 weeks)                                   │
│  ┌──────────────────────────────────────────────┐              │
│  │ 5% traffic: Compressed (with full fallback)  │              │
│  │ 95% traffic: Full precision                  │              │
│  └──────────────────────────────────────────────┘              │
│  Goal: Validate user-facing metrics (CTR, engagement)          │
│                                                                 │
│  Phase 3: Gradual Increase (2-4 weeks)                         │
│  ┌──────────────────────────────────────────────┐              │
│  │ 5% → 25% → 50% → 75% → 100% compressed       │              │
│  └──────────────────────────────────────────────┘              │
│  Goal: Monitor for edge cases, scale issues                    │
│                                                                 │
│  Phase 4: Maintenance                                          │
│  ┌──────────────────────────────────────────────┐              │
│  │ 100% compressed with 1% shadow monitoring    │              │
│  └──────────────────────────────────────────────┘              │
│  Goal: Catch drift, maintain quality                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Fallback Patterns

```python
class ResilientSearch:
    """Search with automatic fallback on quality issues."""
    
    def __init__(
        self,
        compressed_client,
        full_precision_client,
        quality_threshold: float = 0.85
    ):
        self.compressed = compressed_client
        self.full = full_precision_client
        self.threshold = quality_threshold
        self.use_fallback = False
        self.fallback_reason = None
    
    def search(self, query, top_k: int = 10):
        """Search with automatic fallback."""
        if self.use_fallback:
            return self.full.search(query, top_k)
        
        try:
            results = self.compressed.search(query, top_k)
            
            # Quick quality check (e.g., score distribution)
            if self._quality_check_failed(results):
                self.use_fallback = True
                self.fallback_reason = "quality_check_failed"
                return self.full.search(query, top_k)
            
            return results
            
        except Exception as e:
            # Fallback on errors
            self.use_fallback = True
            self.fallback_reason = str(e)
            return self.full.search(query, top_k)
    
    def _quality_check_failed(self, results) -> bool:
        """Quick sanity check on results."""
        if not results:
            return True
        
        # Check if top scores are reasonable
        scores = [r.score for r in results]
        if max(scores) < 0.5:  # Very low confidence
            return True
        
        return False
    
    def reset_fallback(self):
        """Attempt to resume compressed search."""
        self.use_fallback = False
        self.fallback_reason = None
```

### Dual Storage Pattern

Store both compressed and full-precision for flexibility:

```python
class DualStorageIndex:
    """Store compressed for search, full for rescoring."""
    
    def __init__(self, compressed_collection, full_collection):
        self.compressed = compressed_collection
        self.full = full_collection
    
    def insert(self, doc_id: str, embedding: np.ndarray, metadata: dict):
        """Insert into both indices."""
        # Compressed version for fast search
        self.compressed.upsert([{
            "id": doc_id,
            "vector": embedding.tolist(),
            "metadata": metadata
        }])
        
        # Full precision for rescoring (could be on cheaper storage)
        self.full.upsert([{
            "id": doc_id,
            "vector": embedding.tolist(),
            "metadata": metadata
        }])
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        candidates: int = 100,
        rescore: bool = True
    ):
        """Two-stage search with optional rescoring."""
        # Stage 1: Fast compressed search
        compressed_results = self.compressed.search(
            query.tolist(),
            limit=candidates
        )
        
        if not rescore:
            return compressed_results[:top_k]
        
        # Stage 2: Rescore with full precision
        candidate_ids = [r.id for r in compressed_results]
        full_vectors = self.full.fetch(candidate_ids)
        
        # Compute precise similarities
        query_norm = query / np.linalg.norm(query)
        scores = []
        for doc_id, vec in full_vectors.items():
            vec_norm = np.array(vec) / np.linalg.norm(vec)
            sim = np.dot(query_norm, vec_norm)
            scores.append((doc_id, sim))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

---

## Common Pitfalls

### Pitfall 1: Skipping Benchmarks

**Problem:** Deploying compression based on general claims without testing on your data.

**Solution:**
```python
# Always run this before deploying
def validate_compression(benchmark_path: str, compressed_search_fn) -> bool:
    """Gate deployment on benchmark results."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    queries = np.load(benchmark_path.replace('.json', '_queries.npy'))
    
    metrics = evaluate_compression(
        queries,
        compressed_search_fn,
        benchmark["ground_truth"]
    )
    
    recall_10 = metrics["recall@10"]["mean"]
    
    if recall_10 < 0.95:
        raise ValueError(
            f"Compression quality too low: Recall@10 = {recall_10:.3f} "
            f"(minimum: 0.95)"
        )
    
    print(f"✓ Compression validated: Recall@10 = {recall_10:.3f}")
    return True
```

### Pitfall 2: Ignoring Data Distribution

**Problem:** Embedding distributions vary by domain. A codebook trained on news articles won't work well for code.

**Solution:**
- Train codebooks on your actual data
- Retrain when data distribution shifts
- Use domain-specific training samples

### Pitfall 3: Over-Compressing

**Problem:** Choosing maximum compression without considering use case.

**Solution:**

| Application | Max Acceptable Compression | Why |
|-------------|---------------------------|-----|
| E-commerce search | 4x (int8) | Revenue depends on quality |
| Document retrieval | 16x (binary + rescore) | Lower stakes |
| Semantic cache | 32x | Fuzzy matching OK |
| Duplicate detection | 32x | Exact match not needed |

### Pitfall 4: Not Monitoring Drift

**Problem:** Compression quality degrades silently as data changes.

**Solution:**
```python
# Schedule regular quality checks
def weekly_quality_check():
    """Run comprehensive quality check weekly."""
    # Sample recent queries from logs
    recent_queries = get_recent_query_sample(n=1000)
    
    # Run against current compression
    metrics = evaluate_current_compression(recent_queries)
    
    # Compare to baseline
    baseline = load_baseline_metrics()
    
    degradation = baseline["recall@10"]["mean"] - metrics["recall@10"]["mean"]
    
    if degradation > 0.02:  # 2% degradation
        alert_oncall(
            f"Compression quality degraded by {degradation:.1%}. "
            f"Consider retraining codebooks."
        )
```

### Pitfall 5: Rescoring Without Candidates

**Problem:** Binary search with too few candidates for rescoring.

**Solution:**
```python
# Rule of thumb
def calculate_candidates(top_k: int, compression: str) -> int:
    """Calculate appropriate candidate count for rescoring."""
    multipliers = {
        "int8": 1.5,
        "binary_1bit": 3.0,
        "binary_2bit": 2.0,
        "pq": 4.0
    }
    
    multiplier = multipliers.get(compression, 2.0)
    return int(top_k * multiplier)

# Example
top_k = 10
candidates = calculate_candidates(top_k, "binary_1bit")
print(f"For top-{top_k} with binary: fetch {candidates} candidates")
# Output: For top-10 with binary: fetch 30 candidates
```

---

## Quick Reference Checklist

### Before Deploying Compression

- [ ] Created benchmark dataset with ground truth
- [ ] Measured Recall@10 > 0.95 (or your threshold)
- [ ] Tested with production query patterns
- [ ] Validated latency meets requirements
- [ ] Set up monitoring and alerting

### Configuration Recommendations

| Collection Size | Recommended | Oversampling |
|----------------|-------------|--------------|
| < 100K | No compression | N/A |
| 100K - 1M | Scalar (int8) | 1.2-1.5x |
| 1M - 10M | Scalar or Binary + rescore | 1.5-2.0x |
| 10M - 100M | Binary + rescore | 2.0-3.0x |
| > 100M | Binary or PQ + rescore | 3.0-4.0x |

### Monitoring Checklist

- [ ] Shadow testing at 1% of traffic
- [ ] Weekly quality benchmark runs
- [ ] Alerting on recall degradation
- [ ] Dashboard for compression metrics
- [ ] Runbook for quality incidents

---

## Summary

✅ **Benchmark on your data** before any compression deployment  
✅ **Shadow test** to validate quality matches production patterns  
✅ **Roll out gradually** with automatic fallback capability  
✅ **Monitor continuously** for drift and quality degradation  
✅ **Store full-precision** embeddings for rescoring flexibility  
✅ **Match compression level** to your quality requirements

---

**Previous:** [Product Quantization](./06-product-quantization.md)

**Back to:** [Lesson Overview](./00-embedding-compression-quantization.md)

---

<!-- 
Sources Consulted:
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- Production ML Best Practices (internal experience)
-->
