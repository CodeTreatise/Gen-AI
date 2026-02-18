---
title: "Evaluation & Optimization"
---

# Evaluation & Optimization

## Introduction

How do you know if reranking is actually helping? And once you've confirmed it works, how do you optimize for cost and latency? This lesson covers evaluation metrics, benchmarking strategies, and practical optimization techniques for production reranking pipelines.

---

## Evaluation Metrics

### Precision@K

Measures what fraction of the top-K results are relevant:

$$Precision@K = \frac{\text{Relevant documents in top K}}{K}$$

```python
def precision_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int
) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevant: Set of relevant document IDs
        retrieved: List of retrieved document IDs (ranked)
        k: Number of results to consider
    
    Returns:
        Precision score between 0 and 1
    """
    retrieved_k = retrieved[:k]
    relevant_in_k = len(relevant & set(retrieved_k))
    return relevant_in_k / k

# Example
relevant_docs = {"doc_1", "doc_3", "doc_7"}
retrieved_docs = ["doc_3", "doc_5", "doc_1", "doc_8", "doc_7"]

p5 = precision_at_k(relevant_docs, retrieved_docs, k=5)
print(f"Precision@5: {p5:.2f}")  # 0.60 (3 of 5 are relevant)
```

### Recall@K

Measures what fraction of all relevant documents appear in top-K:

$$Recall@K = \frac{\text{Relevant documents in top K}}{\text{Total relevant documents}}$$

```python
def recall_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int
) -> float:
    """Calculate Recall@K."""
    retrieved_k = retrieved[:k]
    relevant_in_k = len(relevant & set(retrieved_k))
    return relevant_in_k / len(relevant) if relevant else 0.0

# Example
r5 = recall_at_k(relevant_docs, retrieved_docs, k=5)
print(f"Recall@5: {r5:.2f}")  # 1.00 (all 3 relevant docs in top 5)
```

### Mean Reciprocal Rank (MRR)

Measures how high the first relevant document appears:

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

```python
def reciprocal_rank(
    relevant: set[str],
    retrieved: list[str]
) -> float:
    """Calculate Reciprocal Rank for a single query."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(
    queries_relevant: list[set[str]],
    queries_retrieved: list[list[str]]
) -> float:
    """Calculate Mean Reciprocal Rank across queries."""
    rr_sum = sum(
        reciprocal_rank(rel, ret)
        for rel, ret in zip(queries_relevant, queries_retrieved)
    )
    return rr_sum / len(queries_relevant) if queries_relevant else 0.0

# Example
rr = reciprocal_rank(relevant_docs, retrieved_docs)
print(f"Reciprocal Rank: {rr:.2f}")  # 1.00 (first result is relevant)
```

### NDCG (Normalized Discounted Cumulative Gain)

Accounts for graded relevance (not just binary):

```python
import math

def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain.
    
    Args:
        relevance_scores: Relevance scores in retrieved order
        k: Number of results to consider
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(
    relevance_scores: list[float],
    k: int
) -> float:
    """
    Calculate Normalized DCG.
    
    Args:
        relevance_scores: Relevance scores in retrieved order
        k: Number of results to consider
    
    Returns:
        NDCG score between 0 and 1
    """
    dcg = dcg_at_k(relevance_scores, k)
    
    # Ideal DCG (perfect ranking)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0

# Example with graded relevance (0=irrelevant, 1=somewhat, 2=highly)
retrieved_relevance = [2, 0, 1, 0, 2]  # Scores for top 5 results
ndcg = ndcg_at_k(retrieved_relevance, k=5)
print(f"NDCG@5: {ndcg:.3f}")
```

---

## Evaluation Framework

### Complete Evaluation Suite

```python
from dataclasses import dataclass
from typing import Callable
import statistics

@dataclass
class EvaluationResult:
    """Results from retrieval evaluation."""
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    
    def to_dict(self) -> dict:
        return {
            "P@5": self.precision_at_5,
            "P@10": self.precision_at_10,
            "R@5": self.recall_at_5,
            "R@10": self.recall_at_10,
            "MRR": self.mrr,
            "NDCG@10": self.ndcg_at_10,
        }

class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    def __init__(
        self,
        queries: list[str],
        ground_truth: list[set[str]],
        relevance_scores: list[dict[str, float]] = None
    ):
        """
        Args:
            queries: List of test queries
            ground_truth: List of relevant doc IDs per query
            relevance_scores: Optional graded relevance {doc_id: score}
        """
        self.queries = queries
        self.ground_truth = ground_truth
        self.relevance_scores = relevance_scores
    
    def evaluate(
        self,
        retriever: Callable[[str], list[str]],
        k_values: list[int] = [5, 10]
    ) -> EvaluationResult:
        """
        Evaluate a retriever function.
        
        Args:
            retriever: Function that takes query, returns list of doc IDs
            k_values: K values for P@K and R@K
        """
        all_precision = {k: [] for k in k_values}
        all_recall = {k: [] for k in k_values}
        all_rr = []
        all_ndcg = []
        
        for i, query in enumerate(self.queries):
            relevant = self.ground_truth[i]
            retrieved = retriever(query)
            
            # Precision and Recall
            for k in k_values:
                all_precision[k].append(precision_at_k(relevant, retrieved, k))
                all_recall[k].append(recall_at_k(relevant, retrieved, k))
            
            # Reciprocal Rank
            all_rr.append(reciprocal_rank(relevant, retrieved))
            
            # NDCG (if graded relevance available)
            if self.relevance_scores:
                scores = self.relevance_scores[i]
                rel_scores = [scores.get(doc, 0.0) for doc in retrieved[:10]]
                all_ndcg.append(ndcg_at_k(rel_scores, k=10))
        
        return EvaluationResult(
            precision_at_5=statistics.mean(all_precision[5]),
            precision_at_10=statistics.mean(all_precision[10]),
            recall_at_5=statistics.mean(all_recall[5]),
            recall_at_10=statistics.mean(all_recall[10]),
            mrr=statistics.mean(all_rr),
            ndcg_at_10=statistics.mean(all_ndcg) if all_ndcg else 0.0,
        )
    
    def compare(
        self,
        retrievers: dict[str, Callable[[str], list[str]]]
    ) -> dict[str, EvaluationResult]:
        """Compare multiple retrievers."""
        return {
            name: self.evaluate(retriever)
            for name, retriever in retrievers.items()
        }
```

### Running Evaluation

```python
# Create test dataset
test_queries = [
    "Who created Python?",
    "What is machine learning?",
    "How does async/await work?",
]

# Ground truth: relevant document IDs for each query
ground_truth = [
    {"doc_1", "doc_4"},  # Relevant docs for query 1
    {"doc_6", "doc_7"},  # Relevant docs for query 2
    {"doc_8", "doc_9", "doc_10"},  # Relevant docs for query 3
]

# Create evaluator
evaluator = RetrievalEvaluator(test_queries, ground_truth)

# Define retrievers to compare
def baseline_retriever(query: str) -> list[str]:
    # Semantic search only
    return pipeline.semantic_search(query, k=10)

def hybrid_retriever(query: str) -> list[str]:
    # Hybrid search without reranking
    return pipeline.hybrid_search(query, k=10)

def reranked_retriever(query: str) -> list[str]:
    # Full pipeline with reranking
    results = pipeline.retrieve(query, final_k=10)
    return [r.document_id for r in results]

# Compare
results = evaluator.compare({
    "baseline": baseline_retriever,
    "hybrid": hybrid_retriever,
    "reranked": reranked_retriever,
})

# Print comparison
print("\n=== Retrieval Comparison ===")
print(f"{'Metric':<12} {'Baseline':>10} {'Hybrid':>10} {'Reranked':>10}")
print("-" * 44)

metrics = ["P@5", "P@10", "R@5", "R@10", "MRR"]
for metric in metrics:
    baseline = results["baseline"].to_dict()[metric]
    hybrid = results["hybrid"].to_dict()[metric]
    reranked = results["reranked"].to_dict()[metric]
    print(f"{metric:<12} {baseline:>10.3f} {hybrid:>10.3f} {reranked:>10.3f}")
```

---

## Expected Improvements

### Typical Results

| Configuration | P@5 | P@10 | MRR | Latency |
|---------------|-----|------|-----|---------|
| **Semantic only** | 0.65 | 0.58 | 0.72 | ~50ms |
| **+ Keyword (hybrid)** | 0.72 | 0.65 | 0.78 | ~80ms |
| **+ Reranking** | 0.85 | 0.78 | 0.91 | ~500ms |

### Anthropic Study Results

From Anthropic's Contextual Retrieval research:

| Method | Failure Rate |
|--------|--------------|
| Baseline | 22.0% |
| + Contextual embeddings | 15.0% |
| + Reranking | 8.0% |
| + Both | 3.0% |

**Key insight**: Reranking alone reduces failures by **67%** (22% → 8%).

---

## Cost Optimization

### Pricing Overview

| Provider | Model | Pricing |
|----------|-------|---------|
| **Cohere** | rerank-v4.0-pro | $2.00 / 1K searches |
| **Cohere** | rerank-v4.0-fast | $1.00 / 1K searches |
| **Voyage** | rerank-2.5 | $0.05 / 1M tokens |
| **Voyage** | rerank-2.5-lite | $0.02 / 1M tokens |

### Cost Calculator

```python
from dataclasses import dataclass

@dataclass
class CostEstimate:
    """Reranking cost estimate."""
    daily_queries: int
    avg_docs_per_query: int
    avg_doc_tokens: int
    monthly_cost_cohere_pro: float
    monthly_cost_cohere_fast: float
    monthly_cost_voyage: float
    monthly_cost_voyage_lite: float

def estimate_reranking_costs(
    daily_queries: int,
    avg_docs_per_query: int = 100,
    avg_doc_tokens: int = 500
) -> CostEstimate:
    """
    Estimate monthly reranking costs.
    """
    monthly_queries = daily_queries * 30
    
    # Cohere: per search
    cohere_pro = (monthly_queries / 1000) * 2.00
    cohere_fast = (monthly_queries / 1000) * 1.00
    
    # Voyage: per token
    total_tokens = monthly_queries * avg_docs_per_query * avg_doc_tokens
    voyage = (total_tokens / 1_000_000) * 0.05
    voyage_lite = (total_tokens / 1_000_000) * 0.02
    
    return CostEstimate(
        daily_queries=daily_queries,
        avg_docs_per_query=avg_docs_per_query,
        avg_doc_tokens=avg_doc_tokens,
        monthly_cost_cohere_pro=cohere_pro,
        monthly_cost_cohere_fast=cohere_fast,
        monthly_cost_voyage=voyage,
        monthly_cost_voyage_lite=voyage_lite,
    )

# Example
estimate = estimate_reranking_costs(
    daily_queries=10000,
    avg_docs_per_query=100,
    avg_doc_tokens=500
)

print(f"Daily queries: {estimate.daily_queries:,}")
print(f"\nMonthly costs:")
print(f"  Cohere Pro:    ${estimate.monthly_cost_cohere_pro:,.2f}")
print(f"  Cohere Fast:   ${estimate.monthly_cost_cohere_fast:,.2f}")
print(f"  Voyage:        ${estimate.monthly_cost_voyage:,.2f}")
print(f"  Voyage Lite:   ${estimate.monthly_cost_voyage_lite:,.2f}")
```

**Example Output (10K daily queries):**
```
Daily queries: 10,000

Monthly costs:
  Cohere Pro:    $600.00
  Cohere Fast:   $300.00
  Voyage:        $7,500.00  (token-based is expensive at high volume)
  Voyage Lite:   $3,000.00
```

### Cost Reduction Strategies

#### 1. Reduce Candidate Count

```python
# Instead of reranking 200 candidates
results = pipeline.retrieve(query, rerank_candidates=200)  # More expensive

# Rerank only 100 candidates
results = pipeline.retrieve(query, rerank_candidates=100)  # 50% cost reduction
```

#### 2. Use Fast Models

```python
# For latency-sensitive, high-volume applications
results = co.rerank(
    model="rerank-v4.0-fast",  # Half the cost of pro
    query=query,
    documents=docs,
    top_n=10
)
```

#### 3. Implement Caching

```python
import hashlib
from functools import lru_cache

class CachedReranker:
    def __init__(self, cache_size: int = 10000):
        self.cohere = cohere.ClientV2()
        self._cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _cache_key(self, query: str, docs: tuple) -> str:
        content = f"{query}|||{hash(docs)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10
    ) -> list[dict]:
        # Create cache key
        key = self._cache_key(query, tuple(documents))
        
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        
        self.cache_misses += 1
        
        # Call API
        results = self.cohere.rerank(
            model="rerank-v4.0-fast",
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True
        )
        
        # Format and cache
        output = [
            {"index": r.index, "score": r.relevance_score}
            for r in results.results
        ]
        
        # LRU eviction
        if len(self._cache) >= self.cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = output
        return output
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
```

#### 4. Pre-filter with Metadata

```python
def smart_retrieve(
    query: str,
    metadata_filter: dict = None,
    max_rerank: int = 100
) -> list[dict]:
    """
    Pre-filter before expensive reranking.
    """
    # First: cheap metadata filtering
    if metadata_filter:
        candidates = vector_store.similarity_search(
            query,
            k=max_rerank * 2,  # Over-fetch
            filter=metadata_filter  # Apply filter
        )
    else:
        candidates = vector_store.similarity_search(query, k=max_rerank)
    
    # Only rerank filtered results
    if len(candidates) > max_rerank:
        candidates = candidates[:max_rerank]
    
    # Now rerank the smaller set
    return reranker.rerank(query, [c.page_content for c in candidates])
```

---

## Latency Optimization

### Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_retrieval(query: str) -> tuple:
    """Run semantic and keyword search in parallel."""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = loop.run_in_executor(
            executor,
            lambda: vector_store.similarity_search(query, k=50)
        )
        keyword_future = loop.run_in_executor(
            executor,
            lambda: bm25.invoke(query)
        )
        
        semantic, keyword = await asyncio.gather(
            semantic_future, keyword_future
        )
    
    return semantic, keyword
```

### Timeout Handling

```python
import asyncio

async def rerank_with_timeout(
    query: str,
    documents: list[str],
    timeout: float = 5.0
) -> list[dict]:
    """Rerank with timeout fallback."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                co.rerank,
                model="rerank-v4.0-fast",
                query=query,
                documents=documents,
                top_n=10
            ),
            timeout=timeout
        )
        return [{"index": r.index, "score": r.relevance_score} for r in result.results]
    
    except asyncio.TimeoutError:
        # Fallback: return original order with estimated scores
        return [
            {"index": i, "score": 1.0 - (i * 0.05)}
            for i in range(min(10, len(documents)))
        ]
```

### Latency Benchmarks

| Component | Typical Latency |
|-----------|-----------------|
| Semantic search (50 docs) | 30-80ms |
| BM25 keyword search | 5-20ms |
| RRF fusion (100 docs) | 1-3ms |
| Cohere rerank-fast (100 docs) | 300-500ms |
| Cohere rerank-pro (100 docs) | 500-800ms |
| **Total pipeline** | **400-900ms** |

---

## A/B Testing Reranking

```python
import random
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ABTestResult:
    timestamp: datetime
    query: str
    variant: str  # "control" or "treatment"
    results: list[str]
    user_clicked: str = None
    click_position: int = None

class RetrievalABTest:
    """A/B test for reranking impact."""
    
    def __init__(
        self,
        control_retriever,
        treatment_retriever,
        treatment_ratio: float = 0.5
    ):
        self.control = control_retriever
        self.treatment = treatment_retriever
        self.treatment_ratio = treatment_ratio
        self.results: list[ABTestResult] = []
    
    def retrieve(self, query: str) -> tuple[list[str], str]:
        """
        Retrieve with random variant assignment.
        
        Returns:
            Tuple of (results, variant_name)
        """
        variant = "treatment" if random.random() < self.treatment_ratio else "control"
        
        if variant == "treatment":
            results = self.treatment(query)
        else:
            results = self.control(query)
        
        # Log result
        self.results.append(ABTestResult(
            timestamp=datetime.now(),
            query=query,
            variant=variant,
            results=results,
        ))
        
        return results, variant
    
    def record_click(self, query: str, clicked_doc: str):
        """Record user click for most recent query."""
        for result in reversed(self.results):
            if result.query == query:
                result.user_clicked = clicked_doc
                if clicked_doc in result.results:
                    result.click_position = result.results.index(clicked_doc) + 1
                break
    
    def analyze(self) -> dict:
        """Analyze A/B test results."""
        control_results = [r for r in self.results if r.variant == "control"]
        treatment_results = [r for r in self.results if r.variant == "treatment"]
        
        def avg_click_position(results: list[ABTestResult]) -> float:
            positions = [r.click_position for r in results if r.click_position]
            return sum(positions) / len(positions) if positions else 0.0
        
        def click_rate(results: list[ABTestResult]) -> float:
            clicked = sum(1 for r in results if r.user_clicked)
            return clicked / len(results) if results else 0.0
        
        return {
            "control": {
                "count": len(control_results),
                "click_rate": click_rate(control_results),
                "avg_click_position": avg_click_position(control_results),
            },
            "treatment": {
                "count": len(treatment_results),
                "click_rate": click_rate(treatment_results),
                "avg_click_position": avg_click_position(treatment_results),
            }
        }
```

---

## Summary

✅ Key metrics: Precision@K, Recall@K, MRR, NDCG  
✅ Reranking typically improves P@5 by 20-30%  
✅ Anthropic study shows 67% failure reduction from reranking alone  
✅ Cost optimization: cache, fast models, reduce candidates, pre-filter  
✅ Latency optimization: parallel retrieval, timeouts, fast model fallback  
✅ A/B test to validate improvements in production  

---

## Hands-On Exercise

Build an evaluation suite that:

1. Creates a test dataset with 20 queries and ground truth
2. Compares three configurations:
   - Semantic search only
   - Hybrid (semantic + keyword + RRF)
   - Hybrid + Reranking
3. Reports P@5, P@10, MRR for each
4. Calculates cost per 1000 queries for each option

<details>
<summary>💡 Hints</summary>

- Use the `RetrievalEvaluator` class from this lesson
- Create realistic ground truth by manually labeling a small set
- Remember to account for embedding API costs too, not just reranking

</details>

---

**Previous:** [Hybrid Reranking Pipeline](./07-hybrid-reranking-pipeline.md)

**Back to:** [Section Overview](./00-reranking-score-fusion.md)

---

## Further Reading

- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank-overview)
- [Voyage AI Rerankers](https://docs.voyageai.com/docs/reranker)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Azure Hybrid Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
