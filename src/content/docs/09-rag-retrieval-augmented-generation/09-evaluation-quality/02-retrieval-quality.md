---
title: "Retrieval Quality Evaluation"
---

# Retrieval Quality Evaluation

## Introduction

Before generation quality can be evaluated, we need to ensure retrieval is working well. Poor retrieval means the LLM has the wrong context, leading to irrelevant or incorrect answers regardless of generation quality.

This lesson covers the key metrics for evaluating retrieval quality in RAG systems.

### What We'll Cover

- Precision and recall for retrieval
- F1 score calculation
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### Prerequisites

- Understanding of RAG retrieval
- Basic statistics concepts
- Vector search fundamentals

---

## Precision and Recall

The fundamental metrics for any retrieval system.

### Definitions

| Metric | Formula | Question Answered |
|--------|---------|-------------------|
| **Precision** | Relevant Retrieved / Total Retrieved | How much of what we retrieved is useful? |
| **Recall** | Relevant Retrieved / Total Relevant | How much of the useful stuff did we find? |

```python
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    precision: float
    recall: float
    f1: float
    retrieved_count: int
    relevant_count: int
    relevant_retrieved: int

def calculate_precision_recall(
    retrieved_ids: list[str],
    relevant_ids: list[str]
) -> RetrievalMetrics:
    """
    Calculate precision, recall, and F1 for retrieval.
    
    Args:
        retrieved_ids: IDs of documents retrieved by the system
        relevant_ids: IDs of documents that are actually relevant (ground truth)
    """
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    
    relevant_retrieved = retrieved_set & relevant_set
    
    # Precision: What fraction of retrieved is relevant?
    precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
    
    # Recall: What fraction of relevant did we retrieve?
    recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
    
    # F1: Harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        retrieved_count=len(retrieved_set),
        relevant_count=len(relevant_set),
        relevant_retrieved=len(relevant_retrieved)
    )

# Example
retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
relevant = ["doc_1", "doc_3", "doc_6", "doc_7"]

metrics = calculate_precision_recall(retrieved, relevant)
print(f"Precision: {metrics.precision:.2f}")  # 0.40 (2/5)
print(f"Recall: {metrics.recall:.2f}")        # 0.50 (2/4)
print(f"F1: {metrics.f1:.2f}")                # 0.44
```

### Precision@K and Recall@K

Evaluate metrics at specific cutoff points:

```python
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Precision considering only top-k retrieved documents."""
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_set)
    return relevant_in_top_k / k if k > 0 else 0.0

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Recall considering only top-k retrieved documents."""
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    relevant_in_top_k = len(top_k & relevant_set)
    return relevant_in_top_k / len(relevant_set) if relevant_set else 0.0

# Example: Evaluate at different k values
retrieved = ["doc_1", "doc_5", "doc_3", "doc_2", "doc_4"]
relevant = ["doc_1", "doc_3", "doc_6"]

for k in [1, 3, 5]:
    p_k = precision_at_k(retrieved, relevant, k)
    r_k = recall_at_k(retrieved, relevant, k)
    print(f"P@{k}: {p_k:.2f}, R@{k}: {r_k:.2f}")

# Output:
# P@1: 1.00, R@1: 0.33
# P@3: 0.67, R@3: 0.67
# P@5: 0.40, R@5: 0.67
```

---

## Mean Reciprocal Rank (MRR)

MRR measures how quickly the system returns the first relevant result.

### Formula

```
MRR = (1/N) × Σ(1/rank_i)
```

Where `rank_i` is the position of the first relevant document for query i.

```python
from typing import Optional

def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """
    Calculate reciprocal rank for a single query.
    
    Returns 1/rank of first relevant document, or 0 if none found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(
    queries: list[dict]
) -> float:
    """
    Calculate MRR across multiple queries.
    
    Each query dict should have:
    - 'retrieved': list of retrieved doc IDs
    - 'relevant': set/list of relevant doc IDs
    """
    if not queries:
        return 0.0
    
    total_rr = 0.0
    for query in queries:
        relevant_set = set(query['relevant'])
        rr = reciprocal_rank(query['retrieved'], relevant_set)
        total_rr += rr
    
    return total_rr / len(queries)

# Example
queries = [
    {
        'retrieved': ['doc_3', 'doc_1', 'doc_2'],
        'relevant': {'doc_1', 'doc_4'}
    },  # First relevant at position 2 → RR = 0.5
    {
        'retrieved': ['doc_1', 'doc_2', 'doc_3'],
        'relevant': {'doc_1', 'doc_2'}
    },  # First relevant at position 1 → RR = 1.0
    {
        'retrieved': ['doc_5', 'doc_6', 'doc_7'],
        'relevant': {'doc_1', 'doc_2'}
    }   # No relevant found → RR = 0.0
]

mrr = mean_reciprocal_rank(queries)
print(f"MRR: {mrr:.3f}")  # (0.5 + 1.0 + 0.0) / 3 = 0.5
```

### Interpreting MRR

| MRR Score | Interpretation |
|-----------|----------------|
| 1.0 | Perfect—relevant doc always first |
| 0.5 | Relevant doc usually in top 2 |
| 0.33 | Relevant doc usually in top 3 |
| 0.1 | Relevant doc around position 10 |
| 0.0 | Never finding relevant documents |

---

## Normalized Discounted Cumulative Gain (NDCG)

NDCG measures ranking quality when documents have graded relevance (not just binary).

### Formula Components

```python
import math
from typing import List

def dcg(relevances: List[float], k: int = None) -> float:
    """
    Discounted Cumulative Gain.
    
    Relevance scores are discounted by position (log2).
    """
    if k is not None:
        relevances = relevances[:k]
    
    dcg_score = 0.0
    for i, rel in enumerate(relevances, start=1):
        # Discount by log2(position + 1)
        dcg_score += rel / math.log2(i + 1)
    
    return dcg_score

def ndcg(
    relevances: List[float],
    k: int = None
) -> float:
    """
    Normalized DCG.
    
    DCG divided by ideal DCG (perfect ranking).
    """
    # Calculate actual DCG
    actual_dcg = dcg(relevances, k)
    
    # Calculate ideal DCG (sorted descending)
    ideal_order = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_order, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg

# Example with graded relevance
# 3 = highly relevant, 2 = relevant, 1 = somewhat relevant, 0 = not relevant

# Good ranking
good_ranking = [3, 2, 3, 0, 1]  # Best docs at top
print(f"NDCG (good): {ndcg(good_ranking):.3f}")  # ~0.95

# Poor ranking
poor_ranking = [0, 1, 0, 3, 2]  # Best docs at bottom
print(f"NDCG (poor): {ndcg(poor_ranking):.3f}")  # ~0.63
```

### NDCG@K for RAG

```python
def calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int
) -> float:
    """
    Calculate NDCG@K for retrieval results.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score (0-3)
        k: Cutoff position
    """
    # Get relevance for retrieved docs
    relevances = [
        relevance_scores.get(doc_id, 0)
        for doc_id in retrieved_ids[:k]
    ]
    
    # Get all relevance scores for ideal ranking
    all_relevances = list(relevance_scores.values())
    ideal_relevances = sorted(all_relevances, reverse=True)[:k]
    
    actual = dcg(relevances)
    ideal = dcg(ideal_relevances)
    
    return actual / ideal if ideal > 0 else 0.0

# Example
retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
relevance = {
    "doc_a": 1,  # somewhat relevant
    "doc_b": 0,  # not relevant
    "doc_c": 3,  # highly relevant
    "doc_d": 2,  # relevant
    "doc_e": 0,  # not relevant
    "doc_f": 3,  # highly relevant (not retrieved!)
}

ndcg_5 = calculate_ndcg_at_k(retrieved, relevance, k=5)
print(f"NDCG@5: {ndcg_5:.3f}")
```

---

## Hit Rate and Success@K

Simple but useful metrics for RAG.

```python
def hit_rate(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """
    Binary: Did we retrieve at least one relevant document?
    """
    return 1.0 if set(retrieved_ids) & relevant_ids else 0.0

def success_at_k(
    queries: list[dict],
    k: int
) -> float:
    """
    Fraction of queries with at least one relevant doc in top-k.
    """
    successes = 0
    
    for query in queries:
        top_k = set(query['retrieved'][:k])
        relevant = set(query['relevant'])
        
        if top_k & relevant:
            successes += 1
    
    return successes / len(queries) if queries else 0.0

# Example
queries = [
    {'retrieved': ['a', 'b', 'c'], 'relevant': {'a', 'd'}},  # Hit at 1
    {'retrieved': ['x', 'y', 'z'], 'relevant': {'a', 'b'}},  # Miss
    {'retrieved': ['p', 'q', 'a'], 'relevant': {'a'}},       # Hit at 3
]

print(f"Success@1: {success_at_k(queries, 1):.2f}")  # 0.33
print(f"Success@3: {success_at_k(queries, 3):.2f}")  # 0.67
```

---

## Complete Retrieval Evaluator

```python
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import math

@dataclass
class RetrievalEvaluation:
    """Complete retrieval evaluation results."""
    query_id: str
    
    # Basic metrics
    precision: float
    recall: float
    f1: float
    
    # Ranking metrics
    reciprocal_rank: float
    ndcg: float
    
    # Position metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    
    # Counts
    retrieved_count: int
    relevant_count: int
    relevant_retrieved: int
    first_relevant_position: Optional[int]

class RetrievalEvaluator:
    """Evaluate retrieval quality with multiple metrics."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate_single(
        self,
        query_id: str,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> RetrievalEvaluation:
        """Evaluate retrieval for a single query."""
        
        retrieved_set = set(retrieved_ids)
        relevant_retrieved = retrieved_set & relevant_ids
        
        # Basic metrics
        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Reciprocal rank
        rr = 0.0
        first_pos = None
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                rr = 1.0 / i
                first_pos = i
                break
        
        # NDCG
        if relevance_scores:
            relevances = [relevance_scores.get(d, 0) for d in retrieved_ids]
        else:
            # Binary relevance
            relevances = [1.0 if d in relevant_ids else 0.0 for d in retrieved_ids]
        
        ndcg_score = self._calculate_ndcg(relevances)
        
        # Precision and Recall at K
        p_at_k = {}
        r_at_k = {}
        for k in self.k_values:
            p_at_k[k] = self._precision_at_k(retrieved_ids, relevant_ids, k)
            r_at_k[k] = self._recall_at_k(retrieved_ids, relevant_ids, k)
        
        return RetrievalEvaluation(
            query_id=query_id,
            precision=precision,
            recall=recall,
            f1=f1,
            reciprocal_rank=rr,
            ndcg=ndcg_score,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            retrieved_count=len(retrieved_set),
            relevant_count=len(relevant_ids),
            relevant_retrieved=len(relevant_retrieved),
            first_relevant_position=first_pos
        )
    
    def evaluate_batch(
        self,
        queries: List[Dict]
    ) -> Dict:
        """
        Evaluate multiple queries and aggregate.
        
        Each query dict should have:
        - 'id': query identifier
        - 'retrieved': list of retrieved doc IDs
        - 'relevant': set/list of relevant doc IDs
        - 'relevance_scores': optional dict of doc_id -> score
        """
        results = []
        
        for query in queries:
            result = self.evaluate_single(
                query_id=query['id'],
                retrieved_ids=query['retrieved'],
                relevant_ids=set(query['relevant']),
                relevance_scores=query.get('relevance_scores')
            )
            results.append(result)
        
        return self._aggregate(results)
    
    def _aggregate(self, results: List[RetrievalEvaluation]) -> Dict:
        """Aggregate evaluation results."""
        n = len(results)
        if n == 0:
            return {}
        
        agg = {
            'count': n,
            'mean_precision': sum(r.precision for r in results) / n,
            'mean_recall': sum(r.recall for r in results) / n,
            'mean_f1': sum(r.f1 for r in results) / n,
            'mrr': sum(r.reciprocal_rank for r in results) / n,
            'mean_ndcg': sum(r.ndcg for r in results) / n,
        }
        
        # P@K and R@K
        for k in self.k_values:
            agg[f'mean_p@{k}'] = sum(r.precision_at_k.get(k, 0) for r in results) / n
            agg[f'mean_r@{k}'] = sum(r.recall_at_k.get(k, 0) for r in results) / n
        
        # Success rate
        agg['success@1'] = sum(1 for r in results if r.first_relevant_position == 1) / n
        agg['hit_rate'] = sum(1 for r in results if r.first_relevant_position is not None) / n
        
        return agg
    
    def _precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        top_k = retrieved[:k]
        hits = sum(1 for d in top_k if d in relevant)
        return hits / k if k > 0 else 0.0
    
    def _recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        top_k = set(retrieved[:k])
        hits = len(top_k & relevant)
        return hits / len(relevant) if relevant else 0.0
    
    def _calculate_ndcg(self, relevances: List[float]) -> float:
        if not relevances:
            return 0.0
        
        # DCG
        dcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(relevances)
        )
        
        # Ideal DCG
        ideal = sorted(relevances, reverse=True)
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )
        
        return dcg / idcg if idcg > 0 else 0.0

# Usage
evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

queries = [
    {
        'id': 'q1',
        'retrieved': ['d1', 'd2', 'd3', 'd4', 'd5'],
        'relevant': {'d1', 'd3', 'd6'},
    },
    {
        'id': 'q2',
        'retrieved': ['d7', 'd8', 'd9', 'd10', 'd11'],
        'relevant': {'d7', 'd8'},
    },
]

results = evaluator.evaluate_batch(queries)
print(f"MRR: {results['mrr']:.3f}")
print(f"Mean P@3: {results['mean_p@3']:.3f}")
print(f"Hit Rate: {results['hit_rate']:.3f}")
```

---

## Hands-on Exercise

### Your Task

Build a `RetrievalBenchmark` class that:
1. Loads benchmark queries with ground truth
2. Runs retrieval and evaluates quality
3. Compares different retrieval configurations
4. Generates comparison report

### Requirements

```python
class RetrievalBenchmark:
    def add_query(
        self,
        query_id: str,
        query_text: str,
        relevant_docs: list[str]
    ) -> None:
        pass
    
    def run_evaluation(
        self,
        retriever_name: str,
        retriever_fn  # callable that takes query_text, returns list[str]
    ) -> dict:
        pass
    
    def compare_retrievers(self) -> str:
        """Generate markdown comparison table."""
        pass
```

<details>
<summary>💡 Hints</summary>

- Store queries in a dictionary by ID
- Keep results per retriever for comparison
- Calculate key metrics: MRR, P@5, R@5, hit rate
- Format comparison as markdown table

</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass
from typing import Callable, Dict, List, Set
import math

@dataclass
class BenchmarkQuery:
    id: str
    text: str
    relevant_docs: Set[str]

@dataclass
class RetrieverResult:
    name: str
    mrr: float
    precision_at_5: float
    recall_at_5: float
    hit_rate: float
    ndcg: float

class RetrievalBenchmark:
    def __init__(self):
        self.queries: Dict[str, BenchmarkQuery] = {}
        self.results: Dict[str, RetrieverResult] = {}
    
    def add_query(
        self,
        query_id: str,
        query_text: str,
        relevant_docs: list[str]
    ) -> None:
        self.queries[query_id] = BenchmarkQuery(
            id=query_id,
            text=query_text,
            relevant_docs=set(relevant_docs)
        )
    
    def run_evaluation(
        self,
        retriever_name: str,
        retriever_fn: Callable[[str], List[str]],
        k: int = 5
    ) -> dict:
        """Run retriever on all queries and evaluate."""
        
        total_rr = 0.0
        total_p_k = 0.0
        total_r_k = 0.0
        total_hits = 0
        total_ndcg = 0.0
        
        for query in self.queries.values():
            # Run retriever
            retrieved = retriever_fn(query.text)
            relevant = query.relevant_docs
            
            # Reciprocal rank
            rr = 0.0
            for i, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rr = 1.0 / i
                    break
            total_rr += rr
            
            # Precision@K
            top_k = retrieved[:k]
            hits_at_k = sum(1 for d in top_k if d in relevant)
            total_p_k += hits_at_k / k
            
            # Recall@K
            total_r_k += hits_at_k / len(relevant) if relevant else 0
            
            # Hit rate
            if set(retrieved) & relevant:
                total_hits += 1
            
            # NDCG (binary relevance)
            relevances = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
            ndcg = self._calculate_ndcg(relevances)
            total_ndcg += ndcg
        
        n = len(self.queries)
        
        result = RetrieverResult(
            name=retriever_name,
            mrr=total_rr / n,
            precision_at_5=total_p_k / n,
            recall_at_5=total_r_k / n,
            hit_rate=total_hits / n,
            ndcg=total_ndcg / n
        )
        
        self.results[retriever_name] = result
        
        return {
            "retriever": retriever_name,
            "mrr": result.mrr,
            "p@5": result.precision_at_5,
            "r@5": result.recall_at_5,
            "hit_rate": result.hit_rate,
            "ndcg": result.ndcg
        }
    
    def _calculate_ndcg(self, relevances: List[float]) -> float:
        if not relevances or sum(relevances) == 0:
            return 0.0
        
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
        ideal = sorted(relevances, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compare_retrievers(self) -> str:
        """Generate markdown comparison table."""
        if not self.results:
            return "No results to compare."
        
        lines = [
            "# Retriever Comparison",
            "",
            "| Retriever | MRR | P@5 | R@5 | Hit Rate | NDCG |",
            "|-----------|-----|-----|-----|----------|------|"
        ]
        
        # Sort by MRR descending
        sorted_results = sorted(
            self.results.values(),
            key=lambda r: r.mrr,
            reverse=True
        )
        
        for r in sorted_results:
            lines.append(
                f"| {r.name} | {r.mrr:.3f} | {r.precision_at_5:.3f} | "
                f"{r.recall_at_5:.3f} | {r.hit_rate:.3f} | {r.ndcg:.3f} |"
            )
        
        # Add winner
        winner = sorted_results[0].name
        lines.extend([
            "",
            f"**Winner:** {winner} (highest MRR)"
        ])
        
        return "\n".join(lines)

# Usage
benchmark = RetrievalBenchmark()

# Add test queries
benchmark.add_query("q1", "What is machine learning?", ["doc_ml_1", "doc_ml_2"])
benchmark.add_query("q2", "How does Python work?", ["doc_py_1", "doc_py_3"])
benchmark.add_query("q3", "What is RAG?", ["doc_rag_1"])

# Mock retrievers
def retriever_a(query: str) -> List[str]:
    return ["doc_ml_1", "doc_other", "doc_ml_2", "doc_py_1", "doc_rag_1"]

def retriever_b(query: str) -> List[str]:
    return ["doc_other", "doc_ml_1", "doc_rag_1", "doc_py_1", "doc_ml_2"]

# Run evaluations
benchmark.run_evaluation("Retriever A", retriever_a)
benchmark.run_evaluation("Retriever B", retriever_b)

# Compare
print(benchmark.compare_retrievers())
```

</details>

---

## Summary

Retrieval quality forms the foundation of RAG performance:

✅ **Precision/Recall** — Basic relevance coverage
✅ **MRR** — How quickly we find first relevant result
✅ **NDCG** — Ranking quality with graded relevance
✅ **Success@K** — Did we find anything useful in top-k?

**Key insight:** High recall with low precision = noisy context. High precision with low recall = missing information.

**Next:** [Answer Quality Evaluation](./03-answer-quality.md)

---

## Further Reading

- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [NDCG Explained](https://towardsdatascience.com/ndcg-is-all-you-need)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)

<!--
Sources Consulted:
- Information retrieval textbooks
- RAGAS retrieval metrics
- BEIR benchmark documentation
-->
