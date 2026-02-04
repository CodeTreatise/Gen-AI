---
title: "Reranking"
---

# Reranking

## Introduction

**Reranking** is the final stage in the Contextual Retrieval pipeline. After retrieving a larger set of candidates (e.g., top-150) with hybrid search, a reranker re-scores and refines the ranking to produce the final top-K results.

This lesson covers how rerankers work and how to implement them.

### What We'll Cover

- What rerankers are and how they differ from embeddings
- Why reranking improves retrieval
- Popular reranking APIs (Cohere, Voyage)
- Implementation examples
- When to use reranking

### Prerequisites

- [Prompt Caching](./07-prompt-caching.md)
- Understanding of the hybrid search pipeline

---

## What is Reranking?

### Embeddings vs Rerankers

```
┌─────────────────────────────────────────────────────────────────┐
│              Embeddings vs Rerankers: Key Difference             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EMBEDDINGS (Bi-Encoder):                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Query ──► [Encoder] ──► Query Embedding ─┐              │  │
│  │                                            │              │  │
│  │  Doc ────► [Encoder] ──► Doc Embedding ───┼► Similarity  │  │
│  │                                            │              │  │
│  │  • Query and doc encoded INDEPENDENTLY                   │  │
│  │  • Fast: Can pre-compute doc embeddings                  │  │
│  │  • Less accurate: No cross-attention                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  RERANKERS (Cross-Encoder):                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  [Query + Doc] ──► [Cross-Encoder] ──► Relevance Score   │  │
│  │                                                          │  │
│  │  • Query and doc processed TOGETHER                      │  │
│  │  • Slow: Must process each query-doc pair                │  │
│  │  • More accurate: Full attention between query & doc     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Cross-Encoders Are More Accurate

```python
# Bi-encoder (embeddings) - separate processing
query_emb = encode("What is ACME's Q2 revenue?")
doc_emb = encode("The company reported $324M in the second quarter")
score = cosine_similarity(query_emb, doc_emb)
# Can't see that "ACME" → "company", "Q2" → "second quarter"

# Cross-encoder (reranker) - joint processing
score = rerank("What is ACME's Q2 revenue?", 
               "The company reported $324M in the second quarter")
# Can attend between "Q2" and "second quarter"
# Can infer "ACME" refers to "the company"
```

---

## Two-Stage Retrieval Architecture

### Why Use Both?

| Stage | Method | Speed | Accuracy | Purpose |
|-------|--------|-------|----------|---------|
| 1 | Hybrid Search | Fast | Good | Cast wide net, get candidates |
| 2 | Reranking | Slow | Best | Refine ranking, select final results |

```
┌─────────────────────────────────────────────────────────────────┐
│              Two-Stage Retrieval Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: "ACME Q2 2023 revenue growth"                           │
│                                                                 │
│  STAGE 1: Hybrid Search (Fast)                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Vector Search ─────┐                                    │  │
│  │                      ├──► Top 150 candidates             │  │
│  │  BM25 Search ───────┘                                    │  │
│  │                                                          │  │
│  │  Time: ~50ms for 100K documents                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  STAGE 2: Reranking (Accurate)                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  150 candidates ──► [Reranker] ──► Top 20 final          │  │
│  │                                                          │  │
│  │  Time: ~200ms for 150 pairs                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  Final: 20 highly relevant chunks for the LLM                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Popular Reranking APIs

### Cohere Rerank

| Model | Best For | Max Docs | Context |
|-------|----------|----------|---------|
| `rerank-v3.5` | Production workloads | 1,000 | 4K tokens |
| `rerank-english-v3.0` | English content | 1,000 | 4K tokens |
| `rerank-multilingual-v3.0` | Non-English | 1,000 | 4K tokens |

### Voyage AI Rerank

| Model | Best For | Max Docs | Context |
|-------|----------|----------|---------|
| `rerank-2` | General purpose | 1,000 | 32K tokens |
| `rerank-2-lite` | Lower latency | 1,000 | 8K tokens |

---

## Cohere Implementation

### Basic Usage

```python
import cohere

def rerank_with_cohere(
    query: str,
    documents: list[str],
    top_n: int = 20,
    model: str = "rerank-v3.5"
) -> list[dict]:
    """
    Rerank documents using Cohere.
    
    Args:
        query: The search query
        documents: List of document texts
        top_n: Number of results to return
        model: Cohere rerank model
    
    Returns:
        List of {index, relevance_score, text}
    """
    co = cohere.ClientV2()
    
    response = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )
    
    results = []
    for item in response.results:
        results.append({
            "index": item.index,
            "relevance_score": item.relevance_score,
            "text": documents[item.index]
        })
    
    return results


# Example usage
query = "ACME Q2 2023 revenue growth"
candidates = [
    "Revenue grew 3% in Q2 2023...",
    "The company expanded to new markets...",
    "Operating expenses decreased by 5%...",
    "ACME Corporation announced record profits...",
    # ... up to 150 candidates
]

reranked = rerank_with_cohere(query, candidates, top_n=5)

for i, r in enumerate(reranked):
    print(f"{i+1}. Score: {r['relevance_score']:.4f}")
    print(f"   {r['text'][:80]}...\n")
```

**Output:**
```
1. Score: 0.9823
   Revenue grew 3% in Q2 2023...

2. Score: 0.8456
   ACME Corporation announced record profits...

3. Score: 0.3421
   Operating expenses decreased by 5%...

4. Score: 0.2134
   The company expanded to new markets...

5. Score: 0.0892
   ...
```

---

## Voyage AI Implementation

### Basic Usage

```python
import voyageai

def rerank_with_voyage(
    query: str,
    documents: list[str],
    top_k: int = 20,
    model: str = "rerank-2"
) -> list[dict]:
    """
    Rerank documents using Voyage AI.
    
    Args:
        query: The search query
        documents: List of document texts
        top_k: Number of results to return
        model: Voyage rerank model
    
    Returns:
        List of {index, relevance_score, document}
    """
    vo = voyageai.Client()
    
    reranking = vo.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=top_k
    )
    
    results = []
    for result in reranking.results:
        results.append({
            "index": result.index,
            "relevance_score": result.relevance_score,
            "document": result.document
        })
    
    return results


# Example usage
query = "What were the risk factors mentioned in the SEC filing?"
candidates = [
    "Market volatility may impact our operations...",
    "Revenue sources are diversified across...",
    "Risk factors include regulatory changes...",
    # ... candidates from hybrid search
]

reranked = rerank_with_voyage(query, candidates, top_k=5)

for i, r in enumerate(reranked):
    print(f"{i+1}. Score: {r['relevance_score']:.4f}")
    print(f"   {r['document'][:80]}...\n")
```

---

## Complete Pipeline with Reranking

### Integrated Example

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class RetrievalResult:
    """Final retrieval result after reranking."""
    text: str
    context: str
    contextualized_text: str
    hybrid_score: float
    rerank_score: float
    final_rank: int


class ContextualRetrievalPipeline:
    """Complete Contextual Retrieval with reranking."""
    
    def __init__(
        self,
        initial_k: int = 150,
        final_k: int = 20,
        hybrid_alpha: float = 0.6,
        reranker: str = "cohere"  # or "voyage"
    ):
        self.initial_k = initial_k
        self.final_k = final_k
        self.hybrid_alpha = hybrid_alpha
        self.reranker = reranker
        
        # Initialize clients
        if reranker == "cohere":
            import cohere
            self.rerank_client = cohere.ClientV2()
        elif reranker == "voyage":
            import voyageai
            self.rerank_client = voyageai.Client()
        
        # Storage (populated by indexing)
        self.chunks: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Two-stage retrieval: hybrid search → reranking.
        """
        # Stage 1: Hybrid search for initial candidates
        candidates = self._hybrid_search(query, k=self.initial_k)
        
        if not candidates:
            return []
        
        # Stage 2: Rerank candidates
        reranked = self._rerank(
            query=query,
            candidates=candidates,
            top_k=self.final_k
        )
        
        return reranked
    
    def _hybrid_search(self, query: str, k: int) -> List[dict]:
        """Perform hybrid vector + BM25 search."""
        # Vector scores
        query_emb = self._embed([query])[0]
        vector_scores = np.dot(self.embeddings, query_emb)
        
        # BM25 scores
        query_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        
        # Normalize
        vector_scores = self._normalize(vector_scores)
        bm25_scores = self._normalize(bm25_scores)
        
        # Combine
        combined = (self.hybrid_alpha * vector_scores + 
                   (1 - self.hybrid_alpha) * bm25_scores)
        
        # Get top-k
        top_indices = np.argsort(combined)[::-1][:k]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                "index": idx,
                "chunk": self.chunks[idx],
                "hybrid_score": float(combined[idx])
            })
        
        return candidates
    
    def _rerank(
        self, 
        query: str, 
        candidates: List[dict],
        top_k: int
    ) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder."""
        
        # Extract texts for reranking
        texts = [c["chunk"]["contextualized_text"] for c in candidates]
        
        if self.reranker == "cohere":
            response = self.rerank_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=texts,
                top_n=top_k,
                return_documents=False
            )
            rerank_results = [
                {"index": r.index, "score": r.relevance_score}
                for r in response.results
            ]
        
        elif self.reranker == "voyage":
            response = self.rerank_client.rerank(
                query=query,
                documents=texts,
                model="rerank-2",
                top_k=top_k
            )
            rerank_results = [
                {"index": r.index, "score": r.relevance_score}
                for r in response.results
            ]
        
        # Build final results
        results = []
        for rank, rr in enumerate(rerank_results):
            orig_idx = rr["index"]
            candidate = candidates[orig_idx]
            chunk = candidate["chunk"]
            
            results.append(RetrievalResult(
                text=chunk["text"],
                context=chunk["context"],
                contextualized_text=chunk["contextualized_text"],
                hybrid_score=candidate["hybrid_score"],
                rerank_score=rr["score"],
                final_rank=rank + 1
            ))
        
        return results
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Create embeddings (implement with your provider)."""
        # ... embedding implementation
        pass
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize for BM25."""
        return text.lower().split()
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize."""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


# Usage
pipeline = ContextualRetrievalPipeline(
    initial_k=150,
    final_k=20,
    reranker="cohere"
)

# ... index documents ...

results = pipeline.retrieve("ACME Q2 2023 revenue growth")

print("Final Results:")
for r in results[:5]:
    print(f"\n{r.final_rank}. Rerank: {r.rerank_score:.4f}, "
          f"Hybrid: {r.hybrid_score:.4f}")
    print(f"   Context: {r.context[:80]}...")
    print(f"   Text: {r.text[:80]}...")
```

---

## Performance Impact

### From Anthropic's Research

| Configuration | Retrieval Failure Rate |
|--------------|----------------------|
| Contextual Embeddings + BM25 | 2.9% |
| **+ Reranking (top-150 → top-20)** | **1.9%** |
| Improvement from reranking | **-34% relative** |

### Latency Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│              Latency Budget for Two-Stage Retrieval              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Hybrid Search                                         │
│  ├── Vector similarity: ~20ms (100K docs)                       │
│  ├── BM25 search: ~30ms                                         │
│  └── Score combination: ~5ms                                    │
│  Subtotal: ~55ms                                                │
│                                                                 │
│  Stage 2: Reranking                                             │
│  ├── 150 docs × ~1.3ms each ≈ 200ms                            │
│  └── (API call overhead included)                               │
│  Subtotal: ~200ms                                               │
│                                                                 │
│  Total: ~255ms                                                  │
│                                                                 │
│  Compare to: Single-stage vector search ~25ms                   │
│  Tradeoff: 10x slower but 67% fewer failures                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Use Reranking

### Use Reranking When

✅ **Accuracy is critical** - customer-facing, legal, medical
✅ **Latency budget allows** - 200-300ms acceptable
✅ **Complex queries** - multi-part, nuanced questions
✅ **High-stakes retrieval** - wrong answer has consequences

### Skip Reranking When

❌ **Ultra-low latency required** - <50ms response needed
❌ **Simple lookups** - exact match queries
❌ **Cost-sensitive** - reranking adds API costs
❌ **Good enough with hybrid** - 2.9% failure rate acceptable

### Cost Comparison

| Provider | Model | Cost per 1K docs |
|----------|-------|-----------------|
| Cohere | rerank-v3.5 | ~$0.002 |
| Voyage | rerank-2 | ~$0.002 |

For 150 candidates per query:
- Cost per query: ~$0.0003
- Cost per 1M queries: ~$300

---

## Best Practices

### Optimal Configuration

```python
# Anthropic's recommended settings
OPTIMAL_CONFIG = {
    "initial_k": 150,      # Retrieve 150 candidates
    "final_k": 20,         # Return top 20 after reranking
    "hybrid_alpha": 0.6,   # 60% vector, 40% BM25
}

# Why 150 → 20?
# - 150 is enough to capture relevant chunks with hybrid search
# - Reranking 150 docs is fast enough (~200ms)
# - 20 final chunks is optimal for most LLM context windows
```

### Handling Edge Cases

```python
def safe_rerank(
    query: str,
    candidates: List[dict],
    top_k: int
) -> List[dict]:
    """Rerank with fallback handling."""
    
    if len(candidates) == 0:
        return []
    
    if len(candidates) <= top_k:
        # Not enough candidates to rerank
        # Just return all sorted by hybrid score
        return sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
    
    try:
        return rerank_with_api(query, candidates, top_k)
    except Exception as e:
        # Fallback to hybrid scores if reranker fails
        print(f"Reranker failed: {e}, using hybrid scores")
        return sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
```

---

## Summary

✅ **Rerankers are cross-encoders** that process query + doc together  
✅ **More accurate than embeddings** due to full attention mechanism  
✅ **Two-stage retrieval:** Hybrid search (150) → Rerank (20)  
✅ **Popular APIs:** Cohere rerank-v3.5, Voyage rerank-2  
✅ **Adds ~200ms latency** but reduces failures by 34%  
✅ Use when **accuracy matters more than latency**

---

**Next:** [Best Practices →](./09-best-practices.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Cohere Rerank Documentation: https://docs.cohere.com/v2/reference/rerank
- Voyage AI Reranker Documentation: https://docs.voyageai.com/docs/reranker
-->
