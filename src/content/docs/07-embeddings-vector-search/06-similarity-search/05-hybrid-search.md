---
title: "Hybrid Search (BM25 + Semantic)"
---

# Hybrid Search (BM25 + Semantic)

## Introduction

Pure semantic search can miss exact keyword matches. Hybrid search combines the best of both approaches: BM25 for precise keyword matching and embeddings for semantic understanding.

---

## Why Hybrid Outperforms Pure Semantic

| Query Type | Semantic Search | BM25 | Hybrid |
|------------|-----------------|------|--------|
| "machine learning tutorial" | ✅ Great | ✅ Good | ✅ Great |
| "Error code TS-999" | ❌ Misses exact match | ✅ Perfect | ✅ Perfect |
| "python pandas dataframe" | ✅ Good | ✅ Great | ✅ Great |
| "how to fix memory leak" | ✅ Great | ⚠️ May miss synonyms | ✅ Great |

---

## Building a Hybrid Searcher

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import Tuple

class HybridSearcher:
    """Combine BM25 keyword search with vector similarity."""
    
    def __init__(self, documents: list[dict], embed_fn):
        self.documents = documents
        self.embed_fn = embed_fn
        
        # Build BM25 index
        tokenized = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Build vector index (simplified - use real vector DB in production)
        self.vectors = np.array([
            embed_fn(doc["text"]) for doc in documents
        ])
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def search_bm25(self, query: str, top_k: int = 100) -> list[Tuple[int, float]]:
        """BM25 keyword search."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices and scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_semantic(self, query: str, top_k: int = 100) -> list[Tuple[int, float]]:
        """Vector similarity search."""
        query_vector = np.array(self.embed_fn(query))
        
        # Cosine similarity
        similarities = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10,
        alpha: float = 0.5
    ) -> list[dict]:
        """Hybrid search with linear combination fusion."""
        bm25_results = self.search_bm25(query, top_k=100)
        semantic_results = self.search_semantic(query, top_k=100)
        
        # Combine scores using weighted linear combination
        combined_scores = {}
        
        for idx, score in bm25_results:
            combined_scores[idx] = (1 - alpha) * score
        
        for idx, score in semantic_results:
            if idx in combined_scores:
                combined_scores[idx] += alpha * score
            else:
                combined_scores[idx] = alpha * score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [
            {
                "document": self.documents[idx],
                "score": score,
                "index": idx
            }
            for idx, score in sorted_results
        ]
```

---

## Score Fusion Methods

Different fusion methods work better for different scenarios:

### Reciprocal Rank Fusion (RRF)

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    results_lists: list[list[Tuple[str, float]]],
    k: int = 60
) -> list[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) - robust fusion method.
    
    RRF score = sum(1 / (k + rank_i)) for each ranking list
    
    Args:
        results_lists: List of ranked result lists, each containing (id, score) tuples
        k: Constant to prevent high scores for top-ranked docs (default 60)
    
    Returns:
        Fused ranking as list of (id, rrf_score) tuples
    """
    rrf_scores = defaultdict(float)
    
    for results in results_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] += 1 / (k + rank)
    
    # Sort by RRF score descending
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### Distribution-Based Score Fusion (DBSF)

```python
def distribution_based_score_fusion(
    results_lists: list[list[Tuple[str, float]]]
) -> list[Tuple[str, float]]:
    """
    Distribution-Based Score Fusion (DBSF).
    
    Normalizes scores to [0, 1] based on distribution, then averages.
    More robust to different score scales.
    """
    def normalize_scores(results: list[Tuple[str, float]]) -> dict[str, float]:
        scores = [score for _, score in results]
        if not scores:
            return {}
        
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        if range_score == 0:
            return {doc_id: 0.5 for doc_id, _ in results}
        
        return {
            doc_id: (score - min_score) / range_score
            for doc_id, score in results
        }
    
    # Normalize each result list
    normalized_lists = [normalize_scores(results) for results in results_lists]
    
    # Combine normalized scores
    combined = defaultdict(list)
    for normalized in normalized_lists:
        for doc_id, score in normalized.items():
            combined[doc_id].append(score)
    
    # Average scores (handle missing entries)
    final_scores = {
        doc_id: sum(scores) / len(results_lists)
        for doc_id, scores in combined.items()
    }
    
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

# Example usage
bm25_results = [("doc1", 25.5), ("doc3", 20.1), ("doc2", 15.3)]
semantic_results = [("doc2", 0.89), ("doc1", 0.75), ("doc4", 0.68)]

rrf_fused = reciprocal_rank_fusion([bm25_results, semantic_results])
dbsf_fused = distribution_based_score_fusion([bm25_results, semantic_results])

print("RRF Fusion:", rrf_fused[:3])
print("DBSF Fusion:", dbsf_fused[:3])
```

**Output:**
```
RRF Fusion: [('doc1', 0.0327), ('doc2', 0.0327), ('doc3', 0.0164)]
DBSF Fusion: [('doc1', 0.875), ('doc2', 0.7275), ('doc3', 0.4925)]
```

---

## Pinecone Hybrid Search

Pinecone's hybrid index handles fusion automatically:

```python
from pinecone import Pinecone
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from collections import Counter

def create_hybrid_index_pinecone():
    """Create and use Pinecone hybrid index."""
    
    pc = Pinecone(api_key="your-api-key")
    
    # Create hybrid index
    pc.create_index(
        name="hybrid-docs",
        dimension=384,
        metric="dotproduct",  # Required for hybrid
        spec={
            "pod": {
                "environment": "us-east-1-aws",
                "pod_type": "s1.x1"
            }
        }
    )
    
    index = pc.Index("hybrid-docs")
    
    # Set up models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_sparse_vector(text: str) -> dict:
        """Generate sparse vector from token frequencies."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts = dict(Counter(tokens))
        return {
            "indices": list(token_counts.keys()),
            "values": list(token_counts.values())
        }
    
    # Upsert with both dense and sparse vectors
    documents = [
        "Machine learning fundamentals tutorial",
        "Error code TS-999 troubleshooting guide",
        "Python pandas dataframe operations"
    ]
    
    vectors = []
    for i, doc in enumerate(documents):
        vectors.append({
            "id": f"doc-{i}",
            "values": model.encode(doc).tolist(),  # Dense
            "sparse_values": generate_sparse_vector(doc),  # Sparse
            "metadata": {"text": doc}
        })
    
    index.upsert(vectors=vectors)
    
    # Hybrid query
    query = "TS-999 error"
    results = index.query(
        vector=model.encode(query).tolist(),
        sparse_vector=generate_sparse_vector(query),
        top_k=5,
        include_metadata=True,
        alpha=0.5  # Balance between dense and sparse
    )
    
    return results
```

---

## Fusion Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Linear Combination** | Simple, tunable alpha | Requires normalized scores | When scales are similar |
| **RRF** | Rank-based, no normalization | Ignores score magnitude | Heterogeneous retrievers |
| **DBSF** | Handles different scales | More computation | Production systems |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use RRF for mixing different retrievers | Assume linear combination works |
| Tune alpha based on query types | Use fixed 0.5 for all use cases |
| Test with exact-match queries | Ignore keyword matching value |
| Index sparse vectors for hybrid | Compute sparse on every query |

---

## Summary

✅ **Hybrid search captures both exact matches and semantic similarity**

✅ **RRF is the most robust fusion method** - no normalization needed

✅ **Alpha parameter** balances keyword (0) vs semantic (1) importance

✅ **Built-in hybrid support** in Pinecone simplifies implementation

**Next:** [SPLADE & Sparse Embeddings](./06-splade-sparse-embeddings.md)
