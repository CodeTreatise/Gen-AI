---
title: "Maximal Marginal Relevance (MMR)"
---

# Maximal Marginal Relevance (MMR)

## Introduction

Sometimes you want diverse results, not just the most similar. Maximal Marginal Relevance (MMR) balances relevance with diversity, ensuring results cover different aspects of a topic rather than repeating the same information.

---

## The Problem: Redundant Results

Without MMR, top results often say the same thing:

```
Query: "Python error handling"
Result 1: "Python uses try/except blocks for exception handling."
Result 2: "Exception handling in Python uses try and except."
Result 3: "The try/except syntax handles exceptions in Python."
```

All three results are highly relevant but redundant—they provide the same information with different wording.

---

## MMR Algorithm

MMR selects documents that are both relevant to the query AND different from already-selected documents:

$$MMR = \lambda \cdot sim(q, d) - (1 - \lambda) \cdot \max_{d' \in S} sim(d, d')$$

Where:
- $\lambda$ (lambda): diversity parameter (0 = max diversity, 1 = max relevance)
- $sim(q, d)$: similarity between query and document
- $S$: set of already-selected documents

---

## MMR Implementation

```python
import numpy as np
from typing import List

def mmr_search(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: List[str],
    k: int = 10,
    lambda_param: float = 0.5,
    candidates_limit: int = 100
) -> List[dict]:
    """
    Maximal Marginal Relevance search for diverse results.
    
    Args:
        query_embedding: Query vector
        document_embeddings: Matrix of document vectors (n_docs, dim)
        documents: List of document texts
        k: Number of results to return
        lambda_param: 0.0 = max diversity, 1.0 = max relevance
        candidates_limit: Number of candidates to consider
    """
    
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Calculate query-document similarities
    query_sims = np.array([
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in document_embeddings
    ])
    
    # Get top candidates by relevance
    candidate_indices = np.argsort(query_sims)[::-1][:candidates_limit]
    
    selected = []
    selected_embeddings = []
    
    while len(selected) < k and len(candidate_indices) > 0:
        mmr_scores = []
        
        for idx in candidate_indices:
            if idx in [s["index"] for s in selected]:
                continue
            
            # Relevance to query
            relevance = query_sims[idx]
            
            # Max similarity to already selected
            if selected_embeddings:
                max_sim_to_selected = max(
                    cosine_similarity(document_embeddings[idx], sel_emb)
                    for sel_emb in selected_embeddings
                )
            else:
                max_sim_to_selected = 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((idx, mmr))
        
        if not mmr_scores:
            break
        
        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        
        selected.append({
            "index": best_idx,
            "text": documents[best_idx],
            "relevance": float(query_sims[best_idx]),
            "mmr_score": float(best_score)
        })
        selected_embeddings.append(document_embeddings[best_idx])
        
        # Remove selected from candidates
        candidate_indices = [i for i in candidate_indices if i != best_idx]
    
    return selected
```

---

## Lambda Value Guide

| Lambda Value | Behavior | Use Case |
|--------------|----------|----------|
| 1.0 | Pure relevance (no diversity) | Q&A, factual lookup |
| 0.7-0.8 | Slightly diverse | Default for most use cases |
| 0.5 | Balanced | Product recommendations |
| 0.3-0.4 | More diverse | Research, exploration |
| 0.0 | Pure diversity (ignore relevance) | Never recommended |

```python
# Example with different lambda values
query = "Python error handling best practices"

# lambda=1.0: Pure relevance (may have redundant results)
results_relevant = mmr_search(query_emb, doc_embs, docs, lambda_param=1.0)

# lambda=0.5: Balanced (default)
results_balanced = mmr_search(query_emb, doc_embs, docs, lambda_param=0.5)

# lambda=0.3: Diverse (covers more topics)
results_diverse = mmr_search(query_emb, doc_embs, docs, lambda_param=0.3)
```

---

## Qdrant Built-in MMR

Qdrant provides MMR as a query option:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

def qdrant_mmr_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 10,
    diversity: float = 0.5
) -> list[dict]:
    """Use Qdrant's built-in MMR."""
    
    # Fetch more candidates for MMR selection
    prefetch_limit = limit * 5
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=prefetch_limit,
        search_params=SearchParams(
            hnsw_ef=128  # Higher for better MMR candidates
        )
    )
    
    # Apply MMR on results
    # (Qdrant's native MMR varies by version - this is application-level)
    if len(results) <= limit:
        return [{"id": h.id, "score": h.score, "payload": h.payload} for h in results]
    
    # Use our MMR implementation on fetched results
    import numpy as np
    doc_vectors = np.array([r.vector for r in results if r.vector])
    
    if len(doc_vectors) == 0:
        return [{"id": h.id, "score": h.score, "payload": h.payload} for h in results[:limit]]
    
    mmr_results = mmr_search(
        np.array(query_vector),
        doc_vectors,
        [r.payload.get("text", "") for r in results],
        k=limit,
        lambda_param=1 - diversity
    )
    
    return [
        {
            "id": results[r["index"]].id,
            "score": r["mmr_score"],
            "payload": results[r["index"]].payload
        }
        for r in mmr_results
    ]
```

---

## When to Use MMR

| Use Case | Lambda Value | Reasoning |
|----------|--------------|-----------|
| Q&A / factual lookup | 0.8-1.0 | Need most relevant answer |
| Product recommendations | 0.4-0.6 | Show variety of options |
| Research exploration | 0.3-0.5 | Cover different perspectives |
| News aggregation | 0.2-0.4 | Avoid duplicate stories |
| RAG context retrieval | 0.5-0.7 | Diverse but relevant chunks |

---

## MMR for RAG Context

When retrieving context for LLM generation, MMR ensures diverse information:

```python
def get_rag_context(
    query: str,
    vector_store,
    embed_fn,
    num_chunks: int = 5,
    diversity: float = 0.3
) -> str:
    """Get diverse context for RAG."""
    
    query_embedding = embed_fn(query)
    
    # Fetch candidates
    candidates = vector_store.search(query_embedding, limit=50)
    
    # Apply MMR
    doc_embeddings = np.array([c["embedding"] for c in candidates])
    mmr_results = mmr_search(
        np.array(query_embedding),
        doc_embeddings,
        [c["text"] for c in candidates],
        k=num_chunks,
        lambda_param=1 - diversity
    )
    
    # Combine into context
    context_chunks = [r["text"] for r in mmr_results]
    return "\n\n---\n\n".join(context_chunks)
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Tune lambda for your use case | Use 0.5 blindly |
| Fetch more candidates than k | Apply MMR to small sets |
| Consider MMR for RAG context | Return redundant chunks to LLM |
| Test diversity vs relevance tradeoff | Maximize diversity at all costs |

---

## Summary

✅ **MMR balances relevance with diversity** using the lambda parameter

✅ **Higher lambda (0.7-1.0)** for factual lookups, lower for exploration

✅ **Fetch 5x candidates** before applying MMR for best selection

✅ **Essential for RAG** to provide diverse context to LLMs

**Next:** [Similarity Search Exercise](./00-similarity-search.md#hands-on-exercise)

---

## Further Reading

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - 67% improvement research
- [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search/) - Dense + sparse vectors
- [Sentence Transformers Cross-Encoders](https://sbert.net/examples/applications/cross-encoder/README.html) - Bi-encoder vs cross-encoder
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank-2) - Production reranking API
- [Voyage AI Rerankers](https://docs.voyageai.com/docs/reranker) - Instruction-following rerankers

<!--
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Pinecone Hybrid Search: https://www.pinecone.io/learn/hybrid-search/
- Voyage AI Rerankers: https://docs.voyageai.com/docs/reranker
- Sentence Transformers Cross-Encoders: https://sbert.net/examples/applications/cross-encoder/README.html
- Cohere Rerank: https://docs.cohere.com/docs/rerank-2
-->
