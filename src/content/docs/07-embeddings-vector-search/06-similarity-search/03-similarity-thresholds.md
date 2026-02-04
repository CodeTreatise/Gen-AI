---
title: "Similarity Thresholds"
---

# Similarity Thresholds

## Introduction

Not all results are worth returning. Similarity thresholds filter out low-quality matches, ensuring users only see relevant content.

---

## Setting Minimum Thresholds

Threshold values depend on your embedding model and use case:

| Embedding Model | Strong Match | Acceptable | Weak |
|-----------------|--------------|------------|------|
| OpenAI ada-002 | > 0.85 | 0.75-0.85 | < 0.75 |
| text-embedding-3-small | > 0.50 | 0.35-0.50 | < 0.35 |
| Voyage-3 | > 0.80 | 0.65-0.80 | < 0.65 |
| BGE models | > 0.75 | 0.60-0.75 | < 0.60 |

> **Note:** text-embedding-3 models use a different similarity scale than earlier models. Always calibrate thresholds empirically with your data.

```python
from dataclasses import dataclass
from typing import Optional
from qdrant_client import QdrantClient

@dataclass
class SearchConfig:
    """Configuration for similarity search."""
    min_score: float = 0.35  # Minimum similarity threshold
    top_k: int = 10
    fallback_to_keyword: bool = True

def search_with_threshold(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    config: SearchConfig
) -> list[dict]:
    """Search with minimum similarity threshold."""
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=config.top_k,
        score_threshold=config.min_score  # Built-in filtering
    )
    
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text", "")
        }
        for hit in results
    ]
```

---

## Dynamic Thresholds

Adjust thresholds based on query characteristics:

```python
def calculate_dynamic_threshold(
    query: str,
    base_threshold: float = 0.35
) -> float:
    """Adjust threshold based on query properties."""
    
    threshold = base_threshold
    
    # Longer queries tend to have lower similarity scores
    word_count = len(query.split())
    if word_count > 10:
        threshold -= 0.05
    elif word_count < 3:
        threshold += 0.05
    
    # Questions often need stricter thresholds
    if query.strip().endswith('?'):
        threshold += 0.02
    
    # Clamp to reasonable range
    return max(0.20, min(0.60, threshold))

# Example
queries = [
    "Python",  # Very short
    "How do I implement retry logic in Python?",  # Medium
    "What are the best practices for handling database connection pooling in a microservices architecture with Python?"  # Long
]

for q in queries:
    threshold = calculate_dynamic_threshold(q)
    print(f"Query: '{q[:50]}...' -> Threshold: {threshold:.2f}")
```

**Output:**
```
Query: 'Python...' -> Threshold: 0.40
Query: 'How do I implement retry logic in Python?...' -> Threshold: 0.37
Query: 'What are the best practices for handling database...' -> Threshold: 0.30
```

---

## Empty Result Handling

Always have a fallback strategy when semantic search returns no results:

```python
async def search_with_fallback(
    query: str,
    vector_client: QdrantClient,
    collection_name: str,
    embed_fn,
    config: SearchConfig
) -> dict:
    """Search with fallback strategies for empty results."""
    
    query_vector = embed_fn(query)
    
    # Primary: semantic search
    results = search_with_threshold(
        vector_client, collection_name, query_vector, config
    )
    
    if results:
        return {
            "results": results,
            "method": "semantic",
            "query": query
        }
    
    # Fallback 1: Lower threshold
    relaxed_config = SearchConfig(
        min_score=config.min_score * 0.7,
        top_k=config.top_k
    )
    results = search_with_threshold(
        vector_client, collection_name, query_vector, relaxed_config
    )
    
    if results:
        return {
            "results": results,
            "method": "semantic_relaxed",
            "query": query,
            "note": "Results may be less relevant"
        }
    
    # Fallback 2: Keyword search (if enabled)
    if config.fallback_to_keyword:
        # This would use a separate keyword index
        return {
            "results": [],
            "method": "keyword_fallback",
            "query": query,
            "suggestion": "Try different search terms"
        }
    
    return {
        "results": [],
        "method": "none",
        "query": query,
        "message": "No relevant results found"
    }
```

---

## Threshold Calibration

Calibrate thresholds using known relevant pairs:

```python
import numpy as np

def calibrate_threshold(
    queries: list[str],
    relevant_docs: list[str],
    irrelevant_docs: list[str],
    embed_fn
) -> dict:
    """Find optimal threshold based on labeled examples."""
    
    relevant_scores = []
    irrelevant_scores = []
    
    for query in queries:
        query_emb = np.array(embed_fn(query))
        
        for doc in relevant_docs:
            doc_emb = np.array(embed_fn(doc))
            score = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            relevant_scores.append(score)
        
        for doc in irrelevant_docs:
            doc_emb = np.array(embed_fn(doc))
            score = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            irrelevant_scores.append(score)
    
    # Find threshold that maximizes separation
    rel_min = np.min(relevant_scores)
    irr_max = np.max(irrelevant_scores)
    
    suggested_threshold = (rel_min + irr_max) / 2
    
    return {
        "suggested_threshold": float(suggested_threshold),
        "relevant_score_range": (float(np.min(relevant_scores)), float(np.max(relevant_scores))),
        "irrelevant_score_range": (float(np.min(irrelevant_scores)), float(np.max(irrelevant_scores))),
        "separation": float(rel_min - irr_max)  # Positive = good separation
    }
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Calibrate thresholds with your data | Use default thresholds blindly |
| Implement fallback strategies | Return empty results without alternatives |
| Adjust thresholds by query type | Use same threshold for all queries |
| Log scores for analysis | Ignore score distributions |

---

## Summary

✅ **Threshold values depend on the model** - calibrate empirically

✅ **Dynamic thresholds** adjust based on query length and type

✅ **Fallback strategies** prevent empty results from frustrating users

✅ **Calibration with labeled data** finds optimal separation

**Next:** [Metadata Filtering](./04-metadata-filtering.md)
