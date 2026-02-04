---
title: "Multi-Vector Representations"
---

# Multi-Vector Representations

## Introduction

Traditional embedding models produce a single vector per document. Multi-vector approaches generate multiple embeddings—one per token or passage segment—enabling more nuanced matching where specific parts of a query can align with specific parts of a document.

---

## Single Vector vs Multi-Vector

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLE VECTOR (Bi-Encoder)                   │
├─────────────────────────────────────────────────────────────────┤
│  Query: "What is ReLU?"     →  [0.2, 0.4, 0.1, ...]  (1 vector) │
│  Doc: "ReLU is max(0,x)..." →  [0.3, 0.5, 0.2, ...]  (1 vector) │
│                                                                 │
│  Score = cosine(query_vec, doc_vec)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-VECTOR (ColBERT)                       │
├─────────────────────────────────────────────────────────────────┤
│  Query: "What is ReLU?"                                         │
│    → [[0.2,...], [0.3,...], [0.1,...], [0.4,...]]  (4 vectors) │
│                                                                 │
│  Doc: "ReLU is max(0,x)..."                                    │
│    → [[0.3,...], [0.5,...], [0.2,...], ...]  (N vectors)       │
│                                                                 │
│  Score = Σ max_sim(query_token, all_doc_tokens)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## ColBERT Architecture

ColBERT (Contextualized Late Interaction over BERT) computes token-level embeddings and uses late interaction for scoring:

```python
import torch
from transformers import AutoModel, AutoTokenizer

class ColBERTEncoder:
    """ColBERT-style multi-vector encoder."""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query into per-token embeddings."""
        inputs = self.tokenizer(
            f"[Q] {query}",  # Query marker
            return_tensors="pt",
            truncation=True,
            max_length=32
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # [num_tokens, hidden_dim]
        return outputs.last_hidden_state[0]
    
    def encode_document(self, doc: str) -> torch.Tensor:
        """Encode document into per-token embeddings."""
        inputs = self.tokenizer(
            f"[D] {doc}",  # Document marker
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0]
    
    def maxsim_score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor
    ) -> float:
        """
        Compute MaxSim score.
        
        For each query token, find max similarity to any document token,
        then sum across all query tokens.
        """
        # Normalize for cosine similarity
        query_norm = query_embs / query_embs.norm(dim=1, keepdim=True)
        doc_norm = doc_embs / doc_embs.norm(dim=1, keepdim=True)
        
        # Similarity matrix [query_tokens, doc_tokens]
        similarity = torch.mm(query_norm, doc_norm.T)
        
        # Max over document tokens for each query token
        max_sims = similarity.max(dim=1).values
        
        # Sum of max similarities
        return max_sims.sum().item()
```

---

## Late Interaction Benefits

| Aspect | Single Vector | Multi-Vector |
|--------|---------------|--------------|
| Storage | 1 vector/doc | N vectors/doc |
| Accuracy | Good | Better |
| Fine-grained matching | ❌ | ✅ |
| Query speed | Fast | Slower |
| Use case | General retrieval | Precision-critical |

---

## Multiple Embeddings Per Document

Another approach: embed different aspects of a document:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def multi_aspect_embedding(document: dict) -> dict:
    """Create multiple embeddings for different document aspects."""
    
    embeddings = {}
    
    # Title embedding (for topic matching)
    if document.get("title"):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=document["title"]
        )
        embeddings["title"] = response.data[0].embedding
    
    # Content embedding (for detail matching)
    if document.get("content"):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=document["content"][:8000]
        )
        embeddings["content"] = response.data[0].embedding
    
    # Summary embedding (for overview matching)
    if document.get("summary"):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=document["summary"]
        )
        embeddings["summary"] = response.data[0].embedding
    
    return embeddings

def multi_aspect_search(
    query_embedding: list[float],
    documents: list[dict],
    weights: dict = None
) -> list[dict]:
    """Search with weighted multi-aspect scoring."""
    
    weights = weights or {"title": 0.3, "content": 0.5, "summary": 0.2}
    query = np.array(query_embedding)
    
    results = []
    for doc in documents:
        score = 0.0
        for aspect, weight in weights.items():
            if aspect in doc.get("embeddings", {}):
                aspect_emb = np.array(doc["embeddings"][aspect])
                sim = np.dot(query, aspect_emb) / (
                    np.linalg.norm(query) * np.linalg.norm(aspect_emb)
                )
                score += weight * sim
        
        results.append({"doc": doc, "score": score})
    
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

---

## Aggregation Strategies

When you have multiple vectors, how do you combine them?

```python
import numpy as np

def aggregate_embeddings(
    embeddings: list[list[float]],
    strategy: str = "mean"
) -> list[float]:
    """Aggregate multiple embeddings into one."""
    
    embs = np.array(embeddings)
    
    if strategy == "mean":
        # Average all embeddings
        return embs.mean(axis=0).tolist()
    
    elif strategy == "max":
        # Element-wise max
        return embs.max(axis=0).tolist()
    
    elif strategy == "weighted_mean":
        # First embedding weighted higher (e.g., title)
        weights = np.array([0.5] + [0.5 / (len(embs) - 1)] * (len(embs) - 1))
        return np.average(embs, axis=0, weights=weights).tolist()
    
    elif strategy == "concat":
        # Concatenate (increases dimensionality)
        return embs.flatten().tolist()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# Usage
chunk_embeddings = [
    [0.1, 0.2, 0.3],  # Chunk 1
    [0.2, 0.3, 0.4],  # Chunk 2
    [0.3, 0.4, 0.5],  # Chunk 3
]

mean_emb = aggregate_embeddings(chunk_embeddings, "mean")
# [0.2, 0.3, 0.4]

max_emb = aggregate_embeddings(chunk_embeddings, "max")
# [0.3, 0.4, 0.5]
```

---

## Qdrant Multi-Vector Support

Qdrant supports multiple named vectors per point:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct
)

client = QdrantClient(":memory:")

# Create collection with multiple vector types
client.create_collection(
    collection_name="documents",
    vectors_config={
        "title": VectorParams(size=384, distance=Distance.COSINE),
        "content": VectorParams(size=384, distance=Distance.COSINE),
        "summary": VectorParams(size=384, distance=Distance.COSINE),
    }
)

# Upsert with multiple vectors
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector={
                "title": title_embedding,
                "content": content_embedding,
                "summary": summary_embedding,
            },
            payload={"doc_id": "doc_001"}
        )
    ]
)

# Search using specific vector
results = client.search(
    collection_name="documents",
    query_vector=("title", query_embedding),  # Search title vectors
    limit=10
)

# Or combine with prefetch and fusion
results = client.query_points(
    collection_name="documents",
    prefetch=[
        {"query": query_embedding, "using": "title", "limit": 50},
        {"query": query_embedding, "using": "content", "limit": 50},
    ],
    query={"fusion": "rrf"},  # Reciprocal Rank Fusion
    limit=10
)
```

---

## ColBERT with Qdrant

Store token embeddings as multi-vectors:

```python
# Create collection for ColBERT-style storage
client.create_collection(
    collection_name="colbert_docs",
    vectors_config={
        "colbert": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config={
                "comparator": "max_sim"  # MaxSim scoring
            }
        )
    }
)

# Upsert with token embeddings
client.upsert(
    collection_name="colbert_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "colbert": [
                    [0.1, 0.2, ...],  # Token 1 embedding
                    [0.2, 0.3, ...],  # Token 2 embedding
                    [0.3, 0.4, ...],  # Token 3 embedding
                    # ... more tokens
                ]
            },
            payload={"text": "Document content..."}
        )
    ]
)

# Query with multi-vector
results = client.query_points(
    collection_name="colbert_docs",
    query=[
        [0.1, 0.2, ...],  # Query token 1
        [0.2, 0.3, ...],  # Query token 2
    ],
    using="colbert",
    limit=10
)
```

---

## Performance Considerations

| Factor | Single Vector | Multi-Vector |
|--------|---------------|--------------|
| Index size | 1× | N× (tokens per doc) |
| Query latency | 1× | 2-5× |
| Accuracy | Baseline | +5-15% on benchmarks |
| Storage | Low | High |

**When to use multi-vector:**
- Precision is critical
- Queries need fine-grained matching
- Storage/latency budget available
- Legal, medical, or technical domains

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use for precision-critical apps | Use for simple keyword matching |
| Compress token embeddings | Store full-precision for all tokens |
| Limit document tokens (256-512) | Store thousands of tokens per doc |
| Combine with two-stage retrieval | Use as only retrieval method |
| Cache query embeddings | Re-encode same queries |

---

## Summary

✅ **Multi-vector** stores multiple embeddings per document (token-level or aspect-level)

✅ **ColBERT** uses late interaction with MaxSim scoring for accuracy

✅ **Multiple aspects** (title, content, summary) enable weighted search

✅ **Aggregation strategies** (mean, max, weighted) combine embeddings

✅ **Higher accuracy** at cost of storage and latency

**Next:** [Hypothetical Document Embeddings](./02-hypothetical-document-embeddings.md)
