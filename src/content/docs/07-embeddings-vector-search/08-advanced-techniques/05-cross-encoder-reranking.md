---
title: "Cross-Encoder Reranking"
---

# Cross-Encoder Reranking

## Introduction

Cross-encoders process query and document together through a single transformer, enabling deep interaction between them. This produces more accurate relevance scores than bi-encoders but at higher computational cost. The solution: use bi-encoders for initial retrieval, cross-encoders for reranking.

> **🤖 AI Context:** Cross-encoders are the "slow but accurate" counterpart to bi-encoders. Production RAG systems typically retrieve 50-100 candidates with bi-encoders, then rerank with cross-encoders to return the top 10.

---

## Bi-Encoder vs Cross-Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                      BI-ENCODER                                 │
├─────────────────────────────────────────────────────────────────┤
│  Query: "What is RAG?"    Document: "RAG combines..."          │
│         ↓                          ↓                            │
│    Encoder                    Encoder                           │
│         ↓                          ↓                            │
│  [0.2, 0.5, ...]           [0.3, 0.4, ...]                     │
│         ↓                          ↓                            │
│         └──── Cosine Similarity ───┘                            │
│                    ↓                                            │
│               Score: 0.85                                       │
│                                                                 │
│  ✅ Fast: Pre-compute document embeddings                       │
│  ❌ Limited: No cross-attention between query and document      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     CROSS-ENCODER                               │
├─────────────────────────────────────────────────────────────────┤
│  Input: "[CLS] What is RAG? [SEP] RAG combines... [SEP]"       │
│                          ↓                                      │
│                    Full Transformer                             │
│              (with cross-attention)                             │
│                          ↓                                      │
│                    Score: 0.92                                  │
│                                                                 │
│  ✅ Accurate: Full interaction between query and document       │
│  ❌ Slow: Must process each query-document pair                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Basic Cross-Encoder Usage

```python
from sentence_transformers import CrossEncoder

# Load pre-trained cross-encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "The weather today is sunny with a high of 75 degrees.",
    "Deep learning uses neural networks with multiple layers.",
]

# Create pairs
pairs = [[query, doc] for doc in documents]

# Get relevance scores
scores = cross_encoder.predict(pairs)
print(scores)
# Output: [0.98, 0.01, 0.85]
```

---

## Two-Stage Retrieval Pipeline

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

class TwoStageRetriever:
    """Bi-encoder retrieval + cross-encoder reranking."""
    
    def __init__(
        self,
        bi_encoder_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.documents = []
        self.embeddings = None
    
    def index(self, documents: list[str]):
        """Index documents with bi-encoder."""
        self.documents = documents
        self.embeddings = self.bi_encoder.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 50
    ) -> list[dict]:
        """Two-stage retrieval: bi-encoder → cross-encoder."""
        
        # Stage 1: Bi-encoder retrieval
        query_embedding = self.bi_encoder.encode(query)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top N candidates for reranking
        top_indices = np.argsort(similarities)[-rerank_top_n:][::-1]
        candidates = [self.documents[i] for i in top_indices]
        
        # Stage 2: Cross-encoder reranking
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Sort by cross-encoder scores
        sorted_indices = np.argsort(rerank_scores)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "document": candidates[idx],
                "score": float(rerank_scores[idx]),
                "original_rank": idx + 1
            })
        
        return results

# Usage
retriever = TwoStageRetriever()
retriever.index(documents)
results = retriever.retrieve("What is RAG?")
```

---

## BGE Reranker

BGE reranker models from BAAI are among the best performing:

```python
# Using FlagEmbedding library
from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",  # Multilingual
    use_fp16=True  # Faster inference
)

query = "What is machine learning?"
documents = [
    "ML is a type of artificial intelligence.",
    "The sun is a star in our solar system.",
    "Neural networks are used in deep learning.",
]

# Score pairs
pairs = [[query, doc] for doc in documents]
scores = reranker.compute_score(pairs, normalize=True)  # 0-1 range
print(scores)
# Output: [0.95, 0.02, 0.78]
```

### BGE Reranker Variants

| Model | Languages | Size | Best For |
|-------|-----------|------|----------|
| `bge-reranker-base` | English | 278M | General English |
| `bge-reranker-large` | English | 560M | High accuracy |
| `bge-reranker-v2-m3` | 100+ | 568M | Multilingual |
| `bge-reranker-v2-gemma` | English | 2B | Maximum accuracy |

---

## Cross-Encoder with Sentence Transformers

```python
from sentence_transformers import CrossEncoder

# MS-MARCO trained models (optimized for search)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# For semantic similarity (not search)
similarity_model = CrossEncoder("cross-encoder/stsb-roberta-large")

# Score documents
scores = cross_encoder.predict([
    ["query", "relevant document"],
    ["query", "irrelevant document"]
])
```

### Popular Cross-Encoder Models

| Model | Task | Size | Notes |
|-------|------|------|-------|
| `ms-marco-MiniLM-L-6-v2` | Search | 22M | Fast, good accuracy |
| `ms-marco-MiniLM-L-12-v2` | Search | 33M | Better accuracy |
| `ms-marco-TinyBERT-L-2-v2` | Search | 4M | Fastest |
| `stsb-roberta-large` | Similarity | 355M | Semantic similarity |

---

## Reranking with Qdrant

Integrate cross-encoder reranking with Qdrant:

```python
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

client = QdrantClient("localhost", port=6333)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def search_with_rerank(
    query: str,
    collection: str,
    top_k: int = 10,
    rerank_candidates: int = 50
) -> list[dict]:
    """Search Qdrant then rerank with cross-encoder."""
    
    # Stage 1: Vector search
    search_results = client.query_points(
        collection_name=collection,
        query=query,  # Qdrant handles embedding
        limit=rerank_candidates
    )
    
    # Extract documents
    candidates = [
        {"id": r.id, "text": r.payload["text"], "vector_score": r.score}
        for r in search_results.points
    ]
    
    # Stage 2: Cross-encoder reranking
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    
    # Combine and sort
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return candidates[:top_k]
```

---

## Batch Reranking for Efficiency

```python
import torch
from sentence_transformers import CrossEncoder

class BatchCrossEncoder:
    """Efficient batched cross-encoder reranking."""
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        
        # Enable GPU if available
        if torch.cuda.is_available():
            self.model.model.to("cuda")
    
    def rerank(
        self,
        query: str,
        documents: list[str]
    ) -> list[tuple[int, float]]:
        """Rerank documents in batches."""
        
        pairs = [[query, doc] for doc in documents]
        
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores)
        
        # Return (index, score) pairs sorted by score
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores

# Usage
reranker = BatchCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
ranked = reranker.rerank("What is RAG?", documents)
```

---

## Performance Comparison

| Stage | Latency (100 docs) | Latency (1000 docs) |
|-------|-------------------|---------------------|
| Bi-encoder retrieval | ~10ms | ~50ms |
| Cross-encoder rerank (50) | ~100ms | ~100ms |
| Cross-encoder rerank (100) | ~200ms | ~200ms |
| Total (bi + rerank 50) | ~110ms | ~150ms |

**Key insight:** Cross-encoder latency scales with candidates, not corpus size.

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Rerank 50-100 candidates | Rerank entire corpus |
| Use smaller models for speed | Always use largest model |
| Batch reranking calls | Call one pair at a time |
| Cache rerank scores | Recompute for same query |
| Truncate long documents | Pass full documents (slow) |

```python
# Truncate documents for efficiency
def truncate_for_rerank(text: str, max_tokens: int = 512) -> str:
    """Truncate text for cross-encoder input."""
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return text
```

---

## Summary

✅ **Cross-encoders** process query+document together for higher accuracy

✅ **Two-stage pipeline**: bi-encoder retrieval → cross-encoder reranking

✅ **BGE reranker** models offer excellent multilingual performance

✅ **Rerank 50-100 candidates** for best accuracy/latency tradeoff

✅ **Batch processing** improves throughput significantly

**Next:** [Clustering Embeddings](./06-clustering-embeddings.md)

---

<!-- 
Sources Consulted:
- SBERT Cross-Encoders: https://www.sbert.net/docs/cross_encoder/usage/usage.html
- BGE Reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3
- FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
-->
