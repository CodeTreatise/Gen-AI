---
title: "Multi-Stage Prefetch Patterns"
---

# Multi-Stage Prefetch Patterns

## Introduction

Multi-stage prefetch uses successive refinement: first retrieve many candidates with fast methods, then re-score with accurate methods. Qdrant's `prefetch` parameter enables this as a single query with multiple stages.

> **🤖 AI Context:** This is the production pattern for high-quality RAG at scale: quantized vectors for initial recall, full vectors for precision, optionally ColBERT for final ranking.

---

## The Multi-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE RETRIEVAL                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Quantized Vectors (Fast)                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 1M vectors → 1000 candidates                                   │ │
│  │ Binary/Scalar quantization, ~5ms                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Stage 2: Full Vectors (Accurate)                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 1000 candidates → 100 candidates                               │ │
│  │ Full float32 vectors, ~10ms                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Stage 3: ColBERT/Cross-Encoder (Precise)                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 100 candidates → 10 final results                              │ │
│  │ Multi-vector or reranking, ~50ms                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Total: ~65ms for 1M vectors with ColBERT-level quality             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Qdrant Prefetch Queries

Basic two-stage prefetch:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

def search_with_prefetch(
    collection: str,
    query_vector: list[float],
    top_k: int = 10
):
    """Two-stage search: quantized → full vectors."""
    
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            # Stage 1: Fast search with quantized vectors
            models.Prefetch(
                query=query_vector,
                using="fast",  # Quantized vector name
                limit=1000     # Retrieve many candidates
            )
        ],
        # Stage 2: Re-score with full vectors
        query=query_vector,
        using="full",          # Full precision vector name
        limit=top_k
    )
    
    return results
```

---

## Setting Up Multi-Vector Collection

```python
from qdrant_client.models import (
    VectorParams, 
    Distance,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType
)

# Create collection with multiple vector types
client.create_collection(
    collection_name="multi_stage",
    vectors_config={
        # Full precision vectors for final scoring
        "full": VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
        # Quantized for fast initial search
        "fast": VectorParams(
            size=1536,
            distance=Distance.COSINE,
            quantization_config=ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True  # Keep in RAM for speed
            )
        ),
        # ColBERT multi-vectors for precision (optional)
        "colbert": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            )
        )
    }
)
```

---

## Three-Stage Pipeline

Add ColBERT for maximum precision:

```python
def search_three_stage(
    collection: str,
    query_embedding: list[float],
    colbert_query_vectors: list[list[float]],
    top_k: int = 10
):
    """Three-stage: quantized → full → ColBERT."""
    
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                # Stage 1: Quantized search
                prefetch=[
                    models.Prefetch(
                        query=query_embedding,
                        using="fast",
                        limit=1000
                    )
                ],
                # Stage 2: Full vector refinement
                query=query_embedding,
                using="full",
                limit=100
            )
        ],
        # Stage 3: ColBERT MaxSim scoring
        query=colbert_query_vectors,
        using="colbert",
        limit=top_k
    )
    
    return results
```

---

## Hybrid Search with Prefetch

Combine dense and sparse in multi-stage:

```python
def hybrid_multi_stage(
    collection: str,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    top_k: int = 10
):
    """Hybrid search with multi-stage prefetch."""
    
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            # Dense search branch
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=100
            ),
            # Sparse search branch
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=100
            )
        ],
        # Final stage: RRF fusion of both
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    
    return results
```

---

## Latency Optimization Strategies

```python
# Strategy 1: Aggressive quantization for stage 1
fast_config = ScalarQuantization(
    type=ScalarType.INT8,
    quantile=0.95,      # More aggressive
    always_ram=True
)

# Strategy 2: Binary quantization (fastest, less accurate)
binary_config = models.BinaryQuantization(
    always_ram=True
)

# Strategy 3: Oversampling at each stage
def search_with_oversample(collection: str, query: list[float], top_k: int = 10):
    return client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=query,
                using="fast",
                limit=top_k * 100  # 100x oversample
            )
        ],
        query=query,
        using="full",
        limit=top_k * 10  # 10x oversample
    )
```

---

## Custom Multi-Stage in Python

For databases without native prefetch:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

class MultiStagePipeline:
    """Custom multi-stage retrieval pipeline."""
    
    def __init__(self):
        # Stage 1: Fast bi-encoder
        self.fast_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Stage 2: Accurate bi-encoder  
        self.full_encoder = SentenceTransformer("all-mpnet-base-v2")
        
        # Stage 3: Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.documents = []
        self.fast_embeddings = None
        self.full_embeddings = None
    
    def index(self, documents: list[str]):
        """Index documents with both encoders."""
        
        self.documents = documents
        self.fast_embeddings = self.fast_encoder.encode(documents)
        self.full_embeddings = self.full_encoder.encode(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_k: int = 1000,
        stage2_k: int = 100
    ) -> list[dict]:
        """Three-stage search."""
        
        # Stage 1: Fast embedding search
        fast_query = self.fast_encoder.encode(query)
        fast_scores = np.dot(self.fast_embeddings, fast_query)
        stage1_indices = np.argsort(fast_scores)[-stage1_k:][::-1]
        
        # Stage 2: Full embedding re-scoring
        full_query = self.full_encoder.encode(query)
        stage1_embeddings = self.full_embeddings[stage1_indices]
        full_scores = np.dot(stage1_embeddings, full_query)
        
        # Get top stage2_k from stage 1 candidates
        top_stage2 = np.argsort(full_scores)[-stage2_k:][::-1]
        stage2_indices = stage1_indices[top_stage2]
        
        # Stage 3: Cross-encoder reranking
        stage2_docs = [self.documents[i] for i in stage2_indices]
        pairs = [[query, doc] for doc in stage2_docs]
        rerank_scores = self.reranker.predict(pairs)
        
        # Final ranking
        final_order = np.argsort(rerank_scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(final_order):
            doc_idx = stage2_indices[idx]
            results.append({
                "document": self.documents[doc_idx],
                "score": float(rerank_scores[idx]),
                "stage1_score": float(fast_scores[doc_idx]),
                "stage2_score": float(full_scores[top_stage2[idx]]),
            })
        
        return results
```

---

## Performance Benchmarks

| Stage Configuration | 1M Docs Latency | Quality |
|---------------------|-----------------|---------|
| Direct full vectors | ~100ms | High |
| Quantized only | ~5ms | Medium |
| Quantized → Full | ~15ms | High |
| Quantized → Full → Rerank | ~115ms | Highest |
| Quantized → ColBERT | ~65ms | Very High |

**Latency breakdown:**
```
Stage 1 (Quantized, 1000 candidates): 5ms
Stage 2 (Full vectors, 100 candidates): 10ms
Stage 3 (ColBERT/Rerank, 10 results): 50-100ms
─────────────────────────────────────────
Total: 65-115ms
```

---

## Choosing Stage Limits

| Collection Size | Stage 1 Limit | Stage 2 Limit | Final |
|-----------------|---------------|---------------|-------|
| <100K | 500 | 50 | 10 |
| 100K-1M | 1000 | 100 | 10 |
| 1M-10M | 2000 | 200 | 10 |
| 10M+ | 5000 | 500 | 10 |

**Rule of thumb:** Each stage reduces by ~10x.

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| 10x reduction per stage | Too aggressive reduction |
| Keep stage 1 in RAM | Disk-based stage 1 |
| Measure each stage latency | Optimize blindly |
| Match query encoding per stage | Use wrong encoder |
| Tune limits for your data | Use generic limits |

---

## Summary

✅ **Multi-stage** balances speed and accuracy

✅ **Quantized → Full → Rerank** is the production pattern

✅ **10x reduction** per stage is a good starting point

✅ **Keep stage 1 in RAM** for lowest latency

✅ **ColBERT** offers precision without cross-encoder cost

**Next:** [Advanced Techniques Overview](./00-advanced-techniques.md)

---

<!-- 
Sources Consulted:
- Qdrant Hybrid Queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
-->
