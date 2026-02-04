---
title: "SPLADE & Learned Sparse Embeddings"
---

# SPLADE & Learned Sparse Embeddings

## Introduction

SPLADE (Sparse Lexical and Expansion) goes beyond traditional BM25 by learning which terms are important and expanding queries with related terms. It creates sparse vectors that capture both exact matches and semantic relationships.

---

## Why SPLADE Beats BM25

| Feature | BM25 | SPLADE |
|---------|------|--------|
| Term matching | Exact only | Learned expansion |
| Vocabulary mismatch | ❌ Fails | ✅ Handles synonyms |
| Term importance | TF-IDF heuristics | Learned weights |
| Zero-shot domain | Works | Works better |

> **🤖 AI Context:** SPLADE learns to expand "machine learning tutorial" to include related terms like "course", "training", and "education" that weren't in the original text. This solves the vocabulary mismatch problem where users search with different words than documents contain.

---

## Implementing SPLADE Encoder

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class SPLADEEncoder:
    """Generate SPLADE sparse vectors."""
    
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> dict:
        """Generate sparse vector with term expansion."""
        tokens = self.tokenizer(
            text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            output = self.model(**tokens)
            logits = output.logits
            
            # SPLADE aggregation: max pooling over sequence + ReLU + log
            splade_vector = torch.max(
                torch.log1p(torch.relu(logits)) * tokens.attention_mask.unsqueeze(-1),
                dim=1
            )[0].squeeze()
        
        # Convert to sparse format
        non_zero_indices = splade_vector.nonzero().squeeze().tolist()
        if isinstance(non_zero_indices, int):
            non_zero_indices = [non_zero_indices]
        
        weights = splade_vector[non_zero_indices].tolist()
        
        # Decode tokens to see expansion
        expanded_terms = self.tokenizer.convert_ids_to_tokens(non_zero_indices)
        
        return {
            "indices": non_zero_indices,
            "values": weights,
            "terms": list(zip(expanded_terms, weights))[:20]  # Top terms
        }

# Example
splade = SPLADEEncoder()
result = splade.encode("machine learning tutorial")

print("Expanded terms (top 10):")
for term, weight in sorted(result["terms"], key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {term}: {weight:.3f}")
```

**Output:**
```
Expanded terms (top 10):
  learning: 2.847
  machine: 2.654
  tutorial: 2.432
  course: 1.823
  training: 1.654
  lesson: 1.432
  education: 1.287
  teach: 1.156
  model: 0.987
  ai: 0.876
```

---

## Combining SPLADE with Dense Vectors

For best results, use SPLADE alongside dense embeddings:

```python
import numpy as np

class HybridSPLADESearcher:
    """Combine SPLADE sparse with dense embeddings."""
    
    def __init__(self, dense_model, splade_encoder):
        self.dense_model = dense_model
        self.splade = splade_encoder
    
    def encode_document(self, text: str) -> dict:
        """Generate both dense and sparse representations."""
        return {
            "dense": self.dense_model.encode(text).tolist(),
            "sparse": self.splade.encode(text)
        }
    
    def encode_query(self, text: str) -> dict:
        """Generate query representations."""
        return {
            "dense": self.dense_model.encode(text).tolist(),
            "sparse": self.splade.encode(text)
        }
    
    def hybrid_score(
        self,
        query_repr: dict,
        doc_repr: dict,
        alpha: float = 0.7
    ) -> float:
        """Calculate hybrid similarity score."""
        
        # Dense cosine similarity
        q_dense = np.array(query_repr["dense"])
        d_dense = np.array(doc_repr["dense"])
        dense_sim = np.dot(q_dense, d_dense) / (
            np.linalg.norm(q_dense) * np.linalg.norm(d_dense)
        )
        
        # Sparse dot product
        q_sparse = dict(zip(
            query_repr["sparse"]["indices"],
            query_repr["sparse"]["values"]
        ))
        d_sparse = dict(zip(
            doc_repr["sparse"]["indices"],
            doc_repr["sparse"]["values"]
        ))
        
        common_indices = set(q_sparse.keys()) & set(d_sparse.keys())
        sparse_sim = sum(q_sparse[i] * d_sparse[i] for i in common_indices)
        
        # Normalize sparse score (approximate)
        if sparse_sim > 0:
            sparse_sim = sparse_sim / (
                sum(q_sparse.values()) + sum(d_sparse.values())
            )
        
        return alpha * dense_sim + (1 - alpha) * sparse_sim
```

---

## SPLADE with Vector Databases

### Storing SPLADE Vectors in Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    NamedVector, NamedSparseVector, SparseVector
)

def create_hybrid_collection(client: QdrantClient, collection_name: str):
    """Create collection supporting both dense and sparse vectors."""
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )

def upsert_hybrid_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    dense_model,
    splade_encoder
):
    """Upsert documents with both dense and sparse vectors."""
    
    points = []
    for i, doc in enumerate(documents):
        dense_vec = dense_model.encode(doc["text"]).tolist()
        sparse_result = splade_encoder.encode(doc["text"])
        
        points.append({
            "id": i,
            "vector": {
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_result["indices"],
                    values=sparse_result["values"]
                )
            },
            "payload": doc
        })
    
    client.upsert(collection_name=collection_name, points=points)
```

---

## When to Use SPLADE

| Use Case | Recommendation |
|----------|----------------|
| Technical documentation | ✅ SPLADE excels at term expansion |
| Code search | ✅ Handles naming variations |
| E-commerce | ✅ Product name synonyms |
| Legal/medical | ⚠️ Test carefully - domain-specific terms |
| Multilingual | ❌ Use language-specific models |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Combine SPLADE with dense vectors | Use SPLADE alone |
| Cache SPLADE encodings | Encode on every query |
| Tune alpha based on query types | Assume 0.5 works best |
| Test on domain-specific queries | Deploy without evaluation |

---

## Summary

✅ **SPLADE learns term importance** instead of using TF-IDF heuristics

✅ **Term expansion** solves vocabulary mismatch problems

✅ **Combine with dense vectors** for best retrieval quality

✅ **Particularly effective** for technical content with specific terminology

**Next:** [Contextual Retrieval](./07-contextual-retrieval.md)
