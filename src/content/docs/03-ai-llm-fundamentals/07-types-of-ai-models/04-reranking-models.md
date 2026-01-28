---
title: "Reranking Models"
---

# Reranking Models

## Introduction

Reranking models refine search results by scoring query-document relevance more accurately than embedding similarity alone. They're essential for high-quality retrieval systems.

### What We'll Cover

- Why reranking matters
- Two-stage retrieval
- Cross-encoder vs bi-encoder
- Leading reranking models

---

## Why Reranking?

### The Retrieval Quality Problem

```
Stage 1: Embedding Search (Fast, but approximate)
──────────────────────────────────────────────────
Query: "How to fix Python memory leak"

Results by embedding similarity:
1. "Python memory management basics" (sim: 0.85)
2. "Memory leak in Python applications" (sim: 0.83)  ← Actually best!
3. "Python programming tutorial" (sim: 0.82)
4. "Debugging memory issues in Python" (sim: 0.81)

Stage 2: Reranking (Slower, but precise)
──────────────────────────────────────────────────
Same results, reranked by relevance:
1. "Memory leak in Python applications" (score: 0.95)  ← Promoted!
2. "Debugging memory issues in Python" (score: 0.89)   ← Promoted!
3. "Python memory management basics" (score: 0.72)
4. "Python programming tutorial" (score: 0.31)          ← Demoted!
```

### Two-Stage Retrieval

```python
from typing import List, Tuple

class TwoStageRetriever:
    """Two-stage retrieve then rerank"""
    
    def __init__(self, embedder, reranker, vector_db):
        self.embedder = embedder
        self.reranker = reranker
        self.vector_db = vector_db
    
    def search(self, query: str, top_k: int = 5, retrieve_n: int = 50) -> List[dict]:
        # Stage 1: Fast retrieval with embeddings
        # Retrieve more than needed (50) for reranking
        query_embedding = self.embedder.embed(query)
        candidates = self.vector_db.search(query_embedding, n=retrieve_n)
        
        # Stage 2: Precise reranking
        # Score each candidate against query
        reranked = self.reranker.rerank(query, candidates)
        
        # Return top results after reranking
        return reranked[:top_k]
```

---

## Cross-Encoder vs Bi-Encoder

### Bi-Encoder (Embeddings)

```
Query: "Python memory leak"    Document: "Fixing memory issues..."
         │                              │
         ▼                              ▼
    ┌─────────┐                   ┌─────────┐
    │ Encoder │                   │ Encoder │
    └────┬────┘                   └────┬────┘
         │                              │
         ▼                              ▼
    [0.2, -0.3, ...]              [0.1, -0.4, ...]
         │                              │
         └──────────┬───────────────────┘
                    ▼
              Cosine Similarity
                    │
                    ▼
                  0.85

Pros: FAST (encode once, compare many)
Cons: Less accurate (no cross-attention)
```

### Cross-Encoder (Reranking)

```
Query + Document: "Python memory leak [SEP] Fixing memory issues..."
                              │
                              ▼
                       ┌─────────────┐
                       │   Encoder   │
                       │ (sees both) │
                       └──────┬──────┘
                              │
                              ▼
                       Relevance Score
                              │
                              ▼
                            0.92

Pros: ACCURATE (full attention between query and doc)
Cons: Slow (must encode every pair)
```

---

## Leading Reranking Models

### Cohere Rerank

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

def rerank_with_cohere(query: str, documents: List[str], top_n: int = 5) -> List[dict]:
    """Rerank documents using Cohere"""
    
    response = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",  # or rerank-multilingual-v3.0
        top_n=top_n
    )
    
    return [
        {
            "text": documents[result.index],
            "score": result.relevance_score,
            "index": result.index
        }
        for result in response.results
    ]

# Example
docs = [
    "Python memory management uses garbage collection",
    "How to identify and fix memory leaks in Python",
    "Introduction to Python programming",
    "Memory profiling tools for Python applications"
]

results = rerank_with_cohere("fix python memory leak", docs, top_n=3)
for r in results:
    print(f"{r['score']:.3f}: {r['text'][:50]}...")
```

### Voyage Rerank

```python
import voyageai

vo = voyageai.Client()

def rerank_with_voyage(query: str, documents: List[str], top_k: int = 5) -> List[dict]:
    """Rerank using Voyage AI"""
    
    response = vo.rerank(
        query=query,
        documents=documents,
        model="rerank-2",
        top_k=top_k
    )
    
    return [
        {
            "text": r.document,
            "score": r.relevance_score,
            "index": r.index
        }
        for r in response.results
    ]
```

### Open Source: Cross-Encoder

```python
from sentence_transformers import CrossEncoder

# Load model (runs locally)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_local(query: str, documents: List[str]) -> List[Tuple[float, str]]:
    """Rerank using local cross-encoder"""
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]
    
    # Score all pairs
    scores = model.predict(pairs)
    
    # Sort by score
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(reverse=True)
    
    return scored_docs

# Example
results = rerank_local("python memory leak", docs)
for score, doc in results[:3]:
    print(f"{score:.3f}: {doc[:50]}...")
```

---

## Integration Patterns

### RAG with Reranking

```python
class RAGWithReranking:
    """RAG system with two-stage retrieval"""
    
    def __init__(self, vector_store, reranker, llm):
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
    
    def query(self, question: str) -> str:
        # Stage 1: Vector search
        initial_results = self.vector_store.similarity_search(
            question, 
            k=20  # Retrieve more for reranking
        )
        
        # Stage 2: Rerank
        documents = [r.page_content for r in initial_results]
        reranked = self.reranker.rerank(question, documents, top_n=5)
        
        # Filter by relevance threshold
        relevant_docs = [r for r in reranked if r['score'] > 0.5]
        
        # Generate answer
        context = "\n\n".join([r['text'] for r in relevant_docs])
        
        return self.llm.generate(
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
```

### Relevance Threshold

```python
def filter_by_relevance(
    reranked_results: List[dict], 
    threshold: float = 0.5
) -> List[dict]:
    """Filter results below relevance threshold"""
    
    return [r for r in reranked_results if r['score'] >= threshold]

# Dynamic threshold based on top score
def adaptive_filter(results: List[dict], ratio: float = 0.5) -> List[dict]:
    """Keep results within ratio of top score"""
    
    if not results:
        return []
    
    top_score = results[0]['score']
    threshold = top_score * ratio
    
    return [r for r in results if r['score'] >= threshold]
```

---

## Performance Considerations

### Speed vs Quality Trade-off

| Approach | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| Embedding only | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | High volume |
| Light reranker | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| Full reranker | ⚡⚡ | ⭐⭐⭐⭐⭐ | Quality critical |

### Batch Processing

```python
def batch_rerank(
    queries: List[str], 
    documents_per_query: List[List[str]]
) -> List[List[dict]]:
    """Batch reranking for efficiency"""
    
    # Cohere supports batching
    all_results = []
    
    for query, docs in zip(queries, documents_per_query):
        results = co.rerank(
            query=query,
            documents=docs,
            model="rerank-english-v3.0",
            top_n=5
        )
        all_results.append(results)
    
    return all_results
```

---

## Hands-on Exercise

### Your Task

Implement and compare retrieval with/without reranking:

```python
import cohere
from openai import OpenAI

openai_client = OpenAI()
co = cohere.Client("YOUR_COHERE_KEY")

# Sample documents
documents = [
    "Python's garbage collector automatically frees memory",
    "How to debug and fix memory leaks in Python applications",
    "Introduction to programming with Python for beginners",
    "Memory profiling tools: tracemalloc and memory_profiler",
    "Best practices for memory-efficient Python code",
    "Python tutorial: variables and data types",
    "Understanding Python's memory allocation model",
    "Web development with Python and Flask framework"
]

def embedding_search(query: str, docs: List[str], top_k: int = 3) -> List[Tuple[float, str]]:
    """Basic embedding similarity search"""
    import numpy as np
    
    # Get embeddings
    response = openai_client.embeddings.create(
        input=[query] + docs,
        model="text-embedding-3-small"
    )
    
    query_emb = response.data[0].embedding
    doc_embs = [response.data[i+1].embedding for i in range(len(docs))]
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embs):
        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        similarities.append((sim, docs[i]))
    
    similarities.sort(reverse=True)
    return similarities[:top_k]

def reranked_search(query: str, docs: List[str], top_k: int = 3) -> List[Tuple[float, str]]:
    """Search with reranking"""
    
    # First get embedding results
    embedding_results = embedding_search(query, docs, top_k=len(docs))
    
    # Rerank all
    response = co.rerank(
        query=query,
        documents=[doc for _, doc in embedding_results],
        model="rerank-english-v3.0",
        top_n=top_k
    )
    
    return [(r.relevance_score, docs[r.index]) for r in response.results]

# Compare
query = "How do I fix memory leak in Python?"

print("=== Embedding Only ===")
for score, doc in embedding_search(query, documents):
    print(f"{score:.3f}: {doc}")

print("\n=== With Reranking ===")
for score, doc in reranked_search(query, documents):
    print(f"{score:.3f}: {doc}")
```

---

## Summary

✅ **Reranking improves precision** over embedding similarity

✅ **Two-stage retrieval**: Fast embeddings → Precise reranking

✅ **Cross-encoders** see query+document together for accuracy

✅ **Cohere Rerank** and **Voyage Rerank** are leading options

✅ **Trade-off**: Speed vs quality based on use case

**Next:** [Classification Models](./05-classification-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Embedding Models](./03-embedding-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Classification Models](./05-classification-models.md) |

