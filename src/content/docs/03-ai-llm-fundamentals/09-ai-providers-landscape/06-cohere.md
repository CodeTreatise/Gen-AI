---
title: "Cohere"
---

# Cohere

## Introduction

Cohere focuses on enterprise AI with specialized models for RAG (Retrieval-Augmented Generation), embeddings, and reranking. Their models are particularly strong for enterprise search and document processing.

### What We'll Cover

- Command R series
- Embed and Rerank models
- RAG optimization
- Enterprise features

---

## Model Lineup

### Current Models

| Model | Purpose | Context |
|-------|---------|---------|
| Command R+ | Flagship generation | 128K |
| Command R | Fast generation | 128K |
| Command | Legacy generation | 4K |
| Embed v3 | Text embeddings | 512 |
| Rerank v3 | Search reranking | 4K |

---

## Command R Series

### Generation API

```python
import cohere

co = cohere.ClientV2(api_key="YOUR_KEY")

def cohere_chat(prompt: str) -> str:
    response = co.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content[0].text

# With system prompt
def cohere_with_system(system: str, prompt: str) -> str:
    response = co.chat(
        model="command-r-plus",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.message.content[0].text
```

### RAG with Documents

```python
def rag_chat(query: str, documents: list) -> str:
    """Chat with document grounding"""
    
    response = co.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": query}],
        documents=documents  # List of {"text": "..."} objects
    )
    
    # Response includes citations
    return {
        "answer": response.message.content[0].text,
        "citations": response.message.citations
    }

# Usage
docs = [
    {"text": "Python was created by Guido van Rossum."},
    {"text": "Python 3.0 was released in 2008."}
]

result = rag_chat("When was Python 3 released?", docs)
print(result["answer"])
print(result["citations"])  # Shows which docs were used
```

---

## Embed Model

### Text Embeddings

```python
def get_embeddings(texts: list, input_type: str = "search_document") -> list:
    """Generate embeddings for texts"""
    
    response = co.embed(
        model="embed-v3.0",
        texts=texts,
        input_type=input_type,  # search_document or search_query
        embedding_types=["float"]
    )
    
    return response.embeddings.float_

# For documents (indexing)
doc_embeddings = get_embeddings(
    ["Document 1 text...", "Document 2 text..."],
    input_type="search_document"
)

# For queries (searching)
query_embedding = get_embeddings(
    ["What is machine learning?"],
    input_type="search_query"
)
```

### Embed v3 Features

```python
embed_v3_features = {
    "dimensions": 1024,  # Default, can compress
    "max_tokens": 512,
    "multilingual": True,  # 100+ languages
    "input_types": [
        "search_document",  # For corpus
        "search_query",     # For queries
        "classification",   # For classifiers
        "clustering"        # For clustering
    ],
    "compression": "Matryoshka-style dimension reduction"
}
```

---

## Rerank Model

### Search Reranking

```python
def rerank_results(query: str, documents: list, top_n: int = 5) -> list:
    """Rerank search results by relevance"""
    
    response = co.rerank(
        model="rerank-v3.0",
        query=query,
        documents=documents,
        top_n=top_n
    )
    
    # Returns documents sorted by relevance
    return [
        {
            "index": result.index,
            "score": result.relevance_score,
            "document": documents[result.index]
        }
        for result in response.results
    ]

# Usage
docs = [
    "Python is a programming language",
    "Snakes are reptiles", 
    "Python was created by Guido van Rossum",
    "The weather is nice today"
]

results = rerank_results("Who created Python?", docs, top_n=2)
# Returns doc 2 and doc 0 as most relevant
```

### Two-Stage Retrieval

```python
class TwoStageRetriever:
    """Embed for recall, rerank for precision"""
    
    def __init__(self):
        self.co = cohere.ClientV2()
    
    def search(self, query: str, documents: list, top_k: int = 10, top_n: int = 3):
        # Stage 1: Embed and retrieve top_k candidates
        query_emb = self.co.embed(
            model="embed-v3.0",
            texts=[query],
            input_type="search_query"
        ).embeddings.float_[0]
        
        doc_embs = self.co.embed(
            model="embed-v3.0",
            texts=documents,
            input_type="search_document"
        ).embeddings.float_
        
        # Simple cosine similarity (use proper vector DB in production)
        import numpy as np
        scores = [np.dot(query_emb, doc_emb) for doc_emb in doc_embs]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        candidates = [documents[i] for i in top_indices]
        
        # Stage 2: Rerank for precision
        reranked = self.co.rerank(
            model="rerank-v3.0",
            query=query,
            documents=candidates,
            top_n=top_n
        )
        
        return [candidates[r.index] for r in reranked.results]
```

---

## Pricing

### Current Pricing

| Model | Pricing |
|-------|---------|
| Command R+ | $2.50 input / $10.00 output per 1M |
| Command R | $0.15 input / $0.60 output per 1M |
| Embed v3 | $0.10 per 1M tokens |
| Rerank v3 | $2.00 per 1K queries |

---

## Enterprise Features

### Cohere Enterprise

```python
enterprise_features = {
    "private_deployment": "Deploy in your cloud",
    "fine_tuning": "Custom model training",
    "sso": "Single sign-on integration",
    "audit_logs": "Compliance logging",
    "sla": "99.9% uptime guarantee",
    "support": "Dedicated support team"
}
```

---

## Best Use Cases

```python
cohere_best_for = [
    "Enterprise search systems",
    "RAG applications with citations",
    "Document retrieval and ranking",
    "Multilingual applications",
    "Semantic search",
    "Content recommendation"
]
```

---

## Summary

✅ **RAG-optimized**: Built-in document grounding with citations

✅ **Strong embeddings**: Embed v3 multilingual, high quality

✅ **Best reranker**: Industry-leading rerank model

✅ **Enterprise focus**: Private deployment, compliance

✅ **Simple pricing**: Predictable costs

**Next:** [Groq](./07-groq.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Mistral](./05-mistral.md) | [AI Providers](./00-ai-providers-landscape.md) | [Groq](./07-groq.md) |

