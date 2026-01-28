---
title: "Embedding Models"
---

# Embedding Models

## Introduction

Embedding models convert text into numerical vectors that capture semantic meaning. These vectors enable semantic search, similarity matching, clustering, and form the foundation of RAG systems.

### What We'll Cover

- What embeddings are and how they work
- Major embedding models
- Dimension sizes and trade-offs
- Specialized embeddings

---

## What Are Embeddings?

### Text to Vector

```python
# Conceptual example
text = "The cat sat on the mat"
embedding = [0.023, -0.041, 0.089, ..., 0.012]  # 1536 dimensions

# Similar meanings → Similar vectors
# "The cat sat on the mat"  →  [0.023, -0.041, ...]
# "A cat was on the rug"    →  [0.025, -0.039, ...]  # Very similar!
# "Stock prices fell today" →  [0.891, 0.234, ...]   # Very different
```

### Visual Representation

```
                    High-dimensional space
                         (projected to 2D)
                              
    "happy" ●                      ● "joyful"
              \                   /
               \                 /
                ●───────────────● 
              "glad"          "cheerful"
                              
                              
                              
    "sad" ●                        ● "unhappy"
            \                     /
             \                   /
              ●─────────────────●
           "depressed"       "melancholy"
```

---

## Major Embedding Models

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Example
embedding = get_embedding("Hello, world!")
print(f"Dimensions: {len(embedding)}")  # 1536
```

**Models:**
- `text-embedding-3-small` — 1536 dimensions, fast, cheap
- `text-embedding-3-large` — 3072 dimensions, more accurate
- `text-embedding-ada-002` — Legacy, 1536 dimensions

### Cohere Embeddings

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

response = co.embed(
    texts=["Hello, world!"],
    model="embed-english-v3.0",
    input_type="search_document"
)

embedding = response.embeddings[0]
print(f"Dimensions: {len(embedding)}")  # 1024
```

**Models:**
- `embed-english-v3.0` — English, 1024 dimensions
- `embed-multilingual-v3.0` — 100+ languages
- `embed-v4.0` — Multimodal (text + images)

### Voyage AI

```python
import voyageai

vo = voyageai.Client()

response = vo.embed(
    ["Hello, world!"],
    model="voyage-3"
)

embedding = response.embeddings[0]
```

**Strengths:** High quality, specialized models for code/law/finance

---

## Dimension Sizes and Trade-offs

### Dimension Comparison

| Dimensions | Storage (float32) | Speed | Quality |
|------------|-------------------|-------|---------|
| 256 | 1 KB | Fastest | Lower |
| 512 | 2 KB | Fast | Good |
| 1024 | 4 KB | Medium | Better |
| 1536 | 6 KB | Slower | High |
| 3072 | 12 KB | Slowest | Highest |

### Matryoshka Embeddings

OpenAI's text-embedding-3 models support variable dimensions:

```python
def get_embedding_flexible(
    text: str, 
    dimensions: int = 1536
) -> list:
    """Get embedding with custom dimensions"""
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",
        dimensions=dimensions  # 256, 512, 1024, 1536, or 3072
    )
    return response.data[0].embedding

# High precision for important queries
important_embedding = get_embedding_flexible("critical search", dimensions=3072)

# Lower dimensions for bulk storage
bulk_embedding = get_embedding_flexible("general document", dimensions=256)
```

### Compression Options

```python
import numpy as np

def compress_embedding(embedding: list, method: str = "float16") -> bytes:
    """Compress embedding for storage"""
    
    arr = np.array(embedding)
    
    if method == "float16":
        # Half precision: 50% size reduction
        return arr.astype(np.float16).tobytes()
    
    elif method == "int8":
        # Quantized: 75% size reduction
        # Scale to -127 to 127 range
        scaled = (arr * 127).astype(np.int8)
        return scaled.tobytes()
    
    elif method == "binary":
        # Binary: 96% size reduction
        # Only store sign bit
        binary = (arr > 0).astype(np.uint8)
        return np.packbits(binary).tobytes()

# Size comparison
embedding = get_embedding("test")
print(f"Float32: {len(embedding) * 4} bytes")
print(f"Float16: {len(compress_embedding(embedding, 'float16'))} bytes")
print(f"Int8: {len(compress_embedding(embedding, 'int8'))} bytes")
print(f"Binary: {len(compress_embedding(embedding, 'binary'))} bytes")
```

---

## Input Types

### Search Optimization

```python
# Cohere uses input_type to optimize embeddings

# For documents being indexed
doc_embedding = co.embed(
    texts=["Machine learning is a subset of AI..."],
    model="embed-english-v3.0",
    input_type="search_document"  # Optimized for being searched
)

# For search queries
query_embedding = co.embed(
    texts=["What is ML?"],
    model="embed-english-v3.0", 
    input_type="search_query"  # Optimized for searching
)
```

### Task-Specific Prefixes

```python
# Some models use prefixes to indicate task

def get_voyage_embedding(text: str, task: str = "document") -> list:
    """Get Voyage embedding with task prefix"""
    
    prefixes = {
        "document": "Represent this document for retrieval: ",
        "query": "Represent this query for retrieval: ",
        "classification": "Classify this text: "
    }
    
    prefixed_text = prefixes[task] + text
    
    response = vo.embed([prefixed_text], model="voyage-3")
    return response.embeddings[0]
```

---

## Specialized Embeddings

### Code Embeddings

```python
# Voyage AI code embedding
code_embedding = vo.embed(
    ["def hello(): return 'world'"],
    model="voyage-code-3"
)

# For code search applications
# Matches semantic meaning, not just text
```

### Multilingual Embeddings

```python
# Cohere multilingual
multilingual = co.embed(
    texts=[
        "Hello, world!",      # English
        "Bonjour le monde!",  # French
        "Hola mundo!",        # Spanish
        "こんにちは世界！"      # Japanese
    ],
    model="embed-multilingual-v3.0",
    input_type="search_document"
)

# All map to similar semantic space
# Cross-lingual search works!
```

### Multimodal Embeddings

```python
# Cohere embed-v4.0: Text + Images in same space

# Text embedding
text_result = co.embed(
    texts=["A photo of a sunset over the ocean"],
    model="embed-v4.0",
    input_type="search_query"
)

# Image embedding
image_result = co.embed(
    images=["base64_encoded_image_data"],
    model="embed-v4.0",
    input_type="image"
)

# Can search images with text queries!
```

---

## Model Selection Criteria

### Decision Framework

```python
def select_embedding_model(requirements: dict) -> str:
    """Select best embedding model"""
    
    if requirements.get("multimodal"):
        return "cohere/embed-v4.0"
    
    if requirements.get("code"):
        return "voyage/voyage-code-3"
    
    if requirements.get("multilingual"):
        return "cohere/embed-multilingual-v3.0"
    
    if requirements.get("budget_constrained"):
        return "openai/text-embedding-3-small"
    
    if requirements.get("max_quality"):
        return "openai/text-embedding-3-large"
    
    # Good default
    return "openai/text-embedding-3-small"
```

### Cost Comparison

| Model | Cost per 1M tokens |
|-------|-------------------|
| text-embedding-3-small | $0.02 |
| text-embedding-3-large | $0.13 |
| Cohere embed-v3 | $0.10 |
| Voyage-3 | $0.06 |

---

## Hands-on Exercise

### Your Task

Build a semantic search demo:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

class SemanticSearch:
    """Simple semantic search with embeddings"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs: list):
        """Add documents to the index"""
        for doc in docs:
            embedding = self._get_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 3) -> list:
        """Search for similar documents"""
        query_embedding = self._get_embedding(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, self.documents[i]))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        return similarities[:top_k]
    
    def _get_embedding(self, text: str) -> list:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test
search = SemanticSearch()

# Add documents
search.add_documents([
    "Python is a programming language known for its simplicity",
    "Machine learning uses algorithms to learn from data",
    "The stock market had a volatile trading session today",
    "Neural networks are inspired by biological brains",
    "The weather forecast predicts rain tomorrow"
])

# Search
results = search.search("How do AI systems learn?")
for score, doc in results:
    print(f"{score:.3f}: {doc}")
```

---

## Summary

✅ **Embeddings** convert text to semantic vectors

✅ **OpenAI text-embedding-3** offers flexible dimensions

✅ **Cohere** specializes in multilingual and multimodal

✅ **Voyage** excels at specialized domains (code, legal)

✅ **Dimension trade-off**: Quality vs storage/speed

✅ **Input types** (query vs document) improve results

**Next:** [Reranking Models](./04-reranking-models.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Code Generation Models](./02-code-generation-models.md) | [Types of AI Models](./00-types-of-ai-models.md) | [Reranking Models](./04-reranking-models.md) |

