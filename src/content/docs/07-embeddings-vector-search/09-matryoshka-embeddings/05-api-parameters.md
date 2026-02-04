---
title: "API Parameters for Dimension Control"
---

# API Parameters for Dimension Control

## Introduction

Each embedding provider implements dimension reduction differently. Some return pre-truncated embeddings from the API, while others require client-side processing. Understanding these differences is crucial for correct implementation.

This lesson covers the exact parameters, code examples, and gotchas for each major provider.

### What We'll Cover

- Provider-specific parameters and syntax
- Complete code examples for each API
- Common mistakes and how to avoid them
- Building a unified wrapper for multiple providers

### Prerequisites

- API keys for providers you want to use
- Understanding of [Supported Models](./03-supported-models.md)

---

## OpenAI: `dimensions` Parameter

OpenAI's `text-embedding-3-*` models accept a `dimensions` parameter directly in the API call.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embedding(text: str, dimensions: int = 3072) -> list[float]:
    """
    Get embedding from OpenAI with specified dimensions.
    
    Args:
        text: Input text to embed
        dimensions: Output dimension (max 3072 for large, 1536 for small)
    
    Returns:
        List of floats representing the embedding
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dimensions  # API returns truncated embedding
    )
    return response.data[0].embedding

# Examples at different dimensions
emb_full = get_openai_embedding("What is machine learning?", dimensions=3072)
emb_half = get_openai_embedding("What is machine learning?", dimensions=1536)
emb_quarter = get_openai_embedding("What is machine learning?", dimensions=768)

print(f"Full: {len(emb_full)} dims")      # 3072
print(f"Half: {len(emb_half)} dims")      # 1536
print(f"Quarter: {len(emb_quarter)} dims") # 768
```

### Batch Processing

```python
def get_openai_embeddings_batch(texts: list[str], dimensions: int = 1024) -> list[list[float]]:
    """Get embeddings for multiple texts in a single API call."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=dimensions
    )
    # Sort by index to maintain order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

texts = [
    "First document about AI",
    "Second document about ML",
    "Third document about deep learning"
]

embeddings = get_openai_embeddings_batch(texts, dimensions=512)
print(f"Got {len(embeddings)} embeddings of {len(embeddings[0])} dimensions")
```

### Key Details

| Aspect | Value |
|--------|-------|
| Parameter name | `dimensions` |
| Type | Integer |
| Max for `text-embedding-3-large` | 3072 |
| Max for `text-embedding-3-small` | 1536 |
| Default | Model's maximum |
| Pre-normalized | ✅ Yes |

> **Important:** OpenAI embeddings are already normalized to unit length. No post-processing required.

---

## Google Gemini: `output_dimensionality` Parameter

Gemini uses `output_dimensionality` along with optional `task_type` for optimization.

### Basic Usage

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

def get_gemini_embedding(
    text: str, 
    dimensions: int = 768,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """
    Get embedding from Gemini with specified dimensions.
    
    Args:
        text: Input text to embed
        dimensions: Output dimension (max 3072)
        task_type: Optimization hint for the model
    
    Returns:
        List of floats (NOT normalized—must normalize after!)
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        output_dimensionality=dimensions,
        task_type=task_type
    )
    return result['embedding']

# Get embedding for a query
query_emb = get_gemini_embedding(
    "What is machine learning?",
    dimensions=768,
    task_type="RETRIEVAL_QUERY"
)

# Get embedding for a document
doc_emb = get_gemini_embedding(
    "Machine learning is a subset of AI that enables systems to learn...",
    dimensions=768,
    task_type="RETRIEVAL_DOCUMENT"
)

print(f"Query embedding: {len(query_emb)} dimensions")
print(f"Document embedding: {len(doc_emb)} dimensions")
```

### Task Types

Gemini optimizes embeddings based on intended use:

| Task Type | Use For |
|-----------|---------|
| `RETRIEVAL_QUERY` | Search queries |
| `RETRIEVAL_DOCUMENT` | Documents in search index |
| `SEMANTIC_SIMILARITY` | Sentence/paragraph comparison |
| `CLASSIFICATION` | Text classification features |
| `CLUSTERING` | Unsupervised grouping |

```python
# Matching task types for retrieval
def search_with_gemini(query: str, documents: list[str], top_k: int = 5):
    """Search documents using Gemini embeddings with matching task types."""
    import numpy as np
    
    # Embed query with RETRIEVAL_QUERY
    q_emb = get_gemini_embedding(query, dimensions=768, task_type="RETRIEVAL_QUERY")
    q_emb = np.array(q_emb)
    q_emb = q_emb / np.linalg.norm(q_emb)  # MUST normalize!
    
    # Embed documents with RETRIEVAL_DOCUMENT
    scores = []
    for i, doc in enumerate(documents):
        d_emb = get_gemini_embedding(doc, dimensions=768, task_type="RETRIEVAL_DOCUMENT")
        d_emb = np.array(d_emb)
        d_emb = d_emb / np.linalg.norm(d_emb)  # MUST normalize!
        
        score = np.dot(q_emb, d_emb)
        scores.append((i, score, doc))
    
    # Return top-k results
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

### Key Details

| Aspect | Value |
|--------|-------|
| Parameter name | `output_dimensionality` |
| Type | Integer |
| Max dimensions | 3072 |
| Default | 768 |
| Recommended values | 768, 1536, 3072 |
| Pre-normalized | ❌ **No** |

> ⚠️ **Critical:** Gemini embeddings are NOT normalized. You MUST normalize them before computing cosine similarity!

---

## Cohere: `output_dimension` Parameter

Cohere's `embed-v4.0` uses `output_dimension` with `input_type` for optimization.

### Basic Usage

```python
import cohere

co = cohere.Client("your-api-key")

def get_cohere_embedding(
    text: str,
    dimensions: int = 1024,
    input_type: str = "search_document"
) -> list[float]:
    """
    Get embedding from Cohere with specified dimensions.
    
    Args:
        text: Input text to embed
        dimensions: Output dimension (256, 512, or 1024)
        input_type: "search_query" or "search_document"
    
    Returns:
        List of floats representing the embedding
    """
    response = co.embed(
        texts=[text],
        model="embed-v4.0",
        input_type=input_type,
        output_dimension=dimensions
    )
    return response.embeddings[0]

# Embed a query
query_emb = get_cohere_embedding(
    "What is machine learning?",
    dimensions=512,
    input_type="search_query"
)

# Embed a document
doc_emb = get_cohere_embedding(
    "Machine learning enables computers to learn from data...",
    dimensions=512,
    input_type="search_document"
)
```

### Batch Processing

```python
def get_cohere_embeddings_batch(
    texts: list[str],
    dimensions: int = 512,
    input_type: str = "search_document"
) -> list[list[float]]:
    """Get embeddings for multiple texts efficiently."""
    response = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type=input_type,
        output_dimension=dimensions
    )
    return response.embeddings

documents = [
    "First document about AI",
    "Second document about ML",
    "Third document about deep learning"
]

embeddings = get_cohere_embeddings_batch(documents, dimensions=256)
```

### Key Details

| Aspect | Value |
|--------|-------|
| Parameter name | `output_dimension` |
| Type | Integer |
| Supported values | 256, 512, 1024 |
| Default | 1024 |
| Pre-normalized | ✅ Yes |

---

## Open-Source: Sentence Transformers

For open-source models, use `truncate_dim` at model load or manual truncation.

### Method 1: Load with Truncation

```python
from sentence_transformers import SentenceTransformer

# Load model with built-in truncation
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    truncate_dim=256  # All outputs will be 256 dims
)

# Encode normally—already truncated
embeddings = model.encode([
    "search_query: What is machine learning?",
    "search_document: ML is a subset of AI..."
])

print(embeddings.shape)  # (2, 256)
```

### Method 2: Manual Truncation

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load full model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Get full embeddings
full_embeddings = model.encode([
    "search_query: What is machine learning?",
    "search_document: ML is a subset of AI..."
])

# Manually truncate to different sizes
def truncate_and_normalize(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings and re-normalize."""
    truncated = embeddings[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / norms

emb_512 = truncate_and_normalize(full_embeddings, 512)
emb_256 = truncate_and_normalize(full_embeddings, 256)
emb_128 = truncate_and_normalize(full_embeddings, 128)

print(f"512-dim shape: {emb_512.shape}")  # (2, 512)
print(f"256-dim shape: {emb_256.shape}")  # (2, 256)
print(f"128-dim shape: {emb_128.shape}")  # (2, 128)
```

### Jina Embeddings with Task Prompts

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    truncate_dim=512
)

# Task-specific encoding
query_embeddings = model.encode(
    ["What is machine learning?"],
    task="retrieval.query"
)

document_embeddings = model.encode(
    ["Machine learning is a field of AI..."],
    task="retrieval.passage"
)
```

---

## Unified Wrapper

Build a single interface for multiple providers:

```python
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: list[str], dimensions: int) -> np.ndarray:
        """Generate embeddings for texts at specified dimensions."""
        pass

class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.max_dims = 3072 if "large" in model else 1536
    
    def embed(self, texts: list[str], dimensions: int) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=min(dimensions, self.max_dims)
        )
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return np.array(embeddings)

class GeminiEmbeddings(EmbeddingProvider):
    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        import google.generativeai as genai
        self.genai = genai
        self.task_type = task_type
    
    def embed(self, texts: list[str], dimensions: int) -> np.ndarray:
        embeddings = []
        for text in texts:
            result = self.genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                output_dimensionality=dimensions,
                task_type=self.task_type
            )
            embeddings.append(result['embedding'])
        
        # Gemini requires normalization
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.max_dims = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str], dimensions: int) -> np.ndarray:
        embeddings = self.model.encode(texts)
        
        if dimensions < self.max_dims:
            embeddings = embeddings[:, :dimensions]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return embeddings

# Factory function
def get_embedder(provider: str, **kwargs) -> EmbeddingProvider:
    """Get embedding provider by name."""
    providers = {
        "openai": OpenAIEmbeddings,
        "gemini": GeminiEmbeddings,
        "sentence-transformers": SentenceTransformerEmbeddings,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](**kwargs)

# Usage
embedder = get_embedder("openai")
embeddings = embedder.embed(["Hello world"], dimensions=256)
```

---

## Parameter Quick Reference

| Provider | Model | Parameter | Type | Values | Normalized |
|----------|-------|-----------|------|--------|------------|
| OpenAI | text-embedding-3-large | `dimensions` | int | 1-3072 | ✅ Yes |
| OpenAI | text-embedding-3-small | `dimensions` | int | 1-1536 | ✅ Yes |
| Google | gemini-embedding-001 | `output_dimensionality` | int | 1-3072 | ❌ No |
| Cohere | embed-v4.0 | `output_dimension` | int | 256, 512, 1024 | ✅ Yes |
| Nomic | nomic-embed-text-v1.5 | `truncate_dim` | int | 1-768 | ⚠️ Manual |
| Jina | jina-embeddings-v3 | `truncate_dim` | int | 1-1024 | ⚠️ Manual |

---

## Common Mistakes

### ❌ Wrong Parameter Name

```python
# WRONG: Different providers use different parameter names
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello",
    output_dimensionality=256  # This is Gemini's parameter!
)

# RIGHT: Use 'dimensions' for OpenAI
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello",
    dimensions=256
)
```

### ❌ Forgetting Task Type Matching

```python
# WRONG: Mismatched task types hurt quality
query_emb = get_gemini_embedding(query, task_type="RETRIEVAL_DOCUMENT")  # Wrong!
doc_emb = get_gemini_embedding(doc, task_type="RETRIEVAL_QUERY")        # Wrong!

# RIGHT: Match task types correctly
query_emb = get_gemini_embedding(query, task_type="RETRIEVAL_QUERY")
doc_emb = get_gemini_embedding(doc, task_type="RETRIEVAL_DOCUMENT")
```

### ❌ Forgetting Gemini Normalization

```python
# WRONG: Using Gemini embeddings without normalization
similarity = np.dot(gemini_emb_a, gemini_emb_b)  # Incorrect!

# RIGHT: Normalize first
gemini_emb_a = gemini_emb_a / np.linalg.norm(gemini_emb_a)
gemini_emb_b = gemini_emb_b / np.linalg.norm(gemini_emb_b)
similarity = np.dot(gemini_emb_a, gemini_emb_b)  # Correct
```

---

## Summary

✅ **OpenAI** uses `dimensions` parameter (embeddings pre-normalized)  
✅ **Gemini** uses `output_dimensionality` with `task_type` (MUST normalize after)  
✅ **Cohere** uses `output_dimension` with `input_type` (embeddings pre-normalized)  
✅ **Sentence Transformers** use `truncate_dim` or manual truncation  
✅ **Always match** query/document task types for retrieval  
✅ **Build unified wrappers** to abstract provider differences

---

**Next:** [Normalization After Truncation →](./06-normalization.md)

---

<!-- 
Sources Consulted:
- OpenAI Embeddings API: https://platform.openai.com/docs/api-reference/embeddings
- Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Cohere Embed API: https://docs.cohere.com/reference/embed
- Sentence Transformers documentation
-->
