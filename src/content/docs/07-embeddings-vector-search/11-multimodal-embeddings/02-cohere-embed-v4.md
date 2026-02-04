---
title: "Cohere embed-v4.0"
---

# Cohere embed-v4.0

## Introduction

Cohere's **embed-v4.0** (released 2025) represents the current state-of-the-art in production multimodal embeddings. It supports text, images, and mixed content in a unified embedding space with support for 100+ languages.

This lesson covers the embed-v4.0 model capabilities, how to use it for multimodal embeddings, and best practices for production deployments.

### What We'll Cover

- embed-v4.0 model capabilities and specifications
- Basic multimodal embedding with Cohere
- Understanding the unified text-image space
- Matryoshka dimension support

### Prerequisites

- [What Are Multimodal Embeddings](./01-what-are-multimodal-embeddings.md)
- Cohere API key
- Python with `cohere` package installed

---

## embed-v4.0 Specifications

### Model Capabilities

| Feature | Details |
|---------|---------|
| **Modalities** | Text, Images, Mixed content |
| **Languages** | 100+ languages supported |
| **Dimensions** | 256, 512, 1024, 1536 (Matryoshka) |
| **Max text** | ~512 tokens |
| **Max images/batch** | 1 per request (API v2) |
| **Image formats** | PNG, JPEG, WebP, GIF |
| **Max image size** | 5MB |

### Comparison with Previous Versions

| Feature | embed-v3.0 | embed-v4.0 |
|---------|------------|------------|
| Text embeddings | ✅ | ✅ |
| Image embeddings | ✅ | ✅ |
| **Mixed content** | ❌ | ✅ |
| **Matryoshka dims** | ❌ | ✅ |
| Languages | 100+ | 100+ |
| MTEB score | ~64% | ~66% |

---

## Getting Started

### Installation

```bash
pip install cohere
```

### Basic Setup

```python
import cohere

# Initialize client (v2 API)
co = cohere.ClientV2(api_key="YOUR_API_KEY")

# Or use environment variable
# export CO_API_KEY="your-api-key"
co = cohere.ClientV2()
```

---

## Text Embeddings

### Basic Text Embedding

```python
import cohere

co = cohere.ClientV2()

# Single text embedding
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    texts=["A beautiful sunset over the ocean"]
)

embedding = response.embeddings.float_[0]
print(f"Dimensions: {len(embedding)}")  # 1536 by default
print(f"First 5 values: {embedding[:5]}")
```

**Output:**
```
Dimensions: 1536
First 5 values: [0.0163, -0.0084, -0.0470, -0.0710, 0.0001]
```

### Batch Text Embeddings

```python
texts = [
    "A beautiful sunset over the ocean",
    "A cat sleeping on a couch",
    "Modern architecture in Tokyo"
]

response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    texts=texts
)

embeddings = response.embeddings.float_
print(f"Generated {len(embeddings)} embeddings")
```

---

## Input Types for Multimodal

Cohere's `input_type` parameter affects how embeddings are optimized:

| Input Type | Use For |
|------------|---------|
| `search_document` | Documents/content to be searched |
| `search_query` | User search queries |
| `classification` | Text classification tasks |
| `clustering` | Clustering/grouping tasks |
| `image` | Image-only embeddings |

> **Important:** For search, always use `search_query` for queries and `search_document` for indexed content.

```python
# Indexing documents (including images)
doc_response = co.embed(
    model="embed-v4.0",
    input_type="search_document",  # For content
    embedding_types=["float"],
    texts=["Product description here"]
)

# Searching
query_response = co.embed(
    model="embed-v4.0",
    input_type="search_query",  # For queries
    embedding_types=["float"],
    texts=["user search query"]
)
```

---

## Matryoshka Dimensions

embed-v4.0 supports Matryoshka representation learning, allowing you to use smaller dimensions without re-embedding:

### Specifying Dimensions

```python
# Get smaller embeddings (faster search, less storage)
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    output_dimension=512,  # Instead of default 1536
    texts=["Sample text"]
)

embedding = response.embeddings.float_[0]
print(f"Dimensions: {len(embedding)}")  # 512
```

### Dimension Trade-offs

| Dimensions | Storage | Search Speed | Quality |
|------------|---------|--------------|---------|
| 256 | 1 KB | Fastest | ~95% of full |
| 512 | 2 KB | Fast | ~97% of full |
| 1024 | 4 KB | Medium | ~99% of full |
| 1536 | 6 KB | Baseline | 100% |

```python
# Compare dimensions
import numpy as np

def embed_with_dims(text: str, dims: int):
    response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=dims,
        texts=[text]
    )
    return response.embeddings.float_[0]

text = "A beautiful sunset over the ocean"

# Get embeddings at different dimensions
emb_256 = embed_with_dims(text, 256)
emb_512 = embed_with_dims(text, 512)
emb_1536 = embed_with_dims(text, 1536)

print(f"256-dim: {len(emb_256)} values")
print(f"512-dim: {len(emb_512)} values")
print(f"1536-dim: {len(emb_1536)} values")
```

---

## Embedding Compression Types

embed-v4.0 supports multiple numeric formats for storage efficiency:

### Available Types

```python
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float", "int8", "ubinary"],  # Multiple types
    texts=["Sample text"]
)

# Access different formats
float_emb = response.embeddings.float_[0]    # Full precision
int8_emb = response.embeddings.int8[0]       # 8-bit integer
binary_emb = response.embeddings.ubinary[0]  # Binary (1-bit per dim)
```

### Compression Comparison

| Type | Bytes/Vector (1536d) | Quality Loss | Use Case |
|------|---------------------|--------------|----------|
| `float` | 6,144 bytes | None | Highest quality |
| `int8` | 1,536 bytes | ~1% | Balanced |
| `uint8` | 1,536 bytes | ~1% | Balanced |
| `binary` | 192 bytes | ~5% | Maximum compression |
| `ubinary` | 192 bytes | ~5% | Maximum compression |

---

## Unified Vector Space Demonstration

### Text and Images in Same Space

The key feature of embed-v4.0 is that text and images share the same embedding space:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed text description
text_response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    texts=["A golden retriever playing in a park"]
)
text_emb = text_response.embeddings.float_[0]

# Embed related image (we'll cover image encoding next lesson)
# For now, assume we have an image embedding
# image_emb = embed_image(golden_retriever_image)

# These will be close in vector space!
# similarity = cosine_similarity(text_emb, image_emb)
# print(f"Text-Image Similarity: {similarity:.4f}")  # ~0.75-0.85
```

### Cross-Modal Search Pattern

```python
class MultimodalSearch:
    """Search across text and images with unified embeddings."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.documents = []
        self.embeddings = []
    
    def add_text(self, text: str, metadata: dict = None):
        """Add text document to index."""
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            texts=[text]
        )
        self.documents.append({"type": "text", "content": text, "metadata": metadata})
        self.embeddings.append(response.embeddings.float_[0])
    
    def add_image(self, image_base64: str, metadata: dict = None):
        """Add image to index (covered in next lesson)."""
        # Image embedding code here
        pass
    
    def search(self, query: str, top_k: int = 5):
        """Search with text query across all content."""
        # Embed query
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_query",  # Note: search_query for queries
            embedding_types=["float"],
            texts=[query]
        )
        query_emb = response.embeddings.float_[0]
        
        # Compute similarities
        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_emb, doc_emb)
            scores.append((i, sim))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {**self.documents[i], "score": score}
            for i, score in scores[:top_k]
        ]
```

---

## Multilingual Support

embed-v4.0 supports 100+ languages in the same embedding space:

```python
# Multilingual text - all in same space
texts = [
    "A beautiful sunset",           # English
    "Un beau coucher de soleil",    # French
    "美しい夕焼け",                   # Japanese
    "Ein wunderschöner Sonnenuntergang",  # German
    "Una hermosa puesta de sol"     # Spanish
]

response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    texts=texts
)

embeddings = response.embeddings.float_

# These should all be similar (same semantic meaning)
for i, text in enumerate(texts):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"'{texts[i][:20]}...' ↔ '{texts[j][:20]}...': {sim:.4f}")
```

**Expected Output:**
```
'A beautiful sunset...' ↔ 'Un beau coucher de s...': 0.9234
'A beautiful sunset...' ↔ '美しい夕焼け...': 0.8876
'A beautiful sunset...' ↔ 'Ein wunderschöner So...': 0.9156
...
```

---

## Best Practices

### 1. Match Input Types

```python
# ✅ CORRECT: search_query for queries, search_document for content
query_emb = embed(query, input_type="search_query")
doc_emb = embed(document, input_type="search_document")

# ❌ WRONG: Same input_type for both
query_emb = embed(query, input_type="search_document")  # Wrong!
```

### 2. Choose Dimensions Based on Scale

```python
# Small dataset (<100K documents): Use full dimensions
output_dimension = 1536

# Medium dataset (100K-1M): Consider 1024
output_dimension = 1024

# Large dataset (>1M): Use 512 or 256
output_dimension = 512
```

### 3. Use Compression for Large Scale

```python
# For very large deployments, use int8 or binary
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["int8"],  # 4x storage reduction
    texts=texts
)
```

### 4. Batch for Efficiency

```python
# ✅ CORRECT: Batch multiple texts
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    texts=["text1", "text2", "text3"]  # Up to 96 texts
)

# ❌ INEFFICIENT: One call per text
for text in texts:
    response = co.embed(texts=[text])  # Slow!
```

---

## Summary

✅ **embed-v4.0** is Cohere's latest multimodal embedding model (2025)  
✅ Supports **text, images, and mixed content** in unified space  
✅ **100+ languages** with cross-lingual similarity  
✅ **Matryoshka dimensions** (256, 512, 1024, 1536)  
✅ **Compression types** for storage efficiency (float, int8, binary)  
✅ Use `search_query` for queries, `search_document` for content

---

**Next:** [Image Embedding with Cohere →](./03-image-embedding-with-cohere.md)

---

<!-- 
Sources Consulted:
- Cohere Embed API Reference: https://docs.cohere.com/reference/embed
- Cohere Multimodal Embeddings Guide: https://docs.cohere.com/docs/multimodal-embeddings
- Cohere Embed documentation: https://docs.cohere.com/docs/cohere-embed
-->
