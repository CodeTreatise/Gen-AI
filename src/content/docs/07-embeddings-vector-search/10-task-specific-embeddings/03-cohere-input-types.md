---
title: "Cohere Input Types"
---

# Cohere Input Types

## Introduction

Cohere's embedding models use the `input_type` parameter to optimize embeddings for specific use cases. With `embed-v4.0`, Cohere offers a streamlined set of input types that cover the most common embedding scenarios, including **multimodal support** for images.

### What We'll Cover

- All Cohere input types with examples
- Text and image embedding patterns
- Mixed content embeddings (new in v4)
- Practical implementation guide

### Prerequisites

- [Why Task Type Matters](./01-why-task-type-matters.md)
- Cohere API access and configuration

---

## Input Type Reference

| Input Type | Description | Use Case |
|------------|-------------|----------|
| `search_query` | Optimize for search queries | User search input |
| `search_document` | Optimize for indexed documents | Document indexing |
| `classification` | Text categorization | Sentiment, spam detection |
| `clustering` | Group similar items | Topic modeling |
| `image` | Image embeddings (v3 only) | Legacy image embedding |

> **Note:** For `embed-v4.0`, use `search_document` when working with images. The `image` type is primarily for `embed-v3.0` models.

---

## Search: Query and Document

### search_query

Use for **user search queries**. Optimized to find matching documents.

```python
import cohere

co = cohere.ClientV2(api_key="your-api-key")

# User's search query
query = "How do neural networks learn?"

response = co.embed(
    texts=[query],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]
)

query_embedding = response.embeddings.float_[0]
print(f"Query embedding: {len(query_embedding)} dimensions")
```

### search_document

Use for **documents being indexed** for later search.

```python
# Documents to index
documents = [
    "Neural networks learn through a process called backpropagation, "
    "where errors are propagated backwards through the network to update weights.",
    
    "Deep learning is a subset of machine learning that uses multiple layers "
    "of neural networks to progressively extract features from raw input.",
    
    "Gradient descent is an optimization algorithm used to minimize the loss "
    "function by iteratively moving toward the steepest descent."
]

response = co.embed(
    texts=documents,
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)

doc_embeddings = response.embeddings.float_
print(f"Indexed {len(doc_embeddings)} documents")
```

### Complete Search Example

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_cohere(query: str, documents: list[str], top_k: int = 3):
    """Search using Cohere asymmetric embeddings."""
    
    # Embed query
    query_response = co.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_emb = np.array(query_response.embeddings.float_[0])
    
    # Embed documents
    doc_response = co.embed(
        texts=documents,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    doc_embs = [np.array(e) for e in doc_response.embeddings.float_]
    
    # Compute similarities
    results = []
    for doc, emb in zip(documents, doc_embs):
        similarity = cosine_similarity(query_emb, emb)
        results.append((doc, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example
query = "how do AI models improve over time"
results = search_cohere(query, documents)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"  {doc[:80]}...")
    print()
```

---

## Classification and Clustering

### classification

Optimized for **separating texts into categories**.

```python
# Training examples for sentiment classifier
training_texts = [
    "I absolutely love this product!",
    "Best purchase I've ever made!",
    "This is terrible, complete waste of money.",
    "Worst experience ever, avoid at all costs.",
    "It's okay, nothing special.",
    "Decent quality for the price."
]
training_labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

# Embed with classification type
response = co.embed(
    texts=training_texts,
    model="embed-v4.0",
    input_type="classification",
    embedding_types=["float"]
)

training_embeddings = response.embeddings.float_

# Train a simple classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training_embeddings, training_labels)

# Classify new text
new_text = "This exceeded all my expectations!"
new_response = co.embed(
    texts=[new_text],
    model="embed-v4.0",
    input_type="classification",
    embedding_types=["float"]
)

prediction = classifier.predict([new_response.embeddings.float_[0]])
print(f"Text: {new_text}")
print(f"Predicted sentiment: {prediction[0]}")  # positive
```

### clustering

Optimized for **grouping similar items together**.

```python
from sklearn.cluster import KMeans

# Documents to cluster
documents = [
    # Technology
    "Python programming best practices",
    "Machine learning model deployment",
    "Web development with React",
    # Health
    "Benefits of regular exercise",
    "Healthy eating habits",
    "Mental health awareness",
    # Finance
    "Stock market investing basics",
    "Personal budgeting tips",
    "Cryptocurrency trading strategies"
]

# Embed with clustering type
response = co.embed(
    texts=documents,
    model="embed-v4.0",
    input_type="clustering",
    embedding_types=["float"]
)

embeddings = response.embeddings.float_

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Display clusters
for cluster_id in range(3):
    print(f"\n📁 Cluster {cluster_id}:")
    for doc, c in zip(documents, clusters):
        if c == cluster_id:
            print(f"  • {doc}")
```

**Output:**
```
📁 Cluster 0:
  • Python programming best practices
  • Machine learning model deployment
  • Web development with React

📁 Cluster 1:
  • Benefits of regular exercise
  • Healthy eating habits
  • Mental health awareness

📁 Cluster 2:
  • Stock market investing basics
  • Personal budgeting tips
  • Cryptocurrency trading strategies
```

---

## Image Embeddings

Cohere supports image embeddings with `embed-v4.0`. Images can be embedded alongside text for multimodal search.

### Basic Image Embedding

```python
import base64
from PIL import Image
from io import BytesIO

def image_to_base64_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format=img.format or "PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Determine MIME type
        format_lower = (img.format or "PNG").lower()
        mime_type = f"image/{format_lower}"
        
        return f"data:{mime_type};base64,{img_base64}"

# Embed an image using the inputs parameter (recommended for v4)
image_url = image_to_base64_url("product_photo.jpg")

input_item = {
    "content": [
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
}

response = co.embed(
    model="embed-v4.0",
    inputs=[input_item],
    input_type="search_document",
    embedding_types=["float"],
    output_dimension=1024
)

image_embedding = response.embeddings.float_[0]
print(f"Image embedding: {len(image_embedding)} dimensions")
```

### Multimodal Search: Text Query → Image Results

```python
# Index product images
product_images = [
    {"id": "prod_1", "path": "red_shoes.jpg", "name": "Red Running Shoes"},
    {"id": "prod_2", "path": "blue_jacket.jpg", "name": "Blue Winter Jacket"},
    {"id": "prod_3", "path": "black_watch.jpg", "name": "Black Smart Watch"},
]

# Embed all product images
for product in product_images:
    image_url = image_to_base64_url(product["path"])
    input_item = {
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
    
    response = co.embed(
        model="embed-v4.0",
        inputs=[input_item],
        input_type="search_document",
        embedding_types=["float"]
    )
    product["embedding"] = np.array(response.embeddings.float_[0])

# Search with text query
query = "athletic footwear in red color"

query_response = co.embed(
    texts=[query],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]
)
query_emb = np.array(query_response.embeddings.float_[0])

# Find matching products
results = []
for product in product_images:
    similarity = cosine_similarity(query_emb, product["embedding"])
    results.append((product["name"], similarity))

results.sort(key=lambda x: x[1], reverse=True)
for name, score in results:
    print(f"{score:.4f}: {name}")
```

---

## Mixed Content Embeddings (v4)

`embed-v4.0` can embed **text and images together** into a single vector.

```python
# Embed product with both image and description
image_url = image_to_base64_url("laptop.jpg")

# Fuse text and image
fused_input = {
    "content": [
        {"type": "text", "text": "15-inch laptop with touchscreen, 16GB RAM"},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
}

response = co.embed(
    model="embed-v4.0",
    inputs=[fused_input],
    input_type="search_document",
    embedding_types=["float"],
    output_dimension=1024
)

# Single embedding capturing both text and image
fused_embedding = response.embeddings.float_[0]
print(f"Fused embedding: {len(fused_embedding)} dimensions")
```

### Use Case: Rich Product Search

```python
# Index products with both descriptions and images
products = [
    {
        "id": "laptop_1",
        "description": "Gaming laptop with RTX 4080, 32GB RAM, 1TB SSD",
        "image_path": "gaming_laptop.jpg"
    },
    {
        "id": "laptop_2", 
        "description": "Ultrabook for professionals, lightweight, all-day battery",
        "image_path": "ultrabook.jpg"
    }
]

for product in products:
    image_url = image_to_base64_url(product["image_path"])
    
    fused_input = {
        "content": [
            {"type": "text", "text": product["description"]},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
    
    response = co.embed(
        model="embed-v4.0",
        inputs=[fused_input],
        input_type="search_document",
        embedding_types=["float"]
    )
    product["embedding"] = response.embeddings.float_[0]

# Search finds products matching BOTH text and visual characteristics
```

---

## Embedding Types (Compression)

Cohere supports multiple output formats for storage optimization:

```python
# Get multiple embedding types in one call
response = co.embed(
    texts=["Sample text for embedding"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float", "int8", "ubinary"]
)

# Access different formats
float_emb = response.embeddings.float_[0]   # Full precision
int8_emb = response.embeddings.int8[0]      # 8-bit quantized
binary_emb = response.embeddings.ubinary[0] # Binary (1-bit)

print(f"Float: {len(float_emb)} floats")
print(f"Int8: {len(int8_emb)} int8 values")
print(f"Binary: {len(binary_emb)} bytes (packed bits)")
```

| Type | Storage per 1024 dims | Retrieval Quality |
|------|----------------------|-------------------|
| `float` | 4,096 bytes | 100% (baseline) |
| `int8` | 1,024 bytes | ~99% |
| `ubinary` | 128 bytes | ~95% |

---

## Matryoshka Support

`embed-v4.0` supports flexible dimensions with the `output_dimension` parameter:

```python
# Generate smaller embeddings
response = co.embed(
    texts=["Sample text"],
    model="embed-v4.0",
    input_type="search_document",
    output_dimension=512,  # Instead of default 1024
    embedding_types=["float"]
)

embedding = response.embeddings.float_[0]
print(f"Dimension: {len(embedding)}")  # 512
```

**Supported dimensions:** 256, 512, 1024, 1536

---

## Complete Example: Multilingual Search

Cohere's models support 100+ languages:

```python
# Multilingual documents
documents = [
    "Machine learning is transforming industries.",      # English
    "L'apprentissage automatique transforme les industries.",  # French
    "Das maschinelle Lernen verändert Branchen.",       # German
    "机器学习正在改变各行各业。",                           # Chinese
    "El aprendizaje automático está transformando industrias."  # Spanish
]

# Index with search_document
doc_response = co.embed(
    texts=documents,
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)
doc_embs = [np.array(e) for e in doc_response.embeddings.float_]

# Search in any language
queries = [
    "How is AI changing business?",           # English
    "Comment l'IA change-t-elle les affaires?",  # French  
    "AI如何改变商业？"                           # Chinese
]

for query in queries:
    query_response = co.embed(
        texts=[query],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_emb = np.array(query_response.embeddings.float_[0])
    
    # All documents should score similarly regardless of language
    print(f"\nQuery: {query}")
    for doc, emb in zip(documents, doc_embs):
        sim = cosine_similarity(query_emb, emb)
        print(f"  {sim:.4f}: {doc[:40]}...")
```

---

## Summary

✅ **Four main input types**: search_query, search_document, classification, clustering  
✅ **Asymmetric search**: Use search_query for queries, search_document for documents  
✅ **Image embeddings**: Use inputs parameter with search_document type  
✅ **Mixed content**: Fuse text and images into single embeddings  
✅ **Multilingual**: 100+ languages supported  
✅ **Compression**: float, int8, and binary output types available

---

**Next:** [Voyage Input Types →](./04-voyage-input-types.md)

---

<!-- 
Sources Consulted:
- Cohere Embeddings documentation: https://docs.cohere.com/docs/embeddings
- Cohere embed-v4.0 features (multimodal, mixed content)
-->
