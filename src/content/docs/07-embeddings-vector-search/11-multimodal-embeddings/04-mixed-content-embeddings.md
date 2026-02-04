---
title: "Mixed Content Embeddings"
---

# Mixed Content Embeddings

## Introduction

One of embed-v4.0's most powerful features is **mixed content embeddings**—the ability to combine text and images into a single embedding. This eliminates complex preprocessing pipelines for documents that contain both modalities.

This lesson covers how to create mixed content embeddings, practical use cases, and best practices for combining text and images effectively.

### What We'll Cover

- The `inputs` parameter for mixed content
- Combining text and images in one embedding
- Document and slide embedding strategies
- Best practices for mixed content

### Prerequisites

- [Image Embedding with Cohere](./03-image-embedding-with-cohere.md)
- Understanding of base64 encoding

---

## Why Mixed Content Matters

### Traditional Approach (Complex)

```
┌─────────────────────────────────────────────────────┐
│           Traditional Document Processing           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PDF/Slide                                          │
│      │                                              │
│      ├──► OCR ──────► Text Extraction ──► Text Emb │
│      │                                              │
│      ├──► Image Extract ──► Image Processing       │
│      │                          │                  │
│      │                          ▼                  │
│      │                    Image Embedding          │
│      │                                              │
│      └──► Layout Analysis ──► Combine? (Hard!)     │
│                                                     │
│  Result: Multiple embeddings, complex pipelines    │
└─────────────────────────────────────────────────────┘
```

### Mixed Content Approach (Simple)

```
┌─────────────────────────────────────────────────────┐
│         Mixed Content Embedding (embed-v4.0)        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PDF/Slide                                          │
│      │                                              │
│      └──► Screenshot + Caption                      │
│                  │                                  │
│                  ▼                                  │
│           inputs=[                                  │
│             {"type": "text", "text": "caption"},   │
│             {"type": "image_url", "image_url":..}  │
│           ]                                         │
│                  │                                  │
│                  ▼                                  │
│         Single Unified Embedding                    │
│                                                     │
│  Result: One embedding captures everything          │
└─────────────────────────────────────────────────────┘
```

---

## The `inputs` Parameter

### Input Object Types

Mixed content uses the `inputs` parameter instead of separate `texts` and `images`:

```python
inputs = [
    # Text object
    {
        "type": "text",
        "text": "Your text content here"
    },
    
    # Image object
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,..."
        }
    }
]
```

### Complete Structure

| Object Type | Required Fields |
|-------------|-----------------|
| Text | `type: "text"`, `text: str` |
| Image | `type: "image_url"`, `image_url: {url: str}` |

---

## Basic Mixed Content Embedding

### Single Item with Text + Image

```python
import cohere
import base64

co = cohere.ClientV2()

def image_to_data_url(path: str) -> str:
    """Convert image file to data URL."""
    with open(path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{data}"

# Create mixed content input
image_data_url = image_to_data_url("product_photo.jpg")

inputs = [
    {
        "type": "text",
        "text": "Nike Air Max 270 Running Shoe - Lightweight mesh upper with Air cushioning"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": image_data_url
        }
    }
]

# Embed mixed content
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",  # Not "image" - this is mixed!
    embedding_types=["float"],
    inputs=inputs
)

embedding = response.embeddings.float_[0]
print(f"Mixed content embedding: {len(embedding)} dimensions")
```

**Output:**
```
Mixed content embedding: 1536 dimensions
```

> **Important:** For mixed content, use `input_type="search_document"` (or other text types), NOT `"image"`.

---

## Multiple Mixed Content Items

### Embedding Multiple Documents

```python
import cohere

co = cohere.ClientV2()

# Prepare multiple document inputs
documents = [
    {
        "title": "Product A - Red Running Shoes",
        "image_path": "products/shoe_red.jpg"
    },
    {
        "title": "Product B - Blue Basketball Shoes",
        "image_path": "products/shoe_blue.jpg"
    },
    {
        "title": "Product C - White Tennis Shoes",
        "image_path": "products/shoe_white.jpg"
    }
]

embeddings = []

for doc in documents:
    # Create input for this document
    inputs = [
        {"type": "text", "text": doc["title"]},
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(doc["image_path"])}
        }
    ]
    
    response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=inputs
    )
    
    embeddings.append({
        "doc": doc,
        "embedding": response.embeddings.float_[0]
    })
    print(f"Embedded: {doc['title']}")

print(f"\nTotal embeddings: {len(embeddings)}")
```

---

## Document and Slide Embedding

### PDF Page as Mixed Content

```python
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image

def pdf_page_to_mixed_input(pdf_path: str, page_num: int) -> list:
    """
    Extract text and render page image from PDF.
    Returns mixed content input for embedding.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Extract text
    text = page.get_text()
    
    # Render page as image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for quality
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert to data URL
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    base64_data = base64.standard_b64encode(buffer.read()).decode('utf-8')
    data_url = f"data:image/png;base64,{base64_data}"
    
    doc.close()
    
    # Create mixed content input
    inputs = []
    
    if text.strip():
        inputs.append({"type": "text", "text": text[:2000]})  # Limit text
    
    inputs.append({
        "type": "image_url",
        "image_url": {"url": data_url}
    })
    
    return inputs


# Embed a PDF page
inputs = pdf_page_to_mixed_input("presentation.pdf", page_num=0)

response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    inputs=inputs
)

print(f"PDF page embedded: {len(response.embeddings.float_[0])} dims")
```

### Presentation Slide Pipeline

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SlideEmbedding:
    """Embedded slide with metadata."""
    slide_number: int
    text_content: str
    embedding: list[float]
    source_file: str


class SlideEmbedder:
    """Embed presentation slides with mixed content."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
    
    def embed_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None
    ) -> list[SlideEmbedding]:
        """Embed all pages of a PDF as mixed content."""
        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), max_pages) if max_pages else len(doc)
        
        embeddings = []
        
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1}/{num_pages}")
            
            # Get mixed input for this page
            inputs = self._page_to_inputs(doc[page_num])
            
            # Embed
            response = self.co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=inputs
            )
            
            # Store result
            text_content = doc[page_num].get_text()[:500]
            embeddings.append(SlideEmbedding(
                slide_number=page_num + 1,
                text_content=text_content,
                embedding=response.embeddings.float_[0],
                source_file=pdf_path
            ))
        
        doc.close()
        return embeddings
    
    def _page_to_inputs(self, page) -> list:
        """Convert PDF page to mixed content inputs."""
        # Extract text
        text = page.get_text()[:2000]
        
        # Render image
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = pix.tobytes("png")
        data_url = f"data:image/png;base64,{base64.standard_b64encode(img_data).decode()}"
        
        inputs = []
        if text.strip():
            inputs.append({"type": "text", "text": text})
        inputs.append({"type": "image_url", "image_url": {"url": data_url}})
        
        return inputs


# Usage
embedder = SlideEmbedder()
slides = embedder.embed_pdf("quarterly_report.pdf", max_pages=20)

print(f"Embedded {len(slides)} slides")
```

---

## Search with Mixed Content

### Text Query to Mixed Documents

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class MixedContentSearch:
    """Search across documents with text+image content."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.documents = []
    
    def add_document(
        self,
        text: str,
        image_path: str,
        metadata: dict = None
    ):
        """Add a document with text and image."""
        # Create mixed input
        inputs = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)}
            }
        ]
        
        # Embed
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            inputs=inputs
        )
        
        self.documents.append({
            "text": text,
            "image_path": image_path,
            "metadata": metadata or {},
            "embedding": response.embeddings.float_[0]
        })
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search with text query."""
        # Embed query (text only)
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query]
        )
        query_emb = response.embeddings.float_[0]
        
        # Score all documents
        results = []
        for doc in self.documents:
            score = cosine_similarity(query_emb, doc["embedding"])
            results.append({
                "text": doc["text"],
                "image_path": doc["image_path"],
                "metadata": doc["metadata"],
                "score": score
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Example usage
search = MixedContentSearch()

# Add product catalog
search.add_document(
    text="Nike Air Max 270 - Lightweight running shoe with Air cushioning",
    image_path="products/nike_air_max.jpg",
    metadata={"category": "running", "brand": "Nike"}
)

search.add_document(
    text="Adidas Ultraboost - Premium running shoe with Boost technology",
    image_path="products/adidas_ultraboost.jpg",
    metadata={"category": "running", "brand": "Adidas"}
)

# Search
results = search.search("comfortable shoes for marathon training")

for i, r in enumerate(results):
    print(f"{i+1}. {r['text'][:50]}... (score: {r['score']:.4f})")
```

---

## Advanced: Weighted Text and Image

### Controlling Modality Influence

When text and image contribute differently, you can adjust by:

```python
def embed_with_emphasis(
    text: str,
    image_path: str,
    text_weight: float = 1.0
) -> list[float]:
    """
    Embed with controllable text emphasis.
    
    Args:
        text: Text description
        image_path: Path to image
        text_weight: How much to emphasize text (0.5-2.0 typical)
    """
    # Adjust text by repetition (simple heuristic)
    if text_weight > 1.0:
        # Repeat key parts of text
        enhanced_text = f"{text} {text}"
    elif text_weight < 1.0:
        # Truncate text
        words = text.split()
        enhanced_text = " ".join(words[:int(len(words) * text_weight)])
    else:
        enhanced_text = text
    
    inputs = [
        {"type": "text", "text": enhanced_text},
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(image_path)}
        }
    ]
    
    response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=inputs
    )
    
    return response.embeddings.float_[0]


# Image-focused embedding (less text influence)
image_focused_emb = embed_with_emphasis(
    "Red shoes",
    "product.jpg",
    text_weight=0.5
)

# Text-focused embedding (more text influence)
text_focused_emb = embed_with_emphasis(
    "Premium Nike Air Max 270 running shoes with excellent cushioning",
    "product.jpg",
    text_weight=1.5
)
```

---

## Best Practices

### 1. Order Matters (Sometimes)

```python
# Text first tends to give slightly more weight to text
inputs = [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {...}}
]

# Image first may slightly emphasize visual features
inputs = [
    {"type": "image_url", "image_url": {...}},
    {"type": "text", "text": "..."}
]
```

### 2. Keep Text Concise

```python
# ✅ Good: Focused text
inputs = [
    {"type": "text", "text": "Red Nike running shoe, Air cushioning"},
    {"type": "image_url", "image_url": {...}}
]

# ❌ Avoid: Long verbose text
inputs = [
    {"type": "text", "text": "This is a very long description that goes on and on about every possible detail of the product which may dilute the visual signal from the image..."},
    {"type": "image_url", "image_url": {...}}
]
```

### 3. Match Query Style

```python
# When indexing mixed content, use search_document
doc_response = co.embed(
    input_type="search_document",
    inputs=[text_obj, image_obj]
)

# When searching, match the input type to query type:
# - Text query? Use search_query with texts parameter
text_query_response = co.embed(
    input_type="search_query",
    texts=["find red shoes"]
)

# - Image query? Use image input type with images parameter
image_query_response = co.embed(
    input_type="image",
    images=[query_image_data_url]
)
```

### 4. Preprocess Images Consistently

```python
def standardize_for_mixed_content(image_path: str) -> str:
    """Consistent image preprocessing for mixed embeddings."""
    img = Image.open(image_path)
    
    # Standardize size
    max_dim = 768
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS
        )
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as JPEG
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    base64_data = base64.standard_b64encode(buffer.read()).decode()
    return f"data:image/jpeg;base64,{base64_data}"
```

---

## Common Pitfalls

### Pitfall 1: Using Wrong Input Type

```python
# ❌ Wrong: input_type="image" for mixed content
response = co.embed(
    input_type="image",  # Wrong!
    inputs=[text_obj, image_obj]
)

# ✅ Correct: Use search_document for mixed
response = co.embed(
    input_type="search_document",
    inputs=[text_obj, image_obj]
)
```

### Pitfall 2: Mixing Parameters

```python
# ❌ Wrong: Can't use both inputs and texts/images
response = co.embed(
    inputs=[...],
    texts=["also this"]  # Error!
)

# ✅ Correct: Use only inputs for mixed content
response = co.embed(
    inputs=[
        {"type": "text", "text": "also this"},
        {"type": "image_url", ...}
    ]
)
```

### Pitfall 3: Oversized Payloads

```python
# ❌ Risk: Large base64 strings in logs/errors
print(inputs)  # May print megabytes of base64!

# ✅ Better: Log metadata only
print(f"Inputs: {len(inputs)} items, types: {[i['type'] for i in inputs]}")
```

---

## Summary

✅ **`inputs` parameter** combines text and images in one embedding  
✅ Text uses `{"type": "text", "text": "..."}` format  
✅ Images use `{"type": "image_url", "image_url": {"url": "data:..."}}`  
✅ Use `input_type="search_document"` for mixed content (not "image")  
✅ Great for **slides, PDFs, product catalogs** with visuals  
✅ Single embedding captures **both textual and visual** information

---

**Next:** [CLIP and Alternatives →](./05-clip-and-alternatives.md)

---

<!-- 
Sources Consulted:
- Cohere Embed API Reference: https://docs.cohere.com/reference/embed
- Cohere Multimodal Embeddings Guide: https://docs.cohere.com/docs/multimodal-embeddings
- PyMuPDF Documentation: https://pymupdf.readthedocs.io/
-->
