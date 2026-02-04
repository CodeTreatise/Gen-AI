---
title: "Multimodal Embedding Use Cases"
---

# Multimodal Embedding Use Cases

## Introduction

Multimodal embeddings unlock powerful applications that weren't possible with text-only systems. When images, text, and video share the same vector space, you can build semantic search and retrieval systems that understand content across modalities.

This lesson explores practical use cases with implementation patterns for each.

### What We'll Cover

- Image search with text queries
- E-commerce product search
- Document understanding with figures
- Visual question answering
- Recommendation systems
- Content moderation

### Prerequisites

- [Mixed Content Embeddings](./04-mixed-content-embeddings.md)
- [CLIP and Alternatives](./05-clip-and-alternatives.md)
- Basic understanding of vector search

---

## Use Case 1: Image Search with Text

### The Problem

Users want to find images using natural language descriptions, not keywords:

```
Traditional keyword search:         Multimodal search:
"red car sunset"                   "sports car driving into
  │                                the sunset on a coastal road"
  ▼                                  │
Results: Random images               ▼
with "red", "car", "sunset"        Results: Semantically
tags somewhere                     matching images
```

### Implementation

```python
import cohere
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ImageResult:
    path: str
    score: float
    metadata: dict


class ImageSearchEngine:
    """Search images with natural language queries."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.index = []
    
    def add_image(self, image_path: str, metadata: dict = None):
        """Index an image with optional metadata."""
        data_url = self._to_data_url(image_path)
        
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[data_url]
        )
        
        self.index.append({
            "path": image_path,
            "embedding": np.array(response.embeddings.float_[0]),
            "metadata": metadata or {}
        })
    
    def search(self, query: str, top_k: int = 10) -> list[ImageResult]:
        """Search images with natural language query."""
        # Embed query as text
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query]
        )
        query_emb = np.array(response.embeddings.float_[0])
        
        # Compute similarities
        results = []
        for item in self.index:
            score = self._cosine_similarity(query_emb, item["embedding"])
            results.append(ImageResult(
                path=item["path"],
                score=score,
                metadata=item["metadata"]
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _to_data_url(self, path):
        import base64
        with open(path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode()
        ext = Path(path).suffix.lower()
        mime = {'png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/jpeg')
        return f"data:{mime};base64,{data}"


# Usage
engine = ImageSearchEngine()

# Index photos
for photo in Path("photos").glob("*.jpg"):
    engine.add_image(str(photo))
    print(f"Indexed: {photo.name}")

# Search
results = engine.search("people celebrating at a birthday party")

for r in results[:5]:
    print(f"  {r.path} (score: {r.score:.4f})")
```

### Real-World Examples

| Query | Returns |
|-------|---------|
| "professional headshot with neutral background" | Portrait photos with plain backgrounds |
| "cozy coffee shop interior" | Café images with warm lighting |
| "mountain landscape at golden hour" | Mountain photos during sunrise/sunset |
| "minimalist modern kitchen" | Clean, contemporary kitchen designs |

---

## Use Case 2: E-Commerce Product Search

### The Problem

Shoppers describe what they want in natural language, but products have structured data and images:

```
User: "comfortable summer dress for beach wedding"

Needs to match:
- Product images (visual style)
- Product titles
- Product descriptions
- Occasion tags
- Material properties
```

### Implementation

```python
from dataclasses import dataclass
from typing import Optional
import cohere
import numpy as np


@dataclass
class Product:
    id: str
    title: str
    description: str
    image_path: str
    price: float
    category: str


class ProductSearch:
    """E-commerce product search with multimodal embeddings."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.products = []
    
    def index_product(self, product: Product):
        """Index product with text+image embedding."""
        # Create mixed content embedding
        inputs = [
            {
                "type": "text",
                "text": f"{product.title}. {product.description}"
            },
            {
                "type": "image_url",
                "image_url": {"url": self._to_data_url(product.image_path)}
            }
        ]
        
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            inputs=inputs
        )
        
        self.products.append({
            "product": product,
            "embedding": np.array(response.embeddings.float_[0])
        })
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        top_k: int = 20
    ) -> list[tuple[Product, float]]:
        """Search products with optional filters."""
        # Embed query
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query]
        )
        query_emb = np.array(response.embeddings.float_[0])
        
        # Score and filter
        results = []
        for item in self.products:
            product = item["product"]
            
            # Apply filters
            if category and product.category != category:
                continue
            if max_price and product.price > max_price:
                continue
            
            score = self._cosine_similarity(query_emb, item["embedding"])
            results.append((product, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def find_similar(self, product_id: str, top_k: int = 10):
        """Find visually/semantically similar products."""
        # Find the product
        source = None
        for item in self.products:
            if item["product"].id == product_id:
                source = item
                break
        
        if not source:
            return []
        
        # Find similar
        results = []
        for item in self.products:
            if item["product"].id == product_id:
                continue
            
            score = self._cosine_similarity(
                source["embedding"],
                item["embedding"]
            )
            results.append((item["product"], score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Usage
search = ProductSearch()

# Index catalog
products = [
    Product("SKU001", "Summer Floral Maxi Dress", 
            "Light and breezy dress perfect for warm weather", 
            "images/dress1.jpg", 89.99, "dresses"),
    Product("SKU002", "Beach Wedding Guest Dress",
            "Elegant flowing dress suitable for outdoor ceremonies",
            "images/dress2.jpg", 129.99, "dresses"),
    # ... more products
]

for p in products:
    search.index_product(p)

# Search
results = search.search(
    "comfortable dress for outdoor summer wedding",
    category="dresses",
    max_price=150.0
)

for product, score in results[:5]:
    print(f"{product.title} - ${product.price} (score: {score:.4f})")
```

### Visual Similarity for "Shop the Look"

```python
# User is viewing product SKU001
similar = search.find_similar("SKU001")

print("You might also like:")
for product, score in similar[:4]:
    print(f"  - {product.title}")
```

---

## Use Case 3: Document Understanding

### The Problem

Technical documents, research papers, and presentations contain figures, charts, and diagrams that are integral to understanding:

```
┌─────────────────────────────────────────┐
│        Traditional Document Search       │
├─────────────────────────────────────────┤
│ Query: "neural network architecture"    │
│                                         │
│ ✗ Misses: Documents where the          │
│   architecture is shown in a diagram   │
│   but not described in text            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│       Multimodal Document Search         │
├─────────────────────────────────────────┤
│ Query: "neural network architecture"    │
│                                         │
│ ✓ Finds: Documents with architecture   │
│   diagrams, even with minimal text     │
└─────────────────────────────────────────┘
```

### Implementation

```python
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List
import cohere
import numpy as np


@dataclass
class DocumentChunk:
    doc_id: str
    page_num: int
    text: str
    has_figures: bool
    embedding: np.ndarray


class DocumentSearchEngine:
    """Search documents understanding text AND figures."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.chunks: List[DocumentChunk] = []
    
    def index_pdf(self, pdf_path: str, doc_id: str):
        """Index PDF with mixed content embeddings."""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Check for images
            images = page.get_images()
            has_figures = len(images) > 0
            
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_data = pix.tobytes("png")
            data_url = f"data:image/png;base64,{self._encode_base64(img_data)}"
            
            # Create mixed content input
            inputs = []
            if text.strip():
                inputs.append({"type": "text", "text": text[:2000]})
            inputs.append({"type": "image_url", "image_url": {"url": data_url}})
            
            # Embed
            response = self.co.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=inputs
            )
            
            chunk = DocumentChunk(
                doc_id=doc_id,
                page_num=page_num + 1,
                text=text[:500],  # Preview
                has_figures=has_figures,
                embedding=np.array(response.embeddings.float_[0])
            )
            self.chunks.append(chunk)
            print(f"Indexed: {doc_id} page {page_num + 1}")
        
        doc.close()
    
    def search(self, query: str, top_k: int = 10):
        """Search across all document pages."""
        response = self.co.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query]
        )
        query_emb = np.array(response.embeddings.float_[0])
        
        results = []
        for chunk in self.chunks:
            score = np.dot(query_emb, chunk.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(chunk.embedding)
            )
            results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Usage
engine = DocumentSearchEngine()

# Index research papers
engine.index_pdf("papers/transformer_paper.pdf", "attention-is-all-you-need")
engine.index_pdf("papers/bert_paper.pdf", "bert")
engine.index_pdf("papers/gpt3_paper.pdf", "gpt3")

# Search for architecture diagrams
results = engine.search("transformer encoder decoder architecture diagram")

for chunk, score in results[:5]:
    fig_indicator = "📊" if chunk.has_figures else ""
    print(f"{chunk.doc_id} p.{chunk.page_num} {fig_indicator} (score: {score:.4f})")
    print(f"  Preview: {chunk.text[:100]}...")
```

---

## Use Case 4: Visual Question Answering (VQA) Foundation

### The Problem

Users want to ask questions about images and get answers based on visual content.

### Retrieval for VQA

```python
class VQARetriever:
    """Retrieve relevant images for visual questions."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.knowledge_base = []  # Images with captions
    
    def add_knowledge(self, image_path: str, caption: str, facts: list[str]):
        """Add image with associated facts."""
        # Embed image
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[self._to_data_url(image_path)]
        )
        
        self.knowledge_base.append({
            "image_path": image_path,
            "caption": caption,
            "facts": facts,
            "embedding": np.array(response.embeddings.float_[0])
        })
    
    def retrieve_for_question(self, question: str, image_path: str = None):
        """Find relevant knowledge for a visual question."""
        # If image provided, embed it
        if image_path:
            response = self.co.embed(
                model="embed-v4.0",
                input_type="image",
                embedding_types=["float"],
                images=[self._to_data_url(image_path)]
            )
            query_emb = np.array(response.embeddings.float_[0])
        else:
            # Use question text
            response = self.co.embed(
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
                texts=[question]
            )
            query_emb = np.array(response.embeddings.float_[0])
        
        # Find similar knowledge
        results = []
        for item in self.knowledge_base:
            score = self._cosine_similarity(query_emb, item["embedding"])
            results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]


# Usage: Building a visual knowledge base
retriever = VQARetriever()

# Add knowledge about landmarks
retriever.add_knowledge(
    "landmarks/eiffel.jpg",
    "Eiffel Tower in Paris",
    ["Built in 1889", "324 meters tall", "Located in Paris, France"]
)

retriever.add_knowledge(
    "landmarks/colosseum.jpg",
    "Roman Colosseum",
    ["Built 70-80 AD", "Largest amphitheater", "Located in Rome, Italy"]
)

# When user asks about an image
question = "How tall is this tower?"
user_image = "user_upload.jpg"

relevant = retriever.retrieve_for_question(question, user_image)
for item, score in relevant:
    print(f"Related: {item['caption']} (score: {score:.4f})")
    print(f"  Facts: {item['facts']}")
```

---

## Use Case 5: Recommendation Systems

### Visual Similarity Recommendations

```python
class VisualRecommender:
    """Recommend items based on visual similarity."""
    
    def __init__(self):
        self.co = cohere.ClientV2()
        self.items = []
    
    def add_item(self, item_id: str, image_path: str, metadata: dict):
        """Add item to recommendation pool."""
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[self._to_data_url(image_path)]
        )
        
        self.items.append({
            "id": item_id,
            "embedding": np.array(response.embeddings.float_[0]),
            "metadata": metadata
        })
    
    def get_recommendations(
        self,
        liked_items: list[str],
        disliked_items: list[str] = None,
        top_k: int = 10
    ):
        """Recommend items based on user preferences."""
        # Build preference vector from liked items
        liked_embeddings = []
        for item in self.items:
            if item["id"] in liked_items:
                liked_embeddings.append(item["embedding"])
        
        if not liked_embeddings:
            return []
        
        # Average liked embeddings
        preference_vector = np.mean(liked_embeddings, axis=0)
        
        # Optionally subtract disliked
        if disliked_items:
            disliked_embeddings = [
                item["embedding"]
                for item in self.items
                if item["id"] in disliked_items
            ]
            if disliked_embeddings:
                dislike_vector = np.mean(disliked_embeddings, axis=0)
                preference_vector = preference_vector - 0.3 * dislike_vector
        
        # Normalize
        preference_vector /= np.linalg.norm(preference_vector)
        
        # Find similar items
        results = []
        seen = set(liked_items) | set(disliked_items or [])
        
        for item in self.items:
            if item["id"] in seen:
                continue
            
            score = np.dot(preference_vector, item["embedding"]) / (
                np.linalg.norm(item["embedding"])
            )
            results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(r[0]["id"], r[0]["metadata"], r[1]) for r in results[:top_k]]


# Usage
recommender = VisualRecommender()

# Add furniture items
items = [
    ("chair001", "images/modern_chair.jpg", {"style": "modern", "price": 299}),
    ("chair002", "images/vintage_chair.jpg", {"style": "vintage", "price": 199}),
    ("sofa001", "images/minimalist_sofa.jpg", {"style": "modern", "price": 899}),
    # ...
]

for item_id, path, meta in items:
    recommender.add_item(item_id, path, meta)

# Get recommendations based on user's liked items
recommendations = recommender.get_recommendations(
    liked_items=["chair001", "sofa001"],  # User liked modern items
    disliked_items=["chair002"]  # Didn't like vintage
)

print("Recommended for you:")
for item_id, meta, score in recommendations[:5]:
    print(f"  {item_id}: {meta['style']} - ${meta['price']} (score: {score:.4f})")
```

---

## Use Case 6: Content Moderation

### Detecting Similar Violating Content

```python
class ContentModerator:
    """Detect content similar to known violations."""
    
    def __init__(self, threshold: float = 0.85):
        self.co = cohere.ClientV2()
        self.violation_db = []  # Known bad content
        self.threshold = threshold
    
    def add_violation(self, image_path: str, violation_type: str):
        """Add known violating content to database."""
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[self._to_data_url(image_path)]
        )
        
        self.violation_db.append({
            "embedding": np.array(response.embeddings.float_[0]),
            "type": violation_type
        })
    
    def check_content(self, image_path: str) -> dict:
        """Check if content is similar to known violations."""
        response = self.co.embed(
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"],
            images=[self._to_data_url(image_path)]
        )
        content_emb = np.array(response.embeddings.float_[0])
        
        # Check against all known violations
        max_similarity = 0
        matched_type = None
        
        for violation in self.violation_db:
            sim = self._cosine_similarity(content_emb, violation["embedding"])
            if sim > max_similarity:
                max_similarity = sim
                matched_type = violation["type"]
        
        is_violation = max_similarity >= self.threshold
        
        return {
            "is_violation": is_violation,
            "confidence": max_similarity,
            "violation_type": matched_type if is_violation else None
        }


# Usage
moderator = ContentModerator(threshold=0.80)

# Train with known violations
moderator.add_violation("violations/spam_image_1.jpg", "spam")
moderator.add_violation("violations/spam_image_2.jpg", "spam")
moderator.add_violation("violations/inappropriate_1.jpg", "inappropriate")

# Check user uploads
result = moderator.check_content("user_upload.jpg")

if result["is_violation"]:
    print(f"⚠️ Flagged: {result['violation_type']} "
          f"(confidence: {result['confidence']:.2%})")
else:
    print("✅ Content approved")
```

---

## Summary

✅ **Image Search**: Natural language queries find semantically matching images  
✅ **E-Commerce**: Combined product text + images improve search relevance  
✅ **Document Understanding**: Find information in figures and diagrams  
✅ **VQA Foundation**: Retrieve relevant knowledge for visual questions  
✅ **Recommendations**: Visual similarity drives "more like this" features  
✅ **Content Moderation**: Detect content similar to known violations

---

**Next:** [Implementation Considerations →](./07-implementation-considerations.md)

---

<!-- 
Sources Consulted:
- Cohere Multimodal Embeddings Guide: https://docs.cohere.com/docs/multimodal-embeddings
- Vector Database Best Practices: https://www.pinecone.io/learn/
- E-commerce Search Patterns: Industry knowledge
-->
