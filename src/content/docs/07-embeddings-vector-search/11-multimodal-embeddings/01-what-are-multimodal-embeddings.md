---
title: "What Are Multimodal Embeddings?"
---

# What Are Multimodal Embeddings?

## Introduction

Traditional embeddings work with a single modality—text goes in, text embeddings come out. **Multimodal embeddings** break this barrier by projecting multiple data types—text, images, video, audio—into the **same vector space**.

This means a picture of a cat and the text "a fluffy orange cat" will have embeddings that are **close together** in vector space, enabling entirely new search and retrieval patterns.

### What We'll Cover

- The unified vector space concept
- How cross-modal search works
- Training approaches for multimodal models
- Why this matters for RAG and search

### Prerequisites

- [Understanding Embeddings](../01-understanding-embeddings/)
- Basic understanding of neural networks

---

## The Unified Vector Space Concept

### Traditional Embeddings: Separate Spaces

With traditional models, different data types exist in incompatible spaces:

```
┌─────────────────────────────────────────────────────────────────┐
│                 TRADITIONAL EMBEDDINGS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TEXT SPACE                      IMAGE SPACE                   │
│   ┌─────────────┐                 ┌─────────────┐               │
│   │  "sunset"   │     ⛔          │  🌅 [img]   │               │
│   │  "beach"    │   INCOMPATIBLE  │  🏖️ [img]   │               │
│   │  "ocean"    │                 │  🌊 [img]   │               │
│   └─────────────┘                 └─────────────┘               │
│                                                                 │
│   ❌ Cannot compare text to images                              │
│   ❌ Separate indexes required                                  │
│   ❌ No cross-modal search                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Multimodal Embeddings: Unified Space

Multimodal models align text and images in the same geometric space:

```
┌─────────────────────────────────────────────────────────────────┐
│                 MULTIMODAL EMBEDDINGS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    UNIFIED VECTOR SPACE                         │
│   ┌─────────────────────────────────────────────────┐           │
│   │                                                 │           │
│   │          "sunset at the beach"  ←───┐          │           │
│   │                  •                   │          │           │
│   │              •  🌅                   │ CLOSE    │           │
│   │                  • 🏖️                │          │           │
│   │                                      │          │           │
│   │     "mountain landscape"  ←─────────┼───────┐  │           │
│   │            •                         │       │  │           │
│   │        • 🏔️                         │ CLOSE │  │           │
│   │            • 🗻                       │       │  │           │
│   │                                             │  │           │
│   └─────────────────────────────────────────────────┘           │
│                                                                 │
│   ✅ Text and images comparable directly                        │
│   ✅ Single unified index                                       │
│   ✅ Cross-modal search enabled                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works: Contrastive Learning

### The Training Objective

Multimodal models are typically trained using **contrastive learning** on paired data (e.g., images with captions):

```
Training Data:
┌─────────────────────────────────────────────┐
│ 🌅 sunset_beach.jpg  ↔  "Beautiful sunset"  │
│ 🐱 cat_sleeping.jpg  ↔  "Cat napping"       │
│ 🚗 red_car.jpg       ↔  "Red sports car"    │
└─────────────────────────────────────────────┘

Training Objective:
• MAXIMIZE similarity between paired image-text
• MINIMIZE similarity between non-paired items
```

### Contrastive Loss Visualization

```python
# Simplified contrastive learning concept
def contrastive_loss(image_embeddings, text_embeddings):
    """
    For a batch of N image-text pairs:
    - Diagonal elements (matched pairs) should be HIGH similarity
    - Off-diagonal elements (mismatched) should be LOW similarity
    """
    # Compute similarity matrix (N x N)
    similarity_matrix = image_embeddings @ text_embeddings.T
    
    # Target: identity matrix (only diagonal = 1)
    # Loss encourages:
    #   similarity[i,i] → HIGH (matched pairs)
    #   similarity[i,j] → LOW  (mismatched, i≠j)
    
    labels = torch.arange(N)
    loss_i2t = cross_entropy(similarity_matrix, labels)  # Image-to-text
    loss_t2i = cross_entropy(similarity_matrix.T, labels)  # Text-to-image
    
    return (loss_i2t + loss_t2i) / 2
```

### The Result: Aligned Spaces

After training:

| Query | Similar Results |
|-------|-----------------|
| Text: "a dog playing fetch" | Images of dogs playing + related captions |
| Image: 🐕 (dog photo) | Similar dog images + "dog playing" text |
| Text: "product packaging design" | Product images + design descriptions |

---

## Cross-Modal Search Patterns

### 1. Text-to-Image Search

The most common pattern: user types text, finds relevant images.

```python
# User query (text)
query = "modern minimalist living room"
query_embedding = embed_text(query)

# Search image index
results = vector_db.search(
    query_embedding,
    collection="interior_design_images"
)

# Returns: Living room images matching the description
```

### 2. Image-to-Text Search

Find text content related to an image:

```python
# User uploads image
image_embedding = embed_image(user_uploaded_image)

# Search text index
results = vector_db.search(
    image_embedding,
    collection="product_descriptions"
)

# Returns: Text descriptions matching the image
```

### 3. Image-to-Image Search

Find visually similar images:

```python
# Reference image
reference_embedding = embed_image(reference_image)

# Search image index
similar_images = vector_db.search(
    reference_embedding,
    collection="image_library"
)
```

### 4. Unified Search

Search across all content types simultaneously:

```python
# Text query
query_embedding = embed_text("sunset photography")

# Search unified index containing both text and images
results = vector_db.search(
    query_embedding,
    collection="all_content"  # Mixed text + images
)

# Returns: Sunset images AND articles about sunset photography
```

---

## The Semantic Bridge

### Why Same Space Matters

The key insight: embeddings measure **semantic meaning**, not surface form.

| Concept | Text Representation | Image Representation |
|---------|---------------------|----------------------|
| "Dog" | Word vectors | Pixel patterns |
| "Happy" | Contextual semantics | Facial expressions, body language |
| "Luxury car" | Brand associations | Visual design cues |

Multimodal models learn that these different representations **refer to the same concept**.

### Geometric Interpretation

```
Vector Space Geometry:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "sports car"  ─────────────●                                  │
│                              │                                  │
│   🚗 [ferrari.jpg]  ─────────●←── These cluster together        │
│                              │                                  │
│   🏎️ [porsche.jpg]  ─────────●                                  │
│                                                                 │
│                                                                 │
│   "flower garden"  ─────────●                                   │
│                              │                                  │
│   🌸 [roses.jpg]  ──────────●←── These cluster together         │
│                              │                                  │
│   🌺 [tulips.jpg]  ─────────●                                   │
│                                                                 │
│   Distance(sports_car_text, ferrari_image) < 0.2                │
│   Distance(sports_car_text, roses_image) > 0.9                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Foundation for Multimodal RAG

### Traditional RAG: Text Only

```
User Query (text) → Text Embeddings → Text Chunks → LLM → Answer
```

### Multimodal RAG: Text + Images

```
User Query (text) → Multimodal Embedding → Text Chunks + Images → Vision LLM → Answer with Visual Context
```

### Example: Product Support

```python
# User asks: "How do I install the wall mount bracket?"

# Traditional RAG: Find text instructions only
text_results = search_text_index(query)

# Multimodal RAG: Find text AND relevant diagrams
multimodal_results = search_unified_index(query)
# Returns: Installation text + diagram images

# Send to vision-capable LLM
response = vision_llm.generate(
    query=query,
    context=multimodal_results  # Includes images!
)
# "To install the wall mount bracket, first locate the 
#  mounting holes as shown in this diagram [image]..."
```

---

## Limitations and Considerations

### Current Limitations

| Limitation | Details |
|------------|---------|
| **Text length** | Often limited (32-77 tokens for CLIP-based models) |
| **Image resolution** | Typically resized to 224×224 or 512×512 |
| **Language support** | Many models English-only or limited multilingual |
| **Abstract concepts** | Better at concrete objects than abstract ideas |
| **OCR text** | May confuse text IN images with image content |

### When Multimodal Helps

✅ Image search with natural language queries  
✅ Product catalogs with visual and text content  
✅ Document understanding (text + figures)  
✅ Visual Q&A systems  
✅ Content-based recommendations  

### When to Stick with Text-Only

✅ Pure text retrieval (documents, articles)  
✅ Long-form content (multimodal often truncates text)  
✅ Non-visual domains (legal, financial text)  
✅ When image processing latency is prohibitive  

---

## Evolution of Multimodal Models

### Brief History

| Year | Model | Significance |
|------|-------|-------------|
| 2021 | CLIP (OpenAI) | First widely-used contrastive multimodal model |
| 2021 | ALIGN (Google) | Scaled to 1.8B image-text pairs |
| 2023 | SigLIP | Improved CLIP with sigmoid loss |
| 2024 | Cohere embed-v3.0 | Production API with text+image |
| 2025 | Cohere embed-v4.0 | Mixed content, 100+ languages |

### Model Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL MODEL                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TEXT INPUT                         IMAGE INPUT                │
│       │                                   │                     │
│       ▼                                   ▼                     │
│   ┌─────────┐                       ┌─────────┐                 │
│   │  Text   │                       │  Image  │                 │
│   │ Encoder │                       │ Encoder │                 │
│   │(Transf.)│                       │ (ViT)   │                 │
│   └────┬────┘                       └────┬────┘                 │
│        │                                  │                     │
│        ▼                                  ▼                     │
│   ┌─────────┐                       ┌─────────┐                 │
│   │Projection│                      │Projection│                │
│   │  Layer   │                      │  Layer   │                │
│   └────┬────┘                       └────┬────┘                 │
│        │                                  │                     │
│        └──────────────┬───────────────────┘                     │
│                       ▼                                         │
│              SHARED EMBEDDING SPACE                             │
│                 (Same dimensions)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

✅ Multimodal embeddings place **text and images in the same vector space**  
✅ **Contrastive learning** on paired data creates semantic alignment  
✅ Enables **cross-modal search**: text→image, image→text, image→image  
✅ Foundation for **multimodal RAG** with visual context  
✅ Current models have limitations (text length, resolution, language)  
✅ Choose multimodal when visual content matters for your use case

---

**Next:** [Cohere embed-v4.0 →](./02-cohere-embed-v4.md)

---

<!-- 
Sources Consulted:
- OpenAI CLIP paper and blog: https://openai.com/index/clip/
- Cohere Multimodal Embeddings: https://docs.cohere.com/docs/multimodal-embeddings
- Contrastive learning research literature
-->
