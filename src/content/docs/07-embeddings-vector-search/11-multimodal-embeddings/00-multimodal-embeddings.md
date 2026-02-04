---
title: "Multimodal Embeddings"
---

# Multimodal Embeddings

## Overview

Multimodal embeddings represent one of the most powerful advances in embedding technology—the ability to embed **text, images, and video into the same vector space**. This enables cross-modal search where you can find images using text queries, or find similar documents regardless of whether they contain text or visual content.

This lesson explores how multimodal embeddings work, the leading providers and models, practical implementation patterns, and real-world use cases.

---

## What You'll Learn

| Sub-lesson | Description |
|------------|-------------|
| [What Are Multimodal Embeddings](./01-what-are-multimodal-embeddings.md) | Unified vector space concept, cross-modal search foundations |
| [Cohere embed-v4.0](./02-cohere-embed-v4.md) | State-of-the-art multimodal embeddings with text and images |
| [Image Embedding with Cohere](./03-image-embedding-with-cohere.md) | Input types, formats, resolution handling, base64 encoding |
| [Mixed Content Embeddings](./04-mixed-content-embeddings.md) | Combining text and images in single embeddings |
| [CLIP and Alternatives](./05-clip-and-alternatives.md) | OpenAI CLIP, Google multimodal, open-source options |
| [Use Cases](./06-use-cases.md) | Image search, product discovery, document understanding |
| [Implementation Considerations](./07-implementation-considerations.md) | Index strategies, storage, latency optimization |

---

## Key Concepts at a Glance

### The Multimodal Promise

```
┌─────────────────────────────────────────────────────────────────┐
│              UNIFIED VECTOR SPACE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   "a photo of a sunset"  ──────┐                                │
│                                 │                               │
│   🌅 [sunset.jpg]  ────────────┼────▶  Same region of          │
│                                 │       vector space            │
│   🎥 [sunset_timelapse.mp4] ──┘                                │
│                                                                 │
│   ✅ Text query finds relevant images                          │
│   ✅ Image query finds similar images AND related text         │
│   ✅ Video segments searchable by text or image                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Provider Comparison

| Provider | Model | Modalities | Dimensions | Languages |
|----------|-------|------------|------------|-----------|
| **Cohere** | embed-v4.0 | Text, Image | 256-1536 (Matryoshka) | 100+ |
| **Google** | multimodalembedding@001 | Text, Image, Video | 128-1408 | English |
| **OpenAI** | CLIP (open source) | Text, Image | 512-768 | English |
| **Open Source** | OpenCLIP, SigLIP | Text, Image | Varies | Multilingual |

### Quick Reference: When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| **Production multimodal RAG** | Cohere embed-v4.0 (best quality, API) |
| **Google Cloud integration** | Vertex AI multimodalembedding@001 |
| **Open source / self-hosted** | OpenCLIP or SigLIP via Hugging Face |
| **Research / experimentation** | CLIP for simplicity and documentation |
| **Video understanding** | Google Vertex AI (native video support) |

---

## Prerequisites

Before starting this lesson, you should understand:

- [Understanding Embeddings](../01-understanding-embeddings/)
- [Generating Embeddings](../03-generating-embeddings/)
- [Task-Specific Embeddings](../10-task-specific-embeddings/)
- Basic image handling in Python (base64 encoding)

---

## Lesson Structure

### Part 1: Foundations
- [What Are Multimodal Embeddings](./01-what-are-multimodal-embeddings.md) — Understand the unified vector space concept

### Part 2: Leading Providers
- [Cohere embed-v4.0](./02-cohere-embed-v4.md) — Production-ready multimodal API
- [Image Embedding with Cohere](./03-image-embedding-with-cohere.md) — Working with images
- [Mixed Content Embeddings](./04-mixed-content-embeddings.md) — Text + image together

### Part 3: Alternatives & Open Source
- [CLIP and Alternatives](./05-clip-and-alternatives.md) — OpenAI CLIP, Google, open source

### Part 4: Practical Application
- [Use Cases](./06-use-cases.md) — Real-world implementations
- [Implementation Considerations](./07-implementation-considerations.md) — Production patterns

---

## Key Takeaways

By the end of this lesson, you will:

✅ Understand how multimodal embeddings unify text and images  
✅ Implement image and mixed-content embedding with Cohere  
✅ Know when to use CLIP vs. API-based providers  
✅ Build cross-modal search (text-to-image, image-to-text)  
✅ Design indexes for multimodal content  
✅ Optimize storage and latency for production  

---

**Next:** [What Are Multimodal Embeddings →](./01-what-are-multimodal-embeddings.md)

---

<!-- 
Sources Consulted:
- Cohere Multimodal Embeddings: https://docs.cohere.com/docs/multimodal-embeddings
- Cohere Embed API: https://docs.cohere.com/reference/embed
- OpenAI CLIP: https://openai.com/index/clip/
- Google Vertex AI Multimodal Embeddings: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
-->
