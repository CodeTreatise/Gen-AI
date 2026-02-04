---
title: "Matryoshka Embeddings & Dimension Reduction"
---

# Matryoshka Embeddings & Dimension Reduction

## Introduction

Have you ever wondered why embeddings always seem to be exactly 768 or 1536 dimensions? What if you could use 256 dimensions for fast filtering and 1536 for precise reranking—all from the *same* embedding? This is exactly what **Matryoshka embeddings** enable.

Named after Russian nesting dolls, Matryoshka embeddings contain smaller, functional representations nested within larger ones. Truncate a 3072-dimensional embedding to 768 dimensions, and you still get a high-quality vector that works for similarity search—not a broken fragment, but a designed feature.

### Why This Matters for AI Applications

Traditional embeddings lock you into a fixed dimension. Want faster search? Train a new model. Need smaller storage? Train another model. Matryoshka embeddings eliminate this constraint:

- **Adaptive precision**: Use small dimensions for fast initial filtering, larger for precise ranking
- **Storage efficiency**: Store documents at 256 dimensions (8x smaller than 2048) with minimal quality loss
- **Cost optimization**: Reduce vector database costs by 50-75% without reindexing

### What We'll Cover

This lesson explores Matryoshka Representation Learning (MRL) and practical dimension reduction:

1. **What are Matryoshka embeddings** - The nesting doll concept and coarse-to-fine encoding
2. **How MRL training works** - Loss computation at multiple dimension cutoffs
3. **Supported models** - OpenAI, Gemini, Cohere, Nomic, and open-source options
4. **Dimension selection strategies** - Quality vs. efficiency tradeoffs with benchmarks
5. **API parameters** - `dimensions`, `output_dimensionality`, and provider-specific options
6. **Normalization requirements** - The critical step most developers miss
7. **Cost-benefit analysis** - Storage, latency, and quality tradeoffs
8. **Migration strategies** - Transitioning existing systems to variable dimensions

### Prerequisites

Before diving in, you should understand:
- Vector embeddings and similarity search ([Lesson 01: What Are Embeddings](../01-understanding-embeddings/01-what-are-embeddings.md))
- Cosine similarity and dot product ([Lesson 02: Vector Similarity](../02-similarity-search/))
- Basic vector database operations ([Lesson 04: Vector Databases](../04-vector-databases/))

---

## Lesson Structure

This lesson is organized into focused sub-lessons:

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01-what-are-matryoshka-embeddings.md](./01-what-are-matryoshka-embeddings.md) | The Matryoshka Concept | Nesting dolls, MRL, coarse-to-fine encoding |
| [02-mrl-training.md](./02-mrl-training.md) | Training Mechanism | Multi-scale loss, dimension cutoffs, information density |
| [03-supported-models.md](./03-supported-models.md) | Available Models | OpenAI, Gemini, Cohere, Nomic, Sentence Transformers |
| [04-dimension-selection.md](./04-dimension-selection.md) | Choosing Dimensions | Quality benchmarks, use case recommendations |
| [05-api-parameters.md](./05-api-parameters.md) | API Integration | Provider-specific parameters and examples |
| [06-normalization.md](./06-normalization.md) | Normalization | Why truncated vectors must be re-normalized |
| [07-cost-benefit-analysis.md](./07-cost-benefit-analysis.md) | Economics | Storage, latency, cost calculations |
| [08-migration-strategies.md](./08-migration-strategies.md) | Production Migration | A/B testing, progressive rollout |

---

## Quick Reference

### Dimension Reduction Quality (Approximate)

| Reduction | Typical Quality Retained | Best For |
|-----------|-------------------------|----------|
| 100% (full) | 100% | Final ranking, high-stakes search |
| 50% | ~99% | General search, most applications |
| 25% | ~97-98% | Fast filtering, initial retrieval |
| 12.5% | ~94-96% | Coarse clustering, rapid prototyping |

### Provider Comparison

| Provider | Model | Max Dimensions | Parameter |
|----------|-------|----------------|-----------|
| OpenAI | `text-embedding-3-large` | 3072 | `dimensions` |
| OpenAI | `text-embedding-3-small` | 1536 | `dimensions` |
| Google | `gemini-embedding-001` | 3072 | `output_dimensionality` |
| Cohere | `embed-v4.0` | 1024 | `output_dimension` |
| Nomic | `nomic-embed-text-v1.5` | 768 | `truncate_dim` |

---

## Key Takeaways

After completing this lesson, you'll understand:

✅ Why Matryoshka embeddings enable adaptive dimensionality  
✅ How MRL training creates information-dense early dimensions  
✅ Which models support dimension reduction and how to use them  
✅ **Critical**: Why you MUST normalize truncated embeddings  
✅ How to calculate cost savings and quality tradeoffs  
✅ Production strategies for migrating to variable dimensions

---

**Start Learning:** [What Are Matryoshka Embeddings →](./01-what-are-matryoshka-embeddings.md)

---

## Further Reading

- [Matryoshka Representation Learning Paper (arXiv 2205.13147)](https://arxiv.org/abs/2205.13147)
- [Sentence Transformers MatryoshkaLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Gemini Embeddings Documentation](https://ai.google.dev/gemini-api/docs/embeddings)

---

*Next: [What Are Matryoshka Embeddings](./01-what-are-matryoshka-embeddings.md)*
