---
title: "Similarity Search"
---

# Similarity Search

## Lesson Overview

You've stored vectors in a database—now comes the moment of truth: finding the right content when a user asks a question. Similarity search is where embeddings prove their value, transforming natural language queries into precise document retrieval.

This lesson covers the complete similarity search pipeline: from generating query embeddings with the correct settings, through retrieval strategies that balance speed and accuracy, to advanced techniques like hybrid search and reranking that push retrieval quality to production-grade levels.

---

## Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Query Embedding Generation](./01-query-embedding-generation.md) | Task types, preprocessing, caching |
| 02 | [Top-K Retrieval](./02-top-k-retrieval.md) | Choosing K, pagination strategies |
| 03 | [Similarity Thresholds](./03-similarity-thresholds.md) | Score filtering, dynamic thresholds |
| 04 | [Metadata Filtering](./04-metadata-filtering.md) | Pre/post-filter, complex queries |
| 05 | [Hybrid Search](./05-hybrid-search.md) | BM25 + semantic, score fusion |
| 06 | [SPLADE & Sparse Embeddings](./06-splade-sparse-embeddings.md) | Learned sparse vectors |
| 07 | [Contextual Retrieval](./07-contextual-retrieval.md) | Anthropic's chunk contextualization |
| 08 | [Reranking](./08-reranking.md) | Cross-encoders, Cohere/Voyage |
| 09 | [Maximal Marginal Relevance](./09-maximal-marginal-relevance.md) | Diversity in results |

---

## Prerequisites

- Understanding of embedding fundamentals (Lessons 01-03)
- Familiarity with vector databases (Lessons 04-05)
- Python and async programming basics
- API keys for embedding providers (OpenAI, Cohere, or Voyage)

---

## Key Concepts

### The Retrieval Pipeline

```mermaid
flowchart LR
    Q[User Query] --> E[Embed Query]
    E --> R[Initial Retrieval]
    R --> F[Filter/Threshold]
    F --> RR[Rerank]
    RR --> D[Diverse Selection]
    D --> Results[Top Results]
```

### What You'll Build

By the end of this lesson, you'll have a production-ready search pipeline that:

- Generates query embeddings with proper task types
- Retrieves candidates using hybrid semantic + keyword search
- Filters by metadata and similarity thresholds
- Reranks for precision using cross-encoders
- Ensures diversity with MMR

---

## Learning Outcomes

After completing this lesson, you will be able to:

- ✅ Generate query embeddings that match your document embeddings
- ✅ Implement hybrid search combining BM25 and semantic similarity
- ✅ Apply reranking to dramatically improve precision
- ✅ Use MMR to balance relevance with diversity
- ✅ Configure thresholds and filters for your use case

---

**Start:** [Query Embedding Generation](./01-query-embedding-generation.md)
