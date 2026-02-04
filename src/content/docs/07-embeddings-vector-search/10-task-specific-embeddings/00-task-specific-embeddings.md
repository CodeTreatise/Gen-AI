---
title: "Task-Specific Embeddings"
---

# Task-Specific Embeddings

## Introduction

Not all embeddings are created equal—even from the same model. **The same text can have different optimal vector representations depending on how you intend to use it.** A search query should be embedded differently than a document. A text destined for classification needs different optimization than one being clustered.

Task-specific embeddings allow you to tell the model *what you're trying to accomplish*, enabling it to generate vectors optimized for that specific use case. This simple parameter change can yield **5-15% improvements in retrieval quality** without any other modifications.

### What We'll Cover

This lesson explores task-specific embeddings across major providers:

| Sub-Lesson | Topic | Key Concepts |
|------------|-------|--------------|
| [01-why-task-type-matters](./01-why-task-type-matters.md) | Why Task Type Matters | Same text → different embeddings, query/document asymmetry |
| [02-gemini-task-types](./02-gemini-task-types.md) | Gemini Task Types | 8 task types including CODE_RETRIEVAL and FACT_VERIFICATION |
| [03-cohere-input-types](./03-cohere-input-types.md) | Cohere Input Types | search_query, search_document, classification, clustering |
| [04-voyage-input-types](./04-voyage-input-types.md) | Voyage Input Types | query, document with transparent prompt prepending |
| [05-openai-approach](./05-openai-approach.md) | OpenAI Approach | General-purpose model with manual prefix strategies |
| [06-best-practices](./06-best-practices.md) | Best Practices | Matching types, avoiding mixing, benchmarking |

### Prerequisites

- [Generating Embeddings](../03-generating-embeddings/00-generating-embeddings.md)
- [Similarity Search](../06-similarity-search/00-similarity-search.md)
- Basic understanding of asymmetric retrieval

---

## Quick Reference

### Provider Task Type Comparison

| Provider | Parameter | Values | Notes |
|----------|-----------|--------|-------|
| **Gemini** | `task_type` | 8 types | Most comprehensive |
| **Cohere** | `input_type` | 5 types | Includes image |
| **Voyage** | `input_type` | query, document | Transparent prompts |
| **OpenAI** | *None* | N/A | General-purpose |
| **Jina** | `task` | retrieval.query, retrieval.passage, etc. | Task LoRA adapters |

### When to Use Each Task Type

| Use Case | Gemini | Cohere | Voyage |
|----------|--------|--------|--------|
| Search queries | `RETRIEVAL_QUERY` | `search_query` | `query` |
| Indexed documents | `RETRIEVAL_DOCUMENT` | `search_document` | `document` |
| Duplicate detection | `SEMANTIC_SIMILARITY` | `classification` | `None` |
| Topic modeling | `CLUSTERING` | `clustering` | `None` |
| Spam/sentiment | `CLASSIFICATION` | `classification` | `None` |
| Code search | `CODE_RETRIEVAL_QUERY` | N/A | Use voyage-code-3 |
| Q&A systems | `QUESTION_ANSWERING` | `search_query` | `query` |
| Fact-checking | `FACT_VERIFICATION` | `search_query` | `query` |

---

## The Core Insight

### Symmetric vs. Asymmetric Embedding

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYMMETRIC EMBEDDING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Text A ──────┐                                                │
│                ├──→ Same Embedding Space ──→ Cosine Similarity │
│   Text B ──────┘                                                │
│                                                                 │
│   Use case: Duplicate detection, paraphrase identification     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   ASYMMETRIC EMBEDDING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ────────→ Query Space ─────┐                           │
│                                    ├──→ Cross-Space Similarity │
│   Document ─────→ Document Space ──┘                           │
│                                                                 │
│   Use case: Search, retrieval, Q&A                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Asymmetry Helps

Queries and documents are fundamentally different:

| Queries | Documents |
|---------|-----------|
| Short (3-10 words) | Long (100-10,000+ words) |
| Questions or fragments | Complete statements |
| Express information need | Contain information |
| "What is..." | "X is defined as..." |

Task-specific embeddings account for these differences.

---

## Quick Start Examples

### Gemini

```python
from google import genai
from google.genai import types

client = genai.Client()

# Embed a query
query_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What are the benefits of meditation?",
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)

# Embed a document
doc_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Meditation has been shown to reduce stress and improve focus...",
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
```

### Cohere

```python
import cohere

co = cohere.ClientV2(api_key="your-key")

# Embed a query
query_response = co.embed(
    texts=["What are the benefits of meditation?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]
)

# Embed documents
doc_response = co.embed(
    texts=["Meditation has been shown to reduce stress..."],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)
```

### Voyage

```python
import voyageai

vo = voyageai.Client()

# Embed a query
query_result = vo.embed(
    texts=["What are the benefits of meditation?"],
    model="voyage-4-large",
    input_type="query"
)

# Embed documents
doc_result = vo.embed(
    texts=["Meditation has been shown to reduce stress..."],
    model="voyage-4-large",
    input_type="document"
)
```

---

## Common Mistakes

| ❌ Mistake | ✅ Solution |
|-----------|------------|
| Using same task type for queries and documents | Use `RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for documents |
| Mixing embeddings from different task types in same index | Keep indexes task-type consistent |
| Not specifying task type at all | Always specify when provider supports it |
| Using `SEMANTIC_SIMILARITY` for search | Use `RETRIEVAL_*` types for search |

---

## Lesson Navigation

**Start with:** [Why Task Type Matters →](./01-why-task-type-matters.md)

**Or jump to a specific provider:**
- [Gemini Task Types](./02-gemini-task-types.md)
- [Cohere Input Types](./03-cohere-input-types.md)
- [Voyage Input Types](./04-voyage-input-types.md)
- [OpenAI Approach](./05-openai-approach.md)

---

**Previous:** [Matryoshka Embeddings ←](../09-matryoshka-embeddings/00-matryoshka-embeddings.md)

**Next:** [Multimodal Embeddings →](../11-multimodal-embeddings.md)

---

<!-- 
Sources Consulted:
- Gemini Embeddings documentation: https://ai.google.dev/gemini-api/docs/embeddings
- Cohere Embeddings documentation: https://docs.cohere.com/docs/embeddings
- Voyage AI Embeddings documentation: https://docs.voyageai.com/docs/embeddings
- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
-->
