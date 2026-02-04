---
title: "Document Chunking"
---

# Document Chunking

## Lesson Overview

Chunking is the art of breaking documents into pieces that balance context preservation with retrieval precision. Done well, chunking enables precise retrieval of exactly the information needed. Done poorly, it destroys context and returns irrelevant fragments.

This lesson covers the complete chunking landscape: from fundamental sizing decisions through advanced techniques like semantic chunking and Anthropic's contextual retrieval method that improves retrieval by 35-67%.

---

## Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Why Chunking Matters](./01-why-chunking-matters.md) | Context limits, retrieval granularity, the "lost in middle" problem |
| 02 | [Chunk Size Considerations](./02-chunk-size-considerations.md) | Optimal ranges, task-specific sizing, embedding model limits |
| 03 | [Overlap Strategies](./03-overlap-strategies.md) | Sliding windows, context preservation, trade-offs |
| 04 | [Structure-Based Chunking](./04-structure-based-chunking.md) | Paragraph, heading, semantic boundaries, code splitting |
| 05 | [Contextual Chunking](./05-contextual-chunking.md) | Anthropic's method, LLM-generated context, 35% improvement |
| 06 | [Managed Chunking Services](./06-managed-chunking-services.md) | Gemini, OpenAI auto-chunking, when to use managed |
| 07 | [Late Chunking](./07-late-chunking.md) | Embed full document first, then chunk representations |
| 08 | [Semantic Chunking](./08-semantic-chunking.md) | AI-assisted boundary detection, embedding clustering |
| 09 | [Chunk Metadata Preservation](./09-chunk-metadata.md) | Source tracking, position info, cross-references |

---

## Prerequisites

- Understanding of embeddings and vector search (Lessons 01-06)
- Familiarity with text processing in Python
- Basic understanding of RAG pipelines

---

## Key Concepts

### The Chunking Pipeline

```mermaid
flowchart LR
    D[Document] --> S[Split Strategy]
    S --> C[Raw Chunks]
    C --> O[Add Overlap]
    O --> M[Attach Metadata]
    M --> X[Add Context]
    X --> E[Embed & Store]
```

### Chunking Strategy Comparison

| Strategy | Complexity | Quality | Best For |
|----------|------------|---------|----------|
| Fixed-size | Low | Basic | Quick prototyping |
| Sentence-based | Low | Good | General text |
| Structure-based | Medium | Better | Formatted documents |
| Semantic | High | Best | Production systems |
| Contextual | High | Excellent | Maximum retrieval quality |

---

## Learning Outcomes

After completing this lesson, you will be able to:

- ✅ Choose optimal chunk sizes for different use cases
- ✅ Implement overlap strategies that preserve context
- ✅ Use structure-aware chunking for documents with headings/sections
- ✅ Apply Anthropic's contextual chunking for 35%+ improvement
- ✅ Understand when to use semantic vs fixed-size chunking
- ✅ Preserve metadata for source tracking and filtering

---

**Start:** [Why Chunking Matters](./01-why-chunking-matters.md)
