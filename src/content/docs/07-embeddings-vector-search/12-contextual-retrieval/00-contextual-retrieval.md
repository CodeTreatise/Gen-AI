---
title: "Contextual Retrieval"
---

# Contextual Retrieval

## Overview

**Contextual Retrieval** is an advanced RAG technique developed by Anthropic that dramatically improves retrieval accuracy by prepending explanatory context to each chunk before embedding. This simple but powerful method addresses the fundamental problem of chunks losing their document context.

This lesson covers the complete Contextual Retrieval methodology, including implementation with hybrid search (BM25), reranking, and cost optimization with prompt caching.

---

## Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [The Context Problem](./01-the-context-problem.md) | Why chunks lose context and how it breaks retrieval |
| 02 | [Contextual Retrieval Solution](./02-contextual-retrieval-solution.md) | The core technique: prepending LLM-generated context |
| 03 | [The Contextualizer Prompt](./03-the-contextualizer-prompt.md) | Designing effective context generation prompts |
| 04 | [Implementation Steps](./04-implementation-steps.md) | End-to-end implementation pipeline |
| 05 | [Hybrid Search with BM25](./05-hybrid-search-bm25.md) | Combining embeddings with lexical search |
| 06 | [Performance Improvements](./06-performance-improvements.md) | Research results and benchmark data |
| 07 | [Prompt Caching](./07-prompt-caching.md) | Cost optimization with Claude's caching |
| 08 | [Reranking](./08-reranking.md) | Boosting accuracy with reranker models |
| 09 | [Best Practices](./09-best-practices.md) | Production recommendations and customization |

---

## Key Concepts

### What is Contextual Retrieval?

```
┌─────────────────────────────────────────────────────────────────┐
│                Traditional RAG vs Contextual Retrieval           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRADITIONAL RAG:                                               │
│                                                                 │
│  Document → [Chunk 1] [Chunk 2] [Chunk 3] → Embed → Index      │
│                                                                 │
│  Problem: "Revenue grew 3%" - Which company? When?             │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  CONTEXTUAL RETRIEVAL:                                          │
│                                                                 │
│  Document → [Chunk 1] → Generate Context → [Context + Chunk 1] │
│                      ↓                                          │
│             "This is from ACME Corp Q2 2023 SEC filing..."     │
│                      ↓                                          │
│             Embed contextualized chunk → Index                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Gains (Anthropic Research)

| Technique | Failure Rate Reduction |
|-----------|------------------------|
| Contextual Embeddings alone | **35% fewer failures** |
| + Contextual BM25 | **49% fewer failures** |
| + Reranking | **67% fewer failures** |

---

## Prerequisites

Before starting this lesson, you should understand:

- **Document chunking** strategies ([Lesson 07](../07-document-chunking/))
- **Embedding models** and vector search ([Lessons 02-06](../02-embedding-models/))
- **Basic RAG architecture** ([Unit 09 Overview](../../09-rag-retrieval-augmented-generation/00-overview.md))
- **Claude API basics** (for context generation)

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. ✅ Explain why traditional chunking loses document context
2. ✅ Implement the Contextual Retrieval preprocessing pipeline
3. ✅ Combine contextual embeddings with BM25 for hybrid search
4. ✅ Use prompt caching to reduce costs by up to 90%
5. ✅ Apply reranking for maximum retrieval accuracy
6. ✅ Choose optimal embedding models and parameters

---

## Quick Reference

### The Contextualizer Prompt

```xml
<document>
{{WHOLE_DOCUMENT}}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else.
```

### Cost Estimate

With Claude's prompt caching:
- **~$1.02 per million document tokens** for contextualization
- Cache reads are 10% of base input token cost
- 5-minute cache lifetime (refreshed on each use)

---

## Tools & Resources

### APIs Used
- **Anthropic Claude** - Context generation with prompt caching
- **Voyage AI / Google Gemini** - Best-performing embedding models
- **Cohere / Voyage Rerankers** - Final ranking step

### Libraries
- `anthropic` - Claude API client
- `rank-bm25` - Python BM25 implementation
- `cohere` - Reranking API
- `voyageai` - Embeddings and reranking

---

**Start:** [The Context Problem →](./01-the-context-problem.md)

---

<!-- 
Sources:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Claude Prompt Caching: https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching
-->
