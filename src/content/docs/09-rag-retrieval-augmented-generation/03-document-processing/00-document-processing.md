---
title: "Document Processing & Chunking Strategies"
---

# Document Processing & Chunking Strategies

## Introduction

Chunking is the art of breaking documents into retrieval-sized pieces. Do it poorly, and your RAG system retrieves fragments that lack context. Do it well, and every chunk carries enough meaning to generate accurate answers.

This lesson explores chunking strategies from basic fixed-size splitting to advanced techniques like Contextual Retrieval and Late Chunking that preserve document context within each chunk.

### What We'll Cover

- Fixed vs semantic chunking approaches
- Handling different document types
- Preserving document structure
- Tables and images in RAG
- Contextual Retrieval (Anthropic method)
- Late Chunking (Jina method)
- AI-powered semantic chunking

### Prerequisites

- Document ingestion pipeline knowledge (Lesson 2)
- Understanding of embeddings (Unit 7)
- Basic NLP concepts

---

## Quick Start

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a text splitter with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split a document
document = """
# Introduction

RAG systems retrieve relevant context before generating responses.
The quality of retrieval depends heavily on how documents are chunked.

## Why Chunking Matters

Chunks that are too large dilute semantic meaning.
Chunks that are too small lose context.
Finding the right balance is crucial for accuracy.
"""

chunks = splitter.split_text(document)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):\n{chunk}\n---")
```

**Output:**
```
Chunk 1 (195 chars):
# Introduction

RAG systems retrieve relevant context before generating responses.
The quality of retrieval depends heavily on how documents are chunked.
---
Chunk 2 (182 chars):
## Why Chunking Matters

Chunks that are too large dilute semantic meaning.
Chunks that are too small lose context.
Finding the right balance is crucial for accuracy.
---
```

---

## The Core Problem

```mermaid
flowchart TD
    subgraph Problem["The Chunking Dilemma"]
        D[Long Document] --> C1[Chunk 1: "The company's revenue grew 3%"]
        D --> C2[Chunk 2: "Its market share increased"]
        D --> C3[Chunk 3: "The CEO announced..."]
    end
    
    subgraph Issue["Lost Context"]
        C1 --> Q1["Which company?"]
        C2 --> Q2["Which company?"]
        C3 --> Q3["Which CEO?"]
    end
    
    subgraph Solution["Solutions Covered"]
        S1["Contextual Retrieval<br/>(prepend context)"]
        S2["Late Chunking<br/>(conditional embeddings)"]
        S3["Semantic Chunking<br/>(AI boundaries)"]
    end
```

When we split documents naively, chunks lose the context that gives them meaning:
- Pronouns ("it", "they", "the company") can't be resolved
- Section context disappears
- Relationships between chunks break

---

## Chunking Strategy Comparison

| Strategy | Description | Best For | Pros | Cons |
|----------|-------------|----------|------|------|
| **Fixed-size** | Split by character/token count | General purpose, simple pipelines | Fast, predictable | Breaks mid-sentence |
| **Recursive** | Split on separators (paragraphs, sentences) | Most text documents | Respects boundaries | May produce uneven chunks |
| **Semantic** | Split on topic changes | Long-form content | Coherent chunks | Slow, requires AI |
| **By Title** | Split on headings/sections | Structured documents | Preserves structure | Needs proper headings |
| **Contextual** | Prepend context to chunks | High-accuracy needs | +49% retrieval accuracy | LLM cost per chunk |
| **Late Chunking** | Conditional embeddings | Long documents | Preserves full context | Requires long-context models |

---

## Key Terminology

| Term | Definition |
|------|------------|
| **Chunk** | A segment of a document sized for retrieval and context windows |
| **Chunk overlap** | Characters/tokens shared between adjacent chunks to maintain continuity |
| **Semantic coherence** | How well a chunk stands alone as a meaningful unit |
| **Retrieval unit** | The granularity at which content is indexed and retrieved |
| **Context window** | Maximum tokens an LLM can process (determines max chunk + prompt size) |
| **i.i.d. embeddings** | Independent and identically distributed — chunks embedded without document context |
| **Conditional embeddings** | Chunk embeddings that incorporate surrounding document context |
| **Contextual retrieval** | Prepending explanatory context to chunks before embedding |

---

## OpenAI Recommendations

OpenAI's file search uses these defaults:
- **Chunk size**: 800 tokens
- **Chunk overlap**: 400 tokens (50%)

> **Note:** High overlap (50%) helps maintain context across chunk boundaries but increases storage and embedding costs.

---

## Lesson Structure

This lesson covers chunking from basic to advanced:

| Section | Topic | Key Concepts |
|---------|-------|--------------|
| [01](./01-chunking-strategies.md) | Chunking Strategies | Fixed, recursive, semantic splitting |
| [02](./02-handling-document-types.md) | Document Type Handling | Type-specific parsers, unified output |
| [03](./03-preserving-document-structure.md) | Structure Preservation | Headings, sections, cross-references |
| [04](./04-table-image-handling.md) | Tables & Images | Conversion, description generation |
| [05](./05-contextual-retrieval.md) | Contextual Retrieval | Anthropic's 67% improvement method |
| [06](./06-late-chunking.md) | Late Chunking | Jina's conditional embedding approach |
| [07](./07-semantic-chunking-ai.md) | AI Semantic Chunking | LLM boundary detection |

---

## Summary

Chunking strategy directly impacts RAG retrieval quality:

✅ **Fixed-size** is fast but naive — use as baseline
✅ **Recursive splitting** respects natural boundaries
✅ **Structure-aware** chunking preserves document organization
✅ **Contextual Retrieval** adds 35-67% accuracy improvement
✅ **Late Chunking** creates context-aware embeddings without LLM calls

**Next:** [Chunking Strategies](./01-chunking-strategies.md) — Master the fundamentals of text splitting.

---

## Further Reading

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - 49-67% improvement methodology
- [Jina Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models) - Conditional embeddings
- [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) - Parser modules
- [Unstructured Chunking](https://docs.unstructured.io/open-source/core-functionality/chunking) - Element-aware chunking

<!--
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Jina Late Chunking: https://jina.ai/news/late-chunking-in-long-context-embedding-models
- LlamaIndex Node Parsers: https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/
- Unstructured Chunking: https://docs.unstructured.io/open-source/core-functionality/chunking
-->
