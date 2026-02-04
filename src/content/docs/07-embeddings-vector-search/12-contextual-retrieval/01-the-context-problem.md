---
title: "The Context Problem"
---

# The Context Problem

## Introduction

Traditional RAG systems split documents into chunks for efficient retrieval, but this chunking process destroys critical context. A chunk that made perfect sense within a document becomes ambiguous when isolated.

This lesson explains the fundamental context problem in RAG and why even the best embedding models cannot fully solve it.

### What We'll Cover

- How chunking destroys document context
- Real examples of context loss
- Why embeddings alone can't fix this
- The impact on retrieval quality

### Prerequisites

- Understanding of document chunking
- Basic RAG architecture knowledge

---

## The Chunking Problem

### How Traditional RAG Works

```
┌─────────────────────────────────────────────────────────────────┐
│                 Traditional RAG Preprocessing                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Document (e.g., SEC Filing)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ACME Corporation Q2 2023 SEC Filing                     │   │
│  │                                                         │   │
│  │ Revenue increased 3% over previous quarter...           │   │
│  │ Operating expenses decreased by 5%...                   │   │
│  │ Net income improved significantly...                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│                     Split into chunks                           │
│                          │                                      │
│          ┌───────────────┼───────────────┐                     │
│          ▼               ▼               ▼                     │
│     ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│     │ Chunk 1 │    │ Chunk 2 │    │ Chunk 3 │                 │
│     │ ~500    │    │ ~500    │    │ ~500    │                 │
│     │ tokens  │    │ tokens  │    │ tokens  │                 │
│     └─────────┘    └─────────┘    └─────────┘                 │
│                                                                 │
│  ⚠️ Context Lost: Which company? Which quarter? Which year?   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Lost

When we chunk documents, we lose:

| Lost Context | Example |
|--------------|---------|
| **Document identity** | Which company is "the company"? |
| **Temporal context** | Which quarter/year is "this quarter"? |
| **Section headers** | Is this about revenue or expenses? |
| **Referential context** | What does "it" or "they" refer to? |
| **Hierarchical position** | Is this a main point or sub-detail? |

---

## Real-World Example

### The SEC Filing Scenario

Imagine you have a collection of quarterly SEC filings from multiple companies embedded in your knowledge base.

**User Query:**
```
"What was the revenue growth for ACME Corp in Q2 2023?"
```

**A relevant chunk might contain:**
```
"The company's revenue grew by 3% over the previous quarter."
```

### The Problem

This chunk is **semantically relevant** to the query—it's about revenue growth. But:

- ❌ Which company? (Could be any company in the corpus)
- ❌ Which quarter? (The chunk doesn't say)
- ❌ Which year? (No year mentioned)
- ❌ Previous quarter compared to what? (Context missing)

Even with a perfect embedding model that understands "revenue growth," retrieval may fail because the chunk lacks identifying context.

---

## Why Embeddings Can't Fix This

### Semantic Similarity ≠ Correct Match

Embedding models excel at capturing **semantic meaning**, but they cannot infer **missing information**:

```python
# These chunks are semantically similar but refer to different companies
chunk_1 = "The company's revenue grew by 3% over the previous quarter."
chunk_2 = "The company's revenue grew by 3% over the previous quarter."

# Query
query = "What was ACME Corp's Q2 2023 revenue growth?"

# Embedding similarity will be high for BOTH chunks!
# But only one is from ACME Corp's filing
```

### The Fundamental Limitation

```
┌─────────────────────────────────────────────────────────────────┐
│              Why Embeddings Alone Fail                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: "ACME Corp Q2 2023 revenue growth"                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Chunk: "The company's revenue grew by 3%"                │  │
│  │                                                          │  │
│  │ Embedding captures:                                      │  │
│  │   ✓ revenue                                              │  │
│  │   ✓ growth                                               │  │
│  │   ✓ percentage increase                                  │  │
│  │                                                          │  │
│  │ Embedding CANNOT capture (not in chunk):                 │  │
│  │   ✗ ACME Corp (company name not present)                │  │
│  │   ✗ Q2 (quarter not mentioned)                          │  │
│  │   ✗ 2023 (year not mentioned)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Result: Semantically relevant but may be WRONG company/time   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## More Context Loss Examples

### Example 1: Technical Documentation

**Full Document Context:**
```
Python 3.12 Release Notes
=========================
New Features:
- Improved error messages
- Performance optimizations

Deprecations:
- The 'distutils' module is deprecated
```

**Isolated Chunk:**
```
"The module is deprecated and will be removed in a future version."
```

**Lost Context:**
- Which module? (`distutils`)
- Which Python version? (3.12)
- When will it be removed? (future version = which?)

### Example 2: Legal Contracts

**Full Document Context:**
```
Agreement between Party A (Microsoft Corporation) and 
Party B (OpenAI, Inc.) dated January 15, 2024

Section 3: Intellectual Property
3.1 All intellectual property developed under this agreement
shall be jointly owned by both parties.
```

**Isolated Chunk:**
```
"All intellectual property developed under this agreement 
shall be jointly owned by both parties."
```

**Lost Context:**
- Who are the parties? (Microsoft and OpenAI)
- When was this agreement? (January 2024)
- What type of agreement? (IP ownership)

### Example 3: Research Papers

**Full Document Context:**
```
Title: "Attention Is All You Need"
Authors: Vaswani et al., Google Brain
Published: NeurIPS 2017

Abstract: We propose a new architecture called Transformer...

Section 3.2.1: Scaled Dot-Product Attention
The output is computed as a weighted sum of the values...
```

**Isolated Chunk:**
```
"The output is computed as a weighted sum of the values, 
where the weight assigned to each value is computed by a 
compatibility function of the query with the corresponding key."
```

**Lost Context:**
- Which paper? (Attention Is All You Need)
- Which architecture? (Transformer)
- Which specific component? (Scaled Dot-Product Attention)

---

## Impact on Retrieval Quality

### Failure Modes

| Failure Mode | Description | Example |
|--------------|-------------|---------|
| **False Positives** | Retrieved chunk is semantically similar but wrong entity | Revenue chunk from wrong company |
| **Ambiguous Results** | Multiple chunks match equally well | Same text appears in multiple docs |
| **Missed Specificity** | Generic match instead of specific | Any "module deprecated" instead of specific module |
| **Temporal Confusion** | Wrong time period retrieved | Q1 data when Q2 was requested |

### Quantified Impact

Anthropic's research showed that traditional RAG (even with excellent embeddings) has a **5.7% failure rate** for top-20 retrieval—meaning ~1 in 18 queries fails to retrieve the correct information.

This may sound small, but in production:
- 1M queries/day = 57,000 failed retrievals daily
- Each failure = incorrect or incomplete answer to user
- Compounds with downstream generation quality

---

## The Scale of the Problem

### Why This Matters More with Scale

```
Documents:     10        100       1,000     10,000
               │         │         │         │
               ▼         ▼         ▼         ▼
Chunks:        50        500       5,000     50,000
               │         │         │         │
               ▼         ▼         ▼         ▼
Ambiguity:    Low     Medium     High    Very High

As corpus grows, more chunks compete for same queries,
making context loss increasingly problematic.
```

### Document Types Most Affected

| Document Type | Context Sensitivity | Why |
|---------------|---------------------|-----|
| **Financial reports** | Very High | Entity + temporal critical |
| **Legal documents** | Very High | Parties + dates + clauses |
| **Technical docs** | High | Version + component specific |
| **Research papers** | High | Paper + section + methodology |
| **Support tickets** | Medium | Customer + product context |
| **General articles** | Lower | Usually self-contained |

---

## Previous Approaches (Limited Success)

### Approach 1: Larger Chunks

```python
# Idea: Use bigger chunks to preserve more context
chunk_size = 2000  # Instead of 500

# Problem: 
# - Still loses document header/metadata
# - Harder to retrieve precise information
# - More noise in retrieved chunks
```

### Approach 2: Chunk Overlap

```python
# Idea: Overlap chunks to preserve boundary context
chunk_overlap = 200  # Overlap tokens

# Problem:
# - Doesn't help with document-level context
# - Increases index size
# - Doesn't add entity/temporal info
```

### Approach 3: Metadata Filtering

```python
# Idea: Add metadata and filter during retrieval
metadata = {
    "company": "ACME Corp",
    "quarter": "Q2",
    "year": "2023"
}

# Problem:
# - Requires structured metadata (often unavailable)
# - Manual annotation doesn't scale
# - User query may not specify filters
```

### Approach 4: Generic Summaries

```python
# Idea: Prepend document summary to each chunk
summary = "This document is a quarterly SEC filing..."

# Problem:
# - Same summary for ALL chunks (not specific)
# - Doesn't help differentiate chunks within document
# - Anthropic tested this: "very limited gains"
```

---

## The Need for a Better Solution

The fundamental insight is that **context must be chunk-specific**, not just document-level. Each chunk needs:

1. **Identity context** - What document is this from?
2. **Positional context** - Where in the document does this appear?
3. **Referential context** - What do pronouns/references mean?
4. **Temporal context** - What time period does this cover?

This is exactly what **Contextual Retrieval** provides—and we'll explore the solution in the next lesson.

---

## Summary

✅ Chunking **destroys document context** that's critical for accurate retrieval  
✅ Lost context includes **entity identity, temporal info, and references**  
✅ **Embeddings capture semantics** but cannot infer missing information  
✅ Traditional workarounds (larger chunks, overlap, metadata) have **limited effectiveness**  
✅ The problem **scales with corpus size** as more chunks compete for queries

---

**Next:** [Contextual Retrieval Solution →](./02-contextual-retrieval-solution.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
-->
