---
title: "The Managed RAG Paradigm"
---

# The Managed RAG Paradigm

## Introduction

The **managed RAG paradigm** represents a shift from "build everything yourself" to "upload and query." Instead of managing vector databases, embedding pipelines, and search infrastructure, you interact with a simple API that handles the entire RAG pipeline automatically.

This lesson explains what managed RAG offers, how it works under the hood, and the fundamental trade-off between control and convenience.

### What We'll Cover

- End-to-end RAG without infrastructure
- Auto-chunking, auto-embedding, auto-indexing
- API-level simplicity
- The control vs convenience trade-off

### Prerequisites

- Basic understanding of RAG (Retrieval-Augmented Generation)
- Familiarity with embeddings and vector search concepts

---

## What is Managed RAG?

### The Traditional RAG Stack

Building a custom RAG system requires assembling multiple components:

```
┌─────────────────────────────────────────────────────────────────┐
│              Traditional RAG Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOUR DOCUMENTS                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Document Parser│ ← You implement (PDF, DOCX, HTML...)       │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Text Chunker   │ ← You configure (size, overlap, strategy)  │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Embedding API  │ ← You call (OpenAI, Cohere, etc.)          │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Vector Database│ ← You provision (Pinecone, Weaviate...)    │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Search + Rank  │ ← You implement (hybrid, reranking)        │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ LLM Generation │ ← You orchestrate (context, prompts)       │
│  └────────────────┘                                            │
│                                                                 │
│  Components: 6+    Infrastructure: Yes    Expertise: High      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Managed RAG Approach

Managed RAG collapses this into two operations:

```
┌─────────────────────────────────────────────────────────────────┐
│              Managed RAG Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOUR DOCUMENTS                                                 │
│       │                                                         │
│       │  ┌─────────────────────────────────────────────────┐   │
│       └─▶│            MANAGED SERVICE                       │   │
│          │  ┌────────────────────────────────────────────┐ │   │
│          │  │ Auto-Parse → Auto-Chunk → Auto-Embed →     │ │   │
│          │  │ Auto-Index → Auto-Search → Auto-Rerank     │ │   │
│          │  └────────────────────────────────────────────┘ │   │
│          │                                                  │   │
│          │  All handled by OpenAI / Google                 │   │
│          └─────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Your Query     │ → "What does the report say about X?"      │
│  └────────────────┘                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │ Response +     │ ← Answer with automatic citations          │
│  │ Citations      │                                            │
│  └────────────────┘                                            │
│                                                                 │
│  Components: 2     Infrastructure: No     Expertise: Low       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Happens Automatically

### 1. Document Parsing

Managed services automatically parse various file formats:

| Format | Handling |
|--------|----------|
| PDF | Text extraction, OCR if needed |
| DOCX/DOC | Structure-aware parsing |
| HTML | Tag stripping, content extraction |
| Markdown | Header-aware parsing |
| Code files | Language-aware parsing |
| Plain text | Direct processing |

```python
# You just upload the file - parsing is automatic
# OpenAI example
file = client.files.create(
    file=open("quarterly_report.pdf", "rb"),
    purpose="assistants"
)
# The service handles PDF parsing internally
```

### 2. Automatic Chunking

Documents are split into chunks optimized for retrieval:

| Provider | Default Chunk Size | Default Overlap |
|----------|-------------------|-----------------|
| OpenAI | 800 tokens | 400 tokens |
| Gemini | Configurable | Configurable |

```python
# OpenAI allows customization
client.vector_stores.file_batches.create(
    vector_store_id="vs_abc123",
    files=[{
        "file_id": "file_xyz",
        "chunking_strategy": {
            "type": "static",
            "max_chunk_size_tokens": 1000,
            "chunk_overlap_tokens": 200
        }
    }]
)
```

### 3. Automatic Embedding

Chunks are embedded using provider-optimized models:

| Provider | Embedding Model | Dimensions |
|----------|-----------------|------------|
| OpenAI | text-embedding-3-large | 256 |
| Gemini | gemini-embedding-001 | — |

### 4. Automatic Indexing

Embeddings are stored and indexed for fast retrieval:

- **Vector indices** for semantic search
- **Keyword indices** for exact matching (hybrid search)
- **Metadata indices** for filtering

### 5. Automatic Search & Reranking

When you query, the service:

1. Rewrites queries for optimal search
2. Runs parallel keyword AND semantic searches
3. Combines results (hybrid search)
4. Reranks to find most relevant chunks
5. Returns results with relevance scores

---

## API-Level Simplicity

### Complete RAG in ~10 Lines

```python
from openai import OpenAI
client = OpenAI()

# Step 1: Create a vector store
vector_store = client.vector_stores.create(name="Knowledge Base")

# Step 2: Upload files (parsing + chunking + embedding automatic)
client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=[open("doc1.pdf", "rb"), open("doc2.docx", "rb")]
)

# Step 3: Query with automatic retrieval
response = client.responses.create(
    model="gpt-4.1",
    input="Summarize the key findings across all documents",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id]
    }]
)

print(response.output_text)
```

### What You Don't Have to Do

| Task | Custom RAG | Managed RAG |
|------|------------|-------------|
| Choose chunking strategy | ✅ Required | ❌ Optional |
| Select embedding model | ✅ Required | ❌ Handled |
| Provision vector database | ✅ Required | ❌ Included |
| Implement hybrid search | ✅ Required | ❌ Built-in |
| Build reranking pipeline | ✅ Required | ❌ Built-in |
| Extract citations | ✅ Required | ❌ Automatic |
| Handle rate limiting | ✅ Required | ❌ Managed |
| Scale infrastructure | ✅ Required | ❌ Managed |

---

## The Control vs Convenience Trade-off

### What You Gain

```
┌─────────────────────────────────────────────────────────────────┐
│              Benefits of Managed RAG                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ SPEED TO MARKET                                             │
│  • Prototype in hours, not weeks                                │
│  • No infrastructure setup                                      │
│  • No DevOps expertise needed                                   │
│                                                                 │
│  ✅ REDUCED COMPLEXITY                                          │
│  • Single API to learn                                          │
│  • No component integration                                     │
│  • Fewer failure points                                         │
│                                                                 │
│  ✅ AUTOMATIC OPTIMIZATION                                      │
│  • Provider-tuned chunking                                      │
│  • Optimized embedding models                                   │
│  • Built-in reranking                                          │
│                                                                 │
│  ✅ BUILT-IN FEATURES                                          │
│  • Automatic citations                                          │
│  • Metadata filtering                                           │
│  • Hybrid search                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What You Give Up

```
┌─────────────────────────────────────────────────────────────────┐
│              Trade-offs of Managed RAG                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ⚠️ LIMITED CONTROL                                             │
│  • Fixed embedding models (can't use custom)                    │
│  • Limited chunking customization                               │
│  • Provider-defined search algorithms                           │
│                                                                 │
│  ⚠️ VENDOR LOCK-IN                                              │
│  • Data in provider's infrastructure                            │
│  • Can't export embeddings                                      │
│  • Migration requires re-processing                             │
│                                                                 │
│  ⚠️ COST AT SCALE                                               │
│  • Storage costs accumulate                                     │
│  • Per-query costs for retrieval                                │
│  • Less cost optimization options                               │
│                                                                 │
│  ⚠️ FEATURE CONSTRAINTS                                         │
│  • May not support latest techniques                            │
│  • No custom reranking models                                   │
│  • Limited multi-tenancy options                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## When Each Approach Wins

### Managed RAG Wins When

| Scenario | Why Managed |
|----------|-------------|
| Proof of concept | Speed matters more than control |
| Small knowledge base (<1GB) | Infrastructure overhead not worth it |
| Simple use cases | Default settings work well |
| Team without ML expertise | No tuning required |
| Tight timeline | Days to deploy, not weeks |

### Custom RAG Wins When

| Scenario | Why Custom |
|----------|------------|
| Specific embedding model needed | Fine-tuned or domain-specific |
| Complex chunking requirements | Document-aware strategies |
| Multi-tenant architecture | Per-customer isolation |
| Regulatory requirements | Data residency, compliance |
| Cost optimization at scale | Bulk embedding, self-hosting |

---

## The Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Managed vs Custom Decision Tree                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  START: Do you need RAG?                                        │
│              │                                                  │
│              ▼                                                  │
│  ┌───────────────────┐                                         │
│  │ Is this a proof   │──YES──▶ Use MANAGED                     │
│  │ of concept?       │                                         │
│  └───────────────────┘                                         │
│              │ NO                                               │
│              ▼                                                  │
│  ┌───────────────────┐                                         │
│  │ Data < 1GB and    │──YES──▶ Use MANAGED                     │
│  │ simple use case?  │                                         │
│  └───────────────────┘                                         │
│              │ NO                                               │
│              ▼                                                  │
│  ┌───────────────────┐                                         │
│  │ Need custom       │──YES──▶ Use CUSTOM                      │
│  │ embeddings?       │                                         │
│  └───────────────────┘                                         │
│              │ NO                                               │
│              ▼                                                  │
│  ┌───────────────────┐                                         │
│  │ Regulatory or     │──YES──▶ Use CUSTOM                      │
│  │ multi-tenant?     │                                         │
│  └───────────────────┘                                         │
│              │ NO                                               │
│              ▼                                                  │
│       Evaluate both based on cost and timeline                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

✅ **Managed RAG** provides end-to-end RAG with a simple API  
✅ **Automatic handling** of parsing, chunking, embedding, indexing, and search  
✅ **Trade-off**: Less control in exchange for simplicity and speed  
✅ **Best for**: Prototypes, small knowledge bases, teams without ML expertise  
✅ **Consider custom when**: You need specific embeddings, complex chunking, or regulatory compliance

---

**Next:** [OpenAI Vector Stores & File Search →](./02-openai-vector-stores.md)

---

<!-- 
Sources Consulted:
- OpenAI File Search: https://platform.openai.com/docs/guides/tools-file-search
- OpenAI Assistants File Search: https://platform.openai.com/docs/assistants/tools/file-search
- Gemini File Search: https://ai.google.dev/gemini-api/docs/file-search
-->
