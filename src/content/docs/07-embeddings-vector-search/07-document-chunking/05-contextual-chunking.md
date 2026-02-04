---
title: "Contextual Chunking"
---

# Contextual Chunking

## Introduction

Contextual chunking, pioneered by Anthropic's research, solves a fundamental RAG problem: chunks lose their document context. A chunk saying "This method achieves 35% improvement" means nothing without knowing what "this" refers to.

> **🤖 AI Context:** Anthropic's contextual retrieval approach showed **35% improvement** with contextual embeddings, **49% with contextual BM25**, and **67% when combined with reranking**.

---

## The Context Problem

Standard chunking creates orphaned text:

```
┌─────────────────────────────────────────────────┐
│ Original Document: "Machine Learning Basics"    │
├─────────────────────────────────────────────────┤
│ Chapter 3: Neural Networks                      │
│                                                 │
│ Section 3.2: Activation Functions               │
│                                                 │
│ The ReLU function is defined as f(x) = max(0,x)│
│ This approach solves the vanishing gradient    │
│ problem that plagued earlier architectures...  │
└─────────────────────────────────────────────────┘
                    │
                    ▼ Standard Chunking
┌─────────────────────────────────────────────────┐
│ Chunk: "This approach solves the vanishing     │
│ gradient problem that plagued earlier          │
│ architectures..."                              │
│                                                 │
│ ❌ What approach? Which architectures?         │
└─────────────────────────────────────────────────┘
```

---

## Anthropic's Contextual Retrieval

Anthropic's solution: prepend LLM-generated context to each chunk:

```python
import anthropic
from typing import Optional

client = anthropic.Anthropic()

CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else."""

def generate_chunk_context(
    document: str,
    chunk: str,
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """Generate context for a chunk using Claude."""
    
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": CONTEXT_PROMPT.format(
                document=document,
                chunk=chunk
            )
        }]
    )
    
    return response.content[0].text

def create_contextual_chunk(
    document: str,
    chunk: str
) -> str:
    """Create a contextualized version of the chunk."""
    
    context = generate_chunk_context(document, chunk)
    return f"{context}\n\n{chunk}"
```

**Example output:**

```
Original chunk:
"This approach solves the vanishing gradient problem that plagued 
earlier architectures..."

Contextualized chunk:
"In the context of neural network activation functions, specifically 
discussing ReLU (Rectified Linear Unit) in Chapter 3 of a machine 
learning basics document:

This approach solves the vanishing gradient problem that plagued 
earlier architectures..."
```

---

## Prompt Caching for Cost Reduction

Processing every chunk with the full document is expensive. Use prompt caching:

```python
import anthropic

client = anthropic.Anthropic()

def batch_contextualize_chunks(
    document: str,
    chunks: list[str],
    model: str = "claude-sonnet-4-20250514"
) -> list[str]:
    """Contextualize multiple chunks with prompt caching."""
    
    contextualized = []
    
    # First request caches the document
    for i, chunk in enumerate(chunks):
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>{document}</document>",
                        "cache_control": {"type": "ephemeral"}  # Cache this
                    },
                    {
                        "type": "text",
                        "text": f"""Here is chunk {i+1}:
<chunk>{chunk}</chunk>

Give a short context to situate this chunk within the document 
for improving search retrieval. Answer only with the context."""
                    }
                ]
            }]
        )
        
        context = response.content[0].text
        contextualized.append(f"{context}\n\n{chunk}")
    
    return contextualized
```

**Cost comparison:**

| Method | Cost per 100 chunks |
|--------|---------------------|
| No caching | ~$2.50 |
| With prompt caching | ~$0.35 (86% reduction) |

> **Note:** Prompt caching requires documents > 1024 tokens and works best for batch processing.

---

## Contextual BM25 + Embeddings

Anthropic found the best results combine contextual embeddings with BM25:

```python
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI

client = OpenAI()

class HybridContextualRetriever:
    """Combine contextual embeddings with BM25 for optimal retrieval."""
    
    def __init__(
        self,
        embedding_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.chunks = []
        self.contextualized_chunks = []
        self.embeddings = []
        self.bm25 = None
    
    def add_document(
        self,
        document: str,
        chunks: list[str],
        contextualized_chunks: list[str]
    ):
        """Index document chunks with both methods."""
        
        self.chunks = chunks
        self.contextualized_chunks = contextualized_chunks
        
        # Create embeddings from contextualized chunks
        self.embeddings = self._embed_chunks(contextualized_chunks)
        
        # Build BM25 index from contextualized chunks
        tokenized = [c.lower().split() for c in contextualized_chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def _embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Create embeddings for chunks."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return np.array([e.embedding for e in response.data])
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """Hybrid search combining embedding similarity and BM25."""
        
        # Embedding search
        query_embedding = self._embed_chunks([query])[0]
        embedding_scores = np.dot(self.embeddings, query_embedding)
        embedding_scores = (embedding_scores + 1) / 2  # Normalize to 0-1
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()  # Normalize
        
        # Combine scores
        combined_scores = (
            self.embedding_weight * embedding_scores +
            self.bm25_weight * bm25_scores
        )
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [
            {
                "chunk": self.chunks[i],  # Return original chunk
                "contextualized": self.contextualized_chunks[i],
                "score": float(combined_scores[i]),
                "embedding_score": float(embedding_scores[i]),
                "bm25_score": float(bm25_scores[i])
            }
            for i in top_indices
        ]
```

---

## Performance Improvements

| Configuration | Failure Rate Reduction |
|---------------|------------------------|
| Baseline (embeddings only) | — |
| + Contextual embeddings | 35% ↓ |
| + Contextual embeddings + BM25 | 49% ↓ |
| + Contextual + BM25 + Reranking | 67% ↓ |

---

## Implementation Patterns

### Pattern 1: Pre-compute Context

```python
def precompute_contextual_chunks(
    documents: list[dict]
) -> list[dict]:
    """Pre-compute context for all chunks during indexing."""
    
    indexed_chunks = []
    
    for doc in documents:
        doc_text = doc["content"]
        doc_chunks = chunk_document(doc_text)
        
        for chunk in doc_chunks:
            context = generate_chunk_context(doc_text, chunk)
            
            indexed_chunks.append({
                "original_chunk": chunk,
                "contextualized_chunk": f"{context}\n\n{chunk}",
                "document_id": doc["id"],
                "context": context  # Store separately for display
            })
    
    return indexed_chunks
```

### Pattern 2: On-Demand Context

```python
def add_context_at_retrieval(
    chunks: list[dict],
    document_store: dict
) -> list[dict]:
    """Add context when retrieving (higher latency, lower storage)."""
    
    enhanced = []
    
    for chunk in chunks:
        doc_id = chunk["document_id"]
        full_doc = document_store.get(doc_id)
        
        if full_doc:
            context = generate_chunk_context(
                full_doc["content"],
                chunk["text"]
            )
            chunk["context"] = context
        
        enhanced.append(chunk)
    
    return enhanced
```

### Pattern 3: Hierarchical Context

```python
def hierarchical_context(
    chunk: str,
    section: str,
    document_title: str
) -> str:
    """Build context from document hierarchy instead of LLM."""
    
    # Fast alternative when LLM context too expensive
    context_parts = []
    
    if document_title:
        context_parts.append(f"Document: {document_title}")
    
    if section:
        context_parts.append(f"Section: {section}")
    
    context = " | ".join(context_parts)
    return f"{context}\n\n{chunk}"
```

---

## Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Prompt caching | 86% | Requires batch processing |
| Smaller model (Haiku) | 90% | Slightly lower quality |
| Hierarchical context | 99% | No semantic understanding |
| Hybrid (LLM for ambiguous) | 70% | Complexity |

```python
def smart_contextualization(
    document: str,
    chunk: str,
    section_title: Optional[str] = None
) -> str:
    """Use LLM only when needed."""
    
    # Check if chunk is self-contained
    ambiguous_patterns = [
        r'^(This|That|These|Those|It|They)\s',
        r'^(The|A|An)\s+(above|following|previous)',
        r'as (mentioned|discussed|shown)',
    ]
    
    needs_context = any(
        re.search(pattern, chunk, re.IGNORECASE)
        for pattern in ambiguous_patterns
    )
    
    if needs_context:
        # Use LLM for ambiguous chunks
        return generate_chunk_context(document, chunk) + "\n\n" + chunk
    elif section_title:
        # Use hierarchy for clear chunks
        return f"From section '{section_title}':\n\n{chunk}"
    else:
        return chunk
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use prompt caching for batches | Generate context per-request |
| Store context separately | Only store combined text |
| Combine with BM25 | Use embeddings alone |
| Cache generated contexts | Regenerate on every query |
| Use hierarchical fallback | Use LLM for every chunk |

---

## Summary

✅ **Context loss** is a major RAG failure mode—chunks lose meaning in isolation

✅ **Anthropic's method** prepends LLM-generated context to each chunk

✅ **35% improvement** with contextual embeddings, **67% with full stack**

✅ **Prompt caching** reduces cost by 86% for batch processing

✅ **Hybrid retrieval** (embeddings + BM25) outperforms either alone

**Next:** [Managed Chunking Services](./06-managed-chunking-services.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
-->
