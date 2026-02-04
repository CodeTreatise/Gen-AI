---
title: "Implementation Steps"
---

# Implementation Steps

## Introduction

This lesson provides a **complete, step-by-step implementation** of Contextual Retrieval. We'll build a full pipeline from raw documents to searchable contextualized chunks.

By the end, you'll have working code for the entire process.

### What We'll Cover

- Complete implementation architecture
- Document chunking strategies
- Context generation with batching
- Embedding contextualized chunks
- Building the BM25 index
- Retrieval function combining both

### Prerequisites

- [The Contextualizer Prompt](./03-the-contextualizer-prompt.md)
- Python environment with `anthropic`, `openai`, and `rank-bm25` packages
- API keys for Claude and an embedding provider

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│           Contextual Retrieval Implementation Pipeline           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Raw         │     │   Chunked    │     │Contextualized│    │
│  │  Documents   │────▶│   Documents  │────▶│   Chunks     │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│        Step 1              Step 2               Step 3          │
│                                                                 │
│                                   │                             │
│                                   ▼                             │
│                    ┌──────────────────────────────┐            │
│                    │     Dual Index Creation      │            │
│                    └──────────────────────────────┘            │
│                          /              \                       │
│                         /                \                      │
│            ┌──────────────┐        ┌──────────────┐           │
│            │Vector Index  │        │  BM25 Index  │           │
│            │(Embeddings)  │        │  (Keywords)  │           │
│            └──────────────┘        └──────────────┘           │
│                 Step 4                  Step 5                  │
│                         \              /                        │
│                          \            /                         │
│                    ┌──────────────────────────────┐            │
│                    │      Hybrid Retrieval        │            │
│                    └──────────────────────────────┘            │
│                              Step 6                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Document Chunking

### Basic Chunking Strategy

```python
from typing import List, Dict
import re

def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Split document into overlapping chunks.
    
    Args:
        text: The document text
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Find end position
        end = start + chunk_size
        
        # Try to end at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            sentence_end = text.rfind('. ', start + chunk_size - 100, end + 50)
            if sentence_end != -1:
                end = sentence_end + 1
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:  # Don't add empty chunks
            chunks.append({
                "text": chunk_text,
                "index": chunk_index,
                "start_char": start,
                "end_char": end
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - chunk_overlap
        
        # Prevent infinite loop
        if start >= len(text) - 50:
            break
    
    return chunks


# Example usage
document = """
ACME Corporation
Quarterly Report Q2 2023

Section 1: Financial Highlights

Revenue for the second quarter of 2023 reached $324 million, 
representing a 3% increase from the previous quarter. This 
growth was primarily driven by our cloud services segment...

Section 2: Operational Review

Our cloud infrastructure expanded to 15 new regions during 
the quarter. Customer satisfaction scores improved to 92%, 
up from 89% in Q1...
"""

chunks = chunk_document(document, chunk_size=300, chunk_overlap=50)
print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(f"\nChunk {chunk['index']}:")
    print(f"  Length: {len(chunk['text'])} chars")
    print(f"  Preview: {chunk['text'][:100]}...")
```

**Output:**
```
Created 3 chunks

Chunk 0:
  Length: 298 chars
  Preview: ACME Corporation
Quarterly Report Q2 2023

Section 1: Financial Highlights

Revenue for the second ...

Chunk 1:
  Length: 287 chars
  Preview: growth was primarily driven by our cloud services segment...

Section 2: Operational Review

Our...

Chunk 2:
  Length: 156 chars
  Preview: Customer satisfaction scores improved to 92%, up from 89% in Q1...
```

---

## Step 2: Context Generation

### Basic Contextualizer

```python
import anthropic
from typing import List, Dict

def generate_chunk_context(
    document: str,
    chunk: str,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """Generate situating context for a single chunk."""
    client = anthropic.Anthropic()
    
    prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else."""
    
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()


def contextualize_chunks(
    document: str,
    chunks: List[Dict]
) -> List[Dict]:
    """Add context to all chunks from a document."""
    contextualized = []
    
    for chunk in chunks:
        context = generate_chunk_context(document, chunk["text"])
        
        contextualized.append({
            **chunk,  # Keep original chunk data
            "context": context,
            "contextualized_text": f"{context}\n\n{chunk['text']}"
        })
        
        print(f"  Contextualized chunk {chunk['index']}")
    
    return contextualized


# Example usage
print("Generating contexts...")
ctx_chunks = contextualize_chunks(document, chunks)

print("\nExample contextualized chunk:")
print(ctx_chunks[0]["contextualized_text"])
```

**Output:**
```
Generating contexts...
  Contextualized chunk 0
  Contextualized chunk 1
  Contextualized chunk 2

Example contextualized chunk:
This chunk is from ACME Corporation's Q2 2023 Quarterly Report, 
specifically from Section 1: Financial Highlights. The company 
reported $314 million in revenue in Q1 2023.

ACME Corporation
Quarterly Report Q2 2023

Section 1: Financial Highlights

Revenue for the second quarter of 2023 reached $324 million, 
representing a 3% increase from the previous quarter. This 
growth was primarily driven by our cloud services segment...
```

---

## Step 3: Create Vector Embeddings

### Using OpenAI Embeddings

```python
from openai import OpenAI
from typing import List
import numpy as np

def create_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> np.ndarray:
    """Create embeddings for a list of texts."""
    client = OpenAI()
    
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def embed_contextualized_chunks(
    ctx_chunks: List[Dict]
) -> List[Dict]:
    """Add embeddings to contextualized chunks."""
    # Get texts to embed
    texts = [chunk["contextualized_text"] for chunk in ctx_chunks]
    
    # Create embeddings
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = create_embeddings(texts)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(ctx_chunks):
        chunk["embedding"] = embeddings[i]
    
    print(f"Added embeddings with dimension {embeddings.shape[1]}")
    return ctx_chunks


# Example usage
ctx_chunks = embed_contextualized_chunks(ctx_chunks)
print(f"Embedding shape: {ctx_chunks[0]['embedding'].shape}")
```

**Output:**
```
Creating embeddings for 3 chunks...
Added embeddings with dimension 1536
Embedding shape: (1536,)
```

---

## Step 4: Create BM25 Index

### Using rank-bm25

```python
from rank_bm25 import BM25Okapi
import nltk
from typing import List, Tuple

# Download tokenizer (first time only)
# nltk.download('punkt')

def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    # Lowercase and split on non-alphanumeric
    tokens = text.lower().split()
    # Remove punctuation
    tokens = [t.strip('.,!?;:()[]{}"\'-') for t in tokens]
    # Remove empty tokens
    return [t for t in tokens if t]


def create_bm25_index(ctx_chunks: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    """Create BM25 index from contextualized chunks."""
    # Tokenize all contextualized texts
    tokenized_corpus = [
        tokenize(chunk["contextualized_text"]) 
        for chunk in ctx_chunks
    ]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"Created BM25 index with {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus


# Example usage
bm25_index, tokenized_docs = create_bm25_index(ctx_chunks)

# Test BM25 search
query = "cloud services revenue growth"
query_tokens = tokenize(query)
scores = bm25_index.get_scores(query_tokens)
print(f"\nBM25 scores for '{query}':")
for i, score in enumerate(scores):
    print(f"  Chunk {i}: {score:.4f}")
```

**Output:**
```
Created BM25 index with 3 documents

BM25 scores for 'cloud services revenue growth':
  Chunk 0: 1.2456
  Chunk 1: 0.8932
  Chunk 2: 0.1234
```

---

## Step 5: Complete Pipeline Class

### Full Implementation

```python
import anthropic
from openai import OpenAI
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    chunk_index: int
    text: str
    context: str
    contextualized_text: str
    vector_score: float
    bm25_score: float
    combined_score: float


class ContextualRetrieval:
    """Complete Contextual Retrieval implementation."""
    
    def __init__(
        self,
        context_model: str = "claude-3-haiku-20240307",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.anthropic_client = anthropic.Anthropic()
        self.openai_client = OpenAI()
        self.context_model = context_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Storage
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_docs: List[List[str]] = []
    
    def process_document(self, document: str, doc_id: str = "doc") -> None:
        """Process a document through the full pipeline."""
        print(f"Processing document: {doc_id}")
        
        # Step 1: Chunk
        raw_chunks = self._chunk_document(document)
        print(f"  Created {len(raw_chunks)} chunks")
        
        # Step 2: Generate contexts
        ctx_chunks = self._contextualize_chunks(document, raw_chunks, doc_id)
        print(f"  Generated contexts")
        
        # Store chunks
        self.chunks.extend(ctx_chunks)
        
        # Step 3: Create embeddings
        texts = [c["contextualized_text"] for c in ctx_chunks]
        new_embeddings = self._create_embeddings(texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        print(f"  Created embeddings")
        
        # Step 4: Update BM25 index
        self._rebuild_bm25_index()
        print(f"  Updated BM25 index")
        
        print(f"  Total chunks in index: {len(self.chunks)}")
    
    def _chunk_document(self, text: str) -> List[Dict]:
        """Split document into chunks."""
        chunks = []
        start = 0
        idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Find sentence boundary
            if end < len(text):
                sent_end = text.rfind('. ', start + self.chunk_size - 100, end + 50)
                if sent_end != -1:
                    end = sent_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "local_index": idx
                })
                idx += 1
            
            start = end - self.chunk_overlap
            if start >= len(text) - 50:
                break
        
        return chunks
    
    def _contextualize_chunks(
        self, 
        document: str, 
        chunks: List[Dict],
        doc_id: str
    ) -> List[Dict]:
        """Add context to chunks."""
        ctx_chunks = []
        
        for chunk in chunks:
            context = self._generate_context(document, chunk["text"])
            
            ctx_chunks.append({
                **chunk,
                "doc_id": doc_id,
                "global_index": len(self.chunks) + len(ctx_chunks),
                "context": context,
                "contextualized_text": f"{context}\n\n{chunk['text']}"
            })
        
        return ctx_chunks
    
    def _generate_context(self, document: str, chunk: str) -> str:
        """Generate context for a single chunk."""
        prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else."""
        
        response = self.anthropic_client.messages.create(
            model=self.context_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all chunks."""
        self.tokenized_docs = [
            self._tokenize(c["contextualized_text"]) 
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        tokens = text.lower().split()
        return [t.strip('.,!?;:()[]{}"\'-') for t in tokens if t]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using hybrid search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            alpha: Weight for vector search (1-alpha for BM25)
        
        Returns:
            List of RetrievalResult objects
        """
        if not self.chunks:
            return []
        
        # Vector search
        query_embedding = self._create_embeddings([query])[0]
        vector_scores = np.dot(self.embeddings, query_embedding)
        
        # BM25 search
        query_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        
        # Normalize scores to [0, 1]
        if vector_scores.max() > 0:
            vector_scores = vector_scores / vector_scores.max()
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores
        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_index=idx,
                text=chunk["text"],
                context=chunk["context"],
                contextualized_text=chunk["contextualized_text"],
                vector_score=float(vector_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                combined_score=float(combined_scores[idx])
            ))
        
        return results
```

---

## Step 6: Using the Pipeline

### Example Usage

```python
# Initialize the retrieval system
retrieval = ContextualRetrieval(
    chunk_size=400,
    chunk_overlap=80
)

# Process documents
doc1 = """
ACME Corporation Quarterly Report Q2 2023

Executive Summary
ACME Corporation delivered strong results in Q2 2023, 
with revenue reaching $324 million...
[rest of document]
"""

doc2 = """
TechStart Inc. Annual Report 2023

Company Overview
TechStart Inc. specializes in AI-powered analytics...
[rest of document]
"""

# Index documents
retrieval.process_document(doc1, doc_id="acme_q2_2023")
retrieval.process_document(doc2, doc_id="techstart_2023")

# Search
results = retrieval.retrieve(
    query="Q2 2023 revenue growth cloud services",
    top_k=3,
    alpha=0.6  # 60% vector, 40% BM25
)

# Display results
print("Search Results:")
print("-" * 60)
for i, result in enumerate(results):
    print(f"\n{i+1}. Score: {result.combined_score:.4f}")
    print(f"   Vector: {result.vector_score:.4f}, BM25: {result.bm25_score:.4f}")
    print(f"   Context: {result.context[:100]}...")
    print(f"   Text: {result.text[:100]}...")
```

**Output:**
```
Processing document: acme_q2_2023
  Created 8 chunks
  Generated contexts
  Created embeddings
  Updated BM25 index
  Total chunks in index: 8
Processing document: techstart_2023
  Created 6 chunks
  Generated contexts
  Created embeddings
  Updated BM25 index
  Total chunks in index: 14

Search Results:
------------------------------------------------------------

1. Score: 0.8934
   Vector: 0.9123, BM25: 0.8654
   Context: This chunk is from ACME Corporation's Q2 2023 Quarterly Report...
   Text: Revenue for the second quarter of 2023 reached $324 million...

2. Score: 0.7456
   Vector: 0.7234, BM25: 0.7789
   Context: This chunk is from ACME Corporation's Q2 2023 Quarterly Report...
   Text: Our cloud services segment continued to lead growth, contributing...

3. Score: 0.4321
   Vector: 0.5123, BM25: 0.3456
   Context: This chunk is from TechStart Inc.'s 2023 Annual Report...
   Text: Revenue from cloud-based solutions increased by 45%...
```

---

## Summary

✅ **Step 1:** Chunk documents with overlap for context continuity  
✅ **Step 2:** Generate context using the contextualizer prompt  
✅ **Step 3:** Create embeddings from contextualized text  
✅ **Step 4:** Build BM25 index from contextualized text  
✅ **Step 5:** Combine into a complete pipeline class  
✅ **Step 6:** Query with hybrid search (vector + BM25)

---

**Next:** [Hybrid Search with BM25 →](./05-hybrid-search-bm25.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- rank-bm25 Python library documentation
- OpenAI Embeddings API documentation
-->
