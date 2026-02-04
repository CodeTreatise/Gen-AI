---
title: "Late Chunking"
---

# Late Chunking

## Introduction

Late chunking inverts traditional RAG: instead of chunking first and embedding each chunk, we embed the full document and then extract chunk representations from the contextualized token embeddings. This preserves document-wide context in every chunk.

---

## Traditional vs Late Chunking

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRADITIONAL CHUNKING                          │
├─────────────────────────────────────────────────────────────────┤
│  Document → Chunk 1 → Embed → [0.2, 0.4, ...]                  │
│           → Chunk 2 → Embed → [0.3, 0.1, ...]   (no context)   │
│           → Chunk 3 → Embed → [0.5, 0.2, ...]                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     LATE CHUNKING                               │
├─────────────────────────────────────────────────────────────────┤
│  Document → Embed full doc → Token embeddings                   │
│                            → Pool chunk 1 tokens → [0.2, 0.4]  │
│                            → Pool chunk 2 tokens → [0.3, 0.1]  │
│                            → Pool chunk 3 tokens → [0.5, 0.2]  │
│                                                                 │
│  Each chunk embedding carries context from entire document!    │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

Late chunking requires models that output **per-token embeddings**:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Models that support token-level embeddings
# - jinaai/jina-embeddings-v2-base-en
# - nomic-ai/nomic-embed-text-v1.5
# - colbert models

model_name = "jinaai/jina-embeddings-v2-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

def late_chunking(
    document: str,
    chunk_boundaries: list[tuple[int, int]],
    pooling: str = "mean"
) -> list[list[float]]:
    """
    Embed full document, then extract chunk embeddings.
    
    Args:
        document: Full document text
        chunk_boundaries: List of (start_char, end_char) tuples
        pooling: How to combine token embeddings ("mean" or "max")
    
    Returns:
        List of chunk embeddings
    """
    
    # Tokenize full document
    inputs = tokenizer(
        document,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True  # Map tokens to characters
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0]
    
    # Get token embeddings for full document
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]
    
    chunk_embeddings = []
    
    for start_char, end_char in chunk_boundaries:
        # Find tokens that fall within this chunk
        token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start >= start_char and tok_end <= end_char:
                token_indices.append(idx)
        
        if token_indices:
            # Pool token embeddings for this chunk
            chunk_tokens = token_embeddings[token_indices]
            
            if pooling == "mean":
                chunk_emb = chunk_tokens.mean(dim=0)
            elif pooling == "max":
                chunk_emb = chunk_tokens.max(dim=0).values
            
            chunk_embeddings.append(chunk_emb.tolist())
    
    return chunk_embeddings
```

---

## Jina AI Late Chunking

Jina AI provides late chunking through their embeddings API:

```python
import requests

def jina_late_chunking(
    document: str,
    chunk_texts: list[str]
) -> list[list[float]]:
    """Use Jina's late chunking via API."""
    
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": "Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        },
        json={
            "model": "jina-embeddings-v3",
            "input": [document],  # Full document
            "late_chunking": True,
            "chunk_spans": [
                {"start": 0, "end": 500},
                {"start": 450, "end": 950},
                # ... more chunk boundaries
            ]
        }
    )
    
    return response.json()["data"]
```

---

## ColBERT-Style Late Interaction

ColBERT extends late chunking to queries, comparing token-by-token:

```python
import torch
from transformers import AutoModel, AutoTokenizer

class ColBERTRetriever:
    """Late interaction retrieval with per-token matching."""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode_document(self, document: str) -> torch.Tensor:
        """Get per-token embeddings for document."""
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0]  # [seq_len, hidden]
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Get per-token embeddings for query."""
        inputs = self.tokenizer(
            f"[Q] {query}",  # Query marker
            return_tensors="pt",
            truncation=True,
            max_length=32
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0]
    
    def score(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor
    ) -> float:
        """
        MaxSim scoring: for each query token, find max similarity
        to any document token, then sum.
        """
        # Cosine similarity matrix [query_len, doc_len]
        query_norm = query_emb / query_emb.norm(dim=1, keepdim=True)
        doc_norm = doc_emb / doc_emb.norm(dim=1, keepdim=True)
        similarity = torch.mm(query_norm, doc_norm.T)
        
        # Max over document tokens for each query token
        max_sims = similarity.max(dim=1).values
        
        # Sum of max similarities
        return max_sims.sum().item()
```

---

## Implementation Considerations

### Memory Requirements

| Approach | Memory per 1000 tokens |
|----------|------------------------|
| Standard embedding | ~6 KB (1536 dims) |
| Late chunking (store tokens) | ~3 MB (1000 × 768 dims) |
| Late chunking (compute on-fly) | ~6 KB + inference time |

### When to Store Token Embeddings

```python
def decide_storage_strategy(
    num_documents: int,
    avg_doc_length: int,
    query_latency_requirement: float
) -> str:
    """Choose between storing token embeddings or recomputing."""
    
    estimated_storage_gb = (
        num_documents * avg_doc_length * 768 * 4  # float32
    ) / (1024 ** 3)
    
    if estimated_storage_gb > 100:
        return "recompute"  # Storage too expensive
    elif query_latency_requirement < 0.1:  # 100ms
        return "store"  # Need fast retrieval
    else:
        return "hybrid"  # Store popular docs, recompute rest
```

---

## Practical Late Chunking Pipeline

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class LateChunk:
    text: str
    embedding: list[float]
    doc_id: str
    start_char: int
    end_char: int

class LateChunkingPipeline:
    """Production late chunking with caching."""
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-en",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(
        self,
        doc_id: str,
        text: str
    ) -> list[LateChunk]:
        """Process document with late chunking."""
        
        # Define chunk boundaries
        boundaries = self._create_boundaries(text)
        
        # Get late chunk embeddings
        embeddings = self._late_embed(text, boundaries)
        
        # Create chunk objects
        chunks = []
        for i, ((start, end), embedding) in enumerate(
            zip(boundaries, embeddings)
        ):
            chunks.append(LateChunk(
                text=text[start:end],
                embedding=embedding,
                doc_id=doc_id,
                start_char=start,
                end_char=end
            ))
        
        return chunks
    
    def _create_boundaries(
        self,
        text: str
    ) -> list[tuple[int, int]]:
        """Create overlapping chunk boundaries."""
        boundaries = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Adjust to word boundary
            if end < len(text):
                space_idx = text.rfind(' ', start, end)
                if space_idx > start:
                    end = space_idx
            
            boundaries.append((start, end))
            start = end - self.chunk_overlap
        
        return boundaries
    
    def _late_embed(
        self,
        text: str,
        boundaries: list[tuple[int, int]]
    ) -> list[list[float]]:
        """Generate embeddings using late chunking."""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embs = outputs.last_hidden_state[0]
        
        chunk_embeddings = []
        
        for start_char, end_char in boundaries:
            # Find tokens in range
            indices = [
                i for i, (ts, te) in enumerate(offset_mapping)
                if ts >= start_char and te <= end_char and ts != te
            ]
            
            if indices:
                chunk_tokens = token_embs[indices]
                chunk_emb = chunk_tokens.mean(dim=0)
                chunk_embeddings.append(chunk_emb.tolist())
            else:
                # Fallback: use CLS token
                chunk_embeddings.append(token_embs[0].tolist())
        
        return chunk_embeddings
```

---

## Comparison with Contextual Chunking

| Aspect | Contextual Chunking | Late Chunking |
|--------|--------------------|--------------| 
| Context source | LLM-generated text | Transformer attention |
| Cost | API calls per chunk | Single embedding call |
| Latency | Higher (LLM calls) | Lower (single pass) |
| Model requirements | Any embedding model | Token-level embeddings |
| Storage | Standard vectors | Standard vectors |
| Quality | Excellent | Good (model dependent) |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use models with long context (8K+) | Use 512-token models |
| Pool with mean for general use | Always use max pooling |
| Define clear chunk boundaries | Let boundaries split words |
| Cache document embeddings | Re-embed on every query |
| Benchmark against standard | Assume late is always better |

---

## Summary

✅ **Late chunking** embeds full document first, extracts chunk representations after

✅ **Each chunk carries context** from the entire document via attention

✅ **Lower cost** than contextual chunking (no LLM calls)

✅ **Requires special models** with per-token embeddings (Jina, ColBERT)

✅ **ColBERT extends** this to query-document late interaction

**Next:** [Semantic Chunking](./08-semantic-chunking.md)
