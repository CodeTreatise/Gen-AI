---
title: "Contextual Retrieval"
---

# Contextual Retrieval

## Introduction

Anthropic's Contextual Retrieval technique adds context to chunks before indexing, dramatically improving retrieval accuracy. It solves the problem of chunks losing context when separated from their source document.

---

## The Problem: Lost Context

Traditional chunking loses document context:

```python
# Original document context
document = """
ACME Corporation Q2 2023 Financial Report

Executive Summary:
Revenue performance exceeded expectations...

Financial Highlights:
The company's revenue grew by 3% over the previous quarter.
Operating margins improved to 15.2%.
"""

# After chunking, this chunk loses context:
chunk = "The company's revenue grew by 3% over the previous quarter."
# Question: Which company? What quarter? What was the previous revenue?
```

When searching for "ACME Q2 revenue growth", the decontextualized chunk may not match well because it doesn't mention "ACME" or "Q2".

---

## Contextual Chunk Generation

Use Claude to add context before embedding:

```python
from anthropic import Anthropic

def generate_contextual_chunk(
    document: str,
    chunk: str,
    client: Anthropic
) -> str:
    """Generate context for a chunk using Claude."""
    
    prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    
    context = response.content[0].text.strip()
    return f"{context} {chunk}"

# Example
client = Anthropic()
document = """ACME Corporation Q2 2023 Financial Report
Executive Summary: Revenue performance exceeded expectations with 3% growth.
The previous quarter's revenue was $314 million.
Financial Highlights: The company's revenue grew by 3% over the previous quarter."""

chunk = "The company's revenue grew by 3% over the previous quarter."

contextualized = generate_contextual_chunk(document, chunk, client)
print(f"Contextualized chunk:\n{contextualized}")
```

**Output:**
```
Contextualized chunk:
This chunk is from ACME Corporation's Q2 2023 Financial Report, discussing financial highlights where the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter.
```

---

## Performance Impact

According to Anthropic's research:

| Method | Retrieval Failure Rate | Improvement |
|--------|------------------------|-------------|
| Standard embeddings | 5.7% | Baseline |
| Contextual Embeddings | 3.7% | 35% better |
| Contextual Embeddings + Contextual BM25 | 2.9% | 49% better |
| + Reranking | 1.9% | 67% better |

---

## Complete Contextual Pipeline

```python
from anthropic import Anthropic
from typing import Callable, Optional

class ContextualRAGPipeline:
    """Complete Contextual Retrieval pipeline."""
    
    def __init__(self, anthropic_client: Anthropic, embed_fn: Callable):
        self.client = anthropic_client
        self.embed_fn = embed_fn
        self.documents: list[str] = []
        self.contextualized_chunks: list[str] = []
        self.original_chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self.bm25_index = None
    
    def _chunk_document(self, text: str, chunk_size: int = 500) -> list[str]:
        """Simple chunking by sentences."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def index_document(self, document: str, chunk_size: int = 500):
        """Index document with contextual chunks."""
        # Chunk the document
        chunks = self._chunk_document(document, chunk_size)
        
        # Generate context for each chunk
        contextualized = []
        for chunk in chunks:
            ctx_chunk = generate_contextual_chunk(
                document, chunk, self.client
            )
            contextualized.append(ctx_chunk)
        
        # Generate embeddings for contextualized chunks
        embeddings = [self.embed_fn(chunk) for chunk in contextualized]
        
        # Store both original and contextualized
        self.documents.append(document)
        self.original_chunks.extend(chunks)
        self.contextualized_chunks.extend(contextualized)
        self.embeddings.extend(embeddings)
        
        # Rebuild BM25 index on contextualized chunks
        self._rebuild_bm25_index()
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from contextualized chunks."""
        from rank_bm25 import BM25Okapi
        tokenized = [chunk.lower().split() for chunk in self.contextualized_chunks]
        self.bm25_index = BM25Okapi(tokenized)
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        use_hybrid: bool = True
    ) -> list[dict]:
        """Search using contextual embeddings."""
        import numpy as np
        
        query_embedding = np.array(self.embed_fn(query))
        
        # Semantic search
        embeddings = np.array(self.embeddings)
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        if use_hybrid and self.bm25_index:
            # BM25 scores
            bm25_scores = self.bm25_index.get_scores(query.lower().split())
            
            # Normalize and combine
            sem_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
            
            combined = 0.5 * sem_norm + 0.5 * bm25_norm
            top_indices = np.argsort(combined)[::-1][:k]
        else:
            top_indices = np.argsort(similarities)[::-1][:k]
        
        return [
            {
                "original_chunk": self.original_chunks[i],
                "contextualized_chunk": self.contextualized_chunks[i],
                "score": float(similarities[i])
            }
            for i in top_indices
        ]
```

---

## Batch Processing with Caching

For production, batch context generation and cache results:

```python
import hashlib
import json
from pathlib import Path

class CachedContextualizer:
    """Cache contextual chunks to avoid repeated API calls."""
    
    def __init__(self, client: Anthropic, cache_dir: str = ".context_cache"):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _cache_key(self, document: str, chunk: str) -> str:
        """Generate cache key."""
        content = f"{document}|||{chunk}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_context(self, document: str, chunk: str) -> str:
        """Get context from cache or generate."""
        key = self._cache_key(document, chunk)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return data["contextualized"]
        
        # Generate and cache
        contextualized = generate_contextual_chunk(document, chunk, self.client)
        cache_file.write_text(json.dumps({
            "chunk": chunk,
            "contextualized": contextualized
        }))
        
        return contextualized
    
    def batch_contextualize(
        self, 
        document: str, 
        chunks: list[str],
        show_progress: bool = True
    ) -> list[str]:
        """Batch process chunks with caching."""
        results = []
        
        for i, chunk in enumerate(chunks):
            if show_progress:
                print(f"Processing chunk {i+1}/{len(chunks)}", end="\r")
            results.append(self.get_context(document, chunk))
        
        if show_progress:
            print()  # New line after progress
        
        return results
```

---

## Cost Considerations

| Component | Cost Factor |
|-----------|-------------|
| Context generation | ~50-100 tokens per chunk (Haiku) |
| Embedding | Same as standard (contextualized text is longer) |
| Storage | ~1.5x (longer chunks) |
| Indexing | One-time, cached |

> **Tip:** Use Claude Haiku for context generation—it's fast and cheap while producing quality context.

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Cache contextualized chunks | Regenerate on every index |
| Use Haiku for cost efficiency | Use Opus for simple context |
| Combine with BM25 for best results | Use only semantic search |
| Keep original chunks for display | Show only contextualized text |

---

## Summary

✅ **Contextual chunks solve the lost context problem**

✅ **49% improvement** when combined with contextual BM25

✅ **67% improvement** when adding reranking

✅ **Cache aggressively** to avoid repeated API costs

**Next:** [Reranking](./08-reranking.md)
