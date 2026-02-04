---
title: "Overlap Strategies"
---

# Overlap Strategies

## Introduction

Overlap ensures that information split across chunk boundaries isn't lost. When a key concept spans two chunks, overlap guarantees at least one chunk contains the complete thought.

---

## Why Overlap Helps

Without overlap, boundaries can split important information:

```python
# Without overlap - information split
chunk_1 = "The API requires authentication. Use Bearer tokens"
chunk_2 = "in the Authorization header. Tokens expire after 1 hour."

# Query: "How do I authenticate with the API?"
# Neither chunk contains complete answer

# With overlap - complete information preserved
chunk_1 = "The API requires authentication. Use Bearer tokens in the Authorization header."
chunk_2 = "Use Bearer tokens in the Authorization header. Tokens expire after 1 hour."

# Now chunk_1 contains the complete auth method
```

---

## Overlap Percentage Guidelines

| Overlap | Tokens (for 500 token chunk) | Trade-off |
|---------|------------------------------|-----------|
| 0% | 0 | May lose boundary context |
| 10% | 50 | Minimal redundancy, good start |
| 15-20% | 75-100 | **Recommended for most cases** |
| 25% | 125 | High redundancy, expensive |
| 50% | 250 | Excessive, doubles storage |

```python
def calculate_overlap(
    chunk_size: int,
    overlap_percentage: float = 0.15
) -> int:
    """Calculate overlap tokens from percentage."""
    return int(chunk_size * overlap_percentage)

# Typical configurations
standard_overlap = calculate_overlap(500, 0.15)   # 75 tokens
aggressive_overlap = calculate_overlap(500, 0.25) # 125 tokens
minimal_overlap = calculate_overlap(500, 0.10)    # 50 tokens
```

---

## Sliding Window Implementation

The sliding window is the most common overlap approach:

```python
import tiktoken

def sliding_window_chunker(
    text: str,
    chunk_size: int = 500,
    overlap: int = 75,
    encoding: str = "cl100k_base"
) -> list[str]:
    """Chunk text with sliding window overlap."""
    
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Slide window by (chunk_size - overlap)
        start += chunk_size - overlap
    
    return chunks

# Example
text = "A " * 1000  # 1000 tokens
chunks = sliding_window_chunker(text, chunk_size=300, overlap=50)
print(f"Created {len(chunks)} chunks")  # 4 chunks with overlap
```

---

## Sentence-Aware Overlap

Better than fixed-token overlap: respect sentence boundaries in the overlap region:

```python
import re

def sentence_aware_chunker(
    text: str,
    target_chunk_size: int = 500,
    overlap_sentences: int = 2
) -> list[str]:
    """Chunk with sentence-boundary aware overlap."""
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = estimate_tokens(sentence)
        
        if current_size + sentence_tokens > target_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap (last N sentences)
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_size = sum(estimate_tokens(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def estimate_tokens(text: str) -> int:
    """Quick token estimate."""
    return len(text) // 4  # Approximate for English
```

---

## Context Preservation Techniques

Beyond simple overlap, preserve context explicitly:

### 1. Prepend Section Headers

```python
def chunk_with_headers(
    text: str,
    chunk_size: int = 500
) -> list[dict]:
    """Preserve section context in each chunk."""
    
    # Track current section hierarchy
    current_h1 = ""
    current_h2 = ""
    
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        # Track headers
        if line.startswith('# '):
            current_h1 = line[2:].strip()
            current_h2 = ""
        elif line.startswith('## '):
            current_h2 = line[3:].strip()
        
        line_tokens = estimate_tokens(line)
        
        if current_size + line_tokens > chunk_size and current_chunk:
            # Create chunk with header context
            header_context = f"[{current_h1}]"
            if current_h2:
                header_context += f" [{current_h2}]"
            
            chunks.append({
                "text": '\n'.join(current_chunk),
                "context": header_context,
                "full_text": f"{header_context}\n\n" + '\n'.join(current_chunk)
            })
            
            current_chunk = []
            current_size = 0
        
        current_chunk.append(line)
        current_size += line_tokens
    
    return chunks
```

### 2. Link Adjacent Chunks

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LinkedChunk:
    id: str
    text: str
    prev_id: Optional[str]
    next_id: Optional[str]
    prev_summary: Optional[str]  # Last sentence of previous chunk
    next_preview: Optional[str]  # First sentence of next chunk

def create_linked_chunks(
    chunks: list[str]
) -> list[LinkedChunk]:
    """Create chunks with navigation context."""
    
    def get_last_sentence(text: str) -> str:
        sentences = text.rstrip('.!?').rsplit('.', 1)
        return sentences[-1].strip() + '.' if len(sentences) > 1 else ""
    
    def get_first_sentence(text: str) -> str:
        match = re.match(r'^[^.!?]+[.!?]', text)
        return match.group() if match else ""
    
    linked = []
    for i, text in enumerate(chunks):
        linked.append(LinkedChunk(
            id=f"chunk_{i}",
            text=text,
            prev_id=f"chunk_{i-1}" if i > 0 else None,
            next_id=f"chunk_{i+1}" if i < len(chunks)-1 else None,
            prev_summary=get_last_sentence(chunks[i-1]) if i > 0 else None,
            next_preview=get_first_sentence(chunks[i+1]) if i < len(chunks)-1 else None
        ))
    
    return linked
```

---

## Overlap vs Chunk Size Trade-offs

More overlap means more storage and embeddings:

```python
def calculate_storage_impact(
    document_tokens: int,
    chunk_size: int,
    overlap: int
) -> dict:
    """Calculate storage and cost impact of overlap."""
    
    step_size = chunk_size - overlap
    num_chunks = (document_tokens - overlap) // step_size + 1
    
    total_tokens_stored = num_chunks * chunk_size
    redundancy_ratio = total_tokens_stored / document_tokens
    
    # Embedding cost (text-embedding-3-small: $0.02 per 1M tokens)
    embedding_cost = (total_tokens_stored / 1_000_000) * 0.02
    
    return {
        "num_chunks": num_chunks,
        "total_tokens": total_tokens_stored,
        "redundancy_ratio": redundancy_ratio,
        "embedding_cost": embedding_cost
    }

# Compare configurations
doc_size = 100_000  # tokens

no_overlap = calculate_storage_impact(doc_size, 500, 0)
# {"num_chunks": 200, "redundancy_ratio": 1.0}

with_overlap = calculate_storage_impact(doc_size, 500, 100)
# {"num_chunks": 250, "redundancy_ratio": 1.25}

high_overlap = calculate_storage_impact(doc_size, 500, 250)
# {"num_chunks": 400, "redundancy_ratio": 2.0}
```

---

## When to Adjust Overlap

| Scenario | Recommended Overlap |
|----------|---------------------|
| Highly structured (clear sections) | 10% or less |
| Flowing prose | 15-20% |
| Technical documentation | 20% |
| Legal/regulatory text | 25% |
| Code with comments | 10-15% |
| Conversational/chat | 20% |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use sentence-aware overlap | Split mid-sentence |
| Preserve section context | Lose heading information |
| Consider storage costs | Use 50% overlap unnecessarily |
| Link adjacent chunks | Treat chunks as isolated |
| Test retrieval quality | Assume more overlap is better |

---

## Summary

✅ **15-20% overlap** is the sweet spot for most use cases

✅ **Sentence-aware overlap** produces cleaner boundaries than fixed-token

✅ **Preserve context** by prepending headers or linking chunks

✅ **Balance quality vs cost** - high overlap increases storage and embeddings

✅ **Adjust for content type** - structured content needs less overlap

**Next:** [Structure-Based Chunking](./04-structure-based-chunking.md)
