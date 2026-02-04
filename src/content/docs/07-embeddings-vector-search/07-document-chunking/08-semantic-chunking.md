---
title: "Semantic Chunking"
---

# Semantic Chunking

## Introduction

Semantic chunking uses AI to detect where topics change, splitting at natural boundaries rather than fixed positions. This creates chunks that contain complete, coherent ideas instead of arbitrarily sliced text.

---

## How Semantic Chunking Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIXED SIZE CHUNKING                          │
├─────────────────────────────────────────────────────────────────┤
│ "Machine learning models need data. | Neural networks are      │
│  inspired by the brain. They use lay│ers of connected nodes..."│
│                                     ↑                           │
│                            Arbitrary cut mid-sentence           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC CHUNKING                            │
├─────────────────────────────────────────────────────────────────┤
│ Chunk 1: "Machine learning models need data."                   │
│ Chunk 2: "Neural networks are inspired by the brain. They      │
│           use layers of connected nodes..."                     │
│                     ↑                                           │
│            Topic change detected                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Embedding-Based Breakpoint Detection

The core algorithm: embed sentences, find where similarity drops:

```python
from openai import OpenAI
import numpy as np
from typing import Optional

client = OpenAI()

def semantic_chunker(
    text: str,
    breakpoint_threshold: float = 0.3,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500
) -> list[str]:
    """
    Split text at semantic breakpoints.
    
    Args:
        text: Input text
        breakpoint_threshold: Similarity drop to trigger split (0-1)
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
    """
    
    # Split into sentences
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return [text]
    
    # Embed all sentences
    embeddings = embed_sentences(sentences)
    
    # Calculate similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    
    # Find breakpoints (significant drops in similarity)
    breakpoints = detect_breakpoints(similarities, breakpoint_threshold)
    
    # Create chunks respecting size limits
    chunks = create_chunks(sentences, breakpoints, min_chunk_size, max_chunk_size)
    
    return chunks

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def embed_sentences(sentences: list[str]) -> list[list[float]]:
    """Batch embed sentences."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences
    )
    return [e.embedding for e in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_breakpoints(
    similarities: list[float],
    threshold: float
) -> list[int]:
    """Find indices where similarity drops below threshold."""
    
    # Calculate mean and std for dynamic threshold
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    # Breakpoint if similarity is threshold std deviations below mean
    dynamic_threshold = mean_sim - (threshold * std_sim)
    
    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < dynamic_threshold:
            breakpoints.append(i + 1)  # Split after sentence i
    
    return breakpoints

def create_chunks(
    sentences: list[str],
    breakpoints: list[int],
    min_size: int,
    max_size: int
) -> list[str]:
    """Create chunks from sentences and breakpoints."""
    
    chunks = []
    current_sentences = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence) // 4  # Rough token estimate
        
        # Check if we should split here
        should_split = (
            i in breakpoints and 
            current_size >= min_size
        ) or (
            current_size + sentence_size > max_size and 
            current_sentences
        )
        
        if should_split:
            chunks.append(' '.join(current_sentences))
            current_sentences = []
            current_size = 0
        
        current_sentences.append(sentence)
        current_size += sentence_size
    
    # Don't forget last chunk
    if current_sentences:
        chunks.append(' '.join(current_sentences))
    
    return chunks
```

---

## LlamaIndex SemanticSplitter

LlamaIndex provides a production-ready implementation:

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document

# Create semantic splitter
embed_model = OpenAIEmbedding()

splitter = SemanticSplitterNodeParser(
    buffer_size=1,              # Sentences to compare
    breakpoint_percentile_threshold=95,  # Sensitivity
    embed_model=embed_model
)

# Process document
document = Document(text=your_text)
nodes = splitter.get_nodes_from_documents([document])

# Access chunks
for node in nodes:
    print(f"Chunk: {node.get_content()[:100]}...")
    print(f"Metadata: {node.metadata}")
```

**Configuration options:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `buffer_size` | 1 | Sentences compared at once |
| `breakpoint_percentile_threshold` | 95 | Higher = fewer splits |
| `embed_model` | Required | Model for embeddings |

---

## Percentile vs Absolute Threshold

```python
def percentile_breakpoints(
    similarities: list[float],
    percentile: int = 95
) -> list[int]:
    """
    Find breakpoints using percentile threshold.
    
    Percentile-based is more robust across different documents:
    - 95th percentile: Very few splits (only major topic changes)
    - 80th percentile: More splits (paragraph-level changes)
    - 50th percentile: Many splits (sentence-level)
    """
    
    # Calculate the threshold as the (100 - percentile) percentile
    # Lower similarities indicate breakpoints
    threshold = np.percentile(similarities, 100 - percentile)
    
    breakpoints = [
        i + 1 for i, sim in enumerate(similarities)
        if sim <= threshold
    ]
    
    return breakpoints

def absolute_breakpoints(
    similarities: list[float],
    threshold: float = 0.5
) -> list[int]:
    """
    Find breakpoints using absolute threshold.
    
    Less robust: a good threshold varies by content type.
    """
    return [
        i + 1 for i, sim in enumerate(similarities)
        if sim < threshold
    ]
```

---

## Windowed Similarity

Compare groups of sentences for more stable detection:

```python
def windowed_semantic_chunker(
    text: str,
    window_size: int = 3,
    step_size: int = 1,
    threshold_percentile: int = 90
) -> list[str]:
    """Use sliding windows for more stable breakpoint detection."""
    
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences)
    
    # Create window embeddings (average of sentences in window)
    window_embeddings = []
    window_indices = []
    
    for i in range(0, len(sentences) - window_size + 1, step_size):
        window_emb = np.mean(embeddings[i:i + window_size], axis=0)
        window_embeddings.append(window_emb)
        window_indices.append(i + window_size // 2)  # Center of window
    
    # Calculate similarities between consecutive windows
    similarities = []
    for i in range(len(window_embeddings) - 1):
        sim = cosine_similarity(
            window_embeddings[i],
            window_embeddings[i + 1]
        )
        similarities.append(sim)
    
    # Find breakpoints
    threshold = np.percentile(similarities, 100 - threshold_percentile)
    breakpoints = set()
    
    for i, sim in enumerate(similarities):
        if sim <= threshold:
            # Map back to sentence index
            breakpoints.add(window_indices[i + 1])
    
    # Create chunks
    chunks = []
    start = 0
    for bp in sorted(breakpoints):
        chunk = ' '.join(sentences[start:bp])
        if chunk:
            chunks.append(chunk)
        start = bp
    
    # Last chunk
    if start < len(sentences):
        chunks.append(' '.join(sentences[start:]))
    
    return chunks
```

---

## Hierarchical Semantic Chunking

Create chunks at multiple granularities:

```python
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)

def hierarchical_semantic_chunker(
    text: str,
    levels: list[int] = [2048, 512, 128]
) -> dict[str, list[str]]:
    """Create semantic chunks at multiple size levels."""
    
    embed_model = OpenAIEmbedding()
    
    results = {}
    
    for level_size in levels:
        # Adjust threshold based on target size
        # Larger chunks = higher threshold (fewer splits)
        threshold = 95 - (2048 - level_size) // 50
        threshold = max(70, min(98, threshold))
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=threshold,
            embed_model=embed_model
        )
        
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        
        # Further split if chunks exceed target size
        final_chunks = []
        for node in nodes:
            content = node.get_content()
            if len(content) // 4 > level_size:
                # Recursively split
                sub_chunks = split_by_size(content, level_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(content)
        
        results[f"level_{level_size}"] = final_chunks
    
    return results
```

---

## Performance Considerations

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Embedding API calls | $$$ for many sentences | Batch embed, cache |
| Sentence count | O(n) embeddings | Pre-filter very short sentences |
| Threshold tuning | Quality varies | Use percentile, not absolute |
| Short documents | Poor statistics | Fall back to fixed-size |

```python
def should_use_semantic_chunking(
    text: str,
    min_sentences: int = 10
) -> bool:
    """Determine if semantic chunking is appropriate."""
    
    sentences = split_into_sentences(text)
    
    # Need enough sentences for meaningful statistics
    if len(sentences) < min_sentences:
        return False
    
    # Check document isn't too uniform (e.g., list of items)
    avg_length = np.mean([len(s) for s in sentences])
    if avg_length < 50:  # Very short sentences
        return False
    
    return True
```

---

## Comparison with Other Methods

| Method | Coherence | Cost | Complexity | Best For |
|--------|-----------|------|------------|----------|
| Fixed-size | Low | Free | Low | Quick prototyping |
| Sentence-based | Medium | Free | Low | Simple prose |
| Structure-based | Medium | Free | Medium | Formatted docs |
| Semantic | High | $$ | High | Production RAG |
| Contextual | Highest | $$$ | Medium | Critical apps |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use percentile thresholds | Use absolute thresholds |
| Batch embed sentences | Embed one at a time |
| Respect min/max sizes | Create tiny/huge chunks |
| Cache embeddings | Re-embed same content |
| Fall back for short docs | Force semantic on lists |

---

## Summary

✅ **Semantic chunking** detects topic changes using embedding similarity

✅ **Percentile thresholds** (95th) are more robust than absolute values

✅ **LlamaIndex SemanticSplitter** provides production-ready implementation

✅ **Windowed comparison** gives more stable breakpoint detection

✅ **Higher cost** than fixed-size, but better retrieval quality

**Next:** [Chunk Metadata](./09-chunk-metadata.md)
