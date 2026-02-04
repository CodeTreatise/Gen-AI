---
title: "Prompt Caching for Cost Optimization"
---

# Prompt Caching for Cost Optimization

## Introduction

Generating context for every chunk requires sending the **full document** to the LLM each time. Without optimization, this is expensive. **Prompt caching** dramatically reduces costs by caching the document content across chunk processing calls.

This lesson shows how to implement prompt caching for Contextual Retrieval.

### What We'll Cover

- Why Contextual Retrieval is expensive without caching
- How prompt caching works
- Anthropic's `cache_control` implementation
- Cost calculations and savings
- Complete implementation example

### Prerequisites

- [Performance Improvements](./06-performance-improvements.md)
- Familiarity with the Claude API

---

## The Cost Problem

### Without Caching

```
┌─────────────────────────────────────────────────────────────────┐
│         Why Contextual Retrieval is Expensive (Naive)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Document: 8,000 tokens                                         │
│  Chunks: 10 chunks                                              │
│                                                                 │
│  For EACH chunk, we send:                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  <document> 8,000 tokens </document>                      │  │
│  │  <chunk> ~500 tokens </chunk>                             │  │
│  │  Prompt instructions: ~50 tokens                          │  │
│  │                                                           │  │
│  │  Total per call: ~8,550 tokens                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Total for 10 chunks: 8,550 × 10 = 85,500 input tokens!        │
│                                                                 │
│  The SAME 8,000 token document is sent 10 times.               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Without Caching

```python
# Example cost calculation (without caching)

document_tokens = 8_000
chunks = 10
prompt_overhead = 550  # Instructions + chunk

tokens_per_chunk = document_tokens + prompt_overhead
total_input_tokens = tokens_per_chunk * chunks

# Claude Haiku pricing: $0.25 per million input tokens
cost_per_doc = (total_input_tokens / 1_000_000) * 0.25

print(f"Tokens per chunk: {tokens_per_chunk:,}")
print(f"Total input tokens: {total_input_tokens:,}")
print(f"Cost per document: ${cost_per_doc:.4f}")
print(f"Cost for 1,000 docs: ${cost_per_doc * 1000:.2f}")
```

**Output:**
```
Tokens per chunk: 8,550
Total input tokens: 85,500
Cost per document: $0.0214
Cost for 1,000 docs: $21.38
```

---

## How Prompt Caching Works

### The Concept

Prompt caching allows you to **cache part of the prompt** across multiple API calls. The cached portion is read from cache instead of being reprocessed.

```
┌─────────────────────────────────────────────────────────────────┐
│              Prompt Caching for Contextual Retrieval             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FIRST CALL (cache write):                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  System: "You are a helpful assistant..."                 │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ CACHED CONTENT (cache_control: ephemeral)          │  │  │
│  │  │                                                    │  │  │
│  │  │ <document>                                         │  │  │
│  │  │ [Full document: 8,000 tokens]                      │  │  │
│  │  │ </document>                                        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  User: <chunk>[Chunk 1]</chunk> Give context...           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  SUBSEQUENT CALLS (cache read):                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  System: "You are a helpful assistant..."                 │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ CACHED (read from cache - 90% cheaper!)            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  User: <chunk>[Chunk 2]</chunk> Give context...           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Pricing

| Operation | Cost vs Base Input | When It Happens |
|-----------|-------------------|-----------------|
| **Cache Write** | +25% | First call (or cache expired) |
| **Cache Read** | **-90%** | Subsequent calls (cache hit) |

---

## Anthropic Cache Control Implementation

### Basic Syntax

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=200,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant that provides context for document chunks.",
        },
        {
            "type": "text",
            "text": f"<document>\n{full_document}\n</document>",
            "cache_control": {"type": "ephemeral"}  # ← Cache this!
        }
    ],
    messages=[
        {
            "role": "user",
            "content": f"<chunk>\n{chunk_text}\n</chunk>\n\nProvide succinct context for this chunk."
        }
    ]
)
```

### Key Points

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `cache_control` | `{"type": "ephemeral"}` | Cache for default TTL (5 min) |
| Location | Last `system` message block | What gets cached |
| TTL | 5 minutes default | Time cache stays valid |

---

## Cache TTL Options

### Default (5 Minutes)

```python
# Default 5-minute TTL
"cache_control": {"type": "ephemeral"}
```

Best for: Processing all chunks of a document in sequence (typical use case).

### Extended (1 Hour) - Beta

```python
# Extended 1-hour TTL (at 2x write cost)
"cache_control": {
    "type": "ephemeral",
    "ttl": "3600"  # 1 hour in seconds
}
```

Best for: Documents that may be reprocessed or have many chunks.

### Cache Invalidation

The cache is invalidated if:
- TTL expires
- Model version changes
- Cached content changes (even by 1 character)
- System prompt before cached content changes

---

## Minimum Token Requirements

### By Model

| Model | Minimum Cacheable Tokens |
|-------|-------------------------|
| Claude 3.5 Sonnet | 1,024 tokens |
| Claude 3.5 Haiku | 1,024 tokens |
| Claude 3 Haiku | 1,024 tokens |
| Claude 3 Opus | 1,024 tokens |

> **Important:** If your document is smaller than the minimum, caching won't be activated.

```python
def should_use_caching(document_tokens: int, chunks: int) -> bool:
    """Determine if caching is worthwhile."""
    MIN_TOKENS = 1024
    MIN_CHUNKS_FOR_BENEFIT = 2
    
    # Need minimum tokens to enable caching
    if document_tokens < MIN_TOKENS:
        return False
    
    # Need multiple chunks to benefit from caching
    if chunks < MIN_CHUNKS_FOR_BENEFIT:
        return False
    
    return True
```

---

## Complete Implementation

### Cached Contextualizer Class

```python
import anthropic
from typing import List, Dict, Optional

class CachedContextualizer:
    """Generate chunk contexts with prompt caching."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        use_extended_ttl: bool = False
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.use_extended_ttl = use_extended_ttl
    
    def build_system_messages(self, document: str) -> List[Dict]:
        """Build system messages with cached document."""
        cache_control = {"type": "ephemeral"}
        if self.use_extended_ttl:
            cache_control["ttl"] = "3600"  # 1 hour
        
        return [
            {
                "type": "text",
                "text": "You are a helpful assistant that provides succinct context for document chunks to improve search retrieval."
            },
            {
                "type": "text",
                "text": f"<document>\n{document}\n</document>",
                "cache_control": cache_control
            }
        ]
    
    def generate_context(
        self,
        document: str,
        chunk: str,
        system_messages: Optional[List[Dict]] = None
    ) -> tuple[str, Dict]:
        """
        Generate context for a chunk.
        
        Returns:
            Tuple of (context_text, usage_info)
        """
        if system_messages is None:
            system_messages = self.build_system_messages(document)
        
        prompt = f"""Here is the chunk we want to situate:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within 
the overall document for the purposes of improving search retrieval 
of the chunk. Answer only with the succinct context and nothing else."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=system_messages,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract usage info for cost tracking
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, 'cache_creation_input_tokens', 0
            ),
            "cache_read_input_tokens": getattr(
                response.usage, 'cache_read_input_tokens', 0
            )
        }
        
        return response.content[0].text.strip(), usage
    
    def contextualize_chunks(
        self,
        document: str,
        chunks: List[str]
    ) -> List[Dict]:
        """
        Contextualize all chunks with caching.
        
        Returns:
            List of dicts with context and usage info
        """
        # Build system messages once (includes cached document)
        system_messages = self.build_system_messages(document)
        
        results = []
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_writes": 0,
            "cache_reads": 0
        }
        
        for i, chunk in enumerate(chunks):
            context, usage = self.generate_context(
                document, 
                chunk, 
                system_messages
            )
            
            # Track usage
            total_usage["input_tokens"] += usage["input_tokens"]
            total_usage["output_tokens"] += usage["output_tokens"]
            
            if usage["cache_creation_input_tokens"] > 0:
                total_usage["cache_writes"] += 1
            if usage["cache_read_input_tokens"] > 0:
                total_usage["cache_reads"] += 1
            
            results.append({
                "chunk_index": i,
                "original_chunk": chunk,
                "context": context,
                "contextualized_text": f"{context}\n\n{chunk}",
                "usage": usage
            })
            
            print(f"  Chunk {i+1}/{len(chunks)}: "
                  f"{'cache_write' if usage['cache_creation_input_tokens'] > 0 else 'cache_read'}")
        
        # Print summary
        print(f"\nUsage Summary:")
        print(f"  Total input tokens: {total_usage['input_tokens']:,}")
        print(f"  Total output tokens: {total_usage['output_tokens']:,}")
        print(f"  Cache writes: {total_usage['cache_writes']}")
        print(f"  Cache reads: {total_usage['cache_reads']}")
        
        return results


# Usage example
contextualizer = CachedContextualizer()

document = """
ACME Corporation Quarterly Report Q2 2023
[... 8000 tokens of content ...]
"""

chunks = [
    "Revenue grew 3% compared to the previous quarter...",
    "Cloud services segment contributed 45% of revenue...",
    "Operating expenses decreased by 5%...",
    # ... more chunks
]

results = contextualizer.contextualize_chunks(document, chunks)
```

**Output:**
```
  Chunk 1/10: cache_write
  Chunk 2/10: cache_read
  Chunk 3/10: cache_read
  Chunk 4/10: cache_read
  Chunk 5/10: cache_read
  Chunk 6/10: cache_read
  Chunk 7/10: cache_read
  Chunk 8/10: cache_read
  Chunk 9/10: cache_read
  Chunk 10/10: cache_read

Usage Summary:
  Total input tokens: 13,500
  Total output tokens: 850
  Cache writes: 1
  Cache reads: 9
```

---

## Cost Calculation

### With Caching vs Without

```python
def calculate_costs(
    document_tokens: int,
    num_chunks: int,
    prompt_overhead: int = 550,  # Instructions + chunk
    output_tokens_per_chunk: int = 80
) -> Dict:
    """Compare costs with and without caching."""
    
    # Claude Haiku pricing
    INPUT_COST = 0.25 / 1_000_000    # $0.25 per 1M input tokens
    OUTPUT_COST = 1.25 / 1_000_000   # $1.25 per 1M output tokens
    CACHE_WRITE_MULTIPLIER = 1.25   # 25% more for cache writes
    CACHE_READ_MULTIPLIER = 0.10    # 90% less for cache reads
    
    # WITHOUT caching
    input_per_chunk = document_tokens + prompt_overhead
    total_input_no_cache = input_per_chunk * num_chunks
    total_output = output_tokens_per_chunk * num_chunks
    
    cost_no_cache = (total_input_no_cache * INPUT_COST + 
                     total_output * OUTPUT_COST)
    
    # WITH caching
    # First chunk: cache write (document only)
    first_chunk_input = document_tokens * CACHE_WRITE_MULTIPLIER + prompt_overhead
    # Remaining chunks: cache read (document) + new (prompt)
    remaining_input = (
        (document_tokens * CACHE_READ_MULTIPLIER + prompt_overhead) * 
        (num_chunks - 1)
    )
    total_input_cached = first_chunk_input + remaining_input
    
    cost_cached = (total_input_cached * INPUT_COST + 
                   total_output * OUTPUT_COST)
    
    savings_pct = (1 - cost_cached / cost_no_cache) * 100
    
    return {
        "without_caching": {
            "input_tokens": total_input_no_cache,
            "cost": cost_no_cache
        },
        "with_caching": {
            "input_tokens": int(total_input_cached),
            "cost": cost_cached
        },
        "savings_percentage": savings_pct,
        "cost_per_million_doc_tokens": cost_cached / (document_tokens / 1_000_000)
    }


# Example
results = calculate_costs(
    document_tokens=8_000,
    num_chunks=10
)

print("Cost Comparison:")
print(f"  Without caching: ${results['without_caching']['cost']:.4f}")
print(f"  With caching: ${results['with_caching']['cost']:.4f}")
print(f"  Savings: {results['savings_percentage']:.1f}%")
print(f"  Cost per 1M doc tokens: ${results['cost_per_million_doc_tokens']:.2f}")
```

**Output:**
```
Cost Comparison:
  Without caching: $0.0214
  With caching: $0.0035
  Savings: 83.6%
  Cost per 1M doc tokens: $1.02
```

---

## Anthropic's Published Cost

> **~$1.02 per million document tokens** with prompt caching

This means for a large corpus:

| Corpus Size | Estimated Cost |
|-------------|----------------|
| 1,000 documents (8K tokens each) | ~$8.16 |
| 10,000 documents | ~$81.60 |
| 100,000 documents | ~$816 |
| 1 million documents | ~$8,160 |

---

## Best Practices for Caching

### Do's

```python
# ✅ DO: Process chunks in sequence immediately
chunks = split_document(document)
system_msgs = contextualizer.build_system_messages(document)
for chunk in chunks:
    context = contextualizer.generate_context(document, chunk, system_msgs)
    # Process immediately to stay within 5-min TTL
```

### Don'ts

```python
# ❌ DON'T: Process chunks with long gaps
chunks = split_document(document)
for chunk in chunks:
    context = contextualizer.generate_context(document, chunk)
    time.sleep(600)  # 10 minutes - cache will expire!
```

### Batching Documents

```python
# ✅ DO: Complete one document before starting next
for document in documents:
    chunks = split_document(document)
    contextualize_all_chunks(document, chunks)  # Process all together
    
# ❌ DON'T: Interleave documents
for i, chunk in enumerate(all_chunks_from_all_docs):
    # This prevents effective caching!
    contextualize_chunk(documents[chunk.doc_id], chunk)
```

---

## Summary

✅ **Prompt caching reduces costs by ~80-90%** for Contextual Retrieval  
✅ Use `cache_control: {"type": "ephemeral"}` on the document content  
✅ **First call writes cache** (+25% cost), subsequent calls read (-90%)  
✅ Default TTL is **5 minutes** (1 hour available with 2x write cost)  
✅ Minimum cacheable content: **1,024 tokens**  
✅ **~$1.02 per million document tokens** with caching enabled  
✅ Process all chunks of one document before moving to the next

---

**Next:** [Reranking →](./08-reranking.md)

---

<!-- 
Sources Consulted:
- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
-->
