---
title: "Best Practices"
---

# Best Practices

## Introduction

This lesson consolidates the best practices for implementing Contextual Retrieval based on Anthropic's research and real-world deployment experience.

### What We'll Cover

- Component selection (models, rerankers)
- Configuration recommendations
- Domain-specific customization
- Production considerations
- Troubleshooting common issues

### Prerequisites

- All previous lessons in this section

---

## Component Selection

### Embedding Models

Based on Anthropic's benchmarks:

| Recommendation | Model | When to Use |
|----------------|-------|-------------|
| **Best Performance** | Voyage AI `voyage-large-2-instruct` | Maximum accuracy needed |
| **Best Performance** | Google `text-embedding-004` | Maximum accuracy needed |
| **Good Balance** | OpenAI `text-embedding-3-large` | General purpose |
| **Budget Option** | OpenAI `text-embedding-3-small` | Cost-sensitive |

```python
# Recommended embedding configuration
EMBEDDING_CONFIG = {
    "production": {
        "model": "voyage-large-2-instruct",  # or text-embedding-004
        "batch_size": 128,
        "max_retries": 3
    },
    "development": {
        "model": "text-embedding-3-small",
        "batch_size": 100,
        "max_retries": 2
    }
}
```

### Context Generator

| Model | Speed | Quality | Cost | Use Case |
|-------|-------|---------|------|----------|
| Claude 3.5 Haiku | Fastest | Good | Lowest | High volume |
| Claude 3.5 Sonnet | Medium | Better | Medium | Balanced |
| Claude 3 Opus | Slowest | Best | Highest | Critical docs |

```python
# Haiku recommended for most use cases
CONTEXT_MODEL = "claude-3-haiku-20240307"
```

### Rerankers

| Model | Max Context | Latency | Best For |
|-------|-------------|---------|----------|
| Cohere `rerank-v3.5` | 4K tokens | Low | Production |
| Voyage `rerank-2` | 32K tokens | Medium | Long documents |

---

## Optimal Configuration

### The "Anthropic Recommended" Setup

```python
RECOMMENDED_CONFIG = {
    # Chunking
    "chunk_size": 400,          # Tokens (not characters)
    "chunk_overlap": 80,        # ~20% overlap
    
    # Context generation
    "context_model": "claude-3-haiku-20240307",
    "context_max_tokens": 150,  # Aim for 50-100 tokens output
    "use_prompt_caching": True,
    
    # Embeddings
    "embedding_model": "voyage-large-2-instruct",
    # or "text-embedding-004"
    
    # Hybrid search
    "hybrid_alpha": 0.6,        # 60% vector, 40% BM25
    "initial_k": 150,           # Candidates for reranking
    
    # Reranking
    "rerank_model": "rerank-v3.5",  # Cohere
    "final_k": 20,              # Final chunks for LLM
}
```

### Configuration Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              Optimal Pipeline Configuration                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Document                                                       │
│     │                                                           │
│     ▼                                                           │
│  ┌────────────────┐                                            │
│  │ Chunking       │  chunk_size=400, overlap=80                │
│  └────────────────┘                                            │
│     │                                                           │
│     ▼                                                           │
│  ┌────────────────┐                                            │
│  │ Contextualize  │  Claude Haiku + Prompt Caching             │
│  └────────────────┘                                            │
│     │                                                           │
│     ├────────────────┬────────────────┐                        │
│     ▼                ▼                │                        │
│  ┌────────────────┐ ┌────────────────┐│                        │
│  │ Vector Index   │ │ BM25 Index     ││                        │
│  │ Voyage/Gemini  │ │ rank-bm25      ││                        │
│  └────────────────┘ └────────────────┘│                        │
│     │                │                │                        │
│     └────────────────┴────────────────┘                        │
│                      │                                          │
│  Query ──────────────▼                                          │
│                ┌────────────────┐                               │
│                │ Hybrid Search  │  alpha=0.6, k=150             │
│                └────────────────┘                               │
│                      │                                          │
│                      ▼                                          │
│                ┌────────────────┐                               │
│                │ Rerank         │  Cohere/Voyage → top 20       │
│                └────────────────┘                               │
│                      │                                          │
│                      ▼                                          │
│                [20 chunks to LLM]                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Domain-Specific Customization

### Custom Context Prompts

```python
DOMAIN_PROMPTS = {
    "legal": """Please provide context that includes:
- Document type (contract, filing, statute)
- Parties involved
- Section/clause identifiers
- Jurisdiction if mentioned
- Key defined terms referenced

Situate this chunk for legal research retrieval.""",

    "medical": """Please provide context that includes:
- Document type (clinical note, study, guideline)
- Patient population or study cohort
- Medical specialty/condition
- Treatment phase or time period
- Related diagnoses mentioned elsewhere

Situate this chunk for clinical retrieval.""",

    "technical": """Please provide context that includes:
- Software/library name and version
- Module, class, or function scope
- Related dependencies
- Configuration requirements
- Platform specifics

Situate this chunk for developer documentation retrieval.""",

    "financial": """Please provide context that includes:
- Company name and ticker
- Reporting period (quarter, year)
- Document type (10-K, 10-Q, earnings)
- Section name
- Key metrics mentioned elsewhere

Situate this chunk for financial research retrieval.""",
}


def get_context_prompt(domain: str) -> str:
    """Get domain-specific context prompt."""
    base_prompt = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

"""
    domain_instruction = DOMAIN_PROMPTS.get(domain, 
        "Please give a short succinct context to situate this chunk.")
    
    return base_prompt + domain_instruction + """

Answer only with the succinct context and nothing else."""
```

### Chunk Size by Content Type

| Content Type | Recommended Size | Overlap | Rationale |
|--------------|-----------------|---------|-----------|
| Dense technical | 300-400 tokens | 20% | High information density |
| Legal contracts | 400-500 tokens | 25% | Clause boundaries important |
| Research papers | 400-500 tokens | 20% | Section structure matters |
| General prose | 500-600 tokens | 15% | More continuous flow |
| Code | 200-300 tokens | 10% | Function boundaries |

---

## Production Considerations

### Rate Limiting & Batching

```python
import asyncio
from typing import List, Dict
import anthropic

class ProductionContextualizer:
    """Production-ready contextualizer with rate limiting."""
    
    def __init__(
        self,
        requests_per_minute: int = 50,
        max_concurrent: int = 10
    ):
        self.client = anthropic.Anthropic()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limit = requests_per_minute
        self._request_times: List[float] = []
    
    async def contextualize_batch(
        self,
        document: str,
        chunks: List[str]
    ) -> List[Dict]:
        """Process chunks with rate limiting."""
        system_msgs = self._build_system_messages(document)
        
        async def process_chunk(idx: int, chunk: str):
            async with self.semaphore:
                await self._wait_for_rate_limit()
                context = await self._generate_context(
                    chunk, system_msgs
                )
                return {"index": idx, "context": context}
        
        tasks = [
            process_chunk(i, chunk) 
            for i, chunk in enumerate(chunks)
        ]
        
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x["index"])
    
    async def _wait_for_rate_limit(self):
        """Enforce rate limiting."""
        import time
        now = time.time()
        
        # Remove old timestamps
        self._request_times = [
            t for t in self._request_times 
            if now - t < 60
        ]
        
        if len(self._request_times) >= self.rate_limit:
            wait_time = 60 - (now - self._request_times[0])
            await asyncio.sleep(wait_time)
        
        self._request_times.append(now)
```

### Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustContextualizer:
    """Contextualizer with robust error handling."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate_context(self, document: str, chunk: str) -> str:
        """Generate context with automatic retry."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=self._build_system_messages(document),
                messages=[{"role": "user", "content": self._build_prompt(chunk)}]
            )
            return response.content[0].text.strip()
        
        except anthropic.RateLimitError:
            # Log and re-raise for retry
            print("Rate limited, retrying...")
            raise
        
        except anthropic.APIError as e:
            # Log error details
            print(f"API error: {e}")
            raise
        
        except Exception as e:
            # Fallback: return empty context rather than fail
            print(f"Unexpected error: {e}")
            return ""  # Graceful degradation
```

### Monitoring & Logging

```python
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ContextualizationMetrics:
    """Track contextualization performance."""
    document_id: str
    num_chunks: int
    total_time_ms: float
    cache_hits: int
    cache_misses: int
    errors: int
    avg_context_length: float


class MonitoredContextualizer:
    """Contextualizer with full monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: List[ContextualizationMetrics] = []
    
    def contextualize_document(
        self,
        document: str,
        chunks: List[str],
        doc_id: str
    ) -> List[Dict]:
        """Process document with metrics collection."""
        start_time = datetime.now()
        results = []
        cache_hits = 0
        cache_misses = 0
        errors = 0
        context_lengths = []
        
        system_msgs = self._build_system_messages(document)
        
        for chunk in chunks:
            try:
                context, usage = self._generate_context(
                    document, chunk, system_msgs
                )
                results.append({
                    "chunk": chunk,
                    "context": context,
                    "contextualized_text": f"{context}\n\n{chunk}"
                })
                
                context_lengths.append(len(context.split()))
                
                if usage.get("cache_read_input_tokens", 0) > 0:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    
            except Exception as e:
                errors += 1
                self.logger.error(f"Error contextualizing chunk: {e}")
                results.append({
                    "chunk": chunk,
                    "context": "",
                    "contextualized_text": chunk  # Fallback
                })
        
        # Record metrics
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        metrics = ContextualizationMetrics(
            document_id=doc_id,
            num_chunks=len(chunks),
            total_time_ms=elapsed_ms,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors,
            avg_context_length=sum(context_lengths) / len(context_lengths) if context_lengths else 0
        )
        self.metrics.append(metrics)
        
        self.logger.info(
            f"Processed {doc_id}: {len(chunks)} chunks, "
            f"{cache_hits} cache hits, {errors} errors, "
            f"{elapsed_ms:.0f}ms"
        )
        
        return results
```

---

## Common Issues & Solutions

### Issue 1: Generic Context

**Problem:** Context is too generic, not chunk-specific.

```python
# ❌ Bad context
"This chunk is from a financial document."

# ✅ Good context  
"This chunk is from ACME Corp's Q2 2023 10-Q filing, 
Section 3: Management's Discussion, discussing revenue 
growth compared to Q1 2023's $314M."
```

**Solution:** Improve the context prompt:

```python
IMPROVED_PROMPT = """Please give a SHORT but SPECIFIC context including:
1. Exact document name/type if known
2. Specific entities (company, person, product names)
3. Specific dates/periods (Q2 2023, not "recent quarter")
4. Section or location in document
5. One key reference point from elsewhere in document

Be specific. Avoid generic phrases like "a financial document" 
or "discusses various topics"."""
```

### Issue 2: Context Too Long

**Problem:** Context overwhelms the chunk content.

```python
# ❌ Context longer than chunk
context = "This chunk... [200 words of context]"
chunk = "[100 words of actual content]"
```

**Solution:** Enforce length limits:

```python
# Add explicit length constraint
CONSTRAINED_PROMPT = """Provide context in 50-80 words maximum.
Focus on: entity, date, section, one reference point.
Answer only with the context, no explanations."""

# Or post-process
def truncate_context(context: str, max_tokens: int = 100) -> str:
    """Truncate context to max tokens."""
    words = context.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return context
```

### Issue 3: Cache Miss Rate High

**Problem:** Cache is being invalidated unexpectedly.

**Solutions:**

```python
# 1. Ensure consistent document formatting
def normalize_document(doc: str) -> str:
    """Normalize document for consistent caching."""
    # Remove trailing whitespace
    doc = "\n".join(line.rstrip() for line in doc.split("\n"))
    # Normalize line endings
    doc = doc.replace("\r\n", "\n")
    return doc

# 2. Process all chunks in sequence without delays
async def process_all_chunks(document: str, chunks: List[str]):
    """Process without gaps to stay within TTL."""
    system_msgs = build_system_messages(document)
    
    # Don't interleave with other documents!
    for chunk in chunks:
        await generate_context(document, chunk, system_msgs)
        # No long delays between chunks

# 3. Use extended TTL for large documents
"cache_control": {"type": "ephemeral", "ttl": "3600"}  # 1 hour
```

### Issue 4: Slow Reranking

**Problem:** Reranking latency too high.

**Solutions:**

| Approach | Impact | Trade-off |
|----------|--------|-----------|
| Reduce initial_k (150 → 100) | -33% latency | Slightly lower recall |
| Use lite reranker | -50% latency | Slightly lower accuracy |
| Parallel batch reranking | -60% latency | Higher API costs |

```python
# Use lite model for latency-sensitive applications
FAST_CONFIG = {
    "rerank_model": "rerank-2-lite",  # Voyage lite
    "initial_k": 100,  # Reduced from 150
    "final_k": 10,     # Reduced from 20
}
```

---

## Pre-Deployment Checklist

```markdown
## Contextual Retrieval Deployment Checklist

### Infrastructure
- [ ] API keys configured for: Claude, Embedding provider, Reranker
- [ ] Rate limiting implemented
- [ ] Retry logic with exponential backoff
- [ ] Error handling with graceful degradation
- [ ] Logging and monitoring in place

### Configuration
- [ ] Chunk size appropriate for content type
- [ ] Context prompt customized for domain
- [ ] Prompt caching enabled
- [ ] Hybrid alpha tuned (start with 0.6)
- [ ] Reranking initial_k and final_k set

### Quality
- [ ] Sample contexts reviewed for specificity
- [ ] Evaluation dataset created
- [ ] Baseline metrics established
- [ ] A/B test framework ready

### Cost
- [ ] Cost per document calculated
- [ ] Cache hit rate monitored
- [ ] Budget alerts configured
```

---

## Summary

✅ **Use Voyage/Gemini embeddings** for best performance  
✅ **Claude Haiku** is recommended for context generation (cost + speed)  
✅ **Always enable prompt caching** for 80-90% cost reduction  
✅ **Hybrid alpha = 0.6** is a good starting point  
✅ **Retrieve 150, rerank to 20** for optimal accuracy/latency  
✅ **Customize context prompts** for your domain  
✅ **Implement robust error handling** and monitoring  
✅ **Test on your data** - results vary by corpus

---

**Lesson Complete!**

Return to [Lesson Overview →](./00-contextual-retrieval.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Cohere Rerank Documentation: https://docs.cohere.com/v2/reference/rerank
- Voyage AI Documentation: https://docs.voyageai.com/
-->
