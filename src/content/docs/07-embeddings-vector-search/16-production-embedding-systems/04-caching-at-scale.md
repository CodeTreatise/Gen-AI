---
title: "Caching at Scale"
---

# Caching at Scale

## Introduction

Caching is critical for embedding system performance. Generating embeddings via API calls adds latency (50-200ms) and costs per token. Vector searches, while fast, still benefit from caching frequent queries. A well-designed caching strategy can reduce latency by 90% and cut embedding API costs significantly.

This lesson covers caching strategies for embeddings, Redis implementation patterns, and cache invalidation approaches.

### What We'll Cover

- Cache architecture for embedding systems
- Query embedding cache
- Document embedding cache
- Redis implementation patterns
- Cache invalidation strategies
- Cache warming and precomputation

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Basic familiarity with Redis
- Knowledge of caching fundamentals (TTL, eviction)

---

## Cache Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Layer Caching Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LAYER 1: Results Cache                 │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  Cache Key: hash(query + filters + top_k)                │  │
│  │  Cache Value: [SearchResult, SearchResult, ...]          │  │
│  │  TTL: 5-15 minutes (frequent updates)                    │  │
│  │  Hit Rate Target: 20-40%                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼ MISS                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LAYER 2: Embedding Cache               │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  Cache Key: hash(text + model)                           │  │
│  │  Cache Value: [float, float, ...] (embedding vector)     │  │
│  │  TTL: 24-72 hours (embeddings don't change)              │  │
│  │  Hit Rate Target: 60-80%                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼ MISS                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LAYER 3: Precomputed                   │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  Popular queries pre-embedded during off-peak            │  │
│  │  Stored in fast access tier                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What to Cache

| Cache Type | Key | Value | TTL | Use Case |
|------------|-----|-------|-----|----------|
| Query embedding | `hash(query_text)` | Embedding vector | 24-72h | Repeated queries |
| Document embedding | `doc_id:chunk_id` | Embedding vector | Long/permanent | Re-embedding checks |
| Search results | `hash(query+filters+k)` | Result list | 5-15min | Popular queries |
| Rerank scores | `hash(query+doc_ids)` | Score list | 5-15min | Expensive reranking |

---

## Query Embedding Cache

### Redis Implementation

```python
import redis
import json
import hashlib
from typing import List, Optional
import numpy as np

class QueryEmbeddingCache:
    """
    Cache query embeddings to avoid repeated API calls.
    """
    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "qembed",
        ttl_hours: int = 24,
        model_version: str = "v1"
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl_hours * 3600
        self.model_version = model_version
    
    def _make_key(self, text: str) -> str:
        """
        Create cache key from query text.
        Include model version to handle model changes.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{self.prefix}:{self.model_version}:{text_hash}"
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for query text.
        """
        key = self._make_key(text)
        cached = self.redis.get(key)
        
        if cached is None:
            return None
        
        # Decompress if using compression
        return json.loads(cached)
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache embedding with TTL.
        """
        key = self._make_key(text)
        value = json.dumps(embedding)
        
        self.redis.setex(key, self.ttl, value)
    
    def get_many(self, texts: List[str]) -> dict:
        """
        Batch get for multiple queries.
        Returns dict mapping text -> embedding (or None if not cached).
        """
        keys = [self._make_key(t) for t in texts]
        
        # Use pipeline for efficiency
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.get(key)
        
        results = pipe.execute()
        
        return {
            text: json.loads(result) if result else None
            for text, result in zip(texts, results)
        }
    
    def set_many(self, text_embeddings: dict) -> None:
        """
        Batch set multiple embeddings.
        """
        pipe = self.redis.pipeline()
        
        for text, embedding in text_embeddings.items():
            key = self._make_key(text)
            value = json.dumps(embedding)
            pipe.setex(key, self.ttl, value)
        
        pipe.execute()


class CachedEmbeddingService:
    """
    Embedding service with transparent caching.
    """
    def __init__(
        self,
        embedding_client,
        cache: QueryEmbeddingCache,
        metrics=None
    ):
        self.embedder = embedding_client
        self.cache = cache
        self.metrics = metrics
    
    def embed(self, text: str) -> List[float]:
        """
        Get embedding with cache-first strategy.
        """
        # Try cache first
        cached = self.cache.get(text)
        
        if cached is not None:
            if self.metrics:
                self.metrics.increment("embedding_cache_hit")
            return cached
        
        # Cache miss - call API
        if self.metrics:
            self.metrics.increment("embedding_cache_miss")
        
        embedding = self.embedder.embed(text)
        
        # Store in cache for next time
        self.cache.set(text, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding with partial cache hits.
        """
        # Check cache for all texts
        cached_results = self.cache.get_many(texts)
        
        # Separate hits and misses
        hits = {t: e for t, e in cached_results.items() if e is not None}
        misses = [t for t in texts if t not in hits]
        
        if self.metrics:
            self.metrics.increment("embedding_cache_hit", len(hits))
            self.metrics.increment("embedding_cache_miss", len(misses))
        
        # Only call API for misses
        if misses:
            new_embeddings = self.embedder.embed_batch(misses)
            
            # Cache new embeddings
            self.cache.set_many(dict(zip(misses, new_embeddings)))
            
            # Merge with cached
            for text, embedding in zip(misses, new_embeddings):
                hits[text] = embedding
        
        # Return in original order
        return [hits[t] for t in texts]
```

### Compressed Embedding Cache

```python
import zlib
import struct
from typing import List

class CompressedEmbeddingCache:
    """
    Cache embeddings with compression for memory efficiency.
    Reduces memory usage by ~50% for float32 embeddings.
    """
    def __init__(self, redis_client: redis.Redis, ttl_hours: int = 24):
        self.redis = redis_client
        self.ttl = ttl_hours * 3600
    
    def _compress(self, embedding: List[float]) -> bytes:
        """Compress embedding to bytes."""
        # Pack floats to binary
        binary = struct.pack(f'{len(embedding)}f', *embedding)
        # Compress with zlib
        return zlib.compress(binary)
    
    def _decompress(self, data: bytes) -> List[float]:
        """Decompress bytes to embedding."""
        binary = zlib.decompress(data)
        count = len(binary) // 4  # 4 bytes per float32
        return list(struct.unpack(f'{count}f', binary))
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get and decompress cached embedding."""
        data = self.redis.get(key)
        if data is None:
            return None
        return self._decompress(data)
    
    def set(self, key: str, embedding: List[float]) -> None:
        """Compress and cache embedding."""
        data = self._compress(embedding)
        self.redis.setex(key, self.ttl, data)


class QuantizedEmbeddingCache:
    """
    Cache embeddings with int8 quantization.
    Reduces memory by 75% with minimal quality loss.
    """
    def __init__(self, redis_client: redis.Redis, ttl_hours: int = 24):
        self.redis = redis_client
        self.ttl = ttl_hours * 3600
    
    def _quantize(self, embedding: List[float]) -> tuple:
        """Quantize float32 to int8."""
        arr = np.array(embedding, dtype=np.float32)
        
        # Get scale factors
        min_val = arr.min()
        max_val = arr.max()
        scale = (max_val - min_val) / 255 if max_val != min_val else 1.0
        
        # Quantize to uint8
        quantized = ((arr - min_val) / scale).astype(np.uint8)
        
        return quantized.tobytes(), float(min_val), float(scale)
    
    def _dequantize(self, data: bytes, min_val: float, scale: float) -> List[float]:
        """Dequantize int8 back to float32."""
        quantized = np.frombuffer(data, dtype=np.uint8)
        return (quantized * scale + min_val).astype(np.float32).tolist()
    
    def set(self, key: str, embedding: List[float]) -> None:
        """Quantize and cache."""
        quantized, min_val, scale = self._quantize(embedding)
        
        # Store quantized data with metadata
        self.redis.hset(key, mapping={
            "data": quantized,
            "min": str(min_val),
            "scale": str(scale)
        })
        self.redis.expire(key, self.ttl)
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get and dequantize."""
        stored = self.redis.hgetall(key)
        if not stored:
            return None
        
        return self._dequantize(
            stored[b"data"],
            float(stored[b"min"]),
            float(stored[b"scale"])
        )
```

---

## Search Results Cache

### Implementation

```python
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import json

@dataclass
class CachedSearchResult:
    id: str
    score: float
    content: str
    metadata: dict

class SearchResultsCache:
    """
    Cache complete search results for repeated queries.
    """
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_minutes: int = 10,
        max_results_cached: int = 50
    ):
        self.redis = redis_client
        self.ttl = ttl_minutes * 60
        self.max_results = max_results_cached
    
    def _make_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict]
    ) -> str:
        """
        Create deterministic cache key from search parameters.
        """
        key_parts = [
            query,
            str(top_k),
            json.dumps(filters, sort_keys=True) if filters else ""
        ]
        key_string = "|".join(key_parts)
        return f"search:{hashlib.sha256(key_string.encode()).hexdigest()[:24]}"
    
    def get(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None
    ) -> Optional[List[CachedSearchResult]]:
        """
        Get cached search results.
        """
        key = self._make_key(query, top_k, filters)
        cached = self.redis.get(key)
        
        if cached is None:
            return None
        
        data = json.loads(cached)
        return [CachedSearchResult(**r) for r in data]
    
    def set(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict],
        results: List[CachedSearchResult]
    ) -> None:
        """
        Cache search results.
        """
        key = self._make_key(query, top_k, filters)
        
        # Only cache up to max_results
        to_cache = results[:self.max_results]
        data = json.dumps([
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata
            }
            for r in to_cache
        ])
        
        self.redis.setex(key, self.ttl, data)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        Use when underlying data changes.
        """
        keys = self.redis.keys(f"search:{pattern}*")
        if keys:
            return self.redis.delete(*keys)
        return 0


class CachedSearchService:
    """
    Search service with results caching.
    """
    def __init__(
        self,
        search_pipeline,
        results_cache: SearchResultsCache,
        embedding_cache: QueryEmbeddingCache
    ):
        self.pipeline = search_pipeline
        self.results_cache = results_cache
        self.embedding_cache = embedding_cache
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        use_cache: bool = True
    ) -> List[CachedSearchResult]:
        """
        Search with multi-layer caching.
        """
        # Layer 1: Check results cache
        if use_cache:
            cached_results = self.results_cache.get(query, top_k, filters)
            if cached_results is not None:
                return cached_results
        
        # Layer 2: Embedding cache (handled by CachedEmbeddingService)
        results = self.pipeline.search(query, top_k, filters)
        
        # Cache results for next time
        if use_cache:
            self.results_cache.set(query, top_k, filters, results)
        
        return results
```

---

## Cache Invalidation Strategies

### Strategy Comparison

| Strategy | Consistency | Complexity | Use Case |
|----------|-------------|------------|----------|
| TTL-based | Eventual | Low | Most queries |
| Event-driven | Strong | Medium | Critical data |
| Versioned keys | Strong | Medium | Model changes |
| Hybrid | Strong | High | Production systems |

### Implementation

```python
from enum import Enum
from typing import Callable, List
import asyncio

class InvalidationStrategy(Enum):
    TTL = "ttl"
    EVENT = "event"
    VERSION = "version"

class CacheInvalidator:
    """
    Multi-strategy cache invalidation.
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.listeners: List[Callable] = []
    
    # === TTL-Based Invalidation ===
    def set_with_ttl(self, key: str, value: str, ttl_seconds: int):
        """Automatic expiration via TTL."""
        self.redis.setex(key, ttl_seconds, value)
    
    # === Event-Driven Invalidation ===
    def invalidate_document(self, doc_id: str):
        """
        Invalidate all caches related to a document.
        Called when document is updated/deleted.
        """
        # Invalidate document embedding cache
        self.redis.delete(f"docembed:{doc_id}:*")
        
        # Invalidate search results that might contain this doc
        # This is expensive - use sparingly
        pattern = f"search:*"
        
        # For production: use Redis SCAN instead of KEYS
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
        
        # Notify listeners
        for listener in self.listeners:
            listener("document_invalidated", doc_id)
    
    def invalidate_by_tag(self, tag: str):
        """
        Invalidate entries by tag.
        Useful for category-based invalidation.
        """
        # Get tagged keys from set
        tagged_keys = self.redis.smembers(f"tag:{tag}")
        
        if tagged_keys:
            # Delete all tagged entries
            self.redis.delete(*tagged_keys)
            # Clear the tag set
            self.redis.delete(f"tag:{tag}")
    
    # === Version-Based Invalidation ===
    def get_current_version(self, namespace: str) -> str:
        """Get current cache version for namespace."""
        version = self.redis.get(f"version:{namespace}")
        return version.decode() if version else "v1"
    
    def increment_version(self, namespace: str) -> str:
        """
        Increment version to invalidate all cached data.
        Old entries become orphaned and expire via TTL.
        """
        new_version = f"v{int(self.redis.incr(f'version_counter:{namespace}'))}"
        self.redis.set(f"version:{namespace}", new_version)
        return new_version
    
    def add_invalidation_listener(self, callback: Callable):
        """Register callback for invalidation events."""
        self.listeners.append(callback)


class TaggedCache:
    """
    Cache with tagging for targeted invalidation.
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def set_with_tags(
        self,
        key: str,
        value: str,
        tags: List[str],
        ttl: int
    ):
        """
        Set value with associated tags for invalidation.
        """
        pipe = self.redis.pipeline()
        
        # Set the value
        pipe.setex(key, ttl, value)
        
        # Add key to each tag set
        for tag in tags:
            pipe.sadd(f"tag:{tag}", key)
            pipe.expire(f"tag:{tag}", ttl)  # Tag expires with data
        
        pipe.execute()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached value."""
        return self.redis.get(key)
    
    def invalidate_tags(self, tags: List[str]):
        """Invalidate all entries with any of the given tags."""
        pipe = self.redis.pipeline()
        
        for tag in tags:
            # Get all keys with this tag
            keys = self.redis.smembers(f"tag:{tag}")
            if keys:
                pipe.delete(*keys)
            pipe.delete(f"tag:{tag}")
        
        pipe.execute()
```

### Event-Driven Invalidation Example

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentEvent:
    event_type: str  # "created", "updated", "deleted"
    doc_id: str
    metadata: Optional[dict] = None

class EventDrivenCacheManager:
    """
    Invalidate caches based on document events.
    """
    def __init__(
        self,
        results_cache: SearchResultsCache,
        embedding_cache: QueryEmbeddingCache,
        invalidator: CacheInvalidator
    ):
        self.results_cache = results_cache
        self.embedding_cache = embedding_cache
        self.invalidator = invalidator
    
    def handle_event(self, event: DocumentEvent):
        """
        Handle document change event.
        """
        if event.event_type == "created":
            # New doc - invalidate search results cache
            # (new doc might be relevant to cached queries)
            self._invalidate_search_cache()
        
        elif event.event_type == "updated":
            # Updated doc - invalidate document embedding + search results
            self._invalidate_document_embedding(event.doc_id)
            self._invalidate_search_cache()
        
        elif event.event_type == "deleted":
            # Deleted doc - full invalidation for this doc
            self._invalidate_document_embedding(event.doc_id)
            self._invalidate_search_cache()
    
    def _invalidate_document_embedding(self, doc_id: str):
        """Invalidate cached embedding for document."""
        pattern = f"docembed:{doc_id}:*"
        self.invalidator.invalidate_by_pattern(pattern)
    
    def _invalidate_search_cache(self):
        """
        Invalidate search results cache.
        
        In production, you might use more targeted invalidation
        based on document tags/categories.
        """
        # Option 1: Invalidate all (simple but aggressive)
        self.invalidator.increment_version("search_results")
        
        # Option 2: Targeted by tag (if document has category tags)
        # self.invalidator.invalidate_by_tag(document.category)
```

---

## Cache Warming

### Precomputation Strategy

```python
import asyncio
from datetime import datetime, timedelta

class CacheWarmer:
    """
    Pre-populate cache with expected queries.
    """
    def __init__(
        self,
        embedding_service,
        cache: QueryEmbeddingCache,
        analytics_client
    ):
        self.embedder = embedding_service
        self.cache = cache
        self.analytics = analytics_client
    
    async def warm_popular_queries(self, top_n: int = 1000):
        """
        Pre-embed most popular queries from analytics.
        Run during off-peak hours.
        """
        # Get popular queries from last 7 days
        popular_queries = await self.analytics.get_top_queries(
            days=7,
            limit=top_n
        )
        
        # Filter out already cached
        uncached = []
        for query in popular_queries:
            if self.cache.get(query["text"]) is None:
                uncached.append(query["text"])
        
        # Batch embed uncached queries
        if uncached:
            print(f"Warming cache with {len(uncached)} queries...")
            
            batch_size = 100
            for i in range(0, len(uncached), batch_size):
                batch = uncached[i:i + batch_size]
                embeddings = self.embedder.embed_batch(batch)
                self.cache.set_many(dict(zip(batch, embeddings)))
                
                # Rate limit to avoid overloading embedding API
                await asyncio.sleep(1)
        
        return {"warmed": len(uncached), "already_cached": len(popular_queries) - len(uncached)}
    
    async def warm_scheduled_content(self, content_items: list):
        """
        Pre-embed content scheduled for publication.
        Run before content goes live.
        """
        for item in content_items:
            # Pre-embed document chunks
            chunks = chunk_document(item["content"])
            
            for chunk in chunks:
                if self.cache.get(chunk.text) is None:
                    embedding = self.embedder.embed(chunk.text)
                    self.cache.set(chunk.text, embedding)
    
    def get_cache_stats(self) -> dict:
        """Get cache health statistics."""
        info = self.cache.redis.info("memory")
        
        return {
            "used_memory_mb": info["used_memory"] / (1024 * 1024),
            "max_memory_mb": info.get("maxmemory", 0) / (1024 * 1024),
            "hit_rate": self._calculate_hit_rate(),
            "keys_count": self.cache.redis.dbsize()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from Redis INFO."""
        info = self.cache.redis.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        
        total = hits + misses
        return hits / total if total > 0 else 0.0
```

---

## Summary

✅ **Multi-layer caching: results → embeddings → precomputed**  
✅ **Compress embeddings to reduce Redis memory usage**  
✅ **Include model version in cache keys for safe upgrades**  
✅ **Use event-driven invalidation for data changes**  
✅ **Warm cache with popular queries during off-peak hours**

---

**Next:** [Scaling Patterns →](./05-scaling-patterns.md)

---

<!-- 
Sources Consulted:
- Redis Documentation: https://redis.io/docs/
- Redis Vector Database: https://redis.io/docs/latest/develop/get-started/vector-database/
- Cache Invalidation Patterns: https://martinfowler.com/bliki/TwoHardThings.html
-->
