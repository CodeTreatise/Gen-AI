---
title: "Caching Strategies"
---

# Caching Strategies

## Introduction

Embedding generation is expensive—both in terms of API costs and latency. A well-designed cache can reduce embedding API calls by 80-99% in many applications, dramatically improving performance and reducing costs.

In this lesson, we'll explore caching strategies for embeddings, from simple in-memory caches to production Redis and database solutions.

### What We'll Cover

- Why cache embeddings (cost and latency benefits)
- Cache key design strategies
- Storage options (memory, file, Redis, database)
- TTL and invalidation policies
- Production caching patterns

### Prerequisites

- Completed previous lessons in this series
- Basic understanding of caching concepts
- Familiarity with Redis (helpful but not required)

---

## Why Cache Embeddings?

### Cost Savings

| Provider | Cost per 1M tokens | 1M Documents (~500 tokens each) |
|----------|-------------------|--------------------------------|
| OpenAI (3-small) | $0.02 | $10 |
| OpenAI (3-large) | $0.13 | $65 |
| Gemini | Free tier / $0.01 | $5 |
| Cohere | $0.10 | $50 |

With caching, you pay once per unique document. Re-embedding the same content is free.

### Latency Improvement

| Operation | Typical Latency |
|-----------|-----------------|
| API call | 100-500ms |
| Redis lookup | 1-5ms |
| In-memory cache | <0.1ms |

> **🤖 AI Context:** In RAG systems, queries often include common phrases or repeated context. Caching query embeddings can reduce p50 latency by 50% or more.

---

## Cache Key Design

The cache key must uniquely identify the embedding. Consider these factors:

### Simple Content Hash

```python
import hashlib

def create_cache_key(text: str) -> str:
    """Create a cache key from text content."""
    # Normalize text first
    normalized = text.strip().lower()
    
    # Create SHA-256 hash
    hash_object = hashlib.sha256(normalized.encode('utf-8'))
    return hash_object.hexdigest()

# Example
text = "Hello, World!"
key = create_cache_key(text)
print(f"Cache key: {key}")
```

**Output:**
```
Cache key: 49cf84bd6...
```

### Including Model Configuration

Different models or configurations produce different embeddings:

```python
def create_full_cache_key(
    text: str,
    model: str,
    dimensions: int | None = None,
    task_type: str | None = None,
) -> str:
    """Create cache key including model configuration."""
    # Normalize text
    normalized_text = text.strip()
    
    # Build configuration string
    config_parts = [
        f"model:{model}",
        f"dims:{dimensions}" if dimensions else "dims:default",
        f"task:{task_type}" if task_type else "task:none",
    ]
    config_string = "|".join(config_parts)
    
    # Combine text and config
    full_content = f"{config_string}||{normalized_text}"
    
    # Hash
    return hashlib.sha256(full_content.encode('utf-8')).hexdigest()

# Same text, different configs = different keys
key1 = create_full_cache_key("Hello", "text-embedding-3-small")
key2 = create_full_cache_key("Hello", "text-embedding-3-small", dimensions=256)
key3 = create_full_cache_key("Hello", "text-embedding-3-large")

print(f"Default dims: {key1[:16]}...")
print(f"256 dims: {key2[:16]}...")
print(f"Large model: {key3[:16]}...")
```

### Key Versioning

Add a version prefix for cache invalidation during upgrades:

```python
CACHE_VERSION = "v1"

def versioned_cache_key(text: str, model: str) -> str:
    """Cache key with version prefix."""
    content_hash = create_cache_key(text)
    return f"{CACHE_VERSION}:{model}:{content_hash}"

# When you change preprocessing or models, bump the version
# Old cache entries will naturally expire or be ignored
```

---

## In-Memory Caching

For simple applications or development:

### Using functools.lru_cache

```python
from functools import lru_cache
from openai import OpenAI

client = OpenAI()

@lru_cache(maxsize=10000)
def get_embedding_cached(text: str, model: str = "text-embedding-3-small") -> tuple:
    """Get embedding with LRU cache. Returns tuple for hashability."""
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    # Return as tuple (lists aren't hashable)
    return tuple(response.data[0].embedding)

# First call - API request
emb1 = get_embedding_cached("Hello, world!")
print("First call: API request made")

# Second call - from cache
emb2 = get_embedding_cached("Hello, world!")
print("Second call: from cache")

# Check cache stats
print(f"Cache info: {get_embedding_cached.cache_info()}")
```

**Output:**
```
First call: API request made
Second call: from cache
Cache info: CacheInfo(hits=1, misses=1, maxsize=10000, currsize=1)
```

### Custom In-Memory Cache

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading

@dataclass
class CacheEntry:
    embedding: list[float]
    created_at: datetime
    hits: int = 0

class EmbeddingCache:
    """Thread-safe in-memory embedding cache with TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0}
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        return datetime.now() - entry.created_at > self.ttl
    
    def get(self, key: str) -> Optional[list[float]]:
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.stats["misses"] += 1
                return None
            
            if self._is_expired(entry):
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            entry.hits += 1
            self.stats["hits"] += 1
            return entry.embedding
    
    def set(self, key: str, embedding: list[float]) -> None:
        with self.lock:
            # Evict if at capacity (simple FIFO for now)
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = CacheEntry(
                embedding=embedding,
                created_at=datetime.now(),
            )
    
    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "size": len(self.cache),
            "hit_rate": f"{hit_rate:.1%}",
        }

# Usage
cache = EmbeddingCache(max_size=5000, ttl_hours=24)

def get_embedding(text: str) -> list[float]:
    key = create_cache_key(text)
    
    # Try cache first
    cached = cache.get(key)
    if cached:
        return cached
    
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    embedding = response.data[0].embedding
    
    # Store in cache
    cache.set(key, embedding)
    
    return embedding
```

---

## File-Based Caching

For persistent caching without external dependencies:

### SQLite Cache

```python
import sqlite3
import json
from contextlib import contextmanager

class SQLiteEmbeddingCache:
    """SQLite-based embedding cache."""
    
    def __init__(self, db_path: str = "embeddings.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created 
                ON embeddings(created_at)
            """)
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get(self, key: str) -> list[float] | None:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update access time
                conn.execute(
                    "UPDATE embeddings SET accessed_at = CURRENT_TIMESTAMP WHERE cache_key = ?",
                    (key,)
                )
                return json.loads(row[0])
            
            return None
    
    def set(self, key: str, embedding: list[float], model: str = "unknown"):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (cache_key, embedding, model)
                VALUES (?, ?, ?)
            """, (key, json.dumps(embedding), model))
    
    def cleanup_old(self, days: int = 30):
        """Remove entries older than specified days."""
        with self._get_conn() as conn:
            conn.execute("""
                DELETE FROM embeddings 
                WHERE created_at < datetime('now', ?)
            """, (f'-{days} days',))
    
    def get_stats(self) -> dict:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT SUM(LENGTH(embedding)) FROM embeddings
            """)
            size = cursor.fetchone()[0] or 0
            
            return {
                "entries": count,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
            }

# Usage
cache = SQLiteEmbeddingCache("./embedding_cache.db")
```

---

## Redis Caching

For production systems with multiple workers:

### Basic Redis Cache

```python
import redis
import json
from typing import Optional

class RedisEmbeddingCache:
    """Redis-based embedding cache for production."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "emb:",
        ttl_seconds: int = 86400 * 7,  # 7 days default
    ):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        self.prefix = prefix
        self.ttl = ttl_seconds
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[list[float]]:
        redis_key = self._make_key(key)
        data = self.client.get(redis_key)
        
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, embedding: list[float]) -> None:
        redis_key = self._make_key(key)
        self.client.setex(
            redis_key,
            self.ttl,
            json.dumps(embedding),
        )
    
    def get_batch(self, keys: list[str]) -> dict[str, list[float] | None]:
        """Get multiple embeddings in one round-trip."""
        redis_keys = [self._make_key(k) for k in keys]
        values = self.client.mget(redis_keys)
        
        return {
            key: json.loads(val) if val else None
            for key, val in zip(keys, values)
        }
    
    def set_batch(self, embeddings: dict[str, list[float]]) -> None:
        """Set multiple embeddings with pipeline."""
        pipe = self.client.pipeline()
        
        for key, embedding in embeddings.items():
            redis_key = self._make_key(key)
            pipe.setex(redis_key, self.ttl, json.dumps(embedding))
        
        pipe.execute()
    
    def get_stats(self) -> dict:
        info = self.client.info("memory")
        keys = self.client.keys(f"{self.prefix}*")
        
        return {
            "embedding_keys": len(keys),
            "used_memory": info.get("used_memory_human"),
        }

# Usage
cache = RedisEmbeddingCache(
    host="localhost",
    port=6379,
    prefix="openai:emb:",
    ttl_seconds=86400 * 30,  # 30 days
)
```

### Batch Operations with Redis

```python
def embed_batch_with_cache(
    texts: list[str],
    cache: RedisEmbeddingCache,
    model: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Embed texts with batch caching."""
    client = OpenAI()
    
    # Generate cache keys
    keys = [create_full_cache_key(t, model) for t in texts]
    
    # Check cache in batch
    cached = cache.get_batch(keys)
    
    # Separate hits and misses
    results = {}
    texts_to_embed = []
    keys_to_embed = []
    
    for text, key in zip(texts, keys):
        if cached.get(key):
            results[key] = cached[key]
        else:
            texts_to_embed.append(text)
            keys_to_embed.append(key)
    
    # Embed misses
    if texts_to_embed:
        response = client.embeddings.create(
            model=model,
            input=texts_to_embed,
        )
        
        new_embeddings = {}
        for key, item in zip(keys_to_embed, response.data):
            results[key] = item.embedding
            new_embeddings[key] = item.embedding
        
        # Cache new embeddings in batch
        cache.set_batch(new_embeddings)
    
    # Return in original order
    return [results[key] for key in keys]
```

---

## Cache Invalidation

### Time-Based (TTL)

```python
# Redis handles TTL automatically
cache = RedisEmbeddingCache(ttl_seconds=86400 * 7)  # Expire after 7 days

# For in-memory, check on access
def get_with_ttl(self, key: str) -> Optional[list[float]]:
    entry = self.cache.get(key)
    if entry and self._is_expired(entry):
        del self.cache[key]
        return None
    return entry.embedding if entry else None
```

### Version-Based

```python
# Bump version when model or preprocessing changes
CACHE_VERSION = "v2"  # Was "v1"

# Old "v1:..." keys are now stale but harmless
# They'll expire naturally via TTL
```

### Explicit Invalidation

```python
def invalidate_document(document_id: str, cache: RedisEmbeddingCache):
    """Invalidate cache when document is updated."""
    # If you store document_id → cache_key mapping
    cache_key = get_cache_key_for_document(document_id)
    cache.client.delete(cache.prefix + cache_key)

def invalidate_all(cache: RedisEmbeddingCache):
    """Nuclear option: clear all embeddings."""
    keys = cache.client.keys(f"{cache.prefix}*")
    if keys:
        cache.client.delete(*keys)
```

---

## Production Caching Pattern

Here's a complete production-ready caching layer:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional
import hashlib
import json

class CacheBackend(Protocol):
    """Interface for cache backends."""
    def get(self, key: str) -> Optional[list[float]]: ...
    def set(self, key: str, embedding: list[float]) -> None: ...
    def get_batch(self, keys: list[str]) -> dict[str, Optional[list[float]]]: ...
    def set_batch(self, embeddings: dict[str, list[float]]) -> None: ...

@dataclass
class CacheConfig:
    version: str = "v1"
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None

class CachedEmbeddingService:
    """Production embedding service with caching."""
    
    def __init__(self, cache: CacheBackend, config: CacheConfig):
        self.cache = cache
        self.config = config
        self.client = OpenAI()
        self.stats = {"cache_hits": 0, "cache_misses": 0, "api_calls": 0}
    
    def _cache_key(self, text: str) -> str:
        content = f"{self.config.version}|{self.config.model}|{self.config.dimensions}|{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def embed(self, text: str) -> list[float]:
        """Get embedding for single text."""
        key = self._cache_key(text)
        
        cached = self.cache.get(key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached
        
        self.stats["cache_misses"] += 1
        self.stats["api_calls"] += 1
        
        kwargs = {"model": self.config.model, "input": text}
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions
        
        response = self.client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding
        
        self.cache.set(key, embedding)
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        keys = [self._cache_key(t) for t in texts]
        cached = self.cache.get_batch(keys)
        
        # Collect results and identify misses
        results = {}
        misses = []
        
        for text, key in zip(texts, keys):
            if cached.get(key):
                results[key] = cached[key]
                self.stats["cache_hits"] += 1
            else:
                misses.append((text, key))
                self.stats["cache_misses"] += 1
        
        # Embed misses
        if misses:
            self.stats["api_calls"] += 1
            
            kwargs = {
                "model": self.config.model,
                "input": [t for t, _ in misses],
            }
            if self.config.dimensions:
                kwargs["dimensions"] = self.config.dimensions
            
            response = self.client.embeddings.create(**kwargs)
            
            new_cache_entries = {}
            for (text, key), item in zip(misses, response.data):
                results[key] = item.embedding
                new_cache_entries[key] = item.embedding
            
            self.cache.set_batch(new_cache_entries)
        
        return [results[key] for key in keys]
    
    def get_stats(self) -> dict:
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.1%}",
            "requests_saved": self.stats["cache_hits"],
        }

# Usage
cache_backend = RedisEmbeddingCache()
config = CacheConfig(version="v1", model="text-embedding-3-small")
service = CachedEmbeddingService(cache_backend, config)

# Single embedding
emb = service.embed("Hello, world!")

# Batch embedding
embeddings = service.embed_batch(["Hello", "World", "Test"])

# Check stats
print(service.get_stats())
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Include model config in cache key | Use text-only keys |
| Version your cache keys | Change preprocessing without invalidation |
| Use batch operations | Make individual Redis calls in loops |
| Set reasonable TTLs | Keep embeddings forever |
| Monitor cache hit rates | Ignore cache performance |
| Pre-warm cache for common queries | Wait for production traffic |

---

## Hands-on Exercise

### Your Task

Build a caching layer that:

1. Supports multiple backends (memory and file)
2. Tracks hit/miss statistics
3. Handles batch operations efficiently
4. Includes cache warming capability

### Requirements

1. Implement `CacheBackend` protocol for both in-memory and SQLite
2. Add statistics tracking (hits, misses, hit rate)
3. Implement `warm_cache(texts)` method for pre-populating
4. Test with at least 100 texts, showing cache improvement

<details>
<summary>💡 Hints</summary>

- Use the protocol class as a base
- Track stats in the service layer, not backend
- For warming, just call `embed_batch` with common queries

</details>

<details>
<summary>✅ Solution</summary>

```python
from typing import Optional, Dict, List
import json
import sqlite3
import time

# In-memory backend
class MemoryCache:
    def __init__(self):
        self.store: Dict[str, list[float]] = {}
    
    def get(self, key: str) -> Optional[list[float]]:
        return self.store.get(key)
    
    def set(self, key: str, embedding: list[float]) -> None:
        self.store[key] = embedding
    
    def get_batch(self, keys: list[str]) -> Dict[str, Optional[list[float]]]:
        return {k: self.store.get(k) for k in keys}
    
    def set_batch(self, embeddings: Dict[str, list[float]]) -> None:
        self.store.update(embeddings)

# SQLite backend
class SQLiteCache:
    def __init__(self, path: str = "cache.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                embedding TEXT
            )
        """)
    
    def get(self, key: str) -> Optional[list[float]]:
        cur = self.conn.execute("SELECT embedding FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None
    
    def set(self, key: str, embedding: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?)",
            (key, json.dumps(embedding))
        )
        self.conn.commit()
    
    def get_batch(self, keys: list[str]) -> Dict[str, Optional[list[float]]]:
        return {k: self.get(k) for k in keys}
    
    def set_batch(self, embeddings: Dict[str, list[float]]) -> None:
        for k, v in embeddings.items():
            self.set(k, v)

# Embedding service with stats
class CachedEmbedder:
    def __init__(self, cache):
        self.cache = cache
        self.stats = {"hits": 0, "misses": 0, "api_calls": 0}
    
    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def embed(self, text: str) -> list[float]:
        key = self._key(text)
        cached = self.cache.get(key)
        
        if cached:
            self.stats["hits"] += 1
            return cached
        
        self.stats["misses"] += 1
        self.stats["api_calls"] += 1
        
        # Simulate API call (replace with real call)
        embedding = [0.1] * 1536  # Placeholder
        
        self.cache.set(key, embedding)
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = self.cache.get_batch(keys)
        
        results = {}
        to_embed = []
        
        for text, key in zip(texts, keys):
            if cached.get(key):
                results[key] = cached[key]
                self.stats["hits"] += 1
            else:
                to_embed.append((text, key))
                self.stats["misses"] += 1
        
        if to_embed:
            self.stats["api_calls"] += 1
            new_entries = {}
            for text, key in to_embed:
                emb = [0.1] * 1536  # Placeholder
                results[key] = emb
                new_entries[key] = emb
            self.cache.set_batch(new_entries)
        
        return [results[key] for key in keys]
    
    def warm_cache(self, texts: list[str]) -> None:
        """Pre-populate cache with common queries."""
        print(f"Warming cache with {len(texts)} texts...")
        self.embed_batch(texts)
        print(f"Cache warmed. Stats: {self.get_stats()}")
    
    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "hit_rate": f"{rate:.1%}"}

# Test
print("Testing with Memory Cache:")
embedder = CachedEmbedder(MemoryCache())

# Warm cache
common_queries = [f"Query {i}" for i in range(50)]
embedder.warm_cache(common_queries)

# Simulate usage (50% repeat queries)
test_queries = common_queries[:25] + [f"New query {i}" for i in range(25)]
start = time.time()
embedder.embed_batch(test_queries)
print(f"Final stats: {embedder.get_stats()}")
print(f"Time: {time.time() - start:.3f}s")
```

</details>

---

## Summary

✅ Caching reduces API costs and latency significantly (80-99% reduction possible)

✅ Include model configuration in cache keys to avoid stale embeddings

✅ Use Redis for production multi-worker deployments

✅ Implement TTL and version-based invalidation

✅ Use batch operations for efficiency—avoid individual cache calls in loops

✅ Monitor hit rates and warm caches for known common queries

**Next:** [Vector Databases](../04-vector-databases/00-vector-databases.md)

---

## Further Reading

- [Redis Caching Best Practices](https://redis.io/docs/manual/patterns/)
- [SQLite as a Cache](https://www.sqlite.org/whentouse.html)
- [Caching Strategies Overview](https://aws.amazon.com/caching/)

---

<!-- 
Sources Consulted:
- Redis Documentation: https://redis.io/docs/
- Python functools: https://docs.python.org/3/library/functools.html
- SQLite Documentation: https://www.sqlite.org/docs.html
-->
