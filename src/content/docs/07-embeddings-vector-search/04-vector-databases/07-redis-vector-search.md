---
title: "Redis Vector Search"
---

# Redis Vector Search

## Introduction

Redis Stack adds vector similarity search to Redis, the world's fastest in-memory database. With sub-millisecond latency and Redis's operational simplicity, it's ideal for real-time recommendations, caching, and session-aware search.

### What We'll Cover

- Redis Stack setup
- Vector index types (HNSW, FLAT)
- Hybrid search with tags and text
- Real-time use cases
- Performance characteristics

### Prerequisites

- Redis Stack or Redis Cloud
- Basic Redis knowledge
- Understanding of embeddings

---

## Redis Stack Setup

### Docker (Local Development)

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

Port 8001 provides RedisInsight—a web-based admin UI.

### Redis Cloud

1. Create account at [redis.com/cloud](https://redis.com/cloud)
2. Create a Redis Stack database (includes Search module)
3. Copy connection string from dashboard

### Connect with Python

```python
import redis
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Or Redis Cloud
r = redis.Redis.from_url("redis://user:password@host:port")

# Test connection
r.ping()  # Returns True
```

---

## Creating Vector Indexes

Redis uses RediSearch under the hood for vector indexing.

### Index Definition

```python
from redis.commands.search.field import VectorField, TextField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Define schema
schema = [
    TextField("title"),
    TextField("content"),
    TagField("category"),
    NumericField("price"),
    VectorField(
        "embedding",
        "HNSW",  # or "FLAT"
        {
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE",
            "M": 16,
            "EF_CONSTRUCTION": 200
        }
    )
]

# Create index on hash keys with prefix "doc:"
r.ft("idx:docs").create_index(
    schema,
    definition=IndexDefinition(
        prefix=["doc:"],
        index_type=IndexType.HASH
    )
)
```

### Index Types

| Type | Best For | Trade-off |
|------|----------|-----------|
| **HNSW** | Large datasets (>10K) | Memory for speed |
| **FLAT** | Small datasets (<10K) | Exact results, slower |

### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Edges per node (8-64) |
| `EF_CONSTRUCTION` | 200 | Build quality (100-500) |
| `EF_RUNTIME` | 10 | Query quality (set at query time) |

---

## Storing Vectors

Redis stores vectors in Hash or JSON format.

### Using Hashes

```python
import numpy as np
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Store document with embedding
doc = {
    "title": "Introduction to Machine Learning",
    "content": "Machine learning is a subset of AI...",
    "category": "AI",
    "price": "29.99"
}

embedding = get_embedding(doc["title"] + " " + doc["content"])
embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

r.hset("doc:1", mapping={
    **doc,
    "embedding": embedding_bytes
})
```

### Batch Insert

```python
# Use pipeline for batch operations
pipe = r.pipeline()

documents = [
    {"id": "1", "title": "ML Basics", "category": "AI"},
    {"id": "2", "title": "Python Tips", "category": "Programming"},
    # ... more documents
]

for doc in documents:
    embedding = get_embedding(doc["title"])
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    pipe.hset(f"doc:{doc['id']}", mapping={
        "title": doc["title"],
        "category": doc["category"],
        "embedding": embedding_bytes
    })

pipe.execute()  # Executes all commands in batch
```

---

## Vector Search Queries

### Basic KNN Search

```python
from redis.commands.search.query import Query

def search(query_text: str, k: int = 5):
    query_embedding = get_embedding(query_text)
    query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    
    q = (
        Query("*=>[KNN $k @embedding $vec AS score]")
        .return_fields("title", "category", "score")
        .sort_by("score")
        .dialect(2)
    )
    
    params = {
        "k": k,
        "vec": query_bytes
    }
    
    results = r.ft("idx:docs").search(q, params)
    return results

# Search
results = search("machine learning tutorials")
for doc in results.docs:
    print(f"{doc.title}: {float(doc.score):.4f}")
```

### With EF_RUNTIME (Quality Tuning)

```python
# Higher EF_RUNTIME = better recall, slower query
q = Query("*=>[KNN $k @embedding $vec EF_RUNTIME 100 AS score]")
```

---

## Hybrid Search

Combine vector similarity with filters and full-text search.

### Filter by Tag

```python
# Search only in "AI" category
q = (
    Query("@category:{AI}=>[KNN $k @embedding $vec AS score]")
    .return_fields("title", "category", "score")
    .sort_by("score")
    .dialect(2)
)
```

### Filter by Numeric Range

```python
# Products under $50
q = (
    Query("@price:[0 50]=>[KNN $k @embedding $vec AS score]")
    .return_fields("title", "price", "score")
    .sort_by("score")
    .dialect(2)
)
```

### Combined Filters

```python
# AI category AND price under $100
q = (
    Query("(@category:{AI} @price:[0 100])=>[KNN $k @embedding $vec AS score]")
    .return_fields("title", "category", "price", "score")
    .sort_by("score")
    .dialect(2)
)
```

### Full-Text + Vector (Hybrid)

```python
# Documents containing "python" ranked by vector similarity
q = (
    Query("@content:python=>[KNN $k @embedding $vec AS score]")
    .return_fields("title", "score")
    .sort_by("score")
    .dialect(2)
)
```

---

## Real-Time Use Cases

Redis's in-memory architecture makes it ideal for real-time scenarios:

### Session-Aware Recommendations

```python
class RealtimeRecommender:
    def __init__(self, redis_client):
        self.r = redis_client
        
    def record_view(self, user_id: str, product_embedding: list[float]):
        """Track user's recent product views"""
        key = f"user:{user_id}:recent_views"
        self.r.lpush(key, np.array(product_embedding, dtype=np.float32).tobytes())
        self.r.ltrim(key, 0, 9)  # Keep last 10 views
        self.r.expire(key, 3600)  # Expire after 1 hour
        
    def get_recommendations(self, user_id: str, k: int = 5):
        """Recommend based on average of recent views"""
        key = f"user:{user_id}:recent_views"
        recent_embeddings = self.r.lrange(key, 0, -1)
        
        if not recent_embeddings:
            return self._popular_products(k)
        
        # Average recent embeddings
        vectors = [np.frombuffer(e, dtype=np.float32) for e in recent_embeddings]
        avg_vector = np.mean(vectors, axis=0)
        
        # Search for similar products
        q = Query("*=>[KNN $k @embedding $vec AS score]")
        results = self.r.ft("idx:products").search(q, {
            "k": k,
            "vec": avg_vector.tobytes()
        })
        
        return results
```

### Semantic Cache

```python
class SemanticCache:
    """Cache LLM responses with semantic matching"""
    
    def __init__(self, redis_client, similarity_threshold: float = 0.95):
        self.r = redis_client
        self.threshold = similarity_threshold
        
    def get_or_generate(self, query: str, generate_fn):
        """Return cached response if similar query exists"""
        query_embedding = get_embedding(query)
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Search for similar cached queries
        q = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .return_fields("response", "score")
            .dialect(2)
        )
        
        results = self.r.ft("idx:cache").search(q, {"vec": query_bytes})
        
        if results.docs and float(results.docs[0].score) >= self.threshold:
            return results.docs[0].response  # Cache hit
        
        # Cache miss - generate new response
        response = generate_fn(query)
        
        # Store in cache
        cache_key = f"cache:{hash(query)}"
        self.r.hset(cache_key, mapping={
            "query": query,
            "response": response,
            "embedding": query_bytes
        })
        self.r.expire(cache_key, 3600)  # 1 hour TTL
        
        return response
```

---

## Performance Characteristics

### Latency Comparison

| Operation | Redis Vector | Typical Vector DB |
|-----------|--------------|-------------------|
| Single query | <1ms | 5-20ms |
| Batch query (100) | <10ms | 50-200ms |
| Insert | <1ms | 1-5ms |

### Memory Usage

| Dimensions | Vectors | Approximate Memory |
|------------|---------|-------------------|
| 1536 | 100K | ~800 MB |
| 1536 | 1M | ~8 GB |
| 1536 | 10M | ~80 GB |

> **Note:** Memory = vectors × dimensions × 4 bytes × ~1.3 (index overhead)

### Scaling Limits

| Metric | Recommended Max |
|--------|-----------------|
| Vectors per shard | 10M |
| Dimensions | 4096 |
| Vector size | 64KB |

---

## Redis vs Purpose-Built Vector DBs

| Feature | Redis Vector | Pinecone/Qdrant |
|---------|--------------|-----------------|
| **Latency** | <1ms | 5-50ms |
| **Scale** | 10M per shard | Billions |
| **Memory model** | In-memory | Disk + cache |
| **Persistence** | RDB/AOF snapshots | Built-in |
| **Best for** | Caching, real-time | Large-scale search |
| **Filtering** | Good | Excellent |

### When to Use Redis Vector

✅ **Use Redis when:**
- Sub-millisecond latency is critical
- You already use Redis in your stack
- Dataset fits in memory (<10M vectors)
- Real-time recommendations or caching

❌ **Consider alternatives when:**
- You have billions of vectors
- Memory costs are a concern
- You need complex metadata filtering
- Disk-based persistence is required

---

## Hands-on Exercise

### Your Task

Build a real-time product recommendation cache:

### Requirements

1. Set up Redis Stack with vector index
2. Insert 50+ products with embeddings
3. Implement view tracking per user
4. Create recommendations based on recent views
5. Add a "similar products" feature

<details>
<summary>💡 Hints</summary>

- Use Redis Lists (`LPUSH`, `LTRIM`) for recent views
- Average embeddings with numpy for user preference
- Exclude already-viewed products from recommendations

</details>

<details>
<summary>✅ Solution</summary>

```python
import redis
import numpy as np
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from openai import OpenAI

# Setup
r = redis.Redis(host='localhost', port=6379, decode_responses=False)
openai_client = OpenAI()

def embed(text):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# 1. Create index
schema = [
    TextField("name"),
    TagField("category"),
    VectorField("embedding", "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE"
    })
]

try:
    r.ft("idx:products").create_index(
        schema,
        definition=IndexDefinition(prefix=["product:"], index_type=IndexType.HASH)
    )
except:
    pass  # Index exists

# 2. Insert products
products = [
    {"id": "1", "name": "Wireless Headphones", "category": "Electronics"},
    {"id": "2", "name": "Running Shoes", "category": "Sports"},
    # ... add more
]

for p in products:
    embedding = embed(p["name"])
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    r.hset(f"product:{p['id']}", mapping={
        "name": p["name"].encode(),
        "category": p["category"].encode(),
        "embedding": embedding_bytes
    })

# 3. View tracking
def track_view(user_id: str, product_id: str):
    # Get product embedding
    embedding = r.hget(f"product:{product_id}", "embedding")
    if embedding:
        key = f"user:{user_id}:views"
        r.lpush(key, embedding)
        r.ltrim(key, 0, 9)
        r.expire(key, 3600)

# 4. Recommendations
def get_recommendations(user_id: str, k: int = 5):
    key = f"user:{user_id}:views"
    recent = r.lrange(key, 0, -1)
    
    if not recent:
        return []
    
    vectors = [np.frombuffer(e, dtype=np.float32) for e in recent]
    avg = np.mean(vectors, axis=0).astype(np.float32).tobytes()
    
    q = (
        Query("*=>[KNN $k @embedding $vec AS score]")
        .return_fields("name", "category", "score")
        .sort_by("score")
        .dialect(2)
    )
    
    return r.ft("idx:products").search(q, {"k": k, "vec": avg})

# 5. Similar products
def similar_products(product_id: str, k: int = 5):
    embedding = r.hget(f"product:{product_id}", "embedding")
    
    q = (
        Query("*=>[KNN $k @embedding $vec AS score]")
        .return_fields("name", "score")
        .sort_by("score")
        .dialect(2)
    )
    
    results = r.ft("idx:products").search(q, {"k": k + 1, "vec": embedding})
    # Exclude self
    return [d for d in results.docs if not d.id.endswith(product_id)][:k]

# Test
track_view("user123", "1")
track_view("user123", "2")
recs = get_recommendations("user123")
print([doc.name for doc in recs.docs])
```

</details>

---

## Summary

✅ Redis Vector Search provides sub-millisecond latency for real-time applications

✅ Best for datasets under 10M vectors that fit in memory

✅ Hybrid search combines vectors with tags, text, and numeric filters

✅ Perfect for caching, session-aware recommendations, and real-time features

✅ Use HNSW for large datasets, FLAT for small exact-match needs

**Next:** [Platform-Integrated Vector Stores](./08-platform-integrated-stores.md)

---

## Further Reading

- [Redis Vector Search Docs](https://redis.io/docs/stack/search/reference/vectors/)
- [RediSearch Query Syntax](https://redis.io/docs/stack/search/reference/query_syntax/)
- [Redis Stack on Docker](https://redis.io/docs/stack/get-started/install/docker/)

---

<!-- 
Sources Consulted:
- Redis Vector Similarity docs: https://redis.io/docs/stack/search/reference/vectors/
- RediSearch query syntax: https://redis.io/docs/stack/search/reference/query_syntax/
- Redis Python client: https://github.com/redis/redis-py
-->
