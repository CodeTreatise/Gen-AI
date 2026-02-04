---
title: "MongoDB Atlas Vector Search"
---

# MongoDB Atlas Vector Search

## Introduction

MongoDB Atlas Vector Search adds semantic search capabilities to your existing MongoDB infrastructure. If you already use MongoDB for your application data, you can add vector search without introducing a separate database.

### What We'll Cover

- Creating vector search indexes
- The `$vectorSearch` aggregation stage
- Pre-filtering with filter indexes
- Hybrid search with text indexes
- Python integration with PyMongo

### Prerequisites

- MongoDB Atlas account (M10+ cluster for production)
- Understanding of MongoDB aggregation pipeline
- Familiarity with embeddings

---

## Creating Vector Search Indexes

Vector search indexes in MongoDB Atlas are created through the Atlas UI, CLI, or programmatically with drivers.

### Index Definition Structure

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "category"
    },
    {
      "type": "filter",
      "path": "created_at"
    }
  ]
}
```

### Index Options

| Option | Values | Description |
|--------|--------|-------------|
| `type` | `vector`, `filter` | Field type for vector search |
| `numDimensions` | 1-8192 | Embedding dimensions |
| `similarity` | `cosine`, `euclidean`, `dotProduct` | Distance metric |
| `quantization` | `none`, `scalar`, `binary` | Compression method |

### Create with PyMongo

```python
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Connect to Atlas
client = MongoClient("mongodb+srv://user:pass@cluster.mongodb.net/")
db = client["mydb"]
collection = db["documents"]

# Define vector search index
search_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine"
            },
            {
                "type": "filter",
                "path": "category"
            },
            {
                "type": "filter",
                "path": "status"
            }
        ]
    },
    name="vector_index",
    type="vectorSearch"
)

# Create the index
collection.create_search_index(search_index_model)
print("Index created - may take a few minutes to build")
```

### Check Index Status

```python
# List all search indexes
indexes = list(collection.list_search_indexes())
for idx in indexes:
    print(f"{idx['name']}: {idx['status']}")
    # Status: PENDING, BUILDING, READY, FAILED
```

---

## The $vectorSearch Stage

`$vectorSearch` is an aggregation pipeline stage that performs vector similarity search.

### Basic Syntax

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": [0.1, 0.2, ...],  # 1536 floats
            "numCandidates": 100,
            "limit": 10
        }
    }
]

results = collection.aggregate(pipeline)
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `index` | ✅ | Name of the vector search index |
| `path` | ✅ | Field containing the vectors |
| `queryVector` | ✅ | The search vector |
| `numCandidates` | ✅ | ANN candidates to consider |
| `limit` | ✅ | Final results to return |
| `filter` | ❌ | Pre-filter on indexed fields |

> **🔑 Key Insight:** `numCandidates` should be 10-20x your `limit` for good recall. Setting `numCandidates: 100, limit: 10` means "consider 100 candidates, return top 10."

### Complete Search Example

```python
from openai import OpenAI
from pymongo import MongoClient

# Initialize clients
openai_client = OpenAI()
mongo_client = MongoClient("mongodb+srv://...")
collection = mongo_client["mydb"]["documents"]

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_documents(query: str, limit: int = 5):
    query_embedding = get_embedding(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit
            }
        },
        {
            "$project": {
                "title": 1,
                "content": 1,
                "category": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return results

# Search
results = search_documents("machine learning algorithms")
for doc in results:
    print(f"{doc['title']}: {doc['score']:.3f}")
```

---

## Pre-Filtering

Pre-filtering narrows the search to a subset of documents BEFORE vector comparison. This is fast because filters use the index.

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"status": {"$eq": "published"}}` |
| `$ne` | Not equals | `{"status": {"$ne": "draft"}}` |
| `$gt`, `$gte` | Greater than | `{"price": {"$gt": 100}}` |
| `$lt`, `$lte` | Less than | `{"rating": {"$lte": 5}}` |
| `$in` | In array | `{"category": {"$in": ["AI", "ML"]}}` |
| `$nin` | Not in array | `{"tag": {"$nin": ["archived"]}}` |
| `$and` | Logical AND | Combine conditions |
| `$or` | Logical OR | Alternative conditions |

### Filtered Search Example

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 10,
            "filter": {
                "$and": [
                    {"category": {"$eq": "technology"}},
                    {"status": {"$eq": "published"}},
                    {"created_at": {"$gte": "2024-01-01"}}
                ]
            }
        }
    }
]
```

> **Warning:** Only fields with `type: "filter"` in your index can be used in the filter. Add filter fields to your index definition before using them.

---

## Hybrid Search

Combine vector search with MongoDB's text search for hybrid semantic + keyword search.

### Setup Text Index

```python
# Create a standard text index alongside your vector index
collection.create_index([("title", "text"), ("content", "text")])
```

### Hybrid Search Pattern

```python
def hybrid_search(
    query: str,
    query_embedding: list[float],
    limit: int = 10,
    vector_weight: float = 0.7
):
    keyword_weight = 1 - vector_weight
    
    # Vector search results
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit * 2
            }
        },
        {
            "$addFields": {
                "vector_score": {"$meta": "vectorSearchScore"},
                "search_type": "vector"
            }
        }
    ]
    
    # Text search results
    text_pipeline = [
        {"$match": {"$text": {"$search": query}}},
        {"$limit": limit * 2},
        {
            "$addFields": {
                "text_score": {"$meta": "textScore"},
                "search_type": "text"
            }
        }
    ]
    
    vector_results = {doc["_id"]: doc for doc in collection.aggregate(vector_pipeline)}
    text_results = {doc["_id"]: doc for doc in collection.aggregate(text_pipeline)}
    
    # Combine and score
    all_ids = set(vector_results.keys()) | set(text_results.keys())
    combined = []
    
    for doc_id in all_ids:
        v_score = vector_results.get(doc_id, {}).get("vector_score", 0)
        t_score = text_results.get(doc_id, {}).get("text_score", 0)
        
        # Normalize text score (typically 0-10+) to 0-1 range
        normalized_text = min(t_score / 10, 1)
        
        hybrid_score = (vector_weight * v_score) + (keyword_weight * normalized_text)
        
        doc = vector_results.get(doc_id) or text_results.get(doc_id)
        doc["hybrid_score"] = hybrid_score
        combined.append(doc)
    
    # Sort by hybrid score
    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined[:limit]
```

---

## Quantization

Reduce memory usage and costs with vector quantization:

### Scalar Quantization (4x compression)

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine",
      "quantization": "scalar"
    }
  ]
}
```

### Binary Quantization (32x compression)

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "euclidean",
      "quantization": "binary"
    }
  ]
}
```

> **Note:** Binary quantization requires `euclidean` similarity. It provides massive compression but may reduce recall for some use cases.

---

## Document Structure Best Practices

### Embedding Storage Patterns

```python
# Pattern 1: Embedding with source document
{
    "_id": ObjectId("..."),
    "title": "Machine Learning Basics",
    "content": "Full content here...",
    "embedding": [0.1, 0.2, ...],  # 1536 floats
    "category": "AI",
    "created_at": datetime.utcnow()
}

# Pattern 2: Separate embeddings collection (for large documents)
# documents collection
{
    "_id": ObjectId("doc123"),
    "title": "Machine Learning Basics",
    "content": "Full content here (100KB+)...",
    "category": "AI"
}

# embeddings collection  
{
    "_id": ObjectId("..."),
    "doc_id": ObjectId("doc123"),
    "embedding": [0.1, 0.2, ...],
    "chunk_index": 0,
    "text": "First chunk of content..."
}
```

### Chunked Document Search

```python
def search_chunked_documents(query_embedding: list[float], limit: int = 5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "chunk_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit * 3  # Get more chunks to dedupe
            }
        },
        {
            "$group": {
                "_id": "$doc_id",
                "best_chunk": {"$first": "$text"},
                "best_score": {"$max": {"$meta": "vectorSearchScore"}}
            }
        },
        {"$sort": {"best_score": -1}},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "documents",
                "localField": "_id",
                "foreignField": "_id",
                "as": "document"
            }
        },
        {"$unwind": "$document"}
    ]
    
    return list(collection.aggregate(pipeline))
```

---

## Performance Tuning

### Index Configuration

```python
# HNSW options for tuning
{
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine",
            "hnswOptions": {
                "maxEdges": 32,           # Default: 16, more = better recall
                "numEdgeCandidates": 200  # Default: 100, more = better quality
            }
        }
    ]
}
```

| Parameter | Default | Range | Trade-off |
|-----------|---------|-------|-----------|
| `maxEdges` | 16 | 16-64 | ↑ = better recall, more memory |
| `numEdgeCandidates` | 100 | 100-3200 | ↑ = better index, slower build |

### Query Optimization

```python
# ✅ Good: Appropriate numCandidates
{"numCandidates": 100, "limit": 10}

# ❌ Bad: Too few candidates (poor recall)
{"numCandidates": 10, "limit": 10}

# ⚠️ Wasteful: Too many candidates (slow)
{"numCandidates": 10000, "limit": 10}
```

---

## Hands-on Exercise

### Your Task

Build a product search system with category filtering:

### Requirements

1. Create a products collection with name, description, category, price, and embedding
2. Create a vector search index with category and price filters
3. Insert 20+ products across 3-4 categories
4. Implement filtered search by category
5. Add a price range filter
6. Display results with similarity scores

<details>
<summary>💡 Hints</summary>

- Index definition needs `type: "filter"` for both category and price
- Use `$and` to combine multiple filter conditions
- `numCandidates` should be 10x your limit

</details>

<details>
<summary>✅ Solution</summary>

```python
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from openai import OpenAI

# Setup
mongo = MongoClient("mongodb+srv://...")
collection = mongo["ecommerce"]["products"]
openai_client = OpenAI()

# 1. Create index
index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine"
            },
            {"type": "filter", "path": "category"},
            {"type": "filter", "path": "price"}
        ]
    },
    name="product_search",
    type="vectorSearch"
)
collection.create_search_index(index_model)

# 2. Insert products
def embed(text):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

products = [
    {"name": "Wireless Headphones", "description": "Premium noise-canceling headphones", "category": "Electronics", "price": 299},
    {"name": "Running Shoes", "description": "Lightweight running shoes for marathons", "category": "Sports", "price": 129},
    # Add more products...
]

for p in products:
    p["embedding"] = embed(f"{p['name']} {p['description']}")
collection.insert_many(products)

# 3. Search function
def search_products(query: str, category: str = None, max_price: float = None, limit: int = 5):
    query_embedding = embed(query)
    
    # Build filter
    filters = []
    if category:
        filters.append({"category": {"$eq": category}})
    if max_price:
        filters.append({"price": {"$lte": max_price}})
    
    filter_clause = {"$and": filters} if filters else None
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "product_search",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
                **({"filter": filter_clause} if filter_clause else {})
            }
        },
        {
            "$project": {
                "name": 1,
                "category": 1,
                "price": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    return list(collection.aggregate(pipeline))

# 4. Test
results = search_products(
    query="comfortable listening music",
    category="Electronics",
    max_price=350
)
for r in results:
    print(f"{r['name']} (${r['price']}): {r['score']:.3f}")
```

</details>

---

## Summary

✅ MongoDB Atlas Vector Search integrates with existing MongoDB data

✅ `$vectorSearch` is an aggregation stage—combine with `$match`, `$project`, `$lookup`

✅ Pre-filtering requires `type: "filter"` in your index definition

✅ Hybrid search combines vector similarity with text search

✅ Quantization reduces costs by 4-32x with some recall trade-off

**Next:** [Redis Vector Search](./07-redis-vector-search.md)

---

## Further Reading

- [MongoDB Atlas Vector Search Docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Vector Search Quick Start](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/)
- [Aggregation Pipeline Reference](https://www.mongodb.com/docs/manual/reference/aggregation/)

---

<!-- 
Sources Consulted:
- MongoDB Atlas Vector Search: https://www.mongodb.com/docs/atlas/atlas-vector-search/
- $vectorSearch reference: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
- Vector search filters: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
-->
