---
title: "Managed Vector Databases"
---

# Managed Vector Databases

## Introduction

Managed vector databases remove the operational burden of running vector search infrastructure. You get automatic scaling, high availability, and zero maintenance—perfect for teams that want to focus on building features rather than managing databases.

In this lesson, we'll explore the major managed vector database providers in 2025, their APIs, features, and how to choose between them.

### What We'll Cover

- Pinecone Serverless: pay-per-query, auto-scaling
- Weaviate Cloud: hybrid search built-in
- Qdrant Cloud: filtering excellence
- Zilliz Cloud: managed Milvus for enterprise
- Feature comparison and selection criteria

### Prerequisites

- Understanding of vector database concepts from the previous lesson
- API keys from at least one provider (for hands-on exercises)

---

## Pinecone Serverless

Pinecone pioneered the purpose-built vector database space. Their serverless offering in 2025 provides pay-per-query pricing with automatic scaling.

### Key Features

- **Integrated embedding models** - Text to vector conversion built-in
- **Namespaces** - Logical partitions within an index
- **Metadata filtering** - Query-time filtering on stored metadata
- **Reranking** - Built-in BGE reranker for improved accuracy
- **Sparse + Dense** - Hybrid search with separate indexes

### Quick Start

```python
from pinecone import Pinecone

# Initialize client
pc = Pinecone(api_key="YOUR_API_KEY")

# Create a dense index with integrated embedding
index_name = "my-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "content"}
        }
    )

# Get index reference
index = pc.Index(index_name)

# Upsert records with text (auto-embedded)
records = [
    {"_id": "doc1", "content": "Introduction to machine learning", "category": "tech"},
    {"_id": "doc2", "content": "Cooking pasta recipes", "category": "food"},
    {"_id": "doc3", "content": "Neural network architectures", "category": "tech"},
]
index.upsert_records("my-namespace", records)

# Semantic search
results = index.search(
    namespace="my-namespace",
    query={
        "top_k": 5,
        "inputs": {"text": "AI and deep learning"}
    }
)

for hit in results['result']['hits']:
    print(f"{hit['_id']}: {hit['_score']:.3f} - {hit['fields']['content']}")
```

### With Your Own Embeddings

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Create index for BYOV (Bring Your Own Vectors)
pc.create_index(
    name="custom-embeddings",
    dimension=1536,  # Match your embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("custom-embeddings")

# Upsert vectors directly
index.upsert(
    vectors=[
        {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"source": "wiki"}},
        {"id": "doc2", "values": [0.3, 0.4, ...], "metadata": {"source": "docs"}},
    ],
    namespace="production"
)

# Query with vector
results = index.query(
    namespace="production",
    vector=[0.15, 0.25, ...],
    top_k=10,
    filter={"source": {"$eq": "wiki"}},
    include_metadata=True
)
```

### Metadata Filtering

Pinecone supports rich filtering syntax:

```python
# Filter examples
filters = {
    # Equality
    "category": {"$eq": "tech"},
    
    # Comparison
    "price": {"$lt": 100},
    
    # In list
    "status": {"$in": ["active", "pending"]},
    
    # Combined (AND)
    "$and": [
        {"category": {"$eq": "tech"}},
        {"price": {"$lt": 100}}
    ],
    
    # Combined (OR)
    "$or": [
        {"category": {"$eq": "tech"}},
        {"featured": {"$eq": True}}
    ]
}

results = index.query(
    vector=query_embedding,
    top_k=10,
    filter=filters
)
```

### Pricing (2025)

| Component | Starter (Free) | Standard | Enterprise |
|-----------|----------------|----------|------------|
| Vectors | 2M | Unlimited | Unlimited |
| Namespaces | 100 | 100,000 | Unlimited |
| Queries/month | 500K | Pay-per-use | Custom |
| Support | Community | Standard | Dedicated |

---

## Weaviate Cloud

Weaviate differentiates with built-in hybrid search (vectors + keywords) and tight model provider integrations.

### Key Features

- **Hybrid search** - BM25 + vector search combined
- **Generative search** - RAG built into queries
- **Multi-tenancy** - Native tenant isolation
- **Module ecosystem** - Integrations with OpenAI, Cohere, etc.
- **GraphQL API** - Flexible query language

### Quick Start

```python
import weaviate
from weaviate.classes.config import Configure
import os

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["WEAVIATE_URL"],
    auth_credentials=os.environ["WEAVIATE_API_KEY"],
)

# Create collection with vectorizer
client.collections.create(
    name="Article",
    vector_config=Configure.Vectors.text2vec_openai(),
    properties=[
        {"name": "title", "data_type": "text"},
        {"name": "content", "data_type": "text"},
        {"name": "category", "data_type": "text"},
    ]
)

# Get collection reference
articles = client.collections.get("Article")

# Add objects (auto-vectorized)
with articles.batch.fixed_size(batch_size=100) as batch:
    batch.add_object(properties={
        "title": "Introduction to Vector Search",
        "content": "Vector search enables semantic similarity...",
        "category": "technology"
    })

# Semantic search
response = articles.query.near_text(
    query="machine learning basics",
    limit=5
)

for obj in response.objects:
    print(obj.properties["title"])
```

### Hybrid Search

Weaviate's killer feature is combining vector search with BM25:

```python
# Hybrid search: combines vector + keyword results
response = articles.query.hybrid(
    query="neural network architecture",
    alpha=0.5,  # 0 = pure BM25, 1 = pure vector
    limit=10
)

# With filters
from weaviate.classes.query import Filter

response = articles.query.hybrid(
    query="machine learning",
    alpha=0.7,
    filters=Filter.by_property("category").equal("technology"),
    limit=10
)
```

### Generative Search (RAG)

```python
from weaviate.classes.generate import GenerativeConfig

# Search + generate in one query
response = articles.generate.near_text(
    query="What is vector search?",
    limit=3,
    grouped_task="Summarize these articles in one paragraph.",
    generative_provider=GenerativeConfig.openai()
)

print(response.generative.text)  # Generated summary
```

---

## Qdrant Cloud

Qdrant excels at complex filtering and offers excellent price/performance for large-scale deployments.

### Key Features

- **Advanced filtering** - Nested conditions, geo-search, full-text
- **Quantization** - Scalar and binary quantization built-in
- **Multi-vectors** - Multiple vectors per point
- **Snapshot/restore** - Easy backup and migration
- **Rust-based** - High performance, memory-efficient

### Quick Start

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="YOUR_API_KEY"
)

# Create collection
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Upsert points
client.upsert(
    collection_name="products",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"name": "Laptop", "price": 999, "category": "electronics"}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"name": "Headphones", "price": 199, "category": "electronics"}
        ),
    ]
)

# Search with filter
results = client.query_points(
    collection_name="products",
    query=[0.15, 0.25, ...],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}},
            {"key": "price", "range": {"lte": 500}}
        ]
    },
    limit=10
)

for point in results.points:
    print(f"{point.id}: {point.score:.3f} - {point.payload}")
```

### Advanced Filtering

Qdrant supports the most powerful filtering in the industry:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Nested filters
complex_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="electronics")),
        Filter(
            should=[
                FieldCondition(key="brand", match=MatchValue(value="Apple")),
                FieldCondition(key="brand", match=MatchValue(value="Samsung")),
            ]
        )
    ],
    must_not=[
        FieldCondition(key="discontinued", match=MatchValue(value=True))
    ]
)

# Geo filter
geo_filter = Filter(
    must=[
        {"key": "location", "geo_radius": {
            "center": {"lat": 40.7128, "lon": -74.0060},
            "radius": 10000  # 10km
        }}
    ]
)

# Full-text filter (requires text index)
text_filter = Filter(
    must=[
        {"key": "description", "match": {"text": "wireless bluetooth"}}
    ]
)
```

---

## Zilliz Cloud (Managed Milvus)

Zilliz Cloud provides managed Milvus, ideal for enterprise workloads requiring billion-scale vector search.

### Key Features

- **Billion-scale** - Proven at massive scale
- **GPU acceleration** - Optional GPU indexes
- **Multiple index types** - IVF, HNSW, DiskANN, SCANN
- **Dynamic schema** - Flexible field definitions
- **Enterprise features** - RBAC, audit logs, compliance

### Quick Start

```python
from pymilvus import MilvusClient

# Connect to Zilliz Cloud
client = MilvusClient(
    uri="https://your-cluster.zillizcloud.com",
    token="YOUR_API_KEY"
)

# Create collection
client.create_collection(
    collection_name="articles",
    dimension=1536,
    metric_type="COSINE"
)

# Insert data
data = [
    {"id": 1, "vector": [0.1, 0.2, ...], "title": "Article 1"},
    {"id": 2, "vector": [0.3, 0.4, ...], "title": "Article 2"},
]
client.insert(collection_name="articles", data=data)

# Search
results = client.search(
    collection_name="articles",
    data=[[0.15, 0.25, ...]],
    limit=10,
    output_fields=["title"]
)

for hits in results:
    for hit in hits:
        print(f"{hit['id']}: {hit['distance']:.3f} - {hit['entity']['title']}")
```

---

## Provider Comparison

| Feature | Pinecone | Weaviate | Qdrant | Zilliz |
|---------|----------|----------|--------|--------|
| **Pricing Model** | Pay-per-query | Capacity-based | Capacity-based | Capacity-based |
| **Free Tier** | 2M vectors | Sandbox | 1GB free | 100K vectors |
| **Hybrid Search** | Separate indexes | Native BM25+vector | Full-text filter | BM25 support |
| **Integrated Embeddings** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Max Dimensions** | 20,000 | Unlimited | Unlimited | 32,768 |
| **Multi-tenancy** | Namespaces | Native | Payload filter | Partitions |
| **Best For** | Simplicity, serverless | Hybrid search, RAG | Complex filters | Enterprise scale |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Start with free tiers to evaluate | Lock into annual contracts early |
| Test with production-like data | Benchmark with random vectors |
| Monitor query latency percentiles | Only look at average latency |
| Use namespaces/collections for isolation | Mix unrelated data |
| Index metadata fields you filter on | Store large text in metadata |

---

## Hands-on Exercise

### Your Task

Set up a Pinecone or Qdrant Cloud account and implement semantic search with metadata filtering:

### Requirements

1. Create a free account with your chosen provider
2. Create an index/collection for movie data
3. Insert 10+ movie records with title, description, genre, year
4. Implement search filtered by genre
5. Compare filtered vs unfiltered results

<details>
<summary>💡 Hints</summary>

- Pinecone: Use `create_index_for_model()` for auto-embedding
- Qdrant: Use `client.create_collection()` with payload fields
- Both have generous free tiers

</details>

<details>
<summary>✅ Solution (Qdrant)</summary>

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# Initialize clients
qdrant = QdrantClient(url="YOUR_URL", api_key="YOUR_KEY")
openai = OpenAI()

# Create collection
qdrant.create_collection(
    collection_name="movies",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Movie data
movies = [
    {"title": "The Matrix", "description": "A hacker discovers reality is a simulation", "genre": "sci-fi", "year": 1999},
    {"title": "Inception", "description": "Thieves enter dreams to steal secrets", "genre": "sci-fi", "year": 2010},
    {"title": "The Godfather", "description": "A crime family saga", "genre": "drama", "year": 1972},
    {"title": "Toy Story", "description": "Toys come to life when humans aren't looking", "genre": "animation", "year": 1995},
    {"title": "Interstellar", "description": "Astronauts travel through a wormhole", "genre": "sci-fi", "year": 2014},
]

# Generate embeddings and insert
points = []
for i, movie in enumerate(movies):
    text = f"{movie['title']}: {movie['description']}"
    response = openai.embeddings.create(model="text-embedding-3-small", input=text)
    
    points.append(PointStruct(
        id=i,
        vector=response.data[0].embedding,
        payload=movie
    ))

qdrant.upsert(collection_name="movies", points=points)

# Search function
def search_movies(query: str, genre: str = None, limit: int = 5):
    response = openai.embeddings.create(model="text-embedding-3-small", input=query)
    query_vector = response.data[0].embedding
    
    query_filter = None
    if genre:
        query_filter = {"must": [{"key": "genre", "match": {"value": genre}}]}
    
    return qdrant.query_points(
        collection_name="movies",
        query=query_vector,
        query_filter=query_filter,
        limit=limit
    )

# Test searches
print("All genres:")
for p in search_movies("space exploration").points:
    print(f"  {p.payload['title']} ({p.payload['genre']}): {p.score:.3f}")

print("\nSci-fi only:")
for p in search_movies("space exploration", genre="sci-fi").points:
    print(f"  {p.payload['title']} ({p.payload['genre']}): {p.score:.3f}")
```

</details>

---

## Summary

✅ Pinecone Serverless offers simplicity with pay-per-query pricing and integrated embeddings

✅ Weaviate Cloud provides native hybrid search combining BM25 and vectors

✅ Qdrant Cloud excels at complex filtering and offers excellent performance

✅ Zilliz Cloud (managed Milvus) handles billion-scale enterprise workloads

✅ Start with free tiers to evaluate before committing

**Next:** [Self-Hosted Options](./03-self-hosted-options.md)

---

## Further Reading

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Weaviate Cloud Docs](https://weaviate.io/developers/weaviate)
- [Qdrant Cloud Docs](https://qdrant.tech/documentation/cloud/)
- [Zilliz Cloud Docs](https://docs.zilliz.com/)

---

<!-- 
Sources Consulted:
- Pinecone quickstart: https://docs.pinecone.io/guides/get-started/quickstart
- Pinecone indexing overview: https://docs.pinecone.io/guides/indexes/understanding-indexes
- Weaviate quickstart: https://docs.weaviate.io/weaviate/quickstart
- Qdrant quickstart: https://qdrant.tech/documentation/quickstart/
- Qdrant filtering: https://qdrant.tech/documentation/concepts/filtering/
-->
