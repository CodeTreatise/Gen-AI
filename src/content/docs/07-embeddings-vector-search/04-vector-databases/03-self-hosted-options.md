---
title: "Self-Hosted Vector Databases"
---

# Self-Hosted Vector Databases

## Introduction

Self-hosted vector databases give you full control over your data, infrastructure, and costs. They're ideal for air-gapped environments, data sovereignty requirements, or when you want to avoid ongoing cloud costs at scale.

In this lesson, we'll explore the major self-hosted options, their deployment patterns, and when to choose each.

### What We'll Cover

- Chroma: Simple, great for prototyping
- Milvus: Production-ready, billion-scale
- Qdrant: Rust-based, excellent filtering
- LanceDB: Embedded, serverless-friendly
- Docker and Kubernetes deployment

### Prerequisites

- Docker installed locally
- Understanding of vector database concepts
- Basic Kubernetes knowledge (for production deployments)

---

## Chroma

Chroma is the easiest vector database to get started with. It runs in-memory, as a persistent local database, or as a client-server deployment.

### Key Features

- **Zero configuration** - Works out of the box
- **Automatic embedding** - Built-in embedding functions
- **Python-native** - Feels like a Python library
- **Multiple persistence modes** - In-memory, local, client-server
- **Open source** - Apache 2.0 license

### Quick Start (In-Memory)

```python
import chromadb

# Ephemeral client (data lost when program ends)
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # cosine, l2, or ip
)

# Add documents (auto-embedded with default model)
collection.add(
    documents=[
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Vectors represent data in high-dimensional space"
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"topic": "AI"},
        {"topic": "programming"},
        {"topic": "math"}
    ]
)

# Query
results = collection.query(
    query_texts=["What is AI?"],
    n_results=2
)

print(results['documents'][0])
print(results['distances'][0])
```

### Persistent Storage

```python
# Persistent client (survives restarts)
client = chromadb.PersistentClient(path="./chroma_data")

# Get or create collection
collection = client.get_or_create_collection(name="my_docs")

# Use upsert to avoid duplicates
collection.upsert(
    documents=["Updated document content"],
    ids=["doc1"],
    metadatas=[{"updated": True}]
)
```

### With Pre-computed Embeddings

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Add with your own embeddings
collection.add(
    embeddings=[
        get_embedding("First document"),
        get_embedding("Second document")
    ],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "api"}, {"source": "api"}]
)

# Query with your own embedding
results = collection.query(
    query_embeddings=[get_embedding("Search query")],
    n_results=5
)
```

### Client-Server Mode (Docker)

```bash
# Run Chroma server
docker run -d \
  --name chroma \
  -p 8000:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma:latest
```

```python
# Connect to server
client = chromadb.HttpClient(host="localhost", port=8000)

# Use exactly like the in-memory client
collection = client.get_or_create_collection("documents")
```

### When to Use Chroma

| ✅ Good For | ❌ Not Ideal For |
|-------------|------------------|
| Prototyping and learning | Billion-scale production |
| Local development | High-availability requirements |
| Small to medium datasets | Complex filtering needs |
| Python-first workflows | Multi-language teams |

---

## Milvus

Milvus is the production workhorse of self-hosted vector databases. It powers search at Xiaomi, Trend Micro, and other enterprises handling billions of vectors.

### Key Features

- **Billion-scale** - Distributed architecture
- **Multiple index types** - IVF, HNSW, DiskANN, GPU indexes
- **Hybrid search** - Dense + sparse vectors
- **Schema-flexible** - Dynamic fields
- **Cloud-native** - Kubernetes-ready

### Quick Start (Docker Compose)

```yaml
# docker-compose.yml
version: '3.5'
services:
  milvus:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
    volumes:
      - ./etcd_data:/etcd
    command: etcd -advertise-client-urls=http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./minio_data:/minio_data
    command: minio server /minio_data
    ports:
      - "9000:9000"
```

```bash
docker compose up -d
```

### Python Client

```python
from pymilvus import MilvusClient

# Connect
client = MilvusClient(uri="http://localhost:19530")

# Create collection
client.create_collection(
    collection_name="documents",
    dimension=1536,
    metric_type="COSINE"
)

# Insert data
data = [
    {"id": 1, "vector": [0.1, 0.2, ...], "text": "Document 1", "category": "tech"},
    {"id": 2, "vector": [0.3, 0.4, ...], "text": "Document 2", "category": "science"},
]
client.insert(collection_name="documents", data=data)

# Create index (for large collections)
client.create_index(
    collection_name="documents",
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 64}
)

# Search
results = client.search(
    collection_name="documents",
    data=[[0.15, 0.25, ...]],
    limit=10,
    filter="category == 'tech'",
    output_fields=["text", "category"]
)
```

### When to Use Milvus

| ✅ Good For | ❌ Not Ideal For |
|-------------|------------------|
| Billion-scale production | Simple prototyping |
| High-throughput requirements | Resource-constrained environments |
| Enterprise deployments | Single-node simplicity |
| GPU acceleration needs | Quick local experiments |

---

## Qdrant (Self-Hosted)

Qdrant self-hosted provides the same powerful filtering as Qdrant Cloud with complete control over your infrastructure.

### Key Features

- **Rust-based** - Memory-efficient, high-performance
- **Advanced filtering** - Best-in-class payload filtering
- **Quantization** - Scalar and binary built-in
- **Snapshots** - Easy backup and restore
- **Multi-vector** - Multiple embeddings per point

### Docker Deployment

```bash
# Simple single-node deployment
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ./qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest
```

### Python Client

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to local instance
client = QdrantClient(url="http://localhost:6333")

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
            payload={"name": "Widget", "price": 29.99, "in_stock": True}
        )
    ]
)

# Search with complex filter
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.query_points(
    collection_name="products",
    query=[0.15, 0.25, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="in_stock", match=MatchValue(value=True)),
            FieldCondition(key="price", range=Range(lte=50.0))
        ]
    ),
    limit=10
)
```

### Distributed Deployment (Kubernetes)

```yaml
# qdrant-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
            - containerPort: 6334
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
          env:
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
  volumeClaimTemplates:
    - metadata:
        name: qdrant-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi
```

---

## LanceDB

LanceDB is an embedded vector database designed for serverless and edge deployments. It stores data in the Lance columnar format, enabling efficient storage and retrieval.

### Key Features

- **Embedded** - No separate server process
- **Lance format** - Columnar storage, fast scans
- **Cloud storage** - Direct S3/GCS integration
- **Multi-modal** - Images, text, audio in same table
- **Zero-copy** - Memory-mapped for speed

### Quick Start

```python
import lancedb
from openai import OpenAI

# Connect to local database
db = lancedb.connect("./lancedb_data")

# Create table with data
data = [
    {"id": 1, "text": "Machine learning basics", "vector": [0.1, 0.2, ...]},
    {"id": 2, "text": "Deep neural networks", "vector": [0.3, 0.4, ...]},
    {"id": 3, "text": "Natural language processing", "vector": [0.5, 0.6, ...]},
]

table = db.create_table("documents", data)

# Search
results = table.search([0.15, 0.25, ...]).limit(5).to_list()

for result in results:
    print(f"{result['id']}: {result['text']} (distance: {result['_distance']:.3f})")
```

### With Pandas Integration

```python
import pandas as pd

# Create from DataFrame
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": ["Doc 1", "Doc 2", "Doc 3"],
    "vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    "category": ["A", "B", "A"]
})

table = db.create_table("from_pandas", df)

# Filter + search
results = (
    table
    .search([0.15, 0.25])
    .where("category = 'A'")
    .limit(10)
    .to_pandas()
)
```

### Cloud Storage Backend

```python
# Connect to S3-backed database
db = lancedb.connect("s3://my-bucket/lancedb")

# Works exactly the same
table = db.create_table("cloud_docs", data)
```

### When to Use LanceDB

| ✅ Good For | ❌ Not Ideal For |
|-------------|------------------|
| Serverless functions | High-concurrency servers |
| Edge deployments | Complex filtering |
| Data science workflows | Enterprise HA requirements |
| Single-process applications | Multi-process writes |

---

## Comparison Summary

| Feature | Chroma | Milvus | Qdrant | LanceDB |
|---------|--------|--------|--------|---------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐ Complex | ⭐⭐ Medium | ⭐ Simple |
| **Scale** | Medium | Billion+ | Billion | Medium |
| **Filtering** | Basic | Good | Excellent | Good |
| **Memory Usage** | Medium | High | Low | Very Low |
| **Best For** | Prototyping | Enterprise | Production | Serverless |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

---

## Deployment Best Practices

### Resource Sizing

| Vector Count | RAM (HNSW) | Storage | Recommendation |
|--------------|------------|---------|----------------|
| < 100K | 1-2 GB | 1 GB | Chroma/LanceDB |
| 100K - 1M | 4-8 GB | 10 GB | Any |
| 1M - 10M | 16-32 GB | 50 GB | Qdrant/Milvus |
| 10M - 100M | 64-128 GB | 200 GB | Milvus/Qdrant |
| 100M+ | Distributed | 1+ TB | Milvus cluster |

### Production Checklist

- [ ] Enable persistence with mounted volumes
- [ ] Configure regular snapshots/backups
- [ ] Set resource limits in container configs
- [ ] Monitor memory and disk usage
- [ ] Configure health checks
- [ ] Plan for index rebuild time during restarts
- [ ] Test disaster recovery procedures

---

## Hands-on Exercise

### Your Task

Deploy Qdrant locally with Docker and implement a document search system:

### Requirements

1. Start Qdrant in Docker
2. Create a collection with 1536 dimensions
3. Insert 20+ text documents with embeddings
4. Implement filtered search by category
5. Create a snapshot for backup

<details>
<summary>✅ Solution</summary>

```bash
# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# Initialize clients
qdrant = QdrantClient(url="http://localhost:6333")
openai_client = OpenAI()

# Create collection
qdrant.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Sample documents
documents = [
    {"text": "Python is great for data science", "category": "programming"},
    {"text": "JavaScript powers the web", "category": "programming"},
    {"text": "Machine learning transforms industries", "category": "AI"},
    {"text": "Neural networks learn patterns", "category": "AI"},
    {"text": "Docker containers simplify deployment", "category": "devops"},
    # ... add more
]

# Generate embeddings and insert
points = []
for i, doc in enumerate(documents):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=doc["text"]
    )
    points.append(PointStruct(
        id=i,
        vector=response.data[0].embedding,
        payload=doc
    ))

qdrant.upsert(collection_name="docs", points=points)

# Search with filter
query_embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input="How to learn AI"
).data[0].embedding

results = qdrant.query_points(
    collection_name="docs",
    query=query_embedding,
    query_filter={"must": [{"key": "category", "match": {"value": "AI"}}]},
    limit=5
)

for point in results.points:
    print(f"{point.payload['text']}: {point.score:.3f}")

# Create snapshot
snapshot = qdrant.create_snapshot(collection_name="docs")
print(f"Snapshot created: {snapshot.name}")
```

</details>

---

## Summary

✅ Chroma is the simplest option—perfect for prototyping and learning

✅ Milvus handles billion-scale production workloads

✅ Qdrant offers the best filtering capabilities and memory efficiency

✅ LanceDB excels in serverless and embedded scenarios

✅ All major options are Apache 2.0 licensed and well-maintained

**Next:** [PostgreSQL with pgvector](./04-postgresql-pgvector.md)

---

## Further Reading

- [Chroma Documentation](https://docs.trychroma.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)

---

<!-- 
Sources Consulted:
- Chroma getting started: https://docs.trychroma.com/docs/overview/getting-started
- Chroma add data: https://docs.trychroma.com/docs/collections/add-data
- Qdrant quickstart: https://qdrant.tech/documentation/quickstart/
- pgvector GitHub: https://github.com/pgvector/pgvector
-->
