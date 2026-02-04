---
title: "Scaling Patterns"
---

# Scaling Patterns

## Introduction

As your embedding system grows, you'll face challenges: more vectors to store, more queries to handle, lower latency requirements, and higher availability demands. Scaling strategies depend on your constraints—whether you're bottlenecked on storage, query throughput, or both.

This lesson covers horizontal scaling patterns for vector databases, including sharding, replication, read replicas, and global distribution strategies.

### What We'll Cover

- Scaling dimensions: storage vs throughput
- Sharding strategies
- Replication for availability
- Read replicas for throughput
- Qdrant distributed deployment
- Global distribution patterns

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Familiarity with distributed systems concepts
- Basic knowledge of Qdrant or similar vector databases

---

## Scaling Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│              Scaling Dimensions                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│          ▲ Query Throughput                                     │
│          │                                                      │
│          │    ┌─────────────────┐                               │
│          │    │   SCALE OUT     │                               │
│          │    │   (Replicas)    │                               │
│          │    │                 │                               │
│          │    │  • Read replicas│                               │
│          │    │  • Load balance │                               │
│          │    │  • Caching      │                               │
│          │    └─────────────────┘                               │
│          │                         ┌─────────────────┐          │
│          │                         │   SCALE BOTH    │          │
│          │                         │   (Shards +     │          │
│          │                         │    Replicas)    │          │
│          │                         └─────────────────┘          │
│    ┌─────────────────┐                                          │
│    │  SINGLE NODE    │    ┌─────────────────┐                   │
│    │  (Baseline)     │    │   SCALE UP      │                   │
│    │                 │    │   (Sharding)    │                   │
│    │  • Development  │    │                 │                   │
│    │  • < 1M vectors │    │  • More shards  │                   │
│    └─────────────────┘    │  • Larger nodes │                   │
│          │                └─────────────────┘                   │
│          └──────────────────────────────────────────▶           │
│                           Storage Capacity                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### When to Scale

| Symptom | Scale Strategy | Action |
|---------|----------------|--------|
| Running out of disk space | Sharding | Add more shards |
| High query latency | Replicas | Add read replicas |
| High write latency | Sharding | Distribute writes |
| Single point of failure | Replication | Add replicas |
| Geographic latency | Distribution | Multi-region deployment |

---

## Sharding Strategies

### What is Sharding?

Sharding splits your vector data across multiple nodes. Each node holds a subset of vectors, and searches query all shards in parallel.

### Automatic Sharding (Qdrant)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create collection with automatic sharding
# shard_number determines parallelism
client.create_collection(
    collection_name="embeddings",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    # Shard configuration
    shard_number=6,           # Number of shards (set at creation)
    replication_factor=2,     # Replicas per shard
    write_consistency_factor=1  # Quorum for writes
)
```

**Shard Number Guidelines:**

| Vector Count | Recommended Shards | Reasoning |
|--------------|-------------------|-----------|
| < 100K | 1-2 | Low overhead |
| 100K - 1M | 2-4 | Balanced |
| 1M - 10M | 4-8 | Good parallelism |
| 10M - 100M | 8-16 | Distributed load |
| > 100M | 16+ | Maximum distribution |

> **Note:** `shard_number` is immutable after collection creation. Plan for growth.

### User-Defined Sharding

For multi-tenant systems, route tenants to specific shards:

```python
from qdrant_client.models import ShardingMethod

# Create collection with custom sharding
client.create_collection(
    collection_name="multi_tenant_embeddings",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    sharding_method=ShardingMethod.CUSTOM,
    shard_number=10
)

# Insert with shard key (tenant isolation)
client.upsert(
    collection_name="multi_tenant_embeddings",
    points=[
        {
            "id": "doc_123",
            "vector": embedding,
            "payload": {"tenant": "acme_corp"},
            "shard_key": "acme_corp"  # Routes to tenant's shard
        }
    ]
)

# Search within tenant's shard only
results = client.search(
    collection_name="multi_tenant_embeddings",
    query_vector=query_embedding,
    limit=10,
    shard_key="acme_corp"  # Only searches this tenant's shard
)
```

### Shard Transfer Methods

When adding nodes to a cluster, shards must be transferred:

| Method | Description | Use Case |
|--------|-------------|----------|
| `stream_records` | Transfer records one by one | Default, safest |
| `snapshot` | Full shard snapshot | Faster for large shards |
| `wal_delta` | Transfer write-ahead log | Minimal downtime |

```python
# Configure shard transfer during cluster operations
# (Usually handled automatically by Qdrant)

# For manual shard movement:
client.update_collection(
    collection_name="embeddings",
    optimizer_config={
        "shard_transfer_method": "snapshot"
    }
)
```

---

## Replication for Availability

### Replication Factor

```python
# Create collection with replication
client.create_collection(
    collection_name="embeddings",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    shard_number=4,
    replication_factor=3  # 3 copies of each shard
)
```

**Replication Guidelines:**

| replication_factor | Failure Tolerance | Cost | Use Case |
|-------------------|-------------------|------|----------|
| 1 | None | 1x | Development |
| 2 | 1 node failure | 2x | Production (basic) |
| 3 | 2 node failures | 3x | Production (recommended) |

### Consistency Levels

Control the trade-off between consistency and availability:

```python
from qdrant_client.models import WriteOrdering, ReadConsistency

# Write with strong consistency
client.upsert(
    collection_name="embeddings",
    points=[...],
    ordering=WriteOrdering.STRONG  # Wait for all replicas
)

# Search with configurable consistency
results = client.search(
    collection_name="embeddings",
    query_vector=query_embedding,
    limit=10,
    consistency=ReadConsistency.MAJORITY  # Read from majority of replicas
)
```

**Consistency Options:**

| Level | Read Behavior | Write Behavior |
|-------|---------------|----------------|
| `WEAK` | Any replica | Return immediately |
| `MEDIUM` | Acknowledge replicas | Wait for some |
| `STRONG` | All replicas agree | Wait for all |
| `MAJORITY` | Majority agrees | Wait for majority |

### Write Consistency Factor

```python
# Control write quorum
client.update_collection(
    collection_name="embeddings",
    write_consistency_factor=2  # Writes must succeed on 2 replicas
)
```

---

## Horizontal Scaling for Query Throughput

### Read Replica Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│              Read Replica Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │   Writes    │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│                             ▼                                   │
│                      ┌─────────────┐                            │
│                      │   Primary   │                            │
│                      │   Cluster   │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              │              │              │                    │
│              ▼              ▼              ▼                    │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│       │ Replica  │   │ Replica  │   │ Replica  │              │
│       │    1     │   │    2     │   │    3     │              │
│       └────┬─────┘   └────┬─────┘   └────┬─────┘              │
│            │              │              │                      │
│            └──────────────┼──────────────┘                      │
│                           │                                     │
│                    ┌──────┴──────┐                              │
│                    │ Load        │                              │
│                    │ Balancer    │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                    ┌──────┴──────┐                              │
│                    │   Reads     │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation with Connection Pool

```python
from typing import List
import random

class ReadReplicaPool:
    """
    Connection pool with read/write splitting.
    """
    def __init__(
        self,
        primary_url: str,
        replica_urls: List[str]
    ):
        self.primary = QdrantClient(url=primary_url)
        self.replicas = [QdrantClient(url=url) for url in replica_urls]
    
    def get_write_client(self) -> QdrantClient:
        """Get client for write operations."""
        return self.primary
    
    def get_read_client(self) -> QdrantClient:
        """Get client for read operations (load balanced)."""
        if not self.replicas:
            return self.primary
        return random.choice(self.replicas)
    
    def search(self, collection: str, query_vector: list, top_k: int = 10):
        """Execute search on replica."""
        client = self.get_read_client()
        return client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )
    
    def upsert(self, collection: str, points: list):
        """Execute upsert on primary."""
        client = self.get_write_client()
        return client.upsert(
            collection_name=collection,
            points=points
        )


class SmartLoadBalancer:
    """
    Load balancer with health checking and latency-based routing.
    """
    def __init__(self, clients: List[QdrantClient]):
        self.clients = clients
        self.health_status = {i: True for i in range(len(clients))}
        self.latencies = {i: 0.0 for i in range(len(clients))}
    
    def get_client(self) -> QdrantClient:
        """Get healthiest, lowest-latency client."""
        healthy = [
            (i, self.latencies[i])
            for i in range(len(self.clients))
            if self.health_status[i]
        ]
        
        if not healthy:
            # Fallback to any client if all unhealthy
            return self.clients[0]
        
        # Sort by latency, pick lowest
        healthy.sort(key=lambda x: x[1])
        return self.clients[healthy[0][0]]
    
    def record_latency(self, client_index: int, latency_ms: float):
        """Update latency tracking."""
        # Exponential moving average
        alpha = 0.3
        self.latencies[client_index] = (
            alpha * latency_ms + 
            (1 - alpha) * self.latencies[client_index]
        )
    
    def mark_unhealthy(self, client_index: int):
        """Mark client as unhealthy."""
        self.health_status[client_index] = False
    
    def mark_healthy(self, client_index: int):
        """Mark client as healthy."""
        self.health_status[client_index] = True
```

---

## Qdrant Distributed Deployment

### Cluster Configuration

```yaml
# docker-compose.yml for Qdrant cluster
version: '3.8'

services:
  qdrant-node-1:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
    ports:
      - "6333:6333"  # REST API
      - "6335:6335"  # P2P communication
    volumes:
      - qdrant_data_1:/qdrant/storage

  qdrant-node-2:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__BOOTSTRAP=http://qdrant-node-1:6335
    ports:
      - "6334:6333"
      - "6336:6335"
    volumes:
      - qdrant_data_2:/qdrant/storage

  qdrant-node-3:
    image: qdrant/qdrant:latest
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__BOOTSTRAP=http://qdrant-node-1:6335
    ports:
      - "6337:6333"
      - "6338:6335"
    volumes:
      - qdrant_data_3:/qdrant/storage

volumes:
  qdrant_data_1:
  qdrant_data_2:
  qdrant_data_3:
```

### Node Count Recommendations

| Nodes | Configuration | Use Case |
|-------|---------------|----------|
| 1 | No clustering | Development only |
| 2 | Balanced cluster | Can tolerate 1 failure with replication_factor=2 |
| 3+ | Production cluster | Full fault tolerance, Raft consensus |

> **Important:** With 2 nodes, if one fails during cluster operations, the remaining node can't reach consensus. Use 3+ nodes for production.

### Cluster Health Monitoring

```python
class ClusterHealthChecker:
    """
    Monitor Qdrant cluster health.
    """
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def get_cluster_info(self) -> dict:
        """Get cluster topology and status."""
        info = self.client.get_cluster_info()
        
        return {
            "peer_id": info.peer_id,
            "peers": {
                peer_id: {
                    "uri": peer.uri,
                    "status": "active"  # Simplified
                }
                for peer_id, peer in info.peers.items()
            },
            "raft_info": {
                "term": info.raft_info.term,
                "commit": info.raft_info.commit,
                "leader": info.raft_info.leader
            }
        }
    
    def check_collection_health(self, collection: str) -> dict:
        """Check collection distribution across cluster."""
        info = self.client.get_collection(collection)
        
        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "points_count": shard.points_count,
                    "state": shard.state
                }
                for shard in info.shards_info
            ]
        }
```

---

## Global Distribution

### Multi-Region Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Global Distribution Pattern                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   US-EAST                    EU-WEST                            │
│   ┌─────────────────┐        ┌─────────────────┐               │
│   │  Primary        │◀──────▶│  Replica        │               │
│   │  Cluster        │  sync  │  Cluster        │               │
│   │                 │        │                 │               │
│   │  • Writes       │        │  • Reads        │               │
│   │  • Reads        │        │  • Writes (opt) │               │
│   └────────┬────────┘        └────────┬────────┘               │
│            │                          │                         │
│            ▼                          ▼                         │
│   ┌─────────────────┐        ┌─────────────────┐               │
│   │   US Users      │        │   EU Users      │               │
│   │   ~20ms latency │        │   ~20ms latency │               │
│   └─────────────────┘        └─────────────────┘               │
│                                                                 │
│   ASIA-PACIFIC                                                  │
│   ┌─────────────────┐                                          │
│   │  Replica        │                                          │
│   │  Cluster        │                                          │
│   │                 │                                          │
│   │  • Reads only   │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │   APAC Users    │                                          │
│   │   ~30ms latency │                                          │
│   └─────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Region-Aware Routing

```python
from typing import Dict
import geoip2.database

class GeoRoutedSearchService:
    """
    Route queries to nearest region.
    """
    def __init__(
        self,
        regional_clients: Dict[str, QdrantClient],
        geoip_db_path: str
    ):
        self.clients = regional_clients  # {"us-east": client, "eu-west": client}
        self.geoip = geoip2.database.Reader(geoip_db_path)
        
        # Map countries to regions
        self.country_to_region = {
            "US": "us-east",
            "CA": "us-east",
            "GB": "eu-west",
            "DE": "eu-west",
            "FR": "eu-west",
            "JP": "asia-pacific",
            "AU": "asia-pacific",
            # ... more mappings
        }
        
        self.default_region = "us-east"
    
    def get_region(self, client_ip: str) -> str:
        """Determine region from client IP."""
        try:
            response = self.geoip.country(client_ip)
            country = response.country.iso_code
            return self.country_to_region.get(country, self.default_region)
        except Exception:
            return self.default_region
    
    def search(
        self,
        query_vector: list,
        client_ip: str,
        collection: str,
        top_k: int = 10
    ):
        """Search in nearest region."""
        region = self.get_region(client_ip)
        client = self.clients[region]
        
        return client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )
```

---

## Scaling the Embedding Service

### Horizontal Scaling with Kubernetes

```yaml
# embedding-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: embedding-service:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: request_latency_p99
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
```

---

## Summary

✅ **Choose sharding for storage scaling, replication for throughput/availability**  
✅ **Set shard_number based on expected vector count (can't change later)**  
✅ **Use replication_factor ≥ 2 for production (3 recommended)**  
✅ **Deploy 3+ nodes for proper Raft consensus in Qdrant**  
✅ **Consider multi-region deployment for global latency requirements**

---

**Next:** [Failure Handling →](./06-failure-handling.md)

---

<!-- 
Sources Consulted:
- Qdrant Distributed Deployment: https://qdrant.tech/documentation/guides/distributed_deployment/
- Qdrant Sharding: https://qdrant.tech/documentation/guides/distributed_deployment/#sharding
- Pinecone Scaling: https://docs.pinecone.io/guides/indexes/understanding-indexes
-->
