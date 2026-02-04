---
title: "Cost and Scaling Considerations"
---

# Cost and Scaling Considerations

## Introduction

Vector database costs can surprise you. Unlike traditional databases where storage dominates, vector databases are compute and memory-intensive. Understanding the cost drivers helps you optimize spending and plan for scale.

### What We'll Cover

- Pricing model breakdown
- Cost optimization strategies
- Scaling patterns
- Capacity planning
- Total cost of ownership

### Prerequisites

- Understanding of vector database options
- Basic cloud cost concepts
- Your project's scale requirements

---

## Pricing Model Breakdown

### Managed Service Components

| Component | What It Covers | Typical Pricing |
|-----------|---------------|-----------------|
| **Storage** | Vector data on disk | $0.05-0.25/GB/month |
| **Compute** | Query processing | $0.05-0.50/hour |
| **Memory** | Index in RAM | Often bundled with compute |
| **Requests** | API calls | $0-2/million queries |
| **Egress** | Data transfer out | $0.05-0.15/GB |

### Provider Pricing Comparison

| Provider | Free Tier | Starter | Production |
|----------|-----------|---------|------------|
| **Pinecone** | 2GB storage, 1M reads | $70+/month | Usage-based |
| **Qdrant Cloud** | 1GB, 1M vectors | $25/month | $0.07/1K vectors |
| **Weaviate Cloud** | 14 days | $25/month | Custom |
| **Zilliz Cloud** | 1 CU free | $65/month | Usage-based |
| **Supabase** | 500MB | $25/month | $0.125/GB |
| **Neon** | 0.5GB | $19/month | Usage-based |

### Self-Hosted Costs

```
Monthly Cost = Compute + Storage + Operations

Compute = (vCPUs × $0.04/hr) × 730 hours
Storage = GB × $0.10/GB/month
Operations = Engineer hours × hourly rate
```

**Example: 1M vectors (1536 dims) self-hosted:**
```
Compute:  8 vCPU, 32GB RAM = ~$200/month
Storage:  ~6GB = ~$0.60/month  
Ops:      2 hours/month × $100 = $200/month
Total:    ~$400/month (vs $70-100 managed)
```

> **Note:** Self-hosted becomes cost-effective at scale (10M+ vectors) when operational overhead is amortized.

---

## Cost Drivers Deep Dive

### 1. Vector Dimensions

Higher dimensions = more storage and compute:

| Dimensions | Memory per 1M vectors | Storage |
|------------|----------------------|---------|
| 384 | ~1.5 GB | ~1.5 GB |
| 768 | ~3 GB | ~3 GB |
| 1536 | ~6 GB | ~6 GB |
| 3072 | ~12 GB | ~12 GB |

**Optimization:**
```python
# Use smaller embedding models when possible
# text-embedding-3-small (1536) vs text-embedding-3-large (3072)

# Or reduce dimensions with Matryoshka embeddings
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="Your text",
    dimensions=512  # Reduced from 1536
)
```

### 2. Index Type

| Index | Memory Usage | Query Speed | Build Time |
|-------|--------------|-------------|------------|
| FLAT | Lowest (vectors only) | Slowest | None |
| IVFFlat | Medium | Medium | Fast |
| HNSW | Highest (graph overhead) | Fastest | Slow |

**HNSW Memory Formula:**
```
Memory = vectors × dims × 4 bytes × (1 + M/16)

Example: 1M vectors × 1536 dims × 4 × 1.5 ≈ 9.2 GB
```

### 3. Replication and Sharding

| Configuration | Cost Multiplier |
|---------------|-----------------|
| Single replica | 1x |
| 2 replicas (HA) | 2x |
| 3 replicas | 3x |
| 2 shards × 2 replicas | 4x |

### 4. Query Patterns

| Pattern | Cost Impact |
|---------|-------------|
| Simple KNN | Low |
| Filtered search | Medium (depends on filter selectivity) |
| Hybrid search | Higher (vector + text) |
| Large `top_k` (100+) | Higher |
| High QPS (1000+) | May need more replicas |

---

## Cost Optimization Strategies

### 1. Choose the Right Embedding Size

```python
# Evaluate quality vs cost trade-off
embedding_options = [
    {"model": "text-embedding-3-small", "dims": 512, "cost": "$"},
    {"model": "text-embedding-3-small", "dims": 1536, "cost": "$$"},
    {"model": "text-embedding-3-large", "dims": 3072, "cost": "$$$"},
]

# Test recall quality with your actual queries before deciding
def benchmark_embedding_quality(model, dims, test_queries, ground_truth):
    # ... measure recall@10 for each configuration
    pass
```

### 2. Use Quantization

Quantization reduces memory and costs by 2-32x:

```python
# Scalar quantization (4x compression, ~2% recall loss)
# Binary quantization (32x compression, ~5% recall loss)

# Pinecone - enable at index creation
pinecone.create_index(
    name="quantized",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # Automatically uses scalar quantization in serverless
)

# Qdrant - enable quantization
client.update_collection(
    collection_name="docs",
    optimizers_config=OptimizersConfigDiff(
        quantization=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8")
        )
    )
)
```

### 3. Implement Caching

```python
from functools import lru_cache
import hashlib

class CachedVectorSearch:
    def __init__(self, vector_db):
        self.db = vector_db
        self.cache = {}
        
    def search(self, query: str, top_k: int = 5) -> list:
        # Cache key from query hash
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding and search
        embedding = get_embedding(query)
        results = self.db.search(embedding, top_k)
        
        # Cache for repeated queries
        self.cache[cache_key] = results
        return results
```

### 4. Use Tiered Storage

```python
# Hot tier: frequently accessed, recent documents
# Cold tier: archive, rarely accessed

class TieredVectorStore:
    def __init__(self, hot_store, cold_store):
        self.hot = hot_store  # Fast, expensive (Pinecone)
        self.cold = cold_store  # Slow, cheap (S3 + LanceDB)
        
    def search(self, query_vector, top_k: int = 5):
        # Search hot tier first
        hot_results = self.hot.search(query_vector, top_k)
        
        if len(hot_results) >= top_k:
            return hot_results
        
        # Fall back to cold tier for remaining
        cold_results = self.cold.search(query_vector, top_k - len(hot_results))
        return hot_results + cold_results
    
    def archive(self, older_than_days: int = 30):
        """Move old documents to cold storage"""
        old_docs = self.hot.query_by_date(older_than_days)
        self.cold.upsert(old_docs)
        self.hot.delete([d.id for d in old_docs])
```

### 5. Optimize Query Patterns

```python
# ❌ Bad: Over-fetching
results = db.search(query, top_k=1000)
filtered = [r for r in results if r.category == "AI"][:10]

# ✅ Good: Pre-filter in database
results = db.search(
    query, 
    top_k=10,
    filter={"category": "AI"}
)

# ❌ Bad: Searching without index optimization
db.search(query, top_k=10)  # Uses default ef_search

# ✅ Good: Tune based on recall needs
db.search(query, top_k=10, params={"ef": 32})  # Lower ef for speed
```

---

## Scaling Patterns

### Horizontal Scaling (Sharding)

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Shard 1 │      │ Shard 2 │      │ Shard 3 │
    │ 0-10M   │      │ 10-20M  │      │ 20-30M  │
    └─────────┘      └─────────┘      └─────────┘
```

**When to shard:**
- Single node memory limit reached
- Query latency degrading
- Write throughput insufficient

### Vertical Scaling

| Vector Count | Recommended Instance |
|--------------|---------------------|
| < 500K | 4 vCPU, 16GB RAM |
| 500K - 2M | 8 vCPU, 32GB RAM |
| 2M - 10M | 16 vCPU, 64GB RAM |
| 10M - 50M | 32 vCPU, 128GB RAM |
| 50M+ | Shard across nodes |

### Read Replicas

```python
# Add read replicas for query throughput
# Write to primary, read from replicas

class ReplicatedVectorStore:
    def __init__(self, primary, replicas: list):
        self.primary = primary
        self.replicas = replicas
        self.replica_index = 0
        
    def upsert(self, vectors):
        # All writes go to primary
        self.primary.upsert(vectors)
        
    def search(self, query_vector, top_k: int):
        # Round-robin across replicas
        replica = self.replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replicas)
        return replica.search(query_vector, top_k)
```

---

## Capacity Planning

### Estimation Worksheet

```python
def estimate_monthly_cost(
    vector_count: int,
    dimensions: int,
    queries_per_day: int,
    provider: str
) -> dict:
    
    # Storage calculation
    storage_gb = (vector_count * dimensions * 4) / (1024**3)
    
    # Memory (with HNSW overhead)
    memory_gb = storage_gb * 1.5
    
    # Provider-specific costs
    if provider == "pinecone":
        # Serverless: $0.25/1M reads, $2/GB storage
        storage_cost = storage_gb * 2
        query_cost = (queries_per_day * 30 / 1_000_000) * 0.25
        
    elif provider == "qdrant_cloud":
        # $0.07 per 1K vectors/month for storage
        storage_cost = (vector_count / 1000) * 0.07
        query_cost = 0  # Included
        
    elif provider == "supabase":
        # $0.125/GB storage
        storage_cost = max(storage_gb, 8) * 0.125  # Min 8GB on paid
        query_cost = 0  # Included
        
    return {
        "storage_gb": storage_gb,
        "memory_required_gb": memory_gb,
        "estimated_monthly_cost": storage_cost + query_cost
    }

# Example
print(estimate_monthly_cost(
    vector_count=1_000_000,
    dimensions=1536,
    queries_per_day=10_000,
    provider="pinecone"
))
```

### Growth Planning

| Growth Rate | Planning Horizon | Action |
|-------------|-----------------|--------|
| < 10% monthly | 12 months | Annual capacity review |
| 10-50% monthly | 6 months | Quarterly scaling plan |
| > 50% monthly | 3 months | Monthly capacity checks |

```python
def project_growth(
    current_vectors: int,
    monthly_growth_rate: float,
    months: int
) -> list[dict]:
    projections = []
    vectors = current_vectors
    
    for month in range(months):
        vectors = int(vectors * (1 + monthly_growth_rate))
        cost = estimate_monthly_cost(vectors, 1536, 10000, "pinecone")
        projections.append({
            "month": month + 1,
            "vectors": vectors,
            "cost": cost["estimated_monthly_cost"]
        })
    
    return projections

# 20% monthly growth for 12 months
projections = project_growth(100_000, 0.20, 12)
for p in projections:
    print(f"Month {p['month']}: {p['vectors']:,} vectors, ${p['cost']:.2f}/month")
```

---

## Total Cost of Ownership

### Managed Service TCO

```
Monthly TCO = Service Fees + Embedding Costs + Engineering Time

Service Fees:   $100/month (Pinecone/Qdrant)
Embedding:      1M documents × $0.02/1K tokens × avg 500 tokens = $10/month
Engineering:    2 hours/month × $150/hour = $300/month
────────────────────────────────────────────────────────────────
Total:          $410/month
```

### Self-Hosted TCO

```
Monthly TCO = Infrastructure + Monitoring + Maintenance + Engineering

Infrastructure: $200/month (compute + storage)
Monitoring:     $50/month (Datadog, etc.)
Maintenance:    4 hours/month × $150/hour = $600/month
Engineering:    8 hours/month × $150/hour = $1,200/month (setup amortized)
────────────────────────────────────────────────────────────────
Total:          $2,050/month (first year)
                $850/month (after setup complete)
```

### Break-Even Analysis

```python
def breakeven_analysis(managed_monthly: float, self_hosted_monthly: float, 
                       self_hosted_setup: float) -> int:
    """Calculate months until self-hosted becomes cheaper"""
    if managed_monthly <= self_hosted_monthly:
        return -1  # Never breaks even
    
    savings_per_month = managed_monthly - self_hosted_monthly
    months = self_hosted_setup / savings_per_month
    return int(months) + 1

# Example
months = breakeven_analysis(
    managed_monthly=500,
    self_hosted_monthly=200,
    self_hosted_setup=5000
)
print(f"Self-hosted breaks even after {months} months")
```

---

## Hands-on Exercise

### Your Task

Create a cost projection for a growing RAG application:

### Scenario

- Starting with 100,000 documents
- Each document averages 500 tokens
- Using text-embedding-3-small (1536 dimensions)
- Expected 20% monthly growth
- 5,000 queries per day
- Budget: $500/month initial, scaling with usage

### Requirements

1. Calculate initial embedding costs
2. Project 12-month vector count
3. Compare 3 provider options
4. Identify when you'll exceed budget
5. Recommend cost optimization strategies

<details>
<summary>✅ Solution</summary>

```python
# Initial embedding costs
initial_docs = 100_000
tokens_per_doc = 500
embedding_cost_per_1k = 0.00002  # $0.02 per 1M tokens

initial_embedding_cost = (initial_docs * tokens_per_doc / 1000) * embedding_cost_per_1k
print(f"Initial embedding cost: ${initial_embedding_cost:.2f}")
# ~$1.00 for initial embeddings

# 12-month projection
monthly_growth = 0.20
vectors = initial_docs
projections = []

for month in range(12):
    vectors = int(vectors * (1 + monthly_growth))
    
    # Calculate for each provider
    costs = {
        "pinecone": (vectors / 1_000_000) * 70 + (5000 * 30 / 1_000_000) * 0.25,
        "qdrant_cloud": (vectors / 1000) * 0.07,
        "supabase": max((vectors * 1536 * 4) / (1024**3), 8) * 0.125 + 25
    }
    
    projections.append({
        "month": month + 1,
        "vectors": vectors,
        **costs
    })

# Print projections
for p in projections:
    print(f"Month {p['month']}: {p['vectors']:,} vectors")
    print(f"  Pinecone: ${p['pinecone']:.2f}")
    print(f"  Qdrant: ${p['qdrant_cloud']:.2f}")
    print(f"  Supabase: ${p['supabase']:.2f}")
    print()

# Budget exceedance
budget = 500
for p in projections:
    if min(p['pinecone'], p['qdrant_cloud'], p['supabase']) > budget:
        print(f"⚠️ Exceed budget in month {p['month']}")
        break

# Recommendations
print("""
Cost Optimization Recommendations:
1. Use text-embedding-3-small with 512 dimensions (1/3 storage)
2. Enable scalar quantization (4x compression)
3. Implement semantic caching for repeated queries
4. Archive old documents to LanceDB/S3 after 90 days
5. Consider Qdrant Cloud - best cost/performance ratio
""")
```

**Results:**
- Month 1: ~$7-30 depending on provider
- Month 6: ~$30-100
- Month 12: ~$60-250
- Budget exceeded: Never within 12 months for this scenario

**Recommendation:** Qdrant Cloud offers the best value with strong free tier transitioning to reasonable pay-as-you-go pricing.

</details>

---

## Summary

✅ Vector dimensions are the biggest cost driver—use smallest that maintains quality

✅ Quantization reduces costs 2-32x with minimal recall impact

✅ Managed services cost more monthly but save engineering time

✅ Self-hosted breaks even at large scale with dedicated DevOps

✅ Implement caching and tiered storage for cost-effective scaling

**Next:** [Migration Strategies](./11-migration-strategies.md)

---

## Further Reading

- [Pinecone Pricing Calculator](https://www.pinecone.io/pricing/)
- [Qdrant Pricing](https://qdrant.tech/pricing/)
- [Vector Database Cost Comparison (2024)](https://benchmark.vectorview.ai/cost)

---

<!-- 
Sources Consulted:
- Pinecone pricing: https://www.pinecone.io/pricing/
- Qdrant pricing: https://qdrant.tech/pricing/
- Supabase pricing: https://supabase.com/pricing
- OpenAI embedding pricing: https://openai.com/pricing
-->
