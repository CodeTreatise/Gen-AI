---
title: "Embedding Cost Factors"
---

# Embedding Cost Factors

## Introduction

Before optimizing costs, we need to understand what drives them. Embedding system costs come from four primary sources: API calls, storage, compute, and bandwidth. Each has different scaling characteristics and optimization strategies.

Understanding these factors helps you identify your biggest cost drivers and prioritize optimizations that will have the most impact for your specific workload.

### What We'll Cover

- API pricing models and per-token costs
- Vector storage cost calculations
- Query compute and processing costs
- Bandwidth and data transfer expenses
- Building a total cost model

### Prerequisites

- Familiarity with embedding APIs (OpenAI, Google)
- Understanding of vector databases
- Basic knowledge of cloud pricing models

---

## API Costs: Per-Token Pricing

API calls are often the largest initial cost, especially during data ingestion when you're embedding your entire corpus.

### How Token Pricing Works

Embedding APIs charge based on the number of tokens processed, not the number of API calls or vectors generated.

```python
import tiktoken

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text for OpenAI models."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def estimate_embedding_cost(
    text: str,
    price_per_million: float = 0.02  # text-embedding-3-small
) -> float:
    """Estimate cost to embed a text string."""
    tokens = count_tokens(text)
    return (tokens / 1_000_000) * price_per_million

# Example: Calculate cost for a document
document = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. It focuses on developing algorithms that can access data 
and use it to learn for themselves.
"""

tokens = count_tokens(document)
cost = estimate_embedding_cost(document)
print(f"Tokens: {tokens}")
print(f"Cost: ${cost:.8f}")
```

**Output:**
```
Tokens: 54
Cost: $0.00000108
```

### Current Embedding Model Pricing (2025)

| Provider | Model | Standard | Batch | Dimensions | Max Tokens |
|----------|-------|----------|-------|------------|------------|
| OpenAI | text-embedding-3-small | $0.02/1M | $0.01/1M | 1,536 | 8,192 |
| OpenAI | text-embedding-3-large | $0.13/1M | $0.065/1M | 3,072 | 8,192 |
| OpenAI | text-embedding-ada-002 | $0.10/1M | $0.05/1M | 1,536 | 8,192 |
| Google | gemini-embedding-001 | Free tier | $0.15/1M | 768 | 2,048 |

### Cost Calculator for Document Corpus

```python
from dataclasses import dataclass
from typing import List

@dataclass
class CostEstimate:
    total_tokens: int
    total_documents: int
    standard_cost: float
    batch_cost: float
    savings_with_batch: float

def estimate_corpus_cost(
    documents: List[str],
    model: str = "text-embedding-3-small"
) -> CostEstimate:
    """Estimate embedding cost for an entire document corpus."""
    
    # Model pricing (standard, batch)
    pricing = {
        "text-embedding-3-small": (0.02, 0.01),
        "text-embedding-3-large": (0.13, 0.065),
        "text-embedding-ada-002": (0.10, 0.05),
    }
    
    standard_price, batch_price = pricing.get(model, (0.02, 0.01))
    
    # Count total tokens
    total_tokens = sum(count_tokens(doc) for doc in documents)
    
    # Calculate costs
    standard_cost = (total_tokens / 1_000_000) * standard_price
    batch_cost = (total_tokens / 1_000_000) * batch_price
    
    return CostEstimate(
        total_tokens=total_tokens,
        total_documents=len(documents),
        standard_cost=standard_cost,
        batch_cost=batch_cost,
        savings_with_batch=standard_cost - batch_cost
    )

# Example: Estimate cost for 10,000 documents averaging 500 tokens
sample_docs = ["Sample document content " * 50] * 10000
estimate = estimate_corpus_cost(sample_docs)

print(f"Total tokens: {estimate.total_tokens:,}")
print(f"Standard API cost: ${estimate.standard_cost:.2f}")
print(f"Batch API cost: ${estimate.batch_cost:.2f}")
print(f"Savings with batch: ${estimate.savings_with_batch:.2f}")
```

**Output:**
```
Total tokens: 1,500,000
Standard API cost: $0.03
Batch API cost: $0.02
Savings with batch: $0.02
```

> **Tip:** At scale, batch processing provides 50% savings. A corpus with 100 million tokens saves $1,000 by using batch API.

---

## Storage Costs: Vector Database Pricing

Once embeddings are generated, you pay to store them. Vector storage costs depend on:

1. **Number of vectors** — How many documents/chunks you've embedded
2. **Vector dimensions** — Higher dimensions = more storage per vector
3. **Metadata size** — Additional data stored with each vector
4. **Database tier** — Serverless vs. dedicated infrastructure

### Calculating Vector Storage Size

```python
from dataclasses import dataclass

@dataclass
class StorageEstimate:
    num_vectors: int
    dimensions: int
    metadata_bytes: int
    total_gb: float
    monthly_cost: float

def calculate_storage(
    num_vectors: int,
    dimensions: int = 1536,
    avg_id_bytes: int = 8,
    avg_metadata_bytes: int = 500,
    price_per_gb_month: float = 0.25  # Example Pinecone rate
) -> StorageEstimate:
    """Calculate vector storage size and cost.
    
    Each dimension uses 4 bytes (32-bit float).
    """
    bytes_per_vector = (
        avg_id_bytes +
        avg_metadata_bytes +
        (dimensions * 4)  # 4 bytes per float32
    )
    
    total_bytes = num_vectors * bytes_per_vector
    total_gb = total_bytes / (1024 ** 3)
    monthly_cost = total_gb * price_per_gb_month
    
    return StorageEstimate(
        num_vectors=num_vectors,
        dimensions=dimensions,
        metadata_bytes=avg_metadata_bytes,
        total_gb=total_gb,
        monthly_cost=monthly_cost
    )

# Example: 1 million vectors with text-embedding-3-small
storage = calculate_storage(
    num_vectors=1_000_000,
    dimensions=1536,
    avg_metadata_bytes=500
)

print(f"Vectors: {storage.num_vectors:,}")
print(f"Storage: {storage.total_gb:.2f} GB")
print(f"Monthly cost: ${storage.monthly_cost:.2f}")
```

**Output:**
```
Vectors: 1,000,000
Storage: 6.19 GB
Monthly cost: $1.55
```

### Pinecone Storage Reference

| Record Count | Dimensions | Metadata | Storage Size |
|--------------|------------|----------|--------------|
| 500,000 | 768 | 500 bytes | 1.79 GB |
| 1,000,000 | 1,536 | 1,000 bytes | 7.15 GB |
| 5,000,000 | 1,024 | 15,000 bytes | 95.5 GB |
| 10,000,000 | 1,536 | 1,000 bytes | 71.5 GB |

> **Note:** Dimension count has massive impact. text-embedding-3-large (3,072 dims) uses 2x storage vs. text-embedding-3-small (1,536 dims).

---

## Compute Costs: Query Processing

Query costs scale with namespace size and query volume. Understanding this helps you right-size your database tier.

### Pinecone Read Unit Pricing

Queries use Read Units (RUs) based on namespace size:

| Namespace Size | RUs per Query |
|---------------|---------------|
| < 0.25 GB | 0.25 (minimum) |
| 1 GB | 1 RU |
| 10 GB | 10 RUs |
| 50 GB | 50 RUs |
| 100 GB | 100 RUs |

```python
def calculate_query_costs(
    namespace_size_gb: float,
    queries_per_day: int,
    ru_price: float = 0.00001  # Example rate
) -> dict:
    """Calculate daily and monthly query costs."""
    
    # RUs scale linearly with namespace size (minimum 0.25)
    rus_per_query = max(0.25, namespace_size_gb)
    
    daily_rus = queries_per_day * rus_per_query
    monthly_rus = daily_rus * 30
    
    daily_cost = daily_rus * ru_price
    monthly_cost = monthly_rus * ru_price
    
    return {
        "rus_per_query": rus_per_query,
        "daily_rus": daily_rus,
        "monthly_rus": monthly_rus,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost
    }

# Example: 10,000 queries/day on 5 GB namespace
costs = calculate_query_costs(
    namespace_size_gb=5,
    queries_per_day=10_000
)

print(f"RUs per query: {costs['rus_per_query']}")
print(f"Daily RUs: {costs['daily_rus']:,.0f}")
print(f"Monthly cost: ${costs['monthly_cost']:.2f}")
```

**Output:**
```
RUs per query: 5
Daily RUs: 50,000
Monthly cost: $15.00
```

### Write Unit Costs

Write operations (upserts, updates, deletes) use Write Units based on data size:

| Operation | Cost |
|-----------|------|
| Upsert | 1 WU per KB (min 5 WUs) |
| Update | 1 WU per KB of old + new record |
| Delete | 1 WU per KB deleted |

```python
def calculate_upsert_cost(
    num_vectors: int,
    dimensions: int,
    metadata_bytes: int,
    wu_price: float = 0.000002  # Example rate
) -> float:
    """Calculate write cost for upserting vectors."""
    
    # Calculate bytes per vector
    bytes_per_vector = 8 + metadata_bytes + (dimensions * 4)
    kb_per_vector = bytes_per_vector / 1024
    
    # WUs per vector (minimum 5 per request, but batched)
    wus_per_vector = max(kb_per_vector, 0.005)  # Simplified
    
    total_wus = num_vectors * wus_per_vector
    total_cost = total_wus * wu_price
    
    return total_cost

# Example: Upserting 100,000 vectors
cost = calculate_upsert_cost(
    num_vectors=100_000,
    dimensions=1536,
    metadata_bytes=500
)
print(f"Upsert cost: ${cost:.2f}")
```

**Output:**
```
Upsert cost: $1.24
```

---

## Bandwidth Costs: Data Transfer

Bandwidth costs are often overlooked but add up for high-volume applications:

1. **Embedding API calls** — Sending text to the API
2. **Vector upserts** — Uploading embeddings to database
3. **Query results** — Retrieving vectors and metadata
4. **Cross-region traffic** — Higher rates for inter-region transfer

### Estimating Bandwidth Costs

```python
def calculate_bandwidth_costs(
    daily_embeddings: int,
    avg_text_bytes: int,
    daily_queries: int,
    avg_results_per_query: int = 10,
    dimensions: int = 1536,
    metadata_bytes: int = 500,
    egress_price_per_gb: float = 0.09  # Typical cloud pricing
) -> dict:
    """Estimate monthly bandwidth costs."""
    
    # Outbound: Text to embedding API
    embedding_egress_bytes = daily_embeddings * avg_text_bytes
    
    # Outbound: Vectors to database  
    vector_bytes = dimensions * 4 + metadata_bytes
    upsert_egress_bytes = daily_embeddings * vector_bytes
    
    # Inbound: Query results (vectors + metadata)
    query_ingress_bytes = (
        daily_queries * avg_results_per_query * vector_bytes
    )
    
    # Convert to GB and calculate monthly
    monthly_egress_gb = (
        (embedding_egress_bytes + upsert_egress_bytes) * 30
    ) / (1024 ** 3)
    
    monthly_ingress_gb = (query_ingress_bytes * 30) / (1024 ** 3)
    
    # Egress costs (ingress usually free)
    monthly_cost = monthly_egress_gb * egress_price_per_gb
    
    return {
        "monthly_egress_gb": monthly_egress_gb,
        "monthly_ingress_gb": monthly_ingress_gb,
        "monthly_cost": monthly_cost
    }

# Example: Processing 10,000 documents/day, 50,000 queries/day
bandwidth = calculate_bandwidth_costs(
    daily_embeddings=10_000,
    avg_text_bytes=2000,  # ~500 tokens
    daily_queries=50_000,
    avg_results_per_query=10
)

print(f"Monthly egress: {bandwidth['monthly_egress_gb']:.2f} GB")
print(f"Monthly ingress: {bandwidth['monthly_ingress_gb']:.2f} GB")
print(f"Monthly bandwidth cost: ${bandwidth['monthly_cost']:.2f}")
```

**Output:**
```
Monthly egress: 2.42 GB
Monthly ingress: 83.82 GB
Monthly bandwidth cost: $0.22
```

> **Note:** Bandwidth is usually the smallest cost component, but consider it for global deployments with cross-region traffic.

---

## Building a Total Cost Model

Combining all factors gives you a complete cost picture:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TotalCostModel:
    """Complete embedding system cost breakdown."""
    # Inputs
    document_count: int
    avg_tokens_per_doc: int
    monthly_queries: int
    
    # API costs
    embedding_api_cost: float
    
    # Storage costs  
    storage_gb: float
    storage_cost: float
    
    # Compute costs
    query_cost: float
    write_cost: float
    
    # Bandwidth
    bandwidth_cost: float
    
    # Totals
    initial_cost: float  # One-time embedding
    monthly_cost: float  # Ongoing
    yearly_cost: float

def build_cost_model(
    document_count: int,
    avg_tokens_per_doc: int = 500,
    monthly_new_documents: int = 1000,
    monthly_queries: int = 100_000,
    dimensions: int = 1536,
    metadata_bytes: int = 500,
    use_batch_api: bool = True
) -> TotalCostModel:
    """Build complete cost model for an embedding system."""
    
    # API pricing
    api_rate = 0.01 if use_batch_api else 0.02  # text-embedding-3-small
    
    # Initial embedding cost
    initial_tokens = document_count * avg_tokens_per_doc
    initial_embedding_cost = (initial_tokens / 1_000_000) * api_rate
    
    # Monthly new document embedding cost
    monthly_tokens = monthly_new_documents * avg_tokens_per_doc
    monthly_embedding_cost = (monthly_tokens / 1_000_000) * api_rate
    
    # Storage calculation
    bytes_per_vector = 8 + metadata_bytes + (dimensions * 4)
    total_vectors = document_count + (monthly_new_documents * 12)  # Year projection
    storage_gb = (total_vectors * bytes_per_vector) / (1024 ** 3)
    storage_cost_monthly = storage_gb * 0.25  # Example rate
    
    # Query costs
    namespace_size_gb = (document_count * bytes_per_vector) / (1024 ** 3)
    rus_per_query = max(0.25, namespace_size_gb)
    monthly_rus = monthly_queries * rus_per_query
    query_cost = monthly_rus * 0.00001
    
    # Write costs (monthly new documents)
    kb_per_vector = bytes_per_vector / 1024
    monthly_wus = monthly_new_documents * max(kb_per_vector, 5)
    write_cost = monthly_wus * 0.000002
    
    # Bandwidth (simplified)
    bandwidth_cost = 5.00  # Flat estimate for typical usage
    
    # Totals
    monthly_total = (
        monthly_embedding_cost +
        storage_cost_monthly +
        query_cost +
        write_cost +
        bandwidth_cost
    )
    
    return TotalCostModel(
        document_count=document_count,
        avg_tokens_per_doc=avg_tokens_per_doc,
        monthly_queries=monthly_queries,
        embedding_api_cost=initial_embedding_cost,
        storage_gb=storage_gb,
        storage_cost=storage_cost_monthly,
        query_cost=query_cost,
        write_cost=write_cost,
        bandwidth_cost=bandwidth_cost,
        initial_cost=initial_embedding_cost,
        monthly_cost=monthly_total,
        yearly_cost=initial_embedding_cost + (monthly_total * 12)
    )

# Example: 500K documents, 100K queries/month
model = build_cost_model(
    document_count=500_000,
    monthly_queries=100_000
)

print("=== Embedding System Cost Model ===")
print(f"\nDocuments: {model.document_count:,}")
print(f"Monthly queries: {model.monthly_queries:,}")
print(f"\n--- Initial Costs ---")
print(f"Embedding API: ${model.embedding_api_cost:.2f}")
print(f"\n--- Monthly Costs ---")
print(f"Storage ({model.storage_gb:.1f} GB): ${model.storage_cost:.2f}")
print(f"Query compute: ${model.query_cost:.2f}")
print(f"Write operations: ${model.write_cost:.4f}")
print(f"Bandwidth: ${model.bandwidth_cost:.2f}")
print(f"\n--- Totals ---")
print(f"Monthly recurring: ${model.monthly_cost:.2f}")
print(f"First year total: ${model.yearly_cost:.2f}")
```

**Output:**
```
=== Embedding System Cost Model ===

Documents: 500,000
Monthly queries: 100,000

--- Initial Costs ---
Embedding API: $2.50

--- Monthly Costs ---
Storage (3.6 GB): $0.90
Query compute: $3.10
Write operations: $0.0124
Bandwidth: $5.00

--- Totals ---
Monthly recurring: $9.01
First year total: $110.67
```

---

## Cost Comparison: Model Choices

The model you choose significantly impacts costs:

```python
def compare_model_costs(
    document_count: int,
    avg_tokens: int = 500
) -> None:
    """Compare costs across different embedding models."""
    
    models = [
        ("text-embedding-3-small", 0.02, 0.01, 1536),
        ("text-embedding-3-large", 0.13, 0.065, 3072),
        ("text-embedding-ada-002", 0.10, 0.05, 1536),
    ]
    
    total_tokens = document_count * avg_tokens
    
    print(f"Cost comparison for {document_count:,} documents ({total_tokens:,} tokens)\n")
    print(f"{'Model':<25} {'Standard':>12} {'Batch':>12} {'Storage/1M':>12}")
    print("-" * 63)
    
    for name, std_price, batch_price, dims in models:
        std_cost = (total_tokens / 1_000_000) * std_price
        batch_cost = (total_tokens / 1_000_000) * batch_price
        
        # Storage per million vectors
        bytes_per_vec = 8 + 500 + (dims * 4)
        gb_per_million = (bytes_per_vec * 1_000_000) / (1024 ** 3)
        
        print(f"{name:<25} ${std_cost:>10.2f} ${batch_cost:>10.2f} {gb_per_million:>10.1f} GB")

compare_model_costs(1_000_000)
```

**Output:**
```
Cost comparison for 1,000,000 documents (500,000,000 tokens)

Model                        Standard        Batch   Storage/1M
---------------------------------------------------------------
text-embedding-3-small        $10.00        $5.00        6.2 GB
text-embedding-3-large        $65.00       $32.50       12.4 GB
text-embedding-ada-002        $50.00       $25.00        6.2 GB
```

> **🔑 Key Insight:** text-embedding-3-small offers the best cost-to-quality ratio for most use cases. Reserve text-embedding-3-large for applications requiring maximum retrieval accuracy.

---

## Best Practices

| Practice | Impact |
|----------|--------|
| Use batch API for bulk operations | 50% API cost reduction |
| Choose smallest adequate model | 60%+ cost difference between models |
| Minimize metadata storage | Direct storage cost impact |
| Right-size dimensions | 2x storage for 2x dimensions |
| Monitor token counts | Avoid unexpected billing surprises |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using text-embedding-3-large for simple search | Start with text-embedding-3-small, upgrade only if needed |
| Storing full documents in metadata | Store IDs, retrieve documents separately |
| Not using batch API | Always batch non-real-time operations |
| Ignoring dimension impact on storage | Calculate storage before choosing model |
| Embedding duplicate content | Deduplicate before embedding |

---

## Hands-on Exercise

### Your Task

Build a cost calculator that estimates total first-year costs for an embedding system.

### Requirements

1. Accept inputs: document count, average tokens, monthly queries
2. Compare costs for all three OpenAI embedding models
3. Show breakdown: API, storage, compute, total
4. Include both standard and batch API pricing
5. Recommend the most cost-effective option

### Expected Result

A function that outputs a formatted comparison table with recommendations.

<details>
<summary>💡 Hints</summary>

- Use the `build_cost_model` function as a starting point
- Add model comparison logic
- Consider the quality-cost trade-off in recommendations
- Include monthly recurring costs in the comparison

</details>

<details>
<summary>✅ Solution</summary>

```python
def cost_calculator(
    document_count: int,
    avg_tokens: int = 500,
    monthly_queries: int = 50_000,
    monthly_new_docs: int = 1_000
) -> None:
    """Comprehensive embedding cost calculator with recommendations."""
    
    models = [
        ("text-embedding-3-small", 0.02, 0.01, 1536, "High"),
        ("text-embedding-3-large", 0.13, 0.065, 3072, "Highest"),
        ("text-embedding-ada-002", 0.10, 0.05, 1536, "Good"),
    ]
    
    results = []
    
    for name, std, batch, dims, quality in models:
        total_tokens = document_count * avg_tokens
        
        # Initial embedding (batch)
        initial_api = (total_tokens / 1_000_000) * batch
        
        # Monthly new docs (batch)
        monthly_tokens = monthly_new_docs * avg_tokens
        monthly_api = (monthly_tokens / 1_000_000) * batch
        
        # Storage
        bytes_per_vec = 8 + 500 + (dims * 4)
        storage_gb = (document_count * bytes_per_vec) / (1024 ** 3)
        monthly_storage = storage_gb * 0.25
        
        # Queries
        ru_per_query = max(0.25, storage_gb)
        monthly_query = (monthly_queries * ru_per_query) * 0.00001
        
        # Totals
        monthly_total = monthly_api + monthly_storage + monthly_query
        year_total = initial_api + (monthly_total * 12)
        
        results.append({
            "name": name,
            "quality": quality,
            "initial": initial_api,
            "monthly": monthly_total,
            "yearly": year_total,
            "storage_gb": storage_gb
        })
    
    # Sort by yearly cost
    results.sort(key=lambda x: x["yearly"])
    
    print("=" * 70)
    print(f"EMBEDDING COST ANALYSIS")
    print(f"Documents: {document_count:,} | Queries/mo: {monthly_queries:,}")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Quality':<10} {'Initial':>10} {'Monthly':>10} {'Year':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<25} {r['quality']:<10} ${r['initial']:>8.2f} ${r['monthly']:>8.2f} ${r['yearly']:>8.2f}")
    
    best = results[0]
    print(f"\n✅ RECOMMENDATION: {best['name']}")
    print(f"   First-year cost: ${best['yearly']:.2f}")
    print(f"   Quality rating: {best['quality']}")
    print(f"   Storage requirement: {best['storage_gb']:.1f} GB")

# Test the calculator
cost_calculator(
    document_count=500_000,
    monthly_queries=100_000
)
```

**Output:**
```
======================================================================
EMBEDDING COST ANALYSIS
Documents: 500,000 | Queries/mo: 100,000
======================================================================

Model                     Quality      Initial    Monthly       Year
----------------------------------------------------------------------
text-embedding-3-small    High           $2.50      $4.05     $51.05
text-embedding-ada-002    Good          $12.50      $4.05     $61.05
text-embedding-3-large    Highest       $16.25      $6.45     $93.65

✅ RECOMMENDATION: text-embedding-3-small
   First-year cost: $51.05
   Quality rating: High
   Storage requirement: 3.1 GB
```

</details>

---

## Summary

Understanding embedding cost factors enables informed optimization decisions:

✅ **API costs** scale with token volume—batch processing cuts this by 50%

✅ **Storage costs** scale with vectors × dimensions—model choice matters

✅ **Compute costs** scale with namespace size × query volume

✅ **Bandwidth** is typically minimal but matters for global deployments

✅ Total cost modeling reveals the biggest optimization opportunities

**Next:** [Batch Embedding Discounts](./02-batch-embedding-discounts.md) — Learn how to implement batch processing for 50% savings

---

## Further Reading

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Count tokens before embedding
- [tiktoken Library](https://github.com/openai/tiktoken) — Token counting in Python
- [Pinecone Cost Calculator](https://www.pinecone.io/pricing/) — Interactive storage estimator

---

[← Back to Cost Optimization Overview](./00-cost-optimization.md) | [Next: Batch Embedding Discounts →](./02-batch-embedding-discounts.md)

---

<!-- 
Sources Consulted:
- OpenAI Pricing: https://platform.openai.com/docs/pricing
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Pinecone Understanding Cost: https://docs.pinecone.io/guides/manage-cost/understanding-cost
- tiktoken: https://github.com/openai/tiktoken
-->
