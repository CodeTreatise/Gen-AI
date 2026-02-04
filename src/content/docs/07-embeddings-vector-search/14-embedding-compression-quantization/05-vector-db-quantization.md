---
title: "Vector Database Quantization Support"
---

# Vector Database Quantization Support

## Introduction

Modern vector databases implement quantization at the storage layer, automatically compressing vectors for you. Each database offers different quantization options with varying trade-offs. This lesson covers the quantization features of Qdrant, Milvus, and pgvector.

Understanding your database's quantization capabilities helps you choose the right compression strategy without building it yourself.

### What We'll Cover

- Qdrant: Scalar, Binary, and Product Quantization
- Milvus: Index-integrated quantization (SQ8, PQ, HNSW variants)
- pgvector: Half-precision and binary quantization
- Comparison matrix for choosing the right option

### Prerequisites

- Understanding of [quantization trade-offs](./03-quantization-tradeoffs.md)
- Familiarity with vector database concepts

---

## Qdrant Quantization

### Overview

Qdrant offers three quantization methods, configurable per collection:

```
┌─────────────────────────────────────────────────────────────────┐
│              Qdrant Quantization Options                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Method          Compression   Quality      Best For            │
│  ────────────────────────────────────────────────────────────   │
│  Scalar (int8)   4x            ~99%         General use        │
│  Binary (1-bit)  32x           ~95%*        High-dim, speed    │
│  Binary (2-bit)  16x           ~97%*        Balance            │
│  Product (PQ)    Up to 64x     ~85-95%      Maximum savings    │
│                                                                 │
│  * With rescoring enabled                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Scalar Quantization

The simplest and most reliable option. Converts float32 → int8.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType
)

client = QdrantClient(host="localhost", port=6333)

# Create collection with scalar quantization
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,       # Clip outliers beyond 99th percentile
            always_ram=True      # Keep quantized vectors in RAM
        )
    )
)
```

**Key parameters:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `type` | Quantization precision | `INT8` |
| `quantile` | Outlier clipping percentile | 0.95-0.99 |
| `always_ram` | Keep quantized in memory | `True` for speed |

### Binary Quantization

Qdrant supports multi-bit binary quantization for better quality:

```python
from qdrant_client.models import (
    BinaryQuantization, BinaryQuantizationConfig
)

# 1-bit binary (maximum compression)
client.create_collection(
    collection_name="documents_binary",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.DOT
    ),
    quantization_config=BinaryQuantization(
        binary=BinaryQuantizationConfig(
            always_ram=True
        )
    )
)
```

> **Note:** Binary quantization in Qdrant works best with dot product distance and high-dimensional vectors (1000+).

### Search with Rescoring

Enable rescoring to recover quality lost to quantization:

```python
from qdrant_client.models import SearchParams, QuantizationSearchParams

# Search with quantization-aware parameters
results = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10,
    search_params=SearchParams(
        quantization=QuantizationSearchParams(
            rescore=True,        # Re-rank with original vectors
            oversampling=2.0     # Fetch 2x candidates for rescoring
        )
    )
)
```

**Oversampling guidelines:**

| Quantization | Recommended Oversampling |
|--------------|--------------------------|
| Scalar (int8) | 1.2-1.5 |
| Binary (1-bit) | 2.0-3.0 |
| Binary (2-bit) | 1.5-2.0 |
| Product (PQ) | 2.0-4.0 |

### Product Quantization

For maximum compression when quality trade-off is acceptable:

```python
from qdrant_client.models import (
    ProductQuantization, ProductQuantizationConfig
)

client.create_collection(
    collection_name="documents_pq",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=ProductQuantizationConfig.CompressionRatio.X32,
            always_ram=True
        )
    )
)
```

---

## Milvus Quantization

### Index-Integrated Approach

Milvus integrates quantization into its index types rather than as a separate configuration:

```
┌─────────────────────────────────────────────────────────────────┐
│              Milvus Index Types with Quantization                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Index Type     Quantization   Compression   Use Case           │
│  ────────────────────────────────────────────────────────────   │
│  FLAT           None           1x            Exact search      │
│  IVF_FLAT       None           1x            Clustered exact   │
│  IVF_SQ8        Scalar (int8)  4x            Balanced          │
│  IVF_PQ         Product        16-64x        High compression  │
│  HNSW           None           1x            Fast, accurate    │
│  HNSW_SQ        Scalar         4x            Fast + compressed │
│  HNSW_PQ        Product        16-64x        HNSW + compression│
│  SCANN          IVF + PQ       Variable      Google's method   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### IVF_SQ8: Scalar Quantization

Combines IVF clustering with int8 quantization:

```python
from pymilvus import (
    Collection, FieldSchema, CollectionSchema, DataType,
    connections, utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields, description="Documents")

# Create collection
collection = Collection("documents", schema)

# Create IVF_SQ8 index
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "IP",      # Inner product (cosine after normalization)
    "params": {
        "nlist": 1024         # Number of clusters
    }
}

collection.create_index("embedding", index_params)
```

### HNSW_SQ: Graph + Scalar Quantization

Combines HNSW's fast search with scalar quantization:

```python
# HNSW with scalar quantization
index_params = {
    "index_type": "HNSW_SQ",
    "metric_type": "IP",
    "params": {
        "M": 16,                    # HNSW connections per node
        "efConstruction": 200,      # Build-time quality
        "sq_type": "SQ8"            # Scalar quantization: SQ6, SQ8, BF16, FP16
    }
}

collection.create_index("embedding", index_params)
```

**SQ type options:**

| Type | Description | Compression | Quality |
|------|-------------|-------------|---------|
| `SQ6` | 6-bit scalar | ~5.3x | ~98% |
| `SQ8` | 8-bit scalar | 4x | ~99% |
| `BF16` | Brain float16 | 2x | ~99.9% |
| `FP16` | Float16 | 2x | ~99.9% |

### IVF_PQ and HNSW_PQ: Product Quantization

For maximum compression:

```python
# IVF with product quantization
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "IP",
    "params": {
        "nlist": 1024,
        "m": 16,           # Number of subvectors (must divide dim evenly)
        "nbits": 8         # Bits per subvector quantization
    }
}

# HNSW with product quantization
index_params = {
    "index_type": "HNSW_PQ",
    "metric_type": "IP",
    "params": {
        "M": 16,
        "efConstruction": 200,
        "m": 16,           # PQ subvectors
        "nbits": 8         # Bits per code
    }
}
```

**PQ parameter tuning:**

| Param | Range | Effect |
|-------|-------|--------|
| `m` | 8-64 | More = better quality, less compression |
| `nbits` | 4-12 | More = better quality, larger codebooks |

### Search Parameters

```python
# Load collection for search
collection.load()

# Search with specific parameters
search_params = {
    "metric_type": "IP",
    "params": {
        "nprobe": 32,     # IVF: clusters to search
        "ef": 64          # HNSW: search quality
    }
}

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10
)
```

---

## pgvector Quantization

### Half-Precision Vectors

pgvector supports float16 via the `halfvec` type:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with half-precision vectors
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding halfvec(1536)    -- Float16, 2 bytes per dimension
);

-- Create HNSW index on halfvec
CREATE INDEX ON documents 
USING hnsw (embedding halfvec_l2_ops);

-- Search (automatic type matching)
SELECT id, content, embedding <-> '[0.1, 0.2, ...]'::halfvec AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Comparison: vector vs halfvec:**

| Aspect | vector (float32) | halfvec (float16) |
|--------|------------------|-------------------|
| Size per dim | 4 bytes | 2 bytes |
| Max dimensions | 16,000 | 16,000 |
| Precision | Full | Reduced |
| Index support | Full | Full |
| Quality | 100% | ~99.9% |

### Binary Vectors

pgvector supports true binary vectors with the `bit` type:

```sql
-- Create table with binary vectors
CREATE TABLE documents_binary (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding bit(1536)         -- 1536 bits = 192 bytes
);

-- Create index for Hamming distance
CREATE INDEX ON documents_binary 
USING hnsw (embedding bit_hamming_ops);

-- Search with Hamming distance
SELECT id, content, embedding <~> B'101010...'::bit(1536) AS hamming_dist
FROM documents_binary
ORDER BY hamming_dist
LIMIT 10;
```

### Binary Quantization Function

Convert float vectors to binary on-the-fly:

```sql
-- binary_quantize converts float vectors to bit vectors
-- Each positive value becomes 1, negative becomes 0

SELECT binary_quantize('[0.5, -0.2, 0.1, -0.8]'::vector);
-- Returns: 1010

-- Two-stage search pattern
WITH binary_candidates AS (
    -- Stage 1: Fast Hamming search on quantized vectors
    SELECT id, embedding
    FROM documents
    ORDER BY binary_quantize(embedding)::bit(1536) <~> 
             binary_quantize($1)::bit(1536)
    LIMIT 100
)
-- Stage 2: Rescore with full precision
SELECT id, embedding <=> $1 AS similarity
FROM binary_candidates
ORDER BY similarity
LIMIT 10;
```

### Complete Two-Stage Example

```sql
-- Create table storing both formats
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),           -- Full precision for rescoring
    embedding_binary bit(1536)        -- Binary for fast search
);

-- Function to insert with automatic binary quantization
CREATE OR REPLACE FUNCTION insert_document(
    p_content TEXT,
    p_embedding vector(1536)
) RETURNS INTEGER AS $$
DECLARE
    new_id INTEGER;
BEGIN
    INSERT INTO documents (content, embedding, embedding_binary)
    VALUES (
        p_content,
        p_embedding,
        binary_quantize(p_embedding)::bit(1536)
    )
    RETURNING id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Create indexes
CREATE INDEX idx_binary ON documents 
USING hnsw (embedding_binary bit_hamming_ops);

CREATE INDEX idx_float ON documents 
USING hnsw (embedding vector_cosine_ops);

-- Two-stage search function
CREATE OR REPLACE FUNCTION search_documents(
    query_vec vector(1536),
    top_k INTEGER DEFAULT 10,
    candidates INTEGER DEFAULT 100
) RETURNS TABLE(id INTEGER, content TEXT, similarity FLOAT) AS $$
BEGIN
    RETURN QUERY
    WITH binary_search AS (
        -- Stage 1: Binary Hamming search
        SELECT d.id, d.embedding
        FROM documents d
        ORDER BY d.embedding_binary <~> binary_quantize(query_vec)::bit(1536)
        LIMIT candidates
    )
    -- Stage 2: Float cosine rescoring
    SELECT bs.id, d.content, 1 - (bs.embedding <=> query_vec) AS similarity
    FROM binary_search bs
    JOIN documents d ON d.id = bs.id
    ORDER BY similarity DESC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;
```

---

## Comparison Matrix

### Feature Comparison

| Feature | Qdrant | Milvus | pgvector |
|---------|--------|--------|----------|
| Scalar (int8) | ✅ | ✅ (SQ8) | ❌ (use halfvec) |
| Float16 | ❌ | ✅ (FP16, BF16) | ✅ (halfvec) |
| Binary | ✅ (1-2 bit) | ❌ | ✅ (bit type) |
| Product Quantization | ✅ | ✅ | ❌ |
| Automatic rescoring | ✅ | ❌ | Manual |
| Oversampling param | ✅ | ❌ | Manual |

### Compression Comparison

| Method | Qdrant | Milvus | pgvector |
|--------|--------|--------|----------|
| 2x | ❌ | FP16/BF16 | halfvec |
| 4x | Scalar | SQ8 | ❌ |
| 16x | Binary (2-bit) | PQ (m=96) | ❌ |
| 32x | Binary (1-bit) | PQ (m=48) | bit |
| 64x | PQ | PQ (m=24) | ❌ |

### Choosing the Right Database

```
┌─────────────────────────────────────────────────────────────────┐
│              Decision Guide                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Need automatic rescoring?                                      │
│       └── Yes → Qdrant                                         │
│                                                                 │
│  Need Product Quantization (>16x)?                             │
│       └── Yes → Qdrant or Milvus                               │
│                                                                 │
│  Already using PostgreSQL?                                      │
│       └── Yes → pgvector (halfvec/bit)                         │
│                                                                 │
│  Need maximum configurability?                                  │
│       └── Yes → Milvus (many index types)                      │
│                                                                 │
│  Simple setup, good defaults?                                   │
│       └── Yes → Qdrant                                         │
│                                                                 │
│  Need SQL interface?                                            │
│       └── Yes → pgvector                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Examples

### Qdrant: Balanced Configuration

```python
# Good for most use cases
client.create_collection(
    collection_name="production",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)

# Search parameters
search_params = SearchParams(
    hnsw_ef=64,
    quantization=QuantizationSearchParams(
        rescore=True,
        oversampling=1.5
    )
)
```

### Milvus: High-Performance Configuration

```python
# Create optimized index
index_params = {
    "index_type": "HNSW_SQ",
    "metric_type": "IP",
    "params": {
        "M": 32,                # More connections for accuracy
        "efConstruction": 256,  # Higher build quality
        "sq_type": "SQ8"
    }
}
collection.create_index("embedding", index_params)

# Search with tuned parameters
search_params = {
    "metric_type": "IP",
    "params": {"ef": 128}       # Higher ef for better recall
}
```

### pgvector: Production Setup

```sql
-- Optimal settings for large collections
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;

-- Create table with both formats
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    embedding_half halfvec(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Partial index for recent documents (example)
CREATE INDEX idx_recent_embedding ON documents 
USING hnsw (embedding_half halfvec_cosine_ops)
WHERE created_at > NOW() - INTERVAL '30 days';

-- Full index
CREATE INDEX idx_all_embedding ON documents 
USING hnsw (embedding_half halfvec_cosine_ops)
WITH (m = 16, ef_construction = 100);
```

---

## Summary

✅ **Qdrant** offers the most flexible quantization with automatic rescoring  
✅ **Milvus** integrates quantization into index types (SQ8, PQ, HNSW variants)  
✅ **pgvector** provides halfvec (2x) and bit (32x) with SQL interface  
✅ **Match compression to your database**—each has different strengths  
✅ **Enable rescoring** to recover quality lost to quantization

---

**Next:** [Product Quantization →](./06-product-quantization.md)

---

<!-- 
Sources Consulted:
- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Milvus Index Documentation: https://milvus.io/docs/index.md
- pgvector GitHub: https://github.com/pgvector/pgvector
-->
