---
title: "PostgreSQL with pgvector"
---

# PostgreSQL with pgvector

## Introduction

pgvector brings vector similarity search to PostgreSQL, the world's most popular open-source relational database. This means you can store embeddings alongside your existing data and query them with familiar SQL—no separate vector database required.

### What We'll Cover

- Installing and configuring pgvector
- Vector data types and operators
- HNSW vs IVFFlat indexes
- Combining SQL filters with vector search
- Performance tuning

### Prerequisites

- PostgreSQL 12+ installed
- Basic SQL knowledge
- Understanding of vector embeddings

---

## Installation

### Linux (Debian/Ubuntu)

```bash
# Install dependencies
sudo apt install postgresql-16-pgvector

# Or compile from source
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### macOS (Homebrew)

```bash
brew install pgvector
```

### Docker

```bash
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Enable the Extension

```sql
-- Connect to your database
CREATE EXTENSION vector;
```

---

## Vector Data Types

pgvector provides four vector types for different use cases:

| Type | Description | Max Dimensions | Use Case |
|------|-------------|----------------|----------|
| `vector` | 32-bit floats | 16,000 | Standard embeddings |
| `halfvec` | 16-bit floats | 16,000 | Memory-constrained |
| `bit` | Binary | 64,000 | Binary embeddings |
| `sparsevec` | Sparse vectors | 16,000 | High-dimensional sparse |

### Creating a Table with Vectors

```sql
-- Standard vector column (1536 dims for OpenAI)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- With half-precision for memory savings
CREATE TABLE documents_compressed (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding halfvec(1536)
);
```

### Inserting Vectors

```sql
-- Insert with vector literal
INSERT INTO documents (content, embedding)
VALUES (
    'Machine learning transforms data into insights',
    '[0.1, 0.2, 0.3, ...]'  -- 1536 floats
);

-- Insert from array
INSERT INTO documents (content, embedding)
VALUES (
    'PostgreSQL is powerful',
    ARRAY[0.1, 0.2, 0.3, ...]::vector
);
```

---

## Distance Operators

pgvector provides operators for different distance metrics:

| Operator | Name | Formula | Index Support |
|----------|------|---------|---------------|
| `<->` | L2 (Euclidean) | √Σ(a-b)² | ✅ HNSW, IVFFlat |
| `<=>` | Cosine distance | 1 - (a·b)/(‖a‖‖b‖) | ✅ HNSW, IVFFlat |
| `<#>` | Negative inner product | -(a·b) | ✅ HNSW, IVFFlat |
| `<+>` | L1 (Manhattan) | Σ\|a-b\| | ❌ No index |

### Basic Vector Search

```sql
-- Find 10 most similar by cosine distance
SELECT id, content, 
       embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;

-- Find by L2 distance
SELECT id, content,
       embedding <-> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'
LIMIT 10;
```

> **Note:** Cosine distance ranges from 0 (identical) to 2 (opposite). For similarity, use `1 - distance`.

---

## Indexing Strategies

Without an index, pgvector performs exact nearest neighbor search (scanning all rows). Indexes enable approximate nearest neighbor (ANN) search for scale.

### HNSW Index

Hierarchical Navigable Small World graphs—the recommended default:

```sql
-- Create HNSW index for cosine distance
CREATE INDEX ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Parameters:**

| Parameter | Default | Description | Trade-off |
|-----------|---------|-------------|-----------|
| `m` | 16 | Connections per node | ↑ = better recall, more memory |
| `ef_construction` | 64 | Build-time search width | ↑ = better quality, slower build |

**Index operator classes:**

| Distance | Operator Class |
|----------|---------------|
| L2 | `vector_l2_ops` |
| Cosine | `vector_cosine_ops` |
| Inner Product | `vector_ip_ops` |

### IVFFlat Index

Inverted File Flat—faster to build, requires more tuning:

```sql
-- Create IVFFlat index
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Parameters:**

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `lists` | rows/1000 | Number of clusters |

### When to Use Which

| Scenario | Recommended Index |
|----------|-------------------|
| < 100K vectors | No index (exact search) |
| 100K - 1M vectors | HNSW |
| > 1M vectors | HNSW or IVFFlat |
| Frequent updates | HNSW |
| One-time bulk load | IVFFlat |

---

## Query-Time Parameters

Tune search behavior at query time:

### HNSW Parameters

```sql
-- Increase search quality (default: 40)
SET hnsw.ef_search = 100;

-- Ensure exact ordering with iterative scan
SET hnsw.iterative_scan = strict_order;

-- Then run your query
SELECT * FROM documents
ORDER BY embedding <=> $1
LIMIT 10;
```

### IVFFlat Parameters

```sql
-- Probe more lists for better recall (default: 1)
SET ivfflat.probes = 10;

SELECT * FROM documents
ORDER BY embedding <=> $1
LIMIT 10;
```

---

## Filtering with SQL

The killer feature of pgvector: combine vector search with SQL filters.

### Pre-filtering (Recommended)

```sql
-- Filter first, then vector search
SELECT id, content, embedding <=> $1 AS distance
FROM documents
WHERE category = 'technology'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> $1
LIMIT 10;
```

### Using Partial Indexes

For frequently filtered columns, create partial indexes:

```sql
-- Index only active documents
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WHERE is_active = true;

-- Query uses the partial index
SELECT * FROM documents
WHERE is_active = true
ORDER BY embedding <=> $1
LIMIT 10;
```

### Expression Indexes

Index computed or cast values:

```sql
-- Index half-precision version
CREATE INDEX ON documents
USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);

-- Query with cast
SELECT * FROM documents
ORDER BY embedding::halfvec(1536) <=> $1::halfvec(1536)
LIMIT 10;
```

---

## Hybrid Search (Vector + Full-Text)

Combine semantic search with keyword matching:

```sql
-- Add full-text search column
ALTER TABLE documents ADD COLUMN search_vector tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ON documents USING gin(search_vector);

-- Hybrid search function
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_count INT DEFAULT 10,
    full_text_weight FLOAT DEFAULT 0.3,
    semantic_weight FLOAT DEFAULT 0.7
)
RETURNS TABLE(id INT, content TEXT, score FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        (full_text_weight * ts_rank(d.search_vector, websearch_to_tsquery(query_text)) +
         semantic_weight * (1 - (d.embedding <=> query_embedding))) AS score
    FROM documents d
    WHERE d.search_vector @@ websearch_to_tsquery(query_text)
       OR (d.embedding <=> query_embedding) < 0.5
    ORDER BY score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Use it
SELECT * FROM hybrid_search(
    'machine learning',
    '[0.1, 0.2, ...]'::vector,
    10,
    0.3,  -- full-text weight
    0.7   -- semantic weight
);
```

---

## Python Integration

### Using psycopg with NumPy

```python
import psycopg
from pgvector.psycopg import register_vector
import numpy as np
from openai import OpenAI

# Connect and register vector type
conn = psycopg.connect("dbname=mydb")
register_vector(conn)

openai_client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

# Insert
embedding = get_embedding("PostgreSQL with vectors")
with conn.cursor() as cur:
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        ("PostgreSQL with vectors", embedding)
    )
conn.commit()

# Search
query_embedding = get_embedding("database with AI")
with conn.cursor() as cur:
    cur.execute("""
        SELECT id, content, embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT 5
    """, (query_embedding, query_embedding))
    
    for row in cur.fetchall():
        print(f"{row[1]}: {row[2]:.4f}")
```

### Using SQLAlchemy

```python
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    embedding = Column(Vector(1536))

engine = create_engine('postgresql://user:pass@localhost/db')

# Create index
with engine.connect() as conn:
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING hnsw (embedding vector_cosine_ops)
    """))
    conn.commit()

# Search
with Session(engine) as session:
    results = (
        session.query(Document)
        .order_by(Document.embedding.cosine_distance(query_embedding))
        .limit(10)
        .all()
    )
```

---

## Performance Optimization

### Memory Settings

```sql
-- Increase for faster index builds
SET maintenance_work_mem = '2GB';

-- Parallel index building
SET max_parallel_maintenance_workers = 4;
```

### Monitoring

```sql
-- Check index size
SELECT pg_size_pretty(pg_relation_size('documents_embedding_idx'));

-- Check if index is being used
EXPLAIN ANALYZE
SELECT * FROM documents
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;

-- Should show "Index Scan using documents_embedding_idx"
```

### Quantization for Memory

```sql
-- Create quantized index (reduces memory ~4x)
CREATE INDEX ON documents
USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);
```

---

## Hands-on Exercise

### Your Task

Set up pgvector and implement a document search with hybrid (vector + full-text) capabilities:

### Requirements

1. Create a table with content, embedding, and full-text search columns
2. Insert 10+ documents with embeddings
3. Create both HNSW and GIN indexes
4. Implement hybrid search combining both methods
5. Compare results from pure vector vs hybrid search

<details>
<summary>💡 Hints</summary>

- Use `GENERATED ALWAYS AS` for the tsvector column
- The `websearch_to_tsquery` function handles natural language queries
- Weight the scores based on your use case (usually 0.7 semantic, 0.3 keyword)

</details>

<details>
<summary>✅ Solution</summary>

```sql
-- 1. Create table
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    search_vector tsvector 
        GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED
);

-- 2. Create indexes
CREATE INDEX articles_embedding_idx ON articles
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX articles_search_idx ON articles
USING gin(search_vector);

-- 3. Insert sample data (pseudo-code for embedding)
INSERT INTO articles (title, content, embedding) VALUES
('PostgreSQL Performance', 'Tips for optimizing PostgreSQL...', '[...]'::vector),
('Vector Databases', 'Understanding vector similarity...', '[...]'::vector);

-- 4. Hybrid search function
CREATE OR REPLACE FUNCTION search_articles(
    query_text TEXT,
    query_vec vector(1536),
    k INT DEFAULT 5
)
RETURNS TABLE(id INT, title TEXT, score FLOAT) AS $$
SELECT 
    a.id,
    a.title,
    (0.3 * COALESCE(ts_rank(search_vector, websearch_to_tsquery(query_text)), 0) +
     0.7 * (1 - (embedding <=> query_vec))) AS score
FROM articles a
ORDER BY score DESC
LIMIT k;
$$ LANGUAGE sql;

-- 5. Compare methods
-- Pure vector
SELECT title, embedding <=> '[...]'::vector AS distance
FROM articles ORDER BY distance LIMIT 5;

-- Hybrid
SELECT * FROM search_articles('PostgreSQL optimization', '[...]'::vector);
```

</details>

---

## Summary

✅ pgvector adds vector search to PostgreSQL with a simple extension

✅ Use HNSW indexes for most workloads—faster queries, easier tuning

✅ Combine vector search with SQL WHERE clauses for powerful filtering

✅ Hybrid search merges semantic understanding with keyword matching

✅ Memory can be reduced ~4x with halfvec quantization

**Next:** [Managed PostgreSQL Vector Services](./05-managed-postgresql-vector.md)

---

## Further Reading

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Supabase Vector Guide](https://supabase.com/docs/guides/ai)

---

<!-- 
Sources Consulted:
- pgvector GitHub README: https://github.com/pgvector/pgvector
- pgvector docs on indexing: https://github.com/pgvector/pgvector#indexing
- PostgreSQL full-text search: https://www.postgresql.org/docs/current/textsearch.html
-->
