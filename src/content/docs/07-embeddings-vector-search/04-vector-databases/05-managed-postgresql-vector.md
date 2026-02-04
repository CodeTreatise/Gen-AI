---
title: "Managed PostgreSQL Vector Services"
---

# Managed PostgreSQL Vector Services

## Introduction

Managed PostgreSQL services with pgvector let you use vector search without managing database infrastructure. You get automatic backups, scaling, and high availability—plus the familiar PostgreSQL experience.

### What We'll Cover

- Supabase Vector
- Neon with pgvector
- AWS RDS for PostgreSQL
- Azure Database for PostgreSQL
- Provider comparison

### Prerequisites

- Understanding of pgvector concepts
- Familiarity with PostgreSQL
- Cloud provider account (for hands-on)

---

## Supabase Vector

Supabase provides PostgreSQL with pgvector pre-installed, plus a generous free tier and excellent developer experience.

### Key Features

- **pgvector pre-enabled** - No setup required
- **Edge Functions** - Serverless compute alongside vectors
- **Built-in Auth** - Row-level security for vectors
- **Dashboard** - Visual SQL editor and table views
- **Real-time** - Subscribe to vector changes

### Setup

1. Create a project at [supabase.com](https://supabase.com)
2. pgvector is already installed—just enable it:

```sql
-- Run in SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

### Creating Tables and Indexes

```sql
-- Create documents table
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for cosine similarity
CREATE INDEX documents_embedding_idx ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: users can only access their own documents
CREATE POLICY "Users access own documents" ON documents
FOR ALL USING (auth.uid() = (metadata->>'user_id')::uuid);
```

### Python Client

```python
from supabase import create_client, Client
from openai import OpenAI

# Initialize clients
supabase: Client = create_client(
    "https://your-project.supabase.co",
    "your-anon-key"
)
openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Insert document
embedding = get_embedding("Machine learning basics")
supabase.table("documents").insert({
    "content": "Machine learning basics",
    "embedding": embedding,
    "metadata": {"category": "AI"}
}).execute()

# Vector search using RPC function
# First, create the function in SQL Editor:
"""
CREATE OR REPLACE FUNCTION search_documents(
    query_embedding vector(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'
)
RETURNS TABLE(id BIGINT, content TEXT, similarity FLOAT)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE (filter = '{}' OR d.metadata @> filter)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""

# Call the function
query_embedding = get_embedding("What is ML?")
results = supabase.rpc("search_documents", {
    "query_embedding": query_embedding,
    "match_count": 5,
    "filter": {"category": "AI"}
}).execute()

for doc in results.data:
    print(f"{doc['content']}: {doc['similarity']:.3f}")
```

### Edge Function for Embeddings

```typescript
// supabase/functions/embed-and-search/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

serve(async (req) => {
  const { query } = await req.json();
  
  // Get embedding from OpenAI
  const embeddingResponse = await fetch(
    "https://api.openai.com/v1/embeddings",
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${Deno.env.get("OPENAI_API_KEY")}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: query,
      }),
    }
  );
  
  const { data } = await embeddingResponse.json();
  const embedding = data[0].embedding;
  
  // Search with Supabase
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );
  
  const { data: results } = await supabase.rpc("search_documents", {
    query_embedding: embedding,
    match_count: 5,
  });
  
  return new Response(JSON.stringify(results), {
    headers: { "Content-Type": "application/json" },
  });
});
```

---

## Neon

Neon offers serverless PostgreSQL with pgvector, featuring instant branching and scale-to-zero capabilities.

### Key Features

- **Serverless** - Scales to zero, pay per use
- **Branching** - Instant database copies for testing
- **pgvector included** - Available on all plans
- **Autoscaling** - Compute scales automatically
- **Connection pooling** - Built-in PgBouncer

### Setup

```python
# Install Neon's serverless driver
# pip install neon-api psycopg[binary]

import psycopg
from pgvector.psycopg import register_vector

# Connect using connection string from Neon dashboard
conn = psycopg.connect(
    "postgresql://user:password@ep-cool-name-123456.us-east-1.aws.neon.tech/neondb?sslmode=require"
)
register_vector(conn)

# Enable pgvector
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
```

### Branch for Testing

```python
from neon_api import NeonAPI

neon = NeonAPI(api_key="your-neon-api-key")

# Create a branch (instant copy of your database)
branch = neon.branches.create(
    project_id="your-project-id",
    branch={
        "name": "test-embeddings",
        "parent_id": "main-branch-id"
    }
)

# Connect to branch for testing
test_conn = psycopg.connect(branch.connection_uri)

# Test changes without affecting production
with test_conn.cursor() as cur:
    cur.execute("DROP INDEX documents_embedding_idx")
    cur.execute("""
        CREATE INDEX documents_embedding_idx ON documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 32, ef_construction = 128)
    """)
    # Run benchmarks...

# Delete branch when done
neon.branches.delete(project_id="your-project-id", branch_id=branch.id)
```

---

## AWS RDS for PostgreSQL

Amazon RDS offers managed PostgreSQL with pgvector support through the `pgvector` extension.

### Key Features

- **Enterprise-grade** - Multi-AZ, automated backups
- **Scalable** - Vertical and read replica scaling
- **IAM integration** - AWS security model
- **pgvector support** - Available on PostgreSQL 15.4+

### Enable pgvector

```sql
-- pgvector is available but not enabled by default
CREATE EXTENSION vector;

-- Check version
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### Connect with boto3 and IAM Auth

```python
import boto3
import psycopg
from pgvector.psycopg import register_vector

def get_rds_auth_token():
    client = boto3.client('rds')
    return client.generate_db_auth_token(
        DBHostname='your-instance.region.rds.amazonaws.com',
        Port=5432,
        DBUsername='iam_user',
        Region='us-east-1'
    )

# Connect with IAM token
conn = psycopg.connect(
    host='your-instance.region.rds.amazonaws.com',
    port=5432,
    dbname='postgres',
    user='iam_user',
    password=get_rds_auth_token(),
    sslmode='require'
)
register_vector(conn)
```

### Performance Recommendations

```sql
-- For large HNSW index builds, increase memory
-- Modify parameter group in AWS Console:
-- maintenance_work_mem = '2GB'
-- max_parallel_maintenance_workers = 4

-- Or set per-session
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;

CREATE INDEX CONCURRENTLY documents_embedding_idx
ON documents USING hnsw (embedding vector_cosine_ops);
```

---

## Azure Database for PostgreSQL

Azure's flexible server supports pgvector with integrated AI capabilities through Azure AI extension.

### Key Features

- **Flexible Server** - pgvector on PostgreSQL 15+
- **Azure AI extension** - Call Azure OpenAI directly from SQL
- **Geo-redundancy** - Built-in disaster recovery
- **Private Link** - VNet integration

### Enable Extensions

```sql
-- Enable pgvector
CREATE EXTENSION vector;

-- Optional: Azure AI extension for embeddings in SQL
CREATE EXTENSION azure_ai;

-- Configure Azure OpenAI
SELECT azure_ai.set_setting('azure_openai.endpoint', 'https://your-resource.openai.azure.com');
SELECT azure_ai.set_setting('azure_openai.subscription_key', 'your-key');
```

### Generate Embeddings in SQL

```sql
-- Create embeddings directly in SQL using Azure OpenAI
SELECT azure_openai.create_embeddings(
    'text-embedding-3-small',
    'Machine learning is fascinating'
);

-- Insert with auto-generated embedding
INSERT INTO documents (content, embedding)
SELECT 
    content,
    azure_openai.create_embeddings('text-embedding-3-small', content)
FROM staging_table;
```

---

## Provider Comparison

| Feature | Supabase | Neon | AWS RDS | Azure |
|---------|----------|------|---------|-------|
| **Free Tier** | 500MB + 2 projects | 3GB | 12 months free | $200 credit |
| **Serverless** | ❌ | ✅ | ❌ | ❌ |
| **pgvector Pre-installed** | ✅ | ✅ | ❌ (enable required) | ❌ |
| **Max Dimensions** | 16,000 | 16,000 | 16,000 | 16,000 |
| **Branching** | ❌ | ✅ | ❌ | ❌ |
| **AI Extensions** | ❌ | ❌ | ❌ | ✅ Azure AI |
| **Best For** | Startups, rapid dev | Dev/test, variable load | Enterprise, AWS shops | Azure enterprises |

### Pricing Comparison (Approximate)

| Provider | Small (1M vectors) | Medium (10M vectors) |
|----------|-------------------|---------------------|
| Supabase | $25/mo (Pro) | $75+/mo |
| Neon | $19/mo | $50+/mo |
| AWS RDS | $30/mo (db.t3.small) | $150+/mo |
| Azure | $35/mo | $160+/mo |

> **Note:** Prices vary by region and configuration. Vector workloads are memory-intensive—plan accordingly.

---

## Migration Patterns

### From Self-Hosted to Managed

```bash
# 1. Export with pg_dump
pg_dump -h localhost -U postgres -d mydb \
  -t documents --no-owner --no-privileges \
  > documents_backup.sql

# 2. Import to managed service
psql -h managed-host.example.com -U admin -d mydb \
  < documents_backup.sql

# 3. Recreate indexes (required after import)
psql -h managed-host.example.com -U admin -d mydb \
  -c "CREATE INDEX documents_embedding_idx ON documents 
      USING hnsw (embedding vector_cosine_ops)"
```

### Between Managed Providers

```python
import psycopg

# Connect to both
source = psycopg.connect("postgresql://source-provider...")
target = psycopg.connect("postgresql://target-provider...")

# Stream data in batches
BATCH_SIZE = 1000

with source.cursor() as src_cur:
    src_cur.execute("SELECT COUNT(*) FROM documents")
    total = src_cur.fetchone()[0]
    
    for offset in range(0, total, BATCH_SIZE):
        src_cur.execute(f"""
            SELECT id, content, embedding, metadata
            FROM documents
            ORDER BY id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """)
        rows = src_cur.fetchall()
        
        with target.cursor() as tgt_cur:
            tgt_cur.executemany("""
                INSERT INTO documents (id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s)
            """, rows)
        target.commit()
        print(f"Migrated {offset + len(rows)}/{total}")
```

---

## Hands-on Exercise

### Your Task

Set up a document search system on Supabase with Row Level Security:

### Requirements

1. Create a Supabase project (free tier)
2. Create a documents table with user ownership
3. Enable RLS with policies for multi-tenant access
4. Create a search RPC function
5. Test that users can only search their own documents

<details>
<summary>💡 Hints</summary>

- Store `user_id` in the metadata JSONB column
- Use `auth.uid()` in RLS policies
- The RPC function runs with invoker's permissions

</details>

<details>
<summary>✅ Solution</summary>

```sql
-- 1. Create table
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    user_id UUID NOT NULL DEFAULT auth.uid(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Create index
CREATE INDEX documents_embedding_idx ON documents
USING hnsw (embedding vector_cosine_ops);

-- 3. Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- 4. Create policies
CREATE POLICY "Users read own docs" ON documents
FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users insert own docs" ON documents
FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users delete own docs" ON documents
FOR DELETE USING (auth.uid() = user_id);

-- 5. Search function (respects RLS)
CREATE OR REPLACE FUNCTION search_my_documents(
    query_embedding vector(1536),
    match_count INT DEFAULT 5
)
RETURNS TABLE(id BIGINT, content TEXT, similarity FLOAT)
LANGUAGE sql
SECURITY INVOKER  -- Uses caller's RLS policies
AS $$
    SELECT 
        d.id,
        d.content,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;
```

```python
# Test multi-tenant access
from supabase import create_client

supabase = create_client(url, key)

# Sign in as user A
supabase.auth.sign_in_with_password({
    "email": "user_a@example.com",
    "password": "password"
})

# User A's search only returns their documents
results = supabase.rpc("search_my_documents", {
    "query_embedding": query_embedding
}).execute()
```

</details>

---

## Summary

✅ Supabase offers the best developer experience with pgvector pre-installed

✅ Neon provides serverless PostgreSQL with instant branching for testing

✅ AWS RDS delivers enterprise reliability with full AWS integration

✅ Azure adds unique AI extensions for in-database embeddings

✅ All providers support the same pgvector SQL—code is portable

**Next:** [MongoDB Atlas Vector Search](./06-mongodb-atlas-vector-search.md)

---

## Further Reading

- [Supabase Vector Guide](https://supabase.com/docs/guides/ai)
- [Neon pgvector Docs](https://neon.tech/docs/extensions/pgvector)
- [AWS RDS pgvector Guide](https://aws.amazon.com/blogs/database/accelerate-hnsw-indexing-and-searching-with-pgvector-on-amazon-rds-for-postgresql/)
- [Azure PostgreSQL Vector Search](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/how-to-use-pgvector)

---

<!-- 
Sources Consulted:
- Supabase Vector docs: https://supabase.com/docs/guides/ai
- Neon pgvector: https://neon.tech/docs/extensions/pgvector
- AWS RDS pgvector blog: https://aws.amazon.com/blogs/database/
- Azure PostgreSQL vector: https://learn.microsoft.com/en-us/azure/postgresql/
-->
