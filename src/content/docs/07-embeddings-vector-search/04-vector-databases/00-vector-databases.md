---
title: "Vector Databases"
---

# Vector Databases

## Overview

Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vectors at scale. While traditional databases excel at exact matches, vector databases enable **similarity search**—finding the most semantically similar items to a query.

This lesson explores the vector database landscape in 2025, from managed cloud services to self-hosted solutions to leveraging your existing database infrastructure.

### Prerequisites

- Completed [Understanding Embeddings](../01-understanding-embeddings/00-understanding-embeddings.md)
- Completed [Generating Embeddings](../03-generating-embeddings/00-generating-embeddings.md)
- Familiarity with database concepts
- Python environment configured

### Lessons in This Section

| Lesson | Topic | Description |
|--------|-------|-------------|
| [01](./01-what-are-vector-databases.md) | What Are Vector Databases | Core concepts, ANN search, index structures |
| [02](./02-managed-vector-databases.md) | Managed Vector Databases | Pinecone, Weaviate Cloud, Qdrant Cloud, Zilliz |
| [03](./03-self-hosted-options.md) | Self-Hosted Options | Chroma, Milvus, Qdrant, LanceDB |
| [04](./04-postgresql-pgvector.md) | PostgreSQL with pgvector | SQL + vectors, HNSW, IVFFlat indexes |
| [05](./05-managed-postgresql-vector.md) | Managed PostgreSQL Vector | Supabase, Neon, AWS RDS, Azure |
| [06](./06-mongodb-atlas-vector-search.md) | MongoDB Atlas Vector Search | $vectorSearch, hybrid search, Atlas integration |
| [07](./07-redis-vector-search.md) | Redis Vector Search | In-memory performance, existing infrastructure |
| [08](./08-platform-integrated-stores.md) | Platform-Integrated Stores | OpenAI Vector Stores, Gemini RAG |
| [09](./09-selection-decision-tree.md) | Database Selection Guide | Scale, budget, infrastructure considerations |
| [10](./10-cost-scaling-considerations.md) | Cost and Scaling | Pricing models, optimization strategies |
| [11](./11-migration-strategies.md) | Migration Strategies | Export, import, zero-downtime migration |

---

## Learning Objectives

By completing this lesson, you will be able to:

- Explain how vector databases differ from traditional databases
- Choose the right vector database for your use case
- Set up and query multiple vector database solutions
- Understand index types (HNSW, IVFFlat) and their tradeoffs
- Design hybrid search combining vectors with metadata filters
- Plan for cost and scaling at different vector counts

---

## Quick Start Decision Guide

**Which vector database should you start with?**

| Scenario | Recommended Solution |
|----------|---------------------|
| **Learning/Prototyping** | Chroma (simple, in-memory) |
| **Already using PostgreSQL** | pgvector extension |
| **Already using MongoDB** | Atlas Vector Search |
| **Production, managed** | Pinecone, Weaviate Cloud, or Qdrant Cloud |
| **High-performance, self-hosted** | Milvus or Qdrant |
| **Serverless/Edge** | LanceDB or Turbopuffer |
| **Using OpenAI Assistants** | OpenAI Vector Stores |

---

## Navigation

**Next:** [What Are Vector Databases](./01-what-are-vector-databases.md)

**Previous:** [Generating Embeddings](../03-generating-embeddings/00-generating-embeddings.md)

---

## Further Reading

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
