---
title: "RAG Cost Optimization"
---

# RAG Cost Optimization

- Embedding cost reduction
  - Batch processing discounts
  - Smaller models for initial retrieval
  - Local embedding models
  - Caching embedded queries
- Storage optimization
  - Quantized embeddings (int8, binary)
  - Index compression
  - Tiered storage (hot/cold)
  - Expiration policies
- LLM cost optimization
  - Shorter context with better retrieval
  - Smaller models for simple queries
  - Caching common responses
  - Prompt caching for repeated context
- Retrieval efficiency
  - Pre-filtering before vector search
  - Approximate nearest neighbors
  - Index sharding and partitioning
  - Query routing to relevant indexes
- Cost comparison (per 1M queries)
  | Component | Low-cost | Standard | Premium |
  |-----------|----------|----------|---------|
  | Embeddings | $0.02 (local) | $0.13 (text-3-small) | $0.65 (text-3-large) |
  | Storage | $0.10/GB/day | $0.25/GB/day | Managed service |
  | LLM | $0.15 (GPT-4.1-mini) | $2.00 (GPT-4.1) | $15.00 (GPT-4.5) |
- Break-even analysis: RAG vs. fine-tuning
  - RAG: lower upfront, per-query cost
  - Fine-tuning: higher upfront, lower per-query
  - RAG wins for: <10K queries/month, frequent updates
  - Fine-tuning wins for: >100K queries/month, static knowledge
