---
title: "Retrieval Strategies"
---

# Retrieval Strategies

- Basic similarity search
  - Top-k retrieval
  - Similarity thresholds
  - Result scoring
  - Simple but effective
- Maximum Marginal Relevance (MMR)
  - Diversity in results
  - Avoiding redundancy
  - Lambda parameter tuning
  - When MMR helps
- Hybrid search (keyword + semantic)
  - BM25 + vector combination
  - Score normalization
  - Fusion techniques (RRF, DBSF)
  - Query routing
  - OpenAI hybrid search weights:
    - Configure via `ranking_options.hybrid_search` object
    - Set `embedding_weight` (e.g., 0.7) for semantic similarity contribution
    - Set `text_weight` (e.g., 0.3) for BM25 keyword matching contribution
    - Weights should sum to 1.0 for normalized scoring
  - Balance semantic understanding with exact matches
- Query rewriting and transformation
  - OpenAI automatic query rewriting (`rewrite_query: true`)
  - Query optimization for embedding search
  - Example transformations:
    | Original query | Rewritten query |
    |----------------|------------------|
    | "What's the height of main office?" | "primary office building height" |
    | "How do I file a complaint?" | "service complaint filing process" |
  - HyDE (Hypothetical Document Embedding)
    - Generate hypothetical answer first
    - Embed the hypothetical answer
    - Search for similar real documents
  - Multi-query generation with LLM
- Multi-query retrieval
  - Query decomposition for complex questions
  - Multiple search passes
  - Result aggregation and deduplication
  - LLM-generated sub-queries
- Parent-child retrieval
  - Small chunks for matching
  - Large chunks for context
  - Hierarchical indexing
  - Context expansion
- Attribute filtering (OpenAI pattern)
  - Pre-retrieval filtering syntax:
    - Comparison filters use object with `type`, `key`, and `value` properties
    - Example: equality check with type "eq", key "region", value "US"
    - Example: greater-than-or-equal with type "gte", key "date", value as Unix timestamp
    - Example: inclusion check with type "in", key "category", value as array of strings
    - Compound filters use type "and" or "or" with nested `filters` array
    - Compound example: AND filter combining department equals "engineering" with date >= timestamp
  - Operators: eq, ne, gt, gte, lt, lte, in, nin
  - Compound: and, or
  - Access control integration
  - Date range and source filtering
- Ranking and score thresholds
  - OpenAI ranker options: `auto`, `default-2024-08-21`
  - Score threshold (0.0 to 1.0) for quality control
  - Higher threshold = more relevant but fewer results
  - Balance precision vs. recall with threshold tuning
