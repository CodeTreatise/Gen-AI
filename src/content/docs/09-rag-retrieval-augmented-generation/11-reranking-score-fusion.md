---
title: "Reranking & Score Fusion"
---

# Reranking & Score Fusion

- Why reranking matters
  - Initial retrieval returns many candidates (100-150)
  - Reranker scores each for relevance
  - Select top-K (e.g., 20) for generation
  - Dramatically improves precision
- Cross-encoder reranking
  - Process query + document together
  - More accurate than bi-encoder similarity
  - Higher latency per document
  - Use after initial retrieval
- Cohere Rerank 3.5
  - State-of-the-art reranking model
  - Supports 100+ languages
  - Reasoning capability for complex queries
  - Semi-structured data handling (JSON, code)
  - Usage:
    - Import cohere library and create Client with api_key
    - Call `co.rerank()` method with required parameters
    - Set model to "rerank-v3.5" for latest version
    - Provide query string for relevance scoring
    - Pass documents as list of retrieved chunks to rerank
    - Set top_n to limit number of results (e.g., 10)
    - Set return_documents=True to include document text in results
    - Results contain reranked documents with relevance scores
- Voyage Reranker
  - rerank-2 and rerank-lite-2 models
  - Optimized for RAG pipelines
  - Low latency options available
- Score fusion techniques
  - Reciprocal Rank Fusion (RRF):
    ```
    RRF_score = Σ 1/(k + rank_i)
    ```
  - Distribution-Based Score Fusion (DBSF):
    - Normalize scores to [0,1] range
    - `normalized = (score - min) / (max - min)`
  - Weighted combination of semantic + keyword scores
- Reranking pipeline
  1. Initial retrieval: 100-150 candidates
  2. Reranking: score all candidates
  3. Select top-20 for context
  4. Generate response
  - 67% improvement over naive retrieval (Anthropic study)
