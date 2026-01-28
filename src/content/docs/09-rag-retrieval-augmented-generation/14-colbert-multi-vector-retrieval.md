---
title: "ColBERT & Multi-Vector Retrieval"
---

# ColBERT & Multi-Vector Retrieval

- ColBERT architecture
  - Late interaction between query and document
  - Token-level matching (not single vector)
  - Each token gets its own embedding
  - MaxSim: maximum similarity per query token
  - Sum of MaxSim scores for final ranking
- Advantages over single-vector
  - Better handling of long documents
  - More precise matching
  - Explainable relevance (which tokens matched)
  - Higher recall for complex queries
- ColBERT v2 and RAGatouille
  - Compressed representations
  - Efficient indexing with PLAID
  - Python library: RAGatouille
    - Import RAGPretrainedModel from ragatouille library
    - Load pretrained model: `RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")`
    - Index documents: call `RAG.index(documents, index_name="my_index")`
    - Search: call `RAG.search(query="...", k=10)` with query and number of results
    - Results include documents ranked by ColBERT's MaxSim scoring
- Multi-vector representations
  - Document → multiple embeddings
  - Capture different aspects/topics
  - Late fusion for scoring
  - Trade-off: storage vs. accuracy
- When to use ColBERT
  - Complex, multi-faceted queries
  - Long documents with diverse content
  - When single-vector retrieval underperforms
  - Explainability requirements
