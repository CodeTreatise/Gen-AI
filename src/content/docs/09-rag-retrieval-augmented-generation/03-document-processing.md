---
title: "Document Processing"
---

# Document Processing

- Chunking strategies revisited
  - RAG-optimized chunk sizes
  - Semantic coherence
  - Retrieval unit size
  - Context window fit
- Handling different document types
  - Type-specific parsers
  - Unified output format
  - Quality validation
  - Fallback handling
- Preserving document structure
  - Heading hierarchy
  - Section relationships
  - List and table context
  - Cross-reference handling
- Table and image handling
  - Table to text conversion
  - Image description generation
  - Chart data extraction
  - Alt text usage
- Multi-modal document processing
  - Combined text and image
  - Vision model integration
  - Layout understanding
  - Rich content preservation
- Contextual Retrieval (Anthropic method)
  - The lost context problem in naive chunking
  - Prepending chunk-specific context before embedding
  - Claude prompt for context generation:
    ```
    <document>{{WHOLE_DOCUMENT}}</document>
    <chunk>{{CHUNK_CONTENT}}</chunk>
    Please give a short succinct context to situate this chunk
    within the overall document for improving search retrieval.
    ```
  - Context typically 50-100 tokens prepended to chunk
  - 35% improvement with contextual embeddings alone
  - 49% improvement combining contextual embeddings + contextual BM25
  - 67% improvement with contextual retrieval + reranking
  - Use prompt caching to reduce context generation costs
  - Custom contextualizer prompts for domain-specific use cases
- Late Chunking (Jina method)
  - Problem: naive chunking produces i.i.d. embeddings losing context
  - Solution: apply transformer to entire document first
  - Generate token-level embeddings with full document context
  - Then apply mean pooling to each chunk boundary
  - Result: conditional chunk embeddings (not independent)
  - Requires long-context embedding models (8K+ tokens)
  - jina-embeddings-v3 supports late chunking natively
  - Effectiveness increases with document length
  - Boundary cues applied after getting token embeddings
  - Comparison:
    | Approach | Context preserved | Embedding type |
    |----------|-------------------|----------------|
    | Naive chunking | Lost | i.i.d. |
    | Contextual retrieval | Prepended text | Enhanced i.i.d. |
    | Late chunking | Full document | Conditional |
- Semantic chunking with AI
  - LLM-based chunk boundary detection
  - Topic coherence analysis
  - Natural break point identification
  - Preserving argument flow
  - Cost vs. quality tradeoffs
