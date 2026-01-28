---
title: "Managed RAG Services"
---

# Managed RAG Services

- OpenAI Vector Stores API
  - Managed vector storage and search
  - Automatic chunking (default: 800 tokens, 400 overlap)
  - Built-in embedding generation
  - Create and search:
    - Import OpenAI from openai library and create client instance
    - Create vector store: call `client.vector_stores.create(name="...")` with descriptive name
    - Upload file: use `client.vector_stores.files.upload_and_poll()` method
    - Pass vector_store_id and file opened in binary read mode ("rb")
    - Method polls until indexing completes automatically
    - Search: call `client.vector_stores.search()` with vector_store_id, query string, and max_num_results
    - Results include matched chunks with relevance scores and file metadata
  - Pricing: Free up to 1GB, $0.10/GB/day beyond
  - Custom chunking strategies per file
  - Attribute filtering for metadata
  - Expiration policies for cost control
- Gemini File Search
  - Upload files directly to Gemini API
  - Automatic grounding in file content
  - Supports PDF, DOCX, TXT, and more
  - Context caching for repeated queries
- AWS Bedrock Knowledge Bases
  - Managed RAG with S3 data sources
  - Automatic sync and indexing
  - OpenSearch or Pinecone backends
  - Built-in chunking strategies
- Azure AI Search + OpenAI
  - Cognitive Search with vector support
  - Semantic ranking built-in
  - Integration with Azure OpenAI
- When to use managed vs. custom
  - Managed: faster setup, less maintenance, limited customization
  - Custom: full control, complex pipelines, specific requirements
