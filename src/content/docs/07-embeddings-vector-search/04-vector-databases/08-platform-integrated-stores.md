---
title: "Platform-Integrated Vector Stores"
---

# Platform-Integrated Vector Stores

## Introduction

AI platforms like OpenAI and Google now include built-in vector storage. These platform-integrated stores simplify RAG implementations by handling embedding generation, storage, and retrieval in a single API call.

### What We'll Cover

- OpenAI Vector Stores (Responses API)
- Google Gemini File Search
- When to use platform stores vs standalone databases
- Limitations and trade-offs

### Prerequisites

- OpenAI API access
- Google AI Studio account
- Understanding of RAG concepts

---

## OpenAI Vector Stores

OpenAI's Responses API includes a file_search tool that automatically creates embeddings, stores them, and retrieves relevant content.

### Key Features

- **Automatic chunking** - Files are split optimally
- **Automatic embedding** - No embedding API calls needed
- **Built-in retrieval** - Search happens server-side
- **Citation support** - Responses include source references
- **Multiple file types** - PDF, DOCX, TXT, MD, and more

### Creating a Vector Store

```python
from openai import OpenAI

client = OpenAI()

# Create a vector store
vector_store = client.vector_stores.create(
    name="product-documentation"
)

print(f"Vector Store ID: {vector_store.id}")
# vs_abc123...
```

### Adding Files

```python
# Upload and add a single file
file = client.files.create(
    file=open("product-manual.pdf", "rb"),
    purpose="assistants"
)

# Add file to vector store
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id
)

# Or upload multiple files at once
file_paths = ["doc1.pdf", "doc2.pdf", "doc3.md"]
file_streams = [open(path, "rb") for path in file_paths]

file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams
)

print(f"Status: {file_batch.status}")
print(f"File counts: {file_batch.file_counts}")
```

### Using with Responses API

```python
# Create a response with file_search tool
response = client.responses.create(
    model="gpt-4o",
    input="What are the warranty terms for the product?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id]
    }]
)

# Access the response
print(response.output_text)

# Get citations
for annotation in response.output[0].content[0].annotations:
    print(f"Source: {annotation.filename}")
    print(f"Quote: {annotation.text}")
```

### Managing Vector Stores

```python
# List all vector stores
stores = client.vector_stores.list()
for store in stores.data:
    print(f"{store.name}: {store.file_counts.total} files")

# List files in a store
files = client.vector_stores.files.list(vector_store_id=vector_store.id)
for f in files.data:
    print(f"{f.id}: {f.status}")

# Delete a file from store
client.vector_stores.files.delete(
    vector_store_id=vector_store.id,
    file_id="file_abc123"
)

# Delete the vector store
client.vector_stores.delete(vector_store_id=vector_store.id)
```

### Expiration Policies

```python
# Auto-expire after 7 days of inactivity
vector_store = client.vector_stores.create(
    name="temp-docs",
    expires_after={
        "anchor": "last_active_at",
        "days": 7
    }
)
```

---

## Google Gemini File Search

Gemini's context caching and file handling enable document-grounded responses.

### Uploading Files

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

# Upload a file
uploaded_file = genai.upload_file(
    path="research-paper.pdf",
    display_name="Research Paper"
)

print(f"File URI: {uploaded_file.uri}")
print(f"State: {uploaded_file.state.name}")
```

### Generating with File Context

```python
# Wait for file processing
import time

while uploaded_file.state.name == "PROCESSING":
    time.sleep(2)
    uploaded_file = genai.get_file(uploaded_file.name)

# Generate with file context
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content([
    "Summarize the key findings from this paper.",
    uploaded_file
])

print(response.text)
```

### Multiple Files

```python
# Upload multiple files
files = []
for path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    f = genai.upload_file(path)
    files.append(f)

# Wait for all to process
for f in files:
    while f.state.name == "PROCESSING":
        time.sleep(2)
        f = genai.get_file(f.name)

# Query across all files
response = model.generate_content([
    "Compare the approaches described in these documents.",
    *files
])
```

### Context Caching (Cost Optimization)

For repeated queries over the same documents:

```python
from google.generativeai import caching

# Create a cached context
cache = caching.CachedContent.create(
    model="models/gemini-1.5-pro-001",
    display_name="product-docs-cache",
    contents=[uploaded_file],
    ttl="3600s"  # 1 hour
)

# Use cached context for queries (cheaper)
cached_model = genai.GenerativeModel.from_cached_content(cache)

response = cached_model.generate_content("What are the main features?")
print(response.text)

# Delete when done
cache.delete()
```

---

## Comparison: Platform Stores vs Standalone

| Feature | OpenAI Vector Stores | Gemini Files | Standalone (Pinecone, etc.) |
|---------|---------------------|--------------|----------------------------|
| **Setup** | Minimal | Minimal | Moderate |
| **Embedding control** | None | None | Full |
| **Chunking control** | Limited | None | Full |
| **Metadata filtering** | None | None | ✅ Advanced |
| **Multi-model** | OpenAI only | Gemini only | Any model |
| **Hybrid search** | ❌ | ❌ | ✅ |
| **Cost visibility** | Bundled | Bundled | Separate |
| **Max files/vectors** | 10K files | Limited | Millions+ |

### When to Use Platform Stores

✅ **Use OpenAI/Gemini built-in when:**
- Rapid prototyping and POCs
- Simple document Q&A
- You're already committed to one platform
- You don't need metadata filtering
- Document count is under 1,000

### When to Use Standalone Vector DBs

✅ **Use Pinecone/Qdrant/etc when:**
- Need metadata filtering
- Multi-modal or hybrid search required
- Using multiple LLM providers
- Cost transparency is important
- Scale beyond 10K documents
- Custom chunking strategies needed

---

## Migration Patterns

### From Platform Store to Standalone

```python
from openai import OpenAI
from pinecone import Pinecone

# Export from OpenAI (conceptual - actual export requires file re-download)
openai_client = OpenAI()
pinecone_client = Pinecone()
index = pinecone_client.Index("my-index")

# List files in vector store
files = openai_client.vector_stores.files.list(
    vector_store_id="vs_abc123"
)

# Re-process each file
for file_ref in files.data:
    # Download file content
    file_content = openai_client.files.content(file_ref.id)
    
    # Chunk and embed with your preferred method
    chunks = your_chunking_function(file_content)
    
    # Upload to standalone database
    for i, chunk in enumerate(chunks):
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.text
        ).data[0].embedding
        
        index.upsert([{
            "id": f"{file_ref.id}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.text,
                "file_id": file_ref.id,
                "chunk_index": i
            }
        }])
```

---

## Cost Considerations

### OpenAI Vector Stores

```
Storage: $0.10 per GB per day
No separate embedding costs (included in storage)
No retrieval costs (included in API call)
```

**Example:** 1GB of documents × 30 days = $3/month

### Gemini Context Caching

```
Caching: $4.50 per million tokens per hour
Generation with cache: 75% discount on input tokens
```

**Example:** 500K tokens cached for 8 hours = $18

### Standalone Comparison

| Provider | Storage (1M vectors) | Query (1M/month) |
|----------|---------------------|------------------|
| Pinecone | ~$70/month | Included |
| Qdrant Cloud | ~$30/month | Included |
| OpenAI Stores | ~$15/month* | Included |

*Assuming 500MB of documents

---

## Best Practices

### Platform Stores

```python
# ✅ Good: Organized file naming
file = client.files.create(
    file=open("2024-Q1-report.pdf", "rb"),
    purpose="assistants"
)

# ✅ Good: Use file batches for multiple files
client.vector_stores.file_batches.upload_and_poll(...)

# ✅ Good: Set expiration for temporary stores
client.vector_stores.create(
    name="temp",
    expires_after={"anchor": "last_active_at", "days": 1}
)

# ❌ Bad: Creating new store per query
# ❌ Bad: Not cleaning up unused stores
```

### Hybrid Approach

```python
# Use platform store for simple Q&A
# Use standalone for filtered search

class HybridSearch:
    def __init__(self):
        self.openai = OpenAI()
        self.pinecone = Pinecone().Index("products")
        
    def search(self, query: str, filters: dict = None):
        if filters:
            # Use standalone for filtered search
            embedding = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            ).data[0].embedding
            
            return self.pinecone.query(
                vector=embedding,
                filter=filters,
                top_k=5
            )
        else:
            # Use platform store for simple search
            return self.openai.responses.create(
                model="gpt-4o",
                input=query,
                tools=[{"type": "file_search", "vector_store_ids": [...]}]
            )
```

---

## Hands-on Exercise

### Your Task

Build a document Q&A system using OpenAI's Vector Stores:

### Requirements

1. Create a vector store for product documentation
2. Upload 3-5 PDF or markdown files
3. Implement a search function that returns citations
4. Create a cleanup function for the vector store

<details>
<summary>💡 Hints</summary>

- Use `file_batches.upload_and_poll` for multiple files
- Citations are in `response.output[0].content[0].annotations`
- Always clean up test vector stores to avoid charges

</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
import os

client = OpenAI()

class DocumentQA:
    def __init__(self, name: str):
        self.vector_store = client.vector_stores.create(name=name)
        print(f"Created vector store: {self.vector_store.id}")
        
    def add_documents(self, file_paths: list[str]):
        """Upload multiple documents to the vector store"""
        file_streams = [open(path, "rb") for path in file_paths]
        
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=self.vector_store.id,
            files=file_streams
        )
        
        # Close file handles
        for f in file_streams:
            f.close()
            
        print(f"Added {batch.file_counts.completed} files")
        return batch
    
    def ask(self, question: str) -> dict:
        """Ask a question and get response with citations"""
        response = client.responses.create(
            model="gpt-4o",
            input=question,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.vector_store.id]
            }]
        )
        
        # Extract text and citations
        content = response.output[0].content[0]
        result = {
            "answer": content.text.value,
            "citations": []
        }
        
        for ann in content.text.annotations:
            result["citations"].append({
                "text": ann.text,
                "file": ann.file_citation.filename if hasattr(ann, 'file_citation') else None
            })
            
        return result
    
    def cleanup(self):
        """Delete the vector store and all files"""
        # Delete files first
        files = client.vector_stores.files.list(
            vector_store_id=self.vector_store.id
        )
        for f in files.data:
            client.files.delete(f.id)
            
        # Delete vector store
        client.vector_stores.delete(vector_store_id=self.vector_store.id)
        print("Cleaned up vector store")

# Usage
qa = DocumentQA("product-docs")
qa.add_documents(["manual.pdf", "faq.md", "specs.txt"])

result = qa.ask("What is the return policy?")
print(result["answer"])
for cite in result["citations"]:
    print(f"  - {cite['file']}: {cite['text'][:50]}...")

qa.cleanup()
```

</details>

---

## Summary

✅ Platform stores (OpenAI, Gemini) simplify RAG with zero infrastructure

✅ Best for prototyping, simple Q&A, and platform-committed projects

✅ Lack metadata filtering and cross-platform portability

✅ Standalone databases offer more control at higher complexity

✅ Hybrid approaches combine platform simplicity with standalone power

**Next:** [Selection Decision Tree](./09-selection-decision-tree.md)

---

## Further Reading

- [OpenAI Vector Stores Guide](https://platform.openai.com/docs/guides/tools-file-search)
- [Gemini File Handling](https://ai.google.dev/gemini-api/docs/files)
- [Gemini Context Caching](https://ai.google.dev/gemini-api/docs/caching)

---

<!-- 
Sources Consulted:
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Vector Stores: https://platform.openai.com/docs/api-reference/vector-stores
- Google Gemini files: https://ai.google.dev/gemini-api/docs/files
-->
