---
title: "OpenAI Vector Stores & File Search"
---

# OpenAI Vector Stores & File Search

## Introduction

OpenAI's **Vector Stores API** and **File Search** tool provide a complete managed RAG solution. You upload documents, OpenAI handles everything else—parsing, chunking, embedding, and search. When you query, the model automatically retrieves relevant content and generates responses with citations.

This lesson covers the complete workflow: creating vector stores, uploading files, querying with file search, and working with citations.

### What We'll Cover

- Creating and managing vector stores
- Uploading files with chunking configuration
- Using the File Search tool in the Responses API
- Working with citations and annotations
- Metadata filtering and ranking options

### Prerequisites

- OpenAI API key
- Basic understanding of RAG concepts
- Python environment with `openai` package installed

---

## Vector Stores Overview

### What is a Vector Store?

A **vector store** is OpenAI's managed container for your searchable documents:

```
┌─────────────────────────────────────────────────────────────────┐
│              OpenAI Vector Store                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Vector Store (vs_abc123)                                       │
│  ├── name: "Product Documentation"                              │
│  ├── status: completed                                          │
│  ├── file_counts:                                               │
│  │   ├── in_progress: 0                                         │
│  │   ├── completed: 5                                           │
│  │   └── failed: 0                                              │
│  └── usage_bytes: 123456                                        │
│                                                                 │
│  Files                                                          │
│  ├── api-reference.pdf → 45 chunks                              │
│  ├── user-guide.docx → 32 chunks                                │
│  ├── faq.md → 18 chunks                                         │
│  ├── release-notes.txt → 12 chunks                              │
│  └── troubleshooting.html → 27 chunks                           │
│                                                                 │
│  Total Chunks: 134                                              │
│  Embedding Model: text-embedding-3-large (256 dimensions)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Specifications

| Attribute | Limit |
|-----------|-------|
| Max file size | 512 MB |
| Max tokens per file | 5,000,000 |
| Supported formats | PDF, DOCX, TXT, HTML, MD, JSON, and more |
| Default chunk size | 800 tokens |
| Default chunk overlap | 400 tokens |
| Embedding model | text-embedding-3-large (256 dims) |

### Pricing

| Component | Cost |
|-----------|------|
| Storage | First 1 GB free, then $0.10/GB/day |
| Search | Included in token usage |

---

## Creating Vector Stores

### Basic Creation

```python
from openai import OpenAI

client = OpenAI()

# Create an empty vector store
vector_store = client.vector_stores.create(
    name="Company Knowledge Base"
)

print(f"Vector Store ID: {vector_store.id}")
print(f"Status: {vector_store.status}")
```

**Output:**
```
Vector Store ID: vs_abc123xyz
Status: completed
```

### Creation with Metadata

```python
# Create with metadata for organization
vector_store = client.vector_stores.create(
    name="Product Documentation v2.0",
    metadata={
        "department": "engineering",
        "version": "2.0",
        "owner": "docs-team"
    }
)
```

### Creation with Files

```python
# Upload files to OpenAI first
file1 = client.files.create(
    file=open("user-guide.pdf", "rb"),
    purpose="assistants"
)
file2 = client.files.create(
    file=open("api-reference.md", "rb"),
    purpose="assistants"
)

# Create vector store with files
vector_store = client.vector_stores.create(
    name="Documentation",
    file_ids=[file1.id, file2.id]
)
```

---

## Chunking Strategies

### Auto Chunking (Default)

```python
# OpenAI determines optimal chunking
vector_store = client.vector_stores.create(
    name="Auto-Chunked Store",
    chunking_strategy={"type": "auto"}
)
```

### Static Chunking (Custom)

```python
# Define your own chunk parameters
vector_store = client.vector_stores.create(
    name="Custom-Chunked Store",
    chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 1000,
            "chunk_overlap_tokens": 200
        }
    }
)
```

### Choosing Chunk Size

| Chunk Size | Best For |
|------------|----------|
| 400-600 tokens | Short, focused answers |
| 800 tokens (default) | General purpose |
| 1000-2000 tokens | Complex, contextual content |

> **Note:** Larger chunks provide more context but may include irrelevant information. Smaller chunks are more precise but may miss context.

---

## Uploading Files

### Single File Upload

```python
# Step 1: Upload to OpenAI Files
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants"
)

# Step 2: Add to vector store
vector_store_file = client.vector_stores.files.create(
    vector_store_id="vs_abc123",
    file_id=file.id
)

print(f"File Status: {vector_store_file.status}")
```

### Batch Upload (Recommended)

```python
# Upload multiple files efficiently
file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id="vs_abc123",
    files=[
        open("doc1.pdf", "rb"),
        open("doc2.docx", "rb"),
        open("doc3.md", "rb")
    ]
)

print(f"Batch Status: {file_batch.status}")
print(f"Files Processed: {file_batch.file_counts.completed}")
```

### Monitoring Upload Progress

```python
import time

# Start upload without waiting
file_batch = client.vector_stores.file_batches.create(
    vector_store_id="vs_abc123",
    file_ids=["file_1", "file_2", "file_3"]
)

# Poll for completion
while file_batch.status in ["in_progress", "queued"]:
    print(f"Status: {file_batch.status}")
    print(f"  Completed: {file_batch.file_counts.completed}")
    print(f"  In Progress: {file_batch.file_counts.in_progress}")
    time.sleep(2)
    file_batch = client.vector_stores.file_batches.retrieve(
        vector_store_id="vs_abc123",
        batch_id=file_batch.id
    )

print(f"Final Status: {file_batch.status}")
```

---

## File Search in Responses API

### Basic File Search

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What are the key features of the product?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }]
)

print(response.output_text)
```

### Controlling Search Results

```python
response = client.responses.create(
    model="gpt-4.1",
    input="Explain the authentication process in detail",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"],
        "max_num_results": 10  # Default: 20 for gpt-4*, 5 for gpt-3.5
    }]
)
```

### Including Raw Search Results

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What does the documentation say about rate limits?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }],
    include=["file_search_call.results"]  # Include retrieved chunks
)

# Access the retrieved chunks
for output in response.output:
    if output.type == "file_search_call":
        print("Retrieved Chunks:")
        for result in output.results:
            print(f"  Score: {result.score:.3f}")
            print(f"  File: {result.file_id}")
            print(f"  Content: {result.content[:100]}...")
```

---

## Working with Citations

### Understanding Annotations

File Search automatically adds citations to responses:

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What are the pricing tiers?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"]
    }]
)

# Access the message output
for output in response.output:
    if output.type == "message":
        # Check for annotations
        for content in output.content:
            if hasattr(content, 'annotations'):
                for annotation in content.annotations:
                    if annotation.type == "file_citation":
                        print(f"Citation: {annotation.file_citation.file_id}")
                        print(f"Quote: {annotation.file_citation.quote}")
```

### Citation Structure

```
┌─────────────────────────────────────────────────────────────────┐
│              Response with Citations                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Response Text:                                                 │
│  "The API supports three pricing tiers: Free, Pro, and         │
│   Enterprise [1]. The Free tier includes 1000 requests         │
│   per month [1], while Pro offers unlimited requests [2]."     │
│                                                                 │
│  Annotations:                                                   │
│  ├── [1] file_citation                                          │
│  │   ├── file_id: "file_abc123"                                │
│  │   └── quote: "Three pricing tiers are available..."         │
│  └── [2] file_citation                                          │
│       ├── file_id: "file_def456"                               │
│       └── quote: "Pro tier: Unlimited API requests..."         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metadata Filtering

### Adding Metadata to Files

```python
# Upload file with metadata
file = client.files.create(
    file=open("quarterly-report-q4.pdf", "rb"),
    purpose="assistants"
)

# Add to vector store with metadata
vector_store_file = client.vector_stores.files.create(
    vector_store_id="vs_abc123",
    file_id=file.id,
    attributes={
        "quarter": "Q4",
        "year": 2024,
        "department": "finance",
        "category": "report"
    }
)
```

### Filtering by Metadata

```python
# Search only specific files
response = client.responses.create(
    model="gpt-4.1",
    input="What were the Q4 2024 revenue numbers?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"],
        "filters": {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "quarter", "value": "Q4"},
                {"type": "eq", "key": "year", "value": 2024}
            ]
        }
    }]
)
```

### Filter Operators

| Operator | Usage | Example |
|----------|-------|---------|
| `eq` | Equals | `{"type": "eq", "key": "status", "value": "active"}` |
| `ne` | Not equals | `{"type": "ne", "key": "draft", "value": true}` |
| `gt` | Greater than | `{"type": "gt", "key": "year", "value": 2020}` |
| `gte` | Greater or equal | `{"type": "gte", "key": "version", "value": 2}` |
| `lt` | Less than | `{"type": "lt", "key": "priority", "value": 5}` |
| `lte` | Less or equal | `{"type": "lte", "key": "age", "value": 30}` |
| `in` | In list | `{"type": "in", "key": "category", "value": ["docs", "guides"]}` |
| `and` | All match | `{"type": "and", "filters": [...]}` |
| `or` | Any match | `{"type": "or", "filters": [...]}` |

---

## Ranking and Hybrid Search

### Ranking Options

```python
response = client.responses.create(
    model="gpt-4.1",
    input="Explain the authentication flow",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_abc123"],
        "ranking_options": {
            "ranker": "auto",  # or "default-2024-11-15"
            "score_threshold": 0.5  # Filter low-relevance results
        }
    }]
)
```

### How Hybrid Search Works

OpenAI's file search uses **hybrid search** combining:

```
┌─────────────────────────────────────────────────────────────────┐
│              Hybrid Search Flow                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query: "How do I authenticate with OAuth?"                │
│                      │                                          │
│         ┌────────────┴────────────┐                            │
│         │                         │                             │
│         ▼                         ▼                             │
│  ┌─────────────┐           ┌─────────────┐                     │
│  │  Semantic   │           │   Keyword   │                     │
│  │   Search    │           │   Search    │                     │
│  │             │           │             │                     │
│  │ Embedding   │           │ BM25/TF-IDF │                     │
│  │ similarity  │           │ matching    │                     │
│  └─────────────┘           └─────────────┘                     │
│         │                         │                             │
│         └────────────┬────────────┘                            │
│                      ▼                                          │
│              ┌─────────────┐                                   │
│              │  Combine &  │                                   │
│              │   Rerank    │                                   │
│              └─────────────┘                                   │
│                      │                                          │
│                      ▼                                          │
│              Top N Results with Scores                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Managing Vector Stores

### Listing Vector Stores

```python
# List all vector stores
vector_stores = client.vector_stores.list()

for vs in vector_stores.data:
    print(f"{vs.name} ({vs.id})")
    print(f"  Files: {vs.file_counts.completed}")
    print(f"  Size: {vs.usage_bytes} bytes")
```

### Updating a Vector Store

```python
# Update name and metadata
updated = client.vector_stores.update(
    vector_store_id="vs_abc123",
    name="Updated Knowledge Base",
    metadata={"version": "2.0"}
)
```

### Deleting a Vector Store

```python
# Delete vector store (also removes files from it)
client.vector_stores.delete(vector_store_id="vs_abc123")
```

> **Warning:** Deleting a vector store removes all indexed content but does NOT delete the underlying files from OpenAI's file storage.

### Removing Files from Vector Store

```python
# Remove specific file (keeps file in OpenAI storage)
client.vector_stores.files.delete(
    vector_store_id="vs_abc123",
    file_id="file_xyz789"
)
```

---

## Complete Example: Document Q&A

```python
from openai import OpenAI

client = OpenAI()

def create_knowledge_base(name: str, files: list[str]) -> str:
    """Create a vector store and upload files."""
    
    # Create vector store
    vector_store = client.vector_stores.create(
        name=name,
        chunking_strategy={
            "type": "static",
            "static": {
                "max_chunk_size_tokens": 800,
                "chunk_overlap_tokens": 200
            }
        }
    )
    
    # Upload files
    file_streams = [open(f, "rb") for f in files]
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams
    )
    
    # Close file handles
    for f in file_streams:
        f.close()
    
    print(f"Created: {vector_store.id}")
    print(f"Files: {file_batch.file_counts.completed}")
    
    return vector_store.id


def ask_question(vector_store_id: str, question: str) -> dict:
    """Query the knowledge base and return answer with citations."""
    
    response = client.responses.create(
        model="gpt-4.1",
        input=question,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": 5
        }],
        include=["file_search_call.results"]
    )
    
    # Extract answer and citations
    result = {
        "answer": response.output_text,
        "citations": [],
        "chunks_used": []
    }
    
    for output in response.output:
        if output.type == "file_search_call":
            for chunk in output.results:
                result["chunks_used"].append({
                    "score": chunk.score,
                    "content": chunk.content[:200]
                })
        
        if output.type == "message":
            for content in output.content:
                if hasattr(content, 'annotations'):
                    for ann in content.annotations:
                        if ann.type == "file_citation":
                            result["citations"].append({
                                "file_id": ann.file_citation.file_id,
                                "quote": ann.file_citation.quote
                            })
    
    return result


# Usage
vs_id = create_knowledge_base(
    name="Product Docs",
    files=["user-guide.pdf", "api-reference.md", "faq.txt"]
)

answer = ask_question(vs_id, "How do I reset my password?")
print(f"Answer: {answer['answer']}")
print(f"Citations: {len(answer['citations'])}")
```

---

## Best Practices

| Practice | Recommendation |
|----------|----------------|
| **File organization** | One topic per file for better retrieval |
| **Chunk size** | Start with default (800), adjust based on results |
| **Metadata** | Add tags for filtering (category, date, version) |
| **Naming** | Use descriptive names for debugging |
| **Cleanup** | Delete unused vector stores to manage costs |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Uploading very large files | Split into smaller, focused documents |
| Not checking file status | Use `upload_and_poll()` to wait for completion |
| Ignoring chunk overlap | Overlap prevents context loss at boundaries |
| Hard-coded file IDs | Store IDs in database for production |

---

## Summary

✅ **Vector Stores** provide managed document storage with automatic chunking and embedding  
✅ **File Search** integrates retrieval directly into the Responses API  
✅ **Citations** are automatically generated with source quotes  
✅ **Metadata filtering** enables targeted searches across document subsets  
✅ **Hybrid search** combines semantic and keyword matching for better results

---

**Next:** [Gemini File Search →](./03-gemini-file-search.md)

---

<!-- 
Sources Consulted:
- OpenAI Vector Stores API Reference: https://platform.openai.com/docs/api-reference/vector-stores
- OpenAI File Search Tool: https://platform.openai.com/docs/guides/tools-file-search
- OpenAI Assistants File Search: https://platform.openai.com/docs/assistants/tools/file-search
-->
