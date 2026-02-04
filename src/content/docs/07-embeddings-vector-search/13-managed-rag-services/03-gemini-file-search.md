---
title: "Gemini File Search"
---

# Gemini File Search

## Introduction

Google's **Gemini File Search** offers a fully managed RAG solution within the Gemini API ecosystem. With `FileSearchStore`, you create searchable collections of documents that Gemini models can query automatically during generation—no external vector database required.

This lesson covers creating file search stores, uploading documents with custom chunking, using metadata filters, and extracting citations from responses.

### What We'll Cover

- Creating and managing FileSearchStores
- Uploading files with chunking configuration
- Using File Search in `generate_content`
- Metadata filtering for targeted retrieval
- Extracting grounding metadata and citations

### Prerequisites

- Google AI API key
- Python environment with `google-genai` package
- Basic understanding of RAG concepts

---

## FileSearchStore Overview

### What is a FileSearchStore?

A **FileSearchStore** is Gemini's managed container for searchable documents:

```
┌─────────────────────────────────────────────────────────────────┐
│              Gemini FileSearchStore                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FileSearchStore                                                │
│  ├── name: "file_search_stores/abc123xyz"                       │
│  ├── display_name: "Product Documentation"                      │
│  ├── state: ACTIVE                                              │
│  └── create_time: 2025-01-15T10:30:00Z                         │
│                                                                 │
│  Files                                                          │
│  ├── user-guide.pdf → Indexed with custom chunking              │
│  ├── api-docs.md → Indexed with metadata                        │
│  └── faq.txt → Indexed with default settings                    │
│                                                                 │
│  Supported Models:                                              │
│  • gemini-3-pro-preview                                         │
│  • gemini-3-flash-preview                                       │
│  • gemini-2.5-pro                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Specifications

| Attribute | Limit |
|-----------|-------|
| Max file size | 100 MB |
| Storage (Free tier) | 1 GB |
| Storage (Tier 1) | 10 GB |
| Storage (Tier 2) | 100 GB |
| Storage (Tier 3) | 1 TB |
| Supported formats | PDF, TXT, MD, HTML, JSON, and more |

### Pricing

| Component | Cost |
|-----------|------|
| Storage | Free |
| Embeddings at indexing | $0.15 per 1M tokens |
| Query embeddings | Free |
| Retrieved tokens | Normal context pricing |

> **Note:** Gemini's pricing model differs significantly from OpenAI—you pay for embedding at index time, not storage.

---

## Setting Up the Client

```python
from google import genai
from google.genai import types

# Initialize client with API key
client = genai.Client(api_key="YOUR_API_KEY")

# Or use environment variable (recommended)
# export GOOGLE_AI_API_KEY="your-key"
client = genai.Client()
```

---

## Creating FileSearchStores

### Basic Creation

```python
# Create a new FileSearchStore
file_search_store = client.file_search_stores.create(
    config={
        'display_name': 'Product Documentation'
    }
)

print(f"Store Name: {file_search_store.name}")
print(f"Display Name: {file_search_store.display_name}")
print(f"State: {file_search_store.state}")
```

**Output:**
```
Store Name: file_search_stores/abc123xyz
Display Name: Product Documentation
State: ACTIVE
```

### Listing Existing Stores

```python
# List all FileSearchStores
stores = client.file_search_stores.list()

for store in stores:
    print(f"{store.display_name}: {store.name}")
    print(f"  State: {store.state}")
```

### Retrieving a Specific Store

```python
# Get store by name
store = client.file_search_stores.get(
    name="file_search_stores/abc123xyz"
)
```

---

## Uploading Files

### Basic File Upload

```python
# Upload a file with default settings
operation = client.file_search_stores.upload_to_file_search_store(
    file='documents/user-guide.pdf',
    file_search_store_name=file_search_store.name,
    config={
        'display_name': 'User Guide v2.0'
    }
)

# Wait for processing to complete
result = operation.result()
print(f"File uploaded: {result.display_name}")
```

### Uploading with Custom Chunking

Gemini supports whitespace-based chunking configuration:

```python
operation = client.file_search_stores.upload_to_file_search_store(
    file='documents/technical-spec.md',
    file_search_store_name=file_search_store.name,
    config={
        'display_name': 'Technical Specification',
        'chunking_config': {
            'white_space_config': {
                'max_tokens_per_chunk': 200,
                'max_overlap_tokens': 20
            }
        }
    }
)

result = operation.result()
```

### Chunking Configuration Options

```
┌─────────────────────────────────────────────────────────────────┐
│              Gemini Chunking Configuration                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  chunking_config:                                               │
│  └── white_space_config:                                        │
│      ├── max_tokens_per_chunk: 200                              │
│      │   • Number of tokens per chunk                           │
│      │   • Smaller = more precise retrieval                     │
│      │   • Larger = more context per chunk                      │
│      │                                                          │
│      └── max_overlap_tokens: 20                                 │
│          • Overlap between consecutive chunks                   │
│          • Prevents context loss at boundaries                  │
│          • Should be ~10% of chunk size                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Uploading with Metadata

```python
# Add custom metadata for filtering
operation = client.file_search_stores.upload_to_file_search_store(
    file='reports/q4-2024.pdf',
    file_search_store_name=file_search_store.name,
    config={
        'display_name': 'Q4 2024 Report',
        'metadata': {
            'quarter': 'Q4',
            'year': '2024',
            'department': 'finance',
            'author': 'John Smith'
        },
        'chunking_config': {
            'white_space_config': {
                'max_tokens_per_chunk': 300,
                'max_overlap_tokens': 30
            }
        }
    }
)
```

---

## Using File Search in Generation

### Basic Usage

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the main features described in the documentation?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name]
                )
            )
        ]
    )
)

print(response.text)
```

### Multiple FileSearchStores

```python
# Search across multiple stores
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Compare the API documentation with the user guide",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[
                        "file_search_stores/api_docs_store",
                        "file_search_stores/user_guide_store"
                    ]
                )
            )
        ]
    )
)
```

---

## Metadata Filtering

### Simple Filter

```python
# Filter by author
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What did John write about performance optimization?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name],
                    metadata_filter="author=John Smith"
                )
            )
        ]
    )
)
```

### Complex Filters

```python
# Multiple filter conditions
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What were the Q4 2024 financial highlights?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name],
                    metadata_filter="quarter=Q4 AND year=2024"
                )
            )
        ]
    )
)
```

### Filter Syntax

| Filter Type | Syntax | Example |
|-------------|--------|---------|
| Equals | `key=value` | `author=John` |
| AND | `key1=val1 AND key2=val2` | `year=2024 AND dept=finance` |
| OR | `key1=val1 OR key2=val2` | `status=draft OR status=review` |

---

## Working with Citations

### Understanding Grounding Metadata

Gemini returns citation information in `grounding_metadata`:

```
┌─────────────────────────────────────────────────────────────────┐
│              Grounding Metadata Structure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  response.candidates[0].grounding_metadata                      │
│  ├── grounding_chunks: [                                        │
│  │   {                                                          │
│  │     chunk: {                                                 │
│  │       text: "The API supports...",                           │
│  │       metadata: { source: "api-docs.md" }                    │
│  │     }                                                        │
│  │   },                                                         │
│  │   ...                                                        │
│  │ ]                                                            │
│  └── grounding_supports: [                                      │
│      {                                                          │
│        segment: { text: "..." },                                │
│        grounding_chunk_indices: [0, 1]                          │
│      }                                                          │
│    ]                                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Extracting Citations

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain the authentication process",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store.name]
                )
            )
        ]
    )
)

# Access grounding metadata
if response.candidates and response.candidates[0].grounding_metadata:
    grounding = response.candidates[0].grounding_metadata
    
    # Get retrieved chunks
    if grounding.grounding_chunks:
        print("Retrieved Chunks:")
        for i, chunk in enumerate(grounding.grounding_chunks):
            print(f"  [{i}] {chunk.chunk.text[:100]}...")
            if chunk.chunk.metadata:
                print(f"      Source: {chunk.chunk.metadata.get('source')}")
    
    # Get support mappings (which response parts use which chunks)
    if grounding.grounding_supports:
        print("\nGrounding Supports:")
        for support in grounding.grounding_supports:
            print(f"  Text: {support.segment.text[:50]}...")
            print(f"  Uses chunks: {support.grounding_chunk_indices}")
```

---

## Managing Files in Stores

### Listing Files

```python
# List files in a FileSearchStore
files = client.file_search_stores.files.list(
    file_search_store_name=file_search_store.name
)

for file in files:
    print(f"File: {file.display_name}")
    print(f"  Name: {file.name}")
    print(f"  State: {file.state}")
```

### Deleting Files

```python
# Remove a file from the store
client.file_search_stores.files.delete(
    name="file_search_stores/abc123/files/file_xyz"
)
```

### Deleting FileSearchStores

```python
# Delete entire store and all files
client.file_search_stores.delete(
    name="file_search_stores/abc123xyz"
)
```

---

## Complete Example: Q&A with Citations

```python
from google import genai
from google.genai import types

client = genai.Client()


def create_knowledge_base(name: str, files: list[dict]) -> str:
    """
    Create a FileSearchStore and upload files.
    
    Args:
        name: Display name for the store
        files: List of dicts with 'path', 'display_name', optional 'metadata'
    
    Returns:
        FileSearchStore name
    """
    # Create store
    store = client.file_search_stores.create(
        config={'display_name': name}
    )
    print(f"Created store: {store.name}")
    
    # Upload files
    for file_info in files:
        config = {
            'display_name': file_info['display_name'],
            'chunking_config': {
                'white_space_config': {
                    'max_tokens_per_chunk': 250,
                    'max_overlap_tokens': 25
                }
            }
        }
        
        if 'metadata' in file_info:
            config['metadata'] = file_info['metadata']
        
        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_info['path'],
            file_search_store_name=store.name,
            config=config
        )
        result = operation.result()
        print(f"  Uploaded: {result.display_name}")
    
    return store.name


def query_with_citations(
    store_name: str,
    question: str,
    metadata_filter: str = None
) -> dict:
    """
    Query the knowledge base and return answer with citations.
    
    Args:
        store_name: FileSearchStore name
        question: User's question
        metadata_filter: Optional filter string
    
    Returns:
        Dict with answer, chunks, and citation info
    """
    # Configure file search
    file_search_config = types.FileSearch(
        file_search_store_names=[store_name]
    )
    
    if metadata_filter:
        file_search_config.metadata_filter = metadata_filter
    
    # Generate response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(file_search=file_search_config)]
        )
    )
    
    # Extract results
    result = {
        'answer': response.text,
        'chunks': [],
        'sources': set()
    }
    
    # Process grounding metadata
    if response.candidates and response.candidates[0].grounding_metadata:
        grounding = response.candidates[0].grounding_metadata
        
        if grounding.grounding_chunks:
            for chunk in grounding.grounding_chunks:
                chunk_info = {
                    'text': chunk.chunk.text,
                    'source': None
                }
                if chunk.chunk.metadata:
                    source = chunk.chunk.metadata.get('source', 'Unknown')
                    chunk_info['source'] = source
                    result['sources'].add(source)
                
                result['chunks'].append(chunk_info)
    
    result['sources'] = list(result['sources'])
    return result


# Usage example
store_name = create_knowledge_base(
    name="Company Docs",
    files=[
        {
            'path': 'docs/user-guide.pdf',
            'display_name': 'User Guide',
            'metadata': {'type': 'guide', 'version': '2.0'}
        },
        {
            'path': 'docs/api-reference.md',
            'display_name': 'API Reference',
            'metadata': {'type': 'reference', 'version': '2.0'}
        },
        {
            'path': 'docs/faq.txt',
            'display_name': 'FAQ',
            'metadata': {'type': 'faq'}
        }
    ]
)

# Query without filter
result = query_with_citations(
    store_name,
    "How do I authenticate with the API?"
)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

# Query with filter
result = query_with_citations(
    store_name,
    "What's covered in the guide?",
    metadata_filter="type=guide"
)
print(f"Filtered Answer: {result['answer']}")
```

---

## Best Practices

| Practice | Recommendation |
|----------|----------------|
| **Chunk size** | Start with 200-300 tokens for precise retrieval |
| **Overlap** | Use ~10% of chunk size to maintain context |
| **Metadata** | Add structured tags for filtering (type, date, author) |
| **Display names** | Use descriptive names for debugging |
| **Store organization** | Separate stores for distinct document collections |

---

## Common Pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Very large chunks | Use smaller chunks (200-300 tokens) for precision |
| No metadata | Add metadata at upload for filtering capability |
| Ignoring operation result | Always call `operation.result()` to ensure completion |
| Single large store | Split into topic-focused stores for better retrieval |

---

## Summary

✅ **FileSearchStore** provides Gemini's managed document storage  
✅ **Custom chunking** via `white_space_config` with token control  
✅ **Metadata filtering** enables targeted searches with simple syntax  
✅ **Grounding metadata** provides citation information automatically  
✅ **Free storage** with pay-per-embedding pricing model

---

**Next:** [OpenAI vs Gemini Comparison →](./04-comparison.md)

---

<!-- 
Sources Consulted:
- Gemini File Search Documentation: https://ai.google.dev/gemini-api/docs/file-search
- Gemini API Reference: https://ai.google.dev/api
-->
