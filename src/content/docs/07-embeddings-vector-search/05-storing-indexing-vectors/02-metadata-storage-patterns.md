---
title: "Metadata Storage Patterns"
---

# Metadata Storage Patterns

## Introduction

Metadata enables filtering, context retrieval, and result enrichment. Well-designed metadata schemas make the difference between a search system that works and one that excels.

---

## Schema Design Patterns

### Pattern 1: Flat Metadata

Simple key-value pairs optimized for fast filtering:

```python
metadata_flat = {
    "title": "Introduction to ML",
    "category": "AI",
    "author": "John Doe",
    "created_at": "2024-01-15",
    "word_count": 1500,
    "is_public": True
}
```

**Best for:** Simple applications with straightforward filtering needs.

### Pattern 2: Nested Metadata

Rich context with hierarchical structure:

```python
metadata_nested = {
    "document": {
        "title": "Introduction to ML",
        "url": "https://example.com/ml-intro"
    },
    "chunk": {
        "index": 5,
        "start_char": 2500,
        "end_char": 3200
    },
    "source": {
        "type": "confluence",
        "space": "Engineering",
        "last_synced": "2024-01-20T10:30:00Z"
    }
}
```

**Best for:** Complex documents requiring full provenance tracking.

### Pattern 3: Minimal Metadata

Cost-optimized with external lookups:

```python
metadata_minimal = {
    "doc_id": "doc123",  # Lookup full details elsewhere
    "chunk_idx": 5
}
```

**Best for:** Large-scale systems where metadata storage costs matter.

---

## Filterable Field Design

| Field Type | Filter Operations | Index Impact |
|------------|-------------------|--------------|
| **String/Tag** | Equals, in list | Low memory |
| **Numeric** | Range, comparison | Low memory |
| **Boolean** | Equals | Very low |
| **Array** | Contains, any match | Medium memory |
| **Text** | Full-text search | High memory |
| **Geo** | Radius, bounding box | Medium memory |

### Recommended Schema

```python
# Design metadata for common filter patterns
metadata = {
    # Fast equality filters
    "category": "technology",      # String tag
    "author_id": "user_456",       # Foreign key
    "status": "published",         # Enum-like
    
    # Range filters
    "created_at": 1705312800,      # Unix timestamp
    "relevance_score": 0.95,       # Numeric
    "word_count": 1500,            # Integer
    
    # Boolean filters
    "is_public": True,
    "has_code": True,
    
    # Array filters (check with vector DB support)
    "tags": ["python", "machine-learning", "tutorial"],
    
    # Context for retrieval (not filtered)
    "text": "The original chunk text for display...",
    "url": "https://..."
}
```

> **Warning:** Not all vector databases support all field types. Check your database's documentation for supported filter operations.

---

## Text Content Storage

Where should you store the original text?

| Strategy | Pros | Cons |
|----------|------|------|
| **Inline in metadata** | Simple, single query | Size limits (40KB typical) |
| **Separate collection** | No size limits | Extra lookup required |
| **External store** | Unlimited, existing infra | Latency, consistency |

### Strategy 1: Inline (Recommended for chunks < 10KB)

```python
record = {
    "id": "chunk_123",
    "vector": [...],
    "metadata": {
        "text": "The actual chunk content...",  # Inline
        "doc_id": "doc_456"
    }
}
```

### Strategy 2: Reference (For large documents)

```python
# Vector DB - minimal metadata
vector_record = {
    "id": "chunk_123",
    "vector": [...],
    "metadata": {
        "doc_id": "doc_456",
        "chunk_idx": 5,
        "char_range": [2500, 3200]
    }
}

# Separate document store
def get_chunk_text(doc_id: str, char_start: int, char_end: int) -> str:
    doc = document_store.get(doc_id)
    return doc["content"][char_start:char_end]
```

### Strategy 3: Hybrid Approach

```python
class HybridContentStore:
    """Store small chunks inline, large chunks externally."""
    
    def __init__(self, max_inline_size: int = 8000):
        self.max_inline_size = max_inline_size
        self.external_store = {}
    
    def create_metadata(self, text: str, doc_id: str) -> dict:
        if len(text.encode('utf-8')) <= self.max_inline_size:
            return {
                "text": text,
                "doc_id": doc_id,
                "storage": "inline"
            }
        else:
            # Store externally
            text_id = f"{doc_id}:text"
            self.external_store[text_id] = text
            return {
                "text_ref": text_id,
                "doc_id": doc_id,
                "storage": "external"
            }
    
    def get_text(self, metadata: dict) -> str:
        if metadata["storage"] == "inline":
            return metadata["text"]
        return self.external_store[metadata["text_ref"]]
```

---

## Metadata for RAG Applications

For RAG (Retrieval-Augmented Generation), include context that helps the LLM:

```python
def create_rag_metadata(
    text: str,
    title: str,
    url: str,
    source_type: str,
    chunk_index: int,
    total_chunks: int,
    section_header: str = None
) -> dict:
    """Create metadata optimized for RAG applications."""
    return {
        # For display and context
        "text": text,
        "title": title,
        "url": url,
        "section": section_header,
        
        # For filtering
        "source_type": source_type,  # "docs", "wiki", "code"
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        
        # For context assembly
        "has_prev": chunk_index > 0,
        "has_next": chunk_index < total_chunks - 1,
        
        # For freshness
        "indexed_at": int(time.time()),
    }
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Store chunk text inline if < 10KB | Store full documents in vector metadata |
| Include source tracking metadata | Forget provenance information |
| Use consistent field naming | Mix naming conventions |
| Index only filterable fields | Create indexes for every field |

---

## Summary

✅ **Flat schemas** enable fast filtering with simple queries

✅ **Nested schemas** provide rich context for complex applications

✅ **Inline text** works for chunks under 10KB

✅ **External storage** handles larger content with extra lookups

**Next:** [Index Types](./03-index-types.md)
