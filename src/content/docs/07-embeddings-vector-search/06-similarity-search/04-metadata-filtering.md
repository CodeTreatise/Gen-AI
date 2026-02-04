---
title: "Metadata Filtering"
---

# Metadata Filtering

## Introduction

Combine vector similarity with attribute filters for precise retrieval. Metadata filtering lets you narrow search results by category, date, source, or any other field.

---

## Pre-filter vs Post-filter

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Pre-filter** | Filter metadata first, then search vectors | High-selectivity filters (few matches) |
| **Post-filter** | Search vectors first, then filter results | Low-selectivity filters (many matches) |

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def search_with_prefilter(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    filters: dict,
    limit: int = 10
) -> list[dict]:
    """Vector search with pre-filtering (database handles both)."""
    
    # Build Qdrant filter from dict
    conditions = []
    
    for field, value in filters.items():
        if isinstance(value, dict):
            # Range filter
            conditions.append(
                FieldCondition(
                    key=field,
                    range=Range(**value)
                )
            )
        else:
            # Exact match
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
    
    query_filter = Filter(must=conditions) if conditions else None
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit
    )
    
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in results]

# Usage examples
# Filter by exact category
results = search_with_prefilter(
    client, "documents", query_vector,
    filters={"category": "technical", "language": "en"}
)

# Filter by date range
results = search_with_prefilter(
    client, "documents", query_vector,
    filters={"created_at": {"gte": 1704067200, "lte": 1735689600}}  # 2024
)
```

---

## Complex Filter Logic

Combine AND, OR, and NOT conditions:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

def build_complex_filter(
    must: list[dict] = None,
    should: list[dict] = None,
    must_not: list[dict] = None
) -> Filter:
    """Build complex Qdrant filter with boolean logic.
    
    Args:
        must: All conditions must match (AND)
        should: At least one condition must match (OR)
        must_not: None of these conditions can match (NOT)
    """
    
    def dict_to_condition(d: dict) -> FieldCondition:
        field = d["field"]
        value = d["value"]
        
        if isinstance(value, list):
            return FieldCondition(key=field, match=MatchAny(any=value))
        return FieldCondition(key=field, match=MatchValue(value=value))
    
    return Filter(
        must=[dict_to_condition(c) for c in (must or [])],
        should=[dict_to_condition(c) for c in (should or [])],
        must_not=[dict_to_condition(c) for c in (must_not or [])]
    )

# Example: Find Python or JavaScript docs, not archived, in tutorial category
complex_filter = build_complex_filter(
    must=[
        {"field": "category", "value": "tutorial"}
    ],
    should=[
        {"field": "language", "value": ["python", "javascript"]}
    ],
    must_not=[
        {"field": "status", "value": "archived"}
    ]
)

results = client.search(
    collection_name="documents",
    query_vector=query_vector,
    query_filter=complex_filter,
    limit=10
)
```

---

## Index Design for Filtered Queries

Create payload indexes for fields you filter on frequently:

```python
from qdrant_client.models import PayloadSchemaType, VectorParams, Distance

def create_optimized_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1536
):
    """Create collection with payload indexes for common filters."""
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    
    # Add payload indexes for filterable fields
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="created_at",
        field_schema=PayloadSchemaType.INTEGER
    )
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    print(f"Created {collection_name} with optimized payload indexes")
```

> **Warning:** Creating too many payload indexes increases storage and slows down upserts. Only index fields you actually filter on.

---

## Common Filter Patterns

### Date Range Filtering

```python
from datetime import datetime, timedelta

def search_recent_documents(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    days_back: int = 30
) -> list[dict]:
    """Search documents from the last N days."""
    
    cutoff = datetime.now() - timedelta(days=days_back)
    cutoff_timestamp = int(cutoff.timestamp())
    
    filter = Filter(
        must=[
            FieldCondition(
                key="created_at",
                range=Range(gte=cutoff_timestamp)
            )
        ]
    )
    
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=filter,
        limit=10
    )
```

### Multi-Category Filtering

```python
def search_in_categories(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    categories: list[str]
) -> list[dict]:
    """Search within specified categories."""
    
    filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchAny(any=categories)
            )
        ]
    )
    
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=filter,
        limit=10
    )
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Create indexes for frequently filtered fields | Index every field |
| Use pre-filtering for selective queries | Post-filter thousands of results |
| Combine filters with vector search | Filter in application code |
| Test filter performance with realistic data | Assume filters are free |

---

## Summary

✅ **Pre-filtering** is handled efficiently by vector databases

✅ **Complex boolean logic** (AND/OR/NOT) supports sophisticated queries

✅ **Payload indexes** dramatically improve filter performance

✅ **Only index fields you filter on** to avoid overhead

**Next:** [Hybrid Search](./05-hybrid-search.md)
