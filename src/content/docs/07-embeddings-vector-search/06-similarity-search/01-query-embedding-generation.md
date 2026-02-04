---
title: "Query Embedding Generation"
---

# Query Embedding Generation

## Introduction

The most critical rule in similarity search: **query embeddings must use the same model as document embeddings**. Mixing models produces meaningless similarity scores because each model creates its own semantic space.

---

## Using the Correct Task Type

Modern embedding models support different task types that optimize the embedding for its purpose:

```python
from openai import OpenAI
import voyageai

client = OpenAI()
voyage = voyageai.Client()

# OpenAI doesn't require task type (same model for query and doc)
def get_openai_query_embedding(query: str) -> list[float]:
    """Generate query embedding using OpenAI."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Voyage uses different input types
def get_voyage_query_embedding(query: str) -> list[float]:
    """Generate query embedding using Voyage with query task type."""
    result = voyage.embed(
        texts=[query],
        model="voyage-3",
        input_type="query"  # Optimized for search queries
    )
    return result.embeddings[0]

def get_voyage_document_embedding(document: str) -> list[float]:
    """Generate document embedding using Voyage."""
    result = voyage.embed(
        texts=[document],
        model="voyage-3",
        input_type="document"  # Optimized for documents
    )
    return result.embeddings[0]
```

**Output:**
```
Query embedding dimension: 1536
Document embedding dimension: 1536
# Dimensions match, models match - ready for similarity search
```

> **🤖 AI Context:** The query/document distinction matters because queries are typically short and question-focused, while documents are longer and contain answers. Some models train asymmetrically to optimize this relationship.

---

## Query Preprocessing

Raw user queries often benefit from preprocessing before embedding:

```python
import re
from typing import Optional

class QueryPreprocessor:
    """Clean and normalize search queries."""
    
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
    
    def clean(self, query: str) -> str:
        """Basic query cleaning."""
        # Lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove special characters but keep meaningful punctuation
        query = re.sub(r'[^\w\s\?\!\.\-]', '', query)
        
        return query
    
    def expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nlp': 'natural language processing',
            'llm': 'large language model',
            'api': 'application programming interface',
            'db': 'database',
        }
        
        words = query.lower().split()
        expanded = [abbreviations.get(word, word) for word in words]
        return ' '.join(expanded)
    
    def process(self, query: str, expand_abbrev: bool = True) -> str:
        """Full preprocessing pipeline."""
        query = self.clean(query)
        if expand_abbrev:
            query = self.expand_abbreviations(query)
        return query

# Usage
preprocessor = QueryPreprocessor()
raw_query = "  How do I use ML   for NLP??  "
clean_query = preprocessor.process(raw_query)
print(f"Original: '{raw_query}'")
print(f"Cleaned: '{clean_query}'")
```

**Output:**
```
Original: '  How do I use ML   for NLP??  '
Cleaned: 'how do i use machine learning for natural language processing?'
```

---

## Query Expansion with LLM

For complex queries, use an LLM to generate additional search terms:

```python
from openai import OpenAI

def expand_query_with_llm(query: str) -> list[str]:
    """Expand query using LLM to generate related search terms."""
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Generate 3 alternative phrasings of the search query.
                Return only the alternatives, one per line, no numbering."""
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    alternatives = response.choices[0].message.content.strip().split('\n')
    return [query] + [alt.strip() for alt in alternatives if alt.strip()]

# Example
original = "How to handle errors in Python?"
expanded = expand_query_with_llm(original)
for q in expanded:
    print(f"  - {q}")
```

**Output:**
```
  - How to handle errors in Python?
  - Python exception handling best practices
  - Try except blocks in Python programming
  - Error management techniques for Python applications
```

---

## Caching Query Embeddings

Frequent queries should be cached to reduce latency and costs:

```python
import hashlib
from functools import lru_cache
from typing import Tuple
import redis
import json

class EmbeddingCache:
    """Cache embeddings in Redis for production use."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 86400  # 24 hours
    
    def _cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return f"emb:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def get(self, text: str, model: str) -> list[float] | None:
        """Retrieve cached embedding."""
        key = self._cache_key(text, model)
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        key = self._cache_key(text, model)
        self.redis.setex(key, self.ttl, json.dumps(embedding))
    
    def get_or_create(
        self, 
        text: str, 
        model: str, 
        embed_fn
    ) -> list[float]:
        """Get from cache or create and cache."""
        embedding = self.get(text, model)
        if embedding is None:
            embedding = embed_fn(text)
            self.set(text, model, embedding)
        return embedding

# For simpler use cases, use lru_cache
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str, model: str = "text-embedding-3-small") -> Tuple[float, ...]:
    """In-memory cached embedding (returns tuple for hashability)."""
    client = OpenAI()
    response = client.embeddings.create(input=text, model=model)
    return tuple(response.data[0].embedding)
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use same model for queries and documents | Mix embedding models |
| Apply correct task type (query vs document) | Ignore task type settings |
| Cache frequent query embeddings | Regenerate embeddings each time |
| Preprocess queries consistently | Apply different preprocessing to queries vs docs |

---

## Summary

✅ **Query and document embeddings must use the same model**

✅ **Use appropriate task types** (query vs document) when supported

✅ **Preprocess queries** to expand abbreviations and clean input

✅ **Cache embeddings** to reduce latency and API costs

**Next:** [Top-K Retrieval](./02-top-k-retrieval.md)
