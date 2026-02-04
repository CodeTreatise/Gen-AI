---
title: "Hypothetical Document Embeddings (HyDE)"
---

# Hypothetical Document Embeddings (HyDE)

## Introduction

HyDE solves a fundamental RAG problem: queries and documents are written differently. A question like "What causes rust?" doesn't semantically match a document stating "Iron oxide forms when iron reacts with oxygen and water." HyDE bridges this gap by generating a hypothetical answer, then searching with that answer's embedding.

> **🤖 AI Context:** HyDE was introduced in the paper "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022) and shows strong performance comparable to fine-tuned retrievers without any training data.

---

## The Query-Document Mismatch Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL SEARCH                           │
├─────────────────────────────────────────────────────────────────┤
│  Query: "What causes rust?"                                     │
│  Query Embedding: [0.2, 0.4, 0.1, ...]                         │
│                                                                 │
│  Documents in corpus:                                           │
│  • "Iron oxide forms when iron reacts with oxygen and water"   │
│    Embedding: [0.5, 0.1, 0.8, ...]                             │
│                                                                 │
│  ❌ Low similarity because question ≠ statement style           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    HyDE SEARCH                                  │
├─────────────────────────────────────────────────────────────────┤
│  Query: "What causes rust?"                                     │
│           ↓ LLM generates hypothetical answer                   │
│  "Rust is caused by the oxidation of iron when exposed to      │
│   oxygen and moisture. Iron oxide, commonly known as rust..."  │
│           ↓ Embed the hypothetical answer                       │
│  HyDE Embedding: [0.5, 0.2, 0.7, ...]                          │
│                                                                 │
│  ✅ High similarity to actual documents about oxidation!        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Basic HyDE Implementation

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical document that would answer the query."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert assistant. Given a question, 
write a short paragraph that would appear in a document answering this question.
Write in a factual, encyclopedic style. Do not say "I" or reference the question.
Just write the answer as if it were from a textbook or Wikipedia."""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def hyde_embed(query: str) -> list[float]:
    """Generate HyDE embedding for a query."""
    
    # Step 1: Generate hypothetical answer
    hypothetical_doc = generate_hypothetical_answer(query)
    
    # Step 2: Embed the hypothetical answer (not the query!)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=hypothetical_doc
    )
    
    return response.data[0].embedding

def hyde_search(
    query: str,
    index,  # Your vector index
    top_k: int = 10
) -> list[dict]:
    """Search using HyDE embedding."""
    
    hyde_embedding = hyde_embed(query)
    results = index.search(hyde_embedding, top_k=top_k)
    
    return results
```

---

## HyDE with Multiple Hypothetical Documents

Generate multiple hypothetical answers and average their embeddings:

```python
def hyde_multi_embed(
    query: str,
    num_hypotheticals: int = 3
) -> list[float]:
    """Generate HyDE embedding from multiple hypothetical answers."""
    
    hypothetical_docs = []
    
    for i in range(num_hypotheticals):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Write a paragraph that answers the following question.
Use variation {i+1} of {num_hypotheticals} - vary your vocabulary and phrasing.
Write factually, as if from a reference document."""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.9  # Higher temp for variation
        )
        hypothetical_docs.append(response.choices[0].message.content)
    
    # Embed all hypothetical documents
    embeddings_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=hypothetical_docs
    )
    
    # Average the embeddings
    embeddings = [e.embedding for e in embeddings_response.data]
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    
    return avg_embedding
```

---

## LlamaIndex HyDE Integration

LlamaIndex provides built-in HyDE support:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# Load documents and create index
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create base query engine
base_engine = index.as_query_engine()

# Wrap with HyDE transformation
hyde_transform = HyDEQueryTransform(include_original=True)
hyde_engine = TransformQueryEngine(
    query_engine=base_engine,
    query_transform=hyde_transform
)

# Query with HyDE
response = hyde_engine.query("What causes rust?")
print(response)
```

---

## When HyDE Helps

| Scenario | HyDE Benefit |
|----------|-------------|
| Questions → Factual docs | High (question ≠ statement) |
| Technical queries | High (jargon bridging) |
| Abstract concepts | Medium (adds context) |
| Cross-lingual | Medium (translation-like) |

**Best use cases:**
- FAQ systems (questions → answers)
- Research paper search
- Technical documentation
- Educational content retrieval

---

## When HyDE Hurts

| Scenario | Problem |
|----------|---------|
| Keyword-heavy queries | LLM may hallucinate different terms |
| Time-sensitive data | LLM knowledge cutoff issues |
| Exact phrase matching | Hypothetical won't contain exact phrases |
| Very short documents | Hypothetical may be longer than corpus docs |
| Named entity queries | "Who is John Smith?" → wrong John Smith |

```python
def should_use_hyde(query: str) -> bool:
    """Heuristic to decide if HyDE will help."""
    
    # Skip HyDE for these patterns
    skip_patterns = [
        r'^who is ',           # Named entity lookups
        r'^what is the (exact|specific)',  # Exact matching
        r'\d{4}',              # Year/date specific
        r'"[^"]+"',            # Quoted phrases
        r'(code|function|api)\s+for',  # Code lookups
    ]
    
    import re
    for pattern in skip_patterns:
        if re.search(pattern, query.lower()):
            return False
    
    # Use HyDE for question-style queries
    question_patterns = [
        r'^(what|why|how|when|where|explain|describe)',
        r'\?$',
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, query.lower()):
            return True
    
    return False  # Default to standard search
```

---

## Hybrid Approach: HyDE + Standard

Combine HyDE and standard embeddings:

```python
def hybrid_hyde_search(
    query: str,
    index,
    hyde_weight: float = 0.6,
    top_k: int = 10
) -> list[dict]:
    """Combine HyDE and standard embeddings."""
    
    # Standard query embedding
    standard_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    standard_emb = np.array(standard_response.data[0].embedding)
    
    # HyDE embedding
    hyde_emb = np.array(hyde_embed(query))
    
    # Weighted combination
    combined_emb = (
        hyde_weight * hyde_emb +
        (1 - hyde_weight) * standard_emb
    )
    
    # Normalize
    combined_emb = combined_emb / np.linalg.norm(combined_emb)
    
    return index.search(combined_emb.tolist(), top_k=top_k)
```

---

## Cost Considerations

| Approach | LLM Calls | Embedding Calls | Latency |
|----------|-----------|-----------------|---------|
| Standard | 0 | 1 | ~50ms |
| HyDE | 1 | 1 | ~500ms |
| Multi-HyDE (3) | 3 | 1 (batched) | ~1000ms |
| Hybrid | 1 | 2 | ~550ms |

**Cost optimization:**
```python
# Cache HyDE results for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_hyde_embed(query: str) -> tuple:
    """Cache HyDE embeddings for repeated queries."""
    embedding = hyde_embed(query)
    return tuple(embedding)  # Tuples are hashable
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Use for question → document search | Use for keyword lookups |
| Cache hypothetical embeddings | Generate for every query |
| Combine with standard search | Use as only retrieval method |
| Tune prompt for your domain | Use generic prompts |
| Test on your specific corpus | Assume universal improvement |

---

## Summary

✅ **HyDE** generates hypothetical answers to bridge query-document gap

✅ **Best for** questions searching factual/encyclopedic content

✅ **Avoid for** named entities, exact phrases, time-sensitive data

✅ **Multi-HyDE** averages multiple hypotheticals for robustness

✅ **Hybrid approach** combines HyDE with standard embeddings

**Next:** [Query Expansion](./03-query-expansion.md)

---

<!-- 
Sources Consulted:
- HyDE Paper: https://arxiv.org/abs/2212.10496
- LlamaIndex HyDE: https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/
-->
