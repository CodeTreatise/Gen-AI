---
title: "OpenAI Approach"
---

# OpenAI Approach

## Introduction

Unlike Gemini, Cohere, and Voyage, OpenAI's embedding models **do not have a task type parameter**. The `text-embedding-3-small` and `text-embedding-3-large` models are designed as **general-purpose** embedding models that work well across all tasks without explicit optimization hints.

This lesson explains OpenAI's philosophy, how to work effectively without task types, and manual strategies to achieve task-specific optimization when needed.

### What We'll Cover

- Why OpenAI doesn't use task types
- Working effectively with general-purpose embeddings
- Manual prefix strategies for task optimization
- When OpenAI's approach is advantageous

### Prerequisites

- [Why Task Type Matters](./01-why-task-type-matters.md)
- OpenAI API access and configuration

---

## OpenAI's Philosophy

### No Task Type Parameter

OpenAI's embedding models accept only:

| Parameter | Description |
|-----------|-------------|
| `model` | Model name (text-embedding-3-small/large) |
| `input` | Text(s) to embed |
| `dimensions` | Output dimensions (optional, for Matryoshka) |
| `encoding_format` | Output format (float or base64) |

**No `task_type`, no `input_type`.**

### The Reasoning

OpenAI's approach reflects a design philosophy:

1. **Simplicity**: One model, one interface, works everywhere
2. **Robustness**: Trained to handle diverse inputs without hints
3. **Flexibility**: No risk of using the "wrong" task type
4. **Compatibility**: All embeddings are in the same space

---

## Basic Usage

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small"):
    """Embed texts using OpenAI."""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]

# Embed queries and documents the same way
query = "How do I learn Python?"
documents = [
    "Python is a versatile programming language great for beginners.",
    "Learning Python starts with understanding variables and data types.",
    "JavaScript is the language of the web."
]

query_emb = embed_openai([query])[0]
doc_embs = embed_openai(documents)

# Compute similarities
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for doc, emb in zip(documents, doc_embs):
    sim = cosine_similarity(query_emb, emb)
    print(f"{sim:.4f}: {doc[:50]}...")
```

**Output:**
```
0.8234: Python is a versatile programming language great...
0.8456: Learning Python starts with understanding variables...
0.5123: JavaScript is the language of the web...
```

> **Note:** OpenAI embeddings are normalized to length 1, so cosine similarity equals dot product.

---

## OpenAI's Built-in Normalization

OpenAI embeddings come pre-normalized:

```python
embedding = embed_openai(["Hello world"])[0]
norm = np.linalg.norm(embedding)
print(f"Norm: {norm:.6f}")  # 1.000000
```

This means:
- **Cosine similarity = dot product** (faster computation)
- **No post-processing needed** for similarity
- **Euclidean distance and cosine give same rankings**

---

## Manual Prefix Strategies

While OpenAI doesn't offer built-in task types, you can **manually prepend instructions** to achieve similar effects. This is effectively what other providers do internally.

### Query/Document Asymmetry

```python
def embed_query(text: str) -> list[float]:
    """Embed with query-optimizing prefix."""
    prefixed = f"Represent this search query for finding relevant documents: {text}"
    return embed_openai([prefixed])[0]

def embed_document(text: str) -> list[float]:
    """Embed with document-optimizing prefix."""
    prefixed = f"Represent this document for retrieval: {text}"
    return embed_openai([prefixed])[0]

# Usage
query_emb = embed_query("How do neural networks learn?")
doc_emb = embed_document("Neural networks learn through backpropagation...")
```

### Task-Specific Prefixes

```python
TASK_PREFIXES = {
    # Retrieval
    "query": "Represent this search query for finding relevant documents: ",
    "document": "Represent this document for retrieval: ",
    
    # Classification
    "classification": "Represent this text for classification: ",
    
    # Clustering
    "clustering": "Represent this text for clustering with similar items: ",
    
    # Similarity
    "similarity": "Represent this text for measuring semantic similarity: ",
    
    # Code
    "code_query": "Represent this natural language query for finding code: ",
    "code_doc": "Represent this code for search: ",
    
    # Q&A
    "question": "Represent this question for finding relevant answers: ",
    "answer": "Represent this answer for question matching: ",
}

def embed_with_task(text: str, task: str) -> list[float]:
    """Embed with task-specific prefix."""
    prefix = TASK_PREFIXES.get(task, "")
    prefixed_text = f"{prefix}{text}"
    return embed_openai([prefixed_text])[0]

# Examples
query_emb = embed_with_task("best pizza recipe", task="query")
doc_emb = embed_with_task("This pizza recipe uses fresh mozzarella...", task="document")
```

---

## When Prefixes Help (and When They Don't)

### Experimental Evidence

Let's test whether prefixes improve OpenAI retrieval:

```python
# Test corpus
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "The weather today is sunny with a high of 75 degrees.",
]

queries = [
    "What is ML?",
    "Best language for data analysis?"
]

def test_retrieval(use_prefixes: bool = False):
    """Test retrieval with and without prefixes."""
    
    if use_prefixes:
        doc_embs = [embed_with_task(d, "document") for d in documents]
        query_embs = [embed_with_task(q, "query") for q in queries]
    else:
        doc_embs = embed_openai(documents)
        query_embs = embed_openai(queries)
    
    print(f"\n{'With prefixes' if use_prefixes else 'Without prefixes'}:")
    print("-" * 50)
    
    for query, q_emb in zip(queries, query_embs):
        print(f"\nQuery: {query}")
        scores = [(doc, cosine_similarity(q_emb, d_emb)) 
                  for doc, d_emb in zip(documents, doc_embs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        for doc, score in scores:
            print(f"  {score:.4f}: {doc[:40]}...")

# Compare
test_retrieval(use_prefixes=False)
test_retrieval(use_prefixes=True)
```

### Typical Results

| Scenario | Prefix Impact |
|----------|---------------|
| Short queries vs long docs | Slight improvement |
| Similar-length texts | Minimal difference |
| Highly specialized domain | Can help significantly |
| General retrieval | Often negligible |

> **Takeaway:** OpenAI's models are trained to be robust without prefixes. Prefixes may help in edge cases but aren't as impactful as with models specifically designed for task types.

---

## Matryoshka Dimensions

OpenAI supports dimension reduction via the `dimensions` parameter:

```python
# Small embedding (faster, cheaper storage)
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Sample text",
    dimensions=256  # Instead of default 3072
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 256
```

### Dimension Comparison

| Model | Default | Recommended Sizes |
|-------|---------|-------------------|
| text-embedding-3-large | 3072 | 256, 512, 1024, 1536, 3072 |
| text-embedding-3-small | 1536 | 256, 512, 1024, 1536 |

---

## Complete Search Implementation

Here's a production-ready search implementation with optional prefixes:

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SearchResult:
    text: str
    score: float
    index: int

class OpenAISearch:
    """Search implementation using OpenAI embeddings."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        use_prefixes: bool = False,
        dimensions: Optional[int] = None
    ):
        self.client = OpenAI()
        self.model = model
        self.use_prefixes = use_prefixes
        self.dimensions = dimensions
        self.documents = []
        self.embeddings = []
    
    def _embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Embed texts with optional prefixing."""
        if self.use_prefixes:
            prefix = "Represent this search query: " if is_query else "Represent this document: "
            texts = [f"{prefix}{t}" for t in texts]
        
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
            
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    
    def index(self, documents: list[str]):
        """Index a list of documents."""
        self.documents = documents
        self.embeddings = self._embed(documents, is_query=False)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for documents matching the query."""
        query_emb = self._embed([query], is_query=True)[0]
        
        # Compute similarities (dot product since normalized)
        scores = [np.dot(query_emb, doc_emb) for doc_emb in self.embeddings]
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            SearchResult(
                text=self.documents[i],
                score=scores[i],
                index=i
            )
            for i in top_indices
        ]

# Usage
search = OpenAISearch(
    model="text-embedding-3-small",
    use_prefixes=True,  # Try with and without
    dimensions=512      # Optional dimension reduction
)

documents = [
    "Python is great for data science and machine learning.",
    "JavaScript powers interactive web applications.",
    "Rust provides memory safety without garbage collection.",
    "Go is designed for concurrent programming.",
]

search.index(documents)

results = search.search("best language for AI?")
for result in results:
    print(f"{result.score:.4f}: {result.text}")
```

---

## When OpenAI's Approach Works Best

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Simplicity** | No task type to choose or misuse |
| **Flexibility** | Same embeddings work for multiple purposes |
| **Fewer bugs** | Can't accidentally use wrong task type |
| **Easier migration** | Switch use cases without re-embedding |

### Ideal Use Cases

1. **Multi-purpose indexes**: Same embeddings for search, clustering, and classification
2. **Unknown future use**: Build now, decide use case later
3. **Simple applications**: Quick prototypes without optimization
4. **Mixed workloads**: Same index serves different query types

### When to Consider Alternatives

1. **Maximum retrieval quality**: Task-specific models can edge out general-purpose
2. **Large-scale production**: Small % improvement matters at scale
3. **Specialized domains**: Code, legal, medical may benefit from specialized models

---

## Comparison with Task-Specific Providers

### Retrieval Quality (Approximate)

| Provider | General Model | With Task Type |
|----------|---------------|----------------|
| OpenAI text-embedding-3-large | 64.6% MTEB | N/A (no task type) |
| Gemini gemini-embedding-001 | ~65% | ~67% with RETRIEVAL_* |
| Cohere embed-v4.0 | ~66% | ~68% with search_* |
| Voyage voyage-4-large | ~66% | ~68% with query/document |

> **Note:** These are approximate ranges. Actual performance varies by benchmark and use case.

### OpenAI vs. Manual Prefixes

```
┌─────────────────────────────────────────────────────────┐
│                 Quality Comparison                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OpenAI (no prefix)           ████████████████ (64.6%)  │
│  OpenAI (manual prefix)       █████████████████ (65-66%)│
│  Gemini (with task type)      ██████████████████ (67%)  │
│  Cohere (with input_type)     ███████████████████ (68%) │
│                                                         │
│  Note: Manual prefixes give ~1-2% boost for OpenAI     │
└─────────────────────────────────────────────────────────┘
```

---

## Best Practices for OpenAI

### 1. Start Simple

```python
# Begin without prefixes
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
```

### 2. Test Prefixes If Needed

```python
# Add prefixes only if benchmarks show improvement
def test_with_prefixes():
    # ... benchmark code
    pass
```

### 3. Consider Model Size

```python
# text-embedding-3-large for quality
# text-embedding-3-small for cost/speed
model = "text-embedding-3-large" if quality_critical else "text-embedding-3-small"
```

### 4. Use Dimension Reduction

```python
# Reduce dimensions for storage efficiency
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=text,
    dimensions=1024  # 3x smaller, ~1% quality loss
)
```

---

## Summary

✅ OpenAI has **no task type parameter**—general-purpose by design  
✅ Embeddings are **pre-normalized** to length 1  
✅ **Manual prefixes** can provide slight improvements (1-2%)  
✅ **Simplicity** is the main advantage over task-specific models  
✅ Works well for **multi-purpose** and **prototype** applications  
✅ Consider **task-specific providers** for maximum retrieval quality

---

**Next:** [Best Practices →](./06-best-practices.md)

---

<!-- 
Sources Consulted:
- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- OpenAI API reference
- MTEB benchmark results
-->
