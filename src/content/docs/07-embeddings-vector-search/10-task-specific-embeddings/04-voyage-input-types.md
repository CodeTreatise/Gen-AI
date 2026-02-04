---
title: "Voyage Input Types"
---

# Voyage Input Types

## Introduction

Voyage AI takes a **transparent approach** to task-specific embeddings. Unlike other providers that treat their optimization as a black box, Voyage documents exactly how their `input_type` parameter works: **prompt prepending**.

This lesson covers Voyage's input types, their specialized domain models, and how understanding the underlying mechanism can help you optimize your embeddings.

### What We'll Cover

- Voyage's input_type parameter
- The transparent prompt prepending mechanism
- Domain-specific models (code, law, finance)
- Flexible dimensions and quantization

### Prerequisites

- [Why Task Type Matters](./01-why-task-type-matters.md)
- Voyage API access and configuration

---

## Input Type Reference

Voyage keeps it simple with three options:

| Input Type | Description | Prepended Prompt |
|------------|-------------|------------------|
| `None` | No optimization | (none) |
| `query` | Search queries | "Represent the query for retrieving supporting documents: " |
| `document` | Indexed documents | "Represent the document for retrieval: " |

> **🔑 Key Insight:** Voyage is fully transparent about what happens when you set `input_type`. They literally prepend instructional text to your input before embedding.

---

## The Prompt Prepending Mechanism

### How It Works

When you set `input_type="query"`, Voyage transforms your input:

```python
# Your input
query = "How do I implement binary search?"

# What actually gets embedded
actual_input = "Represent the query for retrieving supporting documents: How do I implement binary search?"
```

For documents:

```python
# Your input
document = "Binary search is an algorithm that finds the position of a target value..."

# What actually gets embedded  
actual_input = "Represent the document for retrieval: Binary search is an algorithm that finds the position of a target value..."
```

### Why Transparency Matters

This transparency is valuable because:

1. **Reproducibility**: You know exactly what's happening
2. **Debugging**: You can manually prepend prompts to test
3. **Customization**: You can create your own prompts
4. **Compatibility**: Embeddings with and without input_type are compatible

---

## Basic Usage

### Setup

```python
import voyageai
import numpy as np

# Initialize client (uses VOYAGE_API_KEY env variable)
vo = voyageai.Client()
```

### Embedding Queries

```python
# Embed search queries
queries = [
    "How do neural networks learn?",
    "What is backpropagation?",
    "Best practices for training deep learning models"
]

result = vo.embed(
    texts=queries,
    model="voyage-4-large",
    input_type="query"
)

for i, emb in enumerate(result.embeddings):
    print(f"Query {i}: {len(emb)} dimensions")
```

### Embedding Documents

```python
# Embed documents for indexing
documents = [
    "Neural networks learn through iterative adjustment of weights. "
    "The learning process involves forward propagation, loss calculation, "
    "and backpropagation of gradients.",
    
    "Backpropagation is a key algorithm in training neural networks. "
    "It computes gradients of the loss function with respect to weights "
    "by applying the chain rule backwards through the network.",
    
    "Training deep learning models requires careful hyperparameter tuning, "
    "regularization techniques, and proper data preprocessing to achieve "
    "optimal performance."
]

result = vo.embed(
    texts=documents,
    model="voyage-4-large",
    input_type="document"
)

doc_embeddings = result.embeddings
print(f"Indexed {len(doc_embeddings)} documents")
print(f"Tokens used: {result.total_tokens}")
```

---

## Complete Search Example

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_voyage(query: str, documents: list[str], top_k: int = 3):
    """Search using Voyage asymmetric embeddings."""
    
    # Embed query with query type
    query_result = vo.embed(
        texts=[query],
        model="voyage-4-large",
        input_type="query"
    )
    query_emb = np.array(query_result.embeddings[0])
    
    # Embed documents with document type
    doc_result = vo.embed(
        texts=documents,
        model="voyage-4-large",
        input_type="document"
    )
    doc_embs = [np.array(e) for e in doc_result.embeddings]
    
    # Compute similarities
    results = []
    for doc, emb in zip(documents, doc_embs):
        similarity = cosine_similarity(query_emb, emb)
        results.append((doc, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example
results = search_voyage(
    "how do neural networks improve",
    documents
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"  {doc[:60]}...")
    print()
```

---

## Without Input Type (None)

When `input_type` is `None`, Voyage embeds your text directly without any prefix:

```python
# Direct embedding - no optimization
result = vo.embed(
    texts=["Your text here"],
    model="voyage-4-large",
    input_type=None  # or just omit the parameter
)
```

### When to Skip Input Type

| Scenario | Recommendation |
|----------|----------------|
| Semantic similarity (symmetric) | `input_type=None` |
| Custom prompt prepending | `input_type=None` + manual prefix |
| Compatibility testing | `input_type=None` |
| Retrieval (asymmetric) | Use `query`/`document` |

### Compatibility Note

> **Important:** Embeddings generated with and without the `input_type` argument are compatible. You can mix them in the same vector space, though for best retrieval results, use consistent input types.

---

## Domain-Specific Models

Voyage offers specialized models for specific domains, which may provide better results than input_type optimization alone.

### Code Search: voyage-code-3

Optimized for code retrieval tasks:

```python
# Natural language query
code_query = "function to sort array in descending order"

# Code snippets
code_docs = [
    """
def sort_descending(arr):
    '''Sort array in descending order'''
    return sorted(arr, reverse=True)
    """,
    """
def binary_search(arr, target):
    '''Find target in sorted array'''
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
    """,
    """
def merge_sort(arr):
    '''Sort array using merge sort'''
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
    """
]

# Embed query
query_result = vo.embed(
    texts=[code_query],
    model="voyage-code-3",
    input_type="query"
)
query_emb = np.array(query_result.embeddings[0])

# Embed code snippets
code_result = vo.embed(
    texts=code_docs,
    model="voyage-code-3",
    input_type="document"
)

# Search
for code, emb in zip(code_docs, code_result.embeddings):
    similarity = cosine_similarity(query_emb, np.array(emb))
    print(f"Score: {similarity:.4f}")
    print(f"  {code.strip()[:50]}...")
    print()
```

### Legal: voyage-law-2

Optimized for legal document retrieval:

```python
# Legal query
legal_query = "liability for breach of contract"

# Legal documents
legal_docs = [
    "In cases of material breach, the non-breaching party may seek "
    "compensatory damages to cover the loss of the benefit of the bargain.",
    
    "Contributory negligence occurs when a plaintiff's own negligence "
    "contributes to their injury, potentially reducing their recovery.",
    
    "Force majeure clauses excuse performance when extraordinary events "
    "beyond the parties' control prevent fulfillment of contractual obligations."
]

# Use specialized legal model
query_result = vo.embed(
    texts=[legal_query],
    model="voyage-law-2",
    input_type="query"
)

doc_result = vo.embed(
    texts=legal_docs,
    model="voyage-law-2",
    input_type="document"
)

# The legal model understands legal terminology and concepts better
```

### Finance: voyage-finance-2

Optimized for financial document retrieval:

```python
# Financial query
finance_query = "impact of interest rate hikes on bond prices"

# Financial documents
finance_docs = [
    "When central banks raise interest rates, existing bond prices typically "
    "fall as new bonds offer higher yields, making older bonds less attractive.",
    
    "Equity valuations are influenced by discount rates. Higher rates reduce "
    "the present value of future cash flows, pressuring stock prices.",
    
    "Currency exchange rates respond to interest rate differentials between "
    "countries, with higher-rate currencies often appreciating."
]

# Use specialized finance model
query_result = vo.embed(
    texts=[finance_query],
    model="voyage-finance-2",
    input_type="query"
)
```

---

## Flexible Dimensions

Voyage 4 series models support Matryoshka embeddings:

```python
# Available dimensions: 256, 512, 1024 (default), 2048
result = vo.embed(
    texts=["Sample text"],
    model="voyage-4-large",
    input_type="document",
    output_dimension=512  # Smaller for efficiency
)

print(f"Dimension: {len(result.embeddings[0])}")  # 512
```

### Dimension Comparison

| Dimension | Storage (per vector) | Quality (relative) |
|-----------|---------------------|-------------------|
| 2048 | 8 KB | 100% |
| 1024 (default) | 4 KB | ~99.5% |
| 512 | 2 KB | ~98% |
| 256 | 1 KB | ~95% |

---

## Quantization Options

Voyage supports multiple output data types:

```python
# Float (default, highest precision)
float_result = vo.embed(
    texts=["Sample"],
    model="voyage-4-large",
    output_dtype="float"
)

# Int8 (4x smaller)
int8_result = vo.embed(
    texts=["Sample"],
    model="voyage-4-large",
    output_dtype="int8"
)

# Binary (32x smaller)
binary_result = vo.embed(
    texts=["Sample"],
    model="voyage-4-large",
    output_dtype="ubinary"
)
```

| Output Type | Size (1024 dims) | Use Case |
|-------------|------------------|----------|
| `float` | 4,096 bytes | Maximum quality |
| `int8` | 1,024 bytes | Good balance |
| `ubinary` | 128 bytes | Maximum efficiency |

---

## Custom Prompt Prepending

Since Voyage is transparent about their mechanism, you can implement custom task types:

```python
def embed_with_custom_task(texts: list[str], task_prompt: str, model: str = "voyage-4-large"):
    """Embed with a custom task-specific prompt."""
    
    # Prepend custom prompt
    prefixed_texts = [f"{task_prompt}{text}" for text in texts]
    
    # Embed without input_type (we're handling it ourselves)
    result = vo.embed(
        texts=prefixed_texts,
        model=model,
        input_type=None
    )
    
    return result.embeddings

# Custom task: summarization-oriented embedding
summary_docs = embed_with_custom_task(
    texts=["Long document content here..."],
    task_prompt="Represent this document for finding related summaries: "
)

# Custom task: Q&A with specific domain
qa_query = embed_with_custom_task(
    texts=["How do I configure kubernetes?"],
    task_prompt="Represent this DevOps question for finding relevant documentation: "
)
```

### Pre-Built Prompt Library

```python
TASK_PROMPTS = {
    "qa_query": "Represent this question for finding relevant answers: ",
    "qa_answer": "Represent this answer for question matching: ",
    "summary_doc": "Represent this document for summary retrieval: ",
    "summary_query": "Represent this query for finding summarized content: ",
    "code_query": "Represent this programming question for finding code: ",
    "code_doc": "Represent this code snippet for search: ",
    "similar_doc": "Represent this document for finding similar documents: ",
}

def embed_task(texts: list[str], task: str, model: str = "voyage-4-large"):
    """Embed with pre-defined task prompt."""
    prompt = TASK_PROMPTS.get(task, "")
    return embed_with_custom_task(texts, prompt, model)
```

---

## Truncation Handling

Voyage provides control over long text handling:

```python
# Truncate long texts (default: True)
result = vo.embed(
    texts=[very_long_text],
    model="voyage-4-large",
    input_type="document",
    truncation=True  # Truncate to fit context
)

# Raise error for long texts
try:
    result = vo.embed(
        texts=[very_long_text],
        model="voyage-4-large",
        input_type="document",
        truncation=False  # Raise error if too long
    )
except Exception as e:
    print(f"Text too long: {e}")
```

Context limits by model:

| Model | Context Limit |
|-------|---------------|
| voyage-4-large | 32,000 tokens |
| voyage-4 | 32,000 tokens |
| voyage-4-lite | 32,000 tokens |
| voyage-code-3 | 32,000 tokens |
| voyage-law-2 | 16,000 tokens |
| voyage-finance-2 | 32,000 tokens |

---

## Batch Processing

Voyage has generous batch limits:

```python
# Batch embed up to 1000 texts
large_batch = ["Document " + str(i) for i in range(1000)]

result = vo.embed(
    texts=large_batch,
    model="voyage-4-large",
    input_type="document"
)

print(f"Embedded {len(result.embeddings)} documents")
print(f"Total tokens: {result.total_tokens}")
```

### Token Limits by Model

| Model | Max Batch Tokens |
|-------|------------------|
| voyage-4-lite | 1,000,000 |
| voyage-4 | 320,000 |
| voyage-4-large | 120,000 |
| voyage-code-3 | 120,000 |

---

## Summary

✅ **Two input types**: `query` and `document` (plus `None` for direct embedding)  
✅ **Transparent mechanism**: Voyage documents exactly what prompts they prepend  
✅ **Domain models**: Specialized models for code, law, and finance  
✅ **Flexible dimensions**: 256, 512, 1024, 2048 with Matryoshka support  
✅ **Quantization**: float, int8, and binary output types  
✅ **Custom prompts**: Build your own task types using the transparent mechanism

---

**Next:** [OpenAI Approach →](./05-openai-approach.md)

---

<!-- 
Sources Consulted:
- Voyage AI Embeddings documentation: https://docs.voyageai.com/docs/embeddings
- Voyage AI transparent prompt documentation
-->
