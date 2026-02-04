---
title: "Text Embedding Landscape 2025"
---

# Text Embedding Landscape 2025

## Introduction

The text embedding landscape has evolved dramatically. In 2023, OpenAI's `text-embedding-ada-002` dominated. By 2025, we have multiple providers offering models with flexible dimensions, multimodal capabilities, and specialized optimizations. Understanding the current landscape helps you make informed choices.

This lesson surveys the major commercial embedding providers and their flagship models.

### What We'll Cover

- OpenAI's text-embedding-3 series
- Google's Gemini embeddings
- Cohere's embed-v4.0
- Voyage AI's specialized models
- Anthropic's embedding strategy
- When to choose each provider

### Prerequisites

- [Understanding Embeddings](../01-understanding-embeddings/00-understanding-embeddings.md)
- API key for at least one provider (for hands-on)

---

## OpenAI: The Industry Standard

OpenAI offers two third-generation embedding models, both released in early 2024 and still leading in 2025.

### Models

| Model | Dimensions | Context | MTEB Score | Best For |
|-------|------------|---------|------------|----------|
| `text-embedding-3-small` | 1536 (default) | 8192 tokens | 62.3% | Cost-effective production |
| `text-embedding-3-large` | 3072 (default) | 8192 tokens | 64.6% | Maximum quality |

### Key Features

**Flexible Dimensions (Matryoshka)**
Both models support reducing dimensions without retraining:

```python
from openai import OpenAI

client = OpenAI()

# Full dimensions (default)
response_full = client.embeddings.create(
    model="text-embedding-3-large",
    input="Machine learning fundamentals"
)
print(f"Full: {len(response_full.data[0].embedding)} dimensions")  # 3072

# Reduced dimensions
response_reduced = client.embeddings.create(
    model="text-embedding-3-large",
    input="Machine learning fundamentals",
    dimensions=1024  # Reduce to 1024
)
print(f"Reduced: {len(response_reduced.data[0].embedding)} dimensions")  # 1024
```

**Output:**
```
Full: 3072 dimensions
Reduced: 1024 dimensions
```

**Pre-Normalized Vectors**
OpenAI embeddings come normalized to length 1, so:
- Cosine similarity = dot product (faster computation)
- No normalization step needed

```python
import numpy as np

embedding = response_full.data[0].embedding
magnitude = np.linalg.norm(embedding)
print(f"Magnitude: {magnitude:.6f}")  # Very close to 1.0
```

**Output:**
```
Magnitude: 1.000000
```

### When to Choose OpenAI

✅ **Choose OpenAI when:**
- You need reliable, well-documented API
- Maximum compatibility with tools/tutorials
- Strong multilingual performance
- 8K context is sufficient

❌ **Consider alternatives when:**
- You need 32K+ context
- Budget is extremely tight
- You require on-premise deployment
- You need specialized domain models

---

## Google Gemini: Task-Optimized Embeddings

Google's `gemini-embedding-001` (released June 2025) offers task-specific optimization through a `task_type` parameter.

### Model

| Model | Dimensions | Context | MTEB Scores |
|-------|------------|---------|-------------|
| `gemini-embedding-001` | 768, 1536, 3072 | 2048 tokens | 768d: 67.99%, 1536d: 68.17%, 3072d: 68.16% |

### Key Features

**Task Types**
Specify how the embedding will be used for optimized results:

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

# For search queries
query_embedding = genai.embed_content(
    model="models/gemini-embedding-001",
    content="What is machine learning?",
    task_type="RETRIEVAL_QUERY"
)

# For documents being searched
doc_embedding = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Machine learning is a subset of AI that enables systems to learn...",
    task_type="RETRIEVAL_DOCUMENT"
)

# For semantic similarity
similarity_embedding = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Neural networks process information",
    task_type="SEMANTIC_SIMILARITY"
)

print(f"Query embedding: {len(query_embedding['embedding'])} dims")
print(f"Doc embedding: {len(doc_embedding['embedding'])} dims")
```

### Available Task Types

| Task Type | Use Case |
|-----------|----------|
| `RETRIEVAL_QUERY` | Search queries |
| `RETRIEVAL_DOCUMENT` | Documents to be searched |
| `SEMANTIC_SIMILARITY` | Comparing text similarity |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Grouping similar texts |
| `CODE_RETRIEVAL_QUERY` | Code search queries |
| `QUESTION_ANSWERING` | QA systems |
| `FACT_VERIFICATION` | Fact-checking |

**Matryoshka Dimensions**

```python
# Reduce to 768 dimensions
embedding_768 = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Sample text",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=768
)
print(f"Reduced: {len(embedding_768['embedding'])} dims")
```

> **Warning:** Gemini embeddings at 3072d are normalized. Embeddings at 768d and 1536d require **manual normalization** for cosine similarity to work correctly!

```python
import numpy as np

def normalize(embedding):
    """Normalize embedding to unit length."""
    arr = np.array(embedding)
    return (arr / np.linalg.norm(arr)).tolist()

# For 768d or 1536d, always normalize!
embedding_768 = genai.embed_content(
    model="models/gemini-embedding-001",
    content="Sample text",
    output_dimensionality=768
)
normalized = normalize(embedding_768['embedding'])
```

### When to Choose Gemini

✅ **Choose Gemini when:**
- Task-specific optimization matters (QA, classification, retrieval)
- Competitive MTEB scores at lower dimensions (768d ≈ quality of 3072d)
- Integration with Google Cloud ecosystem
- Batch processing (50% discount)

❌ **Consider alternatives when:**
- You need 8K+ context (Gemini = 2048 tokens)
- Consistent normalized output is critical (watch 768/1536d)

---

## Cohere: Multilingual & Multimodal

Cohere's `embed-v4.0` focuses on multilingual support and multimodal capabilities.

### Model

| Model | Dimensions | Languages | Features |
|-------|------------|-----------|----------|
| `embed-v4.0` | 256-1536 | 100+ | Multimodal, Matryoshka, compression |

### Key Features

**Input Type Optimization**

```python
import cohere

co = cohere.Client("your-api-key")

# For queries
query_response = co.embed(
    texts=["What is machine learning?"],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]
)

# For documents
doc_response = co.embed(
    texts=["Machine learning enables computers to learn from data..."],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)

print(f"Query: {len(query_response.embeddings.float[0])} dims")
print(f"Doc: {len(doc_response.embeddings.float[0])} dims")
```

### Input Types

| Input Type | Use Case |
|------------|----------|
| `search_query` | User search queries |
| `search_document` | Documents to index |
| `classification` | Text classification tasks |
| `clustering` | Grouping similar texts |

**Flexible Dimensions (Matryoshka)**

```python
# Reduce dimensions
response = co.embed(
    texts=["Sample text"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    output_dimension=512  # 256, 512, 1024, 1536
)
```

**Compression Options**

```python
# Different precision levels
response = co.embed(
    texts=["Sample text"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["int8", "binary"]  # Compressed formats
)

float_size = 1536 * 4  # 6,144 bytes
int8_size = 1536 * 1   # 1,536 bytes
binary_size = 1536 / 8 # 192 bytes

print(f"Float: {float_size} bytes")
print(f"Int8: {int8_size} bytes (75% smaller)")
print(f"Binary: {binary_size} bytes (97% smaller)")
```

**Multilingual Support**

```python
# Same model handles 100+ languages
texts = [
    "Machine learning is fascinating",
    "机器学习很有趣",  # Chinese
    "Aprendizaje automático es fascinante",  # Spanish
    "機械学習は魅力的です",  # Japanese
]

response = co.embed(
    texts=texts,
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)

# Cross-lingual similarity works out of the box
```

### When to Choose Cohere

✅ **Choose Cohere when:**
- Multilingual support is critical (100+ languages)
- You need multimodal (text + images)
- Storage optimization via compression
- Free tier availability matters

❌ **Consider alternatives when:**
- You only work in English
- Maximum benchmark scores are critical

---

## Voyage AI: Specialized Excellence

Voyage AI focuses on specialized embeddings with excellent benchmark performance.

### Models

| Model | Dimensions | Context | Specialty |
|-------|------------|---------|-----------|
| `voyage-4-large` | 1024 | 32,000 | Best general quality |
| `voyage-4` | 1024 | 32,000 | Balanced performance |
| `voyage-4-lite` | 1024 | 32,000 | Speed optimized |
| `voyage-code-3` | 1024 | 32,000 | Code retrieval |
| `voyage-finance-2` | 1024 | 32,000 | Financial domain |
| `voyage-law-2` | 1024 | 16,000 | Legal domain |

### Key Features

**Long Context (32K Tokens)**

```python
import voyageai

vo = voyageai.Client()

# Embed very long documents (up to 32K tokens!)
long_document = "..." * 10000  # Very long text

response = vo.embed(
    texts=[long_document],
    model="voyage-4-large",
    input_type="document"
)

print(f"Embedded {len(long_document)} character document")
```

**Input Type Optimization**

```python
# Query embedding (adds retrieval-optimized prompt)
query_result = vo.embed(
    texts=["What is transformer architecture?"],
    model="voyage-4-large",
    input_type="query"
)

# Document embedding
doc_result = vo.embed(
    texts=["Transformers use self-attention mechanisms..."],
    model="voyage-4-large",
    input_type="document"
)
```

**Flexible Dimensions**

```python
# Matryoshka support
response = vo.embed(
    texts=["Sample text"],
    model="voyage-4-large",
    input_type="document",
    output_dimension=512  # 256, 512, 1024, 2048
)
```

**Compression/Quantization**

```python
# Multiple output formats
response = vo.embed(
    texts=["Sample text"],
    model="voyage-4-large",
    input_type="document",
    output_dtype="int8"  # float, int8, uint8, binary, ubinary
)
```

### When to Choose Voyage AI

✅ **Choose Voyage AI when:**
- Long context needed (32K tokens vs 8K elsewhere)
- Specialized domains (code, finance, legal)
- High benchmark scores matter
- Flexible output formats needed

❌ **Consider alternatives when:**
- Budget is primary concern (premium pricing)
- OpenAI compatibility is required

---

## Anthropic: Partner Strategy

Anthropic (Claude) does **not** offer a public embedding API. Instead, they partner with embedding providers.

### Official Recommendations

Anthropic recommends:
1. **Voyage AI** - Tight integration, recommended by Anthropic
2. **Cohere** - Strong multilingual support
3. **OpenAI** - Broad compatibility

```python
# Typical RAG setup with Claude + Voyage
import anthropic
import voyageai

# Embeddings from Voyage
voyage = voyageai.Client()
embeddings = voyage.embed(
    texts=["Your documents..."],
    model="voyage-4-large",
    input_type="document"
)

# LLM from Anthropic
claude = anthropic.Anthropic()
response = claude.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## Provider Comparison Matrix

| Feature | OpenAI | Gemini | Cohere | Voyage AI |
|---------|--------|--------|--------|-----------|
| **Max Context** | 8,192 | 2,048 | — | 32,000 |
| **Max Dimensions** | 3,072 | 3,072 | 1,536 | 2,048 |
| **Matryoshka** | ✅ | ✅ | ✅ | ✅ |
| **Task Types** | ❌ | ✅ | ✅ | ✅ |
| **Pre-Normalized** | ✅ | Partial* | ✅ | ✅ |
| **Multimodal** | ❌ | ❌ | ✅ | ✅ |
| **Multilingual** | Good | Good | 100+ | Good |
| **Compression** | ❌ | ❌ | ✅ | ✅ |
| **Domain Models** | ❌ | ❌ | ❌ | ✅ |
| **Self-Host** | ❌ | ❌ | ❌ | ❌ |

*Gemini: 3072d normalized, 768/1536d require manual normalization

---

## Hands-on Exercise

### Your Task

Create an `EmbeddingProviderComparator` that tests multiple providers on the same dataset and reports quality metrics.

### Requirements

1. Accept a list of text pairs with known similarity labels (similar/dissimilar)
2. Embed with at least 2 different providers
3. Calculate cosine similarity for each pair
4. Report which provider best separates similar from dissimilar pairs
5. Measure and report latency

<details>
<summary>💡 Hints (click to expand)</summary>

- Use a simple labeled dataset: `[("text1", "text2", True), ("text3", "text4", False)]`
- Calculate average similarity for "similar" pairs and "dissimilar" pairs
- The best provider has the largest gap between these averages
- Use `time.time()` to measure latency

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
from openai import OpenAI
import voyageai
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class ProviderMetrics:
    name: str
    avg_similar: float
    avg_dissimilar: float
    separation_gap: float
    avg_latency_ms: float

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class EmbeddingProviderComparator:
    """Compare embedding providers on labeled data."""
    
    def __init__(self):
        self.openai_client = OpenAI()
        self.voyage_client = voyageai.Client()
    
    def _embed_openai(self, texts: list[str]) -> tuple[list, float]:
        """Embed with OpenAI, return embeddings and latency."""
        start = time.time()
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        latency = (time.time() - start) * 1000
        embeddings = [e.embedding for e in response.data]
        return embeddings, latency
    
    def _embed_voyage(self, texts: list[str]) -> tuple[list, float]:
        """Embed with Voyage AI, return embeddings and latency."""
        start = time.time()
        response = self.voyage_client.embed(
            texts=texts,
            model="voyage-4-lite",
            input_type="document"
        )
        latency = (time.time() - start) * 1000
        return response.embeddings, latency
    
    def compare(
        self, 
        pairs: list[tuple[str, str, bool]]
    ) -> dict[str, ProviderMetrics]:
        """
        Compare providers on labeled pairs.
        
        pairs: List of (text1, text2, is_similar) tuples
        """
        # Collect all unique texts
        all_texts = []
        for t1, t2, _ in pairs:
            if t1 not in all_texts:
                all_texts.append(t1)
            if t2 not in all_texts:
                all_texts.append(t2)
        
        results = {}
        
        for provider_name, embed_fn in [
            ("OpenAI", self._embed_openai),
            ("Voyage", self._embed_voyage)
        ]:
            try:
                # Embed all texts
                embeddings, latency = embed_fn(all_texts)
                text_to_embedding = dict(zip(all_texts, embeddings))
                
                # Calculate similarities
                similar_scores = []
                dissimilar_scores = []
                
                for t1, t2, is_similar in pairs:
                    sim = cosine_similarity(
                        text_to_embedding[t1],
                        text_to_embedding[t2]
                    )
                    if is_similar:
                        similar_scores.append(sim)
                    else:
                        dissimilar_scores.append(sim)
                
                avg_similar = np.mean(similar_scores)
                avg_dissimilar = np.mean(dissimilar_scores)
                
                results[provider_name] = ProviderMetrics(
                    name=provider_name,
                    avg_similar=avg_similar,
                    avg_dissimilar=avg_dissimilar,
                    separation_gap=avg_similar - avg_dissimilar,
                    avg_latency_ms=latency / len(all_texts)
                )
            except Exception as e:
                print(f"Error with {provider_name}: {e}")
        
        return results
    
    def report(self, results: dict[str, ProviderMetrics]) -> None:
        """Print comparison report."""
        print("=" * 60)
        print("EMBEDDING PROVIDER COMPARISON")
        print("=" * 60)
        
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Avg similarity (similar pairs):    {metrics.avg_similar:.4f}")
            print(f"  Avg similarity (dissimilar pairs): {metrics.avg_dissimilar:.4f}")
            print(f"  Separation gap:                    {metrics.separation_gap:.4f}")
            print(f"  Avg latency per text:              {metrics.avg_latency_ms:.1f}ms")
        
        # Winner
        best = max(results.values(), key=lambda x: x.separation_gap)
        print(f"\n🏆 Best separation: {best.name} (gap: {best.separation_gap:.4f})")


def test_comparator():
    """Test the provider comparator."""
    comparator = EmbeddingProviderComparator()
    
    # Labeled test data
    pairs = [
        # Similar pairs (True)
        ("Machine learning enables predictive models", 
         "AI systems can make predictions from data", True),
        ("Python is a programming language", 
         "Python is used for coding software", True),
        ("The cat sat on the mat", 
         "A feline rested on the rug", True),
        
        # Dissimilar pairs (False)
        ("Machine learning enables predictive models",
         "The recipe calls for two eggs", False),
        ("Python is a programming language",
         "The weather is sunny today", False),
        ("Neural networks process information",
         "My favorite color is blue", False),
    ]
    
    results = comparator.compare(pairs)
    comparator.report(results)

test_comparator()
```

**Output:**
```
============================================================
EMBEDDING PROVIDER COMPARISON
============================================================

OpenAI:
  Avg similarity (similar pairs):    0.8234
  Avg similarity (dissimilar pairs): 0.1456
  Separation gap:                    0.6778
  Avg latency per text:              45.2ms

Voyage:
  Avg similarity (similar pairs):    0.8567
  Avg similarity (dissimilar pairs): 0.1234
  Separation gap:                    0.7333
  Avg latency per text:              52.1ms

🏆 Best separation: Voyage (gap: 0.7333)
```

</details>

---

## Summary

✅ **OpenAI** = Industry standard, great documentation, 8K context  
✅ **Gemini** = Task-specific optimization, competitive at low dimensions  
✅ **Cohere** = Multilingual champion (100+ languages), multimodal, compression  
✅ **Voyage AI** = Long context (32K), specialized domains (code, legal, finance)  
✅ **Anthropic** = No embeddings API—partners with Voyage/Cohere  

**Next:** [Model Specifications & Benchmarks →](./02-model-specifications-benchmarks.md)

---

## Further Reading

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Google Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [Cohere Embeddings](https://docs.cohere.com/reference/embed)
- [Voyage AI Documentation](https://docs.voyageai.com/docs/embeddings)

<!-- 
Sources Consulted:
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Voyage AI Documentation: https://docs.voyageai.com/docs/embeddings
- Google Gemini API: https://ai.google.dev/gemini-api/docs/embeddings
- Cohere Embeddings: https://docs.cohere.com/reference/embed
-->
