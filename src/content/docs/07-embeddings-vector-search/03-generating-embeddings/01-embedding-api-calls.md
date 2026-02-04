---
title: "Embedding API Calls"
---

# Embedding API Calls

## Introduction

Every embedding starts with an API call. Whether you're using OpenAI, Google Gemini, Cohere, or Voyage AI, understanding the request/response format is essential for building robust AI applications.

In this lesson, we'll explore the mechanics of calling embedding APIs across the four major providers, examining their request formats, response structures, error handling patterns, and rate limits.

### What We'll Cover

- Request formats and required parameters for each provider
- Response parsing and extracting embedding vectors
- Error handling and retry strategies
- Rate limits and usage tracking

### Prerequisites

- API keys for at least one embedding provider
- Python 3.10+ with provider SDK installed
- Basic understanding of REST APIs

---

## OpenAI Embeddings API

OpenAI's embeddings endpoint is the most widely used embedding API. Let's examine its structure.

### Request Format

```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY environment variable

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog.",
)

# Extract the embedding vector
embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")
```

**Output:**
```
Embedding dimensions: 1536
```

### Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | Model ID (e.g., `text-embedding-3-small`, `text-embedding-3-large`) |
| `input` | string or array | ✅ | Text to embed. Can be a single string or array of strings |
| `dimensions` | integer | ❌ | Output dimensions (only for `text-embedding-3-*` models) |
| `encoding_format` | string | ❌ | `float` (default) or `base64` for smaller payloads |
| `user` | string | ❌ | Unique user ID for abuse monitoring |

### Response Structure

```python
# Full response structure
print(f"Model: {response.model}")
print(f"Object type: {response.object}")
print(f"Number of embeddings: {len(response.data)}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

**Output:**
```
Model: text-embedding-3-small
Object type: list
Number of embeddings: 1
Prompt tokens: 10
Total tokens: 10
```

### Batch Embedding (Multiple Texts)

```python
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "NLP processes human language.",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)

# Each text gets its own embedding
for i, item in enumerate(response.data):
    print(f"Text {i}: {len(item.embedding)} dimensions")
```

**Output:**
```
Text 0: 1536 dimensions
Text 1: 1536 dimensions
Text 2: 1536 dimensions
```

> **Note:** OpenAI supports up to 8,192 tokens per input and 300,000 total tokens per request. The maximum array size is 2,048 inputs.

---

## Google Gemini Embeddings API

Gemini's embedding API offers task-specific optimization through its `task_type` parameter.

### Request Format

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")  # Or use GOOGLE_API_KEY env var

result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="The quick brown fox jumps over the lazy dog.",
    task_type="RETRIEVAL_DOCUMENT",
)

embedding = result['embedding']
print(f"Embedding dimensions: {len(embedding)}")
```

**Output:**
```
Embedding dimensions: 3072
```

### Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | Model ID (e.g., `models/gemini-embedding-001`) |
| `content` | string or list | ✅ | Text to embed |
| `task_type` | string | ❌ | Optimization hint (see Task Types lesson) |
| `title` | string | ❌ | Document title for `RETRIEVAL_DOCUMENT` task |
| `output_dimensionality` | integer | ❌ | Reduced dimensions (128-3072) |

### Batch Embedding

```python
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "NLP processes human language.",
]

result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=texts,
    task_type="RETRIEVAL_DOCUMENT",
)

# Returns a list of embeddings
for i, emb in enumerate(result['embedding']):
    print(f"Text {i}: {len(emb)} dimensions")
```

**Output:**
```
Text 0: 3072 dimensions
Text 1: 3072 dimensions
Text 2: 3072 dimensions
```

> **Important:** Gemini has a 2,048 token limit per input. The Batch API offers 50% cost reduction for async processing.

---

## Cohere Embed API (v2)

Cohere's v2 API introduces a unified `inputs` parameter that supports both text and images.

### Request Format

```python
import cohere

co = cohere.ClientV2()  # Uses COHERE_API_KEY env var

response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    texts=["The quick brown fox jumps over the lazy dog."],
)

embedding = response.embeddings.float_[0]
print(f"Embedding dimensions: {len(embedding)}")
```

**Output:**
```
Embedding dimensions: 1536
```

### Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | Model ID (e.g., `embed-v4.0`) |
| `input_type` | string | ✅ | `search_document`, `search_query`, `classification`, `clustering`, `image` |
| `texts` | array | ❌ | Array of text strings (max 96) |
| `images` | array | ❌ | Array of base64 data URIs (max 1) |
| `inputs` | array | ❌ | Mixed text+image inputs |
| `embedding_types` | array | ❌ | `float`, `int8`, `uint8`, `binary`, `ubinary` |
| `output_dimension` | integer | ❌ | 256, 512, 1024, or 1536 (default) |
| `truncate` | string | ❌ | `NONE`, `START`, `END` (default) |

### Multiple Embedding Types

Cohere can return embeddings in multiple formats simultaneously:

```python
response = co.embed(
    model="embed-v4.0",
    input_type="search_document",
    texts=["Sample text for embedding."],
    embedding_types=["float", "int8", "binary"],
)

print(f"Float embedding: {len(response.embeddings.float_[0])} dims")
print(f"Int8 embedding: {len(response.embeddings.int8[0])} dims")
print(f"Binary embedding: {len(response.embeddings.binary[0])} dims")
```

**Output:**
```
Float embedding: 1536 dims
Int8 embedding: 1536 dims
Binary embedding: 192 dims
```

> **Note:** Binary embeddings are bit-packed, so their length is 1/8 of the float embedding dimension.

---

## Voyage AI Embeddings API

Voyage AI offers specialized models for different domains (code, law, finance) with a clean Python API.

### Request Format

```python
import voyageai

vo = voyageai.Client()  # Uses VOYAGE_API_KEY env var

result = vo.embed(
    texts=["The quick brown fox jumps over the lazy dog."],
    model="voyage-4-large",
    input_type="document",
)

embedding = result.embeddings[0]
print(f"Embedding dimensions: {len(embedding)}")
print(f"Total tokens: {result.total_tokens}")
```

**Output:**
```
Embedding dimensions: 1024
Total tokens: 10
```

### Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `texts` | string or list | ✅ | Text(s) to embed |
| `model` | string | ✅ | Model name (e.g., `voyage-4-large`, `voyage-code-3`) |
| `input_type` | string | ❌ | `query`, `document`, or `None` |
| `truncation` | bool | ❌ | Truncate over-length texts (default: `True`) |
| `output_dimension` | int | ❌ | 256, 512, 1024 (default), or 2048 |
| `output_dtype` | string | ❌ | `float`, `int8`, `uint8`, `binary`, `ubinary` |

### Batch Limits by Model

| Model | Max Texts | Max Tokens |
|-------|-----------|------------|
| `voyage-4-lite`, `voyage-3.5-lite` | 1,000 | 1M |
| `voyage-4`, `voyage-3.5` | 1,000 | 320K |
| `voyage-4-large`, `voyage-code-3` | 1,000 | 120K |

---

## Error Handling

Robust embedding generation requires proper error handling. Here's a pattern that works across providers:

```python
import time
from openai import OpenAI, RateLimitError, APIError

def embed_with_retry(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Embed texts with exponential backoff retry."""
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]
            
        except RateLimitError as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
            
        except APIError as e:
            if e.status_code >= 500:  # Server error - retry
                wait_time = 2 ** attempt
                print(f"Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Client error - don't retry
    
    raise Exception(f"Failed after {max_retries} attempts")

# Usage
embeddings = embed_with_retry(["Hello, world!"])
```

### Common Error Types

| Error | Cause | Solution |
|-------|-------|----------|
| Rate Limit (429) | Too many requests | Exponential backoff, reduce batch size |
| Invalid Request (400) | Bad input format | Validate input before sending |
| Authentication (401) | Invalid API key | Check environment variables |
| Server Error (5xx) | Provider issue | Retry with backoff |
| Timeout | Slow response | Increase timeout, reduce batch size |

---

## Rate Limits by Provider

| Provider | Requests/min | Tokens/min | Notes |
|----------|--------------|------------|-------|
| OpenAI | 3,000-10,000 | 1M-10M | Varies by tier |
| Gemini | 1,500 | — | 100 RPD for free tier |
| Cohere | 10,000 | — | Production key required |
| Voyage | Varies | — | Contact for enterprise |

> **Tip:** Always check your API dashboard for your specific rate limits, as they vary by account tier.

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Batch multiple texts per request | Make one request per text |
| Implement retry with exponential backoff | Retry immediately after failures |
| Track token usage for cost estimation | Ignore usage metrics |
| Validate input length before sending | Let API truncate silently |
| Store API keys in environment variables | Hardcode API keys |

---

## Hands-on Exercise

### Your Task

Create a unified embedding function that works with multiple providers:

```python
def embed_texts(
    texts: list[str],
    provider: str = "openai",  # "openai", "gemini", "cohere", "voyage"
    **kwargs
) -> list[list[float]]:
    """Embed texts using the specified provider."""
    pass  # Your implementation
```

### Requirements

1. Support all four providers (OpenAI, Gemini, Cohere, Voyage)
2. Handle batch inputs correctly for each provider
3. Include basic error handling
4. Return a consistent output format (list of float lists)

<details>
<summary>💡 Hints</summary>

- Use a dictionary to map provider names to embedding functions
- Each provider has slightly different parameter names (`input` vs `texts` vs `content`)
- Normalize the response format since each provider returns embeddings differently

</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
import google.generativeai as genai
import cohere
import voyageai

def embed_texts(
    texts: list[str],
    provider: str = "openai",
    **kwargs
) -> list[list[float]]:
    """Embed texts using the specified provider."""
    
    if provider == "openai":
        client = OpenAI()
        response = client.embeddings.create(
            model=kwargs.get("model", "text-embedding-3-small"),
            input=texts,
            dimensions=kwargs.get("dimensions"),
        )
        return [item.embedding for item in response.data]
    
    elif provider == "gemini":
        result = genai.embed_content(
            model=kwargs.get("model", "models/gemini-embedding-001"),
            content=texts,
            task_type=kwargs.get("task_type", "RETRIEVAL_DOCUMENT"),
        )
        # Handle single vs batch response
        if isinstance(result['embedding'][0], float):
            return [result['embedding']]
        return result['embedding']
    
    elif provider == "cohere":
        co = cohere.ClientV2()
        response = co.embed(
            model=kwargs.get("model", "embed-v4.0"),
            input_type=kwargs.get("input_type", "search_document"),
            texts=texts,
        )
        return response.embeddings.float_
    
    elif provider == "voyage":
        vo = voyageai.Client()
        result = vo.embed(
            texts=texts,
            model=kwargs.get("model", "voyage-4"),
            input_type=kwargs.get("input_type", "document"),
        )
        return result.embeddings
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Test with each provider
texts = ["Hello, world!", "Goodbye, world!"]

# openai_embeddings = embed_texts(texts, provider="openai")
# gemini_embeddings = embed_texts(texts, provider="gemini")
# cohere_embeddings = embed_texts(texts, provider="cohere")
# voyage_embeddings = embed_texts(texts, provider="voyage")
```

</details>

---

## Summary

✅ OpenAI uses `input` parameter with `dimensions` for size control

✅ Gemini uses `content` with `task_type` for optimization hints

✅ Cohere requires `input_type` and supports multiple `embedding_types`

✅ Voyage offers domain-specific models with `output_dtype` for quantization

✅ Always implement retry logic with exponential backoff for production

**Next:** [Task-Type Specification](./02-task-type-specification.md)

---

## Further Reading

- [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings)
- [Gemini Embeddings Guide](https://ai.google.dev/gemini-api/docs/embeddings)
- [Cohere Embed API v2](https://docs.cohere.com/reference/embed)
- [Voyage AI Documentation](https://docs.voyageai.com/docs/embeddings)

---

<!-- 
Sources Consulted:
- OpenAI Embeddings API: https://platform.openai.com/docs/api-reference/embeddings
- Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Cohere Embed API v2: https://docs.cohere.com/reference/embed
- Voyage AI Embeddings: https://docs.voyageai.com/docs/embeddings
-->
