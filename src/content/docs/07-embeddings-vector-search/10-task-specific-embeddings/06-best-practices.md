---
title: "Task Type Best Practices"
---

# Task Type Best Practices

## Introduction

Choosing and using task types correctly can mean the difference between mediocre and excellent retrieval quality. This lesson consolidates the best practices for working with task-specific embeddings across all providers.

These practices apply whether you're using Gemini's 8 task types, Cohere's input_type, Voyage's transparent prompts, or even OpenAI's manual prefix strategies.

### What We'll Cover

- Matching query and document types consistently
- Avoiding common task type mistakes
- Benchmarking task types for your use case
- Documenting embedding choices
- Migration strategies between providers

### Prerequisites

- Familiarity with at least one task type system (Gemini, Cohere, or Voyage)
- Understanding of [Why Task Type Matters](./01-why-task-type-matters.md)

---

## Practice 1: Always Match Query/Document Types

### The Cardinal Rule

> **ALWAYS use paired task types for queries and documents.**

When using asymmetric embeddings for search:
- Queries → Use query-type embedding
- Documents → Use document-type embedding

### Provider-Specific Pairing

| Provider | Query Type | Document Type |
|----------|------------|---------------|
| **Gemini** | `RETRIEVAL_QUERY` | `RETRIEVAL_DOCUMENT` |
| **Cohere** | `search_query` | `search_document` |
| **Voyage** | `query` | `document` |
| **OpenAI** | "Represent this query..." | "Represent this document..." |

### Example: Correct Pairing

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Base class ensuring correct pairing."""
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed text as a query."""
        pass
    
    @abstractmethod
    def embed_document(self, text: str) -> list[float]:
        """Embed text as a document."""
        pass

class GeminiProvider(EmbeddingProvider):
    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response['embedding']
    
    def embed_document(self, text: str) -> list[float]:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return response['embedding']

class CohereProvider(EmbeddingProvider):
    def __init__(self, client):
        self.client = client
    
    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text],
            model="embed-v4.0",
            input_type="search_query"
        )
        return response.embeddings[0]
    
    def embed_document(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text],
            model="embed-v4.0",
            input_type="search_document"
        )
        return response.embeddings[0]
```

---

## Practice 2: Never Mix Task Types in the Same Index

### The Problem

```
❌ WRONG: Mixed task types in one index
┌─────────────────────────────────────────┐
│ Vector Index                            │
├─────────────────────────────────────────┤
│ doc1 → RETRIEVAL_DOCUMENT embedding     │
│ doc2 → SEMANTIC_SIMILARITY embedding    │  ← WRONG!
│ doc3 → RETRIEVAL_DOCUMENT embedding     │
│ doc4 → CLASSIFICATION embedding         │  ← WRONG!
└─────────────────────────────────────────┘
```

Different task types produce embeddings in **different geometric spaces**. Mixing them causes:
- Inconsistent similarity scores
- Unpredictable ranking
- Degraded retrieval quality

### The Solution

```
✅ CORRECT: Separate indexes per task type
┌─────────────────────────────────────────┐
│ Search Index (RETRIEVAL_DOCUMENT)       │
├─────────────────────────────────────────┤
│ doc1 → RETRIEVAL_DOCUMENT embedding     │
│ doc2 → RETRIEVAL_DOCUMENT embedding     │
│ doc3 → RETRIEVAL_DOCUMENT embedding     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Classification Index (CLASSIFICATION)   │
├─────────────────────────────────────────┤
│ sample1 → CLASSIFICATION embedding      │
│ sample2 → CLASSIFICATION embedding      │
│ sample3 → CLASSIFICATION embedding      │
└─────────────────────────────────────────┘
```

### Code Pattern

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class EmbeddingConfig:
    """Configuration for an embedding index."""
    task_type: str
    model: str
    dimensions: int
    provider: str
    
    def to_metadata(self) -> dict:
        """Convert to metadata for storage."""
        return {
            "task_type": self.task_type,
            "model": self.model,
            "dimensions": self.dimensions,
            "provider": self.provider,
            "version": "1.0"
        }

class VectorIndex:
    """Enforces single task type per index."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.vectors = []
        self.metadata = []
    
    def add(self, embedding: list[float], doc_metadata: dict):
        """Add embedding with validation."""
        if len(embedding) != self.config.dimensions:
            raise ValueError(
                f"Expected {self.config.dimensions} dims, got {len(embedding)}"
            )
        self.vectors.append(embedding)
        self.metadata.append(doc_metadata)
    
    def get_index_config(self) -> dict:
        """Return index configuration for documentation."""
        return self.config.to_metadata()

# Usage
search_config = EmbeddingConfig(
    task_type="RETRIEVAL_DOCUMENT",
    model="gemini-embedding-001",
    dimensions=768,
    provider="gemini"
)

search_index = VectorIndex(search_config)
```

---

## Practice 3: Benchmark Task Types for Your Use Case

### Generic vs. Your Data

Published benchmarks (MTEB, BEIR) use generic datasets. **Your data is unique.**

| Benchmark Says | Your Reality May Be |
|----------------|---------------------|
| "Task type A is 5% better" | Task type B works better for your domain |
| "Provider X leads" | Provider Y handles your language better |
| "768 dims is optimal" | 512 dims is sufficient for your use case |

### Build a Benchmark Pipeline

```python
from dataclasses import dataclass
from typing import Callable
import numpy as np
import json

@dataclass
class BenchmarkQuery:
    query: str
    relevant_doc_ids: list[str]  # Ground truth

@dataclass
class BenchmarkResult:
    task_type: str
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    p95_latency_ms: float

def benchmark_task_type(
    task_type: str,
    embed_query_fn: Callable,
    embed_doc_fn: Callable,
    documents: dict[str, str],
    queries: list[BenchmarkQuery]
) -> BenchmarkResult:
    """Benchmark a single task type configuration."""
    import time
    
    # Embed all documents
    doc_embeddings = {}
    for doc_id, text in documents.items():
        doc_embeddings[doc_id] = embed_doc_fn(text, task_type)
    
    mrr_scores = []
    recall_5_scores = []
    recall_10_scores = []
    latencies = []
    
    for bq in queries:
        # Time query embedding
        start = time.time()
        query_emb = embed_query_fn(bq.query, task_type)
        latencies.append((time.time() - start) * 1000)
        
        # Compute similarities
        scores = []
        for doc_id, doc_emb in doc_embeddings.items():
            sim = np.dot(query_emb, doc_emb)
            scores.append((doc_id, sim))
        
        # Rank by similarity
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        
        # Calculate metrics
        # MRR
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in bq.relevant_doc_ids:
                mrr_scores.append(1 / rank)
                break
        else:
            mrr_scores.append(0)
        
        # Recall@K
        top_5 = set(ranked_ids[:5])
        top_10 = set(ranked_ids[:10])
        relevant = set(bq.relevant_doc_ids)
        
        recall_5_scores.append(len(top_5 & relevant) / len(relevant))
        recall_10_scores.append(len(top_10 & relevant) / len(relevant))
    
    return BenchmarkResult(
        task_type=task_type,
        mrr=np.mean(mrr_scores),
        recall_at_5=np.mean(recall_5_scores),
        recall_at_10=np.mean(recall_10_scores),
        p95_latency_ms=np.percentile(latencies, 95)
    )

# Compare task types
results = []
for task_type in ["RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", "QUESTION_ANSWERING"]:
    result = benchmark_task_type(
        task_type=task_type,
        embed_query_fn=gemini_embed_query,
        embed_doc_fn=gemini_embed_doc,
        documents=your_documents,
        queries=your_test_queries
    )
    results.append(result)

# Print comparison
print(f"{'Task Type':<25} {'MRR':<8} {'R@5':<8} {'R@10':<8} {'P95 ms':<10}")
print("-" * 60)
for r in sorted(results, key=lambda x: x.mrr, reverse=True):
    print(f"{r.task_type:<25} {r.mrr:.4f}   {r.recall_at_5:.4f}   {r.recall_at_10:.4f}   {r.p95_latency_ms:.2f}")
```

**Sample Output:**
```
Task Type                 MRR      R@5      R@10     P95 ms    
------------------------------------------------------------
RETRIEVAL_QUERY           0.8234   0.7856   0.8923   45.23
QUESTION_ANSWERING        0.7912   0.7623   0.8654   46.12
SEMANTIC_SIMILARITY       0.7456   0.7123   0.8234   44.89
```

---

## Practice 4: Document Your Embedding Choices

### Create an Embedding Registry

Every production system should have documented embedding configurations:

```python
# embedding_registry.py
"""
Embedding Configuration Registry
================================

This file documents all embedding configurations used in this project.
DO NOT modify configurations in production without migration planning.

Last Updated: 2025-01-15
"""

EMBEDDING_CONFIGS = {
    "product_search": {
        "provider": "cohere",
        "model": "embed-v4.0",
        "dimensions": 1024,
        "query_type": "search_query",
        "document_type": "search_document",
        "created": "2025-01-01",
        "purpose": "E-commerce product search",
        "notes": "Tested against embed-english-v3.0, 12% better MRR"
    },
    
    "support_tickets": {
        "provider": "voyage",
        "model": "voyage-3",
        "dimensions": 1024,
        "query_type": "query",
        "document_type": "document",
        "created": "2025-01-10",
        "purpose": "Customer support ticket similarity",
        "notes": "Domain-specific model considered but baseline sufficient"
    },
    
    "code_search": {
        "provider": "gemini",
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "query_type": "CODE_RETRIEVAL_QUERY",
        "document_type": "RETRIEVAL_DOCUMENT",
        "created": "2025-01-12",
        "purpose": "Internal code search tool",
        "notes": "Voyage-code-3 tested, Gemini performed 5% better on our codebase"
    }
}

def get_config(name: str) -> dict:
    """Get embedding configuration by name."""
    if name not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown embedding config: {name}")
    return EMBEDDING_CONFIGS[name]

def list_configs() -> list[str]:
    """List all available embedding configurations."""
    return list(EMBEDDING_CONFIGS.keys())
```

### Store Configuration with Index

```python
import json

def create_index_with_metadata(index_name: str, config_name: str):
    """Create a vector index with embedded configuration."""
    config = get_config(config_name)
    
    # Create index (example with ChromaDB)
    collection = chroma_client.create_collection(
        name=index_name,
        metadata={
            "embedding_config": json.dumps(config),
            "created_at": datetime.now().isoformat(),
            "hnsw:space": "cosine"
        }
    )
    
    return collection

def validate_index_config(collection, expected_config: str):
    """Validate that index uses expected configuration."""
    stored = json.loads(collection.metadata.get("embedding_config", "{}"))
    expected = get_config(expected_config)
    
    mismatches = []
    for key in ["provider", "model", "dimensions", "query_type"]:
        if stored.get(key) != expected.get(key):
            mismatches.append(f"{key}: stored={stored.get(key)}, expected={expected.get(key)}")
    
    if mismatches:
        raise ValueError(f"Configuration mismatch:\n" + "\n".join(mismatches))
    
    return True
```

---

## Practice 5: Handle Migration Between Providers

### Migration Scenarios

| Scenario | Strategy |
|----------|----------|
| Changing task type (same provider) | Re-embed all documents, update config |
| Changing provider | Re-embed all documents, test before cutover |
| Changing dimensions | Create parallel index, gradual migration |
| Adding new task type | New index, separate from existing |

### Safe Migration Pattern

```python
from enum import Enum
from typing import Optional
import logging

class MigrationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"

class EmbeddingMigration:
    """Safe migration between embedding configurations."""
    
    def __init__(
        self,
        old_config: EmbeddingConfig,
        new_config: EmbeddingConfig,
        documents: list[dict]  # List of {id, text}
    ):
        self.old_config = old_config
        self.new_config = new_config
        self.documents = documents
        self.status = MigrationStatus.PENDING
        self.logger = logging.getLogger("embedding_migration")
    
    def validate_new_embeddings(self, sample_size: int = 100) -> bool:
        """Validate new embeddings on a sample."""
        sample = self.documents[:sample_size]
        
        # Re-run benchmark queries
        # Compare metrics between old and new
        # Return True if new >= old performance
        
        self.logger.info(f"Validated {sample_size} documents")
        return True  # Implement actual validation
    
    def run_parallel(self, traffic_percentage: float = 10) -> dict:
        """Run old and new embeddings in parallel."""
        # Both systems receive queries
        # Compare results
        # Log discrepancies
        
        return {
            "agreement_rate": 0.95,
            "new_better": 45,
            "old_better": 12,
            "equal": 43
        }
    
    def cutover(self, validate_first: bool = True):
        """Switch from old to new configuration."""
        if validate_first:
            if not self.validate_new_embeddings():
                raise ValueError("Validation failed, aborting migration")
        
        # Atomic switch
        # 1. Stop writes to old index
        # 2. Final sync
        # 3. Switch reads to new index
        # 4. Enable writes on new index
        
        self.status = MigrationStatus.COMPLETE
        self.logger.info("Migration complete")
    
    def rollback(self):
        """Revert to old configuration."""
        # Switch back to old index
        self.status = MigrationStatus.ROLLED_BACK
        self.logger.warning("Migration rolled back")
```

---

## Practice 6: Task Type Selection Flowchart

Use this decision tree to select the right task type:

```
┌─────────────────────────────────────────────────────────────┐
│                   TASK TYPE SELECTION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  What's your primary use case?                              │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   SEARCH    │     │  CLUSTERING  │     │CLASSIFICATION│ │
│  └──────┬──────┘     └──────┬───────┘     └──────┬───────┘ │
│         │                   │                     │         │
│         ▼                   ▼                     ▼         │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │ Short query │     │  CLUSTERING  │     │CLASSIFICATION│ │
│  │ vs long doc?│     │  task type   │     │  task type   │ │
│  └──────┬──────┘     └──────────────┘     └──────────────┘ │
│         │                                                   │
│    YES  │  NO                                              │
│    ▼    ▼                                                  │
│ ┌────────────┐  ┌─────────────────┐                        │
│ │RETRIEVAL_  │  │SEMANTIC_        │                        │
│ │QUERY/DOC   │  │SIMILARITY       │                        │
│ └────────────┘  └─────────────────┘                        │
│                                                             │
│  Special cases:                                             │
│  • Code search → CODE_RETRIEVAL_QUERY                       │
│  • FAQ systems → QUESTION_ANSWERING                         │
│  • Fact checking → FACT_VERIFICATION                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Task Type Cheat Sheet

| Use Case | Gemini | Cohere | Voyage | OpenAI |
|----------|--------|--------|--------|--------|
| Search queries | `RETRIEVAL_QUERY` | `search_query` | `query` | Prefix |
| Search documents | `RETRIEVAL_DOCUMENT` | `search_document` | `document` | Prefix |
| Duplicate detection | `SEMANTIC_SIMILARITY` | — | — | — |
| Topic modeling | `CLUSTERING` | `clustering` | — | — |
| Sentiment analysis | `CLASSIFICATION` | `classification` | — | — |
| Code search | `CODE_RETRIEVAL_QUERY` | — | Use voyage-code-3 | — |
| FAQ matching | `QUESTION_ANSWERING` | — | — | — |
| Fact checking | `FACT_VERIFICATION` | — | — | — |
| Images | — | `image` | — | — |

---

## Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|-----------|---------------------|
| Using same type for queries and documents | Use asymmetric pairs |
| Mixing task types in one index | Separate indexes per task type |
| Choosing based on benchmarks alone | Benchmark on YOUR data |
| Changing task types without re-embedding | Re-embed all documents |
| No documentation of choices | Maintain embedding registry |
| Hardcoding task types | Use configuration patterns |

---

## Hands-on Exercise

### Your Task

Build a task type testing framework that:
1. Accepts a list of test queries with ground truth
2. Tests multiple task type configurations
3. Reports comparative metrics
4. Recommends the best task type for the use case

### Requirements

1. Support at least 2 providers (Gemini and Cohere)
2. Calculate MRR and Recall@10
3. Generate a summary report
4. Handle API errors gracefully

### Expected Result

```
Task Type Benchmark Report
===========================

Dataset: customer_support_queries
Documents: 1,250
Test Queries: 50

Results:
--------
Provider   Task Type              MRR     Recall@10
Gemini     RETRIEVAL_QUERY        0.823   0.912
Gemini     QUESTION_ANSWERING     0.789   0.878
Cohere     search_query           0.834   0.923
Cohere     classification         0.654   0.756

Recommendation: Cohere search_query (highest MRR)
```

<details>
<summary>💡 Hints</summary>

- Use abstract base class for provider interface
- Store test queries as JSON for reproducibility
- Include timing information for latency comparison

</details>

<details>
<summary>✅ Solution</summary>

```python
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class TestQuery:
    text: str
    relevant_ids: list[str]

@dataclass
class TaskTypeConfig:
    provider: str
    query_type: str
    document_type: str

class Provider(ABC):
    @abstractmethod
    def embed(self, text: str, task_type: str) -> list[float]:
        pass

class TaskTypeBenchmark:
    def __init__(self, documents: dict[str, str], queries: list[TestQuery]):
        self.documents = documents
        self.queries = queries
        self.results = []
    
    def test_config(
        self, 
        provider: Provider, 
        config: TaskTypeConfig
    ) -> dict:
        # Embed documents
        doc_embeddings = {
            doc_id: provider.embed(text, config.document_type)
            for doc_id, text in self.documents.items()
        }
        
        mrr_scores = []
        recall_scores = []
        latencies = []
        
        for query in self.queries:
            start = time.time()
            q_emb = provider.embed(query.text, config.query_type)
            latencies.append(time.time() - start)
            
            # Rank documents
            scores = [
                (doc_id, np.dot(q_emb, d_emb))
                for doc_id, d_emb in doc_embeddings.items()
            ]
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, _ in ranked]
            
            # MRR
            for rank, doc_id in enumerate(ranked_ids, 1):
                if doc_id in query.relevant_ids:
                    mrr_scores.append(1 / rank)
                    break
            else:
                mrr_scores.append(0)
            
            # Recall@10
            top_10 = set(ranked_ids[:10])
            relevant = set(query.relevant_ids)
            recall_scores.append(len(top_10 & relevant) / len(relevant))
        
        return {
            "provider": config.provider,
            "query_type": config.query_type,
            "mrr": np.mean(mrr_scores),
            "recall_10": np.mean(recall_scores),
            "avg_latency_ms": np.mean(latencies) * 1000
        }
    
    def run_all(self, configs: list[TaskTypeConfig], providers: dict[str, Provider]):
        for config in configs:
            provider = providers[config.provider]
            result = self.test_config(provider, config)
            self.results.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> str:
        lines = [
            "Task Type Benchmark Report",
            "=" * 30,
            f"Documents: {len(self.documents)}",
            f"Queries: {len(self.queries)}",
            "",
            f"{'Provider':<10} {'Task Type':<25} {'MRR':<8} {'R@10':<8}",
            "-" * 55
        ]
        
        sorted_results = sorted(self.results, key=lambda x: x['mrr'], reverse=True)
        for r in sorted_results:
            lines.append(
                f"{r['provider']:<10} {r['query_type']:<25} "
                f"{r['mrr']:.3f}    {r['recall_10']:.3f}"
            )
        
        best = sorted_results[0]
        lines.append("")
        lines.append(f"Recommendation: {best['provider']} {best['query_type']}")
        
        return "\n".join(lines)
```
</details>

---

## Summary

✅ **Always pair** query and document task types correctly  
✅ **Never mix** task types in the same vector index  
✅ **Benchmark** on your own data, not just published benchmarks  
✅ **Document** all embedding configurations in a registry  
✅ **Plan migrations** carefully with validation and rollback  
✅ Use the **decision flowchart** to select appropriate task types

---

**Next:** [Cross-Encoder Reranking →](../11-cross-encoder-reranking.md)

---

<!-- 
Sources Consulted:
- Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Cohere Embed: https://docs.cohere.com/reference/embed
- Voyage AI: https://docs.voyageai.com/docs/embeddings
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Best practices synthesized from provider documentation and research
-->
