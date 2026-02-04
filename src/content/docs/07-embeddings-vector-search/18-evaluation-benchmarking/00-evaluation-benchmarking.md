---
title: "Evaluation & Benchmarking"
---

# Evaluation & Benchmarking

## Overview

How do you know if your embedding system is actually working? Intuition and spot-checking aren't enough for production systems. This lesson provides the rigorous evaluation framework you need—from standard retrieval metrics to industry benchmarks to end-to-end RAG evaluation.

Evaluation isn't a one-time activity. It's an ongoing discipline that ensures your retrieval quality remains high as your data grows and user queries evolve.

### Why Evaluation Matters

- **Baseline Comparison**: Objectively compare embedding models before deployment
- **Quality Assurance**: Detect degradation before users notice
- **Optimization Decisions**: Measure impact of chunking, reranking, or model changes
- **Stakeholder Communication**: Provide concrete metrics to non-technical stakeholders
- **Continuous Improvement**: Build feedback loops that drive systematic enhancement

---

## Lesson Structure

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Retrieval Quality Metrics](./01-retrieval-quality-metrics.md) | Recall@k, Precision@k, MRR, NDCG, Hit rate |
| 02 | [MTEB Benchmark](./02-mteb-benchmark.md) | Massive Text Embedding Benchmark for model comparison |
| 03 | [BEIR Benchmark](./03-beir-benchmark.md) | Zero-shot retrieval across 18 diverse datasets |
| 04 | [Building Evaluation Datasets](./04-building-evaluation-datasets.md) | Create query-document pairs for your domain |
| 05 | [A/B Testing Retrieval](./05-ab-testing-retrieval.md) | Statistical comparison of retrieval strategies |
| 06 | [End-to-End RAG Evaluation](./06-end-to-end-rag-evaluation.md) | RAGAS, faithfulness, context relevance |
| 07 | [Continuous Monitoring](./07-continuous-monitoring.md) | Drift detection and quality alerts |

---

## The Evaluation Stack

```mermaid
flowchart TD
    subgraph "Component Metrics"
        A[Retrieval Metrics] --> B[Recall@k, Precision@k]
        A --> C[MRR, NDCG]
    end
    
    subgraph "Benchmarks"
        D[MTEB] --> E[Model Selection]
        F[BEIR] --> G[Domain Transfer]
    end
    
    subgraph "End-to-End"
        H[RAG Metrics] --> I[Faithfulness]
        H --> J[Context Relevance]
        H --> K[Answer Relevance]
    end
    
    subgraph "Production"
        L[A/B Testing] --> M[Statistical Significance]
        N[Monitoring] --> O[Drift Detection]
    end
    
    B --> H
    C --> H
    E --> L
    G --> L
    I --> N
    J --> N
    K --> N
```

---

## Core Metrics at a Glance

| Metric | Measures | Range | Higher Is |
|--------|----------|-------|-----------|
| **Recall@k** | Relevant docs found in top-k | 0-1 | Better |
| **Precision@k** | Relevant ratio in top-k | 0-1 | Better |
| **MRR** | Rank of first relevant doc | 0-1 | Better |
| **NDCG@k** | Ranking quality with position weighting | 0-1 | Better |
| **Hit Rate** | Any relevant in top-k (binary) | 0-1 | Better |
| **Faithfulness** | Claims supported by context | 0-1 | Better |
| **Context Precision** | Retrieved contexts that are relevant | 0-1 | Better |
| **Answer Relevance** | Response addresses the query | 0-1 | Better |

---

## Evaluation Tools Landscape

| Tool | Focus | Open Source | Key Features |
|------|-------|-------------|--------------|
| **RAGAS** | RAG evaluation | ✅ Yes | Faithfulness, relevance, context metrics |
| **TruLens** | LLM app evaluation | ✅ Yes | Feedback functions, tracing |
| **DeepEval** | LLM testing | ✅ Yes | Unit tests for LLMs |
| **BEIR** | Retrieval benchmarks | ✅ Yes | 18 diverse datasets |
| **MTEB** | Embedding benchmarks | ✅ Yes | 56+ tasks, leaderboard |
| **LangSmith** | Observability | ❌ No | Tracing, evaluation |
| **Arize Phoenix** | ML observability | ✅ Yes | Embeddings viz, drift |

---

## Quick Start: Basic Evaluation

```python
from typing import List, Set

def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """Calculate Recall@k metric."""
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids
    return len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0

def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """Calculate Precision@k metric."""
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids
    return len(relevant_retrieved) / k if k > 0 else 0.0

def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

# Example usage
retrieved = ["doc_3", "doc_1", "doc_7", "doc_2", "doc_5"]
relevant = {"doc_1", "doc_2", "doc_6"}

print(f"Recall@3: {calculate_recall_at_k(retrieved, relevant, 3):.2f}")
print(f"Precision@3: {calculate_precision_at_k(retrieved, relevant, 3):.2f}")
print(f"MRR: {calculate_mrr(retrieved, relevant):.2f}")
```

**Output:**
```
Recall@3: 0.33
Precision@3: 0.33
MRR: 0.50
```

---

## Learning Path

### Recommended Order

1. **Start with metrics** — Understand what you're measuring
2. **Explore benchmarks** — See how models compare on standard tasks
3. **Build your dataset** — Create evaluation data for your domain
4. **Evaluate end-to-end** — Measure full RAG pipeline quality
5. **Set up A/B testing** — Compare changes with statistical rigor
6. **Implement monitoring** — Catch degradation in production

### Prerequisites

- Understanding of embeddings and vector search
- Familiarity with Python
- Basic statistics knowledge helpful

---

## Key Takeaways

By completing this lesson, you will be able to:

✅ **Calculate standard retrieval metrics** (Recall, Precision, MRR, NDCG)

✅ **Use MTEB and BEIR** to compare embedding models

✅ **Build custom evaluation datasets** for your domain

✅ **Evaluate RAG pipelines** with RAGAS faithfulness and relevance metrics

✅ **Design A/B tests** with proper statistical significance

✅ **Monitor retrieval quality** in production with drift detection

---

## Further Reading

- [RAGAS Documentation](https://docs.ragas.io/) — RAG evaluation framework
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Embedding model comparisons
- [BEIR Benchmark](https://github.com/beir-cellar/beir) — Zero-shot retrieval evaluation

---

**Next:** [Retrieval Quality Metrics](./01-retrieval-quality-metrics.md) — Master the fundamental metrics

---

[← Back to Cost Optimization](../17-cost-optimization/00-cost-optimization.md) | [Start Learning →](./01-retrieval-quality-metrics.md)
