---
title: "Reranking & Score Fusion"
---

# Reranking & Score Fusion

## Section Overview

Initial retrieval is fast but imprecise. When you search a vector database for the top 100 documents, many will be relevant—but they won't be perfectly ordered by relevance. The document at position 50 might actually be more relevant than the one at position 5. This is where **reranking** transforms good retrieval into great retrieval.

Reranking uses more sophisticated (and slower) models to re-score and re-order retrieved documents. Combined with **score fusion** techniques that merge results from multiple retrieval methods, these techniques can improve RAG precision by **67%** according to Anthropic's research.

```mermaid
flowchart LR
    Q[Query] --> RETRIEVE[Initial Retrieval]
    RETRIEVE --> |Top 100-150| FUSION[Score Fusion]
    FUSION --> RERANK[Cross-Encoder Rerank]
    RERANK --> |Top 10-20| CONTEXT[Context for LLM]
    CONTEXT --> GEN[Generate Response]
```

---

## What You'll Learn

| Lesson | Topic | Description |
|--------|-------|-------------|
| 01 | [Why Reranking Matters](./01-why-reranking-matters.md) | The precision problem and two-stage retrieval pattern |
| 02 | [Cross-Encoder Reranking](./02-cross-encoder-reranking.md) | How cross-encoders achieve higher accuracy |
| 03 | [Cohere Rerank](./03-cohere-rerank.md) | Cohere Rerank v4.0 implementation and best practices |
| 04 | [Voyage Reranker](./04-voyage-reranker.md) | Voyage AI rerank-2.5 with instruction-following |
| 05 | [Reciprocal Rank Fusion](./05-reciprocal-rank-fusion.md) | RRF algorithm for combining ranked lists |
| 06 | [Distribution-Based Score Fusion](./06-distribution-based-score-fusion.md) | DBSF for normalizing and combining scores |
| 07 | [Hybrid Reranking Pipeline](./07-hybrid-reranking-pipeline.md) | Complete BM25 + vector + fusion + reranking pipeline |
| 08 | [Evaluation & Optimization](./08-evaluation-optimization.md) | Metrics, best practices, and cost optimization |

---

## Prerequisites

Before starting this section, you should understand:

- **Embedding-based retrieval** — Vector similarity search basics
- **Hybrid search concepts** — Combining keyword and semantic search
- **Basic RAG implementation** — Query → retrieve → generate pattern

---

## Key Concepts

### Bi-Encoder vs Cross-Encoder

| Approach | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| **Bi-encoder** | Fast (~1ms per 1M docs) | Good | Initial retrieval |
| **Cross-encoder** | Slow (~50ms per doc) | Excellent | Reranking top candidates |

### Score Fusion Techniques

- **RRF (Reciprocal Rank Fusion)** — Combines by rank position, not raw scores
- **DBSF (Distribution-Based Score Fusion)** — Normalizes scores to [0,1] before combining

### Leading Reranking APIs

| Provider | Model | Context | Key Feature |
|----------|-------|---------|-------------|
| Cohere | rerank-v4.0-pro | 32K tokens | Structured data (YAML/JSON) |
| Cohere | rerank-v4.0-fast | 32K tokens | Lower latency |
| Voyage | rerank-2.5 | 32K tokens | Instruction-following |
| Voyage | rerank-2.5-lite | 32K tokens | Optimized latency |

---

## Learning Path

1. **Start with theory** — Understand why reranking improves precision (Lesson 01-02)
2. **Learn the APIs** — Implement Cohere and Voyage rerankers (Lesson 03-04)
3. **Master fusion** — Combine results from multiple retrievers (Lesson 05-06)
4. **Build pipelines** — Create production-ready hybrid systems (Lesson 07)
5. **Optimize** — Measure impact and reduce costs (Lesson 08)

---

## Quick Start

If you want to jump straight to implementation:

```python
import cohere

co = cohere.ClientV2()

# Rerank retrieved documents
results = co.rerank(
    model="rerank-v4.0-pro",
    query="What is the capital of France?",
    documents=["Paris is the capital of France.", "London is in England."],
    top_n=1
)

print(results.results[0].relevance_score)  # 0.98+
```

---

**Next:** [Why Reranking Matters](./01-why-reranking-matters.md)
