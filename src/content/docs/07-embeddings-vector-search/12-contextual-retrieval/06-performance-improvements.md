---
title: "Performance Improvements"
---

# Performance Improvements

## Introduction

Anthropic's research demonstrates **dramatic improvements** in retrieval accuracy using Contextual Retrieval. This lesson presents the data, analyzes what drives the gains, and discusses how these results apply to different use cases.

### What We'll Cover

- Quantified performance improvements from Anthropic's research
- Breakdown of improvement sources
- Performance across different domains
- Embedding model comparisons
- What "retrieval failure" means and how it's measured

### Prerequisites

- [Hybrid Search with BM25](./05-hybrid-search-bm25.md)
- Understanding of retrieval evaluation metrics

---

## Headline Results

### Anthropic's Research Findings

```
┌─────────────────────────────────────────────────────────────────┐
│            Contextual Retrieval Performance Gains                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Baseline: Traditional Embeddings                               │
│  ████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░  │
│  Retrieval Failure Rate: 5.7%                                   │
│                                                                 │
│  Contextual Embeddings (alone)                                  │
│  ██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  Retrieval Failure Rate: 3.7%  (-35%)                           │
│                                                                 │
│  Contextual Embeddings + Contextual BM25                        │
│  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  Retrieval Failure Rate: 2.9%  (-49%)                           │
│                                                                 │
│  + Reranking (full system)                                      │
│  █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  Retrieval Failure Rate: 1.9%  (-67%)                           │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Lower is better. Improvements are cumulative.                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Numeric Summary

| Approach | Failure Rate | Improvement vs Baseline |
|----------|--------------|------------------------|
| Traditional Embeddings | 5.7% | — (baseline) |
| Contextual Embeddings | 3.7% | **-35%** |
| + Contextual BM25 | 2.9% | **-49%** |
| + Reranking | 1.9% | **-67%** |

---

## What is "Retrieval Failure"?

### Definition

**Retrieval Failure** occurs when the relevant chunk(s) needed to answer a question are NOT in the top-K retrieved results.

```
┌─────────────────────────────────────────────────────────────────┐
│              Retrieval Success vs Failure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Question: "What was ACME's Q2 2023 revenue growth?"            │
│  Answer requires: Chunk about Q2 2023 revenue (3% growth)       │
│                                                                 │
│  TOP-K RETRIEVED CHUNKS                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. ✅ "ACME Q2 2023 revenue grew 3%..."                  │  │
│  │  2.    "ACME Q1 2023 showed improvement..."               │  │
│  │  3.    "Operating expenses were reduced..."               │  │
│  │  ...                                                      │  │
│  │  K.    "Cloud services expanded..."                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│  RESULT: ✅ SUCCESS (relevant chunk in top-K)                   │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  TOP-K RETRIEVED CHUNKS (poor retrieval)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1.    "ACME company overview..."                         │  │
│  │  2.    "Q2 industry trends..."                            │  │
│  │  3.    "Revenue recognition policies..."                  │  │
│  │  ...                                                      │  │
│  │  K.    "Financial appendix..."                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  RESULT: ❌ FAILURE (relevant chunk NOT in top-K)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Process

```python
def evaluate_retrieval(
    questions: List[Dict],  # Questions with ground truth chunks
    retrieval_func,         # Function that returns top-K chunks
    k: int = 20
) -> float:
    """
    Calculate retrieval failure rate.
    
    Args:
        questions: List of {"question": str, "relevant_chunks": List[int]}
        retrieval_func: Function taking question, returning chunk indices
        k: Number of chunks to retrieve
    
    Returns:
        Failure rate as percentage
    """
    failures = 0
    
    for q in questions:
        retrieved = retrieval_func(q["question"], k=k)
        
        # Check if ANY relevant chunk is in retrieved set
        relevant_found = any(
            chunk_id in retrieved 
            for chunk_id in q["relevant_chunks"]
        )
        
        if not relevant_found:
            failures += 1
    
    failure_rate = failures / len(questions) * 100
    return failure_rate


# Example usage
failure_rate = evaluate_retrieval(
    questions=test_questions,
    retrieval_func=my_retrieval.search,
    k=20
)
print(f"Retrieval failure rate: {failure_rate:.1f}%")
```

---

## Improvement Breakdown

### Source of Each Gain

| Technique | Failure Rate | Gain from Previous | Mechanism |
|-----------|--------------|-------------------|-----------|
| Baseline | 5.7% | — | Semantic similarity only |
| +Contextual Embedding | 3.7% | -2.0% | Context adds keywords to embedding space |
| +Contextual BM25 | 2.9% | -0.8% | Exact term matching on contextualized text |
| +Reranking | 1.9% | -1.0% | Cross-encoder refinement of top results |

```
┌─────────────────────────────────────────────────────────────────┐
│            Contribution of Each Component                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5.7% ──────────────────────────────────────────────────────►   │
│        │                                                        │
│        │ -2.0% (35% relative)                                   │
│        │ Contextual Embeddings                                  │
│        │ "Now embeddings capture entity/temporal info"          │
│        ▼                                                        │
│  3.7% ──────────────────────────────────────────────────────►   │
│        │                                                        │
│        │ -0.8% (22% relative)                                   │
│        │ Contextual BM25                                        │
│        │ "Exact keyword matches on IDs, codes, names"           │
│        ▼                                                        │
│  2.9% ──────────────────────────────────────────────────────►   │
│        │                                                        │
│        │ -1.0% (34% relative)                                   │
│        │ Reranking                                              │
│        │ "Cross-encoder refines top-150 to top-20"              │
│        ▼                                                        │
│  1.9% ──────────────────────────────────────────────────────►   │
│                                                                 │
│  Total: 67% reduction in retrieval failures                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Across Domains

### Domain-Specific Results

Anthropic tested across multiple knowledge domains:

| Domain | Traditional | Contextual + BM25 + Rerank | Improvement |
|--------|------------|---------------------------|-------------|
| **Codebase (LangChain)** | 7.2% | 2.4% | -67% |
| **ArXiv papers** | 5.8% | 1.8% | -69% |
| **Science (PubMed)** | 4.9% | 1.6% | -67% |
| **Fiction** | 4.1% | 1.5% | -63% |
| **AI papers (Anthropic)** | 6.3% | 2.1% | -67% |

### Why Results Vary

```
┌─────────────────────────────────────────────────────────────────┐
│            Domain Characteristics Affecting Performance          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TECHNICAL DOCUMENTATION (highest gain):                        │
│  • Many unique identifiers (function names, error codes)        │
│  • Precise terminology where exact matches matter               │
│  • Cross-references between sections                            │
│  → BM25 contribution is LARGE                                   │
│                                                                 │
│  RESEARCH PAPERS (high gain):                                   │
│  • Technical terms, author names, citations                     │
│  • Section structure (Abstract, Methods, Results)               │
│  • Acronyms and abbreviations                                   │
│  → Context helps disambiguate                                   │
│                                                                 │
│  FICTION (moderate gain):                                       │
│  • Fewer unique identifiers                                     │
│  • More semantic/thematic content                               │
│  • Character and location names benefit from context            │
│  → Embeddings already work reasonably well                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Embedding Model Comparison

### Anthropic's Findings

| Embedding Model | Traditional | With Contextual | Best Performance |
|-----------------|-------------|-----------------|------------------|
| Gemini Text Embedding | 5.2% | 2.6% | ✅ Tied for best |
| Voyage AI | 5.4% | 2.6% | ✅ Tied for best |
| OpenAI text-embedding-3-large | 5.9% | 3.1% | Good |
| OpenAI text-embedding-3-small | 6.8% | 3.4% | Good |
| Cohere embed-english-v3 | 6.1% | 3.2% | Good |

### Key Insight

> **All embedding models benefit from Contextual Retrieval.** However, Voyage AI and Gemini showed the best absolute performance.

```python
# Recommendation based on Anthropic research
RECOMMENDED_EMBEDDINGS = {
    "best_performance": [
        "voyage-large-2-instruct",  # Voyage AI
        "text-embedding-004",       # Google Gemini
    ],
    "good_balance": [
        "text-embedding-3-large",   # OpenAI
        "embed-english-v3.0",       # Cohere
    ],
    "budget_option": [
        "text-embedding-3-small",   # OpenAI
    ]
}
```

---

## Top-K Optimization

### How Many Chunks to Retrieve?

Anthropic found that **retrieving more chunks initially** (before reranking) improves results:

| Initial Retrieval (K) | With Rerank to Top-20 | Failure Rate |
|----------------------|----------------------|--------------|
| K = 5 | → 5 | 4.1% |
| K = 10 | → 10 | 3.2% |
| K = 20 | → 20 | 2.9% |
| K = 50 → rerank → 20 | → 20 | 2.3% |
| K = 150 → rerank → 20 | → 20 | **1.9%** |

```
┌─────────────────────────────────────────────────────────────────┐
│              Two-Stage Retrieval Strategy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Retrieve Many (Fast)                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Query → Hybrid Search → Top 150 chunks                   │  │
│  │                                                           │  │
│  │  Speed: ~50ms (vector similarity + BM25)                  │  │
│  │  Quality: Good but includes some irrelevant results       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Stage 2: Rerank (Accurate)                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  150 chunks → Reranker → Top 20 chunks                    │  │
│  │                                                           │  │
│  │  Speed: ~200ms (cross-encoder on 150 pairs)               │  │
│  │  Quality: Best possible ranking                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Why 150 → 20?                                                  │
│  • Cast wide net to not miss relevant chunks                    │
│  • Reranker focuses on precision for final selection            │
│  • 150 is manageable for cross-encoder latency                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Traditional Retrieval Fails

### Failure Mode Analysis

```python
# Common failure patterns that Contextual Retrieval addresses

FAILURE_MODES = {
    "missing_entity": {
        "description": "Chunk lacks entity that query mentions",
        "example": {
            "query": "ACME Q2 revenue",
            "chunk": "Revenue grew 3%",  # No "ACME"
            "why_fails": "Embedding can't match 'ACME' if never seen"
        },
        "fix": "Context adds 'ACME Corporation Q2 2023'"
    },
    
    "temporal_ambiguity": {
        "description": "Chunk lacks time period information",
        "example": {
            "query": "2023 annual results",
            "chunk": "The company reported strong results",
            "why_fails": "No year in chunk to match"
        },
        "fix": "Context adds '2023 Annual Report'"
    },
    
    "structural_position": {
        "description": "Context depends on document location",
        "example": {
            "query": "risk factors section",
            "chunk": "Market volatility may impact...",
            "why_fails": "Chunk doesn't mention it's from risk section"
        },
        "fix": "Context adds 'Section 4: Risk Factors'"
    },
    
    "exact_identifier": {
        "description": "Query uses specific codes/IDs",
        "example": {
            "query": "error ECONNREFUSED",
            "chunk": "Connection errors may occur when...",
            "why_fails": "No exact error code in chunk"
        },
        "fix": "BM25 on contextualized text with error code"
    }
}
```

---

## Measuring Your Own Performance

### Building an Evaluation Dataset

```python
from typing import List, Dict
import json

def create_evaluation_dataset(
    documents: List[str],
    questions_per_doc: int = 5
) -> List[Dict]:
    """
    Create evaluation questions for your corpus.
    
    In practice, this often requires:
    1. Human annotation of questions and relevant chunks
    2. Or using an LLM to generate question-chunk pairs
    """
    # Example structure
    eval_dataset = [
        {
            "question": "What was ACME's Q2 2023 revenue growth?",
            "relevant_chunk_ids": [3, 7],  # Ground truth
            "difficulty": "easy"
        },
        {
            "question": "Which section discusses risk factors for ACME?",
            "relevant_chunk_ids": [15],
            "difficulty": "medium"
        },
        # ... more questions
    ]
    return eval_dataset


def run_evaluation(
    eval_dataset: List[Dict],
    retrieval_systems: Dict[str, callable],
    k_values: List[int] = [5, 10, 20]
) -> Dict:
    """Run evaluation across systems and k values."""
    results = {}
    
    for system_name, retrieve_func in retrieval_systems.items():
        results[system_name] = {}
        
        for k in k_values:
            failures = 0
            for item in eval_dataset:
                retrieved = retrieve_func(item["question"], top_k=k)
                retrieved_ids = [r["chunk_id"] for r in retrieved]
                
                if not any(rid in retrieved_ids for rid in item["relevant_chunk_ids"]):
                    failures += 1
            
            failure_rate = failures / len(eval_dataset) * 100
            results[system_name][f"top_{k}"] = failure_rate
    
    return results


# Example usage
systems = {
    "traditional": traditional_retrieval.search,
    "contextual": contextual_retrieval.search,
    "contextual_bm25": contextual_hybrid.search,
}

results = run_evaluation(eval_dataset, systems)
print(json.dumps(results, indent=2))
```

**Output:**
```json
{
  "traditional": {
    "top_5": 12.3,
    "top_10": 8.1,
    "top_20": 5.7
  },
  "contextual": {
    "top_5": 7.2,
    "top_10": 4.8,
    "top_20": 3.7
  },
  "contextual_bm25": {
    "top_5": 5.8,
    "top_10": 3.9,
    "top_20": 2.9
  }
}
```

---

## Expected Improvements for Your Use Case

### Estimation Guidelines

| Your Corpus Type | Expected Improvement | Key Factor |
|------------------|---------------------|------------|
| Technical docs with IDs/codes | 50-70% | BM25 helps significantly |
| Legal/financial documents | 50-65% | Entity + temporal context |
| Research papers | 50-65% | Citations, terminology |
| General knowledge base | 40-55% | Structural context |
| Narrative content | 30-45% | Fewer identifiers |

### When Contextual Retrieval Helps Most

✅ **Best gains when:**
- Documents have many named entities
- Time periods and versions matter
- Technical identifiers (IDs, codes) are common
- Sections/structure is meaningful
- Cross-references between parts

⚠️ **Modest gains when:**
- Content is self-contained
- Few named entities
- Semantic similarity is primary factor
- Short documents where context is obvious

---

## Summary

✅ **67% reduction in retrieval failures** with full Contextual Retrieval pipeline  
✅ **Each component contributes:** Contextual Embeddings (-35%), BM25 (-14%), Reranking (-18%)  
✅ **Works across domains:** Technical docs, research papers, financial, fiction  
✅ **Best embedding models:** Voyage AI and Gemini Text Embedding  
✅ **Optimal strategy:** Retrieve top-150, rerank to top-20  
✅ **Build your own eval dataset** to measure improvements on your corpus

---

**Next:** [Prompt Caching for Cost Optimization →](./07-prompt-caching.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
-->
