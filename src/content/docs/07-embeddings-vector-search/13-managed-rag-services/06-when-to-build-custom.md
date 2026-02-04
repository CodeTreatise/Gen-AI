---
title: "When to Build Custom RAG"
---

# When to Build Custom RAG

## Introduction

While managed RAG services excel in simplicity, certain requirements demand the flexibility of a custom solution. Building your own RAG pipeline gives you complete control over every component—from chunking strategies to embedding models to search algorithms.

This lesson identifies scenarios where custom RAG is the better choice and outlines what building custom entails.

### What We'll Cover

- Requirements that necessitate custom RAG
- Technical drivers for customization
- Business and compliance drivers
- The custom RAG component stack
- Migration considerations from managed to custom

### Prerequisites

- Understanding of RAG architecture
- Familiarity with managed RAG limitations
- Basic knowledge of embeddings and vector databases

---

## When Custom RAG Becomes Necessary

### The Tipping Point

```
┌─────────────────────────────────────────────────────────────────┐
│              Managed → Custom Tipping Points                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MANAGED WORKS FINE                  NEED CUSTOM                │
│         │                                  │                    │
│         │                                  │                    │
│  ───────┼──────────────────────────────────┼────────────────▶  │
│         │                                  │              TIME  │
│         │                                  │                    │
│  Triggers for migration:                                        │
│                                                                 │
│  • Default chunking hurts retrieval quality                    │
│  • Need domain-specific embedding model                        │
│  • Multi-tenant isolation required                             │
│  • Data residency / compliance requirements                    │
│  • Cost exceeds custom infrastructure cost                     │
│  • Need features not available (BM25, custom reranking)       │
│  • Scale beyond service limits                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Requirements for Custom RAG

### 1. Custom Chunking Strategies

**Managed limitation:** Fixed chunking based on token count.

**When you need custom:**
- Document-structure-aware chunking (by section, heading)
- Code-aware chunking (by function, class)
- Table and figure handling
- Multi-modal content (images + text)

**Custom solution:**
```python
# Example: Structure-aware markdown chunking
def chunk_by_headers(markdown_text: str) -> list[dict]:
    """Chunk markdown by H2 headers, keeping context."""
    chunks = []
    current_h1 = ""
    current_chunk = ""
    
    for line in markdown_text.split('\n'):
        if line.startswith('# '):
            current_h1 = line[2:]
        elif line.startswith('## '):
            if current_chunk:
                chunks.append({
                    'content': current_chunk,
                    'metadata': {'h1': current_h1}
                })
            current_chunk = f"# {current_h1}\n\n{line}\n"
        else:
            current_chunk += line + '\n'
    
    if current_chunk:
        chunks.append({
            'content': current_chunk,
            'metadata': {'h1': current_h1}
        })
    
    return chunks
```

### 2. Domain-Specific Embeddings

**Managed limitation:** Fixed embedding model (text-embedding-3-large, gemini-embedding).

**When you need custom:**
- Legal, medical, or scientific terminology
- Non-English or multilingual content
- Code and technical documentation
- Domain where general embeddings underperform

**Custom solution:**
```python
# Use specialized embedding models
from sentence_transformers import SentenceTransformer

# Domain-specific options
legal_model = SentenceTransformer('legal-bert-base-uncased')
code_model = SentenceTransformer('microsoft/codebert-base')
multilingual = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Or fine-tune your own
from sentence_transformers import InputExample, losses

train_examples = [
    InputExample(texts=["query", "relevant doc"], label=1.0),
    InputExample(texts=["query", "irrelevant doc"], label=0.0),
]
# Fine-tune on your domain data
```

### 3. Hybrid Search (BM25 + Semantic)

**Managed limitation:** Some providers offer hybrid, but with fixed weights.

**When you need custom:**
- Control over keyword vs semantic balance
- BM25 tuning (k1, b parameters)
- Custom fusion algorithms (RRF, weighted)

**Custom solution:**
```python
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_search(query: str, corpus: list[str], embeddings: np.ndarray,
                  query_embedding: np.ndarray, alpha: float = 0.5) -> list[int]:
    """
    Combine BM25 and semantic search with configurable weights.
    
    alpha: Weight for semantic (1-alpha for BM25)
    """
    # BM25 search
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
    
    # Semantic search
    semantic_scores = np.dot(embeddings, query_embedding)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-6)
    
    # Combine with configurable alpha
    combined = alpha * semantic_norm + (1 - alpha) * bm25_norm
    
    return np.argsort(combined)[::-1].tolist()
```

### 4. Custom Reranking

**Managed limitation:** Provider-defined reranking (if any).

**When you need custom:**
- Cross-encoder reranking for precision
- Domain-specific relevance models
- Multi-stage retrieval pipelines

**Custom solution:**
```python
from sentence_transformers import CrossEncoder

# Cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    """Rerank candidates using cross-encoder."""
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in ranked_indices]
```

---

## Business Requirements for Custom RAG

### 1. Multi-Tenant Architecture

**Managed limitation:** Shared infrastructure, limited isolation options.

**When you need custom:**
- Per-customer data isolation
- Tenant-specific configurations
- Compliance with customer contracts
- Data sovereignty per tenant

**Custom architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Tenant Custom RAG                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Request: { tenant_id: "acme", query: "..." }                  │
│                      │                                          │
│                      ▼                                          │
│              ┌─────────────┐                                   │
│              │ Tenant      │                                   │
│              │ Router      │                                   │
│              └─────────────┘                                   │
│                      │                                          │
│         ┌────────────┼────────────┐                            │
│         ▼            ▼            ▼                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │ Acme     │ │ Beta     │ │ Gamma    │                       │
│  │ Index    │ │ Index    │ │ Index    │                       │
│  │          │ │          │ │          │                       │
│  │ Config:  │ │ Config:  │ │ Config:  │                       │
│  │ chunk:800│ │ chunk:500│ │ chunk:1k │                       │
│  │ model: A │ │ model: B │ │ model: A │                       │
│  └──────────┘ └──────────┘ └──────────┘                       │
│                                                                 │
│  Benefits:                                                      │
│  • Complete data isolation                                      │
│  • Per-tenant configuration                                     │
│  • Independent scaling                                          │
│  • Compliance per customer                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Data Residency & Compliance

**Managed limitation:** Data stored in provider's infrastructure.

**When you need custom:**
- GDPR data residency requirements
- HIPAA compliance (healthcare)
- FedRAMP requirements (government)
- Financial regulations (SOX, PCI-DSS)
- Customer-specific data handling

**Custom approach:**
```python
# Deploy vector database in compliant region
import weaviate

# EU data residency
eu_client = weaviate.Client(
    url="https://weaviate.eu-west-1.your-vpc.com",
    additional_headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
    }
)

# US data residency
us_client = weaviate.Client(
    url="https://weaviate.us-east-1.your-vpc.com",
    additional_headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
    }
)

def get_client_for_region(user_region: str):
    """Route to compliant infrastructure based on user region."""
    if user_region in ['DE', 'FR', 'ES', 'IT']:  # EU
        return eu_client
    return us_client
```

### 3. Cost Optimization at Scale

**Managed limitation:** Pay-per-storage or pay-per-embedding at provider rates.

**When custom saves money:**
- Large datasets (> 10 GB)
- High query volume (> 10K/day)
- Long-term storage needs
- Predictable workloads

**Cost comparison example:**

| Scale | Managed Cost/Month | Custom Cost/Month |
|-------|-------------------|-------------------|
| 1 GB, 1K queries | ~$5 | ~$50 (over-provisioned) |
| 10 GB, 10K queries | ~$35 | ~$100 |
| 100 GB, 100K queries | ~$350 | ~$200 |
| 1 TB, 1M queries | ~$3,500 | ~$500 |

> **Note:** Custom infrastructure has fixed costs that only pay off at scale.

---

## The Custom RAG Component Stack

### Components You'll Build or Integrate

```
┌─────────────────────────────────────────────────────────────────┐
│              Custom RAG Component Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: DOCUMENT PROCESSING                                   │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Parser: PyMuPDF, python-docx, BeautifulSoup              │
│  │ • Chunker: LangChain, custom logic                         │
│  │ • Cleaner: Text normalization, deduplication               │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│  LAYER 2: EMBEDDING                                             │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Model: OpenAI, Cohere, SentenceTransformers, fine-tuned  │
│  │ • Batching: Efficient bulk embedding                       │
│  │ • Caching: Avoid re-embedding unchanged content            │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│  LAYER 3: VECTOR STORAGE                                        │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Database: Pinecone, Weaviate, Qdrant, pgvector, Milvus   │
│  │ • Indexing: HNSW, IVF configuration                        │
│  │ • Metadata: Schema design, filtering indexes               │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│  LAYER 4: RETRIEVAL                                             │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Search: Vector similarity, hybrid, filtered              │
│  │ • Reranking: Cross-encoder, LLM-based                      │
│  │ • Fusion: RRF, weighted combination                        │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│  LAYER 5: GENERATION                                            │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Context: Prompt assembly, token management               │
│  │ • LLM: OpenAI, Anthropic, open-source                      │
│  │ • Citations: Source tracking, quote extraction             │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│  LAYER 6: OPERATIONS                                            │
│  ┌─────────────────────────────────────────────────────────────┐
│  │ • Monitoring: Latency, quality, costs                      │
│  │ • Evaluation: Relevance metrics, A/B testing               │
│  │ • Updates: Incremental indexing, version management        │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Effort Estimate

| Component | Build Time | Maintain Effort |
|-----------|------------|-----------------|
| Document processing | 1-2 weeks | Low |
| Embedding pipeline | 1 week | Low |
| Vector database setup | 1-2 weeks | Medium |
| Retrieval logic | 2-3 weeks | Medium |
| Generation & citations | 1-2 weeks | Low |
| Operations & monitoring | 2-4 weeks | High |
| **Total** | **8-14 weeks** | **Ongoing** |

---

## Migration Path: Managed to Custom

### Step-by-Step Migration

```
┌─────────────────────────────────────────────────────────────────┐
│              Migration Roadmap                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: PREPARATION (Week 1-2)                                │
│  ──────────────────────────────                                 │
│  ☐ Export document inventory from managed service               │
│  ☐ Download original source files                               │
│  ☐ Document current chunking/retrieval behavior                │
│  ☐ Establish quality baseline (test queries + expected results)│
│                                                                 │
│  PHASE 2: INFRASTRUCTURE (Week 3-4)                             │
│  ────────────────────────────────                               │
│  ☐ Provision vector database                                    │
│  ☐ Set up embedding pipeline                                    │
│  ☐ Configure monitoring/logging                                 │
│  ☐ Deploy in staging environment                                │
│                                                                 │
│  PHASE 3: RE-INDEXING (Week 5-6)                                │
│  ────────────────────────────────                               │
│  ☐ Process documents with custom chunking                       │
│  ☐ Generate embeddings (custom or managed API)                  │
│  ☐ Load into vector database                                    │
│  ☐ Verify document counts match                                 │
│                                                                 │
│  PHASE 4: VALIDATION (Week 7-8)                                 │
│  ───────────────────────────────                                │
│  ☐ Run baseline test queries                                    │
│  ☐ Compare retrieval quality                                    │
│  ☐ Tune chunking/search parameters                              │
│  ☐ Performance testing                                          │
│                                                                 │
│  PHASE 5: CUTOVER (Week 9-10)                                   │
│  ─────────────────────────────                                  │
│  ☐ Update application to use custom pipeline                    │
│  ☐ Monitor for issues                                           │
│  ☐ Decommission managed service                                 │
│  ☐ Update documentation                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Migration Risks

| Risk | Mitigation |
|------|------------|
| Quality regression | Extensive A/B testing before cutover |
| Data loss | Verify document counts, checksums |
| Performance issues | Load testing in staging |
| Cost overrun | Budget buffer, phased rollout |
| Timeline slip | Parallel run period, rollback plan |

---

## Decision Checklist

### You Should Build Custom If:

| Requirement | Weight |
|-------------|--------|
| Need custom embedding model | High |
| Multi-tenant isolation required | High |
| Data residency compliance | High |
| Cost optimization at scale (> 10 GB) | Medium |
| Need specific hybrid search tuning | Medium |
| Custom reranking required | Medium |
| Complex document structure | Medium |
| Team has ML/DevOps expertise | Enabler |

**If 2+ High or 3+ Medium requirements → Build Custom**

---

## Summary

✅ **Custom RAG** is necessary for domain-specific embeddings, multi-tenancy, and compliance  
✅ **Technical drivers** include custom chunking, hybrid search control, and reranking  
✅ **Business drivers** include data residency, tenant isolation, and cost at scale  
✅ **Expect 8-14 weeks** to build a production-ready custom RAG pipeline  
✅ **Migration from managed** requires careful planning and quality validation

---

**Decision Rule:**

> If you need custom embeddings, strict multi-tenancy, data residency compliance, or operate at > 10 GB scale—**build custom RAG**.

---

**Next:** [Limitations of Managed RAG →](./07-limitations.md)

---

<!-- 
Sources Consulted:
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Sentence Transformers: https://www.sbert.net/
- Weaviate Documentation: https://weaviate.io/developers/weaviate
- Pinecone Documentation: https://docs.pinecone.io/
-->
