---
title: "Limitations of Managed RAG"
---

# Limitations of Managed RAG

## Introduction

Managed RAG services trade control for convenience. Understanding these limitations is essential—not to avoid managed services entirely, but to make informed decisions about when the trade-offs matter and when they don't.

This lesson catalogs the key limitations of managed RAG services and provides guidance on working around them where possible.

### What We'll Cover

- Control limitations (chunking, embeddings, search)
- Vendor lock-in concerns
- Rate limits and quotas
- Feature constraints
- Workarounds and mitigations

### Prerequisites

- Familiarity with OpenAI Vector Stores and Gemini File Search
- Understanding of custom RAG components

---

## Control Limitations

### 1. Fixed Embedding Models

**The Limitation:**

Managed services use their own embedding models—you cannot substitute them.

| Provider | Embedding Model | Dimensions |
|----------|-----------------|------------|
| OpenAI | text-embedding-3-large | 256 |
| Gemini | gemini-embedding-001 | — |

**Why This Matters:**

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding Model Impact                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  General-purpose embeddings work well for:                      │
│  ✅ Common language patterns                                    │
│  ✅ General knowledge queries                                   │
│  ✅ Standard document types                                     │
│                                                                 │
│  General-purpose embeddings struggle with:                      │
│  ❌ Domain-specific terminology (legal, medical, scientific)   │
│  ❌ Code and technical symbols                                  │
│  ❌ Specialized jargon and acronyms                             │
│  ❌ Non-English languages (variable quality)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Real-World Example:**

```python
# Medical terminology challenge
queries = [
    "What are the contraindications for ACE inhibitors?",
    "Describe the mechanism of action of SSRIs",
    "When is thrombolysis indicated?"
]

# General embeddings may not understand:
# - ACE = Angiotensin-Converting Enzyme
# - SSRIs = Selective Serotonin Reuptake Inhibitors
# - Clinical context of "contraindications"

# Domain-specific embeddings would perform better
```

**Workaround:**
- Use managed RAG for prototyping
- If retrieval quality suffers, migrate to custom with domain-specific embeddings

### 2. Limited Chunking Control

**The Limitation:**

Chunking options are restricted to token-based strategies.

| Provider | Chunking Options |
|----------|------------------|
| OpenAI | auto, static (token-based) |
| Gemini | white_space_config (token-based) |

**What's Missing:**

| Chunking Strategy | Available? |
|-------------------|------------|
| Token-based | ✅ Yes |
| Sentence-based | ❌ No |
| Paragraph-based | ❌ No |
| Semantic-based | ❌ No |
| Structure-aware (headers, sections) | ❌ No |
| Code-aware (functions, classes) | ❌ No |
| Table-preserving | ❌ No |

**Why This Matters:**

```
┌─────────────────────────────────────────────────────────────────┐
│              Chunking Impact on Retrieval                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DOCUMENT:                                                      │
│  ┌──────────────────────────────────────────┐                  │
│  │ # Introduction                           │                  │
│  │ This document describes our API.         │                  │
│  │                                          │                  │
│  │ # Authentication                         │ ← Important      │
│  │ All requests require an API key.         │    section       │
│  │ The key should be passed in the header.  │                  │
│  │                                          │                  │
│  │ ## OAuth Flow                            │                  │
│  │ For user-level access, use OAuth 2.0.    │                  │
│  │ The flow involves three steps:           │                  │
│  │ 1. Redirect to authorization URL         │                  │
│  │ 2. User grants permission                │                  │
│  │ 3. Exchange code for token               │                  │
│  │                                          │                  │
│  │ # Rate Limits                            │ ← Different      │
│  │ Standard tier: 100 requests/minute       │    topic         │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
│  TOKEN-BASED CHUNKING (what managed services do):              │
│  • May split "Authentication" section across chunks            │
│  • May combine unrelated sections in one chunk                 │
│  • Loses document structure context                            │
│                                                                 │
│  STRUCTURE-AWARE CHUNKING (custom only):                       │
│  • Keeps "Authentication" section together                     │
│  • Preserves heading hierarchy                                 │
│  • Better retrieval for section-specific queries               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Workaround:**
- Pre-process documents before upload (split into section-based files)
- Use larger chunks to capture more context
- Accept some retrieval quality trade-off for simplicity

### 3. Limited Search Customization

**The Limitation:**

Search algorithms and ranking are mostly fixed.

| Feature | OpenAI | Gemini |
|---------|--------|--------|
| Hybrid search | ✅ Built-in | ❌ Semantic only |
| BM25 tuning | ❌ No | ❌ No |
| Custom ranker | ❌ No | ❌ No |
| Semantic weight | ❌ Fixed | N/A |
| Score threshold | ✅ Yes | ❌ No |

**Why This Matters:**

Different use cases need different search behaviors:

| Use Case | Ideal Search | Managed Support |
|----------|--------------|-----------------|
| Keyword-heavy (part numbers, codes) | Strong BM25 | Limited |
| Conceptual questions | Strong semantic | ✅ Good |
| Mixed queries | Tunable hybrid | Limited |
| Precision-critical | Custom reranking | ❌ No |

---

## Vendor Lock-In

### Data Portability

**The Challenge:**

Your data becomes tied to the provider's ecosystem.

```
┌─────────────────────────────────────────────────────────────────┐
│              What You Can and Cannot Export                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ EXPORTABLE                     ❌ NOT EXPORTABLE            │
│  ────────────                      ─────────────────            │
│  • Original source files           • Generated embeddings       │
│  • File metadata                   • Chunking decisions         │
│  • Custom metadata you added       • Search indices             │
│                                    • Provider-specific IDs      │
│                                                                 │
│  Impact: Migration requires complete re-processing              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Migration Costs

| Action | Effort |
|--------|--------|
| Download source files | Low (if you kept originals) |
| Re-chunk documents | Medium (implement chunking logic) |
| Re-embed content | High (API costs + time) |
| Re-index in new system | Medium (vector DB setup) |
| Update application code | Medium (API changes) |
| Validate quality | High (testing + tuning) |

### API Dependency

**The Risk:**

Your application depends on APIs that can change.

```python
# Your code today
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "file_search", "vector_store_ids": [...]}]
)

# If OpenAI changes the API:
# - You must update your code
# - Behavior may change subtly
# - Deprecations require migration
```

**Mitigation:**
- Abstract provider-specific code behind interfaces
- Monitor deprecation announcements
- Budget time for API updates

---

## Rate Limits and Quotas

### OpenAI Limits

| Resource | Limit |
|----------|-------|
| Max file size | 512 MB |
| Max tokens per file | 5,000,000 |
| Vector stores per organization | Varies by tier |
| Files per vector store | Varies by tier |
| Concurrent uploads | Limited |

### Gemini Limits

| Resource | Free Tier | Tier 1 | Tier 2 | Tier 3 |
|----------|-----------|--------|--------|--------|
| Storage | 1 GB | 10 GB | 100 GB | 1 TB |
| Max file size | 100 MB | 100 MB | 100 MB | 100 MB |
| Requests/minute | 15 | 1000 | 2000 | 4000 |

### Practical Impact

```
┌─────────────────────────────────────────────────────────────────┐
│              Rate Limit Scenarios                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario 1: Bulk Upload                                        │
│  ─────────────────────────                                      │
│  • 1000 files to upload                                         │
│  • Rate limited to N concurrent                                 │
│  • May take hours to complete                                   │
│  • No parallel processing control                               │
│                                                                 │
│  Scenario 2: High Query Volume                                  │
│  ────────────────────────────                                   │
│  • Peak traffic spike                                           │
│  • Hit queries/minute limit                                     │
│  • Requests queued or rejected                                  │
│  • No burst capacity negotiation                                │
│                                                                 │
│  Scenario 3: Large Document                                     │
│  ────────────────────────────                                   │
│  • 600 MB PDF (over 512 MB limit)                              │
│  • Cannot upload to OpenAI                                      │
│  • Must split document manually                                 │
│  • Lose cross-section retrieval                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Workarounds:**
- Pre-split large documents
- Implement client-side rate limiting
- Use queuing for bulk operations
- Monitor usage to avoid limits

---

## Feature Constraints

### Features You Cannot Have

| Feature | Why It Matters | Alternative |
|---------|----------------|-------------|
| Custom embeddings | Domain-specific quality | Build custom |
| BM25 tuning | Keyword search precision | Build custom |
| Cross-encoder reranking | Answer precision | Build custom |
| Multi-vector representations | ColBERT-style retrieval | Build custom |
| Query expansion | Better recall | Build custom |
| Negative examples | Exclude irrelevant results | Metadata filtering (limited) |

### Features That May Lag

Managed services may not immediately support cutting-edge techniques:

| Technique | Status in Managed |
|-----------|-------------------|
| Contextual embeddings | ❌ Not available |
| Late interaction (ColBERT) | ❌ Not available |
| Hypothetical document embeddings (HyDE) | ❌ Not available |
| Multi-hop retrieval | ❌ Not available |
| Learned sparse representations | ❌ Not available |

---

## Transparency Limitations

### Black Box Behavior

**What You Don't Know:**

```
┌─────────────────────────────────────────────────────────────────┐
│              Hidden Implementation Details                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CHUNKING                                                       │
│  • Exact algorithm used?                   ❓ Unknown           │
│  • How are overlaps handled?               ❓ Unknown           │
│  • Are there preprocessing steps?          ❓ Unknown           │
│                                                                 │
│  EMBEDDING                                                      │
│  • Exact model version?                    Partially known      │
│  • Any preprocessing/normalization?        ❓ Unknown           │
│  • Truncation behavior for long chunks?    ❓ Unknown           │
│                                                                 │
│  SEARCH                                                         │
│  • BM25 parameters (k1, b)?               ❓ Unknown           │
│  • Semantic/keyword weight balance?        ❓ Unknown           │
│  • Reranking algorithm details?            ❓ Unknown           │
│  • Query rewriting rules?                  ❓ Unknown           │
│                                                                 │
│  Impact: Hard to debug retrieval quality issues                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Debugging Challenges

```python
# When retrieval quality is poor, you can check:

# ✅ Available debugging
- Raw chunks returned (with include parameter)
- Relevance scores
- Which files were searched
- Citation quotes

# ❌ Not available for debugging
- Why a chunk was ranked higher than another
- What query rewriting was applied
- BM25 vs semantic contribution
- Why relevant content wasn't retrieved
```

---

## Cost Predictability

### Variable Costs

**OpenAI:**
- Storage: $0.10/GB/day after free tier
- Search: Included in token usage (but varies with results)
- Uncertainty: Token costs depend on retrieved content length

**Gemini:**
- Storage: Free
- Embedding at index: $0.15/1M tokens (one-time)
- Query: Normal context pricing for retrieved tokens
- Uncertainty: Retrieved token count varies

### Surprise Scenarios

| Scenario | Cost Surprise |
|----------|---------------|
| Data grows faster than expected | Storage costs spike (OpenAI) |
| More queries than projected | Token costs increase |
| Frequent reindexing | Re-embedding costs (Gemini) |
| Long documents retrieved | Context token costs spike |

---

## Mitigation Strategies

### For Control Limitations

| Limitation | Mitigation |
|------------|------------|
| Fixed embeddings | Accept for prototyping; plan migration path |
| Limited chunking | Pre-process documents into logical units |
| Fixed search | Use metadata filtering creatively |
| No reranking | Post-process results in your application |

### For Vendor Lock-In

| Risk | Mitigation |
|------|------------|
| API changes | Abstract provider behind interface |
| Data locked | Keep original source files |
| Migration difficulty | Build evaluation dataset for comparison |
| Single provider | Evaluate both OpenAI and Gemini |

### For Rate Limits

| Limit Type | Mitigation |
|------------|------------|
| File size | Pre-split large documents |
| Upload rate | Queue with backoff |
| Query rate | Client-side rate limiting |
| Storage | Monitor and prune unused |

### For Feature Gaps

| Gap | Mitigation |
|-----|------------|
| Custom reranking | Post-process top N results |
| Query expansion | Implement client-side |
| Advanced retrieval | Accept limitations or go custom |

---

## Decision Framework: Are Limitations Acceptable?

### Acceptable When

- Building proof of concept
- Data is < 1 GB
- Default quality is sufficient
- Time-to-market is critical
- Team lacks ML expertise
- Use case is straightforward

### Not Acceptable When

- Domain-specific accuracy is critical
- Multi-tenant isolation required
- Regulatory compliance needed
- Cost optimization essential at scale
- Need specific search behavior
- Debugging capability required

---

## Summary

| Limitation Category | Impact | Workaround Possible? |
|--------------------|--------|---------------------|
| Fixed embeddings | Quality for domains | Limited |
| Limited chunking | Retrieval precision | Partial (pre-process) |
| Fixed search | Query flexibility | Partial (metadata) |
| Vendor lock-in | Migration risk | Partial (abstraction) |
| Rate limits | Scale constraints | Partial (queuing) |
| Feature gaps | Advanced techniques | No |
| Black box | Debugging | Limited |

---

**Key Takeaways:**

✅ **Managed RAG trades control for convenience**—understand what you're giving up  
✅ **Fixed embeddings** are the biggest quality limitation for specialized domains  
✅ **Vendor lock-in** is real—keep source files and build abstraction layers  
✅ **Rate limits** require planning for bulk operations and peak traffic  
✅ **Most limitations are acceptable** for small-to-medium, general-purpose use cases

---

**Lesson Complete!**

[← Back to Lesson Overview](./00-managed-rag-services.md)

---

<!-- 
Sources Consulted:
- OpenAI Vector Stores Limits: https://platform.openai.com/docs/api-reference/vector-stores
- OpenAI File Search: https://platform.openai.com/docs/guides/tools-file-search
- Gemini File Search: https://ai.google.dev/gemini-api/docs/file-search
- Gemini Pricing: https://ai.google.dev/pricing
-->
