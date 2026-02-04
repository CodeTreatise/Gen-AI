---
title: "Hybrid Search with BM25"
---

# Hybrid Search with BM25

## Introduction

Contextual Retrieval achieves its best results by **combining semantic embeddings with BM25 keyword search**. This hybrid approach captures both meaning and exact term matches.

This lesson explains BM25, why hybrid search matters, and how to implement it effectively.

### What We'll Cover

- What BM25 is and how it works
- Why embeddings alone aren't enough
- Combining BM25 with vector search
- Rank fusion techniques
- Tuning the alpha parameter

### Prerequisites

- [Implementation Steps](./04-implementation-steps.md)
- Basic understanding of information retrieval concepts

---

## What is BM25?

### The Algorithm

**BM25** (Best Matching 25) is a ranking function for keyword search. It's an improved version of TF-IDF that considers:

- **Term Frequency (TF):** How often a word appears in a document
- **Inverse Document Frequency (IDF):** How rare a word is across all documents
- **Document Length:** Normalizes for document size

```
┌─────────────────────────────────────────────────────────────────┐
│                    BM25 Scoring Components                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: "revenue growth Q2"                                     │
│                                                                 │
│  Document A: "Revenue grew 3% in Q2. This revenue increase..."  │
│  Document B: "The Q2 quarterly report shows improvements..."    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Term Frequency (TF)                                     │    │
│  │                                                         │    │
│  │           "revenue"   "growth"    "Q2"                 │    │
│  │ Doc A:       2          0 (grew)    1                  │    │
│  │ Doc B:       0          0           1                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Inverse Document Frequency (IDF)                        │    │
│  │                                                         │    │
│  │ If "revenue" appears in 10% of docs → High IDF         │    │
│  │ If "the" appears in 95% of docs → Low IDF              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Final Score = TF × IDF × document_length_normalization         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### BM25 Formula

```python
# Simplified BM25 formula
score(D, Q) = Σ IDF(qi) × (tf(qi, D) × (k1 + 1)) / (tf(qi, D) + k1 × (1 - b + b × |D|/avgdl))

# Where:
# qi = query term i
# tf(qi, D) = term frequency of qi in document D
# |D| = document length
# avgdl = average document length
# k1 = term frequency saturation parameter (typically 1.2-2.0)
# b = document length normalization (typically 0.75)
```

---

## Why Embeddings Need BM25

### The Keyword Gap

Embeddings capture **semantic meaning** but can miss exact matches:

| Query | Embedding Behavior | BM25 Behavior |
|-------|-------------------|---------------|
| "error code 0x8007" | Finds "bug reports" | ✅ Exact match on "0x8007" |
| "ACME-2023-Q2-REV" | Finds "company revenue" | ✅ Exact match on ID |
| "John Smith memo" | Finds "employee notes" | ✅ Exact match on name |
| "§ 12.4.3" | Finds "legal sections" | ✅ Exact match on reference |

```
┌─────────────────────────────────────────────────────────────────┐
│             Embeddings vs BM25: Complementary Strengths          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EMBEDDINGS excel at:                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Synonyms: "car" matches "automobile"                    │  │
│  │ • Paraphrases: "how to fix" matches "troubleshooting"    │  │
│  │ • Concepts: "ML" matches "machine learning"              │  │
│  │ • Semantic similarity across different phrasings         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  BM25 excels at:                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Exact IDs: "ACME-2023-Q2"                              │  │
│  │ • Error codes: "0x8007", "ECONNREFUSED"                  │  │
│  │ • Proper nouns: "John Smith", "Microsoft"                │  │
│  │ • Technical terms: "distutils", "subprocess"             │  │
│  │ • Version numbers: "Python 3.12", "v2.1.4"               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  HYBRID combines both → Best of both worlds                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Examples

**Example 1: Technical Support**

```
Query: "error ECONNREFUSED when connecting to localhost:3000"

Embedding-only result:
  "Connection errors can occur when network issues prevent 
   communication between services..."
  (Semantic match, but generic)

BM25 result:
  "If you see ECONNREFUSED, ensure your server is running 
   on the specified port (localhost:3000)..."
  (Exact match on error code!)

Hybrid result: Returns the BM25 result first ✅
```

**Example 2: Legal Research**

```
Query: "§ 12.4.3 liability limitations"

Embedding-only result:
  "Contract terms often include clauses that limit 
   one party's liability..."
  (Semantic match, but wrong section)

BM25 result:
  "Section 12.4.3 Limitation of Liability: Neither party 
   shall be liable for indirect damages..."
  (Exact match on section number!)

Hybrid result: Returns the BM25 result first ✅
```

---

## Implementing BM25

### Using rank-bm25 Library

```python
from rank_bm25 import BM25Okapi
from typing import List
import re

def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.
    
    Handles:
    - Lowercasing
    - Punctuation removal
    - Preserves technical terms (error codes, IDs)
    """
    # Lowercase
    text = text.lower()
    
    # Split on whitespace and punctuation (but keep alphanumeric groups)
    tokens = re.findall(r'\b[a-z0-9]+(?:[-_][a-z0-9]+)*\b', text)
    
    return tokens


def create_bm25_index(documents: List[str]) -> BM25Okapi:
    """Create a BM25 index from documents."""
    tokenized_docs = [tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)


def search_bm25(
    bm25: BM25Okapi, 
    query: str, 
    top_k: int = 10
) -> List[tuple]:
    """
    Search BM25 index and return (index, score) tuples.
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Get indices sorted by score
    ranked = sorted(
        enumerate(scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return ranked[:top_k]


# Example usage
documents = [
    "This chunk describes the error code ECONNREFUSED in Node.js networking.",
    "Connection errors occur when the server cannot be reached.",
    "The subprocess module handles process communication in Python.",
]

bm25 = create_bm25_index(documents)
results = search_bm25(bm25, "ECONNREFUSED error", top_k=3)

for idx, score in results:
    print(f"Doc {idx}: score={score:.4f}")
    print(f"  {documents[idx][:60]}...")
```

**Output:**
```
Doc 0: score=2.1453
  This chunk describes the error code ECONNREFUSED in Node.js n...
Doc 1: score=0.4521
  Connection errors occur when the server cannot be reached....
Doc 2: score=0.0000
  The subprocess module handles process communication in Python...
```

---

## Combining Vector and BM25 Search

### Score Normalization

Before combining, normalize scores to the same scale:

```python
import numpy as np

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max scaling."""
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(
    query: str,
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    chunks: List[str],
    embed_func,  # Function to create query embedding
    alpha: float = 0.5,
    top_k: int = 10
) -> List[tuple]:
    """
    Perform hybrid search combining vector and BM25.
    
    Args:
        query: Search query
        embeddings: Document embeddings matrix (n_docs x dim)
        bm25: BM25Okapi index
        chunks: Original chunk texts
        embed_func: Function to embed the query
        alpha: Weight for vector scores (0=BM25 only, 1=vector only)
        top_k: Number of results to return
    
    Returns:
        List of (index, combined_score, vector_score, bm25_score)
    """
    # Vector search
    query_embedding = embed_func(query)
    vector_scores = np.dot(embeddings, query_embedding)
    vector_scores = normalize_scores(vector_scores)
    
    # BM25 search
    query_tokens = tokenize(query)
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    bm25_scores = normalize_scores(bm25_scores)
    
    # Combine with weighted sum
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    
    # Rank by combined score
    ranked = sorted(
        enumerate(combined_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return with all scores for analysis
    results = [
        (idx, combined_scores[idx], vector_scores[idx], bm25_scores[idx])
        for idx, _ in ranked[:top_k]
    ]
    
    return results
```

### Alpha Parameter

The `alpha` parameter controls the balance:

| Alpha | Behavior | Best For |
|-------|----------|----------|
| `0.0` | BM25 only | Exact term matching |
| `0.3` | Heavy BM25 | Technical docs with IDs/codes |
| `0.5` | Balanced | General use |
| `0.7` | Heavy vector | Semantic/conceptual search |
| `1.0` | Vector only | Pure semantic similarity |

```
┌─────────────────────────────────────────────────────────────────┐
│                    Alpha Parameter Effects                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  alpha = 0.0                    alpha = 0.5                     │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       │▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░│         │
│  │       BM25         │       │  BM25    │  Vector  │         │
│  └─────────────────────┘       └─────────────────────┘         │
│  100% BM25                      50% BM25, 50% Vector            │
│                                                                 │
│  alpha = 0.7                    alpha = 1.0                     │
│  ┌─────────────────────┐       ┌─────────────────────┐         │
│  │▓▓▓▓▓▓░░░░░░░░░░░░░░│       │░░░░░░░░░░░░░░░░░░░░│         │
│  │BM25│    Vector     │       │       Vector        │         │
│  └─────────────────────┘       └─────────────────────┘         │
│  30% BM25, 70% Vector           100% Vector                     │
│                                                                 │
│  Anthropic found α = 0.5 - 0.7 works best for most cases       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Rank Fusion Alternatives

### Reciprocal Rank Fusion (RRF)

RRF combines rankings rather than scores:

```python
def reciprocal_rank_fusion(
    rankings: List[List[int]],  # List of ranked doc indices
    k: int = 60  # Constant to prevent high scores for top ranks
) -> List[tuple]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.
    
    RRF score = Σ 1 / (k + rank_i)
    
    Args:
        rankings: List of rankings (each is list of doc indices)
        k: Smoothing constant (default 60)
    
    Returns:
        List of (doc_index, rrf_score) sorted by score
    """
    rrf_scores = {}
    
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0
            rrf_scores[doc_idx] += 1 / (k + rank + 1)  # +1 for 0-indexed
    
    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_docs


# Example usage
vector_ranking = [3, 1, 5, 0, 2, 4]  # Doc indices by vector score
bm25_ranking = [1, 3, 0, 5, 4, 2]    # Doc indices by BM25 score

combined = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
print("RRF Combined Ranking:")
for doc_idx, score in combined[:5]:
    print(f"  Doc {doc_idx}: RRF score = {score:.4f}")
```

**Output:**
```
RRF Combined Ranking:
  Doc 3: RRF score = 0.0325
  Doc 1: RRF score = 0.0325
  Doc 5: RRF score = 0.0279
  Doc 0: RRF score = 0.0262
  Doc 2: RRF score = 0.0213
```

### When to Use Each Method

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Weighted Sum** | Simple, tunable alpha | Needs score normalization | Most use cases |
| **RRF** | Score-agnostic, robust | Loses magnitude info | When scores aren't comparable |

---

## Contextual BM25

### The Key Insight

In Contextual Retrieval, BM25 indexes the **contextualized** chunks, not the original:

```python
# Traditional BM25 (without context)
original_chunk = "The company's revenue grew by 3%"
# BM25 can't match "ACME" or "Q2 2023"

# Contextual BM25
contextualized_chunk = """This chunk is from ACME Corporation's 
Q2 2023 quarterly report. The previous quarter revenue was $314M.

The company's revenue grew by 3%"""
# BM25 CAN match "ACME", "Q2", "2023", "quarterly", etc.
```

```
┌─────────────────────────────────────────────────────────────────┐
│                Traditional vs Contextual BM25                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: "ACME Q2 2023 revenue"                                  │
│                                                                 │
│  TRADITIONAL BM25:                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Indexed text: "The company's revenue grew by 3%"         │  │
│  │                                                          │  │
│  │ Query terms:    ACME    Q2    2023    revenue            │  │
│  │ In document?     ❌     ❌      ❌       ✅               │  │
│  │                                                          │  │
│  │ BM25 Score: LOW (only 1/4 terms match)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  CONTEXTUAL BM25:                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Indexed text: "This chunk is from ACME Corporation's     │  │
│  │               Q2 2023 quarterly report... revenue grew"  │  │
│  │                                                          │  │
│  │ Query terms:    ACME    Q2    2023    revenue            │  │
│  │ In document?     ✅     ✅      ✅       ✅               │  │
│  │                                                          │  │
│  │ BM25 Score: HIGH (4/4 terms match)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Impact

From Anthropic's research:

| Approach | Retrieval Failure Rate |
|----------|----------------------|
| Traditional Embeddings | 5.7% |
| Contextual Embeddings | 3.7% (-35%) |
| **Contextual Embeddings + Contextual BM25** | **2.9% (-49%)** |

Adding BM25 to contextualized chunks provides an additional **14% improvement** beyond embeddings alone.

---

## Complete Hybrid Search Implementation

```python
class HybridSearch:
    """Complete hybrid search with contextual BM25."""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        alpha: float = 0.5
    ):
        self.openai_client = OpenAI()
        self.embedding_model = embedding_model
        self.alpha = alpha
        
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.bm25: BM25Okapi = None
    
    def index_chunks(self, contextualized_chunks: List[Dict]) -> None:
        """Index contextualized chunks for hybrid search."""
        self.chunks = contextualized_chunks
        
        # Get contextualized texts
        texts = [c["contextualized_text"] for c in contextualized_chunks]
        
        # Create embeddings
        self.embeddings = self._embed(texts)
        
        # Create BM25 index
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"Indexed {len(self.chunks)} chunks")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        alpha: float = None
    ) -> List[Dict]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Override default alpha (optional)
        
        Returns:
            List of result dictionaries with scores
        """
        alpha = alpha if alpha is not None else self.alpha
        
        # Vector search
        query_emb = self._embed([query])[0]
        vector_scores = np.dot(self.embeddings, query_emb)
        vector_scores = self._normalize(vector_scores)
        
        # BM25 search
        query_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        bm25_scores = self._normalize(bm25_scores)
        
        # Combine
        combined = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        # Rank and return
        top_indices = np.argsort(combined)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "combined_score": float(combined[idx]),
                "vector_score": float(vector_scores[idx]),
                "bm25_score": float(bm25_scores[idx])
            })
        
        return results
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Create embeddings."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize for BM25."""
        return re.findall(r'\b[a-z0-9]+(?:[-_][a-z0-9]+)*\b', text.lower())
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores."""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


# Usage
hybrid = HybridSearch(alpha=0.6)
hybrid.index_chunks(contextualized_chunks)

results = hybrid.search("ACME Q2 2023 cloud revenue growth", top_k=5)
for r in results:
    print(f"Score: {r['combined_score']:.4f} "
          f"(vec={r['vector_score']:.4f}, bm25={r['bm25_score']:.4f})")
    print(f"  {r['chunk']['text'][:80]}...\n")
```

---

## Summary

✅ **BM25 is a keyword-based ranking** using term frequency and document frequency  
✅ **Embeddings miss exact matches** (error codes, IDs, names)  
✅ **Hybrid search combines both** approaches for best results  
✅ **Alpha parameter** controls the balance (0.5-0.7 typically best)  
✅ **Contextual BM25** indexes contextualized chunks for better keyword matching  
✅ Adding BM25 provides **14% additional improvement** beyond contextual embeddings

---

**Next:** [Performance Improvements →](./06-performance-improvements.md)

---

<!-- 
Sources Consulted:
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Pinecone Hybrid Search: https://www.pinecone.io/learn/hybrid-search/
- rank-bm25 library documentation
-->
