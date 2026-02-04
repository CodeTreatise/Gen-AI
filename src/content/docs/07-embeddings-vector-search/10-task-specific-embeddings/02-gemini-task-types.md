---
title: "Gemini Task Types"
---

# Gemini Task Types

## Introduction

Google's Gemini embedding model offers the **most comprehensive task type system** among major providers. With 8 distinct task types, you can precisely specify how your text should be embedded for maximum effectiveness.

The `gemini-embedding-001` model uses the `task_type` parameter to optimize embeddings for specific use cases, from standard retrieval to specialized tasks like code search and fact verification.

### What We'll Cover

- All 8 Gemini task types with examples
- When to use each task type
- Practical implementation patterns
- Task type selection guide

### Prerequisites

- [Why Task Type Matters](./01-why-task-type-matters.md)
- Gemini API access and configuration

---

## Complete Task Type Reference

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `RETRIEVAL_QUERY` | Optimize for search queries | User search input |
| `RETRIEVAL_DOCUMENT` | Optimize for indexed documents | Document indexing |
| `SEMANTIC_SIMILARITY` | Symmetric text comparison | Duplicate detection |
| `CLASSIFICATION` | Text categorization | Sentiment, spam detection |
| `CLUSTERING` | Group similar items | Topic modeling |
| `CODE_RETRIEVAL_QUERY` | Code search queries | Natural language code search |
| `QUESTION_ANSWERING` | Q&A system queries | FAQ, chatbots |
| `FACT_VERIFICATION` | Fact-checking queries | Claim verification |

---

## Retrieval Task Types

### RETRIEVAL_QUERY

Use for **search queries** entered by users. Optimized to match against documents.

```python
from google import genai
from google.genai import types

client = genai.Client()

# User search query
query = "How to implement binary search in Python?"

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)

query_embedding = result.embeddings[0].values
print(f"Query embedding dimension: {len(query_embedding)}")
```

**When to use:**
- User search bar input
- Natural language questions
- Any query that will be matched against indexed documents

### RETRIEVAL_DOCUMENT

Use for **documents being indexed** for later retrieval. Paired with `RETRIEVAL_QUERY`.

```python
# Documents to be indexed
documents = [
    "Binary search is an efficient algorithm for finding an item in a sorted list. "
    "It works by repeatedly dividing the search interval in half.",
    
    "Merge sort is a divide-and-conquer algorithm that divides the input array "
    "into two halves, recursively sorts them, and then merges the sorted halves.",
    
    "Quick sort is an in-place sorting algorithm that uses a pivot element "
    "to partition the array into smaller sub-arrays."
]

# Embed documents for indexing
doc_embeddings = []
for doc in documents:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    doc_embeddings.append(result.embeddings[0].values)

print(f"Indexed {len(doc_embeddings)} documents")
```

**When to use:**
- Building search indexes
- Creating document databases
- Any content that users will search over

### Complete Retrieval Example

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query: str, documents: list[str], top_k: int = 3):
    """Search documents using asymmetric embeddings."""
    
    # Embed query
    query_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_emb = np.array(query_result.embeddings[0].values)
    
    # Embed documents
    results = []
    for doc in documents:
        doc_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=doc,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        doc_emb = np.array(doc_result.embeddings[0].values)
        
        similarity = cosine_similarity(query_emb, doc_emb)
        results.append((doc, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example usage
documents = [
    "Python is a high-level programming language.",
    "Machine learning models learn patterns from data.",
    "Binary search divides the search space in half each iteration."
]

query = "efficient algorithm for sorted data"
results = search(query, documents)

for doc, score in results:
    print(f"Score: {score:.4f} | {doc[:50]}...")
```

---

## Symmetric Comparison Types

### SEMANTIC_SIMILARITY

Use for comparing **two texts of similar nature**. Both texts get the same optimization.

```python
# Comparing similar texts
texts = [
    "The cat sat on the mat.",
    "A feline was resting on the rug.",
    "Dogs are loyal pets.",
]

# Embed all with SEMANTIC_SIMILARITY
embeddings = []
for text in texts:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    embeddings.append(np.array(result.embeddings[0].values))

# Build similarity matrix
import pandas as pd

similarity_matrix = np.zeros((len(texts), len(texts)))
for i in range(len(texts)):
    for j in range(len(texts)):
        similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

df = pd.DataFrame(
    similarity_matrix,
    index=[t[:20] + "..." for t in texts],
    columns=[t[:20] + "..." for t in texts]
)
print(df.round(3))
```

**Output:**
```
                      The cat sat on the...  A feline was restin...  Dogs are loyal pets...
The cat sat on the...                 1.000                   0.892                    0.543
A feline was restin...                0.892                   1.000                    0.521
Dogs are loyal pets...                0.543                   0.521                    1.000
```

**When to use:**
- Duplicate detection
- Paraphrase identification
- Plagiarism detection
- Finding similar documents

---

## Classification and Clustering

### CLASSIFICATION

Optimized for **separating texts into distinct categories**.

```python
# Training examples for a classifier
training_data = [
    ("I love this product! Best purchase ever!", "positive"),
    ("This is absolutely fantastic!", "positive"),
    ("Terrible quality, waste of money.", "negative"),
    ("Worst experience of my life.", "negative"),
    ("The item arrived on time.", "neutral"),
    ("Standard product, nothing special.", "neutral"),
]

# Embed with CLASSIFICATION task type
labeled_embeddings = []
for text, label in training_data:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="CLASSIFICATION")
    )
    labeled_embeddings.append({
        "text": text,
        "label": label,
        "embedding": result.embeddings[0].values
    })

# Now use with any classifier (KNN, SVM, etc.)
from sklearn.neighbors import KNeighborsClassifier

X = [item["embedding"] for item in labeled_embeddings]
y = [item["label"] for item in labeled_embeddings]

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

# Classify new text
new_text = "What an amazing experience!"
new_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=new_text,
    config=types.EmbedContentConfig(task_type="CLASSIFICATION")
)
prediction = classifier.predict([new_result.embeddings[0].values])
print(f"Predicted: {prediction[0]}")  # positive
```

**When to use:**
- Sentiment analysis
- Spam detection
- Topic categorization
- Intent classification

### CLUSTERING

Optimized for **grouping similar items** with clear cluster separation.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Documents to cluster
documents = [
    # Tech cluster
    "Python programming tutorial for beginners",
    "JavaScript framework comparison 2025",
    "Machine learning model deployment",
    # Food cluster
    "Best Italian pasta recipes",
    "How to bake chocolate cake",
    "Healthy meal prep ideas",
    # Sports cluster
    "NBA playoffs predictions",
    "Soccer training techniques",
    "Marathon running tips for beginners"
]

# Embed with CLUSTERING task type
embeddings = []
for doc in documents:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
        config=types.EmbedContentConfig(task_type="CLUSTERING")
    )
    embeddings.append(result.embeddings[0].values)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis', s=100)
for i, doc in enumerate(documents):
    plt.annotate(doc[:20] + "...", (reduced[i, 0], reduced[i, 1]), fontsize=8)
plt.colorbar(scatter, label='Cluster')
plt.title('Document Clustering with CLUSTERING Task Type')
plt.savefig('gemini_clustering.png')
plt.show()

# Print clusters
for cluster_id in range(3):
    print(f"\nCluster {cluster_id}:")
    for doc, c in zip(documents, clusters):
        if c == cluster_id:
            print(f"  - {doc}")
```

**When to use:**
- Topic modeling
- Content organization
- Market segmentation
- Anomaly detection

---

## Specialized Task Types

### CODE_RETRIEVAL_QUERY

Optimized for **natural language queries** that search for code. Pair with `RETRIEVAL_DOCUMENT` for code blocks.

```python
# Natural language query for code
query = "sort a list in descending order"

# Code snippets in your index
code_snippets = [
    """
def sort_descending(lst):
    return sorted(lst, reverse=True)
    """,
    """
def binary_search(arr, target):
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
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    """
]

# Embed query with CODE_RETRIEVAL_QUERY
query_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="CODE_RETRIEVAL_QUERY")
)
query_emb = np.array(query_result.embeddings[0].values)

# Embed code with RETRIEVAL_DOCUMENT
for code in code_snippets:
    code_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=code,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    code_emb = np.array(code_result.embeddings[0].values)
    similarity = cosine_similarity(query_emb, code_emb)
    print(f"Score: {similarity:.4f}")
    print(code[:60].strip() + "...")
    print()
```

**When to use:**
- Code search engines
- IDE code completion
- Documentation search
- Developer tools

### QUESTION_ANSWERING

Optimized for **questions** in Q&A systems. Pair with `RETRIEVAL_DOCUMENT` for answer passages.

```python
# FAQ knowledge base
faq = [
    {
        "question": "What are your business hours?",
        "answer": "We are open Monday through Friday, 9 AM to 5 PM EST."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page and enter your email."
    },
    {
        "question": "What is your return policy?",
        "answer": "Items can be returned within 30 days for a full refund."
    }
]

# Embed FAQ answers as documents
for item in faq:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=item["answer"],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    item["embedding"] = np.array(result.embeddings[0].values)

# User question
user_question = "How can I get my money back?"

# Embed with QUESTION_ANSWERING
question_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=user_question,
    config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING")
)
question_emb = np.array(question_result.embeddings[0].values)

# Find best match
best_match = max(faq, key=lambda x: cosine_similarity(question_emb, x["embedding"]))
print(f"Q: {user_question}")
print(f"A: {best_match['answer']}")
```

**When to use:**
- FAQ chatbots
- Customer support systems
- Knowledge base search
- Virtual assistants

### FACT_VERIFICATION

Optimized for **claims that need verification**. Pair with `RETRIEVAL_DOCUMENT` for evidence passages.

```python
# Evidence documents
evidence_corpus = [
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
    "The Great Wall of China is approximately 21,196 kilometers long.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
]

# Embed evidence
evidence_embeddings = []
for doc in evidence_corpus:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    evidence_embeddings.append({
        "text": doc,
        "embedding": np.array(result.embeddings[0].values)
    })

# Claim to verify
claim = "The Eiffel Tower is over 300 meters tall"

# Embed with FACT_VERIFICATION
claim_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=claim,
    config=types.EmbedContentConfig(task_type="FACT_VERIFICATION")
)
claim_emb = np.array(claim_result.embeddings[0].values)

# Find supporting evidence
for evidence in evidence_embeddings:
    similarity = cosine_similarity(claim_emb, evidence["embedding"])
    if similarity > 0.7:  # Threshold for relevant evidence
        print(f"Relevant evidence (score: {similarity:.3f}):")
        print(f"  {evidence['text']}")
```

**When to use:**
- Fact-checking systems
- Misinformation detection
- News verification
- Research validation

---

## Task Type Selection Guide

```
                          ┌─────────────────┐
                          │  What are you   │
                          │     doing?      │
                          └────────┬────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
      ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
      │  Search/  │          │ Compare   │          │  Analyze  │
      │ Retrieval │          │  Texts    │          │   Texts   │
      └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
            │                      │                      │
     ┌──────┴──────┐               │               ┌──────┴──────┐
     │             │               │               │             │
┌────▼────┐  ┌────▼────┐     ┌────▼────┐     ┌────▼────┐  ┌────▼────┐
│ Query?  │  │  Doc?   │     │ SEMANTIC│     │CLASSIF- │  │CLUSTER- │
│         │  │         │     │SIMILARITY│    │ICATION  │  │  ING    │
└────┬────┘  └────┬────┘     └─────────┘     └─────────┘  └─────────┘
     │            │
┌────┴─────────────┴────┐
│     Query Types       │
├───────────────────────┤
│ General → RETRIEVAL_  │
│           QUERY       │
│                       │
│ Code → CODE_RETRIEVAL_│
│        QUERY          │
│                       │
│ Q&A → QUESTION_       │
│       ANSWERING       │
│                       │
│ Fact → FACT_          │
│        VERIFICATION   │
└───────────────────────┘
```

---

## Batch Processing

Gemini supports batching multiple texts in a single call:

```python
# Batch embed multiple texts with same task type
texts = [
    "First document to embed",
    "Second document to embed", 
    "Third document to embed"
]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)

# Access all embeddings
for i, embedding in enumerate(result.embeddings):
    print(f"Document {i}: {len(embedding.values)} dimensions")
```

---

## Summary

✅ Gemini offers **8 task types** for precise embedding optimization  
✅ **RETRIEVAL_QUERY + RETRIEVAL_DOCUMENT** is the most common pairing  
✅ **SEMANTIC_SIMILARITY** for symmetric comparisons (duplicates)  
✅ **CLASSIFICATION** and **CLUSTERING** for ML tasks  
✅ **CODE_RETRIEVAL_QUERY** for natural language code search  
✅ **QUESTION_ANSWERING** and **FACT_VERIFICATION** for specialized retrieval

---

**Next:** [Cohere Input Types →](./03-cohere-input-types.md)

---

<!-- 
Sources Consulted:
- Gemini Embeddings documentation: https://ai.google.dev/gemini-api/docs/embeddings
- Gemini task types table (official)
-->
