---
title: "Clustering Embeddings"
---

# Clustering Embeddings

## Introduction

Clustering groups similar embeddings together, enabling topic discovery, deduplication, and visualization. Unlike search (finding similar items to a query), clustering finds natural groupings within the entire corpus.

> **🤖 AI Context:** Clustering is essential for RAG data quality: find duplicate chunks, discover topic distributions, and identify content gaps before retrieval issues occur.

---

## Clustering Algorithms for Embeddings

| Algorithm | Best For | Complexity | Cluster Shape |
|-----------|----------|------------|---------------|
| K-Means | Fixed K, spherical clusters | O(n·k·i) | Spherical |
| HDBSCAN | Unknown K, noise detection | O(n log n) | Arbitrary |
| Agglomerative | Hierarchies, dendrograms | O(n²) | Arbitrary |
| Mini-Batch K-Means | Large datasets | O(n·k) | Spherical |

---

## K-Means Clustering

```python
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Python is a programming language.",
    "JavaScript runs in browsers.",
    "AI can generate text and images.",
    "TypeScript adds types to JavaScript.",
]
embeddings = model.encode(documents)

# Cluster
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# View results
for i, (doc, cluster) in enumerate(zip(documents, clusters)):
    print(f"Cluster {cluster}: {doc}")
```

**Output:**
```
Cluster 0: Machine learning is a subset of AI.
Cluster 0: Deep learning uses neural networks.
Cluster 0: AI can generate text and images.
Cluster 1: Python is a programming language.
Cluster 1: JavaScript runs in browsers.
Cluster 1: TypeScript adds types to JavaScript.
```

---

## Finding Optimal K

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_k(embeddings, k_range: range = range(2, 11)):
    """Find optimal number of clusters using silhouette score."""
    
    scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Silhouette score (-1 to 1, higher is better)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        inertias.append(kmeans.inertia_)
    
    # Plot both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k_range, scores, marker='o')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Method')
    
    ax2.plot(k_range, inertias, marker='o')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Inertia (Within-cluster SS)')
    ax2.set_title('Elbow Method')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png')
    
    # Return optimal K
    optimal_k = k_range[np.argmax(scores)]
    return optimal_k

optimal_k = find_optimal_k(embeddings)
print(f"Optimal K: {optimal_k}")
```

---

## HDBSCAN for Unknown Clusters

HDBSCAN automatically determines cluster count and identifies noise:

```python
import hdbscan

def cluster_with_hdbscan(
    embeddings,
    min_cluster_size: int = 5,
    min_samples: int = 3
):
    """Cluster with HDBSCAN - no need to specify K."""
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    
    clusters = clusterer.fit_predict(embeddings)
    
    # -1 indicates noise (no cluster)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Found {n_clusters} clusters")
    print(f"Noise points: {n_noise}")
    
    return clusters, clusterer

clusters, clusterer = cluster_with_hdbscan(embeddings)
```

---

## Topic Discovery from Clusters

```python
from collections import Counter
from openai import OpenAI

client = OpenAI()

def generate_cluster_labels(
    documents: list[str],
    clusters: list[int]
) -> dict[int, str]:
    """Use LLM to generate topic labels for clusters."""
    
    cluster_docs = {}
    for doc, cluster in zip(documents, clusters):
        if cluster == -1:  # Skip noise
            continue
        if cluster not in cluster_docs:
            cluster_docs[cluster] = []
        cluster_docs[cluster].append(doc)
    
    labels = {}
    for cluster_id, docs in cluster_docs.items():
        # Sample docs for labeling
        sample = docs[:5] if len(docs) > 5 else docs
        docs_text = "\n".join(f"- {doc}" for doc in sample)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a short (2-4 word) topic label for these documents."
                },
                {
                    "role": "user",
                    "content": f"Documents:\n{docs_text}\n\nTopic label:"
                }
            ],
            max_tokens=20
        )
        
        labels[cluster_id] = response.choices[0].message.content.strip()
    
    return labels

# Usage
labels = generate_cluster_labels(documents, clusters)
for cluster_id, label in labels.items():
    print(f"Cluster {cluster_id}: {label}")
```

---

## Deduplication with Clustering

Find and remove near-duplicate documents:

```python
from sklearn.metrics.pairwise import cosine_similarity

def find_duplicates(
    embeddings,
    documents: list[str],
    threshold: float = 0.95
) -> list[tuple[int, int, float]]:
    """Find near-duplicate document pairs."""
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    duplicates = []
    n = len(documents)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i][j]
            if sim >= threshold:
                duplicates.append((i, j, sim))
    
    return duplicates

def deduplicate(
    embeddings,
    documents: list[str],
    threshold: float = 0.95
) -> list[str]:
    """Remove near-duplicates, keeping first occurrence."""
    
    duplicates = find_duplicates(embeddings, documents, threshold)
    
    # Track indices to remove (keep lower index)
    to_remove = set()
    for i, j, sim in duplicates:
        to_remove.add(j)  # Remove later duplicate
    
    # Return deduplicated list
    return [doc for i, doc in enumerate(documents) if i not in to_remove]

# Usage
duplicates = find_duplicates(embeddings, documents)
print(f"Found {len(duplicates)} duplicate pairs")

unique_docs = deduplicate(embeddings, documents)
print(f"Reduced from {len(documents)} to {len(unique_docs)} documents")
```

---

## Cluster-Based Retrieval

Use clusters to improve retrieval:

```python
class ClusterRetriever:
    """Retrieve from relevant clusters first."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.clusters = None
        self.centroids = None
    
    def index(
        self,
        documents: list[str],
        n_clusters: int = 10
    ):
        """Index documents and cluster them."""
        
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.embeddings)
        self.centroids = kmeans.cluster_centers_
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        n_clusters: int = 3
    ) -> list[dict]:
        """Search in most relevant clusters."""
        
        query_embedding = self.model.encode(query)
        
        # Find closest clusters
        cluster_sims = cosine_similarity(
            [query_embedding], 
            self.centroids
        )[0]
        top_clusters = np.argsort(cluster_sims)[-n_clusters:]
        
        # Search within those clusters
        results = []
        for cluster_id in top_clusters:
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_embeddings = self.embeddings[cluster_indices]
            
            sims = cosine_similarity([query_embedding], cluster_embeddings)[0]
            
            for idx, sim in zip(cluster_indices, sims):
                results.append({
                    "document": self.documents[idx],
                    "score": float(sim),
                    "cluster": int(cluster_id)
                })
        
        # Sort by score and return top K
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
```

---

## Visualization with UMAP

Reduce dimensions for plotting:

```python
import umap
import matplotlib.pyplot as plt

def visualize_clusters(
    embeddings,
    clusters,
    labels: dict[int, str] = None,
    output_file: str = "clusters.png"
):
    """Visualize clusters in 2D with UMAP."""
    
    # Reduce to 2D
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        metric='cosine'
    )
    coords = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters,
        cmap='tab10',
        alpha=0.6
    )
    
    # Add legend with labels
    if labels:
        handles, _ = scatter.legend_elements()
        legend_labels = [labels.get(i, f"Cluster {i}") for i in sorted(set(clusters)) if i != -1]
        plt.legend(handles, legend_labels, title="Topics")
    
    plt.title("Document Clusters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

visualize_clusters(embeddings, clusters, labels)
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Normalize embeddings first | Cluster raw embeddings |
| Use silhouette score for K | Guess cluster count |
| HDBSCAN for unknown K | Force K-Means always |
| Deduplicate before indexing | Index duplicates |
| Visualize to validate | Trust numbers alone |

---

## Summary

✅ **K-Means** when you know the number of clusters

✅ **HDBSCAN** auto-detects clusters and identifies noise

✅ **Silhouette score** helps find optimal K

✅ **Deduplication** removes near-duplicate documents

✅ **UMAP visualization** validates cluster quality

**Next:** [Parent-Child Retrieval](./07-parent-child-retrieval.md)

---

<!-- 
Sources Consulted:
- Scikit-learn Clustering: https://scikit-learn.org/stable/modules/clustering.html
- HDBSCAN Docs: https://hdbscan.readthedocs.io/
- UMAP Docs: https://umap-learn.readthedocs.io/
-->
