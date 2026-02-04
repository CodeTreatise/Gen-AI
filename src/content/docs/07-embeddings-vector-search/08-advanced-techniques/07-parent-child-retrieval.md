---
title: "Parent-Child Retrieval"
---

# Parent-Child Retrieval

## Introduction

Parent-child retrieval splits documents into small chunks for precise semantic matching, then returns larger parent chunks to provide sufficient context for the LLM. This solves the fundamental chunking dilemma: small chunks match better, large chunks provide more context.

> **🤖 AI Context:** Without parent-child retrieval, you either get precise matches with insufficient context, or imprecise matches with good context. Parent-child gives you both.

---

## The Chunking Dilemma

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMALL CHUNKS (256 tokens)                    │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Precise semantic matching                                   │
│  ✅ Focused retrieval                                           │
│  ❌ Missing surrounding context                                 │
│  ❌ LLM may not understand full picture                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LARGE CHUNKS (2000 tokens)                   │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Rich context for LLM                                        │
│  ✅ Complete information                                        │
│  ❌ Diluted semantic meaning                                    │
│  ❌ Noisy retrieval (irrelevant parts match)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PARENT-CHILD RETRIEVAL                       │
├─────────────────────────────────────────────────────────────────┤
│  🔍 Search: Small "child" chunks (precise matching)             │
│  📄 Return: Large "parent" chunks (rich context)                │
│  ✅ Best of both worlds!                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Basic Parent-Child Implementation

```python
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class Chunk:
    text: str
    parent_id: str
    chunk_id: str
    embedding: list[float] = None

@dataclass
class Parent:
    text: str
    parent_id: str
    children: list[str] = None  # Child chunk IDs

class ParentChildRetriever:
    """Retriever that searches children, returns parents."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.parents: dict[str, Parent] = {}
        self.children: dict[str, Chunk] = {}
        self.child_embeddings = None
        self.child_ids = []
    
    def add_document(
        self,
        document: str,
        parent_size: int = 1000,
        child_size: int = 200,
        child_overlap: int = 50
    ):
        """Add document with parent-child chunking."""
        
        # Split into parent chunks
        parent_chunks = self._split_text(document, parent_size, parent_size // 4)
        
        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{p_idx}"
            
            # Split parent into children
            child_chunks = self._split_text(parent_text, child_size, child_overlap)
            child_ids = []
            
            for c_idx, child_text in enumerate(child_chunks):
                child_id = f"{parent_id}_child_{c_idx}"
                child_ids.append(child_id)
                
                self.children[child_id] = Chunk(
                    text=child_text,
                    parent_id=parent_id,
                    chunk_id=child_id
                )
            
            self.parents[parent_id] = Parent(
                text=parent_text,
                parent_id=parent_id,
                children=child_ids
            )
        
        # Embed all children
        self._embed_children()
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> list[str]:
        """Split text into overlapping chunks."""
        
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _embed_children(self):
        """Generate embeddings for all children."""
        
        self.child_ids = list(self.children.keys())
        texts = [self.children[cid].text for cid in self.child_ids]
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.child_embeddings = np.array(embeddings)
        
        for cid, emb in zip(self.child_ids, embeddings):
            self.children[cid].embedding = emb.tolist()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_children: bool = False
    ) -> list[dict]:
        """Search children, return parents."""
        
        query_embedding = self.model.encode(query)
        
        # Search children
        similarities = np.dot(self.child_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k * 3:][::-1]  # Get more children
        
        # Aggregate by parent
        parent_scores = {}
        parent_children = {}
        
        for idx in top_indices:
            child_id = self.child_ids[idx]
            child = self.children[child_id]
            parent_id = child.parent_id
            score = float(similarities[idx])
            
            if parent_id not in parent_scores:
                parent_scores[parent_id] = score
                parent_children[parent_id] = [child_id]
            else:
                # Keep max score (or could average)
                parent_scores[parent_id] = max(parent_scores[parent_id], score)
                parent_children[parent_id].append(child_id)
        
        # Sort parents by score
        sorted_parents = sorted(
            parent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for parent_id, score in sorted_parents:
            result = {
                "parent_id": parent_id,
                "parent_text": self.parents[parent_id].text,
                "score": score,
            }
            
            if return_children:
                result["matched_children"] = [
                    self.children[cid].text 
                    for cid in parent_children[parent_id]
                ]
            
            results.append(result)
        
        return results
```

---

## Usage Example

```python
# Sample document
document = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience. It focuses on developing 
computer programs that can access data and use it to learn for themselves.

The process begins with observations or data, such as examples, direct 
experience, or instruction. The goal is to allow computers to learn 
automatically without human intervention.

Deep learning is a specialized form of machine learning. It uses neural 
networks with multiple layers to progressively extract higher-level 
features from raw input. For example, in image processing, lower layers 
may identify edges, while higher layers may identify objects.

Neural networks are inspired by the human brain. They consist of 
interconnected nodes organized in layers. Each connection has a weight 
that adjusts as learning proceeds, eventually determining the network's 
output.
"""

# Create retriever and add document
retriever = ParentChildRetriever()
retriever.add_document(document, parent_size=100, child_size=30)

# Search
results = retriever.search("How do neural networks learn?", return_children=True)

for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Parent: {r['parent_text'][:100]}...")
    print(f"Matched children: {len(r['matched_children'])}")
    print("---")
```

---

## LlamaIndex Parent-Child

LlamaIndex has built-in support:

```python
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core import VectorStoreIndex

# Load documents
documents = SimpleDirectoryReader("data/").load_data()

# Create parent splitter (larger chunks)
parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

# Create child splitter (smaller chunks)
child_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

# Parse into parents
parent_nodes = parent_splitter.get_nodes_from_documents(documents)

# For each parent, create children that reference back
all_nodes = []
for parent in parent_nodes:
    # Add parent
    all_nodes.append(parent)
    
    # Create children from parent
    child_nodes = child_splitter.get_nodes_from_documents(
        [parent.get_content()],
        show_progress=False
    )
    
    # Link children to parent using IndexNode
    for child in child_nodes:
        index_node = IndexNode(
            text=child.get_content(),
            index_id=parent.node_id,  # Reference to parent
        )
        all_nodes.append(index_node)

# Create index
index = VectorStoreIndex(all_nodes)

# Query - will automatically return parent context
query_engine = index.as_query_engine()
response = query_engine.query("How do neural networks work?")
```

---

## Multi-Level Hierarchy

Extend to multiple levels for very long documents:

```python
@dataclass
class HierarchicalChunk:
    text: str
    level: int  # 0 = document, 1 = section, 2 = paragraph, 3 = sentence
    chunk_id: str
    parent_id: str | None
    children: list[str]

class HierarchicalRetriever:
    """Multi-level parent-child retrieval."""
    
    LEVELS = {
        0: {"name": "document", "size": 10000},
        1: {"name": "section", "size": 2000},
        2: {"name": "paragraph", "size": 500},
        3: {"name": "sentence", "size": 100},
    }
    
    def __init__(self, search_level: int = 3, return_level: int = 1):
        """
        Args:
            search_level: Level to search (smaller = broader, larger = finer)
            return_level: Level to return (smaller = more context)
        """
        self.search_level = search_level
        self.return_level = return_level
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: dict[str, HierarchicalChunk] = {}
    
    def get_parent_at_level(
        self,
        chunk_id: str,
        target_level: int
    ) -> HierarchicalChunk:
        """Traverse up to find parent at target level."""
        
        chunk = self.chunks[chunk_id]
        
        while chunk.level > target_level:
            if chunk.parent_id is None:
                break
            chunk = self.chunks[chunk.parent_id]
        
        return chunk
```

---

## Qdrant Parent-Child with Payloads

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

client = QdrantClient("localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="parent_child",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Index children with parent references
def index_with_parent_child(
    documents: list[str],
    parent_size: int = 1000,
    child_size: int = 200
):
    points = []
    point_id = 0
    
    for doc_id, document in enumerate(documents):
        # Create parents
        parent_chunks = split_text(document, parent_size)
        
        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"doc{doc_id}_parent{p_idx}"
            
            # Create children
            child_chunks = split_text(parent_text, child_size)
            
            for c_idx, child_text in enumerate(child_chunks):
                embedding = model.encode(child_text)
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "child_text": child_text,
                        "parent_text": parent_text,  # Store full parent
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                    }
                ))
                point_id += 1
    
    client.upsert(collection_name="parent_child", points=points)

# Search and deduplicate parents
def search_parents(query: str, top_k: int = 5):
    results = client.query_points(
        collection_name="parent_child",
        query=model.encode(query).tolist(),
        limit=top_k * 3  # Get more children
    )
    
    # Deduplicate by parent_id, keeping highest score
    seen_parents = {}
    for point in results.points:
        parent_id = point.payload["parent_id"]
        if parent_id not in seen_parents or point.score > seen_parents[parent_id]["score"]:
            seen_parents[parent_id] = {
                "parent_text": point.payload["parent_text"],
                "score": point.score,
                "matched_child": point.payload["child_text"]
            }
    
    # Sort and return
    sorted_results = sorted(
        seen_parents.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]
    
    return sorted_results
```

---

## Choosing Chunk Sizes

| Use Case | Child Size | Parent Size | Ratio |
|----------|------------|-------------|-------|
| FAQ / Short docs | 100-200 | 500-800 | 4-5x |
| Technical docs | 200-400 | 1000-1500 | 4-5x |
| Long-form content | 300-500 | 1500-2500 | 5x |
| Legal / Contracts | 150-300 | 1000-2000 | 6x |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Store full parent text in payload | Re-fetch parents from source |
| Deduplicate parents in results | Return same parent multiple times |
| Use 4-6x ratio for child:parent | Make children too small (<50 tokens) |
| Include overlap in children | Hard boundaries between children |
| Test different size ratios | Use one size for all content |

---

## Summary

✅ **Child chunks** are small for precise semantic matching

✅ **Parent chunks** are large to provide LLM context

✅ **Typical ratio** is 4-6x between parent and child size

✅ **Deduplicate** parents when multiple children match

✅ **Store parent text** in payload for efficient retrieval

**Next:** [Recursive Retrieval](./08-recursive-retrieval.md)

---

<!-- 
Sources Consulted:
- LlamaIndex ParentChild: https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
- Qdrant Payloads: https://qdrant.tech/documentation/concepts/payload/
-->
