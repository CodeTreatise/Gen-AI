---
title: "Chunk Metadata Preservation"
---

# Chunk Metadata Preservation

## Introduction

Chunks without metadata are like puzzle pieces without the box—you can't see the bigger picture. Preserving source information, position, and relationships enables precise retrieval, accurate citations, and context reconstruction.

---

## Essential Metadata Fields

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class ChunkMetadata:
    """Comprehensive metadata for a text chunk."""
    
    # Source identification
    document_id: str
    document_title: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    
    # Position in document
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None
    
    # Structural context
    section_title: Optional[str] = None
    heading_hierarchy: list[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    
    # Content characteristics
    content_type: str = "text"  # text, code, table, list
    language: str = "en"
    word_count: int = 0
    token_count: int = 0
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    document_date: Optional[datetime] = None
    
    # Custom
    tags: list[str] = field(default_factory=list)
    custom: dict = field(default_factory=dict)
```

---

## Metadata Extraction Strategies

### From Document Structure

```python
import re
from typing import Generator

def extract_structural_metadata(
    text: str,
    document_id: str
) -> Generator[dict, None, None]:
    """Extract metadata from markdown structure."""
    
    # Parse headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headings = [(m.start(), len(m.group(1)), m.group(2)) 
                for m in heading_pattern.finditer(text)]
    
    # Track heading hierarchy
    def get_hierarchy(position: int) -> list[str]:
        hierarchy = [""] * 6
        for h_pos, h_level, h_text in headings:
            if h_pos <= position:
                hierarchy[h_level - 1] = h_text
                # Clear lower levels
                for i in range(h_level, 6):
                    hierarchy[i] = ""
        return [h for h in hierarchy if h]
    
    # Chunk the text
    chunks = chunk_text(text)  # Your chunking function
    
    for i, (chunk, start, end) in enumerate(chunks):
        hierarchy = get_hierarchy(start)
        
        yield {
            "text": chunk,
            "metadata": {
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_char": start,
                "end_char": end,
                "section_title": hierarchy[-1] if hierarchy else None,
                "heading_hierarchy": hierarchy,
                "content_type": detect_content_type(chunk)
            }
        }

def detect_content_type(text: str) -> str:
    """Detect the type of content in a chunk."""
    
    if re.search(r'```[\s\S]*?```', text):
        return "code"
    elif re.search(r'^\s*\|.*\|.*\|', text, re.MULTILINE):
        return "table"
    elif re.search(r'^\s*[-*]\s', text, re.MULTILINE):
        return "list"
    elif re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
        return "numbered_list"
    else:
        return "text"
```

### From PDF Documents

```python
import fitz  # PyMuPDF

def extract_pdf_metadata(
    pdf_path: str,
    document_id: str
) -> list[dict]:
    """Extract chunks with page-level metadata from PDF."""
    
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        
        # Chunk the page text
        page_chunks = chunk_text(text)
        
        for chunk_text, start, end in page_chunks:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "source_file": pdf_path,
                    "page_number": page_num,
                    "total_pages": len(doc),
                    "start_char_on_page": start,
                    "end_char_on_page": end,
                    "page_width": page.rect.width,
                    "page_height": page.rect.height
                }
            })
    
    return chunks
```

---

## Cross-Reference Metadata

Link related chunks for context expansion:

```python
from dataclasses import dataclass
from typing import Optional
import hashlib

@dataclass
class LinkedChunk:
    chunk_id: str
    text: str
    embedding: list[float]
    
    # Navigation links
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: list[str] = None
    
    # Semantic links
    related_chunk_ids: list[str] = None  # Similar content
    reference_chunk_ids: list[str] = None  # Cited by this chunk

def create_linked_chunks(
    chunks: list[str],
    document_id: str,
    embeddings: list[list[float]]
) -> list[LinkedChunk]:
    """Create chunks with navigation links."""
    
    linked = []
    
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = generate_chunk_id(document_id, i, text)
        
        linked.append(LinkedChunk(
            chunk_id=chunk_id,
            text=text,
            embedding=embedding,
            prev_chunk_id=(
                generate_chunk_id(document_id, i-1, chunks[i-1])
                if i > 0 else None
            ),
            next_chunk_id=(
                generate_chunk_id(document_id, i+1, chunks[i+1])
                if i < len(chunks) - 1 else None
            ),
            child_chunk_ids=[],
            related_chunk_ids=[]
        ))
    
    return linked

def generate_chunk_id(
    document_id: str,
    index: int,
    text: str
) -> str:
    """Generate stable chunk ID."""
    content = f"{document_id}:{index}:{text[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

## Context Window Expansion

Use metadata to expand retrieved chunks:

```python
from typing import Optional

class ContextExpander:
    """Expand retrieved chunks with surrounding context."""
    
    def __init__(self, chunk_store: dict[str, LinkedChunk]):
        self.store = chunk_store
    
    def expand_context(
        self,
        chunk_id: str,
        window_before: int = 1,
        window_after: int = 1,
        include_parent: bool = True
    ) -> str:
        """Expand a chunk with surrounding context."""
        
        chunk = self.store.get(chunk_id)
        if not chunk:
            return ""
        
        parts = []
        
        # Add parent context (e.g., section header)
        if include_parent and chunk.parent_chunk_id:
            parent = self.store.get(chunk.parent_chunk_id)
            if parent:
                parts.append(f"[Context: {parent.text[:200]}...]\n")
        
        # Add previous chunks
        current = chunk
        prev_texts = []
        for _ in range(window_before):
            if current.prev_chunk_id:
                current = self.store.get(current.prev_chunk_id)
                if current:
                    prev_texts.insert(0, current.text)
        parts.extend(prev_texts)
        
        # Add main chunk
        parts.append(chunk.text)
        
        # Add following chunks
        current = chunk
        for _ in range(window_after):
            if current.next_chunk_id:
                current = self.store.get(current.next_chunk_id)
                if current:
                    parts.append(current.text)
        
        return "\n\n".join(parts)
```

---

## Vector Store Metadata

Store metadata alongside embeddings:

### Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_KEY")
index = pc.Index("documents")

def upsert_with_metadata(
    chunks: list[dict],
    embeddings: list[list[float]]
):
    """Store chunks with full metadata in Pinecone."""
    
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "text": chunk["text"][:1000],  # Pinecone limit
                "document_id": chunk["metadata"]["document_id"],
                "section": chunk["metadata"]["section_title"],
                "page": chunk["metadata"].get("page_number"),
                "chunk_index": chunk["metadata"]["chunk_index"],
                "content_type": chunk["metadata"]["content_type"],
                # Store serialized for complex data
                "hierarchy": "|".join(chunk["metadata"]["heading_hierarchy"])
            }
        })
    
    index.upsert(vectors=vectors)

def query_with_metadata_filter(
    query_embedding: list[float],
    document_id: str = None,
    content_type: str = None,
    top_k: int = 10
) -> list[dict]:
    """Query with metadata filters."""
    
    filter_dict = {}
    if document_id:
        filter_dict["document_id"] = {"$eq": document_id}
    if content_type:
        filter_dict["content_type"] = {"$eq": content_type}
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict if filter_dict else None,
        include_metadata=True
    )
    
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text"),
            "section": match.metadata.get("section"),
            "page": match.metadata.get("page"),
            "hierarchy": match.metadata.get("hierarchy", "").split("|")
        }
        for match in results.matches
    ]
```

### Chroma

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

def add_with_metadata(chunks: list[dict], embeddings: list[list[float]]):
    """Add chunks with metadata to Chroma."""
    
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "document_id": c["metadata"]["document_id"],
                "section": c["metadata"]["section_title"] or "",
                "chunk_index": c["metadata"]["chunk_index"],
                "content_type": c["metadata"]["content_type"],
                "page": c["metadata"].get("page_number") or 0
            }
            for c in chunks
        ]
    )

def query_by_document(
    query_embedding: list[float],
    document_id: str,
    n_results: int = 10
) -> dict:
    """Query within a specific document."""
    
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"document_id": document_id}
    )
```

---

## Citation Generation

Use metadata for accurate citations:

```python
def generate_citation(chunk_metadata: dict) -> str:
    """Generate a citation from chunk metadata."""
    
    parts = []
    
    # Document title
    if chunk_metadata.get("document_title"):
        parts.append(f'"{chunk_metadata["document_title"]}"')
    
    # Section
    if chunk_metadata.get("section_title"):
        parts.append(f'Section: {chunk_metadata["section_title"]}')
    
    # Page
    if chunk_metadata.get("page_number"):
        parts.append(f'p. {chunk_metadata["page_number"]}')
    
    # URL
    if chunk_metadata.get("source_url"):
        parts.append(f'({chunk_metadata["source_url"]})')
    
    return ", ".join(parts)

def format_rag_response(
    answer: str,
    source_chunks: list[dict]
) -> str:
    """Format RAG response with citations."""
    
    citations = []
    for i, chunk in enumerate(source_chunks, 1):
        citation = generate_citation(chunk["metadata"])
        citations.append(f"[{i}] {citation}")
    
    return f"{answer}\n\n**Sources:**\n" + "\n".join(citations)
```

---

## Metadata Schema Best Practices

| Field | Type | Index? | Filter? | Notes |
|-------|------|--------|---------|-------|
| document_id | string | ✅ | ✅ | Always required |
| chunk_index | int | ✅ | ⚠️ | For ordering |
| section_title | string | ❌ | ✅ | Scoped search |
| page_number | int | ❌ | ✅ | PDF navigation |
| content_type | string | ❌ | ✅ | Filter code/text |
| created_at | datetime | ✅ | ✅ | Recency ranking |
| text (snippet) | string | ❌ | ❌ | Display only |

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Store document_id always | Lose source tracking |
| Include position (index, chars) | Store chunks without order |
| Preserve heading hierarchy | Flatten structure |
| Use consistent field names | Vary schema per doc type |
| Store prev/next links | Require re-chunking for context |
| Truncate text in metadata | Store full text as metadata |

---

## Summary

✅ **Essential fields**: document_id, chunk_index, section, position

✅ **Cross-references** (prev/next) enable context window expansion

✅ **Vector store metadata** enables filtered retrieval

✅ **Citations** require source URL, page number, section

✅ **Consistent schema** across documents simplifies querying

**Next:** [Summary and What's Next](./00-document-chunking.md)

---

<!-- 
Sources Consulted:
- Pinecone Metadata Filtering: https://docs.pinecone.io/guides/data/filter-with-metadata
- Chroma Where Filters: https://docs.trychroma.com/guides#using-where-filters
- LlamaIndex Node Metadata: https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
-->
