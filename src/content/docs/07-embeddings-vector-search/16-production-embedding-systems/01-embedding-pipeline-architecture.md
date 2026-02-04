---
title: "Embedding Pipeline Architecture"
---

# Embedding Pipeline Architecture

## Introduction

A production embedding pipeline has three main flows: ingestion (documents → embeddings → database), indexing (organizing for fast search), and query (user input → embedding → search → results). Each flow requires careful design for reliability, performance, and maintainability.

This lesson covers end-to-end pipeline architecture with code examples.

### What We'll Cover

- Ingestion pipeline design
- Async processing with queues
- Query pipeline optimization
- Batch vs real-time processing
- Pipeline orchestration patterns

### Prerequisites

- Understanding of [document chunking](../07-document-chunking/)
- Familiarity with message queues (Kafka, RabbitMQ)
- Basic knowledge of async programming

---

## The Three Pipeline Flows

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding System Data Flows                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FLOW 1: INGESTION (Write Path)                                 │
│  ─────────────────────────────────                              │
│  Source → Extract → Chunk → Embed → Store                       │
│                                                                 │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐│
│  │  S3/   │──▶│  Parse │──▶│ Chunk  │──▶│ Embed  │──▶│ Vector ││
│  │  DB    │   │  Text  │   │  Text  │   │        │   │   DB   ││
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘│
│                                                                 │
│  FLOW 2: QUERY (Read Path)                                      │
│  ─────────────────────────                                      │
│  Query → Embed → Search → Rank → Return                         │
│                                                                 │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐│
│  │ User   │──▶│ Embed  │──▶│ Vector │──▶│ Rerank │──▶│ Results││
│  │ Query  │   │ Query  │   │ Search │   │        │   │        ││
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘│
│                                                                 │
│  FLOW 3: MAINTENANCE (Background)                               │
│  ─────────────────────────────────                              │
│  Re-index → Update → Cleanup → Optimize                         │
│                                                                 │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐             │
│  │ Detect │──▶│ Re-    │──▶│ Delete │──▶│Compact ││
│  │ Stale  │   │ embed  │   │ Old    │   │ Index  ││
│  └────────┘   └────────┘   └────────┘   └────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline

### Basic Ingestion Flow

```python
from dataclasses import dataclass
from typing import List, Optional
import hashlib

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    source: str

@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None

class IngestionPipeline:
    """
    Production-ready ingestion pipeline.
    """
    def __init__(
        self,
        chunker,
        embedding_service,
        vector_db,
        batch_size: int = 100
    ):
        self.chunker = chunker
        self.embedder = embedding_service
        self.vector_db = vector_db
        self.batch_size = batch_size
    
    def process_document(self, document: Document) -> List[Chunk]:
        """Process a single document through the pipeline."""
        # Step 1: Chunk the document
        chunks = self.chunker.chunk(document)
        
        # Step 2: Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Step 3: Store in vector database
        self._store_chunks(chunks)
        
        return chunks
    
    def process_batch(self, documents: List[Document]) -> int:
        """Process multiple documents efficiently."""
        all_chunks = []
        
        # Chunk all documents
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        # Batch embed
        texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Batch store
        self._store_chunks_batch(all_chunks)
        
        return len(all_chunks)
    
    def _store_chunks(self, chunks: List[Chunk]):
        """Store chunks in vector database."""
        vectors = [
            {
                "id": chunk.id,
                "values": chunk.embedding,
                "metadata": {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "content": chunk.content[:1000]  # Store truncated for retrieval
                }
            }
            for chunk in chunks
        ]
        
        self.vector_db.upsert(vectors)
    
    def _store_chunks_batch(self, chunks: List[Chunk]):
        """Batch store with configurable batch size."""
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            self._store_chunks(batch)
```

### Document Change Detection

```python
import hashlib
from datetime import datetime
from typing import Dict, Set

class ChangeDetector:
    """
    Detect document changes to avoid re-embedding unchanged content.
    """
    def __init__(self, metadata_store):
        self.metadata_store = metadata_store
    
    def compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def detect_changes(
        self, 
        documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """
        Categorize documents by change status.
        
        Returns:
            {
                "new": [...],      # Never seen before
                "modified": [...], # Content changed
                "unchanged": [...] # No changes
            }
        """
        result = {"new": [], "modified": [], "unchanged": []}
        
        for doc in documents:
            current_hash = self.compute_hash(doc.content)
            stored = self.metadata_store.get(doc.id)
            
            if stored is None:
                result["new"].append(doc)
            elif stored["content_hash"] != current_hash:
                result["modified"].append(doc)
            else:
                result["unchanged"].append(doc)
        
        return result
    
    def update_metadata(self, document: Document, chunk_ids: List[str]):
        """Record document state for future change detection."""
        self.metadata_store.set(document.id, {
            "content_hash": self.compute_hash(document.content),
            "chunk_ids": chunk_ids,
            "last_indexed": datetime.utcnow().isoformat(),
            "source": document.source
        })
```

---

## Async Processing with Queues

### Queue-Based Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Async Ingestion Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │ Producer │    │            Message Queue                 │   │
│  │          │───▶│  ┌───┬───┬───┬───┬───┬───┬───┬───┐     │   │
│  │ (API/    │    │  │ D │ D │ D │ D │ D │ D │ D │ D │     │   │
│  │  Webhook)│    │  └───┴───┴───┴───┴───┴───┴───┴───┘     │   │
│  └──────────┘    └─────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Worker Pool                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker N │              │   │
│  │  │ Chunk    │  │ Chunk    │  │ Chunk    │              │   │
│  │  │ Embed    │  │ Embed    │  │ Embed    │              │   │
│  │  │ Store    │  │ Store    │  │ Store    │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Vector Database                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation with Celery

```python
from celery import Celery
from celery.exceptions import MaxRetriesExceededError
import logging

app = Celery('embedding_pipeline')
app.config_from_object('celeryconfig')

logger = logging.getLogger(__name__)

@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600
)
def process_document_task(self, document_data: dict):
    """
    Celery task for async document processing.
    
    Features:
    - Automatic retries with exponential backoff
    - Error tracking
    - Idempotent processing
    """
    try:
        document = Document(**document_data)
        
        # Check if already processed (idempotency)
        if is_recently_processed(document.id):
            logger.info(f"Document {document.id} already processed, skipping")
            return {"status": "skipped", "document_id": document.id}
        
        # Process through pipeline
        pipeline = get_pipeline()  # Singleton pattern
        chunks = pipeline.process_document(document)
        
        # Record completion
        mark_processed(document.id)
        
        return {
            "status": "success",
            "document_id": document.id,
            "chunks_created": len(chunks)
        }
        
    except EmbeddingServiceError as e:
        logger.error(f"Embedding failed for {document_data.get('id')}: {e}")
        raise self.retry(exc=e)
    
    except VectorDBError as e:
        logger.error(f"Vector DB failed for {document_data.get('id')}: {e}")
        raise self.retry(exc=e)
    
    except MaxRetriesExceededError:
        # Send to dead letter queue
        send_to_dlq(document_data, "max_retries_exceeded")
        raise

@app.task(bind=True)
def process_batch_task(self, document_ids: List[str]):
    """
    Batch processing task for bulk operations.
    """
    documents = fetch_documents(document_ids)
    
    # Fan out to individual tasks for reliability
    # (one failure doesn't affect others)
    job = group(
        process_document_task.s(doc.to_dict()) 
        for doc in documents
    )
    
    result = job.apply_async()
    return {"batch_id": self.request.id, "task_count": len(documents)}
```

### Producer Pattern

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

class IngestRequest(BaseModel):
    source_url: str
    document_type: str
    priority: str = "normal"

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    """
    API endpoint that queues documents for processing.
    """
    # Validate and extract document
    document = await extract_document(request.source_url)
    
    # Queue for async processing
    if request.priority == "high":
        # High priority: immediate processing
        task = process_document_task.apply_async(
            args=[document.to_dict()],
            queue="high_priority"
        )
    else:
        # Normal priority: batch-friendly queue
        task = process_document_task.apply_async(
            args=[document.to_dict()],
            queue="default"
        )
    
    return {
        "status": "queued",
        "task_id": task.id,
        "document_id": document.id
    }

@app.post("/ingest/batch")
async def ingest_batch(document_urls: List[str]):
    """
    Batch ingestion endpoint.
    """
    documents = await extract_documents_parallel(document_urls)
    
    task = process_batch_task.apply_async(
        args=[[doc.id for doc in documents]]
    )
    
    return {
        "status": "queued",
        "batch_task_id": task.id,
        "document_count": len(documents)
    }
```

---

## Query Pipeline

### Query Flow

```python
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class SearchResult:
    id: str
    score: float
    content: str
    metadata: dict

@dataclass
class QueryMetrics:
    embed_time_ms: float
    search_time_ms: float
    rerank_time_ms: float
    total_time_ms: float
    cache_hit: bool

class QueryPipeline:
    """
    Production query pipeline with caching and metrics.
    """
    def __init__(
        self,
        embedding_service,
        vector_db,
        cache,
        reranker=None
    ):
        self.embedder = embedding_service
        self.vector_db = vector_db
        self.cache = cache
        self.reranker = reranker
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        rerank: bool = True
    ) -> tuple[List[SearchResult], QueryMetrics]:
        """
        Execute search with full metrics tracking.
        """
        start_time = time.perf_counter()
        cache_hit = False
        
        # Step 1: Check cache for query embedding
        cache_key = f"query_embed:{hash(query)}"
        query_embedding = self.cache.get(cache_key)
        
        embed_start = time.perf_counter()
        if query_embedding is None:
            query_embedding = self.embedder.embed(query)
            self.cache.set(cache_key, query_embedding, ttl=3600)
        else:
            cache_hit = True
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Step 2: Vector search
        search_start = time.perf_counter()
        
        # Fetch more than needed if reranking
        fetch_k = top_k * 3 if rerank and self.reranker else top_k
        
        raw_results = self.vector_db.search(
            query_embedding,
            top_k=fetch_k,
            filter=filters
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Step 3: Optional reranking
        rerank_start = time.perf_counter()
        if rerank and self.reranker:
            results = self._rerank(query, raw_results, top_k)
        else:
            results = self._format_results(raw_results[:top_k])
        rerank_time = (time.perf_counter() - rerank_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        metrics = QueryMetrics(
            embed_time_ms=embed_time,
            search_time_ms=search_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            cache_hit=cache_hit
        )
        
        return results, metrics
    
    def _rerank(
        self, 
        query: str, 
        candidates: List, 
        top_k: int
    ) -> List[SearchResult]:
        """Rerank candidates using cross-encoder."""
        texts = [c.metadata.get("content", "") for c in candidates]
        scores = self.reranker.score(query, texts)
        
        # Combine with original scores
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            SearchResult(
                id=c.id,
                score=rerank_score,
                content=c.metadata.get("content", ""),
                metadata=c.metadata
            )
            for c, rerank_score in ranked
        ]
    
    def _format_results(self, raw_results: List) -> List[SearchResult]:
        """Format raw vector DB results."""
        return [
            SearchResult(
                id=r.id,
                score=r.score,
                content=r.metadata.get("content", ""),
                metadata=r.metadata
            )
            for r in raw_results
        ]
```

### Query API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

app = FastAPI()
logger = structlog.get_logger()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[dict] = None
    rerank: bool = True
    
class SearchResponse(BaseModel):
    results: List[SearchResult]
    metrics: QueryMetrics

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search endpoint with comprehensive error handling.
    """
    try:
        pipeline = get_query_pipeline()
        
        results, metrics = pipeline.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            rerank=request.rerank
        )
        
        # Log for observability
        logger.info(
            "search_completed",
            query_length=len(request.query),
            result_count=len(results),
            total_time_ms=metrics.total_time_ms,
            cache_hit=metrics.cache_hit
        )
        
        return SearchResponse(results=results, metrics=metrics)
    
    except EmbeddingServiceError as e:
        logger.error("embedding_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Embedding service unavailable")
    
    except VectorDBError as e:
        logger.error("vectordb_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Search service unavailable")
```

---

## Batch vs Real-Time Processing

### Decision Framework

| Factor | Real-Time | Batch |
|--------|-----------|-------|
| Latency requirement | < 1 second | Hours acceptable |
| Volume | Low-medium | High |
| Cost efficiency | Lower | Higher |
| Complexity | Higher | Lower |
| Use case | User uploads | Bulk imports |

### Hybrid Approach

```python
class HybridIngestionService:
    """
    Combines real-time and batch processing based on load.
    """
    def __init__(
        self,
        pipeline: IngestionPipeline,
        batch_threshold: int = 50,
        batch_interval_seconds: int = 30
    ):
        self.pipeline = pipeline
        self.batch_threshold = batch_threshold
        self.batch_interval = batch_interval_seconds
        self.pending_documents = []
        self.last_flush = time.time()
    
    async def ingest(self, document: Document, priority: str = "normal"):
        """
        Ingest document using appropriate strategy.
        """
        if priority == "high":
            # Real-time: process immediately
            return await self._process_realtime(document)
        else:
            # Batch: queue for batch processing
            return await self._queue_for_batch(document)
    
    async def _process_realtime(self, document: Document):
        """Process single document immediately."""
        chunks = self.pipeline.process_document(document)
        return {"status": "completed", "chunks": len(chunks)}
    
    async def _queue_for_batch(self, document: Document):
        """Add to batch queue, flush if threshold reached."""
        self.pending_documents.append(document)
        
        should_flush = (
            len(self.pending_documents) >= self.batch_threshold or
            time.time() - self.last_flush > self.batch_interval
        )
        
        if should_flush:
            await self._flush_batch()
        
        return {"status": "queued", "queue_size": len(self.pending_documents)}
    
    async def _flush_batch(self):
        """Process all pending documents as batch."""
        if not self.pending_documents:
            return
        
        documents = self.pending_documents
        self.pending_documents = []
        self.last_flush = time.time()
        
        # Process as batch (more efficient)
        count = self.pipeline.process_batch(documents)
        
        logger.info(
            "batch_processed",
            document_count=len(documents),
            chunk_count=count
        )
```

---

## Pipeline Orchestration

### Workflow with Temporal

```python
from temporalio import workflow, activity
from datetime import timedelta

@activity.defn
async def extract_document_activity(source_url: str) -> dict:
    """Extract document from source."""
    document = await extract_document(source_url)
    return document.to_dict()

@activity.defn
async def chunk_document_activity(document_data: dict) -> List[dict]:
    """Chunk document into smaller pieces."""
    document = Document(**document_data)
    chunks = chunker.chunk(document)
    return [c.to_dict() for c in chunks]

@activity.defn
async def embed_chunks_activity(chunk_data: List[dict]) -> List[dict]:
    """Generate embeddings for chunks."""
    texts = [c["content"] for c in chunk_data]
    embeddings = embedder.embed_batch(texts)
    
    for chunk, embedding in zip(chunk_data, embeddings):
        chunk["embedding"] = embedding
    
    return chunk_data

@activity.defn
async def store_chunks_activity(chunk_data: List[dict]) -> int:
    """Store chunks in vector database."""
    vectors = [
        {"id": c["id"], "values": c["embedding"], "metadata": c}
        for c in chunk_data
    ]
    vector_db.upsert(vectors)
    return len(vectors)

@workflow.defn
class DocumentIngestionWorkflow:
    """
    Temporal workflow for document ingestion.
    
    Benefits:
    - Automatic retries per activity
    - Visibility into progress
    - Resume after failures
    """
    @workflow.run
    async def run(self, source_url: str) -> dict:
        # Step 1: Extract
        document_data = await workflow.execute_activity(
            extract_document_activity,
            source_url,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Step 2: Chunk
        chunks = await workflow.execute_activity(
            chunk_document_activity,
            document_data,
            start_to_close_timeout=timedelta(minutes=2)
        )
        
        # Step 3: Embed
        embedded_chunks = await workflow.execute_activity(
            embed_chunks_activity,
            chunks,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Step 4: Store
        stored_count = await workflow.execute_activity(
            store_chunks_activity,
            embedded_chunks,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        return {
            "status": "completed",
            "document_id": document_data["id"],
            "chunks_stored": stored_count
        }
```

---

## Summary

✅ **Ingestion pipelines handle document → embedding → storage flow**  
✅ **Async processing with queues enables scalability and reliability**  
✅ **Query pipelines include caching and optional reranking**  
✅ **Hybrid batch/real-time processing optimizes for different use cases**  
✅ **Workflow orchestration provides visibility and failure recovery**

---

**Next:** [Embedding Versioning →](./02-embedding-versioning.md)

---

<!-- 
Sources Consulted:
- Celery Documentation: https://docs.celeryq.dev/
- Temporal Workflows: https://docs.temporal.io/
- Redis Caching: https://redis.io/docs/
-->
