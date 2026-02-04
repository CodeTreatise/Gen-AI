---
title: "Testing Strategies"
---

# Testing Strategies

## Introduction

Testing embedding systems is challenging because they combine deterministic code (pipeline logic) with probabilistic components (embeddings, similarity scores). You need to test that your pipeline works correctly, that search quality meets expectations, and that the system performs under load—all while dealing with external API dependencies and vector database state.

This lesson covers unit testing, integration testing, retrieval quality regression tests, and load testing strategies for production embedding systems.

### What We'll Cover

- Unit testing embedding pipeline components
- Integration testing with vector databases
- Retrieval quality regression testing
- Load testing and benchmarking
- Testing with mocked vs real services
- CI/CD integration strategies

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Familiarity with pytest and testing concepts
- Basic knowledge of load testing tools

---

## Testing Pyramid for Embedding Systems

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding System Testing Pyramid                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         ▲                                       │
│                        /│\      End-to-End Tests               │
│                       / │ \     • Full system integration       │
│                      /  │  \    • Real APIs (staging)           │
│                     /   │   \   • ~5% of tests                  │
│                    ─────┴─────                                  │
│                   /           \                                 │
│                  /  Integration \   • Vector DB + Cache         │
│                 /    Tests       \  • Mocked embedding API      │
│                /                  \ • ~25% of tests             │
│               ─────────────────────                             │
│              /                     \                            │
│             /      Unit Tests       \  • Pipeline components    │
│            /                         \ • Pure logic             │
│           /                           \• ~70% of tests          │
│          ───────────────────────────────                        │
│                                                                 │
│        + Quality Regression Tests (separate track)              │
│        + Load Tests (separate track)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Unit Testing Pipeline Components

### Testing Document Chunking

```python
import pytest
from typing import List

# Assume this is your chunking implementation
class TextChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        
        return chunks

class TestTextChunker:
    """Unit tests for text chunking."""
    
    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size returns as single chunk."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "Short text"
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_overlap_creates_redundancy(self):
        """Overlapping chunks share content."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 100  # 100 characters
        
        chunks = chunker.chunk(text)
        
        # With chunk_size=50, overlap=10:
        # Chunk 1: 0-50, Chunk 2: 40-90, Chunk 3: 80-100
        assert len(chunks) == 3
        # Check overlap exists
        assert chunks[0][-10:] == chunks[1][:10]
    
    def test_preserves_all_content(self):
        """All original content appears in at least one chunk."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "The quick brown fox jumps over the lazy dog. " * 5
        
        chunks = chunker.chunk(text)
        
        # Reconstruct (accounting for overlap)
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            # Add non-overlapping portion
            reconstructed += chunk[10:]  # Skip overlap
        
        # Original text should be subset of reconstruction
        assert text in reconstructed or reconstructed.startswith(text)
    
    def test_empty_text_returns_empty_list(self):
        """Empty input returns empty output."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        chunks = chunker.chunk("")
        
        assert chunks == [""]  # or [] depending on your implementation
    
    @pytest.mark.parametrize("chunk_size,overlap,expected_chunks", [
        (100, 0, 1),    # No overlap, fits in one chunk
        (50, 0, 2),     # No overlap, splits evenly
        (50, 10, 3),    # With overlap
    ])
    def test_chunk_count_parametrized(self, chunk_size, overlap, expected_chunks):
        """Parametrized test for various chunk configurations."""
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        text = "A" * 100
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == expected_chunks
```

### Testing Embedding Service Wrapper

```python
import pytest
from unittest.mock import Mock, patch
from typing import List

class EmbeddingService:
    """Wrapper around embedding API."""
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
    
    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

class TestEmbeddingService:
    """Unit tests for embedding service wrapper."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock OpenAI client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def embedding_service(self, mock_client):
        """Create embedding service with mock client."""
        return EmbeddingService(mock_client, model="text-embedding-3-small")
    
    def test_embed_single_text(self, embedding_service, mock_client):
        """Test embedding a single text."""
        # Setup mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        result = embedding_service.embed("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input="test text",
            model="text-embedding-3-small"
        )
    
    def test_embed_batch(self, embedding_service, mock_client):
        """Test batch embedding."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        result = embedding_service.embed_batch(["text1", "text2"])
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
    
    def test_embed_preserves_order(self, embedding_service, mock_client):
        """Batch embeddings maintain input order."""
        texts = ["first", "second", "third"]
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[1.0]),
            Mock(embedding=[2.0]),
            Mock(embedding=[3.0])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        result = embedding_service.embed_batch(texts)
        
        assert len(result) == 3
        assert result[0] == [1.0]
        assert result[2] == [3.0]
```

### Testing Search Pipeline Logic

```python
import pytest
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    id: str
    score: float
    content: str

class SearchPipeline:
    """Search pipeline with filtering and reranking."""
    
    def __init__(self, vector_db, reranker=None):
        self.vector_db = vector_db
        self.reranker = reranker
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Execute search with optional filtering and reranking."""
        # Fetch more if reranking
        fetch_k = top_k * 3 if rerank and self.reranker else top_k
        
        results = self.vector_db.search(query_embedding, limit=fetch_k)
        
        # Filter by minimum score
        results = [r for r in results if r.score >= min_score]
        
        # Optional reranking
        if rerank and self.reranker:
            results = self.reranker.rerank(results)
        
        return results[:top_k]

class TestSearchPipeline:
    """Unit tests for search pipeline logic."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database."""
        db = Mock()
        db.search.return_value = [
            SearchResult("doc1", 0.95, "High relevance"),
            SearchResult("doc2", 0.75, "Medium relevance"),
            SearchResult("doc3", 0.45, "Low relevance"),
        ]
        return db
    
    @pytest.fixture
    def mock_reranker(self):
        """Mock reranker that reverses order."""
        reranker = Mock()
        reranker.rerank.side_effect = lambda results: list(reversed(results))
        return reranker
    
    def test_search_returns_top_k_results(self, mock_vector_db):
        """Search returns at most top_k results."""
        pipeline = SearchPipeline(mock_vector_db)
        
        results = pipeline.search([0.1, 0.2], top_k=2, rerank=False)
        
        assert len(results) == 2
    
    def test_search_filters_by_min_score(self, mock_vector_db):
        """Results below min_score are filtered out."""
        pipeline = SearchPipeline(mock_vector_db)
        
        results = pipeline.search([0.1, 0.2], top_k=10, min_score=0.7, rerank=False)
        
        assert len(results) == 2
        assert all(r.score >= 0.7 for r in results)
    
    def test_search_with_reranking(self, mock_vector_db, mock_reranker):
        """Reranking modifies result order."""
        pipeline = SearchPipeline(mock_vector_db, mock_reranker)
        
        results = pipeline.search([0.1, 0.2], top_k=3, rerank=True)
        
        mock_reranker.rerank.assert_called_once()
        # Reranker reverses order
        assert results[0].id == "doc3"  # Was last, now first
    
    def test_search_without_reranker_skips_reranking(self, mock_vector_db):
        """No reranking when reranker not provided."""
        pipeline = SearchPipeline(mock_vector_db, reranker=None)
        
        results = pipeline.search([0.1, 0.2], top_k=3, rerank=True)
        
        # Original order preserved
        assert results[0].id == "doc1"
```

---

## Integration Testing

### Testing with Real Vector Database

```python
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class TestVectorDBIntegration:
    """Integration tests with real Qdrant instance."""
    
    @pytest.fixture(scope="class")
    def qdrant_client(self):
        """Connect to test Qdrant instance."""
        client = QdrantClient(url="http://localhost:6333")
        yield client
    
    @pytest.fixture
    def test_collection(self, qdrant_client):
        """Create and cleanup test collection."""
        collection_name = f"test_collection_{pytest.current_test_name}"
        
        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3,  # Small for testing
                distance=Distance.COSINE
            )
        )
        
        yield collection_name
        
        # Cleanup
        qdrant_client.delete_collection(collection_name)
    
    def test_upsert_and_search(self, qdrant_client, test_collection):
        """Test basic upsert and search operations."""
        # Insert test data
        qdrant_client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "doc1"}),
                PointStruct(id=2, vector=[0.0, 1.0, 0.0], payload={"text": "doc2"}),
                PointStruct(id=3, vector=[0.0, 0.0, 1.0], payload={"text": "doc3"}),
            ]
        )
        
        # Search for similar to first vector
        results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=[0.9, 0.1, 0.0],
            limit=2
        )
        
        assert len(results) == 2
        assert results[0].id == 1  # Most similar
    
    def test_filtered_search(self, qdrant_client, test_collection):
        """Test search with metadata filters."""
        qdrant_client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"category": "A"}),
                PointStruct(id=2, vector=[0.9, 0.1, 0.0], payload={"category": "B"}),
                PointStruct(id=3, vector=[0.8, 0.2, 0.0], payload={"category": "A"}),
            ]
        )
        
        # Search only in category A
        results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=[1.0, 0.0, 0.0],
            query_filter={"must": [{"key": "category", "match": {"value": "A"}}]},
            limit=10
        )
        
        assert len(results) == 2
        assert all(r.payload["category"] == "A" for r in results)
```

### Docker Compose for Test Infrastructure

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  qdrant-test:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    environment:
      - QDRANT__LOG_LEVEL=WARN
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis-test:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## Retrieval Quality Regression Tests

### Golden Query Test Suite

```python
import pytest
import json
from pathlib import Path
from typing import List, Dict

@dataclass
class GoldenQuery:
    query: str
    relevant_doc_ids: List[str]
    relevance_scores: Dict[str, int]  # doc_id -> 0-3 relevance

class TestRetrievalQuality:
    """
    Regression tests for search quality.
    Run against a known test dataset to detect quality degradation.
    """
    
    @pytest.fixture(scope="module")
    def golden_queries(self) -> List[GoldenQuery]:
        """Load golden query test set."""
        path = Path("tests/fixtures/golden_queries.json")
        with open(path) as f:
            data = json.load(f)
        
        return [
            GoldenQuery(
                query=item["query"],
                relevant_doc_ids=item["relevant_docs"],
                relevance_scores=item.get("relevance_scores", {})
            )
            for item in data
        ]
    
    @pytest.fixture(scope="module")
    def search_service(self):
        """Initialize search service with test index."""
        # Use test configuration pointing to test index
        return SearchService(config=TEST_CONFIG)
    
    def test_recall_at_10_above_threshold(
        self, 
        search_service, 
        golden_queries
    ):
        """Recall@10 must be at least 70%."""
        recalls = []
        
        for gq in golden_queries:
            results = search_service.search(gq.query, top_k=10)
            retrieved_ids = {r.id for r in results}
            relevant_ids = set(gq.relevant_doc_ids)
            
            recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
            recalls.append(recall)
        
        avg_recall = sum(recalls) / len(recalls)
        
        assert avg_recall >= 0.70, f"Recall@10 = {avg_recall:.2%}, expected >= 70%"
    
    def test_mrr_above_threshold(
        self,
        search_service,
        golden_queries
    ):
        """Mean Reciprocal Rank must be at least 0.5."""
        mrrs = []
        
        for gq in golden_queries:
            results = search_service.search(gq.query, top_k=10)
            relevant_ids = set(gq.relevant_doc_ids)
            
            mrr = 0.0
            for i, result in enumerate(results, 1):
                if result.id in relevant_ids:
                    mrr = 1.0 / i
                    break
            
            mrrs.append(mrr)
        
        avg_mrr = sum(mrrs) / len(mrrs)
        
        assert avg_mrr >= 0.50, f"MRR = {avg_mrr:.3f}, expected >= 0.50"
    
    def test_no_query_returns_zero_results(
        self,
        search_service,
        golden_queries
    ):
        """Every golden query should return at least one result."""
        zero_result_queries = []
        
        for gq in golden_queries:
            results = search_service.search(gq.query, top_k=10)
            if len(results) == 0:
                zero_result_queries.append(gq.query)
        
        assert len(zero_result_queries) == 0, (
            f"{len(zero_result_queries)} queries returned 0 results: "
            f"{zero_result_queries[:5]}"
        )
    
    @pytest.mark.slow
    def test_ndcg_above_threshold(
        self,
        search_service,
        golden_queries
    ):
        """NDCG@10 must be at least 0.6 (requires graded relevance)."""
        ndcgs = []
        
        for gq in golden_queries:
            if not gq.relevance_scores:
                continue  # Skip queries without graded relevance
            
            results = search_service.search(gq.query, top_k=10)
            
            # Calculate NDCG
            dcg = sum(
                (2 ** gq.relevance_scores.get(r.id, 0) - 1) / np.log2(i + 2)
                for i, r in enumerate(results)
            )
            
            ideal_scores = sorted(
                gq.relevance_scores.values(), 
                reverse=True
            )[:10]
            idcg = sum(
                (2 ** score - 1) / np.log2(i + 2)
                for i, score in enumerate(ideal_scores)
            )
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        if ndcgs:
            avg_ndcg = sum(ndcgs) / len(ndcgs)
            assert avg_ndcg >= 0.60, f"NDCG@10 = {avg_ndcg:.3f}, expected >= 0.60"
```

### Golden Query File Format

```json
// tests/fixtures/golden_queries.json
[
  {
    "query": "how to implement retry logic",
    "relevant_docs": ["doc_retry_patterns", "doc_error_handling", "doc_resilience"],
    "relevance_scores": {
      "doc_retry_patterns": 3,
      "doc_error_handling": 2,
      "doc_resilience": 1
    }
  },
  {
    "query": "vector database scaling",
    "relevant_docs": ["doc_sharding", "doc_replication", "doc_qdrant_cluster"],
    "relevance_scores": {
      "doc_sharding": 3,
      "doc_replication": 2,
      "doc_qdrant_cluster": 3
    }
  }
]
```

---

## Load Testing

### Locust Load Test

```python
# locustfile.py
from locust import HttpUser, task, between
import random

class SearchLoadTest(HttpUser):
    """Load test for search endpoint."""
    
    wait_time = between(0.5, 2.0)  # Seconds between requests
    
    # Sample queries for load testing
    queries = [
        "how to implement caching",
        "vector database performance",
        "embedding model comparison",
        "retry with exponential backoff",
        "circuit breaker pattern",
    ]
    
    @task(10)
    def search(self):
        """Primary search endpoint - 10x weight."""
        query = random.choice(self.queries)
        
        with self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("results", [])) == 0:
                    response.failure("Zero results returned")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def search_with_filters(self):
        """Filtered search - 1x weight."""
        query = random.choice(self.queries)
        
        self.client.post(
            "/search",
            json={
                "query": query,
                "top_k": 10,
                "filters": {"category": "technical"}
            }
        )
    
    @task(1)
    def health_check(self):
        """Health endpoint - 1x weight."""
        self.client.get("/health")

# Run with: locust -f locustfile.py --host=http://localhost:8000
```

### Performance Benchmarks

```python
import pytest
import time
import statistics
from typing import List

class TestPerformanceBenchmarks:
    """Performance benchmarks for embedding pipeline."""
    
    @pytest.fixture
    def search_service(self):
        """Search service configured for benchmarking."""
        return SearchService(config=BENCHMARK_CONFIG)
    
    @pytest.mark.benchmark
    def test_search_latency_p99(self, search_service):
        """P99 search latency must be under 200ms."""
        queries = ["test query"] * 100
        latencies = []
        
        for query in queries:
            start = time.perf_counter()
            search_service.search(query, top_k=10)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        assert p99 < 200, f"P99 latency = {p99:.1f}ms, expected < 200ms"
    
    @pytest.mark.benchmark
    def test_embedding_throughput(self, search_service):
        """Embedding service must handle 100+ embeddings/second."""
        texts = ["Sample text for embedding"] * 100
        
        start = time.perf_counter()
        for text in texts:
            search_service.embed(text)
        duration = time.perf_counter() - start
        
        throughput = len(texts) / duration
        
        assert throughput >= 100, (
            f"Throughput = {throughput:.1f}/sec, expected >= 100/sec"
        )
    
    @pytest.mark.benchmark
    def test_batch_embedding_efficiency(self, search_service):
        """Batch embedding should be at least 2x faster than individual."""
        texts = ["Sample text for embedding"] * 50
        
        # Individual embedding time
        start = time.perf_counter()
        for text in texts:
            search_service.embed(text)
        individual_time = time.perf_counter() - start
        
        # Batch embedding time
        start = time.perf_counter()
        search_service.embed_batch(texts)
        batch_time = time.perf_counter() - start
        
        speedup = individual_time / batch_time
        
        assert speedup >= 2.0, (
            f"Batch speedup = {speedup:.1f}x, expected >= 2.0x"
        )
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Embedding Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run unit tests
        run: pytest tests/unit -v --cov=src
  
  integration-tests:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      
      - name: Wait for services
        run: |
          until curl -s http://localhost:6333/readyz; do sleep 1; done
          until redis-cli ping; do sleep 1; done
      
      - name: Run integration tests
        run: pytest tests/integration -v
        env:
          QDRANT_URL: http://localhost:6333
          REDIS_URL: redis://localhost:6379
  
  quality-regression:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      
      - name: Run quality regression tests
        run: pytest tests/quality -v --tb=short
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          TEST_INDEX_URL: ${{ secrets.TEST_INDEX_URL }}
```

---

## Summary

✅ **Unit test pipeline components with mocked dependencies**  
✅ **Integration test with real vector databases using Docker**  
✅ **Maintain golden query sets for quality regression detection**  
✅ **Benchmark P99 latency and throughput requirements**  
✅ **Automate testing in CI/CD with quality gates**

---

**Previous:** [← Failure Handling](./06-failure-handling.md)  
**Back to:** [Production Embedding Systems Overview](./00-production-embedding-systems.md)

---

<!-- 
Sources Consulted:
- Pytest Documentation: https://docs.pytest.org/
- Locust Load Testing: https://locust.io/
- Testing Best Practices: https://martinfowler.com/articles/practical-test-pyramid.html
-->
