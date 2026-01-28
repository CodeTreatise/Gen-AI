---
title: "File Search Configuration"
---

# File Search Configuration

## Introduction

OpenAI's file search tool uses vector stores to enable semantic search over your documents. Understanding the configuration options helps you optimize search quality, performance, and relevance.

### What We'll Cover

- Vector store creation and management
- Ranking options and score thresholds
- Chunking strategies
- Metadata filtering
- Performance optimization

### Prerequisites

- Understanding of vector embeddings
- File search tool basics
- OpenAI API access

---

## Vector Store Fundamentals

### Creating Vector Stores

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

client = OpenAI()

# Create a basic vector store
store = client.vector_stores.create(
    name="documentation-store"
)

print(f"Store ID: {store.id}")
print(f"Status: {store.status}")
```

### Vector Store Manager

```python
@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    
    name: str
    description: Optional[str] = None
    expires_after_days: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class VectorStoreManager:
    """Manage vector stores for file search."""
    
    def __init__(self):
        self.client = OpenAI()
        self.stores: Dict[str, str] = {}  # name -> id
    
    def create(self, config: VectorStoreConfig) -> str:
        """Create a new vector store."""
        
        create_args = {
            "name": config.name
        }
        
        if config.expires_after_days:
            create_args["expires_after"] = {
                "anchor": "last_active_at",
                "days": config.expires_after_days
            }
        
        if config.metadata:
            create_args["metadata"] = config.metadata
        
        store = self.client.vector_stores.create(**create_args)
        self.stores[config.name] = store.id
        
        return store.id
    
    def get_or_create(self, config: VectorStoreConfig) -> str:
        """Get existing store or create new one."""
        
        # Check cache
        if config.name in self.stores:
            return self.stores[config.name]
        
        # Search existing stores
        stores = self.client.vector_stores.list()
        for store in stores.data:
            if store.name == config.name:
                self.stores[config.name] = store.id
                return store.id
        
        # Create new
        return self.create(config)
    
    def get_status(self, store_id: str) -> dict:
        """Get vector store status."""
        
        store = self.client.vector_stores.retrieve(store_id)
        
        return {
            "id": store.id,
            "name": store.name,
            "status": store.status,
            "file_counts": {
                "total": store.file_counts.total,
                "completed": store.file_counts.completed,
                "in_progress": store.file_counts.in_progress,
                "failed": store.file_counts.failed
            }
        }
    
    def delete(self, store_id: str):
        """Delete a vector store."""
        self.client.vector_stores.delete(store_id)
        
        # Remove from cache
        self.stores = {
            name: sid for name, sid in self.stores.items()
            if sid != store_id
        }


# Usage
manager = VectorStoreManager()

store_id = manager.get_or_create(VectorStoreConfig(
    name="api-documentation",
    description="API reference documentation",
    expires_after_days=30,
    metadata={"version": "v2", "category": "docs"}
))

status = manager.get_status(store_id)
print(f"Files: {status['file_counts']['completed']}/{status['file_counts']['total']}")
```

---

## Ranking Options

### Ranker Configuration

```python
class RankerType(Enum):
    AUTO = "auto"
    DEFAULT_2024_08_21 = "default_2024_08_21"


@dataclass
class RankingOptions:
    """File search ranking configuration."""
    
    ranker: RankerType = RankerType.AUTO
    score_threshold: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> dict:
        return {
            "ranker": self.ranker.value,
            "score_threshold": self.score_threshold
        }


class FileSearchWithRanking:
    """File search with ranking options."""
    
    def __init__(
        self,
        store_id: str,
        ranking_options: RankingOptions = None
    ):
        self.client = OpenAI()
        self.store_id = store_id
        self.ranking = ranking_options or RankingOptions()
    
    def search(
        self,
        query: str,
        max_results: int = 10
    ) -> dict:
        """Search with ranking options."""
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.store_id],
                "max_num_results": max_results,
                "ranking_options": self.ranking.to_dict()
            }],
            input=query
        )
        
        results = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                score = getattr(item, 'score', 0)
                
                # Only include if above threshold
                if score >= self.ranking.score_threshold:
                    results.append({
                        "file": getattr(item, 'file_name', ''),
                        "score": score,
                        "content": getattr(item, 'content', '')
                    })
        
        return {
            "answer": response.output_text,
            "results": sorted(results, key=lambda x: x['score'], reverse=True)
        }


# Usage
search = FileSearchWithRanking(
    store_id="vs_abc123",
    ranking_options=RankingOptions(
        ranker=RankerType.AUTO,
        score_threshold=0.5  # Only high-confidence results
    )
)

result = search.search("How to implement authentication?")
print(f"Found {len(result['results'])} results above threshold")
```

### Dynamic Threshold Adjustment

```python
class AdaptiveThreshold:
    """Dynamically adjust score threshold based on results."""
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        min_results: int = 3,
        max_results: int = 10
    ):
        self.threshold = initial_threshold
        self.min_results = min_results
        self.max_results = max_results
        self.threshold_history: List[float] = []
    
    def adjust(self, result_count: int, avg_score: float) -> float:
        """Adjust threshold based on results."""
        
        self.threshold_history.append(self.threshold)
        
        if result_count < self.min_results:
            # Lower threshold to get more results
            self.threshold = max(0.1, self.threshold - 0.1)
        elif result_count > self.max_results and avg_score > self.threshold:
            # Raise threshold to filter noise
            self.threshold = min(0.9, self.threshold + 0.05)
        
        return self.threshold
    
    def get_recommendation(self) -> str:
        """Get threshold recommendation."""
        
        if len(self.threshold_history) < 5:
            return "Insufficient data for recommendation"
        
        avg_threshold = sum(self.threshold_history[-10:]) / len(self.threshold_history[-10:])
        
        if avg_threshold < 0.3:
            return "Consider improving document quality or query specificity"
        elif avg_threshold > 0.7:
            return "Search is highly selective - good for precision"
        else:
            return f"Optimal threshold range: {avg_threshold:.2f}"


class SmartFileSearch:
    """File search with adaptive thresholds."""
    
    def __init__(self, store_id: str):
        self.client = OpenAI()
        self.store_id = store_id
        self.adaptive = AdaptiveThreshold()
    
    def search(self, query: str) -> dict:
        """Search with adaptive threshold."""
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.store_id],
                "max_num_results": 20,  # Get more, filter later
                "ranking_options": {
                    "ranker": "auto",
                    "score_threshold": 0.0  # Get all, filter ourselves
                }
            }],
            input=query
        )
        
        # Collect all results with scores
        all_results = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                all_results.append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0),
                    "content": getattr(item, 'content', '')
                })
        
        # Filter by threshold
        filtered = [
            r for r in all_results
            if r['score'] >= self.adaptive.threshold
        ]
        
        # Adjust for next time
        if all_results:
            avg_score = sum(r['score'] for r in all_results) / len(all_results)
            self.adaptive.adjust(len(filtered), avg_score)
        
        return {
            "answer": response.output_text,
            "results": filtered,
            "threshold_used": self.adaptive.threshold,
            "total_found": len(all_results)
        }
```

---

## Chunking Strategies

### File Chunking Configuration

```python
class ChunkingStrategy(Enum):
    AUTO = "auto"
    STATIC = "static"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    
    strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    max_chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 400
    
    def to_dict(self) -> dict:
        if self.strategy == ChunkingStrategy.AUTO:
            return {"type": "auto"}
        
        return {
            "type": "static",
            "static": {
                "max_chunk_size_tokens": self.max_chunk_size_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens
            }
        }


class FileUploader:
    """Upload files with chunking configuration."""
    
    def __init__(
        self,
        store_id: str,
        chunking: ChunkingConfig = None
    ):
        self.client = OpenAI()
        self.store_id = store_id
        self.chunking = chunking or ChunkingConfig()
    
    def upload(self, file_path: str) -> str:
        """Upload file with chunking."""
        
        # Upload file
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(
                file=f,
                purpose="assistants"
            )
        
        # Add to vector store with chunking
        self.client.vector_stores.files.create(
            vector_store_id=self.store_id,
            file_id=file_obj.id,
            chunking_strategy=self.chunking.to_dict()
        )
        
        return file_obj.id
    
    def upload_batch(
        self,
        file_paths: List[str],
        wait_for_completion: bool = True
    ) -> List[str]:
        """Upload multiple files."""
        
        file_ids = []
        
        # Upload all files first
        for path in file_paths:
            with open(path, "rb") as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
                file_ids.append(file_obj.id)
        
        # Add to vector store as batch
        batch = self.client.vector_stores.file_batches.create(
            vector_store_id=self.store_id,
            file_ids=file_ids,
            chunking_strategy=self.chunking.to_dict()
        )
        
        if wait_for_completion:
            self._wait_for_batch(batch.id)
        
        return file_ids
    
    def _wait_for_batch(self, batch_id: str, timeout: int = 300):
        """Wait for batch processing."""
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            batch = self.client.vector_stores.file_batches.retrieve(
                vector_store_id=self.store_id,
                batch_id=batch_id
            )
            
            if batch.status == "completed":
                return
            elif batch.status == "failed":
                raise Exception(f"Batch failed: {batch.id}")
            
            time.sleep(2)
        
        raise TimeoutError("Batch processing timeout")


# Usage
uploader = FileUploader(
    store_id="vs_abc123",
    chunking=ChunkingConfig(
        strategy=ChunkingStrategy.STATIC,
        max_chunk_size_tokens=500,  # Smaller for precise retrieval
        chunk_overlap_tokens=100
    )
)

# file_id = uploader.upload("documentation.pdf")
```

### Chunking Optimization

```python
@dataclass
class DocumentType:
    """Document type with optimal chunking."""
    
    file_extension: str
    recommended_chunk_size: int
    recommended_overlap: int
    description: str


DOCUMENT_TYPES = {
    ".pdf": DocumentType(".pdf", 800, 400, "PDF documents"),
    ".md": DocumentType(".md", 600, 200, "Markdown files"),
    ".txt": DocumentType(".txt", 500, 100, "Plain text"),
    ".json": DocumentType(".json", 400, 50, "JSON data"),
    ".py": DocumentType(".py", 300, 100, "Python code"),
    ".html": DocumentType(".html", 700, 300, "HTML pages"),
}


class SmartChunking:
    """Select chunking based on document type."""
    
    @staticmethod
    def get_config(file_path: str) -> ChunkingConfig:
        """Get optimal chunking for file type."""
        
        import os
        _, ext = os.path.splitext(file_path.lower())
        
        doc_type = DOCUMENT_TYPES.get(ext)
        
        if doc_type:
            return ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=doc_type.recommended_chunk_size,
                chunk_overlap_tokens=doc_type.recommended_overlap
            )
        
        # Default for unknown types
        return ChunkingConfig(strategy=ChunkingStrategy.AUTO)
    
    @staticmethod
    def analyze_file(file_path: str) -> dict:
        """Analyze file for chunking recommendations."""
        
        import os
        
        stat = os.stat(file_path)
        _, ext = os.path.splitext(file_path.lower())
        
        # Rough token estimate (4 chars per token)
        estimated_tokens = stat.st_size / 4
        
        doc_type = DOCUMENT_TYPES.get(ext)
        config = SmartChunking.get_config(file_path)
        
        return {
            "file_path": file_path,
            "extension": ext,
            "size_bytes": stat.st_size,
            "estimated_tokens": int(estimated_tokens),
            "estimated_chunks": int(estimated_tokens / config.max_chunk_size_tokens),
            "chunking_config": config,
            "document_type": doc_type.description if doc_type else "Unknown"
        }
```

---

## Metadata Filtering

### Adding Metadata

```python
@dataclass
class FileMetadata:
    """Metadata for searchable file."""
    
    category: str
    version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    created_at: Optional[str] = None
    custom: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API-compatible dict."""
        result = {
            "category": self.category
        }
        
        if self.version:
            result["version"] = self.version
        
        if self.tags:
            result["tags"] = ",".join(self.tags)
        
        if self.source:
            result["source"] = self.source
        
        if self.created_at:
            result["created_at"] = self.created_at
        
        result.update(self.custom)
        
        return result


class MetadataFileUploader:
    """Upload files with metadata."""
    
    def __init__(self, store_id: str):
        self.client = OpenAI()
        self.store_id = store_id
    
    def upload_with_metadata(
        self,
        file_path: str,
        metadata: FileMetadata
    ) -> str:
        """Upload file with metadata."""
        
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(
                file=f,
                purpose="assistants"
            )
        
        # Add to vector store with metadata
        self.client.vector_stores.files.create(
            vector_store_id=self.store_id,
            file_id=file_obj.id,
            metadata=metadata.to_dict()
        )
        
        return file_obj.id


# Usage
uploader = MetadataFileUploader("vs_abc123")

# file_id = uploader.upload_with_metadata(
#     "api-v2-docs.pdf",
#     FileMetadata(
#         category="api-reference",
#         version="2.0",
#         tags=["authentication", "endpoints", "rest"],
#         source="internal",
#         created_at="2025-01-15"
#     )
# )
```

### Filtered Search

```python
@dataclass
class SearchFilter:
    """Filter for file search."""
    
    category: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_filter(self) -> Optional[dict]:
        """Convert to filter expression."""
        conditions = []
        
        if self.category:
            conditions.append({
                "field": "metadata.category",
                "operator": "eq",
                "value": self.category
            })
        
        if self.version:
            conditions.append({
                "field": "metadata.version",
                "operator": "eq",
                "value": self.version
            })
        
        # Tags are stored as comma-separated
        if self.tags:
            for tag in self.tags:
                conditions.append({
                    "field": "metadata.tags",
                    "operator": "contains",
                    "value": tag
                })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {
            "operator": "and",
            "conditions": conditions
        }


class FilteredFileSearch:
    """File search with metadata filtering."""
    
    def __init__(self, store_id: str):
        self.client = OpenAI()
        self.store_id = store_id
    
    def search(
        self,
        query: str,
        filter: SearchFilter = None,
        max_results: int = 10
    ) -> dict:
        """Search with optional filters."""
        
        tool_config = {
            "type": "file_search",
            "vector_store_ids": [self.store_id],
            "max_num_results": max_results
        }
        
        if filter:
            filter_expr = filter.to_filter()
            if filter_expr:
                tool_config["filters"] = filter_expr
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[tool_config],
            input=query
        )
        
        return self._parse_results(response)
    
    def _parse_results(self, response) -> dict:
        """Parse search results."""
        results = []
        
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                results.append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0),
                    "content": getattr(item, 'content', ''),
                    "metadata": getattr(item, 'metadata', {})
                })
        
        return {
            "answer": response.output_text,
            "results": results
        }


# Usage
search = FilteredFileSearch("vs_abc123")

# Search only v2 API docs about authentication
result = search.search(
    "How to authenticate API requests?",
    filter=SearchFilter(
        category="api-reference",
        version="2.0",
        tags=["authentication"]
    )
)
```

---

## Performance Optimization

### Search Performance Monitor

```python
import time
from typing import Callable

@dataclass
class SearchMetrics:
    """Metrics for a search operation."""
    
    query: str
    latency_ms: float
    result_count: int
    avg_score: float
    store_id: str
    filters_used: bool


class PerformanceMonitor:
    """Monitor file search performance."""
    
    def __init__(self):
        self.metrics: List[SearchMetrics] = []
    
    def record(self, metrics: SearchMetrics):
        """Record search metrics."""
        self.metrics.append(metrics)
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        
        if not self.metrics:
            return {"message": "No metrics recorded"}
        
        latencies = [m.latency_ms for m in self.metrics]
        scores = [m.avg_score for m in self.metrics]
        result_counts = [m.result_count for m in self.metrics]
        
        return {
            "total_searches": len(self.metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "avg_result_count": sum(result_counts) / len(result_counts),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "filtered_searches_pct": sum(1 for m in self.metrics if m.filters_used) / len(self.metrics) * 100
        }
    
    def get_slow_queries(self, threshold_ms: float = 2000) -> List[SearchMetrics]:
        """Get queries slower than threshold."""
        return [m for m in self.metrics if m.latency_ms > threshold_ms]


class MonitoredFileSearch:
    """File search with performance monitoring."""
    
    def __init__(
        self,
        store_id: str,
        monitor: PerformanceMonitor = None
    ):
        self.client = OpenAI()
        self.store_id = store_id
        self.monitor = monitor or PerformanceMonitor()
    
    def search(
        self,
        query: str,
        filter: SearchFilter = None,
        max_results: int = 10
    ) -> dict:
        """Search with monitoring."""
        
        start = time.time()
        
        tool_config = {
            "type": "file_search",
            "vector_store_ids": [self.store_id],
            "max_num_results": max_results
        }
        
        if filter and filter.to_filter():
            tool_config["filters"] = filter.to_filter()
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[tool_config],
            input=query
        )
        
        latency_ms = (time.time() - start) * 1000
        
        # Parse results
        results = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                results.append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0)
                })
        
        # Record metrics
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        
        self.monitor.record(SearchMetrics(
            query=query[:100],  # Truncate
            latency_ms=latency_ms,
            result_count=len(results),
            avg_score=avg_score,
            store_id=self.store_id,
            filters_used=filter is not None
        ))
        
        return {
            "answer": response.output_text,
            "results": results,
            "latency_ms": latency_ms
        }


# Usage
monitor = PerformanceMonitor()
search = MonitoredFileSearch("vs_abc123", monitor)

# Perform searches
# result = search.search("authentication guide")
# result = search.search("rate limits")

# Check performance
stats = monitor.get_stats()
print(f"Average latency: {stats.get('avg_latency_ms', 0):.0f}ms")
```

### Optimized Search Configuration

```python
@dataclass
class OptimizedSearchConfig:
    """Optimized configuration based on use case."""
    
    use_case: str
    max_results: int
    score_threshold: float
    ranker: str
    chunking: ChunkingConfig
    
    @classmethod
    def for_precision(cls) -> 'OptimizedSearchConfig':
        """High precision configuration."""
        return cls(
            use_case="precision",
            max_results=5,
            score_threshold=0.7,
            ranker="auto",
            chunking=ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=400,
                chunk_overlap_tokens=100
            )
        )
    
    @classmethod
    def for_recall(cls) -> 'OptimizedSearchConfig':
        """High recall configuration."""
        return cls(
            use_case="recall",
            max_results=20,
            score_threshold=0.3,
            ranker="auto",
            chunking=ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=800,
                chunk_overlap_tokens=400
            )
        )
    
    @classmethod
    def for_code(cls) -> 'OptimizedSearchConfig':
        """Code search configuration."""
        return cls(
            use_case="code",
            max_results=10,
            score_threshold=0.5,
            ranker="auto",
            chunking=ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=300,
                chunk_overlap_tokens=50
            )
        )
    
    @classmethod
    def for_documentation(cls) -> 'OptimizedSearchConfig':
        """Documentation search configuration."""
        return cls(
            use_case="documentation",
            max_results=10,
            score_threshold=0.5,
            ranker="auto",
            chunking=ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=600,
                chunk_overlap_tokens=200
            )
        )
```

---

## Hands-on Exercise

### Your Task

Build a file search system with optimized configuration.

### Requirements

1. Create vector store with appropriate chunking
2. Implement filtered search
3. Add performance monitoring
4. Optimize based on metrics

<details>
<summary>💡 Hints</summary>

- Match chunking to document type
- Use metadata for filtering
- Track latency and score patterns
</details>

<details>
<summary>✅ Solution</summary>

```python
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path
import time

class SearchPurpose(Enum):
    DOCUMENTATION = "documentation"
    CODE = "code"
    FAQ = "faq"
    GENERAL = "general"


@dataclass
class OptimizedFileSearchSystem:
    """Complete file search system with optimization."""
    
    client: OpenAI = field(default_factory=OpenAI)
    store_id: Optional[str] = None
    purpose: SearchPurpose = SearchPurpose.GENERAL
    metrics: List[dict] = field(default_factory=list)
    
    def initialize(
        self,
        name: str,
        purpose: SearchPurpose = SearchPurpose.GENERAL
    ):
        """Initialize vector store."""
        
        self.purpose = purpose
        
        store = self.client.vector_stores.create(
            name=name,
            metadata={
                "purpose": purpose.value,
                "created": time.strftime("%Y-%m-%d")
            }
        )
        
        self.store_id = store.id
        return store.id
    
    def _get_chunking(self) -> ChunkingConfig:
        """Get chunking for purpose."""
        
        configs = {
            SearchPurpose.DOCUMENTATION: ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=600,
                chunk_overlap_tokens=200
            ),
            SearchPurpose.CODE: ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=300,
                chunk_overlap_tokens=50
            ),
            SearchPurpose.FAQ: ChunkingConfig(
                strategy=ChunkingStrategy.STATIC,
                max_chunk_size_tokens=400,
                chunk_overlap_tokens=100
            ),
            SearchPurpose.GENERAL: ChunkingConfig(
                strategy=ChunkingStrategy.AUTO
            )
        }
        
        return configs.get(self.purpose, ChunkingConfig())
    
    def _get_search_config(self) -> dict:
        """Get search config for purpose."""
        
        configs = {
            SearchPurpose.DOCUMENTATION: {
                "max_results": 10,
                "score_threshold": 0.5
            },
            SearchPurpose.CODE: {
                "max_results": 5,
                "score_threshold": 0.6
            },
            SearchPurpose.FAQ: {
                "max_results": 3,
                "score_threshold": 0.7
            },
            SearchPurpose.GENERAL: {
                "max_results": 10,
                "score_threshold": 0.4
            }
        }
        
        return configs.get(self.purpose, configs[SearchPurpose.GENERAL])
    
    def add_files(
        self,
        file_paths: List[str],
        metadata: Dict[str, str] = None
    ) -> List[str]:
        """Add files with optimized chunking."""
        
        if not self.store_id:
            raise ValueError("Initialize store first")
        
        chunking = self._get_chunking()
        file_ids = []
        
        for path in file_paths:
            with open(path, "rb") as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            self.client.vector_stores.files.create(
                vector_store_id=self.store_id,
                file_id=file_obj.id,
                chunking_strategy=chunking.to_dict(),
                metadata=metadata or {}
            )
            
            file_ids.append(file_obj.id)
        
        return file_ids
    
    def search(
        self,
        query: str,
        category_filter: Optional[str] = None
    ) -> dict:
        """Perform optimized search."""
        
        if not self.store_id:
            raise ValueError("Initialize store first")
        
        start = time.time()
        config = self._get_search_config()
        
        tool_config = {
            "type": "file_search",
            "vector_store_ids": [self.store_id],
            "max_num_results": config["max_results"],
            "ranking_options": {
                "ranker": "auto",
                "score_threshold": config["score_threshold"]
            }
        }
        
        if category_filter:
            tool_config["filters"] = {
                "field": "metadata.category",
                "operator": "eq",
                "value": category_filter
            }
        
        response = self.client.responses.create(
            model="gpt-4o",
            tools=[tool_config],
            input=query
        )
        
        latency_ms = (time.time() - start) * 1000
        
        # Parse results
        results = []
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'file_search_result':
                results.append({
                    "file": getattr(item, 'file_name', ''),
                    "score": getattr(item, 'score', 0),
                    "content": getattr(item, 'content', '')[:500]
                })
        
        # Record metrics
        self.metrics.append({
            "query": query[:100],
            "latency_ms": latency_ms,
            "result_count": len(results),
            "avg_score": sum(r['score'] for r in results) / len(results) if results else 0,
            "purpose": self.purpose.value
        })
        
        return {
            "answer": response.output_text,
            "results": results,
            "latency_ms": latency_ms,
            "config_used": config
        }
    
    def get_performance_report(self) -> dict:
        """Get performance report."""
        
        if not self.metrics:
            return {"message": "No searches performed"}
        
        latencies = [m['latency_ms'] for m in self.metrics]
        scores = [m['avg_score'] for m in self.metrics if m['avg_score'] > 0]
        result_counts = [m['result_count'] for m in self.metrics]
        
        return {
            "total_searches": len(self.metrics),
            "purpose": self.purpose.value,
            "performance": {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "max_latency_ms": max(latencies),
                "min_latency_ms": min(latencies)
            },
            "quality": {
                "avg_result_count": sum(result_counts) / len(result_counts),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "zero_result_rate": sum(1 for r in result_counts if r == 0) / len(result_counts)
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        
        recommendations = []
        
        if not self.metrics:
            return ["Perform more searches to get recommendations"]
        
        avg_latency = sum(m['latency_ms'] for m in self.metrics) / len(self.metrics)
        avg_results = sum(m['result_count'] for m in self.metrics) / len(self.metrics)
        avg_score = sum(m['avg_score'] for m in self.metrics if m['avg_score'] > 0) / len([m for m in self.metrics if m['avg_score'] > 0]) if any(m['avg_score'] > 0 for m in self.metrics) else 0
        
        if avg_latency > 3000:
            recommendations.append("High latency - consider reducing max_results")
        
        if avg_results < 2:
            recommendations.append("Low result count - lower score_threshold")
        elif avg_results > 15:
            recommendations.append("Too many results - raise score_threshold")
        
        if avg_score < 0.4:
            recommendations.append("Low relevance - improve document chunking")
        
        if not recommendations:
            recommendations.append("Configuration looks optimal")
        
        return recommendations


# Usage
system = OptimizedFileSearchSystem()

# Initialize for documentation
store_id = system.initialize(
    name="product-docs",
    purpose=SearchPurpose.DOCUMENTATION
)

print(f"Created store: {store_id}")

# Add files
# system.add_files(
#     ["docs/guide.md", "docs/api.md"],
#     metadata={"category": "documentation", "version": "2.0"}
# )

# Search
# result = system.search("How to authenticate?")
# print(f"Answer: {result['answer']}")
# print(f"Latency: {result['latency_ms']:.0f}ms")

# Performance report
# report = system.get_performance_report()
# print(f"\nPerformance Report:")
# print(f"Average latency: {report['performance']['avg_latency_ms']:.0f}ms")
# print(f"Recommendations: {report['recommendations']}")
```

</details>

---

## Summary

✅ Vector stores organize documents for semantic search  
✅ Ranking options control result relevance  
✅ Score thresholds filter low-confidence matches  
✅ Chunking strategies affect retrieval precision  
✅ Metadata enables filtered searches  
✅ Performance monitoring guides optimization

**Next:** [Computer Use Tool](./03-computer-use.md)

---

## Further Reading

- [Vector Stores Guide](https://platform.openai.com/docs/assistants/tools/file-search) — Official documentation
- [Chunking Strategies](https://platform.openai.com/docs/assistants/tools/file-search/chunking) — Chunking options
- [File Search API Reference](https://platform.openai.com/docs/api-reference/vector-stores) — API details
