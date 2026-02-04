---
title: "Embedding Versioning"
---

# Embedding Versioning

## Introduction

Embedding models evolve—new versions improve quality, dimensions change, and fine-tuned models better fit specific domains. Managing these transitions without downtime or data loss requires a versioning strategy.

This lesson covers model version tracking, when and how to re-embed, migration strategies, and A/B testing embedding models in production.

### What We'll Cover

- Version tracking approaches
- When to re-embed your data
- Migration strategies
- A/B testing embedding models
- Rollback procedures

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Familiarity with vector database operations
- Basic knowledge of deployment strategies

---

## Why Version Embeddings?

| Scenario | Impact Without Versioning |
|----------|---------------------------|
| Model upgrade (e.g., `text-embedding-ada-002` → `text-embedding-3-small`) | Search quality degrades—old and new embeddings incompatible |
| Dimension change (e.g., 1536 → 3072) | Database errors or silent failures |
| Fine-tuned model deployment | Can't compare against baseline |
| Bug in embedding pipeline | No way to identify affected data |
| Compliance audit | Can't prove which model generated embeddings |

---

## Version Tracking Strategies

### Strategy 1: Metadata-Based Tracking

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class EmbeddingVersion:
    model_name: str           # e.g., "text-embedding-3-small"
    model_version: str        # e.g., "2024-01"
    dimensions: int           # e.g., 1536
    created_at: datetime
    pipeline_version: str     # Your code version
    config_hash: str          # Hash of chunking/processing config

class VersionedEmbedding:
    """
    Store version metadata with each embedding.
    """
    def __init__(self, embedding_service, version_config: EmbeddingVersion):
        self.embedder = embedding_service
        self.version = version_config
    
    def embed_with_metadata(self, text: str, doc_id: str) -> dict:
        """Generate embedding with full version tracking."""
        embedding = self.embedder.embed(text)
        
        return {
            "id": f"{doc_id}_{self.version.model_version}",
            "values": embedding,
            "metadata": {
                "text": text[:1000],
                "doc_id": doc_id,
                # Version tracking
                "model_name": self.version.model_name,
                "model_version": self.version.model_version,
                "dimensions": self.version.dimensions,
                "embedded_at": datetime.utcnow().isoformat(),
                "pipeline_version": self.version.pipeline_version,
                "config_hash": self.version.config_hash
            }
        }
```

### Strategy 2: Namespace/Collection Per Version

```python
class NamespaceVersioning:
    """
    Use separate namespaces for each embedding version.
    
    Benefits:
    - Clean separation
    - Easy rollback (just switch namespace)
    - Simple comparison for A/B testing
    """
    def __init__(self, vector_db, base_namespace: str = "embeddings"):
        self.db = vector_db
        self.base_namespace = base_namespace
    
    def get_namespace(self, version: str) -> str:
        """Generate versioned namespace name."""
        return f"{self.base_namespace}_v{version}"
    
    def upsert(self, vectors: list, version: str):
        """Insert into version-specific namespace."""
        namespace = self.get_namespace(version)
        self.db.upsert(vectors, namespace=namespace)
    
    def search(self, query_embedding: list, version: str, top_k: int = 10):
        """Search in specific version namespace."""
        namespace = self.get_namespace(version)
        return self.db.query(query_embedding, top_k=top_k, namespace=namespace)
    
    def list_versions(self) -> list:
        """List all embedding versions."""
        stats = self.db.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        
        versions = []
        for ns in namespaces:
            if ns.startswith(self.base_namespace):
                version = ns.replace(f"{self.base_namespace}_v", "")
                versions.append({
                    "version": version,
                    "vector_count": namespaces[ns]["vector_count"]
                })
        
        return sorted(versions, key=lambda x: x["version"], reverse=True)
```

### Strategy 3: Separate Collections (Qdrant)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class CollectionVersioning:
    """
    Use separate collections for each embedding version.
    Preferred for Qdrant deployments.
    """
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def create_versioned_collection(
        self,
        base_name: str,
        version: str,
        dimensions: int
    ) -> str:
        """Create collection for new embedding version."""
        collection_name = f"{base_name}_v{version}"
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=Distance.COSINE
            )
        )
        
        # Store version metadata
        self.client.set_payload(
            collection_name=collection_name,
            payload={
                "version": version,
                "dimensions": dimensions,
                "created_at": datetime.utcnow().isoformat()
            },
            points=["_metadata"]  # Special metadata point
        )
        
        return collection_name
    
    def get_active_collection(self, base_name: str) -> str:
        """Get currently active collection version."""
        # Read from config or alias
        alias = f"{base_name}_active"
        # In production, use collection aliases
        return self.client.get_collection_aliases().get(alias)
```

---

## When to Re-Embed

### Decision Matrix

| Trigger | Action Required | Priority |
|---------|-----------------|----------|
| New embedding model version | Re-embed all | High |
| Dimension change | Must re-embed (incompatible) | Critical |
| Fine-tuned new model | Re-embed for comparison | Medium |
| Bug fix in chunking | Re-embed affected docs | High |
| Improved preprocessing | Optional re-embed | Low |
| Model deprecation notice | Plan migration | Medium |

### Change Detection System

```python
from dataclasses import dataclass
from enum import Enum
import hashlib

class ChangeType(Enum):
    NONE = "none"
    MODEL_UPDATE = "model_update"
    DIMENSION_CHANGE = "dimension_change"
    CONFIG_CHANGE = "config_change"
    CONTENT_CHANGE = "content_change"

@dataclass
class EmbeddingConfig:
    model_name: str
    model_version: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    preprocessing_steps: list
    
    def config_hash(self) -> str:
        """Hash of all configuration affecting embeddings."""
        config_str = f"{self.model_name}:{self.model_version}:{self.dimensions}"
        config_str += f":{self.chunk_size}:{self.chunk_overlap}"
        config_str += f":{sorted(self.preprocessing_steps)}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class ChangeDetector:
    """
    Detect when re-embedding is needed.
    """
    def __init__(self, metadata_store):
        self.store = metadata_store
    
    def detect_changes(
        self,
        current_config: EmbeddingConfig,
        stored_config_hash: str
    ) -> ChangeType:
        """Determine type of change requiring re-embedding."""
        stored = self.store.get_config(stored_config_hash)
        
        if stored is None:
            return ChangeType.CONFIG_CHANGE
        
        # Critical: dimension changes are incompatible
        if current_config.dimensions != stored.dimensions:
            return ChangeType.DIMENSION_CHANGE
        
        # Model version change
        if (current_config.model_name != stored.model_name or
            current_config.model_version != stored.model_version):
            return ChangeType.MODEL_UPDATE
        
        # Config change (chunking, preprocessing)
        if current_config.config_hash() != stored.config_hash():
            return ChangeType.CONFIG_CHANGE
        
        return ChangeType.NONE
    
    def get_affected_documents(
        self,
        change_type: ChangeType,
        config_hash: str
    ) -> list:
        """Get documents that need re-embedding."""
        if change_type == ChangeType.NONE:
            return []
        
        # All documents for model/dimension changes
        if change_type in [ChangeType.MODEL_UPDATE, ChangeType.DIMENSION_CHANGE]:
            return self.store.get_all_document_ids()
        
        # Only documents with old config hash
        return self.store.get_documents_by_config(config_hash)
```

---

## Migration Strategies

### Strategy 1: Blue-Green Migration

```
┌─────────────────────────────────────────────────────────────────┐
│              Blue-Green Embedding Migration                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Parallel Write                                        │
│  ┌─────────────┐                                                │
│  │   Query     │──────┐                                         │
│  │   Traffic   │      │                                         │
│  └─────────────┘      ▼                                         │
│                  ┌─────────┐    ┌─────────────────────────┐    │
│                  │  Blue   │────│ v1 (ada-002) - ACTIVE   │    │
│                  │ (Live)  │    │ 100% read traffic       │    │
│                  └─────────┘    └─────────────────────────┘    │
│                       │                                         │
│  ┌─────────────┐      │                                         │
│  │   Write     │──────┼──────▶ Write to BOTH                   │
│  │   Traffic   │      │                                         │
│  └─────────────┘      ▼                                         │
│                  ┌─────────┐    ┌─────────────────────────┐    │
│                  │  Green  │────│ v2 (3-small) - STANDBY  │    │
│                  │(Standby)│    │ Re-embedding in progress │    │
│                  └─────────┘    └─────────────────────────┘    │
│                                                                 │
│  Phase 2: Traffic Switch                                        │
│  After validation: Route 100% read traffic to Green             │
│  Keep Blue for rollback (7 days)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
class BlueGreenMigration:
    """
    Blue-green deployment for embedding migrations.
    """
    def __init__(self, vector_db, embedding_service):
        self.db = vector_db
        self.embedder = embedding_service
        self.active_version = None
        self.standby_version = None
    
    async def start_migration(
        self,
        from_version: str,
        to_version: str,
        new_embedding_config: EmbeddingConfig
    ):
        """
        Start blue-green migration.
        """
        self.active_version = from_version
        self.standby_version = to_version
        
        # Create new namespace/collection
        await self._create_version_namespace(to_version, new_embedding_config)
        
        # Start background re-embedding
        await self._start_reembedding_job(to_version, new_embedding_config)
        
        return {
            "status": "migration_started",
            "active": from_version,
            "standby": to_version
        }
    
    async def upsert(self, documents: list):
        """
        Write to both versions during migration.
        """
        # Write to active (old model)
        active_embeddings = await self._embed_and_upsert(
            documents, 
            self.active_version,
            use_old_model=True
        )
        
        # Write to standby (new model) if migration in progress
        if self.standby_version:
            standby_embeddings = await self._embed_and_upsert(
                documents,
                self.standby_version,
                use_old_model=False
            )
        
        return {"active": len(active_embeddings), "standby": len(standby_embeddings)}
    
    def search(self, query: str, top_k: int = 10):
        """
        Search against active version only.
        """
        return self._search_version(query, self.active_version, top_k)
    
    async def complete_migration(self, to_version: str):
        """
        Switch traffic to new version.
        """
        # Validate new version is ready
        stats = await self._get_version_stats(to_version)
        if stats["status"] != "ready":
            raise MigrationError(f"Version {to_version} not ready")
        
        # Atomic switch
        old_active = self.active_version
        self.active_version = to_version
        self.standby_version = old_active  # Keep for rollback
        
        return {
            "status": "migration_complete",
            "active": self.active_version,
            "rollback_available": old_active
        }
    
    async def rollback(self):
        """
        Rollback to previous version.
        """
        if not self.standby_version:
            raise MigrationError("No rollback version available")
        
        self.active_version, self.standby_version = (
            self.standby_version, 
            self.active_version
        )
        
        return {"status": "rolled_back", "active": self.active_version}
```

### Strategy 2: Gradual Migration with Shadowing

```python
class GradualMigration:
    """
    Migrate gradually while comparing results.
    """
    def __init__(self, vector_db, old_embedder, new_embedder):
        self.db = vector_db
        self.old_embedder = old_embedder
        self.new_embedder = new_embedder
        self.migration_percentage = 0
    
    async def search_with_shadow(
        self,
        query: str,
        top_k: int = 10
    ) -> dict:
        """
        Search both versions, return active results, log comparison.
        """
        # Always search old (active)
        old_embedding = self.old_embedder.embed(query)
        old_results = self.db.search(
            old_embedding, 
            namespace="v1",
            top_k=top_k
        )
        
        # Shadow search new version
        new_embedding = self.new_embedder.embed(query)
        new_results = self.db.search(
            new_embedding,
            namespace="v2", 
            top_k=top_k
        )
        
        # Log comparison for analysis
        comparison = self._compare_results(old_results, new_results)
        await self._log_comparison(query, comparison)
        
        # Return active results
        return {
            "results": old_results,
            "shadow_comparison": comparison
        }
    
    def _compare_results(self, old: list, new: list) -> dict:
        """Compare result sets for quality analysis."""
        old_ids = set(r.id for r in old)
        new_ids = set(r.id for r in new)
        
        return {
            "overlap": len(old_ids & new_ids) / len(old_ids) if old_ids else 0,
            "only_in_old": list(old_ids - new_ids),
            "only_in_new": list(new_ids - old_ids),
            "position_changes": self._calc_position_changes(old, new)
        }
    
    def _calc_position_changes(self, old: list, new: list) -> dict:
        """Calculate how much result positions changed."""
        old_positions = {r.id: i for i, r in enumerate(old)}
        new_positions = {r.id: i for i, r in enumerate(new)}
        
        changes = {}
        for id in old_positions:
            if id in new_positions:
                changes[id] = new_positions[id] - old_positions[id]
        
        return {
            "avg_position_change": sum(abs(c) for c in changes.values()) / len(changes) if changes else 0,
            "details": changes
        }
```

---

## A/B Testing Embedding Models

### Traffic Splitting

```python
import random
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    test_id: str
    control_version: str      # e.g., "ada-002"
    treatment_version: str    # e.g., "3-small"
    traffic_split: float      # 0.0 to 1.0 (treatment percentage)
    
class EmbeddingABTest:
    """
    A/B test different embedding models in production.
    """
    def __init__(self, vector_db, embedders: dict, metrics_client):
        self.db = vector_db
        self.embedders = embedders  # {"ada-002": ..., "3-small": ...}
        self.metrics = metrics_client
        self.active_tests = {}
    
    def create_test(self, config: ABTestConfig):
        """Start a new A/B test."""
        self.active_tests[config.test_id] = config
        
        self.metrics.track_event("ab_test_started", {
            "test_id": config.test_id,
            "control": config.control_version,
            "treatment": config.treatment_version,
            "split": config.traffic_split
        })
    
    def search(
        self,
        query: str,
        test_id: str,
        user_id: str,
        top_k: int = 10
    ) -> dict:
        """
        Search with A/B test assignment.
        """
        config = self.active_tests.get(test_id)
        if not config:
            raise ValueError(f"Unknown test: {test_id}")
        
        # Consistent assignment per user
        assignment = self._get_assignment(user_id, config)
        version = (
            config.treatment_version 
            if assignment == "treatment" 
            else config.control_version
        )
        
        # Execute search
        embedder = self.embedders[version]
        embedding = embedder.embed(query)
        results = self.db.search(
            embedding,
            namespace=f"v_{version}",
            top_k=top_k
        )
        
        # Track for analysis
        self.metrics.track_event("ab_search", {
            "test_id": test_id,
            "user_id": user_id,
            "assignment": assignment,
            "version": version,
            "result_count": len(results)
        })
        
        return {
            "results": results,
            "assignment": assignment,
            "version": version
        }
    
    def _get_assignment(self, user_id: str, config: ABTestConfig) -> str:
        """
        Deterministic assignment based on user_id.
        Ensures same user always gets same variant.
        """
        hash_input = f"{config.test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 1000) / 1000
        
        return "treatment" if normalized < config.traffic_split else "control"
    
    def get_test_results(self, test_id: str) -> dict:
        """
        Get A/B test results for analysis.
        """
        config = self.active_tests.get(test_id)
        
        control_metrics = self.metrics.get_aggregated(
            test_id=test_id,
            assignment="control"
        )
        
        treatment_metrics = self.metrics.get_aggregated(
            test_id=test_id,
            assignment="treatment"
        )
        
        return {
            "test_id": test_id,
            "control": {
                "version": config.control_version,
                **control_metrics
            },
            "treatment": {
                "version": config.treatment_version,
                **treatment_metrics
            },
            "recommendation": self._get_recommendation(
                control_metrics, 
                treatment_metrics
            )
        }
```

### Metrics Collection for A/B Tests

```python
@dataclass
class SearchMetrics:
    query_latency_ms: float
    result_count: int
    user_clicked: bool
    click_position: int = None
    time_to_click_ms: float = None
    
class ABTestMetricsCollector:
    """
    Collect metrics for A/B test analysis.
    """
    def __init__(self, storage):
        self.storage = storage
    
    def record_search(
        self,
        test_id: str,
        assignment: str,
        user_id: str,
        query: str,
        metrics: SearchMetrics
    ):
        """Record search event with metrics."""
        self.storage.insert({
            "test_id": test_id,
            "assignment": assignment,
            "user_id": user_id,
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": metrics.query_latency_ms,
            "result_count": metrics.result_count,
            "clicked": metrics.user_clicked,
            "click_position": metrics.click_position,
            "time_to_click_ms": metrics.time_to_click_ms
        })
    
    def get_summary(self, test_id: str) -> dict:
        """
        Calculate summary statistics for test.
        """
        data = self.storage.query(test_id=test_id)
        
        control = [d for d in data if d["assignment"] == "control"]
        treatment = [d for d in data if d["assignment"] == "treatment"]
        
        return {
            "control": self._calculate_stats(control),
            "treatment": self._calculate_stats(treatment),
            "statistical_significance": self._calc_significance(
                control, treatment
            )
        }
    
    def _calculate_stats(self, events: list) -> dict:
        """Calculate aggregate statistics."""
        if not events:
            return {}
        
        latencies = [e["latency_ms"] for e in events]
        clicks = [e for e in events if e["clicked"]]
        
        return {
            "total_searches": len(events),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "click_through_rate": len(clicks) / len(events),
            "avg_click_position": (
                sum(c["click_position"] for c in clicks) / len(clicks)
                if clicks else None
            )
        }
```

---

## Rollback Procedures

```python
class VersionRollback:
    """
    Safe rollback procedures for embedding versions.
    """
    def __init__(self, vector_db, config_store):
        self.db = vector_db
        self.config = config_store
    
    async def prepare_rollback(self, from_version: str, to_version: str):
        """
        Prepare for rollback before executing.
        """
        # Verify target version exists and is healthy
        target_stats = await self.db.get_namespace_stats(f"v_{to_version}")
        
        if target_stats["vector_count"] == 0:
            raise RollbackError(f"Target version {to_version} is empty")
        
        # Check for any in-flight writes
        pending = await self.config.get_pending_writes(from_version)
        
        return {
            "ready": len(pending) == 0,
            "pending_writes": len(pending),
            "target_vectors": target_stats["vector_count"],
            "recommendation": (
                "Safe to proceed" if len(pending) == 0 
                else f"Wait for {len(pending)} pending writes"
            )
        }
    
    async def execute_rollback(
        self,
        from_version: str,
        to_version: str,
        reason: str
    ):
        """
        Execute version rollback with audit trail.
        """
        # Record rollback event
        await self.config.record_rollback({
            "from_version": from_version,
            "to_version": to_version,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "initiated_by": get_current_user()
        })
        
        # Update active version pointer
        await self.config.set_active_version(to_version)
        
        # Don't delete old version yet - keep for analysis
        await self.config.mark_version_deprecated(
            from_version,
            retain_days=30
        )
        
        return {
            "status": "rollback_complete",
            "active_version": to_version,
            "deprecated_version": from_version
        }
```

---

## Summary

✅ **Track version metadata (model, dimensions, config hash) with every embedding**  
✅ **Use namespaces or collections to isolate versions**  
✅ **Blue-green migrations enable zero-downtime version switches**  
✅ **A/B testing validates new models before full rollout**  
✅ **Always maintain rollback capability with version retention**

---

**Next:** [Monitoring and Observability →](./03-monitoring-observability.md)

---

<!-- 
Sources Consulted:
- Pinecone Namespaces: https://docs.pinecone.io/guides/indexes/use-namespaces
- Qdrant Collections: https://qdrant.tech/documentation/concepts/collections/
- Feature Flags Best Practices: https://martinfowler.com/articles/feature-toggles.html
-->
