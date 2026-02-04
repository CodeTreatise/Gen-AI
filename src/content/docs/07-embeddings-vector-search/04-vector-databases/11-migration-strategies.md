---
title: "Migration Strategies"
---

# Migration Strategies

## Introduction

Migrating vector databases is more complex than traditional database migrations. You're moving not just data, but embeddings that may need regeneration, indexes that must be rebuilt, and search behavior that must be validated.

### What We'll Cover

- Migration planning and preparation
- Export and import patterns
- Zero-downtime migration strategies
- Validation and testing
- Rollback procedures

### Prerequisites

- Understanding of your source and target databases
- Access to both environments
- Embedding model consistency plan

---

## Migration Planning

### Pre-Migration Checklist

- [ ] Inventory current data (vector count, dimensions, metadata schema)
- [ ] Document current query patterns and performance baselines
- [ ] Verify embedding model compatibility
- [ ] Estimate migration time and resource requirements
- [ ] Plan validation criteria
- [ ] Define rollback triggers
- [ ] Schedule maintenance window (if needed)

### Migration Complexity Assessment

| Factor | Low Complexity | High Complexity |
|--------|---------------|-----------------|
| **Vector count** | < 100K | > 10M |
| **Metadata schema** | Simple, flat | Nested, complex types |
| **Embedding model** | Same model | Different model |
| **Downtime tolerance** | Hours | Zero |
| **Query patterns** | Simple KNN | Complex filters, hybrid |

### Embedding Model Considerations

```python
# ⚠️ Critical: Embeddings from different models are NOT compatible

# If source used text-embedding-ada-002 and target needs text-embedding-3-small:
# You MUST re-embed all documents

def check_embedding_compatibility(source_model: str, target_model: str) -> bool:
    """Check if embeddings can be migrated directly"""
    same_model = source_model == target_model
    same_dimensions = get_dimensions(source_model) == get_dimensions(target_model)
    
    if not same_model:
        print("⚠️ Different embedding models - re-embedding required")
        return False
    
    return same_model and same_dimensions
```

---

## Export Strategies

### 1. Direct Database Export

Most vector databases provide native export tools:

```bash
# Qdrant - Snapshot export
curl -X POST "http://localhost:6333/collections/my_collection/snapshots"

# Response includes snapshot name
# Download: GET /collections/my_collection/snapshots/{snapshot_name}
```

```python
# Qdrant Python export
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Create snapshot
snapshot_info = client.create_snapshot(collection_name="my_collection")
print(f"Snapshot: {snapshot_info.name}")

# Download snapshot
client.download_snapshot(
    collection_name="my_collection",
    snapshot_name=snapshot_info.name,
    path="./backup/my_collection.snapshot"
)
```

### 2. Batch Export to Files

For cross-platform migrations:

```python
import json
from tqdm import tqdm

def export_to_jsonl(source_client, collection: str, output_file: str, batch_size: int = 1000):
    """Export vectors to JSONL format"""
    
    offset = None
    total_exported = 0
    
    with open(output_file, 'w') as f:
        while True:
            # Fetch batch
            results = source_client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )
            
            points, next_offset = results
            
            if not points:
                break
            
            # Write each point as JSON line
            for point in points:
                record = {
                    "id": point.id,
                    "vector": point.vector,
                    "metadata": point.payload
                }
                f.write(json.dumps(record) + '\n')
            
            total_exported += len(points)
            offset = next_offset
            
            print(f"Exported {total_exported} vectors...")
    
    print(f"✅ Exported {total_exported} vectors to {output_file}")
    return total_exported
```

### 3. Streaming Export (Memory-Efficient)

For large datasets that don't fit in memory:

```python
import pyarrow as pa
import pyarrow.parquet as pq

def export_to_parquet(source_client, collection: str, output_dir: str, 
                      batch_size: int = 10000):
    """Stream export to Parquet files"""
    
    offset = None
    file_index = 0
    
    while True:
        points, next_offset = source_client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        
        if not points:
            break
        
        # Convert to Arrow table
        ids = [p.id for p in points]
        vectors = [p.vector for p in points]
        metadata = [json.dumps(p.payload) for p in points]
        
        table = pa.table({
            'id': ids,
            'vector': vectors,
            'metadata': metadata
        })
        
        # Write Parquet file
        output_path = f"{output_dir}/part_{file_index:04d}.parquet"
        pq.write_table(table, output_path)
        
        file_index += 1
        offset = next_offset
        print(f"Written {output_path}")
```

---

## Import Strategies

### 1. Direct Import

```python
def import_from_jsonl(target_client, collection: str, input_file: str, 
                      batch_size: int = 100):
    """Import vectors from JSONL format"""
    
    batch = []
    total_imported = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            batch.append(record)
            
            if len(batch) >= batch_size:
                target_client.upsert(
                    collection_name=collection,
                    points=[
                        {
                            "id": r["id"],
                            "vector": r["vector"],
                            "payload": r["metadata"]
                        }
                        for r in batch
                    ]
                )
                total_imported += len(batch)
                batch = []
                print(f"Imported {total_imported} vectors...")
    
    # Import remaining
    if batch:
        target_client.upsert(collection_name=collection, points=batch)
        total_imported += len(batch)
    
    print(f"✅ Imported {total_imported} vectors")
    return total_imported
```

### 2. Parallel Import (Faster)

```python
import concurrent.futures
from typing import Iterator

def chunked(iterable, size: int) -> Iterator:
    """Yield successive chunks from iterable"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def parallel_import(target_client, collection: str, records: list, 
                    batch_size: int = 100, max_workers: int = 4):
    """Import with parallel workers"""
    
    def import_batch(batch):
        target_client.upsert(collection_name=collection, points=batch)
        return len(batch)
    
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(import_batch, batch) 
            for batch in chunked(records, batch_size)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            total += future.result()
            print(f"Imported {total} vectors...")
    
    return total
```

---

## Zero-Downtime Migration

### Strategy 1: Dual-Write

Write to both old and new databases during migration:

```python
class DualWriteVectorStore:
    def __init__(self, primary, secondary):
        self.primary = primary    # Current production
        self.secondary = secondary  # New database
        self.dual_write_enabled = True
        
    def upsert(self, vectors):
        # Always write to primary
        self.primary.upsert(vectors)
        
        # Write to secondary during migration
        if self.dual_write_enabled:
            try:
                self.secondary.upsert(vectors)
            except Exception as e:
                # Log but don't fail - secondary is not critical yet
                print(f"Secondary write failed: {e}")
    
    def search(self, query_vector, top_k: int):
        # Read from primary only (until cutover)
        return self.primary.search(query_vector, top_k)
    
    def cutover(self):
        """Switch to new database"""
        self.primary, self.secondary = self.secondary, self.primary
        self.dual_write_enabled = False
```

### Strategy 2: Shadow Traffic

Run queries against both databases to validate:

```python
import asyncio
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    query_id: str
    primary_results: list
    secondary_results: list
    recall_match: float
    latency_diff_ms: float

class ShadowTrafficValidator:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self.results = []
        
    async def validate_query(self, query_id: str, query_vector: list, top_k: int):
        """Run query on both and compare"""
        
        # Query primary
        start = time.time()
        primary_results = self.primary.search(query_vector, top_k)
        primary_latency = (time.time() - start) * 1000
        
        # Query secondary
        start = time.time()
        secondary_results = self.secondary.search(query_vector, top_k)
        secondary_latency = (time.time() - start) * 1000
        
        # Calculate recall overlap
        primary_ids = set(r.id for r in primary_results)
        secondary_ids = set(r.id for r in secondary_results)
        overlap = len(primary_ids & secondary_ids)
        recall = overlap / len(primary_ids) if primary_ids else 1.0
        
        result = ComparisonResult(
            query_id=query_id,
            primary_results=primary_results,
            secondary_results=secondary_results,
            recall_match=recall,
            latency_diff_ms=secondary_latency - primary_latency
        )
        
        self.results.append(result)
        return result
    
    def generate_report(self):
        """Summary of shadow traffic validation"""
        recalls = [r.recall_match for r in self.results]
        latencies = [r.latency_diff_ms for r in self.results]
        
        return {
            "total_queries": len(self.results),
            "avg_recall": sum(recalls) / len(recalls),
            "min_recall": min(recalls),
            "avg_latency_diff_ms": sum(latencies) / len(latencies),
            "queries_below_95_recall": sum(1 for r in recalls if r < 0.95)
        }
```

### Strategy 3: Blue-Green with Feature Flags

```python
import os

class FeatureFlaggedVectorStore:
    def __init__(self, blue_store, green_store):
        self.blue = blue_store   # Current production
        self.green = green_store  # New system
        
    @property
    def active_store(self):
        # Feature flag determines which store to use
        use_green = os.getenv("USE_GREEN_VECTOR_STORE", "false").lower() == "true"
        return self.green if use_green else self.blue
    
    def search(self, query_vector, top_k: int):
        return self.active_store.search(query_vector, top_k)
    
    def upsert(self, vectors):
        # Write to both during migration
        self.blue.upsert(vectors)
        self.green.upsert(vectors)

# Cutover process:
# 1. Set USE_GREEN_VECTOR_STORE=true for canary users (1%)
# 2. Monitor metrics, validate results
# 3. Gradually increase to 10%, 50%, 100%
# 4. Disable dual-write once stable
```

---

## Validation Procedures

### Data Integrity Checks

```python
def validate_migration(source, target, sample_size: int = 1000):
    """Validate migration completeness and accuracy"""
    
    results = {
        "count_match": False,
        "sample_vectors_match": 0,
        "sample_metadata_match": 0,
        "issues": []
    }
    
    # Check counts
    source_count = source.count()
    target_count = target.count()
    results["count_match"] = source_count == target_count
    
    if not results["count_match"]:
        results["issues"].append(
            f"Count mismatch: source={source_count}, target={target_count}"
        )
    
    # Sample validation
    sample_ids = source.sample_ids(sample_size)
    
    for id in sample_ids:
        source_point = source.get(id)
        target_point = target.get(id)
        
        if target_point is None:
            results["issues"].append(f"Missing in target: {id}")
            continue
        
        # Check vector equality (with tolerance)
        if vectors_equal(source_point.vector, target_point.vector, tolerance=1e-6):
            results["sample_vectors_match"] += 1
        else:
            results["issues"].append(f"Vector mismatch: {id}")
        
        # Check metadata
        if source_point.metadata == target_point.metadata:
            results["sample_metadata_match"] += 1
        else:
            results["issues"].append(f"Metadata mismatch: {id}")
    
    return results

def vectors_equal(v1: list, v2: list, tolerance: float = 1e-6) -> bool:
    """Compare vectors with floating point tolerance"""
    if len(v1) != len(v2):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(v1, v2))
```

### Search Quality Validation

```python
def validate_search_quality(source, target, test_queries: list, 
                           min_recall: float = 0.95):
    """Validate that search results are consistent"""
    
    results = []
    
    for query in test_queries:
        source_results = source.search(query["vector"], top_k=10)
        target_results = target.search(query["vector"], top_k=10)
        
        source_ids = [r.id for r in source_results]
        target_ids = [r.id for r in target_results]
        
        # Calculate recall
        overlap = len(set(source_ids) & set(target_ids))
        recall = overlap / len(source_ids)
        
        results.append({
            "query_id": query["id"],
            "recall": recall,
            "passed": recall >= min_recall
        })
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"Search validation: {passed}/{total} queries passed (>={min_recall} recall)")
    
    failed_queries = [r for r in results if not r["passed"]]
    if failed_queries:
        print("Failed queries:")
        for q in failed_queries[:10]:
            print(f"  {q['query_id']}: recall={q['recall']:.2f}")
    
    return results
```

---

## Rollback Procedures

### Automatic Rollback Triggers

```python
class MigrationMonitor:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        self.metrics = []
        
    def record_metric(self, name: str, value: float):
        self.metrics.append({"name": name, "value": value, "time": time.time()})
        self.check_thresholds()
        
    def check_thresholds(self):
        # Check error rate
        recent_errors = [m for m in self.metrics[-100:] if m["name"] == "error"]
        error_rate = len(recent_errors) / 100
        
        if error_rate > self.thresholds.get("error_rate", 0.05):
            self.trigger_rollback("Error rate exceeded threshold")
        
        # Check latency
        recent_latencies = [m["value"] for m in self.metrics[-100:] 
                          if m["name"] == "latency"]
        if recent_latencies:
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            if avg_latency > self.thresholds.get("latency_ms", 100):
                self.trigger_rollback("Latency exceeded threshold")
    
    def trigger_rollback(self, reason: str):
        print(f"🚨 ROLLBACK TRIGGERED: {reason}")
        # Switch feature flag
        os.environ["USE_GREEN_VECTOR_STORE"] = "false"
        # Alert team
        send_alert(f"Vector DB migration rolled back: {reason}")
```

### Manual Rollback Checklist

1. **Immediate Actions**
   - [ ] Switch feature flag to old database
   - [ ] Verify old database is receiving traffic
   - [ ] Check old database health metrics

2. **Investigation**
   - [ ] Capture failed queries/errors for analysis
   - [ ] Compare performance metrics before/after
   - [ ] Identify root cause

3. **Recovery**
   - [ ] Fix identified issues
   - [ ] Re-run validation suite
   - [ ] Schedule new migration window

---

## Hands-on Exercise

### Your Task

Plan and execute a migration from Chroma to Qdrant:

### Requirements

1. Export 10,000 sample vectors from Chroma
2. Import to a local Qdrant instance
3. Validate data integrity (count, sample checks)
4. Validate search quality with 10 test queries
5. Document the migration runbook

<details>
<summary>✅ Solution</summary>

```python
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json

# Setup
chroma = chromadb.PersistentClient("./chroma_data")
qdrant = QdrantClient("localhost", port=6333)

source_collection = chroma.get_collection("documents")

# 1. Export from Chroma
def export_chroma():
    results = source_collection.get(
        include=["embeddings", "metadatas", "documents"]
    )
    
    exported = []
    for i in range(len(results["ids"])):
        exported.append({
            "id": results["ids"][i],
            "vector": results["embeddings"][i],
            "metadata": results["metadatas"][i] or {},
            "document": results["documents"][i] if results["documents"] else None
        })
    
    with open("chroma_export.jsonl", "w") as f:
        for record in exported:
            f.write(json.dumps(record) + "\n")
    
    return len(exported)

print(f"Exported {export_chroma()} vectors")

# 2. Create Qdrant collection and import
def import_to_qdrant():
    # Get dimensions from first vector
    with open("chroma_export.jsonl", "r") as f:
        first = json.loads(f.readline())
        dims = len(first["vector"])
    
    # Create collection
    qdrant.recreate_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE)
    )
    
    # Import
    points = []
    with open("chroma_export.jsonl", "r") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            points.append(PointStruct(
                id=i,  # Qdrant needs numeric IDs, store original in payload
                vector=record["vector"],
                payload={
                    "original_id": record["id"],
                    **record["metadata"],
                    "document": record["document"]
                }
            ))
            
            if len(points) >= 100:
                qdrant.upsert(collection_name="documents", points=points)
                points = []
    
    if points:
        qdrant.upsert(collection_name="documents", points=points)
    
    return qdrant.count(collection_name="documents").count

print(f"Imported {import_to_qdrant()} vectors to Qdrant")

# 3. Validate integrity
def validate_integrity():
    source_count = source_collection.count()
    target_count = qdrant.count(collection_name="documents").count
    
    print(f"Source count: {source_count}")
    print(f"Target count: {target_count}")
    print(f"Count match: {source_count == target_count}")
    
    # Sample check
    sample = source_collection.get(limit=10, include=["embeddings"])
    for i, original_id in enumerate(sample["ids"]):
        # Find in Qdrant by original_id
        results = qdrant.scroll(
            collection_name="documents",
            scroll_filter={"must": [{"key": "original_id", "match": {"value": original_id}}]},
            limit=1,
            with_vectors=True
        )
        if results[0]:
            target_vector = results[0][0].vector
            source_vector = sample["embeddings"][i]
            match = all(abs(a-b) < 1e-6 for a, b in zip(source_vector, target_vector))
            print(f"  {original_id}: vector match = {match}")

validate_integrity()

# 4. Validate search quality
def validate_search(num_queries=10):
    # Get random queries from source
    sample = source_collection.get(limit=num_queries, include=["embeddings"])
    
    recalls = []
    for i, query_vector in enumerate(sample["embeddings"]):
        # Search source
        source_results = source_collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        source_ids = set(source_results["ids"][0])
        
        # Search target
        target_results = qdrant.query_points(
            collection_name="documents",
            query=query_vector,
            limit=10
        )
        target_ids = set(p.payload["original_id"] for p in target_results.points)
        
        recall = len(source_ids & target_ids) / len(source_ids)
        recalls.append(recall)
        print(f"Query {i+1}: recall = {recall:.2f}")
    
    avg_recall = sum(recalls) / len(recalls)
    print(f"\nAverage recall: {avg_recall:.2f}")
    return avg_recall >= 0.95

print(f"Search validation passed: {validate_search()}")
```

**Migration Runbook:**
1. Pre-migration: Verify Qdrant is running, disk space available
2. Export: Run export_chroma() - expect ~5 minutes for 10K vectors
3. Import: Run import_to_qdrant() - expect ~2 minutes
4. Validate: Run both validation functions
5. Cutover: Update application config to use Qdrant
6. Monitor: Watch error rates and latency for 24 hours
7. Rollback: If issues, revert config to Chroma

</details>

---

## Summary

✅ Migration planning is critical—assess complexity before starting

✅ Export to portable formats (JSONL, Parquet) for cross-platform migration

✅ Zero-downtime requires dual-write or blue-green deployment

✅ Validate both data integrity AND search quality before cutover

✅ Always have a tested rollback procedure

**Next:** [Back to Vector Databases Overview](./00-vector-databases.md)

---

## Further Reading

- [Qdrant Snapshots Guide](https://qdrant.tech/documentation/concepts/snapshots/)
- [Pinecone Migration Guide](https://docs.pinecone.io/guides/operations/migrate-to-pinecone)
- [Database Migration Best Practices](https://aws.amazon.com/blogs/database/database-migration-what-do-you-need-to-know-before-you-start/)

---

<!-- 
Sources Consulted:
- Qdrant snapshots: https://qdrant.tech/documentation/concepts/snapshots/
- Pinecone migration: https://docs.pinecone.io/
- Database migration patterns: https://martinfowler.com/articles/patterns-of-distributed-systems/
-->
