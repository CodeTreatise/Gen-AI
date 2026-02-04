---
title: "Migration Strategies"
---

# Migration Strategies

## Introduction

Adopting Matryoshka embeddings in an existing system requires careful planning. You can't simply swap out your embeddings overnight—you need strategies for backward compatibility, gradual rollout, and validation.

This lesson covers practical migration approaches, from low-risk "store full, query reduced" patterns to complete system transitions.

### What We'll Cover

- Store full, query reduced strategy
- A/B testing approaches
- Progressive rollout patterns
- Database migration techniques
- Rollback planning

### Prerequisites

- Understanding of [cost-benefit analysis](./07-cost-benefit-analysis.md)
- Experience with vector database operations
- Familiarity with deployment patterns

---

## Migration Approach Overview

### The Three Migration Paths

| Approach | Risk Level | Effort | Best For |
|----------|------------|--------|----------|
| **Store Full, Query Reduced** | 🟢 Low | Low | Testing hypothesis, gradual adoption |
| **A/B Testing** | 🟡 Medium | Medium | Validating quality impact |
| **Full Migration** | 🔴 Higher | High | Complete system transition |

---

## Strategy 1: Store Full, Query Reduced

The safest approach: **store full-dimension embeddings, truncate only at query time**.

### Why This Works

1. **Zero data migration**—your existing embeddings stay the same
2. **Instant rollback**—just stop truncating
3. **Flexible testing**—try different dimensions without re-embedding
4. **Future-proof**—can always access full precision

### Implementation

```python
import numpy as np
from typing import Optional

class FlexibleEmbeddingStore:
    """
    Store full embeddings, query at any dimension.
    """
    
    def __init__(self, full_dimensions: int = 3072):
        self.full_dimensions = full_dimensions
        self.embeddings = {}  # id -> full embedding
        self.query_dimensions = full_dimensions  # Current query dimension
    
    def store(self, doc_id: str, embedding: np.ndarray):
        """Store full-dimension embedding."""
        assert embedding.shape[0] == self.full_dimensions
        # Ensure normalized
        self.embeddings[doc_id] = embedding / np.linalg.norm(embedding)
    
    def set_query_dimensions(self, dims: int):
        """Change query dimension on the fly."""
        assert dims <= self.full_dimensions
        self.query_dimensions = dims
        print(f"Query dimension set to {dims}")
    
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search using current query dimension."""
        # Truncate query
        query_trunc = query[:self.query_dimensions]
        query_trunc = query_trunc / np.linalg.norm(query_trunc)
        
        scores = []
        for doc_id, full_emb in self.embeddings.items():
            # Truncate stored embedding
            doc_trunc = full_emb[:self.query_dimensions]
            doc_trunc = doc_trunc / np.linalg.norm(doc_trunc)
            
            similarity = np.dot(query_trunc, doc_trunc)
            scores.append((doc_id, similarity))
        
        # Return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# Usage example
store = FlexibleEmbeddingStore(full_dimensions=3072)

# Store at full dimension
for doc_id, text in documents.items():
    embedding = model.encode(text)  # Full 3072 dims
    store.store(doc_id, embedding)

# Query at reduced dimension
store.set_query_dimensions(768)
results = store.search(query_embedding, k=10)

# Easy rollback if needed
store.set_query_dimensions(3072)
results_full = store.search(query_embedding, k=10)
```

### Trade-offs

| Pros | Cons |
|------|------|
| ✅ Zero re-embedding cost | ❌ No storage savings |
| ✅ Instant rollback | ❌ Truncation at query time |
| ✅ Test any dimension | ❌ Not compatible with dimension-specific indexes |

> **Note:** This approach works best when storage isn't your bottleneck and you're primarily testing latency/quality tradeoffs.

---

## Strategy 2: A/B Testing

Run parallel systems to validate quality before committing.

### Architecture

```
                        ┌─────────────────────┐
                        │   Load Balancer     │
                        │   (50/50 split)     │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼─────┐                 ┌─────▼─────┐
              │  System A  │                │  System B  │
              │ (3072 dims)│                │ (768 dims) │
              └─────┬─────┘                 └─────┬─────┘
                    │                             │
              ┌─────▼─────┐                 ┌─────▼─────┐
              │ Metrics A │                 │ Metrics B │
              └───────────┘                 └───────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                            ┌──────▼──────┐
                            │  Analysis   │
                            │  Dashboard  │
                            └─────────────┘
```

### Implementation

```python
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    variant: str
    results: list[str]
    latency_ms: float
    timestamp: datetime

class ABTestManager:
    """Manage A/B testing for embedding dimensions."""
    
    def __init__(
        self,
        variant_a_dims: int = 3072,
        variant_b_dims: int = 768,
        traffic_split: float = 0.5  # 50% to each
    ):
        self.variant_a_dims = variant_a_dims
        self.variant_b_dims = variant_b_dims
        self.traffic_split = traffic_split
        self.results_log = []
    
    def assign_variant(self, user_id: str) -> str:
        """
        Deterministically assign user to variant.
        Same user always gets same variant for consistency.
        """
        # Hash user_id for consistent assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (hash_val % 100) / 100 < self.traffic_split:
            return "A"
        return "B"
    
    def get_dimensions(self, variant: str) -> int:
        """Get embedding dimensions for variant."""
        return self.variant_a_dims if variant == "A" else self.variant_b_dims
    
    def log_result(
        self,
        user_id: str,
        variant: str,
        results: list[str],
        latency_ms: float,
        user_clicked: list[str] = None
    ):
        """Log search result for analysis."""
        self.results_log.append({
            "user_id": user_id,
            "variant": variant,
            "dimensions": self.get_dimensions(variant),
            "result_ids": results,
            "latency_ms": latency_ms,
            "clicked": user_clicked or [],
            "timestamp": datetime.now()
        })
    
    def analyze_results(self) -> dict:
        """Analyze A/B test results."""
        variant_a_logs = [r for r in self.results_log if r["variant"] == "A"]
        variant_b_logs = [r for r in self.results_log if r["variant"] == "B"]
        
        def avg_latency(logs):
            return sum(r["latency_ms"] for r in logs) / len(logs) if logs else 0
        
        def click_through_rate(logs):
            if not logs:
                return 0
            clicks = sum(1 for r in logs if r["clicked"])
            return clicks / len(logs)
        
        return {
            "variant_a": {
                "dimensions": self.variant_a_dims,
                "queries": len(variant_a_logs),
                "avg_latency_ms": round(avg_latency(variant_a_logs), 2),
                "ctr": round(click_through_rate(variant_a_logs), 4)
            },
            "variant_b": {
                "dimensions": self.variant_b_dims,
                "queries": len(variant_b_logs),
                "avg_latency_ms": round(avg_latency(variant_b_logs), 2),
                "ctr": round(click_through_rate(variant_b_logs), 4)
            }
        }

# Usage
ab_test = ABTestManager(variant_a_dims=3072, variant_b_dims=768)

# In your search endpoint
def search(user_id: str, query: str):
    variant = ab_test.assign_variant(user_id)
    dims = ab_test.get_dimensions(variant)
    
    start = time.perf_counter()
    results = search_with_dimensions(query, dims)
    latency = (time.perf_counter() - start) * 1000
    
    ab_test.log_result(user_id, variant, results, latency)
    return results
```

### Key Metrics to Track

| Metric | What It Tells You | Action Threshold |
|--------|-------------------|------------------|
| **Click-through rate** | Result relevance | <5% degradation acceptable |
| **Search latency** | Speed improvement | Should be 2-4x better for B |
| **Conversion rate** | Business impact | <1% degradation acceptable |
| **Error rate** | System stability | Any increase is a problem |

### Statistical Significance

```python
from scipy import stats

def is_significant(variant_a_ctr: float, variant_b_ctr: float, 
                   n_a: int, n_b: int, alpha: float = 0.05) -> bool:
    """
    Test if CTR difference is statistically significant.
    Uses a two-proportion z-test.
    """
    # Pooled proportion
    p_pool = (variant_a_ctr * n_a + variant_b_ctr * n_b) / (n_a + n_b)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    
    # Z-score
    z = (variant_a_ctr - variant_b_ctr) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value < alpha, p_value

# Example: After collecting data
result = is_significant(
    variant_a_ctr=0.12,  # 12% CTR for full dims
    variant_b_ctr=0.115, # 11.5% CTR for reduced dims
    n_a=10000,
    n_b=10000
)
print(f"Significant: {result[0]}, p-value: {result[1]:.4f}")
```

---

## Strategy 3: Progressive Rollout

Gradually transition traffic to reduced dimensions.

### Rollout Schedule

```python
from datetime import datetime, timedelta

class ProgressiveRollout:
    """Manage gradual migration to reduced dimensions."""
    
    def __init__(self):
        self.rollout_schedule = [
            {"day": 0, "reduced_pct": 5, "checkpoint": "Initial canary"},
            {"day": 3, "reduced_pct": 10, "checkpoint": "Validate metrics"},
            {"day": 7, "reduced_pct": 25, "checkpoint": "Scale up"},
            {"day": 14, "reduced_pct": 50, "checkpoint": "Half traffic"},
            {"day": 21, "reduced_pct": 75, "checkpoint": "Majority traffic"},
            {"day": 28, "reduced_pct": 90, "checkpoint": "Final validation"},
            {"day": 35, "reduced_pct": 100, "checkpoint": "Complete migration"},
        ]
        self.start_date = None
        self.current_percentage = 0
        self.is_paused = False
    
    def start(self):
        """Begin rollout."""
        self.start_date = datetime.now()
        self.current_percentage = self.rollout_schedule[0]["reduced_pct"]
        print(f"Rollout started at {self.current_percentage}%")
    
    def check_and_advance(self) -> str:
        """Check if we should advance to next stage."""
        if self.is_paused:
            return "Rollout is paused"
        
        days_elapsed = (datetime.now() - self.start_date).days
        
        for stage in reversed(self.rollout_schedule):
            if days_elapsed >= stage["day"]:
                if self.current_percentage < stage["reduced_pct"]:
                    self.current_percentage = stage["reduced_pct"]
                    return f"Advanced to {stage['reduced_pct']}%: {stage['checkpoint']}"
                break
        
        return f"Current: {self.current_percentage}%"
    
    def pause(self, reason: str):
        """Pause rollout if issues detected."""
        self.is_paused = True
        print(f"⚠️ Rollout PAUSED: {reason}")
    
    def rollback(self):
        """Emergency rollback to full dimensions."""
        self.current_percentage = 0
        self.is_paused = True
        print("🔴 ROLLBACK: All traffic on full dimensions")
    
    def should_use_reduced(self, user_id: str) -> bool:
        """Determine if this request uses reduced dimensions."""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (hash_val % 100) < self.current_percentage

# Usage
rollout = ProgressiveRollout()
rollout.start()

# In your application
def get_dimensions_for_user(user_id: str) -> int:
    if rollout.should_use_reduced(user_id):
        return 768  # Reduced
    return 3072  # Full
```

### Monitoring During Rollout

```python
class RolloutMonitor:
    """Monitor key metrics during rollout."""
    
    def __init__(self, rollout: ProgressiveRollout):
        self.rollout = rollout
        self.baseline_metrics = None
        self.alert_thresholds = {
            "latency_increase_pct": 50,  # Alert if latency increases 50%
            "error_rate_increase_pct": 10,
            "ctr_decrease_pct": 5,
        }
    
    def set_baseline(self, latency_p99: float, error_rate: float, ctr: float):
        """Set baseline metrics before rollout."""
        self.baseline_metrics = {
            "latency_p99": latency_p99,
            "error_rate": error_rate,
            "ctr": ctr
        }
    
    def check_metrics(self, current_latency: float, current_error_rate: float, 
                      current_ctr: float) -> list[str]:
        """Check current metrics against baseline, return alerts."""
        alerts = []
        
        # Latency should DECREASE with reduced dims
        # Alert if it's increasing instead
        latency_change = (current_latency - self.baseline_metrics["latency_p99"]) / self.baseline_metrics["latency_p99"] * 100
        if latency_change > self.alert_thresholds["latency_increase_pct"]:
            alerts.append(f"🔴 Latency increased {latency_change:.1f}%")
        
        # Error rate should stay same or decrease
        error_change = (current_error_rate - self.baseline_metrics["error_rate"]) / max(self.baseline_metrics["error_rate"], 0.001) * 100
        if error_change > self.alert_thresholds["error_rate_increase_pct"]:
            alerts.append(f"🔴 Error rate increased {error_change:.1f}%")
        
        # CTR might decrease slightly, but not too much
        ctr_change = (self.baseline_metrics["ctr"] - current_ctr) / self.baseline_metrics["ctr"] * 100
        if ctr_change > self.alert_thresholds["ctr_decrease_pct"]:
            alerts.append(f"🟡 CTR decreased {ctr_change:.1f}%")
        
        return alerts
    
    def auto_pause_if_needed(self, alerts: list[str]):
        """Automatically pause rollout if critical alerts."""
        critical_alerts = [a for a in alerts if "🔴" in a]
        if critical_alerts:
            self.rollout.pause(f"Critical issues: {critical_alerts}")
```

---

## Strategy 4: Full Database Migration

When you're ready for complete transition.

### Migration Pipeline

```python
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class EmbeddingMigrator:
    """Migrate embeddings to new dimension with proper validation."""
    
    def __init__(
        self,
        source_db,
        target_db,
        target_dims: int = 768,
        batch_size: int = 1000
    ):
        self.source_db = source_db
        self.target_db = target_db
        self.target_dims = target_dims
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def migrate_batch(self, doc_ids: list[str]) -> dict:
        """Migrate a batch of documents."""
        success = 0
        failed = []
        
        for doc_id in doc_ids:
            try:
                # Get full embedding
                full_emb = self.source_db.get_embedding(doc_id)
                
                # Truncate and normalize
                truncated = full_emb[:self.target_dims]
                normalized = truncated / np.linalg.norm(truncated)
                
                # Store in target
                self.target_db.store(doc_id, normalized)
                success += 1
                
            except Exception as e:
                failed.append((doc_id, str(e)))
                self.logger.error(f"Failed to migrate {doc_id}: {e}")
        
        return {"success": success, "failed": failed}
    
    def run_migration(self, validate_sample: float = 0.01):
        """Run full migration with progress and validation."""
        all_ids = self.source_db.list_all_ids()
        total = len(all_ids)
        
        self.logger.info(f"Starting migration of {total} documents")
        
        # Batch IDs
        batches = [all_ids[i:i+self.batch_size] 
                   for i in range(0, total, self.batch_size)]
        
        total_success = 0
        total_failed = []
        
        for batch in tqdm(batches, desc="Migrating"):
            result = self.migrate_batch(batch)
            total_success += result["success"]
            total_failed.extend(result["failed"])
        
        self.logger.info(f"Migration complete: {total_success} success, {len(total_failed)} failed")
        
        # Validate sample
        self.validate_migration(sample_size=int(total * validate_sample))
        
        return {"success": total_success, "failed": total_failed}
    
    def validate_migration(self, sample_size: int = 100):
        """Validate migrated embeddings."""
        sample_ids = random.sample(self.source_db.list_all_ids(), sample_size)
        
        errors = []
        for doc_id in sample_ids:
            # Get both versions
            original = self.source_db.get_embedding(doc_id)
            migrated = self.target_db.get_embedding(doc_id)
            
            # Check truncation is correct
            original_truncated = original[:self.target_dims]
            original_truncated = original_truncated / np.linalg.norm(original_truncated)
            
            diff = np.abs(original_truncated - migrated).max()
            if diff > 1e-6:
                errors.append((doc_id, diff))
        
        if errors:
            self.logger.warning(f"Validation found {len(errors)} mismatches")
        else:
            self.logger.info(f"Validation passed: {sample_size} samples checked")

# Usage
migrator = EmbeddingMigrator(
    source_db=old_vector_db,
    target_db=new_vector_db,
    target_dims=768
)

result = migrator.run_migration()
```

### Parallel Databases During Transition

```python
class DualDatabaseSearch:
    """
    Query both old and new database during transition.
    Allows gradual cutover with validation.
    """
    
    def __init__(self, old_db, new_db, new_db_weight: float = 0.0):
        self.old_db = old_db
        self.new_db = new_db
        self.new_db_weight = new_db_weight  # 0.0 = all old, 1.0 = all new
    
    def search(self, query: np.ndarray, k: int = 10) -> list:
        """Search with configurable weights."""
        if self.new_db_weight == 0.0:
            return self.old_db.search(query, k)
        elif self.new_db_weight == 1.0:
            return self.new_db.search(query[:768], k)  # Truncated query
        else:
            # Merge results from both
            old_results = self.old_db.search(query, k * 2)
            new_results = self.new_db.search(query[:768], k * 2)
            
            # Weighted merge (simple version)
            # In practice, you'd use more sophisticated fusion
            return self._merge_results(old_results, new_results, k)
    
    def _merge_results(self, old: list, new: list, k: int) -> list:
        """Merge results with weighted scoring."""
        combined = {}
        
        for rank, (doc_id, score) in enumerate(old):
            combined[doc_id] = combined.get(doc_id, 0) + (1 - self.new_db_weight) * (1 / (rank + 1))
        
        for rank, (doc_id, score) in enumerate(new):
            combined[doc_id] = combined.get(doc_id, 0) + self.new_db_weight * (1 / (rank + 1))
        
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def set_weight(self, weight: float):
        """Adjust weight during transition."""
        self.new_db_weight = weight
        print(f"New DB weight: {weight:.0%}")
```

---

## Rollback Planning

Always have a rollback plan before starting migration.

### Rollback Checklist

```markdown
## Pre-Migration Rollback Preparation

- [ ] Full backup of current embeddings exists
- [ ] Rollback scripts tested in staging
- [ ] Monitoring alerts configured
- [ ] Runbook documented
- [ ] On-call team briefed

## Rollback Triggers

Automatic rollback if ANY of these occur:
- Error rate > baseline + 10%
- P99 latency > baseline + 100%
- CTR < baseline - 10%

## Rollback Steps

1. Set feature flag to disable new system
2. Redirect all traffic to old database
3. Pause any ongoing migration jobs
4. Notify stakeholders
5. Begin incident review
```

### Feature Flag Implementation

```python
class FeatureFlags:
    """Simple feature flag system for embeddings."""
    
    def __init__(self):
        self.flags = {
            "use_reduced_dims": True,
            "reduced_dim_size": 768,
            "enable_new_index": False,
        }
        self._emergency_rollback = False
    
    def get(self, flag_name: str):
        if self._emergency_rollback:
            # During rollback, disable all new features
            return False if "new" in flag_name or "reduced" in flag_name else self.flags.get(flag_name)
        return self.flags.get(flag_name)
    
    def emergency_rollback(self):
        """Trigger emergency rollback."""
        self._emergency_rollback = True
        logging.critical("🚨 EMERGENCY ROLLBACK ACTIVATED")
    
    def clear_rollback(self):
        """Clear emergency rollback state."""
        self._emergency_rollback = False
        logging.info("Emergency rollback cleared")

# Usage
flags = FeatureFlags()

def get_embedding_dimensions():
    if flags.get("use_reduced_dims"):
        return flags.get("reduced_dim_size")
    return 3072  # Full dimensions

# Emergency rollback
if critical_error_detected:
    flags.emergency_rollback()
```

---

## Hands-on Exercise

### Task: Design a Migration Plan

You're migrating a production search system from 3072 to 768 dimensions.

**Current State:**
- 5 million documents
- 1 million queries/day
- P99 latency: 120ms
- CTR: 12%

**Design a migration plan that includes:**

1. Which strategy would you use first?
2. What metrics would you track?
3. What are your rollback triggers?
4. What's your timeline?

<details>
<summary>💡 Solution Approach</summary>

**Recommended Plan:**

1. **Week 1-2: Store Full, Query Reduced**
   - No data changes
   - Test with 5% traffic shadow testing
   - Validate quality offline

2. **Week 3-4: A/B Test**
   - 50/50 split
   - Track CTR, latency, conversions
   - Require statistical significance (p < 0.05)

3. **Week 5-8: Progressive Rollout**
   - 5% → 10% → 25% → 50% → 75% → 90% → 100%
   - 3-day checkpoint at each stage

4. **Week 9-10: Full Migration**
   - Migrate stored embeddings to 768 dims
   - Deprecate old index
   - Maintain rollback capability for 30 days

**Rollback Triggers:**
- Error rate increase > 5%
- P99 latency increase > 50% (60ms increase)
- CTR decrease > 3% (0.36% absolute)

</details>

---

## Summary

✅ **Start with "store full, query reduced"** for zero-risk testing  
✅ **A/B test** to validate quality impact with statistical significance  
✅ **Progressive rollout** minimizes blast radius of issues  
✅ **Always have a rollback plan** before starting migration  
✅ **Monitor key metrics** throughout the transition  
✅ **Feature flags** enable instant rollback if needed

---

**Previous:** [Cost-Benefit Analysis ←](./07-cost-benefit-analysis.md)

**Back to:** [Matryoshka Embeddings Overview](./00-matryoshka-embeddings.md)

---

<!-- 
Sources Consulted:
- Feature flag best practices (LaunchDarkly, Split.io)
- A/B testing methodology (Google, Microsoft ExP)
- Progressive rollout patterns (SRE literature)
-->
