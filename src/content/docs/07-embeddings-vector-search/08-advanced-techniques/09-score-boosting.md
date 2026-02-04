---
title: "Score Boosting & Formula Queries"
---

# Score Boosting & Formula Queries

## Introduction

Score boosting modifies retrieval scores based on document metadata—recency, popularity, category relevance, or geographic distance. Qdrant's formula queries enable complex scoring formulas that combine vector similarity with payload values.

> **🤖 AI Context:** Score boosting is essential for production RAG. A 2-year-old document might be semantically perfect but factually outdated. Boosting lets you balance relevance with freshness.

---

## Qdrant Score Modifiers

Qdrant applies score modifiers to adjust the base similarity score:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

# Search with score boost from payload
results = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, 0.3, ...],  # Query vector
    query_filter=None,
    score_threshold=0.5,
    limit=10,
    # Score modifiers applied after vector search
)
```

---

## Time Decay Functions

Boost recent documents using decay functions:

```python
from datetime import datetime, timedelta
from qdrant_client.models import (
    NamedVector,
    SearchRequest,
    Formula,
    LinearDecay,
    ExponentialDecay,
    GaussDecay
)

# Document ages: boost newer documents
# Linear decay: score reduces linearly over time
# Exponential decay: rapid initial drop, then stabilizes
# Gaussian decay: smooth bell curve decay

def search_with_time_boost(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    time_field: str = "published_at",
    decay_type: str = "exponential",
    half_life_days: int = 30
):
    """Search with time-based score boosting."""
    
    now = datetime.now().timestamp()
    
    # Configure decay based on type
    if decay_type == "linear":
        # Score drops linearly to 0 over decay period
        decay = LinearDecay(
            origin=now,
            scale=half_life_days * 24 * 3600,  # seconds
            offset=0,
            decay=0.5
        )
    elif decay_type == "exponential":
        # Rapid initial drop, slow tail
        decay = ExponentialDecay(
            origin=now,
            scale=half_life_days * 24 * 3600,
            offset=0,
            decay=0.5
        )
    else:  # gaussian
        # Bell curve - smooth center, steep edges
        decay = GaussDecay(
            origin=now,
            scale=half_life_days * 24 * 3600,
            offset=0,
            decay=0.5
        )
    
    # Apply formula query
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        with_payload=True,
        limit=10,
        # Formula combines vector score with decay
        score_modifier=Formula(
            formula=f"vector_score * time_boost",
            variables={
                "time_boost": decay
            },
            defaults={
                "time_boost": 1.0  # Fallback if field missing
            }
        )
    )
    
    return results
```

---

## Popularity Boosting

Boost by view count, likes, or other engagement metrics:

```python
def search_with_popularity_boost(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    popularity_field: str = "view_count",
    boost_factor: float = 0.1
):
    """Boost scores by popularity metrics."""
    
    # Formula: score = vector_score * (1 + log(popularity + 1) * boost_factor)
    # Log dampens extreme values, +1 prevents log(0)
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10,
        score_modifier=Formula(
            formula=f"""
                vector_score * (1 + log({popularity_field} + 1) * {boost_factor})
            """,
            defaults={
                popularity_field: 0
            }
        )
    )
    
    return results
```

---

## Category Weighting

Boost documents in preferred categories:

```python
def search_with_category_boost(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    preferred_categories: list[str],
    boost_amount: float = 1.5
):
    """Boost documents in preferred categories."""
    
    # Pre-filter to preferred categories, or boost their scores
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10,
        query_filter=models.Filter(
            should=[  # OR condition for category match
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=cat)
                )
                for cat in preferred_categories
            ]
        )
    )
    
    # Alternative: boost without filtering
    # This requires a numeric category score in payload
    results_boosted = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10,
        score_modifier=Formula(
            formula="""
                vector_score * category_weight
            """,
            defaults={
                "category_weight": 1.0  # Non-preferred categories
            }
        )
    )
    
    return results
```

---

## Geographic Distance Boosting

Boost documents closer to a location:

```python
from qdrant_client.models import GeoPoint, GeoRadius

def search_with_geo_boost(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    user_lat: float,
    user_lon: float,
    max_distance_km: float = 100
):
    """Boost documents closer to user location."""
    
    # Geo decay: closer = higher score
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10,
        score_modifier=Formula(
            formula="""
                vector_score * geo_boost
            """,
            variables={
                "geo_boost": GaussDecay(
                    origin=GeoPoint(lat=user_lat, lon=user_lon),
                    scale=max_distance_km * 1000,  # meters
                    offset=1000,  # 1km free distance
                    decay=0.5
                )
            },
            defaults={
                "geo_boost": 0.5  # Documents without location
            }
        )
    )
    
    return results
```

---

## Combined Formula Queries

Combine multiple boost factors:

```python
def search_with_combined_boost(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    user_location: tuple[float, float] | None = None,
    weights: dict = None
):
    """Combine multiple boost factors."""
    
    if weights is None:
        weights = {
            "vector": 0.6,
            "recency": 0.2,
            "popularity": 0.1,
            "quality": 0.1
        }
    
    now = datetime.now().timestamp()
    
    # Build formula components
    formula_parts = [
        f"{weights['vector']} * vector_score",
        f"{weights['recency']} * recency_score",
        f"{weights['popularity']} * log(view_count + 1) / 10",
        f"{weights['quality']} * quality_score"
    ]
    
    formula = " + ".join(formula_parts)
    
    variables = {
        "recency_score": ExponentialDecay(
            origin=now,
            scale=30 * 24 * 3600,  # 30 days
            offset=0,
            decay=0.5
        )
    }
    
    if user_location:
        formula_parts.append("0.1 * geo_score")
        variables["geo_score"] = GaussDecay(
            origin=GeoPoint(lat=user_location[0], lon=user_location[1]),
            scale=50000,  # 50km
            offset=1000,
            decay=0.5
        )
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=10,
        score_modifier=Formula(
            formula=formula,
            variables=variables,
            defaults={
                "recency_score": 0.5,
                "view_count": 0,
                "quality_score": 0.5,
                "geo_score": 0.5
            }
        )
    )
    
    return results
```

---

## Custom Python Score Boosting

For vector databases without formula queries:

```python
import numpy as np
from datetime import datetime, timedelta

class ScoreBooster:
    """Custom score boosting logic."""
    
    def __init__(self):
        self.now = datetime.now()
    
    def boost_recency(
        self,
        timestamp: datetime,
        half_life_days: int = 30
    ) -> float:
        """Exponential decay based on age."""
        
        age_days = (self.now - timestamp).days
        decay = 0.5 ** (age_days / half_life_days)
        return decay
    
    def boost_popularity(
        self,
        view_count: int,
        max_views: int = 10000
    ) -> float:
        """Log-scaled popularity boost."""
        
        # Normalize to 0-1 range with log dampening
        normalized = np.log1p(view_count) / np.log1p(max_views)
        return min(normalized, 1.0)
    
    def boost_quality(
        self,
        rating: float,
        num_ratings: int,
        prior_strength: int = 10
    ) -> float:
        """Bayesian average for quality score."""
        
        # Bayesian average: weight toward prior with few ratings
        prior_rating = 3.5
        weighted = (
            (prior_strength * prior_rating + num_ratings * rating) /
            (prior_strength + num_ratings)
        )
        return weighted / 5.0  # Normalize to 0-1
    
    def combine_scores(
        self,
        vector_score: float,
        metadata: dict,
        weights: dict = None
    ) -> float:
        """Combine vector score with metadata boosts."""
        
        if weights is None:
            weights = {
                "vector": 0.6,
                "recency": 0.2,
                "popularity": 0.1,
                "quality": 0.1
            }
        
        recency = self.boost_recency(
            metadata.get("published_at", self.now)
        )
        popularity = self.boost_popularity(
            metadata.get("view_count", 0)
        )
        quality = self.boost_quality(
            metadata.get("rating", 3.5),
            metadata.get("num_ratings", 0)
        )
        
        final_score = (
            weights["vector"] * vector_score +
            weights["recency"] * recency +
            weights["popularity"] * popularity +
            weights["quality"] * quality
        )
        
        return final_score


def rerank_with_boost(
    results: list[dict],
    booster: ScoreBooster
) -> list[dict]:
    """Re-rank results with score boosting."""
    
    for r in results:
        r["boosted_score"] = booster.combine_scores(
            vector_score=r["score"],
            metadata=r["metadata"]
        )
    
    # Sort by boosted score
    results.sort(key=lambda x: x["boosted_score"], reverse=True)
    
    return results
```

---

## Decay Function Comparison

| Function | Behavior | Use Case |
|----------|----------|----------|
| Linear | Steady decline | Equal importance over time |
| Exponential | Fast initial, slow later | Freshness-critical content |
| Gaussian | Slow start/end, fast middle | Balanced relevance window |

```
Score
  1.0 ─┐
      │╲ Exponential
      │ ╲╲
      │  ╲ ╲ Linear
      │   ╲  ╲
      │    ╲   ╲╲
      │     ╲    ╲╲ Gaussian
  0.5 │──────╲─────╲───────
      │       ╲     ╲
      │        ╲     ╲
      │         ╲     ╲
  0.0 └──────────────────→ Time
          Half-life
```

---

## Best Practices

| ✅ Do | ❌ Don't |
|-------|---------|
| Normalize boost factors to 0-1 | Allow unbounded boosts |
| Log-scale popularity counts | Use raw counts directly |
| Set sensible defaults | Fail on missing metadata |
| Test boost weights empirically | Guess weight values |
| Monitor score distributions | Deploy without analysis |

---

## Summary

✅ **Time decay** boosts recent documents

✅ **Popularity boosting** factors in engagement

✅ **Geo boosting** prefers nearby content

✅ **Formula queries** combine multiple factors

✅ **Normalize all boosts** to 0-1 range

**Next:** [Multi-Stage Prefetch](./10-multi-stage-prefetch.md)

---

<!-- 
Sources Consulted:
- Qdrant Hybrid Queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Qdrant Search: https://qdrant.tech/documentation/concepts/search/
-->
