---
title: "Monitoring and Observability"
---

# Monitoring and Observability

## Introduction

Production embedding systems require comprehensive monitoring across three dimensions: **latency** (how fast), **quality** (how good), and **health** (system status). Without observability, you're flying blind—unable to detect degradation, diagnose issues, or optimize performance.

This lesson covers metrics collection, alerting strategies, and integrating with observability platforms like Prometheus, Grafana, and OpenTelemetry.

### What We'll Cover

- Key metrics to monitor
- Prometheus integration for vector databases
- OpenTelemetry instrumentation
- Search quality metrics
- Alerting strategies

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Basic familiarity with Prometheus/Grafana
- Knowledge of observability concepts (metrics, traces, logs)

---

## The Three Pillars of Embedding Observability

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding System Observability                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    METRICS                               │   │
│  │  ─────────────────────────────                          │   │
│  │  • Latency (p50, p95, p99)                              │   │
│  │  • Throughput (requests/sec)                            │   │
│  │  • Error rates                                          │   │
│  │  • Cache hit rates                                      │   │
│  │  • Vector counts                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    TRACES                                │   │
│  │  ─────────────────────────────                          │   │
│  │  Query → Embed → Cache → Search → Rerank → Response     │   │
│  │  [12ms]  [45ms]  [2ms]   [23ms]   [18ms]                │   │
│  │                                                          │   │
│  │  Identify slow spans, distributed request flow          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     LOGS                                 │   │
│  │  ─────────────────────────────                          │   │
│  │  • Structured JSON logs                                 │   │
│  │  • Request/response details                             │   │
│  │  • Error context and stack traces                       │   │
│  │  • Audit trail (who, what, when)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Metrics to Monitor

### Metric Categories

| Category | Metrics | Target |
|----------|---------|--------|
| **Latency** | Embed time, search time, total time | p99 < 100ms |
| **Throughput** | Queries/sec, embeddings/sec | Based on SLA |
| **Errors** | Error rate, timeout rate | < 0.1% |
| **Resources** | Memory, CPU, disk, connections | < 80% utilization |
| **Quality** | Recall, MRR, NDCG | > 0.8 |
| **Business** | Searches with clicks, zero results | Track trends |

### Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
EMBED_LATENCY = Histogram(
    'embedding_latency_seconds',
    'Time to generate embeddings',
    ['model', 'batch_size'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

SEARCH_LATENCY = Histogram(
    'search_latency_seconds',
    'Time to execute vector search',
    ['index', 'top_k'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)

TOTAL_REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Total end-to-end request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

SEARCH_REQUESTS = Counter(
    'search_requests_total',
    'Total number of search requests',
    ['status', 'cache_hit']
)

EMBEDDING_REQUESTS = Counter(
    'embedding_requests_total',
    'Total embedding generation requests',
    ['model', 'status']
)

VECTOR_COUNT = Gauge(
    'vector_count',
    'Number of vectors in index',
    ['index', 'namespace']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate over sliding window',
    ['cache_type']
)

ERROR_RATE = Gauge(
    'error_rate_percent',
    'Error rate percentage',
    ['service']
)

class MetricsCollector:
    """
    Centralized metrics collection for embedding system.
    """
    def __init__(self, port: int = 8000):
        # Start Prometheus metrics server
        start_http_server(port)
    
    def record_embedding(
        self,
        model: str,
        batch_size: int,
        latency: float,
        success: bool
    ):
        """Record embedding generation metrics."""
        EMBED_LATENCY.labels(
            model=model,
            batch_size=str(batch_size)
        ).observe(latency)
        
        EMBEDDING_REQUESTS.labels(
            model=model,
            status="success" if success else "error"
        ).inc()
    
    def record_search(
        self,
        index: str,
        top_k: int,
        latency: float,
        cache_hit: bool,
        success: bool
    ):
        """Record search metrics."""
        SEARCH_LATENCY.labels(
            index=index,
            top_k=str(top_k)
        ).observe(latency)
        
        SEARCH_REQUESTS.labels(
            status="success" if success else "error",
            cache_hit=str(cache_hit).lower()
        ).inc()
    
    def record_request(self, endpoint: str, latency: float):
        """Record full request latency."""
        TOTAL_REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    
    def update_vector_count(self, index: str, namespace: str, count: int):
        """Update vector count gauge."""
        VECTOR_COUNT.labels(index=index, namespace=namespace).set(count)
```

---

## Prometheus Integration for Vector Databases

### Pinecone Prometheus Metrics

Pinecone provides a Prometheus-compatible metrics endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pinecone'
    scheme: https
    authorization:
      type: Bearer
      credentials: '<your-api-key>'
    static_configs:
      - targets:
        - 'metrics.svc-<uuid>.pinecone.io'
```

**Available Pinecone Metrics:**

| Metric | Description |
|--------|-------------|
| `pinecone_db_record_total` | Total vectors in index |
| `pinecone_db_storage_size_bytes` | Storage consumed |
| `pinecone_db_op_query_count` | Total query operations |
| `pinecone_db_op_query_duration_sum` | Total query time (seconds) |
| `pinecone_db_op_upsert_count` | Total upsert operations |
| `pinecone_db_op_fetch_count` | Total fetch operations |
| `pinecone_db_read_unit_count` | Read units consumed |
| `pinecone_db_write_unit_count` | Write units consumed |

**Example PromQL Queries:**

```promql
# Average query latency
rate(pinecone_db_op_query_duration_sum[5m]) 
  / rate(pinecone_db_op_query_count[5m])

# Queries per second
rate(pinecone_db_op_query_count[5m])

# Storage growth rate
rate(pinecone_db_storage_size_bytes[1h])

# Read unit consumption rate
rate(pinecone_db_read_unit_count[5m])
```

### Qdrant Metrics

Qdrant exposes metrics at `/metrics`:

```python
from prometheus_client.parser import text_string_to_metric_families
import requests

class QdrantMetricsCollector:
    """
    Collect and process Qdrant metrics.
    """
    def __init__(self, qdrant_url: str):
        self.metrics_url = f"{qdrant_url}/metrics"
    
    def fetch_metrics(self) -> dict:
        """Fetch and parse Qdrant metrics."""
        response = requests.get(self.metrics_url)
        
        metrics = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                key = f"{sample.name}"
                if sample.labels:
                    key += f"_{sample.labels}"
                metrics[key] = sample.value
        
        return metrics
    
    def get_collection_stats(self, collection: str) -> dict:
        """Get metrics for specific collection."""
        metrics = self.fetch_metrics()
        
        return {
            "vector_count": metrics.get(f"qdrant_points_count{{collection={collection}}}"),
            "segments": metrics.get(f"qdrant_segments_count{{collection={collection}}}"),
            "index_size_bytes": metrics.get(f"qdrant_index_size_bytes{{collection={collection}}}")
        }
```

---

## OpenTelemetry Instrumentation

### Full Request Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from functools import wraps

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Auto-instrument HTTP requests
RequestsInstrumentor().instrument()

class TracedEmbeddingService:
    """
    Embedding service with OpenTelemetry tracing.
    """
    def __init__(self, embedding_client, vector_db, cache):
        self.embedder = embedding_client
        self.vector_db = vector_db
        self.cache = cache
    
    @tracer.start_as_current_span("search_request")
    def search(self, query: str, top_k: int = 10) -> list:
        """
        Search with full distributed tracing.
        """
        span = trace.get_current_span()
        span.set_attribute("query.length", len(query))
        span.set_attribute("query.top_k", top_k)
        
        # Step 1: Check cache
        with tracer.start_as_current_span("cache_lookup") as cache_span:
            cache_key = f"query:{hash(query)}"
            cached_embedding = self.cache.get(cache_key)
            cache_span.set_attribute("cache.hit", cached_embedding is not None)
        
        # Step 2: Generate embedding if not cached
        if cached_embedding is None:
            with tracer.start_as_current_span("generate_embedding") as embed_span:
                embedding = self.embedder.embed(query)
                embed_span.set_attribute("embedding.dimensions", len(embedding))
                self.cache.set(cache_key, embedding, ttl=3600)
        else:
            embedding = cached_embedding
        
        # Step 3: Vector search
        with tracer.start_as_current_span("vector_search") as search_span:
            results = self.vector_db.search(embedding, top_k=top_k)
            search_span.set_attribute("results.count", len(results))
            search_span.set_attribute("results.top_score", results[0].score if results else 0)
        
        span.set_attribute("search.result_count", len(results))
        
        return results

def traced(span_name: str):
    """Decorator for adding tracing to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

### Trace Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│              Search Request Trace                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  search_request                                    [125ms]      │
│  ├── cache_lookup                                  [2ms]        │
│  │   └── cache.hit: false                                       │
│  ├── generate_embedding                            [52ms]       │
│  │   ├── OpenAI API call                           [48ms]       │
│  │   └── embedding.dimensions: 1536                             │
│  ├── vector_search                                 [28ms]       │
│  │   ├── Pinecone query                            [25ms]       │
│  │   └── results.count: 10                                      │
│  └── rerank                                        [43ms]       │
│      └── Cohere rerank                             [40ms]       │
│                                                                 │
│  Attributes:                                                    │
│  • query.length: 45                                             │
│  • search.result_count: 10                                      │
│  • total.latency_ms: 125                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Search Quality Metrics

### Relevance Metrics

```python
from typing import List
import numpy as np

class SearchQualityMetrics:
    """
    Calculate search quality metrics for evaluation.
    """
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Recall@K: What fraction of relevant items were retrieved?
        
        Args:
            retrieved: List of retrieved document IDs (in order)
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider
        
        Returns:
            Recall score between 0 and 1
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / len(relevant_set)
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Precision@K: What fraction of retrieved items were relevant?
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_at_k:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / len(retrieved_at_k)
    
    def mrr(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Mean Reciprocal Rank: Inverse of rank of first relevant result.
        
        MRR = 1/rank of first relevant result
        """
        relevant_set = set(relevant)
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: dict,
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved: List of retrieved document IDs
            relevance_scores: Dict mapping doc_id -> relevance score (0-3)
            k: Number of results to consider
        """
        def dcg(scores: List[float]) -> float:
            return sum(
                (2**score - 1) / np.log2(i + 2)
                for i, score in enumerate(scores)
            )
        
        # Get relevance scores for retrieved docs
        retrieved_scores = [
            relevance_scores.get(doc_id, 0)
            for doc_id in retrieved[:k]
        ]
        
        # Ideal ranking (best possible)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg_score = dcg(retrieved_scores)
        idcg_score = dcg(ideal_scores)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0

class QualityMonitor:
    """
    Continuous quality monitoring with golden queries.
    """
    def __init__(self, search_service, golden_queries: list, metrics_client):
        self.search = search_service
        self.golden_queries = golden_queries  # Pre-labeled test queries
        self.metrics = metrics_client
        self.quality_calculator = SearchQualityMetrics()
    
    async def run_quality_check(self) -> dict:
        """
        Run quality check against golden query set.
        """
        results = {
            "recall_at_10": [],
            "mrr": [],
            "ndcg_at_10": []
        }
        
        for query_data in self.golden_queries:
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]
            relevance_scores = query_data.get("relevance_scores", {})
            
            # Execute search
            search_results = await self.search.search(query, top_k=10)
            retrieved_ids = [r.id for r in search_results]
            
            # Calculate metrics
            results["recall_at_10"].append(
                self.quality_calculator.recall_at_k(retrieved_ids, relevant_docs, 10)
            )
            results["mrr"].append(
                self.quality_calculator.mrr(retrieved_ids, relevant_docs)
            )
            results["ndcg_at_10"].append(
                self.quality_calculator.ndcg_at_k(retrieved_ids, relevance_scores, 10)
            )
        
        # Aggregate
        aggregated = {
            metric: np.mean(scores)
            for metric, scores in results.items()
        }
        
        # Record to monitoring system
        for metric, value in aggregated.items():
            self.metrics.gauge(f"search_quality_{metric}", value)
        
        return aggregated
```

---

## Alerting Strategies

### Alert Rules

```yaml
# prometheus_rules.yml
groups:
  - name: embedding_alerts
    rules:
      # Latency Alerts
      - alert: HighSearchLatency
        expr: |
          histogram_quantile(0.99, 
            rate(search_latency_seconds_bucket[5m])
          ) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Search p99 latency above 500ms"
          description: "P99 search latency is {{ $value }}s"
      
      - alert: CriticalSearchLatency
        expr: |
          histogram_quantile(0.99,
            rate(search_latency_seconds_bucket[5m])
          ) > 2.0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Search p99 latency critically high"
      
      # Error Rate Alerts
      - alert: HighErrorRate
        expr: |
          sum(rate(search_requests_total{status="error"}[5m]))
          / sum(rate(search_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Search error rate above 1%"
      
      - alert: CriticalErrorRate
        expr: |
          sum(rate(search_requests_total{status="error"}[5m]))
          / sum(rate(search_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Search error rate above 5%"
      
      # Cache Alerts
      - alert: LowCacheHitRate
        expr: cache_hit_rate{cache_type="query"} < 0.3
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Query cache hit rate below 30%"
      
      # Quality Alerts
      - alert: SearchQualityDegraded
        expr: search_quality_recall_at_10 < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Search recall@10 dropped below 70%"
      
      # Resource Alerts
      - alert: HighVectorDBLatency
        expr: |
          rate(pinecone_db_op_query_duration_sum[5m])
          / rate(pinecone_db_op_query_count[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Vector DB average latency above 100ms"
```

### Alert Escalation

```python
from enum import Enum
from dataclasses import dataclass

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: Severity
    message: str
    value: float
    threshold: float

class AlertManager:
    """
    Alert routing and escalation.
    """
    def __init__(self, slack_client, pagerduty_client):
        self.slack = slack_client
        self.pagerduty = pagerduty_client
        
        self.escalation_rules = {
            Severity.INFO: ["slack"],
            Severity.WARNING: ["slack", "email"],
            Severity.CRITICAL: ["slack", "pagerduty"]
        }
    
    def route_alert(self, alert: Alert):
        """Route alert to appropriate channels."""
        channels = self.escalation_rules[alert.severity]
        
        if "slack" in channels:
            self._send_slack(alert)
        
        if "pagerduty" in channels:
            self._create_incident(alert)
    
    def _send_slack(self, alert: Alert):
        """Send alert to Slack."""
        color = {
            Severity.INFO: "#36a64f",
            Severity.WARNING: "#ffa500",
            Severity.CRITICAL: "#ff0000"
        }[alert.severity]
        
        self.slack.post_message(
            channel="#embedding-alerts",
            attachments=[{
                "color": color,
                "title": f"[{alert.severity.value.upper()}] {alert.name}",
                "text": alert.message,
                "fields": [
                    {"title": "Current Value", "value": str(alert.value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True}
                ]
            }]
        )
    
    def _create_incident(self, alert: Alert):
        """Create PagerDuty incident for critical alerts."""
        self.pagerduty.trigger_incident(
            title=f"[CRITICAL] {alert.name}",
            body=alert.message,
            severity="critical",
            source="embedding-service"
        )
```

---

## Dashboard Design

### Grafana Dashboard Panels

```json
{
  "title": "Embedding System Overview",
  "panels": [
    {
      "title": "Search Latency (p50, p95, p99)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(search_latency_seconds_bucket[5m]))",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.95, rate(search_latency_seconds_bucket[5m]))",
          "legendFormat": "p95"
        },
        {
          "expr": "histogram_quantile(0.99, rate(search_latency_seconds_bucket[5m]))",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "Request Rate",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(rate(search_requests_total[5m]))",
          "legendFormat": "Searches/sec"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(rate(search_requests_total{status='error'}[5m])) / sum(rate(search_requests_total[5m])) * 100"
        }
      ],
      "thresholds": [
        {"value": 0, "color": "green"},
        {"value": 1, "color": "yellow"},
        {"value": 5, "color": "red"}
      ]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "cache_hit_rate{cache_type='query'} * 100"
        }
      ]
    },
    {
      "title": "Vector Count by Index",
      "type": "stat",
      "targets": [
        {
          "expr": "vector_count",
          "legendFormat": "{{index}}/{{namespace}}"
        }
      ]
    },
    {
      "title": "Search Quality (Recall@10)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "search_quality_recall_at_10"
        }
      ]
    }
  ]
}
```

---

## Summary

✅ **Monitor three dimensions: latency, throughput, and quality**  
✅ **Use Prometheus for metrics collection and alerting**  
✅ **Implement OpenTelemetry for distributed tracing**  
✅ **Track search quality with Recall@K, MRR, and NDCG**  
✅ **Set up tiered alerting with appropriate escalation paths**

---

**Next:** [Caching at Scale →](./04-caching-at-scale.md)

---

<!-- 
Sources Consulted:
- Pinecone Monitoring: https://docs.pinecone.io/integrations/prometheus
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
- Prometheus Best Practices: https://prometheus.io/docs/practices/naming/
-->
