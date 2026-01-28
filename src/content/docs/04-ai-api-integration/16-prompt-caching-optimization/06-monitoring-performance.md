---
title: "Monitoring Performance"
---

# Monitoring Performance

## Introduction

Effective cache monitoring is essential for optimizing costs and latency. This lesson covers tracking cache hit rates, measuring latency improvements, calculating cost savings, and building dashboards for cache performance visibility.

### What We'll Cover

- Cache hit rate tracking
- Latency monitoring
- Cost savings analysis
- Performance dashboards
- Alert configuration

### Prerequisites

- Understanding of caching mechanisms
- Experience with OpenAI or Anthropic APIs
- Python development environment

---

## Cache Hit Rate Tracking

### OpenAI Usage Details

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class CacheMetrics:
    """Metrics from a single request."""
    
    timestamp: datetime
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    cache_key: Optional[str] = None
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate for this request."""
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens
    
    @property
    def is_cache_hit(self) -> bool:
        """Whether this request had any cache hit."""
        return self.cached_tokens > 0


def extract_openai_metrics(response, latency_ms: float) -> CacheMetrics:
    """Extract cache metrics from OpenAI response."""
    
    usage = response.usage
    
    # Get cached tokens from details
    cached = 0
    if hasattr(usage, 'prompt_tokens_details'):
        details = usage.prompt_tokens_details
        cached = getattr(details, 'cached_tokens', 0)
    
    return CacheMetrics(
        timestamp=datetime.now(),
        input_tokens=usage.prompt_tokens,
        cached_tokens=cached,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        model=response.model
    )


def extract_anthropic_metrics(response, latency_ms: float) -> CacheMetrics:
    """Extract cache metrics from Anthropic response."""
    
    usage = response.usage
    
    # Anthropic tracks cache read separately
    cached = getattr(usage, 'cache_read_input_tokens', 0)
    written = getattr(usage, 'cache_creation_input_tokens', 0)
    
    return CacheMetrics(
        timestamp=datetime.now(),
        input_tokens=usage.input_tokens,
        cached_tokens=cached,
        output_tokens=usage.output_tokens,
        latency_ms=latency_ms,
        model=response.model
    )
```

### Aggregated Statistics

```python
from typing import Dict
from collections import defaultdict

@dataclass
class AggregatedCacheStats:
    """Aggregated cache statistics over time."""
    
    metrics: List[CacheMetrics] = field(default_factory=list)
    
    def add(self, metric: CacheMetrics):
        """Add a metric."""
        self.metrics.append(metric)
    
    @property
    def total_requests(self) -> int:
        return len(self.metrics)
    
    @property
    def total_input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.metrics)
    
    @property
    def total_cached_tokens(self) -> int:
        return sum(m.cached_tokens for m in self.metrics)
    
    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate."""
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_input_tokens
    
    @property
    def requests_with_hits(self) -> int:
        """Number of requests that had cache hits."""
        return sum(1 for m in self.metrics if m.is_cache_hit)
    
    @property
    def request_hit_rate(self) -> float:
        """Percentage of requests with cache hits."""
        if not self.metrics:
            return 0.0
        return self.requests_with_hits / len(self.metrics)
    
    def by_model(self) -> Dict[str, dict]:
        """Get statistics grouped by model."""
        
        by_model = defaultdict(lambda: {
            "requests": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0
        })
        
        for m in self.metrics:
            by_model[m.model]["requests"] += 1
            by_model[m.model]["input_tokens"] += m.input_tokens
            by_model[m.model]["cached_tokens"] += m.cached_tokens
            by_model[m.model]["output_tokens"] += m.output_tokens
        
        # Calculate hit rates
        for model, stats in by_model.items():
            if stats["input_tokens"] > 0:
                stats["hit_rate"] = stats["cached_tokens"] / stats["input_tokens"]
            else:
                stats["hit_rate"] = 0.0
        
        return dict(by_model)
    
    def by_cache_key(self) -> Dict[str, dict]:
        """Get statistics grouped by cache key."""
        
        by_key = defaultdict(lambda: {
            "requests": 0,
            "hits": 0,
            "cached_tokens": 0
        })
        
        for m in self.metrics:
            key = m.cache_key or "default"
            by_key[key]["requests"] += 1
            if m.is_cache_hit:
                by_key[key]["hits"] += 1
            by_key[key]["cached_tokens"] += m.cached_tokens
        
        # Calculate hit rates
        for key, stats in by_key.items():
            stats["hit_rate"] = stats["hits"] / stats["requests"] if stats["requests"] > 0 else 0.0
        
        return dict(by_key)
    
    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_requests": self.total_requests,
            "requests_with_hits": self.requests_with_hits,
            "request_hit_rate": f"{self.request_hit_rate:.1%}",
            "total_input_tokens": self.total_input_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "overall_hit_rate": f"{self.overall_hit_rate:.1%}"
        }


# Usage
stats = AggregatedCacheStats()

# Add metrics from requests
# stats.add(extract_openai_metrics(response, latency))

print(stats.summary())
```

---

## Latency Monitoring

### Tracking Request Latency

```python
import time
from contextlib import contextmanager
from typing import Callable

@contextmanager
def timed_request():
    """Context manager for timing requests."""
    start = time.perf_counter()
    result = {"latency_ms": 0}
    try:
        yield result
    finally:
        result["latency_ms"] = (time.perf_counter() - start) * 1000


class LatencyTracker:
    """Track latency with cache correlation."""
    
    def __init__(self):
        self.measurements: List[dict] = []
    
    def record(
        self,
        latency_ms: float,
        cached_tokens: int,
        total_tokens: int,
        model: str
    ):
        """Record a latency measurement."""
        
        self.measurements.append({
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "cache_ratio": cached_tokens / total_tokens if total_tokens > 0 else 0,
            "model": model
        })
    
    def get_latency_by_cache_ratio(self) -> dict:
        """Analyze latency by cache hit ratio."""
        
        buckets = {
            "0%": [],
            "1-25%": [],
            "26-50%": [],
            "51-75%": [],
            "76-99%": [],
            "100%": []
        }
        
        for m in self.measurements:
            ratio = m["cache_ratio"]
            
            if ratio == 0:
                bucket = "0%"
            elif ratio <= 0.25:
                bucket = "1-25%"
            elif ratio <= 0.50:
                bucket = "26-50%"
            elif ratio <= 0.75:
                bucket = "51-75%"
            elif ratio < 1.0:
                bucket = "76-99%"
            else:
                bucket = "100%"
            
            buckets[bucket].append(m["latency_ms"])
        
        # Calculate averages
        result = {}
        for bucket, latencies in buckets.items():
            if latencies:
                result[bucket] = {
                    "count": len(latencies),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies)
                }
        
        return result
    
    def estimate_latency_savings(self) -> dict:
        """Estimate latency savings from caching."""
        
        if not self.measurements:
            return {"error": "No measurements"}
        
        # Compare cached vs uncached requests
        cached = [m for m in self.measurements if m["cache_ratio"] > 0.5]
        uncached = [m for m in self.measurements if m["cache_ratio"] <= 0.5]
        
        if not cached or not uncached:
            return {"note": "Need both cached and uncached requests for comparison"}
        
        avg_cached = sum(m["latency_ms"] for m in cached) / len(cached)
        avg_uncached = sum(m["latency_ms"] for m in uncached) / len(uncached)
        
        savings_ms = avg_uncached - avg_cached
        savings_pct = (savings_ms / avg_uncached) * 100 if avg_uncached > 0 else 0
        
        return {
            "cached_requests": len(cached),
            "uncached_requests": len(uncached),
            "avg_cached_latency_ms": f"{avg_cached:.1f}",
            "avg_uncached_latency_ms": f"{avg_uncached:.1f}",
            "latency_savings_ms": f"{savings_ms:.1f}",
            "latency_savings_pct": f"{savings_pct:.1f}%"
        }


# Usage
tracker = LatencyTracker()

# Record measurements
# with timed_request() as timing:
#     response = client.chat.completions.create(...)
# tracker.record(timing["latency_ms"], cached, total, model)
```

### Percentile Analysis

```python
import statistics

class LatencyAnalyzer:
    """Analyze latency percentiles."""
    
    def __init__(self, measurements: List[float]):
        self.measurements = sorted(measurements)
    
    def percentile(self, p: float) -> float:
        """Get percentile value."""
        if not self.measurements:
            return 0.0
        
        k = (len(self.measurements) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        
        if c >= len(self.measurements):
            return self.measurements[-1]
        
        return self.measurements[f] + (k - f) * (self.measurements[c] - self.measurements[f])
    
    def summary(self) -> dict:
        """Get percentile summary."""
        if not self.measurements:
            return {"error": "No measurements"}
        
        return {
            "count": len(self.measurements),
            "min": f"{min(self.measurements):.1f}ms",
            "p50": f"{self.percentile(50):.1f}ms",
            "p90": f"{self.percentile(90):.1f}ms",
            "p95": f"{self.percentile(95):.1f}ms",
            "p99": f"{self.percentile(99):.1f}ms",
            "max": f"{max(self.measurements):.1f}ms",
            "mean": f"{statistics.mean(self.measurements):.1f}ms",
            "stdev": f"{statistics.stdev(self.measurements):.1f}ms" if len(self.measurements) > 1 else "N/A"
        }


# Usage
latencies = [150, 180, 200, 220, 250, 300, 180, 190, 210, 500]
analyzer = LatencyAnalyzer(latencies)
print(analyzer.summary())
```

---

## Cost Savings Analysis

### Cost Calculator

```python
from enum import Enum

class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class PricingConfig:
    """Pricing configuration per 1M tokens."""
    
    input_rate: float
    cached_rate: float
    output_rate: float
    cache_write_rate: Optional[float] = None  # Anthropic only


# Pricing as of 2025 (example rates)
PRICING = {
    "gpt-4o": PricingConfig(
        input_rate=2.50,
        cached_rate=1.25,  # 50% discount
        output_rate=10.00
    ),
    "gpt-4o-mini": PricingConfig(
        input_rate=0.15,
        cached_rate=0.075,
        output_rate=0.60
    ),
    "o1": PricingConfig(
        input_rate=15.00,
        cached_rate=7.50,
        output_rate=60.00
    ),
    "claude-sonnet-4-20250514": PricingConfig(
        input_rate=3.00,
        cached_rate=0.30,  # 90% discount
        output_rate=15.00,
        cache_write_rate=3.75  # 25% premium
    ),
    "claude-3-5-sonnet-20241022": PricingConfig(
        input_rate=3.00,
        cached_rate=0.30,
        output_rate=15.00,
        cache_write_rate=3.75
    ),
}


class CostAnalyzer:
    """Analyze costs with caching."""
    
    def __init__(self, model: str):
        self.model = model
        self.pricing = PRICING.get(model)
        if not self.pricing:
            raise ValueError(f"Unknown model: {model}")
    
    def calculate_request_cost(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0
    ) -> dict:
        """Calculate cost for a single request."""
        
        uncached_tokens = input_tokens - cached_tokens
        
        # Input costs
        uncached_cost = (uncached_tokens / 1_000_000) * self.pricing.input_rate
        cached_cost = (cached_tokens / 1_000_000) * self.pricing.cached_rate
        output_cost = (output_tokens / 1_000_000) * self.pricing.output_rate
        
        # Cache write cost (Anthropic)
        write_cost = 0
        if self.pricing.cache_write_rate and cache_write_tokens > 0:
            write_cost = (cache_write_tokens / 1_000_000) * self.pricing.cache_write_rate
        
        actual_cost = uncached_cost + cached_cost + output_cost + write_cost
        
        # Cost without caching
        full_input_cost = (input_tokens / 1_000_000) * self.pricing.input_rate
        full_cost = full_input_cost + output_cost
        
        savings = full_cost - actual_cost
        
        return {
            "actual_cost": actual_cost,
            "without_cache": full_cost,
            "savings": savings,
            "savings_pct": (savings / full_cost * 100) if full_cost > 0 else 0,
            "breakdown": {
                "uncached_input": uncached_cost,
                "cached_input": cached_cost,
                "cache_write": write_cost,
                "output": output_cost
            }
        }
    
    def analyze_batch(self, metrics: List[CacheMetrics]) -> dict:
        """Analyze costs for a batch of requests."""
        
        total_actual = 0
        total_without_cache = 0
        
        for m in metrics:
            result = self.calculate_request_cost(
                input_tokens=m.input_tokens,
                cached_tokens=m.cached_tokens,
                output_tokens=m.output_tokens
            )
            total_actual += result["actual_cost"]
            total_without_cache += result["without_cache"]
        
        total_savings = total_without_cache - total_actual
        
        return {
            "total_requests": len(metrics),
            "total_actual_cost": f"${total_actual:.4f}",
            "total_without_cache": f"${total_without_cache:.4f}",
            "total_savings": f"${total_savings:.4f}",
            "savings_pct": f"{(total_savings / total_without_cache * 100):.1f}%" if total_without_cache > 0 else "0%",
            "avg_cost_per_request": f"${total_actual / len(metrics):.6f}" if metrics else "$0"
        }


# Usage
analyzer = CostAnalyzer("gpt-4o")

result = analyzer.calculate_request_cost(
    input_tokens=5000,
    cached_tokens=4000,
    output_tokens=500
)

print(f"Actual cost: ${result['actual_cost']:.4f}")
print(f"Savings: ${result['savings']:.4f} ({result['savings_pct']:.1f}%)")
```

### ROI Tracking

```python
class CacheROITracker:
    """Track return on investment for caching."""
    
    def __init__(self, model: str):
        self.cost_analyzer = CostAnalyzer(model)
        self.requests: List[dict] = []
    
    def record_request(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0
    ):
        """Record a request for ROI tracking."""
        
        cost_result = self.cost_analyzer.calculate_request_cost(
            input_tokens, cached_tokens, output_tokens, cache_write_tokens
        )
        
        self.requests.append({
            "timestamp": datetime.now(),
            **cost_result
        })
    
    def get_roi_summary(self) -> dict:
        """Get ROI summary."""
        
        if not self.requests:
            return {"error": "No requests recorded"}
        
        total_actual = sum(r["actual_cost"] for r in self.requests)
        total_without = sum(r["without_cache"] for r in self.requests)
        total_savings = total_without - total_actual
        
        # Calculate when caching became profitable
        cumulative_savings = 0
        break_even_request = None
        
        for i, r in enumerate(self.requests):
            cumulative_savings += r["savings"]
            if cumulative_savings > 0 and break_even_request is None:
                break_even_request = i + 1
        
        return {
            "total_requests": len(self.requests),
            "total_spent": f"${total_actual:.4f}",
            "would_have_spent": f"${total_without:.4f}",
            "total_saved": f"${total_savings:.4f}",
            "roi_pct": f"{(total_savings / total_actual * 100):.1f}%" if total_actual > 0 else "N/A",
            "break_even_request": break_even_request or "Not yet profitable",
            "avg_savings_per_request": f"${total_savings / len(self.requests):.6f}"
        }


# Usage
roi = CacheROITracker("gpt-4o")

# Simulate requests
# First request - cache write (no savings yet for Anthropic)
roi.record_request(5000, 0, 500, cache_write_tokens=4000)

# Subsequent requests - cache hits
for _ in range(10):
    roi.record_request(5000, 4000, 500)

print(roi.get_roi_summary())
```

---

## Performance Dashboard

### Real-Time Metrics

```python
from collections import deque
from threading import Lock

class RealTimeMetricsDashboard:
    """Real-time cache performance dashboard."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self._lock = Lock()
    
    def record(self, metric: CacheMetrics):
        """Record a metric (thread-safe)."""
        with self._lock:
            self.metrics.append(metric)
    
    def get_current_stats(self) -> dict:
        """Get current window statistics."""
        
        with self._lock:
            if not self.metrics:
                return {"status": "No data"}
            
            metrics_list = list(self.metrics)
        
        # Calculate stats
        total_input = sum(m.input_tokens for m in metrics_list)
        total_cached = sum(m.cached_tokens for m in metrics_list)
        total_output = sum(m.output_tokens for m in metrics_list)
        
        latencies = [m.latency_ms for m in metrics_list]
        avg_latency = sum(latencies) / len(latencies)
        
        hits = sum(1 for m in metrics_list if m.is_cache_hit)
        
        return {
            "window_size": len(metrics_list),
            "requests_with_hits": hits,
            "hit_rate": f"{hits / len(metrics_list) * 100:.1f}%",
            "token_cache_rate": f"{total_cached / total_input * 100:.1f}%" if total_input > 0 else "0%",
            "avg_latency_ms": f"{avg_latency:.1f}",
            "total_input_tokens": total_input,
            "total_cached_tokens": total_cached,
            "total_output_tokens": total_output,
            "oldest": metrics_list[0].timestamp.isoformat() if metrics_list else None,
            "newest": metrics_list[-1].timestamp.isoformat() if metrics_list else None
        }
    
    def get_time_series(self, bucket_minutes: int = 1) -> List[dict]:
        """Get metrics as time series buckets."""
        
        with self._lock:
            metrics_list = list(self.metrics)
        
        if not metrics_list:
            return []
        
        # Group by time bucket
        buckets = defaultdict(list)
        
        for m in metrics_list:
            # Round to bucket
            bucket_ts = m.timestamp.replace(
                second=0,
                microsecond=0,
                minute=(m.timestamp.minute // bucket_minutes) * bucket_minutes
            )
            buckets[bucket_ts].append(m)
        
        # Calculate stats per bucket
        series = []
        for ts in sorted(buckets.keys()):
            bucket_metrics = buckets[ts]
            
            input_tokens = sum(m.input_tokens for m in bucket_metrics)
            cached_tokens = sum(m.cached_tokens for m in bucket_metrics)
            
            series.append({
                "timestamp": ts.isoformat(),
                "requests": len(bucket_metrics),
                "hit_rate": cached_tokens / input_tokens if input_tokens > 0 else 0,
                "avg_latency_ms": sum(m.latency_ms for m in bucket_metrics) / len(bucket_metrics)
            })
        
        return series


# Usage
dashboard = RealTimeMetricsDashboard(window_size=100)

# Record metrics
# dashboard.record(metric)

# Get current stats
print(dashboard.get_current_stats())
```

### Structured Logging

```python
import json
import logging
from typing import Any

class CacheMetricsLogger:
    """Structured logging for cache metrics."""
    
    def __init__(self, logger_name: str = "cache_metrics"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_request(
        self,
        request_id: str,
        model: str,
        cache_key: str,
        metrics: CacheMetrics,
        cost_result: dict
    ):
        """Log request metrics as structured JSON."""
        
        log_entry = {
            "event": "cache_request",
            "timestamp": metrics.timestamp.isoformat(),
            "request_id": request_id,
            "model": model,
            "cache_key": cache_key,
            "input_tokens": metrics.input_tokens,
            "cached_tokens": metrics.cached_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_hit": metrics.is_cache_hit,
            "hit_rate": metrics.hit_rate,
            "latency_ms": metrics.latency_ms,
            "cost_usd": cost_result["actual_cost"],
            "savings_usd": cost_result["savings"]
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_summary(
        self,
        period: str,
        stats: dict
    ):
        """Log periodic summary."""
        
        log_entry = {
            "event": "cache_summary",
            "timestamp": datetime.now().isoformat(),
            "period": period,
            **stats
        }
        
        self.logger.info(json.dumps(log_entry))


# Usage
metrics_logger = CacheMetricsLogger()

# Log individual request
# metrics_logger.log_request(
#     request_id="req_123",
#     model="gpt-4o",
#     cache_key="app_v1",
#     metrics=metric,
#     cost_result=cost
# )
```

---

## Alert Configuration

### Threshold Alerts

```python
from enum import Enum
from typing import Callable

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    name: str
    condition: Callable[[dict], bool]
    severity: AlertSeverity
    message_template: str


class CacheAlertManager:
    """Manage cache performance alerts."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.triggered_alerts: List[dict] = []
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def add_low_hit_rate_rule(
        self,
        threshold: float = 0.3,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """Add rule for low cache hit rate."""
        
        self.add_rule(AlertRule(
            name="low_cache_hit_rate",
            condition=lambda stats: stats.get("hit_rate", 1.0) < threshold,
            severity=severity,
            message_template=f"Cache hit rate below {threshold:.0%}: {{hit_rate:.1%}}"
        ))
    
    def add_high_latency_rule(
        self,
        threshold_ms: float = 2000,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """Add rule for high latency."""
        
        self.add_rule(AlertRule(
            name="high_latency",
            condition=lambda stats: stats.get("avg_latency_ms", 0) > threshold_ms,
            severity=severity,
            message_template=f"Average latency above {threshold_ms}ms: {{avg_latency_ms:.0f}}ms"
        ))
    
    def add_cost_spike_rule(
        self,
        threshold_pct: float = 50,
        severity: AlertSeverity = AlertSeverity.CRITICAL
    ):
        """Add rule for cost spikes."""
        
        self.add_rule(AlertRule(
            name="cost_spike",
            condition=lambda stats: stats.get("cost_increase_pct", 0) > threshold_pct,
            severity=severity,
            message_template=f"Cost increased by more than {threshold_pct}%: {{cost_increase_pct:.1f}}%"
        ))
    
    def evaluate(self, stats: dict) -> List[dict]:
        """Evaluate all rules against current stats."""
        
        alerts = []
        
        for rule in self.rules:
            try:
                if rule.condition(stats):
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "rule": rule.name,
                        "severity": rule.severity.value,
                        "message": rule.message_template.format(**stats),
                        "stats": stats
                    }
                    alerts.append(alert)
                    self.triggered_alerts.append(alert)
            except Exception as e:
                # Log but don't fail on rule evaluation errors
                pass
        
        return alerts
    
    def get_alert_summary(self) -> dict:
        """Get summary of triggered alerts."""
        
        by_severity = defaultdict(int)
        by_rule = defaultdict(int)
        
        for alert in self.triggered_alerts:
            by_severity[alert["severity"]] += 1
            by_rule[alert["rule"]] += 1
        
        return {
            "total_alerts": len(self.triggered_alerts),
            "by_severity": dict(by_severity),
            "by_rule": dict(by_rule)
        }


# Usage
alert_manager = CacheAlertManager()

# Configure rules
alert_manager.add_low_hit_rate_rule(threshold=0.3)
alert_manager.add_high_latency_rule(threshold_ms=2000)
alert_manager.add_cost_spike_rule(threshold_pct=50)

# Evaluate
stats = {
    "hit_rate": 0.25,  # Low!
    "avg_latency_ms": 1500,
    "cost_increase_pct": 10
}

alerts = alert_manager.evaluate(stats)
for alert in alerts:
    print(f"[{alert['severity'].upper()}] {alert['message']}")
```

---

## Hands-on Exercise

### Your Task

Build a comprehensive cache monitoring system.

### Requirements

1. Track cache hits, latency, and costs
2. Provide real-time statistics
3. Calculate ROI over time
4. Generate alerts for anomalies

<details>
<summary>💡 Hints</summary>

- Use dataclasses for structured metrics
- Calculate running averages efficiently
- Compare cached vs uncached performance
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import json

class AlertLevel(Enum):
    INFO = "info"
    WARN = "warning"
    ERROR = "error"


@dataclass
class RequestMetric:
    """Single request metric."""
    
    timestamp: datetime
    model: str
    cache_key: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    savings_usd: float
    
    @property
    def hit_rate(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens
    
    @property
    def is_hit(self) -> bool:
        return self.cached_tokens > 0


@dataclass
class Alert:
    """Alert record."""
    
    timestamp: datetime
    level: AlertLevel
    rule: str
    message: str
    context: Dict


class CacheMonitoringSystem:
    """Comprehensive cache monitoring system."""
    
    # Pricing (per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
        "claude-sonnet-4-20250514": {"input": 3.00, "cached": 0.30, "output": 15.00},
    }
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict] = None
    ):
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self.alerts: List[Alert] = []
        
        # Default thresholds
        self.thresholds = alert_thresholds or {
            "min_hit_rate": 0.3,
            "max_latency_ms": 3000,
            "cost_spike_pct": 100
        }
        
        # Baseline tracking
        self._baseline_cost: Optional[float] = None
        self._baseline_latency: Optional[float] = None
    
    def record(
        self,
        model: str,
        cache_key: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_write_tokens: int = 0
    ) -> RequestMetric:
        """Record a request and return the metric."""
        
        # Calculate cost
        cost_result = self._calculate_cost(
            model, input_tokens, cached_tokens, output_tokens
        )
        
        metric = RequestMetric(
            timestamp=datetime.now(),
            model=model,
            cache_key=cache_key,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_result["actual"],
            savings_usd=cost_result["savings"]
        )
        
        self.metrics.append(metric)
        
        # Check alerts
        self._check_alerts(metric)
        
        return metric
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int
    ) -> Dict:
        """Calculate cost for a request."""
        
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o"])
        
        uncached = input_tokens - cached_tokens
        
        uncached_cost = (uncached / 1_000_000) * pricing["input"]
        cached_cost = (cached_tokens / 1_000_000) * pricing["cached"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        actual = uncached_cost + cached_cost + output_cost
        full = (input_tokens / 1_000_000) * pricing["input"] + output_cost
        
        return {
            "actual": actual,
            "full": full,
            "savings": full - actual
        }
    
    def _check_alerts(self, metric: RequestMetric):
        """Check for alert conditions."""
        
        # Low hit rate (only if we expect caching)
        if metric.input_tokens >= 1024 and metric.hit_rate < self.thresholds["min_hit_rate"]:
            self._add_alert(
                AlertLevel.WARN,
                "low_hit_rate",
                f"Cache hit rate {metric.hit_rate:.1%} below threshold",
                {"hit_rate": metric.hit_rate, "cache_key": metric.cache_key}
            )
        
        # High latency
        if metric.latency_ms > self.thresholds["max_latency_ms"]:
            self._add_alert(
                AlertLevel.WARN,
                "high_latency",
                f"Latency {metric.latency_ms:.0f}ms exceeds threshold",
                {"latency_ms": metric.latency_ms, "model": metric.model}
            )
        
        # Cost spike
        if self._baseline_cost and metric.cost_usd > 0:
            increase_pct = ((metric.cost_usd - self._baseline_cost) / self._baseline_cost) * 100
            if increase_pct > self.thresholds["cost_spike_pct"]:
                self._add_alert(
                    AlertLevel.ERROR,
                    "cost_spike",
                    f"Cost increased {increase_pct:.0f}% from baseline",
                    {"cost": metric.cost_usd, "baseline": self._baseline_cost}
                )
    
    def _add_alert(
        self,
        level: AlertLevel,
        rule: str,
        message: str,
        context: Dict
    ):
        """Add an alert."""
        self.alerts.append(Alert(
            timestamp=datetime.now(),
            level=level,
            rule=rule,
            message=message,
            context=context
        ))
    
    def get_realtime_stats(self) -> Dict:
        """Get real-time statistics."""
        
        if not self.metrics:
            return {"status": "No data"}
        
        metrics_list = list(self.metrics)
        
        # Calculate aggregates
        total_input = sum(m.input_tokens for m in metrics_list)
        total_cached = sum(m.cached_tokens for m in metrics_list)
        total_output = sum(m.output_tokens for m in metrics_list)
        
        total_cost = sum(m.cost_usd for m in metrics_list)
        total_savings = sum(m.savings_usd for m in metrics_list)
        
        latencies = [m.latency_ms for m in metrics_list]
        
        hits = sum(1 for m in metrics_list if m.is_hit)
        
        return {
            "window_requests": len(metrics_list),
            "cache_metrics": {
                "requests_with_hits": hits,
                "request_hit_rate": f"{hits / len(metrics_list):.1%}",
                "token_hit_rate": f"{total_cached / total_input:.1%}" if total_input > 0 else "0%",
                "total_cached_tokens": total_cached
            },
            "latency_metrics": {
                "avg_ms": f"{statistics.mean(latencies):.1f}",
                "p50_ms": f"{statistics.median(latencies):.1f}",
                "p95_ms": f"{sorted(latencies)[int(len(latencies) * 0.95)]:.1f}" if len(latencies) > 20 else "N/A",
                "max_ms": f"{max(latencies):.1f}"
            },
            "cost_metrics": {
                "total_cost": f"${total_cost:.4f}",
                "total_savings": f"${total_savings:.4f}",
                "savings_rate": f"{total_savings / (total_cost + total_savings):.1%}" if (total_cost + total_savings) > 0 else "0%"
            },
            "time_range": {
                "oldest": metrics_list[0].timestamp.isoformat(),
                "newest": metrics_list[-1].timestamp.isoformat()
            }
        }
    
    def get_stats_by_cache_key(self) -> Dict[str, Dict]:
        """Get statistics grouped by cache key."""
        
        by_key = defaultdict(lambda: {
            "requests": 0,
            "hits": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "cost": 0,
            "savings": 0,
            "latencies": []
        })
        
        for m in self.metrics:
            stats = by_key[m.cache_key]
            stats["requests"] += 1
            if m.is_hit:
                stats["hits"] += 1
            stats["input_tokens"] += m.input_tokens
            stats["cached_tokens"] += m.cached_tokens
            stats["cost"] += m.cost_usd
            stats["savings"] += m.savings_usd
            stats["latencies"].append(m.latency_ms)
        
        # Format results
        result = {}
        for key, stats in by_key.items():
            result[key] = {
                "requests": stats["requests"],
                "hit_rate": f"{stats['hits'] / stats['requests']:.1%}",
                "token_cache_rate": f"{stats['cached_tokens'] / stats['input_tokens']:.1%}" if stats['input_tokens'] > 0 else "0%",
                "avg_latency_ms": f"{statistics.mean(stats['latencies']):.1f}",
                "total_cost": f"${stats['cost']:.4f}",
                "total_savings": f"${stats['savings']:.4f}"
            }
        
        return result
    
    def get_roi_analysis(self) -> Dict:
        """Get ROI analysis."""
        
        if not self.metrics:
            return {"status": "No data"}
        
        metrics_list = list(self.metrics)
        
        total_cost = sum(m.cost_usd for m in metrics_list)
        total_savings = sum(m.savings_usd for m in metrics_list)
        would_have_cost = total_cost + total_savings
        
        # Find break-even point
        cumulative = 0
        break_even = None
        for i, m in enumerate(metrics_list):
            cumulative += m.savings_usd
            if cumulative > 0 and break_even is None:
                break_even = i + 1
        
        return {
            "total_requests": len(metrics_list),
            "total_spent": f"${total_cost:.4f}",
            "would_have_spent": f"${would_have_cost:.4f}",
            "total_saved": f"${total_savings:.4f}",
            "roi_percentage": f"{(total_savings / total_cost * 100):.1f}%" if total_cost > 0 else "N/A",
            "break_even_request": break_even or "N/A",
            "avg_savings_per_request": f"${total_savings / len(metrics_list):.6f}"
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        
        recent = self.alerts[-limit:] if len(self.alerts) > limit else self.alerts
        
        return [
            {
                "timestamp": a.timestamp.isoformat(),
                "level": a.level.value,
                "rule": a.rule,
                "message": a.message
            }
            for a in reversed(recent)
        ]
    
    def export_metrics(self) -> List[Dict]:
        """Export all metrics as JSON-serializable dicts."""
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "model": m.model,
                "cache_key": m.cache_key,
                "input_tokens": m.input_tokens,
                "cached_tokens": m.cached_tokens,
                "output_tokens": m.output_tokens,
                "hit_rate": m.hit_rate,
                "latency_ms": m.latency_ms,
                "cost_usd": m.cost_usd,
                "savings_usd": m.savings_usd
            }
            for m in self.metrics
        ]
    
    def set_baseline(self):
        """Set current averages as baseline for anomaly detection."""
        
        if not self.metrics:
            return
        
        metrics_list = list(self.metrics)
        self._baseline_cost = statistics.mean(m.cost_usd for m in metrics_list)
        self._baseline_latency = statistics.mean(m.latency_ms for m in metrics_list)


# Usage demonstration
monitor = CacheMonitoringSystem(window_size=1000)

# Simulate requests
import random

for i in range(50):
    # Simulate varying cache hits
    input_tokens = random.randint(1500, 5000)
    
    # Higher cache rate after first few requests
    if i < 5:
        cached_tokens = 0  # Cache warming
    else:
        cached_tokens = int(input_tokens * random.uniform(0.6, 0.9))
    
    output_tokens = random.randint(200, 800)
    latency = random.uniform(100, 500) if cached_tokens > 0 else random.uniform(300, 1000)
    
    monitor.record(
        model="gpt-4o",
        cache_key="demo_app_v1",
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        latency_ms=latency
    )

# Display results
print("=== Real-Time Stats ===")
stats = monitor.get_realtime_stats()
for category, values in stats.items():
    if isinstance(values, dict):
        print(f"\n{category}:")
        for k, v in values.items():
            print(f"  {k}: {v}")
    else:
        print(f"{category}: {values}")

print("\n=== Stats by Cache Key ===")
for key, stats in monitor.get_stats_by_cache_key().items():
    print(f"\n{key}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

print("\n=== ROI Analysis ===")
roi = monitor.get_roi_analysis()
for k, v in roi.items():
    print(f"{k}: {v}")

print("\n=== Recent Alerts ===")
for alert in monitor.get_recent_alerts(5):
    print(f"[{alert['level'].upper()}] {alert['message']}")
```

</details>

---

## Summary

✅ Track cache hit rates via usage details  
✅ Monitor latency correlation with cache hits  
✅ Calculate actual cost savings  
✅ Build real-time dashboards  
✅ Configure alerts for anomalies

**Next:** [Best Practices](./07-best-practices.md)

---

## Further Reading

- [OpenAI Usage API](https://platform.openai.com/docs/api-reference/usage) — Usage tracking
- [Anthropic Usage](https://docs.anthropic.com/en/api/usage) — Usage details
- [Observability](https://platform.openai.com/docs/guides/production-best-practices) — Production monitoring
