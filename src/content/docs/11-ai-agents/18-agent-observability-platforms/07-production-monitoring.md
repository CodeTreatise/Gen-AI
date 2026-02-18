---
title: "Production Monitoring"
---

# Production Monitoring

## Introduction

Development tracing and evaluation tell you if your agent **can** work. Production monitoring tells you if it **is** working — right now, for real users, at scale. Production brings challenges that don't exist in development: traffic spikes, model provider outages, gradual quality drift, cost overruns, and failure modes you never anticipated.

This lesson covers the metrics, dashboards, alerting patterns, and operational practices that keep agent systems healthy in production. We build monitoring that answers: Is the agent fast? Is it correct? Is it costing too much? Is anything breaking?

### What we'll cover

- Core production metrics for agents
- Success and failure rate tracking
- Latency percentile monitoring (P50/P95/P99)
- Error categorization and alerting
- Cost monitoring and budget enforcement
- Quality drift detection
- Building production dashboards
- Alerting configuration and runbooks

### Prerequisites

- Observability platform setup (Lessons 18-01 through 18-05)
- Evaluation fundamentals (Lesson 18-06)
- Basic understanding of metrics (counters, histograms, percentiles)

---

## Core production metrics

Every production agent system needs these metrics at minimum:

| Metric | Type | What It Answers |
|--------|------|----------------|
| `agent.requests.total` | Counter | How many requests are we handling? |
| `agent.requests.errors` | Counter | How many are failing? |
| `agent.request.duration_seconds` | Histogram | How long do requests take? |
| `agent.tokens.total` | Counter | How many tokens are we consuming? |
| `agent.cost.usd` | Counter | How much are we spending? |
| `agent.tools.calls` | Counter | Which tools are being used? |
| `agent.tools.errors` | Counter | Which tools are failing? |
| `agent.quality.score` | Gauge | What's our current quality level? |

### Implementing core metrics

```python
import time
import functools
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable

@dataclass
class AgentMetrics:
    """Production metrics collector for agent systems."""
    
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list = field(default_factory=list)
    errors_by_type: dict = field(default_factory=lambda: defaultdict(int))
    tool_calls: dict = field(default_factory=lambda: defaultdict(int))
    tool_errors: dict = field(default_factory=lambda: defaultdict(int))
    
    def record_request(self, duration: float, tokens: int, cost: float):
        self.total_requests += 1
        self.latencies.append(duration)
        self.total_tokens += tokens
        self.total_cost += cost
    
    def record_error(self, error_type: str):
        self.total_errors += 1
        self.errors_by_type[error_type] += 1
    
    def record_tool_call(self, tool_name: str, success: bool):
        self.tool_calls[tool_name] += 1
        if not success:
            self.tool_errors[tool_name] += 1
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests
    
    def percentile(self, p: float) -> float:
        """Calculate latency percentile."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "error_rate": f"{self.error_rate:.2%}",
            "p50_latency": f"{self.percentile(50):.2f}s",
            "p95_latency": f"{self.percentile(95):.2f}s",
            "p99_latency": f"{self.percentile(99):.2f}s",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
        }

metrics = AgentMetrics()
```

---

## Success and failure rate tracking

### Wrapping agent calls

```python
from openai import OpenAI

client = OpenAI()

def monitored_agent(question: str) -> str:
    """Agent wrapped with production monitoring."""
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        
        duration = time.time() - start
        usage = response.usage
        
        # Estimate cost (gpt-4o-mini pricing)
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
        
        metrics.record_request(
            duration=duration,
            tokens=usage.total_tokens,
            cost=cost,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        duration = time.time() - start
        error_type = classify_error(e)
        metrics.record_error(error_type)
        metrics.record_request(duration=duration, tokens=0, cost=0)
        raise

# Run some requests
for q in ["What is AI?", "What is ML?", "What is DL?"]:
    answer = monitored_agent(q)
    print(f"Q: {q} → {answer[:50]}...")

print("\n📊 Metrics Summary:")
for k, v in metrics.summary().items():
    print(f"  {k}: {v}")
```

**Output:**
```
Q: What is AI? → Artificial intelligence (AI) refers to the simul...
Q: What is ML? → Machine learning (ML) is a subset of AI that en...
Q: What is DL? → Deep learning (DL) is a subset of machine learn...

📊 Metrics Summary:
  total_requests: 3
  error_rate: 0.00%
  p50_latency: 1.23s
  p95_latency: 1.87s
  p99_latency: 1.87s
  total_tokens: 2541
  total_cost: $0.0015
```

---

## Error categorization

Not all errors are equal. Categorizing them helps prioritize fixes and configure appropriate alerts.

```python
from openai import (
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
)

def classify_error(error: Exception) -> str:
    """Classify agent errors into actionable categories."""
    if isinstance(error, APIConnectionError):
        return "network_error"
    elif isinstance(error, RateLimitError):
        return "rate_limit"
    elif isinstance(error, APITimeoutError):
        return "timeout"
    elif isinstance(error, APIStatusError):
        if error.status_code == 400:
            return "bad_request"
        elif error.status_code == 401:
            return "auth_error"
        elif error.status_code == 500:
            return "provider_error"
        elif error.status_code == 503:
            return "service_unavailable"
        return f"api_error_{error.status_code}"
    elif isinstance(error, ValueError):
        return "validation_error"
    elif isinstance(error, TimeoutError):
        return "agent_timeout"
    else:
        return "unknown_error"
```

### Error severity levels

| Category | Severity | Action |
|----------|----------|--------|
| `network_error` | 🔴 Critical | Check connectivity, alert on-call |
| `rate_limit` | 🟡 Warning | Increase limits, add backoff, alert if sustained |
| `timeout` | 🟡 Warning | Increase timeout, optimize prompt, check model load |
| `provider_error` | 🔴 Critical | Provider outage; switch to fallback model |
| `auth_error` | 🔴 Critical | API key expired or revoked |
| `bad_request` | 🟠 Medium | Input validation issue; fix the prompt/input pipeline |
| `validation_error` | 🟠 Medium | Agent output didn't match expected format |
| `unknown_error` | 🔴 Critical | Unexpected — investigate immediately |

---

## Latency percentile monitoring

Averages hide problems. A P50 of 1.2s might mask a P99 of 15s, meaning 1 in 100 users waits 15 seconds.

### Calculating percentiles

```python
import math

def calculate_percentiles(latencies: list[float]) -> dict:
    """Calculate P50, P95, P99 latency percentiles."""
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0}
    
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    
    def percentile(p):
        idx = math.ceil(n * p / 100) - 1
        return sorted_lat[max(0, min(idx, n - 1))]
    
    return {
        "p50": round(percentile(50), 3),
        "p95": round(percentile(95), 3),
        "p99": round(percentile(99), 3),
    }

# Example
latencies = [1.2, 0.8, 1.5, 2.1, 0.9, 1.1, 3.5, 1.3, 0.7, 15.2]
p = calculate_percentiles(latencies)
print(f"P50: {p['p50']}s | P95: {p['p95']}s | P99: {p['p99']}s")
```

**Output:**
```
P50: 1.2s | P95: 15.2s | P99: 15.2s
```

### Latency budgets

| Tier | P50 Target | P95 Target | P99 Target |
|------|-----------|-----------|-----------|
| **Real-time chat** | < 2s | < 5s | < 10s |
| **Document analysis** | < 10s | < 30s | < 60s |
| **Batch processing** | < 30s | < 120s | < 300s |
| **Background agents** | < 60s | < 300s | < 600s |

---

## Alert configuration

### Alert rules

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    name: str
    condition: Callable[[AgentMetrics], bool]
    severity: AlertSeverity
    message: str

# Define alert rules
alert_rules = [
    AlertRule(
        name="high_error_rate",
        condition=lambda m: m.error_rate > 0.05,
        severity=AlertSeverity.CRITICAL,
        message="Error rate exceeds 5%: {error_rate}",
    ),
    AlertRule(
        name="high_latency",
        condition=lambda m: m.percentile(95) > 10.0,
        severity=AlertSeverity.WARNING,
        message="P95 latency exceeds 10s: {p95}s",
    ),
    AlertRule(
        name="cost_spike",
        condition=lambda m: m.total_cost > 100.0,
        severity=AlertSeverity.WARNING,
        message="Total cost exceeds $100: ${cost}",
    ),
    AlertRule(
        name="rate_limit_storm",
        condition=lambda m: m.errors_by_type.get("rate_limit", 0) > 10,
        severity=AlertSeverity.CRITICAL,
        message="Rate limit errors: {count}",
    ),
]

def check_alerts(metrics: AgentMetrics) -> list[dict]:
    """Evaluate all alert rules against current metrics."""
    triggered = []
    for rule in alert_rules:
        if rule.condition(metrics):
            triggered.append({
                "name": rule.name,
                "severity": rule.severity.value,
                "message": rule.message.format(
                    error_rate=f"{metrics.error_rate:.2%}",
                    p95=f"{metrics.percentile(95):.1f}",
                    cost=f"{metrics.total_cost:.2f}",
                    count=sum(metrics.errors_by_type.values()),
                ),
            })
    return triggered

# Check
alerts = check_alerts(metrics)
for alert in alerts:
    print(f"🚨 [{alert['severity'].upper()}] {alert['name']}: {alert['message']}")
```

### Notification channels

| Channel | Best For | Implementation |
|---------|----------|---------------|
| **Slack** | Team alerts, non-urgent | Webhook POST to Slack API |
| **PagerDuty** | Critical on-call alerts | PagerDuty Events API |
| **Email** | Daily summaries, reports | SMTP or SendGrid |
| **Dashboard** | Real-time visibility | Grafana annotations |
| **SMS** | Critical after-hours | Twilio or PagerDuty |

```python
import requests

def send_slack_alert(alert: dict, webhook_url: str):
    """Send an alert to Slack."""
    color = {
        "critical": "#FF0000",
        "warning": "#FFA500",
        "info": "#0000FF",
    }.get(alert["severity"], "#808080")
    
    payload = {
        "attachments": [{
            "color": color,
            "title": f"🚨 Agent Alert: {alert['name']}",
            "text": alert["message"],
            "fields": [
                {"title": "Severity", "value": alert["severity"], "short": True},
            ],
        }],
    }
    
    requests.post(webhook_url, json=payload)
```

---

## Quality drift detection

Agent quality can degrade over time due to model updates, data distribution shifts, or prompt degradation. Monitor quality continuously.

```python
from collections import deque

class QualityMonitor:
    """Track quality scores over time and detect drift."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.80):
        self.scores = deque(maxlen=window_size)
        self.threshold = threshold
        self.baseline_avg: float = None
    
    def record_score(self, score: float):
        self.scores.append(score)
    
    def set_baseline(self):
        """Set the current average as the baseline."""
        if self.scores:
            self.baseline_avg = sum(self.scores) / len(self.scores)
    
    @property
    def current_avg(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)
    
    @property
    def drift_detected(self) -> bool:
        """True if quality has dropped below threshold."""
        return self.current_avg < self.threshold
    
    @property
    def drift_from_baseline(self) -> float:
        """Percentage drop from baseline."""
        if not self.baseline_avg:
            return 0.0
        return (self.current_avg - self.baseline_avg) / self.baseline_avg
    
    def status(self) -> dict:
        return {
            "current_avg": f"{self.current_avg:.2%}",
            "baseline": f"{self.baseline_avg:.2%}" if self.baseline_avg else "not set",
            "drift": f"{self.drift_from_baseline:+.2%}",
            "below_threshold": self.drift_detected,
            "sample_size": len(self.scores),
        }

# Usage
monitor = QualityMonitor(window_size=50, threshold=0.85)

# Simulate baseline scores
for _ in range(50):
    monitor.record_score(0.90 + (0.05 * (0.5 - __import__('random').random())))
monitor.set_baseline()

# Simulate degraded scores
for _ in range(20):
    monitor.record_score(0.78 + (0.05 * (0.5 - __import__('random').random())))

print("Quality Status:")
for k, v in monitor.status().items():
    print(f"  {k}: {v}")
```

**Output:**
```
Quality Status:
  current_avg: 83.20%
  baseline: 90.50%
  drift: -8.07%
  below_threshold: True
  sample_size: 50
```

---

## Production dashboard layout

### Recommended panels

```
┌─────────────────────────────────────────────────────────┐
│  🟢 Agent Health Dashboard                               │
├─────────────┬──────────────┬──────────────┬─────────────┤
│ Request Rate│  Error Rate  │  P95 Latency │  Cost/Hour  │
│   42 req/s  │    0.3%      │    2.1s      │   $12.50    │
├─────────────┴──────────────┴──────────────┴─────────────┤
│  📈 Requests Over Time (24h)                             │
│  [sparkline graph]                                       │
├─────────────────────────────────────────────────────────┤
│  📊 Latency Distribution         │  🔧 Tool Usage       │
│  P50: 1.2s  P95: 2.1s  P99: 5s  │  search: 45%        │
│  [histogram]                      │  calculator: 30%     │
│                                   │  code_exec: 25%      │
├─────────────────────────────────────────────────────────┤
│  ❌ Errors by Category           │  💰 Cost by Model     │
│  rate_limit: 12                   │  gpt-4o: $8.50      │
│  timeout: 3                       │  gpt-4o-mini: $4.00  │
│  provider_error: 1                │                      │
├─────────────────────────────────────────────────────────┤
│  📋 Recent Alerts                                        │
│  🟡 P95 latency > 5s at 14:23   🔴 Error rate > 5% 14:30│
└─────────────────────────────────────────────────────────┘
```

---

## Best practices

| Practice | Why It Matters |
|----------|----------------|
| Monitor percentiles, not averages | Averages hide tail latency affecting real users |
| Categorize errors by type | Different errors need different responses |
| Set alert thresholds conservatively | Start strict; loosen after you understand normal patterns |
| Track cost per user/feature | Identify expensive workflows before they become budget problems |
| Monitor quality continuously | Model updates can silently degrade output quality |
| Build runbooks for each alert | On-call engineers need clear steps, not just notifications |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| Using averages for latency | Use P50/P95/P99 percentiles |
| Alerting on every single error | Set thresholds (e.g., error rate > 5%, not count > 0) |
| No cost monitoring | Track cost per request and set daily/weekly budget limits |
| No quality monitoring in production | Sample 5–10% of responses and run automated evaluation |
| Alert fatigue from too many notifications | Tune thresholds, aggregate alerts, use escalation policies |
| No runbooks | Every alert should link to a playbook with diagnostic steps |

---

## Hands-on exercise

### Your task

Build a production monitoring wrapper for an agent with metrics collection, error categorization, and alert checking.

### Requirements

1. Implement `AgentMetrics` with request, error, and tool call tracking
2. Create a `classify_error` function for 4+ error types
3. Wrap an agent function with monitoring
4. Define at least 3 alert rules (error rate, latency, cost)
5. Run 10 requests and print a metrics summary with alert status

### Expected result

A metrics summary showing request count, error rate, latency percentiles, cost, and any triggered alerts.

<details>
<summary>💡 Hints (click to expand)</summary>

- Use `time.time()` to measure latency
- `response.usage.total_tokens` gives token count for cost calculation
- Sort latencies and index at `len * percentile / 100` for percentiles
- Use a lambda-based condition in `AlertRule` for flexible thresholds

</details>

<details>
<summary>✅ Solution (click to expand)</summary>

```python
import time
from openai import OpenAI
from collections import defaultdict

client = OpenAI()

class Metrics:
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.latencies = []
        self.tokens = 0
        self.cost = 0.0
        self.error_types = defaultdict(int)
    
    def record(self, duration, tokens, cost):
        self.requests += 1
        self.latencies.append(duration)
        self.tokens += tokens
        self.cost += cost
    
    def record_error(self, etype):
        self.errors += 1
        self.error_types[etype] += 1
    
    def p(self, pct):
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[min(int(len(s) * pct / 100), len(s) - 1)]

m = Metrics()

def agent(q):
    start = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
        )
        dur = time.time() - start
        cost = (r.usage.prompt_tokens * 0.15 + r.usage.completion_tokens * 0.60) / 1e6
        m.record(dur, r.usage.total_tokens, cost)
        return r.choices[0].message.content
    except Exception as e:
        m.record(time.time() - start, 0, 0)
        m.record_error(type(e).__name__)
        return f"Error: {e}"

questions = [
    "What is AI?", "What is ML?", "What is DL?", "What is NLP?",
    "What is CV?", "What is RL?", "What is GAN?", "What is LLM?",
    "What is RAG?", "What is RLHF?",
]

for q in questions:
    agent(q)

# Summary
print(f"Requests: {m.requests}")
print(f"Errors: {m.errors} ({m.errors/max(m.requests,1):.1%})")
print(f"P50: {m.p(50):.2f}s | P95: {m.p(95):.2f}s | P99: {m.p(99):.2f}s")
print(f"Tokens: {m.tokens} | Cost: ${m.cost:.4f}")

# Alerts
if m.errors / max(m.requests, 1) > 0.05:
    print("🚨 ALERT: Error rate > 5%")
if m.p(95) > 10.0:
    print("🚨 ALERT: P95 latency > 10s")
if m.cost > 1.0:
    print("🚨 ALERT: Cost exceeds $1.00")
print("✅ Monitoring check complete")
```

</details>

### Bonus challenges

- [ ] Add a `QualityMonitor` that detects quality drift using sampled LLM-judge scores
- [ ] Send alerts to a Slack webhook
- [ ] Export metrics to Prometheus and create a Grafana dashboard

---

## Summary

✅ **Core metrics** (requests, errors, latency, tokens, cost) provide essential production visibility  
✅ **Error categorization** enables targeted responses — rate limits need backoff, auth errors need key rotation  
✅ **Latency percentiles** (P50/P95/P99) reveal tail latency that averages hide  
✅ **Alert rules** with severity levels and notification channels enable proactive incident response  
✅ **Quality drift detection** catches silent regressions from model updates or data shifts  

**Previous:** [Evaluation Frameworks](./06-evaluation-frameworks.md)  
**Back to:** [Agent Observability Platforms](./00-agent-observability-platforms.md)

---

## Further Reading

- [Grafana + Prometheus Stack](https://grafana.com/docs/grafana/latest/getting-started/) — Dashboard and alerting
- [PagerDuty Alerting Best Practices](https://www.pagerduty.com/resources/learn/alerting-best-practices/) — On-call and escalation
- [Google SRE Book — Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/) — Industry-standard monitoring practices
- [OpenTelemetry Metrics](https://opentelemetry.io/docs/concepts/signals/metrics/) — Vendor-neutral metrics specification

<!--
Sources Consulted:
- Google SRE Book monitoring chapter: https://sre.google/sre-book/monitoring-distributed-systems/
- OpenTelemetry metrics: https://opentelemetry.io/docs/concepts/signals/metrics/
- Grafana alerting docs: https://grafana.com/docs/grafana/latest/alerting/
-->
