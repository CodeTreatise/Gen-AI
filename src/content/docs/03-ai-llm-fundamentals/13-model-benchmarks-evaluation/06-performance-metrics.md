---
title: "Performance Metrics"
---

# Performance Metrics

## Introduction

Beyond quality benchmarks, performance metrics like speed, latency, and cost are critical for production applications. Understanding these metrics helps optimize user experience and operational costs.

### What We'll Cover

- Speed benchmarks (tokens/second)
- Time to First Token (TTFT)
- Latency percentiles
- Cost analysis

---

## Speed Benchmarks

### Tokens Per Second (TPS)

```python
speed_metrics = {
    "input_tps": "Tokens processed per second (prompt)",
    "output_tps": "Tokens generated per second (response)",
    "note": "Output TPS more critical for perceived speed"
}
```

### Typical Output Speeds

| Model | Output TPS | Notes |
|-------|------------|-------|
| GPT-4o | 80-120 | Fast for capability level |
| Claude 3.5 Sonnet | 90-150 | Very fast |
| GPT-4o-mini | 150-200 | Optimized for speed |
| Claude 3 Haiku | 180-250 | Built for speed |
| Groq (Llama 70B) | 300-400 | Hardware optimized |

### Measuring Speed

```python
import time

def measure_generation_speed(client, model: str, prompt: str) -> dict:
    """Measure token generation speed"""
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1  # Approximate
    
    end_time = time.time()
    
    return {
        "ttft_ms": (first_token_time - start_time) * 1000,
        "total_time_s": end_time - start_time,
        "tokens_generated": token_count,
        "tokens_per_second": token_count / (end_time - first_token_time)
    }
```

---

## Time to First Token (TTFT)

### Why TTFT Matters

```
┌─────────────────────────────────────────────────────────────┐
│                    USER PERCEPTION                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Request sent                                                │
│      ↓                                                       │
│  [--------- TTFT ---------]                                 │
│                            ↓                                 │
│                      First token appears                     │
│                            ↓                                 │
│  [------- Generation -------]                               │
│                             ↓                                │
│                       Complete response                      │
│                                                              │
│  TTFT < 500ms   → Feels instant                             │
│  TTFT 500-1000ms → Noticeable delay                         │
│  TTFT 1-3s      → Waiting feeling                           │
│  TTFT > 3s      → Feels slow                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Factors Affecting TTFT

```python
ttft_factors = {
    "model_size": "Larger models = longer TTFT",
    "prompt_length": "Longer prompts = more processing time",
    "server_load": "High traffic increases queue time",
    "geographic_distance": "Farther from servers = higher latency",
    "model_type": "Reasoning models have longer TTFT"
}
```

### TTFT by Provider

| Provider / Model | Typical TTFT | Notes |
|------------------|--------------|-------|
| OpenAI GPT-4o | 300-800ms | Consistent |
| Claude 3.5 Sonnet | 400-900ms | Good |
| Groq | 100-300ms | Very fast (custom hardware) |
| Fireworks | 150-400ms | Optimized serving |
| Together | 200-500ms | Fast |

---

## Latency Percentiles

### Understanding P50, P95, P99

```python
latency_percentiles = {
    "p50": "Median - 50% of requests faster than this",
    "p95": "95th percentile - 95% faster, 5% slower",
    "p99": "99th percentile - tail latency, worst 1%"
}

# Example distribution
example_latencies = {
    "p50": 450,    # ms
    "p95": 1200,   # ms
    "p99": 2500    # ms
}

# Production concern: P99 affects user experience
# If P99 = 5s, 1 in 100 users waits 5+ seconds
```

### Measuring Percentiles

```python
import numpy as np
from collections import defaultdict

class LatencyTracker:
    """Track latency percentiles over time"""
    
    def __init__(self):
        self.latencies = []
    
    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
    
    def get_percentiles(self) -> dict:
        if not self.latencies:
            return {}
        
        arr = np.array(self.latencies)
        return {
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "mean": np.mean(arr),
            "max": np.max(arr)
        }
    
    def report(self):
        p = self.get_percentiles()
        print(f"Latency (n={len(self.latencies)}):")
        print(f"  P50: {p['p50']:.0f}ms")
        print(f"  P95: {p['p95']:.0f}ms")
        print(f"  P99: {p['p99']:.0f}ms")
```

---

## Cost Analysis

### Pricing Models

```python
pricing_models = {
    "per_token": {
        "description": "Pay per input + output tokens",
        "typical": "Most API providers",
        "example": "$5 / 1M input, $15 / 1M output"
    },
    "per_request": {
        "description": "Fixed price per API call",
        "typical": "Some image/audio APIs",
        "example": "$0.01 per request"
    },
    "per_second": {
        "description": "Pay for compute time",
        "typical": "Self-hosted, GPU rental",
        "example": "$2/hour for A100 GPU"
    }
}
```

### Current Pricing Comparison

| Model | Input ($/1M) | Output ($/1M) | Relative Cost |
|-------|--------------|---------------|---------------|
| GPT-4o | $2.50 | $10.00 | $$$ |
| GPT-4o-mini | $0.15 | $0.60 | $ |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $$$ |
| Claude 3 Haiku | $0.25 | $1.25 | $ |
| Gemini 1.5 Flash | $0.075 | $0.30 | $ |
| Groq (Llama 70B) | $0.59 | $0.79 | $$ |

### Cost Calculator

```python
def calculate_monthly_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> dict:
    """Calculate estimated monthly API costs"""
    
    daily_input_tokens = requests_per_day * avg_input_tokens
    daily_output_tokens = requests_per_day * avg_output_tokens
    
    monthly_input_tokens = daily_input_tokens * 30
    monthly_output_tokens = daily_output_tokens * 30
    
    input_cost = (monthly_input_tokens / 1_000_000) * input_price_per_million
    output_cost = (monthly_output_tokens / 1_000_000) * output_price_per_million
    
    return {
        "monthly_tokens": {
            "input": monthly_input_tokens,
            "output": monthly_output_tokens
        },
        "monthly_cost": {
            "input": input_cost,
            "output": output_cost,
            "total": input_cost + output_cost
        },
        "daily_cost": (input_cost + output_cost) / 30,
        "per_request_cost": (input_cost + output_cost) / (requests_per_day * 30)
    }

# Example: Customer support chatbot
cost = calculate_monthly_cost(
    requests_per_day=1000,
    avg_input_tokens=500,
    avg_output_tokens=200,
    input_price_per_million=2.50,  # GPT-4o
    output_price_per_million=10.00
)
print(f"Monthly: ${cost['monthly_cost']['total']:.2f}")
# Monthly: $97.50
```

---

## Cost Optimization Strategies

### Strategies

```python
cost_optimization = {
    "model_routing": {
        "description": "Use cheaper models for simpler tasks",
        "savings": "50-90%",
        "implementation": "Classify query complexity, route accordingly"
    },
    "caching": {
        "description": "Cache common responses",
        "savings": "Variable (depends on repeat rate)",
        "implementation": "Semantic similarity matching for cache lookup"
    },
    "prompt_optimization": {
        "description": "Reduce prompt length",
        "savings": "10-50%",
        "implementation": "Compress system prompts, trim context"
    },
    "batching": {
        "description": "Batch multiple requests when possible",
        "savings": "Reduced overhead",
        "implementation": "Queue similar requests"
    }
}
```

---

## Summary

✅ **TPS**: Tokens per second for generation speed

✅ **TTFT**: Time to first token for perceived responsiveness

✅ **Percentiles**: P50/P95/P99 for production monitoring

✅ **Cost**: Token-based pricing, significant differences between models

✅ **Optimization**: Routing, caching, prompt compression

**Next:** [Interpreting Scores](./07-interpreting-scores.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Chatbot Arena](./05-chatbot-arena.md) | [Benchmarks](./00-model-benchmarks-evaluation.md) | [Interpreting Scores](./07-interpreting-scores.md) |
