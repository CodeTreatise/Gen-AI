---
title: "Full Response Latency"
---

# Full Response Latency

## Introduction

Full response latency measures the total time from request to complete response. Understanding what determines this metric helps you estimate costs, set timeouts, and manage user expectations.

### What We'll Cover

- Total generation time components
- Output length impact
- Model size considerations
- Infrastructure factors
- Estimating and optimizing

---

## Total Generation Time

Full latency consists of TTFT plus token generation time.

### The Equation

```
Full Latency = TTFT + (Output Tokens × Time per Token)
```

### Visual Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL RESPONSE LATENCY                     │
├──────────────────────┬──────────────────────────────────────┤
│        TTFT          │         GENERATION TIME               │
│    (Fixed cost)      │    (Proportional to output)           │
│                      │                                       │
│  • Prompt processing │  • Token 1                            │
│  • Queue time        │  • Token 2                            │
│  • Network latency   │  • Token 3                            │
│                      │  • ...                                │
│    ~500ms-2000ms     │  • Token N                            │
│                      │                                       │
│                      │    N × ~20-50ms per token             │
└──────────────────────┴──────────────────────────────────────┘
```

### Measuring Components

```python
import time

def measure_latency_components(messages: list, model: str = "gpt-4") -> dict:
    """Break down latency into components"""
    
    start = time.time()
    first_token = None
    tokens = []
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    token_times = []
    for chunk in response:
        now = time.time()
        if chunk.choices[0].delta.content:
            if first_token is None:
                first_token = now
            else:
                token_times.append(now - token_times[-1] if token_times else now - first_token)
            tokens.append(chunk.choices[0].delta.content)
    
    end = time.time()
    
    avg_time_per_token = sum(token_times) / len(token_times) if token_times else 0
    
    return {
        "ttft_ms": (first_token - start) * 1000,
        "generation_ms": (end - first_token) * 1000 if first_token else 0,
        "total_ms": (end - start) * 1000,
        "token_count": len(tokens),
        "avg_ms_per_token": avg_time_per_token * 1000,
        "tokens_per_second": len(tokens) / (end - first_token) if first_token and end > first_token else 0
    }

result = measure_latency_components(
    [{"role": "user", "content": "Write a paragraph about AI."}]
)
print(f"TTFT: {result['ttft_ms']:.0f}ms")
print(f"Generation: {result['generation_ms']:.0f}ms")
print(f"Total: {result['total_ms']:.0f}ms")
print(f"Tokens: {result['token_count']}")
print(f"Speed: {result['tokens_per_second']:.1f} tokens/sec")
```

---

## Output Length Impact

Generation time scales linearly with output length.

### The Math

```python
def estimate_generation_time(
    output_tokens: int,
    model: str = "gpt-4",
    ttft_ms: float = 500
) -> float:
    """Estimate total generation time in milliseconds"""
    
    # Approximate tokens per second by model
    tokens_per_second = {
        "gpt-3.5-turbo": 80,
        "gpt-4": 40,
        "gpt-4-turbo": 50,
        "claude-3-haiku": 100,
        "claude-3-sonnet": 60,
        "claude-3-opus": 30,
    }
    
    tps = tokens_per_second.get(model, 50)
    generation_ms = (output_tokens / tps) * 1000
    
    return ttft_ms + generation_ms

# Examples
print("Estimated total time:")
print(f"Short (50 tokens): {estimate_generation_time(50):.0f}ms")
print(f"Medium (200 tokens): {estimate_generation_time(200):.0f}ms")
print(f"Long (1000 tokens): {estimate_generation_time(1000):.0f}ms")

# Short: ~1.75s
# Medium: ~5.5s
# Long: ~25.5s
```

### Real-World Examples

| Response Type | Tokens | GPT-4 Time | GPT-3.5 Time |
|---------------|--------|------------|--------------|
| Yes/No | 5 | ~0.6s | ~0.3s |
| Short answer | 50 | ~1.8s | ~0.9s |
| Paragraph | 150 | ~4.3s | ~2.1s |
| Essay | 500 | ~13s | ~6.5s |
| Long document | 2000 | ~51s | ~25s |

### Practical Implications

```python
def set_appropriate_timeout(
    expected_tokens: int,
    model: str,
    safety_factor: float = 1.5
) -> float:
    """Calculate appropriate timeout for a request"""
    
    estimated_ms = estimate_generation_time(expected_tokens, model)
    timeout_ms = estimated_ms * safety_factor
    
    return max(timeout_ms / 1000, 30)  # Minimum 30 seconds

# Usage
timeout = set_appropriate_timeout(1000, "gpt-4")
print(f"Recommended timeout: {timeout:.0f} seconds")
```

---

## Model Size Considerations

Larger models generate tokens more slowly.

### Token Generation Speed

```python
# Typical tokens per second (TPS) by model class
model_speeds = {
    # Fast (small models)
    "fast": {
        "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-flash"],
        "tps": "60-120 tokens/sec",
        "use_for": "High-volume, low-complexity",
    },
    
    # Balanced
    "balanced": {
        "models": ["gpt-4o", "claude-3-sonnet", "gemini-pro"],
        "tps": "40-70 tokens/sec", 
        "use_for": "Most production use cases",
    },
    
    # Powerful (large models)
    "powerful": {
        "models": ["gpt-4-turbo", "claude-3-opus"],
        "tps": "20-40 tokens/sec",
        "use_for": "Complex reasoning, when quality > speed",
    },
}
```

### Choosing Based on Latency Budget

```python
def select_model_for_latency_budget(
    expected_output_tokens: int,
    max_latency_seconds: float,
    min_quality: str = "medium"
) -> str:
    """
    Select fastest model that meets latency and quality requirements.
    """
    
    models = [
        {"name": "gpt-3.5-turbo", "tps": 80, "quality": "low"},
        {"name": "gpt-4o-mini", "tps": 70, "quality": "medium"},
        {"name": "gpt-4o", "tps": 50, "quality": "high"},
        {"name": "gpt-4-turbo", "tps": 40, "quality": "very_high"},
    ]
    
    quality_order = ["low", "medium", "high", "very_high"]
    min_quality_index = quality_order.index(min_quality)
    
    for model in models:
        if quality_order.index(model["quality"]) < min_quality_index:
            continue
        
        estimated_time = 0.5 + (expected_output_tokens / model["tps"])
        if estimated_time <= max_latency_seconds:
            return model["name"]
    
    return models[-1]["name"]  # Best quality as fallback

# Example
model = select_model_for_latency_budget(
    expected_output_tokens=200,
    max_latency_seconds=5,
    min_quality="medium"
)
print(f"Recommended: {model}")
```

---

## Infrastructure Factors

### Factors Beyond Your Control

```python
infrastructure_factors = {
    "api_load": {
        "description": "Current load on provider's servers",
        "impact": "10-50% latency variation",
        "mitigation": "Retry logic, multiple providers",
    },
    
    "geographic_distance": {
        "description": "Network distance to API server",
        "impact": "20-200ms added latency",
        "mitigation": "Use regional endpoints",
    },
    
    "time_of_day": {
        "description": "Peak vs off-peak hours",
        "impact": "20-100% latency variation",
        "mitigation": "Cache, async processing",
    },
    
    "model_availability": {
        "description": "Model cold start (serverless)",
        "impact": "1-5s for first request",
        "mitigation": "Keep-alive requests",
    },
}
```

### Measuring Variability

```python
import statistics
from datetime import datetime

class LatencyTracker:
    """Track latency patterns over time"""
    
    def __init__(self):
        self.measurements = []
    
    def record(self, latency_ms: float):
        self.measurements.append({
            "latency_ms": latency_ms,
            "timestamp": datetime.now(),
            "hour": datetime.now().hour
        })
    
    def get_stats(self) -> dict:
        if not self.measurements:
            return {}
        
        latencies = [m["latency_ms"] for m in self.measurements]
        
        return {
            "count": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        }
    
    def by_hour(self) -> dict:
        """Group stats by hour of day"""
        by_hour = {}
        for m in self.measurements:
            hour = m["hour"]
            if hour not in by_hour:
                by_hour[hour] = []
            by_hour[hour].append(m["latency_ms"])
        
        return {
            hour: statistics.mean(latencies) 
            for hour, latencies in by_hour.items()
        }
```

---

## Optimization Strategies

### 1. Limit Output Length

```python
def request_with_length_limit(
    prompt: str,
    max_chars: int = 500
) -> str:
    """Request with explicit length constraint"""
    
    # Include length constraint in prompt
    constrained_prompt = f"{prompt}\n\nRespond in {max_chars} characters or less."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": constrained_prompt}],
        max_tokens=max_chars // 4  # Rough chars-to-tokens ratio
    )
    
    return response.choices[0].message.content
```

### 2. Use Streaming for Perceived Speed

```python
# While total time is the same, streaming feels faster
# User starts reading at TTFT, not at completion

async def stream_with_progress(messages: list):
    """Stream with progress indication"""
    
    start = time.time()
    chars_received = 0
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chars_received += len(content)
            print(content, end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\n\n[Completed in {elapsed:.1f}s, {chars_received} characters]")
```

### 3. Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_queries(queries: list) -> list:
    """Run multiple queries in parallel"""
    
    async def query_one(q):
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": q}]
        )
        return response.choices[0].message.content
    
    results = await asyncio.gather(*[query_one(q) for q in queries])
    return results

# Total time ≈ longest single query, not sum of all
```

### 4. Chunked Responses

```python
def chunked_generation(
    prompt: str,
    max_chunk_tokens: int = 200
) -> str:
    """Generate in chunks for interruptibility"""
    
    full_response = ""
    
    while True:
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        if full_response:
            messages.append({
                "role": "assistant", 
                "content": full_response
            })
            messages.append({
                "role": "user",
                "content": "Continue from where you left off. Say 'COMPLETE' when finished."
            })
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=max_chunk_tokens
        )
        
        chunk = response.choices[0].message.content
        
        if "COMPLETE" in chunk:
            full_response += chunk.replace("COMPLETE", "")
            break
        
        full_response += chunk
        
        # Could check with user or add progress here
    
    return full_response
```

---

## Hands-on Exercise

### Your Task

Build a latency estimator and validator:

```python
import time
from openai import OpenAI

client = OpenAI()

class LatencyEstimator:
    """Estimate and validate response latency"""
    
    def __init__(self):
        # Calibrate with your actual observations
        self.model_tps = {
            "gpt-3.5-turbo": 80,
            "gpt-4": 40,
            "gpt-4-turbo": 50,
        }
        self.model_ttft = {
            "gpt-3.5-turbo": 300,
            "gpt-4": 600,
            "gpt-4-turbo": 800,
        }
    
    def estimate(self, model: str, output_tokens: int) -> dict:
        """Estimate latency for given parameters"""
        
        ttft = self.model_ttft.get(model, 500)
        tps = self.model_tps.get(model, 50)
        generation_ms = (output_tokens / tps) * 1000
        total_ms = ttft + generation_ms
        
        return {
            "ttft_ms": ttft,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "recommended_timeout_s": (total_ms * 1.5) / 1000
        }
    
    def measure_and_compare(
        self, 
        model: str, 
        messages: list,
        expected_tokens: int
    ) -> dict:
        """Measure actual latency and compare to estimate"""
        
        estimate = self.estimate(model, expected_tokens)
        
        # Actual measurement
        start = time.time()
        first_token_time = None
        token_count = 0
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
        
        end = time.time()
        
        actual = {
            "ttft_ms": (first_token_time - start) * 1000 if first_token_time else 0,
            "total_ms": (end - start) * 1000,
            "token_count": token_count
        }
        
        return {
            "estimate": estimate,
            "actual": actual,
            "ttft_error": abs(actual["ttft_ms"] - estimate["ttft_ms"]) / estimate["ttft_ms"] * 100,
            "total_error": abs(actual["total_ms"] - estimate["total_ms"]) / estimate["total_ms"] * 100,
        }

# Test the estimator
estimator = LatencyEstimator()

# Estimate for 100 tokens
est = estimator.estimate("gpt-4", 100)
print("Estimate for 100 tokens:")
print(f"  TTFT: {est['ttft_ms']}ms")
print(f"  Total: {est['total_ms']}ms")
print(f"  Timeout: {est['recommended_timeout_s']:.1f}s")

# Measure and compare
result = estimator.measure_and_compare(
    "gpt-4",
    [{"role": "user", "content": "Write 2 sentences about AI."}],
    expected_tokens=50
)

print("\nMeasurement comparison:")
print(f"  Estimated total: {result['estimate']['total_ms']:.0f}ms")
print(f"  Actual total: {result['actual']['total_ms']:.0f}ms")
print(f"  Error: {result['total_error']:.1f}%")
```

---

## Summary

✅ **Full latency** = TTFT + (tokens × time per token)

✅ **Output length** directly impacts generation time

✅ **Larger models** generate tokens more slowly

✅ **Infrastructure varies**: load, location, time of day

✅ **Optimize**: Limit output, use streaming, parallelize

✅ **Set appropriate timeouts** based on expected output

**Next:** [Use Cases by Mode](./07-use-cases.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [First Token Latency](./05-first-token-latency.md) | [Streaming Response Modes](./00-streaming-response-modes.md) | [Use Cases](./07-use-cases.md) |

