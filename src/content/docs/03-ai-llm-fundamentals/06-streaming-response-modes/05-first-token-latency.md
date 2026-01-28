---
title: "First Token Latency (TTFT)"
---

# First Token Latency (TTFT)

## Introduction

Time to First Token (TTFT) measures how quickly a user sees the first piece of response. This metric is crucial for perceived performance in streaming applications.

### What We'll Cover

- What TTFT measures
- Why TTFT matters for UX
- Factors affecting TTFT
- Optimization strategies
- Provider comparisons

---

## What TTFT Measures

TTFT is the time from sending a request to receiving the first token of the response.

### The Timeline

```
Request                First Token              Last Token
   │                        │                        │
   ├────────────────────────┼────────────────────────┤
   │                        │                        │
   │◄─────── TTFT ─────────►│◄─── Generation ───────►│
   │                        │                        │
   │     (Processing)       │    (Token by token)    │
```

### Measuring TTFT

```python
import time
from openai import OpenAI

client = OpenAI()

def measure_ttft(messages: list) -> dict:
    """Measure time to first token and total time"""
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
    
    end_time = time.time()
    
    return {
        "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
        "total_time_ms": (end_time - start_time) * 1000,
        "tokens": token_count,
        "generation_time_ms": (end_time - first_token_time) * 1000 if first_token_time else None,
    }

# Example usage
result = measure_ttft([{"role": "user", "content": "Hello!"}])
print(f"TTFT: {result['ttft_ms']:.0f}ms")
print(f"Total: {result['total_time_ms']:.0f}ms")
```

### Typical TTFT Values

| Model Size | Typical TTFT |
|------------|--------------|
| Small (7B) | 100-300ms |
| Medium (13-30B) | 200-500ms |
| Large (GPT-4, Claude) | 300-1000ms |
| Very Large (GPT-4 Turbo) | 500-1500ms |

---

## Why TTFT Matters

### User Perception

```
Response Time Perception:
─────────────────────────

0-100ms:    Instant ⚡
100-300ms:  Fast ✓
300-1000ms: Noticeable ○
1-3s:       Slow △
>3s:        Frustrating ✗

TTFT directly determines first impression!
```

### The Psychology

```python
# Nielsen Norman Group research shows:
user_perception = {
    "100ms": "Feels instantaneous, no delay perceived",
    "1000ms": "User's flow of thought is maintained",
    "10000ms": "User loses attention span",
}

# With streaming:
# - TTFT determines when user's wait ends
# - Generation after first token feels like progress
# - Low TTFT = responsive system
```

### Business Impact

```python
business_metrics = {
    "engagement": "Users stay longer with fast responses",
    "satisfaction": "Higher CSAT with lower TTFT",
    "conversion": "Faster feels more reliable",
    "abandonment": "High TTFT increases early exits",
}

# Example: Chat application
# TTFT 500ms: "Wow, this is fast!"
# TTFT 3000ms: "Is it working?"
# TTFT 5000ms: *user leaves*
```

---

## Factors Affecting TTFT

### Model Size and Complexity

```python
# Larger models = Higher TTFT
# More parameters to process initial prompt

ttft_by_model = {
    "gpt-3.5-turbo": "~200-400ms",
    "gpt-4": "~400-800ms",
    "gpt-4-turbo": "~500-1000ms",
    "claude-3-haiku": "~200-400ms",
    "claude-3-sonnet": "~400-800ms",
    "claude-3-opus": "~800-2000ms",
}
```

### Input Length

```python
import time

def measure_ttft_by_input_length():
    """TTFT increases with prompt length"""
    
    results = []
    
    for word_count in [10, 100, 500, 1000, 5000]:
        # Create prompt of specified length
        prompt = " ".join(["word"] * word_count)
        
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=10
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                ttft = (time.time() - start) * 1000
                break
        
        results.append({
            "input_words": word_count,
            "ttft_ms": ttft
        })
    
    return results

# Typical results:
# 10 words: ~500ms
# 100 words: ~600ms
# 500 words: ~800ms
# 1000 words: ~1200ms
# 5000 words: ~2500ms
```

### Server Load and Infrastructure

```python
factors_affecting_ttft = {
    "queue_time": "Request waiting in server queue",
    "processing_time": "Initial prompt processing",
    "model_loading": "Cold start for serverless",
    "network_latency": "Distance to API server",
    "time_of_day": "Peak hours = higher load",
    "token_count": "Longer input = more processing",
}

# Peak hours (US business hours) often have higher TTFT
# Consider this in SLA design
```

---

## Optimization Strategies

### 1. Choose the Right Model

```python
def select_model_for_latency(task_type: str) -> str:
    """
    Select model based on latency requirements.
    """
    
    if task_type in ["simple_qa", "classification"]:
        return "gpt-3.5-turbo"  # Fastest
    
    elif task_type in ["general_chat", "summarization"]:
        return "gpt-4o-mini"    # Good balance
    
    elif task_type in ["complex_reasoning", "code"]:
        return "gpt-4o"         # Capable, reasonable latency
    
    else:
        return "gpt-3.5-turbo"  # Default to fast
```

### 2. Minimize Input Size

```python
def optimize_prompt_for_ttft(prompt: str, max_chars: int = 2000) -> str:
    """
    Reduce prompt size without losing meaning.
    """
    
    # Remove excessive whitespace
    prompt = " ".join(prompt.split())
    
    # Truncate if too long
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "..."
    
    return prompt

# Example: Instead of full document, send summary
def prepare_context(document: str) -> str:
    if len(document) > 3000:
        # First paragraph + key sentences
        return extract_key_content(document, max_tokens=500)
    return document
```

### 3. Use Faster Regions

```python
from openai import OpenAI

# Choose API endpoint closer to users
client_us = OpenAI(
    base_url="https://api.openai.com/v1"  # Default
)

client_eu = OpenAI(
    base_url="https://eu.api.openai.com/v1"  # EU endpoint (if available)
)

# Azure OpenAI allows region selection
azure_client = AzureOpenAI(
    azure_endpoint="https://westeurope.openai.azure.com"
)
```

### 4. Caching and Pre-computation

```python
from functools import lru_cache
import hashlib

# Cache system prompts and common prefixes
@lru_cache(maxsize=100)
def get_cached_prefix(prompt_hash: str):
    # Pre-compute common prompts
    pass

# For very low latency: Use streaming with prepared responses
prepared_responses = {
    "greeting": "Hello! How can I help you today?",
    "acknowledgment": "I understand. Let me help you with that.",
}
```

### 5. Show Progress During TTFT

```python
import asyncio

async def ttft_with_indicator(messages: list):
    """Show thinking indicator while waiting for first token"""
    
    thinking_task = asyncio.create_task(show_thinking_dots())
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    first_token = True
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token:
                thinking_task.cancel()  # Stop dots
                print("\r" + " " * 20 + "\r", end="")  # Clear line
                first_token = False
            
            print(chunk.choices[0].delta.content, end="", flush=True)

async def show_thinking_dots():
    """Display thinking animation"""
    dots = [".", "..", "...", ""]
    i = 0
    while True:
        print(f"\rThinking{dots[i % 4]}", end="", flush=True)
        await asyncio.sleep(0.3)
        i += 1
```

---

## Provider Comparisons

### Benchmark Framework

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class TTFTResult:
    provider: str
    model: str
    ttft_ms: float
    total_ms: float

def benchmark_providers(prompt: str) -> List[TTFTResult]:
    """Benchmark TTFT across providers"""
    
    results = []
    
    # OpenAI
    start = time.time()
    # ... measure TTFT
    results.append(TTFTResult("OpenAI", "gpt-4", ttft, total))
    
    # Anthropic
    # ... measure TTFT
    results.append(TTFTResult("Anthropic", "claude-3-sonnet", ttft, total))
    
    # Sort by TTFT
    results.sort(key=lambda x: x.ttft_ms)
    
    return results
```

### Typical Ranges (2024-2025)

| Provider | Model | Typical TTFT |
|----------|-------|--------------|
| OpenAI | GPT-3.5 Turbo | 200-500ms |
| OpenAI | GPT-4 | 400-800ms |
| OpenAI | GPT-4 Turbo | 500-1200ms |
| Anthropic | Claude 3 Haiku | 200-400ms |
| Anthropic | Claude 3 Sonnet | 400-800ms |
| Anthropic | Claude 3 Opus | 800-2000ms |
| Google | Gemini 1.5 Flash | 200-500ms |
| Google | Gemini 1.5 Pro | 400-1000ms |

> **Note:** These are approximate ranges. Actual TTFT varies with load, input size, and network conditions.

---

## Hands-on Exercise

### Your Task

Build a TTFT monitoring dashboard:

```python
import time
from openai import OpenAI
from collections import defaultdict
import statistics

client = OpenAI()

class TTFTMonitor:
    """Monitor TTFT over time"""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def measure(self, model: str, messages: list) -> dict:
        """Measure and record TTFT"""
        
        start = time.time()
        first_token_time = None
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=50
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                    break
        
        ttft_ms = (first_token_time - start) * 1000
        self.history[model].append(ttft_ms)
        
        return {
            "model": model,
            "ttft_ms": ttft_ms,
            "avg_ttft": statistics.mean(self.history[model]),
            "p50": statistics.median(self.history[model]),
            "p95": self._percentile(self.history[model], 95),
            "samples": len(self.history[model])
        }
    
    def _percentile(self, data: list, p: int) -> float:
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def report(self) -> str:
        """Generate TTFT report"""
        lines = ["TTFT Report", "=" * 50]
        
        for model, times in self.history.items():
            lines.append(f"\n{model}:")
            lines.append(f"  Samples: {len(times)}")
            lines.append(f"  Average: {statistics.mean(times):.0f}ms")
            lines.append(f"  P50: {statistics.median(times):.0f}ms")
            lines.append(f"  P95: {self._percentile(times, 95):.0f}ms")
            lines.append(f"  Min: {min(times):.0f}ms")
            lines.append(f"  Max: {max(times):.0f}ms")
        
        return "\n".join(lines)

# Run measurements
monitor = TTFTMonitor()

for i in range(10):
    result = monitor.measure(
        "gpt-4",
        [{"role": "user", "content": f"Hello {i}!"}]
    )
    print(f"Run {i+1}: {result['ttft_ms']:.0f}ms")

print("\n" + monitor.report())
```

---

## Summary

✅ **TTFT** = Time to First Token, crucial for perceived speed

✅ **Typical ranges**: 200ms (fast models) to 2000ms (large models)

✅ **Factors**: Model size, input length, server load, network

✅ **Optimize**: Right model, smaller prompts, closer regions

✅ **Show progress** during TTFT to manage expectations

✅ **Monitor** TTFT in production for SLA compliance

**Next:** [Full Response Latency](./06-full-response-latency.md)

---

## Further Reading

- [OpenAI Latency Guide](https://platform.openai.com/docs/guides/latency-optimization) — Official optimization tips
- [Anthropic Latency](https://docs.anthropic.com/claude/docs/reducing-latency) — Claude-specific guidance

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Streaming Trade-offs](./04-streaming-tradeoffs.md) | [Streaming Response Modes](./00-streaming-response-modes.md) | [Full Response Latency](./06-full-response-latency.md) |

