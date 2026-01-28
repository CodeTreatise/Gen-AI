---
title: "Latency Considerations"
---

# Latency Considerations

## Introduction

Latency—the time between sending a request and receiving a response—directly impacts user experience. Understanding latency factors helps you select models that meet your performance requirements.

### What We'll Cover

- Real-time requirements
- Model size vs latency
- Geographic considerations
- Edge deployment options

---

## Understanding Latency

### Latency Components

```
Total Latency = Network + Queue + TTFT + Generation

┌─────────────────────────────────────────────────────────────┐
│                      LATENCY BREAKDOWN                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Network]     [Queue]      [TTFT]      [Generation]        │
│   20-200ms    0-1000ms+   100-500ms    10-50ms/token        │
│                                                              │
│  Your network  Provider     First       Token-by-token      │
│  to API        wait time    token       generation          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Typical Latencies

| Model | TTFT | Tokens/sec | 100 tokens |
|-------|------|-----------|------------|
| GPT-4o-mini | ~200ms | ~100 | ~1.2s |
| GPT-4o | ~400ms | ~50 | ~2.4s |
| Claude Haiku | ~200ms | ~100 | ~1.2s |
| Claude Sonnet | ~500ms | ~60 | ~2.2s |
| Gemini Flash | ~150ms | ~150 | ~0.8s |
| o1-preview | ~2-10s | ~30 | ~5-15s |

---

## Real-Time Requirements

### Latency Tolerance by Use Case

```python
latency_requirements = {
    "voice_conversation": {
        "max_ttft_ms": 300,
        "max_total_ms": 2000,
        "models": ["gpt-4o-realtime", "gemini-flash"],
        "notes": "Users expect natural conversation flow"
    },
    "chat_interface": {
        "max_ttft_ms": 1000,
        "max_total_ms": 5000,
        "models": ["gpt-4o-mini", "claude-haiku", "gemini-flash"],
        "notes": "Streaming makes longer generation acceptable"
    },
    "autocomplete": {
        "max_ttft_ms": 100,
        "max_total_ms": 300,
        "models": ["gpt-4o-mini", "specialized completion models"],
        "notes": "Must feel instant"
    },
    "code_completion": {
        "max_ttft_ms": 200,
        "max_total_ms": 500,
        "models": ["codestral", "gpt-4o-mini"],
        "notes": "Inline suggestions need speed"
    },
    "document_analysis": {
        "max_ttft_ms": 3000,
        "max_total_ms": 30000,
        "models": ["gpt-4o", "claude-sonnet", "gemini-pro"],
        "notes": "Users expect longer processing"
    },
    "batch_processing": {
        "max_ttft_ms": "N/A",
        "max_total_ms": "hours",
        "models": ["Any - optimize for cost"],
        "notes": "Not user-facing"
    }
}
```

### Measuring Latency

```python
import time
from openai import OpenAI

client = OpenAI()

def measure_latency(prompt: str, model: str) -> dict:
    """Measure detailed latency metrics"""
    
    start_time = time.time()
    first_token_time = None
    tokens_received = 0
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            tokens_received += 1
    
    end_time = time.time()
    
    ttft = (first_token_time - start_time) * 1000 if first_token_time else None
    total = (end_time - start_time) * 1000
    generation_time = (end_time - first_token_time) * 1000 if first_token_time else None
    
    return {
        "ttft_ms": round(ttft, 2) if ttft else None,
        "total_ms": round(total, 2),
        "generation_ms": round(generation_time, 2) if generation_time else None,
        "tokens": tokens_received,
        "tokens_per_second": round(tokens_received / (generation_time / 1000), 1) if generation_time else None
    }

# Benchmark
metrics = measure_latency("Write a haiku about coding", "gpt-4o-mini")
print(f"TTFT: {metrics['ttft_ms']}ms")
print(f"Total: {metrics['total_ms']}ms")
print(f"Speed: {metrics['tokens_per_second']} tokens/sec")
```

---

## Model Size vs Latency

### The Trade-off

```python
model_latency_comparison = {
    "small_models": {
        "examples": ["gpt-4o-mini", "claude-haiku", "gemini-flash"],
        "latency": "Low (100-300ms TTFT)",
        "quality": "Good for most tasks",
        "use_when": "Speed is priority"
    },
    "medium_models": {
        "examples": ["gpt-4o", "claude-sonnet", "gemini-pro"],
        "latency": "Medium (300-800ms TTFT)",
        "quality": "High quality",
        "use_when": "Balanced needs"
    },
    "large_models": {
        "examples": ["o1", "claude-opus"],
        "latency": "High (1-10s+ TTFT)",
        "quality": "Maximum capability",
        "use_when": "Quality over speed"
    }
}
```

### Optimizing for Speed

```python
class LatencyOptimizer:
    """Strategies for reducing latency"""
    
    @staticmethod
    def use_streaming(client, prompt: str, model: str):
        """Stream to improve perceived latency"""
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True  # Returns tokens as generated
        )
    
    @staticmethod
    def limit_output(client, prompt: str, model: str, max_tokens: int = 100):
        """Limit output length for faster completion"""
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
    
    @staticmethod
    def simplify_prompt(prompt: str) -> str:
        """Shorter prompts process faster"""
        # Remove unnecessary context
        # Be direct and specific
        return prompt.strip()
    
    @staticmethod
    def use_caching(prompt: str, cache: dict):
        """Cache common responses"""
        cache_key = hash(prompt)
        if cache_key in cache:
            return cache[cache_key]
        return None
```

---

## Geographic Considerations

### Regional Endpoints

```python
regional_endpoints = {
    "openai": {
        "default": "api.openai.com",
        "azure_regions": [
            "eastus", "westus", "westeurope", 
            "eastasia", "australiaeast"
        ]
    },
    "anthropic": {
        "default": "api.anthropic.com",
        "vertex_regions": ["us-central1", "europe-west4"]
    },
    "google": {
        "vertex_regions": [
            "us-central1", "us-east4", "europe-west4",
            "asia-northeast1", "australia-southeast1"
        ]
    }
}

def select_region(user_location: str) -> dict:
    """Select closest region for lowest latency"""
    
    region_map = {
        "north_america": {
            "azure": "eastus",
            "vertex": "us-central1"
        },
        "europe": {
            "azure": "westeurope",
            "vertex": "europe-west4"
        },
        "asia": {
            "azure": "eastasia",
            "vertex": "asia-northeast1"
        },
        "australia": {
            "azure": "australiaeast",
            "vertex": "australia-southeast1"
        }
    }
    
    return region_map.get(user_location, region_map["north_america"])
```

### Network Optimization

```python
import aiohttp
import asyncio

class OptimizedClient:
    """Client with connection pooling and keep-alive"""
    
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                keepalive_timeout=30
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    async def call_api(self, endpoint: str, data: dict):
        session = await self.get_session()
        async with session.post(endpoint, json=data) as response:
            return await response.json()
```

---

## Edge Deployment

### When to Consider Edge

```python
edge_deployment_criteria = {
    "latency_critical": "Need <50ms response time",
    "offline_required": "Must work without internet",
    "data_privacy": "Data cannot leave device/region",
    "high_volume": "Cost prohibitive at scale with APIs",
    "specialized": "Narrow task with small model sufficient"
}
```

### Edge Options

| Approach | Latency | Cost | Complexity |
|----------|---------|------|------------|
| Cloud API | 200-2000ms | Pay per use | Low |
| Regional API | 100-500ms | Pay per use | Low |
| Self-hosted cloud | 50-200ms | Server costs | Medium |
| Edge server | 20-100ms | Hardware + ops | High |
| On-device | 10-50ms | Device constraints | High |

### Edge Model Selection

```python
edge_models = {
    "mobile_devices": {
        "models": ["phi-3-mini", "gemma-2b", "llama-3-8b-quantized"],
        "considerations": "Memory, battery, thermal"
    },
    "edge_servers": {
        "models": ["llama-3-70b", "mistral-7b", "qwen-14b"],
        "considerations": "GPU availability, cooling"
    },
    "browser": {
        "models": ["gemma-2b-web", "phi-3-mini-web"],
        "considerations": "WASM, WebGPU support"
    }
}

def select_edge_model(
    device_type: str,
    available_memory_gb: float,
    gpu_available: bool
) -> list:
    """Select appropriate edge model"""
    
    suitable = []
    
    if device_type == "mobile":
        if available_memory_gb >= 4:
            suitable.append("phi-3-mini")
        if available_memory_gb >= 8:
            suitable.append("llama-3-8b-q4")
    
    elif device_type == "edge_server":
        if gpu_available:
            if available_memory_gb >= 80:
                suitable.append("llama-3-70b")
            if available_memory_gb >= 40:
                suitable.append("mistral-22b")
        suitable.append("llama-3-8b")
    
    return suitable
```

---

## Latency Budgeting

### Example Budget

```python
def create_latency_budget(target_total_ms: int) -> dict:
    """Create latency budget for request"""
    
    return {
        "network_budget": int(target_total_ms * 0.1),    # 10%
        "queue_budget": int(target_total_ms * 0.1),      # 10%
        "ttft_budget": int(target_total_ms * 0.2),       # 20%
        "generation_budget": int(target_total_ms * 0.6), # 60%
        "total": target_total_ms
    }

# For 2-second target
budget = create_latency_budget(2000)
# network: 200ms, queue: 200ms, ttft: 400ms, generation: 1200ms
```

### Monitoring and Alerting

```python
class LatencyMonitor:
    """Monitor and alert on latency"""
    
    def __init__(self, p50_threshold: int, p99_threshold: int):
        self.p50_threshold = p50_threshold
        self.p99_threshold = p99_threshold
        self.latencies = []
    
    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
        
        if len(self.latencies) > 100:
            self._check_thresholds()
    
    def _check_thresholds(self):
        sorted_latencies = sorted(self.latencies[-100:])
        p50 = sorted_latencies[50]
        p99 = sorted_latencies[99]
        
        if p50 > self.p50_threshold:
            self._alert(f"P50 latency {p50}ms exceeds threshold {self.p50_threshold}ms")
        
        if p99 > self.p99_threshold:
            self._alert(f"P99 latency {p99}ms exceeds threshold {self.p99_threshold}ms")
    
    def _alert(self, message: str):
        print(f"⚠️ ALERT: {message}")
```

---

## Summary

✅ **Know your requirements** - Real-time vs batch

✅ **Measure actual latency** - TTFT + generation time

✅ **Use streaming** - Improves perceived speed

✅ **Consider geography** - Regional endpoints help

✅ **Edge when needed** - For extreme latency needs

**Next:** [API Availability & Reliability](./05-api-availability-reliability.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Context Window Requirements](./03-context-window-requirements.md) | [Model Selection](./00-model-selection-criteria.md) | [API Availability](./05-api-availability-reliability.md) |

