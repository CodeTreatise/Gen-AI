---
title: "OpenAI Caching"
---

# OpenAI Caching

## Introduction

OpenAI provides automatic prompt caching for all models with prompts over 1024 tokens. This lesson covers OpenAI-specific configuration options, retention settings, and optimization strategies.

### What We'll Cover

- Automatic caching behavior
- `prompt_cache_key` for routing
- Retention options (in-memory vs 24h)
- Extended caching configuration
- Responses API caching features

### Prerequisites

- Understanding of cache fundamentals
- OpenAI API access
- Python development environment

---

## Automatic Caching

### Default Behavior

```python
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional

client = OpenAI()

# Automatic caching for prompts >= 1024 tokens
# No configuration needed

large_context = "Your detailed context here... " * 500  # ~2000+ tokens

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": large_context},
        {"role": "user", "content": "Summarize the context"}
    ]
)

# Check cached tokens in usage
usage = response.usage
print(f"Input tokens: {usage.prompt_tokens}")

# Cached tokens available in details
if hasattr(usage, 'prompt_tokens_details'):
    details = usage.prompt_tokens_details
    cached = getattr(details, 'cached_tokens', 0)
    print(f"Cached tokens: {cached}")
```

### Caching Requirements

```python
@dataclass
class CachingRequirements:
    min_tokens: int
    model_support: list
    automatic: bool
    notes: str


OPENAI_CACHING = CachingRequirements(
    min_tokens=1024,
    model_support=[
        "gpt-4o", "gpt-4o-mini",
        "gpt-4-turbo", "gpt-4",
        "o1", "o1-mini", "o1-preview"
    ],
    automatic=True,
    notes="Caching is automatic, no opt-in required"
)


def check_caching_eligibility(
    token_count: int,
    model: str
) -> dict:
    """Check if request is eligible for caching."""
    
    model_base = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
    
    eligible = (
        token_count >= OPENAI_CACHING.min_tokens and
        any(model.startswith(m.split("-")[0]) for m in OPENAI_CACHING.model_support)
    )
    
    return {
        "eligible": eligible,
        "token_count": token_count,
        "min_required": OPENAI_CACHING.min_tokens,
        "model": model,
        "reason": "Meets requirements" if eligible else 
                  f"Need >= {OPENAI_CACHING.min_tokens} tokens" if token_count < OPENAI_CACHING.min_tokens
                  else "Model not supported"
    }


# Check
result = check_caching_eligibility(2000, "gpt-4o")
print(f"Eligible: {result['eligible']}")
print(f"Reason: {result['reason']}")
```

---

## Cache Key Configuration

### Using prompt_cache_key

```python
# Responses API supports explicit cache key
response = client.responses.create(
    model="gpt-4o",
    input="Your query here",
    instructions="Large system prompt...",
    prompt_cache_key="my_app_v2"  # Custom cache key
)

# Same cache key = share cache across requests
# Different users with same app version share cache
```

### Cache Key Strategies

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheKeyStrategy:
    name: str
    key_template: str
    use_case: str


STRATEGIES = [
    CacheKeyStrategy(
        name="version_based",
        key_template="app_v{version}",
        use_case="Share cache across users for same app version"
    ),
    CacheKeyStrategy(
        name="feature_based",
        key_template="feature_{feature_name}_v{version}",
        use_case="Separate cache per feature"
    ),
    CacheKeyStrategy(
        name="tenant_based",
        key_template="tenant_{tenant_id}",
        use_case="Isolate cache per tenant"
    ),
    CacheKeyStrategy(
        name="combined",
        key_template="app_v{version}_{feature}",
        use_case="Balance sharing and isolation"
    ),
]


class CacheKeyManager:
    """Manage cache key generation."""
    
    def __init__(self, app_version: str):
        self.app_version = app_version
    
    def for_version(self) -> str:
        """Global cache key for app version."""
        return f"app_v{self.app_version}"
    
    def for_feature(self, feature: str) -> str:
        """Feature-specific cache key."""
        return f"feature_{feature}_v{self.app_version}"
    
    def for_tenant(self, tenant_id: str) -> str:
        """Tenant-isolated cache key."""
        return f"tenant_{tenant_id}"
    
    def combined(self, feature: str, variant: Optional[str] = None) -> str:
        """Combined cache key."""
        key = f"app_v{self.app_version}_{feature}"
        if variant:
            key += f"_{variant}"
        return key


# Usage
keys = CacheKeyManager("2.0")

print(f"Version key: {keys.for_version()}")
print(f"Feature key: {keys.for_feature('chatbot')}")
print(f"Tenant key: {keys.for_tenant('acme_corp')}")
print(f"Combined: {keys.combined('chatbot', 'premium')}")
```

---

## Retention Options

### In-Memory Caching (Default)

```python
from enum import Enum
from dataclasses import dataclass

class CacheRetention(Enum):
    IN_MEMORY = "in_memory"  # 5-10 minutes
    EXTENDED = "24h"          # 24 hours


@dataclass
class RetentionConfig:
    retention: CacheRetention
    ttl_minutes: int
    cost_multiplier: float
    use_case: str


RETENTION_OPTIONS = {
    CacheRetention.IN_MEMORY: RetentionConfig(
        retention=CacheRetention.IN_MEMORY,
        ttl_minutes=10,
        cost_multiplier=1.0,
        use_case="Bursty traffic, short sessions"
    ),
    CacheRetention.EXTENDED: RetentionConfig(
        retention=CacheRetention.EXTENDED,
        ttl_minutes=60 * 24,
        cost_multiplier=1.0,  # No additional cost
        use_case="Consistent traffic, long-running apps"
    ),
}


def recommend_retention(
    requests_per_minute: float,
    session_duration_hours: float,
    traffic_pattern: str  # "bursty" or "steady"
) -> CacheRetention:
    """Recommend cache retention based on usage pattern."""
    
    # Extended caching benefits steady traffic
    if traffic_pattern == "steady" and requests_per_minute >= 5:
        return CacheRetention.EXTENDED
    
    # Long sessions benefit from extended
    if session_duration_hours >= 2:
        return CacheRetention.EXTENDED
    
    # Default to in-memory for bursty/short
    return CacheRetention.IN_MEMORY


# Example
retention = recommend_retention(
    requests_per_minute=10,
    session_duration_hours=8,
    traffic_pattern="steady"
)

config = RETENTION_OPTIONS[retention]
print(f"Recommended: {config.retention.value}")
print(f"TTL: {config.ttl_minutes} minutes")
print(f"Use case: {config.use_case}")
```

### Extended Caching

```python
# Configure extended caching in request
response = client.responses.create(
    model="gpt-4o",
    input="Your query",
    instructions="Large context...",
    prompt_cache_key="persistent_app",
    prompt_cache_retention="24h"  # Extended retention
)


# Verify extended caching is active
def check_extended_caching(response) -> dict:
    """Check if extended caching is being used."""
    
    usage = response.usage
    
    # Get cached tokens
    input_details = getattr(usage, 'input_tokens_details', None)
    cached = input_details.cached_tokens if input_details else 0
    
    return {
        "input_tokens": usage.input_tokens,
        "cached_tokens": cached,
        "cache_hit_rate": cached / usage.input_tokens if usage.input_tokens > 0 else 0,
        "using_cache": cached > 0
    }
```

---

## Model-Specific Caching

### Caching by Model

```python
from typing import Dict

MODEL_CACHE_SUPPORT = {
    "gpt-4o": {
        "min_tokens": 1024,
        "caching": True,
        "extended": True,
        "discount": 0.50  # 50% cost reduction
    },
    "gpt-4o-mini": {
        "min_tokens": 1024,
        "caching": True,
        "extended": True,
        "discount": 0.50
    },
    "gpt-4-turbo": {
        "min_tokens": 1024,
        "caching": True,
        "extended": True,
        "discount": 0.50
    },
    "o1": {
        "min_tokens": 1024,
        "caching": True,
        "extended": True,
        "discount": 0.50
    },
    "o1-mini": {
        "min_tokens": 1024,
        "caching": True,
        "extended": True,
        "discount": 0.50
    },
}


def get_cache_discount(model: str, cached_tokens: int) -> float:
    """Calculate cost savings from cached tokens."""
    
    config = MODEL_CACHE_SUPPORT.get(model, {})
    discount = config.get("discount", 0)
    
    return cached_tokens * discount


def estimate_request_cost(
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int
) -> dict:
    """Estimate request cost with caching."""
    
    # Example rates (per 1M tokens)
    RATES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
    }
    
    rates = RATES.get(model, {"input": 2.50, "output": 10.00})
    config = MODEL_CACHE_SUPPORT.get(model, {"discount": 0.5})
    
    # Calculate costs
    uncached_tokens = input_tokens - cached_tokens
    
    uncached_cost = (uncached_tokens / 1_000_000) * rates["input"]
    cached_cost = (cached_tokens / 1_000_000) * rates["input"] * (1 - config["discount"])
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    
    total = uncached_cost + cached_cost + output_cost
    full_cost = (input_tokens / 1_000_000) * rates["input"] + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "uncached_cost": uncached_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total,
        "without_cache": full_cost,
        "savings": full_cost - total,
        "savings_percent": (full_cost - total) / full_cost * 100 if full_cost > 0 else 0
    }


# Example
cost = estimate_request_cost(
    model="gpt-4o",
    input_tokens=5000,
    cached_tokens=4000,  # 80% cache hit
    output_tokens=500
)

print(f"Total cost: ${cost['total_cost']:.4f}")
print(f"Without cache: ${cost['without_cache']:.4f}")
print(f"Savings: ${cost['savings']:.4f} ({cost['savings_percent']:.1f}%)")
```

---

## Responses API Caching

### Enhanced Caching Features

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ResponsesCacheConfig:
    cache_key: Optional[str]
    retention: str  # "in_memory" or "24h"
    store: bool  # Store response for chaining


def create_cached_request(
    client,
    input_text: str,
    instructions: str,
    config: ResponsesCacheConfig
):
    """Create request with caching configuration."""
    
    kwargs = {
        "model": "gpt-4o",
        "input": input_text,
        "instructions": instructions,
        "store": config.store
    }
    
    if config.cache_key:
        kwargs["prompt_cache_key"] = config.cache_key
    
    if config.retention:
        kwargs["prompt_cache_retention"] = config.retention
    
    return client.responses.create(**kwargs)


# Usage
config = ResponsesCacheConfig(
    cache_key="chatbot_v2",
    retention="24h",
    store=True
)

response = create_cached_request(
    client,
    input_text="What is Python?",
    instructions="You are a helpful programming tutor.",
    config=config
)
```

### Chaining with Caching

```python
class CachedConversation:
    """Conversation with optimized caching."""
    
    def __init__(
        self,
        client,
        instructions: str,
        cache_key: str
    ):
        self.client = client
        self.instructions = instructions
        self.cache_key = cache_key
        self.last_response_id: Optional[str] = None
        self.cache_stats = {"hits": 0, "misses": 0, "total_cached": 0}
    
    def send(self, message: str) -> str:
        """Send message with caching."""
        
        kwargs = {
            "model": "gpt-4o",
            "input": message,
            "instructions": self.instructions,
            "prompt_cache_key": self.cache_key,
            "prompt_cache_retention": "24h"
        }
        
        if self.last_response_id:
            kwargs["previous_response_id"] = self.last_response_id
        
        response = self.client.responses.create(**kwargs)
        self.last_response_id = response.id
        
        # Track cache stats
        self._update_stats(response)
        
        return response.output_text
    
    def _update_stats(self, response):
        """Update cache statistics."""
        
        usage = response.usage
        details = getattr(usage, 'input_tokens_details', None)
        
        if details:
            cached = getattr(details, 'cached_tokens', 0)
            self.cache_stats["total_cached"] += cached
            
            if cached > 0:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": f"{hit_rate:.1%}",
            "total_requests": total
        }


# Usage
convo = CachedConversation(
    client=client,
    instructions="You are a Python expert. Provide detailed explanations.",
    cache_key="python_tutor_v1"
)

# Conversation
convo.send("What is a decorator?")
convo.send("Show me an example")
convo.send("How do I use functools.wraps?")

print("Cache Stats:", convo.get_stats())
```

---

## Hands-on Exercise

### Your Task

Build a cache-optimized client with monitoring.

### Requirements

1. Configure cache key and retention
2. Track cache hit rates per session
3. Calculate cost savings
4. Provide optimization recommendations

<details>
<summary>💡 Hints</summary>

- Use input_tokens_details for cached tokens
- Track stats per cache key
- Calculate savings based on model rates
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum

class CacheRetention(Enum):
    IN_MEMORY = "in_memory"
    EXTENDED = "24h"


@dataclass
class RequestStats:
    timestamp: datetime
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class CacheKeyStats:
    cache_key: str
    requests: List[RequestStats] = field(default_factory=list)
    
    @property
    def total_requests(self) -> int:
        return len(self.requests)
    
    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.requests)
    
    @property
    def total_cached_tokens(self) -> int:
        return sum(r.cached_tokens for r in self.requests)
    
    @property
    def hit_rate(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_input_tokens
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.requests:
            return 0.0
        return sum(r.latency_ms for r in self.requests) / len(self.requests)


@dataclass
class CacheConfig:
    cache_key: str
    retention: CacheRetention
    model: str = "gpt-4o"


class OptimizedCacheClient:
    """OpenAI client with cache optimization and monitoring."""
    
    # Cost rates per 1M tokens
    RATES = {
        "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
        "o1": {"input": 15.00, "cached": 7.50, "output": 60.00},
    }
    
    def __init__(self, client):
        self.client = client
        self.stats: Dict[str, CacheKeyStats] = {}
    
    def create(
        self,
        input_text: str,
        instructions: str,
        config: CacheConfig,
        **kwargs
    ):
        """Create response with cache monitoring."""
        
        import time
        start = time.time()
        
        request_kwargs = {
            "model": config.model,
            "input": input_text,
            "instructions": instructions,
            "prompt_cache_key": config.cache_key,
            "prompt_cache_retention": config.retention.value,
            **kwargs
        }
        
        response = self.client.responses.create(**request_kwargs)
        
        latency_ms = (time.time() - start) * 1000
        
        # Track stats
        self._track_request(config.cache_key, response, latency_ms)
        
        return response
    
    def _track_request(self, cache_key: str, response, latency_ms: float):
        """Track request statistics."""
        
        if cache_key not in self.stats:
            self.stats[cache_key] = CacheKeyStats(cache_key=cache_key)
        
        usage = response.usage
        details = getattr(usage, 'input_tokens_details', None)
        cached = details.cached_tokens if details else 0
        
        self.stats[cache_key].requests.append(RequestStats(
            timestamp=datetime.now(),
            input_tokens=usage.input_tokens,
            cached_tokens=cached,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms
        ))
    
    def get_stats(self, cache_key: Optional[str] = None) -> dict:
        """Get statistics for a cache key or all keys."""
        
        if cache_key:
            if cache_key not in self.stats:
                return {"error": f"No stats for {cache_key}"}
            return self._format_stats(self.stats[cache_key])
        
        return {
            key: self._format_stats(stats)
            for key, stats in self.stats.items()
        }
    
    def _format_stats(self, stats: CacheKeyStats) -> dict:
        return {
            "cache_key": stats.cache_key,
            "total_requests": stats.total_requests,
            "total_input_tokens": stats.total_input_tokens,
            "total_cached_tokens": stats.total_cached_tokens,
            "hit_rate": f"{stats.hit_rate:.1%}",
            "avg_latency_ms": f"{stats.avg_latency_ms:.1f}"
        }
    
    def calculate_savings(
        self,
        cache_key: str,
        model: str = "gpt-4o"
    ) -> dict:
        """Calculate cost savings for a cache key."""
        
        if cache_key not in self.stats:
            return {"error": f"No stats for {cache_key}"}
        
        stats = self.stats[cache_key]
        rates = self.RATES.get(model, self.RATES["gpt-4o"])
        
        # Calculate costs
        uncached = stats.total_input_tokens - stats.total_cached_tokens
        
        actual_cost = (
            (uncached / 1_000_000) * rates["input"] +
            (stats.total_cached_tokens / 1_000_000) * rates["cached"] +
            sum(r.output_tokens for r in stats.requests) / 1_000_000 * rates["output"]
        )
        
        full_cost = (
            (stats.total_input_tokens / 1_000_000) * rates["input"] +
            sum(r.output_tokens for r in stats.requests) / 1_000_000 * rates["output"]
        )
        
        savings = full_cost - actual_cost
        
        return {
            "cache_key": cache_key,
            "model": model,
            "actual_cost": f"${actual_cost:.4f}",
            "without_cache": f"${full_cost:.4f}",
            "savings": f"${savings:.4f}",
            "savings_percent": f"{(savings / full_cost * 100):.1f}%" if full_cost > 0 else "0%"
        }
    
    def get_recommendations(self, cache_key: str) -> List[str]:
        """Get optimization recommendations."""
        
        recommendations = []
        
        if cache_key not in self.stats:
            return ["No data available for recommendations"]
        
        stats = self.stats[cache_key]
        
        # Hit rate recommendations
        if stats.hit_rate < 0.3:
            recommendations.append(
                "Low cache hit rate (<30%). Consider restructuring prompts "
                "with static content first."
            )
        elif stats.hit_rate < 0.6:
            recommendations.append(
                "Moderate cache hit rate. Review if dynamic content can be "
                "moved to end of prompts."
            )
        else:
            recommendations.append(
                f"Good cache hit rate ({stats.hit_rate:.0%}). Current structure is effective."
            )
        
        # Volume recommendations
        if stats.total_requests < 10:
            recommendations.append(
                "Low request volume. Cache benefits increase with more requests."
            )
        
        # Latency recommendations
        if stats.avg_latency_ms > 2000:
            recommendations.append(
                "High average latency. Consider using extended caching (24h) "
                "for better cache availability."
            )
        
        return recommendations


# Usage
optimized = OptimizedCacheClient(client)

config = CacheConfig(
    cache_key="demo_app_v1",
    retention=CacheRetention.EXTENDED,
    model="gpt-4o"
)

# Make requests
large_instructions = "You are an expert Python tutor. " * 100  # Large system prompt

for query in ["What is a list?", "How do I append?", "Show sorting"]:
    response = optimized.create(
        input_text=query,
        instructions=large_instructions,
        config=config
    )
    print(f"Query: {query[:20]}... -> {len(response.output_text)} chars")

# View stats
print("\nCache Statistics:")
print(optimized.get_stats("demo_app_v1"))

print("\nCost Savings:")
print(optimized.calculate_savings("demo_app_v1", "gpt-4o"))

print("\nRecommendations:")
for rec in optimized.get_recommendations("demo_app_v1"):
    print(f"  • {rec}")
```

</details>

---

## Summary

✅ OpenAI caching is automatic for prompts ≥1024 tokens  
✅ `prompt_cache_key` enables cache routing optimization  
✅ Extended caching (24h) available for sustained workloads  
✅ Responses API provides enhanced caching features  
✅ 50% cost reduction on cached tokens

**Next:** [Anthropic Caching](./03-anthropic-caching.md)

---

## Further Reading

- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — Official guide
- [Responses API](https://platform.openai.com/docs/api-reference/responses) — API reference
- [Pricing](https://openai.com/pricing) — Current pricing
