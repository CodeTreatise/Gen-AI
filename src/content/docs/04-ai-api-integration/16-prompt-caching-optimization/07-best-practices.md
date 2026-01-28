---
title: "Best Practices"
---

# Best Practices

## Introduction

This lesson consolidates best practices for prompt caching across OpenAI and Anthropic. Following these guidelines will help you maximize cache hit rates, minimize costs, and maintain consistent performance.

### What We'll Cover

- Request streaming patterns
- Rate limit considerations
- Cache warming strategies
- Prompt versioning
- Testing and validation
- Common pitfalls

### Prerequisites

- Understanding of caching fundamentals
- Experience with either OpenAI or Anthropic
- Production deployment experience

---

## Request Streaming Patterns

### Maintaining Cache Warmth

```python
from dataclasses import dataclass
from typing import Optional
import time
import threading

@dataclass
class StreamConfig:
    """Configuration for request streaming."""
    
    min_requests_per_minute: float
    cache_ttl_seconds: int
    warmup_interval_seconds: float
    
    @property
    def safe_interval(self) -> float:
        """Safe interval between requests to maintain cache."""
        # Stay well within TTL
        return min(
            self.cache_ttl_seconds * 0.5,
            60 / self.min_requests_per_minute if self.min_requests_per_minute > 0 else 60
        )


# Provider-specific configurations
STREAM_CONFIGS = {
    "openai_default": StreamConfig(
        min_requests_per_minute=6,  # ~1 per 10 seconds
        cache_ttl_seconds=300,       # 5 minutes
        warmup_interval_seconds=60
    ),
    "openai_extended": StreamConfig(
        min_requests_per_minute=0.5,  # ~1 per 2 minutes is fine
        cache_ttl_seconds=86400,      # 24 hours
        warmup_interval_seconds=3600
    ),
    "anthropic_default": StreamConfig(
        min_requests_per_minute=6,
        cache_ttl_seconds=300,
        warmup_interval_seconds=60
    ),
    "anthropic_extended": StreamConfig(
        min_requests_per_minute=1,
        cache_ttl_seconds=3600,
        warmup_interval_seconds=600
    ),
}


class CacheWarmthManager:
    """Manage cache warmth through regular requests."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.last_request_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def record_request(self):
        """Record a request was made."""
        with self._lock:
            self.last_request_time = time.time()
    
    def should_warmup(self) -> bool:
        """Check if a warmup request is needed."""
        with self._lock:
            if self.last_request_time is None:
                return True
            
            elapsed = time.time() - self.last_request_time
            return elapsed > self.config.safe_interval
    
    def time_until_warmup(self) -> float:
        """Seconds until warmup is needed."""
        with self._lock:
            if self.last_request_time is None:
                return 0
            
            elapsed = time.time() - self.last_request_time
            remaining = self.config.safe_interval - elapsed
            return max(0, remaining)


# Usage
manager = CacheWarmthManager(STREAM_CONFIGS["openai_default"])

# Check before making request
if manager.should_warmup():
    print("Cache may be cold, making warmup request")
    # make_request()
    
manager.record_request()
print(f"Next warmup in {manager.time_until_warmup():.0f} seconds")
```

### Traffic Smoothing

```python
from collections import deque
import asyncio

class RateSmoother:
    """Smooth request rate for consistent cache behavior."""
    
    def __init__(
        self,
        target_rpm: float,
        burst_tolerance: int = 3
    ):
        self.target_rpm = target_rpm
        self.burst_tolerance = burst_tolerance
        self.request_times: deque = deque(maxlen=100)
        self._interval = 60.0 / target_rpm
    
    async def wait_if_needed(self):
        """Wait if necessary to smooth traffic."""
        
        if not self.request_times:
            return
        
        # Check recent request rate
        now = time.time()
        recent = [t for t in self.request_times if now - t < 60]
        
        if len(recent) >= self.target_rpm:
            # At rate limit, calculate wait time
            oldest_in_window = min(recent)
            wait_time = 60 - (now - oldest_in_window)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    def record(self):
        """Record a request."""
        self.request_times.append(time.time())
    
    def current_rpm(self) -> float:
        """Calculate current requests per minute."""
        now = time.time()
        recent = [t for t in self.request_times if now - t < 60]
        return len(recent)


# Usage
smoother = RateSmoother(target_rpm=15)

async def make_smoothed_request():
    await smoother.wait_if_needed()
    # response = await make_request()
    smoother.record()
    # return response
```

---

## Rate Limit Considerations

### Cache Key Limits

```python
"""
OpenAI Rate Limit Guidance:
- Keep prefix-key combinations under 15 RPM for optimal caching
- Too many unique cache keys can fragment the cache
- Use consistent cache keys across similar requests
"""

from typing import Dict, Set
from collections import defaultdict

class CacheKeyRateLimiter:
    """Track and limit cache key usage."""
    
    MAX_RPM_PER_KEY = 15
    MAX_UNIQUE_KEYS = 50
    
    def __init__(self):
        self.key_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.active_keys: Set[str] = set()
    
    def check_key(self, cache_key: str) -> dict:
        """Check if a cache key can be used."""
        
        now = time.time()
        
        # Get recent requests for this key
        recent = [t for t in self.key_requests[cache_key] if now - t < 60]
        current_rpm = len(recent)
        
        # Check limits
        issues = []
        
        if current_rpm >= self.MAX_RPM_PER_KEY:
            issues.append(f"Key '{cache_key}' at {current_rpm} RPM (max {self.MAX_RPM_PER_KEY})")
        
        if cache_key not in self.active_keys and len(self.active_keys) >= self.MAX_UNIQUE_KEYS:
            issues.append(f"Too many unique keys ({len(self.active_keys)})")
        
        return {
            "allowed": len(issues) == 0,
            "current_rpm": current_rpm,
            "issues": issues,
            "recommendation": "Use existing key" if issues else "OK"
        }
    
    def record(self, cache_key: str):
        """Record a request for a cache key."""
        self.key_requests[cache_key].append(time.time())
        self.active_keys.add(cache_key)
    
    def get_stats(self) -> dict:
        """Get rate limit statistics."""
        
        now = time.time()
        
        key_stats = {}
        for key in self.active_keys:
            recent = [t for t in self.key_requests[key] if now - t < 60]
            key_stats[key] = len(recent)
        
        return {
            "active_keys": len(self.active_keys),
            "max_keys": self.MAX_UNIQUE_KEYS,
            "key_rpm": key_stats,
            "highest_rpm": max(key_stats.values()) if key_stats else 0
        }


# Usage
limiter = CacheKeyRateLimiter()

check = limiter.check_key("app_v1")
if check["allowed"]:
    # Make request
    limiter.record("app_v1")
else:
    print(f"Rate limited: {check['issues']}")
```

### Handling Rate Limit Errors

```python
from dataclasses import dataclass
from enum import Enum
import random

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Retry configuration."""
    
    strategy: RetryStrategy
    initial_delay_ms: int
    max_delay_ms: int
    max_retries: int
    jitter: bool = True


def calculate_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate delay for retry attempt."""
    
    if config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.initial_delay_ms * (2 ** attempt)
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.initial_delay_ms * (attempt + 1)
    else:
        delay = config.initial_delay_ms
    
    # Cap at max
    delay = min(delay, config.max_delay_ms)
    
    # Add jitter
    if config.jitter:
        delay *= random.uniform(0.5, 1.5)
    
    return delay / 1000  # Convert to seconds


async def with_retry(
    func,
    config: RetryConfig,
    on_retry=None
):
    """Execute function with retry logic."""
    
    last_error = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e
            
            # Check if retryable
            if "rate_limit" not in str(e).lower() and attempt == 0:
                raise
            
            if attempt < config.max_retries:
                delay = calculate_delay(config, attempt)
                
                if on_retry:
                    on_retry(attempt, delay, e)
                
                await asyncio.sleep(delay)
    
    raise last_error


# Usage
retry_config = RetryConfig(
    strategy=RetryStrategy.EXPONENTIAL,
    initial_delay_ms=1000,
    max_delay_ms=60000,
    max_retries=5,
    jitter=True
)

# async def make_request():
#     return await with_retry(
#         lambda: client.chat.completions.create(...),
#         retry_config
#     )
```

---

## Cache Warming Strategies

### Preemptive Warming

```python
from typing import List, Callable
import asyncio

@dataclass
class WarmupTarget:
    """Target for cache warming."""
    
    cache_key: str
    prompt_generator: Callable[[], str]
    priority: int = 1


class CacheWarmer:
    """Proactively warm caches."""
    
    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.targets: List[WarmupTarget] = []
        self.warmed: Set[str] = set()
    
    def add_target(self, target: WarmupTarget):
        """Add a warmup target."""
        self.targets.append(target)
        self.targets.sort(key=lambda t: t.priority)
    
    async def warmup_all(self, max_concurrent: int = 3):
        """Warm all targets."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def warm_one(target: WarmupTarget):
            async with semaphore:
                await self._warm_target(target)
        
        tasks = [warm_one(t) for t in self.targets if t.cache_key not in self.warmed]
        await asyncio.gather(*tasks)
    
    async def _warm_target(self, target: WarmupTarget):
        """Warm a single target."""
        
        try:
            # Generate minimal request to populate cache
            prompt = target.prompt_generator()
            
            await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Hello"}  # Minimal query
                ],
                max_tokens=1  # Minimal response
            )
            
            self.warmed.add(target.cache_key)
            print(f"Warmed cache: {target.cache_key}")
            
        except Exception as e:
            print(f"Failed to warm {target.cache_key}: {e}")
    
    def get_status(self) -> dict:
        """Get warmup status."""
        return {
            "total_targets": len(self.targets),
            "warmed": len(self.warmed),
            "pending": len(self.targets) - len(self.warmed),
            "warmed_keys": list(self.warmed)
        }


# Usage
# warmer = CacheWarmer(client)
# 
# warmer.add_target(WarmupTarget(
#     cache_key="chatbot_v1",
#     prompt_generator=lambda: "You are a helpful assistant..." * 100,
#     priority=1
# ))
# 
# await warmer.warmup_all()
```

### Scheduled Warming

```python
import asyncio
from datetime import datetime, timedelta

class ScheduledCacheWarmer:
    """Schedule regular cache warming."""
    
    def __init__(
        self,
        warmer: CacheWarmer,
        interval_minutes: int = 4  # Less than 5-minute TTL
    ):
        self.warmer = warmer
        self.interval = interval_minutes * 60
        self._running = False
        self._task = None
    
    async def start(self):
        """Start scheduled warming."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
    
    async def stop(self):
        """Stop scheduled warming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _run_loop(self):
        """Run warming loop."""
        while self._running:
            try:
                # Clear warmed set to re-warm
                self.warmer.warmed.clear()
                
                await self.warmer.warmup_all()
                
                print(f"Cache warming complete at {datetime.now()}")
                
            except Exception as e:
                print(f"Warming error: {e}")
            
            await asyncio.sleep(self.interval)


# Usage
# scheduler = ScheduledCacheWarmer(warmer, interval_minutes=4)
# await scheduler.start()
# ... run application ...
# await scheduler.stop()
```

---

## Prompt Versioning

### Version Management

```python
from dataclasses import dataclass
from typing import Dict, Optional
import hashlib

@dataclass
class PromptVersion:
    """Versioned prompt configuration."""
    
    version: str
    content: str
    created_at: datetime
    cache_key: str = ""
    
    def __post_init__(self):
        if not self.cache_key:
            self.cache_key = f"prompt_v{self.version}_{self._content_hash()}"
    
    def _content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:8]


class PromptVersionManager:
    """Manage prompt versions for cache consistency."""
    
    def __init__(self):
        self.versions: Dict[str, PromptVersion] = {}
        self.active_version: Optional[str] = None
    
    def add_version(self, version: str, content: str) -> PromptVersion:
        """Add a new prompt version."""
        
        prompt = PromptVersion(
            version=version,
            content=content,
            created_at=datetime.now()
        )
        
        self.versions[version] = prompt
        
        # Set as active if first version
        if self.active_version is None:
            self.active_version = version
        
        return prompt
    
    def get_active(self) -> Optional[PromptVersion]:
        """Get the active prompt version."""
        if self.active_version:
            return self.versions.get(self.active_version)
        return None
    
    def set_active(self, version: str):
        """Set the active version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        self.active_version = version
    
    def rollback(self, version: str):
        """Rollback to a previous version."""
        self.set_active(version)
    
    def get_cache_key(self) -> str:
        """Get cache key for active version."""
        active = self.get_active()
        return active.cache_key if active else ""
    
    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare two versions."""
        
        p1 = self.versions.get(v1)
        p2 = self.versions.get(v2)
        
        if not p1 or not p2:
            return {"error": "Version not found"}
        
        return {
            "version_1": v1,
            "version_2": v2,
            "same_content": p1.content == p2.content,
            "cache_key_1": p1.cache_key,
            "cache_key_2": p2.cache_key,
            "token_diff": abs(len(p1.content) - len(p2.content)) // 4
        }


# Usage
manager = PromptVersionManager()

manager.add_version("1.0", "You are a helpful assistant. " * 100)
manager.add_version("1.1", "You are an expert assistant. " * 100)
manager.add_version("2.0", "You are a specialized AI. " * 100)

print(f"Active cache key: {manager.get_cache_key()}")

manager.set_active("2.0")
print(f"New cache key: {manager.get_cache_key()}")

print(manager.compare_versions("1.0", "2.0"))
```

### Blue-Green Deployments

```python
class BlueGreenDeployment:
    """Blue-green deployment for prompt changes."""
    
    def __init__(self, version_manager: PromptVersionManager):
        self.manager = version_manager
        self.blue_version: Optional[str] = None
        self.green_version: Optional[str] = None
        self.traffic_split: float = 0.0  # 0 = all blue, 1 = all green
    
    def deploy_green(self, version: str):
        """Deploy new version as green."""
        if version not in self.manager.versions:
            raise ValueError(f"Version {version} not found")
        
        self.blue_version = self.manager.active_version
        self.green_version = version
        self.traffic_split = 0.0  # Start with no traffic to green
    
    def shift_traffic(self, green_percentage: float):
        """Shift traffic to green (0-100)."""
        self.traffic_split = green_percentage / 100
    
    def get_version_for_request(self) -> str:
        """Get version for a request based on traffic split."""
        import random
        
        if self.green_version and random.random() < self.traffic_split:
            return self.green_version
        return self.blue_version or self.manager.active_version
    
    def promote_green(self):
        """Promote green to active, making it the new blue."""
        if self.green_version:
            self.manager.set_active(self.green_version)
            self.blue_version = self.green_version
            self.green_version = None
            self.traffic_split = 0.0
    
    def rollback(self):
        """Rollback to blue."""
        self.green_version = None
        self.traffic_split = 0.0
    
    def get_status(self) -> dict:
        """Get deployment status."""
        return {
            "blue_version": self.blue_version,
            "green_version": self.green_version,
            "traffic_to_green": f"{self.traffic_split * 100:.0f}%",
            "active": self.manager.active_version
        }


# Usage
deployment = BlueGreenDeployment(manager)

# Deploy new version
deployment.deploy_green("2.0")

# Gradually shift traffic
deployment.shift_traffic(10)  # 10% to green
# ... monitor ...
deployment.shift_traffic(50)  # 50% to green
# ... monitor ...
deployment.shift_traffic(100) # 100% to green

# Promote if successful
deployment.promote_green()
```

---

## Testing and Validation

### Cache Behavior Testing

```python
from typing import List, Tuple

class CacheBehaviorTester:
    """Test cache behavior."""
    
    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.results: List[dict] = []
    
    async def test_cache_hit(
        self,
        system_prompt: str,
        test_queries: List[str]
    ) -> dict:
        """Test if cache hits are occurring."""
        
        results = []
        
        for i, query in enumerate(test_queries):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            usage = response.usage
            details = getattr(usage, 'prompt_tokens_details', None)
            cached = details.cached_tokens if details else 0
            
            results.append({
                "query_index": i,
                "input_tokens": usage.prompt_tokens,
                "cached_tokens": cached,
                "is_hit": cached > 0
            })
        
        # Analyze results
        hits = sum(1 for r in results if r["is_hit"])
        
        return {
            "total_queries": len(test_queries),
            "cache_hits": hits,
            "hit_rate": f"{hits / len(test_queries) * 100:.1f}%",
            "first_hit_at": next((i for i, r in enumerate(results) if r["is_hit"]), None),
            "results": results
        }
    
    async def test_prefix_sensitivity(
        self,
        base_prompt: str,
        modifications: List[Tuple[str, str]]  # (name, modified_prompt)
    ) -> dict:
        """Test how modifications affect caching."""
        
        results = []
        
        # First, establish cache with base prompt
        await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": "test"}
            ]
        )
        
        # Test modifications
        for name, modified in modifications:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": modified},
                    {"role": "user", "content": "test"}
                ]
            )
            
            usage = response.usage
            details = getattr(usage, 'prompt_tokens_details', None)
            cached = details.cached_tokens if details else 0
            
            results.append({
                "modification": name,
                "cached_tokens": cached,
                "cache_preserved": cached > 0,
                "cache_rate": cached / usage.prompt_tokens if usage.prompt_tokens > 0 else 0
            })
        
        return {
            "base_prompt_tokens": len(base_prompt) // 4,
            "modifications_tested": len(modifications),
            "cache_preserved": sum(1 for r in results if r["cache_preserved"]),
            "results": results
        }


# Usage
# tester = CacheBehaviorTester(client)
# 
# result = await tester.test_cache_hit(
#     system_prompt="Expert assistant..." * 100,
#     test_queries=["Query 1", "Query 2", "Query 3"]
# )
```

### Performance Validation

```python
class PerformanceValidator:
    """Validate caching performance improvements."""
    
    def __init__(self):
        self.baseline_metrics: List[dict] = []
        self.optimized_metrics: List[dict] = []
    
    def record_baseline(
        self,
        latency_ms: float,
        cost: float,
        tokens: int
    ):
        """Record baseline (no caching) metric."""
        self.baseline_metrics.append({
            "latency_ms": latency_ms,
            "cost": cost,
            "tokens": tokens
        })
    
    def record_optimized(
        self,
        latency_ms: float,
        cost: float,
        tokens: int,
        cached_tokens: int
    ):
        """Record optimized (with caching) metric."""
        self.optimized_metrics.append({
            "latency_ms": latency_ms,
            "cost": cost,
            "tokens": tokens,
            "cached_tokens": cached_tokens
        })
    
    def validate(self) -> dict:
        """Validate performance improvements."""
        
        if not self.baseline_metrics or not self.optimized_metrics:
            return {"error": "Need both baseline and optimized metrics"}
        
        # Calculate averages
        def avg(metrics: List[dict], key: str) -> float:
            return sum(m[key] for m in metrics) / len(metrics)
        
        baseline_latency = avg(self.baseline_metrics, "latency_ms")
        optimized_latency = avg(self.optimized_metrics, "latency_ms")
        
        baseline_cost = sum(m["cost"] for m in self.baseline_metrics)
        optimized_cost = sum(m["cost"] for m in self.optimized_metrics)
        
        # Calculate improvements
        latency_improvement = (baseline_latency - optimized_latency) / baseline_latency * 100
        cost_improvement = (baseline_cost - optimized_cost) / baseline_cost * 100
        
        # Validation thresholds
        EXPECTED_LATENCY_IMPROVEMENT = 30  # At least 30% faster
        EXPECTED_COST_IMPROVEMENT = 40     # At least 40% cheaper
        
        return {
            "baseline": {
                "requests": len(self.baseline_metrics),
                "avg_latency_ms": f"{baseline_latency:.1f}",
                "total_cost": f"${baseline_cost:.4f}"
            },
            "optimized": {
                "requests": len(self.optimized_metrics),
                "avg_latency_ms": f"{optimized_latency:.1f}",
                "total_cost": f"${optimized_cost:.4f}",
                "avg_cache_rate": f"{avg(self.optimized_metrics, 'cached_tokens') / avg(self.optimized_metrics, 'tokens') * 100:.1f}%"
            },
            "improvements": {
                "latency": f"{latency_improvement:.1f}%",
                "cost": f"{cost_improvement:.1f}%"
            },
            "validation": {
                "latency_pass": latency_improvement >= EXPECTED_LATENCY_IMPROVEMENT,
                "cost_pass": cost_improvement >= EXPECTED_COST_IMPROVEMENT,
                "overall_pass": (
                    latency_improvement >= EXPECTED_LATENCY_IMPROVEMENT and
                    cost_improvement >= EXPECTED_COST_IMPROVEMENT
                )
            }
        }
```

---

## Common Pitfalls

### Pitfall Checklist

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Pitfall:
    """Common caching pitfall."""
    
    name: str
    description: str
    check: Callable[[dict], bool]
    fix: str


PITFALLS = [
    Pitfall(
        name="dynamic_prefix",
        description="Dynamic content at start of prompt breaks caching",
        check=lambda ctx: ctx.get("has_timestamp_prefix", False),
        fix="Move timestamps, user IDs, and other dynamic content to end of prompt"
    ),
    Pitfall(
        name="too_short",
        description="Prompt too short for caching (<1024 tokens)",
        check=lambda ctx: ctx.get("token_count", 0) < 1024,
        fix="Add more static context to reach 1024+ token threshold"
    ),
    Pitfall(
        name="inconsistent_tools",
        description="Tool definitions vary between requests",
        check=lambda ctx: not ctx.get("tools_sorted", True),
        fix="Sort tools alphabetically and use consistent definitions"
    ),
    Pitfall(
        name="image_detail_mismatch",
        description="Different image detail levels break cache",
        check=lambda ctx: ctx.get("image_details_inconsistent", False),
        fix="Use consistent detail level (high/low/auto) for all images"
    ),
    Pitfall(
        name="cache_key_explosion",
        description="Too many unique cache keys fragment cache",
        check=lambda ctx: ctx.get("unique_keys", 0) > 50,
        fix="Consolidate cache keys, share across similar requests"
    ),
    Pitfall(
        name="traffic_gaps",
        description="Large gaps in traffic cause cache misses",
        check=lambda ctx: ctx.get("max_gap_seconds", 0) > 300,
        fix="Implement cache warming for low-traffic periods"
    ),
    Pitfall(
        name="no_monitoring",
        description="Not tracking cache performance",
        check=lambda ctx: not ctx.get("monitoring_enabled", False),
        fix="Implement cache hit rate and latency monitoring"
    ),
]


class PitfallChecker:
    """Check for common caching pitfalls."""
    
    def __init__(self):
        self.pitfalls = PITFALLS
    
    def check(self, context: dict) -> dict:
        """Check for pitfalls given context."""
        
        issues = []
        warnings = []
        
        for pitfall in self.pitfalls:
            try:
                if pitfall.check(context):
                    issues.append({
                        "name": pitfall.name,
                        "description": pitfall.description,
                        "fix": pitfall.fix
                    })
            except Exception:
                pass  # Skip if check fails
        
        return {
            "issues_found": len(issues),
            "issues": issues,
            "clean": len(issues) == 0,
            "recommendation": "Address issues above" if issues else "No pitfalls detected"
        }


# Usage
checker = PitfallChecker()

context = {
    "has_timestamp_prefix": True,
    "token_count": 2000,
    "tools_sorted": True,
    "unique_keys": 5,
    "max_gap_seconds": 100,
    "monitoring_enabled": True
}

result = checker.check(context)
print(f"Issues found: {result['issues_found']}")
for issue in result["issues"]:
    print(f"  - {issue['name']}: {issue['fix']}")
```

---

## Summary Checklist

### Production Readiness Checklist

```python
PRODUCTION_CHECKLIST = {
    "prompt_design": [
        "Static content placed at beginning of prompts",
        "Dynamic content placed at end",
        "Prompts exceed 1024 token threshold",
        "Consistent tool definitions (sorted)",
        "Consistent image detail levels",
    ],
    "cache_configuration": [
        "Appropriate TTL selected (default vs extended)",
        "Cache keys are meaningful and consistent",
        "Limited number of unique cache keys",
        "Prompt versioning implemented",
    ],
    "monitoring": [
        "Cache hit rate tracking enabled",
        "Latency monitoring configured",
        "Cost savings tracking active",
        "Alerts configured for anomalies",
    ],
    "operations": [
        "Cache warming strategy for low traffic",
        "Rate limits respected",
        "Retry logic with backoff",
        "Blue-green deployment for changes",
    ],
    "testing": [
        "Cache behavior validated",
        "Performance improvements measured",
        "Regression tests for prompt changes",
    ],
}


def check_readiness(answers: dict) -> dict:
    """Check production readiness."""
    
    results = {}
    total_items = 0
    passed_items = 0
    
    for category, items in PRODUCTION_CHECKLIST.items():
        category_results = []
        
        for item in items:
            total_items += 1
            is_done = answers.get(item, False)
            if is_done:
                passed_items += 1
            
            category_results.append({
                "item": item,
                "done": is_done
            })
        
        results[category] = {
            "items": category_results,
            "complete": sum(1 for r in category_results if r["done"]),
            "total": len(items)
        }
    
    return {
        "categories": results,
        "overall_score": f"{passed_items}/{total_items} ({passed_items/total_items*100:.0f}%)",
        "ready": passed_items == total_items
    }


# Usage
answers = {
    "Static content placed at beginning of prompts": True,
    "Dynamic content placed at end": True,
    "Prompts exceed 1024 token threshold": True,
    # ... fill in rest
}

readiness = check_readiness(answers)
print(f"Production readiness: {readiness['overall_score']}")
```

---

## Hands-on Exercise

### Your Task

Create a caching optimization audit tool.

### Requirements

1. Analyze current caching setup
2. Identify pitfalls and issues
3. Generate recommendations
4. Create action plan

<details>
<summary>💡 Hints</summary>

- Check prompt structure for static/dynamic ordering
- Verify token thresholds
- Analyze cache key usage patterns
</details>

<details>
<summary>✅ Solution</summary>

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import hashlib
import json

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Finding:
    """Audit finding."""
    
    category: str
    severity: Severity
    title: str
    description: str
    recommendation: str
    effort: str  # low, medium, high


@dataclass
class AuditContext:
    """Context for audit."""
    
    prompts: List[str] = field(default_factory=list)
    tools: List[dict] = field(default_factory=list)
    cache_keys: List[str] = field(default_factory=list)
    metrics: List[dict] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class CachingAuditTool:
    """Comprehensive caching optimization audit."""
    
    def __init__(self):
        self.findings: List[Finding] = []
    
    def audit(self, context: AuditContext) -> dict:
        """Run full audit."""
        
        self.findings = []
        
        # Run all checks
        self._audit_prompts(context)
        self._audit_tools(context)
        self._audit_cache_keys(context)
        self._audit_metrics(context)
        self._audit_config(context)
        
        return self._generate_report()
    
    def _audit_prompts(self, context: AuditContext):
        """Audit prompt design."""
        
        for i, prompt in enumerate(context.prompts):
            tokens = len(prompt) // 4
            
            # Check token threshold
            if tokens < 1024:
                self.findings.append(Finding(
                    category="Prompt Design",
                    severity=Severity.HIGH,
                    title=f"Prompt {i} below caching threshold",
                    description=f"Prompt has ~{tokens} tokens, needs 1024+ for caching",
                    recommendation="Add more static context to reach threshold",
                    effort="low"
                ))
            
            # Check for dynamic prefix patterns
            dynamic_patterns = ["timestamp", "datetime", "now()", "current_time"]
            first_100_chars = prompt[:100].lower()
            
            for pattern in dynamic_patterns:
                if pattern in first_100_chars:
                    self.findings.append(Finding(
                        category="Prompt Design",
                        severity=Severity.CRITICAL,
                        title=f"Dynamic content at start of prompt {i}",
                        description=f"Found '{pattern}' near start of prompt",
                        recommendation="Move dynamic content to end of prompt",
                        effort="low"
                    ))
                    break
    
    def _audit_tools(self, context: AuditContext):
        """Audit tool definitions."""
        
        if not context.tools:
            return
        
        # Check if tools are sorted
        tool_names = [t.get("function", {}).get("name", "") for t in context.tools]
        if tool_names != sorted(tool_names):
            self.findings.append(Finding(
                category="Tool Definitions",
                severity=Severity.MEDIUM,
                title="Tools not sorted alphabetically",
                description="Unsorted tools can cause cache misses between requests",
                recommendation="Sort tools by name for consistent ordering",
                effort="low"
            ))
        
        # Check tool complexity
        tool_tokens = len(json.dumps(context.tools)) // 4
        if tool_tokens > 500:
            self.findings.append(Finding(
                category="Tool Definitions",
                severity=Severity.LOW,
                title="Large tool definitions",
                description=f"Tools use ~{tool_tokens} tokens",
                recommendation="Tools are good caching candidates, ensure consistency",
                effort="low"
            ))
    
    def _audit_cache_keys(self, context: AuditContext):
        """Audit cache key usage."""
        
        unique_keys = set(context.cache_keys)
        
        if len(unique_keys) > 50:
            self.findings.append(Finding(
                category="Cache Keys",
                severity=Severity.HIGH,
                title="Too many unique cache keys",
                description=f"Found {len(unique_keys)} unique keys",
                recommendation="Consolidate keys to improve cache hit rates",
                effort="medium"
            ))
        
        if not context.cache_keys:
            self.findings.append(Finding(
                category="Cache Keys",
                severity=Severity.MEDIUM,
                title="No explicit cache keys",
                description="Not using explicit cache keys",
                recommendation="Consider using cache keys for routing optimization",
                effort="low"
            ))
    
    def _audit_metrics(self, context: AuditContext):
        """Audit from metrics."""
        
        if not context.metrics:
            self.findings.append(Finding(
                category="Monitoring",
                severity=Severity.HIGH,
                title="No metrics available",
                description="Cannot analyze cache performance without metrics",
                recommendation="Implement cache metrics collection",
                effort="medium"
            ))
            return
        
        # Calculate hit rate
        total_input = sum(m.get("input_tokens", 0) for m in context.metrics)
        total_cached = sum(m.get("cached_tokens", 0) for m in context.metrics)
        
        if total_input > 0:
            hit_rate = total_cached / total_input
            
            if hit_rate < 0.3:
                self.findings.append(Finding(
                    category="Performance",
                    severity=Severity.HIGH,
                    title=f"Low cache hit rate: {hit_rate:.1%}",
                    description="Less than 30% of tokens are being cached",
                    recommendation="Review prompt structure and cache warming",
                    effort="medium"
                ))
            elif hit_rate < 0.6:
                self.findings.append(Finding(
                    category="Performance",
                    severity=Severity.MEDIUM,
                    title=f"Moderate cache hit rate: {hit_rate:.1%}",
                    description="Room for improvement in cache efficiency",
                    recommendation="Optimize prompt ordering and consistency",
                    effort="low"
                ))
    
    def _audit_config(self, context: AuditContext):
        """Audit configuration."""
        
        config = context.config
        
        if not config.get("monitoring_enabled"):
            self.findings.append(Finding(
                category="Configuration",
                severity=Severity.MEDIUM,
                title="Monitoring not enabled",
                description="Cache performance monitoring is disabled",
                recommendation="Enable monitoring for visibility",
                effort="low"
            ))
        
        if not config.get("cache_warming"):
            self.findings.append(Finding(
                category="Configuration",
                severity=Severity.LOW,
                title="No cache warming configured",
                description="Cache may go cold during low traffic",
                recommendation="Implement cache warming for consistent performance",
                effort="medium"
            ))
        
        if config.get("ttl") == "default" and config.get("traffic_pattern") == "steady":
            self.findings.append(Finding(
                category="Configuration",
                severity=Severity.LOW,
                title="Consider extended TTL",
                description="Steady traffic pattern could benefit from extended TTL",
                recommendation="Enable 24-hour extended caching",
                effort="low"
            ))
    
    def _generate_report(self) -> dict:
        """Generate audit report."""
        
        # Group by severity
        by_severity = {s: [] for s in Severity}
        for finding in self.findings:
            by_severity[finding.severity].append(finding)
        
        # Group by category
        by_category = {}
        for finding in self.findings:
            if finding.category not in by_category:
                by_category[finding.category] = []
            by_category[finding.category].append(finding)
        
        # Calculate score
        severity_weights = {
            Severity.CRITICAL: 10,
            Severity.HIGH: 5,
            Severity.MEDIUM: 2,
            Severity.LOW: 1
        }
        
        total_weight = sum(
            severity_weights[f.severity] for f in self.findings
        )
        
        max_score = 100
        score = max(0, max_score - total_weight)
        
        # Generate action plan
        action_plan = []
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            for finding in by_severity[severity]:
                action_plan.append({
                    "priority": severity.value,
                    "action": finding.recommendation,
                    "effort": finding.effort,
                    "context": finding.title
                })
        
        return {
            "summary": {
                "total_findings": len(self.findings),
                "critical": len(by_severity[Severity.CRITICAL]),
                "high": len(by_severity[Severity.HIGH]),
                "medium": len(by_severity[Severity.MEDIUM]),
                "low": len(by_severity[Severity.LOW]),
                "score": f"{score}/100"
            },
            "findings_by_category": {
                cat: [
                    {
                        "severity": f.severity.value,
                        "title": f.title,
                        "description": f.description,
                        "recommendation": f.recommendation
                    }
                    for f in findings
                ]
                for cat, findings in by_category.items()
            },
            "action_plan": action_plan[:10],  # Top 10 actions
            "grade": self._calculate_grade(score)
        }
    
    def _calculate_grade(self, score: int) -> str:
        """Calculate letter grade from score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


# Usage
auditor = CachingAuditTool()

# Create audit context
context = AuditContext(
    prompts=[
        f"Current time: {datetime.now()}\nYou are an assistant...",  # Bad: dynamic prefix
        "You are an expert assistant. " * 200,  # Good: static, long
    ],
    tools=[
        {"function": {"name": "search"}},
        {"function": {"name": "calculate"}},
    ],
    cache_keys=["app_v1"] * 50 + ["app_v2"] * 30,
    metrics=[
        {"input_tokens": 2000, "cached_tokens": 1500},
        {"input_tokens": 2000, "cached_tokens": 1600},
        {"input_tokens": 2000, "cached_tokens": 0},  # Miss
    ],
    config={
        "monitoring_enabled": True,
        "cache_warming": False,
        "ttl": "default",
        "traffic_pattern": "steady"
    }
)

# Run audit
report = auditor.audit(context)

# Display results
print("=== Caching Optimization Audit ===")
print(f"\nScore: {report['summary']['score']} (Grade: {report['grade']})")
print(f"\nFindings: {report['summary']['total_findings']}")
print(f"  Critical: {report['summary']['critical']}")
print(f"  High: {report['summary']['high']}")
print(f"  Medium: {report['summary']['medium']}")
print(f"  Low: {report['summary']['low']}")

print("\n=== Top Actions ===")
for i, action in enumerate(report['action_plan'][:5], 1):
    print(f"{i}. [{action['priority'].upper()}] {action['action']}")
    print(f"   Context: {action['context']}")
    print(f"   Effort: {action['effort']}")
```

</details>

---

## Summary

✅ Maintain steady request stream for cache warmth  
✅ Keep cache key usage under rate limits  
✅ Implement cache warming for low-traffic periods  
✅ Version prompts for safe deployments  
✅ Test and validate cache behavior regularly

**Next Unit:** [Built-in Tools MCP Integration](../17-built-in-tools-mcp-integration/)

---

## Further Reading

- [OpenAI Production Best Practices](https://platform.openai.com/docs/guides/production-best-practices) — Deployment guide
- [Anthropic Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Caching guide
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits) — Rate limit handling
