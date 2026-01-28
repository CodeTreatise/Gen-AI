---
title: "API Availability and Reliability"
---

# API Availability and Reliability

## Introduction

API reliability directly impacts your application's uptime. Understanding provider SLAs, rate limits, and building resilient systems ensures your AI features remain available.

### What We'll Cover

- Uptime guarantees (SLAs)
- Rate limits and quotas
- Multi-provider strategies
- Handling failures gracefully

---

## Uptime Guarantees

### Provider SLAs

| Provider | SLA | Credits for Downtime |
|----------|-----|---------------------|
| OpenAI | 99.9% (Enterprise) | Yes |
| Anthropic | 99.9% (Scale) | Yes |
| Azure OpenAI | 99.9% | Yes |
| Google Vertex | 99.9% | Yes |
| AWS Bedrock | 99.9% | Yes |

### Understanding SLA Math

```python
def calculate_allowed_downtime(sla_percent: float) -> dict:
    """Calculate allowed downtime from SLA percentage"""
    
    downtime_fraction = (100 - sla_percent) / 100
    
    minutes_per_year = 365.25 * 24 * 60
    minutes_per_month = 30.44 * 24 * 60
    minutes_per_week = 7 * 24 * 60
    
    return {
        "sla": f"{sla_percent}%",
        "yearly_downtime_minutes": round(minutes_per_year * downtime_fraction, 1),
        "monthly_downtime_minutes": round(minutes_per_month * downtime_fraction, 1),
        "weekly_downtime_minutes": round(minutes_per_week * downtime_fraction, 1)
    }

# 99.9% SLA
print(calculate_allowed_downtime(99.9))
# yearly: ~526 minutes (~8.7 hours)
# monthly: ~44 minutes

# 99.99% SLA
print(calculate_allowed_downtime(99.99))
# yearly: ~52 minutes
# monthly: ~4 minutes
```

---

## Rate Limits

### Common Rate Limits

```python
rate_limits = {
    "openai": {
        "gpt-4o": {
            "rpm": 500,       # Requests per minute (Tier 1)
            "tpm": 30000,     # Tokens per minute
            "rpd": 10000      # Requests per day
        },
        "gpt-4o-mini": {
            "rpm": 500,
            "tpm": 200000,
            "rpd": 10000
        }
    },
    "anthropic": {
        "claude-3-5-sonnet": {
            "rpm": 50,        # Varies by tier
            "tpm": 40000,
            "rpd": 1000
        }
    }
}
```

### Rate Limit Handler

```python
import time
from collections import deque

class RateLimiter:
    """Client-side rate limiting"""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_times = deque()
        self.token_counts = deque()
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if we're about to exceed limits"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
            self.token_counts.popleft()
        
        # Check request limit
        if len(self.request_times) >= self.rpm:
            sleep_time = self.request_times[0] - minute_ago
            time.sleep(sleep_time)
        
        # Check token limit
        current_tokens = sum(self.token_counts)
        if current_tokens + estimated_tokens > self.tpm:
            sleep_time = 60 - (now - self.request_times[0])
            time.sleep(max(0, sleep_time))
        
        # Record this request
        self.request_times.append(time.time())
        self.token_counts.append(estimated_tokens)

# Usage
limiter = RateLimiter(requests_per_minute=50, tokens_per_minute=40000)

for prompt in prompts:
    limiter.wait_if_needed(estimated_tokens=500)
    response = client.chat.completions.create(...)
```

### Retry with Backoff

```python
import random
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def call_with_retry(client, **kwargs):
    """Call API with exponential backoff retry"""
    return client.chat.completions.create(**kwargs)

# Manual implementation
def call_with_manual_retry(client, max_retries: int = 5, **kwargs):
    """Manual retry with jitter"""
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(min(wait_time, 60))
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(1)
```

---

## Multi-Provider Strategies

### Fallback Pattern

```python
class MultiProviderClient:
    """Client with automatic fallback"""
    
    def __init__(self):
        self.providers = [
            {"name": "openai", "client": OpenAI(), "model": "gpt-4o"},
            {"name": "anthropic", "client": Anthropic(), "model": "claude-3-5-sonnet"},
            {"name": "google", "client": genai, "model": "gemini-1.5-pro"},
        ]
    
    def chat(self, messages: list, **kwargs) -> str:
        """Try providers in order until one succeeds"""
        
        errors = []
        
        for provider in self.providers:
            try:
                if provider["name"] == "openai":
                    response = provider["client"].chat.completions.create(
                        model=provider["model"],
                        messages=messages,
                        **kwargs
                    )
                    return response.choices[0].message.content
                
                elif provider["name"] == "anthropic":
                    response = provider["client"].messages.create(
                        model=provider["model"],
                        messages=self._convert_to_anthropic(messages),
                        max_tokens=kwargs.get("max_tokens", 1024)
                    )
                    return response.content[0].text
                
                elif provider["name"] == "google":
                    model = provider["client"].GenerativeModel(provider["model"])
                    response = model.generate_content(messages[-1]["content"])
                    return response.text
                    
            except Exception as e:
                errors.append(f"{provider['name']}: {str(e)}")
                continue
        
        raise Exception(f"All providers failed: {errors}")
    
    def _convert_to_anthropic(self, messages: list) -> list:
        """Convert OpenAI format to Anthropic format"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"
        ]
```

### Load Balancing

```python
import random

class LoadBalancedClient:
    """Distribute load across providers"""
    
    def __init__(self):
        self.providers = [
            {"name": "openai", "weight": 0.5, "client": OpenAI()},
            {"name": "anthropic", "weight": 0.3, "client": Anthropic()},
            {"name": "google", "weight": 0.2, "client": None},  # genai
        ]
    
    def select_provider(self) -> dict:
        """Weighted random selection"""
        total = sum(p["weight"] for p in self.providers)
        r = random.uniform(0, total)
        
        cumulative = 0
        for provider in self.providers:
            cumulative += provider["weight"]
            if r <= cumulative:
                return provider
        
        return self.providers[0]
    
    def chat(self, messages: list, **kwargs) -> str:
        """Call weighted random provider"""
        provider = self.select_provider()
        # ... make call to selected provider
```

---

## Handling Degradation

### Circuit Breaker

```python
class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, don't try
    HALF_OPEN = "half_open"  # Testing recovery
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = self.CLOSED
    
    def can_execute(self) -> bool:
        """Check if we should attempt the call"""
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.HALF_OPEN
                return True
            return False
        
        if self.state == self.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record successful call"""
        self.failures = 0
        self.state = self.CLOSED
    
    def record_failure(self):
        """Record failed call"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = self.OPEN

# Usage
breaker = CircuitBreaker()

def call_api_with_breaker(client, **kwargs):
    if not breaker.can_execute():
        raise Exception("Circuit breaker is open")
    
    try:
        result = client.chat.completions.create(**kwargs)
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise
```

### Graceful Degradation

```python
class GracefulDegradation:
    """Provide degraded functionality when API fails"""
    
    def __init__(self):
        self.cache = {}
        self.fallback_responses = {
            "greeting": "Hello! I'm currently experiencing issues. Please try again.",
            "error": "I'm sorry, I couldn't process that request. Please try again later.",
        }
    
    def get_response(self, prompt: str) -> str:
        """Get response with fallbacks"""
        
        # Try cache first
        cache_key = hash(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try API
        try:
            response = self._call_api(prompt)
            self.cache[cache_key] = response
            return response
        except Exception as e:
            # Return appropriate fallback
            return self._get_fallback(prompt, str(e))
    
    def _call_api(self, prompt: str) -> str:
        # Actual API call
        pass
    
    def _get_fallback(self, prompt: str, error: str) -> str:
        """Select appropriate fallback response"""
        
        if "rate_limit" in error.lower():
            return "I'm receiving many requests. Please wait a moment and try again."
        
        if "timeout" in error.lower():
            return "The request took too long. Let me try a simpler response..."
        
        return self.fallback_responses["error"]
```

---

## Monitoring

### Health Checks

```python
class APIHealthMonitor:
    """Monitor API health"""
    
    def __init__(self, providers: list):
        self.providers = providers
        self.health_status = {p: "unknown" for p in providers}
    
    async def check_all(self):
        """Check health of all providers"""
        for provider in self.providers:
            try:
                await self._health_check(provider)
                self.health_status[provider] = "healthy"
            except Exception as e:
                self.health_status[provider] = f"unhealthy: {e}"
    
    async def _health_check(self, provider: str):
        """Simple health check call"""
        # Make minimal API call to check availability
        pass
    
    def get_healthy_providers(self) -> list:
        """Get list of healthy providers"""
        return [p for p, status in self.health_status.items() 
                if status == "healthy"]
```

---

## Summary

✅ **Understand SLAs** - Know your provider's guarantees

✅ **Handle rate limits** - Client-side limiting and retries

✅ **Use fallbacks** - Multiple providers for resilience

✅ **Circuit breakers** - Prevent cascade failures

✅ **Monitor health** - Track provider availability

**Next:** [Compliance & Data Privacy](./06-compliance-data-privacy.md)

---

## Navigation

| Previous | Up | Next |
|----------|-------|------|
| [Latency Considerations](./04-latency-considerations.md) | [Model Selection](./00-model-selection-criteria.md) | [Compliance & Privacy](./06-compliance-data-privacy.md) |

