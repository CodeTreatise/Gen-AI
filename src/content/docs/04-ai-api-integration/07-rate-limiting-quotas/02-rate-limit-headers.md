---
title: "Rate Limit Headers"
---

# Rate Limit Headers

## Introduction

API responses include headers that reveal your current rate limit status. Reading these headers enables proactive management—you can throttle requests before hitting limits rather than reacting to 429 errors.

### What We'll Cover

- Available rate limit headers
- Extracting headers from responses
- Parsing reset times
- Building a header-based limiter

### Prerequisites

- Understanding rate limits
- HTTP header basics

---

## Rate Limit Header Reference

### OpenAI Headers

| Header | Description | Example Value |
|--------|-------------|---------------|
| `x-ratelimit-limit-requests` | Max requests per period | `10000` |
| `x-ratelimit-remaining-requests` | Requests remaining | `9500` |
| `x-ratelimit-reset-requests` | Time until request limit resets | `6s` |
| `x-ratelimit-limit-tokens` | Max tokens per period | `1000000` |
| `x-ratelimit-remaining-tokens` | Tokens remaining | `950000` |
| `x-ratelimit-reset-tokens` | Time until token limit resets | `1m` |
| `retry-after` | Seconds to wait (on 429) | `30` |
| `x-request-id` | Unique request identifier | `req_abc123xyz` |

### Anthropic Headers

| Header | Description |
|--------|-------------|
| `anthropic-ratelimit-requests-limit` | Max requests per period |
| `anthropic-ratelimit-requests-remaining` | Requests remaining |
| `anthropic-ratelimit-requests-reset` | ISO timestamp for reset |
| `anthropic-ratelimit-tokens-limit` | Max tokens per period |
| `anthropic-ratelimit-tokens-remaining` | Tokens remaining |
| `anthropic-ratelimit-tokens-reset` | ISO timestamp for reset |

---

## Extracting Headers

### From OpenAI Responses

```python
from openai import OpenAI

client = OpenAI()

def get_completion_with_headers(messages: list) -> dict:
    """Get completion and extract rate limit headers."""
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )
    
    # Access raw response headers
    # Note: This requires accessing the underlying httpx response
    # The SDK wraps responses, so we track via usage
    
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "model": response.model
    }
```

### Using httpx Directly

```python
import httpx
import os

def get_with_headers(messages: list) -> dict:
    """Make request with direct access to headers."""
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4.1",
        "messages": messages
    }
    
    response = httpx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # Extract rate limit headers
    rate_limits = {
        "requests": {
            "limit": response.headers.get("x-ratelimit-limit-requests"),
            "remaining": response.headers.get("x-ratelimit-remaining-requests"),
            "reset": response.headers.get("x-ratelimit-reset-requests")
        },
        "tokens": {
            "limit": response.headers.get("x-ratelimit-limit-tokens"),
            "remaining": response.headers.get("x-ratelimit-remaining-tokens"),
            "reset": response.headers.get("x-ratelimit-reset-tokens")
        },
        "request_id": response.headers.get("x-request-id")
    }
    
    return {
        "data": response.json(),
        "rate_limits": rate_limits
    }


# Usage
result = get_with_headers([{"role": "user", "content": "Hello"}])
print(f"Requests remaining: {result['rate_limits']['requests']['remaining']}")
print(f"Tokens remaining: {result['rate_limits']['tokens']['remaining']}")
```

---

## Parsing Reset Times

### OpenAI Format

OpenAI uses human-readable durations like `6s`, `1m`, `2m30s`:

```python
import re

def parse_reset_time(reset_str: str) -> float:
    """Parse OpenAI reset time string to seconds."""
    
    if not reset_str:
        return 60.0  # Default fallback
    
    total_seconds = 0.0
    
    # Match patterns like "1m30s", "45s", "2m"
    patterns = [
        (r"(\d+)h", 3600),   # Hours
        (r"(\d+)m", 60),     # Minutes
        (r"(\d+)s", 1),      # Seconds
        (r"(\d+)ms", 0.001)  # Milliseconds
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, reset_str)
        if match:
            total_seconds += float(match.group(1)) * multiplier
    
    return total_seconds if total_seconds > 0 else 60.0


# Examples
print(parse_reset_time("6s"))      # 6.0
print(parse_reset_time("1m"))      # 60.0
print(parse_reset_time("1m30s"))   # 90.0
print(parse_reset_time("500ms"))   # 0.5
```

### Anthropic Format

Anthropic uses ISO 8601 timestamps:

```python
from datetime import datetime, timezone

def parse_anthropic_reset(reset_str: str) -> float:
    """Parse Anthropic reset timestamp to seconds until reset."""
    
    if not reset_str:
        return 60.0
    
    try:
        # Parse ISO 8601 timestamp
        reset_time = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        
        delta = reset_time - now
        return max(0, delta.total_seconds())
    except ValueError:
        return 60.0


# Example
reset_timestamp = "2025-01-15T10:30:00Z"
seconds_until = parse_anthropic_reset(reset_timestamp)
print(f"Seconds until reset: {seconds_until}")
```

---

## Rate Limit Tracker

### Header-Based Tracker

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class RateLimitInfo:
    limit: int
    remaining: int
    reset_seconds: float
    last_updated: float = 0
    
    def update_from_headers(
        self,
        limit: str,
        remaining: str,
        reset: str
    ):
        """Update from response headers."""
        self.limit = int(limit) if limit else self.limit
        self.remaining = int(remaining) if remaining else self.remaining
        self.reset_seconds = parse_reset_time(reset)
        self.last_updated = time.time()
    
    @property
    def percent_used(self) -> float:
        if self.limit == 0:
            return 0
        return (self.limit - self.remaining) / self.limit * 100
    
    @property
    def is_low(self) -> bool:
        """Check if remaining is below 20%."""
        return self.remaining < self.limit * 0.2


class RateLimitTracker:
    """Track rate limits from response headers."""
    
    def __init__(self):
        self.requests = RateLimitInfo(limit=10000, remaining=10000, reset_seconds=60)
        self.tokens = RateLimitInfo(limit=1000000, remaining=1000000, reset_seconds=60)
        self.last_request_id: Optional[str] = None
    
    def update_from_response(self, headers: dict):
        """Update limits from response headers."""
        
        self.requests.update_from_headers(
            headers.get("x-ratelimit-limit-requests"),
            headers.get("x-ratelimit-remaining-requests"),
            headers.get("x-ratelimit-reset-requests")
        )
        
        self.tokens.update_from_headers(
            headers.get("x-ratelimit-limit-tokens"),
            headers.get("x-ratelimit-remaining-tokens"),
            headers.get("x-ratelimit-reset-tokens")
        )
        
        self.last_request_id = headers.get("x-request-id")
    
    def should_wait(self) -> Optional[float]:
        """Check if we should wait before next request."""
        
        if self.requests.remaining <= 0:
            return self.requests.reset_seconds
        
        if self.tokens.remaining < 1000:  # Less than 1K tokens
            return self.tokens.reset_seconds
        
        return None
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        return {
            "requests": {
                "remaining": self.requests.remaining,
                "limit": self.requests.limit,
                "percent_used": round(self.requests.percent_used, 1),
                "reset_in": self.requests.reset_seconds
            },
            "tokens": {
                "remaining": self.tokens.remaining,
                "limit": self.tokens.limit,
                "percent_used": round(self.tokens.percent_used, 1),
                "reset_in": self.tokens.reset_seconds
            },
            "last_request_id": self.last_request_id
        }
```

---

## Proactive Rate Limiting

### Pre-Request Check

```python
import time

class ProactiveRateLimiter:
    """Limit requests based on header information."""
    
    def __init__(self):
        self.tracker = RateLimitTracker()
        self.safety_margin = 0.1  # Keep 10% buffer
    
    def can_make_request(self, estimated_tokens: int = 1000) -> dict:
        """Check if request can proceed safely."""
        
        # Check request limit
        request_headroom = self.tracker.requests.remaining
        request_threshold = self.tracker.requests.limit * self.safety_margin
        
        if request_headroom <= request_threshold:
            return {
                "allowed": False,
                "reason": "Request limit too low",
                "wait_seconds": self.tracker.requests.reset_seconds
            }
        
        # Check token limit
        token_headroom = self.tracker.tokens.remaining
        
        if token_headroom < estimated_tokens:
            return {
                "allowed": False,
                "reason": "Insufficient token headroom",
                "wait_seconds": self.tracker.tokens.reset_seconds
            }
        
        return {"allowed": True}
    
    def make_request(self, request_fn, estimated_tokens: int = 1000):
        """Make request with proactive limiting."""
        
        # Check before making request
        check = self.can_make_request(estimated_tokens)
        
        if not check["allowed"]:
            print(f"Waiting {check['wait_seconds']}s: {check['reason']}")
            time.sleep(check["wait_seconds"])
        
        # Make the request
        response = request_fn()
        
        # Update tracker from response headers
        if hasattr(response, "_raw_response"):
            self.tracker.update_from_response(
                dict(response._raw_response.headers)
            )
        
        return response
```

---

## JavaScript Implementation

```javascript
class RateLimitTracker {
    constructor() {
        this.requests = { limit: 10000, remaining: 10000, resetSeconds: 60 };
        this.tokens = { limit: 1000000, remaining: 1000000, resetSeconds: 60 };
        this.lastRequestId = null;
    }
    
    parseResetTime(resetStr) {
        if (!resetStr) return 60;
        
        let seconds = 0;
        const patterns = [
            { regex: /(\d+)h/, multiplier: 3600 },
            { regex: /(\d+)m(?!s)/, multiplier: 60 },
            { regex: /(\d+)s/, multiplier: 1 },
            { regex: /(\d+)ms/, multiplier: 0.001 }
        ];
        
        for (const { regex, multiplier } of patterns) {
            const match = resetStr.match(regex);
            if (match) {
                seconds += parseFloat(match[1]) * multiplier;
            }
        }
        
        return seconds || 60;
    }
    
    updateFromHeaders(headers) {
        // Request limits
        const reqLimit = headers.get('x-ratelimit-limit-requests');
        const reqRemaining = headers.get('x-ratelimit-remaining-requests');
        const reqReset = headers.get('x-ratelimit-reset-requests');
        
        if (reqLimit) this.requests.limit = parseInt(reqLimit);
        if (reqRemaining) this.requests.remaining = parseInt(reqRemaining);
        if (reqReset) this.requests.resetSeconds = this.parseResetTime(reqReset);
        
        // Token limits
        const tokLimit = headers.get('x-ratelimit-limit-tokens');
        const tokRemaining = headers.get('x-ratelimit-remaining-tokens');
        const tokReset = headers.get('x-ratelimit-reset-tokens');
        
        if (tokLimit) this.tokens.limit = parseInt(tokLimit);
        if (tokRemaining) this.tokens.remaining = parseInt(tokRemaining);
        if (tokReset) this.tokens.resetSeconds = this.parseResetTime(tokReset);
        
        this.lastRequestId = headers.get('x-request-id');
    }
    
    getStatus() {
        return {
            requests: {
                remaining: this.requests.remaining,
                limit: this.requests.limit,
                percentUsed: ((this.requests.limit - this.requests.remaining) / 
                              this.requests.limit * 100).toFixed(1)
            },
            tokens: {
                remaining: this.tokens.remaining,
                limit: this.tokens.limit,
                percentUsed: ((this.tokens.limit - this.tokens.remaining) / 
                              this.tokens.limit * 100).toFixed(1)
            }
        };
    }
}

// Usage with fetch
const tracker = new RateLimitTracker();

async function makeRequestWithTracking(messages) {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'gpt-4.1',
            messages
        })
    });
    
    // Update tracker from headers
    tracker.updateFromHeaders(response.headers);
    
    console.log('Rate limit status:', tracker.getStatus());
    
    return response.json();
}
```

---

## Handling 429 with Headers

```python
from openai import RateLimitError

def handle_rate_limit_with_headers(error: RateLimitError) -> float:
    """Extract wait time from 429 response headers."""
    
    if not hasattr(error, "response") or not error.response:
        return 60.0  # Default wait
    
    headers = error.response.headers
    
    # Check retry-after first (most reliable)
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    
    # Fall back to reset headers
    request_reset = headers.get("x-ratelimit-reset-requests")
    token_reset = headers.get("x-ratelimit-reset-tokens")
    
    wait_times = []
    
    if request_reset:
        wait_times.append(parse_reset_time(request_reset))
    
    if token_reset:
        wait_times.append(parse_reset_time(token_reset))
    
    if wait_times:
        return max(wait_times)  # Wait for the longer reset
    
    return 60.0


# Usage
try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    wait_time = handle_rate_limit_with_headers(e)
    print(f"Rate limited. Waiting {wait_time}s")
    time.sleep(wait_time)
```

---

## Hands-on Exercise

### Your Task

Build a rate limit tracker that warns when approaching limits.

### Requirements

1. Parse OpenAI rate limit headers
2. Track both request and token limits
3. Warn at 80% usage
4. Return wait time when at 100%

### Expected Result

```python
tracker = RateLimitWarner()
tracker.update(headers)

status = tracker.check()
# {'ok': True, 'warning': 'Token usage at 82%'}
# or
# {'ok': False, 'wait_seconds': 30}
```

<details>
<summary>💡 Hints</summary>

- Store limit and remaining for both types
- Calculate percentage: (limit - remaining) / limit
- Return the reset time when remaining is 0
</details>

<details>
<summary>✅ Solution</summary>

```python
import re

class RateLimitWarner:
    def __init__(self, warn_threshold: float = 0.8):
        self.warn_threshold = warn_threshold
        
        self.req_limit = 10000
        self.req_remaining = 10000
        self.req_reset = 60.0
        
        self.tok_limit = 1000000
        self.tok_remaining = 1000000
        self.tok_reset = 60.0
    
    def _parse_reset(self, reset_str: str) -> float:
        """Parse reset time string to seconds."""
        if not reset_str:
            return 60.0
        
        total = 0.0
        for match in re.finditer(r"(\d+)(h|m|s|ms)", reset_str):
            value = float(match.group(1))
            unit = match.group(2)
            
            multipliers = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
            total += value * multipliers.get(unit, 1)
        
        return total if total > 0 else 60.0
    
    def update(self, headers: dict):
        """Update from response headers."""
        
        if headers.get("x-ratelimit-limit-requests"):
            self.req_limit = int(headers["x-ratelimit-limit-requests"])
        if headers.get("x-ratelimit-remaining-requests"):
            self.req_remaining = int(headers["x-ratelimit-remaining-requests"])
        if headers.get("x-ratelimit-reset-requests"):
            self.req_reset = self._parse_reset(headers["x-ratelimit-reset-requests"])
        
        if headers.get("x-ratelimit-limit-tokens"):
            self.tok_limit = int(headers["x-ratelimit-limit-tokens"])
        if headers.get("x-ratelimit-remaining-tokens"):
            self.tok_remaining = int(headers["x-ratelimit-remaining-tokens"])
        if headers.get("x-ratelimit-reset-tokens"):
            self.tok_reset = self._parse_reset(headers["x-ratelimit-reset-tokens"])
    
    def check(self) -> dict:
        """Check current status and return warnings or wait time."""
        
        req_used_pct = (self.req_limit - self.req_remaining) / self.req_limit
        tok_used_pct = (self.tok_limit - self.tok_remaining) / self.tok_limit
        
        # Check if at limit
        if self.req_remaining <= 0:
            return {
                "ok": False,
                "wait_seconds": self.req_reset,
                "reason": "Request limit reached"
            }
        
        if self.tok_remaining <= 0:
            return {
                "ok": False,
                "wait_seconds": self.tok_reset,
                "reason": "Token limit reached"
            }
        
        # Check for warnings
        warnings = []
        
        if req_used_pct >= self.warn_threshold:
            warnings.append(f"Request usage at {req_used_pct*100:.0f}%")
        
        if tok_used_pct >= self.warn_threshold:
            warnings.append(f"Token usage at {tok_used_pct*100:.0f}%")
        
        result = {"ok": True}
        
        if warnings:
            result["warning"] = "; ".join(warnings)
        
        result["status"] = {
            "requests": f"{self.req_remaining}/{self.req_limit}",
            "tokens": f"{self.tok_remaining}/{self.tok_limit}"
        }
        
        return result


# Test
tracker = RateLimitWarner()

# Simulate headers from response
headers = {
    "x-ratelimit-limit-requests": "10000",
    "x-ratelimit-remaining-requests": "1500",  # 85% used
    "x-ratelimit-reset-requests": "30s",
    "x-ratelimit-limit-tokens": "1000000",
    "x-ratelimit-remaining-tokens": "500000",  # 50% used
    "x-ratelimit-reset-tokens": "1m"
}

tracker.update(headers)
print(tracker.check())
# {'ok': True, 'warning': 'Request usage at 85%', 'status': {...}}
```

</details>

---

## Summary

✅ Response headers reveal real-time rate limit status  
✅ Parse reset times from `6s`, `1m30s` format  
✅ Track limits proactively to avoid 429 errors  
✅ Use `retry-after` header when rate limited  
✅ Different providers use different header formats

**Next:** [Tier-Based Limits](./03-tier-based-limits.md)

---

## Further Reading

- [OpenAI Rate Limit Headers](https://platform.openai.com/docs/guides/rate-limits/rate-limit-headers) — Header reference
- [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits) — Anthropic headers
- [HTTP Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers) — MDN reference

<!-- 
Sources Consulted:
- OpenAI Rate Limit Headers: https://platform.openai.com/docs/guides/rate-limits/rate-limit-headers
- Anthropic Rate Limits: https://docs.anthropic.com/en/api/rate-limits
-->
