---
title: "Retry Strategies"
---

# Retry Strategies

## Introduction

Transient errors like rate limits and server issues are best handled with retries. Proper retry strategies use exponential backoff with jitter to avoid overwhelming the API while recovering gracefully.

### What We'll Cover

- Why exponential backoff works
- Implementing backoff algorithms
- Adding jitter for distributed systems
- Respecting retry-after headers
- Maximum retry limits

### Prerequisites

- Common API errors
- Error response parsing

---

## Why Exponential Backoff

### The Problem with Fixed Retries

```python
# ❌ Bad: Fixed retry timing
def bad_retry(client, request_fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return request_fn()
        except RateLimitError:
            time.sleep(1)  # Always 1 second - creates thundering herd
    raise Exception("Max retries exceeded")
```

When many clients retry at the same interval, they all hit the API simultaneously, making congestion worse.

### Exponential Backoff Solution

```mermaid
flowchart LR
    A[Attempt 1] -->|Fail| W1[Wait 1s]
    W1 --> B[Attempt 2]
    B -->|Fail| W2[Wait 2s]
    W2 --> C[Attempt 3]
    C -->|Fail| W4[Wait 4s]
    W4 --> D[Attempt 4]
    D -->|Fail| W8[Wait 8s]
    W8 --> E[Attempt 5]
```

Each retry waits exponentially longer, giving the API time to recover.

---

## Basic Exponential Backoff

```python
import time
from openai import OpenAI, RateLimitError, InternalServerError

client = OpenAI()

def retry_with_backoff(
    request_fn,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """Execute request with exponential backoff on retryable errors."""
    
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return request_fn()
        except (RateLimitError, InternalServerError) as e:
            last_error = e
            
            if attempt == max_retries - 1:
                break  # Don't sleep on last attempt
            
            # Calculate delay (capped at max)
            sleep_time = min(delay, max_delay)
            print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time:.1f}s...")
            
            time.sleep(sleep_time)
            delay *= backoff_factor
    
    raise last_error


# Usage
def make_request():
    return client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )

response = retry_with_backoff(make_request)
```

---

## Adding Jitter

Jitter adds randomness to prevent synchronized retries:

```python
import random

def retry_with_jitter(
    request_fn,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.5
):
    """Exponential backoff with jitter for distributed systems."""
    
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return request_fn()
        except (RateLimitError, InternalServerError) as e:
            last_error = e
            
            if attempt == max_retries - 1:
                break
            
            # Add jitter: ±jitter_factor of the delay
            jitter = delay * jitter_factor * (2 * random.random() - 1)
            sleep_time = min(delay + jitter, max_delay)
            sleep_time = max(0.1, sleep_time)  # At least 100ms
            
            print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time:.2f}s...")
            
            time.sleep(sleep_time)
            delay *= backoff_factor
    
    raise last_error
```

### Jitter Strategies

```python
def calculate_jitter(base_delay: float, strategy: str = "full") -> float:
    """Calculate delay with different jitter strategies."""
    
    if strategy == "full":
        # Full jitter: random between 0 and delay
        return random.uniform(0, base_delay)
    
    elif strategy == "equal":
        # Equal jitter: half base + random half
        return base_delay / 2 + random.uniform(0, base_delay / 2)
    
    elif strategy == "decorrelated":
        # Decorrelated: random between delay and 3x delay
        return random.uniform(base_delay, base_delay * 3)
    
    else:
        return base_delay
```

---

## Respecting Retry-After

APIs may include a `retry-after` header:

```python
def retry_with_header_respect(
    request_fn,
    max_retries: int = 5,
    default_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Retry respecting retry-after headers when present."""
    
    last_error = None
    calculated_delay = default_delay
    
    for attempt in range(max_retries):
        try:
            return request_fn()
        except RateLimitError as e:
            last_error = e
            
            if attempt == max_retries - 1:
                break
            
            # Check for retry-after header
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
            
            if retry_after:
                try:
                    sleep_time = int(retry_after)
                    print(f"Server requested wait: {sleep_time}s")
                except ValueError:
                    sleep_time = calculated_delay
            else:
                sleep_time = calculated_delay
            
            time.sleep(sleep_time)
            calculated_delay *= backoff_factor
        
        except InternalServerError as e:
            # Server errors use calculated backoff
            last_error = e
            
            if attempt == max_retries - 1:
                break
            
            time.sleep(calculated_delay)
            calculated_delay *= backoff_factor
    
    raise last_error
```

---

## Retry Configuration

### Dataclass Configuration

```python
from dataclasses import dataclass, field
from typing import Set, Type

@dataclass
class RetryConfig:
    max_retries: int = 5
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.5
    retryable_errors: Set[Type[Exception]] = field(default_factory=lambda: {
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError
    })
    
    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """Config for aggressive retrying."""
        return cls(
            max_retries=10,
            initial_delay=0.5,
            backoff_factor=1.5,
            max_delay=30.0
        )
    
    @classmethod
    def conservative(cls) -> "RetryConfig":
        """Config for conservative retrying."""
        return cls(
            max_retries=3,
            initial_delay=2.0,
            backoff_factor=3.0,
            max_delay=120.0
        )
```

### Configurable Retry Function

```python
def retry_request(
    request_fn,
    config: RetryConfig = None
):
    """Execute request with configurable retry behavior."""
    
    config = config or RetryConfig()
    last_error = None
    delay = config.initial_delay
    
    for attempt in range(config.max_retries):
        try:
            return request_fn()
        except tuple(config.retryable_errors) as e:
            last_error = e
            
            if attempt == config.max_retries - 1:
                break
            
            # Calculate delay with jitter
            jitter = delay * config.jitter_factor * (2 * random.random() - 1)
            sleep_time = min(delay + jitter, config.max_delay)
            
            time.sleep(max(0.1, sleep_time))
            delay *= config.backoff_factor
        except Exception as e:
            # Non-retryable error
            raise
    
    raise last_error


# Usage
response = retry_request(
    make_request,
    config=RetryConfig.aggressive()
)
```

---

## Retry Decorator

```python
from functools import wraps

def with_retry(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (RateLimitError, InternalServerError)
):
    """Decorator to add retry logic to any function."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        jitter = random.uniform(0, delay * 0.5)
                        time.sleep(delay + jitter)
                        delay *= backoff_factor
            
            raise last_error
        return wrapper
    return decorator


# Usage
@with_retry(max_retries=3)
def get_completion(prompt: str):
    return client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
```

---

## Async Retry

```python
import asyncio

async def async_retry_with_backoff(
    async_request_fn,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Async version of retry with backoff."""
    
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await async_request_fn()
        except (RateLimitError, InternalServerError) as e:
            last_error = e
            
            if attempt == max_retries - 1:
                break
            
            jitter = random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay + jitter)
            delay *= backoff_factor
    
    raise last_error


# Usage
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def get_async_completion():
    return await async_retry_with_backoff(
        lambda: async_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}]
        )
    )
```

---

## JavaScript Implementation

```javascript
async function retryWithBackoff(
    requestFn,
    {
        maxRetries = 5,
        initialDelay = 1000,
        backoffFactor = 2,
        maxDelay = 60000,
        jitterFactor = 0.5,
        retryableErrors = ['rate_limit_error', 'server_error']
    } = {}
) {
    let delay = initialDelay;
    let lastError;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await requestFn();
        } catch (error) {
            lastError = error;
            
            // Check if retryable
            const isRetryable = 
                error.status === 429 ||
                error.status >= 500 ||
                retryableErrors.includes(error.error?.type);
            
            if (!isRetryable || attempt === maxRetries - 1) {
                throw error;
            }
            
            // Check for retry-after header
            let sleepTime = delay;
            const retryAfter = error.headers?.get?.('retry-after');
            if (retryAfter) {
                sleepTime = parseInt(retryAfter) * 1000;
            } else {
                // Add jitter
                const jitter = delay * jitterFactor * (Math.random() * 2 - 1);
                sleepTime = Math.min(delay + jitter, maxDelay);
            }
            
            console.log(`Attempt ${attempt + 1} failed. Retrying in ${sleepTime}ms...`);
            
            await new Promise(resolve => setTimeout(resolve, sleepTime));
            delay *= backoffFactor;
        }
    }
    
    throw lastError;
}

// Usage
const response = await retryWithBackoff(
    () => openai.chat.completions.create({
        model: 'gpt-4.1',
        messages: [{ role: 'user', content: 'Hello' }]
    }),
    { maxRetries: 3 }
);
```

---

## Retry with Logging

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def retry_with_logging(
    request_fn,
    request_id: str = None,
    config: RetryConfig = None
):
    """Retry with detailed logging for monitoring."""
    
    config = config or RetryConfig()
    request_id = request_id or str(datetime.now().timestamp())
    delay = config.initial_delay
    
    for attempt in range(config.max_retries):
        try:
            result = request_fn()
            
            if attempt > 0:
                logger.info(
                    f"Request {request_id} succeeded after {attempt + 1} attempts"
                )
            
            return result
        
        except tuple(config.retryable_errors) as e:
            logger.warning(
                f"Request {request_id} attempt {attempt + 1} failed: "
                f"{type(e).__name__}: {e.message if hasattr(e, 'message') else str(e)}"
            )
            
            if attempt == config.max_retries - 1:
                logger.error(
                    f"Request {request_id} failed after {config.max_retries} attempts"
                )
                raise
            
            jitter = random.uniform(0, delay * config.jitter_factor)
            sleep_time = min(delay + jitter, config.max_delay)
            
            logger.debug(f"Request {request_id} waiting {sleep_time:.2f}s before retry")
            
            time.sleep(sleep_time)
            delay *= config.backoff_factor
        
        except Exception as e:
            logger.error(
                f"Request {request_id} failed with non-retryable error: {e}"
            )
            raise
```

---

## Hands-on Exercise

### Your Task

Implement a retry function with configurable backoff and jitter.

### Requirements

1. Exponential backoff with configurable factor
2. Full jitter (random between 0 and delay)
3. Respect retry-after header
4. Maximum delay cap
5. Logging of retry attempts

### Expected Result

```
Attempt 1 failed. Error: RateLimitError
Waiting 1.23s (retry-after: not specified)
Attempt 2 failed. Error: RateLimitError
Waiting 2.87s (retry-after: not specified)
Attempt 3 succeeded.
```

<details>
<summary>💡 Hints</summary>

- Use random.uniform(0, delay) for full jitter
- Check e.response.headers for retry-after
- min() caps the delay at max_delay
</details>

<details>
<summary>✅ Solution</summary>

```python
import time
import random
import logging

logger = logging.getLogger(__name__)

def smart_retry(
    request_fn,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """Retry with exponential backoff, jitter, and retry-after support."""
    
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = request_fn()
            if attempt > 0:
                logger.info(f"Attempt {attempt + 1} succeeded.")
            return result
        
        except (RateLimitError, InternalServerError) as e:
            last_error = e
            
            logger.warning(f"Attempt {attempt + 1} failed. Error: {type(e).__name__}")
            
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed.")
                break
            
            # Check for retry-after header
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
            
            if retry_after:
                try:
                    sleep_time = float(retry_after)
                    logger.info(f"Waiting {sleep_time}s (retry-after: {retry_after})")
                except ValueError:
                    sleep_time = delay
            else:
                # Full jitter: random between 0 and delay
                sleep_time = random.uniform(0, min(delay, max_delay))
                logger.info(f"Waiting {sleep_time:.2f}s (retry-after: not specified)")
            
            time.sleep(sleep_time)
            delay *= backoff_factor
    
    raise last_error


# Test
def make_failing_request():
    """Simulates a request that might fail."""
    return client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}]
    )

try:
    response = smart_retry(make_failing_request, max_retries=3)
    print(f"Success: {response.choices[0].message.content}")
except Exception as e:
    print(f"Failed after all retries: {e}")
```

</details>

---

## Summary

✅ Exponential backoff prevents overwhelming APIs during outages  
✅ Jitter spreads retry attempts to avoid thundering herd  
✅ Always respect retry-after headers when present  
✅ Cap maximum delay to prevent excessive waits  
✅ Log retry attempts for monitoring and debugging

**Next:** [Graceful Degradation](./04-graceful-degradation.md)

---

## Further Reading

- [Exponential Backoff](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) — AWS best practices
- [Retry Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/retry) — Azure patterns
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits) — Official guide

<!-- 
Sources Consulted:
- AWS Exponential Backoff: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- OpenAI Rate Limits: https://platform.openai.com/docs/guides/rate-limits
-->
