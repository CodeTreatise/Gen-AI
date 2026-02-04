---
title: "Failure Handling"
---

# Failure Handling

## Introduction

Production embedding systems face multiple failure modes: embedding API rate limits, vector database timeouts, network partitions, and service degradation. Robust failure handling ensures your system remains functional—even in degraded mode—rather than failing completely.

This lesson covers retry strategies, circuit breakers, graceful degradation, and fallback patterns for embedding systems.

### What We'll Cover

- Common failure modes
- Retry with exponential backoff
- Circuit breaker pattern
- Graceful degradation strategies
- Fallback patterns
- Error classification and handling

### Prerequisites

- Understanding of [embedding pipeline architecture](./01-embedding-pipeline-architecture.md)
- Familiarity with distributed systems failure modes
- Basic knowledge of HTTP error codes

---

## Common Failure Modes

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding System Failure Points                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Client  │───▶│  Your    │───▶│ Embedding│───▶│ Vector   │ │
│  │  Request │    │  Service │    │ API      │    │ Database │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Failures │    │ Failures │    │ Failures │    │ Failures │ │
│  │──────────│    │──────────│    │──────────│    │──────────│ │
│  │• Timeout │    │• OOM     │    │• 429 Rate│    │• Timeout │ │
│  │• Invalid │    │• Crash   │    │• 500 Err │    │• 503     │ │
│  │  input   │    │• Deadlock│    │• Timeout │    │• Node    │ │
│  └──────────┘    └──────────┘    │• 401 Auth│    │  failure │ │
│                                  └──────────┘    └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Error Classification

| Error Type | HTTP Codes | Retry? | Action |
|------------|------------|--------|--------|
| Client error | 400, 401, 403, 404 | No | Fix request, don't retry |
| Rate limited | 429 | Yes (with backoff) | Back off, respect limits |
| Server error | 500, 502, 503, 504 | Yes | Retry with backoff |
| Timeout | - | Yes | Retry with backoff |
| Network error | - | Yes | Retry with backoff |

---

## Retry with Exponential Backoff

### Implementation

```python
import time
import random
from typing import Callable, TypeVar, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry behavior."""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay with exponential backoff and optional jitter.
    
    Formula: min(base_delay * (2 ** attempt), max_delay) + jitter
    """
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    if config.jitter:
        # Add random jitter: 0-100% of delay
        jitter = random.uniform(0, delay)
        delay += jitter
    
    return delay

def retry_with_backoff(
    config: RetryConfig = RetryConfig(),
    retryable_exceptions: tuple = (Exception,),
    non_retryable_exceptions: tuple = ()
):
    """
    Decorator for retry with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except non_retryable_exceptions as e:
                    # Don't retry these
                    logger.error(f"Non-retryable error: {e}")
                    raise
                
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries + 1} attempts failed. "
                            f"Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# Example usage
class EmbeddingAPIError(Exception):
    """Base exception for embedding API errors."""
    pass

class RateLimitError(EmbeddingAPIError):
    """Rate limit exceeded (429)."""
    pass

class ServerError(EmbeddingAPIError):
    """Server error (5xx)."""
    pass

class AuthenticationError(EmbeddingAPIError):
    """Authentication failed (401/403)."""
    pass

@retry_with_backoff(
    config=RetryConfig(max_retries=3, base_delay=1.0),
    retryable_exceptions=(RateLimitError, ServerError, TimeoutError),
    non_retryable_exceptions=(AuthenticationError,)
)
def call_embedding_api(text: str) -> list:
    """Call embedding API with retry logic."""
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"input": text, "model": "text-embedding-3-small"}
    )
    
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded")
    elif response.status_code >= 500:
        raise ServerError(f"Server error: {response.status_code}")
    elif response.status_code in (401, 403):
        raise AuthenticationError("Invalid API key")
    
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
```

### Async Retry Implementation

```python
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

async def async_retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = RetryConfig(),
    retryable_exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """
    Async retry with exponential backoff.
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_retries:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Async attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
    
    raise last_exception


# Usage example
async def embed_with_retry(text: str) -> list:
    return await async_retry_with_backoff(
        embedding_service.embed,
        text,
        config=RetryConfig(max_retries=3),
        retryable_exceptions=(RateLimitError, ServerError)
    )
```

---

## Circuit Breaker Pattern

### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              Circuit Breaker State Machine                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────┐                              │
│                    │   CLOSED    │                              │
│                    │  (Normal)   │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│         Failures exceed   │                                     │
│         threshold         │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │    OPEN     │                              │
│                    │  (Failing)  │                              │
│                    │             │                              │
│                    │ Fail fast,  │                              │
│                    │ don't call  │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│         After timeout     │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│         Success    │  HALF-OPEN  │    Failure                   │
│         ┌──────────│  (Testing)  │──────────┐                   │
│         │          └─────────────┘          │                   │
│         │                                    │                   │
│         ▼                                    ▼                   │
│    ┌─────────────┐                    ┌─────────────┐           │
│    │   CLOSED    │                    │    OPEN     │           │
│    └─────────────┘                    └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 3        # Successes to close from half-open
    timeout: float = 30.0             # Seconds before half-open
    half_open_max_calls: int = 3      # Test calls in half-open

class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    """
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = CircuitBreakerConfig()
    ):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker.
        """
        with self._lock:
            if not self._can_execute():
                raise CircuitOpenError(
                    f"Circuit {self.name} is open. "
                    f"Retry after {self._time_until_half_open():.1f}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """Check if call is allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._timeout_expired():
                self._transition_to_half_open()
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to open state."""
        logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
        self.state = CircuitState.OPEN
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        logger.info(f"Circuit {self.name} entering half-open state")
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        logger.info(f"Circuit {self.name} closed after successful recovery")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _timeout_expired(self) -> bool:
        """Check if open timeout has expired."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _time_until_half_open(self) -> float:
        """Time remaining until circuit enters half-open."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.config.timeout - elapsed)
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_half_open": self._time_until_half_open() if self.state == CircuitState.OPEN else None
        }

class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""
    pass
```

### Using Circuit Breakers

```python
# Create circuit breakers for external services
embedding_circuit = CircuitBreaker(
    "embedding_api",
    CircuitBreakerConfig(failure_threshold=5, timeout=30)
)

vectordb_circuit = CircuitBreaker(
    "vector_db",
    CircuitBreakerConfig(failure_threshold=3, timeout=60)
)

class ResilientEmbeddingService:
    """
    Embedding service with circuit breaker protection.
    """
    def __init__(self, embedding_client, vector_db):
        self.embedder = embedding_client
        self.vector_db = vector_db
    
    def embed(self, text: str) -> list:
        """Embed with circuit breaker protection."""
        return embedding_circuit.call(self.embedder.embed, text)
    
    def search(self, query_embedding: list, top_k: int = 10) -> list:
        """Search with circuit breaker protection."""
        return vectordb_circuit.call(
            self.vector_db.search,
            query_embedding,
            top_k=top_k
        )
    
    def get_health(self) -> dict:
        """Get health status of all circuits."""
        return {
            "embedding_api": embedding_circuit.get_status(),
            "vector_db": vectordb_circuit.get_status()
        }
```

---

## Graceful Degradation

### Degradation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Cached results | Return stale cached data | Read-heavy workloads |
| Reduced quality | Fewer results, skip reranking | High load periods |
| Fallback service | Use backup embedding model | Primary API down |
| Feature toggle | Disable non-essential features | Partial outage |

### Implementation

```python
from enum import Enum
from dataclasses import dataclass

class DegradationLevel(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    OFFLINE = "offline"

@dataclass
class DegradationConfig:
    skip_reranking: bool = False
    reduce_top_k: bool = False
    use_cached_only: bool = False
    use_fallback_model: bool = False

class GracefulDegradationService:
    """
    Search service with graceful degradation.
    """
    def __init__(
        self,
        primary_embedder,
        fallback_embedder,
        vector_db,
        cache,
        reranker
    ):
        self.primary_embedder = primary_embedder
        self.fallback_embedder = fallback_embedder
        self.vector_db = vector_db
        self.cache = cache
        self.reranker = reranker
        
        self.degradation_level = DegradationLevel.NORMAL
        self._circuit_embedding = CircuitBreaker("embedding")
        self._circuit_vectordb = CircuitBreaker("vectordb")
        self._circuit_reranker = CircuitBreaker("reranker")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True
    ) -> dict:
        """
        Search with automatic degradation based on service health.
        """
        config = self._get_degradation_config()
        
        # Step 1: Get embedding
        try:
            if config.use_fallback_model:
                embedding = self.fallback_embedder.embed(query)
            else:
                embedding = self._circuit_embedding.call(
                    self.primary_embedder.embed, query
                )
        except CircuitOpenError:
            # Fallback to cached embedding or error
            cached = self.cache.get_embedding(query)
            if cached:
                embedding = cached
                config.use_cached_only = True
            else:
                return self._offline_response(query)
        
        # Step 2: Vector search
        search_top_k = top_k // 2 if config.reduce_top_k else top_k * 3
        
        try:
            results = self._circuit_vectordb.call(
                self.vector_db.search,
                embedding,
                top_k=search_top_k
            )
        except CircuitOpenError:
            # Return cached results
            cached_results = self.cache.get_search_results(query)
            if cached_results:
                return {
                    "results": cached_results,
                    "degraded": True,
                    "degradation_reason": "vector_db_unavailable"
                }
            return self._offline_response(query)
        
        # Step 3: Optional reranking
        if rerank and not config.skip_reranking and self.reranker:
            try:
                results = self._circuit_reranker.call(
                    self.reranker.rerank,
                    query,
                    results[:top_k * 2]
                )
            except CircuitOpenError:
                # Skip reranking, use vector similarity only
                logger.warning("Reranking unavailable, using vector scores only")
        
        final_results = results[:top_k]
        
        return {
            "results": final_results,
            "degraded": self.degradation_level != DegradationLevel.NORMAL,
            "degradation_level": self.degradation_level.value
        }
    
    def _get_degradation_config(self) -> DegradationConfig:
        """Determine degradation config based on circuit states."""
        config = DegradationConfig()
        
        # Check each circuit
        if self._circuit_embedding.state != CircuitState.CLOSED:
            config.use_fallback_model = True
            self.degradation_level = DegradationLevel.DEGRADED
        
        if self._circuit_reranker.state != CircuitState.CLOSED:
            config.skip_reranking = True
            self.degradation_level = DegradationLevel.DEGRADED
        
        if self._circuit_vectordb.state != CircuitState.CLOSED:
            config.use_cached_only = True
            self.degradation_level = DegradationLevel.MINIMAL
        
        return config
    
    def _offline_response(self, query: str) -> dict:
        """Return offline response when all options exhausted."""
        return {
            "results": [],
            "degraded": True,
            "degradation_level": DegradationLevel.OFFLINE.value,
            "message": "Search temporarily unavailable. Please try again later."
        }
```

---

## Fallback Patterns

### Multi-Level Fallback

```python
class FallbackEmbeddingService:
    """
    Embedding service with multiple fallback levels.
    """
    def __init__(
        self,
        primary_client,      # OpenAI
        secondary_client,    # Cohere (backup)
        local_model,         # Local sentence-transformers
        cache
    ):
        self.primary = primary_client
        self.secondary = secondary_client
        self.local = local_model
        self.cache = cache
        
        self.providers = [
            ("primary", self.primary),
            ("secondary", self.secondary),
            ("local", self.local)
        ]
    
    def embed(self, text: str) -> dict:
        """
        Embed with cascading fallback.
        
        Order: Cache → Primary → Secondary → Local
        """
        # Try cache first
        cached = self.cache.get(text)
        if cached:
            return {
                "embedding": cached,
                "provider": "cache",
                "fallback_level": 0
            }
        
        # Try each provider in order
        last_error = None
        for level, (name, client) in enumerate(self.providers):
            try:
                embedding = client.embed(text)
                
                # Cache successful embedding
                self.cache.set(text, embedding)
                
                return {
                    "embedding": embedding,
                    "provider": name,
                    "fallback_level": level
                }
            
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {name} failed: {e}")
                continue
        
        # All providers failed
        raise AllProvidersFailedError(
            f"All embedding providers failed. Last error: {last_error}"
        )
    
    def embed_batch(self, texts: list) -> list:
        """
        Batch embed with per-text fallback.
        """
        results = []
        
        for text in texts:
            try:
                result = self.embed(text)
                results.append(result)
            except AllProvidersFailedError:
                results.append({
                    "embedding": None,
                    "error": "all_providers_failed"
                })
        
        return results

class AllProvidersFailedError(Exception):
    """Raised when all fallback providers have failed."""
    pass
```

### Timeout-Based Fallback

```python
import asyncio
from typing import Optional

class TimeoutFallbackService:
    """
    Service with timeout-based fallback to faster option.
    """
    def __init__(
        self,
        primary_service,      # Full quality, slower
        fallback_service,     # Lower quality, faster
        primary_timeout: float = 2.0
    ):
        self.primary = primary_service
        self.fallback = fallback_service
        self.timeout = primary_timeout
    
    async def search(self, query: str, top_k: int = 10) -> dict:
        """
        Try primary with timeout, fallback to faster service.
        """
        try:
            # Try primary with timeout
            result = await asyncio.wait_for(
                self.primary.search(query, top_k),
                timeout=self.timeout
            )
            return {
                "results": result,
                "source": "primary",
                "quality": "full"
            }
        
        except asyncio.TimeoutError:
            logger.warning(f"Primary search timed out after {self.timeout}s")
            
            # Fallback to faster service
            result = await self.fallback.search(query, top_k)
            return {
                "results": result,
                "source": "fallback",
                "quality": "reduced"
            }
```

---

## Health Check Endpoint

```python
from fastapi import FastAPI, Response
from enum import Enum

app = FastAPI()

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    """
    checks = {
        "embedding_api": check_embedding_health(),
        "vector_db": check_vectordb_health(),
        "cache": check_cache_health()
    }
    
    # Determine overall status
    statuses = [c["status"] for c in checks.values()]
    
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
        http_status = 200
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
        http_status = 503
    else:
        overall = HealthStatus.DEGRADED
        http_status = 200  # Still operational
    
    return Response(
        content={
            "status": overall.value,
            "checks": {k: v["status"].value for k, v in checks.items()},
            "details": checks
        },
        status_code=http_status
    )

def check_embedding_health() -> dict:
    """Check embedding API health."""
    circuit = embedding_circuit.get_status()
    
    if circuit["state"] == "closed":
        return {"status": HealthStatus.HEALTHY, "circuit": circuit}
    elif circuit["state"] == "half_open":
        return {"status": HealthStatus.DEGRADED, "circuit": circuit}
    else:
        return {"status": HealthStatus.UNHEALTHY, "circuit": circuit}
```

---

## Summary

✅ **Classify errors: retry server errors, don't retry client errors**  
✅ **Use exponential backoff with jitter to avoid thundering herd**  
✅ **Implement circuit breakers to fail fast during outages**  
✅ **Design graceful degradation: cached results, skip reranking, fallback models**  
✅ **Provide health endpoints that reflect actual service capability**

---

**Next:** [Testing Strategies →](./07-testing-strategies.md)

---

<!-- 
Sources Consulted:
- Pinecone Error Handling: https://docs.pinecone.io/troubleshooting/handling-errors
- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- Retry Best Practices: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
-->
