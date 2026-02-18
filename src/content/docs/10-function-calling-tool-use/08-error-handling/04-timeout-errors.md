---
title: "Timeout and Rate Limit Errors"
---

# Timeout and Rate Limit Errors

## Introduction

Timeouts and rate limits are the most common transient errors in production AI applications. A timeout occurs when a function call takes too long to complete — the external API is slow, the database query is complex, or the network is congested. A rate limit occurs when you've sent too many requests to an API in a given time window. Both are recoverable, but they require different strategies.

Timeout handling requires decisions: Do we retry? Do we return partial results? Do we tell the user to wait? Rate limit handling requires patience: we must back off, wait for our quota to reset, and retry. Getting these wrong leads to cascading failures — retrying too aggressively makes rate limits worse, and retrying timeouts without backoff can overwhelm already-struggling services.

### What we'll cover

- Implementing execution timeouts for function calls
- Detecting and handling API rate limits across providers
- Exponential backoff with jitter
- Partial result handling when functions time out mid-execution
- Provider-specific rate limit responses and headers

### Prerequisites

- Understanding of [execution failures](./03-execution-failures.md) and the `safe_execute` pattern
- Familiarity with [async programming](../../02-python-for-ai-development/09-async-programming/00-async-programming.md)
- Understanding of the [agentic loop](../07-multi-turn-function-calling/01-conversation-flow.md)

---

## Timeout implementation patterns

There are several ways to enforce timeouts on function calls in Python. Each has trade-offs:

### Thread-based timeouts

The most portable approach — works for both I/O-bound and CPU-bound functions:

```python
import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TimeoutResult:
    """Result from a function call with timeout enforcement."""
    success: bool
    result: Any = None
    timed_out: bool = False
    elapsed_seconds: float = 0.0
    error_message: str = ""


def execute_with_timeout(
    handler: Callable,
    arguments: dict,
    timeout_seconds: float = 30.0
) -> TimeoutResult:
    """Execute a function with a hard timeout deadline."""
    start = time.monotonic()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(handler, **arguments)
        
        try:
            result = future.result(timeout=timeout_seconds)
            elapsed = time.monotonic() - start
            return TimeoutResult(
                success=True,
                result=result,
                elapsed_seconds=elapsed
            )
        except concurrent.futures.TimeoutError:
            elapsed = time.monotonic() - start
            future.cancel()
            return TimeoutResult(
                success=False,
                timed_out=True,
                elapsed_seconds=elapsed,
                error_message=(
                    f"Function timed out after {timeout_seconds:.1f}s. "
                    f"The operation may still be running. "
                    f"Try again or use a simpler query."
                )
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return TimeoutResult(
                success=False,
                elapsed_seconds=elapsed,
                error_message=f"Execution failed: {type(exc).__name__}: {exc}"
            )


# Test: function that takes too long
def slow_database_query(query: str, limit: int = 100) -> dict:
    """Simulates a slow database query."""
    time.sleep(10)  # Very slow
    return {"rows": []}

result = execute_with_timeout(
    slow_database_query,
    {"query": "SELECT * FROM large_table", "limit": 100},
    timeout_seconds=3.0
)

print(f"Success: {result.success}")
print(f"Timed out: {result.timed_out}")
print(f"Elapsed: {result.elapsed_seconds:.1f}s")
print(f"Message: {result.error_message}")
```

**Output:**
```
Success: False
Timed out: True
Elapsed: 3.0s
Message: Function timed out after 3.0s. The operation may still be running. Try again or use a simpler query.
```

### Async timeouts

For async function handlers, use `asyncio.wait_for`:

```python
import asyncio


async def execute_with_timeout_async(
    handler: Callable,
    arguments: dict,
    timeout_seconds: float = 30.0
) -> TimeoutResult:
    """Execute an async function with timeout."""
    start = time.monotonic()
    
    try:
        result = await asyncio.wait_for(
            handler(**arguments),
            timeout=timeout_seconds
        )
        elapsed = time.monotonic() - start
        return TimeoutResult(
            success=True,
            result=result,
            elapsed_seconds=elapsed
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return TimeoutResult(
            success=False,
            timed_out=True,
            elapsed_seconds=elapsed,
            error_message=(
                f"Async function timed out after {timeout_seconds:.1f}s."
            )
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return TimeoutResult(
            success=False,
            elapsed_seconds=elapsed,
            error_message=f"{type(exc).__name__}: {exc}"
        )


# Example async function
async def async_api_call(endpoint: str) -> dict:
    await asyncio.sleep(10)  # Simulates slow API
    return {"data": "response"}

# Run it
result = asyncio.run(execute_with_timeout_async(
    async_api_call,
    {"endpoint": "/api/data"},
    timeout_seconds=2.0
))
print(f"Async timeout: timed_out={result.timed_out}, "
      f"elapsed={result.elapsed_seconds:.1f}s")
```

**Output:**
```
Async timeout: timed_out=True, elapsed=2.0s
```

---

## Rate limit detection and handling

Each AI provider returns rate limit information differently. Here's how to detect and respond:

### Provider rate limit responses

| Provider | HTTP Status | Error Type | Retry Header | Unique Behavior |
|----------|------------|------------|--------------|-----------------|
| **OpenAI** | 429 | `RateLimitError` | `Retry-After` | Separate limits for tokens-per-minute and requests-per-minute |
| **Anthropic** | 429 | `rate_limit_error` | `retry-after` | Also has 529 `overloaded_error` for server overload |
| **Gemini** | 429 | `RESOURCE_EXHAUSTED` | — | Per-model, per-region quotas |

### Detecting rate limits in SDK responses

```python
import time
import random


class RateLimitHandler:
    """Handle rate limit errors across providers."""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def execute_with_rate_limit_handling(
        self,
        provider: str,
        api_call: Callable,
        *args,
        **kwargs
    ) -> dict:
        """Execute an API call with automatic rate limit retry."""
        for attempt in range(self.max_retries + 1):
            try:
                return {"success": True, "result": api_call(*args, **kwargs)}
            
            except Exception as exc:
                if not self._is_rate_limit(exc, provider):
                    # Not a rate limit — don't retry
                    return {
                        "success": False,
                        "error": str(exc),
                        "retryable": False
                    }
                
                if attempt >= self.max_retries:
                    return {
                        "success": False,
                        "error": f"Rate limited after {self.max_retries} retries",
                        "retryable": True
                    }
                
                # Calculate backoff with jitter
                delay = self._calculate_backoff(attempt, exc, provider)
                print(f"  ⏳ Rate limited. Waiting {delay:.1f}s "
                      f"(attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(delay)
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def _is_rate_limit(self, exc: Exception, provider: str) -> bool:
        """Check if an exception is a rate limit error."""
        exc_str = str(type(exc).__name__).lower()
        msg = str(exc).lower()
        
        # OpenAI
        if provider == "openai":
            try:
                import openai
                return isinstance(exc, openai.RateLimitError)
            except ImportError:
                return "ratelimit" in exc_str or "429" in msg
        
        # Anthropic — also check for 529 overloaded
        if provider == "anthropic":
            try:
                import anthropic
                return isinstance(exc, (
                    anthropic.RateLimitError,
                    anthropic.InternalServerError  # 529 overloaded
                ))
            except ImportError:
                return "429" in msg or "529" in msg or "rate_limit" in msg
        
        # Gemini
        if provider == "gemini":
            return "resource_exhausted" in msg or "429" in msg
        
        return False
    
    def _calculate_backoff(
        self, attempt: int, exc: Exception, provider: str
    ) -> float:
        """Calculate delay with exponential backoff + jitter."""
        # Try to extract retry-after from headers
        retry_after = self._extract_retry_after(exc)
        
        if retry_after:
            # Use server-provided delay + small jitter
            return retry_after + random.uniform(0, 1)
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s (capped at 60s)
        delay = min(self.base_delay * (2 ** attempt), 60)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, delay * 0.5)
        
        return delay + jitter
    
    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """Extract Retry-After header from exception if available."""
        # OpenAI SDK stores headers on the response
        response = getattr(exc, 'response', None)
        if response:
            headers = getattr(response, 'headers', {})
            retry_after = headers.get('retry-after') or headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return None


# Simulate rate limiting
call_count = 0

def rate_limited_api(query: str) -> dict:
    """Simulates an API that rate-limits the first 2 calls."""
    global call_count
    call_count += 1
    if call_count <= 2:
        raise Exception("429: Rate limit exceeded. Please retry after 1 second.")
    return {"results": [f"Result for '{query}'"]}


handler = RateLimitHandler(max_retries=3, base_delay=1.0)
call_count = 0

result = handler.execute_with_rate_limit_handling(
    "openai",
    rate_limited_api,
    query="test search"
)
print(f"Final result: {result}")
```

**Output:**
```
  ⏳ Rate limited. Waiting 1.3s (attempt 1/3)...
  ⏳ Rate limited. Waiting 2.7s (attempt 2/3)...
Final result: {'success': True, 'result': {'results': ["Result for 'test search'"]}}
```

---

## Exponential backoff with jitter

The standard pattern for retrying transient errors. Without jitter, multiple clients retry at the same time (thundering herd), making the problem worse:

```python
import random
import time


def backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_strategy: str = "full"
) -> float:
    """Calculate backoff delay with jitter.
    
    Strategies:
    - "full": Random between 0 and exponential delay
    - "equal": Half exponential + half random
    - "decorrelated": Random between base and 3x previous delay
    """
    exp_delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter_strategy == "full":
        # Full jitter: uniform random [0, exp_delay]
        return random.uniform(0, exp_delay)
    
    elif jitter_strategy == "equal":
        # Equal jitter: half fixed + half random
        half = exp_delay / 2
        return half + random.uniform(0, half)
    
    elif jitter_strategy == "decorrelated":
        # Decorrelated: random [base, 3 * exp_delay]
        return random.uniform(base_delay, min(3 * exp_delay, max_delay))
    
    return exp_delay  # No jitter


# Show the delay progression
print("Attempt | Full Jitter | Equal Jitter | No Jitter")
print("--------|-------------|--------------|----------")
random.seed(42)
for i in range(6):
    full = backoff_with_jitter(i, jitter_strategy="full")
    equal = backoff_with_jitter(i, jitter_strategy="equal")
    none = min(1.0 * (2 ** i), 60.0)
    print(f"   {i}    |   {full:5.1f}s    |    {equal:5.1f}s     |  {none:5.1f}s")
```

**Output:**
```
Attempt | Full Jitter | Equal Jitter | No Jitter
--------|-------------|--------------|----------
   0    |    0.6s    |     0.8s     |   1.0s
   1    |    0.1s    |     1.5s     |   2.0s
   2    |    3.5s    |     2.7s     |   4.0s
   3    |    2.4s    |     5.1s     |   8.0s
   4    |   12.8s    |    10.3s     |  16.0s
   5    |   25.1s    |    22.7s     |  32.0s
```

> **Tip:** AWS recommends "full jitter" for most use cases. It provides the widest spread of retry times across clients, minimizing collision. "Equal jitter" guarantees a minimum delay, which can be better when you need *some* waiting but don't want to risk near-zero delays.

---

## Partial result handling

Some functions may have partial results when they time out. For example, a search function might have found 3 of 10 results before timing out:

```python
import threading
from typing import Any


class PartialResultCollector:
    """Collect partial results from a function that may time out."""
    
    def __init__(self):
        self._partial_results: list = []
        self._lock = threading.Lock()
        self._complete = False
    
    def add_result(self, item: Any) -> None:
        """Add a partial result (thread-safe)."""
        with self._lock:
            self._partial_results.append(item)
    
    def mark_complete(self) -> None:
        """Mark collection as complete."""
        self._complete = True
    
    @property
    def results(self) -> list:
        with self._lock:
            return list(self._partial_results)
    
    @property
    def is_complete(self) -> bool:
        return self._complete


def execute_with_partial_results(
    handler: Callable,
    arguments: dict,
    collector: PartialResultCollector,
    timeout_seconds: float = 5.0
) -> dict:
    """Execute a function, returning partial results on timeout."""
    
    def wrapped():
        handler(collector=collector, **arguments)
        collector.mark_complete()
    
    thread = threading.Thread(target=wrapped)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    partial = collector.results
    
    if collector.is_complete:
        return {
            "success": True,
            "complete": True,
            "results": partial,
            "count": len(partial)
        }
    else:
        return {
            "success": True,  # Partial success!
            "complete": False,
            "results": partial,
            "count": len(partial),
            "warning": (
                f"Retrieved {len(partial)} results before timeout. "
                f"More results may be available — try a narrower query."
            )
        }


# Simulate: function that produces results incrementally
def search_documents(query: str, collector: PartialResultCollector) -> None:
    """Simulates incremental search — finds results one at a time."""
    documents = [
        {"id": 1, "title": "Python Basics"},
        {"id": 2, "title": "Advanced Python"},
        {"id": 3, "title": "Python for AI"},
        {"id": 4, "title": "Machine Learning"},
        {"id": 5, "title": "Deep Learning"},
    ]
    
    for doc in documents:
        if query.lower() in doc["title"].lower():
            time.sleep(1.5)  # Each result takes 1.5s
            collector.add_result(doc)


collector = PartialResultCollector()
result = execute_with_partial_results(
    search_documents,
    {"query": "Python"},
    collector,
    timeout_seconds=4.0  # Will find ~2 of 3 matching docs
)

print(f"Complete: {result['complete']}")
print(f"Results found: {result['count']}")
for r in result['results']:
    print(f"  - {r['title']}")
if 'warning' in result:
    print(f"Warning: {result['warning']}")
```

**Output:**
```
Complete: False
Results found: 2
  - Python Basics
  - Advanced Python
Warning: Retrieved 2 results before timeout. More results may be available — try a narrower query.
```

> **🤖 AI Context:** Returning partial results is often better than returning nothing. The model can present the available results to the user and offer to search again with different parameters. This keeps the conversation productive even when things go wrong.

---

## Communicating timeouts and rate limits to the model

Format these errors so the model can make good decisions:

```python
import json


def format_timeout_for_model(function_name: str, timeout_result: TimeoutResult, 
                             partial_data: dict | None = None) -> dict:
    """Format a timeout error for model consumption."""
    response = {
        "error": True,
        "error_type": "timeout",
        "function": function_name,
        "message": timeout_result.error_message,
        "elapsed_seconds": round(timeout_result.elapsed_seconds, 1)
    }
    
    if partial_data and partial_data.get("results"):
        response["partial_results"] = partial_data["results"]
        response["message"] += (
            f" However, {len(partial_data['results'])} partial result(s) "
            f"were retrieved before the timeout."
        )
        response["suggestions"] = [
            "You can present the partial results to the user",
            "Try a narrower or simpler query",
            "Try again — the service may be faster now"
        ]
    else:
        response["suggestions"] = [
            "Try again with a simpler query",
            "Inform the user that the service is slow",
            "Try an alternative approach"
        ]
    
    return response


def format_rate_limit_for_model(
    function_name: str,
    retry_after: float | None = None
) -> dict:
    """Format a rate limit error for model consumption."""
    response = {
        "error": True,
        "error_type": "rate_limit",
        "function": function_name,
        "message": (
            f"The API for '{function_name}' is rate-limited. "
            f"Too many requests have been made."
        )
    }
    
    if retry_after:
        response["retry_after_seconds"] = retry_after
        response["message"] += (
            f" Try again after {retry_after:.0f} seconds."
        )
    
    response["suggestions"] = [
        "Inform the user there is a brief delay",
        "Try a different approach that doesn't require this API",
        "Batch multiple requests into fewer calls"
    ]
    
    return response


# Example outputs
timeout_msg = format_timeout_for_model(
    "search_documents",
    TimeoutResult(False, timed_out=True, elapsed_seconds=5.0,
                  error_message="Search timed out after 5.0s"),
    partial_data={"results": [{"title": "Python Basics"}]}
)
print("Timeout response:")
print(json.dumps(timeout_msg, indent=2))

print("\nRate limit response:")
rate_msg = format_rate_limit_for_model("get_weather", retry_after=30)
print(json.dumps(rate_msg, indent=2))
```

**Output:**
```
Timeout response:
{
  "error": true,
  "error_type": "timeout",
  "function": "search_documents",
  "message": "Search timed out after 5.0s However, 1 partial result(s) were retrieved before the timeout.",
  "elapsed_seconds": 5.0,
  "partial_results": [
    {"title": "Python Basics"}
  ],
  "suggestions": [
    "You can present the partial results to the user",
    "Try a narrower or simpler query",
    "Try again — the service may be faster now"
  ]
}

Rate limit response:
{
  "error": true,
  "error_type": "rate_limit",
  "function": "get_weather",
  "message": "The API for 'get_weather' is rate-limited. Too many requests have been made. Try again after 30 seconds.",
  "retry_after_seconds": 30,
  "suggestions": [
    "Inform the user there is a brief delay",
    "Try a different approach that doesn't require this API",
    "Batch multiple requests into fewer calls"
  ]
}
```

---

## Best practices

| Practice | Why it matters |
|----------|----------------|
| Always set execution timeouts | Prevents indefinite hangs in the agentic loop |
| Use exponential backoff with jitter for retries | Prevents thundering herd and distributes load |
| Respect `Retry-After` headers when present | The server knows better than your code when to retry |
| Return partial results when available | Better than nothing — keeps the conversation useful |
| Distinguish timeouts from rate limits | Different causes require different responses |
| Cap maximum retry delay at 60 seconds | Beyond this, inform the user rather than making them wait |

---

## Common pitfalls

| ❌ Mistake | ✅ Solution |
|-----------|-------------|
| No timeout on function execution | Always wrap with `concurrent.futures` (sync) or `asyncio.wait_for` (async) |
| Fixed-delay retries (e.g., always 5s) | Use exponential backoff: 1s, 2s, 4s, 8s, 16s |
| Retrying without jitter | Multiple clients retry simultaneously — add `random.uniform(0, delay/2)` |
| Retrying rate limits aggressively | Respect `Retry-After` headers; back off exponentially |
| Discarding partial results on timeout | Collect results incrementally and return what you have |
| Treating Anthropic 529 (overloaded) as a regular error | It's transient like a rate limit — retry with backoff |
| Not capping maximum backoff delay | Delays can grow to minutes; cap at 60s and inform the user |

---

## Hands-on exercise

### Your task

Build a `TimeoutAwareExecutor` that handles both timeouts and rate limits with proper backoff.

### Requirements

1. Implement execution with configurable timeout per function
2. Add rate limit detection that works for all three providers (check exception type/message)
3. Implement exponential backoff with full jitter
4. Return partial results when a function times out mid-execution
5. Track retry statistics: attempts made, total time waiting, final outcome

### Expected result

The executor gracefully handles timeouts and rate limits, returns partial results when available, and provides retry statistics.

<details>
<summary>💡 Hints</summary>

- Use `concurrent.futures.ThreadPoolExecutor` for timeout enforcement
- For partial results, pass a shared list to the function via `threading.Lock`
- Track `total_wait_time` by summing all backoff delays
- Test rate limits by raising an exception with "429" in the message for the first N calls

</details>

<details>
<summary>✅ Solution</summary>

```python
import time
import random
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExecutionStats:
    attempts: int = 0
    total_wait_seconds: float = 0.0
    timed_out: bool = False
    rate_limited: bool = False
    partial_results: list = field(default_factory=list)
    final_success: bool = False


class TimeoutAwareExecutor:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def execute(
        self,
        name: str,
        handler: Callable,
        args: dict,
        timeout: float = 10.0,
        collect_partial: bool = False
    ) -> tuple[dict, ExecutionStats]:
        stats = ExecutionStats()
        partial_collector = [] if collect_partial else None
        
        for attempt in range(self.max_retries + 1):
            stats.attempts = attempt + 1
            
            # Execute with timeout
            result = self._execute_with_timeout(
                handler, args, timeout, partial_collector
            )
            
            if result["success"]:
                stats.final_success = True
                return result, stats
            
            # Check error type
            if result.get("timed_out"):
                stats.timed_out = True
                if partial_collector:
                    stats.partial_results = list(partial_collector)
                    return {
                        "success": True,
                        "partial": True,
                        "result": stats.partial_results,
                        "warning": f"Timed out with {len(stats.partial_results)} partial results"
                    }, stats
                # Don't retry timeouts by default
                return result, stats
            
            if result.get("rate_limited"):
                stats.rate_limited = True
                if attempt < self.max_retries:
                    delay = self._backoff(attempt)
                    stats.total_wait_seconds += delay
                    print(f"  ⏳ Rate limited, waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            
            # Non-retryable error
            return result, stats
        
        return {"success": False, "error": "Max retries exceeded"}, stats
    
    def _execute_with_timeout(
        self, handler, args, timeout, collector
    ) -> dict:
        if collector is not None:
            args = {**args, "_collector": collector}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(handler, **args)
            try:
                result = future.result(timeout=timeout)
                return {"success": True, "result": result}
            except concurrent.futures.TimeoutError:
                return {"success": False, "timed_out": True,
                        "error": f"Timed out after {timeout}s"}
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate_limit" in msg:
                    return {"success": False, "rate_limited": True,
                            "error": str(exc)}
                return {"success": False, "error": str(exc)}
    
    def _backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), 60)
        return random.uniform(0, delay)  # Full jitter


# Test 1: Rate-limited function
call_count = 0

def rate_limited_fn(query: str, **kwargs) -> dict:
    global call_count
    call_count += 1
    if call_count <= 2:
        raise Exception("429: Rate limit exceeded")
    return {"answer": f"Result for {query}"}


executor = TimeoutAwareExecutor(max_retries=3, base_delay=0.5)
call_count = 0

result, stats = executor.execute(
    "search", rate_limited_fn, {"query": "test"}, timeout=5.0
)
print(f"Result: {result}")
print(f"Stats: attempts={stats.attempts}, "
      f"wait={stats.total_wait_seconds:.1f}s, "
      f"rate_limited={stats.rate_limited}")
```

**Output:**
```
  ⏳ Rate limited, waiting 0.3s...
  ⏳ Rate limited, waiting 0.7s...
Result: {'success': True, 'result': {'answer': 'Result for test'}}
Stats: attempts=3, wait=1.0s, rate_limited=True
```

</details>

### Bonus challenges

- [ ] Add a "budget" timeout: total time across all retries cannot exceed a configured limit
- [ ] Implement sliding-window rate limiting on the client side to avoid hitting server limits

---

## Summary

✅ Always set execution timeouts on function calls — use `concurrent.futures.ThreadPoolExecutor` for sync or `asyncio.wait_for` for async

✅ Exponential backoff with jitter prevents thundering herd when retrying rate limits

✅ Respect server-provided `Retry-After` headers — the server knows best

✅ Partial results are better than nothing — collect results incrementally and return what you have on timeout

✅ Distinguish timeouts from rate limits: timeouts suggest simplifying the query; rate limits require waiting

✅ Anthropic's 529 `overloaded_error` is a rate-limit-like error specific to Anthropic — handle it the same way

**Next:** [Safety Refusals →](./05-safety-refusal.md) — Handling content safety blocks and model refusals during function calling

---

[← Previous: Execution Failures](./03-execution-failures.md) | [Back to Lesson Overview](./00-error-handling.md)

<!-- 
Sources Consulted:
- OpenAI Error Codes: https://platform.openai.com/docs/guides/error-codes
- Anthropic Errors: https://docs.anthropic.com/en/api/errors
- Gemini Troubleshooting: https://ai.google.dev/gemini-api/docs/troubleshooting
- AWS Architecture Blog — Exponential Backoff and Jitter: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
-->
