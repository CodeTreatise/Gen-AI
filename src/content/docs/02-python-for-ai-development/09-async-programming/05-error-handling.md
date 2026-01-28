---
title: "Error Handling in Async Code"
---

# Error Handling in Async Code

## Introduction

Async error handling has unique challenges—exceptions can occur in parallel tasks, cancellations need graceful handling, and cleanup must work correctly across concurrent operations.

### What We'll Cover

- Exception propagation
- TaskGroup (Python 3.11+)
- ExceptionGroup handling
- Graceful shutdown
- Debugging async code

### Prerequisites

- asyncio fundamentals
- Python exception handling

---

## Exception Propagation

### Exceptions in Coroutines

```python
import asyncio

async def might_fail(n: int):
    await asyncio.sleep(0.5)
    if n == 2:
        raise ValueError(f"Task {n} failed!")
    return f"Task {n} succeeded"

async def main():
    try:
        result = await might_fail(2)
    except ValueError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

### Exceptions in gather()

```python
import asyncio

async def task(n):
    await asyncio.sleep(n * 0.1)
    if n == 2:
        raise ValueError(f"Task {n} failed!")
    return n

async def main():
    # Default: First exception propagates, cancels others
    try:
        results = await asyncio.gather(
            task(1),
            task(2),  # Fails
            task(3)
        )
    except ValueError as e:
        print(f"Error: {e}")
    
    # With return_exceptions: All complete, exceptions in results
    results = await asyncio.gather(
        task(1),
        task(2),
        task(3),
        return_exceptions=True
    )
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i+1}: Error - {result}")
        else:
            print(f"Task {i+1}: Success - {result}")

asyncio.run(main())
```

### Exceptions in create_task()

```python
import asyncio

async def background_task():
    await asyncio.sleep(1)
    raise RuntimeError("Background task failed!")

async def main():
    task = asyncio.create_task(background_task())
    
    # Do other work
    await asyncio.sleep(0.5)
    print("Main: still working...")
    
    # Exception raised when awaited
    try:
        await task
    except RuntimeError as e:
        print(f"Main: caught - {e}")

asyncio.run(main())
```

---

## TaskGroup (Python 3.11+)

### Structured Concurrency

```python
import asyncio

async def worker(name: str, delay: float):
    print(f"Worker {name}: starting")
    await asyncio.sleep(delay)
    print(f"Worker {name}: done")
    return f"Result from {name}"

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(worker("A", 1))
        task2 = tg.create_task(worker("B", 2))
        task3 = tg.create_task(worker("C", 1.5))
    
    # All tasks completed when we get here
    print(f"Results: {task1.result()}, {task2.result()}, {task3.result()}")

asyncio.run(main())
```

### TaskGroup Exception Handling

```python
import asyncio

async def might_fail(n: int):
    await asyncio.sleep(0.5)
    if n == 2:
        raise ValueError(f"Task {n} failed!")
    return n

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(might_fail(1))
            tg.create_task(might_fail(2))  # Will fail
            tg.create_task(might_fail(3))
    except* ValueError as eg:
        # ExceptionGroup containing all ValueErrors
        print(f"Caught {len(eg.exceptions)} ValueError(s):")
        for exc in eg.exceptions:
            print(f"  - {exc}")

asyncio.run(main())
```

### Benefits of TaskGroup

| Feature | gather() | TaskGroup |
|---------|----------|-----------|
| Cancel on error | Partial | All tasks cancelled |
| Exception handling | Single or list | ExceptionGroup |
| Cleanup | Manual | Automatic |
| Nested groups | Manual | Natural |

---

## ExceptionGroup (Python 3.11+)

### Handling Multiple Exceptions

```python
import asyncio

async def task(n):
    if n == 1:
        raise ValueError("Value error")
    elif n == 2:
        raise TypeError("Type error")
    elif n == 3:
        raise RuntimeError("Runtime error")
    return n

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task(1))
            tg.create_task(task(2))
            tg.create_task(task(3))
            tg.create_task(task(4))  # This one succeeds
    except* ValueError as eg:
        print(f"ValueErrors: {eg.exceptions}")
    except* TypeError as eg:
        print(f"TypeErrors: {eg.exceptions}")
    except* RuntimeError as eg:
        print(f"RuntimeErrors: {eg.exceptions}")

asyncio.run(main())
```

### Processing All Exceptions

```python
import asyncio

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task(1))
            tg.create_task(task(2))
            tg.create_task(task(3))
    except* Exception as eg:
        for exc in eg.exceptions:
            print(f"Exception: {type(exc).__name__}: {exc}")

asyncio.run(main())
```

---

## Cancellation Handling

### Handling CancelledError

```python
import asyncio

async def cancellable_task():
    try:
        print("Task: Starting work...")
        await asyncio.sleep(10)
        print("Task: Work complete")
    except asyncio.CancelledError:
        print("Task: Cancelled! Cleaning up...")
        # Perform cleanup
        await asyncio.sleep(0.1)
        print("Task: Cleanup done")
        raise  # Re-raise to propagate cancellation

async def main():
    task = asyncio.create_task(cancellable_task())
    
    await asyncio.sleep(1)
    print("Main: Cancelling task...")
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Main: Task was cancelled")

asyncio.run(main())
```

### Shield from Cancellation

```python
import asyncio

async def critical_operation():
    print("Critical: Starting (cannot be cancelled)")
    await asyncio.sleep(2)
    print("Critical: Complete")
    return "important result"

async def main():
    # Shield protects from external cancellation
    task = asyncio.create_task(
        asyncio.shield(critical_operation())
    )
    
    await asyncio.sleep(0.5)
    print("Main: Attempting to cancel...")
    task.cancel()
    
    try:
        result = await task
        print(f"Result: {result}")
    except asyncio.CancelledError:
        print("Main: Task was cancelled (but operation continued)")

asyncio.run(main())
```

---

## Graceful Shutdown

### Shutdown Pattern

```python
import asyncio
import signal

async def worker(name: str, stop_event: asyncio.Event):
    """Worker that checks for shutdown signal."""
    while not stop_event.is_set():
        print(f"Worker {name}: working...")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Continue working
    
    print(f"Worker {name}: shutting down gracefully")

async def main():
    stop_event = asyncio.Event()
    
    # Handle shutdown signal
    def signal_handler():
        print("\nReceived shutdown signal")
        stop_event.set()
    
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    
    # Start workers
    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker("A", stop_event))
        tg.create_task(worker("B", stop_event))
    
    print("All workers stopped")

# asyncio.run(main())
```

### Cleanup with Finally

```python
import asyncio

class AsyncResource:
    async def acquire(self):
        print("Resource: Acquiring...")
        await asyncio.sleep(0.5)
        print("Resource: Acquired")
    
    async def release(self):
        print("Resource: Releasing...")
        await asyncio.sleep(0.5)
        print("Resource: Released")
    
    async def do_work(self):
        print("Resource: Working...")
        await asyncio.sleep(1)
        raise RuntimeError("Work failed!")

async def main():
    resource = AsyncResource()
    
    try:
        await resource.acquire()
        await resource.do_work()
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        await resource.release()

asyncio.run(main())
```

---

## Debugging Async Code

### Debug Mode

```python
import asyncio

# Enable debug mode
asyncio.run(main(), debug=True)

# Or via environment variable
# PYTHONASYNCIODEBUG=1 python script.py
```

### Common Issues

```python
import asyncio

# Issue 1: Forgetting await
async def bad_example():
    result = some_coroutine()  # Missing await!
    # RuntimeWarning: coroutine was never awaited

# Issue 2: Blocking in async
async def blocking_example():
    import time
    time.sleep(5)  # Blocks the entire event loop!
    # Should be: await asyncio.sleep(5)

# Issue 3: Creating coroutine outside async
def sync_function():
    coro = async_function()  # Creates coroutine but can't run it
    # Need to use asyncio.run() or be in async context
```

### Logging Async Operations

```python
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def monitored_task(name: str):
    logger.info(f"Task {name}: starting")
    try:
        await asyncio.sleep(1)
        logger.info(f"Task {name}: completed")
        return f"Result from {name}"
    except asyncio.CancelledError:
        logger.warning(f"Task {name}: cancelled")
        raise
    except Exception as e:
        logger.error(f"Task {name}: error - {e}")
        raise

async def main():
    logger.info("Starting main")
    async with asyncio.TaskGroup() as tg:
        tg.create_task(monitored_task("A"))
        tg.create_task(monitored_task("B"))
    logger.info("All tasks complete")

asyncio.run(main())
```

---

## Error Recovery Pattern

```python
import asyncio
from typing import TypeVar, Callable, Any

T = TypeVar('T')

async def with_retry(
    coro_func: Callable[..., T],
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """Execute coroutine with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait = delay * (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
    
    raise last_exception

# Usage
async def unreliable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("API temporarily unavailable")
    return "Success!"

async def main():
    try:
        result = await with_retry(
            unreliable_api_call,
            max_retries=5,
            delay=0.5,
            exceptions=(ConnectionError,)
        )
        print(f"Got: {result}")
    except ConnectionError:
        print("All retries failed")

asyncio.run(main())
```

---

## Hands-on Exercise

### Your Task

```python
# Build a robust task runner that:
# 1. Runs 5 tasks concurrently using TaskGroup
# 2. Some tasks fail randomly
# 3. Handles multiple exception types
# 4. Logs all errors
# 5. Reports success/failure summary
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    pass

class TimeoutError(Exception):
    pass

async def unreliable_task(task_id: int) -> str:
    """Task that might fail in various ways."""
    logger.info(f"Task {task_id}: starting")
    
    await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # Random failures
    failure = random.random()
    if failure < 0.2:
        raise APIError(f"Task {task_id}: API error")
    elif failure < 0.4:
        raise TimeoutError(f"Task {task_id}: Timeout")
    
    result = f"Task {task_id}: success"
    logger.info(result)
    return result

async def run_with_tracking(task_id: int, results: dict):
    """Wrapper to track individual task results."""
    try:
        result = await unreliable_task(task_id)
        results[task_id] = {"status": "success", "result": result}
    except Exception as e:
        results[task_id] = {"status": "error", "error": str(e), "type": type(e).__name__}
        raise

async def main():
    results = {}
    
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                tg.create_task(run_with_tracking(i, results))
                
    except* APIError as eg:
        logger.error(f"API Errors: {len(eg.exceptions)}")
        for exc in eg.exceptions:
            logger.error(f"  - {exc}")
            
    except* TimeoutError as eg:
        logger.error(f"Timeout Errors: {len(eg.exceptions)}")
        for exc in eg.exceptions:
            logger.error(f"  - {exc}")
            
    except* Exception as eg:
        logger.error(f"Other Errors: {len(eg.exceptions)}")
        for exc in eg.exceptions:
            logger.error(f"  - {type(exc).__name__}: {exc}")
    
    # Summary
    print("\n=== Summary ===")
    successes = sum(1 for r in results.values() if r["status"] == "success")
    failures = sum(1 for r in results.values() if r["status"] == "error")
    
    print(f"Total: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failed: {failures}")
    
    if failures > 0:
        print("\nFailed tasks:")
        for task_id, r in results.items():
            if r["status"] == "error":
                print(f"  Task {task_id}: {r['type']} - {r['error']}")

asyncio.run(main())
```
</details>

---

## Summary

✅ Use **`return_exceptions=True`** in gather() to collect all results
✅ **`TaskGroup`** (Python 3.11+) provides structured concurrency
✅ Handle **`ExceptionGroup`** with `except*` syntax
✅ Always re-raise **`CancelledError`** after cleanup
✅ Use **`asyncio.shield()`** for critical operations
✅ Enable **debug mode** during development

**Next:** [Async Libraries](./06-async-libraries.md)

---

## Further Reading

- [TaskGroup](https://docs.python.org/3/library/asyncio-task.html#asyncio.TaskGroup)
- [ExceptionGroup PEP 654](https://peps.python.org/pep-0654/)

<!-- 
Sources Consulted:
- Python asyncio Docs: https://docs.python.org/3/library/asyncio-task.html
-->
