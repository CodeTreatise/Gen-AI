---
title: "Running Concurrent Tasks"
---

# Running Concurrent Tasks

## Introduction

The real power of async comes from running multiple tasks concurrently. This lesson covers the tools asyncio provides for managing concurrent execution.

### What We'll Cover

- asyncio.gather()
- asyncio.create_task()
- asyncio.wait()
- asyncio.as_completed()
- Timeouts and cancellation

### Prerequisites

- asyncio fundamentals
- async/await syntax

---

## asyncio.gather()

### Running Multiple Coroutines

```python
import asyncio

async def task(name, delay):
    print(f"Task {name}: starting")
    await asyncio.sleep(delay)
    print(f"Task {name}: done")
    return f"Result from {name}"

async def main():
    # Run all concurrently, get results in order
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
        task("C", 3)
    )
    print(f"Results: {results}")

asyncio.run(main())
```

**Output:**
```
Task A: starting
Task B: starting
Task C: starting
Task B: done
Task A: done
Task C: done
Results: ['Result from A', 'Result from B', 'Result from C']
```

### With List Unpacking

```python
import asyncio

async def fetch(url):
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments"
    ]
    
    # Create tasks from list
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for url, result in zip(urls, results):
        print(f"{url}: {result}")

asyncio.run(main())
```

### Handling Exceptions

```python
import asyncio

async def might_fail(n):
    await asyncio.sleep(0.5)
    if n == 2:
        raise ValueError(f"Task {n} failed!")
    return f"Task {n} succeeded"

async def main():
    # Default: First exception cancels all
    try:
        results = await asyncio.gather(
            might_fail(1),
            might_fail(2),  # This will fail
            might_fail(3)
        )
    except ValueError as e:
        print(f"Error: {e}")
    
    # With return_exceptions=True: Collect all results/exceptions
    results = await asyncio.gather(
        might_fail(1),
        might_fail(2),
        might_fail(3),
        return_exceptions=True
    )
    
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"Task {i}: Error - {result}")
        else:
            print(f"Task {i}: {result}")

asyncio.run(main())
```

---

## asyncio.create_task()

### Scheduling Tasks

```python
import asyncio

async def background_task():
    print("Background: starting")
    await asyncio.sleep(2)
    print("Background: done")
    return "Background result"

async def main():
    # Create and schedule task (starts immediately)
    task = asyncio.create_task(background_task())
    
    print("Main: doing other work...")
    await asyncio.sleep(1)
    print("Main: still working...")
    
    # Wait for task to complete
    result = await task
    print(f"Main: got {result}")

asyncio.run(main())
```

**Output:**
```
Background: starting
Main: doing other work...
Main: still working...
Background: done
Main: got Background result
```

### Fire and Forget

```python
import asyncio

async def log_event(event):
    await asyncio.sleep(0.1)  # Simulate async logging
    print(f"Logged: {event}")

async def main():
    # Start task but don't wait
    task = asyncio.create_task(log_event("User logged in"))
    
    # Continue immediately
    print("Processing request...")
    await asyncio.sleep(0.5)
    print("Request done")
    
    # Make sure task completes before exiting
    await task

asyncio.run(main())
```

### Managing Multiple Tasks

```python
import asyncio

async def worker(name, seconds):
    print(f"Worker {name}: starting")
    await asyncio.sleep(seconds)
    print(f"Worker {name}: finished")
    return f"Result from {name}"

async def main():
    # Create tasks
    tasks = [
        asyncio.create_task(worker("A", 2)),
        asyncio.create_task(worker("B", 1)),
        asyncio.create_task(worker("C", 3))
    ]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    print(f"All results: {results}")

asyncio.run(main())
```

---

## asyncio.wait()

### Flexible Task Waiting

```python
import asyncio

async def task(name, delay):
    await asyncio.sleep(delay)
    return f"Task {name}"

async def main():
    tasks = {
        asyncio.create_task(task("A", 2)),
        asyncio.create_task(task("B", 1)),
        asyncio.create_task(task("C", 3))
    }
    
    # Wait for all
    done, pending = await asyncio.wait(tasks)
    
    print(f"Done: {len(done)}, Pending: {len(pending)}")
    for t in done:
        print(f"  Result: {t.result()}")

asyncio.run(main())
```

### Wait for First Complete

```python
import asyncio

async def fetch_from_server(server, delay):
    await asyncio.sleep(delay)
    return f"Data from {server}"

async def main():
    tasks = {
        asyncio.create_task(fetch_from_server("Server1", 2)),
        asyncio.create_task(fetch_from_server("Server2", 1)),
        asyncio.create_task(fetch_from_server("Server3", 3))
    }
    
    # Wait for first to complete
    done, pending = await asyncio.wait(
        tasks, 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Use first result
    first_task = done.pop()
    print(f"First result: {first_task.result()}")
    
    # Cancel remaining
    for task in pending:
        task.cancel()

asyncio.run(main())
```

### Wait Options

| Option | Behavior |
|--------|----------|
| `ALL_COMPLETED` | Wait for all (default) |
| `FIRST_COMPLETED` | Return when any completes |
| `FIRST_EXCEPTION` | Return on first exception |

---

## asyncio.as_completed()

### Process Results as They Arrive

```python
import asyncio

async def fetch(url, delay):
    await asyncio.sleep(delay)
    return f"Data from {url}"

async def main():
    tasks = [
        fetch("url1", 3),
        fetch("url2", 1),
        fetch("url3", 2)
    ]
    
    # Process as each completes (fastest first)
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Got: {result}")

asyncio.run(main())
```

**Output:**
```
Got: Data from url2
Got: Data from url3
Got: Data from url1
```

### With Progress Updates

```python
import asyncio

async def fetch(id, delay):
    await asyncio.sleep(delay)
    return {"id": id, "data": f"result_{id}"}

async def main():
    tasks = [fetch(i, i * 0.5) for i in range(1, 6)]
    total = len(tasks)
    
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        print(f"[{completed}/{total}] Got result: {result['id']}")

asyncio.run(main())
```

---

## Timeouts

### asyncio.wait_for()

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(main())
```

### Timeout with gather()

```python
import asyncio

async def fetch(url, delay):
    await asyncio.sleep(delay)
    return f"Data from {url}"

async def main():
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                fetch("api1", 1),
                fetch("api2", 2),
                fetch("api3", 5)  # This is slow
            ),
            timeout=3.0
        )
        print(results)
    except asyncio.TimeoutError:
        print("Some requests timed out")

asyncio.run(main())
```

### asyncio.timeout (Python 3.11+)

```python
import asyncio

async def main():
    async with asyncio.timeout(2.0):
        await asyncio.sleep(10)  # Will timeout

asyncio.run(main())
```

---

## Task Cancellation

### Cancelling Tasks

```python
import asyncio

async def long_running():
    try:
        print("Task: Starting long operation")
        await asyncio.sleep(10)
        print("Task: Completed")
    except asyncio.CancelledError:
        print("Task: Cancelled!")
        raise  # Re-raise to propagate

async def main():
    task = asyncio.create_task(long_running())
    
    await asyncio.sleep(1)
    print("Main: Cancelling task")
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Main: Task was cancelled")

asyncio.run(main())
```

### Cleanup on Cancellation

```python
import asyncio

async def task_with_cleanup():
    try:
        print("Acquiring resources...")
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("Cleaning up resources...")
        await asyncio.sleep(0.1)  # Cleanup time
        print("Cleanup complete")
        raise
    finally:
        print("Finally block executed")

async def main():
    task = asyncio.create_task(task_with_cleanup())
    await asyncio.sleep(1)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
```

---

## Hands-on Exercise

### Your Task

```python
# Build a concurrent web scraper simulation that:
# 1. "Fetches" 10 URLs with random delays (0.5-2 seconds)
# 2. Has a 3-second overall timeout
# 3. Processes results as they arrive
# 4. Reports which URLs completed and which timed out
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
import random

async def fetch_url(url: str) -> dict:
    """Simulate fetching a URL with random delay."""
    delay = random.uniform(0.5, 2.5)
    await asyncio.sleep(delay)
    return {"url": url, "delay": delay, "status": "success"}

async def main():
    urls = [f"https://example.com/page/{i}" for i in range(10)]
    
    # Create tasks
    tasks = {
        asyncio.create_task(fetch_url(url)): url 
        for url in urls
    }
    
    completed = []
    failed = []
    
    try:
        # Wait with timeout
        done, pending = await asyncio.wait(
            tasks.keys(),
            timeout=3.0
        )
        
        # Process completed
        for task in done:
            if task.exception():
                failed.append(tasks[task])
            else:
                result = task.result()
                completed.append(result)
                print(f"✓ {result['url']} ({result['delay']:.2f}s)")
        
        # Handle pending (timed out)
        for task in pending:
            failed.append(tasks[task])
            task.cancel()
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Report
    print(f"\n=== Summary ===")
    print(f"Completed: {len(completed)}/{len(urls)}")
    print(f"Timed out: {len(failed)}")
    
    if failed:
        print("Failed URLs:")
        for url in failed:
            print(f"  - {url}")

asyncio.run(main())
```
</details>

---

## Summary

✅ **`asyncio.gather()`** runs coroutines concurrently, returns ordered results
✅ **`asyncio.create_task()`** schedules and returns a Task object
✅ **`asyncio.wait()`** for flexible completion handling
✅ **`asyncio.as_completed()`** processes results as they arrive
✅ **`asyncio.wait_for()`** adds timeout to any awaitable
✅ Handle **`CancelledError`** for graceful cleanup

**Next:** [Async Patterns for AI](./04-async-patterns-ai.md)

---

## Further Reading

- [Tasks and Coroutines](https://docs.python.org/3/library/asyncio-task.html)
- [Timeouts](https://docs.python.org/3/library/asyncio-task.html#timeouts)

<!-- 
Sources Consulted:
- Python asyncio Docs: https://docs.python.org/3/library/asyncio-task.html
-->
