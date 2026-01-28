---
title: "asyncio Fundamentals"
---

# asyncio Fundamentals

## Introduction

The `asyncio` module is Python's built-in library for writing concurrent code using the async/await syntax. Understanding its core concepts is essential for async programming.

### What We'll Cover

- async def and await
- Coroutines and awaitables
- asyncio.run()
- Basic async patterns

### Prerequisites

- Python functions
- Understanding of I/O concepts

---

## async def Functions

### Defining Coroutines

```python
import asyncio

# Regular function
def regular_function():
    return "Hello"

# Async function (coroutine function)
async def async_function():
    return "Hello"

# Calling them
result1 = regular_function()      # Returns "Hello"
result2 = async_function()        # Returns a coroutine object!

print(type(result2))  # <class 'coroutine'>
```

### Running Coroutines

```python
import asyncio

async def greet():
    return "Hello, World!"

# ❌ Wrong: Just creates coroutine object
coro = greet()  # RuntimeWarning: coroutine was never awaited

# ✅ Correct: Run with asyncio.run()
result = asyncio.run(greet())
print(result)  # Hello, World!
```

---

## The await Keyword

### Suspending Execution

```python
import asyncio

async def slow_operation():
    print("Starting slow operation...")
    await asyncio.sleep(2)  # Suspends for 2 seconds
    print("Slow operation complete!")
    return 42

async def main():
    result = await slow_operation()  # Wait for it
    print(f"Result: {result}")

asyncio.run(main())
```

**Output:**
```
Starting slow operation...
(2 second pause)
Slow operation complete!
Result: 42
```

### What Can Be Awaited?

```python
import asyncio

# 1. Coroutines
async def my_coroutine():
    return "done"

# 2. Tasks
task = asyncio.create_task(my_coroutine())

# 3. Futures
future = asyncio.Future()

# All are "awaitables"
async def main():
    result1 = await my_coroutine()
    result2 = await task
    # result3 = await future  # Would wait forever unless set
```

### await vs Regular Calls

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "value"}

async def main():
    # ✅ Correct: await the coroutine
    data = await fetch_data()
    print(data)
    
    # ❌ Wrong: Forgot await - gets coroutine object
    # data = fetch_data()  # This is a coroutine, not data!

asyncio.run(main())
```

---

## asyncio.run()

### The Entry Point

```python
import asyncio

async def main():
    print("Running async code")
    await asyncio.sleep(1)
    print("Done!")

# This is the standard entry point
asyncio.run(main())
```

### What asyncio.run() Does

1. Creates a new event loop
2. Runs the coroutine until complete
3. Closes the loop
4. Returns the result

```python
import asyncio

async def compute():
    await asyncio.sleep(1)
    return 42

# Get the return value
result = asyncio.run(compute())
print(f"Result: {result}")  # Result: 42
```

### Jupyter/IPython Note

```python
# In Jupyter notebooks, the event loop is already running
# Use await directly instead of asyncio.run()

# Regular Python script:
asyncio.run(main())

# Jupyter notebook:
await main()
```

---

## Coroutine Objects

### Understanding Coroutines

```python
import asyncio

async def say_hello():
    return "Hello!"

# Creating the coroutine (doesn't run yet)
coro = say_hello()
print(type(coro))  # <class 'coroutine'>
print(coro)        # <coroutine object say_hello at 0x...>

# Running the coroutine
result = asyncio.run(coro)
print(result)  # Hello!
```

### Coroutine Lifecycle

```
1. Define: async def my_func()
2. Create: coro = my_func()  # Returns coroutine object
3. Schedule: task = asyncio.create_task(coro)
4. Execute: await task  # Actually runs
5. Complete: Get result or exception
```

---

## Basic Patterns

### Sequential Execution

```python
import asyncio

async def step(n):
    print(f"Step {n} starting")
    await asyncio.sleep(1)
    print(f"Step {n} done")
    return n

async def main():
    # Sequential: One after another
    result1 = await step(1)
    result2 = await step(2)
    result3 = await step(3)
    print(f"Results: {result1}, {result2}, {result3}")

asyncio.run(main())
# Takes ~3 seconds (1 + 1 + 1)
```

### Concurrent Execution

```python
import asyncio

async def step(n):
    print(f"Step {n} starting")
    await asyncio.sleep(1)
    print(f"Step {n} done")
    return n

async def main():
    # Concurrent: All at once
    results = await asyncio.gather(
        step(1),
        step(2),
        step(3)
    )
    print(f"Results: {results}")

asyncio.run(main())
# Takes ~1 second (all run together)
```

---

## Async With (Context Managers)

### Async Context Managers

```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource...")
        await asyncio.sleep(0.5)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource...")
        await asyncio.sleep(0.5)
    
    async def do_work(self):
        print("Working...")
        await asyncio.sleep(1)

async def main():
    async with AsyncResource() as resource:
        await resource.do_work()

asyncio.run(main())
```

### Common Async Context Managers

```python
import httpx
import aiofiles

async def examples():
    # HTTP client
    async with httpx.AsyncClient() as client:
        response = await client.get('https://api.github.com')
    
    # File I/O
    async with aiofiles.open('file.txt', 'r') as f:
        content = await f.read()
```

---

## Async For (Iterators)

### Async Iterators

```python
import asyncio

class AsyncCounter:
    def __init__(self, stop):
        self.current = 0
        self.stop = stop
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        await asyncio.sleep(0.5)
        self.current += 1
        return self.current

async def main():
    async for num in AsyncCounter(5):
        print(num)

asyncio.run(main())
```

### Async Generators

```python
import asyncio

async def countdown(n):
    while n > 0:
        yield n
        await asyncio.sleep(1)
        n -= 1

async def main():
    async for num in countdown(5):
        print(f"T-minus {num}")
    print("Liftoff!")

asyncio.run(main())
```

---

## Common Mistakes

### Mistake 1: Forgetting await

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "value"}

async def main():
    # ❌ Wrong: Forgot await
    data = fetch_data()
    print(type(data))  # <class 'coroutine'> - Not the data!
    
    # ✅ Correct
    data = await fetch_data()
    print(data)  # {"data": "value"}

asyncio.run(main())
```

### Mistake 2: Using asyncio.run() Inside Async

```python
import asyncio

async def inner():
    return "Hello"

async def outer():
    # ❌ Wrong: Can't call asyncio.run() in async context
    # result = asyncio.run(inner())  # RuntimeError!
    
    # ✅ Correct: Just await
    result = await inner()
    return result

asyncio.run(outer())
```

### Mistake 3: Blocking in Async

```python
import asyncio
import time

async def bad_example():
    # ❌ Wrong: time.sleep blocks the event loop!
    time.sleep(5)  # Nothing else can run during this
    
    # ✅ Correct: Use asyncio.sleep
    await asyncio.sleep(5)  # Other tasks can run

asyncio.run(bad_example())
```

---

## Hands-on Exercise

### Your Task

```python
# Create an async program that:
# 1. Defines an async function that simulates fetching from an API
#    (use asyncio.sleep to simulate network delay)
# 2. Fetches from 3 "APIs" sequentially and measures time
# 3. Fetches from 3 "APIs" concurrently and measures time
# 4. Compares the timing
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
import time

async def fetch_api(api_name: str, delay: float) -> dict:
    """Simulate an API call with delay."""
    print(f"Fetching from {api_name}...")
    await asyncio.sleep(delay)
    print(f"Got response from {api_name}")
    return {"api": api_name, "data": "sample"}

async def sequential():
    """Fetch APIs one by one."""
    start = time.perf_counter()
    
    result1 = await fetch_api("API-1", 1.0)
    result2 = await fetch_api("API-2", 1.5)
    result3 = await fetch_api("API-3", 0.5)
    
    elapsed = time.perf_counter() - start
    return [result1, result2, result3], elapsed

async def concurrent():
    """Fetch APIs all at once."""
    start = time.perf_counter()
    
    results = await asyncio.gather(
        fetch_api("API-1", 1.0),
        fetch_api("API-2", 1.5),
        fetch_api("API-3", 0.5)
    )
    
    elapsed = time.perf_counter() - start
    return results, elapsed

async def main():
    print("=== Sequential ===")
    seq_results, seq_time = await sequential()
    print(f"Sequential time: {seq_time:.2f}s\n")
    
    print("=== Concurrent ===")
    conc_results, conc_time = await concurrent()
    print(f"Concurrent time: {conc_time:.2f}s\n")
    
    print(f"Speedup: {seq_time/conc_time:.1f}x faster!")

asyncio.run(main())
```

**Output:**
```
=== Sequential ===
Fetching from API-1...
Got response from API-1
Fetching from API-2...
Got response from API-2
Fetching from API-3...
Got response from API-3
Sequential time: 3.00s

=== Concurrent ===
Fetching from API-1...
Fetching from API-2...
Fetching from API-3...
Got response from API-3
Got response from API-1
Got response from API-2
Concurrent time: 1.50s

Speedup: 2.0x faster!
```
</details>

---

## Summary

✅ **`async def`** defines a coroutine function
✅ **`await`** suspends execution until awaitable completes
✅ **`asyncio.run()`** is the entry point for async code
✅ Use **`async with`** for async context managers
✅ Use **`async for`** for async iterators
✅ Never use blocking calls (like `time.sleep`) in async code

**Next:** [Concurrent Tasks](./03-concurrent-tasks.md)

---

## Further Reading

- [Coroutines and Tasks](https://docs.python.org/3/library/asyncio-task.html)
- [Async Generators](https://peps.python.org/pep-0525/)

<!-- 
Sources Consulted:
- Python asyncio Docs: https://docs.python.org/3/library/asyncio.html
-->
