---
title: "Understanding Async I/O"
---

# Understanding Async I/O

## Introduction

Before diving into async code, we need to understand when async helps and when it doesn't. Async excels at I/O-bound tasks but won't speed up CPU-heavy work.

### What We'll Cover

- Concurrency vs parallelism
- I/O-bound vs CPU-bound
- The event loop
- When to use async

### Prerequisites

- Python basics

---

## Concurrency vs Parallelism

### Key Difference

```
Concurrency: Dealing with many things at once (structure)
Parallelism: Doing many things at once (execution)
```

### Concurrency (Async)

```python
# Concurrency: One cook, multiple dishes
# Cook starts dish A → while it simmers, starts dish B
# → while B bakes, checks on A → switches between tasks

import asyncio

async def cook_dish(name, time):
    print(f"Starting {name}")
    await asyncio.sleep(time)  # Simulates cooking time
    print(f"Finished {name}")

async def main():
    # Both dishes cook "concurrently"
    await asyncio.gather(
        cook_dish("pasta", 2),
        cook_dish("salad", 1)
    )

asyncio.run(main())
```

**Output:**
```
Starting pasta
Starting salad
Finished salad
Finished pasta
```

### Parallelism (Multiprocessing)

```python
# Parallelism: Multiple cooks, each doing one dish
# Cook A makes pasta while Cook B makes salad simultaneously

from multiprocessing import Pool

def cook_dish(args):
    name, time = args
    import time as t
    print(f"Starting {name}")
    t.sleep(time)
    return f"Finished {name}"

# Requires if __name__ == '__main__':
with Pool(2) as p:
    results = p.map(cook_dish, [("pasta", 2), ("salad", 1)])
```

---

## I/O-Bound vs CPU-Bound

### I/O-Bound Tasks (Use Async)

Waiting for external resources:

```python
# ✅ Good for async
- HTTP requests to APIs
- Database queries
- File reading/writing
- Network operations
- LLM API calls
```

```python
import asyncio
import httpx

async def fetch_urls(urls):
    async with httpx.AsyncClient() as client:
        # All requests happen concurrently
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses
```

### CPU-Bound Tasks (Use Multiprocessing)

Heavy computation:

```python
# ❌ Async won't help here
- Image processing
- Data transformation
- Mathematical calculations
- ML model inference
```

```python
from multiprocessing import Pool

def process_image(image_path):
    # CPU-intensive work
    return transform(load(image_path))

# Use multiple CPU cores
with Pool(4) as p:
    results = p.map(process_image, image_paths)
```

### Decision Guide

| Task Type | Solution | Example |
|-----------|----------|---------|
| **I/O-bound** | `asyncio` | API calls, file I/O |
| **CPU-bound** | `multiprocessing` | Number crunching |
| **Mixed** | Both | Fetch data, then process |

---

## The Event Loop

### What Is It?

The event loop is the core of async Python. It:
1. Runs coroutines
2. Handles I/O callbacks
3. Switches between tasks during `await`

```python
import asyncio

# The event loop runs your async code
asyncio.run(main())  # Creates loop, runs main, closes loop
```

### How It Works

```
1. Start task A
2. Task A hits await (waiting for I/O)
3. Event loop switches to task B
4. Task B hits await
5. Event loop checks: is A ready? is B ready?
6. Whichever is ready, resume it
7. Repeat until all tasks done
```

```python
import asyncio

async def task(name, delay):
    print(f"{name}: starting")
    await asyncio.sleep(delay)  # Yields to event loop
    print(f"{name}: done")

async def main():
    # Both tasks run in the same event loop
    await asyncio.gather(
        task("A", 2),
        task("B", 1)
    )

asyncio.run(main())
```

**Output:**
```
A: starting
B: starting
B: done
A: done
```

### Visualizing the Loop

```
Time →
     0.0   0.5   1.0   1.5   2.0
A:   [start]----wait---------[done]
B:   [start]---[done]
Loop: A→B→(check)→B done→A done
```

---

## When to Use Async

### ✅ Use Async When

```python
# Multiple API calls
async def fetch_all_users():
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get(f'/users/{id}')
            for id in range(100)
        ]
        return await asyncio.gather(*tasks)

# Streaming responses
async def stream_llm_response():
    async for chunk in response.aiter_text():
        print(chunk, end='')

# Web servers handling many requests
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    data = await fetch_from_database()
    return data
```

### ❌ Don't Use Async When

```python
# Pure computation
def calculate_pi(digits):
    # No I/O, no waiting - async won't help
    return compute_pi(digits)

# Blocking libraries
def query_old_database():
    # If library is sync-only, wrapping in async won't help
    return old_db.query("SELECT * FROM users")

# Single sequential task
def process_one_item(item):
    # No benefit to async for one thing
    return transform(item)
```

### Comparison Table

| Scenario | Sync Time | Async Time |
|----------|-----------|------------|
| 10 API calls (1s each) | ~10 seconds | ~1 second |
| 10 CPU tasks (1s each) | ~10 seconds | ~10 seconds |
| 1 API call | ~1 second | ~1 second |

---

## Async vs Threading vs Multiprocessing

### Quick Comparison

| Feature | Async | Threading | Multiprocessing |
|---------|-------|-----------|-----------------|
| **Best for** | I/O-bound | I/O-bound | CPU-bound |
| **GIL** | Yes (one thread) | Yes (limited) | No (separate processes) |
| **Memory** | Lightweight | Medium | Heavy |
| **Complexity** | Medium | High (locks) | Medium |
| **Data sharing** | Easy | Careful (locks) | IPC needed |

### When to Choose What

```python
# Async: Many network requests
async def fetch_many():
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[
            client.get(url) for url in urls
        ])

# Threading: Mixed I/O with some blocking code
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(blocking_io_function, items))

# Multiprocessing: CPU-heavy parallel work
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(cpu_intensive_function, items)
```

---

## Hands-on Exercise

### Your Task

```python
# Determine whether each task is I/O-bound or CPU-bound
# and choose the right approach:

# 1. Fetch 50 weather API responses
# 2. Resize 100 images
# 3. Query 10 different databases
# 4. Train 5 ML models
# 5. Stream responses from 3 LLM APIs

# For each, explain your choice
```

<details>
<summary>✅ Solution</summary>

```python
# 1. Fetch 50 weather API responses
# I/O-bound → async
async def fetch_weather():
    async with httpx.AsyncClient() as client:
        tasks = [client.get(f'/weather/{city}') for city in cities]
        return await asyncio.gather(*tasks)

# 2. Resize 100 images  
# CPU-bound → multiprocessing
from multiprocessing import Pool
def resize_images(image_paths):
    with Pool(processes=8) as pool:
        return pool.map(resize_image, image_paths)

# 3. Query 10 different databases
# I/O-bound → async (with async DB driver)
async def query_databases():
    tasks = [db.execute(query) for db in databases]
    return await asyncio.gather(*tasks)

# 4. Train 5 ML models
# CPU-bound → multiprocessing
def train_models(model_configs):
    with Pool(processes=5) as pool:
        return pool.map(train_model, model_configs)

# 5. Stream responses from 3 LLM APIs
# I/O-bound → async
async def stream_llms(prompts):
    async def stream_one(prompt, api):
        async for chunk in api.stream(prompt):
            yield chunk
    
    # Handle multiple streams concurrently
    tasks = [stream_one(p, api) for p, api in zip(prompts, apis)]
    # Note: Streaming typically handled differently
```
</details>

---

## Summary

✅ **Concurrency** = structure, **Parallelism** = execution
✅ **Async** for I/O-bound tasks (network, files)
✅ **Multiprocessing** for CPU-bound tasks (computation)
✅ The **event loop** switches between tasks at `await`
✅ Async shines with **many concurrent I/O operations**

**Next:** [asyncio Fundamentals](./02-asyncio-fundamentals.md)

---

## Further Reading

- [Python Concurrency](https://realpython.com/python-concurrency/)
- [Async IO in Python](https://realpython.com/async-io-python/)

<!-- 
Sources Consulted:
- Python asyncio Docs: https://docs.python.org/3/library/asyncio.html
-->
