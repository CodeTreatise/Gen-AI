---
title: "Choosing the Right Approach"
---

# Choosing the Right Approach

## Introduction

Choosing between asyncio, threading, and multiprocessing depends on your workload. This lesson provides decision frameworks and benchmarks to help you choose.

### What We'll Cover

- Decision framework
- Benchmarking approaches
- Hybrid patterns
- Memory considerations
- Debugging tips

### Prerequisites

- Understanding of all three approaches

---

## Decision Framework

### Quick Reference

| Task Type | Best Approach | Why |
|-----------|--------------|-----|
| **Many HTTP requests** | asyncio | Single thread, low overhead |
| **File I/O** | threading/asyncio | GIL released during I/O |
| **CPU computation** | multiprocessing | True parallelism |
| **Mixed I/O + CPU** | Hybrid | Combine approaches |
| **Real-time streaming** | asyncio | Event-driven model |

### Flowchart

```
START
  │
  ▼
Is task I/O-bound or CPU-bound?
  │
  ├─► I/O-bound
  │     │
  │     ▼
  │   Many concurrent connections?
  │     │
  │     ├─► Yes ──► asyncio
  │     │
  │     └─► No ──► threading
  │
  └─► CPU-bound
        │
        ▼
      Need to share data frequently?
        │
        ├─► Yes ──► threading (with GIL limitations)
        │
        └─► No ──► multiprocessing
```

---

## Benchmarking the Approaches

### I/O-Bound: HTTP Requests

```python
import asyncio
import threading
import multiprocessing as mp
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

URL = "https://httpbin.org/delay/0.5"
N_REQUESTS = 10

def fetch_sync(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return len(r.read())

async def fetch_async(url):
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
        return len(r.content)

# Sequential
def benchmark_sequential():
    start = time.perf_counter()
    for _ in range(N_REQUESTS):
        fetch_sync(URL)
    return time.perf_counter() - start

# Threading
def benchmark_threading():
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(fetch_sync, [URL] * N_REQUESTS))
    return time.perf_counter() - start

# Asyncio
def benchmark_asyncio():
    async def main():
        tasks = [fetch_async(URL) for _ in range(N_REQUESTS)]
        return await asyncio.gather(*tasks)
    
    start = time.perf_counter()
    asyncio.run(main())
    return time.perf_counter() - start

if __name__ == '__main__':
    print(f"Sequential: {benchmark_sequential():.2f}s")
    print(f"Threading:  {benchmark_threading():.2f}s")
    print(f"Asyncio:    {benchmark_asyncio():.2f}s")
```

**Expected Results:**
```
Sequential: 5.00s   (10 × 0.5s delay)
Threading:  0.60s   (~10x faster)
Asyncio:    0.55s   (slightly faster than threading)
```

### CPU-Bound: Number Crunching

```python
import asyncio
import threading
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_work(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

N = 10_000_000
TASKS = 4

# Sequential
def benchmark_sequential():
    start = time.perf_counter()
    for _ in range(TASKS):
        cpu_work(N)
    return time.perf_counter() - start

# Threading (limited by GIL)
def benchmark_threading():
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=TASKS) as executor:
        list(executor.map(lambda x: cpu_work(N), range(TASKS)))
    return time.perf_counter() - start

# Multiprocessing (true parallelism)
def benchmark_multiprocessing():
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=TASKS) as executor:
        list(executor.map(cpu_work, [N] * TASKS))
    return time.perf_counter() - start

if __name__ == '__main__':
    print(f"Sequential:      {benchmark_sequential():.2f}s")
    print(f"Threading:       {benchmark_threading():.2f}s")
    print(f"Multiprocessing: {benchmark_multiprocessing():.2f}s")
```

**Expected Results (4-core machine):**
```
Sequential:      4.00s
Threading:       4.10s   (No improvement - GIL!)
Multiprocessing: 1.10s   (~4x faster)
```

---

## Hybrid Patterns

### I/O Then CPU Processing

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

async def fetch_data(url):
    """I/O-bound: fetch from API."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

def process_data(data):
    """CPU-bound: heavy processing."""
    # Simulate CPU work
    result = sum(x ** 2 for x in range(100000))
    return {"processed": result, "original": data}

async def hybrid_pipeline(urls):
    """Combine async I/O with multiprocessing CPU work."""
    
    # Phase 1: Fetch all data concurrently (I/O-bound)
    print("Fetching data...")
    fetch_tasks = [fetch_data(url) for url in urls]
    raw_data = await asyncio.gather(*fetch_tasks)
    
    # Phase 2: Process data in parallel (CPU-bound)
    print("Processing data...")
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        process_tasks = [
            loop.run_in_executor(executor, process_data, data)
            for data in raw_data
        ]
        processed = await asyncio.gather(*process_tasks)
    
    return processed

# asyncio.run(hybrid_pipeline(urls))
```

### Async with Thread Pool Fallback

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Some libraries are sync-only
def blocking_database_call(query):
    import time
    time.sleep(0.5)  # Simulate blocking call
    return f"Result for: {query}"

async def async_wrapper(query):
    """Run blocking code in thread pool."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor, 
            blocking_database_call, 
            query
        )

async def main():
    queries = ["SELECT 1", "SELECT 2", "SELECT 3"]
    
    # All blocking calls run concurrently in threads
    results = await asyncio.gather(*[
        async_wrapper(q) for q in queries
    ])
    
    print(results)

asyncio.run(main())
```

---

## Memory Considerations

### Memory Usage by Approach

| Approach | Memory per Task | Suitable For |
|----------|----------------|--------------|
| **asyncio** | ~KB (coroutines are lightweight) | 100,000+ concurrent tasks |
| **threading** | ~8KB (thread stack) | 100s to 1000s of tasks |
| **multiprocessing** | ~MB (full process) | 10s to 100s of tasks |

### Controlling Memory

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Limit workers to control memory
n_io_workers = 50  # For I/O-bound (can be high)
n_cpu_workers = mp.cpu_count()  # For CPU-bound (match cores)

# ThreadPoolExecutor: Lower memory
with ThreadPoolExecutor(max_workers=n_io_workers) as executor:
    pass  # Good for many concurrent I/O tasks

# ProcessPoolExecutor: Higher memory
with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
    pass  # Keep workers limited to CPU count
```

### Large Data with Multiprocessing

```python
import multiprocessing as mp
import numpy as np

def process_chunk(shared_array_info, start, end):
    """Process a slice of shared array."""
    array = np.frombuffer(shared_array_info[0], dtype=np.float64)
    array = array.reshape(shared_array_info[1])
    
    # Process in-place
    for i in range(start, end):
        array[i] *= 2

if __name__ == '__main__':
    # Create shared memory array
    size = 10_000_000
    shared_array = mp.Array('d', size)
    
    # Initialize
    np_array = np.frombuffer(shared_array.get_obj(), dtype=np.float64)
    np_array[:] = np.arange(size)
    
    # Process in parallel (no data copying!)
    n_workers = mp.cpu_count()
    chunk_size = size // n_workers
    
    processes = []
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else size
        p = mp.Process(
            target=process_chunk,
            args=((shared_array.get_obj(), (size,)), start, end)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
```

---

## Debugging Concurrent Code

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Race condition** | Unsynchronized access | Use locks |
| **Deadlock** | Circular lock waiting | Lock ordering |
| **Starvation** | Thread never gets resources | Fair scheduling |
| **Memory leak** | Resources not released | Context managers |

### Debugging Tools

```python
import logging
import threading

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Track thread activity
def worker(name):
    thread = threading.current_thread()
    logging.debug(f"Thread {name} ({thread.name}): starting")
    # ... work ...
    logging.debug(f"Thread {name} ({thread.name}): done")

# Check active threads
print(f"Active threads: {threading.active_count()}")
for t in threading.enumerate():
    print(f"  {t.name}: alive={t.is_alive()}")
```

### Timeouts to Prevent Hangs

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

def slow_task():
    time.sleep(10)
    return "done"

with ThreadPoolExecutor() as executor:
    future = executor.submit(slow_task)
    
    try:
        result = future.result(timeout=2)
    except TimeoutError:
        print("Task timed out!")
        future.cancel()
```

---

## Summary Table

| Criterion | asyncio | threading | multiprocessing |
|-----------|---------|-----------|-----------------|
| **Best for** | I/O-bound, many connections | I/O-bound, simpler code | CPU-bound |
| **GIL** | Single thread (N/A) | Limited | Bypassed |
| **Memory** | Very low | Low | High |
| **Overhead** | Low | Low | High |
| **Debugging** | Different mindset | Race conditions | Pickling issues |
| **Learning curve** | Medium | Low | Low |

---

## Hands-on Exercise

### Your Task

```python
# Profile a mixed workload:
# 1. Fetch data from 5 URLs (I/O-bound)
# 2. Process each response (CPU-bound)
# 3. Implement three versions:
#    a) All sequential
#    b) All threading
#    c) Hybrid (async fetch + process pool)
# 4. Compare times
```

<details>
<summary>✅ Solution</summary>

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Simulate I/O (in real code: HTTP requests)
def simulate_io(n):
    time.sleep(0.5)  # Network delay
    return list(range(n))

# Simulate CPU work
def simulate_cpu(data):
    return sum(x ** 2 for x in data)

# === Sequential ===
def sequential():
    results = []
    for i in range(5):
        data = simulate_io(10000)
        result = simulate_cpu(data)
        results.append(result)
    return results

# === All Threading ===
def all_threading():
    def task(n):
        data = simulate_io(n)
        return simulate_cpu(data)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(task, [10000] * 5))

# === Hybrid ===
async def hybrid():
    loop = asyncio.get_event_loop()
    
    # I/O phase: async with thread pool for blocking I/O
    with ThreadPoolExecutor(5) as io_pool:
        io_tasks = [
            loop.run_in_executor(io_pool, simulate_io, 10000)
            for _ in range(5)
        ]
        data_list = await asyncio.gather(*io_tasks)
    
    # CPU phase: process pool
    with ProcessPoolExecutor(mp.cpu_count()) as cpu_pool:
        cpu_tasks = [
            loop.run_in_executor(cpu_pool, simulate_cpu, data)
            for data in data_list
        ]
        results = await asyncio.gather(*cpu_tasks)
    
    return results

if __name__ == '__main__':
    # Sequential
    start = time.perf_counter()
    r1 = sequential()
    t1 = time.perf_counter() - start
    print(f"Sequential: {t1:.2f}s")
    
    # Threading
    start = time.perf_counter()
    r2 = all_threading()
    t2 = time.perf_counter() - start
    print(f"Threading:  {t2:.2f}s")
    
    # Hybrid
    start = time.perf_counter()
    r3 = asyncio.run(hybrid())
    t3 = time.perf_counter() - start
    print(f"Hybrid:     {t3:.2f}s")
    
    print(f"\nSpeedups: Threading {t1/t2:.1f}x, Hybrid {t1/t3:.1f}x")
```
</details>

---

## Summary

✅ **asyncio** for high-concurrency I/O (many connections)
✅ **threading** for simpler I/O-bound code
✅ **multiprocessing** for CPU-bound work
✅ **Hybrid** patterns combine approaches effectively
✅ Consider **memory** when scaling workers
✅ Use **timeouts** to prevent hangs

**Next:** [AI Applications](./06-ai-applications.md)

---

## Further Reading

- [Concurrency vs Parallelism](https://realpython.com/python-concurrency/)
- [Async IO Guide](https://realpython.com/async-io-python/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/concurrency.html
-->
