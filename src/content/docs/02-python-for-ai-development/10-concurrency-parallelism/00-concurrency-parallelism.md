---
title: "Concurrency & Parallelism"
---

# Concurrency & Parallelism

## Overview

Understanding Python's concurrency model is essential for AI developers. Process large datasets, make concurrent API calls, and run parallel model inference by choosing the right approach.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-python-execution-model.md) | Python Execution Model | GIL, threads, processes |
| [02](./02-threading.md) | Threading Module | Thread class, locks, pools |
| [03](./03-multiprocessing.md) | Multiprocessing | Process, Pool, shared memory |
| [04](./04-concurrent-futures.md) | concurrent.futures | Executors, futures, map |
| [05](./05-choosing-approach.md) | Choosing the Right Approach | I/O vs CPU, hybrid patterns |
| [06](./06-ai-applications.md) | AI Applications | Parallel inference, pipelines |

---

## When to Use What

| Task Type | Approach | Why |
|-----------|----------|-----|
| **API calls** | asyncio/threading | I/O-bound, GIL released |
| **File I/O** | threading/asyncio | I/O-bound |
| **Data processing** | multiprocessing | CPU-bound, bypasses GIL |
| **Model inference** | multiprocessing | CPU-bound computation |
| **Web scraping** | asyncio | Many concurrent connections |

---

## Quick Comparison

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O-bound: Use threads
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_url, urls))

# CPU-bound: Use processes  
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_data, chunks))
```

---

## Prerequisites

Before starting this lesson:
- Python functions
- Basic async concepts helpful
- Understanding of I/O vs CPU work

---

## Start Learning

Begin with [Python Execution Model](./01-python-execution-model.md) to understand the GIL.

---

## Further Reading

- [Python threading Docs](https://docs.python.org/3/library/threading.html)
- [Python multiprocessing Docs](https://docs.python.org/3/library/multiprocessing.html)
- [concurrent.futures Docs](https://docs.python.org/3/library/concurrent.futures.html)
