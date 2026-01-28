---
title: "Threading Module"
---

# Threading Module

## Introduction

Python's `threading` module provides a high-level interface for creating and managing threads. While limited by the GIL for CPU work, threading excels at I/O-bound operations.

### What We'll Cover

- Thread class basics
- Starting and joining threads
- Thread synchronization (Lock, RLock)
- ThreadPoolExecutor
- Thread-safe patterns

### Prerequisites

- Understanding of the GIL

---

## Thread Class Basics

### Creating Threads

```python
import threading
import time

def worker(name, seconds):
    print(f"Thread {name}: starting")
    time.sleep(seconds)
    print(f"Thread {name}: done")

# Create thread
t = threading.Thread(target=worker, args=("A", 2))

# Start thread
t.start()

print("Main: thread started")

# Wait for thread to finish
t.join()

print("Main: thread done")
```

**Output:**
```
Thread A: starting
Main: thread started
Thread A: done
Main: thread done
```

### Thread with Return Value

```python
import threading

def worker(n, results, index):
    """Store result in shared list."""
    result = n * n
    results[index] = result

# Shared storage for results
results = [None, None, None]

threads = []
for i, n in enumerate([10, 20, 30]):
    t = threading.Thread(target=worker, args=(n, results, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Results: {results}")  # [100, 400, 900]
```

---

## Thread Synchronization

### The Problem: Race Conditions

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(100_000):
        counter += 1  # Not atomic!

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Should be 500,000, but isn't!
```

### Solution: Lock

```python
import threading

counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    for _ in range(100_000):
        with lock:  # Only one thread at a time
            counter += 1

threads = [threading.Thread(target=safe_increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Exactly 500,000!
```

### RLock for Reentrant Locking

```python
import threading

rlock = threading.RLock()

def outer():
    with rlock:
        print("Outer acquired lock")
        inner()  # Can acquire same lock again

def inner():
    with rlock:  # Would deadlock with regular Lock!
        print("Inner acquired lock")

outer()
```

---

## Thread-Safe Queue

### Producer-Consumer Pattern

```python
import threading
import queue
import time
import random

def producer(q: queue.Queue, name: str):
    for i in range(5):
        item = f"{name}-item-{i}"
        q.put(item)
        print(f"Producer {name}: added {item}")
        time.sleep(random.uniform(0.1, 0.5))
    q.put(None)  # Signal done

def consumer(q: queue.Queue, name: str):
    while True:
        item = q.get()
        if item is None:
            q.put(None)  # Pass signal
            break
        print(f"Consumer {name}: got {item}")
        q.task_done()

q = queue.Queue(maxsize=3)

# Start threads
t1 = threading.Thread(target=producer, args=(q, "P1"))
t2 = threading.Thread(target=consumer, args=(q, "C1"))
t3 = threading.Thread(target=consumer, args=(q, "C2"))

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()
```

---

## ThreadPoolExecutor

### Managed Thread Pool

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

# Use context manager for cleanup
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit individual tasks
    future = executor.submit(task, 5)
    print(f"Result: {future.result()}")  # 25
    
    # Map over many items
    numbers = range(10)
    results = list(executor.map(task, numbers))
    print(f"Results: {results}")
```

### With URLs

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

def fetch_url(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return url, len(response.read())

urls = [
    "https://python.org",
    "https://github.com",
    "https://stackoverflow.com",
]

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_url, url): url for url in urls}
    
    for future in as_completed(futures):
        url = futures[future]
        try:
            url, size = future.result()
            print(f"{url}: {size:,} bytes")
        except Exception as e:
            print(f"{url}: error - {e}")
```

---

## Thread-Safe Classes

### Thread-Safe Counter

```python
import threading

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def decrement(self):
        with self._lock:
            self._value -= 1
    
    @property
    def value(self):
        with self._lock:
            return self._value

# Usage
counter = ThreadSafeCounter()

def worker():
    for _ in range(10_000):
        counter.increment()

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter.value}")  # Exactly 100,000
```

### Thread-Safe Cache

```python
import threading
from typing import Any, Optional

class ThreadSafeCache:
    def __init__(self, maxsize: int = 100):
        self._cache = {}
        self._lock = threading.RLock()
        self._maxsize = maxsize
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        with self._lock:
            if len(self._cache) >= self._maxsize:
                # Remove oldest item
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = value
    
    def get_or_compute(self, key: str, compute_func):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            value = compute_func()
            self._cache[key] = value
            return value

# Usage
cache = ThreadSafeCache()
cache.set("user:1", {"name": "Alice"})
print(cache.get("user:1"))
```

---

## Daemon Threads

### Background Tasks

```python
import threading
import time

def background_worker():
    """Runs in background, exits when main thread exits."""
    while True:
        print("Background: working...")
        time.sleep(1)

# Daemon thread exits when main thread ends
t = threading.Thread(target=background_worker, daemon=True)
t.start()

# Main thread work
print("Main: doing work")
time.sleep(3)
print("Main: exiting (daemon will stop)")
```

### Use Cases

| Type | Behavior | Use For |
|------|----------|---------|
| **Non-daemon** | Program waits for thread | Critical work |
| **Daemon** | Thread killed when main exits | Background tasks, cleanup |

---

## Thread Local Storage

### Per-Thread Data

```python
import threading

# Thread-local storage
local_data = threading.local()

def worker(name):
    local_data.name = name  # Each thread has own copy
    process()

def process():
    print(f"Processing for: {local_data.name}")

threads = [
    threading.Thread(target=worker, args=("Alice",)),
    threading.Thread(target=worker, args=("Bob",)),
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Hands-on Exercise

### Your Task

```python
# Build a thread-safe rate limiter that:
# 1. Allows max N requests per second
# 2. Blocks if rate limit exceeded
# 3. Works correctly with multiple threads
```

<details>
<summary>✅ Solution</summary>

```python
import threading
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()
        self._lock = threading.Lock()
    
    def acquire(self):
        """Block until request is allowed."""
        while True:
            with self._lock:
                now = time.time()
                
                # Remove old requests
                while self.requests and now - self.requests[0] > self.window:
                    self.requests.popleft()
                
                # Check if we can proceed
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True
            
            # Wait a bit before retry
            time.sleep(0.01)

# Test it
limiter = RateLimiter(max_requests=5, window_seconds=1.0)
request_times = []
lock = threading.Lock()

def make_requests(n):
    for i in range(n):
        limiter.acquire()
        with lock:
            request_times.append(time.time())
        print(f"Request at {time.time():.3f}")

# 10 requests with limit of 5/second
threads = [threading.Thread(target=make_requests, args=(5,)) for _ in range(2)]
start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - start

print(f"\n10 requests took {elapsed:.2f}s (expected ~2s with 5/sec limit)")
```
</details>

---

## Summary

✅ **threading.Thread** for basic thread creation
✅ **Lock** prevents race conditions on shared data
✅ **Queue** for thread-safe producer-consumer patterns
✅ **ThreadPoolExecutor** manages thread pools cleanly
✅ **Daemon threads** for background tasks
✅ **Thread-local storage** for per-thread data

**Next:** [Multiprocessing](./03-multiprocessing.md)

---

## Further Reading

- [threading Documentation](https://docs.python.org/3/library/threading.html)
- [Thread Synchronization](https://docs.python.org/3/library/threading.html#lock-objects)

<!-- 
Sources Consulted:
- Python threading Docs: https://docs.python.org/3/library/threading.html
-->
