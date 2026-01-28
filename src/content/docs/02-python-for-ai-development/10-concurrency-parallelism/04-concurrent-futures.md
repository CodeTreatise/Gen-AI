---
title: "concurrent.futures"
---

# concurrent.futures

## Introduction

The `concurrent.futures` module provides a high-level interface for asynchronous execution using threads or processes. It abstracts away the complexity of managing individual threads or processes.

### What We'll Cover

- Executor interface
- ThreadPoolExecutor
- ProcessPoolExecutor
- Future objects
- as_completed and map
- Exception handling

### Prerequisites

- Threading concepts
- Multiprocessing basics

---

## Executor Interface

### Common Interface

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Both use same interface:
# executor.submit(fn, *args, **kwargs) → Future
# executor.map(fn, *iterables) → Iterator
# executor.shutdown(wait=True)

# Always use context manager
with ThreadPoolExecutor(max_workers=4) as executor:
    # Executor is properly cleaned up
    pass
```

### ThreadPoolExecutor vs ProcessPoolExecutor

| Feature | ThreadPoolExecutor | ProcessPoolExecutor |
|---------|-------------------|---------------------|
| **Best for** | I/O-bound tasks | CPU-bound tasks |
| **GIL** | Limited by GIL | Bypasses GIL |
| **Memory** | Shared memory | Separate memory |
| **Overhead** | Low | Higher (process creation) |
| **Pickling** | Not required | Required |

---

## ThreadPoolExecutor

### Basic Usage

```python
from concurrent.futures import ThreadPoolExecutor
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
    results = executor.map(fetch_url, urls)
    
    for url, size in results:
        print(f"{url}: {size:,} bytes")
```

### With submit()

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(n)
    return n * n

with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit returns a Future immediately
    future1 = executor.submit(task, 2)
    future2 = executor.submit(task, 1)
    
    print("Tasks submitted, doing other work...")
    
    # Get results (blocks until ready)
    print(f"Result 1: {future1.result()}")
    print(f"Result 2: {future2.result()}")
```

---

## ProcessPoolExecutor

### Basic Usage

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def cpu_task(n):
    """CPU-intensive calculation."""
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    data = [1_000_000] * 8
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(cpu_task, data))
    
    print(f"Results: {results}")
```

### With submit()

```python
from concurrent.futures import ProcessPoolExecutor

def heavy_computation(x):
    return sum(i ** 2 for i in range(x))

if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(heavy_computation, n)
            for n in [100_000, 200_000, 300_000]
        ]
        
        for i, future in enumerate(futures):
            print(f"Task {i}: {future.result()}")
```

---

## Future Objects

### Future Methods

```python
from concurrent.futures import ThreadPoolExecutor
import time

def slow_task(n):
    time.sleep(n)
    return f"Done after {n}s"

with ThreadPoolExecutor() as executor:
    future = executor.submit(slow_task, 2)
    
    # Check if done (non-blocking)
    print(f"Done: {future.done()}")  # False
    
    # Check if cancelled
    print(f"Cancelled: {future.cancelled()}")  # False
    
    # Try to cancel (may not succeed if running)
    cancelled = future.cancel()
    print(f"Cancel succeeded: {cancelled}")
    
    # Get result (blocks until ready)
    result = future.result(timeout=5)  # Optional timeout
    print(f"Result: {result}")
```

### Callbacks

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(n)
    return n * n

def on_complete(future):
    print(f"Callback: result = {future.result()}")

with ThreadPoolExecutor() as executor:
    future = executor.submit(task, 2)
    future.add_done_callback(on_complete)
    
    print("Main: continuing...")
    time.sleep(3)
```

---

## as_completed()

### Process Results as They Finish

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def task(task_id):
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    return task_id, delay

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(task, i): i 
        for i in range(10)
    }
    
    # Process in order of completion (fastest first)
    for future in as_completed(futures):
        task_id = futures[future]
        try:
            result_id, delay = future.result()
            print(f"Task {result_id}: completed in {delay:.2f}s")
        except Exception as e:
            print(f"Task {task_id}: failed with {e}")
```

### With Progress Tracking

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_item(item):
    time.sleep(0.5)
    return item * 2

items = list(range(20))

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_item, item): item for item in items}
    
    completed = 0
    total = len(futures)
    
    for future in as_completed(futures):
        completed += 1
        result = future.result()
        print(f"[{completed}/{total}] Completed: {result}")
```

---

## wait()

### Wait for Specific Conditions

```python
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import time

def task(n):
    time.sleep(n)
    return n

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(task, 3),
        executor.submit(task, 1),
        executor.submit(task, 2),
    ]
    
    # Wait for first to complete
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    print(f"First done: {done.pop().result()}")
    
    # Wait for all to complete
    done, pending = wait(futures, return_when=ALL_COMPLETED)
    print(f"All done: {[f.result() for f in done]}")
```

### Return When Options

| Option | Behavior |
|--------|----------|
| `FIRST_COMPLETED` | Return when any future completes |
| `FIRST_EXCEPTION` | Return when any future raises |
| `ALL_COMPLETED` | Return when all futures complete |

---

## Exception Handling

### Handling Exceptions in Futures

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def risky_task(n):
    if n == 3:
        raise ValueError(f"Task {n} failed!")
    return n * n

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(risky_task, i) for i in range(5)]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Success: {result}")
        except ValueError as e:
            print(f"Error: {e}")
```

### Exception in map()

```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    if n == 2:
        raise ValueError("Failed!")
    return n * n

with ThreadPoolExecutor() as executor:
    try:
        # map raises on first exception
        results = list(executor.map(task, range(5)))
    except ValueError as e:
        print(f"Caught: {e}")
```

### Collect All Exceptions

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def task(n):
    if n % 2 == 0:
        raise ValueError(f"Even number: {n}")
    return n * n

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(task, n): n for n in range(6)}
    
    results = []
    errors = []
    
    for future in as_completed(futures):
        n = futures[future]
        try:
            results.append((n, future.result()))
        except Exception as e:
            errors.append((n, e))
    
    print(f"Successes: {results}")
    print(f"Errors: {errors}")
```

---

## Practical Pattern: Rate-Limited API Calls

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

class RateLimitedExecutor:
    def __init__(self, max_workers=5, calls_per_second=2):
        self.executor = ThreadPoolExecutor(max_workers)
        self.delay = 1.0 / calls_per_second
        self.lock = threading.Lock()
        self.last_call = 0
    
    def submit(self, fn, *args, **kwargs):
        def rate_limited_fn():
            with self.lock:
                now = time.time()
                wait = self.delay - (now - self.last_call)
                if wait > 0:
                    time.sleep(wait)
                self.last_call = time.time()
            return fn(*args, **kwargs)
        
        return self.executor.submit(rate_limited_fn)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.executor.shutdown(wait=True)

# Usage
def api_call(n):
    return f"Result {n}"

with RateLimitedExecutor(max_workers=10, calls_per_second=5) as executor:
    futures = [executor.submit(api_call, i) for i in range(20)]
    
    for future in as_completed(futures):
        print(future.result())
```

---

## Hands-on Exercise

### Your Task

```python
# Build a parallel file processor that:
# 1. Takes a list of file paths
# 2. Reads and processes each file in parallel
# 3. Uses as_completed for progress updates
# 4. Handles missing files gracefully
# 5. Returns successful results and errors separately
```

<details>
<summary>✅ Solution</summary>

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def process_file(filepath: str) -> dict:
    """Read and process a file."""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    content = path.read_text()
    
    return {
        "path": filepath,
        "lines": len(content.splitlines()),
        "chars": len(content),
        "words": len(content.split())
    }

def process_files_parallel(filepaths: list[str], max_workers: int = 5):
    """Process files in parallel with progress tracking."""
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, fp): fp 
            for fp in filepaths
        }
        
        total = len(futures)
        completed = 0
        
        for future in as_completed(futures):
            filepath = futures[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                print(f"[{completed}/{total}] ✓ {filepath}")
            except Exception as e:
                errors.append({"path": filepath, "error": str(e)})
                print(f"[{completed}/{total}] ✗ {filepath}: {e}")
    
    return results, errors

# Test
if __name__ == '__main__':
    files = [
        "/etc/passwd",       # Exists (on Linux)
        "/etc/hosts",        # Exists
        "/nonexistent.txt",  # Doesn't exist
    ]
    
    results, errors = process_files_parallel(files)
    
    print(f"\nResults: {len(results)}")
    for r in results:
        print(f"  {r['path']}: {r['lines']} lines")
    
    print(f"\nErrors: {len(errors)}")
    for e in errors:
        print(f"  {e['path']}: {e['error']}")
```
</details>

---

## Summary

✅ **Executor.submit()** returns Future for single task
✅ **Executor.map()** for iterating over multiple inputs
✅ **as_completed()** processes results as they finish
✅ **wait()** for custom completion conditions
✅ **Future.result()** gets result or raises exception
✅ Use **ThreadPoolExecutor** for I/O, **ProcessPoolExecutor** for CPU

**Next:** [Choosing the Right Approach](./05-choosing-approach.md)

---

## Further Reading

- [concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [PEP 3148](https://peps.python.org/pep-3148/)

<!-- 
Sources Consulted:
- Python concurrent.futures Docs: https://docs.python.org/3/library/concurrent.futures.html
-->
