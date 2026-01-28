---
title: "Python Execution Model"
---

# Python Execution Model

## Introduction

Before using concurrency in Python, you need to understand the Global Interpreter Lock (GIL). This knowledge helps you choose the right approach for different tasks.

### What We'll Cover

- What is the GIL?
- How GIL affects threading
- When GIL matters (and doesn't)
- Free-threaded Python (3.13+)

### Prerequisites

- Basic Python knowledge

---

## What is the GIL?

### The Global Interpreter Lock

```python
# The GIL is a mutex that protects access to Python objects
# Only ONE thread can execute Python bytecode at a time

# Even with multiple threads:
import threading

counter = 0

def increment():
    global counter
    for _ in range(1_000_000):
        counter += 1  # Only one thread executes this at a time

# Start 2 threads
t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # May not be 2,000,000 due to race conditions!
```

### Why Does the GIL Exist?

| Reason | Explanation |
|--------|-------------|
| **Memory safety** | Protects Python's memory management |
| **Reference counting** | Python uses ref counting for garbage collection |
| **Simplicity** | Makes C extensions easier to write |
| **Historical** | Designed when multi-core CPUs were rare |

---

## GIL Impact on Threading

### CPU-Bound Tasks (GIL Hurts)

```python
import threading
import time

def cpu_task(n):
    """CPU-intensive calculation."""
    total = 0
    for i in range(n):
        total += i * i
    return total

# Sequential
start = time.perf_counter()
cpu_task(50_000_000)
cpu_task(50_000_000)
seq_time = time.perf_counter() - start
print(f"Sequential: {seq_time:.2f}s")

# Threaded (NO speedup due to GIL!)
start = time.perf_counter()
t1 = threading.Thread(target=cpu_task, args=(50_000_000,))
t2 = threading.Thread(target=cpu_task, args=(50_000_000,))
t1.start()
t2.start()
t1.join()
t2.join()
threaded_time = time.perf_counter() - start
print(f"Threaded: {threaded_time:.2f}s")  # Same or slower!
```

**Output:**
```
Sequential: 4.50s
Threaded: 4.48s  (No improvement!)
```

### I/O-Bound Tasks (GIL Doesn't Hurt)

```python
import threading
import time
import urllib.request

def fetch_url(url):
    """I/O-bound task - network request."""
    with urllib.request.urlopen(url) as response:
        return len(response.read())

urls = ["https://python.org"] * 5

# Sequential
start = time.perf_counter()
for url in urls:
    fetch_url(url)
seq_time = time.perf_counter() - start
print(f"Sequential: {seq_time:.2f}s")

# Threaded (MUCH faster - GIL released during I/O)
start = time.perf_counter()
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
for t in threads:
    t.start()
for t in threads:
    t.join()
threaded_time = time.perf_counter() - start
print(f"Threaded: {threaded_time:.2f}s")
```

**Output:**
```
Sequential: 2.50s
Threaded: 0.52s  (5x faster!)
```

---

## When GIL Matters

### GIL Matters (CPU-Bound)

```python
# ❌ GIL limits these tasks
- Mathematical calculations
- Data transformation
- Image processing
- Model inference (pure Python)
- Cryptography
- Compression

# Solution: Use multiprocessing
from multiprocessing import Pool

with Pool(4) as p:
    results = p.map(cpu_task, data_chunks)
```

### GIL Doesn't Matter (I/O-Bound)

```python
# ✅ GIL released during these operations
- Network requests (HTTP, sockets)
- File I/O (reading, writing)
- Database queries
- Sleep/waiting
- External C library calls (NumPy, etc.)

# Threading works well here
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(10) as executor:
    results = list(executor.map(fetch_url, urls))
```

---

## Visualizing the GIL

### CPU-Bound (Threads Don't Help)

```
Thread 1: [====RUNNING====]................[====RUNNING====]
Thread 2: ................[====RUNNING====]................
          ^-- GIL prevents true parallel execution
          
Time: |---------|---------|---------|---------|
      0         1         2         3         4 seconds
```

### I/O-Bound (Threads Help)

```
Thread 1: [RUN][---WAIT I/O---][RUN]
Thread 2: .....[RUN][---WAIT I/O---][RUN]
Thread 3: ..........[RUN][---WAIT I/O---][RUN]
          ^-- GIL released during I/O wait
          
Time: |---------|---------|
      0         1         2 seconds (all complete!)
```

---

## Multiprocessing: Bypassing the GIL

### Separate Processes = Separate GILs

```python
import multiprocessing as mp
import time

def cpu_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == '__main__':
    # Sequential
    start = time.perf_counter()
    cpu_task(50_000_000)
    cpu_task(50_000_000)
    seq_time = time.perf_counter() - start
    print(f"Sequential: {seq_time:.2f}s")
    
    # Multiprocessing (TRUE parallelism!)
    start = time.perf_counter()
    with mp.Pool(2) as pool:
        pool.map(cpu_task, [50_000_000, 50_000_000])
    mp_time = time.perf_counter() - start
    print(f"Multiprocessing: {mp_time:.2f}s")
```

**Output:**
```
Sequential: 4.50s
Multiprocessing: 2.30s  (2x faster on 2 cores!)
```

---

## Free-Threaded Python (3.13+)

### Experimental No-GIL Build

```python
# Python 3.13 introduces experimental free-threading
# Must be built with --disable-gil flag

# Check if GIL is disabled
import sys
if hasattr(sys, '_is_gil_enabled'):
    print(f"GIL enabled: {sys._is_gil_enabled()}")

# With free-threading, CPU-bound tasks CAN benefit from threads
# But: Not production-ready yet, some packages may break
```

### Current Status (2024)

| Feature | Status |
|---------|--------|
| **Availability** | Python 3.13+ experimental |
| **Stability** | Not recommended for production |
| **Performance** | Some overhead without GIL |
| **Compatibility** | Many C extensions need updates |

---

## Summary Table

| Scenario | Use | Why |
|----------|-----|-----|
| **CPU-bound, need speed** | Multiprocessing | Bypasses GIL |
| **I/O-bound, many tasks** | Threading/asyncio | GIL released during I/O |
| **Simple parallelism** | concurrent.futures | Clean API |
| **Maximum I/O concurrency** | asyncio | Single-threaded, efficient |

---

## Hands-on Exercise

### Your Task

```python
# Demonstrate the GIL's effect:
# 1. Create a CPU-bound function that takes ~1 second
# 2. Run it twice sequentially and time it
# 3. Run it twice with threads and time it
# 4. Run it twice with multiprocessing and time it
# 5. Compare the results
```

<details>
<summary>✅ Solution</summary>

```python
import threading
import multiprocessing as mp
import time

def cpu_work():
    """CPU-bound work (~1 second)."""
    total = 0
    for i in range(20_000_000):
        total += i * i % 1000
    return total

def benchmark_sequential():
    start = time.perf_counter()
    cpu_work()
    cpu_work()
    return time.perf_counter() - start

def benchmark_threaded():
    start = time.perf_counter()
    t1 = threading.Thread(target=cpu_work)
    t2 = threading.Thread(target=cpu_work)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return time.perf_counter() - start

def benchmark_multiprocessing():
    start = time.perf_counter()
    with mp.Pool(2) as pool:
        pool.map(lambda x: cpu_work(), [1, 2])
    return time.perf_counter() - start

if __name__ == '__main__':
    seq = benchmark_sequential()
    print(f"Sequential:      {seq:.2f}s")
    
    thr = benchmark_threaded()
    print(f"Threading:       {thr:.2f}s ({seq/thr:.1f}x)")
    
    proc = benchmark_multiprocessing()
    print(f"Multiprocessing: {proc:.2f}s ({seq/proc:.1f}x)")
```

**Expected Output:**
```
Sequential:      2.00s
Threading:       2.00s (1.0x)  ← GIL prevents speedup
Multiprocessing: 1.05s (1.9x)  ← True parallelism!
```
</details>

---

## Summary

✅ The **GIL** allows only one thread to execute Python bytecode at a time
✅ **Threading** doesn't speed up CPU-bound tasks
✅ **Threading** works great for I/O-bound tasks (GIL released during I/O)
✅ **Multiprocessing** bypasses GIL for true parallelism
✅ **Free-threaded Python** (3.13+) is experimental

**Next:** [Threading Module](./02-threading.md)

---

## Further Reading

- [Python GIL Documentation](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
- [Understanding the GIL](https://realpython.com/python-gil/)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/glossary.html#term-global-interpreter-lock
-->
