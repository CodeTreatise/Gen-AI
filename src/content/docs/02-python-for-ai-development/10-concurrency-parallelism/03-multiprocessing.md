---
title: "Multiprocessing Module"
---

# Multiprocessing Module

## Introduction

The `multiprocessing` module enables true parallelism by running code in separate processes, each with its own Python interpreter and GIL. Essential for CPU-bound tasks.

### What We'll Cover

- Process class basics
- Inter-process communication
- Shared memory
- Pool for parallel mapping
- ProcessPoolExecutor

### Prerequisites

- Understanding of GIL
- Threading concepts

---

## Process Class Basics

### Creating Processes

```python
import multiprocessing as mp
import os

def worker(name):
    print(f"Worker {name}: PID={os.getpid()}")
    return name * 2

if __name__ == '__main__':
    print(f"Main: PID={os.getpid()}")
    
    p = mp.Process(target=worker, args=("A",))
    p.start()
    p.join()
    
    print("Main: done")
```

**Output:**
```
Main: PID=12345
Worker A: PID=12346  (Different process!)
Main: done
```

> **Note:** Always use `if __name__ == '__main__':` guard on Windows.

### Multiple Processes

```python
import multiprocessing as mp
import time

def cpu_task(n):
    """CPU-intensive task."""
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == '__main__':
    # Create processes
    processes = []
    for i in range(4):
        p = mp.Process(target=cpu_task, args=(10_000_000,))
        processes.append(p)
    
    # Start all
    start = time.perf_counter()
    for p in processes:
        p.start()
    
    # Wait for all
    for p in processes:
        p.join()
    
    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.2f}s")
```

---

## Inter-Process Communication

### Queue

```python
import multiprocessing as mp

def producer(queue, items):
    for item in items:
        queue.put(item)
        print(f"Produced: {item}")
    queue.put(None)  # Signal done

def consumer(queue, results):
    while True:
        item = queue.get()
        if item is None:
            break
        result = item * 2
        results.put(result)
        print(f"Consumed: {item} → {result}")

if __name__ == '__main__':
    queue = mp.Queue()
    results = mp.Queue()
    
    p1 = mp.Process(target=producer, args=(queue, [1, 2, 3, 4, 5]))
    p2 = mp.Process(target=consumer, args=(queue, results))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
    # Collect results
    output = []
    while not results.empty():
        output.append(results.get())
    print(f"Results: {output}")
```

### Pipe

```python
import multiprocessing as mp

def sender(conn, messages):
    for msg in messages:
        conn.send(msg)
        print(f"Sent: {msg}")
    conn.send(None)  # Signal done
    conn.close()

def receiver(conn):
    while True:
        msg = conn.recv()
        if msg is None:
            break
        print(f"Received: {msg}")
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    
    p1 = mp.Process(target=sender, args=(parent_conn, ["Hello", "World"]))
    p2 = mp.Process(target=receiver, args=(child_conn,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
```

---

## Shared Memory

### Value and Array

```python
import multiprocessing as mp

def increment(counter, lock):
    for _ in range(10000):
        with lock:
            counter.value += 1

def fill_array(arr, value):
    for i in range(len(arr)):
        arr[i] = value

if __name__ == '__main__':
    # Shared counter
    counter = mp.Value('i', 0)  # 'i' = integer
    lock = mp.Lock()
    
    processes = [
        mp.Process(target=increment, args=(counter, lock))
        for _ in range(4)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(f"Counter: {counter.value}")  # 40000
    
    # Shared array
    arr = mp.Array('d', 10)  # 'd' = double, 10 elements
    p = mp.Process(target=fill_array, args=(arr, 3.14))
    p.start()
    p.join()
    
    print(f"Array: {list(arr)}")
```

### Type Codes

| Code | Type | Size |
|------|------|------|
| `'i'` | int | 4 bytes |
| `'d'` | double | 8 bytes |
| `'f'` | float | 4 bytes |
| `'c'` | char | 1 byte |

---

## Pool for Parallel Mapping

### Basic Pool.map

```python
import multiprocessing as mp

def process_item(item):
    """CPU-bound processing."""
    return item ** 2

if __name__ == '__main__':
    items = list(range(100))
    
    # Use all CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_item, items)
    
    print(f"Processed {len(results)} items")
    print(f"First 10: {results[:10]}")
```

### Pool Methods

```python
import multiprocessing as mp
import time

def slow_task(x):
    time.sleep(0.1)
    return x * x

if __name__ == '__main__':
    with mp.Pool(4) as pool:
        # map: Blocks until all complete, preserves order
        results = pool.map(slow_task, range(10))
        print(f"map: {results}")
        
        # imap: Iterator, lazy evaluation
        for result in pool.imap(slow_task, range(10)):
            print(f"imap result: {result}")
        
        # imap_unordered: Fastest results first
        for result in pool.imap_unordered(slow_task, range(10)):
            print(f"imap_unordered: {result}")
        
        # apply_async: Single task, non-blocking
        future = pool.apply_async(slow_task, (42,))
        result = future.get()  # Blocks until done
        print(f"apply_async: {result}")
```

### Chunking for Large Data

```python
import multiprocessing as mp
import numpy as np

def process_chunk(chunk):
    """Process a chunk of data."""
    return [x ** 2 for x in chunk]

if __name__ == '__main__':
    data = list(range(10_000))
    n_workers = mp.cpu_count()
    
    # Split into chunks
    chunk_size = len(data) // n_workers
    chunks = [
        data[i:i + chunk_size] 
        for i in range(0, len(data), chunk_size)
    ]
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Flatten results
    flat_results = [x for chunk in results for x in chunk]
    print(f"Processed {len(flat_results)} items")
```

---

## ProcessPoolExecutor

### Clean API

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def cpu_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == '__main__':
    items = [1_000_000] * 10
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Submit all tasks
        futures = [executor.submit(cpu_task, item) for item in items]
        
        # Process as completed
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"Task {i}: {result}")
```

### With map

```python
from concurrent.futures import ProcessPoolExecutor

def process(x):
    return x ** 2

if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process, range(100)))
    print(results[:10])
```

---

## Avoiding Pickling Issues

### The Problem

```python
import multiprocessing as mp

class MyClass:
    def process(self, x):
        return x * 2

def worker(obj, x):
    return obj.process(x)

if __name__ == '__main__':
    obj = MyClass()
    
    # This may fail - lambda can't be pickled!
    # pool.map(lambda x: obj.process(x), [1,2,3])
    
    # Use module-level function instead
    with mp.Pool(2) as pool:
        results = pool.starmap(worker, [(obj, x) for x in [1,2,3]])
    print(results)
```

### What Can Be Pickled

| ✅ Can Pickle | ❌ Cannot Pickle |
|--------------|-----------------|
| Built-in types | Lambda functions |
| Module-level functions | Nested functions |
| Classes defined at module level | Inner classes |
| Class instances | Open files |
| Most standard library objects | Database connections |

### Solution: Module-Level Functions

```python
import multiprocessing as mp

# Define at module level
def process_data(data):
    return data * 2

if __name__ == '__main__':
    with mp.Pool(4) as pool:
        results = pool.map(process_data, [1, 2, 3, 4, 5])
    print(results)
```

---

## Hands-on Exercise

### Your Task

```python
# Build a parallel data processor that:
# 1. Takes a large list of numbers
# 2. Splits into chunks for each CPU core
# 3. Processes each chunk in parallel
# 4. Returns combined results
# 5. Compare with sequential processing
```

<details>
<summary>✅ Solution</summary>

```python
import multiprocessing as mp
import time

def process_chunk(chunk):
    """CPU-intensive processing on a chunk."""
    results = []
    for x in chunk:
        # Simulate heavy computation
        result = sum(i * i for i in range(x % 1000))
        results.append(result)
    return results

def parallel_process(data, n_workers=None):
    """Process data in parallel across workers."""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Split into chunks
    chunk_size = max(1, len(data) // n_workers)
    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]
    
    # Process in parallel
    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(process_chunk, chunks)
    
    # Flatten
    return [x for chunk in chunk_results for x in chunk]

def sequential_process(data):
    """Process data sequentially."""
    return process_chunk(data)

if __name__ == '__main__':
    data = list(range(10_000))
    
    # Sequential
    start = time.perf_counter()
    seq_results = sequential_process(data)
    seq_time = time.perf_counter() - start
    print(f"Sequential: {seq_time:.2f}s")
    
    # Parallel
    start = time.perf_counter()
    par_results = parallel_process(data)
    par_time = time.perf_counter() - start
    print(f"Parallel:   {par_time:.2f}s")
    
    # Verify same results
    print(f"Results match: {seq_results == par_results}")
    print(f"Speedup: {seq_time/par_time:.1f}x")
```
</details>

---

## Summary

✅ **Process** runs code in separate interpreter (bypasses GIL)
✅ **Queue/Pipe** for inter-process communication
✅ **Value/Array** for shared memory
✅ **Pool.map** for parallel iteration
✅ **ProcessPoolExecutor** for clean API
✅ Avoid lambda functions (use module-level functions)

**Next:** [concurrent.futures](./04-concurrent-futures.md)

---

## Further Reading

- [multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Shared Memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)

<!-- 
Sources Consulted:
- Python multiprocessing Docs: https://docs.python.org/3/library/multiprocessing.html
-->
