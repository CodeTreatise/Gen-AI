---
title: "Performance Considerations"
---

# Performance Considerations

## Introduction

NumPy is fast, but there are ways to make it even faster—and ways to accidentally slow it down. Understanding vectorization, memory layout, and common pitfalls is essential for high-performance code.

### What We'll Cover

- Vectorization vs loops
- Memory layout and views
- Avoiding common mistakes
- Profiling and optimization

### Prerequisites

- NumPy basics
- Array operations

---

## Vectorization vs Loops

### The Problem with Python Loops

```python
import numpy as np
import time

# Python loop - SLOW
def sum_python(arr):
    total = 0
    for x in arr:
        total += x
    return total

# NumPy vectorized - FAST
def sum_numpy(arr):
    return np.sum(arr)

arr = np.random.random(1000000)

# Benchmark
start = time.time()
sum_python(arr)
print(f"Python loop: {time.time() - start:.4f}s")

start = time.time()
sum_numpy(arr)
print(f"NumPy: {time.time() - start:.4f}s")

# NumPy is typically 10-100x faster!
```

### Vectorization Examples

```python
import numpy as np

arr = np.random.random(1000000)

# ❌ Slow: Python loop
result = []
for x in arr:
    result.append(x ** 2 + 2 * x + 1)
result = np.array(result)

# ✅ Fast: Vectorized
result = arr ** 2 + 2 * arr + 1

# ❌ Slow: List comprehension
result = np.array([x ** 2 for x in arr])

# ✅ Fast: Universal function
result = np.power(arr, 2)
```

### When Loops Are Unavoidable

```python
import numpy as np

# Use numba for loops that can't be vectorized
from numba import jit

@jit(nopython=True)
def custom_operation(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        # Complex logic that can't be vectorized
        if arr[i] > 0:
            result[i] = np.log(arr[i])
        else:
            result[i] = 0
    return result
```

---

## Memory Layout

### C-order vs Fortran-order

```python
import numpy as np

# C-order: row-major (default)
c_arr = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(f"C-order flags:\n{c_arr.flags}")

# Fortran-order: column-major
f_arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f"F-order flags:\n{f_arr.flags}")

# Memory layout matters for performance
# Iterate in memory order for best cache utilization
```

### Row vs Column Iteration

```python
import numpy as np
import time

arr = np.random.random((10000, 10000))

# Row-major iteration (fast for C-order)
start = time.time()
for row in arr:
    row.sum()
print(f"Row iteration: {time.time() - start:.4f}s")

# Column-major iteration (slow for C-order)
start = time.time()
for col in arr.T:
    col.sum()
print(f"Column iteration: {time.time() - start:.4f}s")
```

### Contiguous Arrays

```python
import numpy as np

arr = np.arange(12).reshape(3, 4)

# Check if contiguous
print(arr.flags['C_CONTIGUOUS'])  # True

# Slicing may create non-contiguous view
sliced = arr[:, ::2]  # Every other column
print(sliced.flags['C_CONTIGUOUS'])  # False

# Make contiguous for performance
contiguous = np.ascontiguousarray(sliced)
print(contiguous.flags['C_CONTIGUOUS'])  # True
```

---

## Views vs Copies

### Understanding Views

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# View (shares memory)
view = arr[1:4]
view[0] = 99
print(arr)  # [1 99 3 4 5] - original modified!

# Copy (independent)
copy = arr[1:4].copy()
copy[0] = 100
print(arr)  # [1 99 3 4 5] - original unchanged
```

### When Views Are Created

```python
import numpy as np

arr = np.arange(10)

# Views (no copy)
view1 = arr[2:5]           # Slicing
view2 = arr.reshape(2, 5)  # Reshape (usually)
view3 = arr.T              # Transpose

# Copies (new memory)
copy1 = arr[[1, 3, 5]]     # Fancy indexing
copy2 = arr[arr > 5]       # Boolean indexing
copy3 = arr.copy()         # Explicit copy
```

### Checking for Views

```python
import numpy as np

arr = np.arange(10)
view = arr[2:5]
copy = arr[[1, 3, 5]]

# Check base
print(view.base is arr)  # True (shares memory)
print(copy.base is arr)  # False (independent)

# Check memory sharing
print(np.shares_memory(arr, view))  # True
print(np.shares_memory(arr, copy))  # False
```

---

## Memory Efficiency

### Pre-allocation

```python
import numpy as np

n = 1000000

# ❌ Slow: Growing list
result = []
for i in range(n):
    result.append(i ** 2)
result = np.array(result)

# ✅ Fast: Pre-allocated array
result = np.empty(n)
for i in range(n):
    result[i] = i ** 2

# ✅ Best: Vectorized
result = np.arange(n) ** 2
```

### In-Place Operations

```python
import numpy as np

arr = np.random.random(1000000)

# Creates new array
result = arr * 2

# In-place (reuses memory)
arr *= 2

# Using out parameter
np.multiply(arr, 2, out=arr)
```

### Data Type Selection

```python
import numpy as np
import sys

# Float64 (default) vs Float32
f64 = np.random.random(1000000)
f32 = f64.astype(np.float32)

print(f"float64: {sys.getsizeof(f64.tobytes())} bytes")
print(f"float32: {sys.getsizeof(f32.tobytes())} bytes")
# Float32 uses half the memory

# For integers, use smallest type that fits
small_ints = np.array([1, 2, 3], dtype=np.int8)    # -128 to 127
med_ints = np.array([1, 2, 3], dtype=np.int32)     # ±2 billion
```

---

## Profiling NumPy Code

### Using timeit

```python
import numpy as np
import timeit

arr = np.random.random(100000)

# Time single operation
time_sum = timeit.timeit(lambda: np.sum(arr), number=1000)
print(f"np.sum: {time_sum:.4f}s for 1000 iterations")

# Compare approaches
def approach1():
    return np.sum(arr ** 2)

def approach2():
    return np.dot(arr, arr)

t1 = timeit.timeit(approach1, number=1000)
t2 = timeit.timeit(approach2, number=1000)
print(f"Sum of squares: {t1:.4f}s")
print(f"Dot product: {t2:.4f}s")
```

### Using %timeit in Jupyter

```python
import numpy as np

arr = np.random.random(100000)

# In Jupyter/IPython
%timeit np.sum(arr)
%timeit arr.sum()
```

### Memory Profiling

```python
import numpy as np
import tracemalloc

tracemalloc.start()

# Your code here
arr = np.random.random((1000, 1000))
result = arr @ arr.T

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB")
print(f"Peak: {peak / 1e6:.1f} MB")

tracemalloc.stop()
```

---

## Common Performance Pitfalls

### Pitfall 1: Unnecessary Copies

```python
import numpy as np

arr = np.arange(1000000)

# ❌ Creates copy
doubled = np.array(arr * 2)

# ✅ No extra copy
doubled = arr * 2  # Already returns new array
```

### Pitfall 2: Wrong Axis

```python
import numpy as np

matrix = np.random.random((1000, 10))

# ❌ Wrong axis (sums across rows, returns 10 values)
sums = np.sum(matrix, axis=1)  # Actually correct for row sums

# Be explicit about what you want
row_sums = matrix.sum(axis=1)     # Sum each row
col_sums = matrix.sum(axis=0)     # Sum each column
```

### Pitfall 3: Not Using Out Parameter

```python
import numpy as np

a = np.random.random(1000000)
b = np.random.random(1000000)
result = np.empty(1000000)

# ❌ Allocates new array each time
for _ in range(100):
    c = a + b

# ✅ Reuses existing array
for _ in range(100):
    np.add(a, b, out=result)
```

### Pitfall 4: Python Scalars

```python
import numpy as np

arr = np.random.random(1000000)

# ❌ Python float (slower)
result = arr + 1.0

# ✅ NumPy scalar (faster)
result = arr + np.float64(1.0)

# But usually not noticeable - optimize only if needed
```

---

## Optimization Checklist

```python
# Performance Optimization Checklist

# 1. ✅ Use vectorized operations instead of loops
# 2. ✅ Pre-allocate arrays when size is known
# 3. ✅ Use views instead of copies when possible
# 4. ✅ Choose appropriate dtype (float32 vs float64)
# 5. ✅ Use in-place operations when memory is tight
# 6. ✅ Access arrays in memory order (row-major for C-order)
# 7. ✅ Use numba or Cython for loops that can't be vectorized
# 8. ✅ Profile before optimizing
```

---

## Hands-on Exercise

### Your Task

```python
# Optimize this slow function:
import numpy as np

def slow_normalize(data):
    """Normalize each row to sum to 1."""
    result = []
    for row in data:
        row_sum = 0
        for val in row:
            row_sum += val
        normalized_row = []
        for val in row:
            normalized_row.append(val / row_sum)
        result.append(normalized_row)
    return np.array(result)

# Make it 100x faster using NumPy operations
data = np.random.random((1000, 100))
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np
import timeit

def slow_normalize(data):
    """Slow: Python loops."""
    result = []
    for row in data:
        row_sum = 0
        for val in row:
            row_sum += val
        normalized_row = []
        for val in row:
            normalized_row.append(val / row_sum)
        result.append(normalized_row)
    return np.array(result)

def fast_normalize(data):
    """Fast: Vectorized."""
    row_sums = data.sum(axis=1, keepdims=True)
    return data / row_sums

# Verify same result
data = np.random.random((1000, 100))
slow_result = slow_normalize(data)
fast_result = fast_normalize(data)
print(f"Results equal: {np.allclose(slow_result, fast_result)}")

# Benchmark
t_slow = timeit.timeit(lambda: slow_normalize(data), number=10)
t_fast = timeit.timeit(lambda: fast_normalize(data), number=10)

print(f"Slow: {t_slow:.4f}s")
print(f"Fast: {t_fast:.4f}s")
print(f"Speedup: {t_slow / t_fast:.0f}x")
```
</details>

---

## Summary

✅ **Vectorize** - avoid Python loops when possible
✅ Understand **views vs copies** to manage memory
✅ Use **appropriate dtypes** (float32 saves memory)
✅ **Pre-allocate** arrays and use `out` parameter
✅ Access data in **memory order** for cache efficiency
✅ **Profile first**, optimize second

**Back to:** [NumPy Essentials Overview](./00-numpy-essentials.md)

---

## Further Reading

- [NumPy Performance Tips](https://numpy.org/doc/stable/user/c-info.how-to-extend.html)
- [Numba Documentation](https://numba.pydata.org/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/user/basics.copies.html
-->
