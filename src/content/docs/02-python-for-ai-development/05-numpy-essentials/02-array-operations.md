---
title: "Array Operations"
---

# Array Operations

## Introduction

NumPy's power lies in vectorized operations—applying functions to entire arrays without explicit loops. These operations are faster and more readable than Python loops.

### What We'll Cover

- Element-wise operations
- Arithmetic operations
- Comparison operations
- Aggregation functions
- Universal functions (ufuncs)

### Prerequisites

- NumPy arrays basics

---

## Element-wise Operations

### Basic Arithmetic

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Addition
print(a + b)  # [11 22 33 44]

# Subtraction
print(b - a)  # [9 18 27 36]

# Multiplication
print(a * b)  # [10 40 90 160]

# Division
print(b / a)  # [10. 10. 10. 10.]

# Power
print(a ** 2)  # [1 4 9 16]
```

### Scalar Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4])

# Broadcast scalar to all elements
print(arr + 10)   # [11 12 13 14]
print(arr * 3)    # [3 6 9 12]
print(arr / 2)    # [0.5 1.  1.5 2. ]
print(arr ** 2)   # [1 4 9 16]
```

### In-place Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4], dtype=float)

# Modify in place (no new array created)
arr += 10
print(arr)  # [11. 12. 13. 14.]

arr *= 2
print(arr)  # [22. 24. 26. 28.]

# Using np.add with out parameter
np.add(arr, 1, out=arr)
```

---

## Comparison Operations

### Element-wise Comparisons

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Returns boolean array
print(arr > 3)     # [False False False  True  True]
print(arr == 3)    # [False False  True False False]
print(arr != 2)    # [ True False  True  True  True]
print(arr >= 2)    # [False  True  True  True  True]
```

### Array Comparisons

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 3, 2])

print(a == b)  # [ True False False]
print(a < b)   # [False  True False]

# All elements equal?
print(np.array_equal(a, b))  # False

# Close enough (for floats)?
print(np.allclose([1.0, 2.0], [1.0000001, 2.0]))  # True
```

### Logical Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Combine conditions
mask1 = arr > 2
mask2 = arr < 5

print(mask1 & mask2)  # [False False  True  True False]
print(mask1 | mask2)  # [ True  True  True  True  True]
print(~mask1)         # [ True  True False False False]

# Using np functions
print(np.logical_and(mask1, mask2))
print(np.logical_or(mask1, mask2))
print(np.logical_not(mask1))
```

---

## Aggregation Functions

### Basic Aggregations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(np.sum(arr))   # 15
print(np.prod(arr))  # 120 (product)
print(np.mean(arr))  # 3.0
print(np.std(arr))   # 1.414...
print(np.var(arr))   # 2.0 (variance)
print(np.min(arr))   # 1
print(np.max(arr))   # 5
print(np.argmin(arr))  # 0 (index of min)
print(np.argmax(arr))  # 4 (index of max)
```

### Method Syntax

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Methods work the same
print(arr.sum())   # 15
print(arr.mean())  # 3.0
print(arr.min())   # 1
print(arr.max())   # 5
```

### Axis Parameter

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Sum all elements
print(np.sum(matrix))  # 21

# Sum along rows (axis=1)
print(np.sum(matrix, axis=1))  # [6 15]

# Sum along columns (axis=0)
print(np.sum(matrix, axis=0))  # [5 7 9]
```

### Understanding axis

```
axis=0: operates DOWN columns
axis=1: operates ACROSS rows

[[1, 2, 3],     axis=0: [5, 7, 9] (sum each column)
 [4, 5, 6]]     axis=1: [6, 15] (sum each row)
```

---

## Statistical Functions

```python
import numpy as np

data = np.array([1, 2, 2, 3, 4, 4, 4, 5])

# Central tendency
print(np.mean(data))    # 3.125
print(np.median(data))  # 3.5

# Spread
print(np.std(data))     # 1.25 (standard deviation)
print(np.var(data))     # 1.5625 (variance)

# Percentiles
print(np.percentile(data, 25))  # 2.0 (25th percentile)
print(np.percentile(data, 75))  # 4.0 (75th percentile)

# Range
print(np.ptp(data))     # 4 (peak-to-peak: max - min)
```

### Cumulative Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(np.cumsum(arr))   # [1 3 6 10 15] (cumulative sum)
print(np.cumprod(arr))  # [1 2 6 24 120] (cumulative product)
print(np.diff(arr))     # [1 1 1 1] (differences)
```

---

## Universal Functions (ufuncs)

### Math Functions

```python
import numpy as np

arr = np.array([0, np.pi/4, np.pi/2, np.pi])

# Trigonometric
print(np.sin(arr))  # [0.  0.707 1.  0.]
print(np.cos(arr))  # [1.  0.707 0. -1.]
print(np.tan(arr))

# Exponential and logarithm
arr = np.array([1, 2, 3])
print(np.exp(arr))    # [2.718 7.389 20.085]
print(np.log(arr))    # [0.    0.693 1.099]
print(np.log10(arr))  # [0.    0.301 0.477]
print(np.log2(arr))   # [0.    1.    1.585]

# Power and roots
print(np.sqrt(arr))   # [1.    1.414 1.732]
print(np.power(arr, 2))  # [1 4 9]
```

### Rounding

```python
import numpy as np

arr = np.array([1.4, 2.5, 3.6, -1.5])

print(np.round(arr))    # [1. 2. 4. -2.]
print(np.floor(arr))    # [1. 2. 3. -2.]
print(np.ceil(arr))     # [2. 3. 4. -1.]
print(np.trunc(arr))    # [1. 2. 3. -1.]
print(np.abs(arr))      # [1.4 2.5 3.6 1.5]
```

### Comparison ufuncs

```python
import numpy as np

a = np.array([1, 5, 3])
b = np.array([2, 4, 3])

print(np.maximum(a, b))  # [2 5 3] (element-wise max)
print(np.minimum(a, b))  # [1 4 3] (element-wise min)
print(np.greater(a, b))  # [False  True False]
print(np.equal(a, b))    # [False False  True]
```

---

## NaN Handling

```python
import numpy as np

arr = np.array([1, 2, np.nan, 4, 5])

# Regular functions return nan
print(np.sum(arr))   # nan
print(np.mean(arr))  # nan

# NaN-safe functions
print(np.nansum(arr))   # 12.0
print(np.nanmean(arr))  # 3.0
print(np.nanstd(arr))   # 1.58...
print(np.nanmax(arr))   # 5.0

# Check for NaN
print(np.isnan(arr))  # [False False  True False False]
print(np.any(np.isnan(arr)))  # True
```

---

## Set Operations

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])

print(np.unique(np.array([1, 1, 2, 2, 3])))  # [1 2 3]
print(np.intersect1d(a, b))  # [3 4]
print(np.union1d(a, b))      # [1 2 3 4 5 6]
print(np.setdiff1d(a, b))    # [1 2] (in a but not b)
print(np.in1d(a, b))         # [False False  True  True]
```

---

## Hands-on Exercise

### Your Task

```python
# Given student scores:
scores = np.array([
    [85, 90, 78],   # Student 1: Math, Science, English
    [92, 88, 95],   # Student 2
    [70, 75, 80],   # Student 3
    [88, 92, 85]    # Student 4
])

# Calculate:
# 1. Average score per student
# 2. Average score per subject
# 3. Highest score overall
# 4. Student with highest average
# 5. Normalize scores to 0-100 scale
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np

scores = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [70, 75, 80],
    [88, 92, 85]
])

# 1. Average per student (across columns)
student_avg = np.mean(scores, axis=1)
print(f"Student averages: {student_avg}")
# [84.33 91.67 75.   88.33]

# 2. Average per subject (across rows)
subject_avg = np.mean(scores, axis=0)
print(f"Subject averages: {subject_avg}")
# [83.75 86.25 84.5]

# 3. Highest score overall
highest = np.max(scores)
print(f"Highest score: {highest}")  # 95

# 4. Student with highest average
best_student = np.argmax(student_avg)
print(f"Best student: Student {best_student + 1}")  # Student 2

# 5. Normalize to 0-100 (already 0-100, but general formula)
min_score = np.min(scores)
max_score = np.max(scores)
normalized = (scores - min_score) / (max_score - min_score) * 100
print(f"Normalized:\n{normalized}")
```
</details>

---

## Summary

✅ **Element-wise operations** apply to all elements automatically
✅ Use **axis** parameter to control aggregation direction
✅ **ufuncs** are fast vectorized math functions
✅ Use **nan-safe** functions like `np.nanmean()` for missing data
✅ **Comparison operations** return boolean arrays
✅ Prefer **NumPy functions** over Python loops

**Next:** [Broadcasting](./03-broadcasting.md)

---

## Further Reading

- [Array Operations](https://numpy.org/doc/stable/user/quickstart.html#basic-operations)
- [Universal Functions](https://numpy.org/doc/stable/reference/ufuncs.html)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/user/basics.broadcasting.html
-->
