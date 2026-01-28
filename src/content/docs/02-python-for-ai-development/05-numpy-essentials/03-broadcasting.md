---
title: "Broadcasting"
---

# Broadcasting

## Introduction

Broadcasting is NumPy's powerful mechanism for performing operations on arrays of different shapes. It avoids copying data and enables concise, efficient code.

### What We'll Cover

- Broadcasting rules
- Shape compatibility
- Automatic dimension expansion
- Common patterns
- Avoiding errors

### Prerequisites

- NumPy arrays
- Array operations

---

## What is Broadcasting?

```python
import numpy as np

# Without broadcasting: need matching shapes
a = np.array([1, 2, 3])
b = np.array([10, 10, 10])
print(a + b)  # [11 12 13]

# With broadcasting: scalar expands to match
a = np.array([1, 2, 3])
print(a + 10)  # [11 12 13]

# NumPy "broadcasts" 10 to [10, 10, 10] internally
# No actual memory copy happens!
```

---

## Broadcasting Rules

### The Three Rules

1. **Align shapes from the right**
2. **Dimensions match if equal OR one is 1**
3. **Missing dimensions treated as 1**

```
Shape (3,)    →  (1, 3)   # Add dimension
Shape (3, 1) and (3,) →  (3, 1) and (1, 3)  →  (3, 3)
```

### Compatibility Examples

```python
import numpy as np

# Compatible shapes
(5,)      + (5,)      → (5,)      # Same shape
(5,)      + (1,)      → (5,)      # 1 broadcasts
(3, 4)    + (4,)      → (3, 4)    # Row broadcasts
(3, 4)    + (3, 1)    → (3, 4)    # Column broadcasts
(3, 1)    + (1, 4)    → (3, 4)    # Both broadcast

# Incompatible shapes
(3,)      + (4,)      → Error!   # No dimension is 1
(2, 3)    + (2,)      → Error!   # 3 ≠ 2
```

---

## Basic Broadcasting Examples

### Scalar with Array

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Scalar broadcasts to all elements
print(arr + 10)
# [[11 12 13]
#  [14 15 16]]

print(arr * 2)
# [[ 2  4  6]
#  [ 8 10 12]]
```

### 1D with 2D

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])  # Shape: (2, 3)

row = np.array([10, 20, 30])   # Shape: (3,)

# Row broadcasts across all rows
print(matrix + row)
# [[11 22 33]
#  [14 25 36]]

# Effectively:
# [[1, 2, 3],   +  [[10, 20, 30],   =  [[11, 22, 33],
#  [4, 5, 6]]      [10, 20, 30]]       [14, 25, 36]]
```

### Column Broadcasting

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])    # Shape: (2, 3)

column = np.array([[10],
                   [20]])         # Shape: (2, 1)

# Column broadcasts across all columns
print(matrix + column)
# [[11 12 13]
#  [24 25 26]]

# Or use newaxis
col = np.array([10, 20])[:, np.newaxis]  # (2,) → (2, 1)
print(matrix + col)
```

---

## Practical Examples

### Centering Data (Subtract Mean)

```python
import numpy as np

# Data: 3 samples, 4 features
data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

# Mean of each feature (column)
feature_means = data.mean(axis=0)  # Shape: (4,)
print(feature_means)  # [5. 6. 7. 8.]

# Subtract mean from each sample
centered = data - feature_means
print(centered)
# [[-4. -4. -4. -4.]
#  [ 0.  0.  0.  0.]
#  [ 4.  4.  4.  4.]]
```

### Normalizing Data

```python
import numpy as np

data = np.array([[1, 100, 1000],
                 [2, 200, 2000],
                 [3, 300, 3000]])

# Min-max normalization per column
col_min = data.min(axis=0)
col_max = data.max(axis=0)

normalized = (data - col_min) / (col_max - col_min)
print(normalized)
# [[0.  0.  0. ]
#  [0.5 0.5 0.5]
#  [1.  1.  1. ]]
```

### Distance Calculation

```python
import numpy as np

# Calculate distance from each point to origin
points = np.array([[3, 4],
                   [1, 1],
                   [5, 12]])  # Shape: (3, 2)

# Euclidean distance: sqrt(x² + y²)
distances = np.sqrt(np.sum(points ** 2, axis=1))
print(distances)  # [5. 1.414 13.]
```

### Outer Product

```python
import numpy as np

a = np.array([1, 2, 3])  # Shape: (3,)
b = np.array([10, 20])   # Shape: (2,)

# Create outer product using broadcasting
outer = a[:, np.newaxis] * b[np.newaxis, :]
# Shape: (3, 1) * (1, 2) → (3, 2)

print(outer)
# [[10 20]
#  [20 40]
#  [30 60]]

# Or use np.outer
print(np.outer(a, b))
```

---

## Visualization of Broadcasting

### Adding Column to Each Row

```
Matrix (3, 4):          Row vector (4,):
[[1, 2, 3, 4],          [10, 20, 30, 40]
 [5, 6, 7, 8],              ↓ stretch
 [9, 10, 11, 12]]       [[10, 20, 30, 40],
                         [10, 20, 30, 40],
                         [10, 20, 30, 40]]

Result (3, 4):
[[11, 22, 33, 44],
 [15, 26, 37, 48],
 [19, 30, 41, 52]]
```

### Adding Row to Each Column

```
Matrix (3, 4):          Column vector (3, 1):
[[1, 2, 3, 4],          [[100],
 [5, 6, 7, 8],           [200],
 [9, 10, 11, 12]]        [300]]
                             ↓ stretch
                        [[100, 100, 100, 100],
                         [200, 200, 200, 200],
                         [300, 300, 300, 300]]

Result (3, 4):
[[101, 102, 103, 104],
 [205, 206, 207, 208],
 [309, 310, 311, 312]]
```

---

## Common Patterns

### Row-wise Operations

```python
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Divide each row by its sum
row_sums = data.sum(axis=1, keepdims=True)  # Shape: (2, 1)
normalized = data / row_sums
print(normalized)
# [[0.167 0.333 0.5  ]
#  [0.267 0.333 0.4  ]]
```

### Column-wise Operations

```python
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Subtract column mean
col_means = data.mean(axis=0)  # Shape: (3,)
centered = data - col_means
print(centered)
# [[-1.5 -1.5 -1.5]
#  [ 1.5  1.5  1.5]]
```

### Creating Grids

```python
import numpy as np

x = np.arange(5)  # [0, 1, 2, 3, 4]
y = np.arange(3)  # [0, 1, 2]

# Create grid using broadcasting
xx = x[np.newaxis, :]  # Shape: (1, 5)
yy = y[:, np.newaxis]  # Shape: (3, 1)

# Now can combine
grid = xx + yy * 10
print(grid)
# [[ 0  1  2  3  4]
#  [10 11 12 13 14]
#  [20 21 22 23 24]]

# Or use meshgrid
X, Y = np.meshgrid(x, y)
```

---

## Avoiding Broadcasting Errors

### Common Mistakes

```python
import numpy as np

# Error: shapes (3,) and (4,) not compatible
a = np.array([1, 2, 3])
b = np.array([1, 2, 3, 4])
# a + b  # ValueError!

# Solution: reshape or pad
```

### Checking Compatibility

```python
import numpy as np

def are_broadcastable(shape1, shape2):
    """Check if two shapes are broadcast-compatible."""
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != b and a != 1 and b != 1:
            return False
    return True

print(are_broadcastable((3, 4), (4,)))    # True
print(are_broadcastable((3, 4), (3, 1)))  # True
print(are_broadcastable((3, 4), (2, 4)))  # False
```

### Explicit Broadcasting

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[10], [20]])

# Get broadcast shapes
result_shape = np.broadcast_shapes(a.shape, b.shape)
print(result_shape)  # (2, 3)

# Explicit broadcast (creates views)
a_bc, b_bc = np.broadcast_arrays(a, b)
print(a_bc.shape, b_bc.shape)  # (2, 3) (2, 3)
```

---

## Hands-on Exercise

### Your Task

```python
# Given:
# - 4 cities with (lat, lon) coordinates
# - Calculate pairwise distances between all cities

cities = np.array([
    [40.7, -74.0],   # New York
    [34.0, -118.2],  # Los Angeles
    [41.9, -87.6],   # Chicago
    [29.8, -95.4]    # Houston
])

# Create a 4x4 distance matrix using broadcasting
# Hint: Use Euclidean distance (simplified, not great-circle)
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np

cities = np.array([
    [40.7, -74.0],   # New York
    [34.0, -118.2],  # Los Angeles
    [41.9, -87.6],   # Chicago
    [29.8, -95.4]    # Houston
])

# Reshape for broadcasting
# cities shape: (4, 2)
# We want: diff[i, j] = cities[i] - cities[j]

# Method 1: Using broadcasting
city_a = cities[:, np.newaxis, :]  # Shape: (4, 1, 2)
city_b = cities[np.newaxis, :, :]  # Shape: (1, 4, 2)

diff = city_a - city_b  # Shape: (4, 4, 2)
distances = np.sqrt(np.sum(diff ** 2, axis=2))  # Shape: (4, 4)

print("Distance matrix:")
print(np.round(distances, 1))
# [[  0.   44.6  13.8  22.4]
#  [ 44.6   0.   31.3  23.1]
#  [ 13.8  31.3   0.   14.3]
#  [ 22.4  23.1  14.3   0. ]]

# Method 2: Using scipy (more accurate)
# from scipy.spatial.distance import cdist
# distances = cdist(cities, cities)
```
</details>

---

## Summary

✅ **Broadcasting** enables operations on different-shaped arrays
✅ **Align from right**, dimensions must be equal or 1
✅ Use **`np.newaxis`** or **`[:, None]`** to add dimensions
✅ Use **`keepdims=True`** to preserve dimensions after aggregation
✅ Check compatibility with **`np.broadcast_shapes()`**
✅ No data copying—broadcasting uses views

**Next:** [Linear Algebra](./04-linear-algebra.md)

---

## Further Reading

- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Broadcasting Tutorial](https://numpy.org/doc/stable/user/theory.broadcasting.html)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/user/basics.broadcasting.html
-->
