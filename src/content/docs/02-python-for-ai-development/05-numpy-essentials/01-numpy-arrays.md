---
title: "NumPy Arrays (ndarray)"
---

# NumPy Arrays (ndarray)

## Introduction

The ndarray (N-dimensional array) is NumPy's core data structure. It's a homogeneous, fixed-size container for numerical data that enables fast vectorized operations.

### What We'll Cover

- Creating arrays
- Shape and dimensions
- Data types (dtype)
- Indexing and slicing
- Boolean and fancy indexing

### Prerequisites

- Python lists
- Basic Python

---

## Creating Arrays

### From Python Lists

```python
import numpy as np

# 1D array
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)
# [[1 2 3]
#  [4 5 6]]

# 3D array
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor.shape)  # (2, 2, 2)
```

### Initialization Functions

```python
import numpy as np

# Zeros
zeros = np.zeros((3, 4))  # 3x4 matrix of zeros

# Ones
ones = np.ones((2, 3))  # 2x3 matrix of ones

# Full (any value)
full = np.full((2, 2), 7)  # 2x2 matrix of 7s

# Empty (uninitialized, fast)
empty = np.empty((3, 3))

# Identity matrix
eye = np.eye(4)  # 4x4 identity matrix

# Diagonal
diag = np.diag([1, 2, 3, 4])  # 4x4 diagonal matrix
```

### Sequences

```python
import numpy as np

# Range (like Python range)
arr = np.arange(0, 10, 2)  # [0 2 4 6 8]

# Linspace (n evenly spaced values)
arr = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]

# Logspace (logarithmic spacing)
arr = np.logspace(0, 3, 4)  # [1, 10, 100, 1000]
```

---

## Shape and Dimensions

### Array Attributes

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)    # (2, 3) - rows, columns
print(arr.ndim)     # 2 - number of dimensions
print(arr.size)     # 6 - total elements
print(arr.dtype)    # int64 - data type
print(arr.itemsize) # 8 - bytes per element
```

### Reshaping

```python
import numpy as np

arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# Reshape to 3x4
matrix = arr.reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Use -1 for automatic dimension
matrix = arr.reshape(4, -1)  # 4x3

# Flatten to 1D
flat = matrix.flatten()  # Copy
flat = matrix.ravel()    # View (no copy)
```

### Adding/Removing Dimensions

```python
import numpy as np

arr = np.array([1, 2, 3])  # Shape: (3,)

# Add dimension
row = arr[np.newaxis, :]  # Shape: (1, 3)
col = arr[:, np.newaxis]  # Shape: (3, 1)

# Using expand_dims
row = np.expand_dims(arr, axis=0)  # (1, 3)
col = np.expand_dims(arr, axis=1)  # (3, 1)

# Remove single dimensions
arr = np.array([[[1, 2, 3]]])  # Shape: (1, 1, 3)
squeezed = np.squeeze(arr)      # Shape: (3,)
```

---

## Data Types (dtype)

### Common dtypes

| dtype | Description |
|-------|-------------|
| `int32`, `int64` | Integers |
| `float32`, `float64` | Floats |
| `bool` | Boolean |
| `complex64` | Complex numbers |
| `str_` | Fixed-length strings |

```python
import numpy as np

# Specify dtype at creation
arr = np.array([1, 2, 3], dtype=np.float32)
print(arr.dtype)  # float32

# Common types
ints = np.array([1, 2, 3], dtype=np.int32)
floats = np.array([1, 2, 3], dtype=np.float64)
bools = np.array([1, 0, 1], dtype=np.bool_)
```

### Converting dtypes

```python
import numpy as np

arr = np.array([1.7, 2.3, 3.9])

# Convert to int (truncates)
int_arr = arr.astype(np.int32)
print(int_arr)  # [1 2 3]

# Convert to string
str_arr = arr.astype(str)
print(str_arr)  # ['1.7' '2.3' '3.9']
```

---

## Indexing and Slicing

### Basic Indexing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Single element
print(arr[0])   # 10
print(arr[-1])  # 50

# Slicing [start:stop:step]
print(arr[1:4])    # [20 30 40]
print(arr[::2])    # [10 30 50]
print(arr[::-1])   # [50 40 30 20 10] (reversed)
```

### Multi-dimensional Indexing

```python
import numpy as np

matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# Single element
print(matrix[0, 0])  # 1
print(matrix[1, 2])  # 6

# Row/column
print(matrix[0])      # [1 2 3] (first row)
print(matrix[:, 0])   # [1 4 7] (first column)
print(matrix[1:, :2]) # [[4 5] [7 8]] (slice)
```

### Modifying with Slices

```python
import numpy as np

arr = np.zeros((3, 3))

# Set row
arr[0] = [1, 2, 3]

# Set column
arr[:, 1] = [4, 5, 6]

# Set block
arr[1:, 1:] = [[7, 8], [9, 10]]

print(arr)
```

---

## Boolean Indexing

### Filtering Arrays

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# Boolean mask
mask = arr > 3
print(mask)  # [False False False  True  True  True]

# Apply mask
filtered = arr[mask]
print(filtered)  # [4 5 6]

# One line
filtered = arr[arr > 3]
```

### Multiple Conditions

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# AND: &
result = arr[(arr > 2) & (arr < 5)]
print(result)  # [3 4]

# OR: |
result = arr[(arr < 2) | (arr > 5)]
print(result)  # [1 6]

# NOT: ~
result = arr[~(arr > 3)]
print(result)  # [1 2 3]
```

### Replacing Values

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Replace based on condition
arr[arr > 3] = 0
print(arr)  # [1 2 3 0 0]

# Using np.where
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, 0, arr)  # if > 3: 0, else: original
print(result)  # [1 2 3 0 0]

# np.where with two choices
result = np.where(arr > 3, "big", "small")
print(result)  # ['small' 'small' 'small' 'big' 'big']
```

---

## Fancy Indexing

### Index Arrays

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Select by indices
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# Reorder
print(arr[[4, 3, 2, 1, 0]])  # [50 40 30 20 10]

# Duplicate indices
print(arr[[0, 0, 1, 1]])  # [10 10 20 20]
```

### 2D Fancy Indexing

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select specific elements
rows = [0, 1, 2]
cols = [0, 1, 2]
print(matrix[rows, cols])  # [1 5 9] (diagonal)

# Select entire rows
print(matrix[[0, 2]])  # [[1 2 3] [7 8 9]]
```

---

## Combining Arrays

### Concatenation

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate
c = np.concatenate([a, b])
print(c)  # [1 2 3 4 5 6]

# 2D concatenation
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# Vertical stack
v = np.vstack([m1, m2])  # or np.concatenate([m1, m2], axis=0)

# Horizontal stack
h = np.hstack([m1, m2])  # or np.concatenate([m1, m2], axis=1)
```

### Splitting

```python
import numpy as np

arr = np.arange(9)

# Split into 3 equal parts
parts = np.split(arr, 3)
print(parts)  # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]

# Split at indices
parts = np.split(arr, [2, 5])
# [array([0, 1]), array([2, 3, 4]), array([5, 6, 7, 8])]
```

---

## Hands-on Exercise

### Your Task

```python
# 1. Create a 5x5 matrix with values 1-25
# 2. Extract the diagonal
# 3. Extract all values > 15
# 4. Replace corner values with 0
# 5. Calculate row sums

# Expected output:
# Matrix, diagonal [1, 7, 13, 19, 25], filtered [16-25], corners zeroed
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np

# 1. Create 5x5 matrix
matrix = np.arange(1, 26).reshape(5, 5)
print("Matrix:")
print(matrix)

# 2. Extract diagonal
diagonal = np.diag(matrix)
print(f"\nDiagonal: {diagonal}")  # [1, 7, 13, 19, 25]

# 3. Values > 15
filtered = matrix[matrix > 15]
print(f"\nValues > 15: {filtered}")

# 4. Replace corners with 0
matrix[0, 0] = 0
matrix[0, -1] = 0
matrix[-1, 0] = 0
matrix[-1, -1] = 0
print("\nCorners zeroed:")
print(matrix)

# 5. Row sums
row_sums = matrix.sum(axis=1)
print(f"\nRow sums: {row_sums}")
```
</details>

---

## Summary

✅ **ndarray** is NumPy's core multi-dimensional array
✅ Use **`np.zeros`, `np.ones`, `np.arange`** for initialization
✅ **`.shape`, `.ndim`, `.dtype`** for array info
✅ **Reshape** with `.reshape()` (use -1 for auto-dimension)
✅ **Boolean indexing** for filtering: `arr[arr > 5]`
✅ **Fancy indexing** with index arrays: `arr[[0, 2, 4]]`

**Next:** [Array Operations](./02-array-operations.md)

---

## Further Reading

- [Array Creation](https://numpy.org/doc/stable/user/basics.creation.html)
- [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/user/quickstart.html
-->
