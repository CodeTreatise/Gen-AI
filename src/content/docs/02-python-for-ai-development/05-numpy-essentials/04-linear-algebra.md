---
title: "Linear Algebra"
---

# Linear Algebra

## Introduction

NumPy's linear algebra capabilities are essential for machine learning, data science, and scientific computing. Matrix operations power everything from neural networks to data transformations.

### What We'll Cover

- Matrix multiplication
- Transpose and reshaping
- Matrix inverse and determinant
- Eigenvalues and eigenvectors
- Solving linear equations

### Prerequisites

- NumPy arrays
- Basic linear algebra concepts

---

## Matrix Multiplication

### The @ Operator (Recommended)

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication
C = A @ B
print(C)
# [[19 22]
#  [43 50]]

# (1*5 + 2*7 = 19, 1*6 + 2*8 = 22, ...)
```

### np.dot() and np.matmul()

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# All equivalent for 2D arrays
print(A @ B)
print(np.dot(A, B))
print(np.matmul(A, B))

# np.dot also handles 1D (dot product)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(np.dot(v1, v2))  # 32 (1*4 + 2*5 + 3*6)
```

### Matrix-Vector Multiplication

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])  # Shape: (2, 3)

vector = np.array([1, 2, 3])   # Shape: (3,)

result = matrix @ vector       # Shape: (2,)
print(result)  # [14 32]
# [1*1 + 2*2 + 3*3 = 14, 4*1 + 5*2 + 6*3 = 32]
```

### Element-wise vs Matrix Multiplication

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication (Hadamard product)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Matrix multiplication
print(A @ B)
# [[19 22]
#  [43 50]]
```

---

## Transpose

### Basic Transpose

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

# Using .T attribute
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]

print(A.T.shape)  # (3, 2)

# Using np.transpose
print(np.transpose(A))
```

### Multi-dimensional Transpose

```python
import numpy as np

# 3D array
arr = np.arange(24).reshape(2, 3, 4)  # Shape: (2, 3, 4)

# Swap axes
print(arr.transpose(0, 2, 1).shape)  # (2, 4, 3)
print(arr.transpose(2, 1, 0).shape)  # (4, 3, 2)

# Or use swapaxes
print(np.swapaxes(arr, 1, 2).shape)  # (2, 4, 3)
```

---

## Matrix Properties

### Determinant

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

det = np.linalg.det(A)
print(det)  # -2.0

# Singular matrix has det ≈ 0
singular = np.array([[1, 2],
                     [2, 4]])
print(np.linalg.det(singular))  # ≈ 0
```

### Matrix Rank

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

rank = np.linalg.matrix_rank(A)
print(rank)  # 2 (rows are linearly dependent)

B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
print(np.linalg.matrix_rank(B))  # 3 (full rank)
```

### Trace (Sum of Diagonal)

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(np.trace(A))  # 15 (1 + 5 + 9)
```

---

## Matrix Inverse

### Computing Inverse

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# Inverse
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify: A @ A_inv ≈ I
print(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]
```

### Pseudo-Inverse (for non-square matrices)

```python
import numpy as np

# Non-square matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

# Moore-Penrose pseudo-inverse
A_pinv = np.linalg.pinv(A)
print(A_pinv.shape)  # (3, 2)
```

### Handling Singular Matrices

```python
import numpy as np

# Singular matrix (no inverse exists)
singular = np.array([[1, 2],
                     [2, 4]])

try:
    inv = np.linalg.inv(singular)
except np.linalg.LinAlgError:
    print("Matrix is singular!")
    # Use pseudo-inverse instead
    pinv = np.linalg.pinv(singular)
```

---

## Solving Linear Equations

### System: Ax = b

```python
import numpy as np

# Solve: Ax = b
# 2x + 3y = 8
# 4x + 5y = 14

A = np.array([[2, 3],
              [4, 5]])

b = np.array([8, 14])

# Solve for x
x = np.linalg.solve(A, b)
print(x)  # [1. 2.]

# Verify: A @ x ≈ b
print(A @ x)  # [8. 14.]
```

### Multiple Right-Hand Sides

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# Solve for multiple b vectors
B = np.array([[5, 17],
              [11, 39]])

X = np.linalg.solve(A, B)
print(X)
# [[ 1.  1.]
#  [ 2.  8.]]
```

---

## Eigenvalues and Eigenvectors

### Computing Eigenvalues

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# [5. 2.]

print("Eigenvectors:")
print(eigenvectors)
# [[ 0.894 -0.707]
#  [ 0.447  0.707]]

# Verify: A @ v = λ * v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"A @ v = {A @ v}, λ * v = {lam * v}")
```

### Symmetric Matrices

```python
import numpy as np

# For symmetric matrices, use eigh (faster, numerically stable)
A = np.array([[4, 2],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eigh(A)
print("Eigenvalues:", eigenvalues)  # Sorted ascending
```

---

## Singular Value Decomposition (SVD)

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# SVD: A = U @ S @ V^T
U, s, Vt = np.linalg.svd(A)

print("U shape:", U.shape)   # (3, 3)
print("s shape:", s.shape)   # (2,) - singular values
print("Vt shape:", Vt.shape) # (2, 2)

# Reconstruct A
S = np.zeros_like(A, dtype=float)
S[:2, :2] = np.diag(s)
reconstructed = U @ S @ Vt
print(np.allclose(A, reconstructed))  # True
```

### Low-Rank Approximation

```python
import numpy as np

A = np.random.randn(100, 50)

U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Keep top k singular values
k = 10
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Compression ratio
original_size = A.size
compressed_size = U[:, :k].size + k + Vt[:k, :].size
print(f"Compression: {compressed_size / original_size:.2%}")
```

---

## Norms

```python
import numpy as np

v = np.array([3, 4])

# Vector norms
print(np.linalg.norm(v))       # 5.0 (L2 norm, Euclidean)
print(np.linalg.norm(v, 1))    # 7.0 (L1 norm, Manhattan)
print(np.linalg.norm(v, np.inf))  # 4.0 (Infinity norm, max)

# Matrix norms
A = np.array([[1, 2], [3, 4]])
print(np.linalg.norm(A, 'fro'))  # Frobenius norm
print(np.linalg.norm(A, 2))      # Spectral norm (largest singular value)
```

---

## Practical Example: Linear Regression

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 samples, 3 features
true_weights = np.array([1.5, -2.0, 0.5])
y = X @ true_weights + np.random.randn(100) * 0.1

# Add bias term
X_with_bias = np.c_[np.ones(100), X]

# Solve normal equation: w = (X^T X)^(-1) X^T y
XtX = X_with_bias.T @ X_with_bias
Xty = X_with_bias.T @ y
weights = np.linalg.solve(XtX, Xty)

print("Fitted weights:", weights)
# [~0, 1.5, -2.0, 0.5]  (bias + true weights)
```

---

## Hands-on Exercise

### Your Task

```python
# 1. Create a 3x3 matrix A
# 2. Compute its inverse and verify A @ A_inv ≈ I
# 3. Solve Ax = b where b = [6, 15, 24]
# 4. Compute eigenvalues
# 5. Perform SVD and verify reconstruction

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

b = np.array([6, 15, 24])
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

b = np.array([6, 15, 24])

# 2. Inverse
A_inv = np.linalg.inv(A)
print("Inverse:")
print(A_inv)
print("\nA @ A_inv ≈ I:")
print(np.round(A @ A_inv, 10))

# 3. Solve Ax = b
x = np.linalg.solve(A, b)
print(f"\nSolution x: {x}")
print(f"Verify A @ x = b: {A @ x}")

# 4. Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")

# 5. SVD
U, s, Vt = np.linalg.svd(A)
S = np.diag(s)
reconstructed = U @ S @ Vt
print(f"\nSVD reconstruction matches: {np.allclose(A, reconstructed)}")
print(f"Singular values: {s}")
```
</details>

---

## Summary

✅ Use **`@`** for matrix multiplication
✅ **`.T`** for transpose, **`np.linalg.inv()`** for inverse
✅ **`np.linalg.solve()`** is faster than computing inverse
✅ **`np.linalg.eig()`** for eigenvalues, **`eigh()`** for symmetric
✅ **SVD** is fundamental for dimensionality reduction
✅ Use **`np.linalg.norm()`** for vector/matrix norms

**Next:** [Random Numbers](./05-random-numbers.md)

---

## Further Reading

- [Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/reference/routines.linalg.html
-->
