---
title: "NumPy Essentials"
---

# NumPy Essentials

## Overview

NumPy is the foundation of scientific computing in Python. It provides fast, memory-efficient multi-dimensional arrays and a vast collection of mathematical operations—essential for AI/ML development.

This lesson covers NumPy fundamentals from array creation to linear algebra operations.

---

## What We'll Learn

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| [01](./01-numpy-arrays.md) | NumPy Arrays | Creating, indexing, slicing, reshaping |
| [02](./02-array-operations.md) | Array Operations | Element-wise, aggregations, ufuncs |
| [03](./03-broadcasting.md) | Broadcasting | Shape compatibility, dimension expansion |
| [04](./04-linear-algebra.md) | Linear Algebra | Matrix multiplication, transpose, inverse |
| [05](./05-random-numbers.md) | Random Numbers | Distributions, seeds, sampling |
| [06](./06-performance.md) | Performance | Vectorization, memory, optimization |

---

## Why NumPy?

| Feature | Python Lists | NumPy Arrays |
|---------|--------------|--------------|
| Speed | Slow | 10-100x faster |
| Memory | Inefficient | Contiguous, efficient |
| Operations | Manual loops | Vectorized |
| Broadcasting | ❌ | ✅ |
| Math functions | Limited | Comprehensive |

---

## Quick Start

```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Operations are vectorized
squared = arr ** 2  # [1, 4, 9, 16, 25]

# Multi-dimensional
matrix = np.array([[1, 2], [3, 4]])

# Matrix multiplication
result = matrix @ matrix.T

# Statistics
print(arr.mean(), arr.std())  # 3.0, 1.41...
```

---

## Installation

```bash
pip install numpy
```

```python
import numpy as np

print(np.__version__)  # 1.26.x
```

---

## Prerequisites

Before starting this lesson:
- Python fundamentals
- Basic math (algebra, matrices helpful)

---

## Start Learning

Begin with [NumPy Arrays](./01-numpy-arrays.md) to understand the core data structure.

---

## Further Reading

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
