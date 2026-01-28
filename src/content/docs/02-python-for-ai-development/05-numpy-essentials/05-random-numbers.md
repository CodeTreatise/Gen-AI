---
title: "Random Number Generation"
---

# Random Number Generation

## Introduction

NumPy's random module provides fast, reproducible random number generation—essential for simulations, machine learning, and statistical sampling.

### What We'll Cover

- Random number basics
- Distributions
- Reproducibility with seeds
- Sampling and shuffling
- The new Generator API

### Prerequisites

- NumPy basics
- Basic probability concepts

---

## Basic Random Numbers

### Random Floats

```python
import numpy as np

# Random floats in [0, 1)
arr = np.random.random(5)
print(arr)  # [0.374 0.951 0.732 0.599 0.156]

# Random floats in range [low, high)
arr = np.random.uniform(10, 20, 5)
print(arr)  # Values between 10 and 20

# 2D array
matrix = np.random.random((3, 4))
print(matrix.shape)  # (3, 4)
```

### Random Integers

```python
import numpy as np

# Random integers in [low, high)
arr = np.random.randint(0, 100, 10)
print(arr)  # 10 integers from 0-99

# 2D array of integers
matrix = np.random.randint(1, 7, (3, 4))  # Dice rolls
print(matrix)

# Single random integer
value = np.random.randint(1, 7)  # Single die roll
```

---

## Common Distributions

### Normal (Gaussian) Distribution

```python
import numpy as np

# Standard normal (mean=0, std=1)
arr = np.random.randn(5)
print(arr)

# Custom mean and std
mean, std = 100, 15
scores = np.random.normal(mean, std, 1000)
print(f"Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")
```

### Uniform Distribution

```python
import numpy as np

# Uniform between low and high
arr = np.random.uniform(0, 10, 5)
print(arr)  # Values evenly distributed 0-10
```

### Other Distributions

```python
import numpy as np

# Binomial (coin flips, success/failure)
flips = np.random.binomial(n=10, p=0.5, size=100)
print(f"Average heads: {flips.mean():.1f}")

# Poisson (events per time period)
events = np.random.poisson(lam=5, size=100)
print(f"Average events: {events.mean():.1f}")

# Exponential (time between events)
times = np.random.exponential(scale=1.0, size=100)

# Beta (probabilities)
probs = np.random.beta(a=2, b=5, size=100)
```

---

## Reproducibility with Seeds

### Setting the Seed

```python
import numpy as np

# Set global seed
np.random.seed(42)
print(np.random.random(3))  # Always: [0.374 0.951 0.732]

np.random.seed(42)
print(np.random.random(3))  # Same: [0.374 0.951 0.732]

# Different seed = different sequence
np.random.seed(123)
print(np.random.random(3))  # Different values
```

### Why Seeds Matter

```python
import numpy as np

# Machine Learning: reproducible train/test splits
np.random.seed(42)
indices = np.random.permutation(100)
train_idx = indices[:80]
test_idx = indices[80:]

# Always get same split with same seed
```

---

## The Generator API (Recommended)

### Modern Random Generation

```python
import numpy as np

# Create a Generator with seed
rng = np.random.default_rng(42)

# Use the generator
print(rng.random(5))
print(rng.integers(0, 100, 10))
print(rng.normal(0, 1, 5))
```

### Benefits of Generator

```python
import numpy as np

# Thread-safe, better algorithms
rng = np.random.default_rng(42)

# More intuitive API
print(rng.integers(1, 7, 10))  # Includes high by default (unlike randint)

# Explicitly spawn for parallel work
child_rng = rng.spawn(1)[0]
```

### Common Generator Methods

| Method | Description |
|--------|-------------|
| `random(size)` | Uniform [0, 1) |
| `integers(low, high, size)` | Random integers |
| `normal(mean, std, size)` | Gaussian |
| `uniform(low, high, size)` | Uniform range |
| `choice(arr, size)` | Sample from array |
| `shuffle(arr)` | Shuffle in-place |
| `permutation(arr)` | Return shuffled copy |

---

## Sampling

### Random Choice

```python
import numpy as np

rng = np.random.default_rng(42)

# Sample from array
colors = ['red', 'green', 'blue']
sample = rng.choice(colors, 5)
print(sample)  # ['blue' 'blue' 'red' 'green' 'blue']

# Without replacement
sample = rng.choice(colors, 3, replace=False)
print(sample)  # Each color once

# Weighted sampling
weights = [0.1, 0.3, 0.6]
sample = rng.choice(colors, 100, p=weights)
print(np.unique(sample, return_counts=True))
# Blue appears ~60 times
```

### Sampling Indices

```python
import numpy as np

rng = np.random.default_rng(42)

# Sample indices
data = np.arange(100)
sample_idx = rng.choice(len(data), 10, replace=False)
sample = data[sample_idx]

# Bootstrap sampling (with replacement)
bootstrap_idx = rng.choice(len(data), len(data), replace=True)
bootstrap_sample = data[bootstrap_idx]
```

---

## Shuffling

### Shuffle In-Place

```python
import numpy as np

rng = np.random.default_rng(42)

arr = np.arange(10)
rng.shuffle(arr)  # Modifies arr
print(arr)  # [8 1 5 0 7 2 9 4 3 6]
```

### Return Shuffled Copy

```python
import numpy as np

rng = np.random.default_rng(42)

arr = np.arange(10)
shuffled = rng.permutation(arr)  # Returns copy
print(arr)      # Original unchanged
print(shuffled) # Shuffled copy
```

### Shuffle Multiple Arrays Together

```python
import numpy as np

rng = np.random.default_rng(42)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

# Shuffle indices, apply to both
indices = rng.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# Correspondence maintained
print(X_shuffled)
print(y_shuffled)
```

---

## Practical Examples

### Train/Test Split

```python
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    rng = np.random.default_rng(random_state)
    
    n = len(X)
    indices = rng.permutation(n)
    
    test_count = int(n * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Usage
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

### Monte Carlo Simulation

```python
import numpy as np

def estimate_pi(n_samples=1000000, seed=42):
    """Estimate π using Monte Carlo."""
    rng = np.random.default_rng(seed)
    
    # Random points in unit square
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    
    # Count points inside unit circle
    inside = (x**2 + y**2) <= 1
    
    # π/4 = area of quarter circle / area of square
    return 4 * inside.sum() / n_samples

print(f"Estimated π: {estimate_pi()}")  # ~3.14159
```

### Random Initialization (Neural Networks)

```python
import numpy as np

def xavier_init(shape, seed=42):
    """Xavier/Glorot initialization for neural networks."""
    rng = np.random.default_rng(seed)
    
    fan_in = shape[0]
    fan_out = shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    
    return rng.normal(0, std, shape)

# Initialize weight matrix
weights = xavier_init((784, 256))
print(f"Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
```

---

## Hands-on Exercise

### Your Task

```python
# Create a simulation:
# 1. Generate 1000 samples from normal distribution (mean=50, std=10)
# 2. Bootstrap 100 samples and calculate mean
# 3. Repeat bootstrap 1000 times to get distribution of means
# 4. Calculate 95% confidence interval

# Use seed=42 for reproducibility
```

<details>
<summary>✅ Solution</summary>

```python
import numpy as np

# Set up generator
rng = np.random.default_rng(42)

# 1. Generate original data
original_data = rng.normal(50, 10, 1000)
print(f"Original mean: {original_data.mean():.2f}")

# 2 & 3. Bootstrap sampling
n_bootstrap = 1000
bootstrap_means = []

for _ in range(n_bootstrap):
    # Sample with replacement
    sample = rng.choice(original_data, 100, replace=True)
    bootstrap_means.append(sample.mean())

bootstrap_means = np.array(bootstrap_means)

# 4. 95% confidence interval
lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

print(f"Bootstrap mean: {bootstrap_means.mean():.2f}")
print(f"95% CI: [{lower:.2f}, {upper:.2f}]")

# Visualize
print(f"\nDistribution of bootstrap means:")
print(f"Min: {bootstrap_means.min():.2f}")
print(f"Max: {bootstrap_means.max():.2f}")
print(f"Std: {bootstrap_means.std():.2f}")
```
</details>

---

## Summary

✅ Use **`np.random.default_rng(seed)`** for new code
✅ Set **seeds** for reproducibility
✅ **`rng.choice()`** for sampling, **`rng.permutation()`** for shuffling
✅ **Normal distribution** for continuous data, **binomial** for discrete
✅ Use **`replace=False`** for sampling without replacement
✅ **Generator API** is thread-safe and recommended

**Next:** [Performance](./06-performance.md)

---

## Further Reading

- [Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)

<!-- 
Sources Consulted:
- NumPy Docs: https://numpy.org/doc/stable/reference/random/index.html
-->
