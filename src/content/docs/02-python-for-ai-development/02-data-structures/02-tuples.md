---
title: "Tuples"
---

# Tuples

## Introduction

Tuples are immutable sequences—once created, they cannot be modified. This makes them useful for representing fixed data, returning multiple values from functions, and as dictionary keys.

### What We'll Cover

- Creating and using tuples
- Tuple packing and unpacking
- Named tuples
- When to use tuples vs lists

### Prerequisites

- Lists
- Python fundamentals

---

## Creating Tuples

### Basic Tuple Creation

```python
# With parentheses
point = (10, 20)
rgb = (255, 128, 0)

# Without parentheses (tuple packing)
coordinates = 10, 20, 30

# Single element tuple (comma required!)
single = (42,)      # ✅ Tuple with one element
not_tuple = (42)    # ❌ Just an integer!

# Empty tuple
empty = ()
empty = tuple()

# From iterable
chars = tuple("hello")  # ('h', 'e', 'l', 'l', 'o')
nums = tuple([1, 2, 3]) # (1, 2, 3)
```

### Tuple Immutability

```python
t = (1, 2, 3)

# ❌ Cannot modify
# t[0] = 100  # TypeError: 'tuple' object does not support item assignment
# t.append(4) # AttributeError: 'tuple' object has no attribute 'append'

# ✅ Create new tuple instead
t = (100,) + t[1:]  # (100, 2, 3)

# ⚠️ Nested mutable objects CAN be modified
t = ([1, 2], [3, 4])
t[0].append(5)  # Works! t = ([1, 2, 5], [3, 4])
# But you can't replace the list itself
# t[0] = [10, 20]  # TypeError
```

---

## Indexing and Slicing

### Basic Operations

```python
colors = ("red", "green", "blue", "yellow")

# Indexing
print(colors[0])    # "red"
print(colors[-1])   # "yellow"

# Slicing (returns new tuple)
print(colors[1:3])  # ("green", "blue")
print(colors[:2])   # ("red", "green")
print(colors[::-1]) # ("yellow", "blue", "green", "red")

# Length
print(len(colors))  # 4

# Membership
print("red" in colors)  # True
```

### Tuple Methods

```python
nums = (1, 2, 3, 2, 4, 2, 5)

# Only two methods (tuples are simple!)
print(nums.count(2))   # 3 - count occurrences
print(nums.index(3))   # 2 - find first index
```

---

## Tuple Packing and Unpacking

### Packing

```python
# Multiple values become a tuple
point = 10, 20, 30
print(point)       # (10, 20, 30)
print(type(point)) # <class 'tuple'>
```

### Basic Unpacking

```python
# Unpack into variables
point = (10, 20, 30)
x, y, z = point
print(x, y, z)  # 10 20 30

# Swap values (uses tuple packing/unpacking)
a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1
```

### Extended Unpacking

```python
# * captures remaining elements as list
first, *rest = (1, 2, 3, 4, 5)
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

*start, last = (1, 2, 3, 4, 5)
print(start)  # [1, 2, 3, 4]
print(last)   # 5

first, *middle, last = (1, 2, 3, 4, 5)
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```

### Ignoring Values

```python
# Use _ for values you don't need
point = (10, 20, 30)
x, _, z = point  # Ignore y

# Multiple ignored values
first, *_, last = (1, 2, 3, 4, 5)
print(first, last)  # 1 5
```

### Function Return Values

```python
def get_user():
    return "Alice", 30, "alice@example.com"

# Unpack return value
name, age, email = get_user()

# Or keep as tuple
user = get_user()
print(user[0])  # "Alice"
```

---

## Named Tuples

### Basic Named Tuple

```python
from collections import namedtuple

# Define a named tuple type
Point = namedtuple("Point", ["x", "y"])

# Create instances
p1 = Point(10, 20)
p2 = Point(x=30, y=40)

# Access by name or index
print(p1.x, p1.y)     # 10 20
print(p1[0], p1[1])   # 10 20

# Still immutable
# p1.x = 100  # AttributeError
```

### Named Tuple with Defaults

```python
from collections import namedtuple

# With defaults (Python 3.7+)
User = namedtuple("User", ["name", "age", "email"], defaults=["unknown@example.com"])

u1 = User("Alice", 30)
print(u1)  # User(name='Alice', age=30, email='unknown@example.com')

u2 = User("Bob", 25, "bob@example.com")
print(u2)  # User(name='Bob', age=25, email='bob@example.com')
```

### Named Tuple Methods

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(10, 20)

# Convert to dictionary
print(p._asdict())  # {'x': 10, 'y': 20}

# Create modified copy
p2 = p._replace(x=100)
print(p2)  # Point(x=100, y=20)

# Field names
print(p._fields)  # ('x', 'y')

# Create from iterable
p3 = Point._make([30, 40])
print(p3)  # Point(x=30, y=40)
```

### typing.NamedTuple (Modern Approach)

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    label: str = "origin"

p = Point(10, 20)
print(p)        # Point(x=10, y=20, label='origin')
print(p.x)      # 10
print(p.label)  # "origin"

# With type hints
p2 = Point(x=3.14, y=2.72, label="pi_e")
```

---

## Tuples as Dictionary Keys

```python
# Tuples are hashable (if contents are hashable)
locations = {
    (0, 0): "origin",
    (10, 20): "point A",
    (30, 40): "point B"
}

print(locations[(10, 20)])  # "point A"

# Common use: sparse matrix representation
sparse_matrix = {
    (0, 0): 1,
    (0, 2): 2,
    (1, 1): 3,
    (2, 0): 4
}

# Access with default
value = sparse_matrix.get((0, 1), 0)  # 0 (default)
```

---

## When to Use Tuples vs Lists

| Use Tuples When | Use Lists When |
|-----------------|----------------|
| Data shouldn't change | Data needs modification |
| Returning multiple values | Collection will grow/shrink |
| Dictionary keys needed | Order matters but content changes |
| Heterogeneous data (x, y) | Homogeneous data (scores) |
| Unpacking fixed values | Iterating and processing |

```python
# ✅ Tuple: fixed structure, heterogeneous
user = ("Alice", 30, "alice@example.com")
point = (10, 20)

# ✅ List: variable length, homogeneous
scores = [85, 92, 78, 95]
names = ["Alice", "Bob", "Charlie"]

# ✅ Tuple: function returning multiple values
def divide(a, b):
    return a // b, a % b

# ✅ Tuple: dictionary key
cache = {(x, y): compute(x, y) for x in range(10) for y in range(10)}
```

---

## Performance Considerations

```python
import sys

# Tuples are more memory efficient
list_ex = [1, 2, 3, 4, 5]
tuple_ex = (1, 2, 3, 4, 5)

print(sys.getsizeof(list_ex))   # 120 bytes (varies)
print(sys.getsizeof(tuple_ex))  # 80 bytes (varies)

# Tuple creation is faster
import timeit

print(timeit.timeit("(1, 2, 3, 4, 5)", number=1000000))
print(timeit.timeit("[1, 2, 3, 4, 5]", number=1000000))
# Tuples are typically faster
```

---

## Hands-on Exercise

### Your Task

Create a `Point3D` named tuple and implement distance calculation:

```python
# 1. Create Point3D with x, y, z fields (default z=0)
# 2. Implement function to calculate distance between two points
# 3. Implement function to find the closest point to origin from a list
```

<details>
<summary>✅ Solution</summary>

```python
from typing import NamedTuple
import math

class Point3D(NamedTuple):
    x: float
    y: float
    z: float = 0.0

def distance(p1: Point3D, p2: Point3D) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt(
        (p2.x - p1.x) ** 2 +
        (p2.y - p1.y) ** 2 +
        (p2.z - p1.z) ** 2
    )

def closest_to_origin(points: list[Point3D]) -> Point3D | None:
    """Find the point closest to origin."""
    if not points:
        return None
    
    origin = Point3D(0, 0, 0)
    return min(points, key=lambda p: distance(p, origin))

# Test
p1 = Point3D(1, 2, 3)
p2 = Point3D(4, 5, 6)
print(f"Distance: {distance(p1, p2):.2f}")  # 5.20

points = [
    Point3D(10, 10, 10),
    Point3D(1, 1, 1),
    Point3D(5, 5, 5)
]
closest = closest_to_origin(points)
print(f"Closest: {closest}")  # Point3D(x=1, y=1, z=1)
```
</details>

---

## Summary

✅ **Tuples** are immutable, ordered sequences
✅ Single-element tuple needs trailing comma: `(42,)`
✅ **Packing/unpacking**: `x, y = (10, 20)`
✅ **Extended unpacking**: `first, *rest = items`
✅ **Named tuples** add field names for clarity
✅ Tuples can be **dictionary keys** (lists cannot)
✅ Use tuples for **fixed, heterogeneous** data

**Next:** [Dictionaries](./03-dictionaries.md)

---

## Further Reading

- [Tuples Documentation](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)
- [Named Tuples](https://docs.python.org/3/library/collections.html#collections.namedtuple)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/library/collections.html
-->
