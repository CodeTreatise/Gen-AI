---
title: "Lists"
---

# Lists

## Introduction

Lists are Python's most versatile data structure—ordered, mutable collections that can hold any type of element. Understanding lists deeply is essential for efficient data manipulation.

### What We'll Cover

- Creating and indexing lists
- List methods
- Slicing
- List comprehensions
- Copying (shallow vs deep)
- Sorting

### Prerequisites

- Python fundamentals
- Basic operators

---

## Creating Lists

### Basic List Creation

```python
# Empty list
empty = []
empty = list()

# With elements
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, None]

# From other iterables
chars = list("hello")      # ['h', 'e', 'l', 'l', 'o']
nums = list(range(5))      # [0, 1, 2, 3, 4]

# Nested lists
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
```

### List Multiplication

```python
# Create list with repeated elements
zeros = [0] * 5           # [0, 0, 0, 0, 0]
pattern = [1, 2] * 3      # [1, 2, 1, 2, 1, 2]

# ⚠️ Careful with mutable elements!
wrong = [[]] * 3          # [[], [], []] - same object!
wrong[0].append(1)        # [[1], [1], [1]] - Bug!

# ✅ Use comprehension instead
right = [[] for _ in range(3)]  # [[], [], []] - different objects
right[0].append(1)              # [[1], [], []]
```

---

## Indexing

### Basic Indexing

```python
fruits = ["apple", "banana", "cherry", "date"]
#          0         1         2        3
#         -4        -3        -2       -1

print(fruits[0])    # "apple"
print(fruits[-1])   # "date" (last element)
print(fruits[-2])   # "cherry" (second to last)

# Modify by index
fruits[1] = "blueberry"
print(fruits)  # ["apple", "blueberry", "cherry", "date"]
```

### Nested List Indexing

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(matrix[0][0])   # 1
print(matrix[1][2])   # 6
print(matrix[-1][-1]) # 9

# Modify nested element
matrix[1][1] = 50
```

---

## Slicing

### Basic Slicing

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(nums[2:5])     # [2, 3, 4]
print(nums[:4])      # [0, 1, 2, 3]
print(nums[6:])      # [6, 7, 8, 9]
print(nums[-3:])     # [7, 8, 9]
print(nums[:-2])     # [0, 1, 2, 3, 4, 5, 6, 7]
```

### Slicing with Step

```python
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(nums[::2])     # [0, 2, 4, 6, 8] - every 2nd
print(nums[1::2])    # [1, 3, 5, 7, 9] - odd indices
print(nums[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - reversed
print(nums[5:1:-1])  # [5, 4, 3, 2] - reversed slice
```

### Slice Assignment

```python
nums = [0, 1, 2, 3, 4, 5]

# Replace slice with same length
nums[1:3] = [10, 20]
print(nums)  # [0, 10, 20, 3, 4, 5]

# Replace with different length
nums[1:4] = [100]
print(nums)  # [0, 100, 4, 5]

# Insert without removing
nums[2:2] = [200, 300]
print(nums)  # [0, 100, 200, 300, 4, 5]

# Delete via slice
nums[1:4] = []
print(nums)  # [0, 4, 5]
```

---

## List Methods

### Adding Elements

```python
fruits = ["apple", "banana"]

# append - add to end
fruits.append("cherry")
print(fruits)  # ["apple", "banana", "cherry"]

# insert - add at index
fruits.insert(1, "blueberry")
print(fruits)  # ["apple", "blueberry", "banana", "cherry"]

# extend - add multiple elements
fruits.extend(["date", "elderberry"])
print(fruits)  # ["apple", "blueberry", "banana", "cherry", "date", "elderberry"]

# Difference: append vs extend
a = [1, 2]
a.append([3, 4])    # [1, 2, [3, 4]] - adds list as element
b = [1, 2]
b.extend([3, 4])    # [1, 2, 3, 4] - adds elements
```

### Removing Elements

```python
fruits = ["apple", "banana", "cherry", "banana"]

# remove - first occurrence by value
fruits.remove("banana")
print(fruits)  # ["apple", "cherry", "banana"]

# pop - remove by index (returns element)
last = fruits.pop()     # Removes and returns "banana"
first = fruits.pop(0)   # Removes and returns "apple"
print(fruits)  # ["cherry"]

# clear - remove all
fruits.clear()
print(fruits)  # []

# del - delete by index or slice
nums = [0, 1, 2, 3, 4, 5]
del nums[0]       # [1, 2, 3, 4, 5]
del nums[1:3]     # [1, 4, 5]
del nums[:]       # [] (clear all)
```

### Searching and Counting

```python
nums = [1, 2, 3, 2, 4, 2, 5]

# index - find position (raises ValueError if not found)
print(nums.index(3))     # 2
print(nums.index(2))     # 1 (first occurrence)
print(nums.index(2, 2))  # 3 (start searching from index 2)

# count - count occurrences
print(nums.count(2))     # 3
print(nums.count(10))    # 0

# in - membership test
print(3 in nums)         # True
print(10 in nums)        # False
```

### Other Methods

```python
# reverse - in place
nums = [1, 2, 3]
nums.reverse()
print(nums)  # [3, 2, 1]

# copy - shallow copy
original = [1, 2, 3]
copied = original.copy()
copied.append(4)
print(original)  # [1, 2, 3] - unchanged
```

---

## List Comprehensions

### Basic Syntax

```python
# [expression for item in iterable]

# Without comprehension
squares = []
for x in range(5):
    squares.append(x ** 2)

# With comprehension
squares = [x ** 2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

### With Condition

```python
# [expression for item in iterable if condition]

# Even numbers only
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Filter and transform
words = ["hello", "", "world", "", "python"]
non_empty_upper = [w.upper() for w in words if w]
print(non_empty_upper)  # ["HELLO", "WORLD", "PYTHON"]
```

### With if-else

```python
# [expr1 if condition else expr2 for item in iterable]

nums = [1, 2, 3, 4, 5]
result = ["even" if x % 2 == 0 else "odd" for x in nums]
print(result)  # ["odd", "even", "odd", "even", "odd"]
```

### Nested Comprehensions

```python
# Flatten nested list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create matrix
matrix = [[i * 3 + j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

---

## Copying Lists

### Shallow Copy

```python
# All these create shallow copies
original = [1, 2, [3, 4]]

copy1 = original.copy()
copy2 = original[:]
copy3 = list(original)

# Modifying simple elements
copy1[0] = 100
print(original[0])  # 1 - unchanged

# ⚠️ Nested objects are NOT copied
copy1[2].append(5)
print(original[2])  # [3, 4, 5] - changed!
```

### Deep Copy

```python
import copy

original = [1, 2, [3, 4]]
deep = copy.deepcopy(original)

# Now nested objects are independent
deep[2].append(5)
print(original[2])  # [3, 4] - unchanged
print(deep[2])      # [3, 4, 5]
```

---

## Sorting

### sort() - In Place

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

nums.sort()
print(nums)  # [1, 1, 2, 3, 4, 5, 6, 9]

nums.sort(reverse=True)
print(nums)  # [9, 6, 5, 4, 3, 2, 1, 1]

# Custom key
words = ["banana", "pie", "apple", "cherry"]
words.sort(key=len)
print(words)  # ["pie", "apple", "banana", "cherry"]
```

### sorted() - Returns New List

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# Original unchanged
sorted_nums = sorted(nums)
print(nums)        # [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted_nums) # [1, 1, 2, 3, 4, 5, 6, 9]

# Works on any iterable
chars = sorted("python")
print(chars)  # ['h', 'n', 'o', 'p', 't', 'y']
```

### Custom Sorting

```python
# Sort by multiple criteria
users = [
    {"name": "Charlie", "age": 25},
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

# Sort by age, then by name
users.sort(key=lambda u: (u["age"], u["name"]))
print(users)
# [{"name": "Bob", "age": 25}, {"name": "Charlie", "age": 25}, {"name": "Alice", "age": 30}]
```

---

## Hands-on Exercise

### Your Task

Implement a function to process a list of scores:

```python
# Given scores, return:
# 1. List of scores >= 50 (passing)
# 2. Average of passing scores
# 3. Highest and lowest passing scores

scores = [45, 82, 70, 95, 60, 38, 88, 42, 100, 55]
```

<details>
<summary>✅ Solution</summary>

```python
def analyze_scores(scores: list[int]) -> dict:
    """Analyze a list of scores."""
    passing = [s for s in scores if s >= 50]
    
    if not passing:
        return {"passing": [], "average": 0, "highest": None, "lowest": None}
    
    return {
        "passing": sorted(passing),
        "average": sum(passing) / len(passing),
        "highest": max(passing),
        "lowest": min(passing)
    }

scores = [45, 82, 70, 95, 60, 38, 88, 42, 100, 55]
result = analyze_scores(scores)

print(f"Passing: {result['passing']}")  # [55, 60, 70, 82, 88, 95, 100]
print(f"Average: {result['average']:.1f}")  # 78.6
print(f"Highest: {result['highest']}")  # 100
print(f"Lowest: {result['lowest']}")    # 55
```
</details>

---

## Summary

✅ **Lists** are ordered, mutable, and can contain any type
✅ **Indexing**: `list[0]`, `list[-1]` for first/last
✅ **Slicing**: `list[start:stop:step]`
✅ **Methods**: `append()`, `extend()`, `insert()`, `remove()`, `pop()`
✅ **Comprehensions**: Concise way to create lists
✅ **Shallow vs deep copy**: Use `copy.deepcopy()` for nested lists
✅ **Sorting**: `list.sort()` in-place, `sorted()` returns new list

**Next:** [Tuples](./02-tuples.md)

---

## Further Reading

- [Lists Documentation](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
- [Sorting HOW TO](https://docs.python.org/3/howto/sorting.html)

<!-- 
Sources Consulted:
- Python Docs: https://docs.python.org/3/tutorial/datastructures.html
-->
